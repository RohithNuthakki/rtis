#!/usr/bin/env python
# Enhanced symbolic regression for traffic signal formula discovery

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function

import warnings
warnings.filterwarnings('ignore')

# Configuration
BASE_DIR = "/Users/samnu/Documents/SUMO/traffic_formula/sumo_grid"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
OUTPUT_DIR = os.path.join(BASE_DIR, "formulas")
RANDOM_SEED = 42
TEST_SIZE = 0.3  # 30% of data for testing
def _protected_exponent(x):
    """Protected exponential function to avoid overflow."""
    with np.errstate(over='ignore'):
        return np.where(np.abs(x) < 20, np.exp(x), 0.)  # Limit x to avoid overflow
# Create the function object
protected_exp = make_function(function=_protected_exponent,
                             name='exp',
                             arity=1)



# Enhanced parameters
POPULATION_SIZE = 5000  # Increased from 1000
GENERATIONS = 50  # Increased from 20 
FUNCTION_SET = ('add', 'sub', 'mul', 'div', 'sqrt', 'log',  'sin', 'cos', protected_exp)  # Added more operations
PARSIMONY_COEFFICIENT = 0.005  # Reduced to allow more complex formulas

# Ensure output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def load_data():
    """Load and preprocess the simulation data"""
    # Load all episodes data
    data_file = os.path.join(RESULTS_DIR, "all_episodes.csv")
    
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}. Run simulations first.")
    
    print(f"Loading data from {data_file}...")
    data = pd.read_csv(data_file)
    
    # Filter out global records and focus on intersection decisions
    intersection_data = data[data['tls_id'] != 'global'].copy()
    
    # Add performance metrics (measured after the decision)
    # We want to measure the effect of each decision on traffic flow
    
    # Create a unique identifier for each simulation run and intersection
    intersection_data['sim_id'] = intersection_data['episode'].astype(str) + '_' + \
                                 intersection_data['strategy'].astype(str) + '_' + \
                                 intersection_data['tls_id']
    
    # Sort by simulation and time
    intersection_data = intersection_data.sort_values(['sim_id', 'step'])
    
    # Calculate the improvement in speed after each decision
    intersection_data['next_local_speed'] = intersection_data.groupby('sim_id')['local_avg_speed'].shift(-1)
    intersection_data['speed_improvement'] = intersection_data['next_local_speed'] - intersection_data['local_avg_speed']
    
    # Calculate the relative improvement (percentage)
    # Avoid division by zero
    intersection_data['relative_improvement'] = intersection_data.apply(
        lambda row: (row['speed_improvement'] / row['local_avg_speed']) 
                    if row['local_avg_speed'] > 0 else 0, 
        axis=1
    )
    
    # Drop rows with NaN values (last step of each simulation won't have a next step)
    intersection_data = intersection_data.dropna(subset=['next_local_speed'])
    
    print(f"Preprocessed data: {len(intersection_data)} records")
    print(f"From {intersection_data['episode'].nunique()} episodes")
    
    # Print distribution of speed improvements
    print("\nSpeed improvement statistics:")
    print(intersection_data['speed_improvement'].describe())
    
    return intersection_data

def prepare_features(data):
    """Prepare enhanced features for symbolic regression"""
    # Select and engineer features - now including chosen_phase and more composite features
    features = data[['chosen_phase', 'transition_needed', 'avg_incoming_speed', 
                     'avg_outgoing_speed', 'local_avg_speed', 'global_avg_speed',
                     'num_vehicles']].copy()
    
    # Convert boolean to int
    features['transition_needed'] = features['transition_needed'].astype(int)
    
    # Feature engineering: Create composite features
    # Speed ratios
    features['incoming_to_local_ratio'] = features['avg_incoming_speed'] / features['local_avg_speed'].replace(0, 0.001)
    features['outgoing_to_local_ratio'] = features['avg_outgoing_speed'] / features['local_avg_speed'].replace(0, 0.001)
    features['local_to_global_ratio'] = features['local_avg_speed'] / features['global_avg_speed'].replace(0, 0.001)
    
    # Speed differences
    features['global_local_diff'] = features['global_avg_speed'] - features['local_avg_speed']
    features['incoming_outgoing_diff'] = features['avg_incoming_speed'] - features['avg_outgoing_speed']
    features['incoming_local_diff'] = features['avg_incoming_speed'] - features['local_avg_speed']
    features['outgoing_local_diff'] = features['avg_outgoing_speed'] - features['local_avg_speed']
    
    # Phase-specific features
    features['phase_is_straight'] = ((features['chosen_phase'] == 2) | (features['chosen_phase'] == 3)).astype(int)
    features['phase_is_left'] = ((features['chosen_phase'] == 4) | (features['chosen_phase'] == 5)).astype(int)
    features['phase_is_east_west'] = ((features['chosen_phase'] == 2) | (features['chosen_phase'] == 4)).astype(int)
    features['phase_is_north_south'] = ((features['chosen_phase'] == 3) | (features['chosen_phase'] == 5)).astype(int)
    
    print("\nEngineered features:")
    for col in features.columns:
        print(f"  - {col}")
    
    # Target variable: speed improvement
    target = data['speed_improvement']
    
    # Create train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    
    print(f"\nTraining data: {len(X_train)} samples")
    print(f"Testing data: {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test, features.columns

def run_symbolic_regression(X_train, X_test, y_train, y_test, feature_names):
    """Run enhanced symbolic regression to discover formula"""
    print("\nRunning enhanced symbolic regression to discover formula...")
    print(f"Population size: {POPULATION_SIZE}, Generations: {GENERATIONS}")
    print(f"Function set: {FUNCTION_SET}")
    print(f"Parsimony coefficient: {PARSIMONY_COEFFICIENT}")
    
    # Define the symbolic regressor with enhanced parameters
    est_gp = SymbolicRegressor(
        population_size=POPULATION_SIZE,
        generations=GENERATIONS,
        stopping_criteria=0.01,
        p_crossover=0.7,
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.05,
        p_point_mutation=0.1,
        max_samples=0.9,
        verbose=1,
        parsimony_coefficient=PARSIMONY_COEFFICIENT,
        random_state=RANDOM_SEED,
        function_set=FUNCTION_SET,
        metric='mse',
        const_range=(-10.0, 10.0),
        feature_names=feature_names
    )
    
    # Fit the model
    est_gp.fit(X_train, y_train)
    
    # Evaluate on test data
    y_pred = est_gp.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nBest formula:")
    print(str(est_gp._program))
    print(f"Test RMSE: {np.sqrt(mse):.4f}")
    print(f"Test R²: {r2:.4f}")
    
    # Create a visualization
    plt.figure(figsize=(12, 8))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.xlabel('Actual Speed Improvement')
    plt.ylabel('Predicted Speed Improvement')
    plt.title('Symbolic Regression Performance')
    plt.grid(True)
    
    plot_file = os.path.join(OUTPUT_DIR, 'formula_performance.png')
    plt.savefig(plot_file)
    print(f"Performance plot saved to: {plot_file}")
    
    # Plot feature importances based on their occurrence in the formula
    formula_str = str(est_gp._program)
    feature_occurrences = {}
    
    for i, name in enumerate(feature_names):
        feature_occurrences[name] = formula_str.count(f"X{i}")
    
    # Sort by occurrence count
    sorted_features = {k: v for k, v in sorted(feature_occurrences.items(), key=lambda item: item[1], reverse=True) if v > 0}
    
    if sorted_features:
        plt.figure(figsize=(12, 6))
        plt.barh(list(sorted_features.keys()), list(sorted_features.values()))
        plt.xlabel('Number of Occurrences in Formula')
        plt.title('Feature Importance in Discovered Formula')
        plt.tight_layout()
        importance_file = os.path.join(OUTPUT_DIR, 'feature_importance.png')
        plt.savefig(importance_file)
        print(f"Feature importance plot saved to: {importance_file}")
    
    # Return the best formula
    return est_gp, np.sqrt(mse), r2

def create_safe_operation_functions():
    """Create safe versions of mathematical operations to handle edge cases"""
    code = """
def safe_div(x, y):
    \"\"\"Safe division function to handle division by zero\"\"\"
    if abs(y) < 0.001:
        return x  # Return numerator when denominator is close to zero
    return x / y

def safe_log(x):
    \"\"\"Safe logarithm function to handle negative or zero values\"\"\"
    if x <= 0:
        return 0
    return np.log(x)

def safe_sqrt(x):
    \"\"\"Safe square root function to handle negative values\"\"\"
    if x < 0:
        return 0
    return np.sqrt(x)

def safe_exp(x):
    \"\"\"Safe exponential function to handle overflow\"\"\"
    if x > 20:  # Prevent overflow
        return np.exp(20)
    return np.exp(x)

def safe_sin(x):
    \"\"\"Sine function\"\"\"
    return np.sin(x)

def safe_cos(x):
    \"\"\"Cosine function\"\"\"
    return np.cos(x)
"""
    return code

def create_phase_selection_formula(formula):
    """Convert the discovered formula to a phase selection function with enhanced features"""
    formula_str = str(formula._program)
    
    # Create the operations code
    operations_code = create_safe_operation_functions()
    
    # Create feature engineering code
    feature_engineering_code = """
def engineer_features(chosen_phase, transition_needed, avg_incoming_speed, avg_outgoing_speed, local_avg_speed, global_avg_speed, num_vehicles):
    \"\"\"Create engineered features from raw inputs\"\"\"
    # Original features
    X0 = chosen_phase
    X1 = transition_needed
    X2 = avg_incoming_speed
    X3 = avg_outgoing_speed
    X4 = local_avg_speed
    X5 = global_avg_speed
    X6 = num_vehicles
    
    # Speed ratios
    X7 = safe_div(avg_incoming_speed, local_avg_speed)  # incoming_to_local_ratio
    X8 = safe_div(avg_outgoing_speed, local_avg_speed)  # outgoing_to_local_ratio
    X9 = safe_div(local_avg_speed, global_avg_speed)    # local_to_global_ratio
    
    # Speed differences
    X10 = global_avg_speed - local_avg_speed           # global_local_diff
    X11 = avg_incoming_speed - avg_outgoing_speed      # incoming_outgoing_diff
    X12 = avg_incoming_speed - local_avg_speed         # incoming_local_diff
    X13 = avg_outgoing_speed - local_avg_speed         # outgoing_local_diff
    
    # Phase-specific features
    X14 = 1 if (chosen_phase == 2 or chosen_phase == 3) else 0  # phase_is_straight
    X15 = 1 if (chosen_phase == 4 or chosen_phase == 5) else 0  # phase_is_left
    X16 = 1 if (chosen_phase == 2 or chosen_phase == 4) else 0  # phase_is_east_west
    X17 = 1 if (chosen_phase == 3 or chosen_phase == 5) else 0  # phase_is_north_south
    
    return [X0, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14, X15, X16, X17]
"""
    
    # Create the predict function
    predict_function_code = f"""
def predict_speed_improvement(chosen_phase, transition_needed, avg_incoming_speed, avg_outgoing_speed, local_avg_speed, global_avg_speed, num_vehicles):
    \"\"\"
    Predict the speed improvement based on the discovered formula.
    
    Formula: {formula_str}
    \"\"\"
    # Convert inputs to feature array
    features = engineer_features(chosen_phase, transition_needed, avg_incoming_speed, avg_outgoing_speed, local_avg_speed, global_avg_speed, num_vehicles)
    
    # Extract features for readability
    X0, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14, X15, X16, X17 = features
    
    # Calculate predicted improvement (formula discovered by symbolic regression)
    return {formula_str}
"""
    
    # Create the phase selection function
    selection_function_code = """
def select_best_phase(current_phase, avg_incoming_speed, avg_outgoing_speed, local_avg_speed, global_avg_speed, num_vehicles):
    \"\"\"
    Select the best phase based on predicted speed improvement.
    
    Parameters:
    -----------
    current_phase : int
        The current phase (2-5)
    avg_incoming_speed : float
        Average speed on incoming edges
    avg_outgoing_speed : float
        Average speed on outgoing edges
    local_avg_speed : float
        Average speed around the intersection
    global_avg_speed : float
        Global average speed
    num_vehicles : int
        Number of vehicles in the simulation
        
    Returns:
    --------
    int
        The best phase (2-5) to select next
    \"\"\"
    best_improvement = float('-inf')
    best_phase = current_phase  # Default to keeping current phase to avoid transitions
    
    # Evaluate each possible phase
    for phase in range(2, 6):  # Phases 2-5
        # Check if this would require a transition
        transition_needed = 1 if phase != current_phase else 0
        
        # Predict speed improvement
        improvement = predict_speed_improvement(
            phase, transition_needed, avg_incoming_speed, 
            avg_outgoing_speed, local_avg_speed, global_avg_speed, num_vehicles
        )
        
        # Apply a penalty for transitions to account for the transition phase cost
        if transition_needed:
            improvement -= 0.5  # This penalty might need tuning
        
        if improvement > best_improvement:
            best_improvement = improvement
            best_phase = phase
    
    return best_phase
"""
    
    # Combine all code
    full_code = "import numpy as np\n\n" + operations_code + "\n" + feature_engineering_code + "\n" + predict_function_code + "\n" + selection_function_code
    
    # Save the formula code to a file
    formula_file = os.path.join(OUTPUT_DIR, "phase_selection_formula.py")
    with open(formula_file, 'w') as f:
        f.write(full_code)
    
    print(f"Enhanced phase selection formula saved to {formula_file}")
    
    return full_code

def main():
    """Main execution function"""
    print("=" * 80)
    print("ENHANCED SYMBOLIC REGRESSION FOR TRAFFIC SIGNAL FORMULA DISCOVERY")
    print("=" * 80)
    
    # Step 1: Load and preprocess data
    data = load_data()
    
    # Step 2: Prepare features with engineering
    X_train, X_test, y_train, y_test, feature_names = prepare_features(data)
    
    # Step 3: Run symbolic regression with enhanced parameters
    best_formula, rmse, r2 = run_symbolic_regression(X_train, X_test, y_train, y_test, feature_names)
    
    # Step 4: Create enhanced phase selection formula
    phase_formula = create_phase_selection_formula(best_formula)
    
    # Save metadata about the formula
    metadata_file = os.path.join(OUTPUT_DIR, "formula_metadata.txt")
    with open(metadata_file, 'w') as f:
        f.write(f"Formula: {str(best_formula._program)}\n")
        f.write(f"Test RMSE: {rmse:.4f}\n")
        f.write(f"Test R²: {r2:.4f}\n")
        f.write("\nFeature names:\n")
        for idx, name in enumerate(feature_names):
            f.write(f"X{idx}: {name}\n")
    
    print(f"Formula metadata saved to {metadata_file}")
    print("\nProcess complete!")

if __name__ == "__main__":
    main()