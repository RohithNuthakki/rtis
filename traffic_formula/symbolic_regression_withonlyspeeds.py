#!/usr/bin/env python
# Simplified symbolic regression for traffic signal formula discovery
# Using only pure speed features (incoming, outgoing, and calculated global)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from gplearn.genetic import SymbolicRegressor

# Configuration
BASE_DIR = "/Users/samnu/Documents/SUMO/traffic_formula/sumo_grid"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
OUTPUT_DIR = os.path.join(BASE_DIR, "formulas_pure_speeds")
RANDOM_SEED = 42
TEST_SIZE = 0.3  # 30% of data for testing

# Simplified parameters
POPULATION_SIZE = 1000
GENERATIONS = 20
FUNCTION_SET = ('add', 'sub', 'mul', 'div')  # Basic operations only
PARSIMONY_COEFFICIENT = 0.01  # Favor simpler formulas

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
    
    # Get all available columns
    print("\nAvailable columns in dataset:")
    for col in data.columns:
        print(f"  - {col}")
    
    # Filter to only include intersection data (not global data)
    intersection_data = data[data['tls_id'] != 'global'].copy()
    
    # Calculate global average speed for each episode/step 
    # (average of all intersection incoming and outgoing speeds)
    print("\nCalculating properly defined global average speed...")
    
    # Group by episode, strategy, and step to get all intersections at the same time point
    groups = intersection_data.groupby(['episode', 'strategy', 'step'])
    
    # Calculate global average speed for each group (average of all incoming and outgoing speeds)
    global_speeds = []
    
    for name, group in groups:
        episode, strategy, step = name
        # Calculate global avg speed as average of all incoming and outgoing speeds at this time point
        global_avg_speed = (group['avg_incoming_speed'].mean() + group['avg_outgoing_speed'].mean()) / 2
        
        # Add to each row in this group
        for idx in group.index:
            global_speeds.append({
                'index': idx,
                'calculated_global_avg_speed': global_avg_speed
            })
    
    # Convert to DataFrame and merge back with the original data
    global_speed_df = pd.DataFrame(global_speeds)
    global_speed_df.set_index('index', inplace=True)
    
    # Join the calculated global speeds back to the intersection data
    intersection_data = intersection_data.join(global_speed_df)
    
    # Create a unique identifier for each simulation run and intersection
    intersection_data['sim_id'] = intersection_data['episode'].astype(str) + '_' + \
                                 intersection_data['strategy'].astype(str) + '_' + \
                                 intersection_data['tls_id']
    
    # Sort by simulation and time
    intersection_data = intersection_data.sort_values(['sim_id', 'step'])
    
    # Calculate the improvement in speed after each decision
    intersection_data['next_local_speed'] = intersection_data.groupby('sim_id')['local_avg_speed'].shift(-1)
    intersection_data['speed_improvement'] = intersection_data['next_local_speed'] - intersection_data['local_avg_speed']
    
    # Drop rows with NaN values (last step of each simulation won't have a next step)
    intersection_data = intersection_data.dropna(subset=['next_local_speed', 'calculated_global_avg_speed'])
    
    print(f"Preprocessed data: {len(intersection_data)} records")
    print(f"From {intersection_data['episode'].nunique()} episodes")
    print(f"Global average speed range: {intersection_data['calculated_global_avg_speed'].min():.2f} to {intersection_data['calculated_global_avg_speed'].max():.2f}")
    
    return intersection_data

def prepare_pure_speed_features(data):
    """
    Prepare features using only pure speed measurements:
    1. Incoming local average speed
    2. Outgoing local average speed
    3. Calculated global average speed (average of all intersection incoming and outgoing speeds)
    """
    # Select only the three speed features plus phase and transition info
    features = data[['chosen_phase', 'transition_needed', 
                    'avg_incoming_speed', 'avg_outgoing_speed', 
                    'calculated_global_avg_speed']].copy()
    
    # Convert boolean to int if needed
    if features['transition_needed'].dtype == 'bool':
        features['transition_needed'] = features['transition_needed'].astype(int)
    elif features['transition_needed'].dtype == 'object':
        features['transition_needed'] = (features['transition_needed'] == 'True').astype(int)
    
    # Print features used
    print("\nFeatures used for pure speed symbolic regression:")
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

def run_pure_speed_symbolic_regression(X_train, X_test, y_train, y_test, feature_names):
    """Run symbolic regression with pure speed features"""
    print("\nRunning pure speed symbolic regression...")
    print(f"Population size: {POPULATION_SIZE}, Generations: {GENERATIONS}")
    print(f"Function set: {FUNCTION_SET}")
    print(f"Parsimony coefficient: {PARSIMONY_COEFFICIENT}")
    
    # Define the symbolic regressor
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
    
    print(f"\nBest pure speed formula:")
    formula_str = str(est_gp._program)
    print(formula_str)
    print(f"Test RMSE: {np.sqrt(mse):.4f}")
    print(f"Test R²: {r2:.4f}")
    
    # Create visualization for model performance
    plt.figure(figsize=(12, 8))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    
    # Add regression line
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(y_test, y_pred)
    x_vals = np.array([min(y_test), max(y_test)])
    plt.plot(x_vals, intercept + slope * x_vals, 'g-', linewidth=2)
    
    plt.xlabel('Actual Speed Improvement')
    plt.ylabel('Predicted Speed Improvement')
    plt.title(f'Pure Speed Model Performance (RMSE: {np.sqrt(mse):.4f}, R²: {r2:.4f})')
    plt.grid(True)
    
    plot_file = os.path.join(OUTPUT_DIR, 'pure_speed_formula_performance.png')
    plt.savefig(plot_file)
    print(f"Performance plot saved to: {plot_file}")
    
    # Find important features in the formula
    feature_occurrences = {}
    for i, name in enumerate(feature_names):
        feature_occurrences[name] = formula_str.count(f"X{i}")
    
    # Plot feature importances
    plt.figure(figsize=(10, 6))
    features_used = {name: count for name, count in feature_occurrences.items() if count > 0}
    if features_used:
        bars = plt.bar(features_used.keys(), features_used.values(), color='cornflowerblue')
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('Features')
        plt.ylabel('Occurrences in Formula')
        plt.title('Feature Importance in Pure Speed Formula')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.0f}', ha='center', va='bottom')
        
        plt.tight_layout()
        importance_file = os.path.join(OUTPUT_DIR, 'pure_speed_feature_importance.png')
        plt.savefig(importance_file)
        print(f"Feature importance plot saved to: {importance_file}")
    
    return est_gp, np.sqrt(mse), r2, formula_str

def create_pure_speed_phase_selection_formula(formula_str):
    """Create a phase selection formula using only pure speed features"""
    
    # Add helper functions
    helper_functions = """
# Helper functions for safe operations
def add(x, y):
    return x + y

def sub(x, y):
    return x - y
    
def mul(x, y):
    return x * y
    
def div(x, y):
    if abs(y) < 0.001:
        return x  # Return numerator when denominator is close to zero
    return x / y
"""
    
    # Replace any direct references to calculated_global_avg_speed in the formula string
    modified_formula_str = formula_str.replace("calculated_global_avg_speed", "X4")
    
    # Create the prediction function with compatibility for both global_avg_speed and calculated_global_avg_speed
    predict_function = f"""
def predict_speed_improvement(chosen_phase, transition_needed, avg_incoming_speed, avg_outgoing_speed, global_avg_speed, local_avg_speed=None, num_vehicles=None, incoming_length=200, outgoing_length=200):
    \"\"\"
    Predict speed improvement using only pure speed features.
    
    Formula: {formula_str}
    
    Parameters:
    -----------
    chosen_phase : int
        The phase to evaluate (2-5)
    transition_needed : int
        Whether a transition is needed (0 or 1)
    avg_incoming_speed : float
        Average speed on incoming edges
    avg_outgoing_speed : float
        Average speed on outgoing edges
    global_avg_speed : float
        Global average speed across the network (average of all incoming and outgoing speeds)
        Note: This can be either 'global_avg_speed' or 'calculated_global_avg_speed' from the data
    local_avg_speed : float, optional
        Local average speed (not used in this formula)
    num_vehicles : int, optional
        Number of vehicles (not used in this formula)
    incoming_length : float, optional
        Length of incoming roads (not used in this formula)
    outgoing_length : float, optional
        Length of outgoing roads (not used in this formula)
        
    Returns:
    --------
    float
        Predicted speed improvement
    \"\"\"
    # Define features used in the formula
    X0 = chosen_phase
    X1 = transition_needed
    X2 = avg_incoming_speed
    X3 = avg_outgoing_speed
    X4 = global_avg_speed  # This works with either column name since we use the parameter name
    
    # Return the predicted improvement using the discovered formula
    return {modified_formula_str}
"""
    
    # Create the phase selection function
    select_phase_function = """
def select_best_phase(current_phase, avg_incoming_speed, avg_outgoing_speed, local_avg_speed, global_avg_speed, num_vehicles=None, incoming_length=200, outgoing_length=200):
    \"\"\"
    Select the best phase using the pure speed formula.
    
    Parameters:
    -----------
    current_phase : int
        The current phase (2-5)
    avg_incoming_speed : float
        Average speed on incoming edges
    avg_outgoing_speed : float
        Average speed on outgoing edges
    local_avg_speed : float
        Local average speed around the intersection
    global_avg_speed : float
        Global average speed (average of all intersection incoming and outgoing speeds)
        Note: This can be either 'global_avg_speed' or 'calculated_global_avg_speed' from the data
    num_vehicles : int, optional
        Number of vehicles (not used in this formula)
    incoming_length : float, optional
        Length of incoming roads (not used in this formula)
    outgoing_length : float, optional
        Length of outgoing roads (not used in this formula)
        
    Returns:
    --------
    int
        The best phase (2-5) to select next
    \"\"\"
    best_improvement = float('-inf')
    best_phase = current_phase  # Default to keeping current phase
    
    # Evaluate each possible phase
    for phase in range(2, 6):  # Phases 2-5
        # Check if this would require a transition
        transition_needed = 1 if phase != current_phase else 0
        
        # Predict speed improvement
        improvement = predict_speed_improvement(
            phase, transition_needed, avg_incoming_speed, avg_outgoing_speed,
            global_avg_speed, local_avg_speed, num_vehicles,
            incoming_length, outgoing_length
        )
        
        # Apply a small penalty for transitions
        if transition_needed:
            improvement -= 0.2
        
        if improvement > best_improvement:
            best_improvement = improvement
            best_phase = phase
    
    return best_phase
"""

    # Add max pressure functions for validation
    max_pressure_functions = """
def calculate_max_pressure_score(chosen_phase, avg_incoming_speed, avg_outgoing_speed, 
                                incoming_length=200, outgoing_length=200):
    \"\"\"
    Calculate the Max Pressure score: (incoming speed / incoming length) - (outgoing speed / outgoing length)
    
    This measures the differential pressure across the intersection.
    \"\"\"
    # Avoid division by zero
    if incoming_length <= 0:
        incoming_length = 0.001
    if outgoing_length <= 0:
        outgoing_length = 0.001
        
    # Calculate pressure differential
    incoming_pressure = avg_incoming_speed / incoming_length
    outgoing_pressure = avg_outgoing_speed / outgoing_length
    
    # Return the pressure differential
    return incoming_pressure - outgoing_pressure

def select_max_pressure_phase(current_phase, avg_incoming_speed, avg_outgoing_speed, local_avg_speed=None, 
                             global_avg_speed=None, num_vehicles=None, incoming_length=200, outgoing_length=200):
    \"\"\"
    Select the best phase using the Max Pressure formula.
    
    Parameters:
    -----------
    current_phase : int
        The current phase (2-5)
    avg_incoming_speed : float
        Average speed on incoming edges
    avg_outgoing_speed : float
        Average speed on outgoing edges
    local_avg_speed : float, optional
        Local average speed (not used in this formula)
    global_avg_speed : float, optional
        Global average speed (not used in this formula)
    num_vehicles : int, optional
        Number of vehicles (not used in this formula)
    incoming_length : float, optional
        Length of incoming roads (default 200)
    outgoing_length : float, optional
        Length of outgoing roads (default 200)
        
    Returns:
    --------
    int
        The best phase (2-5) to select next
    \"\"\"
    best_score = float('-inf')
    best_phase = current_phase  # Default to keeping current phase
    
    # Evaluate each possible phase
    for phase in range(2, 6):  # Phases 2-5
        # Calculate Max Pressure score
        score = calculate_max_pressure_score(
            phase, avg_incoming_speed, avg_outgoing_speed, 
            incoming_length, outgoing_length
        )
        
        # Apply a small penalty for transitions
        if phase != current_phase:
            score -= 0.2
        
        if score > best_score:
            best_score = score
            best_phase = phase
    
    return best_phase
"""
    
    # Combine all code
    full_code = "import numpy as np\n\n" + helper_functions + "\n" + predict_function + "\n" + select_phase_function + "\n" + max_pressure_functions
    
    # Save the formula code to a file
    formula_file = os.path.join(OUTPUT_DIR, "pure_speed_phase_selection_formula.py")
    with open(formula_file, 'w') as f:
        f.write(full_code)
    
    # Also save to the directory where the validator looks for it
    os.makedirs(os.path.join(BASE_DIR, "formulas_simple"), exist_ok=True)
    validator_formula_file = os.path.join(BASE_DIR, "formulas_simple", "simplified_phase_selection_formula.py")
    with open(validator_formula_file, 'w') as f:
        f.write(full_code)
    
    print(f"Pure speed phase selection formula saved to {formula_file}")
    print(f"Formula also saved for validator at {validator_formula_file}")
    
    return full_code
def main():
    """Main execution function"""
    print("=" * 80)
    print("SYMBOLIC REGRESSION FOR TRAFFIC SIGNAL FORMULA")
    print("Using only pure speed features (incoming, outgoing, and calculated global)")
    print("=" * 80)
    
    # Step 1: Load and preprocess data
    data = load_data()
    
    # Step 2: Prepare pure speed features
    X_train, X_test, y_train, y_test, feature_names = prepare_pure_speed_features(data)
    
    # Step 3: Run symbolic regression
    best_formula, rmse, r2, formula_str = run_pure_speed_symbolic_regression(X_train, X_test, y_train, y_test, feature_names)
    
    # Step 4: Create phase selection formula
    phase_formula = create_pure_speed_phase_selection_formula(formula_str)
    
    # Save metadata about the formula
    metadata_file = os.path.join(OUTPUT_DIR, "pure_speed_formula_metadata.txt")
    with open(metadata_file, 'w') as f:
        f.write(f"Pure Speed Formula: {formula_str}\n")
        f.write(f"Test RMSE: {rmse:.4f}\n")
        f.write(f"Test R²: {r2:.4f}\n\n")
        f.write("Features used:\n")
        for idx, name in enumerate(feature_names):
            f.write(f"X{idx}: {name}\n")
        f.write("\nThis formula uses only pure speed measurements:\n")
        f.write("1. Incoming local average speed\n")
        f.write("2. Outgoing local average speed\n")
        f.write("3. Calculated global average speed (average of all intersection incoming and outgoing speeds)\n")
    
    print(f"Pure speed formula metadata saved to {metadata_file}")
    print("\nProcess complete!")

if __name__ == "__main__":
    main()