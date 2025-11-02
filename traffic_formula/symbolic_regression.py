#!/usr/bin/env python
# Improved symbolic regression for traffic signal formula discovery

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
OUTPUT_DIR = os.path.join(BASE_DIR, "formulas")
RANDOM_SEED = 42
TEST_SIZE = 0.3  # 30% of data for testing

# Enhanced parameters
POPULATION_SIZE = 2000  # Increased from 1000
GENERATIONS = 35  # Increased from 20
FUNCTION_SET = ('add', 'sub', 'mul', 'div', 'sqrt')  # Added sqrt for nonlinear relationships
PARSIMONY_COEFFICIENT = 0.008  # Reduced to allow more complex formulas

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
    
    # Show statistics on speed improvements
    print("\nSpeed improvement statistics:")
    print(intersection_data['speed_improvement'].describe())
    
    # Plot distribution of speed improvements
    plt.figure(figsize=(10, 6))
    plt.hist(intersection_data['speed_improvement'], bins=50)
    plt.xlabel('Speed Improvement (m/s)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Speed Improvements')
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'speed_improvement_distribution.png'))
    
    print(f"Preprocessed data: {len(intersection_data)} records")
    print(f"From {intersection_data['episode'].nunique()} episodes")
    
    return intersection_data

def prepare_features(data):
    """Prepare features for symbolic regression with comprehensive feature engineering"""
    # Select base features - excluding num_vehicles to match HERE API-like data
    features = data[['chosen_phase', 'transition_needed', 'avg_incoming_speed', 
                     'avg_outgoing_speed', 'local_avg_speed', 'global_avg_speed']].copy()
    
    # Convert boolean to int
    features['transition_needed'] = features['transition_needed'].astype(int)
    
    # Simulate road segment lengths (in real implementation, these would come from the API)
    np.random.seed(42)  # For reproducibility
    incoming_length = np.random.uniform(50, 500, len(features))
    outgoing_length = np.random.uniform(50, 500, len(features))
    
    # Add segment lengths as features
    features['incoming_length'] = incoming_length
    features['outgoing_length'] = outgoing_length
    features['total_segment_length'] = incoming_length + outgoing_length
    
    # Basic speed ratios (with safeguards against division by zero)
    features['local_to_global_ratio'] = features['local_avg_speed'] / features['global_avg_speed'].replace(0, 0.001)
    features['incoming_to_local_ratio'] = features['avg_incoming_speed'] / features['local_avg_speed'].replace(0, 0.001)
    features['outgoing_to_local_ratio'] = features['avg_outgoing_speed'] / features['local_avg_speed'].replace(0, 0.001)
    features['incoming_to_outgoing_ratio'] = features['avg_incoming_speed'] / features['avg_outgoing_speed'].replace(0, 0.001)
    
    # NEW: Local-global speed relationships
    features['local_global_diff'] = features['local_avg_speed'] - features['global_avg_speed']
    features['local_global_sum'] = features['local_avg_speed'] + features['global_avg_speed']
    features['local_global_product'] = features['local_avg_speed'] * features['global_avg_speed']
    
    # Basic speed differences
    features['incoming_outgoing_diff'] = features['avg_incoming_speed'] - features['avg_outgoing_speed']
    features['incoming_local_diff'] = features['avg_incoming_speed'] - features['local_avg_speed']
    features['outgoing_local_diff'] = features['avg_outgoing_speed'] - features['local_avg_speed']
    
    # NEW: Speed sums
    features['incoming_outgoing_sum'] = features['avg_incoming_speed'] + features['avg_outgoing_speed']
    features['incoming_local_sum'] = features['avg_incoming_speed'] + features['local_avg_speed']
    features['outgoing_local_sum'] = features['avg_outgoing_speed'] + features['local_avg_speed']
    
    # NEW: Speed products
    features['incoming_outgoing_product'] = features['avg_incoming_speed'] * features['avg_outgoing_speed']
    features['incoming_local_product'] = features['avg_incoming_speed'] * features['local_avg_speed']
    features['outgoing_local_product'] = features['avg_outgoing_speed'] * features['local_avg_speed']
    
    # Length-normalized speed features
    features['incoming_speed_per_meter'] = features['avg_incoming_speed'] / features['incoming_length']
    features['outgoing_speed_per_meter'] = features['avg_outgoing_speed'] / features['outgoing_length']
    features['local_speed_per_meter'] = features['local_avg_speed'] / features['total_segment_length']
    features['global_speed_per_meter'] = features['global_avg_speed'] / features['total_segment_length']
    
    # Differences for length-normalized speeds
    features['incoming_outgoing_speed_per_meter_diff'] = features['incoming_speed_per_meter'] - features['outgoing_speed_per_meter']
    features['local_global_speed_per_meter_diff'] = features['local_speed_per_meter'] - features['global_speed_per_meter']
    features['incoming_local_speed_per_meter_diff'] = features['incoming_speed_per_meter'] - features['local_speed_per_meter']
    features['outgoing_local_speed_per_meter_diff'] = features['outgoing_speed_per_meter'] - features['local_speed_per_meter']
    
    # Sum of length-normalized speeds
    features['incoming_outgoing_speed_per_meter_sum'] = features['incoming_speed_per_meter'] + features['outgoing_speed_per_meter']
    features['local_global_speed_per_meter_sum'] = features['local_speed_per_meter'] + features['global_speed_per_meter']
    features['all_speed_per_meter_sum'] = features['incoming_speed_per_meter'] + features['outgoing_speed_per_meter'] + features['local_speed_per_meter'] + features['global_speed_per_meter']
    
    # Products of length-normalized speeds
    features['incoming_outgoing_speed_per_meter_product'] = features['incoming_speed_per_meter'] * features['outgoing_speed_per_meter']
    features['local_global_speed_per_meter_product'] = features['local_speed_per_meter'] * features['global_speed_per_meter']
    
    # Travel time estimates (in seconds)
    features['incoming_travel_time'] = features['incoming_length'] / features['avg_incoming_speed'].replace(0, 0.001)
    features['outgoing_travel_time'] = features['outgoing_length'] / features['avg_outgoing_speed'].replace(0, 0.001)
    features['total_travel_time'] = features['incoming_travel_time'] + features['outgoing_travel_time']
    
    # Travel time differences
    features['incoming_outgoing_travel_time_diff'] = features['incoming_travel_time'] - features['outgoing_travel_time']
    features['incoming_total_travel_time_ratio'] = features['incoming_travel_time'] / features['total_travel_time'].replace(0, 0.001)
    features['outgoing_total_travel_time_ratio'] = features['outgoing_travel_time'] / features['total_travel_time'].replace(0, 0.001)
    
    # Travel time products
    features['travel_time_product'] = features['incoming_travel_time'] * features['outgoing_travel_time']
    
    # Combined speed-time metrics
    features['speed_time_balance'] = (features['local_avg_speed'] * features['total_travel_time'])
    features['speed_time_efficiency'] = features['local_avg_speed'] / features['total_travel_time'].replace(0, 0.001)
    
    # Phase-specific features
    features['phase_is_straight'] = ((features['chosen_phase'] == 2) | (features['chosen_phase'] == 3)).astype(int)
    features['phase_is_left'] = ((features['chosen_phase'] == 4) | (features['chosen_phase'] == 5)).astype(int)
    features['phase_is_east_west'] = ((features['chosen_phase'] == 2) | (features['chosen_phase'] == 4)).astype(int)
    features['phase_is_north_south'] = ((features['chosen_phase'] == 3) | (features['chosen_phase'] == 5)).astype(int)
    
    # Print features used
    print("\nFeatures used for symbolic regression:")
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
    """Run symbolic regression to discover formula"""
    print("\nRunning symbolic regression to discover formula...")
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
    
    # Find important features in the formula
    formula_str = str(est_gp._program)
    feature_occurrences = {}
    
    for i, name in enumerate(feature_names):
        feature_occurrences[name] = formula_str.count(f"X{i}")
    
    # Plot feature importances
    plt.figure(figsize=(12, 6))
    features_used = {name: count for name, count in feature_occurrences.items() if count > 0}
    if features_used:
        plt.bar(features_used.keys(), features_used.values())
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('Features')
        plt.ylabel('Occurrences in Formula')
        plt.title('Feature Importance in Discovered Formula')
        plt.tight_layout()
        importance_file = os.path.join(OUTPUT_DIR, 'feature_importance.png')
        plt.savefig(importance_file)
        print(f"Feature importance plot saved to: {importance_file}")
    
    # Return the best formula
    return est_gp, np.sqrt(mse), r2

def create_phase_selection_formula(formula, feature_names):
    """Convert the discovered formula to a phase selection function with comprehensive features"""
    formula_str = str(formula._program)
    
    # Extract the raw formula string without variable replacements
    raw_formula = formula_str
    
    # Create a version with X variables for the actual calculation
    variable_formula = formula_str
    
    # Add helper functions to handle edge cases
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
    
def sqrt(x):
    if x < 0:
        return 0
    return np.sqrt(x)
"""
    
    # Create feature setup code with clear variables
    feature_setup = f"""
def predict_speed_improvement(chosen_phase, transition_needed, avg_incoming_speed, avg_outgoing_speed, local_avg_speed, global_avg_speed, num_vehicles=None, incoming_length=200, outgoing_length=200):
    \"\"\"
    Predict the speed improvement based on the discovered formula.
    
    Formula: {raw_formula}
    \"\"\"
    # Original features
    X0 = chosen_phase
    X1 = transition_needed
    X2 = avg_incoming_speed
    X3 = avg_outgoing_speed
    X4 = local_avg_speed
    X5 = global_avg_speed
    
    # Segment lengths
    X6 = incoming_length
    X7 = outgoing_length
    X8 = incoming_length + outgoing_length  # total_segment_length
    
    # Speed ratios
    X9 = div(local_avg_speed, global_avg_speed)  # local_to_global_ratio
    X10 = div(avg_incoming_speed, local_avg_speed)  # incoming_to_local_ratio
    X11 = div(avg_outgoing_speed, local_avg_speed)  # outgoing_to_local_ratio
    X12 = div(avg_incoming_speed, avg_outgoing_speed)  # incoming_to_outgoing_ratio
    
    # Local-global speed relationships
    X13 = local_avg_speed - global_avg_speed  # local_global_diff
    X14 = local_avg_speed + global_avg_speed  # local_global_sum
    X15 = local_avg_speed * global_avg_speed  # local_global_product
    
    # Basic speed differences and relationships
    X16 = avg_incoming_speed - avg_outgoing_speed  # incoming_outgoing_diff
    X17 = avg_incoming_speed - local_avg_speed  # incoming_local_diff
    X18 = avg_outgoing_speed - local_avg_speed  # outgoing_local_diff
    X19 = avg_incoming_speed + avg_outgoing_speed  # incoming_outgoing_sum
    X20 = avg_incoming_speed + local_avg_speed  # incoming_local_sum
    X21 = avg_outgoing_speed + local_avg_speed  # outgoing_local_sum
    X22 = avg_incoming_speed * avg_outgoing_speed  # incoming_outgoing_product
    X23 = avg_incoming_speed * local_avg_speed  # incoming_local_product
    X24 = avg_outgoing_speed * local_avg_speed  # outgoing_local_product
    
    # Length-normalized speed features
    X25 = div(avg_incoming_speed, incoming_length)  # incoming_speed_per_meter
    X26 = div(avg_outgoing_speed, outgoing_length)  # outgoing_speed_per_meter
    X27 = div(local_avg_speed, (incoming_length + outgoing_length))  # local_speed_per_meter
    X28 = div(global_avg_speed, (incoming_length + outgoing_length))  # global_speed_per_meter
    
    # Differences for length-normalized speeds
    X29 = X25 - X26  # incoming_outgoing_speed_per_meter_diff
    X30 = X27 - X28  # local_global_speed_per_meter_diff
    X31 = X25 - X27  # incoming_local_speed_per_meter_diff
    X32 = X26 - X27  # outgoing_local_speed_per_meter_diff
    
    # Sum of length-normalized speeds
    X33 = X25 + X26  # incoming_outgoing_speed_per_meter_sum
    X34 = X27 + X28  # local_global_speed_per_meter_sum
    X35 = X25 + X26 + X27 + X28  # all_speed_per_meter_sum
    
    # Products of length-normalized speeds
    X36 = X25 * X26  # incoming_outgoing_speed_per_meter_product
    X37 = X27 * X28  # local_global_speed_per_meter_product
    
    # Travel time estimates (in seconds)
    X38 = div(incoming_length, avg_incoming_speed)  # incoming_travel_time
    X39 = div(outgoing_length, avg_outgoing_speed)  # outgoing_travel_time
    X40 = X38 + X39  # total_travel_time
    
    # Travel time differences and relationships
    X41 = X38 - X39  # incoming_outgoing_travel_time_diff
    X42 = div(X38, X40)  # incoming_total_travel_time_ratio
    X43 = div(X39, X40)  # outgoing_total_travel_time_ratio
    X44 = X38 * X39  # travel_time_product
    
    # Combined speed-time metrics
    X45 = local_avg_speed * X40  # speed_time_balance
    X46 = div(local_avg_speed, X40)  # speed_time_efficiency
    
    # Phase-specific features
    X47 = 1 if (chosen_phase == 2 or chosen_phase == 3) else 0  # phase_is_straight
    X48 = 1 if (chosen_phase == 4 or chosen_phase == 5) else 0  # phase_is_left
    X49 = 1 if (chosen_phase == 2 or chosen_phase == 4) else 0  # phase_is_east_west
    X50 = 1 if (chosen_phase == 3 or chosen_phase == 5) else 0  # phase_is_north_south
    
    # Define variables for readability
    # Segment lengths
    incoming_length = X6
    outgoing_length = X7
    total_segment_length = X8
    
    # Speed ratios
    local_to_global_ratio = X9
    incoming_to_local_ratio = X10
    outgoing_to_local_ratio = X11
    incoming_to_outgoing_ratio = X12
    
    # Local-global speed relationships
    local_global_diff = X13
    local_global_sum = X14
    local_global_product = X15
    
    # Basic speed differences and relationships
    incoming_outgoing_diff = X16
    incoming_local_diff = X17
    outgoing_local_diff = X18
    incoming_outgoing_sum = X19
    incoming_local_sum = X20
    outgoing_local_sum = X21
    incoming_outgoing_product = X22
    incoming_local_product = X23
    outgoing_local_product = X24
    
    # Length-normalized speed features
    incoming_speed_per_meter = X25
    outgoing_speed_per_meter = X26
    local_speed_per_meter = X27
    global_speed_per_meter = X28
    
    # Differences for length-normalized speeds
    incoming_outgoing_speed_per_meter_diff = X29
    local_global_speed_per_meter_diff = X30
    incoming_local_speed_per_meter_diff = X31
    outgoing_local_speed_per_meter_diff = X32
    
    # Sum of length-normalized speeds
    incoming_outgoing_speed_per_meter_sum = X33
    local_global_speed_per_meter_sum = X34
    all_speed_per_meter_sum = X35
    
    # Products of length-normalized speeds
    incoming_outgoing_speed_per_meter_product = X36
    local_global_speed_per_meter_product = X37
    
    # Travel time estimates
    incoming_travel_time = X38
    outgoing_travel_time = X39
    total_travel_time = X40
    
    # Travel time differences and relationships
    incoming_outgoing_travel_time_diff = X41
    incoming_total_travel_time_ratio = X42
    outgoing_total_travel_time_ratio = X43
    travel_time_product = X44
    
    # Combined speed-time metrics
    speed_time_balance = X45
    speed_time_efficiency = X46
    
    # Phase-specific features
    phase_is_straight = X47
    phase_is_left = X48
    phase_is_east_west = X49
    phase_is_north_south = X50
    
    # Calculate predicted improvement using the formula
    return {variable_formula}
"""
    
    # Add the select_best_phase function with segment length parameters
    select_best_phase_function = """
def select_best_phase(current_phase, avg_incoming_speed, avg_outgoing_speed, local_avg_speed, global_avg_speed, num_vehicles=None, incoming_length=200, outgoing_length=200):
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
    num_vehicles : int, optional
        Number of vehicles in the simulation (not used in HERE API-like scenario)
    incoming_length : float, optional
        Length of incoming road segment in meters (default 200)
    outgoing_length : float, optional
        Length of outgoing road segment in meters (default 200)
        
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
            avg_outgoing_speed, local_avg_speed, global_avg_speed, 
            num_vehicles, incoming_length, outgoing_length
        )
        
        # Apply a penalty for transitions to account for the transition phase cost
        if transition_needed:
            improvement -= 0.3  # Reduced penalty to encourage more exploration
        
        if improvement > best_improvement:
            best_improvement = improvement
            best_phase = phase
    
    return best_phase
"""
    
    # Combine all code
    full_code = "import numpy as np\n\n" + helper_functions + "\n" + feature_setup + "\n" + select_best_phase_function
    
    # Save the formula code to a file
    formula_file = os.path.join(OUTPUT_DIR, "phase_selection_formula.py")
    with open(formula_file, 'w') as f:
        f.write(full_code)
    
    print(f"Phase selection formula saved to {formula_file}")
    
    return full_code

def main():
    """Main execution function"""
    print("=" * 80)
    print("IMPROVED SYMBOLIC REGRESSION FOR TRAFFIC SIGNAL FORMULA DISCOVERY")
    print("=" * 80)
    
    # Step 1: Load and preprocess data
    data = load_data()
    
    # Step 2: Prepare features with engineering
    X_train, X_test, y_train, y_test, feature_names = prepare_features(data)
    
    # Step 3: Run symbolic regression
    best_formula, rmse, r2 = run_symbolic_regression(X_train, X_test, y_train, y_test, feature_names)
    
    # Step 4: Create phase selection formula
    phase_formula = create_phase_selection_formula(best_formula, feature_names)
    
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