#!/usr/bin/env python
# Comprehensive validation framework for traffic signal control formulas

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import importlib.util
import time
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score

# Configuration
BASE_DIR = "/Users/samnu/Documents/SUMO/traffic_formula/sumo_grid"
FORMULA_DIR = os.path.join(BASE_DIR, "formulas")
# Change this to point to the correct directory
SIMPLE_FORMULA_DIR = os.path.join(BASE_DIR, "formulas_pure_speeds")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
VALIDATION_DIR = os.path.join(BASE_DIR, "validation_comprehensive")

# Ensure output directory exists
if not os.path.exists(VALIDATION_DIR):
    os.makedirs(VALIDATION_DIR)

# Set visualization style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

def load_formula(formula_dir, module_name):
    """Load a phase selection formula module"""
    formula_file = os.path.join(formula_dir, f"{module_name}.py")
    
    if not os.path.exists(formula_file):
        raise FileNotFoundError(f"Formula file not found: {formula_file}")
    
    # Load the module
    spec = importlib.util.spec_from_file_location(module_name, formula_file)
    formula_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(formula_module)
    
    return formula_module

def fixed_cycle_phase(current_phase, step, cycle_length=30, transition_time=5):
    """
    Baseline fixed-cycle phase selection
    """
    # Total cycle is 4 phases * cycle_length + 4 transitions * transition_time
    total_cycle_time = 4 * cycle_length + 4 * transition_time
    
    # Position in the cycle
    cycle_position = step % total_cycle_time
    
    # Determine phase based on position in cycle
    if cycle_position < cycle_length:
        return 2  # First phase
    elif cycle_position < cycle_length + transition_time:
        return 1  # Transition phase
    elif cycle_position < 2 * cycle_length + transition_time:
        return 3  # Second phase
    elif cycle_position < 2 * cycle_length + 2 * transition_time:
        return 1  # Transition phase
    elif cycle_position < 3 * cycle_length + 2 * transition_time:
        return 4  # Third phase
    elif cycle_position < 3 * cycle_length + 3 * transition_time:
        return 1  # Transition phase
    elif cycle_position < 4 * cycle_length + 3 * transition_time:
        return 5  # Fourth phase
    else:
        return 1  # Transition phase back to first

def random_phase(current_phase):
    """
    Random phase selection as another baseline
    """
    if np.random.random() < 0.5:
        return current_phase
    else:
        possible_phases = [p for p in range(2, 6) if p != current_phase]
        return np.random.choice(possible_phases)

def run_comprehensive_validation():
    """Run comprehensive validation of traffic signal formulas"""
    print("Loading simulation data...")
    data_file = os.path.join(RESULTS_DIR, "all_episodes.csv")
    
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    data = pd.read_csv(data_file)
    print(f"Loaded {len(data)} records from {data['episode'].nunique()} episodes")
    
    # Filter to only include intersection data (not global data)
    intersection_data = data[data['tls_id'] != 'global'].copy()
    
    # Convert transition_needed from string to int if needed
    if intersection_data['transition_needed'].dtype == 'bool':
        intersection_data['transition_needed'] = intersection_data['transition_needed'].astype(int)
    elif intersection_data['transition_needed'].dtype == 'object':
        intersection_data['transition_needed'] = (intersection_data['transition_needed'] == 'True').astype(int)
    
    # Calculate actual speed changes after each decision
    print("Calculating actual speed changes...")
    intersection_data['next_local_speed'] = intersection_data.groupby(['episode', 'strategy', 'tls_id'])['local_avg_speed'].shift(-1)
    intersection_data['actual_speed_improvement'] = intersection_data['next_local_speed'] - intersection_data['local_avg_speed']
    
    # Calculate additional metrics
    # Relative speed improvement
    intersection_data['relative_speed_improvement'] = (intersection_data['actual_speed_improvement'] / 
                                                      intersection_data['local_avg_speed'].replace(0, 0.001)) * 100
    
    # Calculate global metrics per episode and step
    global_metrics = data[data['tls_id'] == 'global'].copy()
    global_metrics = global_metrics[['episode', 'strategy', 'step', 'global_avg_speed', 'num_vehicles']]
    
    # Merge global metrics with intersection data
    merged_data = pd.merge(intersection_data, global_metrics, 
                          on=['episode', 'strategy', 'step'], 
                          suffixes=('', '_global'))
    
    # Drop rows with NaN values (last steps won't have next values)
    valid_data = merged_data.dropna(subset=['next_local_speed']).copy()
    print(f"Valid data points: {len(valid_data)}")
    
    # Load formulas
    print("Loading signal control formulas...")
    try:
        # Load complex generated formula
        generated_formula = load_formula(FORMULA_DIR, "phase_selection_formula")
        has_generated_formula = True
    except FileNotFoundError:
        print("Warning: Generated formula not found. Skipping.")
        has_generated_formula = False
    
    try:
        # Load simplified formula - change the module name to match your file
        simplified_formula = load_formula(SIMPLE_FORMULA_DIR, "pure_speed_phase_selection_formula")
        has_simplified_formula = True
    except FileNotFoundError:
        print("Warning: Simplified formula not found. Skipping.")
        has_simplified_formula = False
    
    # Calculate global average speed for validation dataset
    print("\nCalculating global average speeds for validation dataset...")
    # Group by episode, strategy, and step to get all intersections at the same time point
    groups = valid_data.groupby(['episode', 'strategy', 'step'])
    
    # Calculate global average speed for each group (average of all incoming and outgoing speeds)
    global_speeds = []
    
    for name, group in groups:
        episode, strategy, step = name
        # Calculate global avg speed as average of all incoming and outgoing speeds at this time point
        calculated_global_avg_speed = (group['avg_incoming_speed'].mean() + group['avg_outgoing_speed'].mean()) / 2
        
        # Add to each row in this group
        for idx in group.index:
            global_speeds.append({
                'index': idx,
                'calculated_global_avg_speed': calculated_global_avg_speed
            })
    
    # Convert to DataFrame and merge back with the original data
    global_speed_df = pd.DataFrame(global_speeds)
    global_speed_df.set_index('index', inplace=True)
    
    # Join the calculated global speeds back to the valid data
    valid_data = valid_data.join(global_speed_df)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Fixed road segment lengths for consistency
    incoming_length = 200
    outgoing_length = 200
    
    # Apply each formula/strategy to the data
    print("Applying control strategies to predict phases and improvements...")
    
    # Initialize columns for each strategy
    valid_data['phase_fixed_cycle'] = np.nan
    valid_data['phase_random'] = np.nan
    valid_data['phase_max_pressure'] = np.nan
    
    if has_generated_formula:
        valid_data['phase_generated_formula'] = np.nan
        valid_data['improvement_generated_formula'] = np.nan
    
    if has_simplified_formula:
        valid_data['phase_simplified_formula'] = np.nan
    
    # Process in batches to show progress
    total_rows = len(valid_data)
    batch_size = 1000
    num_batches = total_rows // batch_size + (1 if total_rows % batch_size > 0 else 0)
    
    start_time = time.time()
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_rows)
        batch = valid_data.iloc[start_idx:end_idx]
        
        # Apply strategies to each row in the batch
        for idx, row in batch.iterrows():
            try:
                # 1. Fixed cycle baseline
                valid_data.at[idx, 'phase_fixed_cycle'] = fixed_cycle_phase(
                    row['chosen_phase'], row['step'])
                
                # 2. Random baseline
                valid_data.at[idx, 'phase_random'] = random_phase(row['chosen_phase'])
                
                # 3. Max Pressure formula
                if has_simplified_formula:
                    valid_data.at[idx, 'phase_max_pressure'] = simplified_formula.select_max_pressure_phase(
                        row['chosen_phase'],
                        row['avg_incoming_speed'],
                        row['avg_outgoing_speed'],
                        row['local_avg_speed'],
                        row['calculated_global_avg_speed'],  # Using calculated global avg speed
                        row['num_vehicles'],
                        incoming_length,
                        outgoing_length
                    )
                
                # 4. Generated formula (complex)
                if has_generated_formula:
                    # Predicted improvement
                    valid_data.at[idx, 'improvement_generated_formula'] = generated_formula.predict_speed_improvement(
                        row['chosen_phase'],
                        row['transition_needed'],
                        row['avg_incoming_speed'],
                        row['avg_outgoing_speed'],
                        row['local_avg_speed'],
                        row['calculated_global_avg_speed'],
                        row['num_vehicles'],
                        incoming_length,
                        outgoing_length
                    )
                    
                    # Recommended phase
                    valid_data.at[idx, 'phase_generated_formula'] = generated_formula.select_best_phase(
                        row['chosen_phase'],
                        row['avg_incoming_speed'],
                        row['avg_outgoing_speed'],
                        row['local_avg_speed'],
                        row['global_avg_speed'],
                        row['num_vehicles'],
                        incoming_length,
                        outgoing_length
                    )
                
                # 5. Simplified formula
                if has_simplified_formula:
                    valid_data.at[idx, 'phase_simplified_formula'] = simplified_formula.select_best_phase(
                        row['chosen_phase'],
                        row['avg_incoming_speed'],
                        row['avg_outgoing_speed'],
                        row['local_avg_speed'],
                        row['calculated_global_avg_speed'],  # Using calculated global avg speed
                        row['num_vehicles'],
                        incoming_length,
                        outgoing_length
                    )
                
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                print(f"Data: {row[['avg_incoming_speed', 'avg_outgoing_speed', 'local_avg_speed', 'global_avg_speed', 'calculated_global_avg_speed']]}")
        
        elapsed = time.time() - start_time
        progress = (i + 1) / num_batches * 100
        print(f"Progress: {progress:.1f}% ({i+1}/{num_batches} batches) - Time elapsed: {elapsed:.1f}s")
    
    # Create match flags for each strategy
    valid_data['match_original'] = True  # Always matches by definition
    valid_data['match_fixed_cycle'] = valid_data['chosen_phase'] == valid_data['phase_fixed_cycle']
    valid_data['match_random'] = valid_data['chosen_phase'] == valid_data['phase_random']
    valid_data['match_max_pressure'] = valid_data['chosen_phase'] == valid_data['phase_max_pressure']
    
    if has_generated_formula:
        valid_data['match_generated_formula'] = valid_data['chosen_phase'] == valid_data['phase_generated_formula']
    
    if has_simplified_formula:
        valid_data['match_simplified_formula'] = valid_data['chosen_phase'] == valid_data['phase_simplified_formula']
    
    # Statistical analysis of results
    print("\nPerforming statistical analysis...")
    
    # Create summary statistics dataframe
    strategy_names = ['Original (Baseline)', 'Fixed Cycle', 'Random Selection', 'Max Pressure']
    match_columns = ['match_original', 'match_fixed_cycle', 'match_random', 'match_max_pressure']
    
    if has_generated_formula:
        strategy_names.append('Generated Formula')
        match_columns.append('match_generated_formula')
    
    if has_simplified_formula:
        strategy_names.append('Simplified Formula')
        match_columns.append('match_simplified_formula')
    
    # Function to calculate performance statistics for a strategy
    def calculate_performance_stats(data, match_column):
        from scipy import stats as scipy_stats  # Import inside function to avoid scope issues
        
        matches = data[data[match_column]]['actual_speed_improvement']
        non_matches = data[~data[match_column]]['actual_speed_improvement'] if match_column != 'match_original' else None
        
        stats_dict = {
            'count': len(matches),
            'mean_improvement': matches.mean(),
            'median_improvement': matches.median(),
            'std_improvement': matches.std(),
            'min_improvement': matches.min(),
            'max_improvement': matches.max(),
        }
        
        # Add non-match stats if applicable
        if non_matches is not None and len(non_matches) > 0:
            stats_dict.update({
                'non_match_count': len(non_matches),
                'non_match_mean': non_matches.mean(),
                'non_match_std': non_matches.std(),
                # T-test for statistical significance
                't_stat': scipy_stats.ttest_ind(matches, non_matches, equal_var=False)[0] if len(matches) > 1 and len(non_matches) > 1 else np.nan,
                'p_value': scipy_stats.ttest_ind(matches, non_matches, equal_var=False)[1] if len(matches) > 1 and len(non_matches) > 1 else np.nan,
            })
        
        return stats_dict

    # Calculate stats for each strategy
    performance_stats = {}
    for name, column in zip(strategy_names, match_columns):
        performance_stats[name] = calculate_performance_stats(valid_data, column)
    
    
    # Create summary dataframe
    summary_df = pd.DataFrame({
        'Strategy': strategy_names,
        'Count': [performance_stats[name]['count'] for name in strategy_names],
        'Mean Improvement': [performance_stats[name]['mean_improvement'] for name in strategy_names],
        'Std Dev': [performance_stats[name]['std_improvement'] for name in strategy_names],
    })
    
    # Add significance testing where applicable
    for name in strategy_names[1:]:  # Skip the original strategy
        if 'p_value' in performance_stats[name]:
            summary_df.loc[summary_df['Strategy'] == name, 'p-value'] = performance_stats[name]['p_value']
            summary_df.loc[summary_df['Strategy'] == name, 'Significant'] = performance_stats[name]['p_value'] < 0.05
    
    print("\nPerformance Summary:")
    print(summary_df)
    
    # Calculate RMSE and R² for the generated formula's predictions
    if has_generated_formula and 'improvement_generated_formula' in valid_data.columns:
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(valid_data['actual_speed_improvement'], 
                                         valid_data['improvement_generated_formula']))
        
        # Calculate R²
        r2 = r2_score(valid_data['actual_speed_improvement'], 
                      valid_data['improvement_generated_formula'])
        
        print(f"\nGenerated Formula Prediction Performance:")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²: {r2:.4f}")
    
    # Create visualizations
    print("\nCreating comprehensive visualizations...")
    
    # 1. Bar chart of average speed improvement by strategy
    # 1. Bar chart of average speed improvement by strategy
    plt.figure(figsize=(14, 8))

    # Use matplotlib's native bar plot with clear column references
    plt.bar(range(len(summary_df)), summary_df['Mean Improvement'], 
            yerr=summary_df['Std Dev'] / np.sqrt(summary_df['Count']),
            capsize=4, color='steelblue', alpha=0.8)

    # Set x-tick labels
    plt.xticks(range(len(summary_df)), summary_df['Strategy'], rotation=45, ha='right')

    # Add value labels
    for i, v in enumerate(summary_df['Mean Improvement']):
        plt.text(i, v + 0.01, f"{v:.3f}", ha='center')

    plt.title('Average Speed Improvement by Control Strategy')
    plt.xlabel('Strategy')
    plt.ylabel('Speed Improvement (m/s)')
    plt.tight_layout()
    plt.savefig(os.path.join(VALIDATION_DIR, 'strategy_performance_comparison.png'))
    
    # 2. Distribution of speed improvements by strategy
    plt.figure(figsize=(16, 10))
    
    # Create subplot for each strategy
    for i, (name, column) in enumerate(zip(strategy_names, match_columns)):
        plt.subplot(2, 3, i+1)
        data = valid_data[valid_data[column]]['actual_speed_improvement']
        sns.histplot(data, kde=True, bins=30)
        plt.axvline(data.mean(), color='red', linestyle='--', label=f'Mean: {data.mean():.3f}')
        plt.title(f'{name} (n={len(data)})')
        plt.xlabel('Speed Improvement (m/s)')
        plt.ylabel('Frequency')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(VALIDATION_DIR, 'improvement_distributions.png'))
    
    # 3. Actual vs. Predicted improvement scatter plot (for generated formula)
    if has_generated_formula and 'improvement_generated_formula' in valid_data.columns:
        plt.figure(figsize=(12, 8))
        
        # Create a sample if dataset is too large
        if len(valid_data) > 5000:
            plot_data = valid_data.sample(5000, random_state=42)
        else:
            plot_data = valid_data
        
        # Plot scatter with regression line
        sns.regplot(x='actual_speed_improvement', y='improvement_generated_formula', 
                   data=plot_data, scatter_kws={'alpha': 0.3}, line_kws={'color': 'red'})
        
        # Add perfect prediction reference line
        min_val = min(plot_data['actual_speed_improvement'].min(), 
                      plot_data['improvement_generated_formula'].min())
        max_val = max(plot_data['actual_speed_improvement'].max(), 
                      plot_data['improvement_generated_formula'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Prediction')
        
        plt.title(f'Actual vs. Predicted Improvement (RMSE: {rmse:.3f}, R²: {r2:.3f})')
        plt.xlabel('Actual Speed Improvement (m/s)')
        plt.ylabel('Predicted Speed Improvement (m/s)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(VALIDATION_DIR, 'actual_vs_predicted.png'))
    
    # 4. Performance by traffic pattern
    if 'traffic_pattern' in valid_data.columns:
        # Calculate mean improvement by traffic pattern and strategy
        pattern_stats = []
        
        for pattern in valid_data['traffic_pattern'].unique():
            pattern_data = valid_data[valid_data['traffic_pattern'] == pattern]
            
            for name, column in zip(strategy_names, match_columns):
                stats = {
                    'Traffic Pattern': pattern,
                    'Strategy': name,
                    'Count': len(pattern_data[pattern_data[column]]),
                    'Mean Improvement': pattern_data[pattern_data[column]]['actual_speed_improvement'].mean(),
                    'Std Dev': pattern_data[pattern_data[column]]['actual_speed_improvement'].std(),
                }
                pattern_stats.append(stats)
        
        pattern_df = pd.DataFrame(pattern_stats)
        
        # Plot grouped bar chart
        plt.figure(figsize=(16, 10))
        sns.barplot(x='Traffic Pattern', y='Mean Improvement', hue='Strategy', 
                   data=pattern_df, palette='viridis')
        plt.title('Strategy Performance by Traffic Pattern')
        plt.xlabel('Traffic Pattern')
        plt.ylabel('Mean Speed Improvement (m/s)')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(VALIDATION_DIR, 'performance_by_traffic_pattern.png'))
    
    # 5. Phase selection distribution by strategy
    plt.figure(figsize=(14, 8))
    
    # Prepare data for plotting
    phase_columns = ['chosen_phase', 'phase_fixed_cycle', 'phase_random', 'phase_max_pressure']
    phase_labels = ['Original', 'Fixed Cycle', 'Random', 'Max Pressure']
    
    if has_generated_formula:
        phase_columns.append('phase_generated_formula')
        phase_labels.append('Generated Formula')
    
    if has_simplified_formula:
        phase_columns.append('phase_simplified_formula')
        phase_labels.append('Simplified Formula')
    
    # Calculate phase distribution
    phase_dist = []
    for col, label in zip(phase_columns, phase_labels):
        for phase in range(2, 6):  # Phases 2-5
            count = (valid_data[col] == phase).sum()
            percentage = count / len(valid_data) * 100
            phase_dist.append({
                'Strategy': label,
                'Phase': phase,
                'Percentage': percentage
            })
    
    phase_dist_df = pd.DataFrame(phase_dist)
    
    # Plot grouped bar chart
    sns.barplot(x='Strategy', y='Percentage', hue='Phase', data=phase_dist_df, palette='Set2')
    plt.title('Phase Selection Distribution by Strategy')
    plt.xlabel('Strategy')
    plt.ylabel('Percentage Selected (%)')
    plt.legend(title='Phase')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(VALIDATION_DIR, 'phase_distribution.png'))
    
    # Save results to CSV
    output_file = os.path.join(VALIDATION_DIR, 'comprehensive_validation_results.csv')
    valid_data.to_csv(output_file, index=False)
    print(f"Validation results saved to: {output_file}")
    
    # Save summary statistics
    summary_file = os.path.join(VALIDATION_DIR, 'performance_summary.csv')
    summary_df.to_csv(summary_file, index=False)
    print(f"Performance summary saved to: {summary_file}")
    
    # If traffic pattern stats are available, save them too
    if 'traffic_pattern' in valid_data.columns:
        pattern_file = os.path.join(VALIDATION_DIR, 'pattern_performance.csv')
        pattern_df.to_csv(pattern_file, index=False)
        print(f"Traffic pattern performance saved to: {pattern_file}")
    
    # Create detailed validation report
    report_file = os.path.join(VALIDATION_DIR, 'validation_report.txt')
    with open(report_file, 'w') as f:
        f.write("COMPREHENSIVE TRAFFIC SIGNAL FORMULA VALIDATION REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Data points analyzed: {len(valid_data)}\n")
        f.write(f"From {valid_data['episode'].nunique()} episodes\n")
        f.write(f"Across {valid_data['traffic_pattern'].nunique() if 'traffic_pattern' in valid_data.columns else 'N/A'} traffic patterns\n\n")
        
        f.write("STRATEGY PERFORMANCE SUMMARY\n")
        f.write("-"*30 + "\n")
        f.write(summary_df.to_string() + "\n\n")
        
        if has_generated_formula and 'improvement_generated_formula' in valid_data.columns:
            f.write("PREDICTION ACCURACY (GENERATED FORMULA)\n")
            f.write("-"*30 + "\n")
            f.write(f"RMSE: {rmse:.4f}\n")
            f.write(f"R²: {r2:.4f}\n\n")
        
        if 'traffic_pattern' in valid_data.columns:
            f.write("PERFORMANCE BY TRAFFIC PATTERN\n")
            f.write("-"*30 + "\n")
            for pattern in pattern_df['Traffic Pattern'].unique():
                f.write(f"\n{pattern}:\n")
                pattern_summary = pattern_df[pattern_df['Traffic Pattern'] == pattern]
                f.write(pattern_summary.to_string() + "\n")
    
    print(f"Comprehensive validation report saved to: {report_file}")
    
    return valid_data, summary_df

if __name__ == "__main__":
    print("=" * 80)
    print("COMPREHENSIVE TRAFFIC SIGNAL FORMULA VALIDATION")
    print("=" * 80)
    
    try:
        results, summary = run_comprehensive_validation()
        print("\nComprehensive validation completed successfully!")
    except Exception as e:
        print(f"Error during validation: {e}")
        import traceback
        traceback.print_exc()