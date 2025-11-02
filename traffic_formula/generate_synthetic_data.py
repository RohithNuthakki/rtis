#!/usr/bin/env python
# Generate enhanced synthetic simulation data with more variability

import os
import pandas as pd
import numpy as np
import random

# Configuration
BASE_DIR = "/Users/samnu/Documents/SUMO/traffic_formula/sumo_grid"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
NUM_INTERSECTIONS = 25  # 5x5 grid
SIM_DURATION = 1800  # 30 minutes
DECISION_INTERVAL = 15  # seconds
NUM_STRATEGIES = 4
NUM_EPISODES = 10  # Increased from 2 to 10

# Traffic pattern configurations
TRAFFIC_PATTERNS = {
    "morning_rush": {
        "base_speed_range": (3, 10),
        "strategy_factors": [1.3, 1.0, 0.7, 0.6],  # East-West priority
        "time_decay": 0.4,  # Traffic gets worse faster
        "vehicle_factor": 1.5  # More vehicles
    },
    "evening_rush": {
        "base_speed_range": (3, 10),
        "strategy_factors": [1.0, 1.2, 0.7, 0.8],  # North-South priority
        "time_decay": 0.4,
        "vehicle_factor": 1.5
    },
    "off_peak": {
        "base_speed_range": (8, 18),  # Higher speeds
        "strategy_factors": [1.1, 1.1, 1.0, 0.9],  # More balanced
        "time_decay": 0.2,  # Traffic gets worse slower
        "vehicle_factor": 0.8  # Fewer vehicles
    },
    "weekend": {
        "base_speed_range": (7, 16),
        "strategy_factors": [1.1, 1.1, 0.9, 0.9],  # Balanced
        "time_decay": 0.2,
        "vehicle_factor": 0.7
    },
    "night": {
        "base_speed_range": (10, 20),  # Highest speeds
        "strategy_factors": [1.2, 1.2, 1.1, 1.1],  # All strategies work well
        "time_decay": 0.1,  # Traffic remains steady
        "vehicle_factor": 0.5  # Fewest vehicles
    }
}

# Ensure output directory exists
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

def generate_simulation_data():
    """Generate synthetic simulation data with more variability"""
    print("Generating enhanced synthetic simulation data...")
    
    all_data = []
    
    # Generate data for each episode and strategy
    for episode in range(NUM_EPISODES):
        # Select a random traffic pattern for this episode
        pattern_name = random.choice(list(TRAFFIC_PATTERNS.keys()))
        pattern = TRAFFIC_PATTERNS[pattern_name]
        print(f"Generating data for episode {episode}, traffic pattern: {pattern_name}")
        
        for strategy in range(1, NUM_STRATEGIES + 1):
            print(f"  Strategy {strategy}...")
            
            # Base parameters that influence speeds
            base_incoming_speed = random.uniform(*pattern["base_speed_range"])
            base_outgoing_speed = random.uniform(*pattern["base_speed_range"])
            
            # Strategy-specific factors from the traffic pattern
            strategy_factor = pattern["strategy_factors"][strategy-1]
            
            # Generate data for each decision point
            for step in range(0, SIM_DURATION, DECISION_INTERVAL):
                # Time-based factors (traffic increases over time)
                time_factor = 1.0 - (step / SIM_DURATION) * pattern["time_decay"]
                
                # Add some randomness to time factor to simulate unpredictable traffic fluctuations
                time_factor *= random.uniform(0.9, 1.1)
                
                # Global metrics
                # Number of vehicles scales with the traffic pattern and increases over time
                num_vehicles = int((100 + (step / DECISION_INTERVAL) * 2) * pattern["vehicle_factor"])
                
                # Global average speed based on strategy, time, and traffic pattern
                global_avg_speed = max(5, 12 * strategy_factor * time_factor + random.uniform(-1.5, 1.5))
                
                # Add global record
                all_data.append({
                    'episode': episode,
                    'strategy': strategy,
                    'step': step,
                    'tls_id': 'global',
                    'chosen_phase': None,
                    'transition_needed': None,
                    'avg_incoming_speed': None,
                    'avg_outgoing_speed': None,
                    'local_avg_speed': None,
                    'global_avg_speed': global_avg_speed,
                    'num_vehicles': num_vehicles,
                    'traffic_pattern': pattern_name
                })
                
                # Generate data for each intersection
                for tls_id in range(NUM_INTERSECTIONS):
                    # Determine if transition is needed (random)
                    transition_needed = random.choice([True, False])
                    
                    # Calculate speeds based on various factors
                    intersection_factor = random.uniform(0.8, 1.2)  # Some intersections are busier
                    
                    # More variability in speed calculations
                    incoming_speed = max(2, base_incoming_speed * strategy_factor * time_factor * intersection_factor + random.uniform(-3, 3))
                    outgoing_speed = max(2, base_outgoing_speed * strategy_factor * time_factor * intersection_factor + random.uniform(-3, 3))
                    
                    # Create different patterns for incoming vs outgoing speeds
                    if random.random() < 0.3:  # 30% chance of congestion scenario
                        if random.random() < 0.5:  # Incoming congestion
                            incoming_speed = max(1, incoming_speed * random.uniform(0.5, 0.8))
                        else:  # Outgoing congestion
                            outgoing_speed = max(1, outgoing_speed * random.uniform(0.5, 0.8))
                    
                    local_speed = (incoming_speed + outgoing_speed) / 2
                    
                    # Add intersection record
                    all_data.append({
                        'episode': episode,
                        'strategy': strategy,
                        'step': step,
                        'tls_id': str(tls_id),
                        'chosen_phase': strategy + 1,  # Phase 2-5 based on strategy 1-4
                        'transition_needed': transition_needed,
                        'avg_incoming_speed': incoming_speed,
                        'avg_outgoing_speed': outgoing_speed,
                        'local_avg_speed': local_speed,
                        'global_avg_speed': global_avg_speed,
                        'num_vehicles': num_vehicles,
                        'traffic_pattern': pattern_name
                    })
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    # Save to file
    output_file = os.path.join(RESULTS_DIR, "all_episodes.csv")
    df.to_csv(output_file, index=False)
    print(f"Enhanced synthetic data generated and saved to {output_file}")
    print(f"Total data points: {len(df)}")
    print(f"Episodes: {NUM_EPISODES}, Strategies: {NUM_STRATEGIES}, Intersections: {NUM_INTERSECTIONS}")
    
    # Generate statistics
    episode_stats = df[df['tls_id'] == 'global'].groupby(['episode', 'traffic_pattern'])['global_avg_speed'].mean().reset_index()
    print("\nEpisode statistics:")
    print(episode_stats)
    
    return df

if __name__ == "__main__":
    data = generate_simulation_data()
    print("Done!")