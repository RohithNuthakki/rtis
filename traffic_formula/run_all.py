#!/usr/bin/env python
# Main script to run the entire traffic signal optimization workflow

import os
import time
import subprocess
import argparse
import sys

# Set specific directory for this project
PROJECT_DIR = "/Users/samnu/Documents/SUMO/traffic_formula"
BASE_DIR = os.path.join(PROJECT_DIR, "sumo_grid")

def check_requirements():
    """Check if all required tools are installed"""
    try:
        # Check for SUMO
        sumo_home = os.environ.get("SUMO_HOME")
        if not sumo_home:
            # Try common Mac locations
            possible_paths = [
                "/opt/homebrew/opt/sumo/share/sumo",
                "/Applications/SUMO.app/Contents/Home",
                "/usr/local/opt/sumo/share/sumo"
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    os.environ["SUMO_HOME"] = path
                    print(f"SUMO_HOME not found in environment, setting to {path}")
                    break
            else:
                print("Could not find SUMO installation. Please set SUMO_HOME manually.")
                return False
        
        # Check for other Python libraries
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.model_selection import train_test_split
        try:
            import gplearn
        except ImportError:
            print("Warning: gplearn not found. Symbolic regression will not work.")
            print("You can install it with: pip install gplearn")
            return False
        
        print("All required dependencies found!")
        return True
    
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("\nPlease ensure you have the following installed:")
        print("1. SUMO (with SUMO_HOME set in environment)")
        print("2. Python packages: pandas, numpy, matplotlib, scikit-learn, gplearn")
        return False

def run_script(script_name, desc):
    """Run a Python script and handle errors"""
    print(f"\n{'-'*40}")
    print(f"Running: {desc}")
    print(f"{'-'*40}")
    
    script_path = os.path.join(PROJECT_DIR, script_name)
    
    try:
        result = subprocess.run([sys.executable, script_path], check=True)
        if result.returncode == 0:
            print(f"{desc} completed successfully!")
            return True
        else:
            print(f"Error running {desc}. Return code: {result.returncode}")
            return False
    except subprocess.CalledProcessError as e:
        print(f"Error running {desc}: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error running {desc}: {e}")
        return False

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Traffic Signal Optimization Workflow')
    parser.add_argument('--skip-network', action='store_true', help='Skip network generation (use existing files)')
    parser.add_argument('--skip-simulation', action='store_true', help='Skip simulation (use existing data)')
    parser.add_argument('--skip-regression', action='store_true', help='Skip formula discovery (use existing formula)')
    parser.add_argument('--skip-validation', action='store_true', help='Skip formula validation')
    
    return parser.parse_args()

def main():
    """Main execution function"""
    start_time = time.time()
    
    print("=" * 80)
    print("TRAFFIC SIGNAL OPTIMIZATION WORKFLOW")
    print("=" * 80)
    
    # Parse arguments
    args = parse_arguments()
    
    # Check requirements
    if not check_requirements():
        return
    
    print(f"SUMO_HOME is set to: {os.environ.get('SUMO_HOME')}")
    
    # Step 1: Generate network (if not skipped)
    if not args.skip_network:
        if not run_script("grid_generator.py", "Network Generator"):
            return
        
        if not run_script("demand_generator.py", "Traffic Demand Generator"):
            return
    
    # Step 2: Generate synthetic data (instead of running simulations)
    if not args.skip_simulation:
        if not run_script("generate_synthetic_data.py", "Synthetic Data Generator"):
            return
    
    # Step 3: Run symbolic regression (if not skipped)
    if not args.skip_regression:
        if not run_script("symbolic_regression.py", "Symbolic Regression"):
            return
    
    # Step 4: Validate the formula (if not skipped)
    if not args.skip_validation:
        if not run_script("formula_validator.py", "Formula Validator"):
            return
    
    # Calculate total execution time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("\n" + "=" * 80)
    print(f"WORKFLOW COMPLETED in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print("=" * 80)
    
    print("\nResults can be found in:")
    print(f"  - Network files: {BASE_DIR}")
    print(f"  - Simulation data: {os.path.join(BASE_DIR, 'results')}")
    print(f"  - Discovered formula: {os.path.join(BASE_DIR, 'formulas')}")
    print(f"  - Validation results: {os.path.join(BASE_DIR, 'validation')}")

if __name__ == "__main__":
    main()
