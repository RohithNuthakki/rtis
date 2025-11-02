#!/usr/bin/env python
# Helper script to run a single simulation in its own process

import os
import sys
import json
import pandas as pd
import traci
import sumolib
import subprocess
import time

# Run a single simulation with the parameters provided via command line
def run_single_sim():
    # Parse arguments
    sim_type = sys.argv[1]  # "baseline" or "formula"
    episode = int(sys.argv[2])
    
    if sim_type == "baseline":
        strategy_id = int(sys.argv[3])
        select_best_phase = None
    else:  # formula
        strategy_id = None
        # Import the formula module
        formula_file = sys.argv[3]
        import importlib.util
        spec = importlib.util.spec_from_file_location("phase_formula", formula_file)
        formula_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(formula_module)
        select_best_phase = formula_module.select_best_phase
    
    # Load the rest of the parameters from environment variables
    BASE_DIR = os.environ.get("BASE_DIR")
    SIM_DURATION = int(os.environ.get("SIM_DURATION"))
    DECISION_INTERVAL = int(os.environ.get("DECISION_INTERVAL"))
    LOG_INTERVAL = int(os.environ.get("LOG_INTERVAL"))
    SUMO_GUI = os.environ.get("SUMO_GUI") == "True"
    
    # Create a unique port based on process ID to avoid conflicts
    port = 10000 + os.getpid() % 40000
    
    # Define SUMO command
    if SUMO_GUI:
        sumo_binary = "sumo-gui"
        sumo_cmd = [sumo_binary, 
                   "-c", os.path.join(BASE_DIR, "grid.sumocfg"),
                   "--start", "true",
                   "--quit-on-end", "true",
                   "--no-warnings", "true",
                   "--remote-port", str(port)]
    else:
        sumo_binary = "sumo"
        sumo_cmd = [sumo_binary, 
                   "-c", os.path.join(BASE_DIR, "grid.sumocfg"),
                   "--no-warnings", "true",
                   "--no-step-log", "true",
                   "--remote-port", str(port)]
    
    # Run the simulation
    sumo_process = subprocess.Popen(sumo_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for SUMO to initialize
    time.sleep(1.0)
    
    # Try to connect to SUMO via TraCI
    max_retries = 3
    retry_delay = 1.0
    connected = False
    
    for retry in range(max_retries):
        try:
            traci.init(port=port)
            connected = True
            break
        except Exception as e:
            print(f"Retrying in {retry_delay} seconds. Error: {e}")
            time.sleep(retry_delay)
            if sumo_process.poll() is not None:
                stdout, stderr = sumo_process.communicate()
                error_msg = stderr.decode('utf-8')
                if error_msg:
                    print(f"SUMO error: {error_msg}")
                sumo_process = subprocess.Popen(sumo_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                time.sleep(1.0)
    
    if not connected:
        print(f"Failed to connect to SUMO after {max_retries} attempts")
        if sumo_process and sumo_process.poll() is None:
            sumo_process.terminate()
        sys.exit(1)
    
    # Rest of simulation logic from your original functions...
    # (omitted for brevity - you should copy the core logic from your simulation functions)
    
    # Close TraCI and terminate SUMO
    traci.close()
    if sumo_process and sumo_process.poll() is None:
        sumo_process.terminate()
    
    # Save data to output file
    output_file = sys.argv[4]
    pd.DataFrame(step_data).to_csv(output_file, index=False)
    
if __name__ == "__main__":
    run_single_sim()