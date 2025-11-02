 #!/usr/bin/env python
# Validator for the discovered traffic signal formula

import os
import subprocess
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import importlib.util
from traci.exceptions import TraCIException, FatalTraCIError


# Import traci library for TraCI interface
try:
    import traci
    import sumolib
except ImportError:
    raise ImportError("Please make sure SUMO_HOME is in your PATH and the TraCI library is available")

# Configuration
BASE_DIR = "/Users/samnu/Documents/SUMO/traffic_formula/sumo_grid"
FORMULA_DIR = os.path.join(BASE_DIR, "formulas")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
VALIDATION_DIR = os.path.join(BASE_DIR, "validation")
GRID_SIZE = 5
SIM_DURATION = 1800  # 30 minutes
DECISION_INTERVAL = 15  # seconds
LOG_INTERVAL = 15  # seconds
NUM_VALIDATION_EPISODES = 3
SUMO_GUI = False  # Set to False for faster execution

# Ensure output directory exists
if not os.path.exists(VALIDATION_DIR):
    os.makedirs(VALIDATION_DIR)

def cleanup_all_connections():
    """Force cleanup of all TraCI connections and SUMO processes"""
    # Close any existing TraCI connections
    try:
        # Get all labels from the _connections dictionary
        if hasattr(traci, '_connections'):
            for label in list(traci._connections.keys()):
                try:
                    print(f"Closing connection: {label}")
                    traci.close(label)
                except:
                    pass
        
        # Also try to close the default connection
        try:
            traci.close()
        except:
            pass
    except:
        pass
    
    # Kill any lingering SUMO processes
    try:
        import psutil
        for proc in psutil.process_iter(['pid', 'name']):
            if 'sumo' in proc.info['name'].lower():
                print(f"Killing SUMO process: {proc.info}")
                proc.kill()
    except:
        # If psutil isn't available, try using os commands
        try:
            if os.name == 'posix':  # For Unix/Linux/Mac
                os.system("pkill -f sumo")
            elif os.name == 'nt':   # For Windows
                os.system("taskkill /f /im sumo.exe")
                os.system("taskkill /f /im sumo-gui.exe")
        except:
            pass
    
    # Wait a moment to ensure everything is closed
    time.sleep(2)

def load_formula():
    """Load the phase selection formula"""
    formula_file = os.path.join(FORMULA_DIR, "phase_selection_formula.py")
    
    if not os.path.exists(formula_file):
        raise FileNotFoundError(f"Formula file not found: {formula_file}. Run symbolic regression first.")
    
    # Load the module
    spec = importlib.util.spec_from_file_location("phase_formula", formula_file)
    formula_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(formula_module)
    
    return formula_module.select_best_phase

def get_edge_speeds(traci_conn):
    """Get average speed on all edges"""
    edge_speeds = {}
    edge_ids = traci_conn.edge.getIDList()
    
    for edge_id in edge_ids:
        # Get vehicle count on this edge
        vehicles = traci_conn.edge.getLastStepVehicleIDs(edge_id)
        
        if len(vehicles) > 0:
            # Calculate average speed
            avg_speed = traci_conn.edge.getLastStepMeanSpeed(edge_id)
            edge_speeds[edge_id] = avg_speed
        else:
            # No vehicles, use free flow speed
            edge_speeds[edge_id] = traci_conn.edge.getMaxSpeed(edge_id)
    
    return edge_speeds

def get_intersection_incoming_edges(net, tls_id):
    """Get all incoming edges for a traffic light"""
    incoming_edges = []
    for connection in net.getTLS(tls_id).getConnections():
        incoming_edge = connection[0].getFrom().getID()
        if incoming_edge not in incoming_edges:
            incoming_edges.append(incoming_edge)
    return incoming_edges

def get_intersection_outgoing_edges(net, tls_id):
    """Get all outgoing edges for a traffic light"""
    outgoing_edges = []
    for connection in net.getTLS(tls_id).getConnections():
        outgoing_edge = connection[0].getTo().getID()
        if outgoing_edge not in outgoing_edges:
            outgoing_edges.append(outgoing_edge)
    return outgoing_edges

def run_baseline_simulation(episode, strategy_id):
    """Run a baseline simulation with fixed strategy"""
    # Create a unique port number for this simulation
    import random
    base_port = 9000 + (episode * 500) + (strategy_id * 100)
    port = base_port + random.randint(1, 50)
    
    connection_label = f"ep{episode}_strat{strategy_id}"
    try:
        # Try to close the specific connection if it exists
        traci.close(connection_label)
        print(f"Closed existing connection: {connection_label}")
    except:
        # If the connection doesn't exist, this will throw an exception we can ignore
        pass
    
    # Prepare SUMO command
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
    
    # Start SUMO as a subprocess
    sumo_process = subprocess.Popen(sumo_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for SUMO to initialize
    time.sleep(1.0)
    
    # Try to connect to SUMO via TraCI
    max_retries = 3
    retry_delay = 1.0
    connected = False
    
    for retry in range(max_retries):
        try:
            connection_label = f"ep{episode}_strat{strategy_id}"
            traci.init(port=port, label=connection_label)
            connected = True
            break
        except traci.exceptions.FatalTraCIError:
            print(f"Retrying in {retry_delay} seconds")
            time.sleep(retry_delay)
            # Check if SUMO process is still alive
            if sumo_process.poll() is not None:
                # SUMO process has exited, check error output
                stdout, stderr = sumo_process.communicate()
                error_msg = stderr.decode('utf-8')
                if error_msg:
                    print(f"SUMO error: {error_msg}")
                # Start a new process
                sumo_process = subprocess.Popen(sumo_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                time.sleep(1.0)
    
    if not connected:
        print(f"Failed to connect to SUMO after {max_retries} attempts")
        if sumo_process and sumo_process.poll() is None:
            sumo_process.terminate()
        return None
    
    # Load the network
    net = sumolib.net.readNet(os.path.join(BASE_DIR, "grid.net.xml"))
    
    # Get all traffic light IDs
    tls_ids = traci.trafficlight.getIDList(connection_label)
    
    # Baseline phase selection strategy
    def baseline_strategy(tls_id):
        if strategy_id == 1:
            return 2  # Always phase 2
        elif strategy_id == 2:
            return 3  # Always phase 3
        elif strategy_id == 3:
            return 4  # Always phase 4
        elif strategy_id == 4:
            return 5  # Always phase 5
        else:
            # Cycle through phases: 2 -> 3 -> 4 -> 5 -> 2...
            current_phase = traci.trafficlight.getPhase(tls_id, connection_label)
            return (current_phase % 4) + 2
    
    # Prepare data collection
    step_data = []
    decision_steps = list(range(0, SIM_DURATION, DECISION_INTERVAL))
    last_phases = {tls_id: None for tls_id in tls_ids}
    
    # Run the simulation
    for step in range(SIM_DURATION):
        traci.simulationStep(connection_label)
        
        # Make phase decisions at specified intervals
        if step in decision_steps:
            # Get current speeds for all edges
            edge_speeds = get_edge_speeds(traci)
            
            # Calculate average speed
            if len(edge_speeds) > 0:
                global_avg_speed = sum(edge_speeds.values()) / len(edge_speeds)
            else:
                global_avg_speed = 0
            
            # Record data for each intersection
            for tls_id in tls_ids:
                # Get incoming and outgoing edges for this intersection
                incoming_edges = get_intersection_incoming_edges(net, tls_id)
                outgoing_edges = get_intersection_outgoing_edges(net, tls_id)
                
                # Calculate average speed for incoming and outgoing edges
                incoming_speeds = [edge_speeds.get(edge, 0) for edge in incoming_edges if edge in edge_speeds]
                outgoing_speeds = [edge_speeds.get(edge, 0) for edge in outgoing_edges if edge in edge_speeds]
                
                if incoming_speeds:
                    avg_incoming_speed = sum(incoming_speeds) / len(incoming_speeds)
                else:
                    avg_incoming_speed = 0
                
                if outgoing_speeds:
                    avg_outgoing_speed = sum(outgoing_speeds) / len(outgoing_speeds)
                else:
                    avg_outgoing_speed = 0
                
                # Local average speed (both incoming and outgoing)
                local_avg_speed = 0
                if incoming_speeds or outgoing_speeds:
                    local_avg_speed = sum(incoming_speeds + outgoing_speeds) / (len(incoming_speeds) + len(outgoing_speeds))
                
                # Choose phase based on strategy
                target_phase = baseline_strategy(tls_id)
                
                # Check if we need a transition phase
                if last_phases[tls_id] is not None and target_phase != last_phases[tls_id]:
                    # Need transition (Phase 1)
                    traci.trafficlight.setPhase(tls_id, 0, connection_label)  # Phase 1 index is 0
                    transition_needed = True
                else:
                    # Set the target phase directly
                    # Convert from our phase numbering (1-5) to SUMO's index (0-4)
                    traci.trafficlight.setPhase(tls_id, target_phase - 1, connection_label)
                    transition_needed = False
                
                # Update last phase
                last_phases[tls_id] = target_phase
                
                # Record data for this decision point
                step_data.append({
                    'episode': episode,
                    'strategy': f'baseline_{strategy_id}',
                    'step': step,
                    'tls_id': tls_id,
                    'chosen_phase': target_phase,
                    'transition_needed': transition_needed,
                    'avg_incoming_speed': avg_incoming_speed,
                    'avg_outgoing_speed': avg_outgoing_speed,
                    'local_avg_speed': local_avg_speed,
                    'global_avg_speed': global_avg_speed,
                    'num_vehicles': traci.vehicle.getIDCount(connection_label)
                })
        
        # Collect overall data at logging intervals
        if step % LOG_INTERVAL == 0:
            edge_speeds = get_edge_speeds(traci)
            
            if len(edge_speeds) > 0:
                global_avg_speed = sum(edge_speeds.values()) / len(edge_speeds)
            else:
                global_avg_speed = 0
            
            step_data.append({
                'episode': episode,
                'strategy': f'baseline_{strategy_id}',
                'step': step,
                'tls_id': 'global',
                'chosen_phase': None,
                'transition_needed': None,
                'avg_incoming_speed': None,
                'avg_outgoing_speed': None,
                'local_avg_speed': None,
                'global_avg_speed': global_avg_speed,
                'num_vehicles': traci.vehicle.getIDCount(connection_label)
            })
    
    # Close TraCI connection
    traci.close(connection_label)
    
    # Terminate SUMO
    if sumo_process and sumo_process.poll() is None:
        sumo_process.terminate()
    
    # Return collected data
    return pd.DataFrame(step_data)

def run_formula_simulation(episode, select_best_phase):
    """Run a simulation using the discovered formula"""
    import random
    base_port = 9000 + (episode * 500) + 400
    port = base_port + random.randint(1, 50)
    
    connection_label = f"ep{episode}_formula"
    try:
        # Try to close the specific connection if it exists
        traci.close(connection_label)
        print(f"Closed existing connection: {connection_label}")
    except:
        # If the connection doesn't exist, this will throw an exception we can ignore
        pass
    # Prepare SUMO command
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
    
    # Start SUMO as a subprocess
    sumo_process = subprocess.Popen(sumo_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for SUMO to initialize
    time.sleep(1.0)
    
    # Try to connect to SUMO via TraCI
    max_retries = 3
    retry_delay = 1.0
    connected = False
    
    for retry in range(max_retries):
        try:
            traci.init(port=port, label=connection_label)
            connected = True
            break
        except traci.exceptions.FatalTraCIError:
            print(f"Retrying in {retry_delay} seconds")
            time.sleep(retry_delay)
            # Check if SUMO process is still alive
            if sumo_process.poll() is not None:
                # SUMO process has exited, check error output
                stdout, stderr = sumo_process.communicate()
                error_msg = stderr.decode('utf-8')
                if error_msg:
                    print(f"SUMO error: {error_msg}")
                # Start a new process
                sumo_process = subprocess.Popen(sumo_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                time.sleep(1.0)
    
    if not connected:
        print(f"Failed to connect to SUMO after {max_retries} attempts")
        if sumo_process and sumo_process.poll() is None:
            sumo_process.terminate()
        return None
    
    # Load the network
    net = sumolib.net.readNet(os.path.join(BASE_DIR, "grid.net.xml"))
    
    # Get all traffic light IDs
    tls_ids = traci.trafficlight.getIDList(connection_label)
    
    # Prepare data collection
    step_data = []
    decision_steps = list(range(0, SIM_DURATION, DECISION_INTERVAL))
    last_phases = {tls_id: None for tls_id in tls_ids}
    current_phases = {tls_id: 2 for tls_id in tls_ids}  # Start with phase 2 for all intersections
    
    # Run the simulation
    for step in range(SIM_DURATION):
        traci.simulationStep(connection_label)
        
        # Make phase decisions at specified intervals
        if step in decision_steps:
            # Get current speeds for all edges
            edge_speeds = get_edge_speeds(traci.getConnection(connection_label))
            
            # Calculate average speed
            if len(edge_speeds) > 0:
                global_avg_speed = sum(edge_speeds.values()) / len(edge_speeds)
            else:
                global_avg_speed = 0
            
            # Record data for each intersection
            for tls_id in tls_ids:
                # Get incoming and outgoing edges for this intersection
                incoming_edges = get_intersection_incoming_edges(net, tls_id)
                outgoing_edges = get_intersection_outgoing_edges(net, tls_id)
                
                # Calculate average speed for incoming and outgoing edges
                incoming_speeds = [edge_speeds.get(edge, 0) for edge in incoming_edges if edge in edge_speeds]
                outgoing_speeds = [edge_speeds.get(edge, 0) for edge in outgoing_edges if edge in edge_speeds]
                
                if incoming_speeds:
                    avg_incoming_speed = sum(incoming_speeds) / len(incoming_speeds)
                else:
                    avg_incoming_speed = 0
                
                if outgoing_speeds:
                    avg_outgoing_speed = sum(outgoing_speeds) / len(outgoing_speeds)
                else:
                    avg_outgoing_speed = 0
                
                # Local average speed (both incoming and outgoing)
                local_avg_speed = 0
                if incoming_speeds or outgoing_speeds:
                    local_avg_speed = sum(incoming_speeds + outgoing_speeds) / (len(incoming_speeds) + len(outgoing_speeds))
                
                # Use the formula to select the best phase
                current_phase = current_phases[tls_id]
                num_vehicles = traci.vehicle.getIDCount(connection_label)
                
                target_phase = select_best_phase(
                    current_phase, 
                    avg_incoming_speed, 
                    avg_outgoing_speed, 
                    local_avg_speed, 
                    global_avg_speed, 
                    num_vehicles
                )
                
                # Check if we need a transition phase
                if last_phases[tls_id] is not None and target_phase != last_phases[tls_id]:
                    # Need transition (Phase 1)
                    traci.trafficlight.setPhase(tls_id, 0, connection_label)  # Phase 1 index is 0
                    transition_needed = True
                else:
                    # Set the target phase directly
                    # Convert from our phase numbering (1-5) to SUMO's index (0-4)
                    traci.trafficlight.setPhase(tls_id, target_phase - 1, connection_label)
                    transition_needed = False
                
                # Update phases
                last_phases[tls_id] = target_phase
                current_phases[tls_id] = target_phase
                
                # Record data for this decision point
                step_data.append({
                    'episode': episode,
                    'strategy': 'formula',
                    'step': step,
                    'tls_id': tls_id,
                    'chosen_phase': target_phase,
                    'transition_needed': transition_needed,
                    'avg_incoming_speed': avg_incoming_speed,
                    'avg_outgoing_speed': avg_outgoing_speed,
                    'local_avg_speed': local_avg_speed,
                    'global_avg_speed': global_avg_speed,
                    'num_vehicles': traci.vehicle.getIDCount(connection_label)
                })
        
        # Collect overall data at logging intervals
        if step % LOG_INTERVAL == 0:
            edge_speeds = get_edge_speeds(traci.getConnection(connection_label))
            
            if len(edge_speeds) > 0:
                global_avg_speed = sum(edge_speeds.values()) / len(edge_speeds)
            else:
                global_avg_speed = 0
            
            step_data.append({
                'episode': episode,
                'strategy': 'formula',
                'step': step,
                'tls_id': 'global',
                'chosen_phase': None,
                'transition_needed': None,
                'avg_incoming_speed': None,
                'avg_outgoing_speed': None,
                'local_avg_speed': None,
                'global_avg_speed': global_avg_speed,
                'num_vehicles': traci.vehicle.getIDCount(connection_label)
            })
    
    # Close TraCI connection
    traci.close(connection_label)
    
    # Terminate SUMO
    if sumo_process and sumo_process.poll() is None:
        sumo_process.terminate()
    
    # Return collected data
    return pd.DataFrame(step_data)


def run_validation_episode(episode, select_best_phase):
    """Run a validation episode with both baseline and formula simulations"""
    print(f"Running validation episode {episode+1}/{NUM_VALIDATION_EPISODES}...")
    
    # Force cleanup before starting
    cleanup_all_connections()
    
    # Run baseline simulations (one for each fixed phase strategy)
    baseline_results = []
    for strategy_id in range(1, 5):  # Strategies 1-4
        print(f"  Running baseline strategy {strategy_id}...")
        
        # Create temporary output file
        temp_output = os.path.join(VALIDATION_DIR, f"temp_baseline_{episode}_{strategy_id}.csv")
        
        # Set environment variables
        env = os.environ.copy()
        env["BASE_DIR"] = BASE_DIR
        env["SIM_DURATION"] = str(SIM_DURATION)
        env["DECISION_INTERVAL"] = str(DECISION_INTERVAL)
        env["LOG_INTERVAL"] = str(LOG_INTERVAL)
        env["SUMO_GUI"] = str(SUMO_GUI)
        
        # Run the simulation in a separate process
        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "run_single_simulation.py"),
            "baseline",
            str(episode),
            str(strategy_id),
            temp_output
        ]
        
        result = subprocess.run(cmd, env=env)
        
        if result.returncode == 0 and os.path.exists(temp_output):
            baseline_data = pd.read_csv(temp_output)
            baseline_results.append(baseline_data)
            os.remove(temp_output)  # Clean up temp file
        else:
            print(f"  Baseline strategy {strategy_id} failed to run. Skipping...")
    
    # Check if we have any baseline results
    if not baseline_results:
        print("  All baseline strategies failed. Skipping episode...")
        return None
    
    # Run formula simulation
    print("  Running formula-based strategy...")
    temp_output = os.path.join(VALIDATION_DIR, f"temp_formula_{episode}.csv")
    formula_file = os.path.join(FORMULA_DIR, "phase_selection_formula.py")
    
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "run_single_simulation.py"),
        "formula",
        str(episode),
        formula_file,
        temp_output
    ]
    
    result = subprocess.run(cmd, env=env)
    
    if result.returncode == 0 and os.path.exists(temp_output):
        formula_data = pd.read_csv(temp_output)
        os.remove(temp_output)  # Clean up temp file
        # Combine all results
        all_data = pd.concat(baseline_results + [formula_data])
    else:
        print("  Formula strategy failed to run. Using only baseline data...")
        # Combine results from just the baselines
        all_data = pd.concat(baseline_results)
    
    # Save episode data
    output_file = os.path.join(VALIDATION_DIR, f"validation_episode_{episode}.csv")
    all_data.to_csv(output_file, index=False)
    
    print(f"  Episode {episode+1} completed! Data saved to {output_file}")
    
    return all_data

def analyze_results(all_validation_data):
    """Analyze validation results and compare strategies"""
    print("\nAnalyzing validation results...")
    
    # Filter to only global measurements
    global_data = all_validation_data[all_validation_data['tls_id'] == 'global'].copy()
    
    # Calculate average speed for each strategy across all episodes
    strategy_performance = global_data.groupby(['strategy', 'step'])['global_avg_speed'].mean().reset_index()
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    
    # Plot average speed over time for each strategy
    strategies = strategy_performance['strategy'].unique()
    for strategy in strategies:
        strat_data = strategy_performance[strategy_performance['strategy'] == strategy]
        plt.plot(strat_data['step'], strat_data['global_avg_speed'], label=strategy)
    
    plt.xlabel('Simulation Step')
    plt.ylabel('Average Speed (m/s)')
    plt.title('Strategy Performance Comparison')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plot_file = os.path.join(VALIDATION_DIR, 'strategy_comparison.png')
    plt.savefig(plot_file)
    plt.close()
    
    # Calculate overall performance metrics
    performance_summary = global_data.groupby('strategy')['global_avg_speed'].agg(
        ['mean', 'std', 'min', 'max']
    ).reset_index()
    
    # Save summary to file
    summary_file = os.path.join(VALIDATION_DIR, 'performance_summary.csv')
    performance_summary.to_csv(summary_file, index=False)
    
    # Print summary
    print("\nPerformance Summary:")
    print(performance_summary)
    
    # Calculate improvement percentage of formula over best baseline
    formula_perf = None
    if 'formula' in performance_summary['strategy'].values:
        formula_perf = performance_summary[performance_summary['strategy'] == 'formula']['mean'].values[0]
    
    baseline_perfs = performance_summary[performance_summary['strategy'] != 'formula']['mean']
    best_baseline = baseline_perfs.max() if not baseline_perfs.empty else 0
    
    if formula_perf is not None:
        improvement = ((formula_perf - best_baseline) / best_baseline) * 100 if best_baseline > 0 else 0
        
        print(f"\nFormula performance: {formula_perf:.2f} m/s")
        print(f"Best baseline performance: {best_baseline:.2f} m/s")
        print(f"Improvement: {improvement:.2f}%")
        
        return {
            'formula_performance': formula_perf,
            'best_baseline': best_baseline,
            'improvement_percentage': improvement,
            'summary': performance_summary
        }
    else:
        print("\nNo formula performance data available.")
        return {
            'formula_performance': 0,
            'best_baseline': best_baseline,
            'improvement_percentage': 0,
            'summary': performance_summary
        }

def main():
    """Main execution function"""
    print("=" * 80)
    print("TRAFFIC SIGNAL FORMULA VALIDATION")
    print("=" * 80)
    
    # Force cleanup all connections and SUMO processes
    cleanup_all_connections()
    
    # Check for required configuration files
    tls_config = os.path.join(BASE_DIR, "custom_tls.add.xml")
    if not os.path.exists(tls_config):
        print(f"ERROR: Traffic light configuration file not found: {tls_config}")
        print("Make sure you've run the grid generator script first.")
        return
    
    sumocfg = os.path.join(BASE_DIR, "grid.sumocfg")
    if not os.path.exists(sumocfg):
        print(f"ERROR: SUMO configuration file not found: {sumocfg}")
        print("Make sure you've run the grid generator script first.")
        return
    
    # Also check that custom_tls.add.xml is referenced in the sumocfg file
    with open(sumocfg, 'r') as f:
        content = f.read()
        if "custom_tls.add.xml" not in content:
            print("WARNING: custom_tls.add.xml is not referenced in grid.sumocfg")
            print("This may cause traffic light initialization errors.")
            print("Please update your SUMO configuration to include this file.")
    
    # Step 1: Load the formula
    print("\nLoading phase selection formula...")
    try:
        select_best_phase = load_formula()
        print("Formula loaded successfully!")
    except Exception as e:
        print(f"Error loading formula: {e}")
        return
    
    # Step 2: Run validation episodes
    all_validation_data = []
    for episode in range(NUM_VALIDATION_EPISODES):
        episode_data = run_validation_episode(episode, select_best_phase)
        if episode_data is not None:
            all_validation_data.append(episode_data)
    
    # Check if we have any validation data
    if not all_validation_data:
        print("No validation data was collected. Exiting...")
        return
    
    # Combine all validation data
    combined_data = pd.concat(all_validation_data)
    combined_output = os.path.join(VALIDATION_DIR, "all_validation_episodes.csv")
    combined_data.to_csv(combined_output, index=False)
    
    # Step 3: Analyze results
    results = analyze_results(combined_data)
    
    # Write results to file
    with open(os.path.join(VALIDATION_DIR, 'validation_results.txt'), 'w') as f:
        f.write("TRAFFIC SIGNAL FORMULA VALIDATION RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Formula Performance: {results['formula_performance']:.4f} m/s\n")
        f.write(f"Best Baseline Performance: {results['best_baseline']:.4f} m/s\n")
        f.write(f"Improvement: {results['improvement_percentage']:.2f}%\n\n")
        f.write("Performance Summary:\n")
        f.write(results['summary'].to_string(index=False))
    
    print("\nValidation complete!")
    print(f"All results saved to {VALIDATION_DIR}")
    
    if results['improvement_percentage'] > 0:
        print("\nThe discovered formula outperforms the baseline strategies!")
    else:
        print("\nThe formula did not improve over the baseline strategies. Consider refining the approach.")

if __name__ == "__main__":
    # Make sure the SUMO environment is set up
    if "SUMO_HOME" not in os.environ:
        raise EnvironmentError("SUMO_HOME environment variable is not set. Please set it before running this script.")
    
    main()