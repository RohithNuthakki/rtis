#!/usr/bin/env python
# Simplified simulation runner for traffic light optimization

import os
import subprocess
import time
import pandas as pd
import numpy as np
import sys

# Import traci library for TraCI interface
try:
    import traci
    import sumolib
except ImportError:
    raise ImportError("Please make sure SUMO_HOME is in your PATH and the TraCI library is available")

# Configuration
BASE_DIR = "/Users/samnu/Documents/SUMO/traffic_formula/sumo_grid"
GRID_SIZE = 5
SIM_DURATION = 600  # 10 minutes instead of 30 for faster completion
DECISION_INTERVAL = 15  # seconds
LOG_INTERVAL = 15  # seconds
SUMO_GUI = False  # Set to False for faster execution

# Ensure output directories exist
RESULTS_DIR = os.path.join(BASE_DIR, "results")
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# Phase strategies
STRATEGIES = {
    1: "East-West Straight",  # Always phase 2
    2: "North-South Straight",  # Always phase 3
    3: "East-West Left",  # Always phase 4
    4: "North-South Left"  # Always phase 5
}

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

def get_intersection_edges(net, tls_id):
    """Get incoming and outgoing edges for a traffic light"""
    incoming_edges = []
    outgoing_edges = []
    for connection in net.getTLS(tls_id).getConnections():
        incoming_edge = connection[0].getFrom().getID()
        outgoing_edge = connection[0].getTo().getID()
        if incoming_edge not in incoming_edges:
            incoming_edges.append(incoming_edge)
        if outgoing_edge not in outgoing_edges:
            outgoing_edges.append(outgoing_edge)
    return incoming_edges, outgoing_edges

def run_simulation(strategy_id):
    """Run a single simulation with the given phase strategy"""
    print(f"Running simulation with strategy {strategy_id}: {STRATEGIES[strategy_id]}")
    
    # Create a unique port number for this simulation
    port = 9000 + strategy_id
    
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
    print(f"Starting SUMO process on port {port}")
    sumo_process = subprocess.Popen(sumo_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for SUMO to start
    time.sleep(2)
    
    # Connect to SUMO via TraCI with a unique label
    connection_label = f"conn_{strategy_id}"
    try:
        print(f"Connecting to SUMO with label '{connection_label}'...")
        traci.init(port=port, label=connection_label)
    except Exception as e:
        print(f"Failed to connect to SUMO: {e}")
        if sumo_process:
            sumo_process.terminate()
        return None
    
    try:
        # Load the network
        net_file = os.path.join(BASE_DIR, "grid.net.xml")
        print(f"Loading network: {net_file}")
        net = sumolib.net.readNet(net_file)
        
        # Get all traffic light IDs
        tls_ids = traci.trafficlight.getIDList(connection_label)
        print(f"Found {len(tls_ids)} traffic lights")
        
        # Prepare data collection
        step_data = []
        decision_steps = list(range(0, SIM_DURATION, DECISION_INTERVAL))
        last_phases = {tls_id: None for tls_id in tls_ids}
        
        # Choose phase based on strategy
        target_phase = strategy_id + 1  # Strategies 1-4 correspond to phases 2-5
        
        # Run the simulation
        print(f"Starting simulation for {SIM_DURATION} steps...")
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
                    incoming_edges, outgoing_edges = get_intersection_edges(net, tls_id)
                    
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
                    
                    # Check if we need a transition phase
                    current_phase = traci.trafficlight.getPhase(tls_id, connection_label)
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
                        'strategy': strategy_id,
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
                    'strategy': strategy_id,
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
            
            # Print progress occasionally
            if step % 100 == 0:
                print(f"  Simulation progress: {step}/{SIM_DURATION} steps")
        
        print("Simulation completed successfully")
        
        # Close TraCI connection
        print("Closing TraCI connection")
        traci.close(connection_label)
        
        # Terminate SUMO
        print("Terminating SUMO process")
        sumo_process.terminate()
        
        # Return collected data
        return pd.DataFrame(step_data)
    
    except Exception as e:
        print(f"Error during simulation: {e}")
        
        # Try to close TraCI connection
        try:
            traci.close(connection_label)
        except:
            pass
        
        # Terminate SUMO process
        if sumo_process:
            sumo_process.terminate()
        
        return None

def run_all_simulations():
    """Run simulations for all strategies"""
    print("Running simulations for all strategies...")
    
    all_data = []
    
    # Run each strategy simulation sequentially
    for strategy_id in range(1, 5):  # Strategies 1-4
        result = run_simulation(strategy_id)
        if result is not None:
            all_data.append(result)
    
    # Combine all results
    if all_data:
        combined_data = pd.concat(all_data)
        combined_output = os.path.join(RESULTS_DIR, "all_episodes.csv")
        combined_data.to_csv(combined_output, index=False)
        print(f"All simulations completed! Data saved to {combined_output}")
        return combined_data
    else:
        print("No data was collected from any simulation.")
        return None

if __name__ == "__main__":
    # Make sure the SUMO environment is set up
    if "SUMO_HOME" not in os.environ:
        raise EnvironmentError("SUMO_HOME environment variable is not set. Please set it before running this script.")
    
    # Run the network and demand generators first
    print("Setting up the simulation environment...")
    
    # Check if the network and route files exist
    net_file = os.path.join(BASE_DIR, "grid.net.xml")
    route_file = os.path.join(BASE_DIR, "grid.rou.xml")
    
    if not os.path.exists(net_file) or not os.path.exists(route_file):
        print("Network or route files missing. Please run the grid and demand generators first.")
        exit(1)
    
    # Run all simulations
    data = run_all_simulations()
    
    if data is not None:
        print("Simulation complete!")
        print(f"Total data points: {len(data)}")
    else:
        print("Simulation failed to collect any data.")
        exit(1)
