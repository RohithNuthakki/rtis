#!/usr/bin/env python
# Traffic demand generator for SUMO 5x5 grid

import os
import random
import numpy as np

# Configuration
OUTPUT_DIR = "/Users/samnu/Documents/SUMO/traffic_formula/sumo_grid"
GRID_SIZE = 5
SIM_DURATION = 1800  # 30 minutes in seconds
TOTAL_VEHICLES = 500  # Total number of vehicles to generate
VEHICLE_TYPES = ["car", "bus", "truck"]
VEHICLE_DISTRIBUTIONS = [0.80, 0.10, 0.10]  # 80% cars, 10% buses, 10% trucks

# Create the route file
route_file = os.path.join(OUTPUT_DIR, "grid.rou.xml")

def get_edge_ids():
    """Generate edge IDs for a 5x5 grid"""
    edges = []
    
    # Horizontal edges (west to east)
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE - 1):
            edges.append(f"{i}{j}to{i}{j+1}")
    
    # Horizontal edges (east to west)
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE - 1, 0, -1):
            edges.append(f"{i}{j}to{i}{j-1}")
    
    # Vertical edges (south to north)
    for j in range(GRID_SIZE):
        for i in range(GRID_SIZE - 1):
            edges.append(f"{i}{j}to{i+1}{j}")
    
    # Vertical edges (north to south)
    for j in range(GRID_SIZE):
        for i in range(GRID_SIZE - 1, 0, -1):
            edges.append(f"{i}{j}to{i-1}{j}")
    
    return edges

def generate_route():
    """Generate a random route through the network"""
    # Pick random entry and exit points on the grid perimeter
    entry_edges = []
    exit_edges = []
    
    # North perimeter
    for j in range(GRID_SIZE):
        entry_edges.append(f"{GRID_SIZE-1}{j}to{GRID_SIZE-2}{j}")
        exit_edges.append(f"{GRID_SIZE-2}{j}to{GRID_SIZE-1}{j}")
    
    # South perimeter
    for j in range(GRID_SIZE):
        entry_edges.append(f"0{j}to1{j}")
        exit_edges.append(f"1{j}to0{j}")
    
    # East perimeter
    for i in range(GRID_SIZE):
        entry_edges.append(f"{i}{GRID_SIZE-1}to{i}{GRID_SIZE-2}")
        exit_edges.append(f"{i}{GRID_SIZE-2}to{i}{GRID_SIZE-1}")
    
    # West perimeter
    for i in range(GRID_SIZE):
        entry_edges.append(f"{i}0to{i}1")
        exit_edges.append(f"{i}1to{i}0")
    
    # Select random entry and exit edges
    entry_edge = random.choice(entry_edges)
    exit_edge = random.choice(exit_edges)
    
    # Generate a route from entry to exit (simplified approach)
    # In a real implementation, you might want a proper pathfinding algorithm
    all_edges = get_edge_ids()
    route_length = random.randint(3, 10)  # Random route length
    route = [entry_edge]
    
    for _ in range(route_length):
        last_edge = route[-1]
        
        # Get the destination node of the last edge
        to_node = last_edge.split("to")[1]
        
        # Find potential next edges
        next_edges = [edge for edge in all_edges if edge.startswith(to_node + "to")]
        
        # Remove edges that would make a U-turn
        no_uturn_edges = [edge for edge in next_edges if not (edge.split("to")[1] == last_edge.split("to")[0])]
        
        # If there are valid edges, pick one
        if no_uturn_edges:
            route.append(random.choice(no_uturn_edges))
        else:
            # If no valid edges, we're probably at a dead end, add the exit edge
            break
    
    # Ensure the route ends with the exit edge
    route.append(exit_edge)
    
    return " ".join(route)

# Generate vehicle types and routes
with open(route_file, "w") as f:
    f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    f.write('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n')
    
    # Define vehicle types
    f.write('    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="15" guiShape="passenger"/>\n')
    f.write('    <vType id="bus" accel="1.2" decel="2.5" sigma="0.5" length="12" minGap="2.5" maxSpeed="12" guiShape="bus"/>\n')
    f.write('    <vType id="truck" accel="0.8" decel="2.0" sigma="0.5" length="7.5" minGap="3.0" maxSpeed="10" guiShape="truck"/>\n')
    
    # Increasing vehicle flow over time
    # We'll create 500 vehicles with increasing frequency over the 30-minute period
    start_times = np.random.exponential(scale=1800/TOTAL_VEHICLES, size=TOTAL_VEHICLES)
    start_times = np.cumsum(start_times)
    start_times = start_times[start_times <= SIM_DURATION]
    
    # Generate additional vehicles if needed to reach TOTAL_VEHICLES
    while len(start_times) < TOTAL_VEHICLES:
        additional_times = np.random.uniform(0, SIM_DURATION, TOTAL_VEHICLES - len(start_times))
        start_times = np.concatenate([start_times, additional_times])
        start_times = np.sort(start_times)
        start_times = start_times[start_times <= SIM_DURATION]
    
    # Generate vehicles
    for i, depart_time in enumerate(start_times):
        # Choose vehicle type based on distribution
        veh_type = np.random.choice(VEHICLE_TYPES, p=VEHICLE_DISTRIBUTIONS)
        route = generate_route()
        
        # Write vehicle definition
        f.write(f'    <vehicle id="veh{i}" type="{veh_type}" depart="{depart_time:.1f}" departSpeed="max">\n')
        f.write(f'        <route edges="{route}"/>\n')
        f.write('    </vehicle>\n')
    
    f.write('</routes>\n')

print(f"Traffic demand generated: {route_file}")
print(f"Created {TOTAL_VEHICLES} vehicles over {SIM_DURATION} seconds.")
