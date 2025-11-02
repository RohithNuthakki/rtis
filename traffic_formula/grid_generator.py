#!/usr/bin/env python
# Network generator for SUMO 5x5 grid - Simplified version

import os
import subprocess

# Configuration
GRID_SIZE = 5  # 5x5 grid
EDGE_LENGTH = 200  # meters
LANES = 2  # lanes per direction
SPEED = 13.89  # m/s (50 km/h)
OUTPUT_DIR = "/Users/samnu/Documents/SUMO/traffic_formula/sumo_grid"

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Generate the grid network using SUMO's netgenerate
net_file = os.path.join(OUTPUT_DIR, "grid.net.xml")
netgenerate_cmd = [
    "netgenerate",
    "--grid", "true",
    "--grid.x-number", str(GRID_SIZE),
    "--grid.y-number", str(GRID_SIZE),
    "--grid.length", str(EDGE_LENGTH),
    "--default-junction-type", "traffic_light",
    "--default.lanenumber", str(LANES),
    "--default.speed", str(SPEED),
    "--output-file", net_file
]

print("Generating SUMO network...")
print(f"Command: {' '.join(netgenerate_cmd)}")
result = subprocess.run(netgenerate_cmd)
if result.returncode != 0:
    print("ERROR: Failed to generate network. Exiting.")
    exit(1)

print(f"Network generated: {net_file}")

# Create a custom traffic light configuration file to match the specified phases
tls_file = os.path.join(OUTPUT_DIR, "custom_tls.add.xml")
with open(tls_file, "w") as f:
    f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    f.write('<additional xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/additional_file.xsd">\n')
    
    # First, we need to identify all the traffic light IDs in the grid
    # In a grid network, junctions are named as follows:
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            junction_id = f"{i}{j}"
            
            f.write(f'    <tlLogic id="{junction_id}" type="static" programID="custom_program" offset="0">\n')
            
            # Phase 1: Transitional (right turns only) - 5 seconds
            f.write('        <phase duration="5"  state="rrrrrrrrrGrGrGrG"/>\n')  # Simplified for clarity
            
            # Phase 2: East-West straight and all right turns - 15 seconds
            f.write('        <phase duration="15" state="GrGrrrrrGrGrGrG"/>\n')
            
            # Phase 3: North-South straight and all right turns - 15 seconds
            f.write('        <phase duration="15" state="rrrGGrrrGrGrGrG"/>\n')
            
            # Phase 4: East-West left turns and all right turns - 15 seconds
            f.write('        <phase duration="15" state="rrrrGGrrGrGrGrG"/>\n')
            
            # Phase 5: North-South left turns and all right turns - 15 seconds
            f.write('        <phase duration="15" state="GGrrrrrrrGrGrGrG"/>\n')
            
            f.write('    </tlLogic>\n')
    
    f.write('</additional>\n')

print(f"Custom traffic light configuration generated: {tls_file}")

# Create a configuration file
config_file = os.path.join(OUTPUT_DIR, "grid.sumocfg")
with open(config_file, "w") as f:
    f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    f.write('<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">\n')
    f.write('    <input>\n')
    f.write(f'        <net-file value="grid.net.xml"/>\n')
    f.write(f'        <route-files value="grid.rou.xml"/>\n')
    f.write(f'        <additional-files value="custom_tls.add.xml"/>\n')
    f.write('    </input>\n')
    f.write('    <time>\n')
    f.write('        <begin value="0"/>\n')
    f.write('        <end value="1800"/>\n')  # 30 minutes simulation
    f.write('        <step-length value="1"/>\n')
    f.write('    </time>\n')
    f.write('    <processing>\n')
    f.write('        <time-to-teleport value="300"/>\n')  # Teleport after 5 minutes of waiting
    f.write('        <max-depart-delay value="100"/>\n')  # Maximum departure delay
    f.write('    </processing>\n')
    f.write('    <report>\n')
    f.write('        <verbose value="true"/>\n')
    f.write('        <duration-log.statistics value="true"/>\n')
    f.write('        <no-step-log value="true"/>\n')
    f.write('    </report>\n')
    f.write('</configuration>\n')

print(f"SUMO configuration file generated: {config_file}")
print("Network setup complete!")
