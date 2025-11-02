#!/usr/bin/env python3
"""
Manhattan Traffic Simulation using HERE API Data (Revised)
--------------------------------------------------------
This script uses historical traffic data from HERE APIs to create a 
realistic traffic simulation for Manhattan using SUMO.

The simulation directly uses traffic flow data to generate appropriate
traffic demand, incorporating:
- Traffic density and directionality from HERE Traffic Flow API
- Traffic incidents from HERE Traffic Incidents API
- Time-specific patterns (morning/evening rush hour)

Usage:
python manhattan_traffic_simulation.py --time-period evening_rush
"""

import os
import sys
import json
import time
import random
import argparse
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from math import radians, cos, sin, asin, sqrt

# Define workspace path
WORKSPACE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(WORKSPACE, exist_ok=True)
os.chdir(WORKSPACE)

# Setup SUMO environment
def set_sumo_environment():
    """Set up SUMO environment variables and paths"""
    # Try to find SUMO installation
    sumo_home = os.environ.get("SUMO_HOME", None)
    
    if not sumo_home:
        # Check common installation locations
        if sys.platform.startswith('win'):
            possible_paths = [
                r"C:\Program Files (x86)\Eclipse\Sumo",
                r"C:\Program Files\Eclipse\Sumo"
            ]
        elif sys.platform.startswith('darwin'):  # macOS
            possible_paths = [
                "/Applications/sumo",
                "/usr/local/opt/sumo",
                "/opt/homebrew/opt/sumo"
            ]
        else:  # Linux
            possible_paths = [
                "/usr/share/sumo",
                "/usr/local/share/sumo"
            ]
        
        # Check paths
        for path in possible_paths:
            if os.path.isdir(path):
                sumo_home = path
                os.environ["SUMO_HOME"] = path
                print(f"Found SUMO at {path}")
                break
    
    if not sumo_home:
        print("Warning: SUMO_HOME not found. Please make sure SUMO is installed.")
        print("You can download SUMO from: https://www.eclipse.org/sumo/")
        return False
    
    # Add SUMO tools to Python path
    tools_path = os.path.join(sumo_home, "tools")
    if not os.path.isdir(tools_path):
        tools_path = os.path.join(sumo_home, "share", "sumo", "tools")
    
    if os.path.isdir(tools_path):
        if tools_path not in sys.path:
            sys.path.append(tools_path)
            print(f"Added SUMO tools to Python path: {tools_path}")
    else:
        print(f"Warning: Could not find SUMO tools at {tools_path}")
        return False
    
    return True

# Try to set up SUMO environment
if not set_sumo_environment():
    print("Warning: SUMO environment not properly set up. Some functionality may be limited.")

# Try to import SUMO modules
try:
    import sumolib
    from sumolib import checkBinary
    import traci
    print("Successfully imported SUMO libraries.")
except ImportError as e:
    print(f"Error importing SUMO libraries: {e}")
    print("Please make sure SUMO is correctly installed and SUMO_HOME is set.")
    print("Continuing with limited functionality...")

# Simulation parameters
SIMULATION_DURATION = 3600  # 1 hour in seconds
VEHICLE_TYPES = {
    "passenger": 0.6,  # 60% passenger cars
    "taxi": 0.25,      # 25% taxis
    "delivery": 0.1,   # 10% delivery vehicles
    "bus": 0.05        # 5% buses
}

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Manhattan Traffic Simulation using HERE API Data')
    
    parser.add_argument('--time-period', default='evening_rush', 
                      choices=['morning_rush', 'evening_rush'],
                      help='Time period to simulate (morning_rush=8-9am, evening_rush=5-6pm)')
    
    parser.add_argument('--force-regenerate', action='store_true', 
                      help='Force regeneration of all files')
    
    parser.add_argument('--skip-simulation', action='store_true',
                      help='Generate files but skip running the simulation')
    
    parser.add_argument('--gui', action='store_true',
                      help='Run simulation with GUI')
    
    parser.add_argument('--edge-matching-radius', type=float, default=50.0,
                      help='Radius (in meters) for matching HERE segments to SUMO edges')
    
    parser.add_argument('--visualize', action='store_true',
                      help='Create visualizations of the simulation setup')
    
    return parser.parse_args()

def check_here_data_availability(time_period):
    """Check if the required HERE data files are available"""
    required_files = [
        f"traffic_data/processed_traffic_flow_{time_period}.json",
        f"traffic_data/sumo_edge_mapping_{time_period}.json"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("Error: Required HERE data files not found.")
        print("Missing files:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nPlease run here_data_collector.py first to collect traffic data.")
        return False
    
    return True

def download_osm_map():
    """Download OpenStreetMap data for Manhattan"""
    print("Downloading/checking OpenStreetMap data...")
    
    if os.path.exists("manhattan.osm") and not args.force_regenerate:
        print("Using existing OpenStreetMap data.")
        return True
    
    # Try to get the bounding box from the traffic data
    try:
        # Check if any traffic data file exists
        pattern = "traffic_data/traffic_flow_*.json"
        flow_files = list(Path(WORKSPACE).glob(pattern))
        
        if flow_files:
            # Use the first file found
            with open(flow_files[0], "r") as f:
                data = json.load(f)
                bbox = data.get("metadata", {}).get("bbox")
                if bbox:
                    print(f"Using bounding box from traffic data: {bbox}")
                else:
                    # Default Manhattan bounding box
                    bbox = {
                        "min_lat": 40.712178,
                        "min_lon": -74.033341,
                        "max_lat": 40.759722,
                        "max_lon": -73.958424
                    }
        else:
            # Default Manhattan bounding box
            bbox = {
                "min_lat": 40.712178,
                "min_lon": -74.033341,
                "max_lat": 40.759722,
                "max_lon": -73.958424
            }
    except Exception as e:
        print(f"Error reading bounding box from traffic data: {e}")
        # Default Manhattan bounding box
        bbox = {
            "min_lat": 40.712178,
            "min_lon": -74.033341,
            "max_lat": 40.759722,
            "max_lon": -73.958424
        }
    
    # Try using Overpass API for download
    overpass_url = "https://overpass-api.de/api/interpreter"
    
    overpass_query = f"""
    [out:xml];
    (
      way["highway"]({bbox['min_lat']},{bbox['min_lon']},{bbox['max_lat']},{bbox['max_lon']});
      relation["highway"]({bbox['min_lat']},{bbox['min_lon']},{bbox['max_lat']},{bbox['max_lon']});
      way["footway"]({bbox['min_lat']},{bbox['min_lon']},{bbox['max_lat']},{bbox['max_lon']});
      way["pedestrian"]({bbox['min_lat']},{bbox['min_lon']},{bbox['max_lat']},{bbox['max_lon']});
    );
    (._;>;);
    out meta;
    """
    
    try:
        import requests
        print("Using Overpass API to download OSM data...")
        response = requests.post(overpass_url, data=overpass_query)
        
        if response.status_code == 200 and len(response.content) > 1000:
            with open("manhattan.osm", "wb") as f:
                f.write(response.content)
            print("OSM data downloaded successfully via Overpass API.")
            return True
        else:
            print(f"Failed to download OSM data via Overpass API: {response.status_code}")
            
            # Try direct OSM API as a last resort
            bbox_str = f"{bbox['min_lon']},{bbox['min_lat']},{bbox['max_lon']},{bbox['max_lat']}"
            osm_api_url = f"https://api.openstreetmap.org/api/0.6/map?bbox={bbox_str}"
            
            print("Trying OSM API as a last resort...")
            response = requests.get(osm_api_url)
            
            if response.status_code == 200:
                with open("manhattan.osm", "wb") as f:
                    f.write(response.content)
                print("OSM data downloaded successfully via OSM API.")
                return True
            else:
                print(f"Failed to download OSM data via OSM API: {response.status_code}")
                return False
    except Exception as e:
        print(f"Error downloading OSM data: {e}")
        return False

def create_sumo_network():
    """Convert OSM map to SUMO network"""
    print("Converting OSM data to SUMO network...")
    
    if os.path.exists("manhattan.net.xml") and not args.force_regenerate:
        print("Using existing SUMO network.")
        return True
    
    try:
        # Find netconvert
        netconvert = checkBinary("netconvert")
        
        netconvert_cmd = [
            netconvert,
            "--osm", "manhattan.osm",
            "--output-file", "manhattan.net.xml",
            "--geometry.remove", "true",
            "--roundabouts.guess", "true",
            "--junctions.join", "true",
            "--tls.guess", "true",
            "--tls.join", "true",
            "--tls.default-type", "actuated",
            "--no-internal-links", "false",
            "--no-turnarounds", "true",
            "--ramps.guess", "true",
            "--junctions.corner-detail", "5",
            "--walkingareas", "true",
            "--crossings.guess", "true",  # Add pedestrian crossings
            "--sidewalks.guess", "true"   # Add sidewalks
        ]
        
        print(f"Running: {' '.join(netconvert_cmd)}")
        subprocess.run(netconvert_cmd, check=True)
        print("SUMO network created successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error creating SUMO network: {e}")
        return False
    except FileNotFoundError:
        print("netconvert command not found. Check SUMO installation.")
        return False

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371000  # Radius of earth in meters
    return c * r

def match_here_segments_to_sumo_edges(time_period, radius=50.0):
    """
    Match HERE traffic flow segments to SUMO network edges.
    This is crucial for transferring the traffic data to the simulation.
    """
    print(f"Matching HERE segments to SUMO edges for {time_period}...")
    
    # Output file
    output_file = f"traffic_data/matched_edges_{time_period}.json"
    
    if os.path.exists(output_file) and not args.force_regenerate:
        print(f"Using existing matched edges for {time_period}.")
        with open(output_file, "r") as f:
            return json.load(f)
    
    # Load the SUMO network
    try:
        net = sumolib.net.readNet("manhattan.net.xml")
    except Exception as e:
        print(f"Error loading SUMO network: {e}")
        return None
    
    # Load the mapping file
    mapping_file = f"traffic_data/sumo_edge_mapping_{time_period}.json"
    try:
        with open(mapping_file, "r") as f:
            mapping_data = json.load(f)
    except Exception as e:
        print(f"Error loading mapping file: {e}")
        return None
    
    # Track the matching results
    matched_segments = []
    total_matched_edges = 0
    
    # Iterate through each HERE segment
    for segment in mapping_data.get("segments", []):
        # Get start and end points
        start_point = segment.get("start_point")
        end_point = segment.get("end_point")
        
        if not start_point or not end_point:
            continue
        
        # Convert to SUMO coordinates
        start_x, start_y = net.convertLonLat2XY(start_point[1], start_point[0])
        end_x, end_y = net.convertLonLat2XY(end_point[1], end_point[0])
        
        # Find nearby edges for start point
        start_edges = net.getNeighboringEdges(start_x, start_y, radius)
        start_edges.sort(key=lambda x: x[1])  # Sort by distance
        
        # Find nearby edges for end point
        end_edges = net.getNeighboringEdges(end_x, end_y, radius)
        end_edges.sort(key=lambda x: x[1])  # Sort by distance
        
        matched_edges = []
        
        # Find edges that are near both the start and end points
        if start_edges and end_edges:
            # Try to find connected paths
            for start_edge, _ in start_edges[:3]:  # Check top 3 closest start edges
                for end_edge, _ in end_edges[:3]:  # Check top 3 closest end edges
                    
                    # Skip if same edge (would create a self-loop)
                    if start_edge.getID() == end_edge.getID():
                        matched_edges.append(start_edge.getID())
                        continue
                    
                    # Try to find a path between start and end edges
                    try:
                        # Use SUMO's router to find a path
                        router = sumolib.routing.dijkstra.Router(net.getEdges())
                        path = router.compute_path(
                            from_edge=start_edge,
                            to_edge=end_edge
                        )
                        
                        # If path found, add all edges in the path
                        if path:
                            path_edge_ids = [edge.getID() for edge in path]
                            matched_edges.extend(path_edge_ids)
                    except Exception as e:
                        # Failed to find path, just add the start and end edges
                        matched_edges.append(start_edge.getID())
                        matched_edges.append(end_edge.getID())
        
        # If no path found, just use the closest edges
        if not matched_edges:
            if start_edges:
                matched_edges.append(start_edges[0][0].getID())
            if end_edges:
                matched_edges.append(end_edges[0][0].getID())
        
        # Remove duplicates while preserving order
        matched_edges = list(dict.fromkeys(matched_edges))
        
        # Skip if no edges matched
        if not matched_edges:
            continue
        
        # Create matched segment record
        matched_segment = {
            "here_id": segment.get("id"),
            "road_name": segment.get("road_name"),
            "direction": segment.get("direction"),
            "vehicle_count": segment.get("vehicle_count"),
            "vehicle_density": segment.get("vehicle_density"),
            "speed": segment.get("speed"),
            "jam_factor": segment.get("jam_factor"),
            "sumo_edges": matched_edges
        }
        
        matched_segments.append(matched_segment)
        total_matched_edges += len(matched_edges)
    
    # Create result object
    result = {
        "time_period": time_period,
        "matched_segments": matched_segments,
        "total_here_segments": len(mapping_data.get("segments", [])),
        "total_matched_segments": len(matched_segments),
        "total_matched_edges": total_matched_edges,
        "matching_radius": radius
    }
    
    # Save the results
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"Matched {len(matched_segments)} HERE segments to {total_matched_edges} SUMO edges.")
    print(f"Results saved to {output_file}")
    
    return result

def generate_traffic_flows(matched_edges, time_period):
    """
    Generate traffic flows for SUMO simulation based on matched HERE segments.
    This converts the traffic density and directionality data into SUMO flows.
    """
    print(f"Generating traffic flows for {time_period}...")
    
    # Output files
    vtypes_file = f"manhattan_vtypes_{time_period}.xml"
    flows_file = f"manhattan_flows_{time_period}.xml"
    
    # Skip if files exist and we're not forcing regeneration
    if os.path.exists(flows_file) and not args.force_regenerate:
        print(f"Using existing traffic flow files for {time_period}.")
        return True
    
    # Load incidents data if available
    incidents_file = f"traffic_data/processed_incidents_{time_period}.json"
    incidents = None
    if os.path.exists(incidents_file):
        try:
            with open(incidents_file, "r") as f:
                incidents = json.load(f)
            print(f"Loaded {incidents.get('total_incidents', 0)} traffic incidents.")
        except Exception as e:
            print(f"Error loading incidents data: {e}")
    
    # Create vehicle types XML
    with open(vtypes_file, "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n')
        
        # Define vehicle types with realistic parameters
        f.write('    <vType id="passenger" vClass="passenger" color="0,0,1" accel="2.6" decel="4.5" sigma="0.5" length="4.5" minGap="2.5" maxSpeed="15" speedDev="0.1" guiShape="passenger"/>\n')
        f.write('    <vType id="taxi" vClass="taxi" color="1,1,0" accel="2.8" decel="4.5" sigma="0.5" length="4.5" minGap="2.0" maxSpeed="16" speedDev="0.1" guiShape="passenger"/>\n')
        f.write('    <vType id="delivery" vClass="delivery" color="1,0,0" accel="2.4" decel="4.0" sigma="0.5" length="6.5" minGap="3.0" maxSpeed="14" speedDev="0.1" guiShape="delivery"/>\n')
        f.write('    <vType id="bus" vClass="bus" color="0,1,0" accel="2.0" decel="3.5" sigma="0.5" length="12.0" minGap="3.5" maxSpeed="13" speedDev="0.1" guiShape="bus"/>\n')
        
        # Add pedestrian type
        f.write('    <vType id="pedestrian" vClass="pedestrian" color="0.5,0.5,0.5" width="0.6" length="0.5" minGap="0.5" maxSpeed="5.4" guiShape="pedestrian"/>\n')
        
        f.write('</routes>\n')
    
    # Create flows XML file
    with open(flows_file, "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n')
        
        f.write(f'    <include href="{vtypes_file}"/>\n\n')
        
        # Create flows for each HERE segment with SUMO edges
        flow_id = 0
        for segment in matched_edges.get("matched_segments", []):
            sumo_edges = segment.get("sumo_edges", [])
            if not sumo_edges:
                continue
            
            # Skip segments with less than 1 vehicle (unrealistic)
            vehicle_count = max(1, segment.get("vehicle_count", 0))
            
            for vtype, proportion in VEHICLE_TYPES.items():
                vtype_count = max(1, int(vehicle_count * proportion))
                
                # Determine departure timing spread over the simulation duration
                # This creates more realistic traffic patterns
                begin_time = random.randint(0, 300)  # Start within first 5 minutes
                end_time = SIMULATION_DURATION
                
                # Create a flow for each pair of edges (origin → destination)
                # This helps distribute traffic across the network
                for idx, from_edge in enumerate(sumo_edges[:-1]):
                    to_edge = sumo_edges[idx + 1]
                    
                    # Create flow
                    f.write(f'    <flow id="flow_{flow_id}_{vtype}" type="{vtype}" from="{from_edge}" to="{to_edge}" begin="{begin_time}" end="{end_time}" number="{vtype_count}" departLane="best" departSpeed="max"/>\n')
                    flow_id += 1
        
        # Add traffic incidents if available
        if incidents and incidents.get('total_incidents', 0) > 0:
            f.write('\n    <!-- Traffic incidents based on HERE data -->\n')
            
            # Load the SUMO network
            net = sumolib.net.readNet("manhattan.net.xml")
            
            for idx, incident in enumerate(incidents.get('incidents', [])):
                # Skip incidents without sufficient coordinate data
                if len(incident.get('coords', [])) < 2:
                    continue
                
                # Find the closest edge to the incident location
                incident_lat, incident_lon = incident['coords'][0]
                incident_x, incident_y = net.convertLonLat2XY(incident_lon, incident_lat)
                
                # Find edges near the incident
                nearby_edges = net.getNeighboringEdges(incident_x, incident_y, 50)  # 50m radius
                
                if not nearby_edges:
                    continue
                    
                # Sort by distance
                nearby_edges.sort(key=lambda x: x[1])
                edge, distance = nearby_edges[0]
                
                # Skip pedestrian-only edges
                if not edge.allows("passenger"):
                    continue
                
                # Determine severity and type-specific parameters
                severity = incident.get('severity', 0)
                incident_type = incident.get('type', 'UNKNOWN')
                
                # Mild incidents: slow down traffic
                if severity <= 2:
                    # Create a temporary reduced speed area
                    f.write(f'    <variableSpeedSign id="incident_{idx}" lanes="{edge.getID()}_0">\n')
                    f.write(f'        <step time="0" speed="5"/>\n')  # Reduce to 5m/s (18 km/h)
                    f.write(f'        <step time="{SIMULATION_DURATION}" speed="5"/>\n')
                    f.write(f'    </variableSpeedSign>\n')
                    
                # Moderate incidents: close a lane
                elif severity <= 4:
                    # Close the rightmost lane if the edge has multiple lanes
                    if edge.getLaneNumber() > 1:
                        # Create a closing lane
                        f.write(f'    <closingLaneRerouter id="incident_{idx}" lanes="{edge.getID()}_0">\n')
                        f.write(f'        <closingLane startPos="0" endPos="{edge.getLength()}"/>\n')
                        f.write(f'    </closingLaneRerouter>\n')
                    else:
                        # For single lane edges, just slow down traffic significantly
                        f.write(f'    <variableSpeedSign id="incident_{idx}" lanes="{edge.getID()}_0">\n')
                        f.write(f'        <step time="0" speed="2"/>\n')  # Reduce to 2m/s (7.2 km/h)
                        f.write(f'        <step time="{SIMULATION_DURATION}" speed="2"/>\n')
                        f.write(f'    </variableSpeedSign>\n')
                
                # Severe incidents: close the edge
                else:
                    # Create a closing rerouter for the entire edge
                    f.write(f'    <rerouter id="incident_{idx}" edges="{edge.getID()}">\n')
                    f.write(f'        <interval begin="0" end="{SIMULATION_DURATION}">\n')
                    f.write(f'            <closingReroute disallow="all"/>\n')
                    f.write(f'        </interval>\n')
                    f.write(f'    </rerouter>\n')
        
        f.write('</routes>\n')
    
    print(f"Generated traffic flows for {time_period} with {flow_id} total flows.")
    return True

def create_sumo_config(time_period):
    """Create SUMO configuration file"""
    print(f"Creating SUMO configuration for {time_period}...")
    
    config_file = f"manhattan_{time_period}.sumocfg"
    
    # Skip if file exists and we're not forcing regeneration
    if os.path.exists(config_file) and not args.force_regenerate:
        print(f"Using existing SUMO configuration for {time_period}.")
        return True
    
    config = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    <input>
        <net-file value="manhattan.net.xml"/>
        <route-files value="manhattan_flows_{time_period}.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="{SIMULATION_DURATION}"/>
        <step-length value="1.0"/>
    </time>
    <processing>
        <time-to-teleport value="300"/>
        <ignore-junction-blocker value="60"/>
        <collision.action value="warn"/>
        <random value="true"/>
    </processing>
    <report>
        <verbose value="true"/>
        <duration-log.statistics value="true"/>
        <log value="simulation_{time_period}.log"/>
    </report>
    <output>
        <tripinfo-output value="tripinfo_{time_period}.xml"/>
        <summary-output value="summary_{time_period}.xml"/>
        <statistic-output value="statistics_{time_period}.xml"/>
        <queue-output value="queue_{time_period}.xml"/>
        <lanechange-output value="lanechange_{time_period}.xml"/>
    </output>
    <gui_only>
        <gui-settings-file value="gui-settings.xml"/>
        <start value="false"/>
        <quit-on-end value="false"/>
    </gui_only>
</configuration>
"""
    
    # Create GUI settings file if it doesn't exist
    if not os.path.exists("gui-settings.xml"):
        gui_settings = """<?xml version="1.0" encoding="UTF-8"?>
<viewsettings>
    <scheme name="real world"/>
    <delay value="80"/>
    <viewport zoom="1000" x="800" y="800"/>
    <decal file=""/>
    <breakpoints-file value=""/>
    <snapshot file=""/>
</viewsettings>
"""
        with open("gui-settings.xml", "w") as f:
            f.write(gui_settings)
    
    try:
        with open(config_file, "w") as f:
            f.write(config)
            
        print(f"SUMO configuration for {time_period} created successfully.")
        return True
    except Exception as e:
        print(f"Error creating SUMO configuration: {e}")
        return False

def visualize_traffic_setup(matched_edges, time_period):
    """Create visualizations of the traffic setup for analysis"""
    print(f"Creating traffic setup visualizations for {time_period}...")
    
    # Create visualization directory
    viz_dir = os.path.join(WORKSPACE, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Load the SUMO network
    try:
        net = sumolib.net.readNet("manhattan.net.xml")
    except Exception as e:
        print(f"Error loading SUMO network: {e}")
        return False
    
    # Create a map of the matched edges
    plt.figure(figsize=(15, 12))
    
    # Function to get edge coordinates
    def get_edge_coords(edge_id):
        edge = net.getEdge(edge_id)
        if not edge:
            return None, None
        
        # Get shape
        shape = edge.getShape()
        if not shape:
            return None, None
        
        # Extract x and y coordinates
        x = [point[0] for point in shape]
        y = [point[1] for point in shape]
        return x, y
    
    # Plot all edges in light gray
    for edge in net.getEdges():
        x, y = get_edge_coords(edge.getID())
        if x and y:
            plt.plot(x, y, color='lightgray', linewidth=0.5, alpha=0.5)
    
    # Plot matched edges colored by jam factor
    for segment in matched_edges.get("matched_segments", []):
        jam_factor = segment.get("jam_factor", 0)
        color = plt.cm.RdYlGn_r(jam_factor / 10)  # Red for high jam, green for low
        
        # Vehicle density determines line width
        vehicle_density = segment.get("vehicle_density", 10)
        linewidth = min(3, max(0.5, vehicle_density / 10))
        
        # Plot each SUMO edge in this segment
        for edge_id in segment.get("sumo_edges", []):
            x, y = get_edge_coords(edge_id)
            if x and y:
                plt.plot(x, y, color=color, linewidth=linewidth, alpha=0.8)
    
    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r, norm=plt.Normalize(0, 10))
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('Jam Factor')
    
    # Add title and labels
    plt.title(f'SUMO Traffic Setup - {time_period.replace("_", " ").title()}')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f"traffic_setup_{time_period}.png"), dpi=300)
    plt.close()
    
    print(f"Visualization saved to {os.path.join(viz_dir, f'traffic_setup_{time_period}.png')}")
    return True

def run_simulation(time_period):
    """Run the SUMO simulation"""
    print(f"Running SUMO simulation for {time_period}...")
    
    config_file = f"manhattan_{time_period}.sumocfg"
    
    if not os.path.exists(config_file):
        print(f"Error: Configuration file {config_file} not found.")
        return False
    
    try:
        # First, check the routes with DUAROUTER to verify they're valid
        print("Validating routes...")
        flows_file = f"manhattan_flows_{time_period}.xml"
        validated_flows_file = f"manhattan_flows_{time_period}_valid.xml"
        
        duarouter_cmd = [
            checkBinary("duarouter"), 
            "-n", "manhattan.net.xml",
            "-r", flows_file,
            "-o", validated_flows_file,
            "--ignore-errors", "true",
            "--repair", "true"
        ]
        
        # Run duarouter and capture output
        duarouter_result = subprocess.run(duarouter_cmd, 
                                         capture_output=True, 
                                         text=True)
        
        # If duarouter was successful, use the validated routes file
        route_file = validated_flows_file if duarouter_result.returncode == 0 else flows_file
        
        # Update the config file to use the validated routes
        if duarouter_result.returncode == 0:
            print("Routes validated and repaired successfully.")
            
            # Read the existing config
            with open(config_file, "r") as f:
                config_content = f.read()
            
            # Replace the route file
            config_content = config_content.replace(
                f'<route-files value="{flows_file}"/>',
                f'<route-files value="{route_file}"/>'
            )
            
            # Write updated config
            with open(config_file, "w") as f:
                f.write(config_content)
        else:
            print("Warning: Route validation failed. Using original routes.")
            print(f"DUAROUTER output: {duarouter_result.stderr}")
        
        # Now run the simulation
        if args.gui:
            sumo_cmd = [checkBinary("sumo-gui"), "-c", config_file]
        else:
            sumo_cmd = [checkBinary("sumo"), "-c", config_file]
        
        if not args.skip_simulation:
            print(f"Starting SUMO with command: {' '.join(sumo_cmd)}")
            subprocess.run(sumo_cmd, check=True)
            print("Simulation completed successfully.")
            
            # Print information about output files
            print("\nSimulation results available in the following files:")
            print(f"- Log: simulation_{time_period}.log")
            print(f"- Trip info: tripinfo_{time_period}.xml")
            print(f"- Summary: summary_{time_period}.xml")
            print(f"- Statistics: statistics_{time_period}.xml")
            print(f"- Queue data: queue_{time_period}.xml")
            
            print("\nTo visualize the simulation, run:")
            print(f"sumo-gui -c {config_file}")
        else:
            print("Simulation phase skipped as requested.")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running simulation: {e}")
        print(f"Consider running with GUI to diagnose issues: sumo-gui -c {config_file}")
        return False
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Check SUMO installation and make sure it's in your PATH.")
        return False

def main():
    global args
    
    # Parse command line arguments
    args = parse_arguments()
    time_period = args.time_period
    
    print(f"Starting Manhattan traffic simulation for {time_period}...")
    
    # Check for HERE data availability
    if not check_here_data_availability(time_period):
        print("Error: Required HERE data not found. Please run here_data_collector.py first.")
        return False
    
    # Step 1: Download OSM map
    if not download_osm_map():
        print("Failed to download or use OSM map. Exiting.")
        return False
    
    # Step 2: Convert OSM map to SUMO network
    if not create_sumo_network():
        print("Failed to create SUMO network. Exiting.")
        return False
    
    # Step 3: Match HERE segments to SUMO edges
    matched_edges = match_here_segments_to_sumo_edges(time_period, radius=args.edge_matching_radius)
    if not matched_edges:
        print("Failed to match HERE segments to SUMO edges. Exiting.")
        return False
    
    # Step 4: Generate traffic flows
    if not generate_traffic_flows(matched_edges, time_period):
        print("Failed to generate traffic flows. Exiting.")
        return False
    
    # Step 5: Create SUMO configuration
    if not create_sumo_config(time_period):
        print("Failed to create SUMO configuration. Exiting.")
        return False
    
    # Step 6: Create visualizations if requested
    if args.visualize:
        visualize_traffic_setup(matched_edges, time_period)
    
    # Step 7: Run simulation
    run_simulation(time_period)
    
    print("\n=== Manhattan Traffic Simulation Complete ===")
    print(f"All files have been created in: {WORKSPACE}")
    print(f"\nTo analyze the simulation results for {time_period}, check:")
    print(f"- tripinfo_{time_period}.xml")
    print(f"- summary_{time_period}.xml")
    print(f"\nTo visualize the simulation with GUI, run:")
    print(f"sumo-gui -c manhattan_{time_period}.sumocfg")
    
    return True

if __name__ == "__main__":
    main()