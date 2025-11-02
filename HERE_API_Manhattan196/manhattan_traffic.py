
"""
Manhattan Traffic Simulation using HERE API and SUMO
---------------------------------------------------
This script generates a complete traffic simulation for Manhattan intersections
based on HERE API data and prepares it for SUMO simulation.

Default Bounding box: 40.75,-73.98,40.7584,-73.9688
larger Manhattan BB: BB Box for a large area of Manhattan: Latitude: 40.712178, -74.033341; Latitude: 40.759722, -73.958424
Simulation period: 1 hour peak evening traffic (17:00-18:00)

Features:
- Actual traffic flow data from HERE Traffic API
- Realistic directional flows based on HERE Matrix Routing API
- Pedestrian traffic integration
- Separate roundabout conversion utility
- Configurable for machine learning experiments

IMPORTANT: current docstring formatting necessary for compatibility
python manhattan_traffic.py --force-regenerate
Date: March 23, 2025
"""

import os
import sys
import json
import time
import random
import argparse
import requests
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# Define workspace path
WORKSPACE = "/Users/samnu/Documents/SUMO/HERE_API_Manhattan196"
os.makedirs(WORKSPACE, exist_ok=True)
os.chdir(WORKSPACE)


# Setup SUMO environment
# SUMO_HOME should point to the installation directory, not the executable
SUMO_BIN = "/opt/homebrew/bin/sumo"
print(f"SUMO executable detected at: {SUMO_BIN}")



# Infer SUMO_HOME from the executable path
# Typically it's two levels up from the bin directory
if os.path.exists(SUMO_BIN):
    # Get the directory containing the SUMO executable
    bin_dir = os.path.dirname(SUMO_BIN)
    # Go up from /opt/homebrew/bin to /opt/homebrew
    homebrew_dir = os.path.dirname(bin_dir)
    # Look for tools in homebrew/share/sumo
    SUMO_HOME = os.path.join(homebrew_dir, "share", "sumo")
    
    if os.path.exists(os.path.join(SUMO_HOME, "tools")):
        print(f"SUMO_HOME detected at: {SUMO_HOME}")
        if "SUMO_HOME" not in os.environ:
            os.environ["SUMO_HOME"] = SUMO_HOME
            
        # Add SUMO tools to Python path
        tools_path = os.path.join(SUMO_HOME, "tools")
        if tools_path not in sys.path:
            sys.path.append(tools_path)
    else:
        print("Error: SUMO tools directory not found. Please set SUMO_HOME manually.")
        sys.exit(1)
else:
    print("Error: SUMO executable not found. Please install SUMO and set SUMO_HOME.")
    sys.exit(1)

# Now we can import SUMO tools
try:
    import sumolib
    from sumolib import checkBinary
    import traci
except ImportError:
    print("Error importing SUMO libraries. Check if SUMO_HOME is correctly set.")
    sys.exit(1)

def get_sumo_version():
    """Get the installed SUMO version"""
    try:
        # Use the properly set up SUMO executable
        result = subprocess.run([SUMO_BIN, "--version"], 
                               capture_output=True, text=True)
        # Extract the full version string
        version_text = result.stdout.strip()
        # Print the full version for debugging
        print(f"SUMO version output: {version_text}")
        
        # Parse the version from output (typically looks like "SUMO Version v1.x.0")
        if version_text:
            # Try to extract version using regex
            import re
            match = re.search(r"Version\s+(?:v)?(\d+\.\d+\.\d+)", version_text)
            if match:
                version = match.group(1)
            else:
                # Fallback to first line if regex fails
                version = version_text.split('\n')[0]
        else:
            version = "unknown"
            
        print(f"Detected SUMO version: {version}")
        return version
    except Exception as e:
        print(f"Could not determine SUMO version: {e}")
        return "unknown"
sumo_version = get_sumo_version()

# HERE API configuration
HERE_API_KEY = "ejchR9sgdh8wvASbWzfpy3bfkA_PFOo3VA2-eAdQQHo"  # Add  HERE API key here

# Check if API key is set
if not HERE_API_KEY:
    print("Please enter your HERE API key:")
    HERE_API_KEY = input().strip()

# API call counter to ensure we stay within free tier limits
API_CALL_COUNTER = {
    "matrix_routing": 0,
    "traffic_flow": 0,
    "pedestrian": 0
}

# Free tier limits
API_FREE_TIER_LIMITS = {
    "matrix_routing": 2500,  # Matrix Routing API calls
    "traffic_flow": 5000,    # Traffic API calls
    "pedestrian": 1000       # Pedestrian traffic (estimated)
}

# Manhattan bounding box
BBOX = {
    "min_lat": 40.75,
    "min_lon": -73.98,
    "max_lat": 40.7584,
    "max_lon": -73.9688
}

# Simulation parameters
PEAK_HOUR_START = 17  # 5:00 PM
SIMULATION_DURATION = 3600  # 1 hour in seconds
VEHICLE_TYPES = {
    "passenger": 0.6,  # 60% passenger cars
    "taxi": 0.25,      # 25% taxis
    "delivery": 0.1,   # 10% delivery vehicles
    "bus": 0.05        # 5% buses
}

# Step 1: Download OSM map for the specified area
def download_osm_map():
    """Download OpenStreetMap data for the specified bounding box"""
    print("Downloading OpenStreetMap data...")
    
    bbox_str = f"{BBOX['min_lon']},{BBOX['min_lat']},{BBOX['max_lon']},{BBOX['max_lat']}"
    osm_api_url = f"https://api.openstreetmap.org/api/0.6/map?bbox={bbox_str}"
    
    try:
        response = requests.get(osm_api_url)
        if response.status_code == 200:
            with open(os.path.join(WORKSPACE, "manhattan.osm"), "wb") as f:
                f.write(response.content)
            print("OSM data downloaded successfully.")
            return True
        else:
            print(f"Failed to download OSM data: {response.status_code}")
            print("Using Overpass API instead...")
            
            overpass_url = "https://overpass-api.de/api/interpreter"
            overpass_query = f"""
            [out:xml];
            (
              way["highway"]({BBOX['min_lat']},{BBOX['min_lon']},{BBOX['max_lat']},{BBOX['max_lon']});
              relation["highway"]({BBOX['min_lat']},{BBOX['min_lon']},{BBOX['max_lat']},{BBOX['max_lon']});
              way["footway"]({BBOX['min_lat']},{BBOX['min_lon']},{BBOX['max_lat']},{BBOX['max_lon']});
              way["pedestrian"]({BBOX['min_lat']},{BBOX['min_lon']},{BBOX['max_lat']},{BBOX['max_lon']});
            );
            (._;>;);
            out meta;
            """
            
            response = requests.post(overpass_url, data=overpass_query)
            if response.status_code == 200:
                with open(os.path.join(WORKSPACE, "manhattan.osm"), "wb") as f:
                    f.write(response.content)
                print("OSM data downloaded successfully via Overpass API.")
                return True
            else:
                print(f"Failed to download OSM data via Overpass API: {response.status_code}")
                return False
    except Exception as e:
        print(f"Error downloading OSM data: {e}")
        return False

# Step 2: Convert OSM map to SUMO network
def create_sumo_network():
    """Convert OSM map to SUMO network"""
    print("Converting OSM data to SUMO network...")
    
    netconvert_cmd = [
        checkBinary("netconvert"),
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
    
    try:
        subprocess.run(netconvert_cmd, check=True)
        print("SUMO network created successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error creating SUMO network: {e}")
        return False
    except FileNotFoundError:
        print("netconvert command not found. Check SUMO installation.")
        return False

def safe_json_serializable(obj, _already_seen=None):
    """
    Convert objects to JSON serializable format, handling circular references
    and excessive recursion depth.
    """
    # Initialize tracking set for circular references if this is the top level call
    if _already_seen is None:
        _already_seen = set()
    
    # Get object ID to track circular references
    obj_id = id(obj)
    
    # If we've seen this object before, return a placeholder to avoid recursion
    if obj_id in _already_seen:
        return "[Circular Reference]"
    
    # For recursive calls, track this object
    _already_seen.add(obj_id)
    
    # Handle different types
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            # Skip keys that might cause problems
            if k == '_already_seen':
                continue
            # Convert key to string if not already
            k_str = str(k) if not isinstance(k, (str, int, float, bool, type(None))) else k
            try:
                result[k_str] = safe_json_serializable(v, _already_seen)
            except RecursionError:
                result[k_str] = "[Recursion too deep]"
            except Exception as e:
                result[k_str] = f"[Error: {str(e)}]"
        return result
    elif isinstance(obj, (list, tuple, set)):
        try:
            return [safe_json_serializable(item, _already_seen) for item in obj]
        except RecursionError:
            return ["[Recursion too deep]"]
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        # Handle custom objects by using their dictionary or string representation
        try:
            if hasattr(obj, '__dict__'):
                return safe_json_serializable(obj.__dict__, _already_seen)
            else:
                return str(obj)
        except:
            return str(obj)


# Step 3: Extract network information
def extract_network_info():
    """Extract junctions, edges, and connection information from SUMO network"""
    print("Extracting network information...")
    
    try:
        net = sumolib.net.readNet("manhattan.net.xml")
        
        # Extract junctions
        junctions = []
        for junction in net.getNodes():
            if junction.getType() not in ['internal', 'dead_end']:
                is_tls = False
                # More reliable way to check if junction has traffic lights
                if junction.getType() == 'traffic_light':
                    is_tls = True
                    
                # Add the junction with improved has_tls detection
                junctions.append({
                    'id': junction.getID(),
                    'x': junction.getCoord()[0],
                    'y': junction.getCoord()[1],
                    'lat': net.convertXY2LonLat(junction.getCoord()[0], junction.getCoord()[1])[1],
                    'lon': net.convertXY2LonLat(junction.getCoord()[0], junction.getCoord()[1])[0],
                    'type': junction.getType(),
                    'incoming': [edge.getID() for edge in junction.getIncoming()],
                    'outgoing': [edge.getID() for edge in junction.getOutgoing()],
                    'has_tls': is_tls,
                    'traffic_light_id': junction.getID() if is_tls else None
                })
        
        print(f"Found {len(junctions)} junctions in the network.")
        
        # Extract edges
        edges = []
        for edge in net.getEdges():
            if not edge.getID().startswith(':'):  # Skip internal edges
                # Get lane information
                lanes = []
                for lane in edge.getLanes():
                    lane_data = {
                        'id': lane.getID(),
                        'index': lane.getIndex(),
                        'width': lane.getWidth(),
                        'length': lane.getLength(),
                        'speed': lane.getSpeed()
                    }
                    
                    # Handle different SUMO API versions for allowed vehicles
                    try:
                        if hasattr(lane, 'getAllowed'):
                            lane_data['allowed'] = lane.getAllowed()
                        elif hasattr(lane, 'getAllowedVehicleClasses'):
                            lane_data['allowed'] = lane.getAllowedVehicleClasses()
                        elif hasattr(lane, 'getPermissions'):
                            lane_data['allowed'] = lane.getPermissions()
                        else:
                            # Fallback: check specific vehicle classes
                            lane_data['allowed'] = []
                            for vclass in ["passenger", "taxi", "bus", "delivery", "truck", "bicycle", "motorcycle", "pedestrian"]:
                                if hasattr(lane, 'allows') and lane.allows(vclass):
                                    lane_data['allowed'].append(vclass)
                    except Exception as e:
                        # Silent fallback
                        lane_data['allowed'] = []
                    
                    lanes.append(lane_data)
                
                # Add to edges list
                edges.append({
                    'id': edge.getID(),
                    'from': edge.getFromNode().getID(),
                    'to': edge.getToNode().getID(),
                    'length': edge.getLength(),
                    'speed': edge.getSpeed(),
                    'lane_count': edge.getLaneNumber(),
                    'lanes': lanes,
                    'shape': [net.convertXY2LonLat(x, y) for x, y in edge.getShape()],
                    'function': edge.getFunction()
                })
        
        print(f"Found {len(edges)} edges in the network.")
        
        # Extract traffic light information
        tls_list = []
        for tls in net.getTrafficLights():
            connections = []
            try:
                # This is the safer way to get connections in newer SUMO versions
                for connection in tls.getConnections():
                    # Handle different API structures
                    try:
                        conn_data = {
                            'from': connection[0].getFrom().getID(),
                            'to': connection[0].getTo().getID(),
                            'fromLane': connection[0].getFromLane().getIndex(),
                            'toLane': connection[0].getToLane().getIndex(),
                            'linkIndex': connection[1]
                        }
                    except AttributeError:
                        # Alternative approach if the above fails
                        conn_data = {
                            'from': connection[0].getID().split('_')[0],
                            'to': connection[0].getID().split('_')[1] if '_' in connection[0].getID() else '',
                            'linkIndex': connection[1]
                        }
                    connections.append(conn_data)
            except Exception as e:
                # Silent fallback
                pass
            
            tls_list.append({
                'id': tls.getID(),
                'type': 'static',  # Default, will be updated to 'actuated' in the SUMO config
                'connections': connections,
                'programs': {} if not hasattr(tls, 'getPrograms') else tls.getPrograms()
            })
        
        print(f"Found {len(tls_list)} traffic lights in the network.")
        
        # Select entry/exit points (junctions at the boundary of our network)
        entry_exit_points = []
        
        # Get min/max coordinates in the network
        x_coords = [j['x'] for j in junctions]
        y_coords = [j['y'] for j in junctions]
        if x_coords and y_coords:
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            
            # Define boundary margins (5% of total dimensions)
            x_margin = (max_x - min_x) * 0.05
            y_margin = (max_y - min_y) * 0.05
            
            # Find junctions near the boundary
            for junction in junctions:
                x, y = junction['x'], junction['y']
                is_boundary = (
                    abs(x - min_x) < x_margin or 
                    abs(x - max_x) < x_margin or 
                    abs(y - min_y) < y_margin or 
                    abs(y - max_y) < y_margin
                )
                if is_boundary:
                    entry_exit_points.append(junction)
        
        # Ensure we have at least some entry/exit points
        if not entry_exit_points:
            # If no boundary junctions found, use junctions with few connections
            for junction in junctions:
                if len(junction['incoming']) <= 2 or len(junction['outgoing']) <= 2:
                    entry_exit_points.append(junction)
        
        # Take up to 30 entry/exit points for O-D matrix to stay within API limits
        if len(entry_exit_points) > 30:
            import random
            entry_exit_points = random.sample(entry_exit_points, 30)
        
        print(f"Selected {len(entry_exit_points)} entry/exit points for O-D matrix.")
        
        # Prepare data for saving - create a simplified structure
        network_info = {
            'junctions': junctions,
            'edges': edges,
            'traffic_lights': tls_list,
            'entry_exit_points': entry_exit_points
        }
        
        # Make the data JSON serializable
        print("Preparing network data for JSON serialization...")
        serializable_network_info = safe_json_serializable(network_info)
        
        # Save network info
        print("Saving network information to JSON...")
        with open("network_info.json", "w") as f:
            json.dump(serializable_network_info, f, indent=2)
        
        # Save entry/exit points separately for convenience
        with open("entry_exit_points.json", "w") as f:
            serializable_entry_exit = safe_json_serializable(entry_exit_points)
            json.dump(serializable_entry_exit, f, indent=2)
        
        print("Network information saved successfully.")
        
        return network_info
    
    except Exception as e:
        print(f"Error extracting network information: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_network_statistics(network_info):
    """Analyze network data and return detailed statistics"""
    print("Analyzing network statistics...")
    
    # Extract basic data
    junctions = network_info.get('junctions', [])
    edges = network_info.get('edges', [])
    traffic_lights = network_info.get('traffic_lights', [])
    
    # Count different junction types
    total_junctions = len(junctions)
    real_intersections = 0
    virtual_intersections = 0
    traffic_light_junctions = 0
    junction_road_counts = {}  # Count junctions by number of connected roads
    
    for junction in junctions:
        # Check if it's a "real" intersection
        is_virtual = junction['type'] in ['internal', 'dead_end', 'unregulated', 'priority']
        is_traffic_light = junction.get('has_tls', False)
        
        if is_virtual:
            virtual_intersections += 1
        else:
            real_intersections += 1
        
        if is_traffic_light:
            traffic_light_junctions += 1

        # Count by number of roads
        road_count = len(junction['incoming']) + len(junction['outgoing'])
        if road_count in junction_road_counts:
            junction_road_counts[road_count] += 1
        else:
            junction_road_counts[road_count] = 1
    
    # Calculate area - First try based on x/y coordinates
    x_coords = []
    y_coords = []
    for junction in junctions:
        if 'x' in junction and 'y' in junction:
            x_coords.append(junction['x'])
            y_coords.append(junction['y'])
    
    # Debug prints for coordinate values
    if x_coords and y_coords:
        print(f"X coordinate range: min={min(x_coords):.2f}, max={max(x_coords):.2f}, diff={max(x_coords)-min(x_coords):.2f}")
        print(f"Y coordinate range: min={min(y_coords):.2f}, max={max(y_coords):.2f}, diff={max(y_coords)-min(y_coords):.2f}")
    
    # Calculate area in km² from x/y coordinates
    area_km2_xy = 0
    if x_coords and y_coords:
        width_meters = max(x_coords) - min(x_coords)
        height_meters = max(y_coords) - min(y_coords)
        area_km2_xy = (width_meters * height_meters) / 1000000  # Convert m² to km²
        print(f"Area calculated from x/y coordinates: {area_km2_xy:.6f} km²")
    
    # Alternative approach: Calculate area based on geo-coordinates (lat/lon)
    lat_coords = []
    lon_coords = []
    for junction in junctions:
        if 'lat' in junction and 'lon' in junction:
            lat_coords.append(junction['lat'])
            lon_coords.append(junction['lon'])
    
    # Debug prints for geo-coordinate values
    if lat_coords and lon_coords:
        print(f"Lat coordinate range: min={min(lat_coords):.6f}, max={max(lat_coords):.6f}, diff={max(lat_coords)-min(lat_coords):.6f}")
        print(f"Lon coordinate range: min={min(lon_coords):.6f}, max={max(lon_coords):.6f}, diff={max(lon_coords)-min(lon_coords):.6f}")
    
    # Calculate approximate area in km² from lat/lon using Haversine formula for corners
    area_km2_geo = 0
    if lat_coords and lon_coords:
        # Calculate width in km (approximately along equator)
        min_lat, max_lat = min(lat_coords), max(lat_coords)
        min_lon, max_lon = min(lon_coords), max(lon_coords)
        
        # Earth's circumference at equator is about 40,075 km
        # 1 degree of longitude at equator = 40075/360 = 111.32 km
        # 1 degree of longitude at latitude = 111.32 * cos(latitude)
        import math
        avg_lat_radians = math.radians((min_lat + max_lat) / 2)
        lon_km_per_degree = 111.32 * math.cos(avg_lat_radians)
        lat_km_per_degree = 111.32  # Approximately constant
        
        width_km = (max_lon - min_lon) * lon_km_per_degree
        height_km = (max_lat - min_lat) * lat_km_per_degree
        
        area_km2_geo = width_km * height_km
        print(f"Area calculated from geo-coordinates: {area_km2_geo:.6f} km²")
    
    # Choose the best area calculation (prefer geo-based if available)
    area_km2 = area_km2_geo if area_km2_geo > 0 else area_km2_xy
    
    # Calculate densities
    intersection_density = 0
    road_density = 0
    total_roads = len(edges)
    
    if area_km2 > 0:
        intersection_density = real_intersections / area_km2
        road_density = total_roads / area_km2
    
    # Compile statistics
    stats = {
        'total_junctions': total_junctions,
        'real_intersections': real_intersections,
        'virtual_intersections': virtual_intersections,
        'traffic_light_junctions': traffic_light_junctions,
        'traffic_lights': len(traffic_lights),  # Count of actual traffic light objects
        'total_roads': total_roads,
        'area_km2': round(area_km2, 4),
        'intersection_density': round(intersection_density, 2) if area_km2 > 0 else 0,
        'road_density': round(road_density, 2) if area_km2 > 0 else 0,
        'junction_road_counts': junction_road_counts
    }
    
    # Print summary
    print(f"Total junctions/intersections: {total_junctions}")
    print(f"Real intersections: {real_intersections}")
    print(f"Virtual intersections: {virtual_intersections}")
    print(f"Traffic light junctions: {traffic_light_junctions}")#actual traffic light signal controlled intersections
    print(f"Traffic light objects: {len(traffic_lights)}")#are the actual traffic signal controllers (a single controller can manage multiple intersections in complex junctions)
    print(f"Area: {area_km2:.4f} km²")
    print(f"Total roads: {total_roads}")
    print(f"Intersection density: {intersection_density:.2f} per km²")
    print(f"Road density: {road_density:.2f} per km²")
    print(f"Junction road connections: {junction_road_counts}")
    
    return stats

# Step 4: Get actual traffic flow data from HERE Traffic API
def get_traffic_flow_data():
    """Get actual traffic flow data from HERE Traffic API"""
    print("Requesting traffic flow data from HERE Traffic API...")
    
    # Prepare bbox string
    bbox_str = f"{BBOX['min_lon']},{BBOX['min_lat']},{BBOX['max_lon']},{BBOX['max_lat']}"    
    # HERE Traffic Flow API endpoint
    traffic_url = "https://data.traffic.hereapi.com/v7/flow"
    
    params = {
        "apiKey": HERE_API_KEY,
        "locationReferencing": "shape",
        "in": f"bbox:{bbox_str}",
        "functionalClass": [1, 2, 3, 4, 5],  # All road types
        "returnJamFactor": "true",
        "returnTravelTime": "true",
        "returnFreeFlow": "true"
    }
    
    try:
        response = requests.get(traffic_url, params=params)
        API_CALL_COUNTER["traffic_flow"] += 1
        
        if response.status_code == 200:
            data = response.json()
            # Add a timestamp to the data
            data['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            # Save raw traffic data
            with open("traffic_flow_data.json", "w") as f:
                json.dump(data, f, indent=2)
                
            print(f"Traffic flow data retrieved successfully with {len(data.get('results', []))} records.")
            
            # Process traffic flow to extract vehicle counts, speeds, etc.
            traffic_data = process_traffic_flow_data(data)
            
            # Save processed traffic data
            with open("processed_traffic_data.json", "w") as f:
                json.dump(traffic_data, f, indent=2)
                
            return traffic_data
        else:
            print(f"Failed to get traffic flow data: {response.status_code}")
            print(f"Response: {response.text}")
            return None
    except Exception as e:
        print(f"Error getting traffic flow data: {e}")
        return None

def create_sumo_config():
    """Create SUMO configuration file"""
    print("Creating SUMO configuration...")
    
    config = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    <input>
        <net-file value="manhattan.net.xml"/>
        <route-files value="manhattan_flows.xml"/>
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
        <log value="simulation.log"/>
    </report>
    <output>
        <tripinfo-output value="tripinfo.xml"/>
        <summary-output value="summary.xml"/>
        <statistic-output value="statistics.xml"/>
        <queue-output value="queue.xml"/>
        <lanechange-output value="lanechange.xml"/>
    </output>
    <gui_only>
        <gui-settings-file value="gui-settings.xml"/>
        <start value="false"/>
        <quit-on-end value="false"/>
    </gui_only>
</configuration>
"""
    
    # Create GUI settings file
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
    
    try:
        with open("manhattan_sim.sumocfg", "w") as f:
            f.write(config)
            
        with open("gui-settings.xml", "w") as f:
            f.write(gui_settings)
            
        print("SUMO configuration created successfully.")
        return True
    except Exception as e:
        print(f"Error creating SUMO configuration: {e}")
        return False

def calculate_directional_flows(travel_data, entry_exit_points):
    """Calculate directional flows based on travel times"""
    print("Calculating directional flows...")
    
    # Create a lookup table for point coordinates
    point_lookup = {}
    for idx, point in enumerate(entry_exit_points):
        point_lookup[idx] = point
    
    directional_flows = []
    
    # Calculate flow for each origin-destination pair
    for item in travel_data:
        origin_idx = item.get("origin_idx")
        dest_idx = item.get("dest_idx")
        
        # Skip self-loops
        if origin_idx == dest_idx:
            continue
        
        # Get points
        origin = point_lookup.get(origin_idx)
        destination = point_lookup.get(dest_idx)
        
        if not origin or not destination:
            continue
        
        # Calculate direction (N-S, E-W, NE-SW, NW-SE)
        origin_lat, origin_lon = origin["lat"], origin["lon"]
        dest_lat, dest_lon = destination["lat"], destination["lon"]
        
        # Calculate deltas
        d_lat = dest_lat - origin_lat
        d_lon = dest_lon - origin_lon
        
        # Determine direction
        if abs(d_lat) > abs(d_lon) * 2:
            # Primarily North-South
            direction = "N-S" if d_lat > 0 else "S-N"
        elif abs(d_lon) > abs(d_lat) * 2:
            # Primarily East-West
            direction = "E-W" if d_lon > 0 else "W-E"
        elif d_lat * d_lon > 0:
            # Diagonal NE-SW
            direction = "NE-SW" if d_lat > 0 else "SW-NE"
        else:
            # Diagonal NW-SE
            direction = "NW-SE" if d_lat > 0 else "SE-NW"
        
        # Flow weight is inverse proportional to travel time
        travel_time = item.get("travel_time", 1)
        
        # Avoid division by zero
        if travel_time <= 0:
            travel_time = 1
            
        # Calculate flow weight (higher for shorter travel times)
        flow_weight = 1000.0 / travel_time
        
        # Add to directional flows
        directional_flows.append({
            "origin_id": origin["id"],
            "destination_id": destination["id"],
            "origin_idx": origin_idx,
            "destination_idx": dest_idx,
            "travel_time": travel_time,
            "distance": item.get("distance"),
            "direction": direction,
            "flow_weight": flow_weight
        })
    
    return directional_flows


def process_traffic_flow_data(data):
    """Process raw traffic flow data to extract useful metrics"""
    print("Processing traffic flow data...")
    
    # Load network data
    net = sumolib.net.readNet("manhattan.net.xml")
    
    # Initialize storage for processed data
    processed_data = {
        "edges": {},         # Edge-based traffic data
        "total_vehicles": 0, # Estimated total vehicles in the area
        "avg_speed": 0,      # Average speed in the area
        "jam_factor": 0,     # Average jam factor in the area
        "flow_records": 0    # Number of flow records processed
    }
    
    # Process each flow item
    flow_items = data.get("results", [])
    
    jam_factors = []
    speeds = []
    total_length = 0  # Total road length in meters
    
    for item in flow_items:
        # Get traffic flow data
        current_flow = item.get("currentFlow", {})
        
        if not current_flow:
            continue
            
        # Get jam factor, speed
        jam_factor = current_flow.get("jamFactor", 0)
        speed = current_flow.get("speed", 0)
        
        # Get road information
        location = item.get("location", {})
        
        # Handle the new shape structure
        shape_data = location.get("shape", {})
        links = shape_data.get("links", [])
        
        if not links:
            continue
        
        # Process all links
        for link in links:
            points = link.get("points", [])
            if len(points) < 2:
                continue
                
            # Extract coordinates
            coords = []
            for point in points:
                lat = point.get("lat")
                lng = point.get("lng")
                if lat is not None and lng is not None:
                    coords.append((lng, lat))
            
            # Skip if insufficient coordinates
            if len(coords) < 2:
                continue
                
            # Calculate length of segment - use the provided length if available
            segment_length = link.get("length", 0)
            
            # If length not provided, calculate it
            if segment_length <= 0:
                segment_length = 0
                for i in range(len(coords) - 1):
                    x1, y1 = coords[i]
                    x2, y2 = coords[i+1]
                    # Simplified distance calculation (Euclidean)
                    segment_length += ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                
                # Convert to meters (approximate)
                segment_length *= 111000  # Roughly 111km per degree
            
            # Estimate vehicle count based on length, speed, and jam factor

            # Higher jam factor means more vehicles

            # Formula: jam_factor ranges from 0 to 10, with 10 being completely jammed

            # We can estimate vehicles per km based on jam factor:

            # jam_factor 0-2: 10 vehicles/km

            # jam_factor 2-5: 20-50 vehicles/km

            # jam_factor 5-8: 50-100 vehicles/km

            # jam_factor 8-10: 100-200 vehicles/km
            if jam_factor < 2:
                vehicles_per_km = 10 + jam_factor * 5
            elif jam_factor < 5:
                vehicles_per_km = 20 + (jam_factor - 2) * 10
            elif jam_factor < 8:
                vehicles_per_km = 50 + (jam_factor - 5) * 16.7
            else:
                vehicles_per_km = 100 + (jam_factor - 8) * 33.3
                
            # Calculate vehicles for this segment
            segment_vehicles = vehicles_per_km * (segment_length / 1000)
            
            # Find nearest SUMO edge(s)
            nearest_edges = []
            for i in range(len(coords) - 1):
                # Convert to xy coordinates
                x1, y1 = net.convertLonLat2XY(coords[i][0], coords[i][1])
                x2, y2 = net.convertLonLat2XY(coords[i+1][0], coords[i+1][1])
                
                # Find nearest edge for this segment
                radius = 50  # 50m radius
                edges = net.getNeighboringEdges(x1, y1, radius)
                
                if edges:
                    # Sort by distance
                    edges.sort(key=lambda x: x[1])
                    # Take closest edge
                    nearest_edge, distance = edges[0]
                    
                    # Skip pedestrian and special edges
                    if nearest_edge.allows("passenger"):
                        edge_id = nearest_edge.getID()
                        
                        # Initialize edge data if not exists
                        if edge_id not in processed_data["edges"]:
                            processed_data["edges"][edge_id] = {
                                "speed": 0,
                                "jam_factor": 0,
                                "vehicle_count": 0,
                                "records": 0
                            }
                        
                        # Calculate subsegment length (approximate proportion of original segment)
                        subsegment_length = segment_length / (len(coords) - 1)
                        subsegment_vehicles = vehicles_per_km * (subsegment_length / 1000)
                        
                        # Update edge data
                        edge_data = processed_data["edges"][edge_id]
                        edge_data["speed"] = (edge_data["speed"] * edge_data["records"] + speed) / (edge_data["records"] + 1)
                        edge_data["jam_factor"] = (edge_data["jam_factor"] * edge_data["records"] + jam_factor) / (edge_data["records"] + 1)
                        edge_data["vehicle_count"] += subsegment_vehicles
                        edge_data["records"] += 1
                        
                        nearest_edges.append(edge_id)
            
            # Update totals
            processed_data["total_vehicles"] += segment_vehicles
            jam_factors.append(jam_factor)
            speeds.append(speed)
            total_length += segment_length
            processed_data["flow_records"] += 1
    
    # Calculate averages
    if jam_factors:
        processed_data["jam_factor"] = sum(jam_factors) / len(jam_factors)
    if speeds:
        processed_data["avg_speed"] = sum(speeds) / len(speeds)
        
    # Round total vehicles to integer
    processed_data["total_vehicles"] = int(processed_data["total_vehicles"])
    
    # Add density calculation
    if total_length > 0:
        processed_data["vehicle_density"] = processed_data["total_vehicles"] / (total_length / 1000)  # vehicles per km
    else:
        processed_data["vehicle_density"] = 0
    
    print(f"Processed {processed_data['flow_records']} flow records.")
    print(f"Estimated total vehicles: {processed_data['total_vehicles']}")
    print(f"Average jam factor: {processed_data['jam_factor']:.2f}")
    print(f"Average speed: {processed_data['avg_speed']:.2f} km/h")
    print(f"Vehicle density: {processed_data['vehicle_density']:.2f} vehicles/km")
    
    return processed_data

# Step 5: Get pedestrian data using HERE API
def get_pedestrian_data():
    """Get pedestrian data for the specified area"""
    print("Estimating pedestrian counts...")
    
    # Unfortunately, HERE API doesn't directly provide pedestrian counts
    # We'll use a combination of location analytics and time-based estimates
    
    # Load network data
    net = sumolib.net.readNet("manhattan.net.xml")
    
    # Initialize pedestrian data
    pedestrian_data = {
        "crossings": {},  # Pedestrian crossings data
        "total_pedestrians": 0
    }
    
    # Get all pedestrian crossings in the network
    crossings = []
    for edge in net.getEdges():
        # Skip non-pedestrian edges
        if edge.getFunction() == "crossing":
            crossings.append(edge.getID())
    
    print(f"Found {len(crossings)} pedestrian crossings in the network.")
    
    # For Manhattan, typical peak hour pedestrian counts can be estimated
    # Midtown Manhattan can have 3,000-5,000 pedestrians per hour at a busy intersection
    # Since our area is relatively small, let's use a more moderate estimate
    
    total_pedestrians = 10000  # Total pedestrians for the whole area
    
    # Distribute pedestrians across crossings
    if crossings:
        avg_pedestrians_per_crossing = total_pedestrians / len(crossings)
        
        # Distribute with some randomness
        for crossing_id in crossings:
            # Add +/- 30% random variation
            variation = random.uniform(0.7, 1.3)
            pedestrian_count = int(avg_pedestrians_per_crossing * variation)
            
            pedestrian_data["crossings"][crossing_id] = {
                "pedestrian_count": pedestrian_count,
                "flow_rate": pedestrian_count / SIMULATION_DURATION  # pedestrians per second
            }
            
            pedestrian_data["total_pedestrians"] += pedestrian_count
    
    print(f"Estimated total pedestrians: {pedestrian_data['total_pedestrians']}")
    
    # Save pedestrian data
    with open("pedestrian_data.json", "w") as f:
        json.dump(pedestrian_data, f, indent=2)
    
    return pedestrian_data

# Step 6: Request HERE Matrix Routing API for travel times and directions
def get_travel_times(entry_exit_points):
    """Request travel times between entry/exit points using HERE Matrix Routing API"""
    print("Requesting travel times from HERE Matrix Routing API...")
    
    # Prepare the day of week and time for peak hour traffic
    #calculating a future Wednesday at the peak hour time to send 
    # as the departureTime parameter to the Matrix Routing API.
    weekday = 3  # Wednesday, middle of the week
    today = datetime.now()
    # Find next weekday
    days_ahead = weekday - today.weekday()
    if days_ahead <= 0:
        days_ahead += 7
    next_weekday = today + timedelta(days=days_ahead)
    peak_time = next_weekday.replace(hour=PEAK_HOUR_START, minute=0, second=0)
    # Format as ISO 8601 with timezone (RFC 3339 compliant)
    departure_time = peak_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    
    # Structure for HERE Matrix Routing API
    here_matrix_url = "https://matrix.router.hereapi.com/v8/matrix"
    
    # Prepare origins and destinations
    origins = []
    destinations = []
    
    for idx, point in enumerate(entry_exit_points):
        location = {
            "lat": point["lat"],
            "lng": point["lon"]
        }
        origins.append(location)
        destinations.append(location)
    
    # Split requests to stay within free tier limits
    # HERE allows up to 100 origins/destinations points per request in free tier
    max_points_per_request = 15  # Lower this to ensure we stay within limits
    travel_data = []
    
    # Function to process one batch
    def process_batch(batch_origins, batch_destinations):
        params = {
            "apiKey": HERE_API_KEY,
            "async": "false",
            "departureTime": departure_time
        }
        
        payload = {
            "origins": batch_origins,
            "destinations": batch_destinations,
            "regionDefinition": {
                "type": "autoCircle"
            },
            "matrixAttributes": ["travelTimes", "distances"],
            "transportMode": "car",
            "departureTime": departure_time
        }
        
        headers = {'Content-Type': 'application/json'}
        
        try:
            response = requests.post(here_matrix_url, params=params, json=payload, headers=headers)
            API_CALL_COUNTER["matrix_routing"] += 1
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract matrix dimensions and data
                matrix = result.get("matrix", {})
                travel_times = matrix.get("travelTimes", [])
                distances = matrix.get("distances", [])
                num_origins = matrix.get("numOrigins", 0)
                num_destinations = matrix.get("numDestinations", 0)
                """
                travelTimes and distances are 1D arrays with 9 values (3 origins × 3 destinations)
                The matrix dimensions are provided in numOrigins (3) and numDestinations (3)
                Values are organized as a flattened matrix:

                Index 0: Origin 0 to Destination 0
                Index 1: Origin 0 to Destination 1
                """

                batch_data = []
                for i in range(num_origins):
                    for j in range(num_destinations):
                        # Calculate the index in the flattened array
                        flat_index = i * num_destinations + j
                        
                        # Ensure the index is valid
                        if flat_index < len(travel_times) and flat_index < len(distances):
                            travel_time = travel_times[flat_index]
                            distance = distances[flat_index]
                            
                            # Calculate the global indices
                            global_origin_idx = i
                            global_dest_idx = j
                            
                            batch_data.append({
                                "origin_idx": global_origin_idx,
                                "dest_idx": global_dest_idx,
                                "travel_time": travel_time,
                                "distance": distance
                            })
                
                return batch_data
            else:
                print(f"API request failed: {response.status_code}")
                print(f"Response: {response.text}")
                return []
        except Exception as e:
            print(f"Error requesting travel times: {e}")
            return []
    
    # Check if we would exceed free tier limits
    total_api_calls = 0
    for i in range(0, len(origins), max_points_per_request):
        for j in range(0, len(destinations), max_points_per_request):
            total_api_calls += 1
    
    print(f"Estimated API calls required: {total_api_calls}")
    
    if total_api_calls + API_CALL_COUNTER["matrix_routing"] > API_FREE_TIER_LIMITS["matrix_routing"]:
        print("Warning: This would exceed the free tier limit.")
        print(f"Reducing matrix size to stay within limits.")
        
        # Reduce the number of points to stay within limits
        max_api_calls = API_FREE_TIER_LIMITS["matrix_routing"] - API_CALL_COUNTER["matrix_routing"]
        
        # Calculate max points we can process with available API calls
        max_points = int((max_api_calls ** 0.5) * max_points_per_request)
        
        # Trim the origins and destinations lists
        if max_points < len(origins):
            origins = origins[:max_points]
            destinations = destinations[:max_points]
            print(f"Reduced to {len(origins)} origins and destinations.")
    
    # Process in batches to stay within API limits
    for i in range(0, len(origins), max_points_per_request):
        batch_origins = origins[i:i+max_points_per_request]
        
        for j in range(0, len(destinations), max_points_per_request):
            batch_destinations = destinations[j:j+max_points_per_request]
            
            print(f"Requesting batch {i//max_points_per_request + 1}.{j//max_points_per_request + 1}...")
            batch_data = process_batch(batch_origins, batch_destinations)
            
            if batch_data:
                travel_data.extend(batch_data)
            
            # Respect API rate limits
            time.sleep(1)
    
    # Save raw matrix data
    with open("travel_matrix_raw.json", "w") as f:
        json.dump(travel_data, f, indent=2)
    
    # Process the data to create a directional flow matrix
    directional_flows = calculate_directional_flows(travel_data, entry_exit_points)
    
    # Save directional flow data
    with open("directional_flows.json", "w") as f:
        json.dump(directional_flows, f, indent=2)
    
    print(f"Travel time matrix created with {len(travel_data)} entries.")
    print(f"Directional flow data calculated with {len(directional_flows)} origin-destination pairs.")
    
    return {
        'travel_data': travel_data,
        'directional_flows': directional_flows
    }

    
    # Process
    # Step 12: Run the simulation
def run_simulation():
    """Run the SUMO simulation"""
    print("Running SUMO simulation...")
    
    try:
        # First, check the routes with DUAROUTER to verify they're valid
        print("Validating routes...")
        duarouter_cmd = [
            checkBinary("duarouter"), 
            "-n", "manhattan.net.xml",
            "-r", "manhattan_flows.xml",
            "-o", "manhattan_flows_valid.xml",
            "--ignore-errors", "true",
            "--repair", "true"
        ]
        
        # Run duarouter and capture output
        duarouter_result = subprocess.run(duarouter_cmd, 
                                         capture_output=True, 
                                         text=True)
        
        # If duarouter was successful, use the validated routes file
        route_file = "manhattan_flows_valid.xml" if duarouter_result.returncode == 0 else "manhattan_flows.xml"
        
        # Update the config file to use the validated routes
        if duarouter_result.returncode == 0:
            print("Routes validated and repaired successfully.")
            
            # Read the existing config
            with open("manhattan_sim.sumocfg", "r") as f:
                config_content = f.read()
            
            # Replace the route file
            config_content = config_content.replace(
                '<route-files value="manhattan_flows.xml"/>',
                f'<route-files value="{route_file}"/>'
            )
            
            # Write updated config
            with open("manhattan_sim.sumocfg", "w") as f:
                f.write(config_content)
        else:
            print("Warning: Route validation failed. Using original routes.")
            print(f"DUAROUTER output: {duarouter_result.stderr}")
            
        # Now run the simulation
        sumo_cmd = [checkBinary("sumo"), "-c", "manhattan_sim.sumocfg"]
        subprocess.run(sumo_cmd, check=True)
        print("Simulation completed successfully.")
        print("\nTo visualize the simulation, run:")
        print("sumo-gui -c manhattan_sim.sumocfg")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running simulation: {e}")
        print("Consider running with GUI to diagnose issues:")
        print("sumo-gui -c manhattan_sim.sumocfg")
        return False
    except FileNotFoundError:
        print("SUMO command not found. Check SUMO installation.")
        return False


def check_route_validity(from_edge, to_edge, net):
    """Check if there's a valid route between two edges"""
    try:
        # Try to compute a route between the edges
        from_node = net.getEdge(from_edge).getToNode()
        to_node = net.getEdge(to_edge).getFromNode()
        
        # Use SUMO's built-in routing algorithm
        router = sumolib.routing.dijkstra.Router(net.getEdges())
        route = router.computePath(from_node, to_node, 0)
        
        # If we got a route with at least one edge, it's valid
        return len(route) > 0
    except Exception:
        # Any exception means the route is invalid
        return False

def generate_traffic_demand(traffic_data, directional_flows, network_info):
    """Generate SUMO traffic demand files based on actual traffic data and directional flows"""
    print("Generating traffic demand...")
    
    # Get total vehicle count from actual traffic data
    total_vehicles = traffic_data.get("total_vehicles", 0) if traffic_data else 0
    
    # If no traffic data available, use a default value (high density for Manhattan)
    if total_vehicles <= 0:
        total_vehicles = 25000  # Higher estimate for Manhattan density
        print(f"No traffic count available. Using estimated total: {total_vehicles} vehicles")
    else:
        print(f"Using actual traffic count: {total_vehicles} vehicles")
    
    # Calculate flow weights for each origin-destination pair
    if not directional_flows:
        print("No directional flow data available. Using uniform distribution.")
        # Create fallback flow data
        entry_exit_points = network_info.get('entry_exit_points', [])
        if not entry_exit_points:
            print("No entry/exit points available. Cannot generate traffic demand.")
            return False
        
        # Create basic flows from entry points to exit points
        flows = []
        for origin in entry_exit_points:
            for destination in entry_exit_points:
                if origin['id'] != destination['id']:
                    flows.append({
                        "origin": origin['id'],
                        "destination": destination['id'],
                        "count": total_vehicles // (len(entry_exit_points) * (len(entry_exit_points) - 1)),
                        "direction": "unknown"
                    })
    else:
        # Calculate flow weights for each origin-destination pair
        flow_weights = [flow.get("flow_weight", 1.0) for flow in directional_flows]
        total_weight = sum(flow_weights)
        
        # Normalize to get probability distribution
        if total_weight > 0:
            probabilities = [weight / total_weight for weight in flow_weights]
        else:
            # Fallback to uniform distribution
            probabilities = [1.0 / len(directional_flows)] * len(directional_flows)
        
        # Distribute vehicles according to probabilities
        vehicle_counts = np.random.multinomial(total_vehicles, probabilities)
        
        # Create flows
        flows = []
        for idx, count in enumerate(vehicle_counts):
            if count > 0:
                flow = directional_flows[idx]
                
                flows.append({
                    "origin": flow.get("origin_id"),
                    "destination": flow.get("destination_id"),
                    "count": int(count),
                    "direction": flow.get("direction", "unknown")
                })
    
    # Get valid edge IDs from the network
    valid_edges = set()
    for edge in network_info.get('edges', []):
        valid_edges.add(edge['id'])
    
    # # Only create flows with valid edge IDs
    # valid_flows = []
    # for flow in flows:
    #     if flow['origin'] in valid_edges and flow['destination'] in valid_edges:
    #         valid_flows.append(flow)

    # Create junction-to-edge mapping
    junction_edge_map = {}
    for edge in network_info.get('edges', []):
        from_node = edge['from']
        if from_node not in junction_edge_map:
            junction_edge_map[from_node] = []
        junction_edge_map[from_node].append(edge['id'])

    # Create mapping for edges going into each junction
    junction_incoming_map = {}
    for edge in network_info.get('edges', []):
        to_node = edge['to']
        if to_node not in junction_incoming_map:
            junction_incoming_map[to_node] = []
        junction_incoming_map[to_node].append(edge['id'])

    # Load the network for route checking
    net = sumolib.net.readNet("manhattan.net.xml")

    # Modify the flow validation to use junction IDs to find valid edges and check route validity
    valid_flows = []
    invalid_route_count = 0

    for flow in flows:
        # Check if the origin and destination are junctions
        origin_edges = junction_edge_map.get(flow['origin'], [])
        destination_incoming_edges = junction_incoming_map.get(flow['destination'], [])
        
        # If we found valid edges for both, create a valid flow with route checking
        if origin_edges and destination_incoming_edges:
            # Try different combinations of edges until we find a valid route
            route_valid = False
            
            # Try up to 3 origin edges and 3 destination edges
            for origin in origin_edges[:3]:
                for destination in destination_incoming_edges[:3]:
                    if check_route_validity(origin, destination, net):
                        # Found a valid route
                        new_flow = flow.copy()
                        new_flow['origin'] = origin
                        new_flow['destination'] = destination
                        valid_flows.append(new_flow)
                        route_valid = True
                        break  # Break inner loop
                if route_valid:
                    break  # Break outer loop if route found
            
            if not route_valid:
                invalid_route_count += 1
                    
        # If the original IDs are valid edges, check if the route is valid
        elif flow['origin'] in valid_edges and flow['destination'] in valid_edges:
            if check_route_validity(flow['origin'], flow['destination'], net):
                valid_flows.append(flow)
            else:
                invalid_route_count += 1
            
    print(f"Keeping {len(valid_flows)} valid flows out of {len(flows)} total flows")
    print(f"Discarded {invalid_route_count} flows due to invalid routes")

    # Fallback: If no valid flows were created, create some basic ones
    if not valid_flows and total_vehicles > 0:
        print("Warning: No valid flows created. Generating fallback flows.")
        
        # Get a list of all edges that can be used as origins (have outgoing connections)
        origin_edges = []
        destination_edges = []
        
        for edge_id in valid_edges:
            edge = next((e for e in network_info.get('edges', []) if e['id'] == edge_id), None)
            if edge:
                # Skip internal edges and pedestrian-only edges
                if edge.get('function') == 'internal' or not any(any('passenger' in allowed for allowed in lane.get('allowed', [])) for lane in edge.get('lanes', [])):
                    continue
                    
                # Check if this edge has outgoing connections
                to_node = edge.get('to')
                from_node = edge.get('from')
                
                # If we can find the to_node in the from field of other edges, it has outgoing connections
                has_outgoing = any(e.get('from') == to_node for e in network_info.get('edges', []))
                # If we can find the from_node in the to field of other edges, it has incoming connections
                has_incoming = any(e.get('to') == from_node for e in network_info.get('edges', []))
                
                if has_outgoing:
                    origin_edges.append(edge_id)
                if has_incoming:
                    destination_edges.append(edge_id)
        
        # Create basic flows between valid edges
        num_flows = min(20, len(origin_edges), len(destination_edges))  # Limit to 20 flows
        
        if num_flows > 0:
            vehicles_per_flow = total_vehicles // num_flows
            
            for i in range(num_flows):
                origin = origin_edges[i % len(origin_edges)]
                destination = destination_edges[(i + len(destination_edges)//2) % len(destination_edges)]
                
                # Make sure origin and destination are different
                if origin != destination:
                    valid_flows.append({
                        "origin": origin,
                        "destination": destination,
                        "count": vehicles_per_flow,
                        "direction": "fallback"
                    })
            
            print(f"Created {len(valid_flows)} fallback flows")

    # Create vehicle types XML
    with open("manhattan_vtypes.xml", "w") as f:
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
    with open("manhattan_flows.xml", "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n')
        
        f.write('    <include href="manhattan_vtypes.xml"/>\n\n')
        
        # CHANGE THIS: Replace "flows" with "valid_flows"
        # Create flows for each vehicle type according to distribution
        flow_id = 0
        
        for flow in valid_flows:  # CHANGED from "flows" to "valid_flows"
            for vtype, proportion in VEHICLE_TYPES.items():
                vtype_count = max(1, int(flow["count"] * proportion))
                
                # Determine departure timing spread over the simulation duration
                # This creates more realistic traffic patterns
                begin_time = random.randint(0, 300)  # Start within first 5 minutes
                end_time = SIMULATION_DURATION
                
                f.write(f'    <flow id="flow_{flow_id}_{vtype}" type="{vtype}" from="{flow["origin"]}" to="{flow["destination"]}" begin="{begin_time}" end="{end_time}" number="{vtype_count}" departLane="best" departSpeed="max"/>\n')
                flow_id += 1
        
        # Add pedestrian flows if we have pedestrian data
        if 'pedestrian_data' in locals() and pedestrian_data:
            for crossing_id, crossing_data in pedestrian_data.get('crossings', {}).items():
                # Extract pedestrian count
                ped_count = crossing_data.get('pedestrian_count', 0)
                if ped_count > 0:
                    # Create a personFlow (pedestrian group)
                    f.write(f'    <personFlow id="pflow_{crossing_id}" begin="0" end="{SIMULATION_DURATION}" number="{ped_count}">\n')
                    f.write(f'        <walk from="{crossing_id}" to="{crossing_id}" departPos="random" arrivalPos="random"/>\n')
                    f.write(f'    </personFlow>\n')
        
        f.write('</routes>\n')
    
    print(f"Generated traffic demand with {len(valid_flows)} flows for {total_vehicles} vehicles.")
    
    # Save flow information for analysis
    with open("traffic_flows.json", "w") as f:
        json.dump(valid_flows, f, indent=2)  # CHANGED from "flows" to "valid_flows"
    
    return valid_flows  # CHANGED from "flows" to "valid_flows"

def prepare_simulation_from_collected_data():
    """Prepare simulation using previously collected real data"""
    # Create directory for simulation
    os.makedirs("simulation", exist_ok=True)
    
    # Find the latest traffic flow and matrix data
    flow_files = sorted(Path("traffic_data").glob("traffic_flow_*.json"))
    matrix_files = sorted(Path("traffic_data").glob("travel_matrix_*.json"))
    
    if not flow_files or not matrix_files:
        print("No collected traffic data found. Please run collect_traffic_data.py first.")
        return False
    
    latest_flow_file = flow_files[-1]
    latest_matrix_file = matrix_files[-1]
    
    print(f"Using traffic flow data: {latest_flow_file}")
    print(f"Using travel matrix data: {latest_matrix_file}")
    
    # Process the data and generate traffic demand
    with open(latest_flow_file, "r") as f:
        flow_data = json.load(f)
    
    with open(latest_matrix_file, "r") as f:
        matrix_data = json.load(f)
    
    # Process traffic flow data
    processed_flow = process_traffic_flow_data(flow_data)
    
    # Process matrix data
    entry_exit_points = get_entry_exit_points()
    directional_flows = calculate_directional_flows(matrix_data, entry_exit_points)
    
    # Generate traffic demand
    flows = generate_traffic_demand(processed_flow, directional_flows, extract_network_info())
    
    return flows is not None
    
def create_roundabout_converter():
    """Create utility script for converting intersections to roundabouts"""
    print("Creating roundabout converter script...")
    
    script = """#!/usr/bin/env python3
# Roundabout Converter for Manhattan Traffic Simulation
# This script converts selected traffic light junctions to roundabouts

import os
import sys
import json
import random
import argparse
import subprocess
from pathlib import Path

# Define workspace path
WORKSPACE = os.path.dirname(os.path.abspath(__file__))
os.chdir(WORKSPACE)

# Setup SUMO environment (copied from manhattan_traffic.py)
SUMO_HOME = "/opt/homebrew/bin/sumo"
if SUMO_HOME and os.path.exists(SUMO_HOME):
    if "SUMO_HOME" not in os.environ:
        os.environ["SUMO_HOME"] = SUMO_HOME
        
    # Add SUMO tools to Python path
    tools_path = os.path.join(SUMO_HOME, "tools")
    if tools_path not in sys.path:
        sys.path.append(tools_path)
else:
    print("Error: SUMO installation not found. Please install SUMO and set SUMO_HOME.")
    sys.exit(1)

# Import SUMO libraries
try:
    import sumolib
    from sumolib import checkBinary
except ImportError:
    print("Error importing SUMO libraries. Check if SUMO_HOME is correctly set.")
    sys.exit(1)

def load_network_info():
    #Load network information from JSON file
    try:
        with open("network_info.json", "r") as f:
            network_info = json.load(f)
        return network_info
    except Exception as e:
        print(f"Error loading network information: {e}")
        return None

def convert_junction_to_roundabout(junction_id, max_lanes=2, add_signals=True):
    '''Convert a specific junction to a roundabout with customized configuration
    
    Args:
        junction_id: ID of the junction to convert
        max_lanes: Maximum number of lanes for the roundabout (default: 2)
        add_signals: Whether to add traffic signals at roundabout entrances
    '''
    
    print(f"Converting junction {junction_id} to roundabout (max {max_lanes} lanes)...")
    
    # Load original network
    net = sumolib.net.readNet("manhattan.net.xml")
    
    # Check if junction exists
    junction = net.getNode(junction_id)
    if not junction:
        print(f"Junction {junction_id} not found in network.")
        return False
    
    # Create a temporary typefile to control lane numbers
    typefile = "roundabout_types.xml"
    with open(typefile, "w") as f:
        f.write('''<?xml version="1.0" encoding="UTF-8"?>
<types xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/types_file.xsd">
    <type id="roundabout" priority="1" numLanes="{}" speed="13.89"/>
</types>'''.format(max_lanes))
        
    
    # Prepare netconvert command for roundabout conversion
    output_file = "manhattan_roundabouts.net.xml"
    
    netconvert_cmd = [
        checkBinary("netconvert"),
        "--sumo-net-file", "manhattan.net.xml",
        "--output-file", output_file,
        "--roundabouts.guess", "true",
        "--junction.join", "true",
        "--type-files", typefile,
        "--no-turnarounds", "true",
        "--geometry.remove", "true"  # Clean up geometry
    ]
    
    # Add specific junction to convert
    netconvert_cmd.extend(["--roundabouts.explicit", junction_id])
    
    try:
        subprocess.run(netconvert_cmd, check=True)
        print(f"Junction {junction_id} converted to roundabout successfully.")
        
        # Add traffic signals at roundabout entrances if requested
        if add_signals:
            add_roundabout_signals(junction_id, output_file)
        
        # Create new SUMO config for roundabout version
        create_roundabout_config()
        
        # Clean up temporary files
        if os.path.exists(typefile):
            os.remove(typefile)
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error converting junction to roundabout: {e}")
        if os.path.exists(typefile):
            os.remove(typefile)
        return False

def add_roundabout_signals(junction_id, net_file):
    '''Add traffic signals at the entrances of a roundabout
    
    Args:
        junction_id: ID of the roundabout junction
        net_file: Path to the network file
    '''
    
    print(f"Adding traffic signals to roundabout {junction_id}...")
    
    # Load the modified network
    modified_net = sumolib.net.readNet(net_file)
    
    # Find the roundabout node
    roundabout = None
    for node in modified_net.getNodes():
        if node.getID() == junction_id or "roundabout" in node.getID():
            roundabout = node
            break
    
    if not roundabout:
        print(f"Roundabout node {junction_id} not found in modified network.")
        return False
    
    # Create additional file for traffic light control
    tls_file = "roundabout_tls.add.xml"
    with open(tls_file, "w") as f:
        f.write('''<?xml version="1.0" encoding="UTF-8"?>
<additional xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/additional_file.xsd">
    <!-- Traffic light controller for roundabout -->
    <tlLogic id="{}_tls" type="actuated" programID="adaptive" offset="0">
        <phase duration="10000" state="G" minDur="5" maxDur="120"/>
        <phase duration="3"  state="y" minDur="3" maxDur="3"/>
        <phase duration="2"  state="r" minDur="2" maxDur="2"/>
        <param key="show-detectors" value="true"/>
        <param key="file" value="roundabout_tls_state.xml"/>
        <param key="freq" value="10"/>
    </tlLogic>
</additional>'''.format(junction_id))
    
    print(f"Created traffic light controller for roundabout {junction_id}.")
    
    # Update the SUMO config to include the additional file
    update_roundabout_config_with_tls(tls_file)
    
    return True

def update_roundabout_config_with_tls(tls_file):
    '''Update the SUMO configuration to include traffic light control'''
    try:
        # Read existing config
        with open("manhattan_roundabouts.sumocfg", "r") as f:
            config_content = f.read()
        
        # Check if additional files are already defined
        if "<additional-files" in config_content:
            # Update existing additional-files line
            config_content = config_content.replace(
                "<additional-files value=\"", 
                f"<additional-files value=\"{tls_file},"
            )
        else:
            # Add new additional-files line after route-files
            config_content = config_content.replace(
                "</input>",
                f"        <additional-files value=\"{tls_file}\"/>\n    </input>"
            )
        
        # Write updated config
        with open("manhattan_roundabouts.sumocfg", "w") as f:
            f.write(config_content)
        
        print("Updated simulation configuration with traffic light controls.")
        return True
    except Exception as e:
        print(f"Error updating configuration with traffic light controls: {e}")
        return False

def convert_random_junctions(count, max_lanes=2, add_signals=True):
    '''Convert a random selection of junctions to roundabouts
    
    Args:
        count: Number of junctions to convert
        max_lanes: Maximum number of lanes for roundabouts
        add_signals: Whether to add traffic signals at roundabout entrances
    '''
    print(f"Converting {count} random junctions to roundabouts (max {max_lanes} lanes)...")
    
    # Load network info
    network_info = load_network_info()
    if not network_info:
        return False
    
    # Get all traffic light junctions
    tl_junctions = []
    for junction in network_info.get('junctions', []):
        if junction.get('has_tls', False):
            tl_junctions.append(junction)
    
    print(f"Found {len(tl_junctions)} traffic light junctions.")
    
    # Check if we have enough junctions
    if len(tl_junctions) < count:
        print(f"Warning: Requested {count} junctions but only {len(tl_junctions)} available.")
        count = len(tl_junctions)
    
    # Select random junctions
    selected_junctions = random.sample(tl_junctions, count)
    
    # Get junction IDs
    junction_ids = [junction['id'] for junction in selected_junctions]
    
    # Create comma-separated list of junction IDs
    junction_list = ",".join(junction_ids)
    
    # Create a temporary typefile to control lane numbers
    typefile = "roundabout_types.xml"
    with open(typefile, "w") as f:
        f.write('''<?xml version="1.0" encoding="UTF-8"?>
<types xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/types_file.xsd">
    <type id="roundabout" priority="1" numLanes="{}" speed="13.89"/>
</types>'''.format(max_lanes))
    
    # Prepare netconvert command
    output_file = "manhattan_roundabouts.net.xml"
    
    netconvert_cmd = [
        checkBinary("netconvert"),
        "--sumo-net-file", "manhattan.net.xml",
        "--output-file", output_file,
        "--roundabouts.guess", "true",
        "--junction.join", "true",
        "--type-files", typefile,
        "--no-turnarounds", "true",
        "--geometry.remove", "true"  # Clean up geometry
    ]
    
    # Add specific junctions to convert
    netconvert_cmd.extend(["--roundabouts.explicit", junction_list])
    
    try:
        subprocess.run(netconvert_cmd, check=True)
        print(f"Converted {count} junctions to roundabouts successfully.")
        
        # Add traffic signals to roundabouts if requested
        if add_signals:
            for junction_id in junction_ids:
                add_roundabout_signals(junction_id, output_file)
        
        # Save converted junction IDs for reference
        with open("converted_junctions.json", "w") as f:
            json.dump({
                "converted_junctions": junction_ids,
                "max_lanes": max_lanes,
                "add_signals": add_signals
            }, f, indent=2)
        
        # Create new SUMO config for roundabout version
        create_roundabout_config()
        
        # Clean up temporary files
        if os.path.exists(typefile):
            os.remove(typefile)
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error converting junctions to roundabouts: {e}")
        if os.path.exists(typefile):
            os.remove(typefile)
        return False

def convert_all_suitable_junctions(max_lanes=2, add_signals=True):
    '''Convert all suitable junctions to roundabouts
    
    Args:
        max_lanes: Maximum number of lanes for roundabouts
        add_signals: Whether to add traffic signals at roundabout entrances
    '''
    print(f"Converting all suitable junctions to roundabouts (max {max_lanes} lanes)...")
    
    # Create a temporary typefile to control lane numbers
    typefile = "roundabout_types.xml"
    with open(typefile, "w") as f:
        f.write('''<?xml version="1.0" encoding="UTF-8"?>
<types xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/types_file.xsd">
    <type id="roundabout" priority="1" numLanes="{}" speed="13.89"/>
</types>'''.format(max_lanes))
    
    # Prepare netconvert command to let SUMO guess all roundabouts
    output_file = "manhattan_roundabouts.net.xml"
    
    netconvert_cmd = [
        checkBinary("netconvert"),
        "--sumo-net-file", "manhattan.net.xml",
        "--output-file", output_file,
        "--roundabouts.guess", "true",
        "--junction.join", "true",
        "--type-files", typefile,
        "--no-turnarounds", "true",
        "--geometry.remove", "true"  # Clean up geometry
    ]
    
    try:
        subprocess.run(netconvert_cmd, check=True)
        print("Converted suitable junctions to roundabouts.")
        
        # Add traffic signals if requested
        if add_signals:
            # We need to find all the created roundabouts
            modified_net = sumolib.net.readNet(output_file)
            roundabout_nodes = []
            
            for node in modified_net.getNodes():
                if "roundabout" in node.getType().lower():
                    roundabout_nodes.append(node.getID())
            
            print(f"Found {len(roundabout_nodes)} roundabouts to add signals to.")
            
            for node_id in roundabout_nodes:
                add_roundabout_signals(node_id, output_file)
        
        # Create new SUMO config for roundabout version
        create_roundabout_config()
        
        # Clean up temporary files
        if os.path.exists(typefile):
            os.remove(typefile)
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error converting junctions to roundabouts: {e}")
        if os.path.exists(typefile):
            os.remove(typefile)
        return False

def create_roundabout_config():
    '''Create SUMO configuration for roundabout network'''
    config = f'''<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    <input>
        <net-file value="manhattan_roundabouts.net.xml"/>
        <route-files value="manhattan_flows.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="3600"/>
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
        <log value="roundabout_simulation.log"/>
    </report>
    <output>
        <tripinfo-output value="roundabout_tripinfo.xml"/>
        <summary-output value="roundabout_summary.xml"/>
        <statistic-output value="roundabout_statistics.xml"/>
        <queue-output value="roundabout_queue.xml"/>
        <lanechange-output value="roundabout_lanechange.xml"/>
    </output>
    <gui_only>
        <gui-settings-file value="gui-settings.xml"/>
        <start value="false"/>
        <quit-on-end value="false"/>
    </gui_only>
</configuration>
'''
    
    try:
        with open("manhattan_roundabouts.sumocfg", "w") as f:
            f.write(config)
        print("Created roundabout simulation configuration.")
        return True
    except Exception as e:
        print(f"Error creating roundabout configuration: {e}")
        return False

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Convert traffic light junctions to roundabouts')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--junction-id', action='append', help='Specific junction ID to convert')
    group.add_argument('--random', type=int, help='Number of random junctions to convert')
    group.add_argument('--all', action='store_true', help='Convert all suitable junctions')
    
    # Add optional parameters
    parser.add_argument('--max-lanes', type=int, default=2, help='Maximum number of lanes for roundabouts (default: 2)')
    parser.add_argument('--no-signals', action='store_true', help='Do not add traffic signals at roundabout entrances')
    
    args = parser.parse_args()
    add_signals = not args.no_signals
    
    if args.junction_id:
        for junction_id in args.junction_id:
            convert_junction_to_roundabout(junction_id, args.max_lanes, add_signals)
    elif args.random:
        convert_random_junctions(args.random, args.max_lanes, add_signals)
    elif args.all:
        convert_all_suitable_junctions(args.max_lanes, add_signals)
    
    print("Roundabout conversion complete.")
    print("To run the roundabout simulation, use:")
    print("sumo-gui -c manhattan_roundabouts.sumocfg")

if __name__ == "__main__":
    main()
"""
    
    try:
        with open("convert_to_roundabouts.py", "w") as f:
            f.write(script)
        
        # Make script executable
        os.chmod("convert_to_roundabouts.py", 0o755)
        
        print("Roundabout converter script created successfully.")
        return True
    except Exception as e:
        print(f"Error creating roundabout converter script: {e}")
        return False

def create_data_analysis_tools():
    """Create utility script for analyzing simulation results"""
    print("Creating data analysis script...")
    
    script = """#!/usr/bin/env python3
# Results Analysis Script for Manhattan Traffic Simulation
# This script analyzes the output from SUMO simulations

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import xml.etree.ElementTree as ET

# Define workspace path
WORKSPACE = os.path.dirname(os.path.abspath(__file__))
os.chdir(WORKSPACE)

# Setup SUMO environment (similar to manhattan_traffic.py)
SUMO_HOME = "/opt/homebrew/bin/sumo"
if SUMO_HOME and os.path.exists(SUMO_HOME):
    if "SUMO_HOME" not in os.environ:
        os.environ["SUMO_HOME"] = SUMO_HOME
        
    # Add SUMO tools to Python path
    tools_path = os.path.join(SUMO_HOME, "tools")
    if tools_path not in sys.path:
        sys.path.append(tools_path)
else:
    print("Error: SUMO installation not found. Please install SUMO and set SUMO_HOME.")
    sys.exit(1)

# Import SUMO libraries
try:
    import sumolib
except ImportError:
    print("Error importing SUMO libraries. Check if SUMO_HOME is correctly set.")
    sys.exit(1)

def analyze_trip_info(file_path="tripinfo.xml"):
    '''Analyze trip information from SUMO output'''
    print(f"Analyzing trip information from {file_path}...")
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return None
    
    # Parse XML file
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Extract trip data
        trips = []
        for trip in root.findall('tripinfo'):
            trip_data = {
                'id': trip.get('id'),
                'vehicle_type': trip.get('vType'),
                'depart': float(trip.get('depart')),
                'arrival': float(trip.get('arrival')),
                'duration': float(trip.get('duration')),
                'wait_time': float(trip.get('waitingTime')),
                'distance': float(trip.get('routeLength')),
                'avg_speed': float(trip.get('routeLength')) / max(0.1, float(trip.get('duration')))
            }
            trips.append(trip_data)
        
        # Convert to DataFrame
        df = pd.DataFrame(trips)
        
        # Calculate statistics
        stats = {
            'total_trips': len(trips),
            'completed_trips': len(trips),  # All trips in the file are completed
            'avg_duration': df['duration'].mean(),
            'avg_wait_time': df['wait_time'].mean(),
            'avg_distance': df['distance'].mean(),
            'avg_speed': df['avg_speed'].mean(),
            'total_wait_time': df['wait_time'].sum(),
            'max_wait_time': df['wait_time'].max(),
            'min_wait_time': df['wait_time'].min()
        }
        
        # Add statistics by vehicle type
        vtype_stats = {}
        for vtype in df['vehicle_type'].unique():
            vtype_df = df[df['vehicle_type'] == vtype]
            vtype_stats[vtype] = {
                'count': len(vtype_df),
                'avg_duration': vtype_df['duration'].mean(),
                'avg_wait_time': vtype_df['wait_time'].mean(),
                'avg_speed': vtype_df['avg_speed'].mean()
            }
        
        stats['vehicle_types'] = vtype_stats
        
        return {
            'stats': stats,
            'dataframe': df
        }
    
    except Exception as e:
        print(f"Error analyzing trip information: {e}")
        return None

def analyze_queue_data(file_path="queue.xml"):
    '''Analyze queue information from SUMO output'''
    print(f"Analyzing queue information from {file_path}...")
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return None
    
    try:
        # Parse XML file
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Extract queue data for each lane
        queue_data = {}
        
        for step in root.findall('data/timestep'):
            time = float(step.get('time'))
            
            for lane in step.findall('lane'):
                lane_id = lane.get('id')
                queue_length = float(lane.get('queueing_length', 0))
                queue_vehicles = int(lane.get('queueing_vehicles', 0))
                
                if lane_id not in queue_data:
                    queue_data[lane_id] = {
                        'times': [],
                        'queue_lengths': [],
                        'queue_vehicles': []
                    }
                
                queue_data[lane_id]['times'].append(time)
                queue_data[lane_id]['queue_lengths'].append(queue_length)
                queue_data[lane_id]['queue_vehicles'].append(queue_vehicles)
        
        # Calculate statistics for each lane
        lane_stats = {}
        
        for lane_id, data in queue_data.items():
            # Convert to numpy arrays for calculations
            queue_lengths = np.array(data['queue_lengths'])
            queue_vehicles = np.array(data['queue_vehicles'])
            
            lane_stats[lane_id] = {
                'max_queue_length': np.max(queue_lengths),
                'avg_queue_length': np.mean(queue_lengths),
                'max_queued_vehicles': np.max(queue_vehicles),
                'avg_queued_vehicles': np.mean(queue_vehicles),
                'time_with_queue': np.sum(queue_lengths > 0) / len(queue_lengths) * 100  # Percentage of time with queue
            }
        
        # Calculate global statistics
        global_stats = {
            'total_lanes_with_queues': len(lane_stats),
            'max_queue_any_lane': max(s['max_queue_length'] for s in lane_stats.values()) if lane_stats else 0,
            'avg_queue_all_lanes': np.mean([s['avg_queue_length'] for s in lane_stats.values()]) if lane_stats else 0,
            'avg_queued_vehicles': np.mean([s['avg_queued_vehicles'] for s in lane_stats.values()]) if lane_stats else 0
        }
        
        return {
            'lane_stats': lane_stats,
            'global_stats': global_stats,
            'raw_data': queue_data
        }
    
    except Exception as e:
        print(f"Error analyzing queue information: {e}")
        return None

def analyze_summary(file_path="summary.xml"):
    '''Analyze summary information from SUMO output'''
    print(f"Analyzing summary information from {file_path}...")
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return None
    
    try:
        # Parse XML file
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Extract summary data for each timestep
        summary_data = {
            'time': [],
            'vehicles': [],
            'running': [],
            'waiting': [],
            'mean_speed': [],
            'mean_waiting_time': []
        }
        
        for step in root.findall('step'):
            summary_data['time'].append(float(step.get('time')))
            summary_data['vehicles'].append(int(step.get('loaded')))
            summary_data['running'].append(int(step.get('running')))
            summary_data['waiting'].append(int(step.get('waiting')))
            summary_data['mean_speed'].append(float(step.get('meanSpeed', 0)))
            summary_data['mean_waiting_time'].append(float(step.get('meanWaitingTime', 0)))
        
        # Calculate statistics
        stats = {
            'simulation_duration': max(summary_data['time']) if summary_data['time'] else 0,
            'max_vehicles': max(summary_data['vehicles']) if summary_data['vehicles'] else 0,
            'max_running': max(summary_data['running']) if summary_data['running'] else 0,
            'max_waiting': max(summary_data['waiting']) if summary_data['waiting'] else 0,
            'avg_mean_speed': np.mean(summary_data['mean_speed']) if summary_data['mean_speed'] else 0,
            'min_mean_speed': min(summary_data['mean_speed']) if summary_data['mean_speed'] else 0,
            'max_mean_waiting_time': max(summary_data['mean_waiting_time']) if summary_data['mean_waiting_time'] else 0,
            'avg_mean_waiting_time': np.mean(summary_data['mean_waiting_time']) if summary_data['mean_waiting_time'] else 0
        }
        
        return {
            'stats': stats,
            'timeseries': summary_data
        }
    
    except Exception as e:
        print(f"Error analyzing summary information: {e}")
        return None

def analyze_junction_performance(trip_info, queue_data, network_info=None):
    '''Analyze performance at each junction'''
    print("Analyzing junction performance...")
    
    # Load network info if not provided
    if network_info is None:
        try:
            with open("network_info.json", "r") as f:
                network_info = json.load(f)
        except Exception as e:
            print(f"Error loading network information: {e}")
            return None
    
    # Load SUMO network to get lane-junction mapping
    net = sumolib.net.readNet("manhattan.net.xml")
    
    # Create junction-lane mapping
    junction_lanes = {}
    
    for junction in network_info.get('junctions', []):
        junction_id = junction.get('id')
        
        # Get all incoming lanes to this junction
        incoming_lanes = []
        
        for edge_id in junction.get('incoming', []):
            edge = net.getEdge(edge_id)
            if edge:
                for lane in edge.getLanes():
                    incoming_lanes.append(lane.getID())
        
        junction_lanes[junction_id] = incoming_lanes
    
    # Calculate performance metrics for each junction
    junction_performance = []
    
    for junction_id, lanes in junction_lanes.items():
        # Get queue data for incoming lanes
        queue_stats = {}
        max_queue_length = 0
        avg_queue_length = 0
        lanes_with_data = 0
        
        for lane_id in lanes:
            if lane_id in queue_data.get('lane_stats', {}):
                lane_stats = queue_data['lane_stats'][lane_id]
                queue_stats[lane_id] = lane_stats
                
                max_queue_length = max(max_queue_length, lane_stats['max_queue_length'])
                avg_queue_length += lane_stats['avg_queue_length']
                lanes_with_data += 1
        
        # Calculate average if we have data
        if lanes_with_data > 0:
            avg_queue_length /= lanes_with_data
        
        # Find junction in network info to get coordinates
        junction_info = next((j for j in network_info.get('junctions', []) if j.get('id') == junction_id), None)
        
        junction_data = {
            'junction_id': junction_id,
            'max_queue_length': max_queue_length,
            'avg_queue_length': avg_queue_length,
            'lanes_with_data': lanes_with_data,
            'total_lanes': len(lanes),
            'junction_type': junction_info.get('type') if junction_info else 'unknown',
            'has_traffic_light': junction_info.get('has_tls', False) if junction_info else False,
            'coordinates': {
                'lat': junction_info.get('lat') if junction_info else None,
                'lon': junction_info.get('lon') if junction_info else None
            }
        }
        
        # Calculate throughput for this junction (vehicles passing through)
        # This is approximate and would need detailed route information for more accuracy
        throughput = 0
        for lane_id in lanes:
            # Count vehicles that passed through these lanes
            if trip_info and 'dataframe' in trip_info:
                # We don't have direct lane information in trip_info, so this is a rough estimate
                throughput += len(trip_info['dataframe']) / len(junction_lanes)
        
        junction_data['throughput'] = int(throughput)
        
        # Add to results
        junction_performance.append(junction_data)
    
    return junction_performance

def compare_results(baseline_file, current_metrics):
    '''Compare current results with baseline'''
    print(f"Comparing with baseline metrics from {baseline_file}...")
    
    try:
        # Load baseline metrics
        with open(baseline_file, "r") as f:
            baseline = json.load(f)
        
        # Calculate improvements/degradations
        comparison = {
            'global': {},
            'vehicle_types': {},
            'junctions': {}
        }
        
        # Compare global metrics
        for key in current_metrics.get('global', {}):
            if key in baseline.get('global', {}):
                baseline_value = baseline['global'][key]
                current_value = current_metrics['global'][key]
                
                if baseline_value > 0:
                    percent_change = (current_value - baseline_value) / baseline_value * 100
                else:
                    percent_change = 0 if current_value == 0 else float('inf')
                
                comparison['global'][key] = {
                    'baseline': baseline_value,
                    'current': current_value,
                    'diff': current_value - baseline_value,
                    'percent_change': percent_change
                }
        
        # Compare vehicle type metrics
        for vtype in set(list(current_metrics.get('vehicle_types', {}).keys()) + 
                         list(baseline.get('vehicle_types', {}).keys())):
            if vtype in baseline.get('vehicle_types', {}) and vtype in current_metrics.get('vehicle_types', {}):
                comparison['vehicle_types'][vtype] = {}
                
                for key in current_metrics['vehicle_types'][vtype]:
                    if key in baseline['vehicle_types'][vtype]:
                        baseline_value = baseline['vehicle_types'][vtype][key]
                        current_value = current_metrics['vehicle_types'][vtype][key]
                        
                        if baseline_value > 0:
                            percent_change = (current_value - baseline_value) / baseline_value * 100
                        else:
                            percent_change = 0 if current_value == 0 else float('inf')
                        
                        comparison['vehicle_types'][vtype][key] = {
                            'baseline': baseline_value,
                            'current': current_value,
                            'diff': current_value - baseline_value,
                            'percent_change': percent_change
                        }
        
        # Compare junction metrics
        baseline_junctions = {j['junction_id']: j for j in baseline.get('junctions', [])}
        current_junctions = {j['junction_id']: j for j in current_metrics.get('junctions', [])}
        
        comparison['junctions'] = []
        
        for junction_id in set(list(baseline_junctions.keys()) + list(current_junctions.keys())):
            if junction_id in baseline_junctions and junction_id in current_junctions:
                junction_comparison = {
                    'junction_id': junction_id,
                    'metrics': {}
                }
                
                for key in ['max_queue_length', 'avg_queue_length', 'throughput']:
                    if key in baseline_junctions[junction_id] and key in current_junctions[junction_id]:
                        baseline_value = baseline_junctions[junction_id][key]
                        current_value = current_junctions[junction_id][key]
                        
                        if baseline_value > 0:
                            percent_change = (current_value - baseline_value) / baseline_value * 100
                        else:
                            percent_change = 0 if current_value == 0 else float('inf')
                        
                        junction_comparison['metrics'][key] = {
                            'baseline': baseline_value,
                            'current': current_value,
                            'diff': current_value - baseline_value,
                            'percent_change': percent_change
                        }
                
                comparison['junctions'].append(junction_comparison)
        
        return comparison
    
    except Exception as e:
        print(f"Error comparing results: {e}")
        return None

def create_visualizations(results, output_dir="."):
    '''Create visualizations of simulation results'''
    print("Creating visualizations...")
    
    try:
        # Create output directory if needed
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract summary timeseries data
        if 'summary' in results and 'timeseries' in results['summary']:
            timeseries = results['summary']['timeseries']
            
            # Plot vehicles over time
            plt.figure(figsize=(10, 6))
            plt.plot(timeseries['time'], timeseries['running'], label='Running')
            plt.plot(timeseries['time'], timeseries['waiting'], label='Waiting')
            plt.xlabel('Time (s)')
            plt.ylabel('Number of Vehicles')
            plt.title('Vehicles Over Time')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'vehicles_over_time.png'))
            plt.close()
            
            # Plot mean speed over time
            plt.figure(figsize=(10, 6))
            plt.plot(timeseries['time'], timeseries['mean_speed'])
            plt.xlabel('Time (s)')
            plt.ylabel('Mean Speed (m/s)')
            plt.title('Mean Speed Over Time')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'mean_speed_over_time.png'))
            plt.close()
            
            # Plot mean waiting time over time
            plt.figure(figsize=(10, 6))
            plt.plot(timeseries['time'], timeseries['mean_waiting_time'])
            plt.xlabel('Time (s)')
            plt.ylabel('Mean Waiting Time (s)')
            plt.title('Mean Waiting Time Over Time')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'mean_waiting_time_over_time.png'))
            plt.close()
        
        # Create junction performance heatmap
        if 'junctions' in results:
            junction_data = results['junctions']
            
            # Extract data for heatmap
            junction_ids = [j['junction_id'] for j in junction_data]
            queue_lengths = [j['avg_queue_length'] for j in junction_data]
            throughputs = [j['throughput'] for j in junction_data]
            
            # Sort by throughput
            sorted_indices = np.argsort(throughputs)[::-1]  # Descending order
            
            # Take top 20 junctions for readability
            if len(sorted_indices) > 20:
                sorted_indices = sorted_indices[:20]
            
            sorted_junction_ids = [junction_ids[i] for i in sorted_indices]
            sorted_queue_lengths = [queue_lengths[i] for i in sorted_indices]
            sorted_throughputs = [throughputs[i] for i in sorted_indices]
            
            # Create horizontal bar chart for throughput
            plt.figure(figsize=(12, 8))
            y_pos = np.arange(len(sorted_junction_ids))
            plt.barh(y_pos, sorted_throughputs, align='center')
            plt.yticks(y_pos, sorted_junction_ids)
            plt.xlabel('Throughput (vehicles)')
            plt.title('Junction Throughput (Top 20)')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'junction_throughput.png'))
            plt.close()
            
            # Create horizontal bar chart for queue length
            plt.figure(figsize=(12, 8))
            y_pos = np.arange(len(sorted_junction_ids))
            plt.barh(y_pos, sorted_queue_lengths, align='center')
            plt.yticks(y_pos, sorted_junction_ids)
            plt.xlabel('Average Queue Length (m)')
            plt.title('Junction Average Queue Length (Top 20 Junctions by Throughput)')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'junction_queue_length.png'))
            plt.close()
        
        # If we have trip info, create trip statistics visualizations
        if 'trip_info' in results and 'dataframe' in results['trip_info']:
            df = results['trip_info']['dataframe']
            
            # Plot distribution of trip durations
            plt.figure(figsize=(10, 6))
            plt.hist(df['duration'], bins=30)
            plt.xlabel('Trip Duration (s)')
            plt.ylabel('Number of Trips')
            plt.title('Distribution of Trip Durations')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'trip_duration_distribution.png'))
            plt.close()
            
            # Plot distribution of trip distances
            plt.figure(figsize=(10, 6))
            plt.hist(df['distance'], bins=30)
            plt.xlabel('Trip Distance (m)')
            plt.ylabel('Number of Trips')
            plt.title('Distribution of Trip Distances')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'trip_distance_distribution.png'))
            plt.close()
            
            # Plot distribution of trip speeds
            plt.figure(figsize=(10, 6))
            plt.hist(df['avg_speed'], bins=30)
            plt.xlabel('Average Speed (m/s)')
            plt.ylabel('Number of Trips')
            plt.title('Distribution of Trip Speeds')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'trip_speed_distribution.png'))
            plt.close()
            
            # Plot statistics by vehicle type
            if 'vehicle_types' in results['trip_info']['stats']:
                vtype_stats = results['trip_info']['stats']['vehicle_types']
                
                vtypes = list(vtype_stats.keys())
                counts = [vtype_stats[vt]['count'] for vt in vtypes]
                avg_durations = [vtype_stats[vt]['avg_duration'] for vt in vtypes]
                avg_wait_times = [vtype_stats[vt]['avg_wait_time'] for vt in vtypes]
                avg_speeds = [vtype_stats[vt]['avg_speed'] for vt in vtypes]
                
                # Plot vehicle type counts
                plt.figure(figsize=(10, 6))
                plt.bar(vtypes, counts)
                plt.ylabel('Number of Vehicles')
                plt.title('Vehicle Counts by Type')
                plt.savefig(os.path.join(output_dir, 'vehicle_type_counts.png'))
                plt.close()
                
                # Plot vehicle type metrics
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                axes[0].bar(vtypes, avg_durations)
                axes[0].set_ylabel('Avg. Duration (s)')
                axes[0].set_title('Average Trip Duration by Vehicle Type')
                
                axes[1].bar(vtypes, avg_wait_times)
                axes[1].set_ylabel('Avg. Wait Time (s)')
                axes[1].set_title('Average Wait Time by Vehicle Type')
                
                axes[2].bar(vtypes, avg_speeds)
                axes[2].set_ylabel('Avg. Speed (m/s)')
                axes[2].set_title('Average Speed by Vehicle Type')
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'vehicle_type_metrics.png'))
                plt.close()
        
        print(f"Visualizations saved to {output_dir}")
        return True
    
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        return False

def main():
    '''Main execution function for analysis'''
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze SUMO simulation results')
    parser.add_argument('--tripinfo', default='tripinfo.xml', help='Path to tripinfo.xml file')
    parser.add_argument('--queue', default='queue.xml', help='Path to queue.xml file')
    parser.add_argument('--summary', default='summary.xml', help='Path to summary.xml file')
    parser.add_argument('--output-file', default='metrics.json', help='Output file for metrics (JSON)')
    parser.add_argument('--compare', help='Compare with baseline metrics file')
    parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    parser.add_argument('--output-dir', default='visualizations', help='Output directory for visualizations')
    parser.add_argument('--roundabouts', action='store_true', help='Analyze roundabout simulation')
    
    args = parser.parse_args()
    
    # Update file paths for roundabout simulation if needed
    if args.roundabouts:
        if args.tripinfo == 'tripinfo.xml':
            args.tripinfo = 'roundabout_tripinfo.xml'
        if args.queue == 'queue.xml':
            args.queue = 'roundabout_queue.xml'
        if args.summary == 'summary.xml':
            args.summary = 'roundabout_summary.xml'
        if args.output_file == 'metrics.json':
            args.output_file = 'roundabout_metrics.json'
    
    # Load network information
    try:
        with open("network_info.json", "r") as f:
            network_info = json.load(f)
    except Exception as e:
        print(f"Warning: Could not load network information: {e}")
        network_info = None
    
    # Check if converted junctions file exists (for roundabout analysis)
    converted_junctions = []
    if os.path.exists("converted_junctions.json"):
        try:
            with open("converted_junctions.json", "r") as f:
                converted_junctions = json.load(f).get("converted_junctions", [])
            print(f"Found {len(converted_junctions)} converted junctions.")
        except Exception as e:
            print(f"Warning: Could not load converted junctions information: {e}")
    
    # Analyze results
    print("Analyzing simulation results...")
    results = {}
    
    # Analyze trip information
    trip_info = analyze_trip_info(args.tripinfo)
    if trip_info:
        results['trip_info'] = trip_info['stats']
    
    # Analyze queue data
    queue_data = analyze_queue_data(args.queue)
    if queue_data:
        results['queue'] = queue_data['global_stats']
    
    # Analyze summary
    summary = analyze_summary(args.summary)
    if summary:
        results['summary'] = summary['stats']
        results['timeseries'] = summary['timeseries']
    
    # Analyze junction performance
    if trip_info and queue_data:
        junction_performance = analyze_junction_performance(trip_info, queue_data, network_info)
        if junction_performance:
            results['junctions'] = junction_performance
            
            # Add indicator for converted junctions (if doing roundabout analysis)
            if converted_junctions:
                for junction in results['junctions']:
                    junction['converted_to_roundabout'] = junction['junction_id'] in converted_junctions
    
    # Combine all results into a single metrics structure
    global_metrics = {}
    
    # Add trip info metrics
    if 'trip_info' in results:
        for key, value in results['trip_info'].items():
            if key != 'vehicle_types':
                global_metrics[key] = value
    
    # Add queue metrics
    if 'queue' in results:
        for key, value in results['queue'].items():
            global_metrics[key] = value
    
    # Add summary metrics
    if 'summary' in results:
        for key, value in results['summary'].items():
            global_metrics[key] = value
    
    # Create final metrics structure
    metrics = {
        'global': global_metrics,
        'vehicle_types': results.get('trip_info', {}).get('vehicle_types', {}),
        'junctions': results.get('junctions', []),
        'metadata': {
            'simulation_type': 'roundabout' if args.roundabouts else 'baseline',
            'tripinfo_file': args.tripinfo,
            'queue_file': args.queue,
            'summary_file': args.summary,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    }
    
    # Save metrics to file
    try:
        with open(args.output_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {args.output_file}")
    except Exception as e:
        print(f"Error saving metrics: {e}")
    
    # Compare with baseline if requested
    if args.compare:
        comparison = compare_results(args.compare, metrics)
        if comparison:
            comparison_file = f"comparison_{os.path.basename(args.output_file)}"
            try:
                with open(comparison_file, 'w') as f:
                    json.dump(comparison, f, indent=2)
                print(f"Comparison saved to {comparison_file}")
            except Exception as e:
                print(f"Error saving comparison: {e}")
    
    # Create visualizations if requested
    if args.visualize:
        if create_visualizations(results, args.output_dir):
            print(f"Visualizations created in {args.output_dir}")
    
    # Print key metrics
    print("\nKey Metrics:")
    print(f"Total Trips: {global_metrics.get('total_trips', 'N/A')}")
    print(f"Average Trip Duration: {global_metrics.get('avg_duration', 'N/A'):.2f} s")
    print(f"Average Wait Time: {global_metrics.get('avg_wait_time', 'N/A'):.2f} s")
    print(f"Average Speed: {global_metrics.get('avg_speed', 'N/A'):.2f} m/s")
    print(f"Max Queue Length: {global_metrics.get('max_queue_any_lane', 'N/A'):.2f} m")
    
    print("\nAnalysis complete.")

if __name__ == "__main__":
    main()
"""
    
    try:
        with open("analyze_results.py", "w") as f:
            f.write(script)
        
        # Make script executable
        os.chmod("analyze_results.py", 0o755)
        
        print("Data analysis script created successfully.")
        return True
    except Exception as e:
        print(f"Error creating data analysis script: {e}")
        return False

def create_readme():
    """Create README.md file with documentation"""
    print("Creating README.md...")
    
    readme = """# Manhattan Traffic Simulation

This is a traffic simulation for Manhattan intersections based on HERE API data using SUMO (Simulation of Urban MObility).

## Overview

This project creates a realistic traffic simulation for Manhattan intersections using:
- Actual traffic flow data from HERE Traffic API
- Realistic directional flows based on HERE Matrix Routing API
- Pedestrian traffic integration
- Roundabout conversion for comparison studies

## Files

- `manhattan_traffic.py` - Main script to set up and run the simulation
- `manhattan.osm` - OpenStreetMap data for Manhattan
- `manhattan.net.xml` - SUMO network file
- `manhattan_vtypes.xml` - Vehicle type definitions
- `manhattan_flows.xml` - Traffic demand data
- `manhattan_sim.sumocfg` - SUMO simulation configuration
- `convert_to_roundabouts.py` - Utility to convert intersections to roundabouts
- `analyze_results.py` - Data analysis tools

## Prerequisites

- Python 3.6 or higher
- SUMO (https://www.eclipse.org/sumo/)
- HERE API key
- Required Python packages: `numpy`, `pandas`, `requests`

## Running the Simulation

### Basic Simulation

To run the basic simulation with traffic lights:

```bash
sumo -c manhattan_sim.sumocfg
```

For visualization:

```bash
sumo-gui -c manhattan_sim.sumocfg
```

Normal run - will use existing files if they exist

```bash
python manhattan_traffic.py
```

Force regenerate all files

```bash
python manhattan_traffic.py --force-regenerate
```
parser.add_argument('--skip-api-calls', action='store_true', 
                    help='Skip HERE API calls and use estimates instead')
parser.add_argument('--skip-simulation', action='store_true',
                    help='Generate files but skip running the simulation')


### Roundabout Simulations

To convert specific intersections to roundabouts:

```bash
python convert_to_roundabouts.py --junction-id <JUNCTION_ID>
```

Or convert random intersections:

```bash
python convert_to_roundabouts.py --random 5
```

Or convert all intersections:

```bash
python convert_to_roundabouts.py --all
```

Then run:

```bash
sumo-gui -c manhattan_roundabouts.sumocfg
```

### Analyzing Results

After running a simulation, analyze the results:

```bash
python analyze_results.py
```

To create visualizations:

```bash
python analyze_results.py --visualize
```

To compare with baseline:

```bash
python analyze_results.py --compare metrics.json --output-file new_metrics.json
```

## HERE API Usage

This simulation uses the following HERE APIs:

- **Traffic Flow API** - for actual traffic data
- **Matrix Routing API** - for travel times and directional flows

Requests are limited to stay within free tier limits:

- **Matrix Routing API**: 2,500 requests per month
- **Traffic API**: 5,000 requests per month

## Date

March 23, 2025
"""
    
    try:
        with open("README.md", "w") as f:
            f.write(readme)
        print("README.md created successfully.")
        return True
    except Exception as e:
        print(f"Error creating README.md: {e}")
        return False


import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Manhattan Traffic Simulation using HERE API and SUMO')
    parser.add_argument('--force-regenerate', action='store_true', 
                        help='Force regeneration of all files regardless of their existence')
    return parser.parse_args()
def main():
    # Main execution function
    print("Starting Manhattan traffic simulation workflow...")
    
    # Get command line arguments
    args = parse_arguments()
    force_regenerate = args.force_regenerate
    
    if force_regenerate:
        print("Force regeneration enabled - all files will be regenerated")
    
    # Step 1: Download OSM map
    if force_regenerate or not os.path.exists(os.path.join(WORKSPACE, "manhattan.osm")):
        if not download_osm_map():
            print("Failed to download OSM map. Exiting.")
            return False
    else:
        print("Using existing OSM map.")
    
    # Step 2: Convert OSM map to SUMO network
    if force_regenerate or not os.path.exists(os.path.join(WORKSPACE, "manhattan.net.xml")):
        if not create_sumo_network():
            print("Failed to create SUMO network. Exiting.")
            return False
    else:
        print("Using existing SUMO network.")
    
    # Step 3: Extract network information
    network_info = None
    if force_regenerate or not os.path.exists(os.path.join(WORKSPACE, "network_info.json")):
        print("Network info file not found or regeneration forced. Extracting network information...")
        network_info = extract_network_info()
        if not network_info:
            print("Failed to extract network information. Exiting.")
            return False
    else:
        try:
            print("Loading existing network information...")
            with open("network_info.json", "r") as f:
                network_info = json.load(f)
                
            # Load entry/exit points separately if needed
            if not 'entry_exit_points' in network_info and os.path.exists("entry_exit_points.json"):
                with open("entry_exit_points.json", "r") as f:
                    network_info['entry_exit_points'] = json.load(f)
                    
        except json.JSONDecodeError as e:
            print(f"Error loading network_info.json: {e}")
            print("Re-generating network information...")
            network_info = extract_network_info()
            if not network_info:
                print("Failed to extract network information. Exiting.")
                return False
    
    # Step 3.5: Analyze network statistics
    if network_info:
        network_stats = analyze_network_statistics(network_info)
        
        with open("network_statistics.json", "w") as f:
            json.dump(network_stats, f, indent=2)
        print("Network statistics saved to network_statistics.json")

    # Step 4: Get actual traffic flow data
    traffic_data = None
    if force_regenerate or not os.path.exists(os.path.join(WORKSPACE, "processed_traffic_data.json")):
        print("DEBUG: processed_traffic_data.json doesn't exist or regeneration forced")
        # Check if we can make API calls without exceeding limits
        if API_CALL_COUNTER["traffic_flow"] < API_FREE_TIER_LIMITS["traffic_flow"]:
            traffic_data = get_traffic_flow_data()
            if not traffic_data:
                print("Warning: Failed to get traffic flow data. Using estimates instead.")
        else:
            print("Warning: Approaching API limits for traffic flow. Using estimates instead.")
    else:
        print("DEBUG: Using existing processed_traffic_data.json")
        with open("processed_traffic_data.json", "r") as f:
            traffic_data = json.load(f)
    
    # Step 5: Get pedestrian data (estimated)
    pedestrian_data = None
    if force_regenerate or not os.path.exists(os.path.join(WORKSPACE, "pedestrian_data.json")):
        print("DEBUG: pedestrian_data.json doesn't exist or regeneration forced")
        pedestrian_data = get_pedestrian_data()
        if not pedestrian_data:
            print("Warning: Failed to estimate pedestrian data.")
    else:
        print("DEBUG: Using existing pedestrian_data.json")
        with open("pedestrian_data.json", "r") as f:
            pedestrian_data = json.load(f)
    
    # Step 6: Request travel times and directions
    travel_data = None
    directional_flows = None
    if force_regenerate or not os.path.exists(os.path.join(WORKSPACE, "directional_flows.json")):
        print("DEBUG: directional_flows.json doesn't exist or regeneration forced")
        # Check if we can make API calls without exceeding limits
        if API_CALL_COUNTER["matrix_routing"] < API_FREE_TIER_LIMITS["matrix_routing"]:
            print("DEBUG: About to call get_travel_times()")
            travel_results = get_travel_times(network_info['entry_exit_points'])
            if travel_results:
                travel_data = travel_results.get('travel_data')
                directional_flows = travel_results.get('directional_flows')
            else:
                print("Warning: Failed to get travel times. Simulation may not be realistic.")
        else:
            print("Warning: Approaching API limits for matrix routing. Using estimates instead.")
    else:
        print("DEBUG: Using existing directional_flows.json")
        # Load directional flows
        with open("directional_flows.json", "r") as f:
            directional_flows = json.load(f)
        
        # Load travel data if available
        if os.path.exists("travel_matrix_raw.json"):
            with open("travel_matrix_raw.json", "r") as f:
                travel_data = json.load(f)
    
    # Step 7: Generate vehicle demand based on traffic data
    if force_regenerate or not os.path.exists(os.path.join(WORKSPACE, "manhattan_flows.xml")):
        print("DEBUG: manhattan_flows.xml doesn't exist or regeneration forced")
        print("DEBUG: About to call generate_traffic_demand()")
        flows = generate_traffic_demand(traffic_data, directional_flows, network_info)
        if not flows:
            print("Warning: Failed to generate traffic demand with real data.")
            print("Attempting to create basic traffic flows instead...")
            
            # Create very basic flows between random valid edges
            try:
                basic_flows = []
                
                # Get valid edges from the network
                valid_edges = set()
                for edge in network_info.get('edges', []):
                    # Skip internal and special edges
                    if edge.get('function') == 'internal':
                        continue
                    valid_edges.add(edge['id'])
                
                valid_edges = list(valid_edges)
                
                # Create 10 basic flows between random edges
                if len(valid_edges) >= 10:
                    import random
                    total_vehicles = traffic_data.get("total_vehicles", 500) if traffic_data else 500
                    vehicles_per_flow = total_vehicles // 10
                    
                    for i in range(10):
                        origin = random.choice(valid_edges)
                        destination = random.choice(valid_edges)
                        # Make sure origin and destination are different
                        while origin == destination:
                            destination = random.choice(valid_edges)
                            
                        basic_flows.append({
                            "origin": origin,
                            "destination": destination,
                            "count": vehicles_per_flow,
                            "direction": "random"
                        })
                    
                    # Write basic flows to file
                    with open("manhattan_flows.xml", "w") as f:
                        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
                        f.write('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n')
                        
                        # Include vehicle type definitions
                        f.write('    <vType id="passenger" vClass="passenger" color="0,0,1" accel="2.6" decel="4.5" sigma="0.5" length="4.5" minGap="2.5" maxSpeed="15" speedDev="0.1" guiShape="passenger"/>\n')
                        
                        # Create flows
                        flow_id = 0
                        for flow in basic_flows:
                            begin_time = random.randint(0, 300)
                            end_time = 3600
                            f.write(f'    <flow id="flow_{flow_id}" type="passenger" from="{flow["origin"]}" to="{flow["destination"]}" begin="{begin_time}" end="{end_time}" number="{flow["count"]}" departLane="best" departSpeed="max"/>\n')
                            flow_id += 1
                        
                        f.write('</routes>\n')
                    
                    print(f"Created {len(basic_flows)} basic flows as fallback.")
                    flows = basic_flows
                else:
                    print("Error: Not enough valid edges found to create basic flows.")
                    return False
            except Exception as e:
                print(f"Error creating basic flows: {e}")
                return False
    
    # Step 8: Create SUMO configuration with output collectors
    if force_regenerate or not os.path.exists(os.path.join(WORKSPACE, "manhattan_sim.sumocfg")):
        print("DEBUG: manhattan_sim.sumocfg doesn't exist or regeneration forced")
        if not create_sumo_config():
            print("Failed to create SUMO configuration. Exiting.")
            return False
    else:
        print("DEBUG: Using existing manhattan_sim.sumocfg")
    
    # Step 9: Generate roundabout converter script
    if force_regenerate or not os.path.exists(os.path.join(WORKSPACE, "convert_to_roundabouts.py")):
        print("DEBUG: convert_to_roundabouts.py doesn't exist or regeneration forced")
        if not create_roundabout_converter():
            print("Failed to create roundabout converter. Exiting.")
            return False
    else:
        print("DEBUG: Using existing convert_to_roundabouts.py")
    
    # Step 10: Create data analysis tools
    if force_regenerate or not os.path.exists(os.path.join(WORKSPACE, "analyze_results.py")):
        print("DEBUG: analyze_results.py doesn't exist or regeneration forced")
        if not create_data_analysis_tools():
            print("Failed to create data analysis tools. Exiting.")
            return False
    else:
        print("DEBUG: Using existing analyze_results.py")
    
    # Step 11: Create README
    if force_regenerate or not os.path.exists(os.path.join(WORKSPACE, "README.md")):
        print("DEBUG: README.md doesn't exist or regeneration forced")
        if not create_readme():
            print("Failed to create README. Exiting.")
            return False
    else:
        print("DEBUG: Using existing README.md")
    
    # Step 12: Run simulation
    print("DEBUG: About to call run_simulation()")
    run_simulation()
    
    print("\n=== Manhattan Traffic Simulation Setup Complete ===")
    print(f"All files have been created in: {WORKSPACE}")
    print("\nTo analyze the simulation results, run:")
    print("python analyze_results.py")
    print("\nTo visualize the simulation with GUI, run:")
    print("sumo-gui -c manhattan_sim.sumocfg")
    print("\nTo convert traffic lights to roundabouts, run:")
    print("python convert_to_roundabouts.py --random 5")
    print("sumo-gui -c manhattan_roundabouts.sumocfg")
    
    return True

if __name__ == "__main__":
    main()