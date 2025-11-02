#!/usr/bin/env python3
"""
Baseline SUMO Simulation using Real-life HERE API Data
------------------------------------------------------
This script creates a SUMO simulation using real-life traffic data from HERE API.
It incorporates time-varying traffic demand based on multiple snapshots of traffic data.

Features:
- Uses real traffic data from HERE API (8:00am - 9:38am)
- Creates time-varying traffic demand
- Maps HERE API road segments to SUMO network
- Analyzes simulation results and generates visualizations

Usage:
python baseline_sumo_simul.py --osm-file <osm_file> --data-dir <data_dir> --output-dir <output_dir>


Date: April 4, 2025
"""

import os
import sys
import json
import glob
import time
import random
import argparse
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import re
import xml.etree.ElementTree as ET
import networkx as nx
import math
import csv
import traci
import random


# Setup paths
DEFAULT_DATA_DIR = "traffic_data_8_938Fri"
DEFAULT_OUTPUT_DIR = os.path.join(DEFAULT_DATA_DIR, "baseline_sumo_run")
DEFAULT_OSM_FILE = os.path.join(DEFAULT_DATA_DIR, "city.osm")



# Ensure output directory exists
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

# Setup SUMO environment
if "SUMO_HOME" in os.environ:
    SUMO_HOME = os.environ["SUMO_HOME"]
    tools_path = os.path.join(SUMO_HOME, "tools")
    if tools_path not in sys.path:
        sys.path.append(tools_path)
else:
    # Try to infer SUMO_HOME from sumo executable
    SUMO_BIN = "/opt/homebrew/bin/sumo"
    if os.path.exists(SUMO_BIN):
        bin_dir = os.path.dirname(SUMO_BIN)
        homebrew_dir = os.path.dirname(bin_dir)
        SUMO_HOME = os.path.join(homebrew_dir, "share", "sumo")
        
        if os.path.exists(os.path.join(SUMO_HOME, "tools")):
            print(f"SUMO_HOME detected at: {SUMO_HOME}")
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

# Now we can import SUMO libraries
try:
    import sumolib
    from sumolib import checkBinary
except ImportError:
    print("Error importing SUMO libraries. Check if SUMO_HOME is correctly set.")
    sys.exit(1)

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(DEFAULT_OUTPUT_DIR, "simulation.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("baseline_sumo_simul")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate SUMO baseline simulation with TraCI control using HERE API data')
    parser.add_argument('--osm-file', default=DEFAULT_OSM_FILE, help='OpenStreetMap file')
    parser.add_argument('--data-dir', default=DEFAULT_DATA_DIR, help='Directory with HERE API data')
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR, help='Output directory')
    parser.add_argument('--force-regenerate', action='store_true', help='Force regeneration of all files')
    parser.add_argument('--skip-simulation', action='store_true', help='Skip running simulation')
    parser.add_argument('--gui', action='store_true', help='Run SUMO with GUI')
    parser.add_argument('--simulation-period', default=6000, type=int, help='Simulation period in seconds')
    parser.add_argument('--min-lat', default=40.712178, type=float, help='Minimum latitude for OSM map')
    parser.add_argument('--min-lon', default=-74.033341, type=float, help='Minimum longitude for OSM map')
    parser.add_argument('--max-lat', default=40.759722, type=float, help='Maximum latitude for OSM map')
    parser.add_argument('--max-lon', default=-73.958424, type=float, help='Maximum longitude for OSM map')
    # Add new TraCI-specific arguments
    parser.add_argument('--use-traci', action='store_true', default=True,
                       help='Use TraCI for dynamic control (default: True)')
    parser.add_argument('--visualize', action='store_true',
                       help='Use SUMO-GUI for visualization')

    return parser.parse_args()

def find_latest_file(dir_path, pattern):
    """Find the latest file matching a pattern in a directory"""
    files = glob.glob(os.path.join(dir_path, pattern))
    if not files:
        return None
    return max(files, key=os.path.getctime)

def load_json_file(file_path):
    """Load a JSON file"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return None

def download_restricted_osm_map(min_lat, min_lon, max_lat, max_lon, data_dir, osm_filename="manhattan.osm"):
    """Download OpenStreetMap data for a specific area, focusing only on traffic infrastructure"""
    # Create full path for the output file
    output_file = os.path.join(data_dir, osm_filename)
    
    logger.info(f"Downloading restricted OSM map for area: [{min_lat},{min_lon},{max_lat},{max_lon}]")
    logger.info(f"OSM file will be saved to: {output_file}")
    
    import requests
    
    # Use Overpass API with a targeted query for transportation infrastructure only
    overpass_url = "https://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:xml][timeout:300];
    (
    // All major road types for Manhattan
    way["highway"~"^(motorway|trunk|primary|secondary|tertiary|unclassified|residential|service)$"]({min_lat},{min_lon},{max_lat},{max_lon});
    way["highway"="motorway_link"]({min_lat},{min_lon},{max_lat},{max_lon});
    way["highway"="trunk_link"]({min_lat},{min_lon},{max_lat},{max_lon});
    way["highway"="primary_link"]({min_lat},{min_lon},{max_lat},{max_lon});
    way["highway"="secondary_link"]({min_lat},{min_lon},{max_lat},{max_lon});
    
    // Manhattan-specific infrastructure
    way["highway"="living_street"]({min_lat},{min_lon},{max_lat},{max_lon});
    way["highway"="road"]({min_lat},{min_lon},{max_lat},{max_lon});
    
    // Traffic control
    node["highway"="traffic_signals"]({min_lat},{min_lon},{max_lat},{max_lon});
    node["highway"="stop"]({min_lat},{min_lon},{max_lat},{max_lon});
    node["highway"="give_way"]({min_lat},{min_lon},{max_lat},{max_lon});
    
    // Turn restrictions and route relations
    relation["type"="restriction"]({min_lat},{min_lon},{max_lat},{max_lon});
    relation["type"="route"]["route"="road"]({min_lat},{min_lon},{max_lat},{max_lon});
    
    // All nodes from ways (for connectivity)
    node(w)({min_lat},{min_lon},{max_lat},{max_lon});
    );

    // Output with metadata
    out meta;
    """

    # Add logging to validate download
    try:
        response = requests.post(overpass_url, data={"data": overpass_query})
        
        if response.status_code == 200:
            content = response.content
            
            # Parse and analyze the downloaded content
            import xml.etree.ElementTree as ET
            try:
                root = ET.fromstring(content)
                ways = root.findall('.//way')
                nodes = root.findall('.//node')
                relations = root.findall('.//relation')
                
                # Count different highway types
                highway_types = {}
                for way in ways:
                    highway_tag = way.find('.//tag[@k="highway"]')
                    if highway_tag is not None:
                        highway_type = highway_tag.get('v')
                        highway_types[highway_type] = highway_types.get(highway_type, 0) + 1
                
                logger.info(f"Downloaded OSM data analysis:")
                logger.info(f"  Total ways: {len(ways)}")
                logger.info(f"  Total nodes: {len(nodes)}")
                logger.info(f"  Total relations: {len(relations)}")
                logger.info(f"  Highway types: {highway_types}")
                
                # Check for potential connectivity issues
                if len(ways) < 1000:
                    logger.warning(f"Low number of ways ({len(ways)}) - network may be sparse")
                if 'primary' not in highway_types or highway_types.get('primary', 0) < 10:
                    logger.warning("Few primary roads found - may affect connectivity")
                    
            except ET.ParseError as e:
                logger.warning(f"Could not parse OSM XML for analysis: {e}")
            
            with open(output_file, "wb") as f:
                f.write(content)
            logger.info(f"OSM map downloaded successfully to {output_file} ({len(content)} bytes)")
            return output_file
        else:
            logger.error(f"Failed to download OSM map. Status code: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return None
    except Exception as e:
        logger.error(f"Error downloading OSM map: {e}")
        return None


def create_sumo_network(osm_file, output_dir):
    """Convert OSM map to SUMO network"""
    logger.info("Converting OSM data to SUMO network...")
    
    net_file = os.path.join(output_dir, "baseline.net.xml")
    
    # Ensure osm_file is a valid path and has content
    if not os.path.exists(osm_file):
        logger.error(f"OSM file not found: {osm_file}")
        return None
    
    file_size = os.path.getsize(osm_file)
    if file_size < 1000:  # Less than 1KB
        logger.error(f"OSM file is too small ({file_size} bytes), likely invalid")
        return None
    
    logger.info(f"Using OSM file: {osm_file} ({file_size} bytes)")
    
    # Simplified netconvert command focused only on basic road structure
    # We avoid pedestrian and bicycle infrastructure options since we didn't download that data
    netconvert_cmd = [
        checkBinary("netconvert"),
        "--osm", osm_file,
        "--output-file", net_file,
        "--geometry.remove", "true",
        "--roundabouts.guess", "true",
        "--junctions.join", "true",
        "--tls.guess", "true",
        "--tls.join", "true",
        "--tls.default-type", "actuated",
        "--tls.cycle.time", "90",  #  default cycle time
        "--tls.green.time", "31",  #  default green time ; for actuated traffic lights, min -max green time is 5-60s, yellow is 4; detecttion range 0 by default ( no actuation by default)
        "--tls.yellow.time", "4",  # Standard yellow time
        "--tls.red.time", "3",     # All-red time for safety
        "--tls.min-dur", "5",      # Minimum phase duration
        "--tls.max-dur", "60",     # Maximum phase duration for actuated signals
        "--no-internal-links", "false",
        "--no-turnarounds", "true",
        "--ramps.guess", "true",
        "--verbose", "true"
        # Remove pedestrian/bicycle options 
        # "--walkingareas", "true",
        # "--crossings.guess", "true",  
        # "--sidewalks.guess", "true",
    ]
    
    try:
        logger.info(f"Running netconvert: {' '.join(netconvert_cmd)}")
        result = subprocess.run(netconvert_cmd, check=False, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"SUMO network created successfully: {net_file}")
            return net_file
        else:
            logger.error(f"netconvert failed with code {result.returncode}")
            logger.error(f"Error output: {result.stderr}")
            
            # Try even simpler options if the first attempt failed
            logger.info("Trying with even simpler options...")
            simple_cmd = [
                checkBinary("netconvert"),
                "--osm", osm_file,
                "--output-file", net_file,
                "--geometry.remove", "true",
                "--verbose", "true"
            ]
            
            simple_result = subprocess.run(simple_cmd, check=False, capture_output=True, text=True)
            if simple_result.returncode == 0:
                logger.info(f"SUMO network created successfully with simpler options: {net_file}")
                return net_file
            else:
                logger.error(f"Simple netconvert also failed: {simple_result.stderr}")
                return None
    except FileNotFoundError:
        logger.error("netconvert command not found. Check SUMO installation.")
        return None
    # except Exception as e:
    #     logger.error(f"Unexpected error creating SUMO network: {e}")
    #     return None

def get_realtime_files(data_dir):
    """Get all realtime data files sorted by timestamp"""
    realtime_files = glob.glob(os.path.join(data_dir, "realtime_*.json"))
    
    # Sort by timestamp in filename
    def extract_timestamp(filename):
        match = re.search(r'(\d{8}_\d{6})', filename)
        if match:
            return match.group(1)
        return ""
    
    return sorted(realtime_files, key=extract_timestamp)

def get_incidents_files(data_dir):
    """Get all incidents data files sorted by timestamp"""
    incidents_files = glob.glob(os.path.join(data_dir, "incidents_*.json"))
    
    # Sort by timestamp in filename
    def extract_timestamp(filename):
        match = re.search(r'(\d{8}_\d{6})', filename)
        if match:
            return match.group(1)
        return ""
    
    return sorted(incidents_files, key=extract_timestamp)

def extract_timestamp_from_filename(filename):
    """Extract timestamp from filename"""
    match = re.search(r'(\d{8}_\d{6})', filename)
    if match:
        timestamp_str = match.group(1)
        try:
            return datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
        except ValueError:
            logger.warning(f"Could not parse timestamp from {filename}")
    return None

def find_matching_analysis_file(data_dir, timestamp):
    """Find the matching analysis file for a given timestamp"""
    # Convert timestamp to string format
    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
    
    # Look for analysis files
    analysis_files = glob.glob(os.path.join(data_dir, "analysis_*.json"))
    
    # Try to find exact match
    exact_match = [f for f in analysis_files if timestamp_str in f]
    if exact_match:
        logger.info(f"Found exact matching analysis file: {exact_match[0]}")
        return exact_match[0]
    
    # If no exact match, find the closest one (earlier than the timestamp)
    valid_files = []
    for file in analysis_files:
        file_timestamp = extract_timestamp_from_filename(file)
        if file_timestamp and file_timestamp <= timestamp:
            valid_files.append((file, file_timestamp))
    
    if valid_files:
        # Sort by timestamp (descending) and return the most recent one
        valid_files.sort(key=lambda x: x[1], reverse=True)
        logger.info(f"Using closest analysis file: {valid_files[0][0]}")
        return valid_files[0][0]
    
    logger.warning(f"No matching analysis file found for {timestamp}")
    return None

def find_matching_edge_mapping_file(data_dir, timestamp):
    """Find the matching edge mapping file for a given timestamp"""
    mapping_files = glob.glob(os.path.join(data_dir, "sumo_edge_mapping_*.json"))
    if mapping_files:
        return mapping_files[0]
    return None

def find_matching_entry_exit_file(data_dir, timestamp):
    """Find the matching entry/exit points file for a given timestamp"""
    entry_exit_files = glob.glob(os.path.join(data_dir, "sumo_entry_exit_points_*.json"))
    if entry_exit_files:
        return entry_exit_files[0]
    return None

def load_data_for_timestep(data_dir, realtime_file, global_edge_mapping=None, global_entry_exit=None):
    """Load all necessary data for a single timestep, using global mapping data if available"""
    timestamp = extract_timestamp_from_filename(realtime_file)
    if not timestamp:
        logger.error(f"Could not extract timestamp from {realtime_file}")
        return None
    
    # Get the corresponding incidents file
    incidents_file = os.path.join(data_dir, f"incidents_{timestamp.strftime('%Y%m%d_%H%M%S')}.json")
    if not os.path.exists(incidents_file):
        # Try to find the closest incidents file
        incidents_files = get_incidents_files(data_dir)
        incidents_file = None
        min_diff = float('inf')
        for f in incidents_files:
            f_timestamp = extract_timestamp_from_filename(f)
            if f_timestamp:
                diff = abs((f_timestamp - timestamp).total_seconds())
                if diff < min_diff:
                    min_diff = diff
                    incidents_file = f
        
        if not incidents_file or min_diff > 300:  # 5 minutes difference is too much
            logger.warning(f"No matching incidents file found for {realtime_file}")
            incidents_file = None
    
    # Find the corresponding analysis file
    analysis_file = find_matching_analysis_file(data_dir, timestamp)
    if not analysis_file:
        logger.warning(f"No matching analysis file found for {realtime_file}")
    
    # Load all files
    data = {
        'timestamp': timestamp,
        'realtime': load_json_file(realtime_file),
        'incidents': load_json_file(incidents_file) if incidents_file else None,
        'analysis': load_json_file(analysis_file) if analysis_file else None,
        'edge_mapping': global_edge_mapping,  # Use global mapping
        'entry_exit': global_entry_exit       # Use global entry/exit points
    }
    
    return data

def create_duarouter_trips(entry_exit_points, output_file, processed_data):
    """Create trips file for duarouter with lat/lon coordinates and dynamic distribution"""
    # Sort processed data by timestamp to get first and last
    processed_data.sort(key=lambda x: x['timestamp'])
    
    # Calculate simulation period from first to last timestamp
    first_timestamp = processed_data[0]['timestamp']
    last_timestamp = processed_data[-1]['timestamp']
    simulation_period = (last_timestamp - first_timestamp).total_seconds()
    
    # Calculate trip distribution based on jam factor change
    first_jam_factor = processed_data[0].get('avg_jam_factor', 1.0)
    last_jam_factor = processed_data[-1].get('avg_jam_factor', 1.0)
    
    # Calculate percentage increase in jam factor
    if first_jam_factor > 0:
        jam_factor_increase = (last_jam_factor - first_jam_factor) / first_jam_factor
    else:
        jam_factor_increase = 0
    
    # Determine what percentage of trips should start at the beginning
    # If jam factor increased by 30%, then 70% of trips at start, 30% distributed
    # Cap between 50% and 90% for reasonableness
    initial_percentage = max(0.5, min(0.9, 1.0 - jam_factor_increase))
    
    logger.info(f"Simulation period: {simulation_period} seconds")
    logger.info(f"Jam factor change: {first_jam_factor:.2f} to {last_jam_factor:.2f} (change: {jam_factor_increase*100:.1f}%)")
    logger.info(f"Trip distribution: {initial_percentage*100:.1f}% at start, {(1-initial_percentage)*100:.1f}% distributed")
    
    # Filter entry/exit point pairs to ensure minimum distance
    min_route_distance = 1500  # Increased to 1.5km minimum route length
    max_route_distance = 10000  # Added maximum 10km route length to avoid excessively long routes
    valid_pairs = []
    
    # Define haversine distance function if not already defined
    def haversine_distance(lat1, lon1, lat2, lon2):
        """Calculate the great circle distance between two points in meters"""
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371000  # Radius of earth in meters
        return c * r
    
    logger.info("Filtering entry/exit pairs to ensure minimum route distance...")
    for entry in entry_exit_points.get('entry_points', []):
        entry_loc = entry.get('location', [0, 0])
        entry_lat, entry_lon = entry_loc
        
        for exit in entry_exit_points.get('exit_points', []):
            exit_loc = exit.get('location', [0, 0])
            exit_lat, exit_lon = exit_loc
            
            # Calculate direct distance between points
            distance = haversine_distance(entry_lat, entry_lon, exit_lat, exit_lon)
            
            # Only include pairs with sufficient distance
            if distance >= min_route_distance:
                valid_pairs.append((entry, exit))
    
    logger.info(f"Filtered {len(valid_pairs)} valid pairs out of {len(entry_exit_points.get('entry_points', [])) * len(entry_exit_points.get('exit_points', []))} potential pairs")
    
    # If not enough valid pairs, reduce threshold
    if len(valid_pairs) < 50:
        # Try a multi-step approach with distance tiers
        distance_tiers = [1200, 1000, 800, 600]  # Try several minimum distances
        
        for min_dist in distance_tiers:
            valid_pairs = []
            for entry in entry_exit_points.get('entry_points', []):
                entry_loc = entry.get('location', [0, 0])
                entry_lat, entry_lon = entry_loc
                
                for exit in entry_exit_points.get('exit_points', []):
                    exit_loc = exit.get('location', [0, 0])
                    exit_lat, exit_lon = exit_loc
                    
                    distance = haversine_distance(entry_lat, entry_lon, exit_lat, exit_lon)
                    if min_dist <= distance <= max_route_distance:
                        valid_pairs.append((entry, exit))
            
            if len(valid_pairs) >= 50:
                logger.info(f"Found {len(valid_pairs)} valid pairs using minimum distance {min_dist}m")
                break
    
    # Now use valid_pairs instead of all combinations
    with open(output_file, 'w') as f:
        # Write XML header...
        
        # Write entry to exit flows using valid_pairs only
        trip_id = 0
        
        # REPLACE the existing loop with this:
        for entry, exit in valid_pairs:
            entry_lat = entry.get('location', [0, 0])[0]
            entry_lon = entry.get('location', [0, 0])[1]
            entry_weight = entry.get('weight', 1)
            
            exit_lat = exit.get('location', [0, 0])[0]
            exit_lon = exit.get('location', [0, 0])[1]
            exit_weight = exit.get('weight', 1)
            
            flow_weight = entry_weight * exit_weight


    with open(output_file, 'w') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n')
        
        # Write entry to exit flows
        trip_id = 0
        total_trips = len(entry_exit_points.get('entry_points', [])) * len(entry_exit_points.get('exit_points', []))
        
        # Calculate how many trips belong to the initial percentage
        initial_trips = int(total_trips * initial_percentage)
        remaining_trips = total_trips - initial_trips
        
        trip_count = 0
        for entry in entry_exit_points.get('entry_points', []):
            entry_lat = entry.get('location', [0, 0])[0]
            entry_lon = entry.get('location', [0, 0])[1]
            entry_weight = entry.get('weight', 1)
            
            for exit in entry_exit_points.get('exit_points', []):
                exit_lat = exit.get('location', [0, 0])[0]
                exit_lon = exit.get('location', [0, 0])[1]
                exit_weight = exit.get('weight', 1)
                
                flow_weight = entry_weight * exit_weight
                
                # Determine departure time
                if trip_count < initial_trips:
                    # Initial percentage of trips depart at time 0
                    depart_time = 0
                else:
                    # Remaining trips distributed throughout the simulation period
                    # Distribute evenly across the simulation period
                    remaining_index = trip_count - initial_trips
                    depart_time = (remaining_index * simulation_period) / max(1, remaining_trips)
                
                f.write(f'    <trip id="trip_{trip_id}" depart="{depart_time}" fromLonLat="{entry_lon},{entry_lat}" toLonLat="{exit_lon},{exit_lat}" departPos="random" departSpeed="max" departLane="best">\n')
                f.write(f'        <param key="weight" value="{flow_weight}"/>\n')
                f.write('    </trip>\n')
                
                trip_id += 1
                trip_count += 1
        
        f.write('</routes>\n')
    
    logger.info(f"Created duarouter trips file with {trip_id} OD pairs at {output_file}")
    logger.info(f"{initial_percentage*100:.1f}% of trips ({initial_trips}) start at time 0")
    logger.info(f"Remaining {remaining_trips} trips distributed throughout {simulation_period} seconds")
    
    return output_file, simulation_period

def run_duarouter(net_file, trips_file, output_file):
    """Run duarouter with settings optimized for urban route generation"""
    try:
        duarouter = checkBinary("duarouter")
        
        cmd = [
            duarouter,
            "--net-file", net_file,
            "--route-files", trips_file,
            "--output-file", output_file,
            "--ignore-errors", "true",
            "--mapmatch.distance", "100",      # Stricter matching for urban areas, decreased from 500
            "--mapmatch.junctions", "true",    # Include junctions in matching  
            "--repair", "true",                # Repair broken routes
            "--repair.from", "true",           # Repair source positions
            "--repair.to", "true",             # Repair destination positions  
            "--weights.random-factor", "1.1",  # Add route variety
            "--routing-algorithm", "dijkstra", # Use Dijkstra for urban routing
            "--verbose", "true"
        ]
        
        logger.info(f"Running duarouter: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"duarouter completed successfully")
            
            # Analyze generated routes
            analyze_generated_routes(output_file)
            return output_file
        else:
            logger.error(f"duarouter failed: {result.stderr}")
            return None
    except Exception as e:
        logger.error(f"Error running duarouter: {e}")
        return None

def analyze_generated_routes(routes_file):
    """Analyze the routes generated by duarouter"""
    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(routes_file)
        root = tree.getroot()
        
        vehicles = root.findall('.//vehicle')
        routes = root.findall('.//route')
        
        route_lengths = []
        for route in routes:
            edges = route.get('edges', '').split()
            route_lengths.append(len(edges))
        
        if route_lengths:
            avg_edges = sum(route_lengths) / len(route_lengths)
            logger.info(f"Generated routes: {len(vehicles)} vehicles, {len(routes)} routes")
            logger.info(f"Average route length: {avg_edges:.1f} edges")
            logger.info(f"Route edge count range: {min(route_lengths)} - {max(route_lengths)}")
        
    except Exception as e:
        logger.warning(f"Could not analyze routes: {e}")

def map_here_segments_to_sumo_edges(edge_mapping_data, net):
    """Map HERE API segments to SUMO network edges based on coordinates"""
    logger.info("Mapping HERE API segments to SUMO edges...")
    
    # Check if edge_mapping_data has the expected structure
    if not edge_mapping_data or 'segments' not in edge_mapping_data:
        logger.error("Invalid edge mapping data format")
        return {}
    
    segments = edge_mapping_data.get('segments', [])
    logger.info(f"Processing {len(segments)} HERE API segments")
    
    # Create a mapping dictionary
    segment_to_edges = {}
    
    # Max search radius in meters
    max_radius = 50.0
    
    # Count successful mappings
    successful_mappings = 0
    
    for segment in segments:
        segment_id = segment.get('id')
        if not segment_id:
            continue
        
        # Get shape points
        shape_points = segment.get('shape', [])
        if not shape_points or len(shape_points) < 2:
            continue
        
        # Get start and end points
        start_point = shape_points[0]
        end_point = shape_points[-1]
        
        # Extract coordinates
        if len(start_point) >= 2 and len(end_point) >= 2:
            start_lat, start_lon = start_point[0], start_point[1]
            end_lat, end_lon = end_point[0], end_point[1]
            
            try:
                # Convert to SUMO x,y coordinates
                start_x, start_y = net.convertLonLat2XY(start_lon, start_lat)
                end_x, end_y = net.convertLonLat2XY(end_lon, end_lat)
                
                # Find nearby edges for start and end
                start_edges = net.getNeighboringEdges(start_x, start_y, max_radius)
                end_edges = net.getNeighboringEdges(end_x, end_y, max_radius)
                
                # Sort by distance
                if start_edges:
                    start_edges.sort(key=lambda x: x[1])
                if end_edges:
                    end_edges.sort(key=lambda x: x[1])
                
                # Get closest edges
                matched_edges = set()
                
                if start_edges:
                    matched_edges.add(start_edges[0][0].getID())
                
                if end_edges:
                    matched_edges.add(end_edges[0][0].getID())
                
                # Try to find intermediate edges for longer segments
                if len(shape_points) > 2 and len(matched_edges) > 0:
                    # Sample additional points along the segment
                    sample_count = min(5, len(shape_points) - 2)  # Max 5 additional points
                    
                    # Fix the step calculation to avoid division by zero
                    if sample_count == 0:
                        step = 1
                    else:
                        step = max(1, (len(shape_points) - 2) // sample_count)
                    
                    for i in range(1, len(shape_points) - 1, step):
                        if len(matched_edges) >= 5:  # Limit to 5 edges per segment
                            break
                        
                        point = shape_points[i]
                        if len(point) >= 2:
                            mid_lat, mid_lon = point[0], point[1]
                            mid_x, mid_y = net.convertLonLat2XY(mid_lon, mid_lat)
                            mid_edges = net.getNeighboringEdges(mid_x, mid_y, max_radius)
                            
                            if mid_edges:
                                mid_edges.sort(key=lambda x: x[1])
                                matched_edges.add(mid_edges[0][0].getID())
                
                # Store the mapping
                if matched_edges:
                    segment_to_edges[segment_id] = list(matched_edges)
                    successful_mappings += 1
                    
                    # Also update the original segments data for future use
                    segment['sumo_edges'] = list(matched_edges)
            except Exception as e:
                logger.warning(f"Error mapping segment {segment_id}: {e}")
    
    logger.info(f"Successfully mapped {successful_mappings} out of {len(segments)} segments to SUMO edges")
    return segment_to_edges

def process_timestep_data(data, net, segment_to_edges=None):
    """Process data for a single timestep to extract speed information"""
    if not data or not data['realtime']:
        logger.error("Incomplete data for timestep")
        return None
    
    # Use the passed segment_to_edges, or create empty dict if None
    if segment_to_edges is None:
        segment_to_edges = {}

    timestamp = data['timestamp']
    realtime_data = data['realtime']
    incidents_data = data['incidents']
    analysis_data = data.get('analysis', {})
    edge_mapping_data = data.get('edge_mapping', {})
    
    logger.info(f"Processing data for timestep: {timestamp}")
    
    # Extract estimated vehicle count from analysis
    total_vehicles = analysis_data.get('total_vehicles', 0)
    total_vehicles = int(total_vehicles * 0.3)  # Use 30% to avoid overloading

    if total_vehicles <= 0:
        logger.warning("No vehicle count data available, using default")
        total_vehicles = 5000  # Default fallback
    else:
        logger.info(f"Using estimated vehicles from analysis: {total_vehicles}")
    
    # Map traffic data to SUMO edges for speed adjustments
    edge_speeds = {}
    edge_jam_factors = {}
    
    # Debug: check edge_mapping_data structure
    if edge_mapping_data:
        logger.info(f"Processing edge mapping with {len(edge_mapping_data.get('segments', []))} segments")
        
        # Check if segments have SUMO edges
        has_sumo_edges = False
        for segment in edge_mapping_data.get('segments', [])[:100]:  # Check first 100
            if 'sumo_edges' in segment and segment['sumo_edges']:
                has_sumo_edges = True
                logger.info(f"Example segment {segment.get('id')} with SUMO edges: {segment['sumo_edges']}")
                break
        
        if not has_sumo_edges:
            logger.warning("No segments with SUMO edges found in the first 100 segments")
    else:
        logger.warning("No edge mapping data available")
    
    # Get realtime segments
    realtime_segments = realtime_data.get('results', [])
    if not realtime_segments:
        logger.error("No realtime segments found")
        return None
    
    logger.info(f"Realtime data contains {len(realtime_segments)} results")
    
    # Get mapping segments  
    mapping_segments = edge_mapping_data.get('segments', [])
    if not mapping_segments:
        logger.error("No mapping segments found")
        return None
    
    # Process realtime flow data - match by index since realtime segments don't have IDs
    matched_speed_segments = 0
    
    # Match segments by index
    for idx, realtime_segment in enumerate(realtime_segments):
        # Get traffic flow data
        flow = realtime_segment.get('currentFlow', {})
        if not flow:
            continue
        
        # Get jam factor and speed
        jam_factor = flow.get('jamFactor', 0)
        speed = flow.get('speed', 0)
        
        # Skip segments with invalid speeds
        if speed <= 0:
            continue
        
        # Find corresponding mapping segment by index
        if idx < len(mapping_segments):
            mapping_segment = mapping_segments[idx]
            mapped_edges = mapping_segment.get('sumo_edges', [])
            
            # Apply speed data to mapped edges
            if mapped_edges:
                matched_speed_segments += 1
                
                # Log first few matches
                if matched_speed_segments <= 5:
                    logger.info(f"Applied speed {speed} to edges {mapped_edges} for realtime segment index {idx}")
                
                for edge_id in mapped_edges:
                    if edge_id in edge_speeds:
                        # Average with existing speed
                        edge_speeds[edge_id] = (edge_speeds[edge_id] + speed) / 2
                        edge_jam_factors[edge_id] = (edge_jam_factors[edge_id] + jam_factor) / 2
                    else:
                        edge_speeds[edge_id] = speed
                        edge_jam_factors[edge_id] = jam_factor
    
    logger.info(f"Matched speed data for {matched_speed_segments} segments")
    
    
    # Process incidents data to adjust speeds and generate incidents
    incidents = []
    processed_incidents = 0
    
    # Get the correct path to incidents data - could be either 'incidents' or 'results'
    incident_items = None
    if incidents_data:
        if 'incidents' in incidents_data:
            incident_items = incidents_data['incidents']
            logger.info(f"Processing {len(incident_items)} incidents from 'incidents' key")
        elif 'results' in incidents_data:
            incident_items = incidents_data['results']
            logger.info(f"Processing {len(incident_items)} incidents from 'results' key")
        else:
            logger.warning("No recognizable incident data structure found")
        
        # # Print incident structure sample for debugging
        # if incident_items and len(incident_items) > 0:
        #     sample_incident = incident_items[0]
        #     logger.info(f"Sample incident structure: {json.dumps(sample_incident, indent=2)[:500]}...")
        
        # Expand search radius for incident mapping
        incident_search_radius = 200.0  # Increased from 100m to 200m

        # Add fallback for when segment_to_edges is empty by extracting mappings from edge_mapping_data even when direct segment IDs can't be matched.
        if not segment_to_edges and edge_mapping_data and 'segments' in edge_mapping_data:
            logger.info("Building segment_to_edges mapping from edge_mapping_data")
            for segment in edge_mapping_data.get('segments', []):
                if 'id' in segment and 'sumo_edges' in segment and segment['sumo_edges']:
                    segment_to_edges[segment['id']] = segment['sumo_edges']
            logger.info(f"Built mapping with {len(segment_to_edges)} entries")

        if incident_items:
            for incident in incident_items:   
                # Parse incident details - be more flexible with field names
                incident_type = incident.get("type", incident.get("incidentType", 'UNKNOWN'))
                incident_description = incident.get("description", incident.get("incidentDetails", {}).get("description", ''))
                incident_criticality = incident.get("criticality", incident.get("severity", 0))
                
                # # Log a full incident for debugging
                # if processed_incidents == 0:
                #     logger.info(f"Full incident example: {json.dumps(incident, indent=2)}")
                
                # Get affected road segments
                affected_edges = []
                location = incident.get('location', {})
                
                # Different incident data structures - look for shape or links
                if location:
                    # Try to find shape with links
                    shape = location.get('shape', {})
                    if 'links' in shape:
                        for link in shape.get('links', []):
                            link_id = link.get('linkId')
                            if link_id and link_id in segment_to_edges:
                                affected_edges.extend(segment_to_edges[link_id])
                    
                    # If no links found, try to get by coordinates
                    if not affected_edges:
                        # Try multiple location formats
                        point = None
                        
                        # Format 1: Direct point attribute
                        if 'point' in location:
                            point = location.get('point', {})
                            if 'latitude' in point and 'longitude' in point:
                                lat, lon = point.get('latitude'), point.get('longitude')
                        
                        # Format 2: First point in shape
                        elif 'shape' in location:
                            shape = location.get('shape', {})
                            if 'points' in shape and shape['points']:
                                first_point = shape['points'][0]
                                if 'lat' in first_point and 'lng' in first_point:
                                    lat, lon = first_point.get('lat'), first_point.get('lng')
                        
                        # Format 3: Location coords directly in location
                        elif 'lat' in location and 'lon' in location:
                            lat, lon = location.get('lat'), location.get('lon')
                        
                        # Format 4: First link in shape
                        elif 'shape' in location and 'links' in location['shape'] and location['shape']['links']:
                            first_link = location['shape']['links'][0]
                            if 'points' in first_link and first_link['points']:
                                first_point = first_link['points'][0]
                                if 'lat' in first_point and 'lng' in first_point:
                                    lat, lon = first_point.get('lat'), first_point.get('lng')
                        
                        # If we have coordinates, find nearby edges
                        if 'lat' in locals() and 'lon' in locals():
                            try:
                                # Convert to SUMO coordinates
                                x, y = net.convertLonLat2XY(lon, lat)
                                
                                # Find edges near this point with larger radius
                                nearby_edges = net.getNeighboringEdges(x, y, incident_search_radius)
                                if nearby_edges:
                                    # Sort by distance
                                    nearby_edges.sort(key=lambda x: x[1])
                                    
                                    # Add closest edges
                                    for edge, dist in nearby_edges[:5]:  # Take 5 closest edges (increased from 3)
                                        affected_edges.append(edge.getID())
                                        
                                    logger.info(f"Found {len(affected_edges)} affected edges for incident at ({lat:.5f}, {lon:.5f})")
                            except Exception as e:
                                logger.warning(f"Error finding edges for incident point: {e}")
            
                # Apply speed reduction to affected edges
                if affected_edges:
                    # Calculate speed reduction based on criticality
                    reduction_factor = 0.05 * (incident_criticality or 5)  # Default to 5 if not specified
                    
                    for edge_id in affected_edges:
                        if edge_id in edge_speeds:
                            # Reduce speed
                            reduced_speed = edge_speeds[edge_id] * (1 - reduction_factor)
                            edge_speeds[edge_id] = max(5.0, reduced_speed)  # Minimum 5 km/h
                            
                            # Increase jam factor
                            edge_jam_factors[edge_id] = min(10.0, edge_jam_factors.get(edge_id, 0) + incident_criticality / 2)
                    
                    # Record incident for later use
                    incidents.append({
                        'type': incident_type,
                        'description': incident_description,
                        'criticality': incident_criticality,
                        'affected_edges': affected_edges,
                        'speed_reduction': reduction_factor
                    })
                    
                    processed_incidents += 1
                    
                    # Log first few processed incidents
                    if processed_incidents <= 3:
                        logger.info(f"Processed incident: {incident_type}, affecting edges: {affected_edges}")
    
    logger.info(f"Mapped speed data for {len(edge_speeds)} edges")
    logger.info(f"Processed {len(incidents)} incidents")

    # Initialize variables before return
    avg_jam_factor = 0
    total_vehicle_count = 0
    if analysis_data:
        avg_jam_factor = analysis_data.get('avg_jam_factor', 0)
        total_vehicle_count = analysis_data.get('total_vehicles', 0)
    return {
        'timestamp': timestamp,
        'total_vehicles': total_vehicles,
        'edge_speeds': edge_speeds,
        'edge_jam_factors': edge_jam_factors,
        'incidents': incidents,
        'avg_jam_factor': avg_jam_factor,  
        'analysis': analysis_data  # for TraCI to access
    }

def debug_segments_and_realtime(edge_mapping_data, realtime_data, output_dir):
    """Create a debug file showing the relationship between segments and realtime data"""
    logger.info("Creating debug file for segments and realtime data...")
    
    debug_file = os.path.join(output_dir, "segment_realtime_debug.json")
    
    # Get segment IDs from mapping data
    mapping_segment_ids = set()
    segments_with_edges = set()
    
    for segment in edge_mapping_data.get('segments', []):
        segment_id = segment.get('id')
        if segment_id:
            mapping_segment_ids.add(segment_id)
            if 'sumo_edges' in segment and segment['sumo_edges']:
                segments_with_edges.add(segment_id)
    
    # Get segment IDs from realtime data
    realtime_segment_ids = set()
    
    for segment in realtime_data.get('results', []):
        segment_id = segment.get('id')
        if segment_id:
            realtime_segment_ids.add(segment_id)
    
    # Calculate overlaps and differences
    common_segments = mapping_segment_ids.intersection(realtime_segment_ids)
    mappable_segments = common_segments.intersection(segments_with_edges)
    
    only_in_mapping = mapping_segment_ids - realtime_segment_ids
    only_in_realtime = realtime_segment_ids - mapping_segment_ids
    
    # Create report
    report = {
        "total_mapping_segments": len(mapping_segment_ids),
        "total_realtime_segments": len(realtime_segment_ids),
        "segments_with_edges": len(segments_with_edges),
        "common_segments": len(common_segments),
        "mappable_segments": len(mappable_segments),
        "only_in_mapping": len(only_in_mapping),
        "only_in_realtime": len(only_in_realtime),
        "sample_mappable_segments": list(mappable_segments)[:10],
        "sample_only_in_mapping": list(only_in_mapping)[:10],
        "sample_only_in_realtime": list(only_in_realtime)[:10]
    }
    
    # Save to file
    with open(debug_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Debug file created at {debug_file}")
    logger.info(f"Mappable segments: {len(mappable_segments)} out of {len(realtime_segment_ids)} in realtime data")
    
    return report
    
def get_route_edges(net, from_edge, to_edge):
    """Get the route edges between two edges"""
    try:
        # Get the node at the end of the from_edge
        from_node = net.getEdge(from_edge).getToNode()
        # Get the node at the start of the to_edge
        to_node = net.getEdge(to_edge).getFromNode()
        
        # Use Dijkstra's algorithm to find the shortest path
        router = sumolib.routing.dijkstra.Router(net.getEdges())
        route = router.compute_path(from_node, to_node, 0)  # 0 is the departure time
        
        if route:
            # Convert the path to edge IDs
            return [edge.getID() for edge in route]
        else:
            return None
    except Exception as e:
        logger.error(f"Error finding route: {e}")
        return None

def create_time_varying_demand(processed_data, output_dir, simulation_period, duarouter_routes_file=None):
    """Create initial traffic demand files for SUMO with TraCI control"""
    logger.info("Creating initial traffic demand for TraCI-controlled simulation...")
    
    # Sort processed data by timestamp
    processed_data.sort(key=lambda x: x['timestamp'])
    
    # Create vehicle types XML
    vehicle_types_file = os.path.join(output_dir, "baseline_vtypes.xml")
    with open(vehicle_types_file, "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n')
        
        # Vehicle parameters
        vehicle_params = {
            'passenger': {'color': '0,0,1', 'accel': 2.6, 'decel': 4.5, 'length': 4.5, 'minGap': 2.5, 'maxSpeed': 15, 'shape': 'passenger'},
            'taxi': {'color': '1,1,0', 'accel': 2.8, 'decel': 4.5, 'length': 4.5, 'minGap': 2.0, 'maxSpeed': 16, 'shape': 'passenger'},
            'delivery': {'color': '1,0,0', 'accel': 2.4, 'decel': 4.0, 'length': 6.5, 'minGap': 3.0, 'maxSpeed': 14, 'shape': 'delivery'},
            'bus': {'color': '0,1,0', 'accel': 2.0, 'decel': 3.5, 'length': 12.0, 'minGap': 3.5, 'maxSpeed': 13, 'shape': 'bus'},
            'emergency': {'color': '1,0,0', 'accel': 3.0, 'decel': 5.0, 'length': 6.0, 'minGap': 2.5, 'maxSpeed': 20, 'shape': 'emergency'}
        }
        
        for vtype, params in vehicle_params.items():
            f.write(f'    <vType id="{vtype}" vClass="{vtype}" color="{params["color"]}" '
                    f'accel="{params["accel"]}" decel="{params["decel"]}" sigma="0.5" '
                    f'length="{params["length"]}" minGap="{params["minGap"]}" maxSpeed="{params["maxSpeed"]}" '
                    f'speedDev="0.1" guiShape="{params["shape"]}"/>\n')
        
        f.write('</routes>\n')
    
    # Vehicle type distribution
    vehicle_types = {
        'passenger': 0.6,
        'taxi': 0.25,
        'delivery': 0.1,
        'bus': 0.05
    }
    
    # Create minimal initial flows file (TraCI will handle dynamic insertion)
    flows_file = os.path.join(output_dir, "baseline_flows.xml")
    with open(flows_file, "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n')
        f.write('    <include href="baseline_vtypes.xml"/>\n\n')
        
        # Import routes if available
        if duarouter_routes_file and os.path.exists(duarouter_routes_file):
            logger.info(f"Importing routes from duarouter file: {duarouter_routes_file}")
            try:
                tree = ET.parse(duarouter_routes_file)
                root = tree.getroot()
                
                # Check what's in the duarouter file
                vehicle_count = 0
                route_count = 0
                
                # First, look for routes directly
                for route in root.findall('.//route'):
                    edges = route.get('edges')
                    route_id = route.get('id')
                    if edges and route_id:
                        f.write(f'    <route id="{route_id}" edges="{edges}"/>\n')
                        route_count += 1
                
                # Also look for vehicles with routes
                for vehicle in root.findall('.//vehicle'):
                    route = vehicle.find('route')
                    if route is not None:
                        edges = route.get('edges')
                        if edges:
                            route_id = f"route_{vehicle.get('id')}"
                            f.write(f'    <route id="{route_id}" edges="{edges}"/>\n')
                            route_count += 1
                            
                    vehicle_count += 1
                
                logger.info(f"Found {vehicle_count} vehicles and {route_count} routes in duarouter file")
                
            except Exception as e:
                logger.error(f"Error parsing duarouter routes file: {e}")
        
        f.write('</routes>\n')
    
    # Create additional files for incidents only (speed control via TraCI)
    additional_files = []
    
    # Just create incident files, no calibrators needed with TraCI
    for timestep_idx, timestep_data in enumerate(processed_data):
        incidents = timestep_data.get('incidents', [])
        
        if incidents:
            incident_file = os.path.join(output_dir, f"incidents_{timestep_idx}.add.xml")
            with open(incident_file, "w") as f:
                f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
                f.write('<additional>\n')
                
                for incident_idx, incident in enumerate(incidents):
                    affected_edges = incident.get('affected_edges', [])
                    criticality = incident.get('criticalityInfo', {}).get('factor', 0)
                    
                    for edge_id in affected_edges:
                        if criticality >= 7:  # High criticality - close edge
                            timestamp = timestep_data['timestamp']
                            begin_time = (timestamp - processed_data[0]['timestamp']).total_seconds()
                            
                            f.write(f'    <rerouter id="incident_{edge_id}_{incident_idx}" edges="{edge_id}">\n')
                            f.write(f'        <interval begin="{begin_time}" end="{simulation_period}">\n')
                            f.write(f'            <closingReroute id="{edge_id}"/>\n')
                            f.write('        </interval>\n')
                            f.write('    </rerouter>\n')
                
                f.write('</additional>\n')
            additional_files.append(incident_file)
    
    return {
        'vehicle_types_file': vehicle_types_file,
        'flows_file': flows_file,
        'additional_files': additional_files
    }

def validate_speeds_match(sumo_edge_speeds, here_edge_speeds, tolerance=0.1):
    """Validate if SUMO speeds match HERE API speeds within tolerance"""
    matches = {}
    mismatches = {}
    
    for edge_id, here_speed in here_edge_speeds.items():
        if edge_id in sumo_edge_speeds:
            sumo_speed = sumo_edge_speeds[edge_id]
            
            # Calculate percentage difference
            if here_speed > 0:
                diff_percent = abs(sumo_speed - here_speed) / here_speed
                
                if diff_percent <= tolerance:
                    matches[edge_id] = {
                        'sumo': sumo_speed,
                        'here': here_speed,
                        'diff_percent': diff_percent
                    }
                else:
                    mismatches[edge_id] = {
                        'sumo': sumo_speed,
                        'here': here_speed,
                        'diff_percent': diff_percent
                    }
    
    match_rate = len(matches) / (len(matches) + len(mismatches)) if (matches or mismatches) else 0
    
    # Calculate average speeds
    here_values = [v['here'] for v in list(matches.values()) + list(mismatches.values())]
    sumo_values = [v['sumo'] for v in list(matches.values()) + list(mismatches.values())]
    
    here_avg_speed = sum(here_values) / len(here_values) if here_values else 0
    sumo_avg_speed = sum(sumo_values) / len(sumo_values) if sumo_values else 0
    
    return {
        'match_rate': match_rate,
        'matches': matches,
        'mismatches': mismatches,
        'here_avg_speed': here_avg_speed,
        'sumo_avg_speed': sumo_avg_speed
    }

def run_traci_controlled_simulation(net_file, route_files, processed_data, output_dir, 
                                  simulation_period, visualize=False):
    """
    Run SUMO simulation with TraCI control for dynamic speed and flow management
    
    This function manages the simulation in real-time, adjusting vehicle numbers and speeds
    to match HERE API data without restarting the simulation.
    
    Args:
        net_file: SUMO network file
        route_files: Initial route definitions
        processed_data: List of timestep data from HERE API
        output_dir: Directory for output files
        simulation_period: Total simulation time in seconds
        visualize: Whether to use SUMO-GUI
    """
    logger.info("Starting TraCI-controlled simulation...")

    # # Calculate realistic vehicle insertion based on HERE pattern
    # def calculate_vehicle_insertion_rate(current_timestep_data, prev_timestep_data, current_vehicles, step):
    #     """Calculate how many vehicles to insert based on real traffic growth pattern"""
    #     if not prev_timestep_data:
    #         target_total = current_timestep_data.get('total_vehicles', 3000)
    #         return max(0, min(target_total - current_vehicles, 100))
        
    #     # Get real vehicle growth rate
    #     current_total = current_timestep_data.get('total_vehicles', 0)
    #     prev_total = prev_timestep_data.get('total_vehicles', 0)
    #     real_growth = current_total - prev_total
        
    #     real_jam_change = (current_timestep_data.get('avg_jam_factor', 0) - 
    #                     prev_timestep_data.get('avg_jam_factor', 0))
        
    #     # Scale based on network capacity
    #     network_capacity = 4000  # Estimated for Manhattan simulation
    #     current_load_ratio = current_vehicles / network_capacity
    #     capacity_factor = max(0.1, min(1.0, (network_capacity - current_vehicles) / network_capacity))
        
    #     # Adjust insertion based on congestion
    #     if real_jam_change > 0.1:  # Increasing congestion
    #         insertion_multiplier = 1.3  # More aggressive
    #     elif real_jam_change < -0.1:  # Decreasing congestion  
    #         insertion_multiplier = 0.7  # More conservative
    #     else:
    #         insertion_multiplier = 1.0
        
    #     # Calculate target insertion (scale down from real world)
    #     base_insertion = max(0, real_growth * 0.25)  # Use 25% of real growth
    #     adjusted_insertion = int(base_insertion * capacity_factor * insertion_multiplier)
        
    #     # Dynamic cap based on current load
    #     max_insertion = int(150 * (1 - current_load_ratio))
    #     max_insertion = max(50, min(max_insertion, 300))  # Between 50-300 vehicles
        
    #     return min(adjusted_insertion, max_insertion)

    # Use a dictionary to track statistics
    stats = {
        'duarouter_routes_used': 0,
        'dynamic_routes_created': 0
    }
    # Prepare SUMO command
    sumo_binary = 'sumo-gui' if visualize else 'sumo'
    sumo_cmd = [
        sumolib.checkBinary(sumo_binary),
        '--net-file', net_file,
        '--route-files', route_files,
        '--begin', '0',
        '--end', str(simulation_period),
        '--step-length', '1.0',  # 1 second per simulation step
        '--no-warnings', 'true',
        '--no-step-log', 'true',
        '--duration-log.statistics', 'true',
        '--tripinfo-output', os.path.join(output_dir, 'tripinfo.xml'),
        '--summary-output', os.path.join(output_dir, 'summary.xml'),
        '--statistic-output', os.path.join(output_dir, 'statistics.xml'),
        '--queue-output', os.path.join(output_dir, 'queue.xml'),
        # '--queue-output.period', '5',           # Report every 5 seconds
        # '--time-to-teleport', '900',           # Increase from 300 to 900 seconds
        # '--time-to-teleport.highways', '1800', # 30 minutes for highways
        # '--collision.action', 'warn',          # Don't remove vehicles on collision
        # '--ignore-junction-blocker', '120',    # Wait 2 minutes before ignoring blockers
        # '--lateral-resolution', '0.8',        # Better lane change modeling
        # '--step-method.ballistic', 'true',     # More realistic vehicle dynamics
        # '--device.rerouting.probability', '0.3',  # 30% of vehicles can reroute
        # '--device.rerouting.period', '300',    # Reroute every 5 minutes
        # '--vehroute-output', os.path.join(output_dir, 'vehroutes.xml'),  # Track actual routes
    ]
    
    # Add incident files (for edge closures)
    for additional_file in glob.glob(os.path.join(output_dir, '*incidents*.add.xml')):
        sumo_cmd.extend(['--additional-files', additional_file])
    
    # Start TraCI connection
    traci.start(sumo_cmd)
    
    # Debug route loading
    available_routes = list(traci.route.getIDList())
    available_edges = list(traci.edge.getIDList())
    logger.info(f"Routes available in TraCI: {len(available_routes)}")
    logger.info(f"Edges available in TraCI: {len(available_edges)}")
    if available_routes:
        logger.info(f"Sample routes: {available_routes[:5]}")
    if not available_routes and available_edges:
        logger.warning("No routes loaded but edges available. Will use edge-based routing.")

    try:
        step = 0  # Current simulation step (in seconds)
        vehicle_counter = 0  # Counter for unique vehicle IDs
        last_adjustment_time = {}  # Track when we last adjusted each edge's speed
        
        # Get initial vehicle count from HERE data
        # Example: If HERE shows 5000 vehicles at start
        initial_data = processed_data[0]
        initial_vehicles = initial_data.get('total_vehicles', 5000)
        current_vehicles = 0  # Will track actual vehicles in simulation
        
        # Vehicle type distribution for new vehicles
        # Example: 60% passenger cars, 25% taxis, etc.
        vehicle_types = {
            'passenger': 0.6,
            'taxi': 0.25,
            'delivery': 0.1,
            'bus': 0.05
        }
        
        # Track vehicles that have completed their trips
        vehicles_left = 0  # Total count of vehicles that left
        vehicles_teleported = 0  # Count teleported vehicles
        active_vehicles = set()  # Set of currently active vehicle IDs
        waiting_vehicles = 0  # Vehicles waiting to be inserted
        running_vehicles = 0  # Vehicles currently in the simulation
        vehicle_states = {}  # Track detailed vehicle states
        last_vehicle_change_time = 0  # Initialize  variable here
        transition_start_time = 0

        # Main simulation loop - runs every second
        while step < simulation_period:
            # Find which HERE timestep we're currently in
            # Example: If step=3700 and we have data for every hour, we'd be in the 2nd timestep
            current_timestep_data = get_current_here_timestep(processed_data, step)
            # Get simulation statistics
            waiting_vehicles = traci.simulation.getLoadedNumber() - traci.simulation.getDepartedNumber()
            running_vehicles = traci.vehicle.getIDCount()
            
            # Track teleports (indicates jams or other issues)
            current_teleports = traci.simulation.getStartingTeleportNumber()
            if step % 60 == 0:  # Log every minute
                logger.info(f"Step {step}: Running={running_vehicles}, Waiting={waiting_vehicles}, Teleports={current_teleports}")

            if current_timestep_data:
                # Get target values from HERE data for this timestep
                target_total_vehicles = current_timestep_data.get('total_vehicles', initial_vehicles)
                target_edge_speeds = current_timestep_data.get('edge_speeds', {})  # {edge_id: speed_kmh}
                incidents = current_timestep_data.get('incidents', [])
                
                # Track vehicle turnover (vehicles entering and leaving network)
                # Example: If we had {veh1, veh2, veh3} and now have {veh2, veh3, veh4, veh5}
                # then new_vehicles = {veh4, veh5}, left_vehicles = {veh1}
                current_active = set(traci.vehicle.getIDList())
                new_vehicles = current_active - active_vehicles  # Vehicles that just entered
                left_vehicles = active_vehicles - current_active  # Vehicles that just left
                vehicles_left += len(left_vehicles)  # Update total count
                active_vehicles = current_active  # Update active set for next iteration
                current_vehicles = len(active_vehicles)
                

                # Determine if we're in transition period (first 60 seconds of new timestep)
                # Example: If we just switched from timestep 0 to timestep 1 at step=3600,
                # we're in transition until step=3660
                timestep_idx = processed_data.index(current_timestep_data)
                if timestep_idx > 0:
                    prev_timestamp = processed_data[timestep_idx-1]['timestamp']
                    current_timestamp = current_timestep_data['timestamp']
                    timestep_start = (current_timestamp - processed_data[0]['timestamp']).total_seconds()
                    time_since_transition = step - timestep_start
                    in_transition = time_since_transition < 60
                    # Track transition start
                    if time_since_transition == 0:
                        transition_start_time = step
                        logger.info(f"Starting transition to timestep {timestep_idx} at step {step}")
                else:
                    in_transition = step < 60  # First minute of simulation
                
                # Vehicle management - check every 10 seconds during transition, 30 seconds otherwise
                check_interval = 10 if in_transition else 30

                

                # Calculate how many vehicles to add based on speed changes
                # This runs every 30 seconds
                if timestep_idx > 0 and step % check_interval == 0:
                    prev_data = processed_data[timestep_idx-1]
                    prev_total = prev_data.get('total_vehicles', initial_vehicles)
                    
                    # Identify edges affected by incidents (we'll exclude these from speed analysis)
                    incident_edges = set()
                    for incident in incidents:
                        incident_edges.update(incident.get('affected_edges', []))
                    
                    # Get speeds only from non-incident edges
                    # Example: If we have speeds for edges A,B,C,D but C has an incident,
                    # we only consider A,B,D for average speed calculation
                    non_incident_speeds = {k: v for k, v in target_edge_speeds.items() 
                                         if k not in incident_edges}
                    
                    if non_incident_speeds:
                        # Calculate average speed change
                        current_avg_speed = np.mean(list(non_incident_speeds.values()))
                        
                        prev_non_incident = {k: v for k, v in prev_data.get('edge_speeds', {}).items() 
                                           if k not in incident_edges}
                        prev_avg_speed = np.mean(list(prev_non_incident.values())) if prev_non_incident else current_avg_speed
                        
                        # Vehicle insertion logic based on speed change
                        if prev_avg_speed > 0:
                            # Use direct target from HERE data without artificial scaling
                            target_total_vehicles = current_timestep_data.get('total_vehicles', initial_vehicles)
                            vehicles_to_add = max(0, target_total_vehicles - current_vehicles)
                            vehicles_to_add = min(vehicles_to_add, 300)  # Cap per insertion to avoid overload
                            # vehicles_to_add = calculate_vehicle_insertion_rate(
                            #     current_timestep_data, prev_data, current_vehicles, step
                            # )
                        else:
                            vehicles_to_add = max(0, target_total_vehicles - current_vehicles)
                            vehicles_to_add = min(vehicles_to_add, 100)  # Conservative cap
                        # # Vehicle insertion logic based on speed change
                        # if prev_avg_speed > 0:
                        #     speed_ratio = current_avg_speed / prev_avg_speed
                        #     vehicle_difference = target_total_vehicles - current_vehicles
                            
                        #     # Example calculations:
                        #     # If speed decreased from 50 to 40 km/h (ratio = 0.8):
                        #     #   - More congestion, need more vehicles
                        #     #   - If prev had 1000 vehicles, add 1000 * (1.0 - 0.8) = 200 vehicles
                        #     #
                        #     # If speed increased from 40 to 50 km/h (ratio = 1.25):
                        #     #   - Less congestion, vehicles flow out naturally
                        #     #   - Only add minimal vehicles if needed
                            
                        #     # More nuanced vehicle insertion/removal logic
                        #     if speed_ratio < 0.95:  # Speed decreased by more than 5%
                        #         # Need more vehicles
                        #         if vehicle_difference > 0:
                        #             vehicles_to_add = vehicle_difference
                        #         else:
                        #             # Speed decreased but we already have enough vehicles
                        #             vehicles_to_add = int(abs(vehicle_difference) * 0.1)  # Add just a few
                        #     elif speed_ratio > 1.05:  # Speed increased by more than 5%
                        #         # Need fewer vehicles - rely on natural outflow
                        #         if vehicle_difference < 0:
                        #             # We have too many vehicles
                        #             vehicles_to_add = 0
                        #             # Consider removing vehicles if significantly over
                        #             if abs(vehicle_difference) > 0.1 * target_total_vehicles:
                        #                 vehicles_to_remove = min(50, int(abs(vehicle_difference) * 0.2))
                        #                 remove_vehicles(vehicles_to_remove)
                        #                 logger.debug(f"Speed increasing: Removed {vehicles_to_remove} vehicles")
                        #         else:
                        #             # Still need some vehicles but add fewer
                        #             vehicles_to_add = int(vehicle_difference * 0.3)
                        #     else:  # Speed roughly stable
                        #         vehicles_to_add = max(0, int(vehicle_difference * 0.5))
                    
                        #     # Gradual adjustment during transition 
                        #     if in_transition:
                        #         # During transition, be more aggressive but not excessive
                        #         # Scale based on how far through the transition we are
                        #         transition_progress = (step - transition_start_time) / 60.0
                        #         adjustment_factor = 2.0 - transition_progress  # 2.0 to 1.0
                        #         vehicles_to_add = int(vehicles_to_add * adjustment_factor)
                            
                        #     # Debug logging
                        #     if step % 30 == 0:
                        #         logger.debug(f"Step {step}: Current vehicles={current_vehicles}, "
                        #                    f"Target={target_total_vehicles}, "
                        #                    f"Speed ratio={speed_ratio:.2f}, "
                        #                    f"To add={vehicles_to_add}")
                            
                        #     # Warn if no vehicles being added but we're under target
                        #     if vehicles_to_add == 0 and current_vehicles < target_total_vehicles * 0.9:
                        #         logger.warning(f"Step {step}: Under target vehicles "
                        #                      f"({current_vehicles}/{target_total_vehicles}) "
                        #                      f"but not adding due to speed conditions")
                        # else:
                        #     vehicles_to_add = max(0, target_total_vehicles - current_vehicles)
                        #     MAX_INSERTIONS_PER_STEP = 100
                        #     vehicles_to_add = min(vehicles_to_add, MAX_INSERTIONS_PER_STEP)
                    else:
                        vehicles_to_add = max(0, target_total_vehicles - current_vehicles)
                        MAX_INSERTIONS_PER_STEP = 100
                        vehicles_to_add = min(vehicles_to_add, MAX_INSERTIONS_PER_STEP)
                    
                    # Insert vehicles
                    if vehicles_to_add > 0:
                        actually_added, route_type = insert_vehicles(vehicles_to_add, vehicle_types, vehicle_counter)
                        vehicle_counter += actually_added

                        # Update the tracking counters
                        if route_type == "duarouter":
                            stats['duarouter_routes_used'] += actually_added
                        else:  # route_type == "dynamic"
                            stats['dynamic_routes_created'] += actually_added
                        last_vehicle_change_time = step
                        
                        if actually_added < vehicles_to_add:
                            logger.warning(f"Could only add {actually_added}/{vehicles_to_add} vehicles")
                
                # Speed validation and adjustment (more frequent during transition)
                validation_interval = 15 if in_transition else 30

                if step % validation_interval == 0 and step > 0:
                    actual_edge_speeds = measure_edge_speeds(target_edge_speeds)
                    validation = validate_speeds_match(actual_edge_speeds, target_edge_speeds, tolerance=0.1)
                    
                    if validation['match_rate'] < 0.9:
                        logger.info(f"Step {step}: Speed match rate {validation['match_rate']:.1%}")
                        
                        # Adjust speeds with more careful control
                        for edge_id, mismatch_info in validation['mismatches'].items():
                            if edge_id in last_adjustment_time and step - last_adjustment_time[edge_id] < validation_interval:
                                continue
                            
                            actual_speed = mismatch_info['sumo']
                            target_speed = mismatch_info['here']
                            diff_percent = mismatch_info['diff_percent']
                            
                            # Only adjust if significantly off
                            if diff_percent > 0.1:
                                adjust_edge_speed(edge_id, actual_speed, target_speed, in_transition)
                                last_adjustment_time[edge_id] = step

                # ONLY TWO LOGGING POINTS:

                # 1. Log exactly 1 minute after transition starts
                if current_timestep_data and 'time_since_transition' in locals() and in_transition and time_since_transition == 60:
                    logger.info(f"Post-transition measurement (1 minute after transition):")
                    actual_edge_speeds = measure_edge_speeds(target_edge_speeds)
                    validation = validate_speeds_match(actual_edge_speeds, target_edge_speeds, tolerance=0.15)
                    logger.info(f"Post-transition speed match rate: {validation['match_rate']*100:.1f}%")
                    logger.info(f"HERE avg: {validation['here_avg_speed']:.1f} m/s, SUMO avg: {validation['sumo_avg_speed']:.1f} m/s")

                # 2. Log exactly 10 seconds before next transition
                if current_timestep_data and 'time_since_transition' in locals() and timestep_idx < len(processed_data)-1:
                    next_timestamp = processed_data[timestep_idx+1]['timestamp']
                    current_timestamp = current_timestep_data['timestamp']
                    
                    # Only calculate if time_since_transition is defined
                    if 'time_since_transition' in locals():
                        time_to_next = (next_timestamp - current_timestamp).total_seconds() - time_since_transition
                        
                        # Only log when we're exactly 10 seconds away from next transition
                        if not in_transition and time_to_next == 10:
                            logger.info(f"Pre-transition measurement (10 seconds before next timestep):")
                            actual_edge_speeds = measure_edge_speeds(target_edge_speeds)
                            validation = validate_speeds_match(actual_edge_speeds, target_edge_speeds, tolerance=0.1)
                            logger.info(f"Pre-transition speed match rate: {validation['match_rate']*100:.1f}%")
                            logger.info(f"HERE avg: {validation['here_avg_speed']:.1f} m/s, SUMO avg: {validation['sumo_avg_speed']:.1f} m/s")
            
            # Log if vehicles haven't changed for a while
            if step - last_vehicle_change_time > 120:  # 2 minutes
                # Only log once per 10 minutes (600 seconds)
                if (step - last_vehicle_change_time) % 600 == 0:
                    logger.info(f"No vehicle changes for {(step - last_vehicle_change_time)//60} minutes. "
                            f"Current: {current_vehicles}, Outflow rate: {vehicles_left / step:.2f} veh/s")
            
            # ROUTE DEBUGGING
            # Sample vehicles for route analysis
            if step % 300 == 0 and len(traci.vehicle.getIDList()) > 0:  # Every 5 minutes if vehicles exist
                sample_size = min(20, len(traci.vehicle.getIDList()))
                if sample_size > 0:
                    logger.info(f"==== Route Analysis for {sample_size} Sample Vehicles ====")
                    for veh_id in random.sample(traci.vehicle.getIDList(), sample_size):
                        try:
                            route_edges = traci.vehicle.getRoute(veh_id)
                            route_length = 0
                            for edge in route_edges:
                                try:
                                    route_length += traci.lane.getLength(f"{edge}_0")
                                except:
                                    # Sometimes edge doesn't have lane 0
                                    try:
                                        # Try to get any lane of this edge
                                        lanes = traci.edge.getLaneNumber(edge)
                                        if lanes > 0:
                                            route_length += traci.lane.getLength(f"{edge}_{lanes-1}")
                                    except:
                                        pass
                            
                            # Get vehicle's starting and ending edges
                            from_edge = route_edges[0] if route_edges else "unknown"
                            to_edge = route_edges[-1] if len(route_edges) > 1 else "unknown"
                            
                            # Get current speed and position
                            try:
                                current_speed = traci.vehicle.getSpeed(veh_id)
                                distance_traveled = traci.vehicle.getDistance(veh_id)
                                logger.info(f"Vehicle {veh_id}: Route: {len(route_edges)} edges, {route_length:.1f}m, " 
                                            f"From: {from_edge} To: {to_edge}, "
                                            f"Current speed: {current_speed:.1f} m/s, "
                                            f"Distance traveled: {distance_traveled:.1f}m")
                            except:
                                logger.info(f"Vehicle {veh_id}: Route: {len(route_edges)} edges, {route_length:.1f}m, " 
                                            f"From: {from_edge} To: {to_edge}")
                        except Exception as e:
                            logger.info(f"Error analyzing vehicle {veh_id}: {str(e)}")
                            
            # Advance simulation by one second
            traci.simulationStep()
            step += 1
            
            # Log progress every 5 minutes
            if step % 300 == 0:
                logger.info(f"Simulation step {step}: Active vehicles={current_vehicles}, "
                          f"Vehicles left={vehicles_left}, Total spawned={vehicle_counter}")
                logger.info(f"Route creation stats: Duarouter={stats['duarouter_routes_used']}, Dynamic={stats['dynamic_routes_created']}")
                available_routes = len(traci.route.getIDList())
                logger.info(f"Currently available routes: {available_routes}")
                
                # ADD THE VEHICLE SAMPLING CODE RIGHT AFTER
                # Sample vehicles for route analysis
                vehicle_ids = list(traci.vehicle.getIDList())
                if vehicle_ids:
                    sample_vehicles = random.sample(vehicle_ids, min(5, len(vehicle_ids)))
                    
                    logger.info("=== Sample Vehicle Route Analysis ===")
                    for veh_id in sample_vehicles:
                        try:
                            route_id = traci.vehicle.getRouteID(veh_id)
                            route_edges = traci.vehicle.getRoute(veh_id)
                            
                            # Calculate route length
                            route_length = 0
                            for edge in route_edges:
                                try:
                                    # Skip internal edges for length calculation
                                    if not edge.startswith(':'):
                                        route_length += traci.lane.getLength(f"{edge}_0")
                                except:
                                    pass
                            
                            # Get vehicle's origin and destination edges
                            origin_edge = route_edges[0] if route_edges else "unknown"
                            dest_edge = route_edges[-1] if len(route_edges) > 1 else "unknown"
                            
                            logger.info(f"Vehicle {veh_id}: Route ID={route_id}, Edges={len(route_edges)}, "
                                    f"Length={route_length:.1f}m, Origin={origin_edge}, Dest={dest_edge}")
                        except Exception as e:
                            logger.info(f"Error analyzing vehicle {veh_id}: {e}")

    except Exception as e:
        logger.error(f"Error in TraCI simulation: {e}")
        raise
    finally:
        traci.close()
    
def get_current_here_timestep(processed_data, current_step):
    """Find the HERE data timestep for current simulation time with realistic compression"""
    if not processed_data:
        return None
    
    # Compress 95.6 minutes into simulation time (e.g., 30 minutes = 1800 seconds)
    real_duration_minutes = 95.6
    sim_duration_seconds = 1800  # 30 minutes simulation
    compression_factor = sim_duration_seconds / (real_duration_minutes * 60)
    
    # Map simulation step to real time
    real_time_seconds = current_step / compression_factor
    real_time_minutes = real_time_seconds / 60
    
    # Find corresponding timestep
    start_time = processed_data[0]['timestamp']
    
    for i, timestep_data in enumerate(processed_data):
        # Calculate minutes from start for this timestep
        timestep_minutes = (timestep_data['timestamp'] - start_time).total_seconds() / 60
        
        # Check if current simulation time falls in this interval
        if i == len(processed_data) - 1:  # Last timestep
            if real_time_minutes >= timestep_minutes:
                return timestep_data
        else:
            next_timestep_minutes = (processed_data[i+1]['timestamp'] - start_time).total_seconds() / 60
            if timestep_minutes <= real_time_minutes < next_timestep_minutes:
                return timestep_data
    
    return processed_data[-1]  # Return last timestep if beyond range

def insert_vehicles(count, vehicle_types, start_id):
    """Insert vehicles and return how many were actually added. The insert_vehicles function creates routes on-the-fly when needed
       Only used as a fallback when pre-defined routes from duarouter are unavailable or insufficient (getting fileed up)"""

    # Get available routes from the network
    routes = list(traci.route.getIDList())
    
    # If no routes available, try to get edge information for random routing
    if not routes:
        logger.warning("No routes available in TraCI. Attempting edge-based vehicle insertion.")
        edges = list(traci.edge.getIDList())
        edge_candidates = [e for e in edges if not e.startswith(':')]  # Filter out internal edges
        
        if not edge_candidates:
            logger.error("No viable edges found for vehicle insertion")
            return 0
            
        # Precompute edge properties for better selection
        edge_properties = {}
        for edge_id in edge_candidates:
            try:
                lane_count = traci.edge.getLaneNumber(edge_id)
                length = traci.lane.getLength(f"{edge_id}_0") if lane_count > 0 else 0
                speed_limit = traci.lane.getMaxSpeed(f"{edge_id}_0") if lane_count > 0 else 0
                
                # Score edges - prefer longer edges with multiple lanes and higher speed limits
                edge_score = length * lane_count * speed_limit
                
                edge_properties[edge_id] = {
                    'lane_count': lane_count,
                    'length': length,
                    'speed_limit': speed_limit,
                    'score': edge_score
                }
            except:
                # Skip edges that cause errors
                continue
        
        # Filter edges with sufficient properties
        good_edge_candidates = [e for e in edge_properties.keys() 
                               if edge_properties[e]['lane_count'] >= 2 and 
                                  edge_properties[e]['length'] >= 100 and
                                  edge_properties[e]['speed_limit'] >= 10]
        
        if good_edge_candidates:
            edge_candidates = good_edge_candidates
            logger.info(f"Using {len(edge_candidates)} filtered high-quality edges for vehicle insertion")
        
        # Create simple routes between random edges
        added = 0
        for i in range(count):
            # Select vehicle type
            rand_val = random.random()
            cumulative = 0
            selected_type = 'passenger'
            for vtype, prob in vehicle_types.items():
                cumulative += prob
                if rand_val < cumulative:
                    selected_type = vtype
                    break
            
            # Select random origin and destination edges with minimum route length
            from_edge = random.choice(edge_candidates)
            to_edge = random.choice(edge_candidates)

            # Make sure they're different and far enough apart
            attempts = 0
            max_attempts = 50  # Prevent infinite loops
            min_edge_count = 5  # Minimum edges in route
            min_route_length = 1000  # Minimum route length in meters

            while attempts < max_attempts:
                if from_edge == to_edge:
                    to_edge = random.choice(edge_candidates)
                    attempts += 1
                    continue
                    
                # Try to find a route between from_edge and to_edge
                try:
                    # Use SUMO's internal router to compute a route
                    route_edges = traci.simulation.findRoute(from_edge, to_edge).edges
                    
                    # Check if route has enough edges
                    if len(route_edges) >= min_edge_count:
                        # Calculate route length
                        route_length = 0
                        for edge_id in route_edges:
                            try:
                                lane_id = f"{edge_id}_0"
                                edge_length = traci.lane.getLength(lane_id)
                                route_length += edge_length
                            except:
                                pass
                        
                        if route_length >= min_route_length:
                            # Valid route found
                            break
                except:
                    # If route finding fails, try again
                    pass
                
                # Try again with a different to_edge
                to_edge = random.choice(edge_candidates)
                attempts += 1
                
            # If we couldn't find a good route after max attempts, just use what we have
            if attempts >= max_attempts:
                logger.debug(f"Could not find long enough route after {max_attempts} attempts")
            
            vehicle_id = f"dynamic_{start_id + i}"
            
            try:
                # Add a vehicle with a route between two edges
                traci.vehicle.add(
                    vehicle_id,
                    "",  # Empty route - will be set by setRoute
                    typeID=selected_type,
                    depart="now",
                    departLane="best",
                    departSpeed="max"
                )
                
                # Set the route after adding the vehicle
                try:
                    traci.vehicle.setRoute(vehicle_id, route_edges)
                except:
                    # Fallback to simple origin-destination if route setting fails
                    traci.vehicle.setRoute(vehicle_id, [from_edge, to_edge])
                    
                added += 1
                
            except Exception as e:
                # Vehicle insertion failed
                logger.debug(f"Failed to insert vehicle {vehicle_id}: {e}")
                continue

        logger.debug(f"Added {added} vehicles using dynamic routes")

        return added, "dynamic"
    # else:
    #     # Use pre-defined routes from duarouter
    #     added = 0
    #     for i in range(count):
    #         # [existing route usage code]
    #         added += 1
    #     # duarouter_routes_used += added
    #     return added
    
    # Original route-based insertion if routes are available
    added = 0
    for i in range(count):
        # Select vehicle type
        rand_val = random.random()
        cumulative = 0
        selected_type = 'passenger'
        for vtype, prob in vehicle_types.items():
            cumulative += prob
            if rand_val < cumulative:
                selected_type = vtype
                break
        
        # Try to insert vehicle on an existing route
        vehicle_id = f"dynamic_{start_id + i}"
        route_id = random.choice(routes)
        
        try:
            traci.vehicle.add(
                vehicle_id,
                route_id,
                typeID=selected_type,
                depart='now',
                departLane='best',
                departSpeed='max'
            )
            added += 1
        except Exception as e:
            # Route might be full or other issue
            logger.debug(f"Failed to insert vehicle {vehicle_id} on route {route_id}: {e}")
            continue
    logger.debug(f"Added {added} vehicles using duarouter routes")

    return added, "duarouter"

def remove_vehicles(count):
    """Remove vehicles that are close to completing their trips"""
    vehicle_ids = traci.vehicle.getIDList()
    candidates = []
    
    for veh_id in vehicle_ids:
        try:
            # Get route progress
            route_index = traci.vehicle.getRouteIndex(veh_id)
            route = traci.vehicle.getRoute(veh_id)
            
            # If vehicle is in last 10% of route, it's a candidate for removal
            if route_index > len(route) * 0.9:
                distance = traci.vehicle.getDistance(veh_id)
                candidates.append((veh_id, distance))
        except:
            continue
    
    # Sort by distance traveled (remove those who've traveled furthest)
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    removed = 0
    for veh_id, _ in candidates[:count]:
        try:
            traci.vehicle.remove(veh_id)
            removed += 1
        except:
            continue
    
    return removed

def measure_edge_speeds(target_edge_speeds):
    """Measure actual speeds on edges"""
    actual_speeds = {}
    
    for edge_id in target_edge_speeds.keys():
        try:
            vehicles_on_edge = traci.edge.getLastStepVehicleIDs(edge_id)
            if vehicles_on_edge:
                speeds = [traci.vehicle.getSpeed(veh_id) for veh_id in vehicles_on_edge]
                actual_speeds[edge_id] = np.mean(speeds) * 3.6  # m/s to km/h
            else:
                mean_speed = traci.edge.getLastStepMeanSpeed(edge_id)
                if mean_speed >= 0:
                    actual_speeds[edge_id] = mean_speed * 3.6
        except:
            continue
    
    return actual_speeds

def adjust_edge_speed(edge_id, actual_speed, target_speed, in_transition):
    """Adjust speed on an edge"""
    if actual_speed <= 0:
        return
    
    speed_factor = target_speed / actual_speed
    
    # More conservative adjustment to prevent oscillation
    if in_transition:
        adjustment_strength = 0.5  # Not 0.7
    else:
        adjustment_strength = 0.2  # Not 0.3
    
    speed_factor = 1.0 + (speed_factor - 1.0) * adjustment_strength
    
    # Apply to vehicles
    vehicles_on_edge = traci.edge.getLastStepVehicleIDs(edge_id)
    for veh_id in vehicles_on_edge:
        try:
            current_speed = traci.vehicle.getSpeed(veh_id)
            new_speed = current_speed * speed_factor
            max_speed = traci.vehicle.getMaxSpeed(veh_id)
            new_speed = min(new_speed, max_speed)
            new_speed = max(new_speed, 0.1)
            traci.vehicle.setSpeed(veh_id, new_speed)
        except:
            pass
    
    # Set edge speed limit
    try:
        traci.edge.setMaxSpeed(edge_id, target_speed / 3.6)
    except:
        pass


def lookup_route_edges(routes_file, route_id):
    """Look up edges for a route ID in a routes file"""
    try:
        tree = ET.parse(routes_file)
        root = tree.getroot()
        
        route = root.find(f'.//route[@id="{route_id}"]')
        if route is not None:
            edges_str = route.get('edges')
            if edges_str:
                return edges_str.split()
    except Exception as e:
        logger.error(f"Error looking up route edges: {e}")
    
    return None

def create_gui_settings(output_dir):
    """Create SUMO GUI settings file"""
    logger.info("Creating GUI settings file...")
    
    gui_settings_file = os.path.join(output_dir, "gui-settings.xml")
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
    with open(gui_settings_file, "w") as f:
        f.write(gui_settings)
    
    logger.info(f"GUI settings file created: {gui_settings_file}")
    return gui_settings_file

def create_sumo_config(net_file, flows_file, additional_files, output_dir, simulation_period):
    """Create SUMO configuration file"""
    logger.info("Creating SUMO configuration...")
    
    config_file = os.path.join(output_dir, "baseline_sim.sumocfg")
    
    # Create a comma-separated list of additional files
    additional_files_str = ",".join([os.path.basename(f) for f in additional_files])
    
    # Create XML configuration using ElementTree to ensure valid XML
    root = ET.Element("configuration")
    root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    root.set("xsi:noNamespaceSchemaLocation", "http://sumo.dlr.de/xsd/sumoConfiguration.xsd")
    
    # Input section
    input_section = ET.SubElement(root, "input")
    net_file_elem = ET.SubElement(input_section, "net-file")
    net_file_elem.set("value", os.path.basename(net_file))
    route_files_elem = ET.SubElement(input_section, "route-files")
    route_files_elem.set("value", os.path.basename(flows_file))
    if additional_files_str:
        additional_files_elem = ET.SubElement(input_section, "additional-files")
        additional_files_elem.set("value", additional_files_str)
    
    # Time section
    time_section = ET.SubElement(root, "time")
    begin_elem = ET.SubElement(time_section, "begin")
    begin_elem.set("value", "0")
    end_elem = ET.SubElement(time_section, "end")
    end_elem.set("value", str(simulation_period))
    step_elem = ET.SubElement(time_section, "step-length")
    step_elem.set("value", "1.0")
    
    # Processing section
    proc_section = ET.SubElement(root, "processing")
    teleport_elem = ET.SubElement(proc_section, "time-to-teleport")
    teleport_elem.set("value", "300")
    blocker_elem = ET.SubElement(proc_section, "ignore-junction-blocker")
    blocker_elem.set("value", "60")
    collision_elem = ET.SubElement(proc_section, "collision.action")
    collision_elem.set("value", "warn")
    random_elem = ET.SubElement(proc_section, "random")
    random_elem.set("value", "true")
    
    # Report section
    report_section = ET.SubElement(root, "report")
    verbose_elem = ET.SubElement(report_section, "verbose")
    verbose_elem.set("value", "true")
    durlog_elem = ET.SubElement(report_section, "duration-log.statistics")
    durlog_elem.set("value", "true")
    log_elem = ET.SubElement(report_section, "log")
    log_elem.set("value", "simulation.log")
    
    # Output section
    output_section = ET.SubElement(root, "output")
    tripinfo_elem = ET.SubElement(output_section, "tripinfo-output")
    tripinfo_elem.set("value", "tripinfo.xml")
    summary_elem = ET.SubElement(output_section, "summary-output")
    summary_elem.set("value", "summary.xml")
    stats_elem = ET.SubElement(output_section, "statistic-output")
    stats_elem.set("value", "statistics.xml")
    queue_elem = ET.SubElement(output_section, "queue-output")
    queue_elem.set("value", "queue.xml")
    lanechange_elem = ET.SubElement(output_section, "lanechange-output")
    lanechange_elem.set("value", "lanechange.xml")
    # Remove output-prefix from here since we handle it in command line
    
    # GUI section
    gui_section = ET.SubElement(root, "gui_only")
    guisettings_elem = ET.SubElement(gui_section, "gui-settings-file")
    guisettings_elem.set("value", "gui-settings.xml")
    start_elem = ET.SubElement(gui_section, "start")
    start_elem.set("value", "false")
    quitonend_elem = ET.SubElement(gui_section, "quit-on-end")
    quitonend_elem.set("value", "false")
    
    # Convert to string
    tree = ET.ElementTree(root)
    with open(config_file, "wb") as f:
        tree.write(f, encoding="utf-8", xml_declaration=True)
    
    logger.info(f"SUMO configuration file created: {config_file}")
    return config_file

def run_simulation(config_file, output_dir, gui=False):
    """Run the SUMO simulation"""
    logger.info("Running SUMO simulation...")
    
    # Determine which SUMO binary to use
    sumo_binary = checkBinary("sumo-gui" if gui else "sumo")
    
    # Use just the directory name as output prefix
    output_prefix = os.path.basename(output_dir)
    
    # Prepare command - use absolute path for config file
    config_file_abs = os.path.abspath(config_file)
    cmd = [
        sumo_binary,
        "-c", config_file_abs,
        "--output-prefix", output_prefix,
        "--no-warnings", "false",
        "--duration-log.statistics", "true"
    ]
    
    try:
        # Run SUMO - run in the output directory
        logger.info(f"Starting SUMO: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=output_dir)
        logger.info("Simulation completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running simulation: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error running simulation: {e}")
        return False

def analyze_trip_info(file_path):
    """Analyze trip information from SUMO output"""
    logger.info(f"Analyzing trip information from {file_path}...")
    
    # Parse XML file
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
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(trips)
    
    # Calculate statistics
    stats = {
        'total_trips': len(trips),
        'completed_trips': len(trips),  # All trips in the file are completed
        'avg_duration': df['duration'].mean() if not df.empty else 0,
        'avg_wait_time': df['wait_time'].mean() if not df.empty else 0,
        'avg_distance': df['distance'].mean() if not df.empty else 0,
        'avg_speed': df['avg_speed'].mean() if not df.empty else 0,
        'total_wait_time': df['wait_time'].sum() if not df.empty else 0,
        'max_wait_time': df['wait_time'].max() if not df.empty else 0,
        'min_wait_time': df['wait_time'].min() if not df.empty else 0
    }
    
    # Add statistics by vehicle type
    vtype_stats = {}
    if not df.empty:
        for vtype in df['vehicle_type'].unique():
            vtype_df = df[df['vehicle_type'] == vtype]
            vtype_stats[vtype] = {
                'count': len(vtype_df),
                'avg_duration': vtype_df['duration'].mean(),
                'avg_wait_time': vtype_df['wait_time'].mean(),
                'avg_speed': vtype_df['avg_speed'].mean()
            }
    
    stats['vehicle_types'] = vtype_stats
    
    return stats

def analyze_queue_data(file_path):
    """Analyze queue information from SUMO output"""
    logger.info(f"Analyzing queue information from {file_path}...")
    
    # Parse XML file
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # Extract queue data for each lane
    queue_data = {}
    
    for step in root.findall('.//timestep'):
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
        'max_queue_any_lane': max([s['max_queue_length'] for s in lane_stats.values()]) if lane_stats else 0,
        'avg_queue_all_lanes': np.mean([s['avg_queue_length'] for s in lane_stats.values()]) if lane_stats else 0,
        'avg_queued_vehicles': np.mean([s['avg_queued_vehicles'] for s in lane_stats.values()]) if lane_stats else 0
    }
    
    return {
        'lane_stats': lane_stats,
        'global_stats': global_stats
    }

def analyze_summary(file_path):
    """Analyze summary information from SUMO output"""
    logger.info(f"Analyzing summary information from {file_path}...")
    
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
        summary_data['vehicles'].append(int(step.get('loaded', 0)))
        summary_data['running'].append(int(step.get('running', 0)))
        summary_data['waiting'].append(int(step.get('waiting', 0)))
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
    
    return stats

def analyze_results(output_dir, processed_data=None):  # Make processed_data optional
    """Analyze simulation results and generate statistics"""
    logger.info("Analyzing simulation results...")
    
    # Define output files
    tripinfo_file = os.path.join(output_dir, "tripinfo.xml")
    summary_file = os.path.join(output_dir, "summary.xml")
    queue_file = os.path.join(output_dir, "queue.xml")
    statistics_file = os.path.join(output_dir, "statistics.xml")
    
    results = {
        'trip_stats': {},
        'flow_stats': {},
        'queue_stats': {},
        'overall_stats': {}
    }
    
    # Analyze trip information if available
    if os.path.exists(tripinfo_file):
        try:
            logger.info(f"Analyzing trip information from {tripinfo_file}")
            trip_stats = analyze_trip_info(tripinfo_file)
            results['trip_stats'] = trip_stats
        except Exception as e:
            logger.error(f"Error analyzing trip information: {e}")
    else:
        logger.warning(f"Trip information file not found: {tripinfo_file}")
    
    # Analyze summary information if available
    if os.path.exists(summary_file):
        try:
            logger.info(f"Analyzing summary information from {summary_file}")
            summary_stats = analyze_summary(summary_file)
            results['overall_stats'].update(summary_stats)
            
            # Also store raw summary data for timeseries visualization
            tree = ET.parse(summary_file)
            root = tree.getroot()
            
            timeseries = {
                'time': [],
                'vehicles': [],
                'running': [],
                'waiting': [],
                'mean_speed': [],
                'mean_waiting_time': []
            }
            
            for step in root.findall('step'):
                timeseries['time'].append(float(step.get('time')))
                timeseries['vehicles'].append(int(step.get('loaded', 0)))
                timeseries['running'].append(int(step.get('running', 0)))
                timeseries['waiting'].append(int(step.get('waiting', 0)))
                timeseries['mean_speed'].append(float(step.get('meanSpeed', 0)))
                timeseries['mean_waiting_time'].append(float(step.get('meanWaitingTime', 0)))
            
            results['timeseries'] = timeseries
        except Exception as e:
            logger.error(f"Error analyzing summary information: {e}")
    else:
        logger.warning(f"Summary file not found: {summary_file}")
    
    # Analyze queue information if available
    if os.path.exists(queue_file):
        try:
            logger.info(f"Analyzing queue information from {queue_file}")
            queue_stats = analyze_queue_data(queue_file)
            results['queue_stats'] = queue_stats
        except Exception as e:
            logger.error(f"Error analyzing queue information: {e}")
    else:
        logger.warning(f"Queue file not found: {queue_file}")
    
    # Save analysis results
    analysis_file = os.path.join(output_dir, "analysis_results.json")
    try:
        with open(analysis_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Analysis results saved to {analysis_file}")
    except Exception as e:
        logger.error(f"Error saving analysis results: {e}")
    
    return results

def create_comparison_csv(output_dir):
    """
    Create a CSV file with timestep metrics similar to HERE API data format
    for comparison between simulation and real-world data
    """
    logger.info("Creating comparison CSV for validation...")
    
    # Define output file
    csv_file = os.path.join(output_dir, "sumo_metrics_comparison.csv")
    
    # Check if summary file exists (contains timestep data)
    summary_file = os.path.join(output_dir, "summary.xml")
    if not os.path.exists(summary_file):
        logger.error(f"Summary file not found: {summary_file}")
        return None
    
    # Check if tripinfo file exists (for trip times and wait times)
    tripinfo_file = os.path.join(output_dir, "tripinfo.xml")
    if not os.path.exists(tripinfo_file):
        logger.warning(f"Tripinfo file not found: {tripinfo_file}")
    
    # Check if queue file exists
    queue_file = os.path.join(output_dir, "queue.xml")
    if not os.path.exists(queue_file):
        logger.warning(f"Queue file not found: {queue_file}")
    
    # Check if statistics file exists
    statistics_file = os.path.join(output_dir, "statistics.xml")
    if not os.path.exists(statistics_file):
        logger.warning(f"Statistics file not found: {statistics_file}")
    
    try:
        # Parse summary file to get time series data
        tree = ET.parse(summary_file)
        root = tree.getroot()
        
        # Create data structure for CSV
        timestamps = []
        metrics = []
        
        # Process each timestep
        for step in root.findall('step'):
            time = float(step.get('time'))
            
            # Get metrics for this timestep
            running = int(step.get('running', 0))
            vehicles = int(step.get('loaded', 0)) + running
            mean_speed = float(step.get('meanSpeed', 0)) * 3.6  # Convert m/s to km/h
            mean_waiting_time = float(step.get('meanWaitingTime', 0))
            halting_vehicles = int(step.get('halting', 0))
            
            # Only record data at regular intervals (every 300 seconds)
            if time % 300 == 0 or time == 0:
                base_time = datetime(2025, 4, 4, 8, 2, 41)  # First HERE timestamp
                actual_time = base_time + timedelta(seconds=time)
                timestamp = actual_time.strftime('%Y-%m-%d %H:%M:%S')
                timestamps.append(timestamp)
                
                # Create metrics dict for this timestep
                metric = {
                    'timestamp': timestamp,
                    'time': time,
                    'total_vehicles': vehicles,
                    'running_vehicles': running,
                    'mean_speed': mean_speed,
                    'mean_waiting_time': mean_waiting_time,
                    'halting_vehicles': halting_vehicles,
                    'avg_trip_time': 0,
                    'avg_wait_time': 0,
                    'avg_queue_length': 0
                }
                metrics.append(metric)
        
        # Get trip times and wait times from tripinfo file
        if os.path.exists(tripinfo_file):
            # Create a mapping of completed trips by time
            trip_times_by_time = {}
            wait_times_by_time = {}
            
            tripinfo_tree = ET.parse(tripinfo_file)
            tripinfo_root = tripinfo_tree.getroot()
            
            for trip in tripinfo_root.findall('tripinfo'):
                # Get trip details
                arrival_time = float(trip.get('arrival', 0))
                duration = float(trip.get('duration', 0))
                wait_time = float(trip.get('waitingTime', 0))
                
                # Round to nearest 5-minute interval
                time_bucket = (int(arrival_time) // 300) * 300
                
                # Initialize time bucket if not exists
                if time_bucket not in trip_times_by_time:
                    trip_times_by_time[time_bucket] = []
                    wait_times_by_time[time_bucket] = []
                
                # Add trip duration and wait time to bucket
                trip_times_by_time[time_bucket].append(duration)
                wait_times_by_time[time_bucket].append(wait_time)
            
            # Calculate average trip times and wait times for each timestamp
            for i, metric in enumerate(metrics):
                time = metric['time']
                time_bucket = (int(time) // 300) * 300
                
                # Calculate average trip time for this timestamp
                if time_bucket in trip_times_by_time and trip_times_by_time[time_bucket]:
                    metric['avg_trip_time'] = sum(trip_times_by_time[time_bucket]) / len(trip_times_by_time[time_bucket])
                
                # Calculate average wait time for this timestamp
                if time_bucket in wait_times_by_time and wait_times_by_time[time_bucket]:
                    metric['avg_wait_time'] = sum(wait_times_by_time[time_bucket]) / len(wait_times_by_time[time_bucket])
        
        # Get queue data and jam factors
        if os.path.exists(queue_file):
            # Create mapping of queue data by time
            queue_data_by_time = {}
            
            queue_tree = ET.parse(queue_file)
            queue_root = queue_tree.getroot()
            
            for step in queue_root.findall('timestep'):
                step_time = float(step.get('time'))
                time_bucket = (int(step_time) // 300) * 300
                
                # Initialize if not exists
                if time_bucket not in queue_data_by_time:
                    queue_data_by_time[time_bucket] = {
                        'queue_lengths': [],
                        'queue_vehicles': []
                    }
                
                # Process queue data for each lane
                for lane in step.findall('lane'):
                    queue_length = float(lane.get('queueing_length', 0))
                    queue_vehicles = int(lane.get('queueing_vehicles', 0))
                    
                    # Only count non-zero queues
                    if queue_length > 0:
                        queue_data_by_time[time_bucket]['queue_lengths'].append(queue_length)
                    if queue_vehicles > 0:
                        queue_data_by_time[time_bucket]['queue_vehicles'].append(queue_vehicles)
            
            # Calculate network length (for vehicle density)
            network_length_km = 100  # Default 100km
            if os.path.exists(statistics_file):
                stats_tree = ET.parse(statistics_file)
                stats_root = stats_tree.getroot()
                network_elem = stats_root.find('.//network')
                if network_elem is not None:
                    network_length_km = float(network_elem.get('totalLength', 0)) / 1000
            
            # Calculate queue stats and jam factors for each timestamp
            for i, metric in enumerate(metrics):
                time = metric['time']
                time_bucket = (int(time) // 300) * 300
                
                if time_bucket in queue_data_by_time:
                    # Calculate average queue length
                    queue_lengths = queue_data_by_time[time_bucket]['queue_lengths']
                    if queue_lengths:
                        metric['avg_queue_length'] = sum(queue_lengths) / len(queue_lengths)
                    
                    # Calculate approximate jam factor (0-10 scale)
                    # Base it on average queue length and percentage of lanes with queues
                    total_queue_length = sum(queue_lengths)
                    
                    # Calculate jam factor from speed ratio (like HERE API does)
                    if 'mean_speed' in metric and metric['mean_speed'] > 0:
                        # Estimate free flow speed for urban Manhattan (around 25-30 km/h)
                        estimated_free_flow = 27.0  # km/h
                        speed_ratio = metric['mean_speed'] / estimated_free_flow
                        
                        # Convert speed ratio to jam factor (0-10 scale)
                        # Speed ratio 1.0 = jam factor 0, speed ratio 0.2 = jam factor 8
                        jam_factor = max(0, min(10, (1.0 - speed_ratio) * 10))
                        metric['avg_jam_factor'] = jam_factor
                    else:
                        metric['avg_jam_factor'] = 0
                else:
                    metric['avg_jam_factor'] = 0
                
                # Calculate vehicle density (vehicles per km)
                metric['avg_vehicle_density'] = metric['total_vehicles'] / network_length_km
        
        # Get incident count from rerouters (estimate)
        incident_counts = {}
        
        # Look for incident files
        incident_files = glob.glob(os.path.join(output_dir, "incidents_*.add.xml"))
        for incident_file in incident_files:
            try:
                inc_tree = ET.parse(incident_file)
                inc_root = inc_tree.getroot()
                
                # Count rerouters in each time interval
                for rerouter in inc_root.findall('.//rerouter'):
                    for interval in rerouter.findall('.//interval'):
                        begin = float(interval.get('begin', 0))
                        end = float(interval.get('end', 0))
                        
                        # Count as incident for entire interval
                        for t in range(int(begin), int(end), 300):
                            rounded_time = (t // 300) * 300  # Round to nearest 5-minute interval
                            if rounded_time not in incident_counts:
                                incident_counts[rounded_time] = 0
                            incident_counts[rounded_time] += 1
            except Exception as e:
                logger.warning(f"Error parsing incident file {incident_file}: {e}")
        
        # Add incident counts to metrics
        for i, metric in enumerate(metrics):
            time = metric['time']
            time_bucket = (int(time) // 300) * 300
            metric['total_incidents'] = incident_counts.get(time_bucket, 0)
        
        # Write to CSV
        with open(csv_file, 'w', newline='') as f:
            fieldnames = ['timestamp', 'total_vehicles', 'avg_vehicle_density', 
                         'avg_jam_factor', 'mean_speed', 'total_incidents', 
                         'avg_trip_time', 'avg_wait_time', 'avg_queue_length']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for metric in metrics:
                writer.writerow({
                    'timestamp': metric['timestamp'],
                    'total_vehicles': metric['total_vehicles'],
                    'avg_vehicle_density': metric['avg_vehicle_density'],
                    'avg_jam_factor': metric['avg_jam_factor'],
                    'mean_speed': metric['mean_speed'],
                    'total_incidents': metric['total_incidents'],
                    'avg_trip_time': metric['avg_trip_time'],
                    'avg_wait_time': metric['avg_wait_time'],
                    'avg_queue_length': metric['avg_queue_length']
                })
        
        logger.info(f"Comparison CSV created: {csv_file}")
        return csv_file
        
    except Exception as e:
        logger.error(f"Error creating comparison CSV: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def create_visualizations(results, output_dir):
    """Create visualizations of simulation results"""
    logger.info("Creating visualizations...")
    
    # Create visualization directory
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Plot trip statistics
    if 'trip_stats' in results and results['trip_stats']:
        trip_stats = results['trip_stats']
        
        # Plot average trip duration by vehicle type
        if 'vehicle_types' in trip_stats:
            veh_types = list(trip_stats['vehicle_types'].keys())
            avg_durations = [trip_stats['vehicle_types'][vt]['avg_duration'] for vt in veh_types]
            avg_wait_times = [trip_stats['vehicle_types'][vt]['avg_wait_time'] for vt in veh_types]
            counts = [trip_stats['vehicle_types'][vt]['count'] for vt in veh_types]
            
            # Vehicle counts by type
            plt.figure(figsize=(10, 6))
            plt.bar(veh_types, counts)
            plt.title('Vehicle Counts by Type')
            plt.ylabel('Number of Vehicles')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(os.path.join(viz_dir, 'vehicle_counts_by_type.png'))
            plt.close()
            
            # Trip durations by vehicle type
            plt.figure(figsize=(10, 6))
            plt.bar(veh_types, avg_durations)
            plt.title('Average Trip Duration by Vehicle Type')
            plt.ylabel('Duration (seconds)')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(os.path.join(viz_dir, 'avg_duration_by_type.png'))
            plt.close()
            
            # Wait times by vehicle type
            plt.figure(figsize=(10, 6))
            plt.bar(veh_types, avg_wait_times)
            plt.title('Average Wait Time by Vehicle Type')
            plt.ylabel('Wait Time (seconds)')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(os.path.join(viz_dir, 'avg_wait_time_by_type.png'))
            plt.close()
    
    # Plot queue statistics
    if 'queue_stats' in results and 'global_stats' in results['queue_stats']:
        queue_stats = results['queue_stats']
        
        # If we have lane-specific stats, plot the top congested lanes
        if 'lane_stats' in queue_stats and queue_stats['lane_stats']:
            # Sort lanes by maximum queue length
            sorted_lanes = sorted(
                queue_stats['lane_stats'].items(),
                key=lambda x: x[1]['max_queue_length'],
                reverse=True
            )
            
            # Take top 15 lanes for readability
            top_lanes = sorted_lanes[:15]
            lane_ids = [lane[0] for lane in top_lanes]
            max_queue_lengths = [lane[1]['max_queue_length'] for lane in top_lanes]
            avg_queue_lengths = [lane[1]['avg_queue_length'] for lane in top_lanes]
            
            # Maximum queue lengths
            plt.figure(figsize=(12, 8))
            plt.barh(lane_ids, max_queue_lengths)
            plt.title('Maximum Queue Length by Lane (Top 15)')
            plt.xlabel('Queue Length (meters)')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'max_queue_by_lane.png'))
            plt.close()
            
            # Average queue lengths
            plt.figure(figsize=(12, 8))
            plt.barh(lane_ids, avg_queue_lengths)
            plt.title('Average Queue Length by Lane (Top 15)')
            plt.xlabel('Queue Length (meters)')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'avg_queue_by_lane.png'))
            plt.close()
    
    # Plot timeseries data if available from summary
    if 'timeseries' in results:
        timeseries = results['timeseries']
        
        # Plot vehicles over time
        if 'time' in timeseries and 'running' in timeseries:
            plt.figure(figsize=(12, 6))
            plt.plot(timeseries['time'], timeseries['running'], label='Running')
            if 'waiting' in timeseries:
                plt.plot(timeseries['time'], timeseries['waiting'], label='Waiting')
            plt.title('Vehicles Over Time')
            plt.xlabel('Simulation Time (seconds)')
            plt.ylabel('Number of Vehicles')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'vehicles_over_time.png'))
            plt.close()
        
        # Plot mean speed over time
        if 'time' in timeseries and 'mean_speed' in timeseries:
            plt.figure(figsize=(12, 6))
            plt.plot(timeseries['time'], timeseries['mean_speed'])
            plt.title('Mean Speed Over Time')
            plt.xlabel('Simulation Time (seconds)')
            plt.ylabel('Mean Speed (m/s)')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'mean_speed_over_time.png'))
            plt.close()
    
    # Create overall statistics visualization
    if 'trip_stats' in results or 'overall_stats' in results:
        # Combine key statistics
        key_stats = {}
        if 'trip_stats' in results:
            key_stats.update({
                'Total Trips': results['trip_stats'].get('total_trips', 0),
                'Avg Trip Duration (s)': results['trip_stats'].get('avg_duration', 0),
                'Avg Wait Time (s)': results['trip_stats'].get('avg_wait_time', 0),
                'Avg Trip Distance (m)': results['trip_stats'].get('avg_distance', 0),
                'Avg Trip Speed (m/s)': results['trip_stats'].get('avg_speed', 0)
            })
        
        if 'overall_stats' in results:
            key_stats.update({
                'Simulation Duration (s)': results['overall_stats'].get('simulation_duration', 0),
                'Max Vehicles': results['overall_stats'].get('max_vehicles', 0),
                'Avg Mean Speed (m/s)': results['overall_stats'].get('avg_mean_speed', 0)
            })
        
        if 'queue_stats' in results and 'global_stats' in results['queue_stats']:
            key_stats.update({
                'Max Queue Length (m)': results['queue_stats']['global_stats'].get('max_queue_any_lane', 0),
                'Avg Queue Length (m)': results['queue_stats']['global_stats'].get('avg_queue_all_lanes', 0),
                'Lanes with Queues': results['queue_stats']['global_stats'].get('total_lanes_with_queues', 0)
            })
        
        # Create key statistics figure
        plt.figure(figsize=(15, 10))
        plt.bar(key_stats.keys(), key_stats.values())
        plt.title('Key Simulation Statistics')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(viz_dir, 'key_statistics.png'))
        plt.close()
    
    logger.info(f"Visualizations created in {viz_dir}")
    return viz_dir

def find_sumo_edges_for_segment(segment, net, max_radius=50.0):
    """Find SUMO edges corresponding to a HERE API segment"""
    # Get shape points
    shape_points = segment.get('shape', [])
    if not shape_points or len(shape_points) < 2:
        return []
    
    # Get start and end points
    start_point = shape_points[0]
    end_point = shape_points[-1]
    
    matched_edges = set()
    
    # Try to match start and end points
    if len(start_point) >= 2 and len(end_point) >= 2:
        start_lat, start_lon = start_point[0], start_point[1]
        end_lat, end_lon = end_point[0], end_point[1]
        
        try:
            # Convert to SUMO x,y coordinates
            start_x, start_y = net.convertLonLat2XY(start_lon, start_lat)
            end_x, end_y = net.convertLonLat2XY(end_lon, end_lat)
            
            # Find nearby edges for start and end
            start_edges = net.getNeighboringEdges(start_x, start_y, max_radius)
            end_edges = net.getNeighboringEdges(end_x, end_y, max_radius)
            
            # Sort by distance
            if start_edges:
                start_edges.sort(key=lambda x: x[1])
                matched_edges.add(start_edges[0][0].getID())
            
            if end_edges:
                end_edges.sort(key=lambda x: x[1])
                matched_edges.add(end_edges[0][0].getID())
        except Exception as e:
            logger.warning(f"Error finding edges: {e}")
    
    # For longer segments, try to match intermediate points
    if len(shape_points) > 2 and len(matched_edges) > 0:
        # Sample up to 3 additional points along the segment
        sample_count = min(3, len(shape_points) - 2)
        step = max(1, (len(shape_points) - 2) // (sample_count + 1))
        
        for i in range(1, len(shape_points) - 1, step):
            if len(matched_edges) >= 5:  # Limit to 5 edges per segment
                break
            
            point = shape_points[i]
            if len(point) >= 2:
                mid_lat, mid_lon = point[0], point[1]
                try:
                    mid_x, mid_y = net.convertLonLat2XY(mid_lon, mid_lat)
                    mid_edges = net.getNeighboringEdges(mid_x, mid_y, max_radius)
                    
                    if mid_edges:
                        mid_edges.sort(key=lambda x: x[1])
                        matched_edges.add(mid_edges[0][0].getID())
                except Exception:
                    continue
    
    return list(matched_edges)

def create_direct_here_to_sumo_mapping(edge_mapping_data, entry_exit_data, net):
    """Create a direct mapping between HERE API segments and SUMO edges"""
    logger.info("Creating direct HERE API to SUMO edge mapping...")
    
    # Create mapping dictionary
    segment_to_edge = {}
    
    # First, collect all entry and exit points with their coordinates
    all_points = []
    
    # Add entry points
    for i, entry in enumerate(entry_exit_data.get('entry_points', [])):
        if 'location' in entry and len(entry['location']) >= 2:
            lat, lon = entry['location'][0], entry['location'][1]
            point_info = {
                'id': f'entry_{i}',
                'lat': lat,
                'lon': lon,
                'weight': entry.get('weight', 1),
                'type': 'entry',
                'road_name': entry.get('road_name', 'Unknown')
            }
            all_points.append(point_info)
    
    # Add exit points
    for i, exit in enumerate(entry_exit_data.get('exit_points', [])):
        if 'location' in exit and len(exit['location']) >= 2:
            lat, lon = exit['location'][0], exit['location'][1]
            point_info = {
                'id': f'exit_{i}',
                'lat': lat,
                'lon': lon,
                'weight': exit.get('weight', 1),
                'type': 'exit',
                'road_name': exit.get('road_name', 'Unknown')
            }
            all_points.append(point_info)
    
    logger.info(f"Collected {len(all_points)} entry/exit points")
    
    # Get all edges from the network
    all_edges = net.getEdges()
    logger.info(f"Network has {len(all_edges)} edges")
    
    # Create a spatial index for faster lookups
    edge_shapes = {}
    for edge in all_edges:
        edge_id = edge.getID()
        shape = edge.getShape()
        edge_shapes[edge_id] = shape
    
    # For each point, find the best matching edge by name
    points_with_edges = 0
    
    for point in all_points:
        road_name = point.get('road_name', '').lower()
        lat, lon = point.get('lat'), point.get('lon')
        
        # Try to find edges with similar names
        best_match = None
        best_score = 0
        
        for edge in all_edges:
            edge_id = edge.getID()
            # Many SUMO edge IDs contain street names
            if road_name and len(road_name) > 3:
                # Check if road name is part of edge ID (simple string matching)
                if road_name in edge_id.lower():
                    score = len(road_name) / len(edge_id) # Longer match = higher score
                    if score > best_score:
                        best_score = score
                        best_match = edge_id
        
        # If no name match, fall back to coordinate-based matching
        if not best_match:
            # Convert to SUMO coordinates
            try:
                x, y = net.convertLonLat2XY(lon, lat)
                
                # Find the nearest edge - start with a small radius
                radius = 50.0
                max_radius = 500.0
                edges_found = False
                
                while not edges_found and radius <= max_radius:
                    nearby_edges = net.getNeighboringEdges(x, y, radius)
                    if nearby_edges:
                        edges_found = True
                        # Sort by distance
                        nearby_edges.sort(key=lambda x: x[1])
                        best_match = nearby_edges[0][0].getID()
                        logger.debug(f"Found edge {best_match} at distance {nearby_edges[0][1]:.1f}m for {point['id']}")
                    else:
                        radius *= 2
            except Exception as e:
                logger.warning(f"Error finding edge by coordinates for {point['id']}: {e}")
        
        # Store mapping
        if best_match:
            segment_to_edge[point['id']] = best_match
            points_with_edges += 1
            logger.debug(f"Mapped {point['id']} ({point['road_name']}) to edge {best_match}")
        else:
            logger.warning(f"Could not map {point['id']} ({point['road_name']}) to any edge")
    
    logger.info(f"Successfully mapped {points_with_edges} out of {len(all_points)} points to SUMO edges")
    
    # Now map the HERE segments from the edge mapping data
    if edge_mapping_data and 'segments' in edge_mapping_data:
        segments_mapped = 0
        
        for segment in edge_mapping_data['segments']:
            segment_id = segment.get('id')
            if not segment_id:
                continue
            
            # Try to match by road name first
            road_name = segment.get('road_name', '').lower()
            if road_name and len(road_name) > 3:
                # Find edges with matching names
                matching_edges = []
                
                for edge in all_edges:
                    edge_id = edge.getID()
                    if road_name in edge_id.lower():
                        matching_edges.append(edge_id)
                
                if matching_edges:
                    segment_to_edge[segment_id] = matching_edges
                    segment['sumo_edges'] = matching_edges
                    segments_mapped += 1
                    continue
            
            # If no name match, use coordinates from shape
            shape_points = segment.get('shape', [])
            if shape_points and len(shape_points) >= 2:
                # Try to match start and end points to edges
                start_point = shape_points[0]
                end_point = shape_points[-1]
                
                if len(start_point) >= 2 and len(end_point) >= 2:
                    try:
                        # Convert to SUMO coordinates
                        start_x, start_y = net.convertLonLat2XY(start_point[1], start_point[0])
                        end_x, end_y = net.convertLonLat2XY(end_point[1], end_point[0])
                        
                        # Use a fixed radius for edge matching (100 meters)
                        radius = 100.0
                        
                        # Find edges near start and end
                        start_edges = net.getNeighboringEdges(start_x, start_y, radius)
                        end_edges = net.getNeighboringEdges(end_x, end_y, radius)
                        
                        matched_edges = []
                        
                        # Add start edge if found
                        if start_edges:
                            start_edges.sort(key=lambda x: x[1])
                            matched_edges.append(start_edges[0][0].getID())
                        
                        # Add end edge if found and different from start
                        if end_edges:
                            end_edges.sort(key=lambda x: x[1])
                            end_edge_id = end_edges[0][0].getID()
                            if not matched_edges or end_edge_id != matched_edges[0]:
                                matched_edges.append(end_edge_id)
                        
                        if matched_edges:
                            segment_to_edge[segment_id] = matched_edges
                            segment['sumo_edges'] = matched_edges
                            segments_mapped += 1
                    except Exception as e:
                        logger.warning(f"Error matching segment {segment_id} by coordinates: {e}")
        
        logger.info(f"Successfully mapped {segments_mapped} out of {len(edge_mapping_data['segments'])} segments to SUMO edges")
    
    return segment_to_edge

def verify_entry_exit_mapping(entry_exit_data, net, output_dir):
    """Verify that entry/exit points can be mapped to the SUMO network"""
    logger.info("Verifying entry/exit point mapping...")
    
    result = {
        "valid_entries": 0,
        "valid_exits": 0,
        "total_entries": 0,
        "total_exits": 0,
        "invalid_entries": [],
        "invalid_exits": []
    }
    
    # Verify entry points
    entry_points = entry_exit_data.get('entry_points', [])
    result["total_entries"] = len(entry_points)
    
    for i, entry in enumerate(entry_points):
        if 'location' in entry and len(entry['location']) >= 2:
            lat, lon = entry['location'][0], entry['location'][1]
            
            try:
                # Convert to SUMO coordinates
                x, y = net.convertLonLat2XY(lon, lat)
                
                # Try to find edges nearby
                edges = net.getNeighboringEdges(x, y, 100.0)
                
                if edges:
                    # We found at least one edge
                    result["valid_entries"] += 1
                else:
                    result["invalid_entries"].append({
                        "index": i,
                        "road_name": entry.get('road_name', 'Unknown'),
                        "coordinates": [lat, lon]
                    })
            except Exception as e:
                logger.warning(f"Error checking entry point {i}: {e}")
                result["invalid_entries"].append({
                    "index": i,
                    "road_name": entry.get('road_name', 'Unknown'),
                    "coordinates": [lat, lon],
                    "error": str(e)
                })
    
    # Verify exit points
    exit_points = entry_exit_data.get('exit_points', [])
    result["total_exits"] = len(exit_points)
    
    for i, exit in enumerate(exit_points):
        if 'location' in exit and len(exit['location']) >= 2:
            lat, lon = exit['location'][0], exit['location'][1]
            
            try:
                # Convert to SUMO coordinates
                x, y = net.convertLonLat2XY(lon, lat)
                
                # Try to find edges nearby
                edges = net.getNeighboringEdges(x, y, 100.0)
                
                if edges:
                    # We found at least one edge
                    result["valid_exits"] += 1
                else:
                    result["invalid_exits"].append({
                        "index": i,
                        "road_name": exit.get('road_name', 'Unknown'),
                        "coordinates": [lat, lon]
                    })
            except Exception as e:
                logger.warning(f"Error checking exit point {i}: {e}")
                result["invalid_exits"].append({
                    "index": i,
                    "road_name": exit.get('road_name', 'Unknown'),
                    "coordinates": [lat, lon],
                    "error": str(e)
                })
    
    # Save verification result
    with open(os.path.join(output_dir, "entry_exit_verification.json"), 'w') as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"Entry/exit verification result: {result['valid_entries']} valid entries out of {result['total_entries']}")
    logger.info(f"Entry/exit verification result: {result['valid_exits']} valid exits out of {result['total_exits']}")
    
    return result

def compare_segment_ids(realtime_data, edge_mapping_data, output_dir):
    """Compare segment IDs from realtime data and edge mapping data"""
    logger.info("Comparing segment IDs...")
    
    # Collect segment IDs from realtime data
    realtime_ids = []
    for segment in realtime_data.get('results', []):
        segment_id = segment.get('id')
        if segment_id:
            realtime_ids.append(segment_id)
    
    # Collect segment IDs from edge mapping data
    mapping_ids = []
    for segment in edge_mapping_data.get('segments', []):
        segment_id = segment.get('id')
        if segment_id:
            mapping_ids.append(segment_id)
    
    # Write to file for manual inspection
    debug_file = os.path.join(output_dir, "segment_ids_comparison.txt")
    with open(debug_file, 'w') as f:
        f.write("===== REALTIME SEGMENT IDs =====\n")
        for segment_id in realtime_ids[:100]:  # First 100 for brevity
            f.write(f"{segment_id}\n")
        
        f.write("\n===== MAPPING SEGMENT IDs =====\n")
        for segment_id in mapping_ids[:100]:  # First 100 for brevity
            f.write(f"{segment_id}\n")
    
    logger.info(f"Segment ID comparison written to {debug_file}")
    
    # Check if IDs match exactly
    common_exact = set(realtime_ids).intersection(set(mapping_ids))
    logger.info(f"Exact ID matches: {len(common_exact)} out of {len(realtime_ids)} realtime segments")
    
    return {
        "realtime_count": len(realtime_ids),
        "mapping_count": len(mapping_ids),
        "exact_matches": len(common_exact)
    }

def debug_entry_exit_to_sumo_edges(entry_exit_data, net, output_dir):
    """Debug how entry/exit points map to SUMO edges"""
    logger.info("Debugging entry/exit point mapping to SUMO edges...")
    
    debug_file = os.path.join(output_dir, "entry_exit_edge_mapping.json")
    
    # Store mapping results
    entry_mappings = []
    exit_mappings = []
    
    # Process entry points
    for i, entry in enumerate(entry_exit_data.get('entry_points', [])):
        if 'location' in entry and len(entry['location']) >= 2:
            lat, lon = entry['location'][0], entry['location'][1]
            road_name = entry.get('road_name', 'Unknown')
            
            entry_mapping = {
                "index": i,
                "road_name": road_name,
                "location": [lat, lon],
                "mapped_edges": []
            }
            
            try:
                # Convert to SUMO coordinates
                x, y = net.convertLonLat2XY(lon, lat)
                
                # Find edges at increasing radius
                radius = 50.0
                max_radius = 500.0
                found_edges = False
                
                while not found_edges and radius <= max_radius:
                    edges = net.getNeighboringEdges(x, y, radius)
                    if edges:
                        found_edges = True
                        
                        # Sort by distance
                        edges.sort(key=lambda x: x[1])
                        
                        # Get closest edges with distances
                        closest_edges = []
                        for edge, dist in edges[:5]:  # Take top 5
                            edge_info = {
                                "id": edge.getID(),
                                "distance": dist,
                                "num_lanes": len(edge.getLanes()),
                                "length": edge.getLength(),
                                "speed": edge.getSpeed()
                            }
                            closest_edges.append(edge_info)
                        
                        entry_mapping["radius"] = radius
                        entry_mapping["mapped_edges"] = closest_edges
                    else:
                        radius *= 2
                
                if not found_edges:
                    entry_mapping["error"] = f"No edges found within {max_radius}m"
            except Exception as e:
                entry_mapping["error"] = str(e)
            
            entry_mappings.append(entry_mapping)
    
    # Process exit points
    for i, exit in enumerate(entry_exit_data.get('exit_points', [])):
        if 'location' in exit and len(exit['location']) >= 2:
            lat, lon = exit['location'][0], exit['location'][1]
            road_name = exit.get('road_name', 'Unknown')
            
            exit_mapping = {
                "index": i,
                "road_name": road_name,
                "location": [lat, lon],
                "mapped_edges": []
            }
            
            try:
                # Convert to SUMO coordinates
                x, y = net.convertLonLat2XY(lon, lat)
                
                # Find edges at increasing radius
                radius = 50.0
                max_radius = 500.0
                found_edges = False
                
                while not found_edges and radius <= max_radius:
                    edges = net.getNeighboringEdges(x, y, radius)
                    if edges:
                        found_edges = True
                        
                        # Sort by distance
                        edges.sort(key=lambda x: x[1])
                        
                        # Get closest edges with distances
                        closest_edges = []
                        for edge, dist in edges[:5]:  # Take top 5
                            edge_info = {
                                "id": edge.getID(),
                                "distance": dist,
                                "num_lanes": len(edge.getLanes()),
                                "length": edge.getLength(),
                                "speed": edge.getSpeed()
                            }
                            closest_edges.append(edge_info)
                        
                        exit_mapping["radius"] = radius
                        exit_mapping["mapped_edges"] = closest_edges
                    else:
                        radius *= 2
                
                if not found_edges:
                    exit_mapping["error"] = f"No edges found within {max_radius}m"
            except Exception as e:
                exit_mapping["error"] = str(e)
            
            exit_mappings.append(exit_mapping)
    
    # Calculate success rates
    entries_with_edges = sum(1 for e in entry_mappings if e.get("mapped_edges"))
    exits_with_edges = sum(1 for e in exit_mappings if e.get("mapped_edges"))
    
    summary = {
        "total_entries": len(entry_mappings),
        "total_exits": len(exit_mappings),
        "entries_with_edges": entries_with_edges,
        "exits_with_edges": exits_with_edges,
        "entry_success_rate": entries_with_edges / len(entry_mappings) if entry_mappings else 0,
        "exit_success_rate": exits_with_edges / len(exit_mappings) if exit_mappings else 0,
        "entry_mappings": entry_mappings,
        "exit_mappings": exit_mappings,
        "entry_mappings_sample": entry_mappings[:10],  # First 10 for readability
        "exit_mappings_sample": exit_mappings[:10]     # First 10 for readability
    }
    
    # Save to file
    with open(debug_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Entry/exit edge mapping saved to {debug_file}")
    logger.info(f"Entry points: {entries_with_edges}/{len(entry_mappings)} mapped to edges ({summary['entry_success_rate']*100:.1f}%)")
    logger.info(f"Exit points: {exits_with_edges}/{len(exit_mappings)} mapped to edges ({summary['exit_success_rate']*100:.1f}%)")
    
    # Try to create a simple visualization
    try:
        import matplotlib.pyplot as plt
        
        # Create a figure
        plt.figure(figsize=(15, 10))
        
        # Draw network edges lightly in the background
        edge_xs = []
        edge_ys = []
        
        for edge in net.getEdges():
            shape = edge.getShape()
            if shape:
                xs = [p[0] for p in shape]
                ys = [p[1] for p in shape]
                plt.plot(xs, ys, 'gray', linewidth=0.5, alpha=0.2)
        
        # Plot entry points
        for entry in entry_mappings:
            if 'error' not in entry:
                lat, lon = entry['location']
                x, y = net.convertLonLat2XY(lon, lat)
                
                plt.plot(x, y, 'go', markersize=10)
                
                # Plot lines to mapped edges
                for edge_info in entry.get('mapped_edges', [])[:1]:  # Just the closest edge
                    edge_id = edge_info['id']
                    edge = net.getEdge(edge_id)
                    if edge:
                        shape = edge.getShape()
                        if shape:
                            closest_point = min(shape, key=lambda p: math.sqrt((p[0]-x)**2 + (p[1]-y)**2))
                            plt.plot([x, closest_point[0]], [y, closest_point[1]], 'g-', linewidth=1.5, alpha=0.7)
        
        # Plot exit points
        for exit in exit_mappings:
            if 'error' not in exit:
                lat, lon = exit['location']
                x, y = net.convertLonLat2XY(lon, lat)
                
                plt.plot(x, y, 'ro', markersize=10)
                
                # Plot lines to mapped edges
                for edge_info in exit.get('mapped_edges', [])[:1]:  # Just the closest edge
                    edge_id = edge_info['id']
                    edge = net.getEdge(edge_id)
                    if edge:
                        shape = edge.getShape()
                        if shape:
                            closest_point = min(shape, key=lambda p: math.sqrt((p[0]-x)**2 + (p[1]-y)**2))
                            plt.plot([x, closest_point[0]], [y, closest_point[1]], 'r-', linewidth=1.5, alpha=0.7)
        
        # Add legend
        plt.plot([], [], 'go', markersize=10, label='Entry Points')
        plt.plot([], [], 'ro', markersize=10, label='Exit Points')
        plt.plot([], [], 'g-', linewidth=1.5, label='Entry to Edge')
        plt.plot([], [], 'r-', linewidth=1.5, label='Exit to Edge')
        plt.legend()
        
        # Save the visualization
        plt.savefig(os.path.join(output_dir, "entry_exit_edge_mapping.png"), dpi=300)
        plt.close()
        
        logger.info(f"Entry/exit visualization saved to {os.path.join(output_dir, 'entry_exit_edge_mapping.png')}")
    except Exception as e:
        logger.warning(f"Could not create visualization: {e}")
    
    return summary

def debug_incident_processing(incidents_data, edge_mapping_data, net, output_dir):
    """Debug why incidents aren't being applied to edges"""
    logger.info("Debugging incident processing...")
    
    debug_file = os.path.join(output_dir, "incident_debug.json")
    
    # Create segment ID to edges mapping
    segment_to_edges = {}
    for segment in edge_mapping_data.get('segments', []):
        segment_id = segment.get('id')
        sumo_edges = segment.get('sumo_edges', [])
        if segment_id and sumo_edges:
            segment_to_edges[segment_id] = sumo_edges
    
    # Process each incident
    incident_details = []
    total_incidents = 0
    incidents_with_links = 0
    incidents_with_valid_links = 0
    incidents_with_mapped_edges = 0
    
    if incidents_data and 'incidents' in incidents_data:
        total_incidents = len(incidents_data['incidents'])
        
        for i, incident in enumerate(incidents_data['incidents']):
            incident_detail = {
                "index": i,
                "type": incident.get('type', 'UNKNOWN'),
                "criticality": incident.get('criticality', 0),
                "description": incident.get('description', ''),
                "has_location": 'location' in incident,
                "has_shape": False,
                "has_links": False,
                "link_ids": [],
                "affected_edges": [],
                "location_type": "none"
            }
            
            # Check location structure
            location = incident.get('location', {})
            if location:
                # Check if it has shape with links
                shape = location.get('shape', {})
                if shape:
                    incident_detail["has_shape"] = True
                    
                    # Check if shape has links
                    links = shape.get('links', [])
                    if links:
                        incident_detail["has_links"] = True
                        incidents_with_links += 1
                        
                        # Get link IDs
                        link_ids = []
                        mapped_edges = set()
                        
                        for link in links:
                            link_id = link.get('linkId')
                            if link_id:
                                link_ids.append(link_id)
                                
                                # Check if link ID exists in mapping
                                if link_id in segment_to_edges:
                                    incidents_with_valid_links += 1
                                    
                                    # Add mapped edges
                                    for edge_id in segment_to_edges[link_id]:
                                        mapped_edges.add(edge_id)
                        
                        incident_detail["link_ids"] = link_ids
                        incident_detail["affected_edges"] = list(mapped_edges)
                        
                        if mapped_edges:
                            incidents_with_mapped_edges += 1
                        
                        incident_detail["location_type"] = "links"
                
                # If no links found, check for point location
                elif 'point' in location:
                    incident_detail["location_type"] = "point"
                    point = location.get('point', {})
                    
                    if 'latitude' in point and 'longitude' in point:
                        lat = point.get('latitude')
                        lon = point.get('longitude')
                        
                        try:
                            # Convert to SUMO coordinates
                            x, y = net.convertLonLat2XY(lon, lat)
                            
                            # Find edges near this point
                            nearby_edges = net.getNeighboringEdges(x, y, 100.0)
                            if nearby_edges:
                                # Sort by distance
                                nearby_edges.sort(key=lambda x: x[1])
                                
                                # Get closest edges
                                edges = [edge.getID() for edge, dist in nearby_edges[:3]]
                                incident_detail["affected_edges"] = edges
                                
                                if edges:
                                    incidents_with_mapped_edges += 1
                        except Exception as e:
                            incident_detail["error"] = str(e)
            
            incident_details.append(incident_detail)
    
    # Create summary statistics
    summary = {
        "total_incidents": total_incidents,
        "incidents_with_links": incidents_with_links,
        "incidents_with_valid_links": incidents_with_valid_links,
        "incidents_with_mapped_edges": incidents_with_mapped_edges,
        "incident_details": incident_details[:20]  # First 20 for readability
    }
    
    # Save to file
    with open(debug_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Incident debug info saved to {debug_file}")
    logger.info(f"Incidents summary: {total_incidents} total, {incidents_with_links} with links, {incidents_with_mapped_edges} with mapped edges")
    
    return summary
    
def count_traffic_lights(net_file):
    """Count traffic light intersections in the SUMO network"""
    try:
        # Load the network
        net = sumolib.net.readNet(net_file)
        
        # Count nodes with traffic lights
        tl_nodes = 0
        total_nodes = 0
        
        for node in net.getNodes():
            total_nodes += 1
            if node.getType() == "traffic_light":
                tl_nodes += 1
        
        logger.info(f"Traffic light intersections: {tl_nodes} out of {total_nodes} total intersections")
        """ logs show "Traffic light intersections: 790 out of 1471 total intersections" means that out of all the junctions in your SUMO network, about 54% have traffic signals."""
        # Get traffic light programs
        tls = net.getTrafficLights()
        logger.info(f"Traffic light programs: {len(tls)}")
        
        # Sample some traffic light programs
        if tls:
            sample_tl = tls[0]
            programs = sample_tl.getPrograms()
            logger.info(f"Sample traffic light {sample_tl.getID()} has {len(programs)} programs")
            
        return tl_nodes
    except Exception as e:
        logger.error(f"Error counting traffic lights: {e}")
        return 0

def diagnose_routing_issues(net_file, entry_exit_data, output_dir):
    """Diagnose why duarouter is creating such short routes"""
    logger.info("Diagnosing routing issues...")
    
    try:
        net = sumolib.net.readNet(net_file)
        if global_entry_exit:
            routing_diagnosis = diagnose_routing_issues(net_file, global_entry_exit, args.output_dir)
        
        # Test route finding between entry/exit pairs
        entries = entry_exit_data.get('entry_points', [])[:10]
        exits = entry_exit_data.get('exit_points', [])[:5]
        
        routing_diagnosis = {
            'total_tests': 0,
            'successful_routes': 0,
            'failed_routes': 0,
            'short_routes': 0,  # < 500m
            'medium_routes': 0, # 500m - 2km
            'long_routes': 0,   # > 2km
            'route_details': [],
            'network_stats': {
                'total_edges': len(net.getEdges()),
                'total_nodes': len(net.getNodes()),
                'disconnected_components': 0
            }
        }
        
        for i, entry in enumerate(entries):
            for j, exit in enumerate(exits):
                if routing_diagnosis['total_tests'] >= 20:  # Limit tests
                    break
                    
                entry_lat, entry_lon = entry['location']
                exit_lat, exit_lon = exit['location']
                
                try:
                    # Convert to SUMO coordinates
                    entry_x, entry_y = net.convertLonLat2XY(entry_lon, entry_lat)
                    exit_x, exit_y = net.convertLonLat2XY(exit_lon, exit_lat)
                    
                    # Find nearest edges with larger search radius
                    entry_edges = net.getNeighboringEdges(entry_x, entry_y, 200.0)
                    exit_edges = net.getNeighboringEdges(exit_x, exit_y, 200.0)
                    
                    if not entry_edges:
                        logger.warning(f"No entry edges found for entry point {i} within 200m")
                        continue
                    if not exit_edges:
                        logger.warning(f"No exit edges found for exit point {j} within 200m")
                        continue
                    
                    entry_edge = entry_edges[0][0]
                    exit_edge = exit_edges[0][0]
                    
                    # Test multiple routing methods
                    route_found = False
                    route_length = 0
                    route_edges = []
                    
                    # Method 1: SUMO's built-in shortest path
                    try:
                        path_result = net.getShortestPath(entry_edge, exit_edge)
                        if path_result[0]:  # If route found
                            route_edges = path_result[0]
                            route_length = sum(edge.getLength() for edge in route_edges)
                            route_found = True
                            logger.info(f"Route {i}->{j}: {len(route_edges)} edges, {route_length:.1f}m")
                    except Exception as e:
                        logger.debug(f"Shortest path failed for {i}->{j}: {e}")
                    
                    # Method 2: If no route found, check connectivity
                    if not route_found:
                        # Check if edges are in same connected component
                        try:
                            # Simple connectivity test
                            entry_node = entry_edge.getFromNode()
                            exit_node = exit_edge.getToNode()
                            
                            # Try to find any path (even if inefficient)
                            all_routes = net.getShortestPath(entry_edge, exit_edge, maxCost=10000)
                            if all_routes[0]:
                                route_edges = all_routes[0]
                                route_length = sum(edge.getLength() for edge in route_edges)
                                route_found = True
                                logger.info(f"Long route {i}->{j}: {len(route_edges)} edges, {route_length:.1f}m")
                        except:
                            pass
                    
                    # Record results
                    route_detail = {
                        'entry_point': i,
                        'exit_point': j,
                        'entry_edge': entry_edge.getID(),
                        'exit_edge': exit_edge.getID(),
                        'route_found': route_found,
                        'route_length': route_length,
                        'num_edges': len(route_edges) if route_edges else 0,
                        'direct_distance': ((entry_x - exit_x)**2 + (entry_y - exit_y)**2)**0.5
                    }
                    
                    routing_diagnosis['route_details'].append(route_detail)
                    routing_diagnosis['total_tests'] += 1
                    
                    if route_found:
                        routing_diagnosis['successful_routes'] += 1
                        if route_length < 500:
                            routing_diagnosis['short_routes'] += 1
                        elif route_length < 2000:
                            routing_diagnosis['medium_routes'] += 1
                        else:
                            routing_diagnosis['long_routes'] += 1
                    else:
                        routing_diagnosis['failed_routes'] += 1
                        
                except Exception as e:
                    logger.warning(f"Route test {i}->{j} failed: {e}")
                    routing_diagnosis['failed_routes'] += 1
                    routing_diagnosis['total_tests'] += 1
        
        # Save diagnosis
        with open(os.path.join(output_dir, 'routing_diagnosis.json'), 'w') as f:
            json.dump(routing_diagnosis, f, indent=2)
        
        # Log summary
        success_rate = routing_diagnosis['successful_routes'] / max(1, routing_diagnosis['total_tests'])
        avg_length = sum(r['route_length'] for r in routing_diagnosis['route_details'] if r['route_found']) / max(1, routing_diagnosis['successful_routes'])
        
        logger.info(f"Routing diagnosis complete:")
        logger.info(f"  Success rate: {success_rate*100:.1f}%")
        logger.info(f"  Average route length: {avg_length:.1f}m")
        logger.info(f"  Short routes (<500m): {routing_diagnosis['short_routes']}")
        logger.info(f"  Medium routes (500m-2km): {routing_diagnosis['medium_routes']}")
        logger.info(f"  Long routes (>2km): {routing_diagnosis['long_routes']}")
        
        return routing_diagnosis
        
    except Exception as e:
        logger.error(f"Routing diagnosis failed: {e}")
        return None

def main():
    """Main execution function"""
    # Parse command line arguments
    args = parse_arguments()
    segment_to_edges = {}  # Initialize at the start
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("Starting Baseline SUMO Simulation using HERE API data")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Construct the full path to the OSM file
    if os.path.isabs(args.osm_file):
        osm_file = args.osm_file
    else:
        osm_file = os.path.join(args.data_dir, args.osm_file)
    
    logger.info(f"OSM file path: {osm_file}")
   
    # Check if OSM file exists, if not download it
    if not os.path.exists(osm_file) or args.force_regenerate:
        logger.info(f"OSM file {osm_file} not found or force regenerate specified")
        # Use the coordinates to download the map
        osm_file = download_restricted_osm_map(
            args.min_lat, args.min_lon, args.max_lat, args.max_lon, 
            args.data_dir, os.path.basename(args.osm_file)
        )
        if not osm_file:
            logger.error("Failed to download OSM map. Exiting.")
            return False
    else:
        logger.info(f"Using existing OSM file: {osm_file}")

    # Step 1: Create SUMO network from OSM file
    net_file = os.path.join(args.output_dir, "baseline.net.xml")
    if args.force_regenerate or not os.path.exists(net_file):
        net_file = create_sumo_network(args.osm_file, args.output_dir)
        if not net_file:
            logger.error("Failed to create SUMO network. Exiting.")
            return False
    else:
        logger.info(f"Using existing SUMO network: {net_file}")
    
    # Load the SUMO network
    net = sumolib.net.readNet(net_file)
    count_traffic_lights(net_file)

    # Load the SUMO network
    net = sumolib.net.readNet(net_file)
    count_traffic_lights(net_file)

    
    # Check if network has geo-referencing information
    logger.info("Checking SUMO network geo-referencing:")
    try:
        # Test using convertLonLat2XY directly
        test_lon, test_lat = -74.0, 40.7  # Example point near NYC
        test_x, test_y = net.convertLonLat2XY(test_lon, test_lat)
        logger.info(f"Test coordinate conversion: ({test_lon}, {test_lat}) -> ({test_x}, {test_y})")
        
        # Test reverse conversion
        back_lon, back_lat = net.convertXY2LonLat(test_x, test_y)
        logger.info(f"Reverse conversion: ({test_x}, {test_y}) -> ({back_lon}, {back_lat})")
        
        # Check geo-reference by testing bounds
        bbox_min_x, bbox_min_y = net.convertLonLat2XY(args.min_lon, args.min_lat)
        bbox_max_x, bbox_max_y = net.convertLonLat2XY(args.max_lon, args.max_lat)
        logger.info(f"Bounding box in SUMO coordinates: ({bbox_min_x}, {bbox_min_y}) to ({bbox_max_x}, {bbox_max_y})")
        
        # Test finding edges near a point
        test_edges = net.getNeighboringEdges(test_x, test_y, 1000.0)  # large radius
        if test_edges:
            logger.info(f"Found {len(test_edges)} edges near test point")
            logger.info(f"Closest edge: {test_edges[0][0].getID()}, distance: {test_edges[0][1]}")
        else:
            logger.warning("No edges found near test point. Try another point or larger radius")
        
        logger.info("Network geo-referencing test successful")
    except Exception as e:
        logger.error(f"Error testing geo-referencing: {e}")
        logger.warning("Network may not have proper geo-referencing, which could cause mapping failures")
    
    # Load edge mapping and entry/exit files once
    # NEW CODE - Prefer fixed file if it exists
    edge_mapping_file = find_latest_file(args.data_dir, "fixed_edge_mapping_*.json")
    if not edge_mapping_file:
        edge_mapping_file = find_latest_file(args.data_dir, "sumo_edge_mapping_*.json")
        logger.warning("Fixed edge mapping not found, using original file")
    else:
        logger.info("Using fixed edge mapping file")
    entry_exit_file = find_latest_file(args.data_dir, "sumo_entry_exit_points_*.json")
    
    global_edge_mapping = None
    global_entry_exit = None
    
    if edge_mapping_file:
        logger.info(f"Loading global edge mapping from: {edge_mapping_file}")
        global_edge_mapping = load_json_file(edge_mapping_file)
        
        # Map HERE segments to SUMO edges if not already mapped
        if global_edge_mapping and 'segments' in global_edge_mapping:
            logger.info(f"Edge mapping file contains {len(global_edge_mapping['segments'])} segments")
            
            # Check how many segments already have SUMO edges
            segments_with_edges = sum(1 for s in global_edge_mapping['segments'] if 'sumo_edges' in s and s['sumo_edges'])
            logger.info(f"Segments with pre-existing SUMO edges: {segments_with_edges}")
            
            if segments_with_edges < len(global_edge_mapping['segments']):
                logger.info("Mapping remaining HERE segments to SUMO edges...")
                segment_to_edges = map_here_segments_to_sumo_edges(global_edge_mapping, net)
                
                # Check again after mapping
                segments_with_edges = sum(1 for s in global_edge_mapping['segments'] if 'sumo_edges' in s and s['sumo_edges'])
                logger.info(f"Segments with SUMO edges after mapping: {segments_with_edges}")
            
            # Save the updated mapping
            updated_mapping_file = os.path.join(args.output_dir, "updated_edge_mapping.json")
            with open(updated_mapping_file, 'w') as f:
                json.dump(global_edge_mapping, f, indent=2)
            logger.info(f"Saved updated edge mapping to {updated_mapping_file}")
    else:
        logger.warning("No edge mapping file found")
    
    if entry_exit_file:
        logger.info(f"Loading global entry/exit points from: {entry_exit_file}")
        global_entry_exit = load_json_file(entry_exit_file)
        
        if global_entry_exit:
            logger.info(f"Entry/exit file contains {len(global_entry_exit.get('entry_points', []))} entry points and "
                        f"{len(global_entry_exit.get('exit_points', []))} exit points")
            
            # Debug entry/exit point mapping to SUMO edges
            entry_exit_edge_debug = debug_entry_exit_to_sumo_edges(
                global_entry_exit,
                net,
                args.output_dir
            )

            logger.info("Running routing diagnosis to identify short route issues...")
            routing_diagnosis = diagnose_routing_issues(net_file, global_entry_exit, args.output_dir)
            if routing_diagnosis:
                success_rate = routing_diagnosis['successful_routes'] / max(1, routing_diagnosis['total_tests'])
                avg_length = sum(r['route_length'] for r in routing_diagnosis['route_details'] if r['route_found']) / max(1, routing_diagnosis['successful_routes'])
                
                if success_rate < 0.7:
                    logger.warning(f"Low routing success rate ({success_rate*100:.1f}%) - network connectivity issues likely")
                if avg_length < 800:
                    logger.warning(f"Short average route length ({avg_length:.1f}m) - entry/exit points may be too clustered")
    else:
        logger.warning("No entry/exit points file found")
    
    # Create direct mapping between HERE API and SUMO
    if global_edge_mapping and global_entry_exit:
        logger.info("Creating direct mapping between HERE API data and SUMO edges")
        direct_mapping = create_direct_here_to_sumo_mapping(global_edge_mapping, global_entry_exit, net)
        
        # Update the edge mapping with direct mappings
        if direct_mapping:
            # For segments
            if 'segments' in global_edge_mapping:
                for segment in global_edge_mapping['segments']:
                    segment_id = segment.get('id')
                    if segment_id and segment_id in direct_mapping:
                        segment['sumo_edges'] = direct_mapping[segment_id]
            
            # Save updated mapping
            updated_mapping_file = os.path.join(args.output_dir, "direct_edge_mapping.json")
            with open(updated_mapping_file, 'w') as f:
                json.dump(global_edge_mapping, f, indent=2)
            logger.info(f"Saved updated edge mapping with direct mappings to {updated_mapping_file}")

    # Step 2: Load realtime traffic data files
    realtime_files = get_realtime_files(args.data_dir)
    if not realtime_files:
        logger.error("No realtime traffic data files found. Exiting.")
        return False
    
    logger.info(f"Found {len(realtime_files)} realtime traffic data files")
    
    # Also get incidents files for a double-check
    incidents_files = get_incidents_files(args.data_dir)
    logger.info(f"Found {len(incidents_files)} incident data files")
    
    # Step 3: Process each timestep - for speed and incident information
    processed_data = []
    successful_timesteps = 0
    failed_timesteps = 0
    
    # Track timestamps for validation
    processed_timestamps = []
    
    for realtime_file in realtime_files:
        # Extract timestamp for logging
        file_timestamp = extract_timestamp_from_filename(realtime_file)
        logger.info(f"Processing timestep {file_timestamp} from {os.path.basename(realtime_file)}")
        
        # Load data for this timestep, using global mapping
        timestep_data = load_data_for_timestep(args.data_dir, realtime_file, global_edge_mapping, global_entry_exit)
        if not timestep_data:
            logger.warning(f"Could not load complete data for {realtime_file}, skipping")
            failed_timesteps += 1
            continue
        
        # Debug the first timestep in detail
        if successful_timesteps == 0:
            # Check segment IDs
            if 'realtime' in timestep_data and 'edge_mapping' in timestep_data:
                logger.info("Debugging first timestep in detail")
                
                # Compare segment IDs
                compare_result = compare_segment_ids(
                    timestep_data['realtime'], 
                    timestep_data['edge_mapping'], 
                    args.output_dir
                )
                
                # Debug segments and realtime
                debug_report = debug_segments_and_realtime(
                    timestep_data['edge_mapping'],
                    timestep_data['realtime'],
                    args.output_dir
                )
                
                # Debug incident processing
                if 'incidents' in timestep_data:
                    incident_debug = debug_incident_processing(
                        timestep_data['incidents'], 
                        timestep_data['edge_mapping'], 
                        net, 
                        args.output_dir
                    )
        
        # Process the timestep data
        processed_timestep = process_timestep_data(timestep_data, net, segment_to_edges)
        if processed_timestep:
            processed_data.append(processed_timestep)
            processed_timestamps.append(file_timestamp)
            successful_timesteps += 1
            
            # Check for empty edge speeds in first timestep
            if successful_timesteps == 1 and not processed_timestep.get('edge_speeds'):
                logger.warning("First timestep has no edge speeds - there may be a mapping issue")
        else:
            logger.warning(f"Failed to process timestep data for {realtime_file}")
            failed_timesteps += 1
    
    if not processed_data:
        logger.error("No valid processed timestep data. Exiting.")
        return False
    
    logger.info(f"Successfully processed {successful_timesteps} timesteps of traffic data")
    logger.info(f"Failed to process {failed_timesteps} timesteps")
    
    # Validate we have a good spread of timestamps
    if processed_timestamps:
        min_time = min(processed_timestamps)
        max_time = max(processed_timestamps)
        duration = (max_time - min_time).total_seconds() / 60  # minutes
        logger.info(f"Time span of processed data: {duration:.1f} minutes ({min_time} to {max_time})")
    
    # Step 4 & 5: Use the already loaded entry/exit data to create trips file for duarouter
    if not global_entry_exit:
        logger.error("No entry/exit points data available. Cannot create duarouter trips. Exiting.")
        return False
    
    # Create trips file for duarouter
    trips_file = os.path.join(args.output_dir, "geo_trips.xml")
    trips_file, simulation_period = create_duarouter_trips(global_entry_exit, trips_file, processed_data)
    
    # Step 6: Run duarouter to create routes
    routes_file = os.path.join(args.output_dir, "duarouter_routes.xml")
    if args.force_regenerate or not os.path.exists(routes_file):
        routes_file = run_duarouter(net_file, trips_file, routes_file)
        if not routes_file:
            logger.error("Failed to run duarouter. Exiting.")
            return False
    else:
        logger.info(f"Using existing duarouter routes file: {routes_file}")
    
    # Step 7: Create GUI settings
    create_gui_settings(args.output_dir)
    
    # Step 8: Create time-varying traffic demand
    demand_files = create_time_varying_demand(
        processed_data, 
        args.output_dir, 
        args.simulation_period,
        duarouter_routes_file=routes_file
    )
    if not demand_files:
        logger.error("Failed to create traffic demand. Exiting.")
        return False
    
    # Step 9: Create SUMO configuration
    # config_file = create_sumo_config(
    #     args.net_file,
    #     demand_files,
    #     args.output_dir,
    #     args.simulation_period,
    #     visualize=args.visualize
    # )
    config_file = create_sumo_config(
    net_file,  # Changed from args.net_file
    demand_files['flows_file'],  # Extract flows_file from dictionary
    demand_files['additional_files'],  # Extract additional_files
    args.output_dir,
    args.simulation_period
)
   
    if not config_file:
        logger.error("Failed to create SUMO configuration. Exiting.")
        return False
    
    # Step 10: Run simulation (unless skipped)
    if args.use_traci:
        # Run with TraCI control
        route_files = f"{demand_files['flows_file']}"
        run_traci_controlled_simulation(
            net_file,
            route_files,
            processed_data,
            args.output_dir,
            args.simulation_period,
            visualize=args.visualize)
    else:
        logger.error("Failed to generate TraCI Simualtion, using fallback traditional SUMO")
        # Run traditional SUMO (fallback)
        sumo_binary = 'sumo-gui' if args.visualize else 'sumo'
        sumo_cmd = [sumolib.checkBinary(sumo_binary), '-c', config_file]
        logger.info(f"Running SUMO: {' '.join(sumo_cmd)}")
        subprocess.run(sumo_cmd, check=True)
    
    # Step 11: Analyze results
    results = analyze_results(args.output_dir, processed_data)

    # Create comparison CSV
    comparison_csv = create_comparison_csv(args.output_dir)
    if comparison_csv:
        logger.info(f"Created comparison CSV at {comparison_csv}")
    
    # Step 12: Create visualizations
    create_visualizations(results, args.output_dir)
    
    logger.info("Baseline TraCI SUMO Simulation completed.")
    logger.info(f"All output files are in: {args.output_dir}")
    
    # Print key statistics
    if 'trip_stats' in results:
        logger.info("Trip Statistics:")
        logger.info(f"  Total Trips: {results['trip_stats'].get('total_trips', 0)}")
        logger.info(f"  Avg Duration: {results['trip_stats'].get('avg_duration', 0):.2f} s")
        logger.info(f"  Avg Wait Time: {results['trip_stats'].get('avg_wait_time', 0):.2f} s")
        logger.info(f"  Avg Speed: {results['trip_stats'].get('avg_speed', 0):.2f} m/s")
    
    if 'queue_stats' in results and 'global_stats' in results['queue_stats']:
        logger.info("Queue Statistics:")
        logger.info(f"  Max Queue Length: {results['queue_stats']['global_stats'].get('max_queue_any_lane', 0):.2f} m")
        logger.info(f"  Avg Queue Length: {results['queue_stats']['global_stats'].get('avg_queue_all_lanes', 0):.2f} m")
    
    return True
if __name__ == "__main__":
    main()#!/usr/bin/env python3

