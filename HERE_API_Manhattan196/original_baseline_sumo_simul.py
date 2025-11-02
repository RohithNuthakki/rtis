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
    import traci
except ImportError:
    print("Error importing SUMO libraries. Check if SUMO_HOME is correctly set.")
    sys.exit(1)

# Configure logging
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(DEFAULT_OUTPUT_DIR, "simulation.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("baseline_sumo_simul")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Baseline SUMO Simulation using HERE API data')
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
      // Roads and highways
      way["highway"]["highway"!="footway"]["highway"!="cycleway"]["highway"!="path"]["highway"!="bridleway"]["highway"!="steps"]({min_lat},{min_lon},{max_lat},{max_lon});
      
      // Traffic signals and signs
      node["highway"="traffic_signals"]({min_lat},{min_lon},{max_lat},{max_lon});
      node["highway"="stop"]({min_lat},{min_lon},{max_lat},{max_lon});
      node["highway"="give_way"]({min_lat},{min_lon},{max_lat},{max_lon});
      
      // Turn restrictions
      relation["type"="restriction"]({min_lat},{min_lon},{max_lat},{max_lon});
      
      // Add junction information
      node(w)({min_lat},{min_lon},{max_lat},{max_lon});
    );
    
    // Output
    out body;
    """
    
    try:
        logger.info("Sending request to Overpass API...")
        response = requests.post(overpass_url, data={"data": overpass_query})
        
        if response.status_code == 200:
            # Check if response has actual content
            if len(response.content) < 100:
                logger.error("Downloaded OSM data is too small, likely empty")
                return None
                
            with open(output_file, "wb") as f:
                f.write(response.content)
            logger.info(f"OSM map downloaded successfully to {output_file} ({len(response.content)} bytes)")
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
        "--no-internal-links", "false",
        "--no-turnarounds", "true",
        "--ramps.guess", "true",
        # Remove pedestrian/bicycle options 
        # "--walkingareas", "true",
        # "--crossings.guess", "true",  
        # "--sidewalks.guess", "true",
        "--verbose", "true"
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
    except Exception as e:
        logger.error(f"Unexpected error creating SUMO network: {e}")
        return None

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

def load_data_for_timestep(data_dir, realtime_file):
    """Load all necessary data for a single timestep"""
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
    
    # Find the corresponding edge mapping file
    edge_mapping_file = find_matching_edge_mapping_file(data_dir, timestamp)
    if not edge_mapping_file:
        logger.warning(f"No matching edge mapping file found for {realtime_file}")
    
    # Debug the edge mapping structure
    if edge_mapping_file and os.path.exists(edge_mapping_file):
        logger.info(f"Debug: Edge mapping file structure from {os.path.basename(edge_mapping_file)}")
        with open(edge_mapping_file, 'r') as f:
            mapping_data = json.load(f)
            logger.info(f"Debug: Edge mapping file root keys: {list(mapping_data.keys())}")
            if 'segments' in mapping_data:
                logger.info(f"Debug: Edge mapping contains {len(mapping_data['segments'])} segment mappings")
                # Check if segments is a list or dict and handle accordingly
                if isinstance(mapping_data['segments'], list):
                    logger.info(f"Debug: segments is a list with {len(mapping_data['segments'])} items")
                    if len(mapping_data['segments']) > 0:
                        logger.info(f"Debug: First segment structure: {mapping_data['segments'][0].keys() if isinstance(mapping_data['segments'][0], dict) else type(mapping_data['segments'][0])}")
                elif isinstance(mapping_data['segments'], dict):
                    sample_keys = list(mapping_data['segments'].keys())[:5]
                    logger.info(f"Debug: Sample segment keys: {sample_keys}")
                else:
                    logger.info(f"Debug: segments has unexpected type: {type(mapping_data['segments'])}")

    # Find the corresponding entry/exit points file
    entry_exit_file = find_matching_entry_exit_file(data_dir, timestamp)
    if not entry_exit_file:
        logger.warning(f"No matching entry/exit points file found for {realtime_file}")
    
    # Load all files
    data = {
        'timestamp': timestamp,
        'realtime': load_json_file(realtime_file),
        'incidents': load_json_file(incidents_file) if incidents_file else None,
        'analysis': load_json_file(analysis_file) if analysis_file else None,
        'edge_mapping': load_json_file(edge_mapping_file) if edge_mapping_file else None,
        'entry_exit': load_json_file(entry_exit_file) if entry_exit_file else None
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
    """Run duarouter to convert geo-coordinates to edges and generate routes"""
    try:
        # Ensure duarouter is available
        duarouter = checkBinary("duarouter")
        
        # Build the command
        cmd = [
            duarouter,
            "--net-file", net_file,
            "--route-files", trips_file,
            "--output-file", output_file,
            "--ignore-errors", "true",
            "--mapmatch.distance", "500",  # Allow matching up to 500m
            "--verbose", "true"
        ]
        
        # Run duarouter
        logger.info(f"Running duarouter: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"duarouter completed successfully, routes saved to {output_file}")
            return output_file
        else:
            logger.error(f"duarouter failed with code {result.returncode}")
            logger.error(f"Error: {result.stderr}")
            return None
    except Exception as e:
        logger.error(f"Error running duarouter: {e}")
        return None

def process_timestep_data(data, net):
    """Process data for a single timestep to extract speed information"""
    if not data or not data['realtime']:
        logger.error("Incomplete data for timestep")
        return None
    
    timestamp = data['timestamp']
    realtime_data = data['realtime']
    incidents_data = data['incidents']
    analysis_data = data.get('analysis', {})
    edge_mapping = data.get('edge_mapping', {})
    
    logger.info(f"Processing data for timestep: {timestamp}")
    
    # Debug output for segment IDs
    if len(realtime_data.get('results', [])) > 0:
        # Display first few segment structures
        for i in range(min(5, len(realtime_data.get('results', [])))):
            segment = realtime_data.get('results', [])[i]
            logger.info(f"Debug: Realtime segment {i} structure: {segment.keys()}")
            if 'id' in segment:
                logger.info(f"Debug: Segment {i} has ID: {segment['id']}")
            else:
                logger.info(f"Debug: Segment {i} has no explicit ID")
        
        # Check edge mapping keys
        first_few_keys = list(edge_mapping.keys())[:5] if edge_mapping else []
        logger.info(f"Debug: First few edge mapping keys: {first_few_keys}")
        
    # Extract estimated vehicle count from analysis
    total_vehicles = analysis_data.get('total_vehicles', 0)
    if total_vehicles <= 0:
        logger.warning("No vehicle count data available, using default")
        total_vehicles = 5000  # Default fallback
    else:
        logger.info(f"Using estimated vehicles from analysis: {total_vehicles}")
    
    # Map traffic data to SUMO edges for speed adjustments
    edge_speeds = {}
    edge_jam_factors = {}
    
    # Check if edge_mapping has a 'segments' key
    segments_mapping = edge_mapping.get('segments', {})
    if not segments_mapping and isinstance(edge_mapping, dict):
        # If no 'segments' key but edge_mapping is a dict, use it directly
        segments_mapping = edge_mapping
    elif isinstance(segments_mapping, list):
        # Convert list to dict if segments is a list
        # Assuming each item in the list has an identifier we can use as a key
        converted_mapping = {}
        segments_with_edges = 0
        total_edges = 0
        
        for idx, segment in enumerate(segments_mapping):
            if isinstance(segment, dict):
                # Try to find an ID field in the segment
                segment_id = segment.get('id') or segment.get('segment_id') or f"segment_{idx}"
                converted_mapping[segment_id] = segment
                
                # Check if this segment has non-empty sumo_edges
                sumo_edges = segment.get('sumo_edges', [])
                if sumo_edges and len(sumo_edges) > 0:
                    segments_with_edges += 1
                    total_edges += len(sumo_edges)
        
        segments_mapping = converted_mapping
        
        # Log statistics about edge mappings
        logger.info(f"Edge mapping statistics: {segments_with_edges}/{len(segments_mapping)} segments have SUMO edges")
        logger.info(f"Total SUMO edges in mapping: {total_edges}")
        
        # Sample some segments with edges for debugging
        segments_with_edges_sample = []
        segments_without_edges_sample = []
        
        for seg_id, seg_data in segments_mapping.items():
            if seg_data.get('sumo_edges') and len(seg_data.get('sumo_edges', [])) > 0:
                segments_with_edges_sample.append(seg_id)
            else:
                segments_without_edges_sample.append(seg_id)
            
            if len(segments_with_edges_sample) >= 5 and len(segments_without_edges_sample) >= 5:
                break
        
        logger.info(f"Sample segments WITH edges: {segments_with_edges_sample[:5]}")
        logger.info(f"Sample segments WITHOUT edges: {segments_without_edges_sample[:5]}")

    # Log the structure of segments_mapping
    if segments_mapping:
        logger.info(f"Using segments mapping with {len(segments_mapping)} entries")
        # Show sample of keys if available
        sample_keys = list(segments_mapping.keys())[:5] if segments_mapping else []
        logger.info(f"Sample segment mapping keys: {sample_keys}")
    else:
        logger.warning("No valid segments mapping found in edge_mapping")

    # Track how many segments were matched
    matched_segment_count = 0
    mapped_edge_count = 0

    # Process realtime flow data
    for idx, segment in enumerate(realtime_data.get('results', [])):
        # Get location data for matching
        location = segment.get('location', {})
        location_shape = None
        
        # If location has 'shape' attribute, use that
        if isinstance(location, dict) and 'shape' in location:
            location_shape = location['shape']
        
        # Get traffic flow data
        flow = segment.get('currentFlow', {})
        if not flow:
            continue
        
        # Get jam factor and speed
        jam_factor = flow.get('jamFactor', 0)
        speed = flow.get('speed', 0)
        
        # Try to match by location or index-based segment ID
        sumo_edges = []
        # Try different ID formats that might exist in the mapping
        possible_ids = [
            f"segment_{idx}", 
            f"segment_0_{idx}",
            f"segment_{idx}_0",
            f"{idx}"  # Just the index
        ]

        # Also try to match by location if available
        location = segment.get('location', {})
        location_matched = False

        # Try ID-based matching first
        for possible_id in possible_ids:
            if possible_id in segments_mapping:
                mapping_value = segments_mapping[possible_id]
                logger.debug(f"Segment {idx}: Found match with ID '{possible_id}', type: {type(mapping_value)}")
                
                # Handle case where mapping value might be the edges directly or a dict with edges
                if isinstance(mapping_value, list):
                    sumo_edges = mapping_value
                    logger.debug(f"Segment {idx}: mapping_value is list: {sumo_edges}")
                elif isinstance(mapping_value, dict):
                    logger.debug(f"Segment {idx}: mapping_value is dict with keys: {mapping_value.keys()}")
                    # The logs show the key is 'sumo_edges', not 'edges'
                    if 'sumo_edges' in mapping_value:
                        sumo_edges = mapping_value['sumo_edges']
                        logger.debug(f"Segment {idx}: Found sumo_edges in dict: {sumo_edges}")
                    elif 'edges' in mapping_value:
                        sumo_edges = mapping_value['edges']
                        logger.debug(f"Segment {idx}: Found edges in dict: {sumo_edges}")
                    elif 'edge' in mapping_value:
                        sumo_edges = [mapping_value['edge']]  # Single edge
                        logger.debug(f"Segment {idx}: Found edge in dict: {sumo_edges}")
                    else:
                        logger.debug(f"Segment {idx}: Dict contains no edge keys")
                else:
                    # If it's a string, treat it as a single edge ID
                    if isinstance(mapping_value, str):
                        sumo_edges = [mapping_value]
                        logger.debug(f"Segment {idx}: mapping_value is string: {sumo_edges}")
                    else:
                        logger.debug(f"Segment {idx}: Unexpected mapping_value type: {type(mapping_value)}")
                
                matched_segment_count += 1
                break
        else:
            logger.debug(f"Segment {idx}: No ID match found. Tried: {possible_ids}")

        # If no ID match found, try location-based matching
        if not sumo_edges and location:
            # Get shape information from location
            shape = location.get('shape', {})
            if shape and isinstance(shape, dict):
                # Extract coordinates or links
                links = shape.get('links', [])
                
                # Try to match by links
                for link in links:
                    link_id = link.get('linkId')
                    if link_id:
                        # Try to find this link_id in the segments mapping
                        for seg_key, seg_value in segments_mapping.items():
                            if isinstance(seg_value, dict):
                                # Check if this segment has matching link ID in its shape
                                seg_shape = seg_value.get('shape', {})
                                if isinstance(seg_shape, dict):
                                    seg_links = seg_shape.get('links', [])
                                    for seg_link in seg_links:
                                        if seg_link.get('linkId') == link_id:
                                            if 'sumo_edges' in seg_value:
                                                sumo_edges.extend(seg_value['sumo_edges'])
                                                location_matched = True
                                                break
                                    if location_matched:
                                        break
                        if location_matched:
                            matched_segment_count += 1
                            break
                
                # If still no match, try matching by coordinates
                if not sumo_edges and 'coordinates' in location:
                    coords = location['coordinates']
                    if coords and len(coords) >= 2:
                        # Get first coordinate pair
                        start_coord = coords[0]
                        if isinstance(start_coord, list) and len(start_coord) >= 2:
                            start_lat, start_lon = start_coord[0], start_coord[1]
                            
                            # Search for segments with similar start points
                            tolerance = 0.0001  # About 11 meters
                            for seg_key, seg_value in segments_mapping.items():
                                if isinstance(seg_value, dict) and 'start_point' in seg_value:
                                    seg_start = seg_value['start_point']
                                    if abs(seg_start[0] - start_lat) < tolerance and abs(seg_start[1] - start_lon) < tolerance:
                                        if 'sumo_edges' in seg_value:
                                            sumo_edges = seg_value['sumo_edges']
                                            matched_segment_count += 1
                                            break
        
        logger.debug(f"Segment {idx}: After all matching attempts, sumo_edges = {sumo_edges}")

        # If we found edges, map the speed data
        if sumo_edges:
            # Debug: Check what type sumo_edges is and what it contains
            logger.debug(f"Segment {idx}: Found sumo_edges type: {type(sumo_edges)}, content: {sumo_edges}")
            
            # Ensure sumo_edges is a list
            if isinstance(sumo_edges, str):
                sumo_edges = [sumo_edges]
            elif not isinstance(sumo_edges, list):
                logger.warning(f"Segment {idx}: Unexpected sumo_edges type {type(sumo_edges)}")
                continue
            
            # Debug: Check if speed and jam_factor are valid
            logger.debug(f"Segment {idx}: Speed: {speed}, Jam factor: {jam_factor}")
            
            for edge_id in sumo_edges:
                if edge_id is None or edge_id == '':
                    logger.warning(f"Segment {idx}: Skipping empty edge_id")
                    continue
                    
                # Convert edge_id to string if it's not already
                edge_id = str(edge_id)
                
                if edge_id in edge_speeds:
                    # Average with existing speed
                    edge_speeds[edge_id] = (edge_speeds[edge_id] + speed) / 2
                    edge_jam_factors[edge_id] = (edge_jam_factors[edge_id] + jam_factor) / 2
                else:
                    edge_speeds[edge_id] = speed
                    edge_jam_factors[edge_id] = jam_factor
                    mapped_edge_count += 1
            
            logger.debug(f"Segment {idx}: Mapped {len(sumo_edges)} edges, total mapped: {mapped_edge_count}")
        else:
            logger.debug(f"Segment {idx}: No sumo_edges found")
    # # Process realtime flow data
    # for segment in realtime_data.get('results', []):
    #     # Get traffic flow data
    #     flow = segment.get('currentFlow', {})
    #     if not flow:
    #         continue
        
    #     # Get jam factor and speed
    #     jam_factor = flow.get('jamFactor', 0)
    #     speed = flow.get('speed', 0)
        
    #     # Get the SUMO edges from the mapping
    #     segment_id = segment.get('id')
    #     if segment_id and segment_id in edge_mapping:
    #         sumo_edges = edge_mapping[segment_id]
    #         for edge_id in sumo_edges:
    #             if edge_id in edge_speeds:
    #                 # Average with existing speed
    #                 edge_speeds[edge_id] = (edge_speeds[edge_id] + speed) / 2
    #                 edge_jam_factors[edge_id] = (edge_jam_factors[edge_id] + jam_factor) / 2
    #             else:
    #                 edge_speeds[edge_id] = speed
    #                 edge_jam_factors[edge_id] = jam_factor
    
    # Process incidents data to adjust speeds and generate incidents
    incidents = []
    incident_count = 0
    if incidents_data and 'incidents' in incidents_data:
        logger.info(f"Processing {len(incidents_data.get('incidents', []))} incidents")
        
        for incident_idx, incident in enumerate(incidents_data.get('incidents', [])):
            # Extract incident location
            location = incident.get('location', {})
            if not location:
                continue
            
            # Parse incident details
            incident_type = incident.get('type', 'UNKNOWN')
            incident_description = incident.get('description', '')
            incident_criticality = incident.get('criticality', 0)
            
            # Get affected edge IDs
            affected_edges = []
            
            # Debug first incident structure
            if incident_idx == 0:
                logger.info(f"Debug: First incident structure: {incident.keys()}")
                logger.info(f"Debug: First incident location: {str(location)[:100]}...")
            
            # Check if location has shape with links
            if isinstance(location, dict) and 'shape' in location and isinstance(location['shape'], dict):
                shape = location['shape']
                
                # Try to extract links
                links = shape.get('links', [])
                if links:
                    for link in links:
                        link_id = link.get('linkId')
                        # Use segments_mapping instead of edge_mapping directly
                        if link_id and link_id in segments_mapping:
                            affected_edges.extend(segments_mapping[link_id])
            
            # If we still don't have edges, try with raw coordinates
            if not affected_edges and 'coordinates' in location:
                # This would need a way to map coordinates to edges
                # Implement if needed
                pass
            
            # Adjust speeds on affected edges based on incident type and criticality
            if affected_edges:
                # Calculate speed reduction factor based on criticality (1-10 scale)
                # Higher criticality means more speed reduction
                reduction_factor = 0.05 * incident_criticality
                
                # Apply speed reduction to affected edges
                for edge_id in affected_edges:
                    if edge_id in edge_speeds:
                        reduced_speed = edge_speeds[edge_id] * (1 - reduction_factor)
                        edge_speeds[edge_id] = max(5.0, reduced_speed)  # Minimum speed of 5 km/h
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
    
    logger.info(f"Matched speed data for {matched_segment_count} segments")
    logger.info(f"Mapped speed data for {mapped_edge_count} edges")
    logger.info(f"Processed {incident_count} incidents")
    
    return {
        'timestamp': timestamp,
        'total_vehicles': total_vehicles,
        'edge_speeds': edge_speeds,
        'edge_jam_factors': edge_jam_factors,
        'incidents': incidents
    }

# def process_timestep_data(data, net):
#     """Process data for a single timestep to extract flow information"""
#     if not data or not data['realtime'] or not data['analysis'] or not data['edge_mapping'] or not data['entry_exit']:
#         logger.error("Incomplete data for timestep")
#         return None
    
#     timestamp = data['timestamp']
#     realtime_data = data['realtime']
#     incidents_data = data['incidents']
#     analysis_data = data['analysis']
#     edge_mapping = data['edge_mapping']
#     entry_exit = data['entry_exit']
    
#     logger.info(f"Processing data for timestep: {timestamp}")
    
#     # DEBUG: Print entry_exit file structure to understand what's happening
#     logger.info(f"Entry/exit file structure: {json.dumps(entry_exit, indent=2)[:500]}...")
    
#     # Extract estimated vehicle count from analysis
#     total_vehicles = analysis_data.get('total_vehicles', 0)
#     if total_vehicles <= 0:
#         logger.warning("No vehicle count data available, using default")
#         total_vehicles = 5000  # Default fallback
    
#     logger.info(f"Estimated total vehicles: {total_vehicles}")
    
#     # Extract entry/exit points
#     entry_points = entry_exit.get('entry_points', [])
#     exit_points = entry_exit.get('exit_points', [])

#     if not entry_points or not exit_points:
#         logger.error("No entry/exit points found")
#         return None

#     logger.info(f"Found {len(entry_points)} entry points and {len(exit_points)} exit points")

#         # Calculate weights for entry points based on traffic volume
#     entry_weights = []
#     entry_edge_ids = set()  # Keep track of used edge IDs to avoid duplicates
    
#     for i, entry in enumerate(entry_points):
#         weight = entry.get('weight', 1)
        
#         # Get location coordinates
#         location = entry.get('location', [])
#         if len(location) >= 2:
#             lat, lon = location[0], location[1]
            
#             # Find nearest edge
#             edge = find_nearest_edge_by_coordinates(net, lat, lon)
            
#             if edge:
#                 edge_id = edge.getID()
#                 # Check if this edge ID is already used
#                 if edge_id not in entry_edge_ids:
#                     logger.info(f"Entry point {i}: Mapped coordinates [{lat}, {lon}] to edge {edge_id}")
#                     entry_weights.append((edge_id, weight))
#                     entry_edge_ids.add(edge_id)
#                 else:
#                     # If duplicate, try to find neighboring edges instead
#                     logger.info(f"Entry point {i}: Edge {edge_id} already used, finding alternative...")
#                     x, y = convert_geo_to_sumo_coords(net, lat, lon)
#                     if x is not None and y is not None:
#                         edge_candidates = net.getNeighboringEdges(x, y, 100)
#                         found_alternative = False
#                         for candidate, dist in edge_candidates:
#                             candidate_id = candidate.getID()
#                             if candidate_id not in entry_edge_ids:
#                                 logger.info(f"Entry point {i}: Using alternative edge {candidate_id}")
#                                 entry_weights.append((candidate_id, weight))
#                                 entry_edge_ids.add(candidate_id)
#                                 found_alternative = True
#                                 break
#                         if not found_alternative:
#                             logger.warning(f"Entry point {i}: Could not find alternative edge, using original with duplicate ID")
#                             entry_weights.append((edge_id, weight))
#             else:
#                 logger.warning(f"Entry point {i}: Could not find edge near coordinates [{lat}, {lon}]")
#         else:
#             logger.warning(f"Entry point {i}: Invalid location format: {location}")
    
#     # Calculate weights for exit points based on traffic volume
#     exit_weights = []
#     exit_edge_ids = set()  # Keep track of used edge IDs to avoid duplicates
    
#     for i, exit in enumerate(exit_points):
#         weight = exit.get('weight', 1)
        
#         # Get location coordinates
#         location = exit.get('location', [])
#         if len(location) >= 2:
#             lat, lon = location[0], location[1]
            
#             # Find nearest edge
#             edge = find_nearest_edge_by_coordinates(net, lat, lon)
            
#             if edge:
#                 edge_id = edge.getID()
#                 # Check if this edge ID is already used or already in entry edges
#                 if edge_id not in exit_edge_ids and edge_id not in entry_edge_ids:
#                     logger.info(f"Exit point {i}: Mapped coordinates [{lat}, {lon}] to edge {edge_id}")
#                     exit_weights.append((edge_id, weight))
#                     exit_edge_ids.add(edge_id)
#                 else:
#                     # If duplicate, try to find neighboring edges instead
#                     logger.info(f"Exit point {i}: Edge {edge_id} already used, finding alternative...")
#                     x, y = convert_geo_to_sumo_coords(net, lat, lon)
#                     if x is not None and y is not None:
#                         edge_candidates = net.getNeighboringEdges(x, y, 100)
#                         found_alternative = False
#                         for candidate, dist in edge_candidates:
#                             candidate_id = candidate.getID()
#                             if candidate_id not in exit_edge_ids and candidate_id not in entry_edge_ids:
#                                 logger.info(f"Exit point {i}: Using alternative edge {candidate_id}")
#                                 exit_weights.append((candidate_id, weight))
#                                 exit_edge_ids.add(candidate_id)
#                                 found_alternative = True
#                                 break
#                         if not found_alternative:
#                             logger.warning(f"Exit point {i}: Could not find alternative edge, using original with duplicate ID")
#                             exit_weights.append((edge_id, weight))
#             else:
#                 logger.warning(f"Exit point {i}: Could not find edge near coordinates [{lat}, {lon}]")
#         else:
#             logger.warning(f"Exit point {i}: Invalid location format: {location}")
    
#     if not entry_weights or not exit_weights:
#         logger.error("No valid entry/exit edges found")
#         return None
    
#     # Normalize weights
#     entry_total = sum(w for _, w in entry_weights)
#     exit_total = sum(w for _, w in exit_weights)
    
#     normalized_entry_weights = [(edge, weight / entry_total) for edge, weight in entry_weights]
#     normalized_exit_weights = [(edge, weight / exit_total) for edge, weight in exit_weights]
    
#     # Calculate vehicle distribution for this timestep
#     vehicle_distribution = []
    
#     # Calculate the OD flow matrix based on entry/exit weights
#     # More vehicles should flow between high-weight entry and exit points
#     for entry_edge, entry_weight in normalized_entry_weights:
#         for exit_edge, exit_weight in normalized_exit_weights:
#             if entry_edge != exit_edge:  # Avoid self-loops
#                 # Calculate flow for this OD pair
#                 od_weight = entry_weight * exit_weight
#                 vehicle_count = int(total_vehicles * od_weight)
                
#                 if vehicle_count > 0:
#                     # Try to find a route between entry and exit
#                     route_edges = get_route_edges(net, entry_edge, exit_edge)
#                     if route_edges:
#                         vehicle_distribution.append({
#                             'from': entry_edge,
#                             'to': exit_edge,
#                             'count': vehicle_count,
#                             'route': route_edges,
#                             'probability': od_weight
#                         })
    
#     # Map traffic data to SUMO edges for speed adjustments
#     edge_speeds = {}
#     edge_jam_factors = {}
    
#     # Process realtime flow data
#     for segment in realtime_data.get('results', []):
#         # Get traffic flow data
#         flow = segment.get('currentFlow', {})
#         if not flow:
#             continue
        
#         # Get jam factor and speed
#         jam_factor = flow.get('jamFactor', 0)
#         speed = flow.get('speed', 0)
        
#         # Get the SUMO edges from the mapping
#         segment_id = segment.get('id')
#         if segment_id and segment_id in edge_mapping:
#             sumo_edges = edge_mapping[segment_id]
#             for edge_id in sumo_edges:
#                 if edge_id in edge_speeds:
#                     # Average with existing speed
#                     edge_speeds[edge_id] = (edge_speeds[edge_id] + speed) / 2
#                     edge_jam_factors[edge_id] = (edge_jam_factors[edge_id] + jam_factor) / 2
#                 else:
#                     edge_speeds[edge_id] = speed
#                     edge_jam_factors[edge_id] = jam_factor
    
#     # Process incidents data to adjust speeds and generate incidents
#     incidents = []
#     if incidents_data and 'incidents' in incidents_data:
#         for incident in incidents_data.get('incidents', []):
#             # Extract incident location
#             location = incident.get('location', {})
#             if not location:
#                 continue
            
#             # Parse incident details
#             incident_type = incident.get('type', 'UNKNOWN')
#             incident_description = incident.get('description', '')
#             incident_criticality = incident.get('criticality', 0)
            
#             # Get affected edge IDs
#             affected_edges = []
#             if location.get('shape') and location.get('shape', {}).get('links'):
#                 for link in location.get('shape', {}).get('links', []):
#                     link_id = link.get('linkId')
#                     if link_id and link_id in edge_mapping:
#                         affected_edges.extend(edge_mapping[link_id])
            
#             # Adjust speeds on affected edges based on incident type and criticality
#             if affected_edges:
#                 # Calculate speed reduction factor based on criticality (1-10 scale)
#                 # Higher criticality means more speed reduction
#                 reduction_factor = 0.05 * incident_criticality
                
#                 # Apply speed reduction to affected edges
#                 for edge_id in affected_edges:
#                     if edge_id in edge_speeds:
#                         reduced_speed = edge_speeds[edge_id] * (1 - reduction_factor)
#                         edge_speeds[edge_id] = max(5.0, reduced_speed)  # Minimum speed of 5 km/h
#                         # Increase jam factor
#                         edge_jam_factors[edge_id] = min(10.0, edge_jam_factors.get(edge_id, 0) + incident_criticality / 2)
                
#                 # Record incident for later use
#                 incidents.append({
#                     'type': incident_type,
#                     'description': incident_description,
#                     'criticality': incident_criticality,
#                     'affected_edges': affected_edges,
#                     'speed_reduction': reduction_factor
#                 })
    
#     logger.info(f"Created {len(vehicle_distribution)} vehicle flows for timestep {timestamp}")
#     logger.info(f"Mapped speed data for {len(edge_speeds)} edges")
#     logger.info(f"Processed {len(incidents)} incidents")
    
#     return {
#         'timestamp': timestamp,
#         'total_vehicles': total_vehicles,
#         'vehicle_distribution': vehicle_distribution,
#         'edge_speeds': edge_speeds,
#         'edge_jam_factors': edge_jam_factors,
#         'incidents': incidents
#     }

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
    """Create time-varying traffic demand files for SUMO"""
    logger.info("Creating time-varying traffic demand...")
    
    # Sort processed data by timestamp
    processed_data.sort(key=lambda x: x['timestamp'])
    
    # Create vehicle types XML
    vehicle_types_file = os.path.join(output_dir, "baseline_vtypes.xml")
    with open(vehicle_types_file, "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n')
        
        # Define vehicle types with realistic parameters
        f.write('    <vType id="passenger" vClass="passenger" color="0,0,1" accel="2.6" decel="4.5" sigma="0.5" length="4.5" minGap="2.5" maxSpeed="15" speedDev="0.1" guiShape="passenger"/>\n')
        f.write('    <vType id="taxi" vClass="taxi" color="1,1,0" accel="2.8" decel="4.5" sigma="0.5" length="4.5" minGap="2.0" maxSpeed="16" speedDev="0.1" guiShape="passenger"/>\n')
        f.write('    <vType id="delivery" vClass="delivery" color="1,0,0" accel="2.4" decel="4.0" sigma="0.5" length="6.5" minGap="3.0" maxSpeed="14" speedDev="0.1" guiShape="delivery"/>\n')
        f.write('    <vType id="bus" vClass="bus" color="0,1,0" accel="2.0" decel="3.5" sigma="0.5" length="12.0" minGap="3.5" maxSpeed="13" speedDev="0.1" guiShape="bus"/>\n')
        f.write('    <vType id="emergency" vClass="emergency" color="1,0,0" accel="3.0" decel="5.0" sigma="0.5" length="6.0" minGap="2.5" maxSpeed="20" speedDev="0.1" guiShape="emergency"/>\n')
        
        f.write('</routes>\n')
    
    # Vehicle type distribution
    vehicle_types = {
        'passenger': 0.6,  # 60% passenger cars
        'taxi': 0.25,      # 25% taxis
        'delivery': 0.1,   # 10% delivery vehicles
        'bus': 0.05        # 5% buses
    }
    
    # Create a flows file
    flows_file = os.path.join(output_dir, "baseline_flows.xml")
    with open(flows_file, "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n')
        
        f.write('    <include href="baseline_vtypes.xml"/>\n\n')
        
        # Import routes from duarouter if available
        trip_routes = {}
        if duarouter_routes_file and os.path.exists(duarouter_routes_file):
            logger.info(f"Importing routes from duarouter file: {duarouter_routes_file}")
            try:
                # Parse the duarouter routes file
                tree = ET.parse(duarouter_routes_file)
                root = tree.getroot()
                
                # Extract all routes
                for vehicle in root.findall('.//vehicle'):
                    trip_id = vehicle.get('id')
                    route = vehicle.find('route')
                    if route is not None:
                        edges = route.get('edges')
                        # Store route for this trip
                        route_id = f"route_{trip_id}"
                        f.write(f'    <route id="{route_id}" edges="{edges}"/>\n')
                        trip_routes[trip_id] = route_id
                    
                # If no vehicles with routes, check for trips directly
                if not trip_routes:
                    for trip in root.findall('.//trip'):
                        trip_id = trip.get('id')
                        from_edge = trip.get('from')
                        to_edge = trip.get('to')
                        if from_edge and to_edge:
                            # Store from/to for this trip
                            trip_routes[trip_id] = {"from": from_edge, "to": to_edge}
                
                logger.info(f"Imported {len(trip_routes)} routes from duarouter")
                
            except Exception as e:
                logger.error(f"Error parsing duarouter routes file: {e}")
        
        # Process each timestep with varying vehicle counts
        flow_id = 0
        
        # Vehicle distribution for timestamps more realistic:
        # Most vehicles are already in the network at the beginning
        # First timestamp has the highest number of vehicles
        # Later timestamps have less new vehicles proportionally
        
        for timestep_idx, timestep_data in enumerate(processed_data):
            timestamp = timestep_data['timestamp']
            
            # Calculate the begin and end times for this timestep
            if timestep_idx < len(processed_data) - 1:
                begin_time = 0 if timestep_idx == 0 else (timestamp - processed_data[0]['timestamp']).total_seconds()
                end_time = (processed_data[timestep_idx + 1]['timestamp'] - processed_data[0]['timestamp']).total_seconds()
            else:
                # Last timestep - extend to end of simulation
                begin_time = (timestamp - processed_data[0]['timestamp']).total_seconds()
                end_time = simulation_period
            
            # Ensure begin_time is within simulation period
            if begin_time >= simulation_period:
                logger.warning(f"Timestep {timestamp} begins after simulation period, skipping")
                continue
            
            # Cap end_time to simulation period
            end_time = min(end_time, simulation_period)
            
            # Calculate vehicle count for this timestep
            # For first timestep, use full vehicle count
            # For later timesteps, only add the incremental vehicles
            total_vehicles = timestep_data.get('total_vehicles', 5000)
            
            if timestep_idx > 0:
                # Only add the difference from previous timestep
                prev_vehicles = processed_data[timestep_idx-1].get('total_vehicles', 5000)
                incremental_vehicles = max(0, total_vehicles - prev_vehicles)
            else:
                # First timestep - add all vehicles
                incremental_vehicles = total_vehicles
            
            logger.info(f"Creating flows for timestep {timestamp} (simulation time {begin_time}-{end_time})")
            logger.info(f"Adding {incremental_vehicles} new vehicles for this timestep")
            
            if incremental_vehicles <= 0:
                logger.info(f"No new vehicles for this timestep, skipping flow generation")
                continue
            
            # Create flows for each OD pair with the actual vehicle count for this timestep
            if trip_routes:
                # Calculate vehicles per route
                vehicles_per_route = max(1, incremental_vehicles / len(trip_routes))
                
                for trip_id, route_info in trip_routes.items():
                    # Create flows for each vehicle type according to distribution
                    for vtype, proportion in vehicle_types.items():
                        vtype_count = max(1, int(vehicles_per_route * proportion))
                        
                        # Create flow element
                        if isinstance(route_info, str) and route_info.startswith("route_"):
                            # Use pre-computed route
                            f.write(f'    <flow id="flow_{flow_id}_{vtype}" type="{vtype}" route="{route_info}" begin="{begin_time}" end="{end_time}" number="{vtype_count}" departLane="best" departSpeed="max"/>\n')
                        else:
                            # Use from/to
                            f.write(f'    <flow id="flow_{flow_id}_{vtype}" type="{vtype}" from="{route_info["from"]}" to="{route_info["to"]}" begin="{begin_time}" end="{end_time}" number="{vtype_count}" departLane="best" departSpeed="max"/>\n')
                        
                        flow_id += 1
            
            # Add emergency vehicles for incidents if any
            if 'incidents' in timestep_data and timestep_data['incidents']:
                for incident_idx, incident in enumerate(timestep_data['incidents']):
                    if 'affected_edges' in incident and incident['affected_edges']:
                        # Get the first affected edge as the target for emergency vehicle
                        target_edge = incident['affected_edges'][0]
                        
                        # Find a suitable starting point
                        start_edge = None
                        if trip_routes:
                            # Use first route as starting point
                            first_trip = next(iter(trip_routes.items()))
                            if isinstance(first_trip[1], str) and first_trip[1].startswith("route_"):
                                # Get route edges
                                route = root.find(f'.//route[@id="{first_trip[1]}"]')
                                if route is not None:
                                    edges = route.get('edges')
                                    if edges:
                                        edges_list = edges.split()
                                        start_edge = edges_list[0] if edges_list else None
                            elif isinstance(first_trip[1], dict):
                                start_edge = first_trip[1]["from"]
                        
                        if start_edge and start_edge != target_edge:
                            # Create an emergency vehicle flow
                            incident_type = incident.get('type', 'UNKNOWN')
                            criticality = incident.get('criticality', 0)
                            
                            # Criticality affects the number of emergency vehicles
                            num_vehicles = max(1, int(criticality / 3))  # 1-3 vehicles depending on criticality
                            
                            f.write(f'    <flow id="emergency_{incident_idx}_{timestep_idx}" type="emergency" from="{start_edge}" to="{target_edge}" begin="{begin_time}" end="{begin_time + 300}" number="{num_vehicles}" departLane="best" departSpeed="max"/>\n')
        
        f.write('</routes>\n')
    
    # The rest of the function (speed adjustments, incidents) remains the same
    
    # Create a speed adjustment file for each timestep
    additional_files = []
    
    for timestep_idx, timestep_data in enumerate(processed_data):
        timestamp = timestep_data['timestamp']
        edge_speeds = timestep_data['edge_speeds']
        
        if not edge_speeds:
            continue
        
        # Calculate the begin and end times for this timestep
        if timestep_idx < len(processed_data) - 1:
            begin_time = 0 if timestep_idx == 0 else (timestamp - processed_data[0]['timestamp']).total_seconds()
            end_time = (processed_data[timestep_idx + 1]['timestamp'] - processed_data[0]['timestamp']).total_seconds()
        else:
            # Last timestep - extend to end of simulation
            begin_time = (timestamp - processed_data[0]['timestamp']).total_seconds()
            end_time = simulation_period
        
        # Ensure begin_time is within simulation period
        if begin_time >= simulation_period:
            continue
        
        # Cap end_time to simulation period
        end_time = min(end_time, simulation_period)
        
        # Create an additional file for speed adjustments
        add_file = os.path.join(output_dir, f"speeds_{timestep_idx}.add.xml")
        with open(add_file, "w") as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<additional xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/additional_file.xsd">\n')
            
            # Create a time interval for speed adjustments
            f.write(f'    <interval begin="{begin_time}" end="{end_time}">\n')
            
            # Adjust speed for each edge
            for edge_id, speed in edge_speeds.items():
                if speed <= 0:
                    continue
                
                # Convert km/h to m/s
                speed_ms = speed / 3.6
                f.write(f'        <edgeData id="{edge_id}" speed="{speed_ms:.2f}"/>\n')
            
            f.write('    </interval>\n')
            f.write('</additional>\n')
        
        additional_files.append(add_file)
        
        # Create an additional file for incidents
        if 'incidents' in timestep_data and timestep_data['incidents']:
            incidents_file = os.path.join(output_dir, f"incidents_{timestep_idx}.add.xml")
            with open(incidents_file, "w") as f:
                f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
                f.write('<additional xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/additional_file.xsd">\n')
                
                # Create rerouters for each incident
                for incident_idx, incident in enumerate(timestep_data['incidents']):
                    if 'affected_edges' in incident and incident['affected_edges']:
                        # Create a rerouter for this incident
                        rerouter_id = f"rerouter_incident_{incident_idx}_{timestep_idx}"
                        affected_edges = " ".join(incident['affected_edges'])
                        
                        f.write(f'    <rerouter id="{rerouter_id}" edges="{affected_edges}">\n')
                        f.write(f'        <interval begin="{begin_time}" end="{end_time}">\n')
                        
                        # Add closing for major incidents
                        if incident.get('criticality', 0) > 7:
                            for edge_id in incident['affected_edges']:
                                f.write(f'            <closingReroute id="{edge_id}"/>\n')
                        
                        # Add reduced speed area
                        else:
                            for edge_id in incident['affected_edges']:
                                if edge_id in edge_speeds:
                                    # Get original speed and apply reduction
                                    orig_speed = edge_speeds[edge_id]
                                    reduced_speed = orig_speed * (1 - incident.get('speed_reduction', 0.3))
                                    # Convert to m/s
                                    reduced_speed_ms = max(5.0, reduced_speed) / 3.6
                                    
                                    f.write(f'            <speedLimit id="{edge_id}" speed="{reduced_speed_ms:.2f}"/>\n')
                        
                        f.write('        </interval>\n')
                        f.write('    </rerouter>\n')
                
                f.write('</additional>\n')
                
                additional_files.append(incidents_file)
    
    # Create a file listing all additional files
    additionals_list_file = os.path.join(output_dir, "additionals.xml")
    with open(additionals_list_file, "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<configuration>\n')
        f.write('    <input>\n')
        for add_file in additional_files:
            f.write(f'        <additional-files value="{os.path.basename(add_file)}"/>\n')
        f.write('    </input>\n')
        f.write('</configuration>\n')
    
    logger.info(f"Created traffic demand with realistic time-varying vehicle counts")
    logger.info(f"Created {len(additional_files)} additional files (speeds and incidents)")
    
    return {
        'vehicle_types_file': vehicle_types_file,
        'flows_file': flows_file,
        'additional_files': additional_files,
        'additionals_list_file': additionals_list_file
    }
    
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
    
    # Prepare command
    # Prepare command
    cmd = [
        sumo_binary,
        "-c", config_file,
        "--output-prefix", "",  # Remove the output prefix since paths are already correct
        "--no-warnings", "false",
        "--duration-log.statistics", "true"
    ]
    
    try:
        # Run SUMO
        logger.info(f"Starting SUMO: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
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

def analyze_results(output_dir):
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
                timestamp = datetime.fromtimestamp(time).strftime('%Y-%m-%d %H:%M:%S')
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
                    
                    # Approximate jam factor: 0-10 scale based on queue length
                    # Assume 500m total queue length equals jam factor 10
                    jam_factor = min(10, total_queue_length / 50)
                    metric['avg_jam_factor'] = jam_factor
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

def main():
    """Main execution function"""
    # Parse command line arguments
    args = parse_arguments()
    
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
        
        # Load data for this timestep
        timestep_data = load_data_for_timestep(args.data_dir, realtime_file)
        if not timestep_data:
            logger.warning(f"Could not load complete data for {realtime_file}, skipping")
            failed_timesteps += 1
            continue
        
        # Process the timestep data - only for speed data
        processed_timestep = process_timestep_data(timestep_data, net)
        if processed_timestep:
            processed_data.append(processed_timestep)
            processed_timestamps.append(file_timestamp)
            successful_timesteps += 1
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
    
    # Step 4: Find the entry/exit points file
    entry_exit_files = glob.glob(os.path.join(args.data_dir, "sumo_entry_exit_points_*.json"))
    if not entry_exit_files:
        logger.error("No entry/exit points files found. Exiting.")
        return False
    
    entry_exit_file = entry_exit_files[0]
    logger.info(f"Using entry/exit points file: {entry_exit_file}")
    
    # Load entry/exit data
    entry_exit_data = load_json_file(entry_exit_file)
    if not entry_exit_data:
        logger.error(f"Failed to load entry/exit data from {entry_exit_file}")
        return False
    
    # Step 5: Create trips file for duarouter
    trips_file = os.path.join(args.output_dir, "geo_trips.xml")
    trips_file, simulation_period = create_duarouter_trips(entry_exit_data, trips_file, processed_data)
    
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
    demand_files = create_time_varying_demand(processed_data, args.output_dir, simulation_period, routes_file)
    if not demand_files:
        logger.error("Failed to create traffic demand. Exiting.")
        return False
    
    # Step 9: Create SUMO configuration
    config_file = create_sumo_config(
        net_file,
        demand_files['flows_file'],
        demand_files['additional_files'],
        args.output_dir,
        simulation_period
    )
    if not config_file:
        logger.error("Failed to create SUMO configuration. Exiting.")
        return False
    
    # Step 10: Run simulation (unless skipped)
    if not args.skip_simulation:
        success = run_simulation(config_file, args.output_dir, args.gui)
        if not success:
            logger.error("Simulation failed. Continuing with analysis of available outputs.")
    else:
        logger.info("Skipping simulation run as requested.")
    
    # Step 11: Analyze results
    results = analyze_results(args.output_dir)

    # Create comparison CSV
    comparison_csv = create_comparison_csv(args.output_dir)
    if comparison_csv:
        logger.info(f"Created comparison CSV at {comparison_csv}")
    
    # Step 12: Create visualizations
    create_visualizations(results, args.output_dir)
    
    logger.info("Baseline SUMO Simulation completed.")
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



# def main():
#     """Main execution function"""
#     # Parse command line arguments
#     args = parse_arguments()
    
#     # Create output directory
#     os.makedirs(args.output_dir, exist_ok=True)
    
#     logger.info("Starting Baseline SUMO Simulation using HERE API data")
#     logger.info(f"Data directory: {args.data_dir}")
#     logger.info(f"Output directory: {args.output_dir}")
    
#     # Construct the full path to the OSM file
#     if os.path.isabs(args.osm_file):
#         osm_file = args.osm_file
#     else:
#         osm_file = os.path.join(args.data_dir, args.osm_file)
    
#     logger.info(f"OSM file path: {osm_file}")

#     # In main(), before calling create_sumo_network()
#     if os.path.isabs(args.osm_file):
#         osm_file_path = args.osm_file
#     else:
#         osm_file_path = os.path.join(args.data_dir, args.osm_file)

#     logger.info(f"OSM file path: {osm_file_path}")
   
#     # Check if OSM file exists, if not download it
#     if not os.path.exists(osm_file) or args.force_regenerate:
#         logger.info(f"OSM file {osm_file} not found or force regenerate specified")
#         # Use the coordinates to download the map
#         osm_file = download_restricted_osm_map(
#             args.min_lat, args.min_lon, args.max_lat, args.max_lon, 
#             args.data_dir, os.path.basename(args.osm_file)
#         )
#         if not osm_file:
#             logger.error("Failed to download OSM map. Exiting.")
#             return False

#     else:
#         logger.info(f"Using existing OSM file: {osm_file}")

#     # Step 1: Create SUMO network from OSM file
#     net_file = os.path.join(args.output_dir, "baseline.net.xml")
#     if args.force_regenerate or not os.path.exists(net_file):
#         net_file = create_sumo_network(args.osm_file, args.output_dir)
#         if not net_file:
#             logger.error("Failed to create SUMO network. Exiting.")
#             return False
#     else:
#         logger.info(f"Using existing SUMO network: {net_file}")
    
#     # Load the SUMO network
#     net = sumolib.net.readNet(net_file)
    
#     # Step 2: Load realtime traffic data files
#     realtime_files = get_realtime_files(args.data_dir)
#     if not realtime_files:
#         logger.error("No realtime traffic data files found. Exiting.")
#         return False
    
#     # Also get incidents files for a double-check
#     incidents_files = get_incidents_files(args.data_dir)
    
#     logger.info(f"Found {len(realtime_files)} realtime traffic data files")
#     logger.info(f"Found {len(incidents_files)} incident data files")
    
#     # Step 3: Process each timestep
#     processed_data = []
#     successful_timesteps = 0
#     failed_timesteps = 0
    
#     # Track timestamps for validation
#     processed_timestamps = []
    
#     for realtime_file in realtime_files:
#         # Extract timestamp for logging
#         file_timestamp = extract_timestamp_from_filename(realtime_file)
#         logger.info(f"Processing timestep {file_timestamp} from {os.path.basename(realtime_file)}")
        
#         # Load data for this timestep
#         timestep_data = load_data_for_timestep(args.data_dir, realtime_file)
#         if not timestep_data:
#             logger.warning(f"Could not load complete data for {realtime_file}, skipping")
#             failed_timesteps += 1
#             continue
        
#         # Process the timestep data
#         processed_timestep = process_timestep_data(timestep_data, net)
#         if processed_timestep:
#             processed_data.append(processed_timestep)
#             processed_timestamps.append(file_timestamp)
#             successful_timesteps += 1
#         else:
#             logger.warning(f"Failed to process timestep data for {realtime_file}")
#             failed_timesteps += 1
    
#     if not processed_data:
#         logger.error("No valid processed timestep data. Exiting.")
#         return False
    
#     logger.info(f"Successfully processed {successful_timesteps} timesteps of traffic data")
#     logger.info(f"Failed to process {failed_timesteps} timesteps")
    
#     # Validate we have a good spread of timestamps
#     if processed_timestamps:
#         min_time = min(processed_timestamps)
#         max_time = max(processed_timestamps)
#         duration = (max_time - min_time).total_seconds() / 60  # minutes
#         logger.info(f"Time span of processed data: {duration:.1f} minutes ({min_time} to {max_time})")
    
#     # Step 4: Create GUI settings
#     create_gui_settings(args.output_dir)
    
#     # Step 5: Create time-varying traffic demand
#     demand_files = create_time_varying_demand(processed_data, args.output_dir, args.simulation_period)
#     if not demand_files:
#         logger.error("Failed to create traffic demand. Exiting.")
#         return False
    
#     # Step 6: Create SUMO configuration
#     config_file = create_sumo_config(
#         net_file,
#         demand_files['flows_file'],
#         demand_files['additional_files'],
#         args.output_dir,
#         args.simulation_period
#     )
#     if not config_file:
#         logger.error("Failed to create SUMO configuration. Exiting.")
#         return False
    
#     # Step 7: Run simulation (unless skipped)
#     if not args.skip_simulation:
#         success = run_simulation(config_file, args.output_dir, args.gui)
#         if not success:
#             logger.error("Simulation failed. Continuing with analysis of available outputs.")
#     else:
#         logger.info("Skipping simulation run as requested.")
    
#     # Step 8: Analyze results
#     results = analyze_results(args.output_dir)
    
#     # Step 9: Create visualizations
#     create_visualizations(results, args.output_dir)
    
#     logger.info("Baseline SUMO Simulation completed.")
#     logger.info(f"All output files are in: {args.output_dir}")
    
#     # Print key statistics
#     if 'trip_stats' in results:
#         logger.info("Trip Statistics:")
#         logger.info(f"  Total Trips: {results['trip_stats'].get('total_trips', 0)}")
#         logger.info(f"  Avg Duration: {results['trip_stats'].get('avg_duration', 0):.2f} s")
#         logger.info(f"  Avg Wait Time: {results['trip_stats'].get('avg_wait_time', 0):.2f} s")
#         logger.info(f"  Avg Speed: {results['trip_stats'].get('avg_speed', 0):.2f} m/s")
    
#     if 'queue_stats' in results and 'global_stats' in results['queue_stats']:
#         logger.info("Queue Statistics:")
#         logger.info(f"  Max Queue Length: {results['queue_stats']['global_stats'].get('max_queue_any_lane', 0):.2f} m")
#         logger.info(f"  Avg Queue Length: {results['queue_stats']['global_stats'].get('avg_queue_all_lanes', 0):.2f} m")
    
#     return True
