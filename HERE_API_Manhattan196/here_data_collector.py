#!/usr/bin/env python3
"""
HERE Traffic Data Collector (Revised)
------------------------------------
This script collects historical traffic data from HERE APIs for Manhattan:
- Traffic Flow API for typical traffic flow data with directional information
- Traffic Incidents API for historical incident data

Data is collected for typical weekday (Wednesday) patterns:
- Morning rush hour: 8:00-9:00 AM
- Evening rush hour: 5:00-6:00 PM

The data is processed to extract traffic densities, directionality, and congestion
patterns for direct use in SUMO traffic simulation.

Output Files:

traffic_data/traffic_flow_morning_rush.json - Raw traffic flow data (morning)
traffic_data/traffic_flow_evening_rush.json - Raw traffic flow data (evening)
traffic_data/processed_traffic_flow_morning_rush.json - Processed traffic flow data (morning)
traffic_data/processed_traffic_flow_evening_rush.json - Processed traffic flow data (evening)
traffic_data/traffic_incidents_morning_rush.json - Raw traffic incidents (morning)
traffic_data/traffic_incidents_evening_rush.json - Raw traffic incidents (evening)
traffic_data/processed_incidents_morning_rush.json - Processed incident data (morning)
traffic_data/processed_incidents_evening_rush.json - Processed incident data (evening)
traffic_data/sumo_edge_mapping_morning_rush.json - For SUMO integration (morning)
traffic_data/sumo_edge_mapping_evening_rush.json - For SUMO integration (evening)
Various visualization images in traffic_data/visualizations_* directories

Usage:
python here_data_collector.py --api-key YOUR_API_KEY
"""


import os
import sys
import json
import time
import random
import argparse
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt

# Define workspace path
WORKSPACE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(WORKSPACE, "traffic_data"), exist_ok=True)
os.chdir(WORKSPACE)

# HERE API configuration
HERE_API_KEY = None  # Will be set from command line

# API call counter to ensure we stay within free tier limits
API_CALL_COUNTER = {
    "traffic_flow": 0,
    "traffic_incidents": 0
}

# Free tier limits
API_FREE_TIER_LIMITS = {
    "traffic_flow": 5000,    # Traffic API calls per month
    "traffic_incidents": 1000 # Traffic Incidents API calls per month
}

# Larger Manhattan bounding box
BBOX = {
    "min_lat": 40.712178,
    "min_lon": -74.033341,
    "max_lat": 40.759722,
    "max_lon": -73.958424
}

# Time periods for data collection
TIME_PERIODS = {
    "morning_rush": {
        "hour": 8,
        "minute": 0,
        "duration": 60  # minutes
    },
    "evening_rush": {
        "hour": 17,
        "minute": 0,
        "duration": 60  # minutes
    }
}

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Collect traffic data from HERE APIs')
    parser.add_argument('--api-key', required=True, help='HERE API key')
    parser.add_argument('--skip-flow', action='store_true', help='Skip Traffic Flow API calls')
    parser.add_argument('--skip-incidents', action='store_true', help='Skip Traffic Incidents API calls')
    parser.add_argument('--force-regenerate', action='store_true', help='Force regenerate all data')
    parser.add_argument('--visualize', action='store_true', help='Create visualizations of traffic data')
    return parser.parse_args()

def check_api_limits(api_type):
    """Check if we're approaching API call limits"""
    if API_CALL_COUNTER[api_type] >= API_FREE_TIER_LIMITS[api_type]:
        print(f"Warning: {api_type} API call limit reached ({API_FREE_TIER_LIMITS[api_type]} calls)")
        return False
    
    # Warn at 80% of limit
    if API_CALL_COUNTER[api_type] >= 0.8 * API_FREE_TIER_LIMITS[api_type]:
        print(f"Warning: Approaching {api_type} API call limit ({API_CALL_COUNTER[api_type]}/{API_FREE_TIER_LIMITS[api_type]} calls)")
    
    return True

def generate_timestamp_for_prediction(day_of_week=2, hour=8, minute=0):
    """
    Generate a timestamp for the predictive traffic API for a specific day of week and time
    
    Args:
        day_of_week: 0=Monday, 1=Tuesday, ..., 6=Sunday (default is 2=Wednesday)
        hour: Hour of day (0-23)
        minute: Minute of hour (0-59)
    
    Returns:
        ISO 8601 formatted timestamp (RFC 3339 compliant)
    """
    # Get current date
    now = datetime.now(timezone.utc)
    
    # Calculate days until next specified day of week
    days_ahead = day_of_week - now.weekday()
    if days_ahead <= 0:
        days_ahead += 7  # Target next week if today is the target day or we've passed it
    
    # Create the future date
    target_date = now + timedelta(days=days_ahead)
    
    # Set the specific time
    target_date = target_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
    
    # Format according to RFC 3339
    return target_date.isoformat()

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371000  # Radius of earth in meters
    return c * r

def get_traffic_flow_data():
    """Get historical traffic flow data from HERE Traffic API for specified time periods"""
    print("Requesting historical traffic flow data from HERE Traffic API...")
    
    if not check_api_limits("traffic_flow"):
        return None
    
    # IMPORTANT: For HERE API v7, the bbox format must be: 
    # in=bbox:{west longitude},{south latitude},{east longitude},{north latitude}
    # This is the same as: min_lon, min_lat, max_lon, max_lat
    
    bbox_str = f"{BBOX['min_lon']},{BBOX['min_lat']},{BBOX['max_lon']},{BBOX['max_lat']}"
    print(f"Using bounding box: {bbox_str}")
    
    # HERE Traffic Flow API endpoint for historical / predictive data
    traffic_url = "https://data.traffic.hereapi.com/v7/flow"
    
    results = {}
    
    for period_name, period_info in TIME_PERIODS.items():
        # Generate timestamp for prediction
        prediction_time = generate_timestamp_for_prediction(
            day_of_week=2,  # Wednesday
            hour=period_info["hour"],
            minute=period_info["minute"]
        )
        
        output_file = f"traffic_data/traffic_flow_{period_name}.json"
        
        # Skip if file exists and we're not forcing regeneration
        if os.path.exists(output_file) and not args.force_regenerate:
            print(f"Using existing traffic flow data for {period_name} from {output_file}")
            with open(output_file, "r") as f:
                results[period_name] = json.load(f)
            continue
        
        # Try different API parameter combinations
        attempts = [
            # Attempt 1: Standard approach with correct bbox format
            {
                "apiKey": HERE_API_KEY,
                "locationReferencing": "shape",
                "in": f"bbox:{bbox_str}",
                "functionalClass": "1,2,3,4,5",  # All road types, comma-separated string
                "returnJamFactor": "true",
                "returnTravelTime": "true",
                "returnFreeFlow": "true"
            },
            # Attempt 2: Try with predictedTimes
            {
                "apiKey": HERE_API_KEY,
                "locationReferencing": "shape",
                "in": f"bbox:{bbox_str}",
                "functionalClass": "1,2,3,4,5",  # All road types
                "returnJamFactor": "true",
                "returnTravelTime": "true",
                "returnFreeFlow": "true",
                "predictedTimes": prediction_time
            },
            # Attempt 3: Try with different locationReferencing
            {
                "apiKey": HERE_API_KEY,
                "locationReferencing": "olr",
                "in": f"bbox:{bbox_str}",
                "functionalClass": "1,2,3,4,5",
                "returnJamFactor": "true",
                "returnTravelTime": "true",
                "returnFreeFlow": "true"
            }
        ]
        
        success = False
        for attempt_num, params in enumerate(attempts):
            try:
                print(f"Attempt {attempt_num+1} with params: {params}")
                response = requests.get(traffic_url, params=params)
                API_CALL_COUNTER["traffic_flow"] += 1
                
                print(f"Response status: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"Response data: {str(data)[:200]}...")  # Print first 200 chars for debugging
                    
                    # Check if we got any results
                    if 'results' in data and len(data.get('results', [])) > 0:
                        # Add metadata
                        data['metadata'] = {
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'period': period_name,
                            'prediction_time': prediction_time if 'predictedTimes' in params else 'current',
                            'bbox': BBOX,
                            'attempt': attempt_num+1
                        }
                        
                        # Save raw traffic data
                        with open(output_file, "w") as f:
                            json.dump(data, f, indent=2)
                            
                        print(f"Traffic flow data for {period_name} retrieved successfully with {len(data.get('results', []))} records.")
                        results[period_name] = data
                        
                        # Process into traffic flow format for SUMO
                        processed_file = f"traffic_data/processed_traffic_flow_{period_name}.json"
                        processed_data = process_traffic_flow_data(data, period_name)
                        
                        with open(processed_file, "w") as f:
                            json.dump(processed_data, f, indent=2)
                        
                        success = True
                        break  # Break out of attempts loop if successful
                    else:
                        print(f"Request succeeded but no results returned for attempt {attempt_num+1}")
                        if 'errors' in data:
                            print(f"API errors: {data['errors']}")
                else:
                    print(f"Failed to get traffic flow data (attempt {attempt_num+1}): {response.status_code}")
                    print(f"Response: {response.text}")
                    
            except Exception as e:
                print(f"Error in attempt {attempt_num+1}: {e}")
            
            # Sleep to avoid rate limiting
            time.sleep(2)
        
        if not success:
            print(f"ERROR: All attempts failed to get traffic flow data for {period_name}.")
            print("No valid data could be obtained from the HERE Traffic API.")
            print("Possible issues:")
            print("1. Invalid or restricted API key")
            print("2. The bounding box area may be too large or have no traffic data")
            print("3. API service disruption")
            print(f"Bounding box used: {bbox_str}")
            print("Exiting script - fix API issues before continuing.")
            sys.exit(1)  # Exit with error
    
    return results

def calculate_segment_direction(points):
    """Calculate the overall direction of a road segment based on its shape"""
    if len(points) < 2:
        return "unknown"
    
    # Get start and end points
    start_lat, start_lon = points[0]
    end_lat, end_lon = points[-1]
    
    # Calculate deltas
    d_lat = end_lat - start_lat
    d_lon = end_lon - start_lon
    
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
    
    return direction

def process_traffic_flow_data(data, period_name):
    """Process raw traffic flow data into a format suitable for SUMO traffic generation"""
    print(f"Processing traffic flow data for {period_name}...")
    
    # Initialize storage for processed data
    processed_data = {
        "segments": [],         # Road segments with traffic flow data
        "total_vehicles": 0,    # Estimated total vehicles in the area
        "avg_speed": 0,         # Average speed in the area
        "jam_factor": 0,        # Average jam factor in the area
        "flow_records": 0,      # Number of flow records processed
        "directional_flows": {  # Count of flows by direction
            "N-S": 0,
            "S-N": 0,
            "E-W": 0,
            "W-E": 0,
            "NE-SW": 0,
            "SW-NE": 0,
            "NW-SE": 0,
            "SE-NW": 0,
            "unknown": 0
        },
        "major_inflows": [],    # Major entry points with high inflow
        "major_outflows": []    # Major exit points with high outflow
    }
    
    # Process each flow item
    flow_items = data.get("results", [])
    
    jam_factors = []
    speeds = []
    free_flow_speeds = []
    total_length = 0  # Total road length in meters
    
    for item in flow_items:
        # Get traffic flow data - handle predictedFlow for historical/predictive data
        current_flow = item.get("predictedFlow", item.get("currentFlow", {}))
        
        if not current_flow:
            continue
            
        # Get jam factor, speeds
        jam_factor = current_flow.get("jamFactor", 0)
        speed = current_flow.get("speed", 0)
        # Handle freeFlow as a direct float value, not an object with nested properties
        free_flow = current_flow.get("freeFlow", 0)  # This is directly the speed value
        travel_time = current_flow.get("travelTime", 0)
        
        # No need to try to access free_flow_time from a nested structure
        # Set it to 0 as we don't have this information in the actual API response
        free_flow_time = 0
        # Get road information
        location = item.get("location", {})
        
        # Handle the shape structure
        shape_data = location.get("shape", {})
        links = shape_data.get("links", [])
        
        if not links:
            continue
        
        # Process all links
        for link in links:
            # Get link properties
            functional_class = link.get("functionalClass", 5)  # Default to lowest class if not specified
            link_length = link.get("length", 0)
            road_name = link.get("names", [{}])[0].get("value", "Unknown") if link.get("names") else "Unknown"
            
            points = link.get("points", [])
            if len(points) < 2:
                continue
                
            # Extract coordinates
            coords = []
            for point in points:
                lat = point.get("lat")
                lng = point.get("lng")
                if lat is not None and lng is not None:
                    coords.append((lat, lng))
            
            # Skip if insufficient coordinates
            if len(coords) < 2:
                continue
                
            # Calculate segment length if not provided
            if link_length <= 0:
                link_length = 0
                for i in range(len(coords) - 1):
                    lat1, lon1 = coords[i]
                    lat2, lon2 = coords[i+1]
                    # Use Haversine formula for better distance calculation
                    link_length += haversine_distance(lat1, lon1, lat2, lon2)
            
            # Calculate segment direction
            segment_direction = calculate_segment_direction(coords)
            processed_data["directional_flows"][segment_direction] += 1
            
            # Estimate vehicle count based on length, speed, and jam factor
            if jam_factor < 2:
                vehicles_per_km = 10 + jam_factor * 5
            elif jam_factor < 5:
                vehicles_per_km = 20 + (jam_factor - 2) * 10
            elif jam_factor < 8:
                vehicles_per_km = 50 + (jam_factor - 5) * 16.7
            else:
                vehicles_per_km = 100 + (jam_factor - 8) * 33.3
                
            # Calculate vehicles for this segment
            segment_vehicles = vehicles_per_km * (link_length / 1000)
            
            # Create segment record
            segment_record = {
                "id": f"{period_name}_segment_{processed_data['flow_records']}",
                "road_name": road_name,
                "functional_class": functional_class,
                "jam_factor": jam_factor,
                "speed": speed,
                "free_flow_speed": free_flow,
                "travel_time": travel_time,
                "free_flow_time": free_flow_time,
                "length": link_length,
                "vehicle_count": int(segment_vehicles),
                "vehicle_density": vehicles_per_km,
                "direction": segment_direction,
                "coords": coords,
                # Calculate start/end points for connections
                "start_point": coords[0],
                "end_point": coords[-1]
            }
            
            # Is this a major inflow/outflow point?
            if segment_vehicles > 100:  # Arbitrary threshold for "major" flow
                # Determine if inflow or outflow based on direction and position
                is_boundary = False
                start_lat, start_lon = coords[0]
                end_lat, end_lon = coords[-1]
                
                # Check if the segment starts or ends near the boundary
                boundary_margin = 0.001  # Approximately 100 meters
                if (abs(start_lat - BBOX["min_lat"]) < boundary_margin or 
                    abs(start_lat - BBOX["max_lat"]) < boundary_margin or
                    abs(start_lon - BBOX["min_lon"]) < boundary_margin or
                    abs(start_lon - BBOX["max_lon"]) < boundary_margin or
                    abs(end_lat - BBOX["min_lat"]) < boundary_margin or
                    abs(end_lat - BBOX["max_lat"]) < boundary_margin or
                    abs(end_lon - BBOX["min_lon"]) < boundary_margin or
                    abs(end_lon - BBOX["max_lon"]) < boundary_margin):
                    is_boundary = True
                
                if is_boundary:
                    # Determine if this is likely an inflow or outflow
                    # Inflows: moving toward center, outflows: moving toward boundary
                    center_lat = (BBOX["min_lat"] + BBOX["max_lat"]) / 2
                    center_lon = (BBOX["min_lon"] + BBOX["max_lon"]) / 2
                    
                    # Calculate distances from start/end to center
                    start_distance = haversine_distance(start_lat, start_lon, center_lat, center_lon)
                    end_distance = haversine_distance(end_lat, end_lon, center_lat, center_lon)
                    
                    # Moving toward center: potential inflow
                    if start_distance > end_distance:
                        processed_data["major_inflows"].append({
                            "segment_id": segment_record["id"],
                            "location": coords[0],
                            "vehicle_count": int(segment_vehicles),
                            "direction": segment_direction
                        })
                    # Moving away from center: potential outflow
                    else:
                        processed_data["major_outflows"].append({
                            "segment_id": segment_record["id"],
                            "location": coords[-1],
                            "vehicle_count": int(segment_vehicles),
                            "direction": segment_direction
                        })
            
            # Add segment record
            processed_data["segments"].append(segment_record)
            
            # Update totals
            processed_data["total_vehicles"] += segment_vehicles
            jam_factors.append(jam_factor)
            speeds.append(speed)
            if free_flow > 0:
                free_flow_speeds.append(free_flow)
            total_length += link_length
            processed_data["flow_records"] += 1
    
    # Calculate averages
    if jam_factors:
        processed_data["jam_factor"] = sum(jam_factors) / len(jam_factors)
    if speeds:
        processed_data["avg_speed"] = sum(speeds) / len(speeds)
    if free_flow_speeds:
        processed_data["avg_free_flow_speed"] = sum(free_flow_speeds) / len(free_flow_speeds)
        
    # Round total vehicles to integer
    processed_data["total_vehicles"] = int(processed_data["total_vehicles"])
    
    # Calculate average density
    if total_length > 0:
        processed_data["avg_vehicle_density"] = processed_data["total_vehicles"] / (total_length / 1000)  # vehicles per km
    else:
        processed_data["avg_vehicle_density"] = 0
    
    # Add total length
    processed_data["total_road_length"] = total_length
    processed_data["total_road_length_km"] = total_length / 1000
    
    # Limit major inflows/outflows to most significant ones (top 20)
    if len(processed_data["major_inflows"]) > 20:
        processed_data["major_inflows"] = sorted(
            processed_data["major_inflows"], 
            key=lambda x: x["vehicle_count"], 
            reverse=True
        )[:20]
    
    if len(processed_data["major_outflows"]) > 20:
        processed_data["major_outflows"] = sorted(
            processed_data["major_outflows"], 
            key=lambda x: x["vehicle_count"], 
            reverse=True
        )[:20]
    
    # Print summary
    print(f"Processed {processed_data['flow_records']} flow records for {period_name}.")
    print(f"Estimated total vehicles: {processed_data['total_vehicles']}")
    print(f"Average jam factor: {processed_data['jam_factor']:.2f}")
    print(f"Average speed: {processed_data['avg_speed']:.2f} km/h")
    print(f"Average vehicle density: {processed_data['avg_vehicle_density']:.2f} vehicles/km")
    print(f"Directional flows: {processed_data['directional_flows']}")
    print(f"Major inflows: {len(processed_data['major_inflows'])}")
    print(f"Major outflows: {len(processed_data['major_outflows'])}")
    
    return processed_data

def get_traffic_incidents():
    """Get historical traffic incidents from HERE Traffic Incidents API"""
    print("Requesting traffic incidents data from HERE Traffic Incidents API...")
    
    if not check_api_limits("traffic_incidents"):
        return None
    
    # Prepare bbox string - IMPORTANT: For HERE API, the format is west,south,east,north
    bbox_str = f"{BBOX['min_lon']},{BBOX['min_lat']},{BBOX['max_lon']},{BBOX['max_lat']}"
    print(f"Using bounding box for incidents: {bbox_str}")
    
    # HERE Traffic Incidents API endpoint
    incidents_url = "https://data.traffic.hereapi.com/v7/incidents"
    
    results = {}
    
    for period_name, period_info in TIME_PERIODS.items():
        # Generate timestamp for prediction
        prediction_time = generate_timestamp_for_prediction(
            day_of_week=2,  # Wednesday
            hour=period_info["hour"],
            minute=period_info["minute"]
        )
        
        output_file = f"traffic_data/traffic_incidents_{period_name}.json"
        
        # Skip if file exists and we're not forcing regeneration
        if os.path.exists(output_file) and not args.force_regenerate:
            print(f"Using existing traffic incidents data for {period_name} from {output_file}")
            with open(output_file, "r") as f:
                results[period_name] = json.load(f)
            continue
            
        # Try different parameter combinations
        attempts = [
            # Standard approach
            {
                "apiKey": HERE_API_KEY,
                "in": f"bbox:{bbox_str}",
                "locationReferencing": "shape",
                "language": "en-US",
                "incidentDetailsEndTimeSpan": "PT1H",  # Include incidents within the next hour
                "predictedIncidents": "true",  # Get predicted incidents
                "predictedIncidentsTime": prediction_time
            },
            # Try without predicted incidents
            {
                "apiKey": HERE_API_KEY,
                "in": f"bbox:{bbox_str}",
                "locationReferencing": "shape",
                "language": "en-US"
            },
            # Try with different location referencing
            {
                "apiKey": HERE_API_KEY,
                "in": f"bbox:{bbox_str}",
                "locationReferencing": "olr",
                "language": "en-US"
            }
        ]
        
        success = False
        for attempt_num, params in enumerate(attempts):
            try:
                print(f"Incident attempt {attempt_num+1} with params: {params}")
                response = requests.get(incidents_url, params=params)
                API_CALL_COUNTER["traffic_incidents"] += 1
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"Incident response data: {str(data)[:200]}...")  # Print first 200 chars for debugging
                    
                    # Check if we got any results
                    if 'results' in data and len(data.get('results', [])) > 0:
                        # Add metadata
                        data['metadata'] = {
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'period': period_name,
                            'prediction_time': prediction_time if 'predictedIncidents' in params else 'current',
                            'bbox': BBOX,
                            'attempt': attempt_num+1
                        }
                        
                        # Save raw incidents data
                        with open(output_file, "w") as f:
                            json.dump(data, f, indent=2)
                            
                        print(f"Traffic incidents data for {period_name} retrieved successfully with {len(data.get('results', []))} incidents.")
                        results[period_name] = data
                        
                        # Process into simplified format for SUMO
                        processed_file = f"traffic_data/processed_incidents_{period_name}.json"
                        processed_data = process_incidents_data(data, period_name)
                        
                        with open(processed_file, "w") as f:
                            json.dump(processed_data, f, indent=2)
                        
                        success = True
                        break  # Break out of attempts loop if successful
                    else:
                        print(f"Request succeeded but no incidents returned for attempt {attempt_num+1}")
                        # No incidents is okay - might be a quiet day!
                        success = True  # Consider this a success, just with zero incidents
                        
                        # Create empty data structure
                        empty_data = {
                            'results': [],
                            'metadata': {
                                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'period': period_name,
                                'bbox': BBOX,
                                'note': 'No incidents found in this area'
                            }
                        }
                        
                        # Save the empty data
                        with open(output_file, "w") as f:
                            json.dump(empty_data, f, indent=2)
                        
                        # Create empty processed data
                        empty_processed = {
                            "incidents": [],
                            "total_incidents": 0,
                            "incidents_by_type": {},
                            "incidents_by_severity": {}
                        }
                        
                        processed_file = f"traffic_data/processed_incidents_{period_name}.json"
                        with open(processed_file, "w") as f:
                            json.dump(empty_processed, f, indent=2)
                        
                        results[period_name] = empty_data
                        break
                else:
                    print(f"Failed to get incidents data (attempt {attempt_num+1}): {response.status_code}")
                    print(f"Response: {response.text}")
                    
            except Exception as e:
                print(f"Error in attempt {attempt_num+1}: {e}")
            
            # Sleep to avoid rate limiting
            time.sleep(2)
        
        if not success:
            print(f"ERROR: All attempts failed to get traffic incident data for {period_name}.")
            print("No valid data could be obtained from the HERE Traffic API.")
            print("Exiting script - fix API issues before continuing.")
            sys.exit(1)  # Exit with error
    
    return results

def process_incidents_data(data, period_name):
    """Process raw incidents data into a format suitable for SUMO"""
    print(f"Processing traffic incidents data for {period_name}...")
    
    processed_data = {
        "incidents": [],
        "total_incidents": 0,
        "incidents_by_type": {},
        "incidents_by_severity": {}
    }
    
    incidents = data.get("results", [])
    
    for incident in incidents:
        # Get incident details
        incident_type = incident.get("incidentType", "UNKNOWN")
        severity = incident.get("criticality", 0)
        
        # Get location
        location = incident.get("location", {})
        shape_data = location.get("shape", {})
        
        # Get affected road segments
        links = shape_data.get("links", [])
        
        if not links:
            continue
            
        # Process the first link for simplicity
        link = links[0]
        points = link.get("points", [])
        
        if len(points) < 2:
            continue
        
        # Extract coordinates
        coords = []
        for point in points:
            lat = point.get("lat")
            lng = point.get("lng")
            if lat is not None and lng is not None:
                coords.append((lat, lng))
        
        # Skip if insufficient coordinates
        if len(coords) < 2:
            continue
        
        # Create simplified incident record
        processed_incident = {
            "id": incident.get("incidentId", ""),
            "type": incident_type,
            "severity": severity,
            "description": incident.get("incidentDetails", {}).get("description", ""),
            "status": incident.get("incidentDetails", {}).get("status", ""),
            "start_time": incident.get("startTime", ""),
            "end_time": incident.get("endTime", ""),
            "coords": coords
        }
        
        # Add to processed data
        processed_data["incidents"].append(processed_incident)
        processed_data["total_incidents"] += 1
        
        # Count by type
        if incident_type not in processed_data["incidents_by_type"]:
            processed_data["incidents_by_type"][incident_type] = 0
        processed_data["incidents_by_type"][incident_type] += 1
        
        # Count by severity
        if severity not in processed_data["incidents_by_severity"]:
            processed_data["incidents_by_severity"][severity] = 0
        processed_data["incidents_by_severity"][severity] += 1
    
    print(f"Processed {processed_data['total_incidents']} incidents for {period_name}.")
    print(f"Incidents by type: {processed_data['incidents_by_type']}")
    print(f"Incidents by severity: {processed_data['incidents_by_severity']}")
    
    return processed_data

def visualize_traffic_data(period_name):
    """Create traffic data visualizations"""
    print(f"Creating traffic data visualizations for {period_name}...")
    
    # Check if the processed data exists
    flow_file = f"traffic_data/processed_traffic_flow_{period_name}.json"
    if not os.path.exists(flow_file):
        print(f"Error: Processed traffic flow data not found for {period_name}")
        return False
    
    # Load the data
    with open(flow_file, "r") as f:
        flow_data = json.load(f)
    
    # Check if there's any data to visualize
    segments = flow_data.get("segments", [])
    if not segments:
        print(f"Error: No traffic segments found in flow data for {period_name}")
        return False
    
    # Create visualization directory
    viz_dir = f"traffic_data/visualizations_{period_name}"
    os.makedirs(viz_dir, exist_ok=True)
    
    # Create a figure for the traffic flow map
    plt.figure(figsize=(15, 10))
    
    # Plot traffic flow map
    ax = plt.subplot(111)
    
    # Plot each road segment, colored by jam factor
    for segment in segments:
        coords = segment.get("coords", [])
        if len(coords) >= 2:
            # Extract lat/lon pairs
            lats = [coord[0] for coord in coords]
            lons = [coord[1] for coord in coords]
            
            # Color based on jam factor (0-10)
            jam_factor = segment.get("jam_factor", 0)
            color = plt.cm.RdYlGn_r(jam_factor / 10)  # Red for high jam, green for low
            
            # Line width based on vehicle density
            vehicle_density = segment.get("vehicle_density", 10)
            linewidth = min(3, max(0.5, vehicle_density / 50))
            
            # Plot the segment
            plt.plot(lons, lats, color=color, linewidth=linewidth, alpha=0.7)
    
    # Add major inflows and outflows
    for inflow in flow_data.get("major_inflows", []):
        location = inflow.get("location", [0, 0])
        if location:
            plt.plot(location[1], location[0], 'g^', markersize=8, alpha=0.8)
    
    for outflow in flow_data.get("major_outflows", []):
        location = outflow.get("location", [0, 0])
        if location:
            plt.plot(location[1], location[0], 'rv', markersize=8, alpha=0.8)
    
    # Set bounding box
    plt.xlim(BBOX["min_lon"], BBOX["max_lon"])
    plt.ylim(BBOX["min_lat"], BBOX["max_lat"])
    
    # Add a colorbar for the jam factor
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r, norm=plt.Normalize(0, 10))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Jam Factor')
    
    # Add title and labels
    plt.title(f'Traffic Flow Map - {period_name.replace("_", " ").title()}')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    # Add legend
    plt.plot([], [], 'g^', markersize=8, label='Major Inflows')
    plt.plot([], [], 'rv', markersize=8, label='Major Outflows')
    plt.legend()
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f"traffic_flow_map_{period_name}.png"), dpi=300)
    plt.close()
    
    # Create a bar chart for directional flow distribution
    plt.figure(figsize=(10, 6))
    directions = flow_data.get("directional_flows", {})
    
    # Sort by count
    sorted_directions = {k: v for k, v in sorted(directions.items(), key=lambda item: item[1], reverse=True)}
    
    # Create the bar chart
    plt.bar(sorted_directions.keys(), sorted_directions.values())
    plt.title(f'Directional Flow Distribution - {period_name.replace("_", " ").title()}')
    plt.xlabel('Direction')
    plt.ylabel('Number of Road Segments')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f"directional_flow_distribution_{period_name}.png"), dpi=300)
    plt.close()
    
    # Create a scatter plot for speed vs. jam factor
    plt.figure(figsize=(10, 6))
    jam_factors = [segment.get("jam_factor", 0) for segment in segments]
    speeds = [segment.get("speed", 0) for segment in segments]
    
    plt.scatter(jam_factors, speeds, alpha=0.5)
    plt.title(f'Speed vs. Jam Factor - {period_name.replace("_", " ").title()}')
    plt.xlabel('Jam Factor')
    plt.ylabel('Speed (km/h)')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(viz_dir, f"speed_vs_jam_{period_name}.png"), dpi=300)
    plt.close()
    
    # Create a histogram for speed distribution
    plt.figure(figsize=(10, 6))
    plt.hist(speeds, bins=20, alpha=0.7)
    plt.title(f'Speed Distribution - {period_name.replace("_", " ").title()}')
    plt.xlabel('Speed (km/h)')
    plt.ylabel('Number of Road Segments')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(viz_dir, f"speed_distribution_{period_name}.png"), dpi=300)
    plt.close()
    
    # Create a histogram for traffic density distribution
    densities = [segment.get("vehicle_density", 0) for segment in segments]
    plt.figure(figsize=(10, 6))
    plt.hist(densities, bins=20, alpha=0.7)
    plt.title(f'Traffic Density Distribution - {period_name.replace("_", " ").title()}')
    plt.xlabel('Vehicle Density (vehicles/km)')
    plt.ylabel('Number of Road Segments')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(viz_dir, f"density_distribution_{period_name}.png"), dpi=300)
    plt.close()
    
    print(f"Visualizations for {period_name} saved to {viz_dir}.")
    return True

def create_sumo_edge_mapping(flow_data, period_name):
    """
    Create a mapping between HERE traffic flow segments and potential SUMO edges
    This is a pre-processing step to help SUMO simulation use the flow data
    """
    print(f"Creating SUMO edge mapping file for {period_name}...")
    
    segments = flow_data.get("segments", [])
    if not segments:
        print(f"Error: No traffic segments found for {period_name}")
        return False
    
    # Create dictionary to store mapping
    mapping = {
        "segments": [],
        "metadata": {
            "period": period_name,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "total_segments": len(segments)
        }
    }
    
    # Process each segment
    for segment in segments:
        # Extract key information for mapping
        segment_mapping = {
            "id": segment.get("id"),
            "road_name": segment.get("road_name"),
            "direction": segment.get("direction"),
            "vehicle_count": segment.get("vehicle_count"),
            "vehicle_density": segment.get("vehicle_density"),
            "speed": segment.get("speed"),
            "jam_factor": segment.get("jam_factor"),
            # Include start/end coordinates for matching
            "start_point": segment.get("start_point"),
            "end_point": segment.get("end_point"),
            # Placeholder for SUMO edge IDs (to be filled by the simulation script)
            "sumo_edges": []
        }
        
        mapping["segments"].append(segment_mapping)
    
    # Save the mapping file
    output_file = f"traffic_data/sumo_edge_mapping_{period_name}.json"
    with open(output_file, "w") as f:
        json.dump(mapping, f, indent=2)
    
    print(f"SUMO edge mapping file created: {output_file}")
    return mapping

def main():
    global HERE_API_KEY, args
    
    # Parse command line arguments
    args = parse_arguments()
    HERE_API_KEY = args.api_key
    
    print("Starting HERE Traffic Data Collection...")
    print(f"Using HERE API key: {HERE_API_KEY[:5]}...{HERE_API_KEY[-5:]}")
    print(f"Manhattan bounding box: {BBOX}")
    
    # Create directory structure
    os.makedirs("traffic_data", exist_ok=True)
    
    # Step 1: Get traffic flow data
    flow_data = {}
    if not args.skip_flow:
        flow_data = get_traffic_flow_data()
        if not flow_data:
            print("Error: Failed to get traffic flow data.")
            sys.exit(1)
    else:
        print("Skipping Traffic Flow API calls.")
        # Try to load existing data
        for period_name in TIME_PERIODS.keys():
            processed_file = f"traffic_data/processed_traffic_flow_{period_name}.json"
            if os.path.exists(processed_file):
                with open(processed_file, "r") as f:
                    flow_data[period_name] = {"processed": json.load(f)}
                    print(f"Loaded existing processed flow data for {period_name}")
            else:
                print(f"Warning: No existing flow data found for {period_name}")
    
    # Step 2: Get traffic incidents
    incidents_data = {}
    if not args.skip_incidents:
        incidents_data = get_traffic_incidents()
        if not incidents_data:
            print("Warning: Failed to get traffic incidents data.")
    else:
        print("Skipping Traffic Incidents API calls.")
    
    # Step 3: Create SUMO edge mapping for each time period
    for period_name in TIME_PERIODS.keys():
        processed_file = f"traffic_data/processed_traffic_flow_{period_name}.json"
        if os.path.exists(processed_file):
            with open(processed_file, "r") as f:
                period_flow_data = json.load(f)
                create_sumo_edge_mapping(period_flow_data, period_name)
        else:
            print(f"Warning: No processed flow data found for {period_name}, cannot create mapping")
    
    # Step 4: Create visualizations if requested
    if args.visualize:
        for period_name in TIME_PERIODS.keys():
            processed_file = f"traffic_data/processed_traffic_flow_{period_name}.json"
            if os.path.exists(processed_file):
                visualize_traffic_data(period_name)
            else:
                print(f"Warning: No processed flow data found for {period_name}, cannot create visualizations")
    
    print("\n=== HERE Traffic Data Collection Complete ===")
    print(f"Data saved in: {os.path.join(WORKSPACE, 'traffic_data')}")
    print("\nAPI calls made:")
    print(f"  Traffic Flow API: {API_CALL_COUNTER['traffic_flow']} calls")
    print(f"  Traffic Incidents API: {API_CALL_COUNTER['traffic_incidents']} calls")
    print("\nGenerated files:")
    for period_name in TIME_PERIODS.keys():
        print(f"\n{period_name.replace('_', ' ').title()}:")
        flow_file = f"traffic_data/traffic_flow_{period_name}.json"
        processed_flow = f"traffic_data/processed_traffic_flow_{period_name}.json"
        incidents_file = f"traffic_data/traffic_incidents_{period_name}.json"
        processed_incidents = f"traffic_data/processed_incidents_{period_name}.json"
        mapping_file = f"traffic_data/sumo_edge_mapping_{period_name}.json"
        
        if os.path.exists(flow_file):
            print(f"  - {flow_file} (raw flow data)")
        if os.path.exists(processed_flow):
            print(f"  - {processed_flow} (processed flow data)")
        if os.path.exists(incidents_file):
            print(f"  - {incidents_file} (raw incidents)")
        if os.path.exists(processed_incidents):
            print(f"  - {processed_incidents} (processed incidents)")
        if os.path.exists(mapping_file):
            print(f"  - {mapping_file} (for SUMO integration)")
    
    print("\nTo use this data in SUMO simulation, run the manhattan_traffic_simulation.py script.")

if __name__ == "__main__":
    main()