#!/usr/bin/env python3
"""
Process HERE Traffic API Data with Incident Analysis

This script processes the traffic data from a specific timepoint,
filters out congestion caused by incidents, and generates entry/exit points
for SUMO simulation.
output files are (save_dir is currently traffic_data_8_938Fri, in /Users/samnu/Documents/SUMO/HERE_API_Manhattan196/)
Analysis data file: <save_dir>/analysis_<timestamp>.json

Contains the traffic analysis results including jam factors, speeds, vehicle counts, etc.


Top boundary points file: <save_dir>/top_boundary_points_<timestamp>.json

Contains information about the best entry/exit points at the boundaries of the area


Top congestion nodes file: <save_dir>/top_congestion_nodes_<timestamp>.json

Contains information about the most congested internal nodes in the traffic network


SUMO entry/exit points file: <save_dir>/sumo_entry_exit_points_<timestamp>.json

Contains the combined entry and exit points formatted for use in SUMO simulation


SUMO edge mapping file: <save_dir>/sumo_edge_mapping_<timestamp>.json

Maps the HERE API road segments to potential SUMO network edges


Visualization files (if --visualize flag is used):

<save_dir>/visualizations_<timestamp>/traffic_flow_map.png
<save_dir>/visualizations_<timestamp>/entry_exit_points.png
"""

import os
import sys
import json
import re
import glob
import matplotlib.pyplot as plt
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2

# Global variables
BBOX = {
    "min_lat": 40.712178,
    "min_lon": -74.033341,
    "max_lat": 40.759722,
    "max_lon": -73.958424
}

def parse_arguments():
    """Parse command line arguments"""
    import argparse
    parser = argparse.ArgumentParser(description='Process HERE Traffic API data with incident filtering')
    parser.add_argument('--data-dir', default='traffic_data_8_938Fri', help='Directory containing traffic data')
    parser.add_argument('--boundary-count', type=int, default=0, help='Number of top boundary points to use (0 for automatic)')
    parser.add_argument('--congestion-count', type=int, default=0, help='Number of top congested nodes to use (0 for automatic)')
    parser.add_argument('--density-factor', type=float, default=1.0, help='Factor to adjust point density (higher = more points)')
    parser.add_argument('--incident-buffer', type=float, default=0.0005, help='Buffer distance (degrees) around incidents to exclude from congestion points')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations of traffic data')
    return parser.parse_args()

def find_mid_timepoint_files(data_dir):
    """Find the middle timepoint files from a sequence of files, or closest matches if exact files don't exist"""
    print("\n==== Finding Mid-Timepoint Files ====")
    
    # Find all realtime and incident files
    realtime_files = glob.glob(os.path.join(data_dir, "realtime_*.json"))
    incident_files = glob.glob(os.path.join(data_dir, "incidents_*.json"))
    
    if not realtime_files:
        print(f"No realtime files found in {data_dir}")
        return None, None
    
    # Extract timestamps from filenames
    timestamp_pattern = r"(\d{8}_\d{6})"
    all_timestamps = []
    
    for filename in realtime_files:
        match = re.search(timestamp_pattern, filename)
        if match:
            timestamp_str = match.group(1)
            timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            all_timestamps.append((timestamp, timestamp_str, filename))
    
    # Sort timestamps
    all_timestamps.sort()
    
    # Find mid point
    if all_timestamps:
        middle_idx = len(all_timestamps) // 2
        mid_timestamp, mid_timestamp_str, mid_realtime_file = all_timestamps[middle_idx]
        
        print(f"Earliest file timestamp: {all_timestamps[0][0]}")
        print(f"Latest file timestamp: {all_timestamps[-1][0]}")
        print(f"Mid-timepoint: {mid_timestamp}")
        print(f"Mid-timepoint realtime file: {mid_realtime_file}")
        
        # Verify realtime file exists
        if not os.path.exists(mid_realtime_file):
            print(f"Error: Mid-timepoint realtime file not found: {mid_realtime_file}")
            return None, None
        
        # Find closest incident file to the mid timestamp
        closest_incident_file = None
        min_time_diff = float('inf')
        
        for incident_file in incident_files:
            match = re.search(timestamp_pattern, incident_file)
            if match:
                inc_timestamp_str = match.group(1)
                inc_timestamp = datetime.strptime(inc_timestamp_str, "%Y%m%d_%H%M%S")
                
                # Calculate time difference in seconds
                time_diff = abs((inc_timestamp - mid_timestamp).total_seconds())
                
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_incident_file = incident_file
        
        if closest_incident_file:
            print(f"Mid-timepoint incident file: {closest_incident_file}")
            # Verify file exists (should always be true since we found it)
            if not os.path.exists(closest_incident_file):
                print(f"Warning: Mid-timepoint incident file not found: {closest_incident_file}")
                closest_incident_file = None
        else:
            print(f"Warning: No incident files found to match mid-timepoint")
            closest_incident_file = None
        
        return mid_realtime_file, closest_incident_file
    
    return None, Noneestima

def load_json_file(filename):
    """Load a JSON file"""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points in meters"""
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    r = 6371000  # Radius of earth in meters
    return c * r

def calculate_segment_direction(points):
    """Calculate the overall direction of a road segment based on its shape"""
    if len(points) < 2:
        return "unknown"
    
    # Get start and end points
    start_lat = points[0].get("lat", 0)
    start_lon = points[0].get("lng", 0)
    end_lat = points[-1].get("lat", 0)
    end_lon = points[-1].get("lng", 0)
    
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

def analyze_traffic_data(traffic_data, incidents_data, timestamp_str, save_dir):
    """Analyze traffic data to extract key metrics"""
    results = traffic_data.get('results', [])
    
    if not results:
        print("No results to analyze")
        return None
    
    # Collect statistics
    jam_factors = []
    speeds = []
    free_flows = []
    total_length = 0  # Total road length in meters
    total_vehicles = 0  # Estimated total vehicles
    
    # Direction counts
    direction_counts = {
        "N-S": 0, "S-N": 0, "E-W": 0, "W-E": 0,
        "NE-SW": 0, "SW-NE": 0, "NW-SE": 0, "SE-NW": 0
    }
    
    # Road segment counts by functional class
    fc_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    
    # Process incidents data if available
    incidents_processed = {
        "incidents": [],
        "total_incidents": 0,
        "incidents_by_type": {},
        "incidents_by_severity": {}
    }
    
    # Extract incident locations for later filtering
    incident_locations = []
    
    if incidents_data and 'results' in incidents_data:
        incidents = incidents_data.get('results', [])
        
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
                    # Add to incident locations list
                    incident_locations.append((lat, lng, severity))
            
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
            incidents_processed["incidents"].append(processed_incident)
            incidents_processed["total_incidents"] += 1
            
            # Count by type
            if incident_type not in incidents_processed["incidents_by_type"]:
                incidents_processed["incidents_by_type"][incident_type] = 0
            incidents_processed["incidents_by_type"][incident_type] += 1
            
            # Count by severity
            if severity not in incidents_processed["incidents_by_severity"]:
                incidents_processed["incidents_by_severity"][severity] = 0
            incidents_processed["incidents_by_severity"][severity] += 1
    
    for item in results:
        # Get traffic flow data
        current_flow = item.get("currentFlow", {})
        
        if current_flow:
            # Get metrics
            jam_factor = current_flow.get("jamFactor", 0)
            jam_factors.append(jam_factor)
            
            speed = current_flow.get("speed", 0)
            speeds.append(speed)
            
            free_flow = current_flow.get("freeFlow", 0)
            if isinstance(free_flow, (int, float)):
                free_flows.append(free_flow)
        
        # Get road information
        location = item.get("location", {})
        
        # Handle shape data for directional analysis
        shape_data = location.get("shape", {})
        links = shape_data.get("links", [])
        
        for link in links:
            # Get functional class if available
            if "functionalClass" in link:
                fc = link.get("functionalClass")
                if fc in fc_counts:
                    fc_counts[fc] += 1
            
            # Get points to determine direction
            points = link.get("points", [])
            if len(points) >= 2:
                # Calculate direction
                direction = calculate_segment_direction(points)
                if direction in direction_counts:
                    direction_counts[direction] += 1
                
                # Calculate segment length if not provided
                link_length = link.get("length", 0)
                if link_length <= 0:
                    # Estimate length using Haversine distance for all segments
                    link_length = 0
                    for i in range(len(points) - 1):
                        lat1 = points[i].get("lat", 0)
                        lon1 = points[i].get("lng", 0)
                        lat2 = points[i+1].get("lat", 0)
                        lon2 = points[i+1].get("lng", 0)
                        segment_length = haversine_distance(lat1, lon1, lat2, lon2)
                        link_length += segment_length
                
                total_length += link_length
                
                # Estimate vehicle count based on jam factor and length
                if jam_factor < 1:
                    vehicles_per_km = 5 + jam_factor * 5  # 5-10 vehicles/km for free flow
                elif jam_factor < 3:
                    vehicles_per_km = 10 + (jam_factor - 1) * 7  # 10-24 vehicles/km for light congestion
                elif jam_factor < 6:
                    vehicles_per_km = 24 + (jam_factor - 3) * 6  # 24-42 vehicles/km for moderate congestion
                elif jam_factor < 8:
                    vehicles_per_km = 42 + (jam_factor - 6) * 9  # 42-60 vehicles/km for heavy congestion
                else:
                    vehicles_per_km = 60 + (jam_factor - 8) * 10  # 60-80 vehicles/km for extreme congestion
                
                segment_vehicles = int(vehicles_per_km * (link_length / 1000))
                total_vehicles += segment_vehicles
    
    # Calculate average statistics
    avg_jam_factor = sum(jam_factors) / len(jam_factors) if jam_factors else 0
    avg_speed = sum(speeds) / len(speeds) if speeds else 0
    avg_free_flow = sum(free_flows) / len(free_flows) if free_flows else 0
    
    # Calculate vehicle density
    avg_vehicle_density = total_vehicles / (total_length / 1000) if total_length > 0 else 0
    
    # Print summary
    scenario_name = f"analysis_{timestamp_str}"
    print(f"\n=== Traffic Analysis for {scenario_name} ===")
    print(f"Total road segments: {len(results)}")
    print(f"Total road length: {total_length:.2f} meters ({total_length/1000:.2f} km)")
    print(f"Estimated total vehicles: {total_vehicles}")
    print(f"Average vehicle density: {avg_vehicle_density:.2f} vehicles/km")
    print(f"Average jam factor: {avg_jam_factor:.2f}")
    print(f"Average speed: {avg_speed:.2f} km/h")
    print(f"Average free flow speed: {avg_free_flow:.2f} km/h")
    ratio_display = f"{(avg_speed/avg_free_flow):.2f}" if avg_free_flow > 0 else "N/A"
    print(f"Speed ratio (speed/free flow): {ratio_display}")    
    print("\nDirectional distribution:")
    for direction, count in direction_counts.items():
        print(f"  {direction}: {count}")
    
    print("\nFunctional class distribution:")
    for fc, count in fc_counts.items():
        print(f"  Class {fc}: {count}")
    
    # Distribution analysis
    if jam_factors:
        print("\nJam factor distribution:")
        jam_ranges = {"0-2": 0, "2-4": 0, "4-6": 0, "6-8": 0, "8-10": 0}
        for jf in jam_factors:
            if jf < 2: jam_ranges["0-2"] += 1
            elif jf < 4: jam_ranges["2-4"] += 1
            elif jf < 6: jam_ranges["4-6"] += 1
            elif jf < 8: jam_ranges["6-8"] += 1
            else: jam_ranges["8-10"] += 1
        
        for range_name, count in jam_ranges.items():
            percentage = (count / len(jam_factors)) * 100
            print(f"  {range_name}: {count} ({percentage:.1f}%)")
    
    # Print incidents summary if available
    if incidents_processed["total_incidents"] > 0:
        print(f"\nTraffic Incidents Summary:")
        print(f"Total incidents: {incidents_processed['total_incidents']}")
        print("Incidents by type:")
        for inc_type, count in incidents_processed["incidents_by_type"].items():
            print(f"  {inc_type}: {count}")
        print("Incidents by severity:")
        for severity, count in incidents_processed["incidents_by_severity"].items():
            print(f"  {severity}: {count}")
    
    # Save the analysis to a separate file
    analysis = {
        "scenario": scenario_name,
        "segment_count": len(results),
        "total_road_length": total_length,
        "total_road_length_km": total_length / 1000,
        "total_vehicles": total_vehicles,
        "avg_vehicle_density": avg_vehicle_density,
        "avg_jam_factor": avg_jam_factor,
        "avg_speed": avg_speed,
        "avg_free_flow": avg_free_flow,
        "speed_ratio": avg_speed/avg_free_flow if avg_free_flow > 0 else 0,
        "direction_counts": direction_counts,
        "functional_class_counts": fc_counts,
        "incidents": incidents_processed,
        "incident_locations": incident_locations,
        "timestamp": datetime.now().isoformat()
    }
    
    filename = f"{save_dir}/{scenario_name}.json"
    with open(filename, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"Saved analysis to {filename}")
    
    return analysis

def calculate_dynamic_point_counts(analysis_data, bbox, density_factor=1.0):
    """Calculate appropriate number of entry/exit points based on area and road density"""
    # Calculate bounding box area in square kilometers
    width_meters = haversine_distance(bbox["min_lat"], bbox["min_lon"], bbox["min_lat"], bbox["max_lon"])
    height_meters = haversine_distance(bbox["min_lat"], bbox["min_lon"], bbox["max_lat"], bbox["min_lon"])
    area_km2 = (width_meters * height_meters) / 1000000
    
    # Get road density from analysis data
    road_length_km = analysis_data.get("total_road_length_km", 0)
    segment_count = analysis_data.get("segment_count", 0)
    
    # Calculate road density (km/km²)
    road_density = road_length_km / area_km2 if area_km2 > 0 else 0
    
    # Calculate segment density (segments/km²)
    segment_density = segment_count / area_km2 if area_km2 > 0 else 0
    
    # Estimate appropriate counts based on area and density
    # Increased from 3 to 4 to get more points per area
    area_factor = max(1, int(area_km2 * 4))
    
    # Adjust based on road density relative to typical urban areas
    # Typical dense urban road density: ~20-40 km/km²
    calculated_density_factor = max(0.8, min(3.0, road_density / 20))
    
    # Apply user-defined density factor
    # Final counts (with minimums)
    boundary_count = max(16, int(area_factor * calculated_density_factor * density_factor))
    congestion_count = max(16, int(area_factor * calculated_density_factor * density_factor))
    
    print(f"\n=== Dynamic Point Count Calculation ===")
    print(f"Bounding box area: {area_km2:.2f} km²")
    print(f"Road length: {road_length_km:.2f} km")
    print(f"Road density: {road_density:.2f} km/km²")
    print(f"Segment count: {segment_count}")
    print(f"Segment density: {segment_density:.2f} segments/km²")
    print(f"Area factor: {area_factor}")
    print(f"Calculated density factor: {calculated_density_factor:.2f}")
    print(f"User density factor: {density_factor:.2f}")
    print(f"Calculated boundary point count: {boundary_count}")
    print(f"Calculated congestion node count: {congestion_count}")
    
    return boundary_count, congestion_count

def extract_timestamp_from_filename(filename):
    """Extract timestamp from a filename"""
    match = re.search(r"(\d{8}_\d{6})", filename)
    if match:
        return match.group(1)
    return None


            
            

def find_boundary_points(traffic_data, incidents_data, incident_buffer, timestamp_str, save_dir, top_count=30):
    """Find top boundary points with improved outflow detection and incident filtering"""
    results = traffic_data.get('results', [])
    
    if not results:
        print("No results to analyze for boundary points")
        return []
    
    print(f"\n=== Finding Top {top_count} Boundary Points with Enhanced Direction Analysis ===")
    
    # Define boundary margins
    boundary_margin = 0.001  # Approximately 100 meters
    
    # Center point of the bounding box
    center_lat = (BBOX["min_lat"] + BBOX["max_lat"]) / 2
    center_lon = (BBOX["min_lon"] + BBOX["max_lon"]) / 2
    
    # Store all boundary points
    boundary_points = []
    
    # Extract incident locations for filtering
    incident_locations = []
    if incidents_data and 'results' in incidents_data:
        for incident in incidents_data.get('results', []):
            location = incident.get("location", {})
            shape_data = location.get("shape", {})
            links = shape_data.get("links", [])
            
            if not links:
                continue
            
            # Process each link to get all affected points
            for link in links:
                points = link.get("points", [])
                if not points:
                    continue
                
                # Get all points to create a buffer around the incident
                for point in points:
                    lat = point.get("lat")
                    lng = point.get("lng")
                    if lat is not None and lng is not None:
                        # Also store severity for weighted filtering
                        severity = incident.get("criticality", 0)
                        incident_locations.append((lat, lng, severity))
    
    print(f"Found {len(incident_locations)} incident locations for filtering")
    
    for item_idx, item in enumerate(results):
        # Get location
        location = item.get("location", {})
        road_name = location.get("description", "Unknown Road")
        
        # Skip if no shape data
        shape_data = location.get("shape", {})
        links = shape_data.get("links", [])
        
        if not links:
            continue
        
        # Get traffic flow data
        current_flow = item.get("currentFlow", {})
        jam_factor = current_flow.get("jamFactor", 0) if current_flow else 0
        speed = current_flow.get("speed", 0) if current_flow else 0
        
        # Check each link for boundary intersections
        for link_idx, link in enumerate(links):
            points = link.get("points", [])
            if len(points) < 2:
                continue
            
            # Get start and end points
            start_lat = points[0].get("lat", 0)
            start_lon = points[0].get("lng", 0)
            end_lat = points[-1].get("lat", 0)
            end_lon = points[-1].get("lng", 0)
            
            # Check if near boundary
            is_boundary = False
            boundary_type = ""
            boundary_side = ""
            
            # Check each boundary edge
            if abs(start_lat - BBOX["min_lat"]) < boundary_margin:
                is_boundary = True
                boundary_type = "south_start"
                boundary_side = "south"
            elif abs(start_lat - BBOX["max_lat"]) < boundary_margin:
                is_boundary = True
                boundary_type = "north_start"
                boundary_side = "north"
            elif abs(start_lon - BBOX["min_lon"]) < boundary_margin:
                is_boundary = True
                boundary_type = "west_start"
                boundary_side = "west"
            elif abs(start_lon - BBOX["max_lon"]) < boundary_margin:
                is_boundary = True
                boundary_type = "east_start"
                boundary_side = "east"
            elif abs(end_lat - BBOX["min_lat"]) < boundary_margin:
                is_boundary = True
                boundary_type = "south_end"
                boundary_side = "south"
            elif abs(end_lat - BBOX["max_lat"]) < boundary_margin:
                is_boundary = True
                boundary_type = "north_end"
                boundary_side = "north"
            elif abs(end_lon - BBOX["min_lon"]) < boundary_margin:
                is_boundary = True
                boundary_type = "west_end"
                boundary_side = "west"
            elif abs(end_lon - BBOX["max_lon"]) < boundary_margin:
                is_boundary = True
                boundary_type = "east_end"
                boundary_side = "east"
            
            if not is_boundary:
                continue  # Skip if not a boundary segment
            
            # Check if this boundary point is near an incident
            # We'll check both start and end points
            near_incident = False
            for inc_lat, inc_lon, inc_severity in incident_locations:
                # Check distance from each end of the segment to the incident
                start_dist = haversine_distance(start_lat, start_lon, inc_lat, inc_lon)
                end_dist = haversine_distance(end_lat, end_lon, inc_lat, inc_lon)
                
                # Adjust buffer based on incident severity
                # Higher severity means larger buffer
                adjusted_buffer = incident_buffer * (1 + inc_severity * 0.5)
                
                # Convert buffer from degrees to meters (approximate)
                buffer_meters = adjusted_buffer * 111000  # 1 degree ≈ 111 km
                
                if start_dist < buffer_meters or end_dist < buffer_meters:
                    near_incident = True
                    break
            
            if near_incident:
                # Skip this boundary point as it's near an incident
                continue
            
            # Estimate vehicle count based on length and jam factor
            link_length = link.get("length", 0)
            if link_length <= 0:
                # Estimate length using Haversine distance
                link_length = haversine_distance(start_lat, start_lon, end_lat, end_lon)
            
            # Estimate vehicle count based on jam factor and length
            if jam_factor < 2:
                vehicles_per_km = 10 + jam_factor * 5
            elif jam_factor < 5:
                vehicles_per_km = 20 + (jam_factor - 2) * 10
            elif jam_factor < 8:
                vehicles_per_km = 50 + (jam_factor - 5) * 16.7
            else:
                vehicles_per_km = 100 + (jam_factor - 8) * 33.3
            
            segment_vehicles = int(vehicles_per_km * (link_length / 1000))
            
            # Calculate distances from start/end to center
            start_distance = haversine_distance(start_lat, start_lon, center_lat, center_lon)
            end_distance = haversine_distance(end_lat, end_lon, center_lat, center_lon)
            
            # Get direction from road shape
            direction = calculate_segment_direction(points)
            
            # Enhanced direction analysis - incorporate actual direction
            # This better handles roads that angle toward center but are actually outflows
            
            # Determine if segment is parallel to boundary
            is_parallel_to_boundary = False
            if boundary_side in ["north", "south"] and direction in ["E-W", "W-E"]:
                is_parallel_to_boundary = True
            elif boundary_side in ["east", "west"] and direction in ["N-S", "S-N"]:
                is_parallel_to_boundary = True
            
            # Check for one-way streets and major arterials
            is_one_way = False
            one_way_keywords = [
                "fdr", "west side", "avenue", "expressway", "parkway", 
                "one-way", "tunnel", "bridge"
            ]
            
            for keyword in one_way_keywords:
                if keyword.lower() in road_name.lower():
                    is_one_way = True
                    break
            
            # For parallel roads, use explicit direction instead of center distance
            if is_parallel_to_boundary:
                # Use explicit direction to determine flow type
                if direction in ["E-W", "N-S", "NE-SW", "NW-SE"]:
                    flow_type = "outflow"  # These are likely outflow directions for parallel roads
                else:
                    flow_type = "inflow"   # These are likely inflow directions
            elif is_one_way:
                # For one-way roads, prioritize the explicit road direction
                if boundary_type.endswith("_start"):
                    # Road starts at boundary - likely inflow
                    flow_type = "inflow"
                    location_point = (start_lat, start_lon)
                else:
                    # Road ends at boundary - likely outflow
                    flow_type = "outflow"
                    location_point = (end_lat, end_lon)
            else:
                # Default to center distance comparison for regular roads
                if start_distance > end_distance:
                    flow_type = "inflow"  # Moving toward center
                    location_point = (start_lat, start_lon)
                else:
                    flow_type = "outflow"  # Moving away from center
                    location_point = (end_lat, end_lon)
            
            # Create record
            boundary_record = {
                "type": flow_type,
                "location": location_point,
                "vehicle_count": segment_vehicles,
                "direction": direction,
                "boundary_type": boundary_type,
                "boundary_side": boundary_side,
                "is_parallel": is_parallel_to_boundary,
                "is_one_way": is_one_way,
                "jam_factor": jam_factor,
                "speed": speed,
                "length": link_length,
                "road_name": road_name,
                "result_index": item_idx,
                "link_index": link_idx,
                "near_incident": near_incident
            }
            
            boundary_points.append(boundary_record)
    
    # Get inflows and outflows
    inflows = [point for point in boundary_points if point["type"] == "inflow"]
    outflows = [point for point in boundary_points if point["type"] == "outflow"]
    
    # Sort by vehicle count
    sorted_inflows = sorted(inflows, key=lambda x: x["vehicle_count"], reverse=True)
    sorted_outflows = sorted(outflows, key=lambda x: x["vehicle_count"], reverse=True)
    
    # Balance selection while preserving realistic traffic patterns
    top_inflows = sorted_inflows[:min(top_count//2 + top_count//4, len(sorted_inflows))]
    top_outflows = sorted_outflows[:min(top_count//4, len(sorted_outflows))]
    
    # Combine for final result - preserving natural proportions
    top_boundary = top_inflows + top_outflows
    
    print(f"\nFound {len(boundary_points)} total boundary points after filtering out incident-related congestion")
    print(f"Inflows: {len(inflows)}, Outflows: {len(outflows)}")
    print(f"Selected {len(top_inflows)} inflows and {len(top_outflows)} outflows based on realistic traffic patterns")
    
    print("\nTop inflow points:")
    for i, point in enumerate(top_inflows[:5]):
        print(f"  {i+1}. {point['road_name']} ({point['vehicle_count']} vehicles, {point['direction']})")
    
    print("\nTop outflow points:")
    for i, point in enumerate(top_outflows[:5]):
        print(f"  {i+1}. {point['road_name']} ({point['vehicle_count']} vehicles, {point['direction']})")
    
    # Add detailed boundary analysis
    boundary_distribution = {
        'north': 0, 'south': 0, 'east': 0, 'west': 0,
        'corners': 0, 'interior': 0
    }

    for point in boundary_points:
        location = point.get('location', [0, 0])
        lat, lon = location[0], location[1]
        
        # Classify boundary position
        margin = 0.002  # About 200m margin
        
        is_north = abs(lat - BBOX["max_lat"]) < margin
        is_south = abs(lat - BBOX["min_lat"]) < margin  
        is_east = abs(lon - BBOX["max_lon"]) < margin
        is_west = abs(lon - BBOX["min_lon"]) < margin
        
        corner_count = sum([is_north, is_south, is_east, is_west])
        
        if corner_count >= 2:
            boundary_distribution['corners'] += 1
        elif is_north:
            boundary_distribution['north'] += 1
        elif is_south:
            boundary_distribution['south'] += 1
        elif is_east:
            boundary_distribution['east'] += 1
        elif is_west:
            boundary_distribution['west'] += 1
        else:
            boundary_distribution['interior'] += 1
    
    print(f"Boundary point distribution: {boundary_distribution}")

    # Check for clustering issues
    if (boundary_distribution['north'] + boundary_distribution['south'] < 
        boundary_distribution['east'] + boundary_distribution['west']):
        print("WARNING: More east/west entry points than north/south - may indicate clustering")


    # Save analysis
    boundary_analysis = {
        "total_boundary_points": len(boundary_points),
        "inflows": len(inflows),
        "outflows": len(outflows),
        "top_inflows": top_inflows,
        "top_outflows": top_outflows
    }
    
    filename = f"{save_dir}/top_boundary_points_{timestamp_str}.json"
    with open(filename, "w") as f:
        json.dump(boundary_analysis, f, indent=2)
    print(f"Saved top boundary points to {filename}")
    
    return top_boundary

def find_congestion_nodes(traffic_data, incidents_data, incident_buffer, timestamp_str, save_dir, top_count=30):
    """Find most congested nodes with directional analysis and incident filtering"""
    results = traffic_data.get('results', [])
    
    if not results:
        print("No results to analyze for congestion nodes")
        return []
    
    print(f"\n=== Finding Top {top_count} Congested Nodes with Incident Filtering ===")
    
    # Store potential congestion nodes
    congestion_nodes = []
    
    # Extract incident locations for filtering
    incident_locations = []
    if incidents_data and 'results' in incidents_data:
        for incident in incidents_data.get('results', []):
            location = incident.get("location", {})
            shape_data = location.get("shape", {})
            links = shape_data.get("links", [])
            
            if not links:
                continue
            
            # Process each link to get all affected points
            for link in links:
                points = link.get("points", [])
                if not points:
                    continue
                
                # Get all points to create a buffer around the incident
                for point in points:
                    lat = point.get("lat")
                    lng = point.get("lng")
                    if lat is not None and lng is not None:
                        # Also store severity for weighted filtering
                        severity = incident.get("criticality", 0)
                        incident_locations.append((lat, lng, severity))
    
    print(f"Found {len(incident_locations)} incident locations for filtering")
    
    for item_idx, item in enumerate(results):
        # Get location
        location = item.get("location", {})
        road_name = location.get("description", "Unknown Road")
        
        # Skip if no shape data
        shape_data = location.get("shape", {})
        links = shape_data.get("links", [])
        
        if not links:
            continue
        
        # Get traffic flow data
        current_flow = item.get("currentFlow", {})
        if not current_flow:
            continue
            
        jam_factor = current_flow.get("jamFactor", 0)
        speed = current_flow.get("speed", 0)
        free_flow = current_flow.get("freeFlow", 0)
        
        # Skip very low jam factors (uncongested areas)
        if jam_factor < 1.0:
            continue
        
        # Process links to find high congestion points
        for link_idx, link in enumerate(links):
            points = link.get("points", [])
            if len(points) < 2:
                continue
            
            # Use middle point for congestion node
            mid_idx = len(points) // 2
            mid_lat = points[mid_idx].get("lat", 0)
            mid_lon = points[mid_idx].get("lng", 0)
            
            # Check if this congestion node is near an incident
            near_incident = False
            incident_info = None
            
            for inc_lat, inc_lon, inc_severity in incident_locations:
                # Calculate distance from congestion point to incident
                distance = haversine_distance(mid_lat, mid_lon, inc_lat, inc_lon)
                
                # Adjust buffer based on incident severity
                # Higher severity means larger buffer
                adjusted_buffer = incident_buffer * (1 + inc_severity * 0.5)
                
                # Convert buffer from degrees to meters (approximate)
                buffer_meters = adjusted_buffer * 111000  # 1 degree ≈ 111 km
                
                if distance < buffer_meters:
                    near_incident = True
                    incident_info = {
                        "distance": distance,
                        "severity": inc_severity,
                        "location": (inc_lat, inc_lon)
                    }
                    break
            
            # If this congestion is likely caused by an incident, skip it
            if near_incident:
                continue
            
            # Get functional class if available (lower is more important)
            functional_class = link.get("functionalClass", 5)
            
            # Estimate link length
            link_length = link.get("length", 0)
            if link_length <= 0 and len(points) >= 2:
                # Estimate length
                total_length = 0
                for i in range(len(points) - 1):
                    lat1 = points[i].get("lat", 0)
                    lon1 = points[i].get("lng", 0)
                    lat2 = points[i+1].get("lat", 0)
                    lon2 = points[i+1].get("lng", 0)
                    segment_length = haversine_distance(lat1, lon1, lat2, lon2)
                    total_length += segment_length
                link_length = total_length
            
            # Estimate vehicle count based on jam factor and length
            if jam_factor < 2:
                vehicles_per_km = 10 + jam_factor * 5
            elif jam_factor < 5:
                vehicles_per_km = 20 + (jam_factor - 2) * 10
            elif jam_factor < 8:
                vehicles_per_km = 50 + (jam_factor - 5) * 16.7
            else:
                vehicles_per_km = 100 + (jam_factor - 8) * 33.3
            
            segment_vehicles = int(vehicles_per_km * (link_length / 1000))
            
            # Get traffic direction based on actual road shape
            direction = calculate_segment_direction(points)
            
            # Calculate congestion score - combination of jam factor, vehicle count, and road importance
            # Higher is more congested
            congestion_score = (
                jam_factor * 10 +                         # Weight jam factor heavily
                segment_vehicles +                        # Add vehicle count
                (6 - functional_class) * 5 +              # More important roads score higher
                (speed < 10) * 20                         # Very slow speeds boost score
            )
            
            # Only consider significant congestion
            if congestion_score < 20:  # Arbitrary threshold
                continue
            
            # Determine if this congested node should be primarily entry or exit
            # For one-way streets and tunnels, respect the direction
            
            # Get start and end points
            start_lat = points[0].get("lat", 0)
            start_lon = points[0].get("lng", 0)
            end_lat = points[-1].get("lat", 0)
            end_lon = points[-1].get("lng", 0)
            
            # Calculate center point of the bounding box
            center_lat = (BBOX["min_lat"] + BBOX["max_lat"]) / 2
            center_lon = (BBOX["min_lon"] + BBOX["max_lon"]) / 2
            
            # Calculate distances from start/end to center
            start_distance = haversine_distance(start_lat, start_lon, center_lat, center_lon)
            end_distance = haversine_distance(end_lat, end_lon, center_lat, center_lon)
            
            # Check for one-way tunnels, bridges, or major arteries
            is_one_way = False
            is_tunnel_or_bridge = False
            
            # Keywords suggesting one-way, tunnel, bridge
            tunnel_bridge_keywords = [
                "tunnel", "bridge", "expressway", "thruway", "one-way", "highway", 
                "queens midtown", "lincoln", "brooklyn", "williamsburg", "manhattan", 
                "fdr", "west side highway", "henry hudson"
            ]
            
            # Check if road name contains any of the keywords
            for keyword in tunnel_bridge_keywords:
                if keyword.lower() in road_name.lower():
                    is_tunnel_or_bridge = True
                    break
            
            # Assume congestion differential indicates direction
            speed_ratio = speed / free_flow if free_flow > 0 else 0
            
            # Determine node type (entry or exit)
            # Default behavior: highly congested -> primarily exit, moderately congested -> entry
            if is_tunnel_or_bridge:
                # For tunnels and bridges, use direction relative to center to determine type
                if start_distance > end_distance:
                    node_type = "exit"  # Traffic coming into area = exit for simulation
                else:
                    node_type = "entry"  # Traffic going out of area = entry for simulation
            else:
                # For regular roads, use congestion and directional analysis
                # Very low speed ratio indicates congestion that would be an exit point
                if speed_ratio < 0.4 and jam_factor > 5:
                    node_type = "exit"  # High congestion, likely accumulation point
                else:
                    node_type = "entry"  # Moderate congestion, likely source point
            
            # Create record
            node_record = {
                "location": (mid_lat, mid_lon),
                "congestion_score": congestion_score,
                "jam_factor": jam_factor,
                "speed": speed,
                "speed_ratio": speed_ratio,
                "vehicle_count": segment_vehicles,
                "functional_class": functional_class,
                "direction": direction,
                "road_name": road_name,
                "length": link_length,
                "node_type": node_type,
                "is_tunnel_or_bridge": is_tunnel_or_bridge,
                "result_index": item_idx,
                "link_index": link_idx,
                "near_incident": near_incident,
                "incident_info": incident_info
            }
            
            congestion_nodes.append(node_record)
    
    # Sort by congestion score
    sorted_nodes = sorted(congestion_nodes, key=lambda x: x["congestion_score"], reverse=True)
    
    # Get balanced entry/exit points
    entry_nodes = [node for node in sorted_nodes if node["node_type"] == "entry"]
    exit_nodes = [node for node in sorted_nodes if node["node_type"] == "exit"]
    
    # Preserve actual proportions of entry vs exit nodes
    entry_ratio = len(entry_nodes) / (len(entry_nodes) + len(exit_nodes)) if (len(entry_nodes) + len(exit_nodes)) > 0 else 0.5
    entry_count = min(int(top_count * entry_ratio), len(entry_nodes))
    exit_count = min(top_count - entry_count, len(exit_nodes))

    # Ensure we have at least one of each if available
    if entry_count == 0 and len(entry_nodes) > 0:
        entry_count = 1
        exit_count = min(top_count - 1, len(exit_nodes))
    elif exit_count == 0 and len(exit_nodes) > 0:
        exit_count = 1
        entry_count = min(top_count - 1, len(entry_nodes))

    top_entries = entry_nodes[:entry_count]
    top_exits = exit_nodes[:exit_count]
    
    # Combine for final list
    top_nodes = top_entries + top_exits
    
    print(f"\nFound {len(congestion_nodes)} total congestion nodes after filtering out incident-related congestion")
    print(f"Entry nodes: {len(entry_nodes)}, Exit nodes: {len(exit_nodes)}")
    print(f"Selected {len(top_entries)} entry and {len(top_exits)} exit nodes based on real-world proportions")
    
    print("\nTop entry nodes:")
    for i, node in enumerate(top_entries[:5]):
        print(f"  {i+1}. {node['road_name']} (Score: {node['congestion_score']:.1f}, Jam: {node['jam_factor']:.1f})")
    
    print("\nTop exit nodes:")
    for i, node in enumerate(top_exits[:5]):
        print(f"  {i+1}. {node['road_name']} (Score: {node['congestion_score']:.1f}, Jam: {node['jam_factor']:.1f})")
    
    # Save analysis
    congestion_analysis = {
        "total_congestion_nodes": len(congestion_nodes),
        "entry_nodes": len(entry_nodes),
        "exit_nodes": len(exit_nodes),
        "selected_entry_nodes": len(top_entries),
        "selected_exit_nodes": len(top_exits),
        "entry_ratio": entry_ratio,
        "top_entry_nodes": top_entries,
        "top_exit_nodes": top_exits
    }
    
    filename = f"{save_dir}/top_congestion_nodes_{timestamp_str}.json"
    with open(filename, "w") as f:
        json.dump(congestion_analysis, f, indent=2)
    print(f"Saved top congestion nodes to {filename}")
    
    return top_nodes

def deduplicate_points(points, min_distance_meters=500):
    """Remove points that are too close together to improve route diversity"""
    if not points:
        return points
    
    # Sort by weight (descending) to keep highest-weight points
    sorted_points = sorted(points, key=lambda x: x.get('weight', 0), reverse=True)
    
    deduplicated = []
    for point in sorted_points:
        location = point.get('location', [0, 0])
        if len(location) < 2:
            continue
            
        lat, lon = location[0], location[1]
        
        # Check distance to all already selected points
        too_close = False
        for existing in deduplicated:
            existing_loc = existing.get('location', [0, 0])
            if len(existing_loc) >= 2:
                existing_lat, existing_lon = existing_loc[0], existing_loc[1]
                distance = haversine_distance(lat, lon, existing_lat, existing_lon)
                
                if distance < min_distance_meters:
                    too_close = True
                    break
        
        if not too_close:
            deduplicated.append(point)
    
    return deduplicated

def generate_sumo_points(boundary_points, congestion_nodes, timestamp_str, save_dir):
    """Generate authentic entry/exit points for SUMO simulation"""
    print("\n=== Generating SUMO Entry/Exit Points ===")
    
    # Extract boundary inflows and outflows
    boundary_inflows = [point for point in boundary_points if point["type"] == "inflow"]
    boundary_outflows = [point for point in boundary_points if point["type"] == "outflow"]
    
    # Extract congestion entries and exits
    congestion_entries = [node for node in congestion_nodes if node.get("node_type") == "entry"]
    congestion_exits = [node for node in congestion_nodes if node.get("node_type") == "exit"]
    
    # DEDUPLICATION STEP 
    print("Deduplicating boundary points...")
    boundary_inflows = deduplicate_points(boundary_inflows, min_distance_meters=500)
    boundary_outflows = deduplicate_points(boundary_outflows, min_distance_meters=500)
    
    print("Deduplicating congestion points...")  
    congestion_entries = deduplicate_points(congestion_entries, min_distance_meters=400)
    congestion_exits = deduplicate_points(congestion_exits, min_distance_meters=400)
    
    print(f"After deduplication:")
    print(f"  Boundary inflows: {len(boundary_inflows)}")
    print(f"  Boundary outflows: {len(boundary_outflows)}")
    print(f"  Congestion entries: {len(congestion_entries)}")
    print(f"  Congestion exits: {len(congestion_exits)}")

    # Create entry and exit points lists
    entry_points = []
    exit_points = []
    
    # Add boundary inflows as entry points
    for point in boundary_inflows:
        entry_points.append({
            "location": point["location"],
            "weight": point["vehicle_count"],
            "source": "boundary_inflow",
            "road_name": point["road_name"],
            "direction": point["direction"],
            "boundary_side": point.get("boundary_side", "unknown")
        })
    
    # Add boundary outflows as exit points
    for point in boundary_outflows:
        exit_points.append({
            "location": point["location"],
            "weight": point["vehicle_count"],
            "source": "boundary_outflow",
            "road_name": point["road_name"],
            "direction": point["direction"],
            "boundary_side": point.get("boundary_side", "unknown")
        })
    
    # Add congestion entry nodes
    for node in congestion_entries:
        entry_points.append({
            "location": node["location"],
            "weight": int(node["congestion_score"] / 5),  # Convert score to comparable weight
            "source": "congestion_entry",
            "road_name": node["road_name"],
            "direction": node["direction"],
            "congestion_score": node["congestion_score"],
            "jam_factor": node["jam_factor"]
        })
    
    # Add congestion exit nodes
    for node in congestion_exits:
        exit_points.append({
            "location": node["location"],
            "weight": int(node["congestion_score"] / 5),  # Convert score to comparable weight
            "source": "congestion_exit",
            "road_name": node["road_name"],
            "direction": node["direction"],
            "congestion_score": node["congestion_score"],
            "jam_factor": node["jam_factor"]
        })
    
    # Sort by weight
    entry_points = sorted(entry_points, key=lambda x: x["weight"], reverse=True)
    exit_points = sorted(exit_points, key=lambda x: x["weight"], reverse=True)
    
    #to avoid too short trips:
    def filter_close_points(points, min_distance=500):
        """Filter out points that are too close to each other"""
        filtered = []
        for point in points:
            # Check distance to all already filtered points
            too_close = False
            for existing in filtered:
                dist = haversine_distance(
                    point["location"][0], point["location"][1],
                    existing["location"][0], existing["location"][1]
                )
                if dist < min_distance:
                    too_close = True
                    break
            if not too_close:
                filtered.append(point)
        return filtered
    
    # Apply filtering to enforce minimum distances
    entry_points = filter_close_points(entry_points, min_distance=500)  # 500m minimum
    exit_points = filter_close_points(exit_points, min_distance=500)  # 500m minimum

    # Print summary statistics
    boundary_entry_count = sum(1 for p in entry_points if p["source"] == "boundary_inflow")
    boundary_exit_count = sum(1 for p in exit_points if p["source"] == "boundary_outflow")
    congestion_entry_count = sum(1 for p in entry_points if p["source"] == "congestion_entry")
    congestion_exit_count = sum(1 for p in exit_points if p["source"] == "congestion_exit")
    
    print(f"Generated {len(entry_points)} entry points and {len(exit_points)} exit points")
    print(f"Entry points: {boundary_entry_count} from boundary, {congestion_entry_count} from congestion")
    print(f"Exit points: {boundary_exit_count} from boundary, {congestion_exit_count} from congestion")
    
    print("\nTop 5 Entry Points:")
    for i, point in enumerate(entry_points[:5]):
        source = point["source"]
        if "boundary" in source:
            print(f"  {i+1}. BOUNDARY: {point['road_name']}, Weight: {point['weight']}, Direction: {point['direction']}")
        else:
            print(f"  {i+1}. CONGESTION: {point['road_name']}, Weight: {point['weight']}, Jam: {point['jam_factor']:.1f}")
    
    print("\nTop 5 Exit Points:")
    for i, point in enumerate(exit_points[:5]):
        source = point["source"]
        if "boundary" in source:
            print(f"  {i+1}. BOUNDARY: {point['road_name']}, Weight: {point['weight']}, Direction: {point['direction']}")
        else:
            print(f"  {i+1}. CONGESTION: {point['road_name']}, Weight: {point['weight']}, Jam: {point['jam_factor']:.1f}")
    
    # Create realistic entry/exit distribution for SUMO
    sumo_entry_nodes = []
    sumo_exit_nodes = []
    
    # For each entry/exit, create a SUMO-friendly record
    for i, entry in enumerate(entry_points):
        sumo_entry = {
            "id": f"entry_{i+1}",
            "x": entry["location"][1],  # lon
            "y": entry["location"][0],  # lat
            "weight": entry["weight"],
            "road_name": entry["road_name"],
            "source_type": entry["source"]
        }
        sumo_entry_nodes.append(sumo_entry)
    
    for i, exit in enumerate(exit_points):
        sumo_exit = {
            "id": f"exit_{i+1}",
            "x": exit["location"][1],  # lon
            "y": exit["location"][0],  # lat
            "weight": exit["weight"],
            "road_name": exit["road_name"],
            "source_type": exit["source"]
        }
        sumo_exit_nodes.append(sumo_exit)
    
    # Save combined results
    sumo_points = {
        "entry_points": entry_points,
        "exit_points": exit_points,
        "sumo_entry_nodes": sumo_entry_nodes,
        "sumo_exit_nodes": sumo_exit_nodes
    }
    
    filename = f"{save_dir}/sumo_entry_exit_points_{timestamp_str}.json"
    with open(filename, "w") as f:
        json.dump(sumo_points, f, indent=2)
    print(f"Saved SUMO entry/exit points to {filename}")
    
    return sumo_points

def create_sumo_edge_mapping(traffic_data, timestamp_str, save_dir):
    """
    Create a mapping between HERE traffic flow segments and potential SUMO edges
    This is a pre-processing step to help SUMO simulation use the flow data
    """
    print("\n=== Creating SUMO Edge Mapping ===")
    
    results = traffic_data.get('results', [])
    if not results:
        print("No traffic data available for mapping")
        return False
    
    # Create dictionary to store mapping
    mapping = {
        "segments": [],
        "metadata": {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "total_segments": 0,
            "bounding_box": BBOX
        }
    }
    
    segment_count = 0
    
    # Process each segment
    for item_idx, item in enumerate(results):
        # Get traffic flow data
        current_flow = item.get("currentFlow", {})
        jam_factor = current_flow.get("jamFactor", 0) if current_flow else 0
        speed = current_flow.get("speed", 0) if current_flow else 0
        free_flow = current_flow.get("freeFlow", 0) if current_flow else 0
        
        # Get road information
        location = item.get("location", {})
        road_name = location.get("description", "Unknown Road")
        
        # Handle shape data
        shape_data = location.get("shape", {})
        links = shape_data.get("links", [])
        
        if not links:
            continue
        
        for link_idx, link in enumerate(links):
            # Get functional class if available
            functional_class = link.get("functionalClass", 5)
            
            # Get points to determine direction
            points = link.get("points", [])
            if len(points) < 2:
                continue
            
            # Calculate direction
            direction = calculate_segment_direction(points)
            
            # Calculate segment length if not provided
            link_length = link.get("length", 0)
            if link_length <= 0:
                # Estimate length using Haversine distance
                link_length = 0
                for i in range(len(points) - 1):
                    lat1 = points[i].get("lat", 0)
                    lon1 = points[i].get("lng", 0)
                    lat2 = points[i+1].get("lat", 0)
                    lon2 = points[i+1].get("lng", 0)
                    segment_length = haversine_distance(lat1, lon1, lat2, lon2)
                    link_length += segment_length
            
            # Get start and end points
            start_lat = points[0].get("lat", 0)
            start_lon = points[0].get("lng", 0)
            end_lat = points[-1].get("lat", 0)
            end_lon = points[-1].get("lng", 0)
            
            # Estimate vehicle count based on jam factor and length
            if jam_factor < 2:
                vehicles_per_km = 10 + jam_factor * 5
            elif jam_factor < 5:
                vehicles_per_km = 20 + (jam_factor - 2) * 10
            elif jam_factor < 8:
                vehicles_per_km = 50 + (jam_factor - 5) * 16.7
            else:
                vehicles_per_km = 100 + (jam_factor - 8) * 33.3
            
            segment_vehicles = int(vehicles_per_km * (link_length / 1000))
            
            # Create segment mapping
            segment_id = f"segment_{item_idx}_{link_idx}"
            segment_mapping = {
                "id": segment_id,
                "road_name": road_name,
                "functional_class": functional_class,
                "direction": direction,
                "jam_factor": jam_factor,
                "speed": speed,
                "free_flow_speed": free_flow,
                "length": link_length,
                "vehicle_count": segment_vehicles,
                "vehicle_density": vehicles_per_km,
                # Include start/end coordinates for matching
                "start_point": [start_lat, start_lon],
                "end_point": [end_lat, end_lon],
                "shape": [[point.get("lat", 0), point.get("lng", 0)] for point in points],
                # Placeholder for SUMO edge IDs (to be filled by the simulation script)
                "sumo_edges": []
            }
            
            mapping["segments"].append(segment_mapping)
            segment_count += 1
    
    # Update metadata
    mapping["metadata"]["total_segments"] = segment_count
    
    # Save the mapping file
    filename = f"{save_dir}/sumo_edge_mapping_{timestamp_str}.json"
    with open(filename, "w") as f:
        json.dump(mapping, f, indent=2)
    
    print(f"SUMO edge mapping file created: {filename}")
    print(f"Total segments mapped: {segment_count}")
    
    return mapping

def visualize_traffic_data(traffic_data, incidents_data, boundary_points, congestion_nodes, timestamp_str, save_dir):
    """Create traffic data visualizations"""
    print("\n=== Creating Traffic Data Visualizations ===")
    
    # Create visualization directory
    vis_dir = f"{save_dir}/visualizations_{timestamp_str}"
    os.makedirs(vis_dir, exist_ok=True)
    
    # Extract data
    results = traffic_data.get('results', [])
    
    if not results:
        print("No traffic data to visualize")
        return False
    
    # Create a figure for the traffic flow map
    plt.figure(figsize=(15, 10))
    
    # Plot traffic flow map
    ax = plt.subplot(111)
    
    # Track coordinates for each segment for later use
    segments_coords = []
    
    # Plot each road segment, colored by jam factor
    for item in results:
        # Get traffic flow data
        current_flow = item.get("currentFlow", {})
        jam_factor = current_flow.get("jamFactor", 0) if current_flow else 0
        speed = current_flow.get("speed", 0) if current_flow else 0
        
        # Get road information
        location = item.get("location", {})
        road_name = location.get("description", "Unknown Road")
        
        # Handle shape data
        shape_data = location.get("shape", {})
        links = shape_data.get("links", [])
        
        if not links:
            continue
        
        for link in links:
            points = link.get("points", [])
            if len(points) < 2:
                continue
            
            # Extract lat/lon pairs
            lats = [point.get("lat", 0) for point in points]
            lons = [point.get("lng", 0) for point in points]
            
            # Save for later use
            segments_coords.append((lats, lons, jam_factor, speed))
            
            # Color based on jam factor (0-10)
            color = plt.cm.RdYlGn_r(jam_factor / 10)  # Red for high jam, green for low
            
            # Line width based on functional class
            functional_class = link.get("functionalClass", 5)
            linewidth = 4 - (functional_class * 0.5)  # Class 1 = 3.5, Class 5 = 1.5
            
            # Plot the segment
            plt.plot(lons, lats, color=color, linewidth=linewidth, alpha=0.7)
    
    # Plot incidents if available
    if incidents_data and 'results' in incidents_data:
        incidents = incidents_data.get('results', [])
        
        for incident in incidents:
            # Get incident details
            incident_type = incident.get("incidentType", "UNKNOWN")
            severity = incident.get("criticality", 0)
            
            # Get location
            location = incident.get("location", {})
            shape_data = location.get("shape", {})
            
            # Get first link for simplicity
            links = shape_data.get("links", [])
            if not links:
                continue
            
            link = links[0]
            points = link.get("points", [])
            if len(points) < 1:
                continue
            
            # Use first point as incident location
            lat = points[0].get("lat", 0)
            lon = points[0].get("lng", 0)
            
            # Use different markers for different incident types
            if "ACCIDENT" in incident_type:
                marker = 'ro'  # Red circle for accidents
                markersize = 8 + severity  # Bigger marker for more severe incidents
            elif "CONSTRUCTION" in incident_type:
                marker = 'ys'  # Yellow square for construction
                markersize = 7 + severity
            else:
                marker = 'b^'  # Blue triangle for other incidents
                markersize = 6 + severity
            
            # Plot the incident
            plt.plot(lon, lat, marker, markersize=markersize, alpha=0.8)
    
    # Set bounding box
    plt.xlim(BBOX["min_lon"], BBOX["max_lon"])
    plt.ylim(BBOX["min_lat"], BBOX["max_lat"])
    
    # Add a colorbar for the jam factor
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r, norm=plt.Normalize(0, 10))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Jam Factor')
    
    # Add title and labels
    plt.title(f'Traffic Flow Map - {timestamp_str}')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    # Add legend for incidents
    plt.plot([], [], 'ro', markersize=8, label='Accident')
    plt.plot([], [], 'ys', markersize=7, label='Construction')
    plt.plot([], [], 'b^', markersize=6, label='Other Incident')
    plt.legend()
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(f"{vis_dir}/traffic_flow_map.png", dpi=300)
    plt.close()
    
    # Create a second map with entry/exit points
    plt.figure(figsize=(15, 10))
    ax = plt.subplot(111)
    
    # Plot road network lightly for background
    for lats, lons, jam_factor, _ in segments_coords:
        # Use lighter colors for background
        color = plt.cm.Blues(jam_factor / 10)  # Blue scale for background
        plt.plot(lons, lats, color=color, linewidth=0.5, alpha=0.3)
    
    # Plot entry/exit points from boundary analysis
    for point in boundary_points:
        loc = point.get("location", [0, 0])
        if loc:
            if point.get("type") == "inflow":
                marker = 'g^'  # Green triangle for inflows
                plt.plot(loc[1], loc[0], marker, markersize=8, alpha=0.8)
            else:
                marker = 'rv'  # Red triangle for outflows
                plt.plot(loc[1], loc[0], marker, markersize=8, alpha=0.8)
    
    # Plot congestion nodes
    for node in congestion_nodes:
        loc = node.get("location", [0, 0])
        if loc:
            if node.get("node_type") == "entry":
                marker = 'go'  # Green circle for entry nodes
                plt.plot(loc[1], loc[0], marker, markersize=10, alpha=0.8)
            else:
                marker = 'ro'  # Red circle for exit nodes
                plt.plot(loc[1], loc[0], marker, markersize=10, alpha=0.8)
    
    # Plot incidents
    if incidents_data and 'results' in incidents_data:
        incidents = incidents_data.get('results', [])
        
        for incident in incidents:
            location = incident.get("location", {})
            shape_data = location.get("shape", {})
            links = shape_data.get("links", [])
            
            if not links:
                continue
            
            link = links[0]
            points = link.get("points", [])
            if len(points) < 1:
                continue
            
            # Use first point as incident location
            lat = points[0].get("lat", 0)
            lon = points[0].get("lng", 0)
            
            # Plot with yellow X marker with black outline
            plt.plot(lon, lat, 'yx', markersize=10, markeredgewidth=1.5, markeredgecolor='black')
    
    # Set bounding box
    plt.xlim(BBOX["min_lon"], BBOX["max_lon"])
    plt.ylim(BBOX["min_lat"], BBOX["max_lat"])
    
    # Add title and labels
    plt.title(f'SUMO Entry/Exit Points - {timestamp_str}')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    # Add legend
    plt.plot([], [], 'g^', markersize=8, label='Boundary Inflow')
    plt.plot([], [], 'rv', markersize=8, label='Boundary Outflow')
    plt.plot([], [], 'go', markersize=10, label='Congestion Entry')
    plt.plot([], [], 'ro', markersize=10, label='Congestion Exit')
    plt.plot([], [], 'yx', markersize=10, markeredgewidth=1.5, markeredgecolor='black', label='Incident')
    plt.legend()
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(f"{vis_dir}/entry_exit_points.png", dpi=300)
    plt.close()
    
    print(f"Visualizations saved to {vis_dir}/")
    return True

def create_analysis_comparison_csv(all_analyses, save_dir):
    """Create a CSV file comparing analysis metrics across all timestamps"""
    import csv
    
    if not all_analyses:
        print("No analyses to compare")
        return False
    
    # Define fields to include in comparison
    fields = [
        "timestamp", 
        "segment_count", 
        "total_road_length_km",
        "total_vehicles", 
        "avg_vehicle_density", 
        "avg_jam_factor",
        "avg_speed", 
        "avg_free_flow",
        "speed_ratio"
    ]
    
    # Include incident counts if available
    if "incidents" in all_analyses[0] and "total_incidents" in all_analyses[0]["incidents"]:
        fields.append("total_incidents")
    
    # Create CSV file
    csv_filename = f"{save_dir}/timestamp_comparison.csv"
    
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        
        for analysis in all_analyses:
            row = {}
            for field in fields:
                if field == "timestamp":
                    # Extract timestamp from scenario name
                    scenario = analysis.get("scenario", "")
                    # Convert from analysis_YYYYMMDD_HHMMSS to more readable format
                    if scenario.startswith("analysis_"):
                        timestamp = scenario[9:]  # Remove "analysis_" prefix
                        try:
                            dt = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
                            row[field] = dt.strftime("%Y-%m-%d %H:%M:%S")
                        except:
                            row[field] = timestamp
                    else:
                        row[field] = scenario
                elif field == "total_incidents" and "incidents" in analysis:
                    row[field] = analysis["incidents"].get("total_incidents", 0)
                else:
                    row[field] = analysis.get(field, "")
            
            writer.writerow(row)
    
    print(f"Created analysis comparison CSV: {csv_filename}")
    return True


def main():
    """Main execution function"""
    # Parse command line arguments
    args = parse_arguments()
    data_dir = args.data_dir
    
    # For midpoint analysis (original functionality)
    mid_realtime_file, mid_incident_file = find_mid_timepoint_files(data_dir)
    
    if not mid_realtime_file:
        print("Error: Could not find mid-timepoint files.")
        return
    
    # Extract timestamp from filename for midpoint
    timestamp_str = extract_timestamp_from_filename(mid_realtime_file)
    if not timestamp_str:
        print("Error: Could not extract timestamp from filename.")
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print(f"\n==== Processing All Timestamps and Mid Timepoint {timestamp_str} ====")
    
    # Find all realtime and incident files for all-timestamp analysis
    all_realtime_files = glob.glob(os.path.join(data_dir, "realtime_*.json"))
    all_incident_files = glob.glob(os.path.join(data_dir, "incidents_*.json"))
    
    # Sort the files by timestamp
    all_realtime_files.sort()
    all_incident_files.sort()
    
    # Map incident files to realtime files by timestamp
    realtime_to_incident = {}
    
    for realtime_file in all_realtime_files:
        realtime_ts = extract_timestamp_from_filename(realtime_file)
        if not realtime_ts:
            continue
            
        # Find closest incident file
        closest_incident = None
        min_diff = float('inf')
        
        for incident_file in all_incident_files:
            incident_ts = extract_timestamp_from_filename(incident_file)
            if not incident_ts:
                continue
                
            # Calculate time difference
            try:
                realtime_dt = datetime.strptime(realtime_ts, "%Y%m%d_%H%M%S")
                incident_dt = datetime.strptime(incident_ts, "%Y%m%d_%H%M%S")
                diff = abs((realtime_dt - incident_dt).total_seconds())
                
                if diff < min_diff:
                    min_diff = diff
                    closest_incident = incident_file
            except Exception as e:
                print(f"Error parsing timestamp: {e}")
        
        realtime_to_incident[realtime_file] = closest_incident
    
    # Run analysis for all timestamps
    all_analyses = []
    
    for realtime_file in all_realtime_files:
        current_ts = extract_timestamp_from_filename(realtime_file)
        if not current_ts:
            continue
            
        print(f"\n==== Analyzing Timestamp {current_ts} ====")
        
        # Load traffic and incident data
        traffic_data = load_json_file(realtime_file)
        incident_file = realtime_to_incident.get(realtime_file)
        incidents_data = load_json_file(incident_file) if incident_file else None
        
        if not traffic_data:
            print(f"Error: Could not load traffic data from {realtime_file}")
            continue
        
        # Only run analyze_traffic_data for all timestamps
        analysis = analyze_traffic_data(traffic_data, incidents_data, current_ts, data_dir)
        if analysis:
            all_analyses.append(analysis)
    
    # Create comparison CSV
    create_analysis_comparison_csv(all_analyses, data_dir)
    
    # Continue with original functionality for the midpoint file only
    # Load traffic and incident data for midpoint
    traffic_data = load_json_file(mid_realtime_file)
    incidents_data = load_json_file(mid_incident_file) if mid_incident_file else None
    
    if not traffic_data:
        print(f"Error: Could not load traffic data from {mid_realtime_file}")
        return
    
    # Get the midpoint analysis that was already done
    analysis_data = next((a for a in all_analyses if a.get("scenario") == f"analysis_{timestamp_str}"), None)
    
    if not analysis_data:
        print("Warning: Midpoint analysis not found in all analyses, rerunning...")
        analysis_data = analyze_traffic_data(traffic_data, incidents_data, timestamp_str, data_dir)
    
    if not analysis_data:
        print("Error: Failed to analyze traffic data")
        return
    
    # Calculate dynamic point counts if not specified
    boundary_count = args.boundary_count
    congestion_count = args.congestion_count
    
    if boundary_count <= 0 or congestion_count <= 0:
        boundary_count_calc, congestion_count_calc = calculate_dynamic_point_counts(analysis_data, BBOX, args.density_factor)        
        if args.boundary_count <= 0:
            boundary_count = boundary_count_calc
        
        if args.congestion_count <= 0:
            congestion_count = congestion_count_calc
    
    print(f"Using boundary point count: {boundary_count}")
    print(f"Using congestion node count: {congestion_count}")
    
    # Find boundary points with incident filtering
    boundary_points = find_boundary_points(
        traffic_data, 
        incidents_data, 
        args.incident_buffer,
        timestamp_str, 
        data_dir, 
        boundary_count
    )
    
    # Find congestion nodes with incident filtering
    congestion_nodes = find_congestion_nodes(
        traffic_data, 
        incidents_data, 
        args.incident_buffer,
        timestamp_str, 
        data_dir, 
        congestion_count
    )
    
    # Generate SUMO entry/exit points
    if boundary_points or congestion_nodes:
        sumo_points = generate_sumo_points(boundary_points, congestion_nodes, timestamp_str, data_dir)
        
        # Create SUMO edge mapping
        edge_mapping = create_sumo_edge_mapping(traffic_data, timestamp_str, data_dir)
        
        # Generate visualizations if requested
        if args.visualize:
            visualize_traffic_data(traffic_data, incidents_data, boundary_points, congestion_nodes, timestamp_str, data_dir)
        
        print("\n=== Final Summary ===")
        print(f"Timestamp: {timestamp_str}")
        print(f"Generated {len(sumo_points['entry_points'])} entry points and {len(sumo_points['exit_points'])} exit points")
        print("Entries from boundary: %d, from congestion: %d" % (
            sum(1 for p in sumo_points['entry_points'] if p['source'] == 'boundary_inflow'),
            sum(1 for p in sumo_points['entry_points'] if p['source'] == 'congestion_entry')
        ))
        print("Exits from boundary: %d, from congestion: %d" % (
            sum(1 for p in sumo_points['exit_points'] if p['source'] == 'boundary_outflow'),
            sum(1 for p in sumo_points['exit_points'] if p['source'] == 'congestion_exit')
        ))
        
        if incidents_data:
            incident_count = len(incidents_data.get('results', []))
            print(f"Total incidents: {incident_count}")
            print("Note: Congested areas caused by incidents have been filtered out.")
        
        print("\nRecommendation for SUMO simulation:")
        print("1. Use the generated entry/exit points from both boundary segments and congested nodes.")
        print("2. Apply weights to determine vehicle insertion probability at each point.")
        print("3. Use the edge mapping to associate HERE traffic data with SUMO network edges.")
        print("4. Adjust simulation parameters based on the traffic analysis results.")
        
        if args.visualize:
            print(f"5. Check the visualizations in {data_dir}/visualizations_{timestamp_str}/ to verify the data.")
    else:
        print("\nWarning: Could not identify sufficient boundary points or congestion nodes.")
    
    print("\nProcessing complete.")

if __name__ == "__main__":
    main()