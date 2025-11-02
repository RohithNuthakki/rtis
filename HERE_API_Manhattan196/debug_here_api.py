#!/usr/bin/env python3
"""
HERE Traffic API Congestion Analysis

This script analyzes real-time traffic data from the HERE API to identify:
1. Top boundary points for entry/exit by vehicle count
2. Most congested nodes within the area for potential SUMO simulation points
3. Traffic incidents that affect traffic flow
"""

import os
import sys
import json
import requests
import matplotlib.pyplot as plt
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2

# API key
HERE_API_KEY = None

def parse_arguments():
    """Parse command line arguments"""
    import argparse
    parser = argparse.ArgumentParser(description='Analyze congestion in HERE Traffic API data')
    parser.add_argument('--api-key', required=True, help='HERE API key')
    parser.add_argument('--save-dir', default='traffic_analysis', help='Directory to save analysis data')
    parser.add_argument('--min-lat', type=float, default=40.712178, help='Minimum latitude for bounding box')
    parser.add_argument('--min-lon', type=float, default=-74.033341, help='Minimum longitude for bounding box')
    parser.add_argument('--max-lat', type=float, default=40.759722, help='Maximum latitude for bounding box')
    parser.add_argument('--max-lon', type=float, default=-73.958424, help='Maximum longitude for bounding box')
    parser.add_argument('--boundary-count', type=int, default=0, help='Number of top boundary points to use (0 for automatic)')
    parser.add_argument('--congestion-count', type=int, default=0, help='Number of top congested nodes to use (0 for automatic)')
    parser.add_argument('--density-factor', type=float, default=1.0, help='Factor to adjust point density (higher = more points)')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations of traffic data')
    return parser.parse_args()

def get_traffic_data(save_dir):
    """Get real-time traffic flow data from HERE Traffic API"""
    print("\n==== Getting real-time traffic data ====")
    
    # Prepare bbox string
    bbox_str = f"{BBOX['min_lon']},{BBOX['min_lat']},{BBOX['max_lon']},{BBOX['max_lat']}"
    print(f"Using bounding box: {bbox_str}")
    
    # HERE Traffic Flow API endpoint
    traffic_url = "https://data.traffic.hereapi.com/v7/flow"
    
    # Set parameters
    params = {
        "apiKey": HERE_API_KEY,
        "in": f"bbox:{bbox_str}",
        "locationReferencing": "shape",
        "functionalClass": "1,2,3,4,5",
        "returnJamFactor": "true",
        "returnTravelTime": "true",
        "returnFreeFlow": "true"
    }
    
    try:
        response = requests.get(traffic_url, params=params)
        
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            # Basic validation
            if 'results' in data:
                results_count = len(data.get('results', []))
                print(f"Results count: {results_count}")
                
                # Save the data
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{save_dir}/realtime_{timestamp}.json"
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                
                with open(filename, "w") as f:
                    json.dump(data, f, indent=2)
                print(f"Saved data to {filename}")
                
                return data
            else:
                print("No results found in response")
        else:
            print(f"Request failed: {response.status_code}")
            print(f"Response: {response.text}")
    
    except Exception as e:
        print(f"Error: {e}")
    
    return None

def get_incidents_data(save_dir):
    """Get real-time traffic incidents data from HERE Traffic API"""
    print("\n==== Getting real-time traffic incidents data ====")
    
    # Prepare bbox string
    bbox_str = f"{BBOX['min_lon']},{BBOX['min_lat']},{BBOX['max_lon']},{BBOX['max_lat']}"
    
    # HERE Traffic Incidents API endpoint
    incidents_url = "https://data.traffic.hereapi.com/v7/incidents"
    
    # Set parameters
    params = {
        "apiKey": HERE_API_KEY,
        "in": f"bbox:{bbox_str}",
        "locationReferencing": "shape"
    }
    
    try:
        response = requests.get(incidents_url, params=params)
        
        print(f"Incidents response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            # Basic validation
            if 'results' in data:
                incidents_count = len(data.get('results', []))
                print(f"Incidents count: {incidents_count}")
                
                # Save the data
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{save_dir}/incidents_{timestamp}.json"
                
                with open(filename, "w") as f:
                    json.dump(data, f, indent=2)
                print(f"Saved incidents data to {filename}")
                
                return data
            else:
                print("No incidents found in response")
        else:
            print(f"Incidents request failed: {response.status_code}")
            print(f"Response: {response.text}")
    
    except Exception as e:
        print(f"Error getting incidents: {e}")
    
    return None

def analyze_traffic_data(data, incidents_data, scenario_name):
    """Analyze traffic data to extract key metrics"""
    results = data.get('results', [])
    
    if not results:
        print("No results to analyze")
        return
    
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
                if jam_factor < 2:
                    vehicles_per_km = 10 + jam_factor * 5
                elif jam_factor < 5:
                    vehicles_per_km = 20 + (jam_factor - 2) * 10
                elif jam_factor < 8:
                    vehicles_per_km = 50 + (jam_factor - 5) * 16.7
                else:
                    vehicles_per_km = 100 + (jam_factor - 8) * 33.3
                
                segment_vehicles = int(vehicles_per_km * (link_length / 1000))
                total_vehicles += segment_vehicles
    
    # Calculate average statistics
    avg_jam_factor = sum(jam_factors) / len(jam_factors) if jam_factors else 0
    avg_speed = sum(speeds) / len(speeds) if speeds else 0
    avg_free_flow = sum(free_flows) / len(free_flows) if free_flows else 0
    
    # Calculate vehicle density
    avg_vehicle_density = total_vehicles / (total_length / 1000) if total_length > 0 else 0
    
    # Print summary
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
        "timestamp": datetime.now().isoformat()
    }
    
    filename = f"{args.save_dir}/{scenario_name}_analysis.json"
    with open(filename, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"Saved analysis to {filename}")
    
    return analysis

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

def find_boundary_points(data, save_dir, top_count=30):
    """Find top boundary points with improved outflow detection"""
    results = data.get('results', [])
    
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
                "link_index": link_idx
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
    
    print(f"\nFound {len(boundary_points)} total boundary points")
    print(f"Inflows: {len(inflows)}, Outflows: {len(outflows)}")
    print(f"Selected {len(top_inflows)} inflows and {len(top_outflows)} outflows based on realistic traffic patterns")
    
    print("\nTop inflow points:")
    for i, point in enumerate(top_inflows[:5]):
        print(f"  {i+1}. {point['road_name']} ({point['vehicle_count']} vehicles, {point['direction']})")
    
    print("\nTop outflow points:")
    for i, point in enumerate(top_outflows[:5]):
        print(f"  {i+1}. {point['road_name']} ({point['vehicle_count']} vehicles, {point['direction']})")
    
    # Save analysis
    boundary_analysis = {
        "total_boundary_points": len(boundary_points),
        "inflows": len(inflows),
        "outflows": len(outflows),
        "top_inflows": top_inflows,
        "top_outflows": top_outflows
    }
    
    filename = f"{save_dir}/top_boundary_points.json"
    with open(filename, "w") as f:
        json.dump(boundary_analysis, f, indent=2)
    print(f"Saved top boundary points to {filename}")
    
    return top_boundary



def find_congestion_nodes(data, save_dir, top_count=30):
    """Find most congested nodes with directional analysis"""
    results = data.get('results', [])
    
    if not results:
        print("No results to analyze for congestion nodes")
        return []
    
    print(f"\n=== Finding Top {top_count} Congested Nodes with Directional Analysis ===")
    
    # Store potential congestion nodes
    congestion_nodes = []
    
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
                "link_index": link_idx
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
    
    print(f"\nFound {len(congestion_nodes)} total congestion nodes")
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
    
    filename = f"{save_dir}/top_congestion_nodes.json"
    with open(filename, "w") as f:
        json.dump(congestion_analysis, f, indent=2)
    print(f"Saved top congestion nodes to {filename}")
    
    return top_nodes

def generate_sumo_points(boundary_points, congestion_nodes, save_dir):
    """Generate authentic entry/exit points for SUMO simulation"""
    print("\n=== Generating SUMO Entry/Exit Points ===")
    
    # Extract boundary inflows and outflows
    boundary_inflows = [point for point in boundary_points if point["type"] == "inflow"]
    boundary_outflows = [point for point in boundary_points if point["type"] == "outflow"]
    
    # Extract congestion entries and exits
    congestion_entries = [node for node in congestion_nodes if node.get("node_type") == "entry"]
    congestion_exits = [node for node in congestion_nodes if node.get("node_type") == "exit"]
    
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
    
    filename = f"{save_dir}/sumo_entry_exit_points.json"
    with open(filename, "w") as f:
        json.dump(sumo_points, f, indent=2)
    print(f"Saved SUMO entry/exit points to {filename}")
    
    return sumo_points

def calculate_dynamic_point_counts(analysis_data, bbox):
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
    # Guideline: ~1 entry/exit point per 0.25-0.5 km² in dense urban areas
    area_factor = max(1, int(area_km2 * 3))
    
    # Adjust based on road density relative to typical urban areas
    # Typical dense urban road density: ~20-40 km/km²
    density_factor = max(0.5, min(2.0, road_density / 25))
    
    # Apply user-defined density factor
    user_factor = args.density_factor
    
    # Final counts (with minimums)
    boundary_count = max(10, int(area_factor * density_factor * user_factor))
    congestion_count = max(10, int(area_factor * density_factor * user_factor))
    
    print(f"\n=== Dynamic Point Count Calculation ===")
    print(f"Bounding box area: {area_km2:.2f} km²")
    print(f"Road length: {road_length_km:.2f} km")
    print(f"Road density: {road_density:.2f} km/km²")
    print(f"Segment count: {segment_count}")
    print(f"Segment density: {segment_density:.2f} segments/km²")
    print(f"Calculated boundary point count: {boundary_count}")
    print(f"Calculated congestion node count: {congestion_count}")
    
    return boundary_count, congestion_count

def visualize_traffic_data(traffic_data, incidents_data, analysis_data, save_dir):
    """Create traffic data visualizations"""
    print("\n=== Creating Traffic Data Visualizations ===")
    
    # Create visualization directory
    os.makedirs(f"{save_dir}/visualizations", exist_ok=True)
    
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
    plt.title('Traffic Flow Map')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    # Add legend for incidents
    plt.plot([], [], 'ro', markersize=8, label='Accident')
    plt.plot([], [], 'ys', markersize=7, label='Construction')
    plt.plot([], [], 'b^', markersize=6, label='Other Incident')
    plt.legend()
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(f"{save_dir}/visualizations/traffic_flow_map.png", dpi=300)
    plt.close()
    
    # Create a bar chart for directional flow distribution
    plt.figure(figsize=(10, 6))
    
    direction_counts = analysis_data.get("direction_counts", {})
    
    # Sort by count
    sorted_directions = {k: v for k, v in sorted(direction_counts.items(), key=lambda item: item[1], reverse=True)}
    
    # Create the bar chart
    plt.bar(sorted_directions.keys(), sorted_directions.values())
    plt.title('Directional Flow Distribution')
    plt.xlabel('Direction')
    plt.ylabel('Number of Road Segments')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/visualizations/directional_flow_distribution.png", dpi=300)
    plt.close()
    
    # Create a scatter plot for speed vs. jam factor
    plt.figure(figsize=(10, 6))
    
    # Extract speed and jam factors from stored segments
    jam_factors = [jf for _, _, jf, _ in segments_coords]
    speeds = [spd for _, _, _, spd in segments_coords]
    
    plt.scatter(jam_factors, speeds, alpha=0.5)
    plt.title('Speed vs. Jam Factor')
    plt.xlabel('Jam Factor')
    plt.ylabel('Speed (km/h)')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{save_dir}/visualizations/speed_vs_jam.png", dpi=300)
    plt.close()
    
    # Create a histogram for jam factor distribution
    plt.figure(figsize=(10, 6))
    plt.hist(jam_factors, bins=20, alpha=0.7)
    plt.title('Jam Factor Distribution')
    plt.xlabel('Jam Factor')
    plt.ylabel('Number of Road Segments')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{save_dir}/visualizations/jam_factor_distribution.png", dpi=300)
    plt.close()
    
    # Create entry/exit point visualizations
    entry_exit_file = f"{save_dir}/sumo_entry_exit_points.json"
    if os.path.exists(entry_exit_file):
        with open(entry_exit_file, "r") as f:
            sumo_points = json.load(f)
        
        # Create a map with entry/exit points
        plt.figure(figsize=(15, 10))
        ax = plt.subplot(111)
        
        # Plot road network lightly for background
        for lats, lons, _, _ in segments_coords:
            plt.plot(lons, lats, color='lightgray', linewidth=0.5, alpha=0.5)
        
        # Plot entry points
        entry_points = sumo_points.get("entry_points", [])
        for point in entry_points:
            loc = point.get("location", [0, 0])
            if loc:
                if point.get("source") == "boundary_inflow":
                    marker = 'g^'  # Green triangle for boundary inflows
                else:
                    marker = 'go'  # Green circle for congestion entries
                plt.plot(loc[1], loc[0], marker, markersize=8, alpha=0.8)
        
        # Plot exit points
        exit_points = sumo_points.get("exit_points", [])
        for point in exit_points:
            loc = point.get("location", [0, 0])
            if loc:
                if point.get("source") == "boundary_outflow":
                    marker = 'rv'  # Red triangle for boundary outflows
                else:
                    marker = 'ro'  # Red circle for congestion exits
                plt.plot(loc[1], loc[0], marker, markersize=8, alpha=0.8)
        
        # Set bounding box
        plt.xlim(BBOX["min_lon"], BBOX["max_lon"])
        plt.ylim(BBOX["min_lat"], BBOX["max_lat"])
        
        # Add title and labels
        plt.title('SUMO Entry/Exit Points')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        
        # Add legend
        plt.plot([], [], 'g^', markersize=8, label='Boundary Inflow')
        plt.plot([], [], 'go', markersize=8, label='Congestion Entry')
        plt.plot([], [], 'rv', markersize=8, label='Boundary Outflow')
        plt.plot([], [], 'ro', markersize=8, label='Congestion Exit')
        plt.legend()
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(f"{save_dir}/visualizations/entry_exit_points.png", dpi=300)
        plt.close()
    
    print(f"Visualizations saved to {save_dir}/visualizations/")
    return True

def create_sumo_edge_mapping(traffic_data, save_dir):
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
    filename = f"{save_dir}/sumo_edge_mapping.json"
    with open(filename, "w") as f:
        json.dump(mapping, f, indent=2)
    
    print(f"SUMO edge mapping file created: {filename}")
    print(f"Total segments mapped: {segment_count}")
    
    return mapping

def main():
    global HERE_API_KEY, args, BBOX
    
    # Parse command line arguments
    args = parse_arguments()
    HERE_API_KEY = args.api_key
    
    # Update bounding box from arguments
    BBOX = {
        "min_lat": args.min_lat,
        "min_lon": args.min_lon,
        "max_lat": args.max_lat,
        "max_lon": args.max_lon
    }
    
    print(f"HERE Traffic API Congestion Analysis")
    print(f"Using API key: {HERE_API_KEY[:5]}...{HERE_API_KEY[-5:]}")
    print(f"Bounding box: {BBOX['min_lat']},{BBOX['min_lon']} to {BBOX['max_lat']},{BBOX['max_lon']}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Get real-time traffic data
    traffic_data = get_traffic_data(args.save_dir)
    
    # Get real-time incidents data
    incidents_data = get_incidents_data(args.save_dir)
    
    if traffic_data:
        # Analyze traffic data
        analysis_data = analyze_traffic_data(traffic_data, incidents_data, "realtime")
        
        # Calculate dynamic point counts if not specified
        boundary_count = args.boundary_count
        congestion_count = args.congestion_count
        
        if boundary_count <= 0 or congestion_count <= 0:
            boundary_count_calc, congestion_count_calc = calculate_dynamic_point_counts(analysis_data, BBOX)
            
            if args.boundary_count <= 0:
                boundary_count = boundary_count_calc
            
            if args.congestion_count <= 0:
                congestion_count = congestion_count_calc
        
        print(f"Using boundary point count: {boundary_count}")
        print(f"Using congestion node count: {congestion_count}")
        
        # Find top boundary points
        boundary_points = find_boundary_points(traffic_data, args.save_dir, boundary_count)
        
        # Find most congested nodes
        congestion_nodes = find_congestion_nodes(traffic_data, args.save_dir, congestion_count)
        
        # Generate recommended SUMO entry/exit points
        if boundary_points or congestion_nodes:
            sumo_points = generate_sumo_points(boundary_points, congestion_nodes, args.save_dir)
            
            # Create SUMO edge mapping
            edge_mapping = create_sumo_edge_mapping(traffic_data, args.save_dir)
            
            # Generate visualizations if requested
            if args.visualize:
                visualize_traffic_data(traffic_data, incidents_data, analysis_data, args.save_dir)
            
            print("\n=== Final Summary ===")
            print(f"Generated {len(sumo_points['entry_points'])} entry points and {len(sumo_points['exit_points'])} exit points")
            print("\nRecommendation for SUMO simulation:")
            print("1. Use the generated entry/exit points from both boundary segments and congested nodes.")
            print("2. Apply weights to determine vehicle insertion probability at each point.")
            print("3. Use the edge mapping to associate HERE traffic data with SUMO network edges.")
            print("4. Adjust simulation parameters based on the traffic analysis results.")
            
            if args.visualize:
                print(f"5. Check the visualizations in {args.save_dir}/visualizations/ to verify the data.")
        else:
            print("\nWarning: Could not identify sufficient boundary points or congestion nodes.")
    
    print("\nAnalysis complete.")

if __name__ == "__main__":
    main()
