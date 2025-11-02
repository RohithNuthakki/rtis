#!/usr/bin/env python3
"""
HERE Traffic API Data Collection

This script collects real-time traffic data and incidents from the HERE API
every 6 minutes for an hour and saves the results as JSON files.
"""

import os
import sys
import json
import time
import requests
from datetime import datetime

# Directory to save data
SAVE_DIR = "traffic_data"

def get_traffic_flow_data(api_key, bbox):
    """Get real-time traffic flow data from HERE Traffic API"""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Getting traffic flow data...")
    
    # Prepare bbox string
    bbox_str = f"{bbox['min_lon']},{bbox['min_lat']},{bbox['max_lon']},{bbox['max_lat']}"
    
    # HERE Traffic Flow API endpoint
    traffic_url = "https://data.traffic.hereapi.com/v7/flow"
    
    # Set parameters
    params = {
        "apiKey": api_key,
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
                filename = f"{SAVE_DIR}/realtime_{timestamp}.json"
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                
                with open(filename, "w") as f:
                    json.dump(data, f, indent=2)
                print(f"Saved traffic data to {filename}")
                
                return data
            else:
                print("No results found in response")
        else:
            print(f"Request failed: {response.status_code}")
            print(f"Response: {response.text}")
    
    except Exception as e:
        print(f"Error: {e}")
    
    return None

def get_incidents_data(api_key, bbox):
    """Get real-time traffic incidents data from HERE Traffic API
    
    This collects the complete incident data including:
    - Incident type (accident, construction, etc.)
    - Severity/criticality
    - Location details with shape data
    - Incident descriptions
    - Start/end times
    - Status information
    - All other fields returned by the API
    """
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Getting incidents data...")
    
    # Prepare bbox string
    bbox_str = f"{bbox['min_lon']},{bbox['min_lat']},{bbox['max_lon']},{bbox['max_lat']}"
    
    # HERE Traffic Incidents API endpoint
    incidents_url = "https://data.traffic.hereapi.com/v7/incidents"
    
    # Set parameters
    params = {
        "apiKey": api_key,
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
                filename = f"{SAVE_DIR}/incidents_{timestamp}.json"
                
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

def collect_data_periodically(api_key, bbox, interval_minutes=6, duration_minutes=60):
    """Collect traffic data at regular intervals for a specified duration
    
    With interval_minutes=6 and duration_minutes=60, this will perform:
    - Initial collection (t=0)
    - Collection after 6 minutes (t=6)
    - Collection after 12 minutes (t=12)
    - Collection after 18 minutes (t=18)
    - Collection after 24 minutes (t=24)
    - Collection after 30 minutes (t=30)
    - Collection after 36 minutes (t=36)
    - Collection after 42 minutes (t=42)
    - Collection after 48 minutes (t=48)
    - Collection after 54 minutes (t=54)
    - Collection after 60 minutes (t=60)
    
    Total: 11 collections over 1 hour
    """
    print(f"Starting data collection every {interval_minutes} minutes for {duration_minutes} minutes")
    print(f"Will perform a total of {(duration_minutes // interval_minutes) + 1} collections")
    
    # Calculate number of iterations
    iterations = duration_minutes // interval_minutes
    
    # Record start time for consistent intervals
    start_time = datetime.now()
    
    for i in range(iterations + 1):  # +1 to include initial collection
        collection_time = datetime.now()
        print(f"\n=== Collection {i+1}/{iterations+1} at {collection_time.strftime('%H:%M:%S')} ===")
        
        # Get traffic flow data
        traffic_data = get_traffic_flow_data(api_key, bbox)
        
        # Get incidents data
        incidents_data = get_incidents_data(api_key, bbox)
        
        # Record collection statistics
        stats = {
            "timestamp": collection_time.isoformat(),
            "collection_number": i+1,
            "total_collections": iterations + 1,
            "minutes_from_start": i * interval_minutes,
            "traffic_data_results": len(traffic_data.get('results', [])) if traffic_data else 0,
            "incidents_count": len(incidents_data.get('results', [])) if incidents_data else 0
        }
        
        # Save collection stats
        stats_filename = f"{SAVE_DIR}/collection_stats.json"
        
        # Append to stats file
        try:
            if os.path.exists(stats_filename):
                with open(stats_filename, 'r') as f:
                    all_stats = json.load(f)
            else:
                all_stats = []
                
            all_stats.append(stats)
            
            with open(stats_filename, 'w') as f:
                json.dump(all_stats, f, indent=2)
        except Exception as e:
            print(f"Error saving stats: {e}")
        
        # Wait for next interval (except after last iteration)
        if i < iterations:
            # Calculate next collection time (based on start time to prevent drift)
            next_collection_minutes = (i + 1) * interval_minutes
            next_collection_time = start_time.timestamp() + (next_collection_minutes * 60)
            current_time = datetime.now().timestamp()
            
            wait_seconds = max(1, int(next_collection_time - current_time))
            
            print(f"Waiting {wait_seconds} seconds until next collection at {datetime.fromtimestamp(next_collection_time).strftime('%H:%M:%S')}")
            print(f"({next_collection_minutes} minutes from start time)")
            time.sleep(wait_seconds)
    
    print(f"\nData collection completed. All data saved to {SAVE_DIR}/")
    print(f"Total collections performed: {iterations + 1}")

def main():
    # Check for API key
    if len(sys.argv) < 2:
        print("Usage: python here_api_collector.py <HERE_API_KEY>")
        sys.exit(1)
    
    api_key = sys.argv[1]
    
    # Default bounding box for Manhattan, NY
    # Can be customized as needed
    bbox = {
        "min_lat": 40.712178,
        "min_lon": -74.033341,
        "max_lat": 40.759722,
        "max_lon": -73.958424
    }
    
    print(f"HERE Traffic API Data Collection")
    print(f"Using API key: {api_key[:5]}...{api_key[-5:]}")
    print(f"Bounding box: {bbox['min_lat']},{bbox['min_lon']} to {bbox['max_lat']},{bbox['max_lon']}")
    
    # Create save directory
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Start collection every 6 minutes for an hour (total of 11 collections)
    collect_data_periodically(api_key, bbox, interval_minutes=6, duration_minutes=60)

if __name__ == "__main__":
    main()