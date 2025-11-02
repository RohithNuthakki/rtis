import os
import sys
import json
import time
import requests
import datetime
import argparse
import random
from pathlib import Path

# HERE API configuration
HERE_API_KEY = "ejchR9sgdh8wvASbWzfpy3bfkA_PFOo3VA2-eAdQQHo"  # Your API key

# Manhattan bounding box
BBOX = {
    "min_lat": 40.75,
    "min_lon": -73.98,
    "max_lat": 40.7584,
    "max_lon": -73.9688
}

def get_current_traffic():
    """Get current traffic flow data from HERE Traffic API"""
    print("Requesting current traffic flow data from HERE Traffic API...")
    
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
        
        if response.status_code == 200:
            data = response.json()
            # Add a timestamp to the data
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            data['timestamp'] = timestamp
            
            # Create directory for traffic data if it doesn't exist
            os.makedirs("traffic_data", exist_ok=True)
            
            # Save raw traffic data with timestamp
            filename = f"traffic_data/traffic_flow_{timestamp}.json"
            with open(filename, "w") as f:
                json.dump(data, f, indent=2)
                
            print(f"Traffic flow data retrieved successfully with {len(data.get('results', []))} records.")
            print(f"Saved to {filename}")
            
            return data
        else:
            print(f"Failed to get traffic flow data: {response.status_code}")
            print(f"Response: {response.text}")
            return None
    except Exception as e:
        print(f"Error getting traffic flow data: {e}")
        return None

def get_entry_exit_points():
    """Load or generate entry/exit points for the network area"""
    # Try to load from existing file
    if os.path.exists("entry_exit_points.json"):
        try:
            with open("entry_exit_points.json", "r") as f:
                return json.load(f)
        except:
            print("Could not load existing entry_exit_points.json")
    
    # Generate new points based on bounding box
    print("Generating entry/exit points along the network boundary...")
    
    points = []
    
    # Create points along the edges of the bounding box
    min_lat, max_lat = BBOX["min_lat"], BBOX["max_lat"]
    min_lon, max_lon = BBOX["min_lon"], BBOX["max_lon"]
    
    # Number of points to generate on each edge
    num_points = 3
    
    # Bottom edge (min_lat)
    for i in range(num_points):
        lon = min_lon + (max_lon - min_lon) * (i + 0.5) / num_points
        points.append({
            "id": f"south_{i}",
            "lat": min_lat,
            "lon": lon,
            "type": "boundary"
        })
    
    # Top edge (max_lat)
    for i in range(num_points):
        lon = min_lon + (max_lon - min_lon) * (i + 0.5) / num_points
        points.append({
            "id": f"north_{i}",
            "lat": max_lat,
            "lon": lon,
            "type": "boundary"
        })
    
    # Left edge (min_lon)
    for i in range(num_points):
        lat = min_lat + (max_lat - min_lat) * (i + 0.5) / num_points
        points.append({
            "id": f"west_{i}",
            "lat": lat,
            "lon": min_lon,
            "type": "boundary"
        })
    
    # Right edge (max_lon)
    for i in range(num_points):
        lat = min_lat + (max_lat - min_lat) * (i + 0.5) / num_points
        points.append({
            "id": f"east_{i}",
            "lat": lat,
            "lon": max_lon,
            "type": "boundary"
        })
    
    # Save to file
    os.makedirs("traffic_data", exist_ok=True)
    with open("traffic_data/generated_entry_exit_points.json", "w") as f:
        json.dump(points, f, indent=2)
    
    return points

def get_matrix_routing(entry_exit_points):
    """Get current travel times between entry/exit points using HERE Matrix Routing API"""
    print("Requesting origin-destination matrix from HERE Matrix Routing API...")
    
    # Current time for departure
    now = datetime.datetime.now()
    # Format as ISO 8601 with timezone (RFC 3339 compliant)
    departure_time = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    
    # HERE Matrix Routing API endpoint
    matrix_url = "https://matrix.router.hereapi.com/v8/matrix"
    
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
    max_points_per_request = 10  # Lower to ensure we stay within limits
    travel_data = []
    
    # Process in batches
    batch_count = 0
    
    for i in range(0, len(origins), max_points_per_request):
        batch_origins = origins[i:i+max_points_per_request]
        
        for j in range(0, len(destinations), max_points_per_request):
            batch_destinations = destinations[j:j+max_points_per_request]
            
            batch_count += 1
            print(f"Requesting batch {batch_count}: {len(batch_origins)} origins, {len(batch_destinations)} destinations")
            
            params = {
                "apiKey": HERE_API_KEY
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
                response = requests.post(matrix_url, params=params, json=payload, headers=headers)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Save raw response for debugging
                    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                    os.makedirs("traffic_data", exist_ok=True)
                    with open(f"traffic_data/matrix_response_{timestamp}_batch_{batch_count}.json", "w") as f:
                        json.dump(result, f, indent=2)
                    
                    # Extract travelTimes and distances
                    matrix = result.get("matrix", {})
                    travel_times = matrix.get("travelTimes", [])
                    distances = matrix.get("distances", [])
                    
                    # Process the matrix data
                    for origin_idx, origin_times in enumerate(travel_times):
                        for dest_idx, time in enumerate(origin_times):
                            # Calculate global indices
                            global_origin_idx = i + origin_idx
                            global_dest_idx = j + dest_idx
                            
                            # Skip self-loops
                            if global_origin_idx == global_dest_idx:
                                continue
                                
                            distance = distances[origin_idx][dest_idx] if distances and origin_idx < len(distances) and dest_idx < len(distances[origin_idx]) else None
                            
                            travel_data.append({
                                "origin_idx": global_origin_idx,
                                "dest_idx": global_dest_idx,
                                "origin_id": entry_exit_points[global_origin_idx]["id"],
                                "dest_id": entry_exit_points[global_dest_idx]["id"],
                                "travel_time": time,
                                "distance": distance,
                                "timestamp": timestamp
                            })
                    
                    print(f"Batch {batch_count} successful: Added {len(origin_times) * len(origin_times)} entries")
                else:
                    print(f"API request failed: {response.status_code}")
                    print(f"Response: {response.text}")
                    # Continue processing other batches
            except Exception as e:
                print(f"Error requesting travel times: {e}")
                # Continue with other batches
            
            # Respect API rate limits
            time.sleep(1)
    
    # Save raw matrix data
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    with open(f"traffic_data/travel_matrix_{timestamp}.json", "w") as f:
        json.dump(travel_data, f, indent=2)
    
    print(f"Origin-destination matrix created with {len(travel_data)} entries.")
    return travel_data

def collect_traffic_over_time(interval_minutes=15, duration_hours=1):
    """Collect traffic data at regular intervals for a specified duration"""
    print(f"Collecting traffic data every {interval_minutes} minutes for {duration_hours} hours...")
    
    start_time = time.time()
    end_time = start_time + (duration_hours * 3600)
    
    collection_count = 0
    
    # Get entry/exit points (only needed once)
    entry_exit_points = get_entry_exit_points()
    print(f"Using {len(entry_exit_points)} entry/exit points for origin-destination matrix")
    
    while time.time() < end_time:
        print(f"\nCollection #{collection_count + 1}")
        
        # Get traffic flow data
        flow_data = get_current_traffic()
        
        # Get matrix routing data (only every other time to reduce API calls)
        if collection_count % 2 == 0:
            matrix_data = get_matrix_routing(entry_exit_points)
        else:
            print("Skipping matrix routing collection this cycle to reduce API calls")
        
        if flow_data:
            collection_count += 1
        
        # Calculate time to sleep until next collection
        next_collection = start_time + (collection_count * interval_minutes * 60)
        time_to_sleep = max(0, next_collection - time.time())
        
        if time_to_sleep > 0 and time.time() + time_to_sleep < end_time:
            print(f"Waiting {time_to_sleep:.1f} seconds until next collection...")
            time.sleep(time_to_sleep)
        else:
            # If we've reached the end time, break
            if time.time() >= end_time:
                break
    
    print(f"\nCompleted {collection_count} traffic data collections.")
    return collection_count

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Collect real-time traffic data from HERE API")
    parser.add_argument("--interval", type=int, default=15, help="Collection interval in minutes (default: 15)")
    parser.add_argument("--duration", type=int, default=1, help="Total collection duration in hours (default: 1)")
    parser.add_argument("--single", action="store_true", help="Collect a single sample instead of continuous collection")
    parser.add_argument("--flow-only", action="store_true", help="Only collect traffic flow data (no matrix routing)")
    parser.add_argument("--matrix-only", action="store_true", help="Only collect matrix routing data (no traffic flow)")
    
    args = parser.parse_args()
    
    if args.single:
        print("Collecting a single traffic data sample...")
        if not args.matrix_only:
            get_current_traffic()
        
        if not args.flow_only:
            entry_exit_points = get_entry_exit_points()
            get_matrix_routing(entry_exit_points)
    else:
        collect_traffic_over_time(args.interval, args.duration)
    
    print("\nTraffic data collection complete.")
    
if __name__ == "__main__":
    main()