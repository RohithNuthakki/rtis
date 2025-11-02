import os
import sys
import json
import time
import requests
from datetime import datetime, timedelta

"""
travelTimes and distances are 1D arrays with 9 values (3 origins × 3 destinations)
The matrix dimensions are provided in numOrigins (3) and numDestinations (3)
Values are organized as a flattened matrix:

Index 0: Origin 0 to Destination 0
Index 1: Origin 0 to Destination 1
Index 2: Origin 0 to Destination 2
Index 3: Origin 1 to Destination 0
And so on...
"""

# HERE API key
HERE_API_KEY = "ejchR9sgdh8wvASbWzfpy3bfkA_PFOo3VA2-eAdQQHo"

# Test coordinates - these are example points in Manhattan
test_points = [
    {"lat": 40.7514, "lng": -73.9796},
    {"lat": 40.7554, "lng": -73.9842},
    {"lat": 40.7574, "lng": -73.9702}
]

def debug_matrix_routing():
    """Debug the Matrix Routing API call"""
    print("Debugging HERE Matrix Routing API...")
    
    # Prepare the day of week and time
    weekday = 3  # Wednesday
    today = datetime.now()
    days_ahead = weekday - today.weekday()
    if days_ahead <= 0:
        days_ahead += 7
    next_weekday = today + timedelta(days=days_ahead)
    peak_time = next_weekday.replace(hour=17, minute=0, second=0)
    # Format as ISO 8601 with timezone (RFC 3339 compliant)
    departure_time = peak_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    
    # Structure for HERE Matrix Routing API
    here_matrix_url = "https://matrix.router.hereapi.com/v8/matrix"
    
    # Prepare origins and destinations
    origins = []
    destinations = []
    
    for point in test_points:
        location = {
            "lat": point["lat"],
            "lng": point["lng"]  # Note: HERE API uses "lng" not "lon"
        }
        origins.append(location)
        destinations.append(location)
    
    print(f"Using {len(origins)} test points")
    print(f"Origin points: {json.dumps(origins, indent=2)}")
    
    # Prepare request parameters
    params = {
        "apiKey": HERE_API_KEY,
        "async": "false"
    }
    
    payload = {
        "origins": origins,
        "destinations": destinations,
        "regionDefinition": {
            "type": "autoCircle"
        },
        "matrixAttributes": ["travelTimes", "distances"],
        "transportMode": "car",
        "departureTime": departure_time
    }
    
    headers = {'Content-Type': 'application/json'}
    
    try:
        print("Sending request to HERE Matrix Routing API...")
        response = requests.post(here_matrix_url, params=params, json=payload, headers=headers)
        
        print(f"Response status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            # Save raw response for debugging
            with open("debug_matrix_response.json", "w") as f:
                json.dump(result, f, indent=2)
            
                        # Get the matrix data
            matrix = result["matrix"]
            travel_times = matrix.get("travelTimes", [])
            distances = matrix.get("distances", [])
            num_origins = matrix.get("numOrigins", 0)
            num_destinations = matrix.get("numDestinations", 0)
            
            print(f"\nMatrix dimensions: {num_origins} origins × {num_destinations} destinations")
            
            # Process the flattened matrix into a more usable format
            travel_data = []
            
            for i in range(num_origins):
                for j in range(num_destinations):
                    # Calculate the index in the flattened array
                    flat_index = i * num_destinations + j
                    
                    # Ensure the index is valid
                    if flat_index < len(travel_times) and flat_index < len(distances):
                        travel_time = travel_times[flat_index]
                        distance = distances[flat_index]
                        
                        travel_data.append({
                            "origin_idx": i,
                            "dest_idx": j,
                            "travel_time": travel_time,
                            "distance": distance
                        })
            
            # Print the processed travel data
            print(f"\nExtracted {len(travel_data)} travel data entries:")
            for entry in travel_data:
                print(f"  Origin {entry['origin_idx']} to Destination {entry['dest_idx']}: " +
                      f"{entry['travel_time']} seconds, {entry['distance']} meters")
            
            return True
        else:
            print(f"API request failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"Error making matrix routing API call: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_matrix_routing()