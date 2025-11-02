# Manhattan Traffic Simulation: Deployment and Model Training Guide

This guide explains how to deploy the Manhattan traffic simulation and use it for model training. 

## Prerequisites

- macOS with M3 chip
- Python 3.6 or higher
- SUMO installed at `/opt/homebrew/bin/sumo`
- HERE API key
- Required Python packages: `numpy`, `pandas`, `requests`

## Deployment Steps

1. **Create Project Directory**

```bash
mkdir -p /Users/samnu/Documents/SUMO/HERE_API_Manhattan196
cd /Users/samnu/Documents/SUMO/HERE_API_Manhattan196
```

2. **Save the Manhattan Traffic Script**

Save the full `manhattan_traffic.py` script to the project directory.

3. **Run the Script**

```bash
cd /Users/samnu/Documents/SUMO/HERE_API_Manhattan196
python3 -m venv venv
source venv/bin/activate
pip install numpy pandas requests
which sumo
python manhattan_traffic.py
```

During the first run, the script will:
- Download OpenStreetMap data for Manhattan
- Convert it to SUMO network format
- Request traffic data from HERE API (you'll need to enter your API key)
- Generate traffic demand based on real-world data
- Create all necessary configuration files

4. **Verify Installation**

After successful setup, you should see the following files in your directory:
- `manhattan.osm`
- `manhattan.net.xml`
- `manhattan_vtypes.xml`
- `manhattan_flows.xml`
- `manhattan_sim.sumocfg`
- Various data files (`traffic_flow_data.json`, `processed_traffic_data.json`, etc.)
- Utility scripts (`convert_to_roundabouts.py`, `analyze_results.py`)

## Running Simulations

### Baseline Simulation

Run the baseline simulation with traffic lights:

```bash
# Run with GUI (for visualization)
sumo-gui -c manhattan_sim.sumocfg

# Run without GUI (for faster processing)
sumo -c manhattan_sim.sumocfg
```
# Normal run - will use existing files if they exist
python manhattan_traffic.py

# Force regenerate all files
python manhattan_traffic.py --force-regenerate

<!-- parser.add_argument('--skip-api-calls', action='store_true', 
                    help='Skip HERE API calls and use estimates instead')
parser.add_argument('--skip-simulation', action='store_true',
                    help='Generate files but skip running the simulation') -->



### Roundabout Conversion Simulations

For testing roundabout conversions:

```bash
# Convert specific intersections to roundabouts
python convert_to_roundabouts.py --junction-id [JUNCTION_ID]

# Convert 5 random intersections
python convert_to_roundabouts.py --random 5

# Convert all suitable intersections
python convert_to_roundabouts.py --all
```

Then run the modified simulation:

```bash
sumo-gui -c manhattan_roundabouts.sumocfg
```

## Analyzing Results for Model Training

After each simulation run, analyze the results:

```bash
python analyze_results.py
```

This will generate a `metrics.json` file with comprehensive traffic metrics including:
- Average speeds
- Travel times
- Wait times
- Vehicle throughput
- Per-intersection metrics

To create visualizations: 
```bash
python analyze_results.py --visualize
```


For comparing different strategies:

```bash
python analyze_results.py --compare metrics.json --output-file new_metrics.json
```

## Model Training Workflow

### 1. Generate Training Data

Run multiple simulations with different configurations:

```bash
# Run baseline simulation and collect metrics
sumo -c manhattan_sim.sumocfg
python analyze_results.py --output-file baseline_metrics.json

# Run with 5 random roundabouts and collect metrics
python convert_to_roundabouts.py --random 5
sumo -c manhattan_roundabouts.sumocfg
python analyze_results.py --output-file random5_metrics.json

# Run with 10 random roundabouts and collect metrics
python convert_to_roundabouts.py --random 10
sumo -c manhattan_roundabouts.sumocfg
python analyze_results.py --output-file random10_metrics.json

# Continue with more variations...
```

### 2. Prepare Features for Model Training

Create a dataset from the simulation results:

```python
import json
import pandas as pd

# Load all results
baseline = json.load(open('baseline_metrics.json'))
random5 = json.load(open('random5_metrics.json'))
random10 = json.load(open('random10_metrics.json'))

# Extract features for each intersection
def extract_intersection_features(metrics_file, converted=False):
    data = json.load(open(metrics_file))
    features = []
    
    for intersection in data['intersections']:
        features.append({
            'junction_id': intersection['junction_id'],
            'avg_speed': intersection.get('avg_speed', 0),
            'throughput': intersection.get('throughput', 0),
            'avg_waiting_time': intersection.get('avg_waiting_time', 0),
            'converted_to_roundabout': converted
        })
    
    return features

# Create training dataset
training_data = extract_intersection_features('baseline_metrics.json', False)
training_data.extend(extract_intersection_features('random5_metrics.json', True))
training_data.extend(extract_intersection_features('random10_metrics.json', True))

# Save as CSV
pd.DataFrame(training_data).to_csv('training_data.csv', index=False)
```

### 3. Train a Model

Train a simple model to predict which intersections would benefit from roundabout conversion:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load data
df = pd.read_csv('training_data.csv')

# Prepare features and target
X = df[['avg_speed', 'throughput', 'avg_waiting_time']]
y = df['converted_to_roundabout']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

# Save model
import pickle
with open('roundabout_prediction_model.pkl', 'wb') as f:
    pickle.dump(model, f)
```

### 4. Use Model to Optimize Traffic

Use the trained model to identify which intersections would benefit most from roundabout conversion:

```python
import pickle
import json
import pandas as pd

# Load model and baseline data
with open('roundabout_prediction_model.pkl', 'rb') as f:
    model = pickle.load(f)
    
baseline = json.load(open('baseline_metrics.json'))

# Extract features for prediction
features = []
junction_ids = []

for intersection in baseline['intersections']:
    junction_ids.append(intersection['junction_id'])
    features.append([
        intersection.get('avg_speed', 0),
        intersection.get('throughput', 0),
        intersection.get('avg_waiting_time', 0)
    ])

# Make predictions
X = pd.DataFrame(features, columns=['avg_speed', 'throughput', 'avg_waiting_time'])
predictions = model.predict_proba(X)[:, 1]  # Probability of benefiting from roundabout

# Rank junctions by prediction
results = list(zip(junction_ids, predictions))
results.sort(key=lambda x: x[1], reverse=True)

# Print top candidates for roundabout conversion
print("Top 10 intersections for roundabout conversion:")
for junction_id, score in results[:10]:
    print(f"Junction {junction_id}: {score:.4f} probability of improvement")

# Save optimized list
with open('optimized_roundabouts.txt', 'w') as f:
    for junction_id, score in results[:10]:
        f.write(f"{junction_id}\n")
```

### 5. Validate Optimized Solution

Test the model's recommendations:

```bash
# Convert the top 10 predicted intersections
python convert_to_roundabouts.py --junction-id $(head -n 1 optimized_roundabouts.txt) \
                               --junction-id $(head -n 2 optimized_roundabouts.txt | tail -n 1) \
                               # ... and so on for all 10

# Run simulation with optimized roundabouts
sumo -c manhattan_roundabouts.sumocfg

# Analyze results
python analyze_results.py --compare baseline_metrics.json --output-file optimized_metrics.json
```

## Using Model Results for HERE API Optimization

Since your main goal is to optimize traffic flow based on HERE API data, you can extract the key metrics from your model and map them back to the real-world parameters available in the HERE API:

1. **Average Speed**: This directly maps to the speed data in HERE Traffic API
2. **Travel Time**: Maps to travel time data in Matrix Routing API
3. **Congestion Levels**: Maps to jam factor in HERE Traffic API

By understanding which intersections benefit most from modifications, you can develop a model that recommends real-world traffic optimizations based solely on HERE API data, without needing the full SUMO simulation in production.

## Troubleshooting

### Common Issues

1. **HERE API Key Issues**
   - Error: "Failed to get traffic flow data"
   - Solution: Verify your HERE API key is valid and has the necessary permissions

2. **SUMO Path Issues**
   - Error: "SUMO command not found"
   - Solution: Update the SUMO path in the script to match your installation

3. **Missing Python Packages**
   - Error: "ModuleNotFoundError"
   - Solution: Install required packages: `pip install numpy pandas requests`

4. **Simulation Errors**
   - Error: "Error during netconvert"
   - Solution: Check the downloaded OSM data for completeness

### Getting Help

For more assistance:
- SUMO Documentation: https://sumo.dlr.de/docs/
- HERE API Documentation: https://developer.here.com/documentation
