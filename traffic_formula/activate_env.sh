#!/bin/bash
# Activation script for the traffic signal formula discovery project

# Activate virtual environment
source "/Users/samnu/Documents/SUMO/traffic_formula/venv/bin/activate"

# Set SUMO_HOME
export SUMO_HOME="/opt/homebrew/opt/sumo/share/sumo"

# Add SUMO tools to PYTHONPATH
export PYTHONPATH="$PYTHONPATH:$SUMO_HOME/tools"

echo "Project environment activated!"
echo "  - Virtual environment: /Users/samnu/Documents/SUMO/traffic_formula/venv"
echo "  - SUMO_HOME: $SUMO_HOME"
echo "  - Project directory: /Users/samnu/Documents/SUMO/traffic_formula"
echo ""
echo "To run the project:"
echo "  cd /Users/samnu/Documents/SUMO/traffic_formula"
echo "  python run_all.py         # Run without GUI (faster)"
echo "  python run_all.py --gui   # Run with GUI visualization"
