#!/bin/bash

# Screen Detection - Desktop Version
echo "🖥️  Starting Desktop Screen Detection..."
echo "Make sure you're in the project root directory"

# Activate virtual environment
source screenwatch-venv/bin/activate

# Navigate to main folder and run
cd main
python main.py

echo "Desktop application closed." 