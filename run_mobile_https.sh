#!/bin/bash

# Screen Detection - Mobile Camera Version (HTTPS)
echo "🔒📱 Starting Mobile Camera Screen Detection (HTTPS)..."
echo "This version uses HTTPS for camera permissions on mobile!"
echo "Make sure you're in the project root directory"

# Activate virtual environment
source screenwatch-venv/bin/activate

# Navigate to web folder and run HTTPS mobile app
cd web
python web_app_https.py

echo "HTTPS mobile camera application stopped." 