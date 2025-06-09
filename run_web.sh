#!/bin/bash

# Screen Detection - Mobile Camera Version
echo "📱 Starting Mobile Camera Screen Detection..."
echo "This version uses your PHONE'S camera for detection!"
echo "Make sure you're in the project root directory"

# Activate virtual environment
source screenwatch-venv/bin/activate

# Navigate to web folder and run mobile app
cd web
python web_app.py

echo "Mobile camera application stopped." 