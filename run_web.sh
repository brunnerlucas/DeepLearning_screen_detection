#!/bin/bash

# Screen Detection - Web Version
echo "📱 Starting Web Screen Detection..."
echo "Make sure you're in the project root directory"

# Activate virtual environment
source screenwatch-venv/bin/activate

# Navigate to web folder and run
cd web
python web_app.py

echo "Web application stopped." 