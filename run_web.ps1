#!/bin/bash

# Screen Detection - Mobile Camera Version
Write-Host "Starting Mobile Camera Screen Detection..."
Write-Host "This version uses your PHONE'S camera for detection!"
Write-Host "Make sure you're in the project root directory"

# Activate virtual environment
& '.\screenwatch-venv\Scripts\Activate.ps1'

# Navigate to web folder and run mobile app
Set-Location -Path '.\web'
python '.\web_app.py'

Write-Host "Mobile camera application stopped." 