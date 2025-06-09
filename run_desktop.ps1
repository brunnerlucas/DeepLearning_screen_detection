# Screen Detection - Desktop Version
Write-Host "Starting Desktop Screen Detection..."
Write-Host "Make sure you're in the project root directory"

# Activate virtual environment
& '.\screenwatch-venv\Scripts\Activate.ps1'

# Navigate to web folder and run mobile app
Set-Location -Path '.\main'
python 'main.py'

Write-Host "Desktop application closed." 