Write-Host 'Starting Mobile Camera Screen Detection (HTTPS)...'
Write-Host 'This version uses HTTPS for camera permissions on mobile!'
Write-Host 'Make sure you are in the project root directory'

# Optional: activate the virtual environment
& '.\screenwatch-venv\Scripts\Activate.ps1'

Set-Location -Path '.\web'
python '.\web_app_https.py'

Write-Host 'HTTPS mobile camera application stopped.'
