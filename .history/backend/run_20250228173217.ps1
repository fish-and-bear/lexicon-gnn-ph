# Set environment variables
$env:PYTHONPATH = (Get-Location).Path
$env:FLASK_APP = "app.py"
$env:FLASK_ENV = "development"
$env:FLASK_DEBUG = 1
$env:DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/fil_dict_db"
$env:REDIS_ENABLED = "false"

Write-Host "Starting Flask development server..."
Write-Host "Environment variables set:"
Write-Host "PYTHONPATH: $env:PYTHONPATH"
Write-Host "FLASK_APP: $env:FLASK_APP"
Write-Host "FLASK_ENV: $env:FLASK_ENV"
Write-Host "DATABASE_URL: $env:DATABASE_URL"

# Start the Flask server
python -m flask run --host=0.0.0.0 --port=10000 