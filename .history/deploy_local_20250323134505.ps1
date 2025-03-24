# Function to check if a command exists
function Test-Command($cmdname) {
    return [bool](Get-Command -Name $cmdname -ErrorAction SilentlyContinue)
}

Write-Host "Starting local deployment..."

# Check prerequisites
Write-Host "Checking prerequisites..."

if (-not (Test-Command "psql")) {
    Write-Host "PostgreSQL is not installed. Please install PostgreSQL first."
    exit 1
}

if (-not (Test-Command "python")) {
    Write-Host "Python is not installed. Please install Python first."
    exit 1
}

if (-not (Test-Command "npm")) {
    Write-Host "Node.js/npm is not installed. Please install Node.js first."
    exit 1
}

# Setup database
Write-Host "Setting up database..."
$env:PGPASSWORD = "postgres"  # Set PostgreSQL password
# Only create database if it doesn't exist
$dbExists = psql -U postgres -tAc "SELECT 1 FROM pg_database WHERE datname='fil_dict_db'"
if (-not $dbExists) {
    Write-Host "Database does not exist, creating it..."
    psql -U postgres -c "CREATE DATABASE fil_dict_db OWNER postgres;"
}

# Setup backend
Write-Host "Setting up backend..."
Set-Location backend
if (-not (Test-Path "venv")) {
    python -m venv venv
}
.\venv\Scripts\Activate

# Install requirements
Write-Host "Installing Python dependencies..."
python -m pip install --upgrade pip
pip install -r requirements.txt

# Run migrations with recreate option
Write-Host "Running database migrations with recreate option..."
python migrate.py --recreate

# Set PYTHONPATH
$env:PYTHONPATH = "$PWD;$env:PYTHONPATH"

# Run tests
Write-Host "Running tests..."
python -m pytest tests/

# Start backend server
Write-Host "Starting backend server..."
$backendProcess = Start-Process -FilePath "python" -ArgumentList "app.py" -NoNewWindow -PassThru
Set-Location ..

# Setup frontend
Write-Host "Setting up frontend..."

# Create package.json if it doesn't exist
if (-not (Test-Path "package.json")) {
    @{
        name = "word-relationship-webapp"
        version = "0.1.0"
        private = $true
        scripts = @{
            start = "react-scripts start"
            build = "react-scripts build"
            test = "react-scripts test"
            eject = "react-scripts eject"
        }
    } | ConvertTo-Json | Set-Content "package.json"
}

# Install dependencies
npm install --legacy-peer-deps
npm install --save-dev @babel/plugin-proposal-private-property-in-object --legacy-peer-deps

# Update browserslist
Write-Host "Updating browserslist..."
npx update-browserslist-db@latest

# Build and start frontend
Write-Host "Building frontend..."
npm run build

Write-Host "Starting frontend server..."
$frontendProcess = Start-Process -FilePath "cmd" -ArgumentList "/c npm start" -NoNewWindow -PassThru

Write-Host "Deployment complete!"
Write-Host "Backend running on http://localhost:10000"
Write-Host "Frontend running on http://localhost:3000"
Write-Host "Press Enter to stop the servers..."

# Wait for user input
$null = Read-Host

# Cleanup
if ($backendProcess) { 
    try {
        Stop-Process -Id $backendProcess.Id -ErrorAction SilentlyContinue
    } catch {
        Write-Host "Backend process already stopped"
    }
}

if ($frontendProcess) { 
    try {
        Stop-Process -Id $frontendProcess.Id -Force -ErrorAction SilentlyContinue
        Get-Process -Name "node" | Where-Object { $_.MainWindowTitle -eq "" } | Stop-Process -Force
    } catch {
        Write-Host "Frontend process already stopped"
    }
}

deactivate

Write-Host "Servers stopped." 