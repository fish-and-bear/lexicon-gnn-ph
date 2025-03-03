# PowerShell script for local development on Windows
# Requires PowerShell 5.0 or later

# Script configuration
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# Color configuration
$Colors = @{
    Success = "Green"
    Error = "Red"
    Warning = "Yellow"
    Info = "Cyan"
}

# Logging functions
function Write-Log {
    param(
        [string]$Message,
        [string]$Color = $Colors.Info
    )
    Write-Host "[$([DateTime]::Now.ToString('yyyy-MM-dd HH:mm:ss'))] $Message" -ForegroundColor $Color
}

function Write-Success {
    param([string]$Message)
    Write-Log -Message $Message -Color $Colors.Success
}

function Write-Error {
    param([string]$Message)
    Write-Log -Message "ERROR: $Message" -Color $Colors.Error
}

function Write-Warning {
    param([string]$Message)
    Write-Log -Message "WARNING: $Message" -Color $Colors.Warning
}

# Check Docker daemon
function Test-DockerDaemon {
    Write-Success "Checking Docker daemon..."
    try {
        $null = docker info
        return $true
    }
    catch {
        Write-Error "Docker daemon is not running. Please start Docker Desktop first."
        Write-Warning "If Docker Desktop is not installed, download it from: https://www.docker.com/products/docker-desktop"
        return $false
    }
}

# Check requirements
function Test-Requirements {
    Write-Success "Checking requirements..."
    
    # Check if Docker Desktop is installed
    if (-not (Get-Command "docker" -ErrorAction SilentlyContinue)) {
        Write-Error "Docker Desktop is not installed. Please install from: https://www.docker.com/products/docker-desktop"
        return $false
    }
    
    # Check if Docker daemon is running
    if (-not (Test-DockerDaemon)) {
        return $false
    }
    
    # Check other requirements
    $requirements = @(
        @{
            Name = "Docker Compose"
            Command = "docker-compose --version"
            Link = "https://docs.docker.com/compose/install/"
        },
        @{
            Name = "Python 3"
            Command = "python --version"
            Link = "https://www.python.org/downloads/"
        },
        @{
            Name = "Node.js"
            Command = "node --version"
            Link = "https://nodejs.org/"
        }
    )

    $allRequirementsMet = $true
    foreach ($req in $requirements) {
        try {
            $null = Invoke-Expression $req.Command
            Write-Success "$($req.Name) is installed"
        }
        catch {
            Write-Error "$($req.Name) is not installed"
            Write-Warning "Please install from: $($req.Link)"
            $allRequirementsMet = $false
        }
    }

    return $allRequirementsMet
}

# Setup environment
function Initialize-Environment {
    Write-Success "Setting up environment..."
    
    # Warn about data preservation
    Write-Warning "Note: Dictionary data in the data/ directory is always preserved."
    Write-Warning "Use .\dev.ps1 clean-all only if you really need to delete dictionary files."
    
    # Create local env file if it doesn't exist
    if (-not (Test-Path ".env.local")) {
        Write-Success "Creating .env.local..."
        if (Test-Path ".env.example") {
            Copy-Item ".env.example" ".env.local"
            Write-Success "Created .env.local from .env.example"
        }
        else {
            # Create default .env.local
            @"
# Application
VERSION=dev
NODE_ENV=development
FLASK_ENV=development
FLASK_DEBUG=1

# URLs
REACT_APP_API_BASE_URL=http://localhost:10000/api/v2
ALLOWED_ORIGINS=http://localhost:3000

# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=dictionary_dev
DB_USER=postgres
DB_PASSWORD=postgres
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/dictionary_dev

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=redis
REDIS_URL=redis://:redis@localhost:6379/0

# Monitoring
GRAFANA_PASSWORD=admin
"@ | Out-File -FilePath ".env.local" -Encoding utf8
            Write-Success "Created default .env.local"
        }
    }
    
    # Verify required environment variables
    $requiredVars = @(
        "REDIS_PASSWORD",
        "GRAFANA_PASSWORD",
        "DB_USER",
        "DB_PASSWORD",
        "DB_NAME"
    )
    
    $envContent = Get-Content ".env.local" | Where-Object { $_ -match '=' }
    $envVars = @{}
    $envContent | ForEach-Object {
        $key, $value = $_ -split '=', 2
        $envVars[$key.Trim()] = $value.Trim('"', "'")
    }
    
    $missingVars = $requiredVars | Where-Object { -not $envVars.ContainsKey($_) -or [string]::IsNullOrWhiteSpace($envVars[$_]) }
    if ($missingVars) {
        Write-Error "Missing required environment variables in .env.local: $($missingVars -join ', ')"
        Write-Warning "Please update .env.local with the required variables"
        return $false
    }
    
    # Create necessary directories
    @("logs") | ForEach-Object {
        if (-not (Test-Path $_)) {
            New-Item -ItemType Directory -Path $_ | Out-Null
            Write-Success "Created directory: $_"
        }
    }
    
    # Ensure data directory exists but don't clean it
    if (-not (Test-Path "data")) {
        New-Item -ItemType Directory -Path "data" | Out-Null
        Write-Success "Created directory: data"
        Write-Warning "Please add your dictionary files to the data/ directory"
    }
    
    return $true
}

# Setup database
function Initialize-Database {
    Write-Success "Setting up database..."
    
    # Load environment variables
    $envContent = Get-Content ".env.local" | Where-Object { $_ -match '=' }
    $envVars = @{}
    $envContent | ForEach-Object {
        $key, $value = $_ -split '=', 2
        $envVars[$key] = $value.Trim('"', "'")
    }
    
    # Wait for database to be ready with timeout
    $maxAttempts = 30
    $attempt = 1
    $ready = $false
    $timeout = [System.Diagnostics.Stopwatch]::StartNew()
    $maxTimeout = [TimeSpan]::FromMinutes(5)
    
    while (-not $ready -and $attempt -le $maxAttempts -and $timeout.Elapsed -lt $maxTimeout) {
        Write-Warning "Waiting for database (attempt $attempt/$maxAttempts, elapsed: $($timeout.Elapsed.ToString('mm\:ss')))..."
        try {
            $null = docker-compose -f docker-compose.local.yml exec -T db pg_isready -U $envVars.DB_USER -d $envVars.DB_NAME
            $ready = $true
        }
        catch {
            if ($timeout.Elapsed -ge $maxTimeout) {
                throw "Database initialization timed out after $($maxTimeout.TotalMinutes) minutes"
            }
            Start-Sleep -Seconds 2
            $attempt++
        }
    }
    
    if (-not $ready) {
        throw "Database failed to become ready after $maxAttempts attempts"
    }
    
    # Run migrations with timeout
    Write-Success "Running database migrations..."
    $migrationTimeout = [System.Diagnostics.Stopwatch]::StartNew()
    $maxMigrationTimeout = [TimeSpan]::FromMinutes(10)
    
    try {
        $job = Start-Job -ScriptBlock {
            docker-compose -f docker-compose.local.yml exec -T backend alembic upgrade head
        }
        
        while (-not $job.HasMoreData -and $migrationTimeout.Elapsed -lt $maxMigrationTimeout) {
            Start-Sleep -Seconds 5
        }
        
        if ($migrationTimeout.Elapsed -ge $maxMigrationTimeout) {
            Stop-Job $job
            throw "Database migrations timed out after $($maxMigrationTimeout.TotalMinutes) minutes"
        }
        
        Receive-Job $job
    }
    finally {
        Remove-Job $job -Force -ErrorAction SilentlyContinue
    }
    
    # Load initial data if needed with timeout
    if (Test-Path "backend/setup_db.sql") {
        Write-Success "Loading initial data..."
        $initTimeout = [System.Diagnostics.Stopwatch]::StartNew()
        $maxInitTimeout = [TimeSpan]::FromMinutes(5)
        
        try {
            $job = Start-Job -ScriptBlock {
                docker-compose -f docker-compose.local.yml exec -T db psql -U $envVars.DB_USER -d $envVars.DB_NAME -f /docker-entrypoint-initdb.d/setup_db.sql
            }
            
            while (-not $job.HasMoreData -and $initTimeout.Elapsed -lt $maxInitTimeout) {
                Start-Sleep -Seconds 5
            }
            
            if ($initTimeout.Elapsed -ge $maxInitTimeout) {
                Stop-Job $job
                throw "Initial data load timed out after $($maxInitTimeout.TotalMinutes) minutes"
            }
            
            Receive-Job $job
        }
        finally {
            Remove-Job $job -Force -ErrorAction SilentlyContinue
        }
    }
}

# Start development environment
function Start-DevEnvironment {
    Write-Success "Starting development environment..."
    
    # Check Docker daemon
    if (-not (Test-DockerDaemon)) {
        return $false
    }
    
    try {
        # Build and start containers
        Write-Success "Building and starting containers..."
        docker-compose -f docker-compose.local.yml up -d --build
        
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to start containers"
            return $false
        }
        
        # Setup database
        Initialize-Database
        
        # Show service status
        docker-compose -f docker-compose.local.yml ps
        
        # Show access URLs
        Write-Success "Development environment is ready!"
        @{
            "Frontend" = "http://localhost:3000"
            "Backend API" = "http://localhost:10000"
            "Grafana" = "http://localhost:3001"
            "Prometheus" = "http://localhost:9090"
            "Jaeger UI" = "http://localhost:16686"
        }.GetEnumerator() | ForEach-Object {
            Write-Host "$($_.Key): " -NoNewline -ForegroundColor $Colors.Success
            Write-Host $_.Value
        }
        
        return $true
    }
    catch {
        Write-Error "Error starting development environment: $($_.Exception.Message)"
        return $false
    }
}

# Stop development environment
function Stop-DevEnvironment {
    Write-Success "Stopping development environment..."
    docker-compose -f docker-compose.local.yml down
}

# Clean development environment
function Clear-DevEnvironment {
    Write-Success "Cleaning development environment..."
    docker-compose -f docker-compose.local.yml down -v
    
    # Only clean logs, preserve data
    if (Test-Path "logs") {
        Get-ChildItem -Path "logs" -File | Remove-Item -Force
        Write-Success "Cleaned: logs/*"
    }
    
    # Create data directory if it doesn't exist, but don't clean it
    if (-not (Test-Path "data")) {
        New-Item -ItemType Directory -Path "data" | Out-Null
        Write-Success "Created directory: data"
    }
}

# Clean development environment with data (admin only)
function Clear-DevEnvironmentWithData {
    Write-Warning "This will delete ALL data including dictionary files. Are you sure? (y/N)"
    $confirm = Read-Host
    if ($confirm -eq 'y') {
        Write-Success "Cleaning complete development environment..."
        docker-compose -f docker-compose.local.yml down -v
        
        @("logs/*") | ForEach-Object {
            if (Test-Path $_) {
                Remove-Item -Path $_ -Recurse -Force
                Write-Success "Cleaned: $_"
            }
        }
        
        if (Test-Path "data") {
            Get-ChildItem -Path "data" -File | Remove-Item -Force
            Write-Success "Cleaned: data/*"
        }
    } else {
        Write-Success "Operation cancelled"
    }
}

# Watch logs
function Watch-Logs {
    Write-Success "Watching logs..."
    docker-compose -f docker-compose.local.yml logs -f
}

# Show help
function Show-Help {
    @"
Usage: .\dev.ps1 [command]

Commands:
  start     Start development environment
  stop      Stop development environment
  clean     Clean development environment (preserves data files)
  clean-all Clean everything including dictionary data (requires confirmation)
  logs      Watch logs from all services
  help      Show this help message

Note: Dictionary data in the data/ directory is preserved by default.
      Use clean-all only if you really need to delete dictionary files.
"@ | Write-Host
}

# Main execution
try {
    $command = $args[0]
    
    if (-not $command) {
        Write-Error "No command specified"
        Show-Help
        exit 1
    }
    
    switch ($command) {
        "start" {
            if (-not (Test-Requirements)) {
                exit 1
            }
            if (-not (Initialize-Environment)) {
                exit 1
            }
            if (-not (Start-DevEnvironment)) {
                exit 1
            }
        }
        "stop" {
            if (Test-DockerDaemon) {
                Stop-DevEnvironment
            }
            else {
                Write-Warning "Docker is not running, nothing to stop"
            }
        }
        "clean" {
            if (Test-DockerDaemon) {
                Clear-DevEnvironment
            }
            else {
                Write-Warning "Docker is not running, performing partial cleanup..."
                # Only clean logs directory
                if (Test-Path "logs") {
                    Get-ChildItem -Path "logs" -File | Remove-Item -Force
                    Write-Success "Cleaned: logs/*"
                }
            }
        }
        "clean-all" {
            if ((Test-DockerDaemon)) {
                Clear-DevEnvironmentWithData
            }
            else {
                Write-Error "Docker must be running for complete cleanup"
                exit 1
            }
        }
        "logs" {
            if (-not (Test-DockerDaemon)) {
                Write-Error "Docker is not running, cannot show logs"
                exit 1
            }
            Watch-Logs
        }
        { $_ -in "help","--help","-h" } {
            Show-Help
        }
        default {
            Write-Error "Unknown command: $command"
            Show-Help
            exit 1
        }
    }
}
catch {
    Write-Error $_.Exception.Message
    exit 1
} 