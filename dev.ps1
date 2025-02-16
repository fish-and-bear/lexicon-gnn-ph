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

# Check requirements
function Test-Requirements {
    Write-Success "Checking requirements..."
    
    $requirements = @(
        @{
            Name = "Docker Desktop"
            Command = "docker --version"
            Link = "https://www.docker.com/products/docker-desktop"
        },
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

    if (-not $allRequirementsMet) {
        throw "Missing requirements"
    }
}

# Setup environment
function Initialize-Environment {
    Write-Success "Setting up environment..."
    
    # Create local env file if it doesn't exist
    if (-not (Test-Path ".env.local")) {
        Write-Success "Creating .env.local..."
        if (Test-Path ".env.example") {
            Copy-Item ".env.example" ".env.local"
        }
        else {
            Write-Warning "No .env.example found, skipping..."
        }
    }
    
    # Create necessary directories
    @("logs", "data") | ForEach-Object {
        if (-not (Test-Path $_)) {
            New-Item -ItemType Directory -Path $_ | Out-Null
            Write-Success "Created directory: $_"
        }
    }
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
    
    # Wait for database to be ready
    $maxAttempts = 30
    $attempt = 1
    $ready = $false
    
    while (-not $ready -and $attempt -le $maxAttempts) {
        Write-Warning "Waiting for database (attempt $attempt/$maxAttempts)..."
        try {
            $null = docker-compose -f docker-compose.local.yml exec -T db pg_isready -U $envVars.DB_USER -d $envVars.DB_NAME
            $ready = $true
        }
        catch {
            Start-Sleep -Seconds 2
            $attempt++
        }
    }
    
    if (-not $ready) {
        throw "Database failed to become ready"
    }
    
    # Run migrations
    Write-Success "Running database migrations..."
    docker-compose -f docker-compose.local.yml exec -T backend alembic upgrade head
    
    # Load initial data if needed
    if (Test-Path "backend/setup_db.sql") {
        Write-Success "Loading initial data..."
        docker-compose -f docker-compose.local.yml exec -T db psql -U $envVars.DB_USER -d $envVars.DB_NAME -f /docker-entrypoint-initdb.d/setup_db.sql
    }
}

# Start development environment
function Start-DevEnvironment {
    Write-Success "Starting development environment..."
    
    # Build and start containers
    docker-compose -f docker-compose.local.yml up -d --build
    
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
    
    @("logs/*", "data/*") | ForEach-Object {
        if (Test-Path $_) {
            Remove-Item -Path $_ -Recurse -Force
            Write-Success "Cleaned: $_"
        }
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
  start    Start development environment
  stop     Stop development environment
  clean    Clean development environment (removes volumes and logs)
  logs     Watch logs from all services
  help     Show this help message
"@ | Write-Host
}

# Main execution
try {
    $command = $args[0]
    switch ($command) {
        "start" {
            Test-Requirements
            Initialize-Environment
            Start-DevEnvironment
        }
        "stop" {
            Stop-DevEnvironment
        }
        "clean" {
            Clear-DevEnvironment
        }
        "logs" {
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