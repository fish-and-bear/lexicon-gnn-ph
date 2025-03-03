# PowerShell script to start the application
# Handles directory navigation and npm start

# Navigate to the parent directory if needed
$currentPath = (Get-Location).Path
if ($currentPath.EndsWith("fil-relex")) {
    Write-Host "Navigating to parent directory..." -ForegroundColor Cyan
    Set-Location ..
}

# Start the application
Write-Host "Starting the application..." -ForegroundColor Green
npm start 