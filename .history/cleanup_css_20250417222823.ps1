# PowerShell script to clean up old CSS files after migrating to modular CSS
# Run this script only after confirming the new modular CSS structure works correctly

Write-Host "Cleaning up old component CSS files after modular CSS migration..."

# List of component CSS files to remove
$cssFilesToRemove = @(
    "src\components\WordDetails.css",
    "src\components\WordExplorer.css",
    "src\components\WordGraph.css",
    "src\components\BaybayinStatistics.css",
    "src\components\Header.css",
    "src\components\Tabs.css"
)

# Remove each file if it exists
foreach ($file in $cssFilesToRemove) {
    if (Test-Path $file) {
        Remove-Item $file
        Write-Host "Removed: $file"
    } else {
        Write-Host "File not found: $file"
    }
}

# Clean up duplicate index.css files if needed
if (Test-Path "src\index.css" -and (Test-Path "src\styles\index.css")) {
    # Keep only the src\styles\index.css version
    Remove-Item "src\index.css"
    Write-Host "Removed duplicate index.css file from src root"
}

Write-Host "CSS cleanup complete!" 