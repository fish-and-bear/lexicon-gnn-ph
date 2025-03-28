Write-Host "Analyzing Kaikki files..."
$tagalogEntries = Get-Content -Path data/kaikki.jsonl -TotalCount 100 | ForEach-Object { $_ | ConvertFrom-Json -ErrorAction SilentlyContinue }
