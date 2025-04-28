# FilRelex API Endpoint Testing Script
# This script tests the main API endpoints of the FilRelex Dictionary API

$baseUrl = "http://localhost:5000"
$testResults = @()

function Test-Endpoint {
    param (
        [string]$Name,
        [string]$Endpoint,
        [string]$Method = "GET",
        [string]$Body = "",
        [scriptblock]$ValidationCheck = { $true }
    )
    
    Write-Host "Testing: $Name ($Endpoint)..." -ForegroundColor Yellow
    
    try {
        $params = @{
            Uri = $Endpoint
            Method = $Method
            ContentType = "application/json"
            ErrorAction = "Stop"
        }
        
        if ($Method -in "POST", "PUT" -and $Body) {
            $params.Body = $Body
        }
        
        $response = Invoke-RestMethod @params
        $isValid = & $ValidationCheck $response
        
        if ($isValid) {
            Write-Host "✅ SUCCESS: $Name" -ForegroundColor Green
            $testResults += @{Name = $Name; Status = "Success"; Response = $response}
            return $response
        } else {
            Write-Host "❌ VALIDATION FAILED: $Name" -ForegroundColor Red
            $testResults += @{Name = $Name; Status = "Validation Failed"; Response = $response}
            return $null
        }
    } catch {
        Write-Host "❌ ERROR: $Name - $($_.Exception.Message)" -ForegroundColor Red
        $testResults += @{Name = $Name; Status = "Error: $($_.Exception.Message)"; Response = $null}
        return $null
    }
}

Write-Host "Starting FilRelex API Tests..." -ForegroundColor Cyan
Write-Host "Base URL: $baseUrl" -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan

# 1. Test health endpoint
$health = Test-Endpoint -Name "Health Check" -Endpoint "$baseUrl/health" -ValidationCheck {
    param($r)
    return $r.status -eq "ok"
}

# 2. Test word lookup for "aklat"
$word = "aklat"
$wordResponse = Test-Endpoint -Name "Word Lookup" -Endpoint "$baseUrl/words/$word" -ValidationCheck {
    param($r)
    return $r.lemma -eq $word
}

# 3. Test word definitions v2 API
$wordDefResponse = Test-Endpoint -Name "Word Definitions (v2)" -Endpoint "$baseUrl/api/v2/words/$word/definitions" -ValidationCheck {
    param($r)
    return $r.count -gt 0
}

if ($wordDefResponse) {
    Write-Host "  - Found $($wordDefResponse.count) definition(s) for '$word'" -ForegroundColor Cyan
    Write-Host "  - First definition: $($wordDefResponse.results[0].definition_text)" -ForegroundColor Cyan
}

# 4. Test definitions query with parameters
$defsResponse = Test-Endpoint -Name "Definitions Query" -Endpoint "$baseUrl/api/v2/definitions?limit=5" -ValidationCheck {
    param($r)
    return $r.count -gt 0 -and $r.results.Count -le 5
}

if ($defsResponse) {
    Write-Host "  - Found $($defsResponse.count) total definitions (showing first 5)" -ForegroundColor Cyan
}

# 5. Test search functionality
$searchTerm = "libro"
$searchResponse = Test-Endpoint -Name "Search" -Endpoint "$baseUrl/search?q=$searchTerm&limit=3" -ValidationCheck {
    param($r)
    return $r.count -gt 0
}

if ($searchResponse) {
    Write-Host "  - Found $($searchResponse.count) search result(s) for '$searchTerm'" -ForegroundColor Cyan
    if ($searchResponse.count -gt 0) {
        Write-Host "  - First result: $($searchResponse.results[0].lemma)" -ForegroundColor Cyan
    }
}

# 6. Test v2 search API
$searchV2Response = Test-Endpoint -Name "Search v2" -Endpoint "$baseUrl/api/v2/search?q=$searchTerm&limit=3" -ValidationCheck {
    param($r)
    return $r.count -gt 0
}

if ($searchV2Response) {
    Write-Host "  - Found $($searchV2Response.count) search result(s) for '$searchTerm' in v2 API" -ForegroundColor Cyan
}

# 7. Test random word
$randomResponse = Test-Endpoint -Name "Random Word" -Endpoint "$baseUrl/random" -ValidationCheck {
    param($r)
    return $r.lemma -ne $null
}

if ($randomResponse) {
    Write-Host "  - Random word: $($randomResponse.lemma)" -ForegroundColor Cyan
}

# 8. Test parts of speech
$posResponse = Test-Endpoint -Name "Parts of Speech" -Endpoint "$baseUrl/parts_of_speech" -ValidationCheck {
    param($r)
    return $r.count -gt 0
}

if ($posResponse) {
    Write-Host "  - Found $($posResponse.count) parts of speech" -ForegroundColor Cyan
}

# 9. Test statistics
$statsResponse = Test-Endpoint -Name "Statistics" -Endpoint "$baseUrl/statistics" -ValidationCheck {
    param($r)
    return $r.word_count -ne $null
}

if ($statsResponse) {
    Write-Host "  - Dictionary contains $($statsResponse.word_count) words" -ForegroundColor Cyan
}

# 10. Test the /api/v2/words endpoint with pagination
$wordsResponse = Test-Endpoint -Name "Words List (v2)" -Endpoint "$baseUrl/api/v2/words?limit=10" -ValidationCheck {
    param($r)
    return $r.count -gt 0 -and $r.results.Count -le 10
}

if ($wordsResponse) {
    Write-Host "  - Found $($wordsResponse.count) total words (showing first 10)" -ForegroundColor Cyan
}

# Summary
Write-Host "`n===============================================" -ForegroundColor Cyan
Write-Host "Test Summary:" -ForegroundColor Cyan
$successCount = ($testResults | Where-Object { $_.Status -eq "Success" }).Count
$failureCount = ($testResults | Where-Object { $_.Status -ne "Success" }).Count

Write-Host "Total Tests: $($testResults.Count)" -ForegroundColor White
Write-Host "Successful:  $successCount" -ForegroundColor Green
Write-Host "Failed:      $failureCount" -ForegroundColor Red

if ($failureCount -gt 0) {
    Write-Host "`nFailed Tests:" -ForegroundColor Red
    $testResults | Where-Object { $_.Status -ne "Success" } | ForEach-Object {
        Write-Host "- $($_.Name): $($_.Status)" -ForegroundColor Red
    }
}

Write-Host "`nAPI Testing Complete!" -ForegroundColor Cyan 