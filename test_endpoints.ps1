# Test Filipino Dictionary API Endpoints
Write-Host "Testing Filipino Dictionary API Endpoints..." -ForegroundColor Cyan

$baseUrl = "http://localhost:5000"

# Test 1: Basic health check
Write-Host "`nTesting health endpoint..." -ForegroundColor Yellow
$response = Invoke-RestMethod -Uri "$baseUrl/health" -Method Get
Write-Host "Status: $($response.status)" -ForegroundColor Green
Write-Host "Message: $($response.message)" -ForegroundColor Green

# Test 2: Word lookup by lemma
Write-Host "`nTesting word lookup..." -ForegroundColor Yellow
$word = "aklat"
try {
    $response = Invoke-RestMethod -Uri "$baseUrl/words/$word" -Method Get
    Write-Host "Word lookup successful: $($response.word.lemma)" -ForegroundColor Green
    Write-Host "Definitions count: $($response.word.definitions.Count)" -ForegroundColor Green
} catch {
    Write-Host "Word lookup failed: $_" -ForegroundColor Red
}

# Test 3: Get definitions for a specific word
Write-Host "`nTesting word definitions endpoint..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$baseUrl/api/v2/words/$word/definitions" -Method Get
    Write-Host "Word definitions endpoint successful" -ForegroundColor Green
    Write-Host "Definitions count: $($response.definitions.Count)" -ForegroundColor Green
    
    if ($response.definitions.Count -gt 0) {
        Write-Host "Sample definition: $($response.definitions[0].definition_text)" -ForegroundColor Green
    }
} catch {
    Write-Host "Word definitions endpoint failed: $_" -ForegroundColor Red
}

# Test 4: Get definitions with query parameters
Write-Host "`nTesting definitions query endpoint..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$baseUrl/api/v2/definitions?limit=5" -Method Get
    Write-Host "Definitions query endpoint successful" -ForegroundColor Green
    Write-Host "Definitions count: $($response.definitions.Count)" -ForegroundColor Green
    Write-Host "Total count: $($response.total)" -ForegroundColor Green
    
    if ($response.definitions.Count -gt 0) {
        Write-Host "Sample definition: $($response.definitions[0].definition_text)" -ForegroundColor Green
    }
} catch {
    Write-Host "Definitions query endpoint failed: $_" -ForegroundColor Red
}

# Test 5: Search functionality
Write-Host "`nTesting search endpoint..." -ForegroundColor Yellow
$searchTerm = "libro"
try {
    $response = Invoke-RestMethod -Uri "$baseUrl/search?q=$searchTerm&limit=3" -Method Get
    Write-Host "Search endpoint successful" -ForegroundColor Green
    Write-Host "Results count: $($response.results.Count)" -ForegroundColor Green
    Write-Host "Total count: $($response.total)" -ForegroundColor Green
    
    if ($response.results.Count -gt 0) {
        Write-Host "Sample result: $($response.results[0].lemma)" -ForegroundColor Green
    }
} catch {
    Write-Host "Search endpoint failed: $_" -ForegroundColor Red
}

# Test 6: Get random word
Write-Host "`nTesting random word endpoint..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$baseUrl/random" -Method Get
    Write-Host "Random word endpoint successful" -ForegroundColor Green
    Write-Host "Random word: $($response.word.lemma)" -ForegroundColor Green
} catch {
    Write-Host "Random word endpoint failed: $_" -ForegroundColor Red
}

# Test 7: Get parts of speech
Write-Host "`nTesting parts of speech endpoint..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$baseUrl/parts_of_speech" -Method Get
    Write-Host "Parts of speech endpoint successful" -ForegroundColor Green
    Write-Host "Parts of speech count: $($response.parts_of_speech.Count)" -ForegroundColor Green
} catch {
    Write-Host "Parts of speech endpoint failed: $_" -ForegroundColor Red
}

# Test 8: Get statistics
Write-Host "`nTesting statistics endpoint..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$baseUrl/statistics" -Method Get
    Write-Host "Statistics endpoint successful" -ForegroundColor Green
    Write-Host "Total words: $($response.total_words)" -ForegroundColor Green
} catch {
    Write-Host "Statistics endpoint failed: $_" -ForegroundColor Red
}

Write-Host "`nAll tests completed." -ForegroundColor Cyan 