# FilRelex Database Migration Compatibility Test Script
# This script tests the API's handling of definition_metadata and missing columns

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

# Function to create a new definition
function Create-Definition {
    param (
        [string]$WordId,
        [string]$Text,
        [hashtable]$Metadata = $null,
        [string]$PartOfSpeechId = $null
    )
    
    $defBody = @{
        word_id = $WordId
        definition_text = $Text
    }
    
    if ($Metadata) {
        $defBody.definition_metadata = $Metadata
    }
    
    if ($PartOfSpeechId) {
        $defBody.part_of_speech_id = $PartOfSpeechId
    }
    
    $jsonBody = $defBody | ConvertTo-Json
    
    return Test-Endpoint -Name "Create Definition" -Endpoint "$baseUrl/api/v2/definitions" -Method "POST" -Body $jsonBody -ValidationCheck {
        param($r)
        return $r.id -ne $null
    }
}

Write-Host "Starting FilRelex Migration Compatibility Tests..." -ForegroundColor Cyan
Write-Host "Base URL: $baseUrl" -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan

# 1. Get Parts of Speech for use in testing
$posResponse = Test-Endpoint -Name "Get Parts of Speech" -Endpoint "$baseUrl/parts_of_speech" -ValidationCheck {
    param($r)
    return $r.count -gt 0
}

if (-not $posResponse) {
    Write-Host "Cannot proceed with testing - unable to retrieve parts of speech" -ForegroundColor Red
    exit
}

$posId = $posResponse.results[0].id
Write-Host "Using part of speech ID: $posId" -ForegroundColor Cyan

# 2. Create a test word to use
$testWord = "test_word_$(Get-Random)"
$wordBody = @{
    lemma = $testWord
    language_code = "fil"
} | ConvertTo-Json

$testWordResponse = Test-Endpoint -Name "Create Test Word" -Endpoint "$baseUrl/api/v2/words" -Method "POST" -Body $wordBody -ValidationCheck {
    param($r)
    return $r.id -ne $null
}

if (-not $testWordResponse) {
    Write-Host "Cannot proceed with testing - unable to create test word" -ForegroundColor Red
    exit
}

$wordId = $testWordResponse.id
Write-Host "Created test word with ID: $wordId" -ForegroundColor Cyan

# 3. Test creating definition with null metadata
$nullMetadataResponse = Create-Definition -WordId $wordId -Text "Test definition with null metadata" -PartOfSpeechId $posId

if ($nullMetadataResponse) {
    Write-Host "  - Created definition with null metadata, ID: $($nullMetadataResponse.id)" -ForegroundColor Cyan
    
    # Verify we can retrieve it
    $defId = $nullMetadataResponse.id
    $retrieveResponse = Test-Endpoint -Name "Retrieve Definition (null metadata)" -Endpoint "$baseUrl/api/v2/definitions/$defId" -ValidationCheck {
        param($r)
        return $r.id -eq $defId
    }
    
    if ($retrieveResponse) {
        $actualMetadata = $retrieveResponse.definition_metadata
        Write-Host "  - Retrieved definition, metadata type: $($actualMetadata.GetType().Name)" -ForegroundColor Cyan
        Write-Host "  - Metadata value: $($actualMetadata | ConvertTo-Json -Compress)" -ForegroundColor Cyan
    }
}

# 4. Test creating definition with empty metadata object
$emptyMetadataResponse = Create-Definition -WordId $wordId -Text "Test definition with empty metadata" -Metadata @{} -PartOfSpeechId $posId

if ($emptyMetadataResponse) {
    Write-Host "  - Created definition with empty metadata, ID: $($emptyMetadataResponse.id)" -ForegroundColor Cyan
    
    # Verify we can retrieve it
    $defId = $emptyMetadataResponse.id
    $retrieveResponse = Test-Endpoint -Name "Retrieve Definition (empty metadata)" -Endpoint "$baseUrl/api/v2/definitions/$defId" -ValidationCheck {
        param($r)
        return $r.id -eq $defId
    }
    
    if ($retrieveResponse) {
        $actualMetadata = $retrieveResponse.definition_metadata
        Write-Host "  - Retrieved definition, metadata type: $($actualMetadata.GetType().Name)" -ForegroundColor Cyan
        Write-Host "  - Metadata value: $($actualMetadata | ConvertTo-Json -Compress)" -ForegroundColor Cyan
    }
}

# 5. Test creating definition with complex metadata
$complexMetadata = @{
    etymology = @{
        origin = "Spanish"
        original_word = "libro"
    }
    usage_notes = @(
        "Formal contexts",
        "Academic settings"
    )
    examples = @(
        @{
            text = "Maganda ang aklat na iyon."
            translation = "That book is beautiful."
        },
        @{
            text = "Bumili ako ng aklat."
            translation = "I bought a book."
        }
    )
    tags = @("literature", "education", "publishing")
}

$complexMetadataResponse = Create-Definition -WordId $wordId -Text "Test definition with complex metadata" -Metadata $complexMetadata -PartOfSpeechId $posId

if ($complexMetadataResponse) {
    Write-Host "  - Created definition with complex metadata, ID: $($complexMetadataResponse.id)" -ForegroundColor Cyan
    
    # Verify we can retrieve it
    $defId = $complexMetadataResponse.id
    $retrieveResponse = Test-Endpoint -Name "Retrieve Definition (complex metadata)" -Endpoint "$baseUrl/api/v2/definitions/$defId" -ValidationCheck {
        param($r)
        return $r.id -eq $defId
    }
    
    if ($retrieveResponse) {
        $actualMetadata = $retrieveResponse.definition_metadata
        Write-Host "  - Retrieved definition, metadata type: $($actualMetadata.GetType().Name)" -ForegroundColor Cyan
        # Check if the complex metadata was preserved
        $hasEtymology = $actualMetadata.etymology -ne $null
        $hasExamples = $actualMetadata.examples -ne $null
        $examplesCount = if ($actualMetadata.examples) { $actualMetadata.examples.Count } else { 0 }
        
        Write-Host "  - Metadata has etymology: $hasEtymology" -ForegroundColor Cyan
        Write-Host "  - Metadata has examples: $hasExamples (count: $examplesCount)" -ForegroundColor Cyan
    }
}

# 6. Test the search endpoint that depends on definition_metadata
$searchResponse = Test-Endpoint -Name "Search with metadata" -Endpoint "$baseUrl/api/v2/search?q=$testWord&include_metadata=true" -ValidationCheck {
    param($r)
    return $r.count -gt 0
}

if ($searchResponse) {
    Write-Host "  - Found $($searchResponse.count) search result(s) for '$testWord' with metadata" -ForegroundColor Cyan
    if ($searchResponse.results.Count -gt 0 -and $searchResponse.results[0].definitions.Count -gt 0) {
        $hasMetadata = $searchResponse.results[0].definitions[0].definition_metadata -ne $null
        Write-Host "  - First definition has metadata: $hasMetadata" -ForegroundColor Cyan
    }
}

# 7. Test creating a definition without part_of_speech_id to test backward compatibility
$noPartOfSpeechResponse = Create-Definition -WordId $wordId -Text "Test definition without part of speech"

if ($noPartOfSpeechResponse) {
    Write-Host "  - Created definition without part of speech, ID: $($noPartOfSpeechResponse.id)" -ForegroundColor Cyan
    
    # Verify we can retrieve it
    $defId = $noPartOfSpeechResponse.id
    $retrieveResponse = Test-Endpoint -Name "Retrieve Definition (no part of speech)" -Endpoint "$baseUrl/api/v2/definitions/$defId" -ValidationCheck {
        param($r)
        return $r.id -eq $defId
    }
    
    if ($retrieveResponse) {
        $hasPosId = $retrieveResponse.part_of_speech_id -ne $null
        Write-Host "  - Retrieved definition, has part_of_speech_id: $hasPosId" -ForegroundColor Cyan
    }
}

# 8. Update a definition with new metadata
if ($nullMetadataResponse) {
    $defId = $nullMetadataResponse.id
    $updateMetadata = @{
        updated = $true
        version = 2
        notes = "This metadata was updated through the API"
    }
    
    $updateBody = @{
        definition_metadata = $updateMetadata
    } | ConvertTo-Json
    
    $updateResponse = Test-Endpoint -Name "Update Definition Metadata" -Endpoint "$baseUrl/api/v2/definitions/$defId" -Method "PUT" -Body $updateBody -ValidationCheck {
        param($r)
        return $r.id -eq $defId
    }
    
    if ($updateResponse) {
        Write-Host "  - Updated definition metadata, new value: $($updateResponse.definition_metadata | ConvertTo-Json -Compress)" -ForegroundColor Cyan
    }
}

# 9. Test an API endpoint that handles definition_links
if ($complexMetadataResponse) {
    $defId = $complexMetadataResponse.id
    $linksResponse = Test-Endpoint -Name "Get Definition Links" -Endpoint "$baseUrl/api/v2/definitions/$defId/links" -ValidationCheck {
        param($r)
        return $true  # We're just testing if it works without errors
    }
    
    if ($linksResponse) {
        Write-Host "  - Successfully retrieved definition links" -ForegroundColor Cyan
        Write-Host "  - Links count: $($linksResponse.count)" -ForegroundColor Cyan
    }
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

Write-Host "`nMigration Compatibility Testing Complete!" -ForegroundColor Cyan 