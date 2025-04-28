#region Script Configuration
$ErrorActionPreference = "Stop" # Stop on most errors

# --- File Paths ---
$kaikkiTlFile = "data/kaikki.jsonl"
$kaikkiCebFile = "data/kaikki-ceb.jsonl"

# --- Baybayin Config ---
$targetLangCodeTl = "tl"
$targetScriptTagBaybayin = "Baybayin" # Tag used in 'forms'
$targetScriptNameBaybayin = "Baybayin" # Name used in 'scripts' or head_templates

# --- Badlit Config ---
$targetLangCodeCeb = "ceb"
$targetScriptTagBadlit = "Badlit" # Assumed tag used in 'forms' for Cebuano
$targetScriptNameBadlit = "Badlit" # Assumed name used in 'scripts' or head_templates

#endregion

#region Helper Function (Simplified Script Extraction)

function Get-ScriptInfo {
    param(
        [Parameter(Mandatory=$true)]
        [psobject]$Entry,

        [Parameter(Mandatory=$true)]
        [string]$TargetScriptTag, # e.g., "Baybayin" or "Badlit"

        [Parameter(Mandatory=$true)]
        [string]$TargetScriptName # e.g., "Baybayin" or "Badlit"
    )

    $scriptForm = $null
    $romanized = $null
    $foundIn = $null

    # 1. Check 'forms' array (Primary location)
    if ($Entry.PSObject.Properties.Name -contains 'forms' -and $Entry.forms -is [array]) {
        foreach ($form_data in $Entry.forms) {
            if ($form_data -is [psobject] -and
                $form_data.PSObject.Properties.Name -contains 'tags' -and
                $form_data.tags -is [array] -and
                $form_data.tags -contains $TargetScriptTag) {

                $formText = $null
                if ($form_data.PSObject.Properties.Name -contains 'form') {
                    $formText = $form_data.form -as [string]
                }

                if (-not [string]::IsNullOrWhiteSpace($formText)) {
                    # Basic cleanup (remove potential prefixes - adapt as needed)
                    $cleanedForm = $formText.Trim()
                    # Example prefix removal (less robust than Python version)
                    if ($cleanedForm.ToLower().StartsWith("spelling ")) { $cleanedForm = $cleanedForm.Substring("spelling ".Length).TrimStart() }
                    if ($cleanedForm.ToLower().StartsWith("script ")) { $cleanedForm = $cleanedForm.Substring("script ".Length).TrimStart() }
                    if ($cleanedForm.ToLower().StartsWith($TargetScriptTag.ToLower() + " ")) { $cleanedForm = $cleanedForm.Substring(($TargetScriptTag + " ").Length).TrimStart() }

                    # Basic validation: Check for non-ASCII characters (indicative of script)
                    if ($cleanedForm -match '[^\x00-\x7F]') {
                        $scriptForm = $cleanedForm
                        # Get explicit romanization if present
                        if ($form_data.PSObject.Properties.Name -contains 'roman') { $romanized = $form_data.roman }
                        if (-not $romanized -and $form_data.PSObject.Properties.Name -contains 'romanization') { $romanized = $form_data.romanization }
                        $foundIn = "forms"
                        break # Found in forms, stop checking other forms for this entry
                    }
                }
            }
        }
    }

    # 2. Fallback: Check 'head_templates' (More complex, simplified check)
    if (-not $scriptForm -and $Entry.PSObject.Properties.Name -contains 'head_templates' -and $Entry.head_templates -is [array]) {
       foreach ($template in $Entry.head_templates) {
           if ($template -is [psobject] -and $template.PSObject.Properties.Name -contains 'expansion') {
               $expansion = $template.expansion -as [string]
               if ($expansion) {
                   # Simple pattern match (less precise than Python regex)
                   # Escape the script name for the regex pattern
                   $escapedScriptName = [System.Text.RegularExpressions.Regex]::Escape($TargetScriptName)
                   # Refined pattern based on observed format: (Badlit spelling <chars>)
                   # Need to escape parentheses for PowerShell regex
                   $pattern = "\($escapedScriptName spelling\s+([\u1700-\u171F\s]+)\)" # Match U+1700 range chars and spaces inside parens
                   if ($expansion -match $pattern) {
                        $matchesValue = $matches[1].Trim() # Capture match before validation check overwrites $matches
                        # Basic validation
                        if (-not [string]::IsNullOrWhiteSpace($matchesValue) -and $matchesValue -match '[\u1700-\u171F]') {
                            $scriptForm = $matchesValue
                            $romanized = $null # Romanization unlikely here
                            $foundIn = "head_templates"
                            break # Found in templates
                        }
                   }
               }
           }
       }
    }

     # 3. Fallback: Check top-level 'scripts' array (Less common)
    if (-not $scriptForm -and $Entry.PSObject.Properties.Name -contains 'scripts' -and $Entry.scripts -is [array]) {
         foreach ($script_entry in $Entry.scripts) {
            if ($script_entry -is [psobject] -and
                $script_entry.PSObject.Properties.Name -contains 'name' -and
                $script_entry.name -eq $TargetScriptName) { # Match script name

                if($script_entry.PSObject.Properties.Name -contains 'text'){
                     $scriptForm = $script_entry.text
                }
                if($script_entry.PSObject.Properties.Name -contains 'roman'){
                     $romanized = $script_entry.roman
                }
                $foundIn = "scripts"
                break # Found in scripts array
            }
         }
    }

    # Return result as a hashtable or PSObject
    if ($scriptForm) {
        return [PSCustomObject]@{
            ScriptForm = $scriptForm
            Romanized  = $romanized
            FoundIn    = $foundIn
        }
    } else {
        return $null
    }
}

#endregion

#region Main Processing Logic

$allFoundScripts = [System.Collections.Generic.List[PSObject]]::new()
$totalLinesProcessed = 0
$jsonErrorCount = 0

# --- Process Baybayin (Tagalog) ---
Write-Host "Analyzing Baybayin script info in '$kaikkiTlFile' for lang '$targetLangCodeTl'..."
if (-not (Test-Path $kaikkiTlFile)) {
    Write-Warning "File not found: '$kaikkiTlFile'. Skipping Baybayin analysis."
} else {
    try {
        # Use explicit UTF-8 encoding for reading
        $streamReader = [System.IO.StreamReader]::new($kaikkiTlFile, [System.Text.Encoding]::UTF8)
        while (($line = $streamReader.ReadLine()) -ne $null) {
            $totalLinesProcessed++
            if (-not [string]::IsNullOrWhiteSpace($line)) {
                try {
                    $entry = $line | ConvertFrom-Json -ErrorAction Stop

                    # Check if it's the target language
                    if ($entry.PSObject.Properties.Name -contains 'lang_code' -and $entry.lang_code -eq $targetLangCodeTl) {
                        $word = $entry.word
                        $scriptResult = Get-ScriptInfo -Entry $entry -TargetScriptTag $targetScriptTagBaybayin -TargetScriptName $targetScriptNameBaybayin

                        if ($scriptResult) {
                            $allFoundScripts.Add([PSCustomObject]@{
                                File       = (Get-Item $kaikkiTlFile).Name # Store only filename
                                Word       = $word
                                Lang       = $entry.lang_code
                                ScriptName = $targetScriptNameBaybayin
                                ScriptForm = $scriptResult.ScriptForm
                                Romanized  = $scriptResult.Romanized
                                FoundIn    = $scriptResult.FoundIn
                            })
                        }
                    }
                } catch [System.Text.Json.JsonException] {
                    $jsonErrorCount++
                    Write-Verbose "Skipping line $($totalLinesProcessed) due to JSON parsing error: $($_.Exception.Message)" # Use Verbose
                } catch {
                     Write-Warning "Skipping line $($totalLinesProcessed) due to unexpected error: $($_.Exception.Message)"
                }
            }
            # Progress indicator
            if ($totalLinesProcessed % 10000 -eq 0) { Write-Host "..processed $totalLinesProcessed lines from $($kaikkiTlFile)" -NoNewline; Write-Host "`r" -NoNewline }
        }
        $streamReader.Close()
        Write-Host "`nFinished processing $kaikkiTlFile ($totalLinesProcessed lines)"
    } catch {
        Write-Error "An error occurred processing '$kaikkiTlFile': $($_.Exception.Message)"
        if ($streamReader -ne $null -and -not $streamReader.EndOfStream) { $streamReader.Close() } # Ensure closure on error
    }
}


# --- Process Badlit (Cebuano) ---
Write-Host "`nAnalyzing Badlit script info in '$kaikkiCebFile' for lang '$targetLangCodeCeb'..."
$cebLinesProcessed = 0
if (-not (Test-Path $kaikkiCebFile)) {
    Write-Warning "File not found: '$kaikkiCebFile'. Skipping Badlit analysis."
} else {
     try {
        # Use explicit UTF-8 encoding for reading
        $streamReaderCeb = [System.IO.StreamReader]::new($kaikkiCebFile, [System.Text.Encoding]::UTF8)
        $currentTotalLines = $totalLinesProcessed # Track total lines before starting this file
        while (($line = $streamReaderCeb.ReadLine()) -ne $null) {
            $cebLinesProcessed++
            $totalLinesProcessed++
            if (-not [string]::IsNullOrWhiteSpace($line)) {
                try {
                    $entry = $line | ConvertFrom-Json -ErrorAction Stop

                    # Check if it's the target language
                    if ($entry.PSObject.Properties.Name -contains 'lang_code' -and $entry.lang_code -eq $targetLangCodeCeb) {
                        $word = $entry.word
                        $scriptResult = Get-ScriptInfo -Entry $entry -TargetScriptTag $targetScriptTagBadlit -TargetScriptName $targetScriptNameBadlit

                         if ($scriptResult) {
                            $allFoundScripts.Add([PSCustomObject]@{
                                File       = (Get-Item $kaikkiCebFile).Name # Store only filename
                                Word       = $word
                                Lang       = $entry.lang_code
                                ScriptName = $targetScriptNameBadlit
                                ScriptForm = $scriptResult.ScriptForm
                                Romanized  = $scriptResult.Romanized
                                FoundIn    = $scriptResult.FoundIn
                            })
                        }
                    }
                } catch [System.Text.Json.JsonException] {
                    $jsonErrorCount++
                     Write-Verbose "Skipping line $($cebLinesProcessed) (CEB) due to JSON parsing error: $($_.Exception.Message)" # Use Verbose
                } catch {
                     Write-Warning "Skipping line $($cebLinesProcessed) (CEB) due to unexpected error: $($_.Exception.Message)"
                }
            }
             # Progress indicator
            if ($cebLinesProcessed % 10000 -eq 0) { Write-Host "..processed $cebLinesProcessed lines from $($kaikkiCebFile)" -NoNewline; Write-Host "`r" -NoNewline }
        }
        $streamReaderCeb.Close()
        Write-Host "`nFinished processing $kaikkiCebFile ($cebLinesProcessed lines)"
    } catch {
        Write-Error "An error occurred processing '$kaikkiCebFile': $($_.Exception.Message)"
         if ($streamReaderCeb -ne $null -and -not $streamReaderCeb.EndOfStream) { $streamReaderCeb.Close() } # Ensure closure on error
    }
}


# --- Output Results ---
# Use a consistent format for better parsing later if needed
Write-Host "`n--- ANALYSIS SUMMARY ---"
Write-Host "Total Lines Processed: $totalLinesProcessed"
Write-Host "JSON Parsing Errors: $jsonErrorCount"
Write-Host "Script Entries Found: $($allFoundScripts.Count)"

if ($allFoundScripts.Count -gt 0) {
    Write-Host "`n--- FOUND SCRIPT DETAILS ---"
    # Output as JSON for easier parsing by other tools or myself
    $allFoundScripts | ConvertTo-Json -Depth 5
} else {
    Write-Host "No relevant script information (Baybayin or Badlit) found in the specified files."
}

Write-Host "`n--- SCRIPT ANALYSIS FINISHED ---"

#endregion 