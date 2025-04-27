$filePath = "data/kaikki-ceb.jsonl"
$sampleSize = 5000  # Analyze more entries for thorough coverage
$fieldStats = @{}
$posStats = @{}
$entriesWithBaybayin = 0
$entriesWithoutBaybayin = 0
$entriesWithBadlit = 0
$entriesWithoutBadlit = 0
$entryCount = 0
$missingFieldsEntries = @{}
$senseStats = @{}
$relationStats = @{}
$etymologyTemplateStats = @{}
$senseTotalCount = 0
$maxSensesPerEntry = 0
$entriesWithMultipleSenses = 0
$scriptStats = @{}

Write-Host "Analyzing $sampleSize entries from $filePath..."

Get-Content -Path $filePath | Select-Object -First $sampleSize | ForEach-Object {
    $entryCount++
    if ($entryCount % 100 -eq 0) {
        Write-Host "Processed $entryCount entries..." 
    }
    
    try {
        $entry = $_ | ConvertFrom-Json
        
        # Count occurrences of each top-level field
        $entry.PSObject.Properties.Name | ForEach-Object {
            if (-not $fieldStats.ContainsKey($_)) {
                $fieldStats[$_] = 0
            }
            $fieldStats[$_]++
        }
        
        # Track parts of speech
        if ($entry.pos) {
            if (-not $posStats.ContainsKey($entry.pos)) {
                $posStats[$entry.pos] = 0
            }
            $posStats[$entry.pos]++
        }
        
        # Check for script presence
        $hasBaybayin = $false
        $hasBadlit = $false
        $scriptTypes = @()
        if ($entry.forms) {
            foreach ($form in $entry.forms) {
                if ($form.tags) {
                    foreach ($tag in $form.tags) {
                        if ($tag -eq "Baybayin") {
                            $hasBaybayin = $true
                            if (-not $scriptTypes.Contains("Baybayin")) {
                                $scriptTypes += "Baybayin"
                            }
                        }
                        elseif ($tag -eq "Badlit") {
                            $hasBadlit = $true
                            if (-not $scriptTypes.Contains("Badlit")) {
                                $scriptTypes += "Badlit"
                            }
                        }
                        
                        # Track all script types for comprehensive analysis
                        if ($tag -match "script$" -or $tag -in @("Baybayin", "Badlit", "Hanunoo", "Buhid", "Tagbanwa")) {
                            if (-not $scriptStats.ContainsKey($tag)) {
                                $scriptStats[$tag] = 0
                            }
                            $scriptStats[$tag]++
                        }
                    }
                }
            }
        }
        
        if ($hasBaybayin) {
            $entriesWithBaybayin++
        } else {
            $entriesWithoutBaybayin++
        }
        
        if ($hasBadlit) {
            $entriesWithBadlit++
        } else {
            $entriesWithoutBadlit++
        }
        
        # Check for missing fields that should be handled by processor
        $fieldsToCheck = @("word", "pos", "senses", "forms", "sounds", "hyphenation", "etymology_text", "derived")
        foreach ($field in $fieldsToCheck) {
            if (-not $entry.PSObject.Properties.Name.Contains($field)) {
                if (-not $missingFieldsEntries.ContainsKey($field)) {
                    $missingFieldsEntries[$field] = 0
                }
                $missingFieldsEntries[$field]++
            }
        }
        
        # Analyze sense structure
        if ($entry.senses -and $entry.senses.Count -gt 0) {
            $senseCount = $entry.senses.Count
            $senseTotalCount += $senseCount
            
            # Track max senses per entry
            if ($senseCount -gt $maxSensesPerEntry) {
                $maxSensesPerEntry = $senseCount
            }
            
            # Count entries with multiple senses
            if ($senseCount -gt 1) {
                $entriesWithMultipleSenses++
            }
            
            # Examine sense fields
            foreach ($sense in $entry.senses) {
                if ($sense -is [System.Management.Automation.PSCustomObject]) {
                    foreach ($property in $sense.PSObject.Properties.Name) {
                        if (-not $senseStats.ContainsKey($property)) {
                            $senseStats[$property] = 0
                        }
                        $senseStats[$property]++
                    }
                    
                    # Count relation types inside senses
                    foreach ($relType in @("synonyms", "antonyms", "hypernyms", "hyponyms", "related")) {
                        if ($sense.$relType -and $sense.$relType.Count -gt 0) {
                            $relKey = "sense_$relType"
                            if (-not $relationStats.ContainsKey($relKey)) {
                                $relationStats[$relKey] = 0
                            }
                            $relationStats[$relKey]++
                        }
                    }
                }
            }
        }
        
        # Analyze top-level relation fields
        foreach ($relType in @("synonyms", "antonyms", "related", "derived", "descendants")) {
            if ($entry.$relType -and $entry.$relType.Count -gt 0) {
                $relKey = "top_$relType"
                if (-not $relationStats.ContainsKey($relKey)) {
                    $relationStats[$relKey] = 0
                }
                $relationStats[$relKey]++
            }
        }
        
        # Analyze etymology templates
        if ($entry.etymology_templates -and $entry.etymology_templates.Count -gt 0) {
            foreach ($template in $entry.etymology_templates) {
                if ($template -is [System.Management.Automation.PSCustomObject] -and $template.name) {
                    $templateName = $template.name
                    if (-not $etymologyTemplateStats.ContainsKey($templateName)) {
                        $etymologyTemplateStats[$templateName] = 0
                    }
                    $etymologyTemplateStats[$templateName]++
                }
            }
        }
        
    } catch {
        Write-Host "Error processing entry: $_"
    }
}

# Output results
Write-Host "`n==== Field Statistics ===="
$fieldStats.GetEnumerator() | Sort-Object -Property Value -Descending | ForEach-Object {
    $percentage = [math]::Round(($_.Value / $entryCount) * 100, 2)
    Write-Host "$($_.Key): $($_.Value) ($percentage%)"
}

Write-Host "`n==== Parts of Speech ===="
$posStats.GetEnumerator() | Sort-Object -Property Value -Descending | ForEach-Object {
    $percentage = [math]::Round(($_.Value / $entryCount) * 100, 2)
    Write-Host "$($_.Key): $($_.Value) ($percentage%)"
}

Write-Host "`n==== Script Statistics ===="
$baybayinPercentage = [math]::Round(($entriesWithBaybayin / $entryCount) * 100, 2)
$noBaybayinPercentage = [math]::Round(($entriesWithoutBaybayin / $entryCount) * 100, 2)
Write-Host "Entries with Baybayin script: $entriesWithBaybayin ($baybayinPercentage%)"
Write-Host "Entries without Baybayin script: $entriesWithoutBaybayin ($noBaybayinPercentage%)"

$badlitPercentage = [math]::Round(($entriesWithBadlit / $entryCount) * 100, 2)
$noBadlitPercentage = [math]::Round(($entriesWithoutBadlit / $entryCount) * 100, 2)
Write-Host "Entries with Badlit script: $entriesWithBadlit ($badlitPercentage%)"
Write-Host "Entries without Badlit script: $entriesWithoutBadlit ($noBadlitPercentage%)"

Write-Host "`n==== Script Type Details ===="
$scriptStats.GetEnumerator() | Sort-Object -Property Value -Descending | ForEach-Object {
    $percentage = [math]::Round(($_.Value / $entryCount) * 100, 2)
    Write-Host "$($_.Key): $($_.Value) ($percentage%)"
}

Write-Host "`n==== Sense Structure Statistics ===="
Write-Host "Total number of senses across all entries: $senseTotalCount"
Write-Host "Average senses per entry: $([math]::Round($senseTotalCount / $entryCount, 2))"
Write-Host "Maximum senses per entry: $maxSensesPerEntry"
Write-Host "Entries with multiple senses: $entriesWithMultipleSenses ($([math]::Round(($entriesWithMultipleSenses / $entryCount) * 100, 2))%)"

Write-Host "`n==== Sense Field Statistics ===="
$senseStats.GetEnumerator() | Sort-Object -Property Value -Descending | ForEach-Object {
    $percentage = [math]::Round(($_.Value / $senseTotalCount) * 100, 2)
    Write-Host "$($_.Key): $($_.Value) ($percentage%)"
}

Write-Host "`n==== Relation Type Statistics ===="
$relationStats.GetEnumerator() | Sort-Object -Property Value -Descending | ForEach-Object {
    $percentage = [math]::Round(($_.Value / $entryCount) * 100, 2)
    Write-Host "$($_.Key): $($_.Value) ($percentage%)"
}

Write-Host "`n==== Etymology Template Statistics ===="
$etymologyTemplateStats.GetEnumerator() | Sort-Object -Property Value -Descending | Select-Object -First 15 | ForEach-Object {
    $percentage = [math]::Round(($_.Value / $entryCount) * 100, 2)
    Write-Host "$($_.Key): $($_.Value) ($percentage%)"
}

if ($missingFieldsEntries.Count -gt 0) {
    Write-Host "`n==== Missing Fields ===="
    $missingFieldsEntries.GetEnumerator() | Sort-Object -Property Value -Descending | ForEach-Object {
        $percentage = [math]::Round(($_.Value / $entryCount) * 100, 2)
        Write-Host "$($_.Key): Missing in $($_.Value) entries ($percentage%)"
    }
}

# Analyze a specific entry in detail
Write-Host "`n==== Detailed Analysis of First Entry ===="
$firstEntry = Get-Content -Path $filePath -TotalCount 1 | ConvertFrom-Json

# Output general information
Write-Host "Word: $($firstEntry.word)"
Write-Host "POS: $($firstEntry.pos)"
Write-Host "Language: $($firstEntry.lang) ($($firstEntry.lang_code))"

# Check for forms
if ($firstEntry.forms) {
    Write-Host "`nForms:"
    foreach ($form in $firstEntry.forms) {
        Write-Host "  - $($form.form) (Tags: $($form.tags -join ', '))"
    }
}

# Check for sounds
if ($firstEntry.sounds) {
    Write-Host "`nSounds:"
    foreach ($sound in $firstEntry.sounds) {
        $soundInfo = $sound | ConvertTo-Json -Compress
        Write-Host "  - $soundInfo"
    }
}

# Check for senses
if ($firstEntry.senses) {
    Write-Host "`nSenses: $($firstEntry.senses.Count)"
    foreach ($sense in $firstEntry.senses) {
        Write-Host "  - ID: $($sense.id)"
        if ($sense.glosses) {
            Write-Host "    Glosses: $($sense.glosses -join '; ')"
        }
        if ($sense.tags) {
            Write-Host "    Tags: $($sense.tags -join ', ')"
        }
        if ($sense.examples) {
            Write-Host "    Examples: $($sense.examples.Count)"
            foreach ($example in $sense.examples | Select-Object -First 2) {
                Write-Host "      - $($example.text)"
            }
            if ($sense.examples.Count -gt 2) {
                Write-Host "      - ... and $($sense.examples.Count - 2) more"
            }
        }
    }
}

# Check for derived terms
if ($firstEntry.derived) {
    Write-Host "`nDerived Terms: $($firstEntry.derived.Count)"
    foreach ($term in $firstEntry.derived | Select-Object -First 5) {
        Write-Host "  - $($term.word)"
    }
    if ($firstEntry.derived.Count -gt 5) {
        Write-Host "  - ... and $($firstEntry.derived.Count - 5) more"
    }
}

# Check for etymology templates if present
if ($firstEntry.etymology_templates) {
    Write-Host "`nEtymology Templates: $($firstEntry.etymology_templates.Count)"
    foreach ($template in $firstEntry.etymology_templates) {
        Write-Host "  - Name: $($template.name)"
        if ($template.args) {
            Write-Host "    Args: $($template.args | ConvertTo-Json -Compress)"
        }
    }
}

Write-Host "`nAnalysis complete." 