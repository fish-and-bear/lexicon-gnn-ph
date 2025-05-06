<#
  Summarize-LogErrors.ps1
  Streams huge logs, groups identical errors, prints each just once.

  ▸ Console:  “### signature” + first traceback line
  ▸ -OutFile: full traceback for that single representative block
#>

param(
    [Parameter(Mandatory)][string]$LogPath,
    [string]$OutFile
)

if (-not (Test-Path $LogPath)) {
    Write-Error "Log '$LogPath' not found."; exit 1
}

# ─── Regex helpers ────────────────────────────────────────────────────────────
$errHeaderRegex = '^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - ERROR -'
$exceptionLine  = '^(?<exc>[A-Za-z0-9_]+Error: .+)$'  # last line of traceback

# ─── Data stores ──────────────────────────────────────────────────────────────
$seen      = @{}    # canonicalSig → List[string] (first traceback example)
$inBlock   = $false
$block     = [System.Collections.Generic.List[string]]::new()

# ─── Helper: canonicalize a signature so near‑duplicates collapse ─────────────
function CanonicalizeSig([string]$sig) {
    $sig.Trim().ToLower() -replace '\s+', ' '   # squeeze whitespace, lower‑case
}

# ─── Helper: flush one block into $seen if new ───────────────────────────────
function FlushBlock {
    param([hashtable]$Seen, [Collections.Generic.List[string]]$Block)

    if (-not $Block.Count) { return }

    # 1. Try classic “XError: …” line
    $sig = $null
    for ($i = $Block.Count-1; $i -ge 0; $i--) {
        if ($Block[$i] -match $exceptionLine) {
            $sig = $matches.exc
            break
        }
    }

    # 2. Fallback to message after “- ERROR -”
    if (-not $sig -and $Block[0] -match '- ERROR - (?<msg>.+)$') {
        $sig = $matches.msg
    }
    if (-not $sig) { $sig = '[unknown]' }

    # Canonical signature to avoid duplicates that differ only by spacing/case
    $canon = CanonicalizeSig $sig

    if (-not $Seen.ContainsKey($canon)) {
        # first time we meet this signature → keep its full traceback
        $Seen[$canon] = $Block.ToArray()   # copy, don’t reference
    }

    $Block.Clear()
}

# ─── Stream read (never loads entire file) ───────────────────────────────────
Get-Content $LogPath -ReadCount 1000 | ForEach-Object {
    foreach ($line in $_) {
        if ($line -match $errHeaderRegex) {
            if ($inBlock) { FlushBlock -Seen $seen -Block $block }
            $inBlock = $true
        }
        if ($inBlock) { [void]$block.Add($line) }
    }
}
if ($inBlock) { FlushBlock -Seen $seen -Block $block }

# ─── Output unique signatures ────────────────────────────────────────────────
$writer = if ($OutFile) {
    [System.IO.StreamWriter]::new($OutFile, $false, [Text.UTF8Encoding]::new())
}

foreach ($pair in $seen.GetEnumerator() | Sort-Object Key) {
    $sigBlock = $pair.Value
    $header   = "### $($sigBlock[0] -replace '.* - ERROR - ','').Trim()"

    Write-Host $header -ForegroundColor Cyan
    if ($writer) { $writer.WriteLine($header) }

    # first traceback line after header (skip header itself)
    $intro = ($sigBlock | Select-Object -Skip 1 | Select-Object -First 1)
    Write-Host "    $intro"
    Write-Host

    if ($writer) {
        $sigBlock | ForEach-Object { $writer.WriteLine($_) }
        $writer.WriteLine(('-'*80))
    }
}

if ($writer) {
    $writer.Dispose()
    Write-Host "`nSaved $($seen.Count) unique error(s) to '$OutFile'."
}
