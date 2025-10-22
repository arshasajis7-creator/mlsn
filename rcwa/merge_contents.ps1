# merge_contents.ps1
# Save this file in the root folder and run run_merge.bat (provided separately).
# This script:
#  - creates a tree map of the folder (as a Python triple-quote block)
#  - appends contents of script-like files (txt,json,py,md,...)
#  - writes separators/metadata in the format you requested
#  - saves the merged file as "<rootfolder> - file contents.txt" (UTF8)

# Ensure script stops on errors
$ErrorActionPreference = 'Stop'

# Root and output paths
$Root = (Get-Location).ProviderPath
$RootName = Split-Path -Leaf $Root
$OutFile = Join-Path $Root ("$RootName - file contents.txt")

# If output exists, remove it first
if (Test-Path $OutFile) {
    Remove-Item -Force $OutFile
}

# 1) Build tree map (use built-in tree.exe for Windows consoles)
try {
    $treeText = & tree /F $Root 2>&1 | Out-String
} catch {
    # fallback: simple recursive listing if tree.exe unavailable
    function Get-Tree {
        param($Path, $Prefix = "")
        Get-ChildItem -LiteralPath $Path -Force | ForEach-Object {
            if ($_.PSIsContainer) {
                "$Prefix+ $_"
                Get-Tree -Path $_.FullName -Prefix ("$Prefix    ")
            } else {
                "$Prefix- $_"
            }
        }
    }
    $treeText = Get-Tree -Path $Root | Out-String
}

# Write Python triple-quote comment block with the tree inside (UTF8)
$headerBlock = @()
$headerBlock += '"""'
$headerBlock += $treeText.TrimEnd("`r","`n")
$headerBlock += '"""'
$headerBlock += ''  # blank line

$headerBlock | Out-File -FilePath $OutFile -Encoding UTF8

# 2) Append contents of relevant files
$exts = @('.txt','.json','.py','.md','.yml','.yaml','.ini','.cfg','.csv','.tsv','.bat','.ps1','.sh','.toml','.xml','.html','.css','.js')
$allFiles = Get-ChildItem -Path $Root -Recurse -File -Force | Where-Object { $exts -contains ($_.Extension.ToLower()) } | Sort-Object FullName

$first = $true

foreach ($f in $allFiles) {
    # Skip the merged output itself if it matches an extension filter
    if ($f.FullName -ieq $OutFile) { continue }

    # Relative path from root
    $rel = $f.FullName.Substring($Root.Length).TrimStart('\','/')

    # Metadata
    $size = $f.Length
    $mtime = $f.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss')
    # Try to get file version if any (for executables/assemblies). Most text files will be null.
    try {
        $ver = (Get-Item -LiteralPath $f.FullName).VersionInfo.FileVersion
    } catch {
        $ver = $null
    }
    if ([string]::IsNullOrWhiteSpace($ver)) { $ver = 'n/a' }

    # Write separators if not first file
    if (-not $first) {
        Add-Content -LiteralPath $OutFile -Value "% The end of previous file" -Encoding UTF8
        Add-Content -LiteralPath $OutFile -Value "%█████████████████████████████████████████████████" -Encoding UTF8
        Add-Content -LiteralPath $OutFile -Value "%█████████████████████████████████████████████████" -Encoding UTF8
        Add-Content -LiteralPath $OutFile -Value "%█████████████████████████████████████████████████" -Encoding UTF8
        Add-Content -LiteralPath $OutFile -Value "%█████████████████████████████████████████████████" -Encoding UTF8
    }

    $metaLine = "% the start of the next file,"
    $metaLine2 = "%Address and name of the file from root folder: $rel | version: $ver | size: $size bytes | modified date and time: $mtime"

    Add-Content -LiteralPath $OutFile -Value $metaLine -Encoding UTF8
    Add-Content -LiteralPath $OutFile -Value $metaLine2 -Encoding UTF8

    # Read file content raw (try-best)
    try {
        $content = Get-Content -LiteralPath $f.FullName -Raw -ErrorAction Stop
        # Ensure final newline between files
        if (-not $content.EndsWith("`n")) { $content += "`r`n" }
        # Append content preserving text
        # Use Out-File -Append to control encoding
        $content | Out-File -FilePath $OutFile -Append -Encoding UTF8
    } catch {
        # If -Raw fails (binary or other), fallback to get bytes and write a hex-ish placeholder
        $errNote = "% [Could not read file as text; binary or unreadable]"
        Add-Content -LiteralPath $OutFile -Value $errNote -Encoding UTF8
    }

    $first = $false
}

# Final message to console
Write-Host "Done. Output: $OutFile"
