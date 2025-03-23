# PowerShell script to process RPG files and generate RAFT datasets

# Default parameters
$docsDir = "$env:USERPROFILE\train_source"
$outputDir = "$env:USERPROFILE\train_output"
$docType = "md"
$questions = 5
$distractors = 3
$pValue = 0.8
$chunkSize = 1024
$qgModel = "google/flan-t5-large"
$cotModel = "google/flan-t5-xl"
$qaModel = "deepset/deberta-v3-large-squad2"
$pythonEnv = $null  # Default to no specific environment

# Process command line arguments
for ($i = 0; $i -lt $args.Count; $i++) {
    switch ($args[$i]) {
        "--docs-dir" {
            $docsDir = $args[++$i]
        }
        "--output-dir" {
            $outputDir = $args[++$i]
        }
        "--questions" {
            $questions = $args[++$i]
        }
        "--distractors" {
            $distractors = $args[++$i]
        }
        "--p" {
            $pValue = $args[++$i]
        }
        "--chunk-size" {
            $chunkSize = $args[++$i]
        }
        "--qg-model" {
            $qgModel = $args[++$i]
        }
        "--cot-model" {
            $cotModel = $args[++$i]
        }
        "--qa-model" {
            $qaModel = $args[++$i]
        }
        "--python-env" {
            $pythonEnv = $args[++$i]
        }
        default {
            Write-Host "Unknown parameter: $($args[$i])"
            exit 1
        }
    }
}

# Create output directory if it doesn't exist
if (!(Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir | Out-Null
}

# Check if directory exists
if (!(Test-Path $docsDir)) {
    Write-Host "Error: Directory $docsDir does not exist."
    exit 1
}

# Activate virtual environment if specified
if ($pythonEnv) {
    $activateScript = Join-Path $pythonEnv "Scripts\Activate.ps1"
    if (Test-Path $activateScript) {
        Write-Host "Activating Python virtual environment: $pythonEnv"
        & $activateScript
    }
    else {
        Write-Host "Error: Virtual environment activation script not found at $activateScript"
        exit 1
    }
}

# Create a log file
$logFile = Join-Path $outputDir "processing_log.txt"
"Starting processing at $(Get-Date)" | Out-File -FilePath $logFile

# Track completion status
"Processing Status:" | Out-File -FilePath $logFile -Append

# Create a combined JSONL file for all results
$combinedJsonl = Join-Path $outputDir "combined_raft_data.jsonl"
"" | Out-File -FilePath $combinedJsonl

# Function to get file extension
function Get-FileExtension {
    param (
        [string]$filePath
    )
    return [System.IO.Path]::GetExtension($filePath).TrimStart('.')
}

# Function to process a file
function Invoke-FileProcessing {
    param (
        [string]$file
    )
    
    $filename = Split-Path $file -Leaf
    $filenameNoExt = [System.IO.Path]::GetFileNameWithoutExtension($file)
    $extension = Get-FileExtension -filePath $file
    
    # Set doctype based on file extension
    switch ($extension) {
        "md" { $doctype = "md" }
        "txt" { $doctype = "txt" }
        "pdf" { $doctype = "pdf" }
        "json" { $doctype = "json" }
        default {
            Write-Host "Unsupported file type: $extension for file $filename"
            return $false
        }
    }
    
    $outputPath = Join-Path $outputDir $filenameNoExt
    $jsonlPath = "$outputPath.jsonl"
    
    Write-Host "Processing $filename (doctype: $doctype)"
    Write-Host "  - Output: $outputPath"
        
    # Run the script
    $pythonScript = Join-Path $PSScriptRoot "raft_training_data_gen.py"
    $pythonExe = if ($pythonEnv) { Join-Path $pythonEnv "Scripts\python.exe" } else { "python" }
    & $pythonExe $pythonScript `
        --datapath "$file" `
        --output "$outputPath" `
        --doctype "$doctype" `
        --questions "$questions" `
        --distractors "$distractors" `
        --p "$pValue" `
        --chunk_size "$chunkSize" `
        --qg-model "$qgModel" `
        --cot-model "$cotModel" `
        --qa-model "$qaModel"
    
    $status = $LASTEXITCODE
    if ($status -eq 0) {
        "  ${filename}: Completed successfully at $(Get-Date)" | Out-File -FilePath $logFile -Append
        
        # Append content to combined JSONL if file exists
        if (Test-Path $jsonlPath) {
            Get-Content $jsonlPath | Out-File -FilePath $combinedJsonl -Append
            Write-Host "  - Added to combined JSONL file"
            return $true
        } else {
            Write-Host "  - Warning: JSONL file not found at $jsonlPath"
            return $false
        }
    } else {
        "  ${filename}: Failed with status $status at $(Get-Date)" | Out-File -FilePath $logFile -Append
        return $false
    }
}

# Process each file in the directory
Write-Host "Starting to process files in $docsDir"
$fileCount = 0
$successCount = 0

Get-ChildItem -Path $docsDir -File | ForEach-Object {
    $fileCount++
    $success = Invoke-FileProcessing -file $_.FullName
    if ($success) {
        $successCount++
    }
}

# Log completion
Write-Host "Processing complete. Processed $successCount/$fileCount files successfully."
"Processed $successCount/$fileCount files successfully at $(Get-Date)" | Out-File -FilePath $logFile -Append
Write-Host "Combined RAFT data saved to: $combinedJsonl"

# Optional: Shuffle the combined JSONL for better training
if (Test-Path $combinedJsonl) {
    $fileContent = Get-Content -Path $combinedJsonl
    if ($fileContent.Count -gt 0) {
        Write-Host "Shuffling combined data..."
        $tempFile = Join-Path $outputDir "temp_combined.jsonl"
        $fileContent | Get-Random -Count $fileContent.Count | Out-File -FilePath $tempFile
        Move-Item -Path $tempFile -Destination $combinedJsonl -Force
        Write-Host "Combined data shuffled successfully."
    }
}

Write-Host "All processing complete."

# Deactivate virtual environment if it was activated
if ($pythonEnv) {
    # In PowerShell, deactivate is available as a function after activation
    if (Get-Command deactivate -ErrorAction SilentlyContinue) {
        deactivate
        Write-Host "Python virtual environment deactivated."
    }
}