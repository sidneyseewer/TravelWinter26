# Model Download Script for OPUS-MT Models
# Downloads and extracts models only if they don't already exist

$ErrorActionPreference = "Stop"

# Get script directory (Translator folder)
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$languagesJsonPath = Join-Path $scriptDir "languages.json"

# Load model definitions from languages.json
if (-not (Test-Path $languagesJsonPath)) {
    Write-Host "Error: languages.json not found at $languagesJsonPath" -ForegroundColor Red
    exit 1
}

try {
    $languagesConfig = Get-Content $languagesJsonPath -Raw | ConvertFrom-Json
    $models = $languagesConfig.models
    if ($null -eq $models -or $models.Count -eq 0) {
        Write-Host "Error: No models found in languages.json" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "Error: Failed to parse languages.json: $_" -ForegroundColor Red
    exit 1
}
$modelsDir = Join-Path $scriptDir "models"
$zipsDir = Join-Path $scriptDir "models-zips"

# Create directories if they don't exist
if (-not (Test-Path $modelsDir)) {
    New-Item -ItemType Directory -Path $modelsDir | Out-Null
    Write-Host "Created models directory: $modelsDir" -ForegroundColor Green
}

if (-not (Test-Path $zipsDir)) {
    New-Item -ItemType Directory -Path $zipsDir | Out-Null
    Write-Host "Created ZIP storage directory: $zipsDir" -ForegroundColor Green
}

Write-Host "`nStarting model download and extraction process..." -ForegroundColor Cyan
Write-Host "Models directory: $modelsDir" -ForegroundColor Gray
Write-Host "ZIP storage: $zipsDir`n" -ForegroundColor Gray

foreach ($model in $models) {
    $zipPath = Join-Path $zipsDir $model.zipFile
    $extractPath = Join-Path $modelsDir $model.dirName
    $configPath = Join-Path $extractPath "config.json"
    
    Write-Host "Processing: $($model.dirName)" -ForegroundColor Yellow
    
    # Check if model is already extracted
    if (Test-Path $configPath) {
        Write-Host "  ✓ Model already extracted, skipping: $($model.dirName)" -ForegroundColor Green
        continue
    }
    
    # Check if ZIP exists, download if not
    if (-not (Test-Path $zipPath)) {
        Write-Host "  Downloading ZIP: $($model.url)" -ForegroundColor Cyan
        try {
            $ProgressPreference = 'SilentlyContinue'
            Invoke-WebRequest -Uri $model.url -OutFile $zipPath -UseBasicParsing
            Write-Host "  ✓ Download complete: $($model.zipFile)" -ForegroundColor Green
        } catch {
            Write-Host "  ✗ Download failed: $_" -ForegroundColor Red
            continue
        }
    } else {
        Write-Host "  ✓ ZIP already exists, skipping download: $($model.zipFile)" -ForegroundColor Green
    }
    
    # Extract ZIP if model directory doesn't exist
    if (-not (Test-Path $extractPath)) {
        Write-Host "  Extracting ZIP to: $extractPath" -ForegroundColor Cyan
        $extractionSuccess = $false
        try {
            # Create target directory
            New-Item -ItemType Directory -Path $extractPath -Force | Out-Null
            
            # Extract ZIP
            Expand-Archive -Path $zipPath -DestinationPath $extractPath -Force
            
            # Some ZIPs might have a nested structure, check and flatten if needed
            $extractedItems = Get-ChildItem -Path $extractPath
            if ($extractedItems.Count -eq 1 -and $extractedItems[0].PSIsContainer) {
                # Single directory inside, likely the model name - flatten it
                $nestedDir = $extractedItems[0].FullName
                Write-Host "  Flattening nested directory structure..." -ForegroundColor Gray
                Get-ChildItem -Path $nestedDir | Move-Item -Destination $extractPath -Force
                Remove-Item -Path $nestedDir -Force
            }
            
            Write-Host "  ✓ Extraction complete: $($model.dirName)" -ForegroundColor Green
            $extractionSuccess = $true
        } catch {
            Write-Host "  ✗ Extraction failed: $_" -ForegroundColor Red
            # Clean up partial extraction
            if (Test-Path $extractPath) {
                Remove-Item -Path $extractPath -Recurse -Force -ErrorAction SilentlyContinue
            }
        }
        
        if (-not $extractionSuccess) {
            continue
        }
    } else {
        Write-Host "  ✓ Model directory already exists, skipping extraction: $($model.dirName)" -ForegroundColor Green
    }
    
    # Verify extraction
    if (Test-Path $configPath) {
        Write-Host "  ✓ Model verified: $($model.dirName)`n" -ForegroundColor Green
    } else {
        Write-Host "  ⚠ Warning: Model extracted but config.json not found: $($model.dirName)`n" -ForegroundColor Yellow
    }
}

Write-Host "`nModel download and extraction process completed!" -ForegroundColor Cyan
Write-Host "Models are available in: $modelsDir" -ForegroundColor Gray
Write-Host "ZIP files are stored in: $zipsDir (for future use)`n" -ForegroundColor Gray

