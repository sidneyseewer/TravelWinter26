#!/bin/bash
# Model Download Script for OPUS-MT Models
# Downloads and extracts models only if they don't already exist

set -e

# Get script directory (Translator folder)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Allow environment variables to override paths (useful for Docker)
LANGUAGES_JSON="${LANGUAGES_JSON:-$SCRIPT_DIR/languages.json}"
MODELS_DIR="${MODELS_DIR:-$SCRIPT_DIR/models}"
ZIPS_DIR="${ZIPS_DIR:-$SCRIPT_DIR/models-zips}"

# Check if languages.json exists
if [ ! -f "$LANGUAGES_JSON" ]; then
    echo "Error: languages.json not found at $LANGUAGES_JSON" >&2
    exit 1
fi

# Create directories if they don't exist
mkdir -p "$MODELS_DIR"
mkdir -p "$ZIPS_DIR"

echo ""
echo "Starting model download and extraction process..."
echo "Models directory: $MODELS_DIR"
echo "ZIP storage: $ZIPS_DIR"
echo ""

# Function to download and extract a model
download_and_extract() {
    local code=$1
    local url=$2
    local zip_file=$3
    local dir_name=$4
    
    local zip_path="$ZIPS_DIR/$zip_file"
    local extract_path="$MODELS_DIR/$dir_name"
    local config_path="$extract_path/config.json"
    
    echo "Processing: $dir_name"
    
    # Check if model is already extracted (has config.json)
    if [ -f "$config_path" ]; then
        echo "  ✓ Model already extracted, skipping: $dir_name"
        return 0
    fi
    
    # Check if ZIP exists, download if not
    if [ ! -f "$zip_path" ]; then
        echo "  Downloading ZIP: $url"
        if curl -L -o "$zip_path" "$url"; then
            echo "  ✓ Download complete: $zip_file"
        else
            echo "  ✗ Download failed: $zip_file"
            return 1
        fi
    else
        echo "  ✓ ZIP already exists, skipping download: $zip_file"
    fi
    
    # Extract ZIP if config.json doesn't exist (even if directory exists, it might be incomplete)
    if [ ! -f "$config_path" ]; then
        # Remove directory if it exists but is incomplete
        if [ -d "$extract_path" ]; then
            echo "  Removing incomplete extraction directory..."
            rm -rf "$extract_path"
        fi
        echo "  Extracting ZIP to: $extract_path"
        mkdir -p "$extract_path"
        
        # Extract ZIP
        if unzip -q "$zip_path" -d "$extract_path"; then
            # Some ZIPs might have a nested structure, check and flatten if needed
            extracted_items=$(ls -A "$extract_path")
            if [ $(echo "$extracted_items" | wc -l) -eq 1 ]; then
                nested_dir="$extract_path/$(echo "$extracted_items" | head -n1)"
                if [ -d "$nested_dir" ]; then
                    echo "  Flattening nested directory structure..."
                    mv "$nested_dir"/* "$extract_path"/ 2>/dev/null || true
                    rmdir "$nested_dir" 2>/dev/null || true
                fi
            fi
            echo "  ✓ Extraction complete: $dir_name"
        else
            echo "  ✗ Extraction failed: $dir_name"
            rm -rf "$extract_path"
            return 1
        fi
    fi
    
    # Verify extraction
    if [ -f "$config_path" ]; then
        echo "  ✓ Model verified: $dir_name"
    else
        echo "  ⚠ Warning: Model extracted but config.json not found: $dir_name"
    fi
    echo ""
}

# Parse languages.json and download/extract all models
# Try to use jq if available, otherwise use Python
if command -v jq &> /dev/null; then
    # Use jq to parse JSON
    model_count=$(jq '.models | length' "$LANGUAGES_JSON")
    if [ "$model_count" -eq 0 ]; then
        echo "Error: No models found in languages.json" >&2
        exit 1
    fi
    
    for i in $(seq 0 $((model_count - 1))); do
        code=$(jq -r ".models[$i].code" "$LANGUAGES_JSON")
        url=$(jq -r ".models[$i].url" "$LANGUAGES_JSON")
        zip_file=$(jq -r ".models[$i].zipFile" "$LANGUAGES_JSON")
        dir_name=$(jq -r ".models[$i].dirName" "$LANGUAGES_JSON")
        
        download_and_extract "$code" "$url" "$zip_file" "$dir_name"
    done
elif command -v python3 &> /dev/null; then
    # Use Python to parse JSON and output model data for bash to process
    # Use process substitution to avoid subshell issues
    while IFS='|' read -r code url zip_file dir_name; do
        if [ -n "$code" ] && [ -n "$url" ] && [ -n "$zip_file" ] && [ -n "$dir_name" ]; then
            download_and_extract "$code" "$url" "$zip_file" "$dir_name"
        fi
    done < <(python3 << EOF
import json
import sys
import os

languages_json = "$LANGUAGES_JSON"

try:
    with open(languages_json, 'r') as f:
        config = json.load(f)
    
    models = config.get('models', [])
    if not models:
        print("Error: No models found in languages.json", file=sys.stderr)
        sys.exit(1)
    
    for model in models:
        code = model.get('code', '')
        url = model.get('url', '')
        zip_file = model.get('zipFile', '')
        dir_name = model.get('dirName', '')
        print(f"{code}|{url}|{zip_file}|{dir_name}")
except Exception as e:
    print(f"Error parsing languages.json: {e}", file=sys.stderr)
    sys.exit(1)
EOF
    )
else
    echo "Error: Neither jq nor python3 is available. Please install one of them." >&2
    exit 1
fi

echo ""
echo "Model download and extraction process completed!"
echo "Models are available in: $MODELS_DIR"
echo "ZIP files are stored in: $ZIPS_DIR (for future use)"
echo ""

