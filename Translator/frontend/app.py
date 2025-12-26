"""
FastAPI backend for MarianMT translation service.
"""

import json
import os
from pathlib import Path
from typing import Optional, Dict, List
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from transformers import MarianMTModel, MarianTokenizer

# Define functions first
def get_models_directory() -> Path:
    """Get the models directory path (../models relative to this script)."""
    script_dir = Path(__file__).parent.absolute()
    models_dir = script_dir.parent / "models"
    return models_dir

def get_flags_directory() -> Path:
    """Get the flags directory path (../flags relative to this script)."""
    script_dir = Path(__file__).parent.absolute()
    flags_dir = script_dir.parent / "flags"
    return flags_dir

app = FastAPI(title="MarianMT Translation Service")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Mount flags directory
flags_dir = get_flags_directory()
if flags_dir.exists():
    app.mount("/flags", StaticFiles(directory=str(flags_dir)), name="flags")


# Global models dictionary - load all at startup
loaded_models: Dict[str, Dict] = {}  # {model_name: {"model": model_obj, "tokenizer": tokenizer_obj, "path": path}}

# Current active model
current_model_name = None

# Available models cache
available_models: Dict[str, Dict] = {}

class TranslationRequest(BaseModel):
    text: str
    source_lang: Optional[str] = None
    target_lang: Optional[str] = None
    model_path: Optional[str] = None

class TranslationResponse(BaseModel):
    original_text: str
    translated_text: str
    model_path: str


def is_model_complete(model_path: Path) -> bool:
    """Check if a model directory is complete (HuggingFace format or has model files)."""
    try:
        # Check if directory only has README.md (incomplete)
        files = list(model_path.iterdir())
        if len(files) == 1 and any(f.name == "README.md" for f in files):
            return False
        
        # Check for HuggingFace format (has config.json)
        has_config = (model_path / "config.json").exists()
        has_tokenizer_config = (model_path / "tokenizer_config.json").exists()
        
        # Check for model weights (HuggingFace format)
        has_model_weights = any([
            (model_path / "pytorch_model.bin").exists(),
            (model_path / "model.safetensors").exists(),
            (model_path / "model.bin").exists(),
            (model_path / "pytorch_model.bin.index.json").exists()  # Sharded model
        ])
        
        # HuggingFace format is complete
        if has_config and has_tokenizer_config and has_model_weights:
            return True
        
        # If it has config.json, consider it complete (might work)
        if has_config:
            return True
        
        # Check for OPUS-MT format files (decoder.yml, .npz files, vocab files)
        has_decoder = (model_path / "decoder.yml").exists()
        has_npz = any(f.suffix == ".npz" for f in model_path.iterdir() if f.is_file())
        has_vocab = any([
            (model_path / "opus.bpe32k-bpe32k.vocab.yml").exists(),
            (model_path / "vocab.yml").exists(),
            (model_path / "source.bpe").exists(),
            (model_path / "target.bpe").exists(),
            (model_path / "source.spm").exists(),
            (model_path / "target.spm").exists()
        ])
        
        # OPUS-MT format - mark as complete, we'll try to load it
        # (might be in HuggingFace cache or we'll try the identifier format)
        if has_decoder or has_npz or has_vocab:
            return True
        
        return False
    except Exception as e:
        return False

def scan_available_models() -> Dict[str, Dict]:
    """Scan the models directory for folders starting with 'en-'."""
    models_dir = get_models_directory()
    available = {}
    
    if not models_dir.exists():
        print(f"Warning: Models directory not found at {models_dir}")
        return available
    
    # Find all directories starting with "en-"
    for item in models_dir.iterdir():
        if item.is_dir() and item.name.startswith("en-"):
            # Check if it's complete (has all files, not just README)
            is_complete = is_model_complete(item)
            available[item.name] = {
                "path": item,
                "complete": is_complete,
                "name": item.name
            }
    
    return available

def load_single_model(model_name: str):
    """Load a single model on demand (lazy loading)."""
    global loaded_models, available_models
    
    # Check if already loaded
    if model_name in loaded_models:
        return loaded_models[model_name]
    
    if not available_models:
        available_models.update(scan_available_models())
    
    if model_name not in available_models:
        raise FileNotFoundError(f"Model {model_name} not found in available models")
    
    info = available_models[model_name]
    
    if not info["complete"]:
        raise RuntimeError(f"Model {model_name} is incomplete (missing required files)")
    
    model_path_obj = info.get("path")
    
    # Fix path if None
    if model_path_obj is None:
        # Reconstruct path from model name
        models_dir = get_models_directory()
        model_path_obj = models_dir / model_name
        print(f"  WARNING: Path was None, reconstructed: {model_path_obj}")
    
    # Ensure it's a Path object
    if not isinstance(model_path_obj, Path):
        model_path_obj = Path(model_path_obj)
    
    if not model_path_obj.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path_obj}")
    
    model_path_str = str(model_path_obj.absolute())
    
    # Check if it's HuggingFace format (has config.json)
    has_config = (model_path_obj / "config.json").exists()
    
    print(f"  Loading {model_name} from {model_path_str}...", end=" ", flush=True)
    
    # Track error messages for better error reporting
    error_msg = None
    
    # Try loading from local directory first (HuggingFace format)
    if has_config:
        try:
            tokenizer = MarianTokenizer.from_pretrained(model_path_str, local_files_only=True)
            model = MarianMTModel.from_pretrained(model_path_str, local_files_only=True)
            
            loaded_models[model_name] = {
                "model": model,
                "tokenizer": tokenizer,
                "path": model_path_str
            }
            print("✓")
            return loaded_models[model_name]
        except Exception as e:
            error_msg = str(e)
            print(f"\n  ⚠ Local load failed: {error_msg[:60]}")
            print(f"  Trying HuggingFace identifier format...", end=" ", flush=True)
    
    # If local directory doesn't work, try HuggingFace model identifier format
    # OPUS-MT models on HuggingFace use: Helsinki-NLP/opus-mt-{src}-{tgt}
    # Extract language codes from model name (e.g., "en-fi" -> "en", "fi")
    parts = model_name.split("-", 1)
    if len(parts) == 2:
        src_lang, tgt_lang = parts[0], parts[1]
        hf_model_id = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
        
        try:
            # Try loading using HuggingFace identifier with local_files_only
            # This will work if the model is in the HuggingFace cache or local directory
            print(f"  Trying {hf_model_id}...", end=" ", flush=True)
            tokenizer = MarianTokenizer.from_pretrained(hf_model_id, local_files_only=True)
            model = MarianMTModel.from_pretrained(hf_model_id, local_files_only=True)
            
            loaded_models[model_name] = {
                "model": model,
                "tokenizer": tokenizer,
                "path": hf_model_id
            }
            print("✓")
            return loaded_models[model_name]
        except Exception as e2:
            error_msg2 = str(e2)
            print(f"✗")
            # If both fail, try loading from local path even without config.json
            # (some models might work without explicit config)
            if not has_config:
                print(f"  ⚠ Model is in OPUS-MT format. Trying direct path load...", end=" ", flush=True)
                try:
                    tokenizer = MarianTokenizer.from_pretrained(model_path_str, local_files_only=True)
                    model = MarianMTModel.from_pretrained(model_path_str, local_files_only=True)
                    
                    loaded_models[model_name] = {
                        "model": model,
                        "tokenizer": tokenizer,
                        "path": model_path_str
                    }
                    print("✓")
                    return loaded_models[model_name]
                except Exception as e3:
                    local_error = error_msg[:60] if error_msg else "N/A (no config)"
                    final_error = f"Failed to load model. Local path failed: {local_error}. HuggingFace ID failed: {error_msg2[:60]}. Direct load failed: {str(e3)[:60]}"
                    print(f"✗")
                    raise RuntimeError(final_error)
            else:
                raise RuntimeError(f"Failed to load model: {error_msg2}")
    else:
        # No config.json, try HuggingFace identifier format
        parts = model_name.split("-", 1)
        if len(parts) == 2:
            src_lang, tgt_lang = parts[0], parts[1]
            hf_model_id = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
            
            try:
                print(f"  Trying HuggingFace format: {hf_model_id}...", end=" ", flush=True)
                tokenizer = MarianTokenizer.from_pretrained(hf_model_id, local_files_only=True)
                model = MarianMTModel.from_pretrained(hf_model_id, local_files_only=True)
                
                loaded_models[model_name] = {
                    "model": model,
                    "tokenizer": tokenizer,
                    "path": hf_model_id
                }
                print("✓")
                return loaded_models[model_name]
            except Exception as e:
                error_msg = str(e)
                print(f"✗")
                raise RuntimeError(f"Model {model_name} is not in HuggingFace format and not found in local cache. Error: {error_msg[:80]}")
        else:
            raise RuntimeError(f"Model {model_name} is not in HuggingFace format (missing config.json) and model name format is invalid.")


@app.on_event("startup")
async def startup_event():
    """Scan for models when the application starts (lazy loading on demand)."""
    print("=" * 70)
    print("🌍 MarianMT Translation Service")
    print("=" * 70)
    
    # Scan for available models (but don't load them yet)
    print("\n🔍 Scanning for models...")
    available_models.update(scan_available_models())
    
    if not available_models:
        models_dir = get_models_directory()
        print(f"  ⚠ No models found in: {models_dir}")
        print(f"  Please ensure models are in the models directory.")
    else:
        print(f"  Found {len(available_models)} model directory(ies):")
        complete_count = 0
        incomplete_count = 0
        for model_name, info in sorted(available_models.items()):
            status = "✓ Complete" if info["complete"] else "✗ Incomplete"
            if info["complete"]:
                complete_count += 1
            else:
                incomplete_count += 1
            print(f"    - {model_name} [{status}]")
        
        print(f"\n  Summary: {complete_count} complete, {incomplete_count} incomplete")
        print(f"  Models will be loaded on demand when flags are clicked.")
    
    print("=" * 70)
    print("🌐 Web interface available at: http://localhost:8000")
    print("=" * 70)

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend HTML."""
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": len(loaded_models),
        "current_model": current_model_name,
        "available_models": len(available_models),
        "complete_models": sum(1 for m in available_models.values() if m["complete"])
    }

@app.post("/translate", response_model=TranslationResponse)
async def translate(request: TranslationRequest):
    """Translate text using the MarianMT model."""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    # Determine which model to use
    model_name_to_use = None
    
    # First try current_model_name (set by flag click)
    if current_model_name and current_model_name in available_models:
        model_name_to_use = current_model_name
    
    # Then try model_path from request
    if not model_name_to_use and request.model_path:
        path_str = str(request.model_path)
        # Check if it's a model name directly
        if path_str in available_models:
            model_name_to_use = path_str
        else:
            # Try to extract model name from path
            try:
                path_parts = Path(path_str).parts
                for part in reversed(path_parts):
                    if part.startswith("en-") and part in available_models:
                        model_name_to_use = part
                        break
            except:
                pass
    
    # Fallback to first available complete model
    if not model_name_to_use:
        complete_models = [name for name, info in available_models.items() if info["complete"]]
        if complete_models:
            model_name_to_use = complete_models[0]
            print(f"  No model specified, using first available: {model_name_to_use}")
    
    if not model_name_to_use:
        raise HTTPException(
            status_code=400,
            detail="No model available. Please ensure models are in the models directory."
        )
    
    try:
        # Load model if not already loaded
        if model_name_to_use not in loaded_models:
            print(f"  Loading model {model_name_to_use} for translation...")
            load_single_model(model_name_to_use)
        
        model_info = loaded_models[model_name_to_use]
        model = model_info["model"]
        tokenizer = model_info["tokenizer"]
        
        # Tokenize input
        inputs = tokenizer([request.text], return_tensors="pt", padding=True, truncation=True)
        
        # Translate
        translated = model.generate(**inputs)
        
        # Decode output
        translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
        
        return TranslationResponse(
            original_text=request.text,
            translated_text=translated_text,
            model_path=model_name_to_use
        )
    
    except Exception as e:
        error_msg = str(e)
        print(f"  Translation error: {error_msg}")
        raise HTTPException(
            status_code=500,
            detail=f"Translation failed: {error_msg}"
        )

class SelectModelRequest(BaseModel):
    model_name: str

@app.post("/api/select-model")
async def select_model_endpoint(request: SelectModelRequest):
    """Select a model (load it if needed)."""
    global current_model_name
    
    model_name = request.model_name
    print(f"\n  Flag clicked: {model_name}")
    
    if not available_models:
        available_models.update(scan_available_models())
    
    if model_name not in available_models:
        error_msg = f"Model {model_name} not found in available models"
        print(f"  ✗ {error_msg}")
        raise HTTPException(status_code=404, detail=error_msg)
    
    model_info = available_models[model_name]
    
    if not model_info["complete"]:
        error_msg = f"Model {model_name} is incomplete (missing required files or not in HuggingFace format)"
        print(f"  ✗ {error_msg}")
        raise HTTPException(status_code=400, detail=error_msg)
    
    try:
        # Load model if not already loaded
        if model_name not in loaded_models:
            print(f"  Loading model {model_name}...")
            load_single_model(model_name)
        else:
            print(f"  Model {model_name} already loaded")
        
        current_model_name = model_name
        print(f"  ✓ Model {model_name} selected and ready")
        
        return {
            "status": "success",
            "model": model_name,
            "loaded": model_name in loaded_models
        }
    except Exception as e:
        error_msg = str(e)
        print(f"  ✗ Failed to load model: {error_msg}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model: {error_msg}"
        )

@app.get("/api/models")
async def list_models():
    """List available models in the models directory (directories starting with 'en-')."""
    # Refresh available models
    available_models.update(scan_available_models())
    
    flags_dir = get_flags_directory()
    
    models_list = []
    for name, info in available_models.items():
        lang_code = info["name"].split("-")[1] if "-" in info["name"] else None
        
        # Only include if flag exists
        if not lang_code:
            continue
            
        flag_file = flags_dir / f"{lang_code}.svg"
        if not flag_file.exists():
            continue  # Skip if no flag
        
        flag_path = f"/flags/{lang_code}.svg"
        
        models_list.append({
            "name": info["name"],
            "path": str(info["path"]),
            "complete": info["complete"],
            "is_current": (current_model_name == info["name"]),
            "lang_code": lang_code,
            "flag_path": flag_path
        })
    
    # Sort: complete models first, then incomplete
    models_list.sort(key=lambda x: (not x["complete"], x["name"]))
    
    return {
        "models": models_list,
        "current_model": current_model_name,
        "loaded_models": list(loaded_models.keys()),
        "total": len(models_list),
        "complete_count": sum(1 for m in models_list if m["complete"])
    }

@app.get("/api/languages")
async def get_languages():
    """Get all available language models (directories starting with 'en-')."""
    # Refresh available models
    available_models.update(scan_available_models())
    
    languages_data = {
        "models": [],
        "language_pairs": []
    }
    
    for model_name, info in available_models.items():
        # Extract language pair from model name (e.g., "en-de" -> en, de)
        parts = model_name.split("-")
        
        source = parts[0] if len(parts) > 0 else None
        target = parts[1] if len(parts) > 1 else None
        
        languages_data["models"].append({
            "model_name": model_name,
            "model_path": str(info["path"]),
            "source_language": source,
            "target_language": target,
            "complete": info["complete"]
        })
        
        if source and target:
            languages_data["language_pairs"].append({
                "source": source,
                "target": target,
                "model_name": model_name,
                "model_path": str(info["path"]),
                "complete": info["complete"]
            })
    
    return languages_data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

