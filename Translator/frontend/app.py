"""
FastAPI backend for MarianMT translation service.
Simplified and Linux-compatible version.
"""

import json
import io
import hashlib
import wave
from pathlib import Path
from typing import Optional, Dict, List
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from transformers import MarianMTModel, MarianTokenizer
import numpy as np
import onnxruntime as ort
import subprocess
import shutil

# Path helpers
def get_base_dir() -> Path:
    """Get the base directory (parent of frontend)."""
    return Path(__file__).parent.parent.absolute()

def get_models_directory() -> Path:
    """Get the models directory path."""
    return get_base_dir() / "models"

def get_flags_directory() -> Path:
    """Get the flags directory path."""
    return get_base_dir() / "flags"

def get_voices_directory() -> Path:
    """Get the voices directory path."""
    return get_base_dir() / "voices"

app = FastAPI(title="MarianMT Translation Service")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Mount flags directory
flags_dir = get_flags_directory()
if flags_dir.exists():
    app.mount("/flags", StaticFiles(directory=str(flags_dir)), name="flags")

# Global state
loaded_models: Dict[str, Dict] = {}
current_model_name: Optional[str] = None
available_models: Dict[str, Dict] = {}
available_voices: Dict[str, List[Dict]] = {}
loaded_voice_models: Dict[str, Dict] = {}
audio_cache: Dict[str, bytes] = {}

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
    """Check if a model directory is complete."""
    if not model_path.exists() or not model_path.is_dir():
        return False
    
    # Skip if only README.md exists
    files = list(model_path.iterdir())
    if len(files) == 1 and any(f.name == "README.md" for f in files):
        return False
    
    # Check for HuggingFace format
    has_config = (model_path / "config.json").exists()
    has_weights = any([
        (model_path / "pytorch_model.bin").exists(),
        (model_path / "model.safetensors").exists(),
        (model_path / "model.bin").exists(),
        (model_path / "pytorch_model.bin.index.json").exists()
    ])
    
    if has_config and has_weights:
        return True
    
    # Check for OPUS-MT format indicators
    has_decoder = (model_path / "decoder.yml").exists()
    has_npz = any(f.suffix == ".npz" for f in model_path.iterdir() if f.is_file())
    
    return has_decoder or has_npz

def scan_available_models() -> Dict[str, Dict]:
    """Scan the models directory for folders starting with 'en-'."""
    models_dir = get_models_directory()
    available = {}
    
    if not models_dir.exists():
        return available
    
    for item in models_dir.iterdir():
        if item.is_dir() and item.name.startswith("en-"):
            # Ensure path is always an absolute Path object
            model_path = item.absolute() if isinstance(item, Path) else Path(item).absolute()
            available[item.name] = {
                "path": model_path,
                "complete": is_model_complete(model_path),
                "name": item.name
            }
    
    return available

def scan_available_voices(lang_code: Optional[str] = None) -> Dict[str, List[Dict]]:
    """Scan the voices directory for available voices.
    
    Args:
        lang_code: If provided, only scan voices for this language. If None, scan all languages.
    """
    voices_dir = get_voices_directory()
    voices_dict = {}
    
    if not voices_dir.exists():
        return voices_dict
    
    # Determine which languages to scan
    if lang_code:
        # Only scan the specified language
        lang_dirs = [voices_dir / lang_code] if (voices_dir / lang_code).exists() else []
    else:
        # Scan all languages
        lang_dirs = [d for d in voices_dir.iterdir() if d.is_dir()]
    
    # Structure: voices/{lang_code}/{locale}/{voice_name}/{quality}/
    for lang_dir in lang_dirs:
        if not lang_dir.is_dir():
            continue
        
        current_lang_code = lang_dir.name
        voices_list = []
        
        for locale_dir in lang_dir.iterdir():
            if not locale_dir.is_dir():
                continue
            
            for voice_name_dir in locale_dir.iterdir():
                if not voice_name_dir.is_dir():
                    continue
                
                voice_name = voice_name_dir.name
                
                for quality_dir in voice_name_dir.iterdir():
                    if not quality_dir.is_dir():
                        continue
                    
                    quality = quality_dir.name
                    onnx_files = list(quality_dir.glob("*.onnx"))
                    
                    for onnx_file in onnx_files:
                        json_file = onnx_file.with_suffix(".onnx.json")
                        if not json_file.exists():
                            continue
                        
                        # Check if ONNX file is valid (not a Git LFS pointer)
                        # Skip Git LFS pointers - they won't work and just clutter the UI
                        try:
                            file_size = onnx_file.stat().st_size
                            if file_size < 1024:  # Less than 1KB is suspicious
                                with open(onnx_file, 'rb') as f:
                                    first_bytes = f.read(100)
                                    if b'version https://git-lfs.github.com/spec/v1' in first_bytes:
                                        # Skip Git LFS pointers - they're not actual files
                                        print(f"  Skipping {onnx_file.name}: Git LFS pointer ({file_size} bytes) - file not downloaded")
                                        continue
                        except (OSError, IOError) as e:
                            print(f"  Warning: Cannot check {onnx_file.name}: {e}")
                            # Continue anyway - let the ONNX runtime handle validation
                        
                        voice_key = onnx_file.stem
                        
                        try:
                            with open(json_file, 'r', encoding='utf-8') as f:
                                config = json.load(f)
                            
                            voices_list.append({
                                "key": voice_key,
                                "name": voice_name,
                                "quality": quality,
                                "locale": locale_dir.name,
                                "onnx_path": str(onnx_file),
                                "json_path": str(json_file),
                                "config": config,
                                "display_name": f"{voice_name} ({quality})"
                            })
                        except Exception as e:
                            print(f"Warning: Failed to load voice config {json_file}: {e}")
                            continue
        
        if voices_list:
            voices_dict[current_lang_code] = voices_list
    
    return voices_dict

def load_voice_model(voice_key: str, lang_code: str) -> Dict:
    """Load a voice ONNX model."""
    global loaded_voice_models
    
    if voice_key in loaded_voice_models:
        return loaded_voice_models[voice_key]
    
    # Find the voice in available voices - only scan for this language
    if lang_code not in available_voices:
        scanned = scan_available_voices(lang_code=lang_code)
        available_voices.update(scanned)
    
    if lang_code not in available_voices:
        raise FileNotFoundError(f"No voices found for language: {lang_code}")
    
    voice_info = None
    for voice in available_voices[lang_code]:
        if voice["key"] == voice_key:
            voice_info = voice
            break
    
    if not voice_info:
        raise FileNotFoundError(f"Voice {voice_key} not found for language {lang_code}")
    
    onnx_path = voice_info["onnx_path"]
    config = voice_info["config"]
    
    # Check if ONNX file exists and is valid
    onnx_file = Path(onnx_path)
    if not onnx_file.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
    
    # Check if file is a Git LFS pointer - detect and provide helpful error
    try:
        file_size = onnx_file.stat().st_size
        if file_size < 1024:  # Suspiciously small
            with open(onnx_file, 'rb') as f:
                first_bytes = f.read(200)
                if b'version https://git-lfs.github.com/spec/v1' in first_bytes:
                    # Extract the expected file size from the LFS pointer
                    try:
                        lfs_content = first_bytes.decode('utf-8')
                        size_line = [l for l in lfs_content.split('\n') if l.startswith('size ')][0]
                        expected_size = int(size_line.split()[1])
                        expected_size_mb = expected_size / (1024 * 1024)
                        oid_line = [l for l in lfs_content.split('\n') if l.startswith('oid sha256:')][0]
                        file_hash = oid_line.split()[1]
                        
                        # Check if files are tracked in git
                        import subprocess
                        try:
                            result = subprocess.run(
                                ['git', 'ls-files', str(onnx_file)],
                                capture_output=True,
                                text=True,
                                timeout=2
                            )
                            is_tracked = bool(result.stdout.strip())
                        except:
                            is_tracked = False
                        
                        if is_tracked:
                            error_msg = (
                                f"ONNX file is a Git LFS pointer (file not downloaded).\n"
                                f"Current size: {file_size} bytes\n"
                                f"Expected size: {expected_size_mb:.1f} MB\n\n"
                                f"To fix this:\n"
                                f"1. Make sure Git LFS is installed: sudo apt-get install git-lfs\n"
                                f"2. Initialize Git LFS in the repo: git lfs install\n"
                                f"3. Fetch and checkout LFS files: git lfs fetch && git lfs checkout\n"
                                f"   Or pull specific file: git lfs pull --include='{onnx_file.name}'\n\n"
                                f"File path: {onnx_path}"
                            )
                        else:
                            error_msg = (
                                f"ONNX file is a Git LFS pointer (file not downloaded).\n"
                                f"Current size: {file_size} bytes\n"
                                f"Expected size: {expected_size_mb:.1f} MB\n"
                                f"File hash: {file_hash[:16]}...\n\n"
                                f"This file is not tracked in the git repository.\n"
                                f"The actual ONNX file needs to be downloaded manually.\n"
                                f"These files are typically large (~10-70MB each) and need to be\n"
                                f"downloaded from the original source where you obtained the voice models.\n\n"
                                f"File path: {onnx_path}\n\n"
                                f"Note: This is not a Linux or protobuf compatibility issue.\n"
                                f"The file simply hasn't been downloaded yet."
                            )
                        raise RuntimeError(error_msg)
                    except (ValueError, IndexError, UnicodeDecodeError):
                        raise RuntimeError(
                            f"ONNX file is a Git LFS pointer (file not downloaded). "
                            f"File size: {file_size} bytes (expected ~10-70MB). "
                            f"Please install Git LFS and run: git lfs pull --include='{onnx_file.name}'"
                        )
    except RuntimeError:
        raise  # Re-raise our custom error
    except (UnicodeDecodeError, PermissionError, OSError):
        # If we can't read it, continue - might be binary or valid
        pass
    
    # Load ONNX model
    try:
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    except Exception as e:
        error_msg = str(e)
        # Check if it's a protobuf parsing error (usually means Git LFS pointer)
        if "INVALID_PROTOBUF" in error_msg or "Protobuf parsing failed" in error_msg:
            file_size = onnx_file.stat().st_size
            if file_size < 1024:  # Very small file
                raise RuntimeError(
                    f"ONNX file is a Git LFS pointer (file not downloaded). "
                    f"File size: {file_size} bytes (expected ~10-70MB). "
                    f"Please download the actual file using:\n"
                    f"  git lfs pull --include='{onnx_file.name}'\n"
                    f"Or manually download the file to: {onnx_path}\n"
                    f"Original error: {error_msg}"
                )
        raise RuntimeError(
            f"Failed to load ONNX model from {onnx_path}: {error_msg}. "
            f"File may be corrupted or incomplete."
        )
    
    loaded_voice_models[voice_key] = {
        "session": session,
        "config": config,
        "info": voice_info
    }
    
    return loaded_voice_models[voice_key]

def phonemize_text_espeak(text: str, voice: str) -> str:
    """Phonemize text using espeak-ng via subprocess."""
    # Find espeak-ng binary (Linux: espeak-ng, fallback: espeak)
    espeak_bin = shutil.which("espeak-ng") or shutil.which("espeak")
    
    if not espeak_bin:
        raise RuntimeError(
            "espeak-ng not found. Install with: sudo apt-get install espeak-ng "
            "(Debian/Ubuntu) or sudo yum install espeak-ng (RHEL/CentOS)"
        )
    
    try:
        result = subprocess.run(
            [espeak_bin, "-q", "-x", "-v", voice, text],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
            encoding="utf-8"
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        raise RuntimeError("espeak-ng phonemization timed out")
    except subprocess.CalledProcessError as e:
        stderr_msg = e.stderr if isinstance(e.stderr, str) else e.stderr.decode('utf-8', errors='ignore')
        raise RuntimeError(f"espeak-ng failed: {stderr_msg}")
    except FileNotFoundError:
        raise RuntimeError("espeak-ng binary not found")

def synthesize_speech(text: str, voice_key: str, lang_code: str) -> bytes:
    """Synthesize speech from text using a Piper ONNX model."""
    # Check cache
    normalized_text = text.strip()
    if not normalized_text:
        raise ValueError("Text cannot be empty")
    
    cache_key = f"{voice_key}:{hashlib.md5(normalized_text.encode('utf-8')).hexdigest()}"
    if cache_key in audio_cache:
        return audio_cache[cache_key]
    
    # Load voice model
    voice_model = load_voice_model(voice_key, lang_code)
    session = voice_model["session"]
    config = voice_model["config"]
    
    # Get phoneme type and espeak voice
    phoneme_type = config.get("phoneme_type", "espeak")
    espeak_voice = config.get("espeak", {}).get("voice", lang_code.split("_")[0] if "_" in lang_code else lang_code)
    
    # Debug: Log the input text
    print(f"[TTS DEBUG] Input text: {repr(normalized_text)}")
    print(f"[TTS DEBUG] Voice: {voice_key}, Language: {lang_code}, Espeak voice: {espeak_voice}")
    
    # Phonemize text
    if phoneme_type == "text":
        phonemes_str = normalized_text
    else:
        try:
            phonemes_str = phonemize_text_espeak(normalized_text, espeak_voice)
            print(f"[TTS DEBUG] Phonemes: {repr(phonemes_str)}")
        except Exception as e:
            print(f"[TTS DEBUG] Phonemization error: {e}")
            # Fallback to language code only
            lang_only = lang_code.split("_")[0] if "_" in lang_code else lang_code
            phonemes_str = phonemize_text_espeak(normalized_text, lang_only)
            print(f"[TTS DEBUG] Phonemes (fallback): {repr(phonemes_str)}")
    
    # Apply phoneme mapping (from config)
    phoneme_map = config.get("phoneme_map", {})
    for src, dst in phoneme_map.items():
        phonemes_str = phonemes_str.replace(src, dst)
    
    # Pre-process: normalize common espeak-ng output issues
    # Convert newlines/tabs to spaces (they're just separators in espeak output)
    phonemes_str = phonemes_str.replace('\n', ' ').replace('\t', ' ')
    # Normalize multiple spaces to single space
    import re
    phonemes_str = re.sub(r' +', ' ', phonemes_str)
    
    # CRITICAL: Handle espeak special markers that aren't in phoneme_id_map
    # The '~' character is a nasalization marker in espeak but may not be in the map
    # In Piper, these markers are typically removed or handled by the phoneme_map
    # If not mapped, we should remove them to avoid skipping characters
    # Remove unmapped special markers that would cause issues
    # But first, let's see what's actually in the map
    phoneme_id_map = config.get("phoneme_id_map", {})
    # Remove characters that definitely won't be in the map (after checking)
    # We'll handle this during conversion instead
    
    print(f"[TTS DEBUG] Phonemes after mapping and normalization: {repr(phonemes_str)}")
    
    # Convert phonemes to IDs
    phoneme_id_map = config.get("phoneme_id_map", {})
    if not phoneme_id_map:
        raise RuntimeError("phoneme_id_map not found in config")
    
    phoneme_ids = []
    
    # Add start token
    if "^" in phoneme_id_map:
        token_ids = phoneme_id_map["^"]
        phoneme_ids.extend([int(x) for x in (token_ids if isinstance(token_ids, list) else [token_ids])])
    
    # Convert phonemes character by character
    # Important: Process each character individually, including spaces
    # Some espeak phonemes use uppercase or special characters that need mapping
    unmapped_chars = []
    for char in phonemes_str:
        # Try exact match first
        if char in phoneme_id_map:
            token_ids = phoneme_id_map[char]
            phoneme_ids.extend([int(x) for x in (token_ids if isinstance(token_ids, list) else [token_ids])])
        elif char.isspace():
            # Handle spaces - add multiple space tokens for longer pauses between words
            # This creates better word separation and clearer pronunciation
            if " " in phoneme_id_map:
                token_ids = phoneme_id_map[" "]
                space_ids = [int(x) for x in (token_ids if isinstance(token_ids, list) else [token_ids])]
                # Add double spaces for longer pauses between words (better clarity)
                phoneme_ids.extend(space_ids)
                phoneme_ids.extend(space_ids)  # Double space for pause
            # If space not in map, skip it (don't add anything)
        else:
            # Try case-insensitive match for letters
            if char.isalpha():
                char_lower = char.lower()
                char_upper = char.upper()
                if char_lower in phoneme_id_map:
                    token_ids = phoneme_id_map[char_lower]
                    phoneme_ids.extend([int(x) for x in (token_ids if isinstance(token_ids, list) else [token_ids])])
                    print(f"[TTS DEBUG] Mapped '{char}' -> '{char_lower}' (case conversion)")
                elif char_upper in phoneme_id_map:
                    token_ids = phoneme_id_map[char_upper]
                    phoneme_ids.extend([int(x) for x in (token_ids if isinstance(token_ids, list) else [token_ids])])
                    print(f"[TTS DEBUG] Mapped '{char}' -> '{char_upper}' (case conversion)")
                else:
                    # Character not in map - collect for warning
                    if char not in unmapped_chars:
                        unmapped_chars.append(char)
            else:
                # Special character - try common mappings
                # Common espeak special characters and their mappings
                special_mappings = {
                    '~': '~',  # Nasalization - might be in map as is
                    '&': '&',  # Special marker
                    '\n': ' ',  # Newline -> space
                    '\t': ' ',  # Tab -> space
                }
                
                mapped = False
                if char in special_mappings:
                    mapped_char = special_mappings[char]
                    if mapped_char in phoneme_id_map:
                        token_ids = phoneme_id_map[mapped_char]
                        phoneme_ids.extend([int(x) for x in (token_ids if isinstance(token_ids, list) else [token_ids])])
                        print(f"[TTS DEBUG] Mapped '{char}' -> '{mapped_char}' (special char mapping)")
                        mapped = True
                
                if not mapped:
                    # Character not in map - for espeak markers like ~ (nasalization), 
                    # we should skip them silently as they're modifiers, not phonemes
                    # But log them for debugging
                    if char not in unmapped_chars:
                        unmapped_chars.append(char)
                    # Don't add anything to phoneme_ids - these are modifiers, not phonemes
    
    if unmapped_chars:
        print(f"[TTS DEBUG] Warning: {len(unmapped_chars)} unmapped characters: {[f'{c}(ord={ord(c)})' for c in set(unmapped_chars)][:10]}")
        print(f"[TTS DEBUG] These characters are being skipped, which may cause incomplete audio!")
    
    # Add end token
    if "$" in phoneme_id_map:
        token_ids = phoneme_id_map["$"]
        phoneme_ids.extend([int(x) for x in (token_ids if isinstance(token_ids, list) else [token_ids])])
    
    if not phoneme_ids:
        raise RuntimeError("Failed to convert phonemes to IDs - no phoneme IDs generated")
    
    print(f"[TTS DEBUG] Phoneme IDs count: {len(phoneme_ids)}, first 20: {phoneme_ids[:20]}")
    print(f"[TTS DEBUG] Phoneme IDs last 20: {phoneme_ids[-20:]}")
    print(f"[TTS DEBUG] Total phoneme sequence length: {len(phonemes_str)} chars -> {len(phoneme_ids)} IDs")
    
    # Prepare ONNX inputs
    # Shape should be [1, sequence_length] for phoneme IDs
    phoneme_array = np.array([phoneme_ids], dtype=np.int64)
    sequence_length = phoneme_array.shape[1]
    
    print(f"[TTS DEBUG] Phoneme array shape: {phoneme_array.shape}, sequence_length: {sequence_length}")
    
    inference_config = config.get("inference", {})
    base_length_scale = float(inference_config.get("length_scale", 1.0))
    base_noise_scale = float(inference_config.get("noise_scale", 0.667))
    base_noise_w = float(inference_config.get("noise_w", 0.8))
    
    # CRITICAL: Adjust parameters for clearer, slower speech with better pronunciation
    # Based on Piper TTS best practices:
    # - Higher length_scale = slower speech with better vowel pronunciation
    # - Lower noise_scale = clearer, less variation (better for vowels)
    # - Slightly lower noise_w = more stable pronunciation
    
    if len(phoneme_ids) < 15:
        # Very short sequences need much higher length_scale
        length_scale = max(base_length_scale, 2.5)
        noise_scale = min(base_noise_scale, 0.5)  # Lower noise for clarity
        noise_w = min(base_noise_w, 0.7)  # Slightly lower for stability
        print(f"[TTS DEBUG] Very short sequence ({len(phoneme_ids)} tokens) - length_scale: {length_scale}, noise_scale: {noise_scale}")
    elif len(phoneme_ids) < 25:
        # Short sequences - moderate increase
        length_scale = max(base_length_scale, 1.6)  # Increased for better pacing
        noise_scale = min(base_noise_scale, 0.55)  # Lower noise for clarity
        noise_w = min(base_noise_w, 0.75)
        print(f"[TTS DEBUG] Short sequence ({len(phoneme_ids)} tokens) - length_scale: {length_scale}, noise_scale: {noise_scale}")
    else:
        # Longer sequences - slower, clearer speech with better vowel pronunciation
        # Use higher length_scale for slower pace and better vowel clarity
        length_scale = max(base_length_scale, 1.5)  # Increased from 1.3 for better clarity
        noise_scale = min(base_noise_scale, 0.6)  # Lower noise for clearer vowels
        noise_w = min(base_noise_w, 0.75)  # Slightly lower for stability
        print(f"[TTS DEBUG] Longer sequence ({len(phoneme_ids)} tokens) - length_scale: {length_scale}, noise_scale: {noise_scale} for clearer speech")
    
    print(f"[TTS DEBUG] Inference params: length_scale={length_scale}, noise_scale={noise_scale}, noise_w={noise_w}")
    
    # Build inputs based on model signature
    model_inputs = session.get_inputs()
    print(f"[TTS DEBUG] Model inputs: {[(inp.name, inp.type, inp.shape) for inp in model_inputs]}")
    
    inputs = {}
    
    for inp in model_inputs:
        inp_name = inp.name.lower()
        inp_type = str(inp.type).lower()
        
        if 'int64' in inp_type or 'long' in inp_type:
            if 'length' in inp_name and 'scale' not in inp_name:
                inputs[inp.name] = np.array([sequence_length], dtype=np.int64)
                print(f"[TTS DEBUG] Assigned length: {inp.name} = {sequence_length}")
            else:
                inputs[inp.name] = phoneme_array
                print(f"[TTS DEBUG] Assigned phonemes: {inp.name} = shape {phoneme_array.shape}")
        elif 'float' in inp_type:
            if 'scales' == inp_name:
                inputs[inp.name] = np.array([length_scale, noise_scale, noise_w], dtype=np.float32)
                print(f"[TTS DEBUG] Assigned scales: {inp.name} = [{length_scale}, {noise_scale}, {noise_w}]")
            elif 'length_scale' in inp_name:
                inputs[inp.name] = np.array([length_scale], dtype=np.float32)
                print(f"[TTS DEBUG] Assigned length_scale: {inp.name} = {length_scale}")
            elif 'noise_scale' in inp_name and 'w' not in inp_name:
                inputs[inp.name] = np.array([noise_scale], dtype=np.float32)
                print(f"[TTS DEBUG] Assigned noise_scale: {inp.name} = {noise_scale}")
            elif 'noise_w' in inp_name or ('noise' in inp_name and 'w' in inp_name):
                inputs[inp.name] = np.array([noise_w], dtype=np.float32)
                print(f"[TTS DEBUG] Assigned noise_w: {inp.name} = {noise_w}")
            else:
                inputs[inp.name] = np.array([length_scale], dtype=np.float32)
                print(f"[TTS DEBUG] Assigned default float: {inp.name} = {length_scale}")
    
    # Fallback: positional assignment
    if len(inputs) < len(model_inputs):
        print(f"[TTS DEBUG] Using fallback assignment (only {len(inputs)}/{len(model_inputs)} inputs assigned)")
        input_names = [inp.name for inp in model_inputs]
        if input_names[0] not in inputs:
            inputs[input_names[0]] = phoneme_array
            print(f"[TTS DEBUG] Fallback: {input_names[0]} = phoneme_array")
        for i, name in enumerate(input_names[1:], 1):
            if name not in inputs:
                if i == 1:
                    inputs[name] = np.array([length_scale], dtype=np.float32)
                    print(f"[TTS DEBUG] Fallback: {name} = length_scale")
                elif i == 2:
                    inputs[name] = np.array([noise_scale], dtype=np.float32)
                    print(f"[TTS DEBUG] Fallback: {name} = noise_scale")
                elif i == 3:
                    inputs[name] = np.array([noise_w], dtype=np.float32)
                    print(f"[TTS DEBUG] Fallback: {name} = noise_w")
    
    print(f"[TTS DEBUG] Final inputs: {[(k, v.shape, v.dtype) for k, v in inputs.items()]}")
    
    # Run inference
    print(f"[TTS DEBUG] Running ONNX inference...")
    outputs = session.run(None, inputs)
    
    # Get audio output - should be first output
    if len(outputs) == 0:
        raise RuntimeError("ONNX model produced no output")
    
    audio = outputs[0]
    print(f"[TTS DEBUG] Raw output shape: {audio.shape}, dtype: {audio.dtype}")
    print(f"[TTS DEBUG] Number of outputs: {len(outputs)}")
    if len(outputs) > 1:
        for i, out in enumerate(outputs[1:], 1):
            print(f"[TTS DEBUG] Output {i} shape: {out.shape}, dtype: {out.dtype}")
    
    # Flatten if needed and convert to float32
    # Piper models typically output shape (batch, channels, samples) or (batch, samples)
    # Shape (1, 1, 1, N) means: batch=1, channels=1, time_steps=1, samples=N
    # We need to extract the actual audio samples
    original_shape = audio.shape
    print(f"[TTS DEBUG] Original output shape: {original_shape}")
    
    # Handle different output shapes
    if len(audio.shape) == 4:
        # Shape (batch, channels, time, samples) - take the last dimension
        # OR shape (batch, channels, mel, audio) - need to check
        # For now, flatten everything and take all samples
        audio = audio.flatten()
        print(f"[TTS DEBUG] 4D tensor - flattened to {len(audio)} samples")
    elif len(audio.shape) == 3:
        # Shape (batch, channels, samples)
        audio = audio.flatten()
        print(f"[TTS DEBUG] 3D tensor - flattened to {len(audio)} samples")
    elif len(audio.shape) == 2:
        # Shape (batch, samples)
        audio = audio.flatten()
        print(f"[TTS DEBUG] 2D tensor - flattened to {len(audio)} samples")
    else:
        # Already 1D or something else
        audio = audio.flatten()
        print(f"[TTS DEBUG] Flattened to {len(audio)} samples")
    
    audio = audio.astype(np.float32)
    
    print(f"[TTS DEBUG] Audio after processing: {len(audio)} samples")
    print(f"[TTS DEBUG] Audio after flatten: length={len(audio)}, min={audio.min():.4f}, max={audio.max():.4f}, mean={audio.mean():.4f}, std={audio.std():.4f}")
    
    # Get sample rate early for duration checks
    audio_config = config.get("audio", {})
    config_sample_rate = int(audio_config.get("sample_rate", 22050))
    if config_sample_rate in [16000, 22050, 44100]:
        sample_rate = config_sample_rate
    else:
        sample_rate = 22050
    
    # Check if audio seems too short
    expected_min_duration = len(phoneme_ids) * 0.05  # Rough estimate: 50ms per phoneme
    actual_duration = len(audio) / sample_rate
    if actual_duration < expected_min_duration:
        print(f"[TTS DEBUG] WARNING: Audio duration ({actual_duration:.2f}s) is much shorter than expected ({expected_min_duration:.2f}s)")
        print(f"[TTS DEBUG] This suggests the model may have cut off early or the output format is incorrect")
    
    # Remove any NaN or Inf values
    if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
        print(f"[TTS DEBUG] Warning: Audio contains NaN or Inf values, replacing with zeros")
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Normalize and convert to int16
    # Piper models output audio in range approximately [-1, 1], but we should normalize to be safe
    audio_max = np.abs(audio).max()
    if audio_max > 0:
        # Normalize to [-1, 1] range
        audio_normalized = audio / audio_max
        # Clamp to [-1, 1] to avoid overflow
        audio_normalized = np.clip(audio_normalized, -1.0, 1.0)
        # Convert to int16: multiply by 32767 (max int16 value)
        audio = (audio_normalized * 32767).astype(np.int16)
        print(f"[TTS DEBUG] Normalized with max={audio_max:.4f}")
    else:
        # Silent audio - create zeros
        print(f"[TTS DEBUG] Warning: Audio is silent (max=0)")
        audio = np.zeros(len(audio), dtype=np.int16)
    
    print(f"[TTS DEBUG] Audio after normalization: min={audio.min()}, max={audio.max()}, mean={audio.mean():.2f}, non-zero samples={np.count_nonzero(audio)}")
    
    # Sample rate already determined above, just log it
    print(f"[TTS DEBUG] Config sample rate: {config_sample_rate} Hz")
    print(f"[TTS DEBUG] Using sample rate: {sample_rate} Hz")
    print(f"[TTS DEBUG] Audio length: {len(audio)} samples, Duration: {len(audio)/sample_rate:.2f} seconds")
    
    # Warn if duration seems too short
    duration = len(audio) / sample_rate
    if duration < 0.5:
        print(f"[TTS DEBUG] WARNING: Audio duration ({duration:.2f}s) seems very short for text: {repr(normalized_text[:50])}")
    
    # Create WAV file with proper format
    wav_buffer = io.BytesIO()
    try:
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit (2 bytes per sample)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio.tobytes())
        
        # Ensure buffer is at the beginning
        wav_buffer.seek(0)
        audio_bytes = wav_buffer.getvalue()
        
        print(f"[TTS DEBUG] WAV file created: {len(audio_bytes)} bytes")
        
        # Validate WAV file (should start with "RIFF" and contain "WAVE")
        if len(audio_bytes) < 12 or audio_bytes[:4] != b'RIFF' or audio_bytes[8:12] != b'WAVE':
            raise RuntimeError("Invalid WAV file format generated")
        
        # Cache the audio
        audio_cache[cache_key] = audio_bytes
        
        return audio_bytes
    except Exception as e:
        print(f"[TTS DEBUG] Error creating WAV file: {e}")
        raise RuntimeError(f"Failed to create WAV file: {str(e)}")
    finally:
        wav_buffer.close()

def load_single_model(model_name: str):
    """Load a single model on demand (lazy loading)."""
    global loaded_models, available_models
    
    print(f"\n[DEBUG] load_single_model called for: {model_name}")
    print(f"[DEBUG] model_name type: {type(model_name)}, value: {repr(model_name)}")
    
    if model_name in loaded_models:
        print(f"[DEBUG] Model {model_name} already loaded")
        return loaded_models[model_name]
    
    # Always refresh available models to ensure paths are correct
    print(f"[DEBUG] Scanning for available models...")
    scanned = scan_available_models()
    print(f"[DEBUG] Scanned {len(scanned)} models")
    
    # Completely replace available_models to ensure clean state
    available_models.clear()
    for name, info in scanned.items():
        # Ensure path is always set and is a Path object
        if "path" not in info or info["path"] is None:
            info["path"] = (get_models_directory() / name).absolute()
        elif not isinstance(info["path"], Path):
            info["path"] = Path(info["path"]).absolute()
        else:
            info["path"] = info["path"].absolute()
        available_models[name] = info
    
    if model_name not in available_models:
        print(f"[DEBUG] Model {model_name} not in available_models. Available: {list(available_models.keys())}")
        raise FileNotFoundError(f"Model {model_name} not found")
    
    info = available_models[model_name]
    print(f"[DEBUG] Model info: {info}")
    if not info["complete"]:
        raise RuntimeError(f"Model {model_name} is incomplete")
    
    # Always construct path from model name to avoid any path storage issues
    # This is more reliable than relying on stored paths
    models_dir = get_models_directory()
    print(f"[DEBUG] Models directory: {models_dir}")
    print(f"[DEBUG] Models directory exists: {models_dir.exists()}")
    
    model_path = (models_dir / model_name).absolute()
    print(f"[DEBUG] Constructed model_path: {model_path}")
    print(f"[DEBUG] model_path type: {type(model_path)}")
    print(f"[DEBUG] model_path exists: {model_path.exists()}")
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    
    # Convert to string - this should never be None
    model_path_str = str(model_path)
    print(f"[DEBUG] model_path_str: {repr(model_path_str)}")
    print(f"[DEBUG] model_path_str type: {type(model_path_str)}")
    print(f"[DEBUG] model_path_str length: {len(model_path_str) if model_path_str else 'N/A'}")
    
    # Final validation
    if not model_path_str or not isinstance(model_path_str, str):
        raise ValueError(f"CRITICAL: Invalid model path string for {model_name}: {type(model_path_str)}")
    
    # Check if model is in HuggingFace format (has config.json) or OPUS-MT format
    has_config = (model_path / "config.json").exists()
    print(f"Loading {model_name}...", end=" ", flush=True)
    
    # Initialize variables
    tokenizer = None
    model = None
    load_path = model_path_str
    
    try:
        # Try loading from local path first
        print(f"[DEBUG] Trying local path first: {load_path}")
        tokenizer = MarianTokenizer.from_pretrained(load_path, local_files_only=True)
        model = MarianMTModel.from_pretrained(load_path, local_files_only=True)
        print(f"[DEBUG] Successfully loaded from local path")
    except Exception as local_error:
        # If local path fails and it's OPUS-MT format, try HuggingFace identifier
        if not has_config:
            parts = model_name.split("-", 1)
            if len(parts) == 2:
                src_lang, tgt_lang = parts[0], parts[1]
                hf_model_id = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
                print(f"[DEBUG] Local load failed, trying HuggingFace identifier: {hf_model_id}")
                print(f"[DEBUG] Local error: {str(local_error)[:100]}")
                
                # Check if we already have it saved locally (from previous download)
                local_config = model_path / "config.json"
                if local_config.exists():
                    # We have local files - try loading from local (might have been converted)
                    try:
                        tokenizer = MarianTokenizer.from_pretrained(model_path_str, local_files_only=True)
                        model = MarianMTModel.from_pretrained(model_path_str, local_files_only=True)
                        load_path = model_path_str
                        print(f"[DEBUG] Successfully loaded from previously saved local model")
                    except Exception:
                        # Local files exist but might be incomplete - will try cache/download
                        pass
                
                if model is None or tokenizer is None:
                    # Try HuggingFace cache first
                    try:
                        tokenizer = MarianTokenizer.from_pretrained(hf_model_id, local_files_only=True)
                        model = MarianMTModel.from_pretrained(hf_model_id, local_files_only=True)
                        print(f"[DEBUG] Successfully loaded from HuggingFace cache")
                        
                        # Save to local directory for future use (only once)
                        print(f"[DEBUG] Saving model to local directory for future offline use...")
                        try:
                            tokenizer.save_pretrained(model_path_str)
                            model.save_pretrained(model_path_str)
                            load_path = model_path_str
                            print(f"[DEBUG] Model saved locally to: {model_path_str}")
                        except Exception as save_error:
                            print(f"[DEBUG] Warning: Failed to save model locally: {str(save_error)[:100]}")
                            load_path = hf_model_id  # Use HuggingFace identifier as fallback
                    except Exception as hf_cache_error:
                        # Not in cache - download and save locally (only once)
                        print(f"[DEBUG] Model not in cache, downloading from HuggingFace (one-time download)...")
                        tokenizer = MarianTokenizer.from_pretrained(hf_model_id, local_files_only=False)
                        model = MarianMTModel.from_pretrained(hf_model_id, local_files_only=False)
                        
                        # Save to local directory for future offline use
                        print(f"[DEBUG] Saving downloaded model to local directory...")
                        tokenizer.save_pretrained(model_path_str)
                        model.save_pretrained(model_path_str)
                        load_path = model_path_str
                        print(f"[DEBUG] Model downloaded and saved to: {model_path_str}")
            else:
                raise RuntimeError(f"Invalid model name format: {model_name}. Local load failed: {str(local_error)[:150]}")
        else:
            # HuggingFace format but local load failed
            raise RuntimeError(f"Failed to load HuggingFace format model from local path: {str(local_error)[:150]}")
    
    # Ensure model and tokenizer are loaded
    if model is None or tokenizer is None:
        raise RuntimeError(f"Failed to load model {model_name}: model or tokenizer is None")
    
    loaded_models[model_name] = {
        "model": model,
        "tokenizer": tokenizer,
        "path": load_path
    }
    print("✓")
    return loaded_models[model_name]


@app.on_event("startup")
async def startup_event():
    """Scan for models when the application starts."""
    print("=" * 70)
    print("🌍 MarianMT Translation Service")
    print("=" * 70)
    
    available_models.update(scan_available_models())
    
    if not available_models:
        print(f"⚠ No models found in: {get_models_directory()}")
    else:
        complete = sum(1 for m in available_models.values() if m["complete"])
        print(f"Found {len(available_models)} model(s): {complete} complete, {len(available_models) - complete} incomplete")
        print("Models will be loaded on demand")
    
    print("=" * 70)
    print("🌐 Web interface: http://localhost:8000")
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
    model_name = current_model_name
    
    if not model_name and request.model_path:
        path_str = str(request.model_path)
        if path_str in available_models:
            model_name = path_str
        else:
            # Extract model name from path
            path_obj = Path(path_str)
            for part in reversed(path_obj.parts):
                if part.startswith("en-") and part in available_models:
                    model_name = part
                    break
    
    if not model_name:
        complete_models = [name for name, info in available_models.items() if info["complete"]]
        if complete_models:
            model_name = complete_models[0]
        else:
            raise HTTPException(status_code=400, detail="No model available")
    
    try:
        if model_name not in loaded_models:
            load_single_model(model_name)
        
        model_info = loaded_models[model_name]
        
        # Ensure model and tokenizer are available
        if "model" not in model_info or "tokenizer" not in model_info:
            raise HTTPException(status_code=500, detail=f"Model {model_name} is not properly loaded")
        
        tokenizer = model_info["tokenizer"]
        model = model_info["model"]
        
        inputs = tokenizer([request.text], return_tensors="pt", padding=True, truncation=True)
        translated = model.generate(**inputs)
        translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
        
        return TranslationResponse(
            original_text=request.text,
            translated_text=translated_text,
            model_path=model_name
        )
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"\n✗ Translation error:")
        print(f"  Error type: {type(e).__name__}")
        print(f"  Error message: {str(e)}")
        print(f"  Full traceback:\n{error_trace}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

class SelectModelRequest(BaseModel):
    model_name: str

@app.post("/api/select-model")
async def select_model_endpoint(request: SelectModelRequest):
    """Select a model (load it if needed)."""
    global current_model_name
    
    model_name = request.model_name
    
    # Always refresh to ensure paths are current
    available_models.update(scan_available_models())
    
    if model_name not in available_models:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    if not available_models[model_name]["complete"]:
        raise HTTPException(status_code=400, detail=f"Model {model_name} is incomplete")
    
    try:
        if model_name not in loaded_models:
            load_single_model(model_name)
        
        current_model_name = model_name
        return {"status": "success", "model": model_name, "loaded": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

@app.get("/api/models")
async def list_models():
    """List available models."""
    available_models.update(scan_available_models())
    flags_dir = get_flags_directory()
    
    models_list = []
    for name, info in available_models.items():
        lang_code = name.split("-")[1] if "-" in name else None
        if not lang_code:
            continue
        
        flag_file = flags_dir / f"{lang_code}.svg"
        if not flag_file.exists():
            continue
        
        model_path = info.get("path")
        if model_path is None:
            model_path = get_models_directory() / name
        
        models_list.append({
            "name": name,
            "path": str(model_path),
            "complete": info["complete"],
            "is_current": (current_model_name == name),
            "lang_code": lang_code,
            "flag_path": f"/flags/{lang_code}.svg"
        })
    
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
    """Get all available language models."""
    available_models.update(scan_available_models())
    
    models_list = []
    pairs_list = []
    
    for model_name, info in available_models.items():
        parts = model_name.split("-")
        source = parts[0] if len(parts) > 0 else None
        target = parts[1] if len(parts) > 1 else None
        
        model_path = info.get("path")
        if model_path is None:
            model_path = get_models_directory() / model_name
        
        model_data = {
            "model_name": model_name,
            "model_path": str(model_path),
            "source_language": source,
            "target_language": target,
            "complete": info["complete"]
        }
        
        models_list.append(model_data)
        if source and target:
            pairs_list.append(model_data)
    
    return {"models": models_list, "language_pairs": pairs_list}

@app.get("/api/voices/{lang_code}")
async def get_voices(lang_code: str):
    """Get available voices for a language."""
    global available_voices
    
    # Only scan voices for the requested language (lazy loading)
    if lang_code not in available_voices:
        scanned = scan_available_voices(lang_code=lang_code)
        available_voices.update(scanned)
    
    if lang_code not in available_voices:
        return {"voices": [], "lang_code": lang_code}
    
    voices_list = [
        {
            "key": v["key"],
            "name": v["name"],
            "quality": v["quality"],
            "display_name": v["display_name"]
        }
        for v in available_voices[lang_code]
    ]
    
    voices_list.sort(key=lambda x: (x["name"], x["quality"]))
    return {"voices": voices_list, "lang_code": lang_code}

class SynthesizeRequest(BaseModel):
    text: str
    voice_key: str
    lang_code: str

@app.post("/api/synthesize")
async def synthesize_endpoint(request: SynthesizeRequest):
    """Synthesize speech from text using a voice model."""
    try:
        print(f"[TTS] Synthesis request: text={repr(request.text[:50])}..., voice={request.voice_key}, lang={request.lang_code}")
        
        # Synthesize speech (this is synchronous and will complete before returning)
        audio_data = synthesize_speech(request.text, request.voice_key, request.lang_code)
        
        print(f"[TTS] Synthesis complete: {len(audio_data)} bytes")
        
        # Return as Response (not StreamingResponse) to ensure complete file
        from fastapi.responses import Response
        return Response(
            content=audio_data,
            media_type="audio/wav",
            headers={
                "Content-Type": "audio/wav",
                "Content-Length": str(len(audio_data)),
                "Content-Disposition": 'inline; filename="speech.wav"'
            }
        )
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[TTS ERROR] Synthesis failed: {str(e)}")
        print(f"[TTS ERROR] Traceback:\n{error_trace}")
        raise HTTPException(status_code=500, detail=f"Speech synthesis failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

