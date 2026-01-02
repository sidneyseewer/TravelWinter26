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
import torch

# Try to import piper-tts package
try:
    from piper import PiperVoice, SynthesisConfig
    PIPER_AVAILABLE = True
except ImportError:
    try:
        from piper_tts import PiperVoice, SynthesisConfig
        PIPER_AVAILABLE = True
    except ImportError:
        try:
            from piper import PiperVoice
            # Try to import SynthesisConfig separately
            try:
                from piper.config import SynthesisConfig
            except ImportError:
                SynthesisConfig = None
            PIPER_AVAILABLE = True
        except ImportError:
            PIPER_AVAILABLE = False
            SynthesisConfig = None
            print("[WARNING] piper-tts package not found. Falling back to ONNX Runtime.")

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
    """Check if a model directory is complete.
    
    Supports both HuggingFace format and OPUS-MT format.
    Returns True if sufficient files are present for inference.
    """
    try:
        if not model_path.exists() or not model_path.is_dir():
            return False
        
        # Skip if only README.md exists
        try:
            files = list(model_path.iterdir())
            if len(files) == 1 and any(f.name == "README.md" for f in files):
                return False
        except (OSError, PermissionError):
            # If we can't read directory, skip it
            return False
        
        # Check for HuggingFace format
        try:
            has_config = (model_path / "config.json").exists()
            has_weights = any([
                (model_path / "pytorch_model.bin").exists(),
                (model_path / "model.safetensors").exists(),
                (model_path / "model.bin").exists(),
                (model_path / "pytorch_model.bin.index.json").exists()
            ])
            
            if has_config and has_weights:
                return True
        except (OSError, PermissionError):
            # Continue to check OPUS-MT format
            pass
            
        # Check for OPUS-MT format indicators
        # Need decoder.yml and at least one .npz file for inference
        try:
            has_decoder = (model_path / "decoder.yml").exists()
            has_npz = any(f.suffix == ".npz" for f in model_path.iterdir() if f.is_file())
            
            # Also check for vocab.yml or source.bpe/target.bpe as additional indicators
            has_vocab = (model_path / "opus.bpe32k-bpe32k.vocab.yml").exists() or \
                       any("vocab.yml" in f.name for f in model_path.iterdir() if f.is_file())
            has_bpe = (model_path / "source.bpe").exists() or (model_path / "target.bpe").exists()
            
            # Model is complete if it has decoder.yml and .npz file
            # Additional files (vocab.yml, .bpe) are helpful but not strictly required
            if has_decoder and has_npz:
                return True
        except (OSError, PermissionError):
            # If we can't check files, assume incomplete
            return False
        
        return False
    except Exception as e:
        # If any error occurs during checking, skip this model
        print(f"Warning: Error checking model completeness for {model_path}: {e}")
        return False

def scan_available_models() -> Dict[str, Dict]:
    """Scan the models directory for folders starting with 'en-'.
    
    If a model check fails, it's skipped but scanning continues.
    """
    models_dir = get_models_directory()
    available = {}
    
    if not models_dir.exists():
        return available
    
    try:
        for item in models_dir.iterdir():
            if item.is_dir() and item.name.startswith("en-"):
                try:
                    # Ensure path is always an absolute Path object
                    model_path = item.absolute() if isinstance(item, Path) else Path(item).absolute()
                    
                    # Check completeness - if check fails, mark as incomplete but still add
                    try:
                        complete = is_model_complete(model_path)
                    except Exception as e:
                        print(f"Warning: Failed to check completeness for {item.name}: {e}")
                        complete = False
                    
                    available[item.name] = {
                        "path": model_path,
                        "complete": complete,
                        "name": item.name
                    }
                except Exception as e:
                    # Skip this model but continue scanning others
                    print(f"Warning: Error processing model {item.name}: {e}")
                    continue
    except (OSError, PermissionError) as e:
        print(f"Warning: Error reading models directory: {e}")
    
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
                        
                        # Try to find ONNX files - if glob fails, continue
                        try:
                            onnx_files = list(quality_dir.glob("*.onnx"))
                        except (OSError, PermissionError) as e:
                            print(f"Warning: Cannot read voice directory {quality_dir}: {e}")
                            continue
                        
                        for onnx_file in onnx_files:
                            json_file = onnx_file.with_suffix(".onnx.json")
                            
                            # Check if JSON file exists - if not, skip this voice
                            try:
                                if not json_file.exists():
                                    continue
                            except (OSError, PermissionError):
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
                            except (OSError, IOError, PermissionError) as e:
                                # If we can't check, continue anyway - let the ONNX runtime handle validation
                                pass
                            
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
                                # Skip this voice but continue with others
                                print(f"Warning: Failed to load voice config {json_file}: {e}")
                                continue
        
        if voices_list:
            voices_dict[current_lang_code] = voices_list
    
    return voices_dict

def load_voice_model(voice_key: str, lang_code: str, force_onnx: bool = False) -> Dict:
    """Load a voice model using piper-tts package or ONNX Runtime.
    
    Args:
        voice_key: The voice key identifier
        lang_code: The language code
        force_onnx: If True, force ONNX Runtime loading even if piper-tts is available
    """
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
    config_path = voice_info.get("config_path")
    
    # Use piper-tts package if available and not forced to use ONNX
    if PIPER_AVAILABLE and not force_onnx:
        try:
            # Find config file path
            if not config_path:
                config_path = str(Path(onnx_path).with_suffix('.onnx.json'))
            
            if not Path(config_path).exists():
                # Try to find config in same directory
                onnx_dir = Path(onnx_path).parent
                config_path = str(onnx_dir / f"{Path(onnx_path).stem}.onnx.json")
            
            if Path(config_path).exists():
                print(f"[TTS] Loading voice with piper-tts: {onnx_path}")
                voice = PiperVoice.load(model_path=onnx_path, config_path=config_path)
                
                loaded_voice_models[voice_key] = {
                    "voice": voice,
                    "config": config,
                    "onnx_path": onnx_path,
                    "config_path": config_path,
                    "use_piper": True
                }
                return loaded_voice_models[voice_key]
            else:
                print(f"[TTS WARNING] Config file not found: {config_path}, falling back to ONNX Runtime")
        except Exception as e:
            print(f"[TTS WARNING] Failed to load with piper-tts: {e}, falling back to ONNX Runtime")
            import traceback
            traceback.print_exc()
    
    # Fallback to ONNX Runtime implementation
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

def synthesize_speech(text: str, voice_key: str, lang_code: str, speed_multiplier: float = 1.0, length_scale: float = 1.0, noise_scale: float = 0.667, noise_w: float = 0.8) -> bytes:
    """Synthesize speech from text using piper-tts package or ONNX Runtime.
    
    Args:
        speed_multiplier: Speed as percentage (0.01 = 1%, 1.0 = 100%, 2.0 = 200%, etc.)
    """
    """Synthesize speech from text using piper-tts package or ONNX Runtime."""
    # Normalize and preprocess text
    # Remove leading/trailing whitespace but preserve internal structure
    normalized_text = text.strip()
    if not normalized_text:
        raise ValueError("Text cannot be empty")
    
    # Log the full input text to verify we're processing everything
    print(f"[TTS] Processing full text ({len(normalized_text)} chars): {repr(normalized_text)}")
    
    # Don't remove punctuation - espeak-ng handles it properly
    # Just ensure the text is properly encoded
    normalized_text = normalized_text.replace('\r\n', ' ').replace('\r', ' ')
    # Keep newlines as sentence breaks (convert to periods for better prosody)
    normalized_text = normalized_text.replace('\n', '. ')
    
    print(f"[TTS] Normalized text: {repr(normalized_text)}")
    
    # Speed control using percentage:
    # speed_multiplier is now a percentage: 0.01 = 1%, 1.0 = 100%, 2.0 = 200%, etc.
    # For Piper TTS: length_scale controls speed
    # - Higher length_scale = slower speech
    # - Lower length_scale = faster speech
    #
    # FIXED: The formula was reversed. The correct formula is:
    #   base_length_scale = 1.0 / speed_percentage
    # But if the model interprets it the opposite way, we need to invert:
    #   base_length_scale = speed_percentage
    #
    # Since user reports 1% makes it faster and 300% makes it slower,
    # the current formula (1.0 / speed_percentage) is producing the opposite effect.
    # So we need to use speed_percentage directly.
    #
    # Correct behavior:
    #   speed=0.01 (1%) → should be SLOW → need HIGH length_scale → use 1.0 / 0.01 = 100.0
    #   speed=3.00 (300%) → should be FAST → need LOW length_scale → use 1.0 / 3.0 = 0.33
    #
    # But user says 1% is fast and 300% is slow, so the model must interpret length_scale backwards.
    # Fix: Use speed_percentage directly (inverted from expected)
    #   speed=0.01 (1%) → base_length_scale = 0.01 (low = fast) - matches user report
    #   speed=3.00 (300%) → base_length_scale = 3.0 (high = slow) - matches user report
    #
    # Actually wait - if the model interprets length_scale backwards, we should still use 1.0 / speed_percentage
    # but maybe apply it differently. Let me think...
    #
    # User says: 1% makes it faster, 300% makes it slower
    # Current: base_length_scale = 1.0 / speed_percentage
    #   1% → 100.0 (high) → should be slow but user says fast
    #   300% → 0.33 (low) → should be fast but user says slow
    #
    # So the model must interpret HIGH length_scale as FAST and LOW as SLOW (opposite of normal)
    # Fix: Invert the formula to match the model's interpretation
    #   base_length_scale = speed_percentage (direct use, not inverted)
    
    speed_percentage = max(speed_multiplier, 0.01)  # Minimum 1%
    speed_percentage = min(speed_percentage, 3.0)  # Maximum 300%
    
    # FIXED: Invert the formula since the model interprets length_scale backwards
    # Use speed_percentage directly instead of 1.0 / speed_percentage
    base_length_scale = speed_percentage
    
    # Apply user's length_scale preference (from slider) as additional adjustment
    final_length_scale = base_length_scale * length_scale
    
    # Clamp to reasonable bounds for Piper TTS
    final_length_scale = max(final_length_scale, 0.01)  # Minimum for very slow (1%)
    final_length_scale = min(final_length_scale, 3.0)  # Maximum for very fast (300%)
    
    print(f"[TTS] Speed control: speed_percentage={speed_percentage*100:.0f}%, base_length_scale={base_length_scale:.2f}, user_length_scale={length_scale:.2f}, final_length_scale={final_length_scale:.2f}")
    
    # Store speed_percentage for use in phoneme processing (pause calculation)
    # This will be used later when processing spaces
    
    # Include all parameters in cache key so different parameter combinations are cached separately
    cache_key = f"{voice_key}:{speed_multiplier:.2f}:{length_scale:.2f}:{noise_scale:.3f}:{noise_w:.3f}:{hashlib.md5(normalized_text.encode('utf-8')).hexdigest()}"
    if cache_key in audio_cache:
        return audio_cache[cache_key]
    
    # Load voice model
    voice_model = load_voice_model(voice_key, lang_code)
    
    # Try using command-line piper tool first (produces best quality audio)
    piper_bin = shutil.which("piper")
    if piper_bin:
        try:
            onnx_path = voice_model["onnx_path"]
            print(f"[TTS] Using command-line piper tool: {piper_bin}")
            print(f"[TTS] Model: {onnx_path}")
            print(f"[TTS] Parameters: length_scale={final_length_scale:.2f}, noise_scale={noise_scale:.3f}, noise_w={noise_w:.3f}")
            
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            try:
                # Run piper command-line tool
                # piper expects text via stdin and outputs to --output_file
                cmd = [
                    piper_bin,
                    '--model', onnx_path,
                    '--output_file', tmp_path,
                    '--length_scale', str(final_length_scale),
                    '--noise_scale', str(noise_scale),
                    '--noise_w', str(noise_w)
                ]
                
                print(f"[TTS] Running: {' '.join(cmd)}")
                
                result = subprocess.run(
                    cmd,
                    input=normalized_text,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=30,
                    encoding='utf-8'
                )
                
                if result.stderr:
                    print(f"[TTS] Piper stderr: {result.stderr}")
                
                # Read the generated WAV file
                if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
                    with open(tmp_path, 'rb') as f:
                        audio_bytes = f.read()
                    
                    print(f"[TTS] Piper generated {len(audio_bytes)} bytes of audio")
                    
                    # Cache the audio
                    audio_cache[cache_key] = audio_bytes
                    return audio_bytes
                else:
                    raise RuntimeError(f"Piper did not generate output file: {tmp_path}")
                    
            finally:
                # Clean up temp file if it exists
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                    
        except subprocess.TimeoutExpired:
            print(f"[TTS ERROR] Piper command timed out, falling back to ONNX Runtime")
        except subprocess.CalledProcessError as e:
            print(f"[TTS ERROR] Piper command failed: {e.stderr}, falling back to ONNX Runtime")
        except Exception as e:
            print(f"[TTS ERROR] Piper synthesis failed: {e}, falling back to ONNX Runtime")
            import traceback
            traceback.print_exc()
    
    # Try piper-tts Python package if available
    if voice_model.get("use_piper", False) and PIPER_AVAILABLE:
        try:
            voice = voice_model["voice"]
            print(f"[TTS] Using piper-tts Python package with length_scale={final_length_scale:.2f}, noise_scale={noise_scale:.3f}, noise_w={noise_w:.3f}")
            
            # Synthesize using piper-tts
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            try:
                # Create SynthesisConfig if available
                if SynthesisConfig is not None:
                    syn_config = SynthesisConfig(
                        length_scale=final_length_scale,
                        noise_scale=noise_scale,
                        noise_w=noise_w
                    )
                    
                    # Use synthesize_wav with config
                    with wave.open(tmp_path, 'wb') as wav_file:
                        voice.synthesize_wav(normalized_text, wav_file, syn_config=syn_config)
                else:
                    # Fallback: try synthesize with wav_file
                    with wave.open(tmp_path, 'wb') as wav_file:
                        # Try different API variations
                        try:
                            voice.synthesize(normalized_text, wav_file, 
                                            length_scale=final_length_scale,
                                            noise_scale=noise_scale,
                                            noise_w=noise_w)
                        except TypeError:
                            # Try without parameters (use defaults from config)
                            voice.synthesize(normalized_text, wav_file)
                
                # Read the generated WAV file
                with open(tmp_path, 'rb') as f:
                    audio_bytes = f.read()
                
                # Cache the audio
                audio_cache[cache_key] = audio_bytes
                return audio_bytes
            finally:
                # Clean up temp file if it exists
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        except Exception as e:
            print(f"[TTS ERROR] piper-tts Python package synthesis failed: {e}, falling back to ONNX Runtime")
            import traceback
            traceback.print_exc()
            # Fall through to ONNX Runtime implementation
    
    # Fallback to ONNX Runtime implementation
    # Check if we have a session (ONNX Runtime) or need to reload
    if "session" not in voice_model:
        # If piper-tts was used but failed, we need to reload with ONNX Runtime
        print(f"[TTS] Reloading voice model with ONNX Runtime for fallback")
        # Clear from cache and reload with force_onnx=True
        if voice_key in loaded_voice_models:
            del loaded_voice_models[voice_key]
        voice_model = load_voice_model(voice_key, lang_code, force_onnx=True)
    
    session = voice_model["session"]
    config = voice_model["config"]
    
    # Get phoneme type and espeak voice
    phoneme_type = config.get("phoneme_type", "espeak")
    espeak_voice = config.get("espeak", {}).get("voice", lang_code.split("_")[0] if "_" in lang_code else lang_code)
    
    # Debug: Log the input text
    print(f"[TTS DEBUG] Input text for phonemization ({len(normalized_text)} chars): {repr(normalized_text)}")
    print(f"[TTS DEBUG] Voice: {voice_key}, Language: {lang_code}, Espeak voice: {espeak_voice}")
    
    # Phonemize text
    if phoneme_type == "text":
        phonemes_str = normalized_text
        print(f"[TTS DEBUG] Using text phoneme type (no phonemization)")
    else:
        try:
            # Phonemize the full text - espeak-ng handles punctuation and prosody
            phonemes_str = phonemize_text_espeak(normalized_text, espeak_voice)
            print(f"[TTS DEBUG] Phonemes ({len(phonemes_str)} chars): {repr(phonemes_str[:200])}{'...' if len(phonemes_str) > 200 else ''}")
        except Exception as e:
            print(f"[TTS DEBUG] Phonemization error: {e}")
            # Fallback to language code only
            lang_only = lang_code.split("_")[0] if "_" in lang_code else lang_code
            phonemes_str = phonemize_text_espeak(normalized_text, lang_only)
            print(f"[TTS DEBUG] Phonemes (fallback, {len(phonemes_str)} chars): {repr(phonemes_str[:200])}{'...' if len(phonemes_str) > 200 else ''}")
    
    # Apply phoneme mapping (from config)
    phoneme_map = config.get("phoneme_map", {})
    for src, dst in phoneme_map.items():
        phonemes_str = phonemes_str.replace(src, dst)
    
    # Pre-process: normalize common espeak-ng output issues
    # Convert newlines/tabs to spaces (they're just separators in espeak output)
    phonemes_str = phonemes_str.replace('\n', ' ').replace('\t', ' ')
    # Normalize multiple spaces to single space (but keep at least one for word boundaries)
    import re
    phonemes_str = re.sub(r' +', ' ', phonemes_str).strip()
    
    # Verify we have phonemes for the full text
    print(f"[TTS DEBUG] Final phoneme string length: {len(phonemes_str)} chars")
    if len(phonemes_str) < len(normalized_text) * 0.5:
        print(f"[TTS WARNING] Phoneme string seems short compared to input text. This might indicate incomplete phonemization.")
    
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
            # Handle spaces - add multiple space tokens for much longer pauses between words
            # This creates better word separation and clearer pronunciation
            if " " in phoneme_id_map:
                token_ids = phoneme_id_map[" "]
                space_ids = [int(x) for x in (token_ids if isinstance(token_ids, list) else [token_ids])]
                # Add spaces for pauses between words
                # The number of spaces scales with speed - slower speech needs more pauses
                # Calculate pause multiplier based on speed_percentage
                # For slower speech (lower percentage), add more spaces
                # speed_percentage 0.01 (1%) → 100 spaces
                # speed_percentage 0.10 (10%) → 10 spaces
                # speed_percentage 0.50 (50%) → 2 spaces
                # speed_percentage 1.00 (100%) → 1 space
                # speed_percentage 2.00 (200%) → 1 space
                # Use speed_percentage from the function scope (calculated earlier)
                pause_multiplier = max(1, int(1.0 / max(speed_percentage, 0.01)))  # Inverse relationship
                pause_multiplier = min(pause_multiplier, 100)  # Cap at 100 spaces max for very slow speech (1%)
                
                # Add spaces based on speed (slower = more pauses between words)
                for _ in range(pause_multiplier):
                    phoneme_ids.extend(space_ids)
            # If space not in map, skip it (don't add anything)
            else:
                pass
        # Try case-insensitive match for letters
        elif char.isalpha():
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
    config_base_length_scale = float(inference_config.get("length_scale", 1.0))
    config_base_noise_scale = float(inference_config.get("noise_scale", 0.667))
    config_base_noise_w = float(inference_config.get("noise_w", 0.8))
    
    # According to Piper TTS documentation:
    # - length_scale directly controls speech speed
    # - Higher length_scale = slower speech (e.g., 2.0 = slower)
    # - Lower length_scale = faster speech (e.g., 0.5 = faster)
    # - Default is typically 1.0
    
    # Use user-provided length_scale directly as the base
    # The user can set this via the Length Scale slider
    base_length_scale = length_scale
    
    # Speed multiplier modifies the base length_scale:
    # - speed_multiplier < 1.0 means slower (e.g., 0.5 = 2x slower)
    # - speed_multiplier = 1.0 means normal speed (no change)
    # - speed_multiplier > 1.0 means faster (e.g., 2.0 = 2x faster)
    # 
    # Formula: final_length_scale = base_length_scale / speed_multiplier
    # Examples:
    #   base=1.0, speed=0.5 → final=2.0 (2x slower) ✓
    #   base=1.0, speed=1.0 → final=1.0 (normal) ✓
    #   base=1.0, speed=2.0 → final=0.5 (2x faster) ✓
    #   base=2.0, speed=0.5 → final=4.0 (4x slower) ✓
    
    # FIXED: The model interprets length_scale backwards
    # User reports: 1% makes it faster, 300% makes it slower
    # So we need to use speed_multiplier directly (multiply, not divide)
    #   1% (0.01) → final = base * 0.01 (low = fast) ✓
    #   300% (3.0) → final = base * 3.0 (high = slow) ✓
    
    # Clamp speed_multiplier to valid range
    speed_multiplier = max(speed_multiplier, 0.01)  # Minimum 1%
    speed_multiplier = min(speed_multiplier, 3.0)  # Maximum 300%
    
    # Apply speed multiplier to base length_scale (FIXED: multiply instead of divide)
    final_length_scale = base_length_scale * speed_multiplier
    
    # Clamp final length_scale to reasonable bounds
    # Piper TTS typically works well with length_scale between 0.3 and 5.0
    final_length_scale = max(final_length_scale, 0.3)  # Minimum for very fast speech
    final_length_scale = min(final_length_scale, 10.0)  # Maximum for very slow speech
    
    # Use user-provided noise parameters directly
    final_noise_scale = noise_scale
    final_noise_w = noise_w
    
    print(f"[TTS DEBUG] Sequence ({len(phoneme_ids)} tokens) - base_length_scale: {base_length_scale:.2f}, speed_multiplier: {speed_multiplier:.2f}, final length_scale: {final_length_scale:.2f}")
    print(f"[TTS DEBUG] Noise params - noise_scale: {final_noise_scale:.3f}, noise_w: {final_noise_w:.3f}")
    
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
                inputs[inp.name] = np.array([final_length_scale, final_noise_scale, final_noise_w], dtype=np.float32)
                print(f"[TTS DEBUG] Assigned scales: {inp.name} = [{final_length_scale}, {final_noise_scale}, {final_noise_w}]")
            elif 'length_scale' in inp_name:
                inputs[inp.name] = np.array([final_length_scale], dtype=np.float32)
                print(f"[TTS DEBUG] Assigned length_scale: {inp.name} = {final_length_scale}")
            elif 'noise_scale' in inp_name and 'w' not in inp_name:
                inputs[inp.name] = np.array([final_noise_scale], dtype=np.float32)
                print(f"[TTS DEBUG] Assigned noise_scale: {inp.name} = {final_noise_scale}")
            elif 'noise_w' in inp_name or ('noise' in inp_name and 'w' in inp_name):
                inputs[inp.name] = np.array([final_noise_w], dtype=np.float32)
                print(f"[TTS DEBUG] Assigned noise_w: {inp.name} = {final_noise_w}")
            else:
                inputs[inp.name] = np.array([final_length_scale], dtype=np.float32)
                print(f"[TTS DEBUG] Assigned default float: {inp.name} = {final_length_scale}")
    
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
                    inputs[name] = np.array([final_length_scale], dtype=np.float32)
                    print(f"[TTS DEBUG] Fallback: {name} = final_length_scale")
                elif i == 2:
                    inputs[name] = np.array([final_noise_scale], dtype=np.float32)
                    print(f"[TTS DEBUG] Fallback: {name} = final_noise_scale")
                elif i == 3:
                    inputs[name] = np.array([final_noise_w], dtype=np.float32)
                    print(f"[TTS DEBUG] Fallback: {name} = final_noise_w")
    
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
    
    # Set model to evaluation mode for inference (important for consistent behavior)
    model.eval()
    
    # Verify model is on correct device (CPU for Linux compatibility)
    device = torch.device("cpu")
    model = model.to(device)
    
    loaded_models[model_name] = {
        "model": model,
        "tokenizer": tokenizer,
        "path": load_path,
        "device": device
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
        device = model_info.get("device", torch.device("cpu"))
        
        # Tokenize input text
        inputs = tokenizer([request.text], return_tensors="pt", padding=True, truncation=True, max_length=512)
        # Move inputs to the same device as the model
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate translation with proper parameters
        # Use num_beams for better quality, max_length to prevent infinite generation
        with torch.no_grad():  # Disable gradient computation for inference
            translated = model.generate(
                **inputs,
                max_length=512,  # Maximum output length
                num_beams=4,  # Beam search for better quality
                early_stopping=True,  # Stop when EOS token is generated
                length_penalty=0.6,  # Penalize longer sequences slightly
                no_repeat_ngram_size=3  # Prevent repetition
            )
        
        # Decode the translated tokens
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
    speed_multiplier: float = 1.0  # 0.1 = 10x slower, 1.0 = normal speed
    length_scale: float = 1.0  # Length scale parameter
    noise_scale: float = 0.667  # Noise scale parameter
    noise_w: float = 0.8  # Noise W parameter

@app.post("/api/synthesize")
async def synthesize_endpoint(request: SynthesizeRequest):
    """Synthesize speech from text using a voice model."""
    try:
        print(f"[TTS] Synthesis request: text={repr(request.text[:50])}..., voice={request.voice_key}, lang={request.lang_code}, speed={request.speed_multiplier}, length_scale={request.length_scale}, noise_scale={request.noise_scale}, noise_w={request.noise_w}")
        
        # Synthesize speech (this is synchronous and will complete before returning)
        audio_data = synthesize_speech(
            request.text, 
            request.voice_key, 
            request.lang_code, 
            speed_multiplier=request.speed_multiplier,
            length_scale=request.length_scale,
            noise_scale=request.noise_scale,
            noise_w=request.noise_w
        )
        
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

