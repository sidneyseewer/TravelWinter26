/**
 * Standalone Node.js Translation Server
 * Works independently from Python backend
 * Uses @xenova/transformers for translation and onnxruntime-node for TTS
 */

import express from 'express';
import cors from 'cors';
import path from 'path';
import fs from 'fs-extra';
import { pipeline, env } from '@xenova/transformers';
import wavefileModule from 'wavefile';
// wavefile exports an object with WaveFile property: { default: { WaveFile: class } }
const WaveFile = wavefileModule.default?.WaveFile || wavefileModule.WaveFile;
import { fileURLToPath } from 'url';
import { spawn } from 'child_process';
import crypto from 'crypto';

// Lazy load onnxruntime-node (only needed for TTS, not translation)
let ort = null;
let ortLoadError = null;

try {
    const ortModule = await import('onnxruntime-node');
    ort = ortModule.default || ortModule;
    if (ort && typeof ort.InferenceSession !== 'undefined') {
        console.log('[TTS] onnxruntime-node module imported successfully');
    } else {
        throw new Error('onnxruntime-node module loaded but InferenceSession not found');
    }
} catch (error) {
    ortLoadError = error;
    ort = null;
    console.error('[TTS] ERROR: onnxruntime-node failed to import. TTS features will be unavailable.');
    console.error('[TTS] Error details:', error.message);
    if (error.stack) {
        console.error('[TTS] Stack trace:', error.stack);
    }
}

// Get __dirname equivalent in ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Allow remote model downloads for caching (will cache for offline use)
env.allowRemoteModels = true;
// Don't set localModelPath - we'll use absolute paths directly

const app = express();
const PORT = process.env.PORT || 3000;

// Get project directories
const PROJECT_ROOT = path.resolve(__dirname, '..');
const MODELS_DIR = path.join(PROJECT_ROOT, 'models');
const VOICES_DIR = path.join(PROJECT_ROOT, 'voices');
const FLAGS_DIR = path.join(PROJECT_ROOT, 'flags');
const PUBLIC_DIR = path.join(__dirname, 'public');

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Serve static files
app.use('/static', express.static(path.join(PUBLIC_DIR, 'static')));
app.use('/flags', express.static(FLAGS_DIR));

// In-memory caches
const loadedModels = new Map();
const loadedVoices = new Map();
const availableModels = new Map();
const availableVoices = new Map();
const ttsCache = new Map(); // Cache for TTS audio: key -> audio buffer

// Input validation helpers
function sanitizeString(input, maxLength = 10000) {
    if (typeof input !== 'string') {
        throw new Error('Input must be a string');
    }
    // Remove null bytes and control characters (except newlines and tabs)
    let sanitized = input.replace(/[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]/g, '');
    // Limit length
    if (sanitized.length > maxLength) {
        sanitized = sanitized.substring(0, maxLength);
    }
    return sanitized.trim();
}

function sanitizeModelName(input) {
    if (typeof input !== 'string') {
        throw new Error('Model name must be a string');
    }
    // Only allow alphanumeric, hyphens, underscores, and forward slashes
    if (!/^[a-zA-Z0-9_\-/]+$/.test(input)) {
        throw new Error('Invalid model name format');
    }
    return input;
}

function sanitizeVoiceKey(input) {
    if (typeof input !== 'string') {
        throw new Error('Voice key must be a string');
    }
    // Only allow alphanumeric, hyphens, underscores, and forward slashes
    if (!/^[a-zA-Z0-9_\-/]+$/.test(input)) {
        throw new Error('Invalid voice key format');
    }
    return input;
}

function sanitizeLangCode(input) {
    if (typeof input !== 'string') {
        throw new Error('Language code must be a string');
    }
    // Only allow lowercase letters, underscores, and hyphens (e.g., 'en', 'fi_FI', 'en-US')
    if (!/^[a-z_\-]+$/.test(input)) {
        throw new Error('Invalid language code format');
    }
    return input;
}

function sanitizeFloat(input, min = 0, max = 10, defaultValue = 1.0) {
    if (input === undefined || input === null) {
        return defaultValue;
    }
    const num = parseFloat(input);
    if (isNaN(num)) {
        return defaultValue;
    }
    return Math.max(min, Math.min(max, num));
}

// Download model from HuggingFace using Python
async function downloadModelFromHuggingFace(modelName, modelPath, progressCallback = null) {
    return new Promise((resolve, reject) => {
        // Check if model is already complete before downloading
        const configPath = path.join(modelPath, 'config.json');
        const hasWeights = (
            fs.existsSync(path.join(modelPath, 'model.safetensors')) ||
            fs.existsSync(path.join(modelPath, 'pytorch_model.bin')) ||
            fs.existsSync(path.join(modelPath, 'model.bin'))
        );
        
        if (fs.existsSync(configPath) && hasWeights) {
            if (progressCallback) progressCallback('Model already exists, skipping download');
            resolve('already_exists');
            return;
        }
        
        const langCode = modelName.split('-')[1] || '';
        // Try common HuggingFace model IDs
        const possibleModelIds = [
            `Helsinki-NLP/opus-mt-en-${langCode}`,
            `Helsinki-NLP/opus-mt-${modelName}`,
            `Helsinki-NLP/opus-mt-tc-${modelName}`
        ];
        
        const scriptContent = [
            'import sys',
            'import json',
            'from pathlib import Path',
            'from transformers import MarianMTModel, MarianTokenizer',
            '',
            `model_name = "${modelName}"`,
            `model_path = r"${modelPath.replace(/\\/g, '\\\\')}"`,
            `possible_ids = ${JSON.stringify(possibleModelIds)}`,
            '',
            'Path(model_path).mkdir(parents=True, exist_ok=True)',
            '',
            'success = False',
            'error_msg = None',
            'model_id_used = None',
            '',
            'for model_id in possible_ids:',
            '    try:',
            '        print(f"PROGRESS:Attempting to download {model_id}...", file=sys.stderr, flush=True)',
            '        ',
            '        # Download with progress',
            '        print(f"PROGRESS:Downloading tokenizer...", file=sys.stderr, flush=True)',
            '        tokenizer = MarianTokenizer.from_pretrained(model_id, local_files_only=False)',
            '        ',
            '        print(f"PROGRESS:Downloading model weights...", file=sys.stderr, flush=True)',
            '        model = MarianMTModel.from_pretrained(model_id, local_files_only=False)',
            '        ',
            '        print(f"PROGRESS:Saving to local directory...", file=sys.stderr, flush=True)',
            '        tokenizer.save_pretrained(model_path)',
            '        model.save_pretrained(model_path)',
            '        ',
            '        print(f"PROGRESS:Download complete", file=sys.stderr, flush=True)',
            '        success = True',
            '        model_id_used = model_id',
            '        break',
            '    except Exception as e:',
            '        error_msg = str(e)',
            '        print(f"PROGRESS:Failed to download {model_id}: {error_msg}", file=sys.stderr, flush=True)',
            '        continue',
            '',
            'if success:',
            '    print(json.dumps({"success": True, "model_id": model_id_used}))',
            'else:',
            '    print(json.dumps({"success": False, "error": error_msg or "Model not found on HuggingFace"}))',
            '    sys.exit(1)'
        ].join('\n');
        
        const venvPython = path.join(PROJECT_ROOT, '.venv', 'bin', 'python3');
        const pythonPath = fs.existsSync(venvPython) ? venvPython : 'python3';
        const pythonProcess = spawn(pythonPath, ['-c', scriptContent]);
        
        let stdout = '';
        let stderr = '';
        let timeoutId = null;
        
        // Set timeout (10 minutes for model downloads)
        const timeout = 10 * 60 * 1000;
        timeoutId = setTimeout(() => {
            pythonProcess.kill();
            reject(new Error(`Download timeout after ${timeout / 1000 / 60} minutes`));
        }, timeout);
        
        pythonProcess.stdout.on('data', (data) => {
            stdout += data.toString();
        });
        
        pythonProcess.stderr.on('data', (data) => {
            const output = data.toString();
            stderr += output;
            
            // Parse progress messages
            const lines = output.split('\n');
            for (const line of lines) {
                if (line.startsWith('PROGRESS:')) {
                    const message = line.substring(9).trim();
                    if (progressCallback) {
                        progressCallback(message);
                    } else {
                        console.log(`    ${message}`);
                    }
                }
            }
        });
        
        pythonProcess.on('close', (code) => {
            if (timeoutId) clearTimeout(timeoutId);
            
            if (code !== 0) {
                reject(new Error(`Download failed: ${stderr || 'Unknown error'}`));
                return;
            }
            
            try {
                const result = JSON.parse(stdout.trim());
                if (result.success) {
                    resolve(result.model_id);
                } else {
                    reject(new Error(result.error || 'Download failed'));
                }
            } catch (e) {
                reject(new Error(`Failed to parse download result: ${e.message}`));
            }
        });
        
        pythonProcess.on('error', (error) => {
            if (timeoutId) clearTimeout(timeoutId);
            reject(new Error(`Failed to start Python process: ${error.message}`));
        });
    });
}

// Download voice from HuggingFace using Python
async function downloadVoiceFromHuggingFace(voiceKey, langCode, targetPath, progressCallback = null) {
    return new Promise((resolve, reject) => {
        // Check if voice files already exist
        const onnxExists = fs.existsSync(targetPath);
        const jsonExists = fs.existsSync(targetPath + '.json');
        
        if (onnxExists && jsonExists) {
            // Verify it's not a Git LFS pointer
            try {
                const stats = fs.statSync(targetPath);
                if (stats.size >= 1024) {
                    // File exists and is not a pointer
                    if (progressCallback) progressCallback('Voice already exists, skipping download');
                    resolve('already_exists');
                    return;
                }
            } catch (e) {
                // Continue with download
            }
        }
        
        // Extract locale, voice name, and quality from voiceKey (format: fi_FI-harri-low)
        const parts = voiceKey.split('-');
        const locale = parts[0] || langCode;
        const voiceName = parts.length > 1 ? parts.slice(0, -1).join('-') : voiceKey;
        const quality = parts[parts.length - 1] || 'low';
        
        const scriptContent = [
            'import sys',
            'import json',
            'from pathlib import Path',
            'from huggingface_hub import hf_hub_download',
            'import shutil',
            '',
            `voice_key = "${voiceKey}"`,
            `lang_code = "${langCode}"`,
            `locale = "${locale}"`,
            `voice_name = "${voiceName}"`,
            `quality = "${quality}"`,
            `target_path = r"${targetPath.replace(/\\/g, '\\\\')}"`,
            '',
            'Path(target_path).parent.mkdir(parents=True, exist_ok=True)',
            '',
            'try:',
            '    # Try different possible paths in rhasspy/piper-voices',
            '    possible_paths = [',
            '        f"{lang_code}/{voice_key}.onnx",',
            '        f"{locale}/{voice_key}.onnx",',
            '        f"{lang_code}/{locale}/{voice_name}/{quality}/{voice_key}.onnx",',
            '    ]',
            '    ',
            '    onnx_path = None',
            '    json_path = None',
            '    ',
            '    for file_path in possible_paths:',
            '        try:',
            '            print(f"PROGRESS:Trying path: {file_path}", file=sys.stderr, flush=True)',
            '            print(f"PROGRESS:Downloading ONNX file...", file=sys.stderr, flush=True)',
            '            onnx_path = hf_hub_download(',
            '                repo_id="rhasspy/piper-voices",',
            '                filename=file_path,',
            '                local_dir=None',
            '            )',
            '            print(f"PROGRESS:Downloading JSON config...", file=sys.stderr, flush=True)',
            '            json_path = hf_hub_download(',
            '                repo_id="rhasspy/piper-voices",',
            '                filename=file_path + ".json",',
            '                local_dir=None',
            '            )',
            '            print(f"PROGRESS:Copying files to target location...", file=sys.stderr, flush=True)',
            '            shutil.copy2(onnx_path, target_path)',
            '            shutil.copy2(json_path, target_path + ".json")',
            '            print(f"PROGRESS:Download complete", file=sys.stderr, flush=True)',
            '            break',
            '        except Exception as e:',
            '            print(f"PROGRESS:Path {file_path} failed: {str(e)}", file=sys.stderr, flush=True)',
            '            continue',
            '    ',
            '    if onnx_path and json_path:',
            '        print(json.dumps({"success": True}))',
            '    else:',
            '        print(json.dumps({"success": False, "error": "Voice not found on HuggingFace"}))',
            '        sys.exit(1)',
            'except Exception as e:',
            '    print(json.dumps({"success": False, "error": str(e)}))',
            '    sys.exit(1)'
        ].join('\n');
        
        const venvPython = path.join(PROJECT_ROOT, '.venv', 'bin', 'python3');
        const pythonPath = fs.existsSync(venvPython) ? venvPython : 'python3';
        const pythonProcess = spawn(pythonPath, ['-c', scriptContent]);
        
        let stdout = '';
        let stderr = '';
        let timeoutId = null;
        
        // Set timeout (5 minutes for voice downloads)
        const timeout = 5 * 60 * 1000;
        timeoutId = setTimeout(() => {
            pythonProcess.kill();
            reject(new Error(`Download timeout after ${timeout / 1000 / 60} minutes`));
        }, timeout);
        
        pythonProcess.stdout.on('data', (data) => {
            stdout += data.toString();
        });
        
        pythonProcess.stderr.on('data', (data) => {
            const output = data.toString();
            stderr += output;
            
            // Parse progress messages
            const lines = output.split('\n');
            for (const line of lines) {
                if (line.startsWith('PROGRESS:')) {
                    const message = line.substring(9).trim();
                    if (progressCallback) {
                        progressCallback(message);
                    } else {
                        console.log(`    ${message}`);
                    }
                }
            }
        });
        
        pythonProcess.on('close', (code) => {
            if (timeoutId) clearTimeout(timeoutId);
            
            if (code !== 0) {
                reject(new Error(`Download failed: ${stderr || 'Unknown error'}`));
                return;
            }
            
            try {
                const result = JSON.parse(stdout.trim());
                if (result.success) {
                    resolve(true);
                } else {
                    reject(new Error(result.error || 'Download failed'));
                }
            } catch (e) {
                reject(new Error(`Failed to parse download result: ${e.message}`));
            }
        });
        
        pythonProcess.on('error', (error) => {
            if (timeoutId) clearTimeout(timeoutId);
            reject(new Error(`Failed to start Python process: ${error.message}`));
        });
    });
}

// Initialize on startup
async function initialize() {
    console.log('Initializing server...');
    console.log(`Scanning models directory: ${MODELS_DIR}`);
    try {
        await scanModels();
    } catch (error) {
        console.error(`Error scanning models: ${error.message}`);
        console.log('Continuing with available models...');
    }
    console.log(`Scanning voices directory: ${VOICES_DIR}`);
    try {
        await scanVoices();
    } catch (error) {
        console.error(`Error scanning voices: ${error.message}`);
        console.log('Continuing without voices...');
    }
}

// Scan models directory
async function scanModels() {
    availableModels.clear();
    
    if (!await fs.pathExists(MODELS_DIR)) {
        console.log('Models directory not found');
        return;
    }
    
    let entries;
    try {
        entries = await fs.readdir(MODELS_DIR, { withFileTypes: true });
    } catch (error) {
        console.error(`Error reading models directory: ${error.message}`);
        return;
    }
    
    for (const entry of entries) {
        if (entry.isDirectory() && entry.name.startsWith('en-')) {
            try {
            const modelPath = path.join(MODELS_DIR, entry.name);
            const configPath = path.join(modelPath, 'config.json');
            const tokenizerJsonPath = path.join(modelPath, 'tokenizer.json');
            
            let hasConfig = false;
            let hasTokenizerJson = false;
            let hasWeights = false;
            let hasDecoder = false;
            let hasNpz = false;
            
            // Check for HuggingFace format (non-fatal)
            try {
                hasConfig = await fs.pathExists(configPath);
                hasTokenizerJson = await fs.pathExists(tokenizerJsonPath);
                
                hasWeights = (
                    await fs.pathExists(path.join(modelPath, 'model.safetensors')) ||
                    await fs.pathExists(path.join(modelPath, 'pytorch_model.bin')) ||
                    await fs.pathExists(path.join(modelPath, 'model.bin'))
                );
            } catch (error) {
                // Continue to check OPUS-MT format
            }
            
            // Check for OPUS-MT format (non-fatal)
            try {
                hasDecoder = await fs.pathExists(path.join(modelPath, 'decoder.yml'));
                
                // Check for .npz files
                try {
                    const files = await fs.readdir(modelPath);
                    hasNpz = files.some(f => f.endsWith('.npz'));
                } catch (error) {
                    // If we can't read directory, skip npz check
                }
            } catch (error) {
                // Continue
            }
            
            // Check if model uses SentencePiece (.spm files) - these should use Python
            let hasSentencePiece = false;
            try {
                const files = await fs.readdir(modelPath);
                hasSentencePiece = files.some(f => f.endsWith('.spm'));
            } catch (error) {
                // Skip if we can't read directory
            }
            
            // Mark as complete if it has HuggingFace format (config + weights) OR OPUS-MT format (decoder + npz)
            const complete = (hasConfig && hasWeights) || (hasDecoder && hasNpz);
            const langCode = entry.name.split('-')[1] || '';
            const flagPath = `/flags/${langCode}.svg`;
            
            // All models use @xenova/transformers (no Python dependencies)
            const usePython = false;
            
            availableModels.set(entry.name, {
                name: entry.name,
                path: modelPath,
                lang_code: langCode,
                flag_path: flagPath,
                complete: complete,
                hasTokenizerJson: hasTokenizerJson,
                usePython: false  // Always use JavaScript/transformers
            });
            } catch (error) {
                console.error(`  Error processing model ${entry.name}: ${error.message}`);
                console.log(`  Skipping ${entry.name}, continuing with other models`);
            }
        }
    }
    
    // Log scan results and attempt downloads for incomplete models
    const completeModels = Array.from(availableModels.values()).filter(m => m.complete);
    const incompleteModels = Array.from(availableModels.values()).filter(m => !m.complete);
    
    if (completeModels.length > 0) {
        console.log(`Found ${completeModels.length} complete model(s):`);
        completeModels.forEach(m => {
            const hasTokenizer = m.hasTokenizerJson ? ' (has tokenizer.json)' : ' (SentencePiece)';
            console.log(`  [COMPLETE] ${m.name} (${m.lang_code}) [JavaScript/@xenova/transformers]${hasTokenizer}`);
            console.log(`    Path: ${m.path}`);
        });
    }
    
    if (incompleteModels.length > 0) {
        console.log(`Found ${incompleteModels.length} incomplete model(s):`);
        for (const m of incompleteModels) {
            try {
                const missing = [];
                
                // Check HuggingFace format
                try {
                    if (!await fs.pathExists(path.join(m.path, 'config.json'))) {
                        missing.push('config.json');
                    }
                    const hasWeights = (
                        await fs.pathExists(path.join(m.path, 'model.safetensors')) ||
                        await fs.pathExists(path.join(m.path, 'pytorch_model.bin')) ||
                        await fs.pathExists(path.join(m.path, 'model.bin'))
                    );
                    if (!hasWeights) missing.push('model weights');
                } catch (error) {
                    // Continue to check OPUS-MT format
                }
                
                // Check OPUS-MT format
                try {
                    const hasDecoder = await fs.pathExists(path.join(m.path, 'decoder.yml'));
                    if (!hasDecoder) missing.push('decoder.yml');
                    
                    let hasNpz = false;
                    try {
                        const files = await fs.readdir(m.path);
                        hasNpz = files.some(f => f.endsWith('.npz'));
                    } catch (error) {
                        // Skip
                    }
                    if (!hasNpz) missing.push('.npz file');
                } catch (error) {
                    // Skip
                }
                
                console.log(`  [INCOMPLETE] ${m.name} (${m.lang_code})`);
                console.log(`    Path: ${m.path}`);
                if (missing.length > 0) {
                    console.log(`    Missing: ${missing.join(', ')}`);
                }
                
                // Skip download attempts - models should be pre-installed
                console.log(`    Skipping incomplete model ${m.name} - will not be available for translation`);
            } catch (error) {
                console.error(`    Error processing incomplete model ${m.name}: ${error.message}`);
                console.log(`    Skipping ${m.name}, continuing with other models`);
            }
        }
    }
    
    if (availableModels.size === 0) {
        console.log('No models found in models directory');
    }
}

// Scan voices directory
async function scanVoices(langCode = null) {
    if (!langCode) {
        availableVoices.clear();
    }
    
    if (!await fs.pathExists(VOICES_DIR)) {
        if (!langCode) {
            console.log('Voices directory not found');
        }
        return;
    }
    
    // Get language codes from available models (only scan voices for languages with models)
    const modelLangCodes = new Set();
    for (const modelInfo of availableModels.values()) {
        if (modelInfo.complete && modelInfo.lang_code) {
            modelLangCodes.add(modelInfo.lang_code);
        }
    }
    
    // If a specific langCode is requested, check if we have a model for it
    if (langCode) {
        if (!modelLangCodes.has(langCode)) {
            // No model for this language, skip voice scan
            return;
        }
    } else {
        // No models available at all, skip voice scan
        if (modelLangCodes.size === 0) {
            console.log('No models available, skipping voice scan');
            return;
        }
    }
    
    // Get all voice language directories
    const allLangDirs = (await fs.readdir(VOICES_DIR, { withFileTypes: true }))
        .filter(e => e.isDirectory())
        .map(e => ({ path: path.join(VOICES_DIR, e.name), lang: e.name }));
    
    // Filter to only include languages that have models
    const langDirs = langCode 
        ? (modelLangCodes.has(langCode) ? [path.join(VOICES_DIR, langCode)] : [])
        : allLangDirs
            .filter(dir => modelLangCodes.has(dir.lang))
            .map(dir => dir.path);
    
    // Log which voice directories are being scanned
    if (!langCode) {
        const skippedLangs = allLangDirs
            .filter(dir => !modelLangCodes.has(dir.lang))
            .map(dir => dir.lang);
        
        if (langDirs.length > 0) {
            console.log(`Scanning voices for ${langDirs.length} language(s) with models:`);
            langDirs.forEach(langDir => {
                const lang = path.basename(langDir);
                console.log(`  - ${lang}`);
            });
        }
        
        if (skippedLangs.length > 0) {
            console.log(`Skipping ${skippedLangs.length} voice language(s) without models: ${skippedLangs.join(', ')}`);
        }
    }
    
    for (const langDir of langDirs) {
        if (!await fs.pathExists(langDir)) continue;
        
        const lang = path.basename(langDir);
        if (langCode && lang !== langCode) continue;
        
        const voicesList = [];
        
        try {
            const localeDirs = (await fs.readdir(langDir, { withFileTypes: true }))
                .filter(e => e.isDirectory())
                .map(e => path.join(langDir, e.name));
            
            for (const localeDir of localeDirs) {
                try {
                    const locale = path.basename(localeDir);
                    let voiceDirs = [];
                    try {
                        voiceDirs = (await fs.readdir(localeDir, { withFileTypes: true }))
                            .filter(e => e.isDirectory())
                            .map(e => path.join(localeDir, e.name));
                    } catch (error) {
                        console.log(`  Warning: Cannot read locale directory ${localeDir}: ${error.message}`);
                        continue;
                    }
                    
                    for (const voiceDir of voiceDirs) {
                        try {
                            const voiceName = path.basename(voiceDir);
                            let qualityDirs = [];
                            try {
                                qualityDirs = (await fs.readdir(voiceDir, { withFileTypes: true }))
                                    .filter(e => e.isDirectory())
                                    .map(e => path.join(voiceDir, e.name));
                            } catch (error) {
                                console.log(`  Warning: Cannot read voice directory ${voiceDir}: ${error.message}`);
                                continue;
                            }
                            
                            for (const qualityDir of qualityDirs) {
                                try {
                                    const quality = path.basename(qualityDir);
                                    let onnxFiles = [];
                                    try {
                                        onnxFiles = (await fs.readdir(qualityDir))
                                            .filter(f => f.endsWith('.onnx'));
                                    } catch (error) {
                                        console.log(`  Warning: Cannot read quality directory ${qualityDir}: ${error.message}`);
                                        continue;
                                    }
                                    
                                    for (const onnxFile of onnxFiles) {
                                        // Normalize paths to absolute paths for cross-platform compatibility (Windows/Linux)
                                        const onnxPath = path.resolve(qualityDir, onnxFile);
                                        const jsonPath = path.resolve(qualityDir, onnxFile + '.json');
                                        const voiceKey = path.basename(onnxFile, '.onnx');
                                        
                                        // Check if files are complete
                                        const hasOnnx = await fs.pathExists(onnxPath);
                                        const hasJson = await fs.pathExists(jsonPath);
                                        
                                        // Skip Git LFS pointers - only flag as pointer if file is small AND contains LFS pointer text
                                        let isLfsPointer = false;
                                        let fileSize = 0;
                                        if (hasOnnx) {
                                            try {
                                                const stats = await fs.stat(onnxPath);
                                                fileSize = stats.size;
                                                // ONNX files should be at least several KB - files < 1KB are likely LFS pointers
                                                if (stats.size < 1024) {
                                                    // Small file - check if it's an LFS pointer
                                                    const firstBytes = await fs.readFile(onnxPath, { encoding: 'utf8', flag: 'r' });
                                                    if (firstBytes.includes('version https://git-lfs.github.com/spec/v1')) {
                                                        isLfsPointer = true;
                                                        console.log(`  [LFS POINTER] ${voiceKey}: File size ${stats.size} bytes, contains LFS pointer text`);
                                                    } else {
                                                        // Very small file but not LFS pointer - might be corrupted
                                                        console.log(`  [WARNING] ${voiceKey}: File size ${stats.size} bytes, but not an LFS pointer - may be corrupted`);
                                                    }
                                                } else {
                                                    // File is large enough to be valid - not an LFS pointer
                                                    // ONNX files are typically several MB, so anything >= 1KB that's not an LFS pointer is likely valid
                                                }
                                            } catch (e) {
                                                console.log(`  [WARNING] ${voiceKey}: Error checking file: ${e.message}`);
                                                // Continue - don't mark as LFS pointer if we can't check
                                            }
                                        }
                                        
                                        // Debug logging for server diagnosis
                                        if (!hasOnnx || !hasJson || isLfsPointer) {
                                            console.log(`  [VOICE CHECK] ${voiceKey}: hasOnnx=${hasOnnx}, hasJson=${hasJson}, isLfsPointer=${isLfsPointer}, fileSize=${fileSize} bytes`);
                                            console.log(`  [VOICE CHECK] ONNX path: ${onnxPath}`);
                                            console.log(`  [VOICE CHECK] JSON path: ${jsonPath}`);
                                        }
                                        
                                        // If incomplete or LFS pointer, try to download
                                        if ((!hasOnnx || !hasJson || isLfsPointer) && !langCode) {
                                            console.log(`  [INCOMPLETE VOICE] ${voiceKey} (${lang}/${locale}/${voiceName}/${quality})`);
                                            if (isLfsPointer) {
                                                console.log(`    Git LFS pointer detected, attempting download...`);
                                            } else {
                                                console.log(`    Missing: ${!hasOnnx ? 'onnx file' : ''}${!hasOnnx && !hasJson ? ', ' : ''}${!hasJson ? 'json file' : ''}`);
                                                console.log(`    Attempting to download from HuggingFace...`);
                                            }
                                            
                                            try {
                                                const progressCallback = (message) => {
                                                    console.log(`    ${message}`);
                                                };
                                                
                                                const result = await downloadVoiceFromHuggingFace(voiceKey, lang, onnxPath, progressCallback);
                                                
                                                if (result === 'already_exists') {
                                                    console.log(`    Voice already exists locally`);
                                                } else {
                                                    console.log(`    Successfully downloaded ${voiceKey}`);
                                                }
                                                
                                                // Re-check after download
                                                const newHasOnnx = await fs.pathExists(onnxPath);
                                                const newHasJson = await fs.pathExists(jsonPath);
                                                
                                                if (newHasOnnx && newHasJson) {
                                                    // Verify it's not a pointer
                                                    let isValid = true;
                                                    try {
                                                        const stats = await fs.stat(onnxPath);
                                                        if (stats.size < 1024) {
                                                            const firstBytes = await fs.readFile(onnxPath);
                                                            if (firstBytes.toString().includes('version https://git-lfs.github.com/spec/v1')) {
                                                                isValid = false;
                                                            }
                                                        }
                                                    } catch (e) {
                                                        isValid = false;
                                                    }
                                                    
                                                    if (isValid) {
                                                        // Try to load config
                                                        try {
                                                            const config = await fs.readJson(jsonPath);
                                                            voicesList.push({
                                                                key: voiceKey,
                                                                name: voiceName,
                                                                quality: quality,
                                                                locale: locale,
                                                                onnx_path: onnxPath,
                                                                json_path: jsonPath,
                                                                config: config,
                                                                display_name: `${voiceName} (${quality})`
                                                            });
                                                            console.log(`    Voice ${voiceKey} is now complete`);
                                                        } catch (e) {
                                                            console.log(`    Voice ${voiceKey} downloaded but config invalid: ${e.message}`);
                                                        }
                                                    } else {
                                                        console.log(`    Voice ${voiceKey} still appears to be a Git LFS pointer`);
                                                    }
                                                } else {
                                                    console.log(`    Voice ${voiceKey} still incomplete after download`);
                                                }
                                            } catch (error) {
                                                console.log(`    Failed to download: ${error.message}`);
                                                console.log(`    Skipping ${voiceKey}, continuing with other voices`);
                                            }
                                            continue;
                                        }
                                        
                                        // If complete, add to list
                                        if (hasOnnx && hasJson && !isLfsPointer) {
                                            try {
                                                const config = await fs.readJson(jsonPath);
                                                
                                                voicesList.push({
                                                    key: voiceKey,
                                                    name: voiceName,
                                                    quality: quality,
                                                    locale: locale,
                                                    onnx_path: onnxPath,
                                                    json_path: jsonPath,
                                                    config: config,
                                                    display_name: `${voiceName} (${quality})`
                                                });
                                            } catch (e) {
                                                // Skip invalid configs but continue
                                                console.log(`  Warning: Failed to load voice config for ${voiceKey}: ${e.message}`);
                                            }
                                        }
                                    }
                                } catch (error) {
                                    // Skip this quality directory but continue with others
                                    console.log(`  Warning: Error processing quality directory ${qualityDir}: ${error.message}`);
                                }
                            }
                        } catch (error) {
                            // Skip this voice directory but continue with others
                            console.log(`  Warning: Error processing voice directory ${voiceDir}: ${error.message}`);
                        }
                    }
                } catch (error) {
                    // Skip this locale directory but continue with others
                    console.log(`  Warning: Error processing locale directory ${localeDir}: ${error.message}`);
                }
            }
        } catch (e) {
            // Skip languages with errors but continue
            console.log(`  Warning: Error scanning voices for language ${lang}: ${e.message}`);
        }
        
        if (voicesList.length > 0) {
            availableVoices.set(lang, voicesList);
        }
    }
    
    // Log scan results (only on initial scan, not per-language)
    if (!langCode) {
        const totalVoices = Array.from(availableVoices.values()).reduce((sum, v) => sum + v.length, 0);
        if (totalVoices > 0) {
            console.log(`Found ${totalVoices} complete voice(s) across ${availableVoices.size} language(s):`);
            for (const [lang, voices] of availableVoices.entries()) {
                console.log(`  [COMPLETE] ${lang}: ${voices.length} voice(s)`);
                voices.forEach(v => {
                    console.log(`    * ${v.key} (${v.locale}/${v.name}/${v.quality})`);
                    console.log(`      Path: ${v.onnx_path}`);
                });
            }
        } else {
            console.log('No complete voices found in voices directory');
        }
    }
}

// Load translation model
async function loadTranslationModel(modelName) {
    if (loadedModels.has(modelName)) {
        return loadedModels.get(modelName);
    }
    
    const modelInfo = availableModels.get(modelName);
    if (!modelInfo || !modelInfo.complete) {
        throw new Error(`Model ${modelName} not found or incomplete`);
    }
    
    // Load with @xenova/transformers (no Python dependencies)
    try {
        const modelPath = path.resolve(modelInfo.path);
        
        const translator = await pipeline('translation', modelPath, {
            device: 'cpu',
            local_files_only: true
        });
        
        loadedModels.set(modelName, {
            translator,
            modelInfo,
            usePython: false  // Always false, no Python dependencies
        });
        
        return loadedModels.get(modelName);
    } catch (error) {
        throw new Error(`Failed to load model: ${error.message}`);
    }
}

// Python translation removed - all models now use @xenova/transformers

// Translate text - uses @xenova/transformers only (no Python dependencies)
async function translateText(text, modelName) {
    const model = await loadTranslationModel(modelName);
    
    // Always use @xenova/transformers (no Python dependencies)
    const result = await model.translator(text, {
        max_length: 512,
        num_beams: 4,
        early_stopping: true
    });
    const translated = result[0].translation_text;
    
    // Log translation in clean format
    const timestamp = new Date().toISOString();
    console.log(`[${timestamp}] Translation: "${text.substring(0, 50)}${text.length > 50 ? '...' : ''}" -> "${translated.substring(0, 50)}${translated.length > 50 ? '...' : ''}" (${modelName})`);
    
    return translated;
}

// Load TTS voice
async function loadVoiceModel(voiceKey, langCode) {
    const cacheKey = `${langCode}:${voiceKey}`;
    
    if (loadedVoices.has(cacheKey)) {
        return loadedVoices.get(cacheKey);
    }
    
    const voices = availableVoices.get(langCode) || [];
    const voice = voices.find(v => v.key === voiceKey);
    
    if (!voice) {
        throw new Error(`Voice ${voiceKey} not found for language ${langCode}`);
    }
    
    if (!ort) {
        const errorMsg = ortLoadError 
            ? `TTS is not available: onnxruntime-node failed to load: ${ortLoadError.message}. TTS requires onnxruntime-node to be properly installed.`
            : 'TTS is not available: ONNX Runtime is not loaded.';
        throw new Error(errorMsg);
    }
    
        try {
            // Normalize path to absolute path for cross-platform compatibility (Windows/Linux)
            const onnxPath = path.resolve(voice.onnx_path);
            
            // Verify file exists before attempting to load
            if (!await fs.pathExists(onnxPath)) {
                throw new Error(`ONNX file not found: ${onnxPath}`);
            }
            
            const session = await ort.InferenceSession.create(onnxPath, {
                executionProviders: ['cpu']
            });
            
            const config = voice.config;
            
            loadedVoices.set(cacheKey, {
                session,
                config,
                voice
            });
            
            return loadedVoices.get(cacheKey);
        } catch (error) {
            throw new Error(`Failed to load voice: ${error.message}`);
        }
}

// Phonemize text using espeak-ng
async function phonemizeTextEspeak(text, voice) {
    return new Promise((resolve, reject) => {
        const espeakBin = 'espeak-ng'; // Try espeak-ng first
        
        // Helper function to run espeak
        const runEspeak = (espeakPath) => {
            const espeakProcess = spawn(espeakPath, ['-q', '-x', '-v', voice, text]);
            let stdout = '';
            let stderr = '';
            
            espeakProcess.stdout.on('data', (data) => {
                stdout += data.toString();
            });
            
            espeakProcess.stderr.on('data', (data) => {
                stderr += data.toString();
            });
            
            espeakProcess.on('close', (code) => {
                if (code !== 0) {
                    reject(new Error(`espeak-ng failed: ${stderr}`));
                } else {
                    resolve(stdout.trim());
                }
            });
            
            espeakProcess.on('error', (error) => {
                reject(new Error(`Failed to start espeak: ${error.message}`));
            });
        };
        
        // Use 'where' on Windows, 'which' on Unix/Linux
        const whichCmd = process.platform === 'win32' ? 'where' : 'which';
        const whichProcess = spawn(whichCmd, [espeakBin]);
        let whichOutput = '';
        
        whichProcess.stdout.on('data', (data) => {
            whichOutput += data.toString();
        });
        
        whichProcess.stderr.on('data', () => {
            // Ignore stderr from which/where
        });
        
        whichProcess.on('error', (error) => {
            // If 'which'/'where' command fails, fallback to espeak-ng directly
            runEspeak(espeakBin);
        });
        
        whichProcess.on('close', (code) => {
            // On Windows, 'where' returns 0 if found, non-zero if not found
            // On Unix, 'which' returns 0 if found, non-zero if not found
            // Extract the first line of output (path to executable)
            let espeakPath = whichOutput.trim().split(/[\r\n]+/)[0].trim();
            if (!espeakPath || code !== 0) {
                espeakPath = 'espeak'; // Fallback to espeak
            }
            runEspeak(espeakPath);
        });
    });
}

// Generate cache key for TTS
function generateTTSCacheKey(text, voiceKey, langCode, lengthScale, noiseScale, noiseW) {
    // Normalize parameters to avoid cache misses due to floating point precision
    const normalizedLengthScale = Math.round(lengthScale * 100) / 100;
    const normalizedNoiseScale = Math.round(noiseScale * 1000) / 1000;
    const normalizedNoiseW = Math.round(noiseW * 1000) / 1000;
    const keyString = `${text}|${voiceKey}|${langCode}|${normalizedLengthScale}|${normalizedNoiseScale}|${normalizedNoiseW}`;
    return crypto.createHash('sha256').update(keyString).digest('hex');
}

// Check if piper CLI tool is available
async function checkPiperCLI() {
    return new Promise((resolve) => {
        const piperProcess = spawn('piper', ['--version'], { stdio: 'ignore' });
        piperProcess.on('close', (code) => {
            resolve(code === 0);
        });
        piperProcess.on('error', () => {
            resolve(false);
        });
    });
}

// Synthesize speech using piper CLI tool (standalone binary, no dependencies)
async function synthesizeSpeechWithPiperCLI(text, voiceKey, langCode, lengthScale, noiseScale, noiseW, cacheKey) {
    return new Promise(async (resolve, reject) => {
        try {
            // Get voice info
            const voices = availableVoices.get(langCode) || [];
            const voice = voices.find(v => v.key === voiceKey);
            
            if (!voice) {
                throw new Error(`Voice ${voiceKey} not found for language ${langCode}`);
            }
            
            const onnxPath = voice.onnx_path;
            
            // Normalize text
            let normalizedText = text.trim();
            if (!normalizedText) {
                throw new Error('Text cannot be empty');
            }
            normalizedText = normalizedText.replace(/\r\n/g, ' ').replace(/\r/g, ' ');
            normalizedText = normalizedText.replace(/\n/g, '. ');
            
            // Create temp file for output
            const os = await import('os');
            const tmpPath = path.join(os.tmpdir(), `piper-${Date.now()}-${Math.random().toString(36).substring(7)}.wav`);
            
            try {
                // Run piper CLI
                const piperProcess = spawn('piper', [
                    '--model', onnxPath,
                    '--output_file', tmpPath,
                    '--length_scale', lengthScale.toString(),
                    '--noise_scale', noiseScale.toString(),
                    '--noise_w', noiseW.toString()
                ]);
                
                let stderr = '';
                piperProcess.stderr.on('data', (data) => {
                    stderr += data.toString();
                });
                
                piperProcess.stdin.write(normalizedText);
                piperProcess.stdin.end();
                
                piperProcess.on('close', async (code) => {
                    try {
                        if (code !== 0) {
                            throw new Error(`Piper CLI failed with code ${code}: ${stderr}`);
                        }
                        
                        // Read the generated WAV file
                        const wavBuffer = await fs.readFile(tmpPath);
                        
                        if (wavBuffer.length === 0) {
                            throw new Error('Piper CLI generated empty audio file');
                        }
                        
                        // Cache the audio
                        if (ttsCache.size < 100) {
                            ttsCache.set(cacheKey, wavBuffer);
                        } else {
                            const firstKey = ttsCache.keys().next().value;
                            ttsCache.delete(firstKey);
                            ttsCache.set(cacheKey, wavBuffer);
                        }
                        
                        // Clean up temp file
                        await fs.unlink(tmpPath).catch(() => {});
                        
                        resolve(wavBuffer);
                    } catch (error) {
                        await fs.unlink(tmpPath).catch(() => {});
                        reject(error);
                    }
                });
                
                piperProcess.on('error', async (error) => {
                    await fs.unlink(tmpPath).catch(() => {});
                    reject(new Error(`Failed to start piper CLI: ${error.message}`));
                });
            } catch (error) {
                await fs.unlink(tmpPath).catch(() => {});
                reject(error);
            }
        } catch (error) {
            reject(error);
        }
    });
}

// Synthesize speech using Node.js native ONNX Runtime or piper CLI fallback
async function synthesizeSpeech(text, voiceKey, langCode, lengthScale = 1.0, noiseScale = 0.667, noiseW = 0.8) {
    // Check cache first
    const cacheKey = generateTTSCacheKey(text, voiceKey, langCode, lengthScale, noiseScale, noiseW);
    if (ttsCache.has(cacheKey)) {
        return ttsCache.get(cacheKey);
    }
    
    // Try piper CLI tool first (standalone binary, no dependencies)
    const hasPiperCLI = await checkPiperCLI();
    if (hasPiperCLI) {
        try {
            return await synthesizeSpeechWithPiperCLI(text, voiceKey, langCode, lengthScale, noiseScale, noiseW, cacheKey);
        } catch (error) {
            console.log(`[TTS] Piper CLI failed: ${error.message}, trying ONNX Runtime...`);
        }
    }
    
    // Fallback to ONNX Runtime
    if (!ort) {
        const errorMsg = ortLoadError 
            ? `TTS is not available: onnxruntime-node failed to load: ${ortLoadError.message}. TTS requires onnxruntime-node to be properly installed.`
            : 'TTS is not available: ONNX Runtime is not loaded.';
        throw new Error(errorMsg);
    }
    
    try {
        // Normalize text
        let normalizedText = text.trim();
        if (!normalizedText) {
            throw new Error('Text cannot be empty');
        }
        normalizedText = normalizedText.replace(/\r\n/g, ' ').replace(/\r/g, ' ');
        normalizedText = normalizedText.replace(/\n/g, '. ');
        
        // Load voice model (this will fail if onnxruntime-node native bindings aren't available)
        let voiceModel;
        try {
            voiceModel = await loadVoiceModel(voiceKey, langCode);
        } catch (error) {
            // If loadVoiceModel fails due to native binding issues, provide clear error
            if (error.message && (error.message.includes('cannot run') || error.message.includes('.node') || error.message.includes('DLL'))) {
                throw new Error('TTS is not available: onnxruntime-node native bindings cannot be loaded. This requires Visual C++ Redistributables on Windows. TTS functionality is disabled.');
            }
            throw error;
        }
        const config = voiceModel.config;
        const session = voiceModel.session;
        
        // Get phoneme type and espeak voice
        const phonemeType = config.phoneme_type || 'espeak';
        const espeakConfig = config.espeak || {};
        const espeakVoice = espeakConfig.voice || langCode.split('_')[0];
        
        // Phonemize text
        let phonemesStr;
        if (phonemeType === 'text') {
            phonemesStr = normalizedText;
        } else {
            phonemesStr = await phonemizeTextEspeak(normalizedText, espeakVoice);
        }
        
        // Apply phoneme mapping
        const phonemeMap = config.phoneme_map || {};
        for (const [src, dst] of Object.entries(phonemeMap)) {
            phonemesStr = phonemesStr.replace(new RegExp(src.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'g'), dst);
        }
        
        // Normalize phonemes
        phonemesStr = phonemesStr.replace(/\n/g, ' ').replace(/\t/g, ' ');
        phonemesStr = phonemesStr.replace(/ +/g, ' ').trim();
        
        // Convert phonemes to IDs
        const phonemeIdMap = config.phoneme_id_map;
        if (!phonemeIdMap) {
            throw new Error('phoneme_id_map not found in config');
        }
        
        const phonemeIds = [];
        
        // Add start token
        if (phonemeIdMap['^']) {
            const tokenIds = Array.isArray(phonemeIdMap['^']) ? phonemeIdMap['^'] : [phonemeIdMap['^']];
            phonemeIds.push(...tokenIds.map(x => parseInt(x)));
        }
        
        // Convert phonemes character by character
        for (const char of phonemesStr) {
            if (char in phonemeIdMap) {
                const tokenIds = Array.isArray(phonemeIdMap[char]) ? phonemeIdMap[char] : [phonemeIdMap[char]];
                phonemeIds.push(...tokenIds.map(x => parseInt(x)));
            } else if (/\s/.test(char)) {
                // Handle spaces
                if (' ' in phonemeIdMap) {
                    const tokenIds = Array.isArray(phonemeIdMap[' ']) ? phonemeIdMap[' '] : [phonemeIdMap[' ']];
                    const spaceIds = tokenIds.map(x => parseInt(x));
                    phonemeIds.push(...spaceIds);
                }
            } else if (/[a-zA-Z]/.test(char)) {
                // Try case-insensitive match
                const charLower = char.toLowerCase();
                const charUpper = char.toUpperCase();
                if (charLower in phonemeIdMap) {
                    const tokenIds = Array.isArray(phonemeIdMap[charLower]) ? phonemeIdMap[charLower] : [phonemeIdMap[charLower]];
                    phonemeIds.push(...tokenIds.map(x => parseInt(x)));
                } else if (charUpper in phonemeIdMap) {
                    const tokenIds = Array.isArray(phonemeIdMap[charUpper]) ? phonemeIdMap[charUpper] : [phonemeIdMap[charUpper]];
                    phonemeIds.push(...tokenIds.map(x => parseInt(x)));
                }
                // Skip unmapped characters
            }
        }
        
        // Add end token
        if (phonemeIdMap['$']) {
            const tokenIds = Array.isArray(phonemeIdMap['$']) ? phonemeIdMap['$'] : [phonemeIdMap['$']];
            phonemeIds.push(...tokenIds.map(x => parseInt(x)));
        }
        
        if (phonemeIds.length === 0) {
            throw new Error('Failed to convert phonemes to IDs - no phoneme IDs generated');
        }
        
        // Prepare ONNX inputs
        const sequenceLength = phonemeIds.length;
        const finalLengthScale = Math.max(0.3, Math.min(10.0, lengthScale));
        const finalNoiseScale = noiseScale;
        const finalNoiseW = noiseW;
        
        // Get model input names - onnxruntime-node API
        let inputNames = [];
        try {
            // Try session.inputNames (array of strings)
            if (session.inputNames && Array.isArray(session.inputNames)) {
                inputNames = session.inputNames;
            } 
            // Try session.inputMetadata (object with input names as keys)
            else if (session.inputMetadata && typeof session.inputMetadata === 'object') {
                inputNames = Object.keys(session.inputMetadata);
            }
            // Try getInputs() method (returns array of input metadata objects)
            else if (typeof session.getInputs === 'function') {
                const inputList = session.getInputs();
                inputNames = inputList.map(inp => inp.name || inp);
            }
        } catch (e) {
            console.log(`[TTS DEBUG] Error getting input names: ${e.message}`);
        }
        
        console.log(`[TTS DEBUG] Model input names: ${JSON.stringify(inputNames)}`);
        
        const inputs = {};
        
        // Create phoneme array as Int64Array (BigInt64Array for ONNX)
        const phonemeArray = new BigInt64Array(phonemeIds.map(x => BigInt(x)));
        const phonemeTensor = new ort.Tensor('int64', phonemeArray, [1, sequenceLength]);
        
        // Assign inputs based on names or position
        if (inputNames.length > 0) {
            for (const inputName of inputNames) {
                const inputNameLower = inputName.toLowerCase();
                if (inputNameLower === 'input' || inputNameLower.includes('phoneme') || inputNameLower.includes('sequence') || inputNameLower.includes('token')) {
                    inputs[inputName] = phonemeTensor;
                } else if (inputNameLower === 'scales') {
                    const scalesArray = new Float32Array([finalLengthScale, finalNoiseScale, finalNoiseW]);
                    inputs[inputName] = new ort.Tensor('float32', scalesArray, [3]);
                } else if (inputNameLower.includes('length_scale')) {
                    const lengthScaleArray = new Float32Array([finalLengthScale]);
                    inputs[inputName] = new ort.Tensor('float32', lengthScaleArray, [1]);
                } else if (inputNameLower.includes('noise_scale') && !inputNameLower.includes('w')) {
                    const noiseScaleArray = new Float32Array([finalNoiseScale]);
                    inputs[inputName] = new ort.Tensor('float32', noiseScaleArray, [1]);
                } else if (inputNameLower.includes('noise_w') || (inputNameLower.includes('noise') && inputNameLower.includes('w'))) {
                    const noiseWArray = new Float32Array([finalNoiseW]);
                    inputs[inputName] = new ort.Tensor('float32', noiseWArray, [1]);
                } else if (inputNameLower.includes('length') && !inputNameLower.includes('scale')) {
                    const lengthArray = new BigInt64Array([BigInt(sequenceLength)]);
                    inputs[inputName] = new ort.Tensor('int64', lengthArray, [1]);
                }
            }
        }
        
        // Ensure 'input' is assigned if it exists in inputNames but wasn't matched
        if (inputNames.includes('input') && !inputs.hasOwnProperty('input')) {
            inputs['input'] = phonemeTensor;
        }
        
        // Fallback: positional assignment (common Piper TTS model inputs)
        // Most Piper models expect: [phonemes, length_scale, noise_scale, noise_w]
        if (Object.keys(inputs).length === 0) {
            inputs[inputNames[0] || 'input'] = phonemeTensor;
            if (inputNames[1]) {
                const lengthScaleArray = new Float32Array([finalLengthScale]);
                inputs[inputNames[1]] = new ort.Tensor('float32', lengthScaleArray, [1]);
            }
            if (inputNames[2]) {
                const noiseScaleArray = new Float32Array([finalNoiseScale]);
                inputs[inputNames[2]] = new ort.Tensor('float32', noiseScaleArray, [1]);
            }
            if (inputNames[3]) {
                const noiseWArray = new Float32Array([finalNoiseW]);
                inputs[inputNames[3]] = new ort.Tensor('float32', noiseWArray, [1]);
            }
        }
        
        // Run ONNX inference
        const outputs = await session.run(inputs);
        
        if (!outputs) {
            throw new Error('ONNX model produced no output');
        }
        
        console.log(`[TTS DEBUG] Outputs type: ${outputs.constructor.name}, is Array: ${Array.isArray(outputs)}, is Map: ${outputs instanceof Map}`);
        
        // Get audio output - onnxruntime-node returns Map or array
        let audioTensor = null;
        if (outputs instanceof Map) {
            // Get first value from Map
            const outputValues = Array.from(outputs.values());
            audioTensor = outputValues[0];
            console.log(`[TTS DEBUG] Output Map keys: ${Array.from(outputs.keys()).join(', ')}`);
        } else if (Array.isArray(outputs)) {
            audioTensor = outputs[0];
        } else if (typeof outputs === 'object') {
            // Try to get first property value
            const outputKeys = Object.keys(outputs);
            if (outputKeys.length > 0) {
                audioTensor = outputs[outputKeys[0]];
            }
        }
        
        if (!audioTensor) {
            throw new Error('ONNX model produced no output - could not extract audio tensor');
        }
        
        console.log(`[TTS DEBUG] Audio tensor type: ${audioTensor.constructor.name}, has data: ${!!audioTensor.data}, has dims: ${!!audioTensor.dims}`);
        
        let audio;
        
        // Extract data from tensor - ONNX Runtime returns Tensor objects
        if (audioTensor && audioTensor.data !== undefined) {
            // Tensor has .data property
            if (audioTensor.data instanceof Float32Array) {
                audio = audioTensor.data;
            } else if (audioTensor.data instanceof Float64Array) {
                audio = new Float32Array(audioTensor.data);
            } else if (Array.isArray(audioTensor.data)) {
                audio = new Float32Array(audioTensor.data);
            } else if (audioTensor.data instanceof ArrayBuffer) {
                audio = new Float32Array(audioTensor.data);
            } else if (Buffer.isBuffer(audioTensor.data)) {
                audio = new Float32Array(audioTensor.data.buffer, audioTensor.data.byteOffset, audioTensor.data.length / 4);
            } else {
                // Try to extract from buffer
                try {
                    audio = new Float32Array(Buffer.from(audioTensor.data));
                } catch (e) {
                    throw new Error(`Unexpected audio output data format: ${typeof audioTensor.data}, ${e.message}`);
                }
            }
        } else if (audioTensor instanceof Float32Array) {
            audio = audioTensor;
        } else if (Array.isArray(audioTensor)) {
            audio = new Float32Array(audioTensor);
        } else {
            throw new Error(`Unexpected audio output format from ONNX model: ${typeof audioTensor}, constructor: ${audioTensor.constructor.name}`);
        }
        
        // Flatten if needed (handle multi-dimensional arrays)
        // Piper TTS models typically output audio as [1, samples] or [samples]
        if (audioTensor.dims && audioTensor.dims.length > 1) {
            const totalLength = audioTensor.dims.reduce((a, b) => a * b, 1);
            if (audio.length !== totalLength) {
                // Reshape needed - take only what we need
                const flattened = new Float32Array(totalLength);
                const copyLength = Math.min(audio.length, totalLength);
                flattened.set(audio.subarray(0, copyLength));
                audio = flattened;
            }
        }
        
        // Clip audio to [-1.0, 1.0] range (Piper TTS outputs are already in this range)
        // Don't normalize by max - just clip to prevent distortion
        for (let i = 0; i < audio.length; i++) {
            audio[i] = Math.max(-1.0, Math.min(1.0, audio[i]));
        }
        
        // Convert to Int16 PCM (16-bit signed integer)
        // Range: -32768 to 32767 for 16-bit audio
        const audioInt16 = new Int16Array(audio.length);
        for (let i = 0; i < audio.length; i++) {
            // Clamp to [-1, 1] and convert to int16
            const clamped = Math.max(-1.0, Math.min(1.0, audio[i]));
            audioInt16[i] = Math.round(clamped * 32767);
        }
        
        // Get sample rate from config (Piper TTS models typically use 22050 Hz)
        const audioConfig = config.audio || {};
        let sampleRate = audioConfig.sample_rate || 22050;
        
        // Validate sample rate
        if (![16000, 22050, 44100, 48000].includes(sampleRate)) {
            console.log(`[TTS] Warning: Invalid sample rate ${sampleRate}, using 22050`);
            sampleRate = 22050;
        }
        
        // Create WAV file using WaveFile library
        // Parameters: channels (1 = mono), sampleRate, bitDepth ('16' = 16-bit), audioData
        const wav = new WaveFile();
        wav.fromScratch(1, sampleRate, '16', audioInt16);
        const wavBuffer = Buffer.from(wav.toBuffer());
        
        // Cache the audio
        if (ttsCache.size < 100) {
            ttsCache.set(cacheKey, wavBuffer);
        } else {
            const firstKey = ttsCache.keys().next().value;
            ttsCache.delete(firstKey);
            ttsCache.set(cacheKey, wavBuffer);
        }
        
        return wavBuffer;
    } catch (error) {
        console.error(`[TTS] Synthesis error: ${error.message}`);
        throw error;
    }
}

// API Routes
app.get('/health', (req, res) => {
    res.json({
        status: 'ok',
        models_loaded: loadedModels.size,
        voices_loaded: loadedVoices.size,
        available_models: availableModels.size,
        available_voices: Array.from(availableVoices.values()).reduce((sum, v) => sum + v.length, 0)
    });
});

app.get('/api/models', (req, res) => {
    // Only return models that can actually be loaded
    const models = Array.from(availableModels.values())
        .filter(m => m.complete)  // Only show complete models
        .map(m => ({
            name: m.name,
            path: m.path,
            lang_code: m.lang_code,
            flag_path: m.flag_path,
            complete: m.complete,
            is_current: loadedModels.has(m.name)
        }));
    
    res.json({
        models,
        current_model_path: null
    });
});

app.post('/api/select-model', async (req, res) => {
    try {
        const { model_name } = req.body;
        
        if (!model_name) {
            return res.status(400).json({ error: 'model_name is required' });
        }
        
        // Sanitize model name
        const sanitizedModelName = sanitizeModelName(model_name);
        
        await loadTranslationModel(sanitizedModelName);
        res.json({ status: 'ok', model_name: sanitizedModelName });
    } catch (error) {
        if (error.message.includes('Invalid') || error.message.includes('must be')) {
            res.status(400).json({ error: error.message });
        } else {
            res.status(500).json({ error: error.message });
        }
    }
});

app.post('/translate', async (req, res) => {
    try {
        const { text, model_path } = req.body;
        
        if (!text) {
            return res.status(400).json({ error: 'text is required' });
        }
        
        // Sanitize input
        const sanitizedText = sanitizeString(text, 10000);
        if (!sanitizedText) {
            return res.status(400).json({ error: 'Text cannot be empty' });
        }
        
        // Find model to use
        let modelName = null;
        if (model_path) {
            const sanitizedPath = sanitizeModelName(path.basename(model_path));
            if (availableModels.has(sanitizedPath)) {
                modelName = sanitizedPath;
            }
        }
        
        if (!modelName) {
            // Use first available model
            const models = Array.from(availableModels.values()).filter(m => m.complete);
            if (models.length === 0) {
                return res.status(400).json({ error: 'No models available' });
            }
            modelName = models[0].name;
        }
        
        const translated = await translateText(sanitizedText, modelName);
        
        res.json({
            translated_text: translated,
            model_name: modelName
        });
    } catch (error) {
        console.error(`[TRANSLATE ERROR] ${error.message}`);
        console.error(`[TRANSLATE ERROR] Stack:`, error.stack);
        if (error.message.includes('Invalid') || error.message.includes('must be')) {
            res.status(400).json({ error: error.message });
        } else {
            res.status(500).json({ error: error.message });
        }
    }
});

app.get('/api/voices/:langCode', async (req, res) => {
    try {
        const { langCode } = req.params;
        
        // Sanitize language code
        const sanitizedLangCode = sanitizeLangCode(langCode);
        
        await scanVoices(sanitizedLangCode);
        const voices = availableVoices.get(sanitizedLangCode) || [];
        res.json({ voices });
    } catch (error) {
        if (error.message.includes('Invalid') || error.message.includes('must be')) {
            res.status(400).json({ error: error.message });
        } else {
            res.status(500).json({ error: error.message });
        }
    }
});

// Also support the format used by frontend
app.get('/api/voices', async (req, res) => {
    const { lang_code } = req.query;
    
    if (lang_code) {
        try {
            await scanVoices(lang_code);
            const voices = availableVoices.get(lang_code) || [];
            res.json({ voices });
        } catch (error) {
            res.status(500).json({ error: error.message });
        }
    } else {
        res.json({ voices: {} });
    }
});

app.post('/api/synthesize', async (req, res) => {
    try {
        const { text, voice_key, lang_code, length_scale, noise_scale, noise_w } = req.body;
        
        if (!text || !voice_key || !lang_code) {
            return res.status(400).json({ error: 'text, voice_key, and lang_code are required' });
        }
        
        // Sanitize all inputs
        const sanitizedText = sanitizeString(text, 5000);
        if (!sanitizedText) {
            return res.status(400).json({ error: 'Text cannot be empty' });
        }
        
        const sanitizedVoiceKey = sanitizeVoiceKey(voice_key);
        const sanitizedLangCode = sanitizeLangCode(lang_code);
        const sanitizedLengthScale = sanitizeFloat(length_scale, 0.1, 5.0, 1.0);
        const sanitizedNoiseScale = sanitizeFloat(noise_scale, 0.0, 2.0, 0.667);
        const sanitizedNoiseW = sanitizeFloat(noise_w, 0.0, 2.0, 0.8);
        
        console.log(`[TTS] Synthesizing speech: text="${sanitizedText.substring(0, 50)}...", voice="${sanitizedVoiceKey}", lang="${sanitizedLangCode}"`);
        
        const audioBuffer = await synthesizeSpeech(
            sanitizedText, 
            sanitizedVoiceKey, 
            sanitizedLangCode, 
            sanitizedLengthScale,
            sanitizedNoiseScale,
            sanitizedNoiseW
        );
        
        res.setHeader('Content-Type', 'audio/wav');
        res.send(audioBuffer);
    } catch (error) {
        console.error(`[TTS] Synthesis error: ${error.message}`);
        if (error.message.includes('Invalid') || error.message.includes('must be')) {
            res.status(400).json({ error: error.message });
        } else {
            res.status(500).json({ error: error.message || 'Speech synthesis failed' });
        }
    }
});

// Serve config.json
app.get('/api/config', async (req, res) => {
    try {
        const configPath = path.join(__dirname, 'config.json');
        const config = await fs.readJson(configPath);
        res.json(config);
    } catch (error) {
        // Silent error - config is optional
        res.status(500).json({ error: 'Failed to load configuration' });
    }
});

// Serve main page
app.get('/', async (req, res) => {
    try {
        const htmlPath = path.join(PUBLIC_DIR, 'templates', 'index.html');
        if (await fs.pathExists(htmlPath)) {
            const html = await fs.readFile(htmlPath, 'utf-8');
            res.send(html);
        } else {
            res.status(404).send('Frontend not found');
        }
    } catch (error) {
        res.status(500).send(`Error: ${error.message}`);
    }
});

// Start server
async function start() {
    try {
        await initialize();
    } catch (error) {
        console.error(`Initialization error: ${error.message}`);
        console.log('Starting server anyway with available resources...');
    }
    
    const server = app.listen(PORT, () => {
        const usableModels = Array.from(availableModels.values()).filter(m => m.complete).length;
        const totalVoices = Array.from(availableVoices.values()).reduce((sum, v) => sum + v.length, 0);
        console.log('');
        console.log('Server initialized successfully');
        console.log(`Server running on http://localhost:${PORT}`);
        console.log(`Summary: ${usableModels} usable model(s), ${totalVoices} voice(s) available`);
        console.log('');
        
        // Heartbeat log every 2 minutes
        setInterval(() => {
            const timestamp = new Date().toISOString();
            console.log(`[${timestamp}] Server heartbeat - still running`);
        }, 2 * 60 * 1000);
    });
    
    server.on('error', (error) => {
        if (error.code === 'EADDRINUSE') {
            console.error(`Port ${PORT} is already in use. Please stop the other process or use a different port.`);
            process.exit(1);
        } else {
            console.error(`Server error: ${error.message}`);
            throw error;
        }
    });
}

start().catch((error) => {
    console.error(`Fatal error starting server: ${error.message}`);
    process.exit(1);
});
