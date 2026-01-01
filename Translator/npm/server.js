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
import ort from 'onnxruntime-node';
import WaveFile from 'wavefile';
import { fileURLToPath } from 'url';
import { spawn } from 'child_process';
import crypto from 'crypto';

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
    await scanModels();
    console.log(`Scanning voices directory: ${VOICES_DIR}`);
    await scanVoices();
}

// Scan models directory
async function scanModels() {
    availableModels.clear();
    
    if (!await fs.pathExists(MODELS_DIR)) {
        console.log('Models directory not found');
        return;
    }
    
    const entries = await fs.readdir(MODELS_DIR, { withFileTypes: true });
    
    for (const entry of entries) {
        if (entry.isDirectory() && entry.name.startsWith('en-')) {
            const modelPath = path.join(MODELS_DIR, entry.name);
            const configPath = path.join(modelPath, 'config.json');
            const tokenizerJsonPath = path.join(modelPath, 'tokenizer.json');
            const hasConfig = await fs.pathExists(configPath);
            const hasTokenizerJson = await fs.pathExists(tokenizerJsonPath);
            
            const hasWeights = (
                await fs.pathExists(path.join(modelPath, 'model.safetensors')) ||
                await fs.pathExists(path.join(modelPath, 'pytorch_model.bin')) ||
                await fs.pathExists(path.join(modelPath, 'model.bin'))
            );
            
            // Mark as complete if it has config and weights (can use Python for inference)
            // We'll use Python subprocess for models without tokenizer.json
            const complete = hasConfig && hasWeights;
            const langCode = entry.name.split('-')[1] || '';
            const flagPath = `/flags/${langCode}.svg`;
            
            availableModels.set(entry.name, {
                name: entry.name,
                path: modelPath,
                lang_code: langCode,
                flag_path: flagPath,
                complete: complete,
                hasTokenizerJson: hasTokenizerJson,
                usePython: !hasTokenizerJson  // Use Python for SentencePiece models
            });
        }
    }
    
    // Log scan results and attempt downloads for incomplete models
    const completeModels = Array.from(availableModels.values()).filter(m => m.complete);
    const incompleteModels = Array.from(availableModels.values()).filter(m => !m.complete);
    
    if (completeModels.length > 0) {
        console.log(`Found ${completeModels.length} complete model(s):`);
        completeModels.forEach(m => {
            const method = m.usePython ? 'Python' : 'JavaScript';
            const hasTokenizer = m.hasTokenizerJson ? ' (has tokenizer.json)' : ' (SentencePiece)';
            console.log(`  [COMPLETE] ${m.name} (${m.lang_code}) [${method}]${hasTokenizer}`);
            console.log(`    Path: ${m.path}`);
        });
    }
    
    if (incompleteModels.length > 0) {
        console.log(`Found ${incompleteModels.length} incomplete model(s):`);
        for (const m of incompleteModels) {
            const missing = [];
            if (!await fs.pathExists(path.join(m.path, 'config.json'))) missing.push('config.json');
            const hasWeights = (
                await fs.pathExists(path.join(m.path, 'model.safetensors')) ||
                await fs.pathExists(path.join(m.path, 'pytorch_model.bin')) ||
                await fs.pathExists(path.join(m.path, 'model.bin'))
            );
            if (!hasWeights) missing.push('model weights');
            
            console.log(`  [INCOMPLETE] ${m.name} (${m.lang_code})`);
            console.log(`    Path: ${m.path}`);
            console.log(`    Missing: ${missing.join(', ')}`);
            
            // Attempt to download from HuggingFace
            console.log(`    Attempting to download from HuggingFace...`);
            try {
                const progressCallback = (message) => {
                    console.log(`    ${message}`);
                };
                
                const modelId = await downloadModelFromHuggingFace(m.name, m.path, progressCallback);
                
                if (modelId === 'already_exists') {
                    console.log(`    Model already exists locally`);
                } else {
                    console.log(`    Successfully downloaded from HuggingFace: ${modelId}`);
                }
                
                // Re-check completeness
                const newHasConfig = await fs.pathExists(path.join(m.path, 'config.json'));
                const newHasWeights = (
                    await fs.pathExists(path.join(m.path, 'model.safetensors')) ||
                    await fs.pathExists(path.join(m.path, 'pytorch_model.bin')) ||
                    await fs.pathExists(path.join(m.path, 'model.bin'))
                );
                const newHasTokenizerJson = await fs.pathExists(path.join(m.path, 'tokenizer.json'));
                
                if (newHasConfig && newHasWeights) {
                    m.complete = true;
                    m.hasTokenizerJson = newHasTokenizerJson;
                    m.usePython = !newHasTokenizerJson;
                    console.log(`    Model ${m.name} is now complete`);
                } else {
                    console.log(`    Model ${m.name} still incomplete after download`);
                }
            } catch (error) {
                console.log(`    Failed to download: ${error.message}`);
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
                const locale = path.basename(localeDir);
                const voiceDirs = (await fs.readdir(localeDir, { withFileTypes: true }))
                    .filter(e => e.isDirectory())
                    .map(e => path.join(localeDir, e.name));
                
                for (const voiceDir of voiceDirs) {
                    const voiceName = path.basename(voiceDir);
                    const qualityDirs = (await fs.readdir(voiceDir, { withFileTypes: true }))
                        .filter(e => e.isDirectory())
                        .map(e => path.join(voiceDir, e.name));
                    
                    for (const qualityDir of qualityDirs) {
                        const quality = path.basename(qualityDir);
                        const onnxFiles = (await fs.readdir(qualityDir))
                            .filter(f => f.endsWith('.onnx'));
                        
                        for (const onnxFile of onnxFiles) {
                            const onnxPath = path.join(qualityDir, onnxFile);
                            const jsonPath = onnxPath + '.json';
                            const voiceKey = path.basename(onnxFile, '.onnx');
                            
                            // Check if files are complete
                            const hasOnnx = await fs.pathExists(onnxPath);
                            const hasJson = await fs.pathExists(jsonPath);
                            
                            // Skip Git LFS pointers
                            let isLfsPointer = false;
                            if (hasOnnx) {
                                try {
                                    const stats = await fs.stat(onnxPath);
                                    if (stats.size < 1024) {
                                        const firstBytes = await fs.readFile(onnxPath);
                                        if (firstBytes.toString().includes('version https://git-lfs.github.com/spec/v1')) {
                                            isLfsPointer = true;
                                        }
                                    }
                                } catch (e) {
                                    // Continue
                                }
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
                                    // Silent error - skip invalid configs
                                }
                            }
                        }
                    }
                }
            }
        } catch (e) {
            // Silent error - skip languages with errors
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
    
    // If model uses Python (SentencePiece), just mark as ready
    if (modelInfo.usePython) {
        loadedModels.set(modelName, {
            translator: null,  // Will use Python subprocess
            modelInfo,
            usePython: true
        });
        return loadedModels.get(modelName);
    }
    
    // Load with @xenova/transformers for models with tokenizer.json
    try {
        const modelPath = path.resolve(modelInfo.path);
        
        const translator = await pipeline('translation', modelPath, {
            device: 'cpu',
            local_files_only: true
        });
        
        loadedModels.set(modelName, {
            translator,
            modelInfo,
            usePython: false
        });
        
        return loadedModels.get(modelName);
    } catch (error) {
        throw new Error(`Failed to load model: ${error.message}`);
    }
}

// Translate text using Python subprocess for SentencePiece models
async function translateWithPython(text, modelPath) {
    return new Promise((resolve, reject) => {
        // Escape paths and text for Python
        const escapedModelPath = modelPath.replace(/\\/g, '\\\\').replace(/"/g, '\\"');
        const escapedText = JSON.stringify(text);
        
        const scriptContent = [
            'import sys',
            'import json',
            'import os',
            'from pathlib import Path',
            '',
            '# Add frontend to path',
            'frontend_path = Path(__file__).parent if "__file__" in globals() else Path.cwd()',
            'if not frontend_path.name == "frontend":',
            '    frontend_path = frontend_path / "frontend"',
            'sys.path.insert(0, str(frontend_path))',
            '',
            'from transformers import MarianMTModel, MarianTokenizer',
            'import torch',
            '',
            `model_path = r"${escapedModelPath}"`,
            `text = ${escapedText}`,
            '',
            'try:',
            '    tokenizer = MarianTokenizer.from_pretrained(model_path, local_files_only=True)',
            '    model = MarianMTModel.from_pretrained(model_path, local_files_only=True)',
            '    model.eval()',
            '    model = model.to("cpu")',
            '    ',
            '    with torch.no_grad():',
            '        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)',
            '        translated = model.generate(**inputs, max_length=512, num_beams=4, early_stopping=True)',
            '        result = tokenizer.decode(translated[0], skip_special_tokens=True)',
            '    ',
            '    print(json.dumps({"translated_text": result}))',
            'except Exception as e:',
            '    print(json.dumps({"error": str(e)}), file=sys.stderr)',
            '    sys.exit(1)'
        ].join('\n');
        
        // Try to use venv Python first, fallback to system Python
        const venvPython = path.join(PROJECT_ROOT, '.venv', 'bin', 'python3');
        const pythonPath = fs.existsSync(venvPython) ? venvPython : 'python3';
        const frontendDir = path.join(PROJECT_ROOT, 'frontend');
        const pythonProcess = spawn(pythonPath, ['-c', scriptContent], {
            cwd: frontendDir,
            env: { ...process.env, PYTHONPATH: frontendDir }
        });
        let stdout = '';
        let stderr = '';
        
        pythonProcess.stdout.on('data', (data) => {
            stdout += data.toString();
        });
        
        pythonProcess.stderr.on('data', (data) => {
            stderr += data.toString();
        });
        
        pythonProcess.on('close', (code) => {
            if (code !== 0) {
                reject(new Error(`Python translation failed: ${stderr || 'Unknown error'}`));
                return;
            }
            
            try {
                const output = stdout.trim();
                if (!output) {
                    reject(new Error('Python returned empty output'));
                    return;
                }
                
                const result = JSON.parse(output);
                if (result.error) {
                    reject(new Error(result.error));
                } else {
                    resolve(result.translated_text);
                }
            } catch (e) {
                reject(new Error(`Failed to parse Python output: ${e.message}. Output: ${stdout}`));
            }
        });
        
        pythonProcess.on('error', (error) => {
            reject(new Error(`Failed to start Python process: ${error.message}. Make sure Python 3 and transformers are installed.`));
        });
    });
}

// Translate text
async function translateText(text, modelName) {
    const model = await loadTranslationModel(modelName);
    
    let translated;
    // Use Python for SentencePiece models
    if (model.usePython) {
        translated = await translateWithPython(text, model.modelInfo.path);
    } else {
        // Use @xenova/transformers for models with tokenizer.json
        const result = await model.translator(text, {
            max_length: 512,
            num_beams: 4,
            early_stopping: true
        });
        translated = result[0].translation_text;
    }
    
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
    
        try {
            const session = await ort.InferenceSession.create(voice.onnx_path, {
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
        const pythonProcess = spawn('which', [espeakBin]);
        let whichOutput = '';
        
        pythonProcess.stdout.on('data', (data) => {
            whichOutput += data.toString();
        });
        
        pythonProcess.on('close', (code) => {
            const espeakPath = whichOutput.trim() || 'espeak'; // Fallback to espeak
            
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

// Synthesize speech using Python subprocess (uses existing files, same as translation)
async function synthesizeSpeech(text, voiceKey, langCode, lengthScale = 1.0, noiseScale = 0.667, noiseW = 0.8) {
    // Check cache first
    const cacheKey = generateTTSCacheKey(text, voiceKey, langCode, lengthScale, noiseScale, noiseW);
    if (ttsCache.has(cacheKey)) {
        return ttsCache.get(cacheKey);
    }
    return new Promise((resolve, reject) => {
        const escapedText = JSON.stringify(text);
        
        const scriptContent = [
            'import sys',
            'import json',
            'import base64',
            '',
            'from app import synthesize_speech',
            '',
            `text = ${escapedText}`,
            `voice_key = "${voiceKey}"`,
            `lang_code = "${langCode}"`,
            `length_scale = ${lengthScale}`,
            `noise_scale = ${noiseScale}`,
            `noise_w = ${noiseW}`,
            '',
            'try:',
            '    audio_bytes = synthesize_speech(',
            '        text,',
            '        voice_key,',
            '        lang_code,',
            '        speed_multiplier=1.0,',
            '        length_scale=length_scale,',
            '        noise_scale=noise_scale,',
            '        noise_w=noise_w',
            '    )',
            '    ',
            '    # Output as base64 for safe transfer',
            '    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")',
            '    print(json.dumps({"audio": audio_b64}))',
            'except Exception as e:',
            '    import traceback',
            '    print(json.dumps({"error": str(e), "traceback": traceback.format_exc()}), file=sys.stderr)',
            '    sys.exit(1)'
        ].join('\n');
        
        // Try to use venv Python first, fallback to system Python
        const venvPython = path.join(PROJECT_ROOT, '.venv', 'bin', 'python3');
        const pythonPath = fs.existsSync(venvPython) ? venvPython : 'python3';
        const frontendDir = path.join(PROJECT_ROOT, 'frontend');
        const pythonProcess = spawn(pythonPath, ['-c', scriptContent], {
            cwd: frontendDir,
            env: { ...process.env, PYTHONPATH: frontendDir }
        });
        let stdout = '';
        let stderr = '';
        
        pythonProcess.stdout.on('data', (data) => {
            stdout += data.toString();
        });
        
        pythonProcess.stderr.on('data', (data) => {
            stderr += data.toString();
        });
        
        pythonProcess.on('close', (code) => {
            if (code !== 0) {
                reject(new Error(`Python TTS failed: ${stderr || 'Unknown error'}`));
                return;
            }
            
            try {
                // Python prints debug messages to stdout, then JSON on the last line
                // Extract only the last line which should be the JSON
                const lines = stdout.trim().split('\n');
                const lastLine = lines[lines.length - 1];
                
                if (!lastLine) {
                    reject(new Error('Python returned empty output'));
                    return;
                }
                
                const result = JSON.parse(lastLine);
                if (result.error) {
                    reject(new Error(result.error));
                } else {
                    // Decode base64 audio
                    const audioBuffer = Buffer.from(result.audio, 'base64');
                    
                    // Store in cache (limit cache size to prevent memory issues)
                    if (ttsCache.size < 100) {
                        ttsCache.set(cacheKey, audioBuffer);
                    } else {
                        // Remove oldest entry (simple FIFO)
                        const firstKey = ttsCache.keys().next().value;
                        ttsCache.delete(firstKey);
                        ttsCache.set(cacheKey, audioBuffer);
                    }
                    
                    resolve(audioBuffer);
                }
            } catch (e) {
                reject(new Error(`Failed to parse Python output: ${e.message}`));
            }
        });
        
        pythonProcess.on('error', (error) => {
            reject(new Error(`Failed to start Python process: ${error.message}. Make sure Python 3 and required packages are installed.`));
        });
    });
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
    await initialize();
    
    app.listen(PORT, () => {
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
}

start().catch(console.error);
