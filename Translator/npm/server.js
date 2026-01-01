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

// Initialize on startup
async function initialize() {
    await scanModels();
    await scanVoices();
}

// Scan models directory
async function scanModels() {
    availableModels.clear();
    
    if (!await fs.pathExists(MODELS_DIR)) {
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
    
    // Silent scan - no logging
}

// Scan voices directory
async function scanVoices(langCode = null) {
    if (!langCode) {
        availableVoices.clear();
    }
    
    if (!await fs.pathExists(VOICES_DIR)) {
        return;
    }
    
    const langDirs = langCode 
        ? [path.join(VOICES_DIR, langCode)]
        : (await fs.readdir(VOICES_DIR, { withFileTypes: true }))
            .filter(e => e.isDirectory())
            .map(e => path.join(VOICES_DIR, e.name));
    
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
                            
                            // Skip Git LFS pointers
                            try {
                                const stats = await fs.stat(onnxPath);
                                if (stats.size < 1024) {
                                    const firstBytes = await fs.readFile(onnxPath);
                                    if (firstBytes.toString().includes('version https://git-lfs.github.com/spec/v1')) {
                                        continue;
                                    }
                                }
                            } catch (e) {
                                // Continue
                            }
                            
                            if (await fs.pathExists(jsonPath)) {
                                try {
                                    const config = await fs.readJson(jsonPath);
                                    const voiceKey = path.basename(onnxFile, '.onnx');
                                    
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
    
    if (!langCode) {
    // Silent scan - no logging
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
        console.log(`Server running on http://localhost:${PORT}`);
        console.log(`Models: ${usableModels} usable, Voices: ${totalVoices} available`);
        
        // Heartbeat log every 2 minutes
        setInterval(() => {
            const timestamp = new Date().toISOString();
            console.log(`[${timestamp}] Server heartbeat - still running`);
        }, 2 * 60 * 1000);
    });
}

start().catch(console.error);
