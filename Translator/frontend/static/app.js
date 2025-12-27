let availableModels = [];
let currentModelPath = null;
let selectedModel = null;
let currentView = 'main'; // 'main' or 'chat'
let currentChatLanguage = null;
let availableVoices = [];
let selectedVoice = null;
let lastTranslatedText = null; // Store the last translated text for consistent speaking
let historyVoices = {}; // Store voice selections per language code
let historyLengthScales = {}; // Store length scale per language code

// No emoji mapping needed - we use SVG flags

function getLanguageName(langCode) {
    const langNames = {
        'af': 'Afrikaans', 'ar': 'Arabic', 'bg': 'Bulgarian', 'bn': 'Bengali', 'ca': 'Catalan',
        'cs': 'Czech', 'da': 'Danish', 'de': 'German', 'el': 'Greek', 'es': 'Spanish',
        'et': 'Estonian', 'fa': 'Persian', 'fi': 'Finnish', 'fr': 'French', 'he': 'Hebrew',
        'hi': 'Hindi', 'hr': 'Croatian', 'hu': 'Hungarian', 'id': 'Indonesian', 'it': 'Italian',
        'ja': 'Japanese', 'ko': 'Korean', 'lt': 'Lithuanian', 'lv': 'Latvian', 'mk': 'Macedonian',
        'ms': 'Malay', 'mt': 'Maltese', 'nl': 'Dutch', 'no': 'Norwegian', 'pl': 'Polish',
        'pt': 'Portuguese', 'ro': 'Romanian', 'ru': 'Russian', 'sk': 'Slovak', 'sl': 'Slovenian',
        'sq': 'Albanian', 'sr': 'Serbian', 'sv': 'Swedish', 'sw': 'Swahili', 'ta': 'Tamil',
        'th': 'Thai', 'tr': 'Turkish', 'uk': 'Ukrainian', 'ur': 'Urdu', 'vi': 'Vietnamese',
        'zh': 'Chinese'
    };
    return langNames[langCode.toLowerCase()] || langCode.toUpperCase();
}

// Load available models
async function loadModels() {
    try {
        const response = await fetch('/api/models');
        const data = await response.json();
        
        availableModels = data.models || [];
        currentModelPath = data.current_model_path;
        
        // Display flags
        displayFlags();
        
    } catch (error) {
        console.error('Failed to load models:', error);
    }
}

// Display flags for available models
function displayFlags() {
    const flagsContainer = document.getElementById('flagsContainer');
    const incompleteContainer = document.getElementById('incompleteFlagsContainer');
    const modelsSection = document.getElementById('modelsSection');
    const incompleteSection = document.getElementById('incompleteSection');
    
    // Separate complete and incomplete models
    const completeModels = availableModels.filter(m => m.complete && m.flag_path);
    const incompleteModels = availableModels.filter(m => !m.complete && m.flag_path);
    
    // Display complete models
    if (completeModels.length > 0) {
        modelsSection.style.display = 'block';
        flagsContainer.innerHTML = '';
        
        completeModels.forEach(model => {
            const flag = document.createElement('div');
            const langCode = model.lang_code || model.name.split('-')[1] || '';
            const langName = getLanguageName(langCode);
            
            flag.className = 'flag complete';
            if (model.is_current) {
                flag.classList.add('active');
            }
            
            flag.innerHTML = `
                <img src="${model.flag_path}" alt="${langCode.toUpperCase()}" />
                <div class="flag-label">${langCode.toUpperCase()}</div>
            `;
            
            flag.title = `Select ${langName} (${langCode.toUpperCase()})`;
            flag.onclick = () => selectModel(model);
            
            flagsContainer.appendChild(flag);
        });
    } else {
        modelsSection.style.display = 'none';
    }
    
    // Display incomplete models below translate button
    if (incompleteModels.length > 0) {
        incompleteSection.style.display = 'block';
        incompleteContainer.innerHTML = '';
        
        incompleteModels.forEach(model => {
            const flag = document.createElement('div');
            const langCode = model.lang_code || model.name.split('-')[1] || '';
            const langName = getLanguageName(langCode);
            
            flag.className = 'flag incomplete';
            
            flag.innerHTML = `
                <img src="${model.flag_path}" alt="${langCode.toUpperCase()}" />
                <div class="flag-label">${langCode.toUpperCase()}</div>
            `;
            
            flag.title = `${langName} (${langCode.toUpperCase()}) - Model incomplete (missing files)`;
            
            incompleteContainer.appendChild(flag);
        });
    } else {
        incompleteSection.style.display = 'none';
    }
}

// Select a model (load it but don't translate yet)
async function selectModel(model) {
    const statusDiv = document.getElementById('status');
    
    try {
        // Remove previous selection
        document.querySelectorAll('.flag.selected').forEach(flag => {
            flag.classList.remove('selected');
        });
        
        // Mark as selected immediately for visual feedback
        const flags = document.querySelectorAll('.flag');
        flags.forEach(flag => {
            const label = flag.querySelector('.flag-label');
            if (label && label.textContent === (model.lang_code || model.name.split('-')[1] || '').toUpperCase()) {
                flag.classList.add('selected');
            }
        });
        
        selectedModel = model;
        
        // Show loading status
        statusDiv.textContent = `Loading ${model.name}...`;
        statusDiv.style.display = 'block';
        statusDiv.style.background = '#fff3cd';
        statusDiv.style.color = '#856404';
        
        // Load the model (but don't translate yet)
        const response = await fetch('/api/select-model', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                model_name: model.name
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to load model');
        }
        
        const data = await response.json();
        console.log(`Model ${model.name} loaded:`, data);
        
        // Update status
        statusDiv.textContent = `${model.name} ready for translation`;
        statusDiv.style.background = '#e8f5e9';
        statusDiv.style.color = '#2e7d32';
        
        // Load voices for this language
        await loadVoicesForLanguage(model.lang_code);
        
    } catch (error) {
        console.error('Failed to select model:', error);
        statusDiv.textContent = `Failed to load: ${error.message}`;
        statusDiv.style.background = '#fee';
        statusDiv.style.color = '#c33';
        
        // Remove selection on error
        document.querySelectorAll('.flag.selected').forEach(flag => {
            flag.classList.remove('selected');
        });
        selectedModel = null;
    }
}

// Load voices for a language
async function loadVoicesForLanguage(langCode) {
    const voiceSelector = document.getElementById('voiceSelector');
    const voiceSelect = document.getElementById('voiceSelect');
    const speakBtn = document.getElementById('speakBtn');
    
    // Disable speak button while loading voices
    if (speakBtn) {
        speakBtn.disabled = true;
    }
    
    try {
        const response = await fetch(`/api/voices/${langCode}`);
        const data = await response.json();
        
        availableVoices = data.voices || [];
        
        // Clear existing options
        voiceSelect.innerHTML = '';
        
        if (availableVoices.length > 0) {
            // Show voice selector
            voiceSelector.style.display = 'block';
            
            // Add voices to dropdown
            availableVoices.forEach((voice, index) => {
                const option = document.createElement('option');
                option.value = voice.key;
                option.textContent = voice.display_name || voice.name || voice.key;
                voiceSelect.appendChild(option);
            });
            
            // Select first voice by default
            selectedVoice = availableVoices[0];
            voiceSelect.value = selectedVoice.key;
            
            // Add change handler
            voiceSelect.onchange = function() {
                const selectedKey = voiceSelect.value;
                selectedVoice = availableVoices.find(v => v.key === selectedKey);
                // Update speak button state based on whether we have translated text
                if (speakBtn) {
                    const translatedText = document.getElementById('translatedText');
                    const hasTranslatedText = translatedText && translatedText.textContent.trim().length > 0;
                    speakBtn.disabled = !(selectedVoice && hasTranslatedText);
                }
            };
        } else {
            // Hide voice selector if no voices available
            voiceSelector.style.display = 'none';
            selectedVoice = null;
        }
        
        // Update speak button state - keep disabled until translation is done
        if (speakBtn) {
            speakBtn.disabled = true;
        }
    } catch (error) {
        console.error('Failed to load voices:', error);
        voiceSelector.style.display = 'none';
        selectedVoice = null;
        if (speakBtn) {
            speakBtn.disabled = true;
        }
    }
}

// Speak translated text
async function speakText(text, voice = null, langCode = null, lengthScale = null, buttonElement = null) {
    // Use provided parameters or fall back to global selections
    const useVoice = voice || selectedVoice;
    const useLangCode = langCode || (selectedModel ? selectedModel.lang_code : null);
    const useLengthScale = lengthScale !== null ? lengthScale : parseFloat(document.getElementById('lengthScaleSlider').value);
    
    if (!useVoice || !useLangCode) {
        alert('Please select a language and voice first');
        return;
    }
    
    const speakBtn = buttonElement || document.getElementById('speakBtn');
    const originalText = speakBtn.textContent;
    
    try {
        speakBtn.disabled = true;
        speakBtn.textContent = '...';
        
        // Only use length scale parameter (removed speed, noise scale, noise w)
        const lengthScaleValue = useLengthScale; // 0.5 to 5.0
        
        const response = await fetch('/api/synthesize', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: text,
                voice_key: useVoice.key,
                lang_code: useLangCode,
                speed_multiplier: 1.0,
                length_scale: lengthScaleValue,
                noise_scale: 0.667,
                noise_w: 0.8
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Speech synthesis failed');
        }
        
        // Get audio data
        const audioBlob = await response.blob();
        console.log('Audio blob received:', audioBlob.size, 'bytes, type:', audioBlob.type);
        
        // Validate audio blob
        if (audioBlob.size === 0) {
            throw new Error('Received empty audio file');
        }
        
        // Store audio blob for history
        lastAudioBlob = audioBlob;
        
        // Also store in a more accessible way for history
        window.lastGeneratedAudio = audioBlob;
        
        // Play the audio
        const audioUrl = URL.createObjectURL(audioBlob);
        
        // Create audio element with proper configuration for full playback
        const audio = new Audio();
        audio.preload = 'auto'; // Preload entire audio file
        audio.src = audioUrl;
        
        // Keep reference to prevent garbage collection during playback
        window.currentAudio = audio;
        
        console.log('Loading audio (', audioBlob.size, 'bytes)...');
        
        // Wait for metadata (includes duration)
        await new Promise((resolve, reject) => {
            audio.addEventListener('loadedmetadata', () => {
                console.log('Metadata loaded - duration:', audio.duration, 'seconds');
                resolve();
            }, { once: true });
            
            audio.addEventListener('error', (e) => {
                console.error('Audio load error:', e, audio.error);
                reject(new Error('Failed to load audio: ' + (audio.error ? audio.error.message : 'Unknown error')));
            }, { once: true });
            
            setTimeout(() => {
                if (audio.readyState >= 1) {
                    resolve();
                } else {
                    reject(new Error('Audio loading timeout'));
                }
            }, 10000);
        });
        
        // Wait for audio to be buffered
        if (audio.readyState < 4) {
            await new Promise((resolve) => {
                audio.addEventListener('canplaythrough', resolve, { once: true });
                setTimeout(resolve, 2000);
            });
        }
        
        console.log('Audio ready - readyState:', audio.readyState, 'duration:', audio.duration, 'seconds');
        
        // Setup handlers BEFORE playing
        audio.onended = function() {
            console.log('Playback completed -', audio.duration, 'seconds played');
            URL.revokeObjectURL(audioUrl);
            window.currentAudio = null;
            speakBtn.disabled = false;
            speakBtn.textContent = originalText;
        };
        
        audio.onerror = function(e) {
            console.error('Playback error:', e, audio.error);
            URL.revokeObjectURL(audioUrl);
            window.currentAudio = null;
            speakBtn.disabled = false;
            speakBtn.textContent = originalText;
            alert('Failed to play audio: ' + (audio.error ? audio.error.message : 'Unknown error'));
        };
        
        // Log progress (throttled to every second)
        let lastLogTime = 0;
        audio.ontimeupdate = function() {
            const now = Date.now();
            if (now - lastLogTime > 1000 && audio.duration > 0) {
                const percent = (audio.currentTime / audio.duration * 100).toFixed(1);
                console.log('Playing:', audio.currentTime.toFixed(2) + 's /', audio.duration.toFixed(2) + 's (' + percent + '%)');
                lastLogTime = now;
            }
        };
        
        // Play audio
        try {
            await audio.play();
            console.log('Playback started - duration:', audio.duration, 'seconds');
        } catch (playError) {
            console.error('Playback error:', playError);
            URL.revokeObjectURL(audioUrl);
            window.currentAudio = null;
            throw new Error('Failed to play audio: ' + playError.message);
        }
        
    } catch (error) {
        console.error('Failed to speak text:', error);
        alert(`Failed to speak text: ${error.message}`);
        speakBtn.disabled = false;
        speakBtn.textContent = originalText;
    }
}

// Check health status on load
async function checkHealth() {
    try {
        const response = await fetch('/health');
        const data = await response.json();
        
        const statusDiv = document.getElementById('status');
        if (data.models_loaded > 0) {
            statusDiv.textContent = `${data.models_loaded} model(s) loaded. Current: ${data.current_model || 'none'}`;
            statusDiv.style.display = 'block';
            statusDiv.style.background = '#e8f5e9';
            statusDiv.style.color = '#2e7d32';
        } else {
            statusDiv.textContent = `No models loaded (${data.complete_models} complete found). Check server logs.`;
            statusDiv.style.display = 'block';
            statusDiv.style.background = '#fff3cd';
            statusDiv.style.color = '#856404';
        }
    } catch (error) {
        console.error('Health check failed:', error);
        const statusDiv = document.getElementById('status');
        statusDiv.textContent = 'Error checking service status';
        statusDiv.style.display = 'block';
        statusDiv.style.background = '#fee';
        statusDiv.style.color = '#c33';
    }
}

async function translate() {
    const inputText = document.getElementById('inputText').value.trim();
    const translateBtn = document.getElementById('translateBtn');
    const loading = document.getElementById('loading');
    const result = document.getElementById('result');
    const error = document.getElementById('error');
    const translatedText = document.getElementById('translatedText');
    const speakBtn = document.getElementById('speakBtn');

    // Clear previous results
    result.style.display = 'none';
    error.style.display = 'none';
    
    // Disable speak button during translation
    speakBtn.disabled = true;

    if (!inputText) {
        error.textContent = 'Please enter some text to translate.';
        error.style.display = 'block';
        return;
    }

    // Show loading state
    translateBtn.disabled = true;
    loading.classList.add('active');

    try {
        // Use selected model if available, otherwise let server choose first available
        const modelPath = selectedModel ? (selectedModel.path || selectedModel.name) : null;
        
        const response = await fetch('/translate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: inputText,
                model_path: modelPath
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || 'Translation failed');
        }

        // Show result with the actual translated text
        const translatedContent = data.translated_text || '';
        translatedText.textContent = translatedContent;
        result.style.display = 'block';
        
        // Store the translated text for consistent speaking
        lastTranslatedText = translatedContent.trim();
        
        // Show voice/TTS section
        const voiceTtsSection = document.getElementById('voiceTtsSection');
        if (voiceTtsSection) {
            voiceTtsSection.style.display = 'block';
        }
        
        // Enable speak button only after successful translation and if voice is available
        const speakBtn = document.getElementById('speakBtn');
        if (speakBtn) {
            speakBtn.disabled = !(selectedVoice && lastTranslatedText.length > 0);
        }

        // Automatically speak the translated text after successful translation
        if (selectedVoice && lastTranslatedText.length > 0) {
            // Small delay to ensure UI updates first
            setTimeout(async () => {
                await speakText(lastTranslatedText);
                
                // Save translation to history after audio is generated
                if (selectedModel) {
                    const langCode = selectedModel.lang_code || selectedModel.name.split('-')[1] || 'unknown';
                    await saveTranslationToHistory(langCode, inputText, translatedContent, selectedModel);
                }
            }, 300);
        } else {
            // Save without audio if no voice selected
            if (selectedModel) {
                const langCode = selectedModel.lang_code || selectedModel.name.split('-')[1] || 'unknown';
                await saveTranslationToHistory(langCode, inputText, translatedContent, selectedModel);
            }
        }

    } catch (err) {
        error.textContent = `Error: ${err.message}`;
        error.style.display = 'block';
        // Keep speak button disabled on error
        speakBtn.disabled = true;
    } finally {
        translateBtn.disabled = false;
        loading.classList.remove('active');
    }
}

// Allow Enter key to trigger translation (Ctrl+Enter or Cmd+Enter)
const inputText = document.getElementById('inputText');
if (inputText) {
    inputText.addEventListener('keydown', function(e) {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            translate();
        }
    });
}

// Attach translate button click handler
const translateBtn = document.getElementById('translateBtn');
if (translateBtn) {
    translateBtn.addEventListener('click', translate);
}

// Store last audio blob for download
let lastAudioBlob = null;

// Download audio function removed - no longer needed

// Attach speak button click handler
const speakBtn = document.getElementById('speakBtn');
if (speakBtn) {
    speakBtn.addEventListener('click', function() {
        // Use the stored translated text for consistent speaking
        if (lastTranslatedText && selectedVoice && !this.disabled) {
            speakText(lastTranslatedText);
        }
    });
}

// Download button removed

// Parameter slider update handlers - update display values as user moves sliders
function setupParameterSlider(sliderId, valueId, formatFn) {
    const slider = document.getElementById(sliderId);
    const valueDisplay = document.getElementById(valueId);
    
    if (slider && valueDisplay) {
        slider.addEventListener('input', function() {
            const value = parseFloat(this.value);
            valueDisplay.textContent = formatFn ? formatFn(value) : value.toFixed(3);
        });
        
        // Initialize display
        const initialValue = parseFloat(slider.value);
        valueDisplay.textContent = formatFn ? formatFn(initialValue) : initialValue.toFixed(3);
    }
}

// Setup parameter slider (only length scale now)
setupParameterSlider('lengthScaleSlider', 'lengthScaleValue', (v) => v.toFixed(2));

// Reset length scale button
const resetLengthScaleBtn = document.getElementById('resetLengthScaleBtn');
if (resetLengthScaleBtn) {
    resetLengthScaleBtn.addEventListener('click', function() {
        const lengthScaleSlider = document.getElementById('lengthScaleSlider');
        const lengthScaleValue = document.getElementById('lengthScaleValue');
        if (lengthScaleSlider && lengthScaleValue) {
            lengthScaleSlider.value = 1.0;
            lengthScaleValue.textContent = '1.00';
        }
    });
}

// Translation History Functions
async function saveTranslationToHistory(langCode, original, translated, model) {
    const history = getTranslationHistory();
    const langKey = langCode.toLowerCase();
    
    if (!history[langKey]) {
        history[langKey] = {
            langCode: langCode,
            langName: getLanguageName(langCode),
            flagPath: model.flag_path || `/flags/${langCode}.svg`,
            translations: []
        };
    }
    
    // Convert audio blob to base64 for storage if available
    let audioData = null;
    if (window.lastGeneratedAudio) {
        try {
            audioData = await blobToBase64(window.lastGeneratedAudio);
        } catch (e) {
            console.log('Could not save audio to history:', e);
        }
    }
    
    history[langKey].translations.unshift({
        id: Date.now(),
        original: original,
        translated: translated,
        timestamp: new Date().toISOString(),
        audioData: audioData
    });
    
    // Keep only last 100 translations per language
    if (history[langKey].translations.length > 100) {
        history[langKey].translations = history[langKey].translations.slice(0, 100);
    }
    
    localStorage.setItem('translationHistory', JSON.stringify(history));
    updateHistoryList();
}

// Helper function to convert blob to base64
function blobToBase64(blob) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(blob);
    });
}

function getTranslationHistory() {
    const stored = localStorage.getItem('translationHistory');
    return stored ? JSON.parse(stored) : {};
}

function updateHistoryList() {
    const history = getTranslationHistory();
    const historyList = document.getElementById('historyList');
    
    // Sort languages by most recent translation
    const languages = Object.values(history).sort((a, b) => {
        const aTime = a.translations[0]?.timestamp || '';
        const bTime = b.translations[0]?.timestamp || '';
        return bTime.localeCompare(aTime);
    });
    
    if (languages.length === 0) {
        historyList.innerHTML = '<div class="empty-history">No translation history yet</div>';
        return;
    }
    
    historyList.innerHTML = languages.map(lang => {
        const lastTranslation = lang.translations[0];
        const lastTime = lastTranslation ? new Date(lastTranslation.timestamp).toLocaleString() : '';
        return `
            <div class="history-language-item" onclick="openChatView('${lang.langCode}')">
                <div class="lang-name">${lang.langName} (${lang.langCode.toUpperCase()})</div>
                <div class="lang-count">${lang.translations.length} translation${lang.translations.length !== 1 ? 's' : ''}</div>
                ${lastTime ? `<div class="lang-time">Last: ${lastTime}</div>` : ''}
            </div>
        `;
    }).join('');
}

// Play audio from history
async function playHistoryAudio(event, translationId) {
    event.stopPropagation(); // Prevent opening chat view
    
    const button = event.target;
    const originalText = button.textContent;
    
    try {
        button.disabled = true;
        button.textContent = '...';
        
        // Find the translation in history
        const history = getTranslationHistory();
        let translation = null;
        
        for (const lang of Object.values(history)) {
            translation = lang.translations.find(t => t.id == translationId);
            if (translation) break;
        }
        
        if (!translation || !translation.audioData) {
            alert('Audio not found');
            return;
        }
        
        // Convert base64 back to blob
        const response = await fetch(translation.audioData);
        const audioBlob = await response.blob();
        
        // Play the audio
        const audioUrl = URL.createObjectURL(audioBlob);
        const audio = new Audio();
        audio.preload = 'auto';
        audio.src = audioUrl;
        
        audio.onended = function() {
            URL.revokeObjectURL(audioUrl);
            button.disabled = false;
            button.textContent = originalText;
        };
        
        audio.onerror = function(e) {
            console.error('Audio playback error:', e, audio.error);
            URL.revokeObjectURL(audioUrl);
            button.disabled = false;
            button.textContent = originalText;
            alert('Failed to play audio');
        };
        
        await audio.play();
        
    } catch (error) {
        console.error('Failed to play history audio:', error);
        button.disabled = false;
        button.textContent = originalText;
        alert('Failed to play audio: ' + error.message);
    }
}

function openChatView(langCode) {
    const history = getTranslationHistory();
    const langKey = langCode.toLowerCase();
    const langData = history[langKey];
    
    if (!langData) {
        return;
    }
    
    currentChatLanguage = langCode;
    currentView = 'chat';
    
    // Update chat header
    document.getElementById('chatFlag').src = langData.flagPath;
    document.getElementById('chatTitle').textContent = `${langData.langName} (${langCode.toUpperCase()})`;
    
    // Display translations
    const chatHistory = document.getElementById('chatHistory');
    if (langData.translations.length === 0) {
        chatHistory.innerHTML = '<div class="empty-history">No translations yet for this language</div>';
    } else {
        chatHistory.innerHTML = langData.translations.map(t => {
            const time = new Date(t.timestamp).toLocaleString();
            const hasAudio = t.audioData ? true : false;
            return `
                <div class="translation-pair">
                    <div class="original">${escapeHtml(t.original)}</div>
                    <div class="translated-with-controls">
                        <div class="translated">${escapeHtml(t.translated)}</div>
                        ${hasAudio ? `<button class="history-play-btn" onclick="playHistoryAudio(event, ${t.id})" title="Play audio">🔊</button>` : ''}
                    </div>
                    <div class="timestamp">${time}</div>
                </div>
            `;
        }).join('');
    }
    
    // Switch views
    document.getElementById('mainView').classList.add('hidden');
    document.getElementById('chatView').classList.add('active');
    closeSideMenu();
}

function backToMainView() {
    currentView = 'main';
    currentChatLanguage = null;
    document.getElementById('mainView').classList.remove('hidden');
    document.getElementById('chatView').classList.remove('active');
}

function toggleSideMenu() {
    const overlay = document.getElementById('sideMenuOverlay');
    const menu = document.getElementById('sideMenu');
    overlay.classList.toggle('active');
    menu.classList.toggle('active');
    updateHistoryList();
}

function closeSideMenu() {
    const overlay = document.getElementById('sideMenuOverlay');
    const menu = document.getElementById('sideMenu');
    overlay.classList.remove('active');
    menu.classList.remove('active');
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Load models and check health on page load
loadModels();
checkHealth();
updateHistoryList();

