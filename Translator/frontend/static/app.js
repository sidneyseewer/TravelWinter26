let availableModels = [];
let currentModelPath = null;
let selectedModel = null;
let currentView = 'main'; // 'main' or 'chat'
let currentChatLanguage = null;

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
        statusDiv.textContent = `✓ ${model.name} ready for translation`;
        statusDiv.style.background = '#e8f5e9';
        statusDiv.style.color = '#2e7d32';
        
    } catch (error) {
        console.error('Failed to select model:', error);
        statusDiv.textContent = `✗ Failed to load: ${error.message}`;
        statusDiv.style.background = '#fee';
        statusDiv.style.color = '#c33';
        
        // Remove selection on error
        document.querySelectorAll('.flag.selected').forEach(flag => {
            flag.classList.remove('selected');
        });
        selectedModel = null;
    }
}

// Check health status on load
async function checkHealth() {
    try {
        const response = await fetch('/health');
        const data = await response.json();
        
        const statusDiv = document.getElementById('status');
        if (data.models_loaded > 0) {
            statusDiv.textContent = `✓ ${data.models_loaded} model(s) loaded. Current: ${data.current_model || 'none'}`;
            statusDiv.style.display = 'block';
            statusDiv.style.background = '#e8f5e9';
            statusDiv.style.color = '#2e7d32';
        } else {
            statusDiv.textContent = `⚠ No models loaded (${data.complete_models} complete found). Check server logs.`;
            statusDiv.style.display = 'block';
            statusDiv.style.background = '#fff3cd';
            statusDiv.style.color = '#856404';
        }
    } catch (error) {
        console.error('Health check failed:', error);
        const statusDiv = document.getElementById('status');
        statusDiv.textContent = '⚠ Error checking service status';
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

    // Clear previous results
    result.style.display = 'none';
    error.style.display = 'none';

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

        // Show result
        translatedText.textContent = data.translated_text;
        result.style.display = 'block';

        // Save translation to history
        if (selectedModel) {
            const langCode = selectedModel.lang_code || selectedModel.name.split('-')[1] || 'unknown';
            saveTranslationToHistory(langCode, inputText, data.translated_text, selectedModel);
        }

    } catch (err) {
        error.textContent = `Error: ${err.message}`;
        error.style.display = 'block';
    } finally {
        translateBtn.disabled = false;
        loading.classList.remove('active');
    }
}

// Allow Enter key to trigger translation (Ctrl+Enter or Cmd+Enter)
document.getElementById('inputText').addEventListener('keydown', function(e) {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        translate();
    }
});

// Attach translate button click handler
document.getElementById('translateBtn').addEventListener('click', translate);

// Translation History Functions
function saveTranslationToHistory(langCode, original, translated, model) {
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
    
    history[langKey].translations.unshift({
        original: original,
        translated: translated,
        timestamp: new Date().toISOString()
    });
    
    // Keep only last 100 translations per language
    if (history[langKey].translations.length > 100) {
        history[langKey].translations = history[langKey].translations.slice(0, 100);
    }
    
    localStorage.setItem('translationHistory', JSON.stringify(history));
    updateHistoryList();
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
            return `
                <div class="translation-pair">
                    <div class="original">${escapeHtml(t.original)}</div>
                    <div class="translated">${escapeHtml(t.translated)}</div>
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

