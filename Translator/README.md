# MarianMT Translation Service

A Python-based translation service using MarianMT models from the OPUS-MT project.

## Architecture

- **FastAPI web service** that loads models from a local `models` folder
- Models are expected to be in directories starting with `en_` (e.g., `en_de`, `en_it`, `en_nl`, `en_ru`)
- Models should be placed in the `models` folder at the same level as the `frontend` folder

## Quick Start

1. **Install dependencies:**
   ```bash
   cd frontend
   pip install -r requirements.txt
   ```

2. **Prepare your models:**
   - Place your MarianMT models in the `models` folder (same level as `frontend`)
   - Models should be in directories starting with `en_` (e.g., `en_de`, `en_it`, `en_nl`, `en_ru`)
   - Each model directory should contain the standard MarianMT model files (config.json, tokenizer files, etc.)

3. **Run the application:**
   ```bash
   # Option 1: Use the run script (Windows)
   run.bat
   
   # Option 2: Use the run script (Linux/Mac)
   ./run.sh
   
   # Option 3: Run directly
   cd frontend
   python app.py
   ```

4. **Access the web interface:**
   - The service will scan for models and display them in the console
   - Open your browser to `http://localhost:8000` (link is shown in the console output)

5. **The application will:**
   - Automatically scan the `../models` folder for directories starting with `en_`
   - Display all available models on the webpage
   - Load the first available model by default
   - Allow you to switch between models via the dropdown

## Model Setup

### Model Directory Structure

Place your models in the `models` folder with the following structure:

```
Translator/
├── models/
│   ├── en_de/          # English to German
│   │   ├── config.json
│   │   ├── tokenizer files...
│   │   └── model files...
│   ├── en_it/          # English to Italian
│   ├── en_nl/          # English to Dutch
│   └── en_ru/          # English to Russian
└── frontend/
```

### Model Naming Convention

- Model directories must start with `en_` followed by the target language code
- Examples: `en_de`, `en_it`, `en_nl`, `en_ru`, `en_fr`, `en_es`
- The system automatically scans and lists all directories matching this pattern

### Getting Models

Models can be obtained from:
- The [OPUS-MT training repository](https://github.com/Helsinki-NLP/OPUS-MT-train) by Helsinki-NLP
- Hugging Face: https://huggingface.co/models?library=marian
- Pre-trained models are available under CC-BY 4.0 license

## API Endpoints

### `GET /`
Web interface for translation

### `POST /translate`
Translate text

**Request:**
```json
{
  "text": "Hello, world!",
  "model_path": "/app/models/en_de"  // Optional: specify which model to use
}
```

**Response:**
```json
{
  "original_text": "Hello, world!",
  "translated_text": "Hallo, Welt!",
  "model_path": "/app/models/en_de"
}
```

### `GET /health`
Check service health and model status

### `GET /api/models`
List available models in the models directory

### `GET /api/languages`
Get all available language models (directories starting with `en_`)

**Response:**
```json
{
  "models": [
    {
      "model_name": "en_de",
      "model_path": "/app/models/en_de",
      "source_language": "en",
      "target_language": "de"
    }
  ],
  "language_pairs": [
    {
      "source": "en",
      "target": "de",
      "model_name": "en_de",
      "model_path": "/app/models/en_de"
    }
  ]
}
```

## Development

### Running the application:
```bash
cd frontend
python app.py
```

### Running with custom host/port:
```bash
cd frontend
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Installing dependencies:
```bash
cd frontend
pip install -r requirements.txt
```

### Model directory:
The application looks for models in `../models/` relative to the `frontend` directory.

## Building an App

The FastAPI backend provides a REST API that you can integrate into any application:

- **Web apps**: Use the `/translate` endpoint with fetch/axios
- **Mobile apps**: Call the API from your mobile framework
- **CLI tools**: Use curl or httpie to interact with the API
- **Other services**: Integrate via HTTP requests

Example curl:
```bash
curl -X POST http://localhost:8000/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world!"}'
```

## File Structure

```
Translator/
├── docker-compose.yml
├── models/              # Place your model directories here (en_de, en_it, etc.)
│   ├── en_de/
│   ├── en_it/
│   └── ...
└── frontend/
    ├── Dockerfile
    ├── requirements.txt
    ├── app.py
    ├── templates/
    │   └── index.html
    └── static/
```

## Notes

- **Local models**: Models must be placed in the `models` folder before running the application
- **Auto-detection**: The system automatically scans `../models/` for directories starting with `en_`
- **Model switching**: You can switch between models using the dropdown on the web interface
- **Model display**: All available models and their paths are displayed on the webpage
- **Startup logging**: The service logs all found models and the web interface URL when starting
- **Lazy loading**: Models are loaded on first use or when switched
- **Relative paths**: The application uses relative paths (`../models/`) so it works when run from the `frontend` directory

## Model Sources

The models are from the [OPUS-MT training repository](https://github.com/Helsinki-NLP/OPUS-MT-train) by Helsinki-NLP, distributed under CC-BY 4.0 license. Pre-trained models are available on Hugging Face.

To use models from Hugging Face, download them using the transformers library and place them in the `models` folder with the appropriate naming (e.g., `en_de` for English to German models).

