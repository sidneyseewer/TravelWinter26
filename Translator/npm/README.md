# Translation Service - Standalone Node.js Server

A fully standalone Node.js/Express server for offline translation and text-to-speech. Works independently from Python - uses `@xenova/transformers` for translation and `onnxruntime-node` for TTS.

## Features

- ✅ **Standalone Operation**: No Python backend required
- ✅ **Translation**: Uses HuggingFace Transformers models (MarianMT)
- ✅ **Text-to-Speech**: Uses ONNX Runtime for Piper TTS models
- ✅ **Responsive UI**: Mobile-friendly interface
- ✅ **Offline**: All models and voices loaded from local directories

## Prerequisites

- Node.js 18+ and npm
- Models in `../models/` directory (same structure as Python version)
- Voices in `../voices/` directory (same structure as Python version)
- Flags in `../flags/` directory

## Installation

```bash
cd npm
npm install
```

## Usage

### Start the server:

```bash
npm start
```

Or for development with auto-reload:

```bash
npm run dev
```

The server will start on port 3000 by default.

### Access the application:

Open your browser to: http://localhost:3000

## Project Structure

```
npm/
├── server.js          # Main server file
├── package.json       # Dependencies
├── public/
│   ├── templates/
│   │   └── index.html # Frontend HTML
│   └── static/
│       ├── app.js     # Frontend JavaScript
│       └── style.css  # Responsive CSS
└── README.md
```

## API Endpoints

- `GET /health` - Server health check
- `GET /api/models` - List available translation models
- `POST /api/select-model` - Load a translation model
- `POST /translate` - Translate text
- `GET /api/voices/:langCode` - List available voices for a language
- `POST /api/synthesize` - Generate speech from text

## Environment Variables

- `PORT` - Server port (default: 3000)

## Notes

- Models are loaded on-demand (lazy loading)
- Voices are cached in memory after first load
- The server automatically scans the `models/` and `voices/` directories on startup
- Translation uses HuggingFace Transformers (compatible with MarianMT models)
- TTS uses ONNX Runtime for Piper TTS models

## Troubleshooting

### Models not loading

- Ensure models are in `../models/` directory
- Check that model directories start with `en-` (e.g., `en-fi`, `en-fr`)
- Verify models have `config.json` and weight files

### Voices not found

- Ensure voices are in `../voices/` directory
- Check voice structure: `voices/{lang}/{locale}/{voice_name}/{quality}/*.onnx`
- Verify `.onnx` and `.onnx.json` files exist (not Git LFS pointers)

### Translation errors

- Some OPUS-MT format models may need conversion to HuggingFace format
- Check console logs for detailed error messages

## Mobile Support

The frontend is fully responsive and optimized for mobile devices:
- Touch-friendly interface
- Responsive layout that adapts to screen size
- Mobile-optimized controls and buttons
