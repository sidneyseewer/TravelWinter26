# Quick Start Guide

## Prerequisites

Install Node.js and npm:

```bash
# On Ubuntu/Debian:
sudo apt update
sudo apt install nodejs npm

# Or use NodeSource repository for latest version:
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs

# Verify installation:
node --version
npm --version
```

## Setup

1. Navigate to the npm directory:
   ```bash
   cd npm
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

## Running

### Option 1: With Python Backend (Full Functionality)

1. Start Python backend (in one terminal):
   ```bash
   cd ../frontend
   python3 app.py
   ```

2. Start Node.js server (in another terminal):
   ```bash
   cd npm
   npm start
   ```

3. Access at: http://localhost:3000

### Option 2: Node.js Only (Limited - Frontend Only)

```bash
cd npm
npm start
```

Note: Translation and TTS will not work without the Python backend.

## Development Mode

For auto-reload during development:

```bash
npm run dev
```

## Environment Variables

You can customize the server with environment variables:

```bash
PORT=3000 PYTHON_BACKEND_URL=http://localhost:8000 npm start
```

Or create a `.env` file:
```
PORT=3000
PYTHON_BACKEND_URL=http://localhost:8000
```

