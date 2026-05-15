# 🎵 GenreScope — Music Genre Classifier

An AI-powered music genre classifier that listens to audio (recorded or uploaded) and identifies the genre using a pre-trained deep learning model.

---

## ✨ Features

- **Live Recording** — Record up to 30 seconds directly in the browser
- **File Upload** — Drag & drop or browse MP3, WAV, FLAC, OGG, M4A files
- **AI Classification** — Uses `mtg-upf/discogs-maest-30s-pw-73e-ts` (trained on 400+ genres)
- **Audio Analysis** — Tempo (BPM), musical key, energy, duration
- **Confidence Scores** — Shows top 8 genre predictions with bar charts
- **Beautiful UI** — Dark aesthetic with animated vinyl record, waveform visualizer

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│              Browser (React + Vite)          │
│  ┌──────────────┐   ┌────────────────────┐  │
│  │  Mic Record  │   │   File Upload/DnD  │  │
│  └──────┬───────┘   └────────┬───────────┘  │
│         └──────────┬─────────┘              │
│                    │ FormData POST /classify │
└────────────────────┼────────────────────────┘
                     │
┌────────────────────▼────────────────────────┐
│           FastAPI Backend (Python)           │
│  ┌──────────────────────────────────────┐   │
│  │  pydub: Convert to WAV (mono 22kHz)  │   │
│  ├──────────────────────────────────────┤   │
│  │  HuggingFace Transformers Pipeline   │   │
│  │  Model: discogs-maest-30s-pw-73e-ts  │   │
│  ├──────────────────────────────────────┤   │
│  │  librosa: BPM, Key, Energy features  │   │
│  └──────────────────────────────────────┘   │
│  Returns: { predictions, features, ... }     │
└─────────────────────────────────────────────┘
```

---

## 📋 Requirements

| Tool | Min Version | Purpose |
|------|-------------|---------|
| Python | 3.9+ | Backend runtime |
| Node.js | 18+ | Frontend dev server |
| ffmpeg | Any | Audio conversion (via pydub) |

### Install ffmpeg

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt update && sudo apt install ffmpeg -y
```

**Windows:**
Download from https://ffmpeg.org/download.html and add to PATH.

---

## 🚀 Step-by-Step Setup

### Step 1: Clone / Download the project

```bash
# If you have it as a zip, extract it
# Then enter the project folder:
cd music-genre-classifier
```

### Step 2: Start the Backend

Open **Terminal 1**:

```bash
cd backend
chmod +x start.sh
./start.sh
```

**What this does:**
1. Creates a Python virtual environment (`venv/`)
2. Installs all Python packages
3. Starts FastAPI on `http://localhost:8000`
4. On first run, downloads the AI model (~500MB from HuggingFace) — this takes 2–5 minutes

**Manual setup (alternative):**
```bash
cd backend
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

**Verify backend is running:**
```
http://localhost:8000/health
→ {"status":"ok","model_loaded":true}
```

> ⚠️ Wait for `model_loaded: true` before using the app (model loads in ~30-60s after server starts)

---

### Step 3: Start the Frontend

Open **Terminal 2**:

```bash
cd frontend
chmod +x start.sh
./start.sh
```

**Manual setup (alternative):**
```bash
cd frontend
npm install
npm run dev
```

Frontend starts at: **http://localhost:5173**

---

### Step 4: Use the App

1. Open **http://localhost:5173** in your browser
2. **Record tab:** Click the microphone button → play music near your mic → click Stop → click "Analyze Genre"
3. **Upload tab:** Drag & drop an audio file or click to browse → click "Analyze Genre"
4. Wait 5–30 seconds (depends on file size)
5. See genre predictions with confidence scores and audio features!

---

## 🤖 AI Model Details

**Primary:** `mtg-upf/discogs-maest-30s-pw-73e-ts`
- Trained on Discogs dataset with 400+ genre/style tags
- Architecture: Music Audio Efficient Spectrogram Transformer (MAEST)
- Input: 30 seconds of audio at 16kHz
- Output: Top genre predictions with confidence scores

**Fallback:** `dima806/music_genres_classification`
- CNN-based classifier
- 10 major genres (blues, classical, country, disco, hip-hop, jazz, metal, pop, reggae, rock)

---

## 📁 Project Structure

```
music-genre-classifier/
├── backend/
│   ├── main.py              # FastAPI app + classification logic
│   ├── requirements.txt     # Python dependencies
│   └── start.sh             # Setup & run script
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Main UI component
│   │   ├── index.css        # Global styles & animations
│   │   ├── main.jsx         # React entry point
│   │   ├── components/
│   │   │   ├── GenreResult.jsx      # Results display
│   │   │   └── WaveformVisualizer.jsx
│   │   └── hooks/
│   │       └── useAudioRecorder.js  # Mic recording logic
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js       # Dev proxy → backend
│   └── start.sh
└── README.md
```

---

## 🔧 Troubleshooting

**`Model not loaded yet` error:**
- Backend just started. Wait 60 seconds and try again.

**`Microphone access denied`:**
- Allow microphone in browser permissions (click 🔒 in address bar)

**`Failed to process audio file`:**
- Make sure ffmpeg is installed: `ffmpeg -version`
- Try a different audio format (WAV works most reliably)

**Backend won't start / import errors:**
```bash
# Make sure you're in the venv
source backend/venv/bin/activate
pip install -r backend/requirements.txt --force-reinstall
```

**Port already in use:**
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9
# Or change port:
uvicorn main:app --port 8001
```

---

## 🎨 Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18, Vite 5 |
| Backend | Python, FastAPI, Uvicorn |
| ML Model | HuggingFace Transformers, PyTorch |
| Audio Processing | librosa, pydub, soundfile |
| Styling | Pure CSS (no UI libraries) |

---

## 🚢 Production Deployment

For production, build the frontend and serve it:

```bash
# Build frontend
cd frontend && npm run build

# Serve with FastAPI (add to main.py)
from fastapi.staticfiles import StaticFiles
app.mount("/", StaticFiles(directory="../frontend/dist", html=True))

# Run without --reload
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 2
```
