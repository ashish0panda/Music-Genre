import os
import io
import time
import math
import tempfile
import numpy as np
import librosa
import soundfile as sf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydub import AudioSegment
import logging
from scipy.ndimage import binary_dilation
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Music Genre Classifier API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# GTZAN genre labels — must match the order used during training
# (preprocess.py walks the dataset_path alphabetically via os.walk)
# ---------------------------------------------------------------------------
GENRE_LABELS = [
    "blues", "classical", "country", "disco",
    "hiphop", "jazz", "metal", "pop", "reggae", "rock"
]

# MFCC extraction params — must match preprocess.py exactly
SAMPLE_RATE       = 22050
DURATION          = 30          # seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
N_MFCC            = 40
N_FFT             = 2048
HOP_LENGTH        = 512
NUM_SEGMENTS      = 10

# Path to the saved Keras model (place the file next to main.py or update the path)
MODEL_PATH = "best_genre_classifier.keras"

keras_model = None  # loaded at startup


# ---------------------------------------------------------------------------
# Startup: load the Keras model
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def load_model():
    global keras_model
    try:
        import tensorflow as tf
        keras_model = tf.keras.models.load_model(MODEL_PATH)
        logger.info(f"✓ Loaded local Keras model from '{MODEL_PATH}'")
        keras_model.summary(print_fn=logger.info)
    except Exception as e:
        logger.error(f"✗ Failed to load Keras model: {e}")
        keras_model = None


# ---------------------------------------------------------------------------
# MFCC extraction (mirrors preprocess.py segment logic)
# ---------------------------------------------------------------------------

def extract_mfcc_segments(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Split the audio into NUM_SEGMENTS equal chunks, extract an MFCC matrix
    for each, and return an array of shape (num_valid_segments, frames, N_MFCC).
    This replicates exactly what preprocess.py stored in data.json.
    """
    num_samples_per_segment = int(SAMPLES_PER_TRACK / NUM_SEGMENTS)
    expected_frames = math.ceil(num_samples_per_segment / HOP_LENGTH)

    segments = []
    for s in range(NUM_SEGMENTS):
        start  = num_samples_per_segment * s
        finish = start + num_samples_per_segment

        chunk = y[start:finish]
        if len(chunk) < num_samples_per_segment:
            # Pad the last short chunk so it always produces a full MFCC matrix
            chunk = np.pad(chunk, (0, num_samples_per_segment - len(chunk)))

        mfcc = librosa.feature.mfcc(
            y=chunk, sr=sr,
            n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH
        )
        mfcc = mfcc.T  # shape: (frames, N_MFCC)

        if len(mfcc) == expected_frames:
            segments.append(mfcc)

    if not segments:
        raise ValueError("Could not extract any valid MFCC segments from the audio.")

    return np.array(segments)  # (S, frames, N_MFCC)


# ---------------------------------------------------------------------------
# Keras-based genre classification
# ---------------------------------------------------------------------------

def classify_with_keras(y: np.ndarray, sr: int) -> list[dict]:
    """
    Returns a list of {genre, confidence} dicts sorted by confidence (desc).
    Averages softmax probabilities over all extracted segments.
    """
    if keras_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Please try again.")

    # Pad / trim to exactly 30 s so segment math is consistent
    target_len = SAMPLES_PER_TRACK
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    segments = extract_mfcc_segments(y, sr)          # (S, frames, N_MFCC)
    X = segments[..., np.newaxis]                     # (S, frames, N_MFCC, 1)

    probs = keras_model.predict(X, verbose=0)         # (S, 10)
    avg_probs = probs.mean(axis=0)                    # (10,)

    results = [
        {
            "genre":      GENRE_LABELS[i],
            "confidence": round(float(avg_probs[i]) * 100, 1),
        }
        for i in range(len(GENRE_LABELS))
    ]
    results.sort(key=lambda x: -x["confidence"])
    return results


# ---------------------------------------------------------------------------
# Mood inference (unchanged from original)
# ---------------------------------------------------------------------------

def infer_mood(y: np.ndarray, sr: int, tempo: float, key_idx: int) -> list[dict]:
    moods = defaultdict(float)

    if tempo > 140:
        moods["Energetic"] += 0.9; moods["Upbeat"] += 0.7; moods["Happy"] += 0.5
    elif tempo > 110:
        moods["Happy"] += 0.7; moods["Upbeat"] += 0.6; moods["Energetic"] += 0.4
    elif tempo > 80:
        moods["Calm"] += 0.5; moods["Romantic"] += 0.4; moods["Melancholic"] += 0.3
    else:
        moods["Calm"] += 0.8; moods["Romantic"] += 0.7
        moods["Melancholic"] += 0.6; moods["Sad"] += 0.5

    chroma      = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    major_profile = np.roll(np.array([1,0,1,0,1,1,0,1,0,1,0,1], dtype=float), key_idx)
    minor_profile = np.roll(np.array([1,0,1,1,0,1,0,1,1,0,1,0], dtype=float), key_idx)
    is_minor = (np.dot(chroma_mean, minor_profile) > np.dot(chroma_mean, major_profile))

    if is_minor:
        moods["Sad"] += 0.6; moods["Melancholic"] += 0.7
        moods["Romantic"] += 0.5; moods["Dark"] += 0.4; moods["Happy"] -= 0.3
    else:
        moods["Happy"] += 0.6; moods["Uplifting"] += 0.5
        moods["Energetic"] += 0.3; moods["Sad"] -= 0.3

    rms = float(np.mean(librosa.feature.rms(y=y)))
    if rms > 0.08:
        moods["Energetic"] += 0.6; moods["Aggressive"] += 0.4
    elif rms > 0.03:
        moods["Upbeat"] += 0.3
    else:
        moods["Calm"] += 0.6; moods["Peaceful"] += 0.5; moods["Romantic"] += 0.3

    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    if centroid > 3000:
        moods["Bright"] += 0.5; moods["Happy"] += 0.3
    elif centroid < 1500:
        moods["Dark"] += 0.4; moods["Melancholic"] += 0.3; moods["Romantic"] += 0.2

    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    if zcr < 0.04:
        moods["Smooth"] += 0.5; moods["Romantic"] += 0.4; moods["Calm"] += 0.3
    elif zcr > 0.12:
        moods["Aggressive"] += 0.4; moods["Energetic"] += 0.3

    result = [{"mood": m, "score": round(min(s, 1.0), 3)}
              for m, s in moods.items() if s > 0.2]
    result.sort(key=lambda x: -x["score"])
    if result:
        top = result[0]["score"]
        for r in result:
            r["confidence"] = round((r["score"] / top) * 100, 1)
    return result[:6]


# ---------------------------------------------------------------------------
# Audio processing helpers (unchanged from original)
# ---------------------------------------------------------------------------

def convert_to_wav(file_bytes: bytes, content_type: str, filename: str = "") -> bytes:
    fmt_map = {
        "audio/mpeg": "mp3", "audio/mp3": "mp3",
        "audio/wav": "wav", "audio/wave": "wav", "audio/x-wav": "wav",
        "audio/ogg": "ogg", "audio/flac": "flac",
        "audio/aac": "aac", "audio/m4a": "m4a", "audio/x-m4a": "m4a",
        "audio/webm": "webm", "video/webm": "webm",
    }
    fmt = fmt_map.get(content_type)
    if not fmt and filename:
        ext = filename.rsplit(".", 1)[-1].lower()
        fmt = ext if ext in fmt_map.values() else "mp3"
    if not fmt:
        fmt = "mp3"

    tmp_in = tmp_out = None
    try:
        with tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False) as f:
            f.write(file_bytes); tmp_in = f.name
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_out = f.name
        audio = AudioSegment.from_file(tmp_in, format=fmt)
        audio = audio.set_channels(1).set_frame_rate(22050)
        audio.export(tmp_out, format="wav")
        with open(tmp_out, "rb") as f:
            return f.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not process audio: {str(e)}")
    finally:
        for p in [tmp_in, tmp_out]:
            if p:
                try: os.unlink(p)
                except: pass


def load_audio(wav_bytes: bytes, sr: int = 22050):
    buf = io.BytesIO(wav_bytes)
    y, orig_sr = sf.read(buf, dtype="float32", always_2d=False)
    if orig_sr != sr:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
    return y, sr


def trim_and_gate(y: np.ndarray, sr: int) -> np.ndarray:
    y_trimmed, _ = librosa.effects.trim(y, top_db=25)
    if len(y_trimmed) < sr:
        return y_trimmed
    frame_length, hop_length = 2048, 512
    rms = librosa.feature.rms(y=y_trimmed, frame_length=frame_length, hop_length=hop_length)[0]
    max_rms = np.max(rms)
    if max_rms < 0.001:
        return y_trimmed
    mask = binary_dilation(rms > (max_rms * 0.20), iterations=8)
    y_gated = np.zeros_like(y_trimmed)
    for i, keep in enumerate(mask):
        s, e = i * hop_length, min(i * hop_length + frame_length, len(y_trimmed))
        if keep:
            y_gated[s:e] = y_trimmed[s:e]
    ratio = np.sqrt(np.mean(y_gated**2)) / (np.sqrt(np.mean(y_trimmed**2)) + 1e-9)
    return y_gated if ratio >= 0.30 else y_trimmed


def extract_features(y: np.ndarray, sr: int) -> tuple[dict, float, int]:
    features = {}
    tempo, key_idx = 120.0, 0
    try:
        tempo_arr, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(tempo_arr)
        features["tempo_bpm"] = round(tempo, 1)

        chroma  = librosa.feature.chroma_cqt(y=y, sr=sr)
        key_names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
        key_idx = int(np.argmax(chroma.mean(axis=1)))
        features["key"] = key_names[key_idx]

        features["energy"]       = round(float(np.mean(librosa.feature.rms(y=y))) * 100, 2)
        features["duration_sec"] = round(len(y) / sr, 2)

        centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        features["brightness_hz"] = round(centroid, 0)
    except Exception as e:
        logger.warning(f"Feature extraction error: {e}")
    return features, tempo, key_idx


# ---------------------------------------------------------------------------
# Main classify logic
# ---------------------------------------------------------------------------

def process_audio_classify(wav_bytes: bytes) -> dict:
    if keras_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Please try again in a moment.")

    y, sr = load_audio(wav_bytes)

    rms_overall = float(np.sqrt(np.mean(y**2)))
    logger.info(f"Input RMS: {rms_overall:.5f}, duration: {len(y)/sr:.1f}s")
    if rms_overall < 0.002:
        raise HTTPException(status_code=400,
                            detail="Audio is too quiet. Please record closer to the music source.")

    y_clean = trim_and_gate(y, sr)
    if len(y_clean) < sr * 2:
        raise HTTPException(status_code=400,
                            detail="Not enough music detected. Please ensure music is playing clearly.")

    # Clip / pad to 30 s for the classifier (feature extraction uses full clean audio)
    max_samples = sr * 30
    if len(y_clean) > max_samples:
        start  = (len(y_clean) - max_samples) // 2
        y_clip = y_clean[start:start + max_samples]
    else:
        y_clip = y_clean

    # --- Genre classification ---
    start_time = time.time()
    logger.info("Running Keras genre classifier...")
    predictions = classify_with_keras(y_clip, sr)
    inference_time = round(time.time() - start_time, 2)
    logger.info(f"Inference done in {inference_time}s — top genre: {predictions[0]['genre']}")

    # --- Audio features + mood ---
    features, tempo, key_idx = extract_features(y_clip, sr)
    moods = infer_mood(y_clip, sr, tempo, key_idx)

    return {
        "predictions":       predictions,
        "top_genre":         predictions[0]["genre"] if predictions else "Unknown",
        "confidence":        predictions[0]["confidence"] if predictions else 0,
        "features":          features,
        "moods":             moods,
        "model_used":        "best_genre_classifier.keras",
        "inference_time_sec": inference_time,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    status = "loaded" if keras_model is not None else "not loaded"
    return {"message": "Music Genre Classifier API v3", "model": MODEL_PATH, "model_status": status}


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": keras_model is not None, "model_path": MODEL_PATH}


@app.post("/classify")
async def classify_audio(file: UploadFile = File(...)):
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(file_bytes) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large. Max 50MB.")

    logger.info(f"Processing: {file.filename}, {file.content_type}, {len(file_bytes)} bytes")
    wav_bytes = convert_to_wav(file_bytes, file.content_type or "audio/mpeg", file.filename or "")
    result    = process_audio_classify(wav_bytes)

    return JSONResponse(content={"success": True, "filename": file.filename, **result})