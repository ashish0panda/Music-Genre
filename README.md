# Music Genre Classification (GTZAN)

Course project for **IIT Ropar — 2nd Semester, Machine Learning**.

The repository contains two related bodies of work on the GTZAN music-genre dataset:

1. **A comparative study notebook** ([music_genre_classification_full_colab.ipynb](music_genre_classification_full_colab.ipynb)) covering the five assignment tasks — handcrafted features, PCA, a spectrogram CNN, pretrained audio embeddings, and a comparative analysis.
2. **A standalone deep-learning pipeline plus a web application** — Python scripts that turn raw `.wav` audio into MFCC / mel-spectrogram tensors, train Keras CNNs on them, and a FastAPI + React app ("GenreScope") that classifies a recorded or uploaded clip in the browser.

---

## 1. Dataset

The [GTZAN genre collection](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) is expected under `Data/` (git-ignored — download it separately):

```
Data/
├── genres_original/       # 10 genres × 100 × 30 s .wav clips (jazz has 99 usable files)
│   ├── blues/blues.00000.wav ...
│   └── ... classical, country, disco, hiphop, jazz, metal, pop, reggae, rock
├── images_original/       # pre-rendered mel-spectrogram PNGs, one per track
├── features_30_sec.csv    # 1 000 rows  × 58 handcrafted features
└── features_3_sec.csv     # 9 990 rows × 58 handcrafted features (3 s chunks)
```

Genre label order is alphabetical everywhere in the project:

```
blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock
```

---

## 2. Repository layout

| Path | What it is |
| --- | --- |
| [preprocess.py](preprocess.py) | Extracts **40 MFCCs** per frame from each track, split into 10 segments of 3 s, and writes `data.json`. |
| [preprocess_1.py](preprocess_1.py) | Memory-optimised variant: **128-band mel spectrograms** with 4× data augmentation (original, noise, pitch shift, time stretch), saved as compressed `data.npz`. |
| [train.py](train.py) | Baseline Keras CNN on the MFCC tensors (70/30 split) → `best_genre_classifier.keras`. |
| [train_1.py](train_1.py) | Same model plus a full plotting/reporting suite; writes PNGs into `training_plots/`. |
| [train_2.py](train_2.py) | Deeper, regularised CNN (L2 + GlobalAveragePooling) trained on the augmented mel data with a 70/15/15 split → `best_model.keras`. |
| [visualize.py](visualize.py) | Loads a trained model, regenerates the confusion matrix, the class-wise report and Conv2D weight histograms. |
| [music_genre_classification_full_colab.ipynb](music_genre_classification_full_colab.ipynb) | The 5-task comparative study (scikit-learn + PyTorch + HuggingFace). |
| [web/backend/](web/backend/) | FastAPI inference service (`main.py`) with the trained Keras model bundled next to it. |
| [web/frontend/](web/frontend/) | React 18 + Vite single-page app: record from the mic or drop a file, see genre / mood / audio features. |
| [Music_Genre_classification.pdf](Music_Genre_classification.pdf) | Project problem statement / report. |

Generated artefacts (`Data/`, `*.keras`, `*.npz`, `*.json`, `training_plots/`, `music_genre_project_outputs/`, `node_modules/`) are git-ignored; only the source is tracked.

---

## 3. Notebook — the five tasks

Everything uses one **song-level split** (700 train / 100 val / 200 test songs) so that 3-second chunks of the same track never straddle the split.

### Task 1 — Handcrafted features (SVM-RBF and Random Forest)

| Data | Model | Aggregation | Accuracy | Macro F1 |
| --- | --- | --- | --- | --- |
| 30 s | SVM (RBF) | — | 0.750 | 0.750 |
| 30 s | Random Forest | — | 0.790 | 0.787 |
| 3 s | SVM (RBF) | majority vote | 0.795 | 0.795 |
| 3 s | SVM (RBF) | probability average | 0.795 | 0.794 |
| **3 s** | **SVM (RBF)** | **confidence-weighted vote** | **0.800** | **0.799** |
| 3 s | Random Forest | confidence-weighted vote | 0.795 | 0.792 |

The 3-second setup wins after song-level aggregation: ~10× more training examples per song improves generalisation, and confidence-weighted voting is the best of the three aggregation rules.

### Task 2 — PCA on the best Task 1 pipeline (3 s + SVM-RBF)

| Components | Accuracy | Macro F1 |
| --- | --- | --- |
| 0.95 variance | 0.790 | 0.786 |
| **0.99 variance** | **0.805** | **0.803** |
| 50 | 0.800 | 0.798 |
| 30 | 0.765 | 0.760 |
| 20 | 0.780 | 0.777 |
| 10 | 0.745 | 0.737 |

Keeping 99 % of the variance slightly *beats* the un-reduced baseline (de-noising + decorrelation help the RBF kernel); compressing below ~30 components starts discarding genre-discriminative information.

### Task 3 — Spectrogram 2D CNN (PyTorch)

A 4-block CNN (32→64→128→256 channels, BatchNorm + max-pool, 224×224 inputs, 70 epochs) trained from scratch on `images_original/`, with Grad-CAM explainability. **Song-level test accuracy 0.660, macro F1 0.649** — only 699 training images is very little for a from-scratch CNN, which is exactly the point the segment-based Keras pipeline below addresses.

### Task 4 — Pretrained audio embeddings + MLP

768-dimensional [MERT-v1-95M](https://huggingface.co/m-a-p/MERT-v1-95M) embeddings per track, cached to `music_genre_project_outputs/cache/`, classified with an MLP: **test accuracy 0.835, macro F1 0.833** — the best result in the notebook.

### Task 5 — Comparative analysis

Pretrained representations (Task 4) beat handcrafted features (Tasks 1–2), which in turn beat the small from-scratch spectrogram CNN (Task 3). `rock`, `country` and `disco` are the consistently confused classes across every method; `classical` and `metal` are the easiest.

---

## 4. Standalone Keras pipeline

Two independent pipelines live at the repo root. They share the 3-second segmentation idea: each 30 s track becomes 10 segments, so ~1 000 tracks yield ~9 986 training examples.

### Pipeline A — MFCC (deployed in the web app)

```bash
python preprocess.py     # Data/genres_original -> data.json  (~2 GB of MFCC-40 JSON)
python train.py          # -> best_genre_classifier.keras
# or, with plots and a classification report:
python train_1.py        # -> best_genre_classifier.keras + training_plots/
python visualize.py      # -> confusion_matrix.png, class_accuracy_report.txt, weights_distribution_*.png
```

Feature config: `sr = 22050 Hz`, `n_mfcc = 40`, `n_fft = 2048`, `hop_length = 512`, 10 segments/track → each example is a **130 × 40** matrix.

Architecture (394 K parameters):

```
Conv2D(256, 3×3) → MaxPool(3×3, s2) → BatchNorm
Conv2D(128, 3×3) → MaxPool(3×3, s2) → BatchNorm
Conv2D( 64, 2×2) → MaxPool(2×2, s2) → BatchNorm
Flatten → Dense(64, relu) → Dropout(0.4) → Dense(10, softmax)
Adam(1e-4), sparse categorical cross-entropy, batch 32
Callbacks: ModelCheckpoint(val_accuracy) + EarlyStopping(15) + ReduceLROnPlateau(0.5, 5)
```

Segment-level test results (`class_accuracy_report.txt`, 2 996 held-out segments) — **83 % accuracy, 0.83 macro F1**:

| Genre | Precision | Recall | F1 |
| --- | --- | --- | --- |
| blues | 0.90 | 0.84 | 0.87 |
| classical | 0.94 | 0.95 | 0.95 |
| country | 0.79 | 0.73 | 0.76 |
| disco | 0.77 | 0.79 | 0.78 |
| hiphop | 0.85 | 0.84 | 0.85 |
| jazz | 0.83 | 0.94 | 0.88 |
| metal | 0.84 | 0.95 | 0.89 |
| pop | 0.80 | 0.89 | 0.84 |
| reggae | 0.87 | 0.78 | 0.82 |
| rock | 0.73 | 0.65 | 0.69 |

An earlier 13-MFCC run of the same architecture is kept as `best_genre_classifier_1.keras` (input 130 × 13); its full training log is in [output.txt](output.txt) (best val accuracy 0.820).

### Pipeline B — Augmented mel spectrograms

```bash
python preprocess_1.py   # -> data.npz (X, y, mapping) — augmented, ~3 GB compressed
python train_2.py        # -> best_model.keras + loss_graph.png, accuracy_graph.png, confusion_matrix.png
```

Each example is a **130 × 128** log-mel patch; augmentation (Gaussian noise, +2 semitone pitch shift, 0.9× time stretch) quadruples the dataset. The model normalises on-GPU with a leading `BatchNormalization`, uses three L2-regularised conv blocks with progressive dropout and `GlobalAveragePooling2D` instead of `Flatten`, and is trained on a stratified 70/15/15 train/val/test split.

---

## 5. Web app — GenreScope

`web/` is a two-process app: a FastAPI service that runs the MFCC model, and a Vite dev server that proxies `/classify` and `/health` to it.

### Backend

```bash
cd web/backend
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install tensorflow                            # see the note below
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Endpoints:

| Method | Route | Description |
| --- | --- | --- |
| `GET` | `/` | Service banner + whether the model loaded |
| `GET` | `/health` | `{status, model_loaded, model_path}` |
| `POST` | `/classify` | multipart `file` upload (≤ 50 MB) → genre ranking, moods and audio features |

What a request does:

1. `pydub`/ffmpeg converts the upload (mp3, wav, ogg, flac, aac, m4a, webm) to mono 22 050 Hz WAV.
2. Silence is trimmed and an RMS noise gate keeps only the musical parts; too-quiet or too-short clips are rejected with a 400.
3. The audio is centre-cropped/padded to 30 s and split into 10 segments; MFCCs are extracted with **exactly the same parameters as `preprocess.py`** and the softmax outputs are averaged across segments.
4. `librosa` additionally reports tempo (BPM), key, energy, brightness and duration, and a rule-based `infer_mood` maps tempo / major-minor chroma / RMS / spectral centroid / ZCR onto moods such as *Energetic*, *Melancholic* or *Romantic*.

Response shape:

```json
{
  "success": true,
  "filename": "clip.mp3",
  "top_genre": "jazz",
  "confidence": 71.4,
  "predictions": [{"genre": "jazz", "confidence": 71.4}, "..."],
  "features": {"tempo_bpm": 96.3, "key": "F", "energy": 4.21, "brightness_hz": 1830, "duration_sec": 30.0},
  "moods": [{"mood": "Calm", "score": 0.9, "confidence": 100.0}],
  "model_used": "best_genre_classifier.keras",
  "inference_time_sec": 0.42
}
```

### Frontend

```bash
cd web/frontend
npm install
npm run dev        # http://localhost:5173
```

React 18 + Vite, no CSS framework — styling is inline with a dark neon theme. `useAudioRecorder` captures up to 30 s of `audio/webm` from the microphone via `MediaRecorder`, `WaveformVisualizer` draws the live level, and `GenreResult` renders the ranked genres, mood chips and feature pills. The Vite dev server proxies API calls to `localhost:8000`, and the backend's CORS allowlist covers ports 5173 and 3000.

`start.sh` scripts in both folders do the same setup steps in one command (bash/WSL/Git Bash).

---

## 6. Requirements

- **Python 3.10+** with `tensorflow`, `librosa`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `soundfile`, `scipy`
- **PyTorch + torchvision + transformers** for notebook Tasks 3 and 4
- **ffmpeg** on `PATH` (pydub needs it to decode mp3/m4a/webm uploads)
- **Node.js 18+** for the frontend
- A CUDA GPU is optional but strongly recommended: a full MFCC training run is ~100 epochs × ~16 s/epoch on GPU.

---

## 7. Notes and known quirks

- `web/backend/requirements.txt` **does not list TensorFlow**, although `main.py` needs it to load the model — install it manually. The `torch`/`transformers` entries in that file are leftovers and are not imported by the service.
- `preprocess.py` and `preprocess_1.py` derive the genre name with `dirpath.split("/")`, which does not split Windows paths; the *label indices* still follow `os.walk`'s alphabetical order, but the stored `mapping` strings can come out as full paths on Windows. The web backend therefore hard-codes the alphabetical `GENRE_LABELS` list instead of reading the mapping.
- The intermediate feature dumps are large: `data.json` variants are 0.6–2 GB and `data.npz` is ~3 GB. They are regenerable and git-ignored.
- GTZAN's `jazz.00054.wav` is corrupt in most distributions; the loaders skip failures, which is why some counts show 999 tracks instead of 1 000.
- Model files at the root correspond to different feature sets — `best_genre_classifier.keras` (130 × 40 MFCC, the deployed one), `best_genre_classifier_1.keras` (130 × 13 MFCC), `best_model.keras` (130 × 128 mel). Loading a model with the wrong preprocessing will fail on the input shape.
