# ==============================
# preprocess_optimized.py
# Memory-Optimized Preprocessing Pipeline
# ==============================

import os
import librosa
import numpy as np

DATASET_PATH = "./data/genres_original"
NPZ_PATH = "data.npz" # Changed to .npz for efficient storage

SAMPLE_RATE = 22050
DURATION = 30
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

def save_mel_spectrogram(
    dataset_path,
    npz_path,
    n_mels=128,
    n_fft=2048,
    hop_length=512,
    num_segments=10
):

    mapping = []
    mel_spectrograms = []
    labels = []

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_vectors_per_segment = int(np.ceil(samples_per_segment / hop_length))

    print("Processing dataset...")

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        if dirpath != dataset_path:
            semantic_label = dirpath.split("/")[-1]
            mapping.append(semantic_label)
            print(f"\nProcessing: {semantic_label}")

            for f in filenames:
                if not f.endswith(".wav"):
                    continue

                file_path = os.path.join(dirpath, f)

                try:
                    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                    # ==============================
                    # DATA AUGMENTATION
                    # ==============================
                    augmented_signals = [
                        signal,                                              # Original
                        signal + 0.005 * np.random.randn(len(signal)),       # Noise Injection
                        librosa.effects.pitch_shift(signal, sr=sr, n_steps=2), # Pitch Shift
                        librosa.effects.time_stretch(signal, rate=0.9)       # Time Stretch
                    ]

                    # ==============================
                    # PROCESS EACH AUGMENTED SIGNAL
                    # ==============================
                    for aug_signal in augmented_signals:
                        for s in range(num_segments):
                            start = samples_per_segment * s
                            finish = start + samples_per_segment
                            segment = aug_signal[start:finish]

                            # Skip short segments
                            if len(segment) < samples_per_segment:
                                continue

                            mel = librosa.feature.melspectrogram(
                                y=segment, sr=sr, n_fft=n_fft,
                                hop_length=hop_length, n_mels=n_mels
                            )

                            mel_db = librosa.power_to_db(mel, ref=np.max).T

                            if len(mel_db) == expected_num_vectors_per_segment:
                                # Keep as numpy arrays, DO NOT use .tolist()
                                mel_spectrograms.append(mel_db)
                                labels.append(i - 1)

                except Exception as e:
                    print(f"Error processing {file_path}")
                    print(e)

    # Convert lists of arrays to a single numpy array
    X = np.array(mel_spectrograms)
    y = np.array(labels)
    mapping_arr = np.array(mapping)
    
    print(f"\nShape of X: {X.shape}")
    print(f"Shape of y: {y.shape}")

    # Save as compressed NumPy array (.npz)
    np.savez_compressed(npz_path, X=X, y=y, mapping=mapping_arr)
    print(f"\nSaved successfully to {npz_path}")

if __name__ == "__main__":
    save_mel_spectrogram(DATASET_PATH, NPZ_PATH)