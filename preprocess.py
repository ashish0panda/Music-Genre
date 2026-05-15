import os
import json
import librosa
import math
import numpy as np

# Update this path to where you extracted the GTZAN dataset
DATASET_PATH = "./data/genres_original"
JSON_PATH = "data.json"
SAMPLE_RATE = 22050
DURATION = 30 # Measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

def save_mfcc(dataset_path, json_path, n_mfcc=40, n_fft=2048, hop_length=512, num_segments=10):
    """Extracts MFCCs from GTZAN dataset and saves them into a JSON file."""
    
    # Dictionary to store data
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": []
    }

    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)

    # Loop through all genre sub-folders
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        if dirpath is not dataset_path:
            genre_label = dirpath.split("/")[-1]
            data["mapping"].append(genre_label)
            print(f"\nProcessing: {genre_label}")

            for f in filenames:
                if f.endswith('.wav'):
                    file_path = os.path.join(dirpath, f)
                    try:
                        signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                        
                        # Process segments extracting MFCCs
                        for s in range(num_segments):
                            start_sample = num_samples_per_segment * s
                            finish_sample = start_sample + num_samples_per_segment

                            mfcc = librosa.feature.mfcc(y=signal[start_sample:finish_sample], 
                                                        sr=sr, 
                                                        n_fft=n_fft, 
                                                        n_mfcc=n_mfcc, 
                                                        hop_length=hop_length)
                            mfcc = mfcc.T

                            # Store MFCC segment if it has expected length
                            if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                                data["mfcc"].append(mfcc.tolist())
                                data["labels"].append(i-1)
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
        print(f"\nSuccessfully saved data to {json_path}")
        
# Run the extraction
if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH)