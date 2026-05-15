import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf

# Constants
JSON_PATH = "data.json"
MODEL_PATH = "best_genre_classifier.keras"
GENRES = ["blues", "classical", "country", "disco", "hiphop", 
          "jazz", "metal", "pop", "reggae", "rock"]

def load_test_data(data_path):
    """Loads the dataset and recreates the exact same test split used in training."""
    with open(data_path, "r") as fp:
        data = json.load(fp)
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    
    # Must use the exact same random_state=42 to get the same test data!
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Add channel dimension
    X_test = X_test[..., np.newaxis]
    return X_test, y_test

print("Loading test data and model...")
X_test, y_test = load_test_data(JSON_PATH)
model = tf.keras.models.load_model(MODEL_PATH)

print("\nGenerating predictions...")
y_pred_probabilities = model.predict(X_test)
y_pred = np.argmax(y_pred_probabilities, axis=1)

# ==========================================
# 1. GENERATE CONFUSION MATRIX GRAPH
# ==========================================
print("\nPlotting Confusion Matrix...")
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=GENRES, yticklabels=GENRES)
plt.title('Genre Classification Confusion Matrix', fontsize=16)
plt.ylabel('Actual Genre', fontsize=12)
plt.xlabel('Predicted Genre', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
print("Saved -> 'confusion_matrix.png'")

# ==========================================
# 2. GENERATE CLASS-WISE ACCURACY REPORT
# ==========================================
print("\nCalculating Class-wise Accuracy...")
report = classification_report(y_test, y_pred, target_names=GENRES)

# Print to terminal
print("\n--- Classification Report ---")
print(report)

# Save to a text file for the report
with open("class_accuracy_report.txt", "w") as f:
    f.write("Model Classification Report\n")
    f.write("===========================\n\n")
    f.write(report)
print("Saved -> 'class_accuracy_report.txt'")

# ==========================================
# 3. GENERATE MODEL WEIGHTS DISTRIBUTION GRAPH
# ==========================================
# This looks highly technical and is great for academic/engineering reports!
print("\nExtracting Conv2D Layer Weights...")

# Loop through every layer in the model
for layer in model.layers:
    # Check if the current layer is a Convolutional 2D layer
    if isinstance(layer, tf.keras.layers.Conv2D):
        layer_name = layer.name
        print(f"Processing weights for: {layer_name}...")
        
        # Extract weights (get_weights returns [weights, biases], so we grab index 0)
        weights = layer.get_weights()[0]
        
        # Flatten the multi-dimensional weight matrix into a 1D array for the histogram
        flattened_weights = weights.flatten()

        # Create the plot
        plt.figure(figsize=(8, 5))
        plt.hist(flattened_weights, bins=50, color='#835af1', alpha=0.75, edgecolor='black')
        plt.title(f'Distribution of Weights - {layer_name}', fontsize=14)
        plt.xlabel('Weight Value')
        plt.ylabel('Frequency')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        # Save the file dynamically using the layer's actual name
        filename = f'weights_distribution_{layer_name}.png'
        plt.savefig(filename, dpi=300)
        plt.close() # Close the plot to free memory before starting the next loop
        
        print(f"Saved -> '{filename}'")

print("\nAll visualizations complete! Check your folder for the new files.")