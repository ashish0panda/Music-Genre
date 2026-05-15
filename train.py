import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Quick GPU Check
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"Hardware accelerated with: {gpus[0]}")
else:
    print("Warning: Running on CPU. Training will be slow.")

JSON_PATH = "data.json"

def load_data(data_path):
    with open(data_path, "r") as fp:
        data = json.load(fp)
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    return X, y

print("Loading data...")
X, y = load_data(JSON_PATH)

# Train/Test Split (70% Training, 30% Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Add a channel dimension for the CNN
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

input_shape = (X_train.shape[1], X_train.shape[2], 1)

# Build the Modified CNN Model
model = keras.Sequential([
    Conv2D(256, (3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D((3, 3), strides=(2, 2), padding='same'),
    BatchNormalization(),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((3, 3), strides=(2, 2), padding='same'),
    BatchNormalization(),

    Conv2D(64, (2, 2), activation='relu'),
    MaxPooling2D((2, 2), strides=(2, 2), padding='same'),
    BatchNormalization(),

    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.4), # <-- Increased to 0.4 for better generalization
    Dense(10, activation='softmax')
])


optimizer = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

if __name__ == "__main__":
   # 2. Set up the dynamic callbacks
    checkpoint = ModelCheckpoint(
        filepath="best_genre_classifier.keras", 
        monitor="val_accuracy", 
        save_best_only=True, 
        mode="max"
    )
    
    # Stops training early if validation accuracy doesn't improve for 15 epochs
    early_stop = EarlyStopping(
        monitor="val_accuracy",
        patience=15,
        restore_best_weights=True
    )
    
    # Drops the learning rate if validation accuracy plateaus for 5 epochs
    lr_scheduler = ReduceLROnPlateau(
        monitor="val_accuracy",
        factor=0.5,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )

    # 3. Pass all callbacks to the fit function
    history = model.fit(
        X_train, y_train, 
        validation_data=(X_test, y_test), 
        batch_size=32, 
        epochs=150, # Increased to 150, but EarlyStopping will likely cut it off sooner
        callbacks=[checkpoint, early_stop, lr_scheduler]
    )