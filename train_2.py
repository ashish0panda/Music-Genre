# ==============================
# train.py
# Memory-Optimized CNN for Genre Classification
# ==============================

import gc
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Dense,
    Dropout,
    GlobalAveragePooling2D
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

from tensorflow.keras.regularizers import l2


# ==============================
# GPU CHECK
# ==============================

gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print(f"Using GPU: {gpus[0]}")
else:
    print("Running on CPU")


NPZ_PATH = "data.npz"


# ==============================
# LOAD DATA
# ==============================

def load_data(data_path):
    print("Loading dataset into memory...")
    data = np.load(data_path)
    X = data['X']
    y = data['y']
    mapping = data['mapping']
    
    data.close() 
    return X, y, mapping


X, y, mapping = load_data(NPZ_PATH)


# ==============================
# MEMORY-SAFE SPLIT (No Duplication)
# ==============================
print("Splitting data without duplicating memory...")

# We split indices first, NOT the massive array
indices = np.arange(len(X))

train_idx, temp_idx, y_train, y_temp = train_test_split(
    indices, y, test_size=0.3, random_state=42, stratify=y
)

val_idx, test_idx, y_val, y_test = train_test_split(
    temp_idx, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# Now we extract the exact pieces we need
X_train = X[train_idx]
X_val = X[val_idx]
X_test = X[test_idx]

# DESTROY the original massive array to free up 2.5+ GB of RAM instantly
del X
del indices
del train_idx
del temp_idx
del val_idx
del test_idx
gc.collect()

print("Memory cleared successfully!")


# ==============================
# ADD CHANNEL DIMENSION
# ==============================

X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]
X_test = X_test[..., np.newaxis]

input_shape = (
    X_train.shape[1],
    X_train.shape[2],
    1
)

print("\nInput Shape:", input_shape)
print(f"Training samples: {X_train.shape[0]}")
print(f"Validation samples: {X_val.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}\n")


# ==============================
# IMPROVED CNN MODEL
# ==============================

num_classes = len(np.unique(y_train))

model = keras.Sequential([

    # GPU ON-THE-FLY NORMALIZATION
    # This completely replaces CPU normalization. The GPU will normalize 
    # the data automatically as it feeds it into the network!
    BatchNormalization(input_shape=input_shape),

    # BLOCK 1
    Conv2D(
        32,
        (3, 3),
        padding='same',
        kernel_regularizer=l2(0.001)
    ),
    BatchNormalization(),
    keras.layers.Activation('relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.2),

    # BLOCK 2
    Conv2D(
        64,
        (3, 3),
        padding='same',
        kernel_regularizer=l2(0.001)
    ),
    BatchNormalization(),
    keras.layers.Activation('relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    # BLOCK 3
    Conv2D(
        128,
        (3, 3),
        padding='same',
        kernel_regularizer=l2(0.001)
    ),
    BatchNormalization(),
    keras.layers.Activation('relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    # GLOBAL POOLING
    GlobalAveragePooling2D(),

    # DENSE
    Dense(
        128,
        activation='relu',
        kernel_regularizer=l2(0.001)
    ),
    Dropout(0.4),

    # OUTPUT
    Dense(
        num_classes,
        activation='softmax'
    )
])


# ==============================
# COMPILE MODEL
# ==============================

optimizer = keras.optimizers.Adam(learning_rate=0.0001)
loss = tf.keras.losses.SparseCategoricalCrossentropy()

model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=['accuracy']
)

model.summary()


# ==============================
# CALLBACKS
# ==============================

checkpoint = ModelCheckpoint(
    "best_model.keras",
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)


# ==============================
# TRAINING
# ==============================

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    batch_size=32,
    epochs=100,
    callbacks=[checkpoint, early_stop, lr_scheduler]
)


# ==============================
# FINAL TEST ACCURACY
# ==============================

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\n==============================")
print(f"Final Test Accuracy: {test_acc * 100:.2f}%")
print(f"==============================\n")


# ==============================
# PLOTTING & EVALUATION
# ==============================
print("Generating and saving evaluation plots...")

# 1. Loss (Error) Graph
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('loss_graph.png', bbox_inches='tight', dpi=300)
plt.close()
print("- Saved loss_graph.png")

# 2. Accuracy Graph
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('accuracy_graph.png', bbox_inches='tight', dpi=300)
plt.close()
print("- Saved accuracy_graph.png")

# 3. Confusion Matrix
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=mapping, yticklabels=mapping)
plt.title('Confusion Matrix on Test Data')
plt.ylabel('True Genre')
plt.xlabel('Predicted Genre')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
plt.close()
print("- Saved confusion_matrix.png")
print("\nAll done!")