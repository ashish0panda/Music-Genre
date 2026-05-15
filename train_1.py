import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import os

# Quick GPU Check
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"Hardware accelerated with: {gpus[0]}")
else:
    print("Warning: Running on CPU. Training will be slow.")

JSON_PATH = "data.json"

GENRE_LABELS = [
    "blues", "classical", "country", "disco",
    "hiphop", "jazz", "metal", "pop", "reggae", "rock"
]

PLOTS_DIR = "training_plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


def load_data(data_path):
    with open(data_path, "r") as fp:
        data = json.load(fp)
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    return X, y


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_loss(history):
    """Training loss vs Validation loss."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history.history["loss"],     label="Train Loss",      linewidth=2)
    ax.plot(history.history["val_loss"], label="Validation Loss", linewidth=2, linestyle="--")
    ax.set_title("Training vs Validation Loss", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "loss_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_accuracy(history):
    """Training accuracy vs Validation accuracy."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history.history["accuracy"],     label="Train Accuracy",      linewidth=2)
    ax.plot(history.history["val_accuracy"], label="Validation Accuracy", linewidth=2, linestyle="--")
    ax.set_title("Training vs Validation Accuracy", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "accuracy_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_confusion_matrix(model, X_test, y_test):
    """Normalized confusion matrix."""
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=GENRE_LABELS, yticklabels=GENRE_LABELS,
        linewidths=0.5, ax=ax, vmin=0, vmax=1
    )
    ax.set_title("Confusion Matrix (Normalized)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Genre", fontsize=11)
    ax.set_ylabel("True Genre", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")
    return y_pred


def plot_per_class_accuracy(y_test, y_pred):
    """Bar chart of per-class accuracy."""
    cm = confusion_matrix(y_test, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(GENRE_LABELS, per_class_acc, color=plt.cm.tab10.colors, edgecolor="black")
    ax.set_title("Per-Class Accuracy", fontsize=14, fontweight="bold")
    ax.set_xlabel("Genre")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.axhline(per_class_acc.mean(), color="red", linestyle="--", label=f"Mean: {per_class_acc.mean():.2%}")
    ax.legend()
    for bar, val in zip(bars, per_class_acc):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                f"{val:.0%}", ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "per_class_accuracy.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_learning_rate(history):
    """Learning rate over epochs (captured via history if available)."""
    if "lr" not in history.history:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history.history["lr"], linewidth=2, color="darkorange")
    ax.set_title("Learning Rate Schedule", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "learning_rate.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_loss_and_accuracy_combined(history):
    """Single figure with both curves side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    ax1.plot(history.history["loss"],     label="Train",      linewidth=2)
    ax1.plot(history.history["val_loss"], label="Validation", linewidth=2, linestyle="--")
    ax1.set_title("Loss", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    ax2.plot(history.history["accuracy"],     label="Train",      linewidth=2)
    ax2.plot(history.history["val_accuracy"], label="Validation", linewidth=2, linestyle="--")
    ax2.set_title("Accuracy", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1)
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax2.legend(); ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.suptitle("Training Overview", fontsize=15, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "training_overview.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Data + model (same as original)
# ---------------------------------------------------------------------------

print("Loading data...")
X, y = load_data(JSON_PATH)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train = X_train[..., np.newaxis]
X_test  = X_test[..., np.newaxis]

input_shape = (X_train.shape[1], X_train.shape[2], 1)

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
    Dropout(0.4),
    Dense(10, activation='softmax')
])

optimizer = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

if __name__ == "__main__":
    checkpoint = ModelCheckpoint(
        filepath="best_genre_classifier.keras",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max"
    )

    early_stop = EarlyStopping(
        monitor="val_accuracy",
        patience=15,
        restore_best_weights=True
    )

    lr_scheduler = ReduceLROnPlateau(
        monitor="val_accuracy",
        factor=0.5,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        batch_size=32,
        epochs=150,
        callbacks=[checkpoint, early_stop, lr_scheduler]
    )

    # -----------------------------------------------------------------------
    # Save all visualizations
    # -----------------------------------------------------------------------
    print(f"\nSaving plots to '{PLOTS_DIR}/'...")

    plot_loss(history)                               # loss_curve.png
    plot_accuracy(history)                           # accuracy_curve.png
    plot_loss_and_accuracy_combined(history)         # training_overview.png
    plot_learning_rate(history)                      # learning_rate.png
    y_pred = plot_confusion_matrix(model, X_test, y_test)  # confusion_matrix.png
    plot_per_class_accuracy(y_test, y_pred)          # per_class_accuracy.png

    print("\nAll plots saved.")
    print(classification_report(y_test, y_pred, target_names=GENRE_LABELS))