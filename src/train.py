# src/train.py
# Complete training pipeline for Cats vs Dogs (0 = Cat, 1 = Dog).
# Run from the project root:  python src/train.py

import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save figures without opening a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split

try:
    import tensorflow as tf
except ImportError:
    raise SystemExit("TensorFlow is not installed. Run: pip install -r requirements.txt")

# ---------------- Settings ----------------
SEED = 42
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 5
FINE_TUNE_EPOCHS = 3
FINE_TUNE_LAYERS = 20

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = PROJECT_ROOT / "data" / "train"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_PATH = MODELS_DIR / "cats_vs_dogs_mobilenetv2.keras"

AUTOTUNE = tf.data.AUTOTUNE
CLASS_NAMES = ["Cat", "Dog"]

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def build_dataframe():
    # Scan data/train and infer labels from filenames (cat -> 0, dog -> 1).
    if not TRAIN_DIR.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {TRAIN_DIR}\n"
                                "Place your images in data/train/ (cat.0.jpg, dog.0.jpg, ...).")
    rows = []
    for p in sorted(TRAIN_DIR.iterdir()):
        if not p.is_file() or p.suffix.lower() not in (".jpg", ".jpeg"):
            continue
        name = p.name.lower()
        if name.startswith("cat"):
            rows.append((str(p), 0, "Cat"))
        elif name.startswith("dog"):
            rows.append((str(p), 1, "Dog"))
    df = pd.DataFrame(rows, columns=["filepath", "label", "class_name"])
    if df.empty:
        raise ValueError("No cat/dog images found in data/train/.")
    if (df["label"] == 0).sum() == 0:
        raise ValueError("No cat images found (files must start with 'cat').")
    if (df["label"] == 1).sum() == 0:
        raise ValueError("No dog images found (files must start with 'dog').")

    # Drop unreadable/corrupt images so training does not crash midway.
    bad = []
    for fp in df["filepath"]:
        try:
            with Image.open(fp) as im:
                im.verify()
        except Exception:
            bad.append(fp)
    if bad:
        print(f"Removing {len(bad)} corrupt image(s).")
        df = df[~df["filepath"].isin(bad)].reset_index(drop=True)
    return df


def load_image(filepath, label):
    image_bytes = tf.io.read_file(filepath)
    image = tf.image.decode_jpeg(image_bytes, channels=3)  # channels=3 -> RGB
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0             # normalize to 0-1
    return image, label


def make_dataset(dataframe, training):
    ds = tf.data.Dataset.from_tensor_slices(
        (dataframe["filepath"].values, dataframe["label"].values.astype("float32")))
    if training:
        ds = ds.shuffle(1000, seed=SEED)
    ds = ds.map(load_image, num_parallel_calls=AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)


def build_model():
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ], name="data_augmentation")  # only active during training

    pretrained = True
    try:
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
    except Exception as e:
        print("WARNING: could not download ImageNet weights (", e, ")")
        print("Falling back to random weights. Accuracy will be lower.")
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=IMG_SIZE + (3,), include_top=False, weights=None)
        pretrained = False
    base_model.trainable = not pretrained  # freeze only if pretrained

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = data_augmentation(inputs)
    x = tf.keras.layers.Rescaling(scale=2.0, offset=-1.0)(x)  # 0-1 -> -1..1 (MobileNetV2 format)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs, name="cats_vs_dogs_mobilenetv2")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model, base_model, pretrained


def plot_history(acc, val_acc, loss, val_loss, split_epoch):
    for name, train_vals, val_vals in [("accuracy", acc, val_acc), ("loss", loss, val_loss)]:
        plt.figure(figsize=(8, 5))
        epochs = range(1, len(train_vals) + 1)
        plt.plot(epochs, train_vals, marker="o", label=f"Training {name.title()}")
        plt.plot(epochs, val_vals, marker="o", label=f"Validation {name.title()}")
        if split_epoch:
            plt.axvline(split_epoch, color="gray", linestyle="--", label="Fine-tuning starts")
        plt.title(f"Training vs Validation {name.title()}")
        plt.xlabel("Epoch")
        plt.ylabel(name.title())
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(RESULTS_DIR / f"training_{name}.png", dpi=150, bbox_inches="tight")
        plt.close()


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "predictions").mkdir(parents=True, exist_ok=True)

    df = build_dataframe()
    print(f"Total: {len(df)} | Cats: {(df.label == 0).sum()} | Dogs: {(df.label == 1).sum()}")

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["label"])
    print(f"Train: {len(train_df)} | Validation: {len(val_df)}")
    train_ds = make_dataset(train_df, training=True)
    val_ds = make_dataset(val_df, training=False)

    model, base_model, pretrained = build_model()
    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2,
                                                  restore_best_weights=True)
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[early_stop])
    acc = list(history.history["accuracy"]); val_acc = list(history.history["val_accuracy"])
    loss = list(history.history["loss"]); val_loss_hist = list(history.history["val_loss"])
    split_epoch = None

    # Optional fine-tuning of the last layers (kept only if it improves val_loss)
    if pretrained:
        stage1_best = min(val_loss_hist)
        weights_before = model.get_weights()
        try:
            base_model.trainable = True
            for layer in base_model.layers[:-FINE_TUNE_LAYERS]:
                layer.trainable = False
            model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                          loss="binary_crossentropy", metrics=["accuracy"])
            early_stop_ft = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2,
                                                             restore_best_weights=True)
            ft = model.fit(train_ds, validation_data=val_ds, epochs=FINE_TUNE_EPOCHS,
                           callbacks=[early_stop_ft])
            if min(ft.history["val_loss"]) > stage1_best:
                print("Fine-tuning did not help. Restoring the original frozen model.")
                model.set_weights(weights_before)
            else:
                split_epoch = len(acc)
                acc += ft.history["accuracy"]; val_acc += ft.history["val_accuracy"]
                loss += ft.history["loss"]; val_loss_hist += ft.history["val_loss"]
        except Exception as e:
            print("Fine-tuning failed:", e, "-> using the original frozen model.")
            model.set_weights(weights_before)

    plot_history(acc, val_acc, loss, val_loss_hist, split_epoch)

    # ---------------- Evaluation ----------------
    val_loss, _ = model.evaluate(val_ds, verbose=0)
    y_prob = model.predict(val_ds, verbose=0).ravel()
    y_true = val_df["label"].values.astype(int)
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "val_loss": float(val_loss),
        "total_images": int(len(df)),
        "training_samples": int(len(train_df)),
        "validation_samples": int(len(val_df)),
    }
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0)
    print(report)
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    with open(RESULTS_DIR / "metrics.txt", "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
        f.write("\n" + report)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_xlabel("Predicted label"); ax.set_ylabel("Actual label")
    ax.set_title("Confusion Matrix (Validation Set)")
    fig.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    model.save(MODEL_PATH)
    print("\n" + "=" * 40 + "\nFINAL RESULTS\n" + "=" * 40)
    print(f"Dataset size: {len(df)}\nTraining samples: {len(train_df)}\nValidation samples: {len(val_df)}")
    print(f"Accuracy: {metrics['accuracy']:.4f}\nPrecision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}\nF1-score: {metrics['f1_score']:.4f}")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
