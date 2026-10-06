# src/predict.py
# Predict Cat or Dog for one image.
# Usage (from the project root):  python src/predict.py path/to/image.jpg

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import tensorflow as tf
except ImportError:
    sys.exit("TensorFlow is not installed. Run: pip install -r requirements.txt")

IMG_SIZE = (128, 128)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "cats_vs_dogs_mobilenetv2.keras"


def predict(image_path, model):
    # Same preprocessing as training: RGB -> resize 128x128 -> scale to 0-1.
    img = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(img, dtype="float32") / 255.0
    prob_dog = float(model.predict(np.expand_dims(arr, 0), verbose=0)[0][0])
    label = "Dog" if prob_dog >= 0.5 else "Cat"
    confidence = prob_dog if label == "Dog" else 1 - prob_dog
    return label, confidence


def main():
    parser = argparse.ArgumentParser(description="Classify an image as Cat or Dog.")
    parser.add_argument("image_path", help="Path to a .jpg/.png image")
    args = parser.parse_args()

    if not Path(args.image_path).is_file():
        sys.exit(f"Image not found: {args.image_path}")
    if not MODEL_PATH.is_file():
        sys.exit(f"Model file not found: {MODEL_PATH}\nTrain first: python src/train.py")

    model = tf.keras.models.load_model(MODEL_PATH)
    label, confidence = predict(args.image_path, model)
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence * 100:.2f}%")


if __name__ == "__main__":
    main()
