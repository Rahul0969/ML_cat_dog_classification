# Cats vs Dogs Image Classification

## Problem Statement
Given a photograph, decide whether it shows a **cat** or a **dog**. This is a binary image-classification problem.

## Objective
Build a complete, reproducible deep-learning pipeline that loads the data efficiently, trains a CNN (MobileNetV2 transfer learning), evaluates it with standard metrics, and predicts on new images.

## Dataset
About 25,000 labeled JPEG images in one folder, `data/train/`, named `cat.0.jpg`, `cat.1.jpg`, ..., `dog.0.jpg`, `dog.1.jpg`, ...
There are no class subfolders, so labels are inferred from filenames: `cat*` → 0 (Cat), `dog*` → 1 (Dog).

## Technologies
- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Pillow
- Scikit-learn

## Methodology
```
Dataset
  ↓
DataFrame (filepath, label, class_name)
  ↓
Train/Validation Split (80/20, stratified, seed 42)
  ↓
Preprocessing (decode JPEG → RGB → resize 128×128 → normalize 0–1)
  ↓
Data Augmentation (flip, rotation, zoom; training only)
  ↓
MobileNetV2 (frozen, then optional fine-tuning)
  ↓
Training (EarlyStopping on val_loss)
  ↓
Evaluation
  ↓
Prediction
```
Images are streamed in batches of 32 with a `tf.data` pipeline (parallel map + prefetch), so the whole dataset is never held in RAM.

## Evaluation Metrics
- **Accuracy** – share of all predictions that are correct.
- **Precision** – of images predicted Dog, how many really are dogs.
- **Recall** – of all real dogs, how many were found.
- **F1-score** – harmonic mean of precision and recall.
- **Confusion matrix** – table of actual vs. predicted classes.

Results (generated after training and saved in `results/metrics.json`):

| Metric | Value |
|---|---|
| Accuracy | Generated after training |
| Precision | Generated after training |
| Recall | Generated after training |
| F1-score | Generated after training |

## Project Structure
```
ML_1_CatsVsDogs/
├── data/
│   └── train/               # cat.0.jpg ... dog.0.jpg ...
├── notebooks/
│   └── cats_vs_dogs_classification.ipynb
├── models/
│   └── cats_vs_dogs_mobilenetv2.keras   # created after training
├── results/
│   ├── class_distribution.png
│   ├── sample_images.png
│   ├── training_accuracy.png
│   ├── training_loss.png
│   ├── confusion_matrix.png
│   ├── metrics.json
│   ├── metrics.txt
│   └── predictions/
│       ├── predictions.png
│       └── predictions.csv
├── src/
│   ├── train.py
│   └── predict.py
├── README.md
├── TASK1_SUMMARY.md
└── requirements.txt
```

## Installation
```
pip install -r requirements.txt
```

## Running
1. Put the images in `data/train/`.
2. In a terminal: `cd ML_1_CatsVsDogs`, then `jupyter notebook`. Chrome opens automatically.
3. Open `notebooks/cats_vs_dogs_classification.ipynb`.
4. Choose **Kernel → Restart & Run All**, or run cells one at a time with **Shift + Enter**.

Alternatively, train from the terminal: `python src/train.py`

## Prediction
Inside the notebook, call:
```python
predict_image("../data/train/dog.1.jpg")
```
From the terminal (after training):
```
python src/predict.py path/to/image.jpg
```
Output:
```
Prediction: Dog
Confidence: 94.20%
```
