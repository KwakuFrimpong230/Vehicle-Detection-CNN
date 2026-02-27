# 🚗 vehicle-detection-cnn

> A lightweight dual-head CNN that classifies vehicle types and predicts bounding boxes from traffic surveillance images using TensorFlow.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13.1-orange?style=flat-square&logo=tensorflow)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8.1-green?style=flat-square&logo=opencv)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Demo](#-demo)
- [Features](#-features)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Limitations & Future Work](#-limitations--future-work)
- [License](#-license)

---

## 🔍 Overview

This project implements a **multi-task deep learning model** for vehicle detection in traffic surveillance imagery. A single lightweight CNN simultaneously solves two tasks:

| Task | Output | Loss |
|------|--------|------|
| **Vehicle Classification** | 11-class softmax probability vector | Categorical Cross-Entropy |
| **Bounding Box Localization** | `[xmin, ymin, xmax, ymax]` normalized to `[0, 1]` | Mean Squared Error |

The model is designed to run in **CPU-only, memory-constrained environments** (~1 GB RAM), making it suitable for edge deployment without GPU hardware.

---

## 🎬 Demo

```
Input Image (96×96)
        │
        ▼
┌──────────────────────┐
│  LightweightVehicle  │──► Class: car (89% confidence)
│      Detector        │──► BBox: [0.31, 0.22, 0.74, 0.65]  IoU: 0.61
└──────────────────────┘
     181,807 params
       0.73 MB
```

Sample inference output — green box = ground truth, red dashed = predicted:

```
GT: car        | Pred: car (95%)   | IoU: 0.72  ✅
GT: bus        | Pred: bus (88%)   | IoU: 0.63  ✅
GT: pickup_truck | Pred: car (71%) | IoU: 0.31  ⚠️
```

---

## ✨ Features

- 🏗️ **Dual-head architecture** — single forward pass for both classification and localization
- ⚡ **Ultra-lightweight** — 181,807 parameters, ~0.73 MB model size
- 🔧 **CPU-friendly** — trains and infers without GPU hardware
- 📊 **Comprehensive evaluation** — accuracy, IoU, MAE, confusion matrix, per-class F1
- 💾 **SavedModel export** — ready for TensorFlow Lite conversion
- 📈 **Training callbacks** — EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

---

## 📦 Dataset

| Property | Value |
|----------|-------|
| Total images | 110,000 |
| Total annotations | 351,549 |
| Classes | 11 vehicle types |
| Annotation format | `image_id, label, xmin, ymin, xmax, ymax` |
| Training subset used | 5,000 images (memory constraint) |

### Vehicle Classes

```
car · pickup_truck · motorized_vehicle · bus · articulated_truck
work_van · pedestrian · single_unit_truck · non-motorized_vehicle
bicycle · motorcycle
```

> ⚠️ The dataset is **heavily imbalanced** — `car` accounts for **66.4%** of all annotations.

---

## 🧠 Model Architecture

```
Input (96×96×3)
    │
    ▼
Conv Block 1:    Conv2D(32) → BatchNorm → ReLU → MaxPool   [96→48]
    │
    ▼
Sep Block 2:     SeparableConv2D(64) → BatchNorm → MaxPool  [48→24]
    │
    ▼
Sep Block 3:     SeparableConv2D(128) → BatchNorm → MaxPool [24→12]
    │
    ▼
Sep Block 4:     SeparableConv2D(256) → BatchNorm           [12×12]
    │
    ▼
Shared:          GlobalAveragePooling2D → Dense(256) → Dropout(0.4)
    │
    ├──────────────────────┬──────────────────────┐
    ▼                      ▼
Head 1 (Classification)    Head 2 (BBox Regression)
Dense(128) → Dropout(0.3)  Dense(128) → Dropout(0.3)
Dense(11, softmax)         Dense(4, sigmoid)
```

**Why SeparableConv2D?**  
Depthwise-separable convolutions factorize a standard convolution into a depthwise spatial filter + a pointwise 1×1 projection, reducing parameters by ~8–9× compared to standard Conv2D at equivalent filter counts.

**Total Parameters:** 181,807 (180,847 trainable + 960 BatchNorm statistics)

---

## 📊 Results

### Test Set Performance (N = 750)

| Metric | Value |
|--------|-------|
| Classification Accuracy | **70.1%** |
| Weighted F1-Score | 0.67 |
| Bounding Box MAE | 0.1019 |
| Mean IoU (mIoU) | **0.254** |
| IoU ≥ 0.50 (standard) | 15.2% |
| IoU ≥ 0.25 (lenient) | 47.2% |

### Per-Class F1 Scores

| Class | F1 | Support |
|-------|----|---------|
| Bus | **0.84** | 60 |
| Car | **0.80** | 439 |
| Bicycle | 0.75 | 5 |
| Articulated Truck | 0.59 | 35 |
| Pickup Truck | 0.47 | 140 |
| Motorized Vehicle | 0.26 | 22 |
| Motorcycle | 0.00 | 4 |
| Work Van | 0.00 | 29 |
| Single Unit Truck | 0.00 | 12 |
| Non-Motorized Vehicle | 0.00 | 4 |

> Minority classes (motorcycle, work van, single unit truck, non-motorized vehicle) achieve zero recall due to severe class imbalance in the training subset.

### Training Summary

- **Optimizer:** Adam (lr=1e-3, with ReduceLROnPlateau)
- **Epochs trained:** 29 (EarlyStopping, best weights from epoch 19)
- **Loss weights:** Classification × 1.0, BBox MSE × 5.0
- **Final val loss:** 1.041 | **Final val accuracy:** 70.1%

---

## 📁 Project Structure

```
vehicle-detection-cnn/
│
├── Object_Detection_Organized.ipynb   # Main notebook (full pipeline)
│
├── vehicle_detection_project/
│   ├── data/
│   │   └── images/
│   │       └── Images/                # Extracted .jpg files
│   ├── models/
│   │   ├── best_model.h5              # Best checkpoint weights
│   │   ├── vehicle_detector_final/    # TF SavedModel format
│   │   └── class_names.json           # Class index → label mapping
│   └── results/
│       ├── 01_class_distribution.png
│       ├── 02_sample_images_gt.png
│       ├── 03_training_history.png
│       ├── 04_confusion_matrix.png
│       └── 05_iou_distribution.png
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.8+
- pip

### Clone the Repository

```bash
git clone https://github.com/<your-username>/vehicle-detection-cnn.git
cd vehicle-detection-cnn
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt`**
```
tensorflow==2.13.1
numpy==1.23.5
opencv-python==4.8.1.78
pandas
matplotlib
seaborn
scikit-learn
```

---

## 🚀 Usage

### Running the Full Notebook

```bash
jupyter notebook Object_Detection_Organized.ipynb
```

### Inference on a New Image

```python
import tensorflow as tf
import numpy as np
import cv2
import json

# Load model and class names
model = tf.saved_model.load('vehicle_detection_project/models/vehicle_detector_final')
with open('vehicle_detection_project/models/class_names.json') as f:
    class_names = json.load(f)

# Preprocess image
img = cv2.imread('your_image.jpg')
img = cv2.resize(img, (96, 96))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = (img / 255.0).astype(np.float32)
img = np.expand_dims(img, axis=0)  # Add batch dimension

# Run inference
cls_probs, bbox = model(img)
predicted_class = class_names[str(np.argmax(cls_probs))]
confidence = float(np.max(cls_probs))
xmin, ymin, xmax, ymax = bbox[0].numpy()

print(f"Class:      {predicted_class} ({confidence:.1%})")
print(f"Bounding Box: [{xmin:.3f}, {ymin:.3f}, {xmax:.3f}, {ymax:.3f}]")
```

---

## 🏋️ Training

### Data Setup

1. Place `Images.zip` at `/voc/work/Images.zip` (or update `ZIP_PATH` in the notebook)
2. Place `labels.csv` at `/voc/work/labels.csv` (or update `LABELS_CSV`)

### Key Configuration (Cell 32)

```python
IMG_SIZE    = 96      # Input resolution (increase for better bbox quality)
MAX_IMAGES  = 5000    # Subsample cap (increase if RAM allows)
BATCH_SIZE  = 32
EPOCHS      = 50      # EarlyStopping will terminate earlier
```

### Loss Weights

```python
loss_weights = {
    'class_output': 1.0,
    'bbox_output':  5.0   # Higher weight needed — bbox MSE values are small
}
```

### Expected Training Time

| Hardware | Approx. Time per Epoch | Total (29 epochs) |
|----------|------------------------|-------------------|
| CPU only | ~24 seconds | ~12 minutes |
| GPU (T4)  | ~3 seconds  | ~1.5 minutes |

---

## 📉 Evaluation

The notebook generates the following evaluation artifacts automatically:

| File | Description |
|------|-------------|
| `01_class_distribution.png` | Bar + pie chart of annotation counts |
| `02_sample_images_gt.png` | 10 sample images with ground truth boxes |
| `03_training_history.png` | Accuracy, total loss, and BBox MAE curves |
| `04_confusion_matrix.png` | 10×10 heatmap on test set |
| `05_iou_distribution.png` | Histogram of IoU scores with threshold markers |

### IoU Computation

```python
def compute_iou(true_box, pred_box):
    ix1 = max(true_box[0], pred_box[0])
    iy1 = max(true_box[1], pred_box[1])
    ix2 = min(true_box[2], pred_box[2])
    iy2 = min(true_box[3], pred_box[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area1 = (true_box[2] - true_box[0]) * (true_box[3] - true_box[1])
    area2 = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0
```

---

## ⚠️ Limitations & Future Work

### Current Limitations

- **Class imbalance** — `car` dominates; minority classes (motorcycle, work van) achieve 0% recall
- **Single-box prediction** — only the largest bounding box per image is targeted; multi-object scenes are not handled
- **Low input resolution** — 96×96 limits fine-grained feature learning
- **CPU-only training** — constrains dataset size and model capacity

### Planned Improvements

- [ ] Apply **focal loss** or **class-weighted cross-entropy** for minority class recall
- [ ] Replace MSE with **GIoU / DIoU loss** for better localization
- [ ] Implement **multi-scale anchor-based detection** (SSD-style)
- [ ] Fine-tune from **MobileNetV2** ImageNet weights via transfer learning
- [ ] Train at **224×224** resolution with GPU
- [ ] Add **data augmentation** (flips, brightness jitter, mosaic)
- [ ] Export to **TensorFlow Lite** for embedded deployment

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">
  <sub>Built as part of a Deep Learning for Computer Vision capstone project · February 2026</sub>
</div>
