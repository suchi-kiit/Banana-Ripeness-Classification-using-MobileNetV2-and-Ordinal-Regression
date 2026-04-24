# Banana Ripeness Classification

A deep learning project for classifying banana ripeness stages using MobileNetV2 with a custom ordinal regression approach and LAB color space preprocessing.

## Overview

This project trains a convolutional neural network to classify bananas into ripeness stages (unripe, ripe, overripe) from images. It uses ordinal encoding to preserve the natural ordering between ripeness stages and applies a custom LAB color space transformation layer to improve color robustness.

## Features

- MobileNetV2 backbone with transfer learning
- Custom RGB-to-LAB color space preprocessing layer
- Ordinal regression output with a custom loss function
- Grad-CAM and Score-CAM visualizations for model explainability
- Lighting robustness evaluation (dark, bright, warm)
- Ablation study across image perturbations (dark, bright, blur)
- Cosine similarity and SSIM analysis across ripeness classes

## Dataset

The dataset is sourced from [Roboflow Universe](https://universe.roboflow.com/roboflow-universe-projects/banana-ripeness-classification/dataset/6).

To download it, you need a Roboflow API key (see Setup below).

## Setup

### 1. Get a Roboflow API Key

1. Go to [https://roboflow.com](https://roboflow.com) and create a free account.
2. Navigate to your workspace settings to find your API key.
3. Replace `"your-api-key"` in the script with your actual key:

```python
rf = Roboflow(api_key="your-api-key")
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run in Google Colab

This project is designed for Google Colab with Google Drive mounted for saving model checkpoints. Open `banana_ripeness_classification.py` in Colab and run all cells.

The trained model is saved to:
```
/content/drive/MyDrive/best_mobilenetv2_run3.keras
```

## Model Architecture

- Input: 224x224 RGB image
- Preprocessing: RGB to LAB color space conversion + noise augmentation
- Backbone: MobileNetV2 (ImageNet weights, last 50 layers unfrozen)
- Head: GlobalAveragePooling2D → Dense(128, ReLU) → Dropout(0.3) → Dense(3, Sigmoid)
- Output: Ordinal vector of length 3, decoded to class index via threshold (0.5)

## Ordinal Encoding

Classes are encoded as cumulative binary vectors:

| Class | Encoding |
|-------|----------|
| Unripe | [0, 0, 0] |
| Ripe | [1, 0, 0] |
| Overripe | [1, 1, 0] |

The predicted class index is the sum of values above the 0.5 threshold.

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | ~98% |
| Adjacent Accuracy | ~100% |
| MARE | ~0.02 |

Lighting robustness results will vary based on your dataset split.

## Visualizations

Grad-CAM and Score-CAM are available for any uploaded image. In the final cells of the script, you will be prompted to upload an image via the Colab file upload dialog, and both heatmaps will be displayed.

## Project Structure

```
banana_ripeness_classification.py   Main training and evaluation script
requirements.txt                    Python dependencies
README.md                           Project documentation
```
