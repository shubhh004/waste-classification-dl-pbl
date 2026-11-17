# Intelligent Waste Classification Using Deep Learning

CSE0711 Deep Learning - Individual PBL
Shubh Chaurasia (BETN1CS23074), ITM University, Gwalior

## Overview
Classifies waste images into 6 classes (cardboard, glass, metal, paper, plastic, trash)
using a custom CNN and MobileNetV2 transfer learning, with Grad-CAM for explainability.

## Dataset
TrashNet (Kaggle: "Garbage Classification"), 2527 images, 6 classes.
Split: 70% train / 15% validation / 15% test (stratified, seed 42).

## Results
| Model | Test Accuracy | Macro F1 |
|---|---|---|
| Custom CNN | 50.26% | 47.34 |
| MobileNetV2 (frozen) | 87.63% | 85.81 |
| MobileNetV2 (fine-tuned) | 89.21% | 87.56 |

Results vary slightly between training runs because of random initialisation and augmentation.

## Files
- Waste_Classification_PBL.ipynb : full code and outputs
- *.png : training curves, confusion matrices, Grad-CAM and error analysis figures
- model_comparison.csv : metrics of all three models

## How to run
1. Open the notebook in Google Colab and enable a T4 GPU.
2. Upload the dataset zip and extract it to /content/data.
3. Run all cells.
