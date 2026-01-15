# Facial Emotion Detection System 🎭

## Overview

This project implements a **real-time facial emotion recognition system** using a **hybrid CNN–BiLSTM neural network architecture**.  
It provides a **FastAPI backend** that captures webcam images, detects faces, and classifies human emotions into seven categories:

**Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral**

The system performs real-time emotion inference and returns both the predicted emotion and confidence scores.

---

## 🧠 System Architecture

### Backend (FastAPI)

The backend is built using **FastAPI** and exposes an API endpoint for real-time emotion prediction.

#### API Endpoint

- **`/predict`**
  - Captures an image from the webcam
  - Detects a face in the frame
  - Processes the image
  - Returns the predicted emotion with confidence scores

---

## 🧬 Neural Network Model

The emotion detection model uses a **hybrid CNN–BiLSTM architecture**, combining spatial feature extraction with sequential learning.

### 🔹 Convolutional Neural Network (CNN)

Used for spatial feature extraction from facial images:

- Multiple `Conv2D` layers with increasing filters: **32 → 64 → 128**
- `BatchNormalization` for training stability
- `MaxPooling2D` for dimensionality reduction

---

### 🔹 Bidirectional LSTM (BiLSTM)

Used for sequential feature learning:

- CNN feature maps are reshaped before LSTM input
- Two LSTM layers:
  - 128 units
  - 64 units
- Captures temporal dependencies in extracted features

---

### 🔹 Fully Connected Layers

- Dense layer with **200 units** and ReLU activation
- `Dropout (0.6)` for regularization
- Output layer with **7 units** and softmax activation for emotion probabilities

---

## 🔄 Processing Pipeline

### 1️⃣ Image Capture
- Webcam frames captured using **OpenCV**
- Platform-specific camera optimizations applied
- Resolution set to **1280×720** for better face detection

---

### 2️⃣ Face Detection
- Frames converted to grayscale
- **Haar Cascade Classifier** used for face detection
- First detected face is extracted for emotion analysis

---

### 3️⃣ Image Preprocessing
- Face resized to **48×48 pixels** (model input size)
- Pixel values normalized to range **[0, 1]**
- Image reshaped to match model input format:

