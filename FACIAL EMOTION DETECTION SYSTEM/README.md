# Facial Emotion Recognition System

## Overview

This project implements a real-time facial emotion recognition system using a hybrid CNN-BiLSTM neural network architecture. It provides a FastAPI backend that captures webcam images, detects faces, and classifies emotions into seven categories: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

## Architecture

### Backend (FastAPI)

The backend is built with FastAPI and exposes an API endpoint for emotion prediction:

- `/predict`: Captures an image from the webcam, processes it, and returns the detected emotion with confidence scores.

### Neural Network Model

The emotion detection model uses a hybrid architecture combining:

1. **Convolutional Neural Network (CNN)** layers for feature extraction:
   - Multiple Conv2D layers with increasing filter sizes (32 → 64 → 128)
   - BatchNormalization for training stability
   - MaxPooling2D for dimensionality reduction

2. **Bidirectional LSTM** layers for sequential feature learning:
   - Reshape operations to prepare CNN features for LSTM processing
   - Two LSTM layers (128 units and 64 units) for temporal dependencies

3. **Dense layers** for classification:
   - 200-unit dense layer with ReLU activation
   - Dropout (0.6) for regularization
   - 7-unit output layer with softmax activation for emotion probabilities

## Processing Pipeline

1. **Image Capture**:
   - The system captures frames from the webcam using OpenCV with platform-specific optimizations
   - Resolution is set to 1280×720 for better face detection

2. **Face Detection**:
   - Captured images are converted to grayscale
   - Haar Cascade Classifier detects faces in the image
   - The first detected face is extracted for emotion analysis

3. **Image Preprocessing**:
   - Face is resized to 48×48 pixels (model input size)
   - Pixel values are normalized to [0,1] range
   - Image is reshaped to match model input shape (batch, height, width, channels)

4. **Emotion Prediction**:
   - Preprocessed face image is passed through the neural network
   - The model outputs probability scores for 7 emotion classes
   - The emotion with highest probability is selected as the prediction

5. **Result Format**:
   - Primary emotion and confidence score
   - Probability distribution across all emotions

## Technical Implementation Details

### Platform Compatibility

The system includes platform-specific optimizations:
- Windows: Disables MSMF priority
- macOS: Uses AVFoundation backend for camera capture

### Model Training

The model is pre-trained on the FER2013 dataset and weights are loaded from `fer2013_bilstm_cnn.h5`.

### CORS Support

The API includes CORS middleware for cross-origin requests, enabling web clients to interact with the backend.

## Deployment

The FastAPI application runs on port 2000 and can be started directly by executing the script. 