# Audio Deepfake Detection System

## Project Description
This repository contains an implementation of deep learning models for detecting synthetic voice manipulations, developed as part of Momenta's assessment process. The system analyzes audio samples to distinguish between genuine human speech and AI-generated deepfakes.

## Key Features
- Implements multiple state-of-the-art detection approaches
- Supports both spectrogram-based and raw waveform analysis
- Includes data preprocessing pipelines for common audio datasets
- Modular architecture for easy experimentation

## Installation

### Requirements
- Python 3.8 or higher
- PyTorch 1.10+
- Librosa
- NVIDIA GPU (recommended for training)

### Setup
```bash
git clone https://github.com/yourusername/audio-deepfake-detection.git
cd audio-deepfake-detection
pip install -r requirements.txt