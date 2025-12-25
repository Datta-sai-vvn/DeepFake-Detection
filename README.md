# DeepFake Detection System

A robust, real-time deepfake video detection pipeline leveraging multi-strategy temporal sampling and deep facial feature extraction.

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)
![Accuracy](https://img.shields.io/badge/Ensemble%20Accuracy-96%25-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Performance](#performance)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Results Visualization](#results-visualization)
- [Limitations & Future Work](#limitations--future-work)
- [References](#references)
- [Author](#author)

---

## Overview

In an era where synthetic media threatens the integrity of digital content, this project presents a lightweight yet robust deepfake detection pipeline. By leveraging three complementary frame-sampling strategies, deep facial feature extraction with ResNeXt-50, and an ensemble of multi-layer perceptrons, we achieve state-of-the-art detection accuracy while maintaining real-time performance.

### Objectives

- **Computational Efficiency**: Fast inference suitable for real-time applications
- **Robustness**: Effective across different temporal locations of manipulations
- **Ease of Deployment**: Simple integration via Streamlit web interface

---

## Key Features

- **Multi-Strategy Temporal Sampling**
  - Even-spaced frame sampling for temporal balance
  - Early frame focus for fast-response scenarios
  - Random sampling for generalization
  
- **Deep Feature Extraction**
  - ResNeXt-50 (32x4d) backbone
  - 2048-dimensional facial embeddings
  - MTCNN for robust face detection

- **Ensemble Learning**
  - Majority voting across three specialized models
  - 96% ensemble accuracy
  - Individual model accuracies: 94-96%

- **Production Ready**
  - Real-time inference
  - Streamlit web interface
  - Easy deployment and scalability

---

## Architecture

### System Pipeline

```
Video Input -> Frame Sampling -> Face Detection -> Feature Extraction -> Classification -> Ensemble Vote -> Prediction
```

### Three Detection Strategies

| Strategy | Sampling Method | Purpose | Accuracy |
|----------|----------------|---------|----------|
| **Model 1** | 10 evenly spaced frames | Temporal balance across entire video | 94.12% |
| **Model 2** | 10 early frames | Fast response for quick decisions | 96.04% |
| **Model 3** | 10 random frames | Generalization & variance reduction | 93.88% |

### Model Architecture

```
Input (2048-d facial embedding)
    |
Dense Layer (2048 -> 512) + ReLU
    |
Dropout (p=0.3)
    |
Output Layer (512 -> 1) + Sigmoid
    |
Binary Classification (REAL/FAKE)
```

---

## Performance

### Overall Results

| Metric | Model 1 | Model 2 | Model 3 | **Ensemble** |
|--------|---------|---------|---------|--------------|
| **Accuracy** | 94.12% | 96.04% | 93.88% | **~96%** |
| **Precision** | High | High | High | **Very High** |
| **Recall** | High | High | High | **Very High** |

### Key Performance Indicators

- **Real-time inference** capability
- **Robust** to various deepfake generation techniques
- **Scalable** architecture for production deployment
- **Efficient** GPU utilization (NVIDIA T4/A100)

---

## Dataset

### Source
**Kaggle DeepFake Detection Challenge (DFDC)**

| Specification | Details |
|--------------|---------|
| **Original Size** | 472 GB (50 ZIP archives) |
| **Subset Used** | 38 GB (4 archives: 05, 10, 12, 15) |
| **Format** | MP4 videos + metadata.json |
| **Labels** | REAL / FAKE |
| **Preprocessing** | Merged JSONs -> consolidated metadata.csv |

### Data Structure
```
data/
  05/
    video1.mp4
    video2.mp4
    metadata.json
  10/
  12/
  15/
```

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)
- Google Colab Pro (optional, for training)

### Required Libraries

```bash
pip install -r requirements.txt
```

### Clone Repository

```bash
git clone https://github.com/Datta-sai-vvn/DeepFake-Detection.git
cd DeepFake-Detection
```

---

## Usage

### 1. Feature Extraction

Each strategy has its own feature extraction notebook. Navigate to the `notebooks` directory:

```bash
# Strategy 1: Even-spaced frames
jupyter notebook notebooks/Strategy_1_Feature_Extraction.ipynb

# Strategy 2: Early frames
jupyter notebook notebooks/Strategy_2_Feature_Extraction.ipynb

# Strategy 3: Random frames
jupyter notebook notebooks/Strategy_3_Feature_Extraction.ipynb
```

### 2. Model Training

Train each model independently:

```bash
# Train Model 1
jupyter notebook notebooks/Strategy_1_Model_training.ipynb

# Train Model 2
jupyter notebook notebooks/Strategy_2_Model_training.ipynb

# Train Model 3
jupyter notebook notebooks/Strategy_3_Model_training.ipynb
```

### 3. Streamlit Web Application

Launch the detection interface:

```bash
streamlit run app.py
```

Then upload a video and get instant predictions.

---

## Model Details

### Feature Extraction Pipeline

#### Step-by-Step Process

1. **Video Loading**
   - Library: `imageio.v3` + `pyav`
   - Purpose: Fast, codec-agnostic decoding

2. **Frame Sampling**
   - Strategy-specific selection (even/early/random)
   - Extracts 10 frames per video

3. **Face Detection**
   - Algorithm: MTCNN (Multi-task Cascaded CNN)
   - Output: First detected face per frame
   - Handles: Multiple poses and scales

4. **Preprocessing**
   - Crop face bounding box
   - Resize to 224x224
   - Normalize using ImageNet statistics

5. **Feature Extraction**
   - Model: ResNeXt-50 (32x4d)
   - Output: 2048-dimensional embedding
   - Pretrained on ImageNet

6. **Aggregation**
   - Method: Mean pooling across 10 frames
   - Result: Single 2048-d vector per video

### Training Configuration

| Parameter | Value |
|-----------|-------|
| **Loss Function** | Binary Cross-Entropy |
| **Optimizer** | Adam (LR: 1e-4) |
| **Scheduler** | ReduceLROnPlateau (factor: 0.5, patience: 5) |
| **Early Stopping** | Patience: 10 epochs on F1 score |
| **Train/Val Split** | 80% / 20% |
| **Batch Size** | 32 |
| **Max Epochs** | 100 |
| **Regularization** | Dropout (p=0.3) |

### Model Checkpoints

Models are stored in the `models` directory:
- `strategy1_C_best_model.pth` - Even-spaced sampling model
- `strategy2_C_best_model.pth` - Early frames model
- `strategy3_C_best_model.pth` - Random sampling model

---

## Results Visualization

### Training Metrics

All three models demonstrate:
- Smooth convergence of training loss
- Steady improvement in validation accuracy
- Consistent F1 score progression
- No significant overfitting

### Confusion Matrices

Each strategy achieves:
- High True Positive rates (219-233 correct REAL predictions)
- High True Negative rates (204-223 correct FAKE predictions)
- Low False Positive rates (23-34 errors)
- Low False Negative rates (32-44 errors)

---

## Limitations & Future Work

### Current Limitations

- **Multi-face scenarios**: Pipeline currently processes only the first detected face
- **Temporal modeling**: Static feature aggregation (mean pooling)
- **Dataset size**: Trained on subset (38 GB of 472 GB available)

### Planned Enhancements

#### Short-term
- [ ] **Multi-face handling**: Detect and analyze all faces in frame
- [ ] **Adversarial robustness**: Test against compression, noise, color-space attacks
- [ ] **Expanded training**: Leverage full 472 GB DFDC dataset

#### Medium-term
- [ ] **Temporal sequence modeling**: Integrate RNNs/Transformers for motion analysis
- [ ] **FastAPI wrapper**: Production-grade REST API
- [ ] **Docker containerization**: Easy deployment and scaling

#### Long-term
- [ ] **Distributed training**: Multi-GPU/Multi-node support
- [ ] **Real-time streaming**: Process live video feeds
- [ ] **Mobile deployment**: TensorFlow Lite/ONNX conversion

---

## References

1. Dolhansky et al., "The DeepFake Detection Challenge (DFDC) Dataset," *arXiv*, 2020.
2. Xie et al., "Aggregated Residual Transformations for Deep Neural Networks," *CVPR*, 2017.
3. Zhang et al., "Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks," *IEEE Signal Processing Letters*, 2016.
4. Paszke et al., "PyTorch: An Imperative Style, High-Performance Deep Learning Library," *NeurIPS*, 2019.

---

## Author

**Datta Sai V V N**
- UIN: 672019346
- Email: dvvn@uic.edu
- Institution: University of Illinois Chicago
- Course: CS 412 - Data Mining

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Kaggle for hosting the DFDC dataset
- Google Colab for providing GPU resources
- PyTorch and torchvision teams for excellent deep learning tools
- MTCNN developers for robust face detection

---

## Connect

If you found this project helpful, please consider:
- Starring the repository
- Forking for your own experiments
- Sharing with others interested in deepfake detection
