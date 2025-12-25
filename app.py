# --- Enhanced DeepFake Detection Streamlit App ---
# Pure UI Enhancement - No Logic Changes

import streamlit as st
import torch
import torch.nn as nn
import torchvision
import numpy as np
import tempfile
import cv2
from pathlib import Path
from collections import Counter
import random
from mtcnn import MTCNN
from torchvision import transforms
import imageio.v3 as iio
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="DeepFake Detection System",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS Styling ---
st.markdown("""
<style>
    /* Import Modern Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main Container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    
    /* Hero Section */
    .hero-section {
        text-align: center;
        padding: 60px 20px;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(168, 85, 247, 0.1));
        border-radius: 24px;
        margin-bottom: 40px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .hero-title {
        font-size: 56px;
        font-weight: 900;
        margin-bottom: 16px;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        background-size: 200% 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: gradient-shift 6s ease infinite;
    }
    
    @keyframes gradient-shift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .hero-subtitle {
        font-size: 20px;
        color: rgba(255, 255, 255, 0.6);
        font-weight: 400;
        letter-spacing: 0.5px;
    }
    
    /* Upload Section */
    .upload-container {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 40px;
        border: 2px dashed rgba(99, 102, 241, 0.3);
        margin-bottom: 30px;
        transition: all 0.3s ease;
    }
    
    .upload-container:hover {
        border-color: rgba(99, 102, 241, 0.6);
        background: rgba(255, 255, 255, 0.05);
    }
    
    /* Video Player Container */
    .video-container {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        margin: 20px auto;
        max-width: 400px; /* Limit max width for compactness */
    }
    
    /* Result Cards */
    .result-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(16px);
        border-radius: 16px;
        padding: 30px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        animation: fadeInUp 0.6s ease-out forwards;
    }
    
    .result-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 36px rgba(0, 0, 0, 0.4);
        border-color: rgba(255, 255, 255, 0.2);
    }
    
    /* Fade In Animation */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Final Result Card */
    .final-result-card {
        text-align: center;
        padding: 40px 20px;
        border-radius: 20px;
        margin-bottom: 30px;
        position: relative;
        overflow: hidden;
        animation: fadeInUp 0.8s ease-out forwards;
    }
    
    .final-result-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255, 255, 255, 0.1) 0%, transparent 70%);
        animation: pulse 4s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 0.8; }
    }
    
    .result-text {
        font-size: 42px; /* Smaller font for mobile */
        font-weight: 900;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 2px;
        position: relative;
        z-index: 1;
        text-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    .result-subtitle {
        font-size: 16px;
        margin-top: 10px;
        opacity: 0.9;
        position: relative;
        z-index: 1;
        font-weight: 500;
    }
    
    /* Model Result Card */
    .model-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(12px);
        border-radius: 12px;
        padding: 20px; /* Reduced padding */
        margin-bottom: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
        animation: fadeInUp 0.6s ease-out forwards;
    }
    
    .model-card:hover {
        transform: translateX(4px);
        border-color: rgba(255, 255, 255, 0.2);
    }
    
    .model-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
    }
    
    .model-name {
        font-size: 16px;
        font-weight: 600;
        color: rgba(255, 255, 255, 0.95); /* Improved visibility */
        margin: 0;
    }
    
    .prediction-badge {
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Confidence Meter */
    .confidence-meter {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        height: 12px; /* Thinner bar */
        overflow: hidden;
        position: relative;
        margin-bottom: 8px;
    }
    
    .confidence-fill {
        height: 100%;
        border-radius: 8px;
        transition: width 1.2s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .confidence-fill::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
        animation: shimmer 2s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    .confidence-fill.authentic {
        background: linear-gradient(90deg, #10b981, #14b8a6);
        box-shadow: 0 0 10px rgba(16, 185, 129, 0.5);
    }
    
    .confidence-fill.synthetic {
        background: linear-gradient(90deg, #ef4444, #f43f5e);
        box-shadow: 0 0 10px rgba(239, 68, 68, 0.5);
    }
    
    .confidence-text {
        margin-top: 4px;
        color: rgba(255, 255, 255, 0.6);
        font-size: 12px;
        font-weight: 500;
        text-align: right; /* Right align for cleaner look */
    }
    
    /* Processing Section */
    .processing-container {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(99, 102, 241, 0.2);
        margin: 20px 0;
    }
    
    .processing-title {
        color: white;
        font-size: 20px;
        font-weight: 700;
        margin-bottom: 20px;
        text-align: center;
    }
    
    .stage-container {
        margin-bottom: 16px;
    }
    
    .stage-header {
        display: flex;
        justify-content: space-between;
        margin-bottom: 6px;
    }
    
    .stage-name {
        color: white;
        font-size: 14px;
        font-weight: 500;
    }
    
    .stage-progress {
        color: rgba(255, 255, 255, 0.7);
        font-size: 13px;
    }
    
    .progress-bar-container {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        overflow: hidden;
        height: 6px;
    }
    
    .progress-bar {
        background: linear-gradient(90deg, #6366f1, #a855f7);
        height: 100%;
        transition: width 0.5s ease;
        box-shadow: 0 0 10px rgba(99, 102, 241, 0.6);
    }
    
    /* Section Headers */
    .section-header {
        color: rgba(255, 255, 255, 0.9); /* Increased opacity */
        font-size: 24px;
        font-weight: 700;
        margin: 30px 0 20px 0;
        text-align: center;
        text-shadow: 0 2px 4px rgba(0,0,0,0.5); /* Drop shadow for contrast */
    }
    
    /* Divider */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        margin: 30px 0;
    }
    
    /* Info Box */
    .info-box {
        background: rgba(59, 130, 246, 0.1);
        border-left: 3px solid #3b82f6;
        padding: 16px;
        border-radius: 8px;
        margin: 16px 0;
    }
    
    .info-text {
        color: rgba(255, 255, 255, 0.9);
        font-size: 14px;
        margin: 0;
    }

    /* Mobile Responsiveness */
    @media (max-width: 600px) {
        .hero-title { font-size: 36px; }
        .hero-subtitle { font-size: 16px; }
        .result-text { font-size: 32px; }
        .upload-container { padding: 20px; }
    }
</style>
""", unsafe_allow_html=True)

# --- Settings (Same as Original) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- MLP Classifier Definition (Unchanged) ---
class MLPClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.classifier(x).squeeze(1)

# --- Load Feature Extractor (Unchanged) ---
feature_extractor = torchvision.models.resnext50_32x4d(weights=torchvision.models.ResNeXt50_32X4D_Weights.DEFAULT)
feature_extractor.fc = nn.Identity()
feature_extractor = feature_extractor.to(device).eval()

# --- Load MLP Models (Unchanged) ---
strategy_models = {}
for i in range(1, 4):
    model = MLPClassifier().to(device)
    model.load_state_dict(torch.load(f"models/strategy{i}_C_best_model.pth", map_location=device))
    model.eval()
    strategy_models[f"strategy_{i}"] = model

# --- Preprocessing Transform (Unchanged) ---
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Frame Sampling Functions (Unchanged) ---
def get_frame_indices(total_frames, strategy):
    if total_frames == 0:
        return []
    if total_frames < 10:
        return list(range(total_frames))
    
    if strategy == "strategy_1":
        return np.linspace(0, total_frames - 1, 10, dtype=int)
    elif strategy == "strategy_2":
        return np.arange(0, total_frames, total_frames // 10)[:10]
    elif strategy == "strategy_3":
        return np.random.choice(range(total_frames), 10, replace=False)
    return []

# --- Efficient Frame Extraction (Unchanged) ---
def extract_specific_frames(video_path, indices):
    cap = cv2.VideoCapture(video_path)
    frames = []
    indices = set(indices)
    
    current_frame = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if current_frame in indices:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            
        current_frame += 1
        
        if current_frame > max(indices, default=0):
            break
            
    cap.release()
    return frames

# --- Face Detection (Unchanged) ---
def detect_faces(frames):
    detector = MTCNN()
    faces = []
    for frame in frames:
        detections = detector.detect_faces(frame)
        if detections:
            x, y, w, h = detections[0]['box']
            x, y = max(0, x), max(0, y)
            face = frame[y:y+h, x:x+w]
            if face.size > 0:
                faces.append(face)
    return faces

# --- Feature Extraction (Unchanged) ---
@torch.no_grad()
def extract_features(faces):
    if len(faces) == 0:
        return None
    tensors = torch.stack([transform(face) for face in faces]).to(device)
    features = feature_extractor(tensors)
    return features.mean(dim=0).cpu().numpy()

# ===== STREAMLIT UI =====

# Hero Section
st.markdown("""
<div class='hero-section'>
    <h1 class='hero-title'>DeepFake Detection System</h1>
    <p class='hero-subtitle'>Advanced ensemble learning for authentic video verification</p>
</div>
""", unsafe_allow_html=True)

# Upload Section
st.markdown("<div class='upload-container'>", unsafe_allow_html=True)
uploaded_video = st.file_uploader("Upload a Video for Analysis", type=["mp4", "avi", "mov", "mpeg4"], label_visibility="collapsed")
st.markdown("</div>", unsafe_allow_html=True)

if uploaded_video:
    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(uploaded_video.read())
        video_path = tmp_file.name

    # Display Video (Resized to be small/compact)
    st.markdown("""
    <div style='display: flex; justify-content: center;'>
        <div class='video-container'>
    """, unsafe_allow_html=True)
    st.video(uploaded_video)
    st.markdown("""
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Info Box
    st.markdown("""
    <div class='info-box'>
        <p class='info-text'>Analyzing video using three specialized detection strategies with ensemble voting...</p>
    </div>
    """, unsafe_allow_html=True)

    # Processing Section
    with st.spinner(""):
        # Get Total Frame Count
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if total_frames == 0:
            st.error("Could not read video frames. Please try another video.")
            st.stop()

        # Processing Stages Animation
        stages = [
            "Loading Video",
            "Extracting Frames",
            "Detecting Faces",
            "Analyzing Features",
            "Generating Predictions"
        ]
        
        progress_placeholder = st.empty()
        
        # Show processing stages
        for idx, stage in enumerate(stages):
            progress_pct = ((idx + 1) / len(stages)) * 100
            
            progress_html = f"""
            <div class='processing-container'>
                <div class='processing-title'>Processing Video</div>
            """
            
            for s_idx, s_name in enumerate(stages):
                s_progress = 100 if s_idx < idx else (100 if s_idx == idx else 0)
                progress_html += f"""
                <div class='stage-container'>
                    <div class='stage-header'>
                        <span class='stage-name'>{s_name}</span>
                        <span class='stage-progress'>{s_progress:.0f}%</span>
                    </div>
                    <div class='progress-bar-container'>
                        <div class='progress-bar' style='width: {s_progress}%;'></div>
                    </div>
                </div>
                """
            
            progress_html += "</div>"
            progress_placeholder.markdown(progress_html, unsafe_allow_html=True)
            time.sleep(0.3)
        
        # Process video with all strategies
        results = {}
        
        for strategy_name, model in strategy_models.items():
            indices = get_frame_indices(total_frames, strategy_name)
            frames = extract_specific_frames(video_path, indices)
            faces = detect_faces(frames)

            if not faces:
                continue
            
            features = extract_features(faces)
            if features is None:
                continue

            input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                prob = model(input_tensor).item()

            if prob > 0.5:
                prediction = "FAKE"
                confidence = prob
            else:
                prediction = "REAL"
                confidence = 1 - prob

            results[strategy_name] = (prediction, confidence)
        
        # Clear progress
        progress_placeholder.empty()

    # ===== DISPLAY RESULTS =====
    if results:
        # Calculate ensemble prediction
        predictions = [v[0] for v in results.values()]
        final_prediction = Counter(predictions).most_common(1)[0][0]

        # Divider
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # Final Result Display
        if final_prediction == "REAL":
            result_text = "AUTHENTIC"
            gradient = "linear-gradient(135deg, #10b981, #14b8a6)"
            text_color = "#ecfdf5"
        else:
            result_text = "SYNTHETIC"
            gradient = "linear-gradient(135deg, #ef4444, #f43f5e)"
            text_color = "#fef2f2"

        st.markdown(f"""
        <div class='final-result-card' style='background: {gradient};'>
            <h2 class='result-text' style='color: {text_color};'>{result_text}</h2>
            <p class='result-subtitle' style='color: {text_color};'>Analysis Complete</p>
        </div>
        """, unsafe_allow_html=True)

        # Section Header for Individual Results
        st.markdown("<h3 class='section-header'>Individual Model Results</h3>", unsafe_allow_html=True)

        # Display each model result
        # Map strategy names to neat display names
        strategy_map = {
            "strategy_1": "Model 1",
            "strategy_2": "Model 2",
            "strategy_3": "Model 3"
        }

        for idx, (strategy_name, (pred, conf)) in enumerate(results.items()):
            # Determine styling
            if pred == "REAL":
                badge_class = "authentic"
                badge_color = "background: linear-gradient(135deg, #10b981, #14b8a6); color: #ecfdf5;"
                label = "AUTHENTIC"
            else:
                badge_class = "synthetic"
                badge_color = "background: linear-gradient(135deg, #ef4444, #f43f5e); color: #fef2f2;"
                label = "SYNTHETIC"
            
            # Format strategy name using the map, default to original if not found
            display_name = strategy_map.get(strategy_name, strategy_name.replace('_', ' ').title())
            
            # Animation delay for staggered effect
            delay = idx * 0.15
            
            st.markdown(f"""
            <div class='model-card' style='animation-delay: {delay}s;'>
                <div class='model-header'>
                    <h4 class='model-name'>{display_name}</h4>
                    <span class='prediction-badge' style='{badge_color}'>{label}</span>
                </div>
                <div class='confidence-meter'>
                    <div class='confidence-fill {badge_class}' style='width: {conf*100}%;'></div>
                </div>
                <p class='confidence-text'>Confidence: {conf*100:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class='info-box' style='background: rgba(239, 68, 68, 0.1); border-left: 4px solid #ef4444;'>
            <p class='info-text'>No predictions could be made. Please ensure the video contains visible faces.</p>
        </div>
        """, unsafe_allow_html=True)
