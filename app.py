# --- DeepFake Detection Streamlit App (Final Live Feature Extraction Version) ---

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
from facenet_pytorch import MTCNN
from torchvision import transforms
import imageio.v3 as iio

# --- Settings ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Initialize Face Detector (Global) ---
# keep_all=False ensures we mainly focus on the primary face if multiple are found,
# or we can manually select. select_largest=False is default but we will handle selection.
# Using a global instance prevents re-initialization on every frame/call.
mtcnn_detector = MTCNN(keep_all=False, select_largest=True, device=device)

# --- MLP Classifier Definition ---
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

# --- Load Feature Extractor ---
feature_extractor = torchvision.models.resnext50_32x4d(weights=torchvision.models.ResNeXt50_32X4D_Weights.DEFAULT)
feature_extractor.fc = nn.Identity()
feature_extractor = feature_extractor.to(device).eval()

# --- Load MLP Models ---
strategy_models = {}
for i in range(1, 4):
    model = MLPClassifier().to(device)
    model.load_state_dict(torch.load(f"models/strategy{i}_C_best_model.pth", map_location=device))
    model.eval()
    strategy_models[f"strategy_{i}"] = model

# --- Preprocessing Transform ---
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Frame Sampling Functions ---
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

# --- Efficient Frame Extraction ---
def extract_specific_frames(video_path, indices):
    cap = cv2.VideoCapture(video_path)
    frames = []
    indices = set(indices) # Optimize lookup
    
    current_frame = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if current_frame in indices:
            # Convert BGR (OpenCV) to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            
        current_frame += 1
        
        # Stop if we went past the max index needed (optimization)
        if current_frame > max(indices, default=0):
            break
            
    cap.release()
    return frames

# --- Face Detection ---
def detect_faces(frames):
    faces = []
    for frame in frames:
        # facenet-pytorch expects PIL or numpy. detect returns boxes, probs.
        # frame is numpy RGB.
        try:
            boxes, _ = mtcnn_detector.detect(frame)
        except Exception as e:
            # Fallback or skip if detection fails internally
            continue

        if boxes is not None and len(boxes) > 0:
            # Take the first face (highest probability/largest depending on config)
            # facenet-pytorch boxes are [x1, y1, x2, y2]
            box = boxes[0]
            x1, y1, x2, y2 = map(int, box)
            
            # Clamp coordinates to frame dimensions
            h_frame, w_frame = frame.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w_frame, x2)
            y2 = min(h_frame, y2)
            
            # Extract face
            if x2 > x1 and y2 > y1:
                face = frame[y1:y2, x1:x2]
                if face.size > 0:
                    faces.append(face)
    return faces

# --- Feature Extraction ---
@torch.no_grad()
def extract_features(faces):
    if len(faces) == 0:
        return None
    tensors = torch.stack([transform(face) for face in faces]).to(device)
    features = feature_extractor(tensors)
    return features.mean(dim=0).cpu().numpy()

# --- Streamlit Frontend ---
st.set_page_config(page_title="DeepFake Detection", layout="centered")
st.title("DeepFake Detection Ensemble App (Live Extraction)")

uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi"])

if uploaded_video:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        # Read and write in chunks to avoid memory overflow with large files
        chunk_size = 4 * 1024 * 1024 # 4MB chunks
        while True:
            chunk = uploaded_video.read(chunk_size)
            if not chunk:
                break
            tmp_file.write(chunk)
        video_path = tmp_file.name

    st.video(uploaded_video)

    st.info(f"Extracting frames and features...")

    with st.spinner("Processing video..."):
        results = {}

        # Get Total Frame Count
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if total_frames == 0:
             st.error("Could not read video frames.")
             st.stop()

        for strategy_name, model in strategy_models.items():
            # 1. Determine which frame indices to read
            indices = get_frame_indices(total_frames, strategy_name)
            
            # 2. Read ONLY those frames
            frames = extract_specific_frames(video_path, indices)
            
            # 3. Detect Faces
            faces = detect_faces(frames)

            if not faces:
                st.warning(f"No faces detected for {strategy_name}. Skipping...")
                continue
            
            # 4. Extract Features
            features = extract_features(faces)

            if features is None:
                st.warning(f"Feature extraction failed for {strategy_name}. Skipping...")
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

    # --- Display Final Ensemble Result ---
    if results:
        predictions = [v[0] for v in results.values()]
        final_prediction = Counter(predictions).most_common(1)[0][0]

        st.markdown("---")
        st.subheader("Final Ensemble Result")

        if final_prediction == "REAL":
            st.markdown(
                f"""
                <div style='background-color: #d4edda; padding: 20px; border-radius: 10px;'>
                    <h2 style='color: #155724; text-align: center;'>FINAL RESULT: REAL</h2>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div style='background-color: #f8d7da; padding: 20px; border-radius: 10px;'>
                    <h2 style='color: #721c24; text-align: center;'>FINAL RESULT: FAKE</h2>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("---")
        st.subheader("Individual Model Results")

        for strategy_name, (pred, conf) in results.items():
            with st.container():
                if pred == "REAL":
                    st.markdown(
                        f"""
                        <div style='background-color: #d4edda; padding: 10px; border-radius: 8px; margin-bottom: 10px;'>
                            <h5 style='color: #155724;'>{strategy_name} - REAL ({conf*100:.2f}% confidence)</h5>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div style='background-color: #f8d7da; padding: 10px; border-radius: 8px; margin-bottom: 10px;'>
                            <h5 style='color: #721c24;'>{strategy_name} - FAKE ({conf*100:.2f}% confidence)</h5>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                st.progress(conf)

    else:
        st.error("No predictions made. Check if faces are detected properly.")
