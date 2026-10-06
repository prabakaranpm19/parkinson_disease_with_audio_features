import os
import glob
import numpy as np
import pandas as pd
import scipy.io.wavfile as wav
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from features import extract_features_with_breakdown

print("Setting up directory structure...")
for d in ["data/raw_audio/0", "data/raw_audio/1", "data/processed", "models"]:
    os.makedirs(d, exist_ok=True)

def generate_synthetic_voice(file_path, label, duration=1.5, sr=22050):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    f0 = 150.0  # fundamental frequency
    
    if label == 0:
        # Healthy: clean, stable sine wave
        signal = np.sin(2 * np.pi * f0 * t)
    else:
        # Parkinson's: introduce voice impairments
        tremor_f = 6.0
        tremor = 5.0 * np.sin(2 * np.pi * tremor_f * t)
        jitter = 2.0 * np.sin(2 * np.pi * 35.0 * t)
        phase = 2 * np.pi * (f0 * t + tremor * t + jitter * t)
        shimmer = 1.0 + 0.15 * np.sin(2 * np.pi * 8.0 * t)
        signal = shimmer * np.sin(phase)
        noise = np.random.normal(0, 0.1, len(signal))
        signal = signal + noise
        
    signal = signal / np.max(np.abs(signal))
    signal_int16 = (signal * 32767).astype(np.int16)
    wav.write(file_path, sr, signal_int16)

healthy_dir = "data/raw_audio/0"
pd_dir = "data/raw_audio/1"

print("Generating synthetic speech audio files...")
for i in range(25):
    generate_synthetic_voice(os.path.join(healthy_dir, f"healthy_{i+1}.wav"), label=0)
    generate_synthetic_voice(os.path.join(pd_dir, f"parkinson_{i+1}.wav"), label=1)

print("Extracting features using features.py...")
dataset_features = []
dataset_labels = []

for f in glob.glob(os.path.join(healthy_dir, "*.wav")):
    vec, _ = extract_features_with_breakdown(f)
    dataset_features.append(vec.flatten())
    dataset_labels.append(0)

for f in glob.glob(os.path.join(pd_dir, "*.wav")):
    vec, _ = extract_features_with_breakdown(f)
    dataset_features.append(vec.flatten())
    dataset_labels.append(1)

X = np.array(dataset_features)
y = np.array(dataset_labels)

feature_names = (
    [f"mfcc_{i+1}" for i in range(20)] + 
    ["spectral_centroid", "spectral_rolloff", "spectral_bandwidth", "zcr", "chroma", "rms"]
)

df = pd.DataFrame(X, columns=feature_names)
df['status'] = y
df.to_csv("data/processed/parkinsons_extracted_features.csv", index=False)

print(f"Dataset shape: {X.shape}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

xgb_clf = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

xgb_clf.fit(X_scaled, y)

xgb_model_path = "models/xgboost_parkinsons_model.joblib"
scaler_path = "models/standard_scaler.joblib"

joblib.dump(xgb_clf, xgb_model_path)
joblib.dump(scaler, scaler_path)

print("Model and Scaler trained and saved successfully with 26 features!")
