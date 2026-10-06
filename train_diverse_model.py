import os
import glob
import numpy as np
import pandas as pd
import scipy.io.wavfile as wav
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from features import extract_features_with_breakdown

print("Setting up directory structure...")
for d in ["data/raw_audio/0", "data/raw_audio/1", "data/processed", "models"]:
    os.makedirs(d, exist_ok=True)

def generate_diverse_synthetic_voice(file_path, label, duration=1.5, sr=22050):
    """
    Generates a realistic synthetic speech signal with continuous parameter variations 
    (varying pitch, tremor, jitter, shimmer, and noise levels).
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # 1. Randomize Fundamental Frequency (F0) across human speech pitch range (90 Hz - 240 Hz)
    f0 = np.random.uniform(90.0, 240.0)
    
    # Subtle natural pitch drift over time
    f0_drift = np.random.uniform(-3.0, 3.0) * t
    
    if label == 0:
        # Healthy: Stable voice with tiny natural micro-variations
        jitter_amp = np.random.uniform(0.05, 0.3)
        jitter_freq = np.random.uniform(10.0, 25.0)
        jitter = jitter_amp * np.sin(2 * np.pi * jitter_freq * t)
        
        phase = 2 * np.pi * ((f0 + f0_drift) * t + jitter * t)
        
        # Micro shimmer (loudness variation 0.5% - 2%)
        shimmer_amp = np.random.uniform(0.005, 0.02)
        shimmer = 1.0 + shimmer_amp * np.sin(2 * np.pi * 5.0 * t)
        
        # Main harmonic signal + 2nd & 3rd harmonics
        signal = shimmer * (
            1.0 * np.sin(phase) + 
            0.3 * np.sin(2 * phase) + 
            0.1 * np.sin(3 * phase)
        )
        
        # High Harmonics-to-Noise Ratio (tiny white noise)
        noise_level = np.random.uniform(0.001, 0.008)
        noise = np.random.normal(0, noise_level, len(signal))
        signal = signal + noise

    else:
        # Parkinson's Disease: Continuous spectrum of dysphonia severity
        # Severity factor between 0.2 (mild) and 1.0 (severe)
        severity = np.random.uniform(0.2, 1.0)
        
        # Vocal Tremor (frequency modulation between 4.0 Hz and 7.5 Hz)
        tremor_f = np.random.uniform(4.0, 7.5)
        tremor_depth = np.random.uniform(2.0, 8.0) * severity
        tremor = tremor_depth * np.sin(2 * np.pi * tremor_f * t)
        
        # Jitter: pitch micro-instability
        jitter_amp = np.random.uniform(1.0, 4.0) * severity
        jitter_freq = np.random.uniform(30.0, 50.0)
        jitter = jitter_amp * np.sin(2 * np.pi * jitter_freq * t)
        
        phase = 2 * np.pi * ((f0 + f0_drift) * t + tremor * t + jitter * t)
        
        # Shimmer: amplitude micro-instability (5% - 25%)
        shimmer_freq = np.random.uniform(6.0, 12.0)
        shimmer_amp = np.random.uniform(0.05, 0.25) * severity
        shimmer = 1.0 + shimmer_amp * np.sin(2 * np.pi * shimmer_freq * t)
        
        # Signal with tremor, jitter, and shimmer
        signal = shimmer * (
            1.0 * np.sin(phase) + 
            0.2 * np.sin(2 * phase) + 
            0.05 * np.sin(3 * phase)
        )
        
        # Additive noise (elevated breathiness / lower HNR)
        noise_level = np.random.uniform(0.03, 0.18) * severity
        noise = np.random.normal(0, noise_level, len(signal))
        signal = signal + noise

    # Normalize amplitude
    max_amp = np.max(np.abs(signal))
    if max_amp > 0:
        signal = signal / max_amp

    # Convert to 16-bit PCM integer WAV format and save
    signal_int16 = (signal * 32767).astype(np.int16)
    wav.write(file_path, sr, signal_int16)

# Clean out old static synthetic files
for old_file in glob.glob("data/raw_audio/*/*.wav"):
    try:
        os.remove(old_file)
    except Exception:
        pass

print("Generating 100 Healthy & 100 Parkinsonian diverse WAV files...")
healthy_dir = "data/raw_audio/0"
pd_dir = "data/raw_audio/1"

for i in range(100):
    generate_diverse_synthetic_voice(os.path.join(healthy_dir, f"healthy_{i+1}.wav"), label=0)
    generate_diverse_synthetic_voice(os.path.join(pd_dir, f"parkinson_{i+1}.wav"), label=1)

print("Extracting features using features.py...")
dataset_features = []
dataset_labels = []

for f in glob.glob(os.path.join(healthy_dir, "*.wav")):
    vec_df, _ = extract_features_with_breakdown(f)
    dataset_features.append(vec_df.iloc[0].values)
    dataset_labels.append(0)

for f in glob.glob(os.path.join(pd_dir, "*.wav")):
    vec_df, _ = extract_features_with_breakdown(f)
    dataset_features.append(vec_df.iloc[0].values)
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

print(f"Extracted dataset shape: {X.shape}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# XGBoost with probability calibration & smooth depth
base_xgb = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

# Apply CalibratedClassifierCV for smooth continuous probabilities
calibrated_clf = CalibratedClassifierCV(estimator=base_xgb, method='sigmoid', cv=5)
calibrated_clf.fit(X_train_scaled, y_train)

# Save calibrated model and scaler
xgb_model_path = "models/xgboost_parkinsons_model.joblib"
scaler_path = "models/standard_scaler.joblib"

joblib.dump(calibrated_clf, xgb_model_path)
joblib.dump(scaler, scaler_path)

print("\nCalibrated Classifier and Scaler saved successfully!")

# Test probabilities on test samples
test_probs = calibrated_clf.predict_proba(X_test_scaled)[:, 1]
print("\nSample Probabilities on Test Set (0=Healthy, 1=Parkinson's):")
for idx in range(10):
    true_cls = "Healthy" if y_test[idx] == 0 else "Parkinson's"
    prob_pd = test_probs[idx] * 100
    print(f"Sample {idx+1:02d} [{true_cls}]: Risk Probability = {prob_pd:.1f}%")
