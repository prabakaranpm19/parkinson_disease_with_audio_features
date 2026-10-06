import os
import traceback
import librosa
import numpy as np
import pandas as pd
import scipy.io.wavfile as wavfile

FEATURE_NAMES = (
    [f"mfcc_{i+1}" for i in range(20)] + 
    ["spectral_centroid", "spectral_rolloff", "spectral_bandwidth", "zcr", "chroma", "rms"]
)

def extract_features_with_breakdown(file_path, target_sr=22050):
    """
    Extracts a 26-element feature vector and returns both the DataFrame (1, 26)
    and a structured breakdown dictionary for UI visualization.
    Includes robust fallback for multi-format audio loading.
    """
    y = None
    sr = target_sr
    
    # Primary loader: Librosa
    try:
        y, sr = librosa.load(file_path, sr=target_sr)
    except Exception as e1:
        print(f"[WARN] librosa.load failed: {e1}. Trying scipy.io.wavfile fallback...")
        try:
            sr_in, data = wavfile.read(file_path)
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            elif data.dtype == np.int32:
                data = data.astype(np.float32) / 2147483648.0
            elif data.dtype == np.uint8:
                data = (data.astype(np.float32) - 128.0) / 128.0
            else:
                data = data.astype(np.float32)
                
            if data.ndim > 1:
                data = np.mean(data, axis=1)
                
            if sr_in != target_sr:
                data = librosa.resample(data, orig_sr=sr_in, target_sr=target_sr)
                
            y, sr = data, target_sr
        except Exception as e2:
            print(f"[ERROR] All audio loading methods failed for {file_path}")
            traceback.print_exc()
            raise RuntimeError(f"Could not decode audio file ({type(e1).__name__}). Please ensure file is a valid 16-bit PCM WAV or standard uncompressed audio format.")

    # Trim leading/trailing silence (top_db=20)
    try:
        y, _ = librosa.effects.trim(y, top_db=20)
    except Exception:
        pass
    
    # Ensure audio signal is non-empty
    if y is None or len(y) == 0:
        y = np.zeros(target_sr)
    
    # Extract Librosa features
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20), axis=1)
    spec_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    spec_rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    spec_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    chroma = float(np.mean(librosa.feature.chroma_stft(y=y, sr=sr)))
    rms = float(np.mean(librosa.feature.rms(y=y)))
    
    # Flatten 26-element raw vector
    raw_vector = np.hstack([
        mfccs,
        spec_centroid,
        spec_rolloff,
        spec_bandwidth,
        zcr,
        chroma,
        rms
    ]).reshape(1, -1)
    
    # Wrap in pandas DataFrame with exact feature names matching StandardScaler
    vector_df = pd.DataFrame(raw_vector, columns=FEATURE_NAMES)
    
    # Human-readable metrics dictionary for dashboard rendering
    breakdown = {
        'mfccs': mfccs.tolist(),
        'spectral_centroid_hz': round(spec_centroid, 2),
        'spectral_rolloff_hz': round(spec_rolloff, 2),
        'spectral_bandwidth_hz': round(spec_bandwidth, 2),
        'zero_crossing_rate': round(zcr, 4),
        'chroma_stft': round(chroma, 4),
        'rms_energy': round(rms, 4)
    }
    
    return vector_df, breakdown
