import os
import tempfile
import numpy as np
import librosa
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

# Load the trained model and scaler
MODEL_PATH = os.path.join("models", "xgboost_parkinsons_model.joblib")
SCALER_PATH = os.path.join("models", "standard_scaler.joblib")

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    print("[WARNING] Trained models not found. Please run the notebook cells or run_notebook.py first.")
    model = None
    scaler = None
else:
    print(f"Loading model from {MODEL_PATH}...")
    model = joblib.load(MODEL_PATH)
    print(f"Loading scaler from {SCALER_PATH}...")
    scaler = joblib.load(SCALER_PATH)

def extract_audio_features(file_path):
    """
    Extracts the acoustic features exactly matching the training pipeline.
    """
    # Load audio (downsample/fix rate to 22050 Hz)
    y, sr = librosa.load(file_path, sr=22050)
    
    # Extract raw descriptors
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y=y)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    
    # Compute means along timeline axis (axis=1) and concatenate
    features = np.concatenate([
        np.mean(mfccs, axis=1),
        np.mean(spectral_centroid, axis=1),
        np.mean(spectral_rolloff, axis=1),
        np.mean(spectral_bandwidth, axis=1),
        np.mean(zcr, axis=1),
        np.mean(chroma_stft, axis=1),
        np.mean(rms, axis=1)
    ])
    
    # Structured representation of features for frontend consumption
    feature_breakdown = {
        "mfccs": np.mean(mfccs, axis=1).tolist(),
        "spectral_centroid": float(np.mean(spectral_centroid)),
        "spectral_rolloff": float(np.mean(spectral_rolloff)),
        "spectral_bandwidth": float(np.mean(spectral_bandwidth)),
        "zcr": float(np.mean(zcr)),
        "chroma": np.mean(chroma_stft, axis=1).tolist(),
        "rms": float(np.mean(rms))
    }
    
    return features, feature_breakdown

@app.route('/')
def serve_index():
    return app.send_static_file('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({
            "success": False,
            "error": "ML models are not loaded. Run notebook training first."
        }), 500

    if 'audio' not in request.files:
        return jsonify({
            "success": False,
            "error": "No audio file provided in request."
        }), 400
        
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({
            "success": False,
            "error": "Empty filename."
        }), 400

    # Save audio file to a temporary location
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, "uploaded_speech.wav")
    
    try:
        audio_file.save(temp_path)
        
        # Extract features
        features, breakdown = extract_audio_features(temp_path)
        
        # Standardize features using the training scaler
        scaled_features = scaler.transform([features])
        
        # Perform prediction and retrieve class probabilities
        pred_label = int(model.predict(scaled_features)[0])
        probabilities = model.predict_proba(scaled_features)[0].tolist()
        
        # Clean up the file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return jsonify({
            "success": True,
            "prediction": pred_label,
            "label": "Parkinson's Disease Detected" if pred_label == 1 else "Healthy / No Signs Detected",
            "confidence": float(probabilities[pred_label]),
            "probabilities": {
                "healthy": float(probabilities[0]),
                "parkinsons": float(probabilities[1])
            },
            "metrics": breakdown
        })
        
    except Exception as e:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        return jsonify({
            "success": False,
            "error": f"Failed to analyze audio: {str(e)}"
        }), 500

if __name__ == '__main__':
    # Run the server on all interfaces at port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
