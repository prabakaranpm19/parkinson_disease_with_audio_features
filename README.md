# 🧠 Parkinson's Disease Speech Detection System

An end-to-end signal processing and machine learning pipeline to detect Parkinson's Disease (PD) from vocal recordings. By analyzing speech patterns and clinical vocal biomarkers—such as micro-instabilities in pitch/amplitude, vocal tremors, and harmonics-to-noise ratios—this system leverages signal processing (`librosa`) and gradient boosting (`XGBoost`) to classify voice samples.

---

## 🚀 Key Features

*   **Clinical Speech Biomarker Simulator**: Generates synthetic vocal waves modeled on healthy controls vs. Parkinson's patients (incorporating frequency/amplitude modulations like jitter, shimmer, 4–7 Hz tremor frequency, and additive noise to simulate breathiness/lower HNR).
*   **Feature Extraction Pipeline**: Extracts high-dimensional acoustic descriptors using `librosa`:
    *   **20 Mel-Frequency Cepstral Coefficients (MFCCs)**
    *   **Spectral Centroid, Rolloff, and Bandwidth**
    *   **Zero Crossing Rate (ZCR)**
    *   **Chroma Short-Time Fourier Transform (Chroma STFT)**
    *   **Root-Mean-Square (RMS) Energy**
*   **Machine Learning Classifiers**: Compares performance between **XGBoost Classifier** and **Random Forest Classifier** baselines.
*   **Flask REST API**: Serves a real-time HTTP server to accept audio file uploads (`.wav`), dynamically extract features, standardize them with a pre-fit `StandardScaler`, and predict the probability of Parkinson's Disease.

---

## 📁 Repository Structure

```filepath
├── app.py                            # Flask REST API server
├── parkinsons_speech_detection.ipynb # Jupyter notebook for the entire E2E training pipeline
├── uci_dataset.zip                   # Original UCI Parkinson's Speech dataset package
└── README.md                         # Project documentation
```

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/prabakaranpm19/parkinson_disease_with_audio_features.git
cd parkinson_disease_with_audio_features
```

### 2. Install Dependencies
Ensure you have Python 3.8+ installed. Install all required libraries:
```bash
pip install numpy pandas scikit-learn xgboost matplotlib seaborn joblib soundfile scipy librosa Flask flask-cors
```

---

## ⚙️ How to Use

### Step 1: Model Training & Feature Extraction
Run the Jupyter notebook `parkinsons_speech_detection.ipynb` to execute the full pipeline:
1.  Verify the workspace directories (`data/`, `models/`).
2.  Extract the UCI speech dataset.
3.  Simulate clinical voice recordings for feature engineering.
4.  Standardize features and train the classifier models.
5.  Save the trained XGBoost model and scaler (saved to `models/xgboost_parkinsons_model.joblib` and `models/standard_scaler.joblib`).

### Step 2: Run the Web Server
Launch the Flask backend server:
```bash
python app.py
```
By default, the server runs on `http://localhost:5000`.

### Step 3: Test the API Endpoint
Send a `POST` request to `http://localhost:5000/api/predict` with an audio file in the form-data parameter `audio`.

**Example request (cURL):**
```bash
curl -X POST -F "audio=@path_to_vocal_sample.wav" http://localhost:5000/api/predict
```

**Example JSON Response:**
```json
{
  "success": true,
  "prediction": 1,
  "label": "Parkinson's Disease Detected",
  "confidence": 0.942,
  "probabilities": {
    "healthy": 0.058,
    "parkinsons": 0.942
  },
  "metrics": {
    "mfccs": [...],
    "spectral_centroid": 1250.45,
    "spectral_rolloff": 2400.12,
    "spectral_bandwidth": 1150.32,
    "zcr": 0.045,
    "chroma": [...],
    "rms": 0.082
  }
}
```

---

## 📊 Acoustic Features Breakdown

*   **MFCCs**: Represents the short-term power spectrum of a sound, showing voice resonance properties.
*   **Spectral Centroid**: Indicates the "center of gravity" of the spectrum, mapping to speech brightness.
*   **Spectral Rolloff / Bandwidth**: Measures spectral shape and frequency spread.
*   **Zero Crossing Rate (ZCR)**: The rate at which the signal changes sign, helping identify voice noisiness.
*   **Chroma STFT**: Projects the sound spectrum onto 12 bin semitones to measure pitch class characteristics.
*   **RMS**: Mean root energy representing amplitude variations.
