# ParkinsonsSpeech.AI — Speech-Based Parkinson's Disease Screening Application

**ParkinsonsSpeech.AI** is a machine learning and digital signal processing (DSP) application that screens for vocal dysphonia and early acoustic biomarkers associated with Parkinson's Disease.

---

## Key Features & Upgrades (Pass 1 Core Production Enhancements)

1. **Real Clinical Acoustic Biomarkers (`biomarkers.py`):**
   - Calculates true clinical voice parameters: **Jitter (local %)**, **Shimmer (local %)**, **Harmonics-to-Noise Ratio (HNR in dB)**, and **Noise-to-Harmonics Ratio (NHR)**.
   - Primary engine uses **Praat Parselmouth** C++ bindings with an automatic **Librosa PYIN** autocorrelation fallback.
   - Exposes both legacy heuristic index and real clinical biomarkers for backward compatibility.

2. **Database Persistence Layer (`database.py`):**
   - **SQLAlchemy ORM** models: `User`, `Prediction`, `Feedback`.
   - SQLite by default (`parkinson.db`) for local development; PostgreSQL ready via `DATABASE_URL` environment variable.
   - Saves `sample_id`, 26-dim `feature_vector`, `ml_probability`, `acoustic_index`, `biomarkers`, `risk_score`, `risk_label`, and `timestamp` for every prediction.
   - **History Endpoint (`GET /api/history`):** Retrieve past predictions and trends with optional `?user_id=...` filter.

3. **Input Validation & Content Security (`app.py`):**
   - **File Size Limit:** Enforces 25MB upload ceiling (`MAX_CONTENT_LENGTH`).
   - **Header Magic Byte Sniffing:** Verifies true binary signatures (`b'RIFF'`, `b'OggS'`, `b'fLaC'`, `b'ID3'`, `b'\x1a\x45\xdf\xa3'`) before processing to prevent non-audio ingestion.
   - **Duration Rejection:** Rejects files exceeding 30 seconds limit.
   - **Silent Audio Rejection:** Rejects recordings with RMS energy `< 0.003` to avoid low-confidence false predictions.

---

## API Documentation

### 1. `POST /api/predict`
Ingests audio recording and returns Parkinson's risk assessment.

* **Headers:** `Content-Type: multipart/form-data`
* **Form Field:** `audio` (file: WAV, MP3, OGG, FLAC, or WebM)
* **Optional Form Field:** `user_id` (string)
* **Response Example:**
```json
{
  "success": true,
  "sample_id": "a3f12e84-7c9b-4b12-98ef-123456789abc",
  "prediction": 1,
  "label": "HIGH RISK / PARKINSONIAN BIOMARKERS",
  "confidence": 0.8425,
  "probabilities": {
    "healthy": 0.1575,
    "parkinsons": 0.8425
  },
  "biomarkers": {
    "jitter_percent": 2.45,
    "shimmer_percent": 6.80,
    "hnr_db": 14.2,
    "nhr": 0.038,
    "clinical_anomaly_index": 0.72,
    "method": "praat_parselmouth"
  },
  "metrics": {
    "spectral_centroid_hz": 1850.4,
    "zero_crossing_rate": 0.048,
    "rms_energy": 0.042
  },
  "disclaimer": "Screening tool only — not a clinical medical diagnosis."
}
```

### 2. `GET /api/history`
Returns history of past predictions.

* **Query Parameters:** `?user_id=anonymous` (optional)
* **Response Example:**
```json
{
  "success": true,
  "count": 1,
  "history": [
    {
      "sample_id": "a3f12e84-7c9b-4b12-98ef-123456789abc",
      "user_id": "anonymous",
      "filename": "live_recording.wav",
      "risk_score": 0.8425,
      "risk_label": "HIGH RISK / PARKINSONIAN BIOMARKERS",
      "timestamp": "2026-10-06T18:00:00.000000"
    }
  ]
}
```

### 3. `GET /api/health`
Health check and database connection status.

---

## Environment Variables

Configure via environment or `.env` file:

```env
DATABASE_URL=sqlite:///parkinson.db
SECRET_KEY=your-secret-key-here
PORT=5000
```

---

## Medical Disclaimer
**ParkinsonsSpeech.AI** is designed purely as an educational research pre-screening demo. It is not a certified diagnostic device and should not replace professional neurological evaluation.
