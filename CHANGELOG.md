# Changelog

All notable changes to the **ParkinsonsSpeech.AI** project will be documented in this file.

## [1.1.0] - 2026-10-06 (Pass 1 Core Production Enhancements)

### Added
- **Real Acoustic Biomarkers Module ([`biomarkers.py`](file:///d:/pbl/parkinson/biomarkers.py)):**
  - Integrated Praat Parselmouth C++ bindings and Librosa PYIN fallback engine to compute true clinical parameters: Jitter (local %), Shimmer (local %), Harmonics-to-Noise Ratio (HNR in dB), and Noise-to-Harmonics Ratio (NHR).
  - Formulated continuous clinical anomaly index calibrated against UCI Parkinson's patient baseline distributions.
- **SQLAlchemy Persistence Layer ([`database.py`](file:///d:/pbl/parkinson/database.py)):**
  - Created `User`, `Prediction`, and `Feedback` models.
  - Added default SQLite database (`parkinson.db`) with PostgreSQL environment configuration support (`DATABASE_URL`).
  - Implemented automatic prediction history logging for every `/api/predict` execution.
- **Prediction History API ([`app.py`](file:///d:/pbl/parkinson/app.py)):**
  - Added `GET /api/history` endpoint to retrieve past user predictions and timestamps.
- **Input Validation & Content Security ([`app.py`](file:///d:/pbl/parkinson/app.py)):**
  - Implemented 25MB maximum upload payload enforcement (`MAX_CONTENT_LENGTH`).
  - Added binary header magic byte sniffing (`sniff_audio_format`) for WAV, MP3, OGG, FLAC, and WebM format verification.
  - Added audio duration restriction (max 30 seconds).
  - Added silent/near-silent audio rejection (RMS energy `< 0.003`) returning clear HTTP 400 error responses.

### Changed
- **Hybrid Scoring Blend ([`app.py`](file:///d:/pbl/parkinson/app.py)):**
  - Updated risk calculation to blend 50% XGBoost ML model probability + 25% legacy heuristic index + 25% clinical Praat biomarker index while maintaining full backward contract compatibility.
- **System Documentation ([`README.md`](file:///d:/pbl/parkinson/README.md), [`PROJECT_DOCUMENTATION.md`](file:///d:/pbl/parkinson/PROJECT_DOCUMENTATION.md)):**
  - Updated API specifications, database configurations, and architectural sitemaps.
