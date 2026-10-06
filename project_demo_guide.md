# Project Demo Presentation Guide: Parkinson's Speech Detection System

This guide is designed to help you present your project demo successfully. It breaks down the technical details into simple, easy-to-understand terms and provides a ready-to-use speaking script.

---

## 1. What is the Project? (The Core Idea)

This project is a **Parkinson's Speech Detection System** (named **ParkinsonsSpeech.AI**). 
* **The Problem:** Parkinson's Disease (PD) is a progressive neurological disorder that affects movement, but one of its earliest indicators is **Dysphonia** (difficulty speaking or changes in voice). Patients often experience quiet, breathy, unstable, or tremulous voices long before other motor symptoms appear.
* **The Solution:** We built a machine learning-powered web application that analyzes a person's voice in real-time (either through a live microphone recording or an uploaded audio file) to detect acoustic biomarkers associated with Parkinson's Disease.
* **Why it matters:** It acts as a **non-invasive, cost-effective, and rapid pre-screening tool** that can run in a web browser.

---

## 2. How We Created the Project (The Architecture)

Our system is split into three main components:

1. **The Machine Learning Model (The Brain):**
   * We evaluated two classifiers: **Random Forest** and **XGBoost Classifier**.
   * We selected **XGBoost** as our active model because of its high accuracy and efficiency in handling tabular acoustic features.
   * The models were trained and saved (`joblib` files) alongside a `StandardScaler` which normalizes user inputs so they match the format of the training data.

2. **The Audio Processing Pipeline (The Ear):**
   * Because raw Parkinson's audio datasets are highly restricted due to medical privacy, we built a **Synthetic Voice Generator** in Python to simulate clinical voice disorders.
   * **Healthy voices** were generated as clean, stable waves.
   * **Parkinson's voices** were injected with clinical dysphonia indicators:
     * *Vocal Tremor:* Pitch modulation between 4 to 7 Hz.
     * *Jitter:* Tiny random changes in frequency (pitch instability).
     * *Shimmer:* Tiny random changes in amplitude (loudness instability).
     * *Breathiness:* Additive white Gaussian noise, lowering the Harmonics-to-Noise Ratio (HNR).
   * We used **Librosa** (a Python library for audio analysis) to extract mathematical features from these recordings.

3. **The Web Interface (The Face):**
   * **Frontend:** Built with HTML, Vanilla CSS (styled with a premium glassmorphic dark theme and glowing accents), and interactive JavaScript.
   * **Visualizers:** A live canvas-based waveform visualizer that reacts when you speak, and an interactive graph showing Mel-Frequency Cepstral Coefficients (MFCCs).
   * **Backend:** A **Flask server (Python)** that handles the audio upload, extracts features on the fly, scales them, runs the XGBoost model, and sends the risk index and breakdown back to the screen.

---

## 3. How We Extracted the Project Features (The Pipeline)

When a user records their voice or uploads a WAV file, the system executes these steps:

1. **User inputs voice:** The user speaks into the microphone (live recording) or uploads a WAV file.
2. **Audio ingestion:** The Flask server receives the raw audio data.
3. **Resampling:** The audio is loaded and downsampled to 22,050 Hz using `librosa.load` to guarantee consistent sampling rates.
4. **Feature Extraction:** Librosa extracts mathematical feature vectors (MFCCs, Spectral Centroid, ZCR, RMS, etc.) across the audio duration.
5. **Mean Pooling:** We calculate the mean value of each acoustic feature over time to create a single static feature array.
6. **Standard Scaling:** The features are normalized using our fitted `StandardScaler` to match the model's expected inputs.
7. **XGBoost Inference:** The XGBoost model calculates class probabilities (healthy vs. Parkinson's).
8. **JSON Response:** The backend returns the prediction outcome, confidence levels, and the breakdown of acoustic metrics.
9. **UI Update:** The frontend dynamically renders the gauge score, active visual feedback, and color-coded meters.

---

## 4. Voice Features Explained in Simple Terms

To identify Parkinsonian dysphonia, our system extracts specific voice characteristics:

* **Mel-Frequency Cepstral Coefficients (MFCCs):**
  * *What it is:* The "shape" or resonance of the vocal tract (mouth, throat, tongue).
  * *Why we use it:* People with Parkinson's have stiffened vocal muscles, which alters how they articulate sounds. MFCCs capture these subtle changes in the texture and timbre of the voice.
* **RMS Energy (Root Mean Square Energy):**
  * *What it is:* The volume or loudness stability of the signal.
  * *Why we use it:* Parkinson's patients often exhibit vocal weakness (quieter voice) and struggle to maintain a constant, stable loudness level.
* **Spectral Centroid (Vocal Brightness):**
  * *What it is:* The "center of mass" of the sound frequencies (whether the voice sounds deep/heavy or bright/high-pitched).
  * *Why we use it:* A stiffened larynx shifts the concentration of voice energy, altering vocal brightness.
* **Zero Crossing Rate (ZCR):**
  * *What it is:* How fast the sound wave changes between positive and negative values (measuring noise density).
  * *Why we use it:* Breathy, whispery voices have higher noise levels, which increases the ZCR.
* **Spectral Rolloff & Bandwidth:**
  * *What it is:* Metrics indicating where high-frequency noise begins and how wide the frequency spread is.
  * *Why we use it:* They help identify the presence of vocal friction and breathiness over harmonic tones.
* **Chroma STFT:**
  * *What it is:* How the voice maps across the 12 musical pitches.
  * *Why we use it:* It captures the melodic quality and monotone pitch characteristics often found in Parkinson's speech.

---

## 5. Ready-to-Use Presentation Script (Spoken Points)

Here is a step-by-step script you can follow during your live demo.

### Slide 1: Introduction
> **"Good morning/afternoon everyone. Today, I am presenting our project: ParkinsonsSpeech.AI, which is an intelligent system designed to detect Parkinson's Disease from voice recordings.**
>
> **Parkinson's Disease is a neurological disorder, but it affects the larynx and vocal muscles early on, causing vocal instabilities. Our system acts as a non-invasive screening tool using advanced speech analysis and Machine Learning."**

### Slide 2: How We Created It & Built the Model
> **"To build this project, we designed a pipeline in Python. Because raw clinical speech audio is private and difficult to collect, we wrote a Synthetic Voice Generator. This generator simulates the speech patterns of healthy people and Parkinson's patients.**
>
> **Specifically, it models clinical biomarkers like Jitter—which is pitch instability, Shimmer—which is loudness instability, and Vocal Tremor—which is a periodic modulation of the voice. We then extracted features using Librosa and trained an XGBoost Classifier, which serves as the core machine learning model in our backend."**

### Slide 3: Live Demo walkthrough (Perform the Demo Now)
> *(Open the web browser and show the interface)*
> **"This is our user interface. It has a modern, dark-theme layout. I can either upload a WAV file of a voice recording, or record directly using the microphone.**
>
> **Let's perform a live test. I will click 'Start Recording' and hold a sustained vowel sound like 'ahhh'. This is recommended for acoustic stability."**
>
> *(Click record, make a sound for 2 seconds, click stop, and let it run)*

### Slide 4: Explaining the Results & Voice Features
> *(When the result is shown on the screen)*
> **"As you can see, the server processed the audio and returned a Diagnostic Report. It shows a Risk Index percentage and a diagnostic label.**
>
> **Behind the scenes, the system extracted several key features:**
> * **First, the RMS Energy, which tells us how stable the volume was.**
> * **Second, the Spectral Centroid, which measures the brightness of the voice.**
> * **Third, the Zero Crossing Rate, which flags any breathiness or whisper-like noise.**
> * **And finally, the MFCCs—which are shown in the bar chart below. These coefficients represent the shape of my vocal tract during the recording.**
>
> **All of these features are fed into our scaled XGBoost model to instantly calculate the risk probability. This shows how web technology and AI can work together to create accessible healthcare screening tools. Thank you!"**
