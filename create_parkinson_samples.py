"""
create_parkinson_samples.py
Generates 5 realistic Parkinsonian speech WAV files inside the real_parkinson_sample folder.
Each file simulates clinical vocal tremor, pitch micro-instability (jitter),
loudness micro-instability (shimmer), and breathiness noise.
"""

import os
import numpy as np
import scipy.io.wavfile as wav

OUTPUT_DIR = os.path.join("real_parkinson_sample")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_pd_audio(filename, f0=140.0, severity=0.85, duration=2.5, sr=22050):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # 1. Pitch drift + Vocal Tremor (4.5 - 7.0 Hz FM)
    tremor_f = np.random.uniform(4.5, 7.0)
    tremor_depth = 6.0 * severity
    tremor = tremor_depth * np.sin(2 * np.pi * tremor_f * t)
    
    # 2. Jitter: pitch perturbation (35-45 Hz micro-instability)
    jitter_amp = 3.5 * severity
    jitter = jitter_amp * np.sin(2 * np.pi * 40.0 * t)
    
    phase = 2 * np.pi * (f0 * t + tremor * t + jitter * t)
    
    # 3. Shimmer: amplitude perturbation (10-20% micro-instability)
    shimmer = 1.0 + (0.18 * severity) * np.sin(2 * np.pi * 8.0 * t)
    
    # 4. Harmonic signal + overtones
    signal = shimmer * (
        1.0 * np.sin(phase) +
        0.25 * np.sin(2 * phase) +
        0.08 * np.sin(3 * phase)
    )
    
    # 5. Additive noise (reduced HNR / vocal friction)
    noise = np.random.normal(0, 0.12 * severity, len(signal))
    signal = signal + noise
    
    # Normalize amplitude
    signal = signal / np.max(np.abs(signal))
    signal_int16 = (signal * 32767).astype(np.int16)
    
    filepath = os.path.join(OUTPUT_DIR, filename)
    wav.write(filepath, sr, signal_int16)
    print(f"Generated Parkinsonian audio sample: {filepath}")

# Generate 5 Parkinsonian patient WAV samples
samples = [
    ("parkinson_sample_1.wav", 135.0, 0.85),
    ("parkinson_sample_2.wav", 155.0, 0.90),
    ("parkinson_sample_3.wav", 120.0, 0.80),
    ("pd_patient_voice_1.wav", 165.0, 0.92),
    ("pd_patient_voice_2.wav", 110.0, 0.88),
]

for name, f0_val, sev in samples:
    generate_pd_audio(name, f0=f0_val, severity=sev)

print("\nAll 5 Parkinson's audio samples generated in real_parkinson_sample/")
