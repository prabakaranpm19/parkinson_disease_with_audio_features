/**
 * ParkinsonsSpeech.AI - Clinical Diagnostic Dashboard Logic
 * Handles Web Audio API raw PCM WAV recording, canvas oscilloscope, API communication,
 * arch gauge needle animation, and acoustic biomarker spectrum rendering.
 */

document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Element References ---
    
    // Header Metrics
    const latencyValEl = document.getElementById('latency-val');
    const statusTextEl = document.getElementById('status-text');
    const statusIndicatorEl = document.querySelector('.status-indicator');

    // Tab Switcher
    const tabRecordBtn = document.getElementById('tab-record');
    const tabUploadBtn = document.getElementById('tab-upload');
    const contentRecordPane = document.getElementById('content-record');
    const contentUploadPane = document.getElementById('content-upload');

    // Recording Controls
    const waveformCanvas = document.getElementById('waveform-canvas');
    const canvasPlaceholder = document.getElementById('canvas-placeholder');
    const recordingTimerEl = document.getElementById('recording-timer');
    const timerStatusHintEl = document.getElementById('timer-status-hint');
    const recProgressBar = document.getElementById('rec-progress');
    const btnRecordAction = document.getElementById('btn-record-action');
    const btnRecordText = document.getElementById('btn-record-text');
    const audioPreviewEl = document.getElementById('audio-preview');
    const previewWrapper = document.getElementById('preview-wrapper');

    // Upload Controls
    const dropZone = document.getElementById('drop-zone');
    const btnBrowseTrigger = document.getElementById('btn-browse-trigger');
    const fileInput = document.getElementById('file-input');
    const fileInfoContainer = document.getElementById('file-info-container');
    const selectedFileName = document.getElementById('selected-file-name');
    const btnAnalyzeFile = document.getElementById('btn-analyze-file');

    // Diagnostic States
    const stateIdle = document.getElementById('state-idle');
    const stateLoading = document.getElementById('state-loading');
    const stateReport = document.getElementById('state-report');
    const loadingStatusText = document.getElementById('loading-status-text');

    // Gauge & Report Elements
    const gaugeNeedleGroup = document.getElementById('gauge-needle-group');
    const gaugeFillArc = document.getElementById('gauge-fill-arc');
    const riskScoreEl = document.getElementById('risk-score');
    const diagnosisLabelEl = document.getElementById('diagnosis-label');
    const diagnosisDescEl = document.getElementById('diagnosis-description');
    const probHealthyValEl = document.getElementById('prob-healthy-val');
    const probHealthyBarEl = document.getElementById('prob-healthy-bar');
    const probPdValEl = document.getElementById('prob-pd-val');
    const probPdBarEl = document.getElementById('prob-pd-bar');

    // Feature Cards
    const mfccCanvas = document.getElementById('mfcc-canvas');
    const metricChromaEl = document.getElementById('metric-chroma');
    const metricCentroidEl = document.getElementById('metric-centroid');
    const evalPitchEl = document.getElementById('eval-pitch');
    
    const metricRolloffEl = document.getElementById('metric-rolloff');
    const metricBandwidthEl = document.getElementById('metric-bandwidth');
    const barBrightnessEl = document.getElementById('bar-brightness');

    const metricZcrEl = document.getElementById('metric-zcr');
    const metricRmsEl = document.getElementById('metric-rms');
    const evalNoiseEl = document.getElementById('eval-noise');

    // --- State Variables ---
    let audioContext = null;
    let analyser = null;
    let mediaStream = null;
    let scriptProcessor = null;
    let pcmBuffers = [];
    let isRecording = false;
    let recCountdownTimer = null;
    let animFrameId = null;
    let selectedFile = null;
    const RECORDING_DURATION = 3.0; // 3 Seconds strict phonation timer

    // --- 1. Latency & System Health Ping ---
    async function checkHealth() {
        const start = performance.now();
        try {
            const response = await fetch('/api/health');
            const latency = Math.round(performance.now() - start);
            if (response.ok) {
                latencyValEl.textContent = `${latency} ms`;
                statusTextEl.textContent = 'System Ready';
                if (statusIndicatorEl) statusIndicatorEl.className = 'status-indicator ready';
            } else {
                statusTextEl.textContent = 'Degraded';
            }
        } catch (e) {
            latencyValEl.textContent = 'Offline';
            statusTextEl.textContent = 'Disconnected';
        }
    }
    checkHealth();
    setInterval(checkHealth, 15000);

    // --- 2. Tab Switcher Logic ---
    tabRecordBtn.addEventListener('click', () => {
        tabRecordBtn.classList.add('active');
        tabUploadBtn.classList.remove('active');
        contentRecordPane.classList.remove('hidden');
        contentUploadPane.classList.add('hidden');
    });

    tabUploadBtn.addEventListener('click', () => {
        tabUploadBtn.classList.add('active');
        tabRecordBtn.classList.remove('active');
        contentUploadPane.classList.remove('hidden');
        contentRecordPane.classList.add('hidden');
    });

    // --- 3. Web Audio API Oscilloscope Visualization ---
    function initAudioContext() {
        if (!audioContext) {
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
        }
    }

    function setupOscilloscope(stream) {
        initAudioContext();
        if (audioContext.state === 'suspended') {
            audioContext.resume();
        }

        const source = audioContext.createMediaStreamSource(stream);
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 2048;
        source.connect(analyser);

        // Connect script processor for raw PCM collection
        scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);
        scriptProcessor.onaudioprocess = (e) => {
            if (!isRecording) return;
            const inputData = e.inputBuffer.getChannelData(0);
            pcmBuffers.push(new Float32Array(inputData));
        };

        source.connect(scriptProcessor);
        scriptProcessor.connect(audioContext.destination);

        drawOscilloscope();
    }

    function drawOscilloscope() {
        if (!waveformCanvas) return;
        const ctx = waveformCanvas.getContext('2d');
        const width = waveformCanvas.width = waveformCanvas.parentElement.clientWidth;
        const height = waveformCanvas.height = waveformCanvas.parentElement.clientHeight;

        const bufferLength = analyser ? analyser.frequencyBinCount : 0;
        const dataArray = new Uint8Array(bufferLength);

        function renderFrame() {
            animFrameId = requestAnimationFrame(renderFrame);

            ctx.fillStyle = '#090d12';
            ctx.fillRect(0, 0, width, height);

            // Draw center grid line
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(0, height / 2);
            ctx.lineTo(width, height / 2);
            ctx.stroke();

            if (!analyser || !isRecording) {
                // Draw idle sine placeholder line
                ctx.lineWidth = 2;
                ctx.strokeStyle = 'rgba(56, 189, 248, 0.3)';
                ctx.beginPath();
                const time = Date.now() * 0.003;
                for (let x = 0; x < width; x++) {
                    const y = height / 2 + Math.sin(x * 0.02 + time) * 6;
                    if (x === 0) ctx.moveTo(x, y);
                    else ctx.lineTo(x, y);
                }
                ctx.stroke();
                return;
            }

            analyser.getByteTimeDomainData(dataArray);

            ctx.lineWidth = 2.5;
            ctx.strokeStyle = '#38bdf8'; // Glowing cyan
            ctx.shadowBlur = 10;
            ctx.shadowColor = '#38bdf8';

            ctx.beginPath();
            const sliceWidth = width * 1.0 / bufferLength;
            let x = 0;

            for (let i = 0; i < bufferLength; i++) {
                const v = dataArray[i] / 128.0;
                const y = v * height / 2;

                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
                x += sliceWidth;
            }

            ctx.lineTo(width, height / 2);
            ctx.stroke();
            ctx.shadowBlur = 0;
        }

        if (animFrameId) cancelAnimationFrame(animFrameId);
        renderFrame();
    }

    // --- 4. 16-Bit PCM WAV Encoder ---
    function encodeWAV(samples, sampleRate) {
        const buffer = new ArrayBuffer(44 + samples.length * 2);
        const view = new DataView(buffer);

        function writeString(view, offset, string) {
            for (let i = 0; i < string.length; i++) {
                view.setUint8(offset + i, string.charCodeAt(i));
            }
        }

        /* RIFF identifier */
        writeString(view, 0, 'RIFF');
        /* RIFF chunk length */
        view.setUint32(4, 36 + samples.length * 2, true);
        /* RIFF type */
        writeString(view, 8, 'WAVE');
        /* format chunk identifier */
        writeString(view, 12, 'fmt ');
        /* format chunk length */
        view.setUint32(16, 16, true);
        /* sample format (raw PCM) */
        view.setUint16(20, 1, true);
        /* channel count (mono) */
        view.setUint16(22, 1, true);
        /* sample rate */
        view.setUint32(24, sampleRate, true);
        /* byte rate (sampleRate * 2) */
        view.setUint32(28, sampleRate * 2, true);
        /* block align (2) */
        view.setUint16(32, 2, true);
        /* bits per sample (16) */
        view.setUint16(34, 16, true);
        /* data chunk identifier */
        writeString(view, 36, 'data');
        /* data chunk length */
        view.setUint32(40, samples.length * 2, true);

        // Float32 to Int16 PCM conversion
        let offset = 44;
        for (let i = 0; i < samples.length; i++) {
            const s = Math.max(-1, Math.min(1, samples[i]));
            view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
            offset += 2;
        }

        return new Blob([view], { type: 'audio/wav' });
    }

    // --- 5. Live Microphone Recording Logic ---
    btnRecordAction.addEventListener('click', async () => {
        if (isRecording) {
            stopRecording();
        } else {
            await startRecording();
        }
    });

    async function startRecording() {
        try {
            pcmBuffers = [];
            mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
            
            setupOscilloscope(mediaStream);
            isRecording = true;

            // UI updates for recording state
            btnRecordAction.classList.add('recording');
            btnRecordText.textContent = 'Stop Recording';
            canvasPlaceholder.style.opacity = '0';
            timerStatusHintEl.textContent = 'Hold sustained "Aaaaaah"...';

            // Start 3-second countdown timer
            let elapsed = 0.0;
            recProgressBar.style.width = '0%';
            recordingTimerEl.textContent = '00:03.0';

            recCountdownTimer = setInterval(() => {
                elapsed += 0.1;
                const remaining = Math.max(0, RECORDING_DURATION - elapsed);
                const progressPct = Math.min(100, (elapsed / RECORDING_DURATION) * 100);

                recProgressBar.style.width = `${progressPct}%`;
                recordingTimerEl.textContent = `00:0${remaining.toFixed(1)}`;

                if (elapsed >= RECORDING_DURATION) {
                    stopRecording();
                }
            }, 100);

        } catch (err) {
            alert(`Microphone Access Error: ${err.message}. Please check browser permissions.`);
        }
    }

    function stopRecording() {
        if (!isRecording) return;
        isRecording = false;
        clearInterval(recCountdownTimer);

        if (scriptProcessor) {
            scriptProcessor.disconnect();
            scriptProcessor = null;
        }

        if (mediaStream) {
            mediaStream.getTracks().forEach(track => track.stop());
        }

        btnRecordAction.classList.remove('recording');
        btnRecordText.textContent = 'Start Recording (Hold "Aaaah")';
        timerStatusHintEl.textContent = 'Phonation sample captured. Processing...';
        canvasPlaceholder.style.opacity = '1';

        // Merge raw PCM buffers into float32 array
        let totalLength = pcmBuffers.reduce((acc, buf) => acc + buf.length, 0);
        let mergedSamples = new Float32Array(totalLength);
        let offset = 0;
        for (let buf of pcmBuffers) {
            mergedSamples.set(buf, offset);
            offset += buf.length;
        }

        const currentSr = audioContext ? audioContext.sampleRate : 22050;
        const wavBlob = encodeWAV(mergedSamples, currentSr);

        // Preview audio player
        const audioUrl = URL.createObjectURL(wavBlob);
        audioPreviewEl.src = audioUrl;
        previewWrapper.classList.remove('hidden');

        // Send genuine WAV blob to backend API
        uploadAndAnalyzeAudio(wavBlob, "live_recording.wav");
    }

    // --- 6. File Upload Logic ---
    btnBrowseTrigger.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleSelectedFile(e.target.files[0]);
        }
    });

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleSelectedFile(e.dataTransfer.files[0]);
        }
    });

    function handleSelectedFile(file) {
        selectedFile = file;
        selectedFileName.textContent = file.name;
        fileInfoContainer.classList.remove('hidden');
    }

    btnAnalyzeFile.addEventListener('click', () => {
        if (selectedFile) {
            uploadAndAnalyzeAudio(selectedFile, selectedFile.name);
        }
    });

    // --- 7. API Ingestion & Prediction ---
    async function uploadAndAnalyzeAudio(fileOrBlob, filename) {
        // Show Loading State
        stateIdle.classList.add('hidden');
        stateReport.classList.add('hidden');
        stateLoading.classList.remove('hidden');

        loadingStatusText.textContent = 'Extracting 26 Librosa acoustic features...';

        const formData = new FormData();
        formData.append('audio', fileOrBlob, filename);

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!data.success) {
                throw new Error(data.error || 'Server processing error');
            }

            // Render Results
            renderDiagnosticReport(data);

        } catch (err) {
            let msg = err.message || 'Server processing error';
            if (msg.includes('Failed to fetch') || msg.includes('NetworkError')) {
                msg = 'Could not connect to backend server at http://localhost:5000.\nPlease ensure the server is running by executing "py app.py" in your terminal.';
            }
            alert(`Analysis Error: ${msg}`);
            stateLoading.classList.add('hidden');
            stateIdle.classList.remove('hidden');
        }
    }

    // --- 8. Diagnostic Report & Arch Gauge Needle Animation ---
    function renderDiagnosticReport(data) {
        stateLoading.classList.add('hidden');
        stateReport.classList.remove('hidden');

        const prediction = data.prediction; // 0 or 1
        const probabilities = data.probabilities || { healthy: 0.5, parkinsons: 0.5 };
        const features = data.features || data.metrics || {};

        const riskPercentage = Math.round(probabilities.parkinsons * 1000) / 10; // e.g., 84.5%
        const isHighRisk = prediction === 1 || probabilities.parkinsons >= 0.45;

        // 1. Animate Arch Gauge Needle (-90deg to +90deg)
        const needleAngle = -90 + (riskPercentage / 100.0) * 180;
        gaugeNeedleGroup.style.transform = `rotate(${needleAngle}deg)`;

        // 2. Animate Arc Dashoffset (251.2 total stroke-dasharray)
        const arcOffset = 251.2 - (riskPercentage / 100.0) * 251.2;
        gaugeFillArc.style.strokeDashoffset = arcOffset;

        // 3. Score & Badge Text
        riskScoreEl.textContent = `${riskPercentage.toFixed(1)}%`;

        if (isHighRisk) {
            diagnosisLabelEl.textContent = 'HIGH RISK / PARKINSONIAN BIOMARKERS';
            diagnosisLabelEl.className = 'badge-status high-risk';
            diagnosisDescEl.textContent = 'Acoustic biomarkers indicate vocal tremor, pitch micro-instability, and elevated noise density.';
        } else {
            diagnosisLabelEl.textContent = 'LOW RISK / HEALTHY';
            diagnosisLabelEl.className = 'badge-status low-risk';
            diagnosisDescEl.textContent = 'Acoustic parameters align with normal vocal tract stability and harmonic stability.';
        }

        // 4. Probability Bars
        const healthyPct = (probabilities.healthy * 100).toFixed(1);
        const pdPct = (probabilities.parkinsons * 100).toFixed(1);

        probHealthyValEl.textContent = `${healthyPct}%`;
        probHealthyBarEl.style.width = `${healthyPct}%`;

        probPdValEl.textContent = `${pdPct}%`;
        probPdBarEl.style.width = `${pdPct}%`;

        // 5. Render Feature Cards
        renderFeatureBreakdown(features);
    }

    // --- 9. Render Acoustic Feature Cards & MFCC Canvas Spectrum ---
    function renderFeatureBreakdown(features) {
        // Card 1: MFCC Spectrum Canvas
        if (features.mfccs && Array.isArray(features.mfccs)) {
            drawMfccCanvas(features.mfccs);
        }

        // Card 2: Pitch & Tonality
        const chroma = features.chroma_stft !== undefined ? features.chroma_stft : '--';
        const centroid = features.spectral_centroid_hz !== undefined ? features.spectral_centroid_hz : '--';

        metricChromaEl.textContent = chroma;
        metricCentroidEl.textContent = `${centroid} Hz`;

        if (chroma > 0.45 || centroid > 2200) {
            evalPitchEl.textContent = 'Micro-Instability Detected';
            evalPitchEl.className = 'eval-pill abnormal';
        } else {
            evalPitchEl.textContent = 'Stable Pitch Pattern';
            evalPitchEl.className = 'eval-pill normal';
        }

        // Card 3: Voice Brightness & Bandwidth
        const rolloff = features.spectral_rolloff_hz !== undefined ? features.spectral_rolloff_hz : '--';
        const bandwidth = features.spectral_bandwidth_hz !== undefined ? features.spectral_bandwidth_hz : '--';

        metricRolloffEl.textContent = `${rolloff} Hz`;
        metricBandwidthEl.textContent = `${bandwidth} Hz`;

        // Scale bar ratio (0 - 5000 Hz)
        const pctBrightness = Math.min(100, Math.max(10, (rolloff / 5000) * 100));
        barBrightnessEl.style.width = `${pctBrightness}%`;

        // Card 4: Breathing Noise & Energy
        const zcr = features.zero_crossing_rate !== undefined ? features.zero_crossing_rate : '--';
        const rms = features.rms_energy !== undefined ? features.rms_energy : '--';

        metricZcrEl.textContent = zcr;
        metricRmsEl.textContent = rms;

        if (zcr > 0.07 || rms < 0.015) {
            evalNoiseEl.textContent = 'Elevated Breathiness';
            evalNoiseEl.className = 'eval-pill abnormal';
        } else {
            evalNoiseEl.textContent = 'Normal HNR / Low Noise';
            evalNoiseEl.className = 'eval-pill normal';
        }
    }

    // Render 20-Bar MFCC Canvas Spectrum
    function drawMfccCanvas(mfccs) {
        if (!mfccCanvas) return;
        const ctx = mfccCanvas.getContext('2d');
        const width = mfccCanvas.width = mfccCanvas.parentElement.clientWidth;
        const height = mfccCanvas.height = mfccCanvas.parentElement.clientHeight;

        ctx.clearRect(0, 0, width, height);

        const numBars = mfccs.length; // 20
        const padding = 3;
        const barWidth = (width - (numBars + 1) * padding) / numBars;

        // Find min/max for normalization
        let minVal = Math.min(...mfccs);
        let maxVal = Math.max(...mfccs);
        if (minVal === maxVal) { minVal = -100; maxVal = 100; }

        for (let i = 0; i < numBars; i++) {
            const val = mfccs[i];
            
            // Normalize bar height between 10% and 90% of canvas height
            const normalizedRatio = (val - minVal) / (maxVal - minVal);
            const barHeight = Math.max(6, normalizedRatio * (height - 20));
            
            const x = padding + i * (barWidth + padding);
            const y = height - barHeight - 12;

            let color = val >= 0 ? '#38bdf8' : '#e3b341';
            if (i === 0) color = '#58a6ff'; // MFCC 1 baseline

            ctx.fillStyle = color;
            ctx.shadowBlur = 6;
            ctx.shadowColor = color;

            ctx.beginPath();
            ctx.roundRect(x, y, barWidth, barHeight, [3, 3, 0, 0]);
            ctx.fill();

            ctx.shadowBlur = 0;
            ctx.fillStyle = '#6e7681';
            ctx.font = '9px "JetBrains Mono", monospace';
            ctx.textAlign = 'center';
            if (i % 2 === 0) {
                ctx.fillText(`${i + 1}`, x + barWidth / 2, height - 2);
            }
        }
    }

    // Initialize Canvas Oscilloscope Idle Loop
    drawOscilloscope();
});
