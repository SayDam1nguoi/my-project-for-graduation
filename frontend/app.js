// Interview Analysis System - Full Features
// JavaScript for all tabs

const API_URL = 'http://localhost:8000';

// ===== TAB SWITCHING =====
function switchTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });

    // Remove active from all buttons
    document.querySelectorAll('.tab-button').forEach(btn => {
        btn.classList.remove('active');
    });

    // Show selected tab
    document.getElementById(`tab-${tabName}`).classList.add('active');

    // Activate button
    event.target.classList.add('active');
}

// ===== UTILITY FUNCTIONS =====
function showStatus(elementId, message, type = 'info') {
    const statusEl = document.getElementById(elementId);
    statusEl.textContent = message;
    statusEl.className = `status-message show ${type}`;
}

function hideStatus(elementId) {
    const statusEl = document.getElementById(elementId);
    statusEl.classList.remove('show');
}

function setupFileInput(inputId, fileNameId, buttonId, uploadAreaId) {
    const input = document.getElementById(inputId);
    const fileName = document.getElementById(fileNameId);
    const button = document.getElementById(buttonId);
    const uploadArea = document.getElementById(uploadAreaId);

    // File input change
    input.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            fileName.textContent = `📹 ${file.name}`;
            button.disabled = false;
        }
    });

    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            input.files = files;
            fileName.textContent = `📹 ${files[0].name}`;
            button.disabled = false;
        }
    });
}

// ===== TAB 1: EMOTION RECOGNITION =====
setupFileInput('emotionVideoInput', 'emotionFileName', 'emotionAnalyzeBtn', 'emotionUploadArea');

document.getElementById('emotionAnalyzeBtn').addEventListener('click', async () => {
    const input = document.getElementById('emotionVideoInput');
    const file = input.files[0];
    if (!file) return;

    showStatus('emotionStatus', '🔄 Đang phân tích cảm xúc... (có thể mất vài phút)', 'info');
    document.getElementById('emotionResults').style.display = 'none';

    try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${API_URL}/api/analyze-sync`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const result = await response.json();

        // Display emotion results
        displayEmotionResults(result);

        // Also display focus details if available
        if (result.details && result.details.focus) {
            displayFocusDetails(result.details.focus);
        }

        showStatus('emotionStatus', '✅ Phân tích hoàn tất!', 'success');

    } catch (error) {
        console.error('Error:', error);
        showStatus('emotionStatus', `❌ Lỗi: ${error.message}`, 'error');
    }
});

function displayEmotionResults(result) {
    const resultsDiv = document.getElementById('emotionResults');
    const gridDiv = document.getElementById('emotionResultsGrid');

    gridDiv.innerHTML = `
        <div class="result-card">
            <h3>😊 Điểm Cảm Xúc</h3>
            <div class="score">${result.scores.emotion.toFixed(1)}</div>
        </div>
        <div class="result-card">
            <h3>👁️ Điểm Tập Trung</h3>
            <div class="score">${result.scores.focus.toFixed(1)}</div>
        </div>
        <div class="result-card">
            <h3>📊 Điểm Tổng</h3>
            <div class="score">${result.scores.total.toFixed(1)}</div>
        </div>
        <div class="result-card">
            <h3>⭐ Đánh Giá</h3>
            <div class="score" style="font-size: 1.5em;">${result.rating}</div>
        </div>
    `;

    resultsDiv.style.display = 'block';
}

function displayFocusDetails(focusDetails) {
    // Display focus details if available
    if (!focusDetails) return;

    const resultsDiv = document.getElementById('emotionResults');

    // Add focus details section
    const focusSection = document.createElement('div');
    focusSection.style.marginTop = '20px';
    focusSection.style.padding = '15px';
    focusSection.style.background = '#252525';
    focusSection.style.borderRadius = '10px';

    focusSection.innerHTML = `
        <h3 style="color: #667eea; margin-bottom: 15px;">📊 Chi Tiết Tập Trung</h3>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
            <div>
                <strong>⏱️ Thời gian tập trung:</strong><br>
                ${focusDetails.focused_time || 0}s (${focusDetails.focused_rate || 0}%)
            </div>
            <div>
                <strong>⚠️ Thời gian mất tập trung:</strong><br>
                ${focusDetails.distracted_time || 0}s (${focusDetails.distracted_rate || 0}%)
            </div>
            <div>
                <strong>🔢 Số lần mất tập trung:</strong><br>
                ${focusDetails.distracted_count || 0} lần
            </div>
            <div>
                <strong>📈 Điểm trung bình:</strong><br>
                ${focusDetails.average_attention || 0}/10
            </div>
        </div>
    `;

    resultsDiv.appendChild(focusSection);
}

// ===== TAB 2: VIDEO TRANSCRIPTION =====
setupFileInput('videoInput', 'videoFileName', 'videoTranscribeBtn', 'videoUploadArea');

document.getElementById('videoTranscribeBtn').addEventListener('click', async () => {
    const input = document.getElementById('videoInput');
    const file = input.files[0];
    if (!file) return;

    showStatus('videoStatus', '🔄 Đang chuyển đổi audio trong video sang text... (có thể mất vài phút)', 'info');
    document.getElementById('videoTranscript').style.display = 'none';

    try {
        const formData = new FormData();
        formData.append('file', file);

        // Call transcription endpoint
        const response = await fetch(`${API_URL}/api/transcribe-video`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const result = await response.json();

        // Display transcript
        const transcriptText = result.transcript_with_timestamps || result.transcript || 'Không có transcript';
        const wordCount = result.word_count || 0;
        const duration = result.duration || 0;
        const langDisplay = result.language_display || result.language || 'unknown';
        const segments = result.segments || 0;

        document.getElementById('videoTranscriptText').innerHTML = `
            <div style="margin-bottom: 15px; padding: 10px; background: #252525; border-radius: 5px;">
                <strong>📊 Thông tin:</strong><br>
                Số từ: ${wordCount} | Thời lượng: ${duration.toFixed(1)}s | Ngôn ngữ: ${langDisplay} | Segments: ${segments}
            </div>
            <div style="white-space: pre-wrap;">${transcriptText}</div>
        `;

        document.getElementById('videoTranscript').style.display = 'block';
        showStatus('videoStatus', '✅ Chuyển đổi hoàn tất!', 'success');

    } catch (error) {
        console.error('Error:', error);
        showStatus('videoStatus', `❌ Lỗi: ${error.message}. Kiểm tra video có audio không?`, 'error');
    }
});

// ===== TAB 3: AUDIO TRANSCRIPTION =====
setupFileInput('audioInput', 'audioFileName', 'audioTranscribeBtn', 'audioUploadArea');

document.getElementById('audioTranscribeBtn').addEventListener('click', async () => {
    const input = document.getElementById('audioInput');
    const file = input.files[0];
    if (!file) return;

    showStatus('audioStatus', '🔄 Đang chuyển đổi audio sang text...', 'info');
    document.getElementById('audioTranscript').style.display = 'none';

    try {
        const formData = new FormData();
        formData.append('file', file);

        // Call audio transcription endpoint
        const response = await fetch(`${API_URL}/api/transcribe-audio`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const result = await response.json();

        // Display transcript
        const transcriptText = result.transcript || 'Không có transcript';
        const wordCount = result.word_count || 0;

        document.getElementById('audioTranscriptText').innerHTML = `
            <div style="margin-bottom: 15px; padding: 10px; background: #252525; border-radius: 5px;">
                <strong>📊 Thông tin:</strong><br>
                Số từ: ${wordCount}
            </div>
            <div style="white-space: pre-wrap;">${transcriptText}</div>
        `;

        document.getElementById('audioTranscript').style.display = 'block';
        showStatus('audioStatus', '✅ Chuyển đổi hoàn tất!', 'success');

    } catch (error) {
        console.error('Error:', error);
        showStatus('audioStatus', `❌ Lỗi: ${error.message}`, 'error');
    }
});

document.getElementById('audioTranscribeBtn').addEventListener('click', async () => {
    const input = document.getElementById('audioInput');
    const file = input.files[0];
    if (!file) return;

    showStatus('audioStatus', '🔄 Đang chuyển đổi audio... (có thể mất vài phút)', 'info');
    document.getElementById('audioTranscript').style.display = 'none';

    try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${API_URL}/api/analyze-sync`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const result = await response.json();

        // Extract transcript
        const transcript = result.details?.clarity?.transcription_text ||
            result.details?.content?.transcript ||
            'Không có transcript';

        // Display transcript
        document.getElementById('audioTranscriptText').textContent = transcript;
        document.getElementById('audioTranscript').style.display = 'block';
        showStatus('audioStatus', '✅ Chuyển đổi hoàn tất!', 'success');

    } catch (error) {
        console.error('Error:', error);
        showStatus('audioStatus', `❌ Lỗi: ${error.message}`, 'error');
    }
});

// Copy transcript function
function copyTranscript(type) {
    const textId = type === 'video' ? 'videoTranscriptText' : 'audioTranscriptText';
    const text = document.getElementById(textId).textContent;

    navigator.clipboard.writeText(text).then(() => {
        alert('✅ Đã copy transcript!');
    }).catch(err => {
        console.error('Copy failed:', err);
        alert('❌ Không thể copy');
    });
}

// ===== TAB 4: SCORE SUMMARY =====
setupFileInput('summaryVideoInput', 'summaryFileName', 'summaryAnalyzeBtn', 'summaryUploadArea');

// Weight controls
const weightInputs = ['weightEmotion', 'weightFocus', 'weightClarity', 'weightContent'];
weightInputs.forEach(id => {
    document.getElementById(id).addEventListener('input', updateTotalWeight);
});

function updateTotalWeight() {
    const total = weightInputs.reduce((sum, id) => {
        return sum + parseInt(document.getElementById(id).value || 0);
    }, 0);

    document.getElementById('totalWeight').textContent = total;

    // Highlight if not 100%
    const totalEl = document.getElementById('totalWeight');
    if (total !== 100) {
        totalEl.style.color = '#f5576c';
    } else {
        totalEl.style.color = '#43e97b';
    }
}

document.getElementById('summaryAnalyzeBtn').addEventListener('click', async () => {
    const input = document.getElementById('summaryVideoInput');
    const file = input.files[0];
    if (!file) return;

    // Check weights
    const total = weightInputs.reduce((sum, id) => {
        return sum + parseInt(document.getElementById(id).value || 0);
    }, 0);

    if (total !== 100) {
        showStatus('summaryStatus', '⚠️ Tổng trọng số phải bằng 100%!', 'warning');
        return;
    }

    showStatus('summaryStatus', '🔄 Đang phân tích toàn diện... (có thể mất vài phút)', 'info');
    document.getElementById('summaryResults').style.display = 'none';

    try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${API_URL}/api/analyze-sync`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const result = await response.json();

        // Get custom weights
        const weights = {
            emotion: parseInt(document.getElementById('weightEmotion').value) / 100,
            focus: parseInt(document.getElementById('weightFocus').value) / 100,
            clarity: parseInt(document.getElementById('weightClarity').value) / 100,
            content: parseInt(document.getElementById('weightContent').value) / 100
        };

        // Calculate custom total score
        const customTotal = (
            result.scores.emotion * weights.emotion +
            result.scores.focus * weights.focus +
            result.scores.clarity * weights.clarity +
            result.scores.content * weights.content
        );

        // Display results
        displaySummaryResults(result, customTotal);
        showStatus('summaryStatus', '✅ Phân tích hoàn tất!', 'success');

    } catch (error) {
        console.error('Error:', error);
        showStatus('summaryStatus', `❌ Lỗi: ${error.message}`, 'error');
    }
});

function displaySummaryResults(result, customTotal) {
    // Total score
    document.getElementById('summaryTotalScore').textContent = customTotal.toFixed(1);

    // Rating based on custom total
    let rating;
    if (customTotal >= 9.0) rating = 'XUẤT SẮC';
    else if (customTotal >= 8.0) rating = 'RẤT TỐT';
    else if (customTotal >= 7.0) rating = 'TỐT';
    else if (customTotal >= 6.0) rating = 'KHÁ';
    else if (customTotal >= 5.0) rating = 'TRUNG BÌNH';
    else rating = 'CẦN CẢI THIỆN';

    document.getElementById('summaryRating').textContent = rating;

    // Individual scores
    document.getElementById('summaryEmotionScore').textContent = result.scores.emotion.toFixed(1);
    document.getElementById('summaryFocusScore').textContent = result.scores.focus.toFixed(1);
    document.getElementById('summaryClarityScore').textContent = result.scores.clarity.toFixed(1);
    document.getElementById('summaryContentScore').textContent = result.scores.content.toFixed(1);

    // Show results
    document.getElementById('summaryResults').style.display = 'block';
}

function resetSummary() {
    document.getElementById('summaryVideoInput').value = '';
    document.getElementById('summaryFileName').textContent = '';
    document.getElementById('summaryAnalyzeBtn').disabled = true;
    document.getElementById('summaryResults').style.display = 'none';
    hideStatus('summaryStatus');
}

// ===== TAB 0: REAL-TIME CAMERA =====
let cameraStream = null;
let cameraMediaRecorder = null;
let cameraRecordedChunks = [];
let cameraRecordingStartTime = null;
let cameraTimerInterval = null;
let faceApiModelsLoaded = false;
let detectionInterval = null;

// Focus tracking variables
let cameraFocusHistory = [];
let cameraTotalFocusedTime = 0;
let cameraTotalDistractedTime = 0;
let cameraDistractedEvents = 0;
let cameraCurrentlyDistracted = false;
let cameraStartTime = null;

// Load face-api.js models
async function loadFaceApiModels() {
    if (faceApiModelsLoaded) return true;

    try {
        showStatus('cameraStatusMsg', '⏳ Đang tải AI models... (chỉ lần đầu, ~10-15 giây)', 'info');

        // Try multiple CDN sources
        const MODEL_URLS = [
            'https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.12/model',
            'https://justadudewhohacks.github.io/face-api.js/models'
        ];

        let loaded = false;
        for (const MODEL_URL of MODEL_URLS) {
            try {
                console.log(`Trying to load models from: ${MODEL_URL}`);

                await Promise.all([
                    faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
                    faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),  // CẦN CHO HEAD POSE
                    faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL)
                ]);

                loaded = true;
                console.log(`✅ Face-api.js models loaded from: ${MODEL_URL}`);
                break;
            } catch (err) {
                console.warn(`Failed to load from ${MODEL_URL}:`, err);
                continue;
            }
        }

        if (!loaded) {
            throw new Error('Could not load models from any CDN');
        }

        faceApiModelsLoaded = true;
        return true;
    } catch (error) {
        console.error('❌ Failed to load models:', error);
        showStatus('cameraStatusMsg', '❌ Không thể tải AI models. Vui lòng kiểm tra kết nối internet.', 'error');
        return false;
    }
}

// Start Camera
document.getElementById('cameraStartBtn').addEventListener('click', async () => {
    try {
        // Load models first
        showStatus('cameraStatusMsg', '⏳ Đang khởi động camera...', 'info');

        // Load models FIRST and WAIT
        const modelsLoaded = await loadFaceApiModels();
        if (!modelsLoaded) {
            throw new Error('Không thể tải AI models. Vui lòng thử lại.');
        }

        // Request camera access
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: 'user'
            },
            audio: true // Include audio for recording
        });

        const videoElement = document.getElementById('cameraVideo');
        videoElement.srcObject = cameraStream;

        // Wait for video to be ready
        await new Promise(resolve => {
            videoElement.onloadedmetadata = () => {
                resolve();
            };
        });

        // Update UI
        document.getElementById('cameraStartBtn').disabled = true;
        document.getElementById('cameraStopBtn').disabled = false;
        document.getElementById('cameraRecordBtn').disabled = false;
        document.getElementById('cameraStatus').textContent = 'Đang hoạt động';
        document.getElementById('cameraStatus').style.color = '#43e97b';

        showStatus('cameraStatusMsg', '✅ Camera đã bật! Đang phát hiện cảm xúc real-time...', 'success');

        // Start real-time face detection with LAUNCHER FORMULA
        // Wait a bit for video to stabilize
        setTimeout(() => {
            startRealTimeFaceDetection_Launcher();
        }, 500);

    } catch (error) {
        console.error('Camera error:', error);
        showStatus('cameraStatusMsg', '❌ Không thể truy cập camera. Vui lòng cho phép quyền truy cập.', 'error');
    }
});

// Stop Camera
document.getElementById('cameraStopBtn').addEventListener('click', () => {
    stopCamera();
});

function stopCamera() {
    // Stop detection
    if (detectionInterval) {
        clearInterval(detectionInterval);
        detectionInterval = null;
    }

    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }

    const videoElement = document.getElementById('cameraVideo');
    videoElement.srcObject = null;

    // Clear canvas
    const canvas = document.getElementById('cameraCanvas');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Reset focus tracking
    cameraFocusHistory = [];
    cameraTotalFocusedTime = 0;
    cameraTotalDistractedTime = 0;
    cameraDistractedEvents = 0;
    cameraCurrentlyDistracted = false;
    cameraStartTime = null;

    // Update UI
    document.getElementById('cameraStartBtn').disabled = false;
    document.getElementById('cameraStopBtn').disabled = true;
    document.getElementById('cameraRecordBtn').disabled = true;
    document.getElementById('cameraStatus').textContent = 'Chưa bật';
    document.getElementById('cameraStatus').style.color = '#888';
    document.getElementById('cameraFaceCount').textContent = '0';
    document.getElementById('cameraEmotion').textContent = '-';
    document.getElementById('cameraFocusScore').textContent = '-';
    document.getElementById('cameraFocusedTime').textContent = '0s';
    document.getElementById('cameraDistractedTime').textContent = '0s';
    document.getElementById('cameraDistractedCount').textContent = '0';

    showStatus('cameraStatusMsg', '⏹ Camera đã tắt', 'info');
}

// Start Recording
document.getElementById('cameraRecordBtn').addEventListener('click', () => {
    startCameraRecording();
});

// Stop Recording
document.getElementById('cameraStopRecordBtn').addEventListener('click', () => {
    stopCameraRecording();
});

function startCameraRecording() {
    if (!cameraStream) return;

    cameraRecordedChunks = [];

    // Create MediaRecorder
    try {
        cameraMediaRecorder = new MediaRecorder(cameraStream, {
            mimeType: 'video/webm;codecs=vp9'
        });
    } catch (e) {
        // Fallback to default codec
        cameraMediaRecorder = new MediaRecorder(cameraStream);
    }

    cameraMediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
            cameraRecordedChunks.push(event.data);
        }
    };

    cameraMediaRecorder.onstop = () => {
        // Create blob and upload
        const blob = new Blob(cameraRecordedChunks, { type: 'video/webm' });
        uploadCameraRecording(blob);
    };

    cameraMediaRecorder.start();
    cameraRecordingStartTime = Date.now();

    // Update UI
    document.getElementById('cameraRecordBtn').style.display = 'none';
    document.getElementById('cameraStopRecordBtn').style.display = 'inline-block';
    document.getElementById('cameraRecordingIndicator').style.display = 'block';
    document.getElementById('cameraTimer').style.display = 'block';

    // Start timer
    cameraTimerInterval = setInterval(updateCameraTimer, 1000);

    showStatus('cameraStatusMsg', '⏺ Đang ghi hình...', 'info');
}

function stopCameraRecording() {
    if (cameraMediaRecorder && cameraMediaRecorder.state !== 'inactive') {
        cameraMediaRecorder.stop();
    }

    // Stop timer
    if (cameraTimerInterval) {
        clearInterval(cameraTimerInterval);
        cameraTimerInterval = null;
    }

    // Update UI
    document.getElementById('cameraRecordBtn').style.display = 'inline-block';
    document.getElementById('cameraStopRecordBtn').style.display = 'none';
    document.getElementById('cameraRecordingIndicator').style.display = 'none';

    showStatus('cameraStatusMsg', '⏹ Đã dừng ghi. Đang upload và phân tích...', 'info');
}

function updateCameraTimer() {
    if (!cameraRecordingStartTime) return;

    const elapsed = Math.floor((Date.now() - cameraRecordingStartTime) / 1000);
    const minutes = Math.floor(elapsed / 60);
    const seconds = elapsed % 60;

    document.getElementById('cameraTimerValue').textContent =
        `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
}

async function uploadCameraRecording(blob) {
    try {
        showStatus('cameraStatusMsg', '📤 Đang upload và phân tích video...', 'info');

        const formData = new FormData();
        formData.append('file', blob, 'camera-recording.webm');

        const response = await fetch(`${API_URL}/api/analyze-sync`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const result = await response.json();

        showStatus('cameraStatusMsg', '✅ Phân tích hoàn tất!', 'success');
        displayCameraResults(result);

    } catch (error) {
        console.error('Upload error:', error);
        showStatus('cameraStatusMsg', `❌ Lỗi khi upload: ${error.message}`, 'error');
    }
}

function displayCameraResults(result) {
    // Extract focus details
    const focusDetails = result.details && result.details.focus ? result.details.focus : {};

    // Show results in alert with focus details
    const message = `📊 Kết Quả Phân Tích Camera:

Điểm Tổng: ${result.scores.total.toFixed(1)}/10
Rating: ${result.rating}

Chi tiết:
- Cảm xúc: ${result.scores.emotion.toFixed(1)}/10
- Tập trung: ${result.scores.focus.toFixed(1)}/10
- Rõ ràng: ${result.scores.clarity.toFixed(1)}/10
- Nội dung: ${result.scores.content.toFixed(1)}/10

📊 Chi tiết tập trung:
- Thời gian tập trung: ${focusDetails.focused_time || 0}s (${focusDetails.focused_rate || 0}%)
- Thời gian mất tập trung: ${focusDetails.distracted_time || 0}s (${focusDetails.distracted_rate || 0}%)
- Số lần mất tập trung: ${focusDetails.distracted_count || 0} lần
- Điểm trung bình: ${focusDetails.average_attention || 0}/10`;

    alert(message);
}

// Real-time face detection with face-api.js
async function startRealTimeFaceDetection() {
    const video = document.getElementById('cameraVideo');
    const canvas = document.getElementById('cameraCanvas');

    // Set canvas size to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const displaySize = { width: video.videoWidth, height: video.videoHeight };
    faceapi.matchDimensions(canvas, displaySize);

    // Start tracking time
    cameraStartTime = Date.now();

    // Detection loop
    detectionInterval = setInterval(async () => {
        if (!cameraStream) {
            clearInterval(detectionInterval);
            return;
        }

        try {
            // Detect faces with expressions
            const detections = await faceapi
                .detectAllFaces(video, new faceapi.TinyFaceDetectorOptions())
                .withFaceExpressions();

            // Calculate focus score (0-10)
            let focusScore = 0;

            if (detections && detections.length > 0) {
                // Clear canvas
                const ctx = canvas.getContext('2d');
                ctx.clearRect(0, 0, canvas.width, canvas.height);

                // Resize detections to match display
                const resizedDetections = faceapi.resizeResults(detections, displaySize);

                // Draw detections
                resizedDetections.forEach(detection => {
                    const box = detection.detection.box;

                    // Draw bounding box
                    ctx.strokeStyle = '#43e97b';
                    ctx.lineWidth = 3;
                    ctx.strokeRect(box.x, box.y, box.width, box.height);

                    // Get dominant emotion
                    const expressions = detection.expressions;
                    const dominantEmotion = Object.keys(expressions).reduce((a, b) =>
                        expressions[a] > expressions[b] ? a : b
                    );
                    const confidence = (expressions[dominantEmotion] * 100).toFixed(0);

                    // Draw emotion label
                    const emotionEmoji = getEmotionEmojiFromName(dominantEmotion);
                    const label = `${emotionEmoji} ${dominantEmotion} (${confidence}%)`;

                    ctx.fillStyle = '#43e97b';
                    ctx.font = 'bold 20px Arial';
                    ctx.fillText(label, box.x, box.y - 10);
                });

                // Update stats
                document.getElementById('cameraFaceCount').textContent = detections.length;

                // Show dominant emotion of first face
                const firstFace = detections[0];
                const expressions = firstFace.expressions;
                const dominantEmotion = Object.keys(expressions).reduce((a, b) =>
                    expressions[a] > expressions[b] ? a : b
                );
                const emotionEmoji = getEmotionEmojiFromName(dominantEmotion);

                document.getElementById('cameraEmotion').textContent = emotionEmoji;
                document.getElementById('cameraEmotion').style.fontSize = '2em';

                // Calculate focus score based on face position and size
                const box = firstFace.detection.box;
                const centerX = box.x + box.width / 2;
                const centerY = box.y + box.height / 2;
                const videoCenterX = displaySize.width / 2;
                const videoCenterY = displaySize.height / 2;

                // Calculate deviation from center (0-1)
                const deviationX = Math.abs(centerX - videoCenterX) / (displaySize.width / 2);
                const deviationY = Math.abs(centerY - videoCenterY) / (displaySize.height / 2);
                const maxDeviation = Math.max(deviationX, deviationY);

                // Calculate face size ratio (ideal: 0.3-0.5 of frame)
                const faceArea = box.width * box.height;
                const frameArea = displaySize.width * displaySize.height;
                const sizeRatio = faceArea / frameArea;

                // Focus score components (0-10 scale)
                const positionScore = (1 - Math.min(maxDeviation, 1)) * 10; // 10 if centered
                const sizeScore = sizeRatio > 0.1 && sizeRatio < 0.6 ? 10 : 5; // 10 if good size

                // Combined focus score
                focusScore = (positionScore * 0.7 + sizeScore * 0.3);

                // Track focused/distracted time
                if (focusScore >= 6.0) {
                    cameraTotalFocusedTime += 0.1; // 100ms interval
                    if (cameraCurrentlyDistracted) {
                        cameraCurrentlyDistracted = false;
                    }
                } else {
                    cameraTotalDistractedTime += 0.1;
                    if (!cameraCurrentlyDistracted) {
                        cameraCurrentlyDistracted = true;
                        cameraDistractedEvents++;
                    }
                }

            } else {
                // No face detected - count as distracted
                const ctx = canvas.getContext('2d');
                ctx.clearRect(0, 0, canvas.width, canvas.height);

                document.getElementById('cameraFaceCount').textContent = '0';
                document.getElementById('cameraEmotion').textContent = '-';
                document.getElementById('cameraEmotion').style.fontSize = '1.5em';

                focusScore = 0;
                cameraTotalDistractedTime += 0.1;
                if (!cameraCurrentlyDistracted) {
                    cameraCurrentlyDistracted = true;
                    cameraDistractedEvents++;
                }
            }

            // Update focus UI
            cameraFocusHistory.push(focusScore);
            if (cameraFocusHistory.length > 30) {
                cameraFocusHistory.shift(); // Keep last 30 samples (3 seconds)
            }

            const avgFocusScore = cameraFocusHistory.reduce((a, b) => a + b, 0) / cameraFocusHistory.length;
            document.getElementById('cameraFocusScore').textContent = avgFocusScore.toFixed(1);
            document.getElementById('cameraFocusedTime').textContent = `${cameraTotalFocusedTime.toFixed(1)}s`;
            document.getElementById('cameraDistractedTime').textContent = `${cameraTotalDistractedTime.toFixed(1)}s`;
            document.getElementById('cameraDistractedCount').textContent = cameraDistractedEvents;

            // Color code focus score
            const focusScoreEl = document.getElementById('cameraFocusScore');
            if (avgFocusScore >= 7.5) {
                focusScoreEl.style.color = '#43e97b'; // Green - focused
            } else if (avgFocusScore >= 6.0) {
                focusScoreEl.style.color = '#fee140'; // Yellow - slightly distracted
            } else {
                focusScoreEl.style.color = '#f5576c'; // Red - distracted
            }

        } catch (error) {
            console.error('Detection error:', error);
        }

    }, 100); // Detect every 100ms (10 FPS)
}

function getEmotionEmojiFromName(emotion) {
    const emojiMap = {
        'happy': '😊',
        'sad': '😢',
        'angry': '😠',
        'fearful': '😨',
        'disgusted': '🤢',
        'surprised': '😲',
        'neutral': '😐'
    };
    return emojiMap[emotion] || '😐';
}

// Cleanup camera on page unload
window.addEventListener('beforeunload', () => {
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
    }
    if (detectionInterval) {
        clearInterval(detectionInterval);
    }
});

// ===== CHECK API ON LOAD =====
window.addEventListener('load', async () => {
    try {
        const response = await fetch(`${API_URL}/health`);
        if (!response.ok) {
            throw new Error('API not responding');
        }
        console.log('✅ API is running');
    } catch (error) {
        console.error('❌ API not available:', error);
        alert('⚠️ Không thể kết nối với API.\n\nVui lòng chạy: python api/main.py');
    }
});


// ===== TAB 5: DUAL PERSON COMPARISON =====
let dualPerson1Stream = null;
let dualPerson2Stream = null;
let dualComparisonInterval = null;

document.getElementById('dualStartBtn').addEventListener('click', async () => {
    try {
        showStatus('dualStatus', '⏳ Đang khởi động camera và screen capture...', 'info');

        // Start camera for person 1
        dualPerson1Stream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
            audio: false
        });

        const video1 = document.getElementById('dualPerson1Video');
        video1.srcObject = dualPerson1Stream;

        // Start screen capture for person 2
        dualPerson2Stream = await navigator.mediaDevices.getDisplayMedia({
            video: { width: { ideal: 1280 }, height: { ideal: 720 } },
            audio: false
        });

        const canvas2 = document.getElementById('dualPerson2Canvas');
        const video2 = document.createElement('video');
        video2.srcObject = dualPerson2Stream;
        video2.play();

        // Draw screen capture to canvas
        const ctx2 = canvas2.getContext('2d');
        const drawScreen = () => {
            if (dualPerson2Stream && dualPerson2Stream.active) {
                canvas2.width = video2.videoWidth;
                canvas2.height = video2.videoHeight;
                ctx2.drawImage(video2, 0, 0, canvas2.width, canvas2.height);
                requestAnimationFrame(drawScreen);
            }
        };
        video2.onloadedmetadata = () => {
            drawScreen();
        };

        // Update UI
        document.getElementById('dualStartBtn').disabled = true;
        document.getElementById('dualStopBtn').disabled = false;
        document.getElementById('dualExportBtn').disabled = false;

        showStatus('dualStatus', '✅ Đang so sánh 2 người...', 'success');

        // Start comparison updates
        dualComparisonInterval = setInterval(updateDualComparison, 2000);

    } catch (error) {
        console.error('Dual person error:', error);
        showStatus('dualStatus', `❌ Lỗi: ${error.message}`, 'error');
    }
});

document.getElementById('dualStopBtn').addEventListener('click', () => {
    stopDualPerson();
});

document.getElementById('dualExportBtn').addEventListener('click', () => {
    alert('📊 Chức năng xuất báo cáo đang được phát triển!');
});

function stopDualPerson() {
    if (dualPerson1Stream) {
        dualPerson1Stream.getTracks().forEach(track => track.stop());
        dualPerson1Stream = null;
    }

    if (dualPerson2Stream) {
        dualPerson2Stream.getTracks().forEach(track => track.stop());
        dualPerson2Stream = null;
    }

    if (dualComparisonInterval) {
        clearInterval(dualComparisonInterval);
        dualComparisonInterval = null;
    }

    document.getElementById('dualStartBtn').disabled = false;
    document.getElementById('dualStopBtn').disabled = true;
    document.getElementById('dualExportBtn').disabled = true;

    showStatus('dualStatus', '⏹ Đã dừng so sánh', 'info');
}

function updateDualComparison() {
    // Simulated comparison (in real app, would use face-api.js)
    const emotions = ['happy', 'neutral', 'sad', 'surprised'];
    const person1Emotion = emotions[Math.floor(Math.random() * emotions.length)];
    const person2Emotion = emotions[Math.floor(Math.random() * emotions.length)];

    document.getElementById('dualPerson1Emotion').textContent = person1Emotion;
    document.getElementById('dualPerson2Emotion').textContent = person2Emotion;

    const comparisonText = `
        Người 1: ${person1Emotion} | Người 2: ${person2Emotion}
        ${person1Emotion === person2Emotion ? '✅ Cảm xúc giống nhau' : '⚠️ Cảm xúc khác nhau'}
    `;

    document.getElementById('dualComparisonResults').textContent = comparisonText;
}

// ===== TAB 7: VIDEO CALL =====
// Video call functionality is in separate videocall.html/videocall.js
// This tab just provides a link to open the video call interface

// ===== AUDIO RECORDING =====
let audioRecorder = null;
let audioRecordedChunks = [];
let audioRecordingStartTime = null;
let audioTimerInterval = null;

document.getElementById('audioStartRecordBtn').addEventListener('click', async () => {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

        audioRecorder = new MediaRecorder(stream);
        audioRecordedChunks = [];

        audioRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioRecordedChunks.push(event.data);
            }
        };

        audioRecorder.onstop = () => {
            const blob = new Blob(audioRecordedChunks, { type: 'audio/webm' });
            const url = URL.createObjectURL(blob);

            const audioPlayback = document.getElementById('audioPlayback');
            audioPlayback.src = url;
            audioPlayback.style.display = 'block';

            document.getElementById('audioPlayBtn').disabled = false;
        };

        audioRecorder.start();
        audioRecordingStartTime = Date.now();

        document.getElementById('audioStartRecordBtn').disabled = true;
        document.getElementById('audioStopRecordBtn').disabled = false;
        document.getElementById('audioRecordTimer').style.display = 'block';

        audioTimerInterval = setInterval(updateAudioTimer, 1000);

        showStatus('audioStatus', '⏺ Đang thu âm...', 'info');

    } catch (error) {
        console.error('Audio recording error:', error);
        showStatus('audioStatus', `❌ Lỗi: ${error.message}`, 'error');
    }
});

document.getElementById('audioStopRecordBtn').addEventListener('click', () => {
    if (audioRecorder && audioRecorder.state !== 'inactive') {
        audioRecorder.stop();
        audioRecorder.stream.getTracks().forEach(track => track.stop());
    }

    if (audioTimerInterval) {
        clearInterval(audioTimerInterval);
        audioTimerInterval = null;
    }

    document.getElementById('audioStartRecordBtn').disabled = false;
    document.getElementById('audioStopRecordBtn').disabled = true;
    document.getElementById('audioRecordTimer').style.display = 'none';

    showStatus('audioStatus', '✅ Đã dừng thu âm', 'success');
});

document.getElementById('audioPlayBtn').addEventListener('click', () => {
    const audioPlayback = document.getElementById('audioPlayback');
    audioPlayback.play();
});

function updateAudioTimer() {
    if (!audioRecordingStartTime) return;

    const elapsed = Math.floor((Date.now() - audioRecordingStartTime) / 1000);
    const minutes = Math.floor(elapsed / 60);
    const seconds = elapsed % 60;

    document.getElementById('audioRecordTimerValue').textContent =
        `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
}

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    stopDualPerson();

    if (audioRecorder && audioRecorder.state !== 'inactive') {
        audioRecorder.stop();
    }
});
