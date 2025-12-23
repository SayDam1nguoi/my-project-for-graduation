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

// ===== TAB 2: VIDEO TRANSCRIPTION =====
setupFileInput('videoInput', 'videoFileName', 'videoTranscribeBtn', 'videoUploadArea');

document.getElementById('videoTranscribeBtn').addEventListener('click', async () => {
    const input = document.getElementById('videoInput');
    const file = input.files[0];
    if (!file) return;

    showStatus('videoStatus', '🔄 Đang chuyển đổi video... (có thể mất vài phút)', 'info');
    document.getElementById('videoTranscript').style.display = 'none';

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

        // Extract transcript from details
        const transcript = result.details?.clarity?.transcription_text ||
            result.details?.content?.transcript ||
            'Không có transcript';

        // Display transcript
        document.getElementById('videoTranscriptText').textContent = transcript;
        document.getElementById('videoTranscript').style.display = 'block';
        showStatus('videoStatus', '✅ Chuyển đổi hoàn tất!', 'success');

    } catch (error) {
        console.error('Error:', error);
        showStatus('videoStatus', `❌ Lỗi: ${error.message}`, 'error');
    }
});

// ===== TAB 3: AUDIO TRANSCRIPTION =====
setupFileInput('audioInput', 'audioFileName', 'audioTranscribeBtn', 'audioUploadArea');

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

// Load face-api.js models
async function loadFaceApiModels() {
    if (faceApiModelsLoaded) return true;

    try {
        showStatus('cameraStatusMsg', '⏳ Đang tải AI models... (chỉ lần đầu)', 'info');

        const MODEL_URL = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.12/model';

        await Promise.all([
            faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
            faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL)
        ]);

        faceApiModelsLoaded = true;
        console.log('✅ Face-api.js models loaded');
        return true;
    } catch (error) {
        console.error('❌ Failed to load models:', error);
        showStatus('cameraStatusMsg', '⚠️ Không thể tải AI models. Sẽ dùng simulated detection.', 'warning');
        return false;
    }
}

// Start Camera
document.getElementById('cameraStartBtn').addEventListener('click', async () => {
    try {
        // Load models first
        showStatus('cameraStatusMsg', '⏳ Đang khởi động camera...', 'info');
        await loadFaceApiModels();

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

        // Start real-time face detection
        startRealTimeFaceDetection();

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

    // Update UI
    document.getElementById('cameraStartBtn').disabled = false;
    document.getElementById('cameraStopBtn').disabled = true;
    document.getElementById('cameraRecordBtn').disabled = true;
    document.getElementById('cameraStatus').textContent = 'Chưa bật';
    document.getElementById('cameraStatus').style.color = '#888';
    document.getElementById('cameraFaceCount').textContent = '0';
    document.getElementById('cameraEmotion').textContent = '-';

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
    // Show results in alert
    const message = `📊 Kết Quả Phân Tích Camera:

Điểm Tổng: ${result.scores.total.toFixed(1)}/10
Rating: ${result.rating}

Chi tiết:
- Cảm xúc: ${result.scores.emotion.toFixed(1)}/10
- Tập trung: ${result.scores.focus.toFixed(1)}/10
- Rõ ràng: ${result.scores.clarity.toFixed(1)}/10
- Nội dung: ${result.scores.content.toFixed(1)}/10`;

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

            } else {
                // No face detected
                const ctx = canvas.getContext('2d');
                ctx.clearRect(0, 0, canvas.width, canvas.height);

                document.getElementById('cameraFaceCount').textContent = '0';
                document.getElementById('cameraEmotion').textContent = '-';
                document.getElementById('cameraEmotion').style.fontSize = '1.5em';
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
