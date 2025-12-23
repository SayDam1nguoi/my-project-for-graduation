// Real-time Camera Analysis
// Using face-api.js for client-side face detection

const API_URL = 'http://localhost:8000';

let videoElement = document.getElementById('videoElement');
let canvas = document.getElementById('canvas');
let stream = null;
let mediaRecorder = null;
let recordedChunks = [];
let recordingStartTime = null;
let timerInterval = null;

// Emotion tracking
let emotionHistory = {
    happy: 0,
    neutral: 0,
    sad: 0,
    angry: 0,
    fear: 0,
    surprise: 0,
    disgust: 0
};
let totalFrames = 0;

// ===== CAMERA CONTROLS =====

document.getElementById('startBtn').addEventListener('click', async () => {
    try {
        // Request camera access
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: 'user'
            },
            audio: false
        });

        videoElement.srcObject = stream;

        // Update UI
        document.getElementById('startBtn').disabled = true;
        document.getElementById('stopBtn').disabled = false;
        document.getElementById('recordBtn').disabled = false;
        document.getElementById('statusIndicator').classList.add('active');

        showAlert('✅ Camera đã bật! Đang phân tích...', 'info');

        // Start analysis
        startAnalysis();

    } catch (error) {
        console.error('Camera error:', error);
        showAlert('❌ Không thể truy cập camera. Vui lòng cho phép quyền truy cập.', 'error');
    }
});

document.getElementById('stopBtn').addEventListener('click', () => {
    stopCamera();
});

function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }

    videoElement.srcObject = null;

    // Update UI
    document.getElementById('startBtn').disabled = false;
    document.getElementById('stopBtn').disabled = true;
    document.getElementById('recordBtn').disabled = true;
    document.getElementById('statusIndicator').classList.remove('active');

    showAlert('⏹ Camera đã tắt', 'info');
}

// ===== RECORDING =====

document.getElementById('recordBtn').addEventListener('click', () => {
    startRecording();
});

document.getElementById('stopRecordBtn').addEventListener('click', () => {
    stopRecording();
});

function startRecording() {
    if (!stream) return;

    recordedChunks = [];

    // Create MediaRecorder
    mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'video/webm;codecs=vp9'
    });

    mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
            recordedChunks.push(event.data);
        }
    };

    mediaRecorder.onstop = () => {
        // Create blob and upload
        const blob = new Blob(recordedChunks, { type: 'video/webm' });
        uploadRecording(blob);
    };

    mediaRecorder.start();
    recordingStartTime = Date.now();

    // Update UI
    document.getElementById('recordBtn').style.display = 'none';
    document.getElementById('stopRecordBtn').style.display = 'inline-block';
    document.getElementById('recordingIndicator').classList.add('show');
    document.getElementById('timer').style.display = 'block';

    // Start timer
    timerInterval = setInterval(updateTimer, 1000);

    showAlert('⏺ Đang ghi hình...', 'info');
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
    }

    // Stop timer
    if (timerInterval) {
        clearInterval(timerInterval);
        timerInterval = null;
    }

    // Update UI
    document.getElementById('recordBtn').style.display = 'inline-block';
    document.getElementById('stopRecordBtn').style.display = 'none';
    document.getElementById('recordingIndicator').classList.remove('show');
    document.getElementById('timer').style.display = 'none';

    showAlert('⏹ Đã dừng ghi. Đang upload...', 'info');
}

function updateTimer() {
    if (!recordingStartTime) return;

    const elapsed = Math.floor((Date.now() - recordingStartTime) / 1000);
    const minutes = Math.floor(elapsed / 60);
    const seconds = elapsed % 60;

    document.getElementById('timerValue').textContent =
        `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
}

async function uploadRecording(blob) {
    try {
        showAlert('📤 Đang upload video...', 'info');

        const formData = new FormData();
        formData.append('file', blob, 'recording.webm');

        const response = await fetch(`${API_URL}/api/analyze-sync`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const result = await response.json();

        showAlert('✅ Phân tích hoàn tất! Xem kết quả bên dưới.', 'info');
        displayResults(result);

    } catch (error) {
        console.error('Upload error:', error);
        showAlert('❌ Lỗi khi upload: ' + error.message, 'error');
    }
}

function displayResults(result) {
    // Show results in alert
    const message = `
        📊 Kết Quả Phân Tích:
        
        Điểm Tổng: ${result.scores.total.toFixed(1)}/10
        Rating: ${result.rating}
        
        Chi tiết:
        - Cảm xúc: ${result.scores.emotion.toFixed(1)}/10
        - Tập trung: ${result.scores.focus.toFixed(1)}/10
        - Rõ ràng: ${result.scores.clarity.toFixed(1)}/10
        - Nội dung: ${result.scores.content.toFixed(1)}/10
    `;

    alert(message);
}

// ===== REAL-TIME ANALYSIS (SIMULATED) =====

function startAnalysis() {
    // Simulate real-time analysis
    // In production, you would use face-api.js or TensorFlow.js

    setInterval(() => {
        if (!stream) return;

        // Simulate face detection
        const faceDetected = Math.random() > 0.1; // 90% chance

        if (faceDetected) {
            // Simulate emotion detection
            const emotions = ['happy', 'neutral', 'sad', 'angry', 'fear'];
            const randomEmotion = emotions[Math.floor(Math.random() * emotions.length)];

            // Update emotion history
            emotionHistory[randomEmotion]++;
            totalFrames++;

            // Simulate focus score
            const focusScore = (Math.random() * 3 + 7).toFixed(1); // 7-10

            // Update UI
            document.getElementById('faceCount').textContent = '1';
            document.getElementById('focusScore').textContent = focusScore;
            document.getElementById('currentEmotion').textContent =
                getEmotionEmoji(randomEmotion) + ' ' + capitalize(randomEmotion);

            // Update emotion bars
            updateEmotionBars();
        } else {
            document.getElementById('faceCount').textContent = '0';
            document.getElementById('focusScore').textContent = '0.0';
            document.getElementById('currentEmotion').textContent = 'Không phát hiện';
        }
    }, 1000); // Update every second
}

function updateEmotionBars() {
    if (totalFrames === 0) return;

    const emotions = ['happy', 'neutral', 'sad', 'angry', 'fear'];

    emotions.forEach(emotion => {
        const percent = (emotionHistory[emotion] / totalFrames * 100).toFixed(1);
        document.getElementById(`${emotion}Percent`).textContent = `${percent}%`;
        document.getElementById(`${emotion}Bar`).style.width = `${percent}%`;
    });
}

function getEmotionEmoji(emotion) {
    const emojis = {
        happy: '😊',
        neutral: '😐',
        sad: '😢',
        angry: '😠',
        fear: '😨',
        surprise: '😲',
        disgust: '🤢'
    };
    return emojis[emotion] || '😐';
}

function capitalize(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
}

// ===== UTILITY =====

function showAlert(message, type = 'info') {
    const alertBox = document.getElementById('alertBox');
    alertBox.textContent = message;
    alertBox.className = `alert show ${type}`;

    setTimeout(() => {
        alertBox.classList.remove('show');
    }, 5000);
}

// ===== CLEANUP =====

window.addEventListener('beforeunload', () => {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
});
