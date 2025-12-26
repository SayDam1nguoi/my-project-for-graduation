// Video Call with Emotion Analysis - OPTIMIZED VERSION
// WebRTC + Face-api.js with performance improvements

const CONFIG = {
    SIGNALING_SERVER: 'ws://localhost:8001',
    DETECTION_INTERVAL: 200,
    RECONNECT_DELAY: 3000,
    MAX_RECONNECT_ATTEMPTS: 5,
    ICE_SERVERS: [
        { urls: 'stun:stun.l.google.com:19302' },
        { urls: 'stun:stun1.l.google.com:19302' },
        { urls: 'stun:stun2.l.google.com:19302' }
    ]
};

// Centralized state management
const state = {
    ws: null,
    roomId: null,
    userId: null,
    peerConnection: null,
    localStream: null,
    remoteStream: null,
    faceApiLoaded: false,
    connectionStartTime: null,
    framesAnalyzed: 0,
    localEmotionHistory: {},
    remoteEmotionHistory: {},
    reconnectAttempts: 0,
    detectionIntervals: [],
    isJoining: false
};

// Initialize
window.addEventListener('load', () => {
    loadFaceApiModels();
    setupEventListeners();
});

async function loadFaceApiModels() {
    if (state.faceApiLoaded) return true;

    try {
        showStatus('⏳ Đang tải AI models...', 'info');

        const MODEL_URL = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.12/model';

        await Promise.race([
            Promise.all([
                faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
                faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL)
            ]),
            new Promise((_, reject) => setTimeout(() => reject(new Error('Timeout')), 30000))
        ]);

        state.faceApiLoaded = true;
        console.log('✅ Face-api.js models loaded');
        showStatus('✅ Sẵn sàng!', 'success');
        return true;

    } catch (error) {
        console.error('❌ Failed to load models:', error);
        showStatus('⚠️ Không thể tải AI models', 'error');
        setTimeout(loadFaceApiModels, 5000);
        return false;
    }
}

function setupEventListeners() {
    document.getElementById('roomIdInput')?.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') joinRoom();
    });
}

async function joinRoom() {
    if (state.isJoining) return;
    state.isJoining = true;

    const input = document.getElementById('roomIdInput').value.trim();

    if (input && !/^[a-zA-Z0-9_-]+$/.test(input)) {
        showStatus('❌ Room ID không hợp lệ', 'error');
        state.isJoining = false;
        return;
    }

    state.userId = 'user_' + Math.random().toString(36).substr(2, 9);

    try {
        if (input) {
            state.roomId = input;
        } else {
            const controller = new AbortController();
            setTimeout(() => controller.abort(), 5000);

            const response = await fetch('http://localhost:8001/api/create-room', {
                method: 'POST',
                signal: controller.signal
            });

            if (!response.ok) throw new Error('Server error');
            const data = await response.json();
            state.roomId = data.room_id;
        }

        state.localStream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } },
            audio: { echoCancellation: true, noiseSuppression: true }
        });

        document.getElementById('localVideo').srcObject = state.localStream;

        connectToSignalingServer();

        document.getElementById('roomSetup').style.display = 'none';
        document.getElementById('callInterface').style.display = 'block';
        document.getElementById('currentRoomId').textContent = state.roomId;

        showStatus(`✅ Đã tham gia: ${state.roomId}`, 'success');

        await startLocalEmotionDetection();

        state.connectionStartTime = Date.now();
        setInterval(updateConnectionTime, 1000);

    } catch (error) {
        console.error('Join error:', error);
        showStatus(`❌ ${error.message}`, 'error');
    } finally {
        state.isJoining = false;
    }
}

function connectToSignalingServer() {
    state.ws = new WebSocket(`${CONFIG.SIGNALING_SERVER}/ws/${state.roomId}/${state.userId}`);

    state.ws.onopen = () => {
        console.log('✅ Connected');
        updateConnectionStatus('connecting');
        state.reconnectAttempts = 0;
    };

    state.ws.onmessage = async (event) => {
        try {
            const message = JSON.parse(event.data);
            await handleSignalingMessage(message);
        } catch (error) {
            console.error('Message error:', error);
        }
    };

    state.ws.onerror = () => showStatus('❌ Lỗi kết nối', 'error');

    state.ws.onclose = () => {
        updateConnectionStatus('disconnected');
        if (state.reconnectAttempts++ < CONFIG.MAX_RECONNECT_ATTEMPTS) {
            setTimeout(connectToSignalingServer, CONFIG.RECONNECT_DELAY);
        }
    };
}

async function handleSignalingMessage(message) {
    const handlers = {
        'room-state': () => message.users.length > 0 && createOffer(message.users[0]),
        'offer': () => handleOffer(message),
        'answer': () => handleAnswer(message),
        'ice-candidate': () => handleIceCandidate(message),
        'emotion-data': () => handleRemoteEmotionData(message),
        'user-left': () => handleUserLeft(message)
    };

    const handler = handlers[message.type];
    if (handler) await handler();
}

async function createOffer(targetUserId) {
    state.peerConnection = new RTCPeerConnection({ iceServers: CONFIG.ICE_SERVERS });

    state.localStream.getTracks().forEach(track => {
        state.peerConnection.addTrack(track, state.localStream);
    });

    state.peerConnection.ontrack = (event) => {
        state.remoteStream = event.streams[0];
        document.getElementById('remoteVideo').srcObject = state.remoteStream;
        updateConnectionStatus('connected');
        startRemoteEmotionDetection();
    };

    state.peerConnection.onicecandidate = (event) => {
        if (event.candidate && state.ws.readyState === WebSocket.OPEN) {
            state.ws.send(JSON.stringify({
                type: 'ice-candidate',
                candidate: event.candidate,
                target: targetUserId
            }));
        }
    };

    const offer = await state.peerConnection.createOffer();
    await state.peerConnection.setLocalDescription(offer);

    state.ws.send(JSON.stringify({ type: 'offer', offer, target: targetUserId }));
}

async function handleOffer(message) {
    state.peerConnection = new RTCPeerConnection({ iceServers: CONFIG.ICE_SERVERS });

    state.localStream.getTracks().forEach(track => {
        state.peerConnection.addTrack(track, state.localStream);
    });

    state.peerConnection.ontrack = (event) => {
        state.remoteStream = event.streams[0];
        document.getElementById('remoteVideo').srcObject = state.remoteStream;
        updateConnectionStatus('connected');
        startRemoteEmotionDetection();
    };

    state.peerConnection.onicecandidate = (event) => {
        if (event.candidate && state.ws.readyState === WebSocket.OPEN) {
            state.ws.send(JSON.stringify({
                type: 'ice-candidate',
                candidate: event.candidate,
                target: message.from
            }));
        }
    };

    await state.peerConnection.setRemoteDescription(new RTCSessionDescription(message.offer));
    const answer = await state.peerConnection.createAnswer();
    await state.peerConnection.setLocalDescription(answer);

    state.ws.send(JSON.stringify({ type: 'answer', answer, target: message.from }));
}

async function handleAnswer(message) {
    await state.peerConnection.setRemoteDescription(new RTCSessionDescription(message.answer));
}

async function handleIceCandidate(message) {
    try {
        await state.peerConnection.addIceCandidate(new RTCIceCandidate(message.candidate));
    } catch (error) {
        console.error('ICE error:', error);
    }
}

async function startLocalEmotionDetection() {
    if (!state.faceApiLoaded) return;

    const video = document.getElementById('localVideo');
    const canvas = document.getElementById('localCanvas');

    if (video.readyState < 2) {
        video.addEventListener('loadeddata', startLocalEmotionDetection, { once: true });
        return;
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    faceapi.matchDimensions(canvas, { width: video.videoWidth, height: video.videoHeight });

    const intervalId = setInterval(async () => {
        try {
            const detections = await faceapi
                .detectAllFaces(video, new faceapi.TinyFaceDetectorOptions({ scoreThreshold: 0.5 }))
                .withFaceExpressions();

            if (detections?.length > 0) {
                const expressions = detections[0].expressions;
                const emotion = Object.keys(expressions).reduce((a, b) =>
                    expressions[a] > expressions[b] ? a : b
                );
                const confidence = (expressions[emotion] * 100).toFixed(0);

                requestAnimationFrame(() => {
                    document.getElementById('localEmotion').textContent = getEmotionEmoji(emotion);
                    document.getElementById('localConfidence').textContent = `${emotion} (${confidence}%)`;
                });

                state.localEmotionHistory[emotion] = (state.localEmotionHistory[emotion] || 0) + 1;
                updateTopEmotion('local');

                if (state.ws?.readyState === WebSocket.OPEN) {
                    state.ws.send(JSON.stringify({ type: 'emotion-data', emotion, confidence }));
                }

                document.getElementById('framesAnalyzed').textContent = ++state.framesAnalyzed;
            }
        } catch (error) {
            console.error('Detection error:', error);
        }
    }, CONFIG.DETECTION_INTERVAL);

    state.detectionIntervals.push(intervalId);
}

async function startRemoteEmotionDetection() {
    if (!state.faceApiLoaded) return;

    const video = document.getElementById('remoteVideo');
    const canvas = document.getElementById('remoteCanvas');

    if (video.readyState < 2) {
        video.addEventListener('loadeddata', startRemoteEmotionDetection, { once: true });
        return;
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    faceapi.matchDimensions(canvas, { width: video.videoWidth, height: video.videoHeight });

    const intervalId = setInterval(async () => {
        try {
            const detections = await faceapi
                .detectAllFaces(video, new faceapi.TinyFaceDetectorOptions({ scoreThreshold: 0.5 }))
                .withFaceExpressions();

            if (detections?.length > 0) {
                const expressions = detections[0].expressions;
                const emotion = Object.keys(expressions).reduce((a, b) =>
                    expressions[a] > expressions[b] ? a : b
                );
                const confidence = (expressions[emotion] * 100).toFixed(0);

                requestAnimationFrame(() => {
                    document.getElementById('remoteEmotion').textContent = getEmotionEmoji(emotion);
                    document.getElementById('remoteConfidence').textContent = `${emotion} (${confidence}%)`;
                });

                state.remoteEmotionHistory[emotion] = (state.remoteEmotionHistory[emotion] || 0) + 1;
                updateTopEmotion('remote');
            }
        } catch (error) {
            console.error('Detection error:', error);
        }
    }, CONFIG.DETECTION_INTERVAL);

    state.detectionIntervals.push(intervalId);
}

function handleRemoteEmotionData(message) {
    document.getElementById('remoteEmotion').textContent = getEmotionEmoji(message.emotion);
    document.getElementById('remoteConfidence').textContent = `${message.emotion} (${message.confidence}%)`;
    state.remoteEmotionHistory[message.emotion] = (state.remoteEmotionHistory[message.emotion] || 0) + 1;
    updateTopEmotion('remote');
}

function handleUserLeft() {
    showStatus('User đã rời phòng', 'info');
    state.peerConnection?.close();
    state.peerConnection = null;
    document.getElementById('remoteVideo').srcObject = null;
    updateConnectionStatus('disconnected');
}

function leaveRoom() {
    state.detectionIntervals.forEach(id => clearInterval(id));
    state.detectionIntervals = [];

    state.peerConnection?.close();
    state.localStream?.getTracks().forEach(track => track.stop());
    state.ws?.close();

    Object.assign(state, {
        peerConnection: null,
        localStream: null,
        remoteStream: null,
        ws: null,
        connectionStartTime: null,
        framesAnalyzed: 0,
        localEmotionHistory: {},
        remoteEmotionHistory: {}
    });

    document.getElementById('callInterface').style.display = 'none';
    document.getElementById('roomSetup').style.display = 'block';
    document.getElementById('roomIdInput').value = '';
    document.getElementById('localVideo').srcObject = null;
    document.getElementById('remoteVideo').srcObject = null;

    showStatus('👋 Đã rời phòng', 'info');
}

function getEmotionEmoji(emotion) {
    const map = {
        happy: '😊', sad: '😢', angry: '😠',
        fearful: '😨', disgusted: '🤢', surprised: '😲', neutral: '😐'
    };
    return map[emotion] || '😐';
}

function updateConnectionStatus(status) {
    const el = document.getElementById('connectionStatus');
    if (el) {
        el.className = `connection-status ${status}`;
        el.textContent = status.charAt(0).toUpperCase() + status.slice(1);
    }
}

function updateConnectionTime() {
    if (!state.connectionStartTime) return;

    const elapsed = Math.floor((Date.now() - state.connectionStartTime) / 1000);
    const el = document.getElementById('connectionTime');
    if (el) {
        el.textContent = `${String(Math.floor(elapsed / 60)).padStart(2, '0')}:${String(elapsed % 60).padStart(2, '0')}`;
    }
}

function updateTopEmotion(type) {
    const history = type === 'local' ? state.localEmotionHistory : state.remoteEmotionHistory;
    if (Object.keys(history).length === 0) return;

    const topEmotion = Object.keys(history).reduce((a, b) => history[a] > history[b] ? a : b);
    const el = document.getElementById(type === 'local' ? 'localTopEmotion' : 'remoteTopEmotion');
    if (el) el.textContent = `${getEmotionEmoji(topEmotion)} ${topEmotion}`;
}

function showStatus(message, type) {
    const el = document.getElementById('statusMessage');
    if (el) {
        el.textContent = message;
        el.className = `status-message show ${type}`;
        setTimeout(() => el.classList.remove('show'), 5000);
    }
}

window.addEventListener('beforeunload', leaveRoom);
