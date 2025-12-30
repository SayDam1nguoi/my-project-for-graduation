// ===== CÔNG THỨC TÍNH FOCUS SCORE GIỐNG LAUNCHER =====
// FocusScore = (FacePresence × 40% + GazeFocus × 30% + HeadFocus × 20% + DriftScore × 10%) × 10

// Real-time face detection with LAUNCHER FORMULA
async function startRealTimeFaceDetection_Launcher() {
    const video = document.getElementById('cameraVideo');
    const canvas = document.getElementById('cameraCanvas');

    // Set canvas size to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const displaySize = { width: video.videoWidth, height: video.videoHeight };
    faceapi.matchDimensions(canvas, displaySize);

    // Start tracking time
    cameraStartTime = Date.now();

    // Drift tracking (for Drift Score component)
    let driftEvents = 0;
    let lastDriftResetTime = Date.now();
    let lastFacePresent = true;

    console.log('🚀 Starting real-time face detection with LAUNCHER formula...');

    // Detection loop
    detectionInterval = setInterval(async () => {
        if (!cameraStream) {
            clearInterval(detectionInterval);
            return;
        }

        try {
            // Detect faces with expressions AND landmarks
            const detections = await faceapi
                .detectAllFaces(video, new faceapi.TinyFaceDetectorOptions())
                .withFaceLandmarks()
                .withFaceExpressions();

            // ===== CÔNG THỨC 4 THÀNH PHẦN =====
            let facePresenceScore = 0;
            let gazeFocusScore = 0;
            let headFocusScore = 0;
            let driftScore = 1.0;

            if (detections && detections.length > 0) {
                // ===== 1. FACE PRESENCE (40%) =====
                facePresenceScore = 1.0;
                lastFacePresent = true;

                const detection = detections[0];
                const box = detection.detection.box;
                const landmarks = detection.landmarks;

                // Clear canvas
                const ctx = canvas.getContext('2d');
                ctx.clearRect(0, 0, canvas.width, canvas.height);

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

                // Update emotion stats
                document.getElementById('cameraFaceCount').textContent = detections.length;
                document.getElementById('cameraEmotion').textContent = emotionEmoji;
                document.getElementById('cameraEmotion').style.fontSize = '2em';

                // ===== 2. GAZE FOCUS (30%) =====
                const leftEye = landmarks.getLeftEye();
                const rightEye = landmarks.getRightEye();
                const eyeCenterX = (leftEye[0].x + rightEye[0].x) / 2;
                const eyeCenterY = (leftEye[0].y + rightEye[0].y) / 2;

                const h_ratio = eyeCenterX / displaySize.width;
                const v_ratio = eyeCenterY / displaySize.height;

                const h_deviation = Math.abs(h_ratio - 0.5);
                const v_deviation = Math.abs(v_ratio - 0.5);

                if (h_deviation < 0.15 && v_deviation < 0.15) {
                    gazeFocusScore = 1.0;
                } else if (h_deviation > 0.35 || v_deviation > 0.35) {
                    gazeFocusScore = 0.0;
                } else {
                    const max_deviation = Math.max(h_deviation, v_deviation);
                    gazeFocusScore = 1.0 - ((max_deviation - 0.15) / (0.35 - 0.15));
                    gazeFocusScore = Math.max(0.0, Math.min(1.0, gazeFocusScore));
                }

                // ===== 3. HEAD FOCUS (20%) =====
                const nose = landmarks.getNose();
                const mouth = landmarks.getMouth();

                const noseTip = nose[3];
                const noseToLeft = Math.sqrt(Math.pow(noseTip.x - leftEye[0].x, 2) + Math.pow(noseTip.y - leftEye[0].y, 2));
                const noseToRight = Math.sqrt(Math.pow(noseTip.x - rightEye[0].x, 2) + Math.pow(noseTip.y - rightEye[0].y, 2));
                const faceWidth = Math.sqrt(Math.pow(rightEye[0].x - leftEye[0].x, 2) + Math.pow(rightEye[0].y - leftEye[0].y, 2));

                let yaw = 0;
                if (faceWidth > 0) {
                    const yaw_ratio = (noseToLeft - noseToRight) / faceWidth;
                    yaw = yaw_ratio * 90;
                }

                const mouthCenter = mouth[3];
                const eyeToMouth = Math.sqrt(Math.pow(mouthCenter.x - eyeCenterX, 2) + Math.pow(mouthCenter.y - eyeCenterY, 2));
                const noseToEye = Math.sqrt(Math.pow(noseTip.x - eyeCenterX, 2) + Math.pow(noseTip.y - eyeCenterY, 2));
                const noseToMouth = Math.sqrt(Math.pow(noseTip.x - mouthCenter.x, 2) + Math.pow(noseTip.y - mouthCenter.y, 2));

                let pitch = 0;
                if (eyeToMouth > 0) {
                    const pitch_ratio = (noseToEye - noseToMouth) / eyeToMouth;
                    pitch = pitch_ratio * 45;
                }

                const d_eye_x = rightEye[0].x - leftEye[0].x;
                const d_eye_y = rightEye[0].y - leftEye[0].y;
                const roll = Math.atan2(d_eye_y, d_eye_x) * 180 / Math.PI;

                const max_angle = Math.max(Math.abs(yaw), Math.abs(pitch), Math.abs(roll));

                if (max_angle < 15.0) {
                    headFocusScore = 1.0;
                } else if (max_angle > 25.0) {
                    headFocusScore = 0.0;
                } else {
                    headFocusScore = 1.0 - ((max_angle - 15.0) / (25.0 - 15.0));
                    headFocusScore = Math.max(0.0, Math.min(1.0, headFocusScore));
                }

                // Draw debug info
                ctx.fillStyle = '#667eea';
                ctx.font = '12px Arial';
                ctx.fillText(`Yaw: ${yaw.toFixed(1)}° Pitch: ${pitch.toFixed(1)}° Roll: ${roll.toFixed(1)}°`, box.x, box.y + box.height + 20);

            } else {
                // ===== KHÔNG CÓ MẶT =====
                facePresenceScore = 0.0;
                gazeFocusScore = 0.0;
                headFocusScore = 0.0;

                if (lastFacePresent) {
                    driftEvents++;
                    lastFacePresent = false;
                }

                const ctx = canvas.getContext('2d');
                ctx.clearRect(0, 0, canvas.width, canvas.height);

                document.getElementById('cameraFaceCount').textContent = '0';
                document.getElementById('cameraEmotion').textContent = '-';
                document.getElementById('cameraEmotion').style.fontSize = '1.5em';
            }

            // ===== 4. DRIFT SCORE (10%) =====
            const currentTime = Date.now();
            if (currentTime - lastDriftResetTime > 60000) {
                driftEvents = 0;
                lastDriftResetTime = currentTime;
            }

            const max_allowed_drifts = 3;
            driftScore = 1.0 - Math.min(1.0, driftEvents / max_allowed_drifts);

            // ===== TÍNH ĐIỂM (0-10) =====
            const instant_score_normalized = (
                facePresenceScore * 0.40 +
                gazeFocusScore * 0.30 +
                headFocusScore * 0.20 +
                driftScore * 0.10
            );

            const focusScore = instant_score_normalized * 10.0;

            // Track time
            if (focusScore >= 6.0) {
                cameraTotalFocusedTime += 0.1;
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

            // Update UI
            cameraFocusHistory.push(focusScore);
            if (cameraFocusHistory.length > 30) {
                cameraFocusHistory.shift();
            }

            const avgFocusScore = cameraFocusHistory.reduce((a, b) => a + b, 0) / cameraFocusHistory.length;
            document.getElementById('cameraFocusScore').textContent = avgFocusScore.toFixed(1);
            document.getElementById('cameraFocusedTime').textContent = `${cameraTotalFocusedTime.toFixed(1)}s`;
            document.getElementById('cameraDistractedTime').textContent = `${cameraTotalDistractedTime.toFixed(1)}s`;
            document.getElementById('cameraDistractedCount').textContent = cameraDistractedEvents;

            // Color code
            const focusScoreEl = document.getElementById('cameraFocusScore');
            if (avgFocusScore >= 7.5) {
                focusScoreEl.style.color = '#43e97b';
            } else if (avgFocusScore >= 6.0) {
                focusScoreEl.style.color = '#fee140';
            } else {
                focusScoreEl.style.color = '#f5576c';
            }

        } catch (error) {
            console.error('❌ Detection error:', error);
            // Show error on canvas
            const ctx = canvas.getContext('2d');
            ctx.fillStyle = '#f5576c';
            ctx.font = '16px Arial';
            ctx.fillText('Error: ' + error.message, 10, 30);
        }

    }, 100);
}
