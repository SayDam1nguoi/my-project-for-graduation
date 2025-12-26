# 🏗️ System Architecture - Camera Integration

## Tổng quan hệ thống

```
┌─────────────────────────────────────────────────────────────────┐
│                         WEB FRONTEND                            │
│                      (frontend/app.html)                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ 📹 Camera    │  │ 😊 Emotion   │  │ 📹 Video     │         │
│  │   Trực Tiếp  │  │   Recognition│  │   Transcript │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐                           │
│  │ 🎤 Audio     │  │ 📊 Score     │                           │
│  │   Transcript │  │   Summary    │                           │
│  └──────────────┘  └──────────────┘                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP POST
                              │ multipart/form-data
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         FASTAPI BACKEND                         │
│                        (api/main.py)                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Endpoints:                                                     │
│  • GET  /health                                                 │
│  • POST /api/upload                                             │
│  • POST /api/analyze                                            │
│  • GET  /api/status/{job_id}                                    │
│  • GET  /api/results/{job_id}                                   │
│  • POST /api/analyze-sync  ← Camera uses this                   │
│  • GET  /api/jobs                                               │
│  • DELETE /api/jobs/{job_id}                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ Process video
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      PYTHON CORE ENGINE                         │
│                    (src/ directory)                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Video Analysis (src/video_analysis/)                    │   │
│  │ • emotion_scoring_engine.py                             │   │
│  │ • attention_detector.py                                 │   │
│  │ • video_transcription_coordinator.py                    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Speech Analysis (src/speech_analysis/)                  │   │
│  │ • integrated_speech_evaluator.py                        │   │
│  │ • interview_content_evaluator.py                        │   │
│  │ • hallucination_filter.py                               │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Evaluation (src/evaluation/)                            │   │
│  │ • integrated_interview_evaluator.py                     │   │
│  │ • overall_interview_scorer.py                           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Camera Feature Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    CAMERA FEATURE FLOW                          │
└─────────────────────────────────────────────────────────────────┘

1. USER ACTION: Click "Bật Camera"
   ↓
2. BROWSER: Request camera permission
   ↓
3. WEBRTC: Access camera stream
   ↓
4. VIDEO ELEMENT: Display preview (mirrored)
   ↓
5. JAVASCRIPT: Start simulated face detection
   ↓
6. USER ACTION: Click "Bắt Đầu Ghi"
   ↓
7. MEDIARECORDER: Start recording (video + audio)
   ↓
8. TIMER: Update every second
   ↓
9. USER ACTION: Click "Dừng Ghi & Phân Tích"
   ↓
10. MEDIARECORDER: Stop recording
    ↓
11. BLOB: Create video blob (WebM format)
    ↓
12. FETCH API: Upload to /api/analyze-sync
    ↓
13. FASTAPI: Receive video file
    ↓
14. PYTHON ENGINE: Process video
    │
    ├─→ Extract frames
    ├─→ Detect faces & emotions
    ├─→ Analyze attention/focus
    ├─→ Extract audio
    ├─→ Transcribe speech
    ├─→ Evaluate clarity
    ├─→ Evaluate content
    └─→ Calculate scores
    ↓
15. FASTAPI: Return JSON results
    ↓
16. JAVASCRIPT: Display results in alert
    ↓
17. USER: View scores & rating
```

## Data Flow

```
┌──────────────┐
│   Browser    │
│   Camera     │
└──────┬───────┘
       │ Video Stream
       ▼
┌──────────────┐
│ MediaRecorder│
│   (WebM)     │
└──────┬───────┘
       │ Blob
       ▼
┌──────────────┐
│  Fetch API   │
│  (Upload)    │
└──────┬───────┘
       │ HTTP POST
       ▼
┌──────────────┐
│   FastAPI    │
│  /analyze    │
└──────┬───────┘
       │ File
       ▼
┌──────────────┐
│ Video Engine │
│  (Process)   │
└──────┬───────┘
       │ Scores
       ▼
┌──────────────┐
│   FastAPI    │
│  (Response)  │
└──────┬───────┘
       │ JSON
       ▼
┌──────────────┐
│  JavaScript  │
│  (Display)   │
└──────────────┘
```

## Technology Stack

### Frontend
```
┌─────────────────────────────────────┐
│ HTML5                               │
│ • Video element                     │
│ • Canvas (future: face overlay)    │
│ • File input                        │
├─────────────────────────────────────┤
│ CSS3                                │
│ • Flexbox/Grid layout               │
│ • Animations (pulse, spin)          │
│ • Dark theme                        │
├─────────────────────────────────────┤
│ JavaScript (Vanilla)                │
│ • WebRTC API                        │
│ • MediaRecorder API                 │
│ • Fetch API                         │
│ • DOM manipulation                  │
└─────────────────────────────────────┘
```

### Backend
```
┌─────────────────────────────────────┐
│ FastAPI                             │
│ • REST API                          │
│ • CORS middleware                   │
│ • File upload handling              │
│ • Async/await                       │
├─────────────────────────────────────┤
│ Python 3.8+                         │
│ • OpenCV (video processing)         │
│ • DeepFace (emotion detection)      │
│ • Whisper (speech-to-text)          │
│ • Custom scoring algorithms         │
└─────────────────────────────────────┘
```

## File Structure

```
project/
├── frontend/
│   ├── app.html                    # Main frontend (5 tabs)
│   ├── app.js                      # All JavaScript logic
│   ├── camera.html                 # Standalone camera demo
│   ├── camera.js                   # Camera demo logic
│   ├── index.html                  # Simple frontend
│   ├── README.md                   # Simple frontend docs
│   ├── FULL_FEATURES.md            # Full frontend docs
│   ├── CAMERA_FEATURE.md           # Camera feature docs
│   ├── INTEGRATION_COMPLETE.md     # Integration summary
│   ├── QUICK_START_CAMERA.md       # Quick start guide
│   └── SYSTEM_ARCHITECTURE.md      # This file
│
├── api/
│   ├── main.py                     # FastAPI application
│   ├── requirements.txt            # API dependencies
│   ├── README.md                   # API documentation
│   ├── test_api.py                 # API tests (Python)
│   └── test_api_windows.ps1        # API tests (PowerShell)
│
├── src/
│   ├── video_analysis/             # Video processing
│   ├── speech_analysis/            # Speech processing
│   ├── evaluation/                 # Scoring logic
│   └── ...
│
├── config/
│   ├── emotion_scoring_config.yaml # Emotion weights
│   └── ...
│
└── docs/
    ├── EMOTION_ONLY_SYSTEM.md      # Emotion-only docs
    ├── FOCUS_SCORING_EXPLAINED.md  # Focus algorithm
    └── SCORING_SYSTEM.md           # Overall scoring
```

## API Endpoints Detail

### POST /api/analyze-sync
**Used by Camera Feature**

Request:
```http
POST /api/analyze-sync HTTP/1.1
Host: localhost:8000
Content-Type: multipart/form-data; boundary=----WebKitFormBoundary

------WebKitFormBoundary
Content-Disposition: form-data; name="file"; filename="recording.webm"
Content-Type: video/webm

[binary video data]
------WebKitFormBoundary--
```

Response:
```json
{
  "job_id": "abc123",
  "status": "completed",
  "scores": {
    "emotion": 8.5,
    "focus": 7.2,
    "clarity": 6.8,
    "content": 7.5,
    "total": 7.5
  },
  "rating": "TỐT",
  "details": {
    "emotion": { ... },
    "focus": { ... },
    "clarity": { ... },
    "content": { ... }
  }
}
```

## Security Considerations

### Frontend
- ✅ Camera permission required
- ✅ HTTPS recommended (localhost OK for dev)
- ✅ No sensitive data stored in browser
- ✅ Video blob cleared after upload

### Backend
- ✅ CORS enabled (localhost only)
- ✅ File size limits (100MB)
- ✅ File type validation
- ⚠️ In-memory storage (use Redis/DB for production)
- ⚠️ No authentication (add JWT for production)

## Performance

### Frontend
- Video resolution: 1280x720 (ideal)
- Recording format: WebM (VP9)
- Face detection: 1 FPS (simulated)
- Upload: Async (non-blocking)

### Backend
- Video processing: ~30-60 seconds
- Emotion detection: ~5-10 seconds
- Speech transcription: ~20-40 seconds
- Total: ~1-2 minutes per video

## Future Enhancements

### 1. Real-time Face Detection
```javascript
// Using face-api.js
const detections = await faceapi
  .detectAllFaces(video)
  .withFaceExpressions();
```

### 2. Live Emotion Chart
```javascript
// Using Chart.js
const chart = new Chart(ctx, {
  type: 'line',
  data: emotionTimeline
});
```

### 3. WebSocket for Real-time Updates
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');
ws.onmessage = (event) => {
  updateStats(JSON.parse(event.data));
};
```

### 4. Video Playback with Annotations
```javascript
// Overlay emotion labels on video
ctx.fillText(emotion, x, y);
```

## Deployment

### Development
```bash
# Backend
python api/main.py

# Frontend
# Open app.html in browser
```

### Production
```bash
# Backend (with Gunicorn)
gunicorn api.main:app -w 4 -k uvicorn.workers.UvicornWorker

# Frontend (with Nginx)
nginx -c nginx.conf
```

## Monitoring

### Logs
- Frontend: Browser console
- Backend: Uvicorn logs
- Python: Custom logging

### Metrics
- Upload success rate
- Processing time
- Error rate
- User engagement

---

**System Status: ✅ Fully Operational**
