# 📊 Tích Hợp Focus Scoring Lên Web

## Tổng Quan

Đã tích hợp phần quét tập trung (focus scoring) lên web, bao gồm:
- ✅ Điểm tập trung (0-10)
- ✅ Thời gian tập trung (giây)
- ✅ Thời gian mất tập trung (giây)
- ✅ Số lần mất tập trung
- ✅ Tỷ lệ tập trung/mất tập trung (%)

## Các Thay Đổi

### 1. Sửa Text Quét Bị Ngược

**Vấn đề**: Text bị ngược khi dùng camera trực tiếp do CSS `transform: scaleX(-1)`

**Giải pháp**: Đã xóa mirror effect trong `frontend/camera.html`

```css
/* TRƯỚC */
#videoElement {
    transform: scaleX(-1); /* Mirror effect */
}

/* SAU */
#videoElement {
    /* Removed mirror effect to fix text orientation */
}
```

### 2. Thêm API Endpoint Mới

**Endpoint**: `POST /api/analyze-focus`

**Request**:
```bash
curl -X POST http://localhost:8000/api/analyze-focus \
  -F "file=@video.mp4"
```

**Response**:
```json
{
  "job_id": "abc123",
  "status": "completed",
  "filename": "video.mp4",
  "focus_score": 8.5,
  "focused_time": 45.2,
  "distracted_time": 4.8,
  "total_distraction_time": 4.8,
  "distracted_count": 3,
  "focused_rate": 90.4,
  "distracted_rate": 9.6,
  "duration": 50.0,
  "total_frames": 1500,
  "analyzed_frames": 300,
  "distraction_events": [
    {
      "start_frame": 100,
      "end_frame": 150,
      "duration": 1.6
    }
  ],
  "created_at": "2025-12-29T10:00:00"
}
```

### 3. Cập Nhật Frontend

#### Camera.html - Thêm Hiển Thị Focus Details

```html
<div class="stat-card">
    <h3>Thời Gian Tập Trung</h3>
    <div class="stat-value" id="focusedTime">0s</div>
    <div class="stat-label">Focused Time</div>
</div>

<div class="stat-card">
    <h3>Thời Gian Mất Tập Trung</h3>
    <div class="stat-value" id="distractedTime">0s</div>
    <div class="stat-label">Distracted Time</div>
</div>

<div class="stat-card">
    <h3>Số Lần Mất Tập Trung</h3>
    <div class="stat-value" id="distractedCount">0</div>
    <div class="stat-label">Distraction Events</div>
</div>
```

#### Camera.js - Real-time Tracking

```javascript
// Focus tracking
let focusHistory = [];
let totalFocusedTime = 0;
let totalDistractedTime = 0;
let distractedEvents = 0;
let currentlyDistracted = false;

// Update every second
if (focusScore >= 6.0) {
    totalFocusedTime += 1;
} else {
    totalDistractedTime += 1;
    if (!currentlyDistracted) {
        distractedEvents++;
        currentlyDistracted = true;
    }
}
```

#### App.js - Hiển Thị Focus Details

```javascript
function displayFocusDetails(focusDetails) {
    const focusSection = document.createElement('div');
    focusSection.innerHTML = `
        <h3>📊 Chi Tiết Tập Trung</h3>
        <div>
            <strong>⏱️ Thời gian tập trung:</strong>
            ${focusDetails.focused_time}s (${focusDetails.focused_rate}%)
        </div>
        <div>
            <strong>⚠️ Thời gian mất tập trung:</strong>
            ${focusDetails.distracted_time}s (${focusDetails.distracted_rate}%)
        </div>
        <div>
            <strong>🔢 Số lần mất tập trung:</strong>
            ${focusDetails.distracted_count} lần
        </div>
    `;
}
```

### 4. Cập Nhật API Response

**Endpoint**: `POST /api/analyze-sync`

Đã thêm focus details vào response:

```json
{
  "scores": {
    "emotion": 8.5,
    "focus": 7.8,
    "clarity": 8.0,
    "content": 7.5,
    "total": 7.95
  },
  "details": {
    "focus": {
      "score": 7.8,
      "focused_time": 45.2,
      "distracted_time": 4.8,
      "distracted_count": 3,
      "focused_rate": 90.4,
      "distracted_rate": 9.6,
      "average_attention": 7.8
    }
  }
}
```

## Cách Sử Dụng

### 1. Khởi Động API

```bash
python api/main.py
```

### 2. Mở Web Interface

```bash
# Mở trình duyệt
http://localhost:8000/docs  # API docs
```

Hoặc mở file HTML trực tiếp:
```bash
# Camera trực tiếp
frontend/camera.html

# Phân tích video
frontend/app.html
```

### 3. Test API

```bash
python api/test_focus_api.py
```

## Công Thức Tính Focus Score

Xem chi tiết tại: [FOCUS_SCORING_EXPLAINED.md](./FOCUS_SCORING_EXPLAINED.md)

**Công thức chính**:
```
FocusScore = (
    FacePresence × 40% +
    GazeFocus    × 30% +
    HeadFocus    × 20% +
    DriftScore   × 10%
) × 10
```

**4 Thành phần**:
1. **Face Presence (40%)**: Có mặt trong khung hình
2. **Gaze Focus (30%)**: Nhìn thẳng vào camera
3. **Head Focus (20%)**: Giữ đầu thẳng
4. **Drift Score (10%)**: Không ngó nghiêng quá nhiều

## Thang Điểm

- **8-10**: Tập trung tốt (Focused)
- **6-8**: Hơi mất tập trung (Slightly Distracted)
- **4-6**: Mất tập trung (Distracted)
- **0-4**: Rất mất tập trung (Very Distracted)

## Ví Dụ Kết Quả

### Camera Trực Tiếp

```
📊 Trạng Thái Real-time:
- Điểm tập trung: 8.5/10
- Thời gian tập trung: 45s
- Thời gian mất tập trung: 5s
- Số lần mất tập trung: 2 lần
- Cảm xúc hiện tại: 😊 Happy
```

### Phân Tích Video

```
📊 Kết Quả Phân Tích:

Điểm Tổng: 7.9/10
Rating: RẤT TỐT

Chi tiết:
- Cảm xúc: 8.5/10
- Tập trung: 7.8/10
- Rõ ràng: 8.0/10
- Nội dung: 7.5/10

📊 Chi tiết tập trung:
- Thời gian tập trung: 45.2s (90.4%)
- Thời gian mất tập trung: 4.8s (9.6%)
- Số lần mất tập trung: 3 lần
- Điểm trung bình: 7.8/10
```

## Lưu Ý

1. **Camera Mirror**: Đã tắt mirror effect để text không bị ngược
2. **Real-time Tracking**: Cập nhật mỗi giây (1000ms)
3. **Simulated Detection**: Hiện tại dùng simulated data, cần tích hợp face-api.js hoặc MediaPipe cho production
4. **API Response**: Focus details được trả về trong `details.focus`

## Tích Hợp Face Detection (Production)

Để có kết quả chính xác, cần tích hợp face detection library:

### Option 1: face-api.js (Client-side)

```javascript
// Load models
await faceapi.nets.tinyFaceDetector.loadFromUri('/models');
await faceapi.nets.faceExpressionNet.loadFromUri('/models');

// Detect faces
const detections = await faceapi
    .detectAllFaces(video, new faceapi.TinyFaceDetectorOptions())
    .withFaceExpressions();
```

### Option 2: MediaPipe (Server-side)

```python
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection()

results = face_detection.process(frame)
```

## Troubleshooting

### 1. Text vẫn bị ngược

**Giải pháp**: Xóa cache trình duyệt (Ctrl+Shift+R)

### 2. Focus details không hiển thị

**Kiểm tra**:
- API có trả về `details.focus` không?
- Console có lỗi JavaScript không?
- Function `displayFocusDetails()` có được gọi không?

### 3. API lỗi 500

**Kiểm tra**:
- Dependencies đã cài đủ chưa? (`pip install -r requirements.txt`)
- Video có hợp lệ không?
- Log API: `python api/main.py` để xem lỗi chi tiết

## Tài Liệu Tham Khảo

- [FOCUS_SCORING_EXPLAINED.md](./FOCUS_SCORING_EXPLAINED.md) - Chi tiết công thức tính điểm
- [API README](../api/README.md) - Hướng dẫn API
- [Frontend README](../frontend/README.md) - Hướng dẫn frontend

## Changelog

### 2025-12-29
- ✅ Sửa text quét bị ngược (xóa mirror effect)
- ✅ Thêm API endpoint `/api/analyze-focus`
- ✅ Tích hợp focus details vào `/api/analyze-sync`
- ✅ Thêm hiển thị focus details trên camera.html
- ✅ Thêm hiển thị focus details trên app.html
- ✅ Thêm real-time tracking cho camera trực tiếp
- ✅ Thêm test script `test_focus_api.py`
