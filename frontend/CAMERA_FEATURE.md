# 📹 Camera Trực Tiếp - Real-time Camera Feature

## Tính năng

Chức năng quét mặt qua camera trực tiếp đã được tích hợp vào web frontend.

### Các chức năng chính:

1. **Bật/Tắt Camera**
   - Truy cập camera của thiết bị
   - Hiển thị video preview (mirror mode)
   - Phát hiện khuôn mặt real-time (simulated)

2. **Ghi Hình**
   - Ghi video với audio
   - Hiển thị thời gian ghi
   - Indicator đỏ khi đang ghi

3. **Phân Tích**
   - Upload video sau khi ghi xong
   - Phân tích toàn diện qua API
   - Hiển thị kết quả (emotion, focus, clarity, content)

4. **Real-time Stats** (Simulated)
   - Số khuôn mặt phát hiện
   - Cảm xúc hiện tại
   - Trạng thái camera

## Cách sử dụng

### 1. Khởi động API
```bash
python api/main.py
```

### 2. Mở Frontend
```bash
# Mở file trong browser
frontend/app.html
```

### 3. Sử dụng Camera
1. Click tab "📹 Camera Trực Tiếp"
2. Click "📹 Bật Camera" → Cho phép quyền truy cập camera
3. Click "⏺ Bắt Đầu Ghi" → Bắt đầu ghi hình
4. Click "⏹ Dừng Ghi & Phân Tích" → Upload và phân tích

### 4. Xem Kết Quả
- Kết quả hiển thị trong popup alert
- Bao gồm: điểm tổng, rating, và 4 điểm chi tiết

## Công nghệ

### Frontend
- **WebRTC**: Truy cập camera
- **MediaRecorder API**: Ghi video
- **JavaScript**: Xử lý logic

### Backend
- **FastAPI**: API endpoint `/api/analyze-sync`
- **Python**: Xử lý video và phân tích

## Lưu ý

### Browser Support
- Chrome/Edge: ✅ Full support
- Firefox: ✅ Full support
- Safari: ⚠️ Cần test (có thể cần codec khác)

### Permissions
- Cần cho phép quyền truy cập camera
- Cần cho phép quyền truy cập microphone (cho audio)

### Performance
- Video codec: VP9 (fallback to default)
- Resolution: 1280x720 (ideal)
- Real-time detection: Simulated (1 FPS)

## Nâng cấp tương lai

### Client-side Face Detection
Có thể tích hợp thư viện như:
- **face-api.js**: Face detection + emotion recognition
- **TensorFlow.js**: Custom models
- **MediaPipe**: Google's solution

### Example với face-api.js:
```javascript
// Load models
await faceapi.nets.tinyFaceDetector.loadFromUri('/models');
await faceapi.nets.faceExpressionNet.loadFromUri('/models');

// Detect faces
const detections = await faceapi
    .detectAllFaces(videoElement, new faceapi.TinyFaceDetectorOptions())
    .withFaceExpressions();

// Update UI with real detections
```

## Troubleshooting

### Camera không bật
- Kiểm tra quyền truy cập trong browser settings
- Đảm bảo không có app khác đang dùng camera
- Thử refresh page

### Recording không hoạt động
- Kiểm tra browser support cho MediaRecorder
- Thử codec khác (webm, mp4)
- Kiểm tra dung lượng disk

### Upload lỗi
- Đảm bảo API đang chạy (`python api/main.py`)
- Kiểm tra CORS settings
- Kiểm tra file size (có thể quá lớn)

## Demo Flow

```
1. User clicks "Bật Camera"
   ↓
2. Browser requests camera permission
   ↓
3. Camera stream displays (mirrored)
   ↓
4. Real-time face detection starts (simulated)
   ↓
5. User clicks "Bắt Đầu Ghi"
   ↓
6. MediaRecorder starts recording
   ↓
7. Timer shows recording duration
   ↓
8. User clicks "Dừng Ghi & Phân Tích"
   ↓
9. Video blob created and uploaded to API
   ↓
10. API analyzes video (emotion, focus, clarity, content)
   ↓
11. Results displayed in alert popup
```

## Code Structure

### HTML (app.html)
- Video element với mirror transform
- Control buttons
- Stats display
- Recording indicator

### JavaScript (app.js)
- Camera control logic
- MediaRecorder setup
- Upload handling
- Real-time analysis (simulated)

### API (api/main.py)
- `/api/analyze-sync` endpoint
- Video processing
- Score calculation
