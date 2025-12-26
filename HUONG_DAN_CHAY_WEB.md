# 🚀 HƯỚNG DẪN CHẠY WEB - CỰC KỲ ĐỠN GIẢN

## ⚡ Quick Start (2 Bước)

### Bước 1: Khởi động API Backend

**Mở Terminal/PowerShell và chạy:**

```bash
cd api
python main.py
```

**Hoặc (nếu đang ở thư mục gốc):**

```bash
python api/main.py
```

**Đợi thông báo:**
```
INFO:     Started server process [xxxxx]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

✅ **API đã sẵn sàng!**

---

### Bước 2: Mở Frontend

**Option A: Double-click (Đơn giản nhất)**
1. Mở File Explorer
2. Vào thư mục `frontend/`
3. **Double-click file `app.html`**
4. Browser sẽ tự động mở

**Option B: Right-click**
1. Right-click file `frontend/app.html`
2. Chọn **"Open with"** → **Chrome** (hoặc Edge)

**Option C: Kéo thả**
1. Mở Chrome/Edge
2. Kéo file `app.html` vào browser

✅ **Web đã chạy!**

---

## 🎯 Sử dụng Web

### Tab 1: 📹 Camera Trực Tiếp (Real-time Emotion Detection)

**Bước 1: Bật Camera**
1. Click tab **"📹 Camera Trực Tiếp"**
2. Click button **"📹 Bật Camera"**
3. Cho phép quyền camera/microphone
4. Đợi AI models load (~5-10 giây, lần đầu)

**Bước 2: Xem Real-time Detection**
- ✅ Khung xanh xuất hiện quanh mặt
- ✅ Label cảm xúc: `😊 happy (85%)`
- ✅ Stats cập nhật real-time

**Bước 3: Ghi Video & Phân Tích**
1. Click **"⏺ Bắt Đầu Ghi"**
2. Nói vài câu (10-30 giây)
3. Click **"⏹ Dừng Ghi & Phân Tích"**
4. Đợi kết quả (~1-2 phút)
5. Xem điểm tổng + 4 điểm chi tiết

---

### Tab 2: 😊 Nhận Diện Cảm Xúc

1. Click tab **"😊 Nhận Diện Cảm Xúc"**
2. Upload video (kéo thả hoặc click chọn)
3. Click **"🚀 Phân Tích Cảm Xúc"**
4. Đợi kết quả
5. Xem điểm cảm xúc chi tiết

---

### Tab 3: 📹 Chuyển Đổi Video

1. Click tab **"📹 Chuyển Đổi Video"**
2. Upload video
3. Click **"🎬 Chuyển Đổi Video"**
4. Đợi transcription
5. Copy transcript

---

### Tab 4: 🎤 Chuyển Đổi Audio

1. Click tab **"🎤 Chuyển Đổi Audio"**
2. Upload audio (WAV, MP3)
3. Click **"🎙️ Chuyển Đổi Audio"**
4. Đợi transcription
5. Copy transcript

---

### Tab 5: 📊 Tổng Hợp Điểm

1. Click tab **"📊 Tổng Hợp Điểm"**
2. Upload video phỏng vấn
3. Điều chỉnh trọng số (phải tổng = 100%):
   - 😊 Cảm xúc: 5%
   - 👁️ Tập trung: 20%
   - 🗣️ Rõ ràng: 35%
   - 📝 Nội dung: 40%
4. Click **"🚀 Phân Tích Toàn Diện"**
5. Xem điểm tổng + rating + 4 điểm chi tiết

---

## 🎭 Tính năng Real-time Emotion Detection

### 7 Cảm xúc được phát hiện:
1. 😊 **Happy** - Vui vẻ
2. 😢 **Sad** - Buồn
3. 😠 **Angry** - Tức giận
4. 😨 **Fearful** - Sợ hãi
5. 🤢 **Disgusted** - Ghê tởm
6. 😲 **Surprised** - Ngạc nhiên
7. 😐 **Neutral** - Bình thường

### Thử các biểu cảm:
- **Cười** → 😊 happy (80-95%)
- **Buồn** → 😢 sad (70-90%)
- **Tức giận** → 😠 angry (60-85%)
- **Ngạc nhiên** → 😲 surprised (70-90%)
- **Bình thường** → 😐 neutral (60-80%)

---

## 🐛 Troubleshooting

### API không chạy
```
Lỗi: "Cannot connect to API"

Giải pháp:
1. Check API đang chạy: python api/main.py
2. Check port 8000 không bị chiếm
3. Thử: http://localhost:8000/health
```

### Camera không bật
```
Lỗi: "Không thể truy cập camera"

Giải pháp:
1. Cho phép quyền camera/microphone
2. Đóng các app khác đang dùng camera
3. Thử browser khác (Chrome recommended)
4. Refresh page
```

### Models không load
```
Lỗi: "Failed to load models"

Giải pháp:
1. Check internet connection (cần lần đầu)
2. Đợi thêm 10-20 giây
3. Refresh page
4. Check console (F12) for errors
```

### Upload lỗi
```
Lỗi: "Upload failed"

Giải pháp:
1. Check API đang chạy
2. Check file size (<100MB)
3. Check file format (MP4, AVI, MOV, WebM)
4. Check console logs (F12)
```

### Không detect được mặt
```
Lỗi: Không có khung xanh

Giải pháp:
1. Cải thiện ánh sáng
2. Nhìn thẳng vào camera
3. Di chuyển gần camera hơn
4. Đảm bảo mặt rõ ràng
```

---

## 📊 Kiểm tra API đang chạy

**Mở browser và truy cập:**

```
http://localhost:8000/health
```

**Kết quả mong đợi:**
```json
{
  "status": "healthy",
  "message": "Interview Analysis API is running"
}
```

**Hoặc xem API docs:**
```
http://localhost:8000/docs
```

---

## 🌐 Browser Support

| Browser | Camera | Recording | Upload | Overall |
|---------|--------|-----------|--------|---------|
| Chrome  | ✅     | ✅        | ✅     | ✅ Recommended |
| Edge    | ✅     | ✅        | ✅     | ✅ Recommended |
| Firefox | ✅     | ✅        | ✅     | ✅ Good |
| Safari  | ⚠️     | ⚠️        | ✅     | ⚠️ Limited |

**Khuyến nghị:** Chrome hoặc Edge

---

## 📚 Documentation

### Quick Guides
- `frontend/TEST_REALTIME_EMOTION.md` - Test real-time emotion (1 phút)
- `frontend/TEST_NOW.md` - Test camera feature (2 phút)
- `frontend/QUICK_START_CAMERA.md` - Hướng dẫn camera chi tiết

### Technical Docs
- `frontend/REALTIME_EMOTION_DETECTION.md` - Chi tiết kỹ thuật
- `frontend/CAMERA_FEATURE.md` - Camera feature overview
- `frontend/SYSTEM_ARCHITECTURE.md` - Kiến trúc hệ thống

### API Docs
- `api/README.md` - API documentation
- `http://localhost:8000/docs` - Swagger UI (khi API chạy)

---

## 🎯 Demo Flow

```
1. Khởi động API
   ↓
2. Mở app.html trong browser
   ↓
3. Click "Camera Trực Tiếp" tab
   ↓
4. Bật camera
   ↓
5. Đợi models load (~5-10 giây)
   ↓
6. Xem real-time emotion detection!
   ↓
7. Khung xanh + label xuất hiện
   ↓
8. Thử các biểu cảm khác nhau
   ↓
9. Ghi video & phân tích
   ↓
10. Xem kết quả chi tiết
```

---

## 💡 Tips

### Để có kết quả tốt nhất:

**Camera:**
- ✅ Ánh sáng tốt (mặt rõ ràng)
- ✅ Nhìn thẳng vào camera
- ✅ Khoảng cách vừa phải (~50cm)
- ✅ Background đơn giản

**Recording:**
- ✅ Nói rõ ràng, không quá nhanh
- ✅ Độ dài: 10-30 giây (tối ưu)
- ✅ Tránh nhiễu âm
- ✅ Microphone tốt

**Upload:**
- ✅ File size < 100MB
- ✅ Format: MP4, AVI, MOV, WebM
- ✅ Resolution: 720p-1080p
- ✅ Có audio rõ ràng

---

## 🚀 Performance

### Expected Processing Time:
- **Real-time detection**: ~100ms per frame (10 FPS)
- **Video upload**: ~5-10 seconds
- **Video processing**: ~1-2 minutes
- **Total**: ~2-3 minutes per video

### Resource Usage:
- **CPU**: ~20-30% (real-time detection)
- **Memory**: ~100-150MB (browser)
- **Network**: ~2.5MB (models, first time only)

---

## 🎉 Tổng kết

**Chạy web cực kỳ đơn giản:**

1. ✅ `python api/main.py` (Terminal)
2. ✅ Double-click `frontend/app.html` (File Explorer)
3. ✅ Enjoy! 🎭

**Tính năng:**
- ✅ Real-time emotion detection (AI thật!)
- ✅ Video recording & analysis
- ✅ Speech-to-text transcription
- ✅ Comprehensive scoring
- ✅ 5 tabs đầy đủ chức năng

**Sẵn sàng test ngay!** 🚀

---

**Version:** 2.1 (Real-time Emotion Detection)

**Last Updated:** December 2024

**Happy analyzing! 🎉**
