# 🚀 BẮT ĐẦU TẠI ĐÂY

## Cách chạy Web - CỰC KỲ ĐƠN GIẢN

### Option 1: Tự động (Khuyến nghị) ⭐

**Double-click một trong hai file:**

```
start_web.bat       (Windows - đơn giản)
start_web.ps1       (PowerShell - đầy đủ)
```

✅ **Xong!** API sẽ tự động chạy và browser sẽ mở.

---

### Option 2: Thủ công (2 bước)

**Bước 1: Khởi động API**

Mở Terminal/PowerShell:
```bash
python api/main.py
```

Đợi thông báo:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Bước 2: Mở Frontend**

Double-click file:
```
frontend/app.html
```

✅ **Xong!**

---

## 🎯 Sử dụng

### Tab 1: 📹 Camera Trực Tiếp

1. Click tab **"📹 Camera Trực Tiếp"**
2. Click **"📹 Bật Camera"**
3. Cho phép quyền camera
4. Đợi AI models load (~5-10 giây)
5. **Xem real-time emotion detection!**
   - Khung xanh quanh mặt
   - Label: `😊 happy (85%)`
   - Stats cập nhật real-time

### Ghi Video & Phân Tích

1. Click **"⏺ Bắt Đầu Ghi"**
2. Nói vài câu (10-30 giây)
3. Click **"⏹ Dừng Ghi & Phân Tích"**
4. Đợi kết quả (~1-2 phút)
5. Xem điểm tổng + 4 điểm chi tiết

---

## 🎭 7 Cảm xúc được phát hiện

| Emoji | Emotion | Tiếng Việt |
|-------|---------|------------|
| 😊 | happy | Vui vẻ |
| 😢 | sad | Buồn |
| 😠 | angry | Tức giận |
| 😨 | fearful | Sợ hãi |
| 🤢 | disgusted | Ghê tởm |
| 😲 | surprised | Ngạc nhiên |
| 😐 | neutral | Bình thường |

**Thử các biểu cảm khác nhau và xem AI phản ứng!**

---

## 🐛 Lỗi thường gặp

### API không chạy
```
Lỗi: "Cannot connect to API"
→ Chạy: python api/main.py
```

### Camera không bật
```
Lỗi: "Không thể truy cập camera"
→ Cho phép quyền camera/microphone
→ Thử browser khác (Chrome)
```

### Models không load
```
Lỗi: "Failed to load models"
→ Check internet (cần lần đầu)
→ Đợi thêm 10-20 giây
→ Refresh page
```

---

## 📚 Tài liệu đầy đủ

- **HUONG_DAN_CHAY_WEB.md** - Hướng dẫn chi tiết
- **frontend/TEST_REALTIME_EMOTION.md** - Test guide
- **frontend/CAMERA_FEATURE.md** - Camera features
- **api/README.md** - API documentation

---

## 🌐 Browser Support

| Browser | Status |
|---------|--------|
| Chrome  | ✅ Recommended |
| Edge    | ✅ Recommended |
| Firefox | ✅ Good |
| Safari  | ⚠️ Limited |

---

## 🎉 Tổng kết

**3 cách chạy web:**

1. ⭐ **Double-click `start_web.bat`** (Tự động)
2. 📝 **`python api/main.py`** + **Double-click `app.html`** (Thủ công)
3. 📖 **Xem `HUONG_DAN_CHAY_WEB.md`** (Chi tiết)

**Tính năng:**
- ✅ Real-time emotion detection (AI thật!)
- ✅ 7 emotions với confidence scores
- ✅ Video recording & analysis
- ✅ Speech-to-text transcription
- ✅ Comprehensive scoring

**Sẵn sàng test ngay!** 🚀

---

**Version:** 2.1 (Real-time Emotion Detection)

**Happy analyzing! 🎉**
