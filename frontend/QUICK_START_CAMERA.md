# 🚀 Quick Start - Camera Feature

## Bắt đầu nhanh trong 3 bước

### Bước 1: Khởi động API
```bash
cd api
python main.py
```

Đợi thông báo:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Bước 2: Mở Frontend
```bash
# Mở file trong browser (Chrome/Edge recommended)
frontend/app.html
```

Hoặc double-click file `app.html`

### Bước 3: Sử dụng Camera
1. Click tab **"📹 Camera Trực Tiếp"**
2. Click **"📹 Bật Camera"** → Cho phép quyền truy cập
3. Click **"⏺ Bắt Đầu Ghi"** → Nói vài câu
4. Click **"⏹ Dừng Ghi & Phân Tích"** → Chờ kết quả

## Demo Video Flow

```
┌─────────────────────────────────────┐
│  1. Bật Camera                      │
│     ↓                               │
│  2. Video preview hiển thị          │
│     ↓                               │
│  3. Stats cập nhật real-time        │
│     ↓                               │
│  4. Bắt đầu ghi (⏺)                 │
│     ↓                               │
│  5. Timer đếm thời gian             │
│     ↓                               │
│  6. Dừng ghi & phân tích            │
│     ↓                               │
│  7. Upload to API                   │
│     ↓                               │
│  8. Kết quả hiển thị                │
└─────────────────────────────────────┘
```

## Kết quả mẫu

```
📊 Kết Quả Phân Tích Camera:

Điểm Tổng: 7.5/10
Rating: TỐT

Chi tiết:
- Cảm xúc: 8.5/10
- Tập trung: 7.2/10
- Rõ ràng: 6.8/10
- Nội dung: 7.5/10
```

## Lưu ý

✅ **Browser**: Chrome hoặc Edge (recommended)
✅ **Permissions**: Cho phép camera + microphone
✅ **API**: Phải chạy trước khi test
✅ **Network**: Localhost (không cần internet)

## Troubleshooting nhanh

| Vấn đề | Giải pháp |
|--------|-----------|
| Camera không bật | Cho phép quyền trong browser settings |
| API lỗi | Chạy `python api/main.py` |
| Upload lỗi | Kiểm tra API đang chạy |
| Video không ghi | Thử browser khác (Chrome) |

## Tính năng

✅ Real-time camera preview (mirrored)
✅ Face detection (simulated)
✅ Emotion tracking (simulated)
✅ Video recording với audio
✅ Upload & analysis
✅ Comprehensive scoring

## Nâng cấp (Optional)

Muốn face detection thật?
```bash
# Install face-api.js
npm install face-api.js

# Download models
# https://github.com/justadudewhohacks/face-api.js-models
```

Thêm vào HTML:
```html
<script src="node_modules/face-api.js/dist/face-api.min.js"></script>
```

## Support

Có vấn đề? Xem:
- `frontend/CAMERA_FEATURE.md` - Documentation đầy đủ
- `frontend/INTEGRATION_COMPLETE.md` - Chi tiết integration
- `api/README.md` - API documentation

---

**Enjoy! 🎉**
