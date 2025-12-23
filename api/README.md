# Interview Analysis API

FastAPI backend để phân tích video phỏng vấn.

## 🚀 Quick Start

### 1. Cài đặt dependencies

```bash
pip install -r api/requirements.txt
```

### 2. Chạy API

```bash
python api/main.py
```

API sẽ chạy tại: `http://localhost:8000`

### 3. Mở API Documentation

Truy cập: `http://localhost:8000/docs`

FastAPI tự động tạo UI để test!

---

## 📡 API Endpoints

### 1. Health Check

```bash
GET /
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00",
  "total_jobs": 5
}
```

---

### 2. Upload Video (Async)

```bash
POST /api/upload
Content-Type: multipart/form-data
Body: file=@video.mp4
```

**Response:**
```json
{
  "job_id": "abc12345",
  "status": "uploaded",
  "message": "Video uploaded successfully"
}
```

---

### 3. Start Analysis (Background)

```bash
POST /api/analyze/{job_id}
```

**Response:**
```json
{
  "job_id": "abc12345",
  "status": "processing",
  "message": "Analysis started"
}
```

---

### 4. Check Status

```bash
GET /api/status/{job_id}
```

**Response:**
```json
{
  "job_id": "abc12345",
  "status": "processing"
}
```

Status values:
- `uploaded` - Đã upload, chưa phân tích
- `processing` - Đang phân tích
- `completed` - Hoàn thành
- `failed` - Lỗi

---

### 5. Get Results

```bash
GET /api/results/{job_id}
```

**Response:**
```json
{
  "job_id": "abc12345",
  "status": "completed",
  "filename": "interview.mp4",
  "scores": {
    "emotion": 8.5,
    "focus": 7.2,
    "clarity": 8.0,
    "content": 9.0,
    "total": 8.48
  },
  "rating": "RẤT TỐT",
  "details": {
    "emotion": {...},
    "focus": {...},
    "clarity": {...},
    "content": {...}
  },
  "created_at": "2024-01-01T00:00:00",
  "completed_at": "2024-01-01T00:05:00"
}
```

---

### 6. Analyze Sync (One-shot)

```bash
POST /api/analyze-sync
Content-Type: multipart/form-data
Body: file=@video.mp4
```

Upload và phân tích ngay lập tức (không dùng background task).

**Warning:** Request có thể mất vài phút!

**Response:** Giống `/api/results/{job_id}`

---

### 7. List All Jobs

```bash
GET /api/jobs
```

**Response:**
```json
{
  "total": 5,
  "jobs": [
    {
      "job_id": "abc12345",
      "status": "completed",
      "filename": "interview.mp4",
      "created_at": "2024-01-01T00:00:00"
    }
  ]
}
```

---

### 8. Delete Job

```bash
DELETE /api/jobs/{job_id}
```

Xóa job và file video.

---

## 🧪 Testing với curl

### Test 1: Health check
```bash
curl http://localhost:8000/health
```

### Test 2: Upload video
```bash
curl -X POST -F "file=@path/to/video.mp4" http://localhost:8000/api/upload
```

Response:
```json
{"job_id": "abc12345", "status": "uploaded"}
```

### Test 3: Start analysis
```bash
curl -X POST http://localhost:8000/api/analyze/abc12345
```

### Test 4: Check status
```bash
curl http://localhost:8000/api/status/abc12345
```

### Test 5: Get results
```bash
curl http://localhost:8000/api/results/abc12345
```

### Test 6: One-shot analysis
```bash
curl -X POST -F "file=@path/to/video.mp4" http://localhost:8000/api/analyze-sync
```

---

## 🔄 Workflow

### Async (Recommended cho video dài)

```
1. Upload video
   POST /api/upload
   → job_id

2. Start analysis
   POST /api/analyze/{job_id}
   → status: processing

3. Poll status (mỗi 2-5 giây)
   GET /api/status/{job_id}
   → status: processing | completed | failed

4. Get results
   GET /api/results/{job_id}
   → scores, rating, details
```

### Sync (Cho video ngắn hoặc testing)

```
1. Upload & analyze
   POST /api/analyze-sync
   → scores, rating, details (ngay lập tức)
```

---

## 🧪 Testing với Postman

1. Import collection từ `http://localhost:8000/openapi.json`
2. Hoặc dùng UI tại `http://localhost:8000/docs`

---

## 📁 File Structure

```
api/
├── main.py              # FastAPI app
├── requirements.txt     # Dependencies
├── README.md           # This file
└── api_uploads/        # Uploaded videos (auto-created)
```

---

## 🐛 Troubleshooting

### Lỗi: Module not found

```bash
# Chạy từ root directory
cd /path/to/project
python api/main.py
```

### Lỗi: Port 8000 đã được sử dụng

```python
# Đổi port trong main.py
uvicorn.run(app, host="0.0.0.0", port=8001)
```

### Lỗi: CORS

Nếu frontend không gọi được API, check CORS settings trong `main.py`:

```python
allow_origins=["http://localhost:3000"]  # Frontend URL
```

---

## 🚀 Next Steps

1. ✅ Test API với Postman/curl
2. ✅ Tạo frontend (React/Vue)
3. ✅ Deploy lên Render/Railway
4. ✅ Add authentication (JWT)
5. ✅ Add database (PostgreSQL/MongoDB)
6. ✅ Add caching (Redis)

---

## 📝 Notes

- API lưu kết quả trong memory (restart = mất data)
- Production: dùng Redis hoặc Database
- Video files được lưu trong `api_uploads/`
- Có thể xóa file sau khi xử lý để tiết kiệm dung lượng
