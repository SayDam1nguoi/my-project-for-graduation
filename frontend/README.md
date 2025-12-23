# 🎨 Frontend - Interview Analysis System

Frontend đơn giản bằng HTML + CSS + JavaScript thuần (không cần build).

## 🚀 Cách Chạy (Cực Kỳ Đơn Giản)

### Bước 1: Đảm bảo API đang chạy

```bash
python api/main.py
```

API phải chạy tại: `http://localhost:8000`

### Bước 2: Mở frontend

**Option A: Double-click file**
- Mở file `frontend/index.html` bằng browser

**Option B: Dùng Python HTTP Server**
```bash
cd frontend
python -m http.server 3000
```
Sau đó mở: `http://localhost:3000`

**Option C: Dùng Live Server (VS Code)**
- Cài extension "Live Server"
- Right-click `index.html` → "Open with Live Server"

---

## 🎯 Cách Sử Dụng

1. **Upload Video**
   - Click "Chọn Video" hoặc kéo thả video vào
   - Chọn file video (MP4, AVI, MOV)

2. **Phân Tích**
   - Click "Phân Tích Video"
   - Đợi 2-5 phút (tùy độ dài video)

3. **Xem Kết Quả**
   - Điểm tổng (0-10)
   - Rating (XUẤT SẮC, RẤT TỐT, TỐT, etc.)
   - 4 điểm chi tiết:
     - 😊 Cảm Xúc
     - 👁️ Tập Trung
     - 🗣️ Rõ Ràng
     - 📝 Nội Dung

4. **Phân Tích Video Khác**
   - Click "Phân Tích Video Khác"

---

## ✨ Tính Năng

✅ **Drag & Drop**: Kéo thả video vào
✅ **Real-time Status**: Hiển thị tiến trình
✅ **Beautiful UI**: Giao diện đẹp, gradient
✅ **Responsive**: Hoạt động trên mobile
✅ **Error Handling**: Xử lý lỗi tốt
✅ **No Build Required**: Chạy ngay, không cần npm

---

## 🎨 Screenshots

### Upload Screen
```
┌─────────────────────────────────────┐
│   Interview Analysis System         │
│   Phân tích video phỏng vấn bằng AI │
├─────────────────────────────────────┤
│                                     │
│   📹 Upload Video Phỏng Vấn        │
│                                     │
│   [  📁 Chọn Video  ]              │
│                                     │
│   [🚀 Phân Tích Video]             │
│                                     │
└─────────────────────────────────────┘
```

### Results Screen
```
┌─────────────────────────────────────┐
│         Điểm Tổng: 8.5/10          │
│           RẤT TỐT                   │
├─────────────────────────────────────┤
│  😊 Cảm Xúc    👁️ Tập Trung       │
│     8.5           7.2               │
│                                     │
│  🗣️ Rõ Ràng    📝 Nội Dung        │
│     8.0           9.0               │
└─────────────────────────────────────┘
```

---

## 🔧 Customization

### Đổi màu chủ đạo

Tìm trong `index.html`:
```css
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
```

Thay bằng màu khác:
```css
/* Blue */
background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);

/* Green */
background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);

/* Pink */
background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
```

### Đổi API URL

Nếu API chạy ở port khác:
```javascript
const API_URL = 'http://localhost:8001'; // Đổi port
```

---

## 🐛 Troubleshooting

### Lỗi: "Không thể kết nối với API"

**Nguyên nhân**: API chưa chạy

**Giải pháp**:
```bash
python api/main.py
```

### Lỗi: CORS

**Nguyên nhân**: Browser block request

**Giải pháp**: Dùng HTTP server thay vì mở file trực tiếp
```bash
cd frontend
python -m http.server 3000
```

### Video không upload được

**Kiểm tra**:
- File size < 100MB
- Format: MP4, AVI, MOV
- Video có âm thanh

---

## 📱 Mobile Support

Frontend responsive, hoạt động tốt trên:
- ✅ Desktop (Chrome, Firefox, Edge)
- ✅ Tablet
- ✅ Mobile (iOS Safari, Android Chrome)

---

## 🚀 Next Steps

1. ✅ Frontend đã hoạt động
2. ⏭️ Deploy lên Vercel/Netlify (miễn phí)
3. ⏭️ Thêm tính năng:
   - History (lịch sử phân tích)
   - Export PDF report
   - Compare candidates
   - Real-time progress bar

---

## 📦 File Structure

```
frontend/
├── index.html          # Main file (all-in-one)
└── README.md          # This file
```

Chỉ 1 file HTML duy nhất! Không cần thư mục khác.

---

## 🎓 Tech Stack

- **HTML5**: Structure
- **CSS3**: Styling (Gradients, Flexbox, Grid)
- **Vanilla JavaScript**: Logic (Fetch API, DOM)
- **No frameworks**: Không dùng React/Vue/Angular
- **No build tools**: Không cần Webpack/Vite

---

## 💡 Tips

1. **Test nhanh**: Mở file HTML trực tiếp
2. **Development**: Dùng Live Server
3. **Production**: Deploy lên Vercel/Netlify
4. **Customize**: Tất cả code trong 1 file, dễ sửa

---

## 📝 License

Free to use!
