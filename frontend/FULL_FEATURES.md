# 🎯 Frontend Đầy Đủ Tính Năng

Frontend với **4 tabs** giống GUI desktop hiện tại.

## 🚀 Cách Chạy

### Bước 1: Chạy API
```bash
python api/main.py
```

### Bước 2: Mở Frontend
```bash
# Option A: Double-click
start frontend/app.html

# Option B: HTTP Server
cd frontend
python -m http.server 3000
# Mở: http://localhost:3000/app.html
```

---

## ✨ 4 Tabs Chính

### 1. 😊 Nhận Diện Cảm Xúc
- Upload video phỏng vấn
- Phân tích cảm xúc và tập trung
- Hiển thị điểm emotion, focus, total
- Đánh giá tổng quan

**Tính năng:**
- Drag & drop video
- Real-time status
- Kết quả dạng cards

### 2. 📹 Chuyển Đổi Video
- Upload video
- Chuyển đổi sang text (transcript)
- Hiển thị transcript đầy đủ
- Copy transcript

**Tính năng:**
- Hỗ trợ nhiều format (MP4, AVI, MOV)
- Transcript box với scroll
- Copy to clipboard

### 3. 🎤 Chuyển Đổi Audio
- Upload file audio (WAV, MP3)
- Chuyển đổi sang text
- Hiển thị transcript
- Copy transcript

**Tính năng:**
- Hỗ trợ audio files
- Whisper ASR
- Copy to clipboard

### 4. 📊 Tổng Hợp Điểm
- Upload video phỏng vấn
- Phân tích toàn diện (4 tiêu chí)
- Custom trọng số (%)
- Hiển thị điểm tổng + rating

**Tính năng:**
- ⚖️ Điều chỉnh trọng số:
  - 😊 Cảm Xúc (%)
  - 👁️ Tập Trung (%)
  - 🗣️ Rõ Ràng (%)
  - 📝 Nội Dung (%)
- Tự động tính điểm tổng
- Validation (tổng = 100%)
- Hiển thị rating (XUẤT SẮC, RẤT TỐT, etc.)

---

## 🎨 Giao Diện

### Dark Theme
- Background: #0f0f0f
- Cards: #1a1a1a, #252525
- Text: #e0e0e0
- Accent: Gradient (purple-pink)

### Responsive
- Desktop: 4 columns grid
- Tablet: 2 columns
- Mobile: 1 column

### Components
- ✅ Tabs navigation
- ✅ Upload areas (drag & drop)
- ✅ Status messages (info, success, error, warning)
- ✅ Result cards
- ✅ Progress indicators
- ✅ Weight controls (sliders)

---

## 📊 So Sánh Với GUI Desktop

| Tính Năng | Desktop GUI | Web Frontend |
|-----------|-------------|--------------|
| Nhận diện cảm xúc | ✅ | ✅ |
| Chuyển đổi video | ✅ | ✅ |
| Chuyển đổi audio | ✅ | ✅ |
| Tổng hợp điểm | ✅ | ✅ |
| Custom trọng số | ✅ | ✅ |
| Real-time camera | ✅ | ❌ (chưa) |
| Screen capture | ✅ | ❌ (chưa) |
| Export PDF | ✅ | ❌ (chưa) |

---

## 🔧 Customization

### Đổi màu theme

Trong `app.html`, tìm:
```css
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
```

Thay bằng:
```css
/* Blue theme */
background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);

/* Green theme */
background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
```

### Đổi API URL

Trong `app.js`:
```javascript
const API_URL = 'http://localhost:8001'; // Đổi port
```

### Thêm preset weights

Trong tab "Tổng Hợp Điểm", thêm buttons:
```html
<button onclick="setWeights(40, 35, 20, 5)">Default</button>
<button onclick="setWeights(45, 30, 20, 5)">Technical</button>
<button onclick="setWeights(35, 35, 20, 10)">Sales</button>
```

Thêm function:
```javascript
function setWeights(content, clarity, focus, emotion) {
    document.getElementById('weightContent').value = content;
    document.getElementById('weightClarity').value = clarity;
    document.getElementById('weightFocus').value = focus;
    document.getElementById('weightEmotion').value = emotion;
    updateTotalWeight();
}
```

---

## 🚀 Tính Năng Nâng Cao (Có Thể Thêm)

### 1. Real-time Camera
```javascript
// Sử dụng WebRTC
navigator.mediaDevices.getUserMedia({ video: true })
```

### 2. Progress Bar
```javascript
// Upload progress
xhr.upload.addEventListener('progress', (e) => {
    const percent = (e.loaded / e.total) * 100;
    updateProgressBar(percent);
});
```

### 3. History
```javascript
// Lưu vào localStorage
localStorage.setItem('history', JSON.stringify(results));
```

### 4. Export PDF
```javascript
// Sử dụng jsPDF
const doc = new jsPDF();
doc.text('Interview Analysis Report', 10, 10);
doc.save('report.pdf');
```

### 5. Compare Candidates
```javascript
// So sánh nhiều ứng viên
const candidates = [result1, result2, result3];
displayComparison(candidates);
```

---

## 📱 Mobile Support

Frontend responsive, hoạt động tốt trên:
- ✅ Desktop (Chrome, Firefox, Edge, Safari)
- ✅ Tablet (iPad, Android tablets)
- ✅ Mobile (iOS, Android)

**Lưu ý:**
- Upload file trên mobile có thể chậm hơn
- Recommend dùng WiFi cho video lớn

---

## 🐛 Troubleshooting

### Lỗi: "Không thể kết nối với API"
→ Chạy: `python api/main.py`

### Lỗi: CORS
→ Dùng HTTP server: `python -m http.server 3000`

### Lỗi: "Tổng trọng số phải bằng 100%"
→ Điều chỉnh các trọng số sao cho tổng = 100%

### Video không upload được
→ Check:
- File size < 100MB
- Format: MP4, AVI, MOV
- Video có âm thanh

---

## 📦 File Structure

```
frontend/
├── app.html           # Main HTML (4 tabs)
├── app.js            # JavaScript logic
├── index.html        # Simple version (1 tab)
├── README.md         # Simple version docs
└── FULL_FEATURES.md  # This file
```

---

## 🎓 Tech Stack

- **HTML5**: Structure, semantic tags
- **CSS3**: Flexbox, Grid, Gradients, Animations
- **Vanilla JavaScript**: Fetch API, DOM manipulation
- **No frameworks**: Pure HTML/CSS/JS
- **No build tools**: Chạy ngay

---

## 💡 Tips

1. **Development**: Dùng Live Server (VS Code extension)
2. **Testing**: Dùng Chrome DevTools
3. **Debugging**: Check Console (F12)
4. **Performance**: Compress videos trước khi upload

---

## 🎉 Kết Luận

Frontend đầy đủ tính năng, giống GUI desktop:
- ✅ 4 tabs chính
- ✅ Dark theme đẹp
- ✅ Responsive
- ✅ Custom weights
- ✅ Real-time status
- ✅ Error handling

**Ready to use!** 🚀
