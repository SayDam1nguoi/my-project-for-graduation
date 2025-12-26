# 🎯 Interview Analysis System - Emotion Scanner 2

**Hệ thống phân tích phỏng vấn toàn diện với AI** - Đánh giá ứng viên qua video, audio và cảm xúc real-time.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Mục lục

- [Tổng quan](#-tổng-quan)
- [Tính năng chính](#-tính-năng-chính)
- [Kiến trúc hệ thống](#-kiến-trúc-hệ-thống)
- [Công nghệ sử dụng](#-công-nghệ-sử-dụng)
- [Quick Start](#-quick-start)
- [Hệ thống chấm điểm](#-hệ-thống-chấm-điểm)
- [API Documentation](#-api-documentation)
- [Kết quả & Metrics](#-kết-quả--metrics)
- [Limitations & Future Work](#-limitations--future-work)

---

## 🎯 Tổng quan

**Interview Analysis System** là hệ thống AI toàn diện để phân tích và đánh giá ứng viên trong quá trình phỏng vấn. Hệ thống cung cấp cả **Desktop GUI (Python)** và **Web Application** với đầy đủ tính năng.

### 🎭 Điểm nổi bật

- ✅ **Real-time Emotion Detection** - Phát hiện 7 cảm xúc với AI (face-api.js + DeepFace)
- ✅ **Video Analysis** - Phân tích cảm xúc, tập trung, hành vi từ video
- ✅ **Speech-to-Text** - Chuyển đổi audio/video sang text (Whisper)
- ✅ **Content Evaluation** - Đánh giá nội dung câu trả lời (Sentence Transformers)
- ✅ **Comprehensive Scoring** - Tính điểm tổng hợp từ 4 tiêu chí
- ✅ **Web & Desktop** - Cả 2 nền tảng với đầy đủ tính năng
- ✅ **Privacy-friendly** - Client-side processing cho web


---

## 🚀 Tính năng chính

### 1. 📹 Real-time Emotion Detection (Web)

**Tính năng mới nhất!** Phát hiện cảm xúc real-time trên browser với AI thật.

- **7 cảm xúc**: Happy, Sad, Angry, Fearful, Disgusted, Surprised, Neutral
- **Visual feedback**: Bounding box xanh + label với confidence score
- **10 FPS detection**: Smooth, không lag
- **Client-side**: Không upload frames lên server (privacy-friendly)
- **Technology**: face-api.js (TensorFlow.js)

```
Workflow:
Camera → WebRTC → face-api.js → Canvas overlay → Real-time display
```

### 2. 😊 Emotion Recognition (Desktop & Web)

Phân tích cảm xúc từ video với độ chính xác cao.

**Desktop (Python):**
- DeepFace với multiple models (VGG-Face, Facenet, OpenFace)
- MTCNN face detection (>95% accuracy)
- GPU acceleration (CUDA)
- Multi-face support

**Web (JavaScript):**
- face-api.js (TinyFaceDetector + FaceExpressionNet)
- Real-time detection (10 FPS)
- Bounding box + emotion label
- Confidence scores (0-100%)

**Scoring:**
- Emotion stability score (0-10)
- Weighted by emotion type
- Temporal smoothing

### 3. 👁️ Focus & Attention Analysis

Đánh giá mức độ tập trung của ứng viên.

**4 Components:**
1. **Face Presence** (40%): Khuôn mặt có trong frame
2. **Gaze Focus** (30%): Nhìn vào camera
3. **Head Focus** (20%): Đầu thẳng
4. **Drift Score** (10%): Không nhìn đi chỗ khác

**Formula:**
```
Focus Score = (FacePresence×40% + GazeFocus×30% + HeadFocus×20% + DriftScore×10%) × 10
```

**Output:** Score 0-10

### 4. 🗣️ Speech Clarity Analysis

Đánh giá độ rõ ràng khi nói.

**Metrics:**
- **Speech rate**: Tốc độ nói (words/minute)
- **Filler words**: Từ ngập ngừng (ừm, à, ...)
- **Pauses**: Độ dài các khoảng lặng
- **Articulation**: Phát âm rõ ràng

**Technology:**
- Whisper (OpenAI) - Speech-to-text
- Custom analysis algorithms
- Vietnamese language support

**Scoring:**
- Optimal speech rate: 120-160 wpm
- Filler word penalty
- Pause analysis
- Final score: 0-10

### 5. 📝 Content Evaluation

Đánh giá nội dung câu trả lời.

**Method:** Embedding-based Similarity (NOT RAG)

**5 Standard Questions:**
1. **Problem Solving** (35%): Giải quyết vấn đề khó
2. **Deadline Management** (20%): Làm việc dưới áp lực
3. **Teamwork** (20%): Làm việc nhóm
4. **Communication** (15%): Thuyết phục người khác
5. **Achievement** (10%): Thành tựu tự hào

**Each question has 3-5 sample answers**

**Scoring Process:**
```
1. Compute embeddings (Sentence Transformers)
2. Calculate cosine similarity with sample answers
3. Take MAX similarity (best match)
4. Convert to score (0-10) with smooth interpolation
5. Weight by question importance
```

**Model:** `paraphrase-multilingual-MiniLM-L12-v2`

### 6. 📊 Comprehensive Scoring

Tính điểm tổng hợp từ 4 tiêu chí.

**4 Criteria:**
- 😊 **Emotion** (5%): Ổn định cảm xúc
- 👁️ **Focus** (20%): Tập trung
- 🗣️ **Clarity** (35%): Rõ ràng
- 📝 **Content** (40%): Nội dung

**Formula:**
```
Total Score = Emotion×5% + Focus×20% + Clarity×35% + Content×40%
```

**Custom Weights:** User có thể điều chỉnh (phải tổng = 100%)

**Rating Scale:**
- 9.0-10.0: XUẤT SẮC
- 8.0-8.9: RẤT TỐT
- 7.0-7.9: TỐT
- 6.0-6.9: KHÁ
- 5.0-5.9: TRUNG BÌNH
- <5.0: CẦN CẢI THIỆN


---

## 🏗️ Kiến trúc hệ thống

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERFACES                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────┐         ┌──────────────────┐        │
│  │  Desktop GUI     │         │  Web Application │        │
│  │  (Python/Tkinter)│         │  (HTML/CSS/JS)   │        │
│  └────────┬─────────┘         └────────┬─────────┘        │
│           │                            │                   │
└───────────┼────────────────────────────┼───────────────────┘
            │                            │
            │                            │ HTTP/REST
            ▼                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    BACKEND SERVICES                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────┐         ┌──────────────────┐        │
│  │  Core Engine     │         │  FastAPI Server  │        │
│  │  (Python)        │◄────────┤  (REST API)      │        │
│  └────────┬─────────┘         └──────────────────┘        │
│           │                                                 │
│           ├─► Video Analysis                               │
│           ├─► Speech Analysis                              │
│           ├─► Emotion Detection                            │
│           └─► Scoring Engine                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
            │
            │ Process
            ▼
┌─────────────────────────────────────────────────────────────┐
│                    AI MODELS & LIBRARIES                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  • DeepFace (Emotion Detection)                            │
│  • MTCNN (Face Detection)                                  │
│  • Whisper (Speech-to-Text)                                │
│  • Sentence Transformers (Content Evaluation)              │
│  • face-api.js (Web Real-time Detection)                   │
│  • TensorFlow / PyTorch                                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Video/Audio Input
  ↓
┌─────────────────────┐
│  Preprocessing      │
│  - Extract frames   │
│  - Extract audio    │
│  - Normalize        │
└──────────┬──────────┘
           │
           ├─────────────────────────────────┐
           │                                 │
           ▼                                 ▼
┌──────────────────────┐         ┌──────────────────────┐
│  Video Analysis      │         │  Audio Analysis      │
│  - Face detection    │         │  - Speech-to-text    │
│  - Emotion recog     │         │  - Clarity metrics   │
│  - Attention track   │         │  - Content eval      │
└──────────┬───────────┘         └──────────┬───────────┘
           │                                 │
           └─────────────┬───────────────────┘
                         ▼
                ┌─────────────────┐
                │  Score Engine   │
                │  - Weighted sum │
                │  - Rating       │
                └────────┬────────┘
                         ▼
                ┌─────────────────┐
                │  Final Report   │
                │  - Total score  │
                │  - 4 scores     │
                │  - Rating       │
                │  - Details      │
                └─────────────────┘
```


---

## 🛠️ Công nghệ sử dụng

### Backend (Python)

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Face Detection** | MTCNN | Phát hiện khuôn mặt (>95% accuracy) |
| **Emotion Recognition** | DeepFace | Nhận diện 7 cảm xúc |
| **Speech-to-Text** | Whisper (OpenAI) | Chuyển đổi audio sang text |
| **Content Evaluation** | Sentence Transformers | Đánh giá nội dung câu trả lời |
| **API Server** | FastAPI | REST API backend |
| **GUI** | Tkinter | Desktop application |
| **Deep Learning** | TensorFlow, PyTorch | Model training & inference |

### Frontend (Web)

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Face Detection** | face-api.js | Real-time face detection |
| **Emotion Recognition** | FaceExpressionNet | 7 emotions với confidence |
| **Camera Access** | WebRTC API | Truy cập camera |
| **Recording** | MediaRecorder API | Ghi video với audio |
| **UI Framework** | Vanilla JavaScript | No frameworks, pure JS |
| **Styling** | CSS3 | Dark theme, responsive |

### Models & Algorithms

**Emotion Detection:**
- VGG-Face, Facenet, OpenFace (Desktop)
- TinyFaceDetector + FaceExpressionNet (Web)

**Speech-to-Text:**
- Whisper Large V3 (Vietnamese optimized)
- Custom vocabulary support

**Content Evaluation:**
- Model: `paraphrase-multilingual-MiniLM-L12-v2`
- Cosine similarity matching
- Smooth interpolation scoring

**Attention Detection:**
- Head pose estimation
- Gaze direction tracking
- Temporal smoothing


---

## 🚀 Quick Start

### Option 1: Web Application (Khuyến nghị)

**Bước 1: Khởi động API**
```bash
python api/main.py
```

**Bước 2: Mở Frontend**
- Double-click `start_web.bat` (Windows)
- Hoặc double-click `frontend/app.html`

**Bước 3: Sử dụng**
1. Click tab "📹 Camera Trực Tiếp"
2. Click "📹 Bật Camera"
3. Cho phép quyền camera
4. Xem real-time emotion detection!

### Option 2: Desktop GUI

```bash
# Cài đặt dependencies
pip install -r requirements.txt

# Khởi động GUI
python launcher.py
```

### Cài đặt đầy đủ

**1. Clone repository**
```bash
git clone https://github.com/your-repo/emotion-scanner-2.git
cd emotion-scanner-2
```

**2. Tạo virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

**3. Cài đặt dependencies**
```bash
pip install -r requirements.txt
```

**4. Download models (optional)**
```bash
# DeepFace models sẽ tự động download khi chạy lần đầu
# Whisper models cũng tự động download
```

**5. Chạy ứng dụng**
```bash
# Desktop GUI
python launcher.py

# Web API
python api/main.py

# Web Frontend
# Mở frontend/app.html trong browser
```

### Requirements

**Minimum:**
- Python 3.8+
- 4GB RAM
- CPU: Intel i5 hoặc tương đương

**Recommended:**
- Python 3.9+
- 8GB RAM
- GPU: NVIDIA GTX 1060+ (CUDA support)
- SSD storage

**Browser (for Web):**
- Chrome 90+ (recommended)
- Edge 90+
- Firefox 88+
- Safari 14+ (limited support)


---

## 📊 Hệ thống chấm điểm

### Scoring Formula

```
Total Score = Emotion×5% + Focus×20% + Clarity×35% + Content×40%
```

### 1. Emotion Score (0-10)

**Calculation:**
```python
emotion_stability = 1.0 - emotion_variance
emotion_positivity = (happy + surprise) / total_frames
emotion_score = (emotion_stability × 0.6 + emotion_positivity × 0.4) × 10
```

**Factors:**
- Emotion stability (60%): Ít thay đổi cảm xúc
- Emotion positivity (40%): Nhiều cảm xúc tích cực

### 2. Focus Score (0-10)

**Formula:**
```
Focus = (FacePresence×40% + GazeFocus×30% + HeadFocus×20% + DriftScore×10%) × 10
```

**Components:**
- **Face Presence** (40%): Mặt có trong frame
- **Gaze Focus** (30%): Nhìn vào camera
- **Head Focus** (20%): Đầu thẳng
- **Drift Score** (10%): Không nhìn đi chỗ khác

### 3. Clarity Score (0-10)

**Metrics:**
- Speech rate: 120-160 wpm (optimal)
- Filler words: <5% (good)
- Pause duration: <2s (good)
- Articulation: Clear pronunciation

**Scoring:**
```python
speech_rate_score = calculate_speech_rate_score(wpm)
filler_penalty = filler_word_count × 0.5
pause_penalty = long_pause_count × 0.3
clarity_score = max(0, 10 - filler_penalty - pause_penalty)
```

### 4. Content Score (0-10)

**Method:** Embedding-based Similarity

**Process:**
1. Compute embeddings for applicant answer
2. Compute embeddings for 3-5 sample answers
3. Calculate cosine similarity
4. Take MAX similarity (best match)
5. Convert to score with smooth interpolation

**Similarity → Score Mapping:**
- 0.85-1.0 → 9.0-10.0 (Xuất sắc)
- 0.75-0.85 → 7.5-9.0 (Rất tốt)
- 0.65-0.75 → 6.0-7.5 (Tốt)
- 0.50-0.65 → 4.0-6.0 (Trung bình)
- 0.30-0.50 → 2.0-4.0 (Yếu)
- 0.0-0.30 → 0.0-2.0 (Rất yếu)

### Rating Scale

| Score | Rating | Decision |
|-------|--------|----------|
| 9.0-10.0 | XUẤT SẮC | Tuyển ngay |
| 8.0-8.9 | RẤT TỐT | Tuyển |
| 7.0-7.9 | TỐT | Tuyển có điều kiện |
| 6.0-6.9 | KHÁ | Xem xét |
| 5.0-5.9 | TRUNG BÌNH | Xem xét kỹ |
| <5.0 | CẦN CẢI THIỆN | Không tuyển |

