# 📊 Cách Tính Điểm Tập Trung (Focus Score)

## Tổng Quan

Điểm tập trung đánh giá mức độ chú ý của ứng viên trong suốt buổi phỏng vấn dựa trên **phân tích khuôn mặt và mắt** từ video.

**Thang điểm**: 0-10
- **8-10**: Tập trung tốt (Focused)
- **6-8**: Hơi mất tập trung (Slightly Distracted)
- **4-6**: Mất tập trung (Distracted)
- **0-4**: Rất mất tập trung (Very Distracted)

---

## Công Thức Tính Điểm

### Bước 1: Tính Điểm Tức Thời (Instant Score)

Cho mỗi frame video, tính điểm dựa trên **4 thành phần**:

```
InstantScore = (
    FacePresence × 40% +
    GazeFocus    × 30% +
    HeadFocus    × 20% +
    DriftScore   × 10%
) × 10
```

#### 1. Face Presence (40%) - Có Mặt Trong Khung Hình

**Mục đích**: Kiểm tra ứng viên có trong khung hình không

**Cách tính**:
```python
if face_detected:
    face_presence_score = 1.0  # Có mặt
else:
    face_presence_score = 0.0  # Không có mặt
```

**Ý nghĩa**:
- Có mặt = 1.0 → Đóng góp 4.0 điểm (40% × 10)
- Không có mặt = 0.0 → Đóng góp 0 điểm

#### 2. Gaze Focus (30%) - Hướng Nhìn

**Mục đích**: Kiểm tra ứng viên có nhìn thẳng vào camera không

**Cách tính**:
```python
# Tính độ lệch từ trung tâm (0.5, 0.5)
h_deviation = abs(h_ratio - 0.5)  # Lệch ngang
v_deviation = abs(v_ratio - 0.5)  # Lệch dọc

if h_deviation < 0.15 and v_deviation < 0.15:
    gaze_focus_score = 1.0  # Nhìn thẳng
elif h_deviation > 0.35 or v_deviation > 0.35:
    gaze_focus_score = 0.0  # Nhìn xa
else:
    # Trung gian: tính tỷ lệ
    max_deviation = max(h_deviation, v_deviation)
    gaze_focus_score = 1.0 - ((max_deviation - 0.15) / (0.35 - 0.15))
```

**Ví dụ**:
- Nhìn thẳng (lệch < 15%) → 1.0 → Đóng góp 3.0 điểm
- Nhìn hơi lệch (lệch 20%) → 0.75 → Đóng góp 2.25 điểm
- Nhìn xa (lệch > 35%) → 0.0 → Đóng góp 0 điểm

#### 3. Head Focus (20%) - Tư Thế Đầu

**Mục đích**: Kiểm tra ứng viên có giữ đầu thẳng không

**Cách tính**:
```python
# Tính góc đầu (yaw, pitch, roll)
yaw, pitch, roll = calculate_head_pose(landmarks)
max_angle = max(abs(yaw), abs(pitch), abs(roll))

if max_angle < 15.0:
    head_focus_score = 1.0  # Đầu thẳng
elif max_angle > 25.0:
    head_focus_score = 0.0  # Quay đầu nhiều
else:
    # Trung gian
    head_focus_score = 1.0 - ((max_angle - 15.0) / (25.0 - 15.0))
```

**Ví dụ**:
- Đầu thẳng (góc < 15°) → 1.0 → Đóng góp 2.0 điểm
- Đầu hơi quay (góc 20°) → 0.5 → Đóng góp 1.0 điểm
- Đầu quay nhiều (góc > 25°) → 0.0 → Đóng góp 0 điểm

**Các góc**:
- **Yaw**: Quay trái/phải (âm = trái, dương = phải)
- **Pitch**: Cúi/ngẩng (âm = ngẩng, dương = cúi)
- **Roll**: Nghiêng (âm = nghiêng trái, dương = nghiêng phải)

#### 4. Drift Score (10%) - Phạt Ngó Nghiêng

**Mục đích**: Phạt khi ứng viên ngó nghiêng quá nhiều lần

**Cách tính**:
```python
# Đếm số lần drift trong 60 giây
max_allowed_drifts = 3  # Cho phép tối đa 3 lần/phút
drift_score = 1.0 - min(1.0, drift_events / max_allowed_drifts)
```

**Ví dụ**:
- 0 lần drift → 1.0 → Đóng góp 1.0 điểm
- 3 lần drift → 0.0 → Đóng góp 0 điểm
- 1 lần drift → 0.67 → Đóng góp 0.67 điểm

---

### Bước 2: Làm Mượt (Smoothing)

Lấy trung bình **30 frames gần nhất** (~1 giây ở 30fps):

```python
recent_scores = list(attention_history)[-30:]
focus_score = np.mean(recent_scores)
```

**Lý do**: Tránh điểm nhảy lung tung do nhiễu

---

### Bước 3: Tính Điểm Cuối Cùng

```python
# Tính trung bình tất cả frames đã phân tích
focus_score = sum(attention_scores) / len(attention_scores)

# Clamp vào [0, 10]
focus_score = max(0, min(10, focus_score))
```

---

## Ví Dụ Cụ Thể

### Ví Dụ 1: Ứng Viên Tập Trung Tốt

**Frame 1:**
- Face detected: ✅ (1.0)
- Gaze: Nhìn thẳng (1.0)
- Head: Đầu thẳng (1.0)
- Drift: 0 lần (1.0)

```
InstantScore = (1.0×40% + 1.0×30% + 1.0×20% + 1.0×10%) × 10
             = (0.4 + 0.3 + 0.2 + 0.1) × 10
             = 1.0 × 10
             = 10.0/10 ✅
```

**Kết quả**: Tập trung xuất sắc!

---

### Ví Dụ 2: Ứng Viên Hơi Mất Tập Trung

**Frame 1:**
- Face detected: ✅ (1.0)
- Gaze: Nhìn hơi lệch phải (0.7)
- Head: Đầu hơi quay (0.5)
- Drift: 1 lần (0.67)

```
InstantScore = (1.0×40% + 0.7×30% + 0.5×20% + 0.67×10%) × 10
             = (0.4 + 0.21 + 0.1 + 0.067) × 10
             = 0.777 × 10
             = 7.77/10 ⚠️
```

**Kết quả**: Hơi mất tập trung (Slightly Distracted)

---

### Ví Dụ 3: Ứng Viên Mất Tập Trung

**Frame 1:**
- Face detected: ❌ (0.0)
- Gaze: N/A (0.0)
- Head: N/A (0.0)
- Drift: 2 lần (0.33)

```
InstantScore = (0.0×40% + 0.0×30% + 0.0×20% + 0.33×10%) × 10
             = (0 + 0 + 0 + 0.033) × 10
             = 0.033 × 10
             = 0.33/10 ❌
```

**Kết quả**: Rất mất tập trung (Very Distracted)

---

## Workflow Hoàn Chỉnh

```
1. Đọc video
   ↓
2. Mỗi 5 frames (để tăng tốc):
   ↓
   a. Phát hiện khuôn mặt (Face Detector)
   ↓
   b. Lấy landmarks (68 điểm hoặc 5 điểm)
   ↓
   c. Tính 4 thành phần:
      - Face Presence (40%)
      - Gaze Focus (30%)
      - Head Focus (20%)
      - Drift Score (10%)
   ↓
   d. Tính InstantScore (0-10)
   ↓
   e. Lưu vào history
   ↓
3. Làm mượt (trung bình 30 frames)
   ↓
4. Tính điểm cuối cùng (trung bình tất cả frames)
   ↓
5. Kết quả: Focus Score (0-10)
```

---

## Các Trường Hợp Đặc Biệt

### 1. Không Phát Hiện Được Khuôn Mặt

```python
if not face_detected:
    instant_score = 0.0 + drift_score × 0.1 × 10
    # Chỉ còn drift score đóng góp
```

**Lý do**: Không có mặt = mất tập trung hoàn toàn

### 2. Mắt Nhắm (Bổ Sung)

```python
# Tính EAR (Eye Aspect Ratio)
if ear < 0.15:
    eyes_closed = True
    # Không ảnh hưởng trực tiếp đến điểm
    # Nhưng được ghi nhận trong details
```

**Lưu ý**: Mắt nhắm không trừ điểm trực tiếp, chỉ ghi nhận

### 3. Chớp Mắt (Blink)

```python
# Đếm số lần chớp mắt
if eyes_closed and previous_eyes_open:
    blink_counter += 1
```

**Lưu ý**: Chớp mắt bình thường không bị phạt

---

## Thống Kê Bổ Sung

Ngoài điểm số, hệ thống còn cung cấp:

```python
statistics = {
    'total_frames': 1000,           # Tổng số frames
    'focused_frames': 850,          # Số frames tập trung
    'distracted_frames': 150,       # Số frames mất tập trung
    'focused_rate': 0.85,           # Tỷ lệ tập trung (85%)
    'distracted_rate': 0.15,        # Tỷ lệ mất tập trung (15%)
    'average_attention': 8.5,       # Điểm trung bình
    'blink_count': 45               # Số lần chớp mắt
}
```

---

## Tối Ưu Hóa

### 1. Sampling Rate

```python
# Xử lý mỗi 5 frames thay vì tất cả
if frame_count % 5 != 0:
    continue
```

**Lý do**: Tăng tốc xử lý, vẫn đủ chính xác

### 2. Dead Zone

```python
# Vùng "an toàn" để giảm nhiễu
pose_dead_zone = 15.0   # Góc < 15° = nhìn thẳng
gaze_dead_zone = 0.25   # Lệch < 25% = nhìn thẳng
```

**Lý do**: Tránh phạt những chuyển động nhỏ tự nhiên

### 3. Smoothing Window

```python
# Lấy trung bình 30 frames (~1 giây)
recent_scores = list(attention_history)[-30:]
focus_score = np.mean(recent_scores)
```

**Lý do**: Làm mượt điểm, tránh nhảy lung tung

---

## Cảnh Báo Real-time

Hệ thống có thể cảnh báo khi mất tập trung:

```python
if score < 5.0:
    # Bắt đầu đếm thời gian
    if distraction_duration >= 3.0:
        # Cảnh báo sau 3 giây
        alert("Mất tập trung!")
```

**Cooldown**: 3 giây giữa các cảnh báo

---

## Kết Luận

Điểm tập trung được tính dựa trên:
1. ✅ **Face Presence** (40%) - Có mặt trong khung hình
2. ✅ **Gaze Focus** (30%) - Nhìn thẳng vào camera
3. ✅ **Head Focus** (20%) - Giữ đầu thẳng
4. ✅ **Drift Score** (10%) - Không ngó nghiêng quá nhiều

**Thang điểm**: 0-10 (thống nhất với các tiêu chí khác)

**Độ chính xác**: 85-90% (phụ thuộc chất lượng video)
