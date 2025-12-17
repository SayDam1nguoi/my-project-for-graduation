# Hệ Thống Tính Điểm Phỏng Vấn

## Tổng Quan

Hệ thống đánh giá phỏng vấn dựa trên **4 tiêu chí chính**, tất cả đều trên thang **0-10 điểm**.

## 4 Tiêu Chí Đánh Giá

### 1. Cảm Xúc (Emotion) - 5%
- **Thang điểm**: 0-10
- **Trọng số**: 5%
- **Module**: `EmotionScoringEngine`
- **Đánh giá**: 4 tiêu chí con
  - Emotion Stability (Ổn định cảm xúc): 40%
  - Emotion-Content Alignment (Khớp cảm xúc & nội dung): 35%
  - Positive Ratio (Tỷ lệ tích cực): 15%
  - Negative Overload (Mức tiêu cực vượt ngưỡng): 10%

### 2. Tập Trung (Focus) - 20%
- **Thang điểm**: 0-10
- **Trọng số**: 20%
- **Module**: `AttentionDetector`
- **Đánh giá**: Sự tập trung, chú ý trong suốt buổi phỏng vấn

### 3. Rõ Ràng (Clarity) - 35%
- **Thang điểm**: 0-10
- **Trọng số**: 35%
- **Module**: `SpeechClarityAnalyzer`
- **Đánh giá**: Độ rõ ràng lời nói, phát âm, tốc độ nói

### 4. Nội Dung (Content) - 40%
- **Thang điểm**: 0-10
- **Trọng số**: 40%
- **Module**: `InterviewContentEvaluator`
- **Đánh giá**: Chất lượng câu trả lời, độ phù hợp với câu hỏi

## Công Thức Tính Điểm Tổng

```
Total Score = (Emotion × 5%) + (Focus × 20%) + (Clarity × 35%) + (Content × 40%)
```

### Ví Dụ Tính Toán

```
Emotion:  8.0/10 → 8.0 × 0.05 = 0.40
Focus:    7.5/10 → 7.5 × 0.20 = 1.50
Clarity:  8.5/10 → 8.5 × 0.35 = 2.98
Content:  9.0/10 → 9.0 × 0.40 = 3.60
─────────────────────────────────────
Total:    0.40 + 1.50 + 2.98 + 3.60 = 8.48/10
```

## Đóng Góp Tối Đa Của Từng Tiêu Chí

| Tiêu Chí | Điểm Tối Đa | Trọng Số | Đóng Góp Tối Đa |
|----------|-------------|----------|-----------------|
| Emotion  | 10          | 5%       | 0.50 điểm       |
| Focus    | 10          | 20%      | 2.00 điểm       |
| Clarity  | 10          | 35%      | 3.50 điểm       |
| Content  | 10          | 40%      | 4.00 điểm       |
| **Tổng** | -           | **100%** | **10.00 điểm**  |

## Trọng Số Theo Vị Trí

Hệ thống hỗ trợ custom trọng số theo từng vị trí:

### Default (Mặc định)
- **Emotion: 5%**, **Focus: 20%**, **Clarity: 35%**, **Content: 40%**
- Cân bằng giữa các tiêu chí, ưu tiên nội dung và rõ ràng

### Technical (Kỹ thuật)
- **Emotion: 5%**, **Focus: 20%**, **Clarity: 30%**, **Content: 45%**
- Ưu tiên: Nội dung kỹ thuật chuyên sâu

### Sales (Bán hàng)
- **Emotion: 10%**, **Focus: 20%**, **Clarity: 35%**, **Content: 35%**
- Ưu tiên: Cảm xúc tích cực và kỹ năng giao tiếp

### Customer Service (Chăm sóc khách hàng)
- **Emotion: 10%**, **Focus: 20%**, **Clarity: 40%**, **Content: 30%**
- Ưu tiên: Rõ ràng trong giao tiếp và thái độ tích cực

### Management (Quản lý)
- **Emotion: 5%**, **Focus: 20%**, **Clarity: 30%**, **Content: 45%**
- Ưu tiên: Nội dung chiến lược và tầm nhìn

## Cấu Hình

### Emotion Scoring Config
File: `config/emotion_scoring_config.yaml`

```yaml
# Trọng số 4 tiêu chí con (tổng = 1.0)
criterion_weights:
  emotion_stability: 0.40
  emotion_content_alignment: 0.35
  positive_ratio: 0.15
  negative_overload: 0.10

# Multiplier = 1.0 để giữ thang 0-10
final_score_multiplier: 1.0
```

### Overall Scorer Config
File: `src/evaluation/overall_interview_scorer.py`

```python
DEFAULT_WEIGHTS = {
    'emotion': 0.05,   # 5%
    'focus': 0.20,     # 20%
    'clarity': 0.35,   # 35%
    'content': 0.40    # 40%
}
```

## Quy Trình Tính Điểm

### 1. Thu Thập Điểm Từ Các Module
```python
emotion_score = emotion_engine.score_video_interview(...)  # 0-10
focus_score = attention_detector.calculate_attention(...)  # 0-10
clarity_score = speech_analyzer.analyze_clarity(...)       # 0-10
content_score = content_evaluator.evaluate_answers(...)    # 0-10
```

### 2. Tính Điểm Tổng
```python
total_score = overall_scorer.calculate_score(
    emotion_score=emotion_score,
    focus_score=focus_score,
    clarity_score=clarity_score,
    content_score=content_score
)
```

### 3. Kết Quả
```python
InterviewScore(
    emotion_score=8.0,
    focus_score=7.5,
    clarity_score=8.5,
    content_score=9.0,
    total_score=8.48,
    overall_rating="RẤT TỐT"
)
```

## Đánh Giá Tổng Quan

| Điểm Số | Đánh Giá | Khuyến Nghị |
|---------|----------|-------------|
| 9.0-10  | XUẤT SẮC | Tuyển dụng ngay |
| 8.0-8.9 | RẤT TỐT  | Tuyển dụng |
| 7.0-7.9 | TỐT      | Tuyển dụng có điều kiện |
| 6.0-6.9 | KHÁ      | Xem xét thêm |
| 5.0-5.9 | TRUNG BÌNH | Cân nhắc |
| < 5.0   | CẦN CẢI THIỆN | Không tuyển |

## Lưu Ý Quan Trọng

1. **Tất cả điểm đều 0-10**: Không có scale khác nhau giữa các tiêu chí
2. **Trọng số áp dụng 1 lần**: Chỉ ở tầng tổng hợp cuối cùng
3. **Có thể custom**: Trọng số có thể điều chỉnh theo vị trí hoặc yêu cầu
4. **Tổng trọng số = 100%**: Luôn đảm bảo tổng các trọng số = 100%

## Code Example

```python
from src.evaluation.integrated_interview_evaluator import IntegratedInterviewEvaluator

# Khởi tạo evaluator
evaluator = IntegratedInterviewEvaluator(position_type='default')

# Đánh giá phỏng vấn
score, details = evaluator.evaluate_video_interview(
    video_path='interview.mp4',
    candidate_id='CANDIDATE_001',
    answers={'q1': 'answer1', 'q2': 'answer2'}
)

# Kết quả
print(f"Total Score: {score.total_score}/10")
print(f"Rating: {score.overall_rating}")
print(f"Emotion: {score.emotion_score}/10 (5%)")
print(f"Focus: {score.focus_score}/10 (20%)")
print(f"Clarity: {score.clarity_score}/10 (35%)")
print(f"Content: {score.content_score}/10 (40%)")
```

## GUI Usage

Trong GUI, người dùng có thể:
1. Xem điểm từng tiêu chí (0-10)
2. Custom trọng số (%) cho từng tiêu chí
3. Xem điểm tổng tự động cập nhật
4. Chọn preset theo vị trí (Default, Technical, Sales, etc.)

Trọng số mặc định trong GUI:
- Nội dung: 40%
- Rõ ràng: 35%
- Tập trung: 20%
- Cảm xúc: 5%
- **Tổng: 100%**
