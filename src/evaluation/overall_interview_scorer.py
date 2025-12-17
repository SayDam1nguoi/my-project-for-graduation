"""
Overall Interview Scorer

Kết hợp 4 đầu điểm để tạo điểm tổng hợp cho phỏng vấn:
1. Cảm xúc (Emotion) - 5% (0-10 điểm)
2. Tập trung (Focus/Attention) - 20% (0-10 điểm)
3. Rõ ràng lời nói (Speech Clarity) - 35% (0-10 điểm)
4. Nội dung (Content) - 40% (0-10 điểm)

Tất cả điểm đều trên thang 0-10, trọng số được áp dụng khi tính tổng.
"""

import logging
from typing import Dict, Optional, List
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class InterviewScore:
    """Điểm đánh giá phỏng vấn tổng hợp."""
    
    # Điểm từng tiêu chí (0-10)
    emotion_score: float
    focus_score: float
    clarity_score: float
    content_score: float
    
    # Trọng số từng tiêu chí
    emotion_weight: float = 0.25
    focus_weight: float = 0.25
    clarity_weight: float = 0.25
    content_weight: float = 0.25
    
    # Điểm tổng (0-10)
    total_score: float = 0.0
    
    # Đánh giá tổng quan
    overall_rating: str = ""
    
    # Chi tiết từng tiêu chí
    details: Dict = None
    
    def __post_init__(self):
        """Tính điểm tổng sau khi khởi tạo."""
        self.calculate_total_score()
        self.determine_rating()
    
    def calculate_total_score(self):
        """Tính điểm tổng có trọng số."""
        self.total_score = (
            self.emotion_score * self.emotion_weight +
            self.focus_score * self.focus_weight +
            self.clarity_score * self.clarity_weight +
            self.content_score * self.content_weight
        )
        
        # Đảm bảo trong khoảng 0-10
        self.total_score = max(0.0, min(10.0, self.total_score))
    
    def determine_rating(self):
        """Xác định đánh giá tổng quan."""
        if self.total_score >= 9.0:
            self.overall_rating = "XUẤT SẮC"
        elif self.total_score >= 8.0:
            self.overall_rating = "RẤT TỐT"
        elif self.total_score >= 7.0:
            self.overall_rating = "TỐT"
        elif self.total_score >= 6.0:
            self.overall_rating = "KHÁ"
        elif self.total_score >= 5.0:
            self.overall_rating = "TRUNG BÌNH"
        else:
            self.overall_rating = "CẦN CẢI THIỆN"


class OverallInterviewScorer:
    """
    Hệ thống chấm điểm tổng hợp cho phỏng vấn.
    
    Kết hợp 4 tiêu chí (tất cả đều 0-10 điểm):
    1. Cảm xúc (Emotion) - 5% - Đánh giá biểu cảm, thái độ
    2. Tập trung (Focus) - 20% - Đánh giá sự tập trung, chú ý
    3. Rõ ràng (Clarity) - 35% - Đánh giá độ rõ ràng lời nói
    4. Nội dung (Content) - 40% - Đánh giá chất lượng câu trả lời
    
    Công thức: Total = (E×5% + F×20% + C×35% + N×40%) = 0-10 điểm
    """
    
    # Cấu hình trọng số mặc định
    # Emotion: 5%, Focus: 20%, Clarity: 35%, Content: 40%
    DEFAULT_WEIGHTS = {
        'emotion': 0.05,   # 5% - Cảm xúc (0-10 điểm)
        'focus': 0.20,     # 20% - Tập trung (0-10 điểm)
        'clarity': 0.35,   # 35% - Rõ ràng lời nói (0-10 điểm)
        'content': 0.40    # 40% - Nội dung (0-10 điểm)
    }
    
    # Cấu hình trọng số cho các vị trí khác nhau
    POSITION_WEIGHTS = {
        'technical': {  # Vị trí kỹ thuật
            'emotion': 0.05,   # 5% - Cảm xúc ít quan trọng
            'focus': 0.20,     # 20% - Tập trung
            'clarity': 0.30,   # 30% - Rõ ràng
            'content': 0.45    # 45% - Nội dung quan trọng nhất
        },
        'sales': {  # Vị trí bán hàng
            'emotion': 0.10,   # 10% - Cảm xúc quan trọng hơn
            'focus': 0.20,     # 20% - Tập trung
            'clarity': 0.35,   # 35% - Rõ ràng quan trọng
            'content': 0.35    # 35% - Nội dung
        },
        'customer_service': {  # Vị trí chăm sóc khách hàng
            'emotion': 0.10,   # 10% - Cảm xúc quan trọng
            'focus': 0.20,     # 20% - Tập trung
            'clarity': 0.40,   # 40% - Rõ ràng rất quan trọng
            'content': 0.30    # 30% - Nội dung
        },
        'management': {  # Vị trí quản lý
            'emotion': 0.05,   # 5% - Cảm xúc
            'focus': 0.20,     # 20% - Tập trung
            'clarity': 0.30,   # 30% - Rõ ràng
            'content': 0.45    # 45% - Nội dung quan trọng nhất
        },
        'default': DEFAULT_WEIGHTS
    }
    
    def __init__(self, position_type: str = 'default'):
        """
        Khởi tạo scorer.
        
        Args:
            position_type: Loại vị trí (technical, sales, customer_service, management, default)
        """
        self.position_type = position_type
        self.weights = self.POSITION_WEIGHTS.get(position_type, self.DEFAULT_WEIGHTS)
        
        logger.info(f"Initialized OverallInterviewScorer for position: {position_type}")
        logger.info(f"Weights: {self.weights}")
    
    def calculate_score(
        self,
        emotion_score: float,
        focus_score: float,
        clarity_score: float,
        content_score: float,
        custom_weights: Optional[Dict[str, float]] = None
    ) -> InterviewScore:
        """
        Tính điểm tổng hợp.
        
        Args:
            emotion_score: Điểm cảm xúc (0-10)
            focus_score: Điểm tập trung (0-10)
            clarity_score: Điểm rõ ràng (0-10)
            content_score: Điểm nội dung (0-10)
            custom_weights: Trọng số tùy chỉnh (optional)
            
        Returns:
            InterviewScore object
        """
        # Sử dụng trọng số tùy chỉnh hoặc mặc định
        weights = custom_weights if custom_weights else self.weights
        
        # Tạo InterviewScore
        score = InterviewScore(
            emotion_score=emotion_score,
            focus_score=focus_score,
            clarity_score=clarity_score,
            content_score=content_score,
            emotion_weight=weights['emotion'],
            focus_weight=weights['focus'],
            clarity_weight=weights['clarity'],
            content_weight=weights['content']
        )
        
        # Thêm chi tiết
        score.details = self._create_details(score)
        
        logger.info(f"Calculated overall score: {score.total_score:.2f}/10 - {score.overall_rating}")
        
        return score
    
    def _create_details(self, score: InterviewScore) -> Dict:
        """Tạo chi tiết đánh giá."""
        return {
            'scores': {
                'emotion': {
                    'score': score.emotion_score,
                    'weight': score.emotion_weight,
                    'contribution': score.emotion_score * score.emotion_weight,
                    'rating': self._get_rating(score.emotion_score)
                },
                'focus': {
                    'score': score.focus_score,
                    'weight': score.focus_weight,
                    'contribution': score.focus_score * score.focus_weight,
                    'rating': self._get_rating(score.focus_score)
                },
                'clarity': {
                    'score': score.clarity_score,
                    'weight': score.clarity_weight,
                    'contribution': score.clarity_score * score.clarity_weight,
                    'rating': self._get_rating(score.clarity_score)
                },
                'content': {
                    'score': score.content_score,
                    'weight': score.content_weight,
                    'contribution': score.content_score * score.content_weight,
                    'rating': self._get_rating(score.content_score)
                }
            },
            'strengths': self._identify_strengths(score),
            'weaknesses': self._identify_weaknesses(score),
            'recommendations': self._generate_recommendations(score)
        }
    
    def _get_rating(self, score: float) -> str:
        """Lấy đánh giá cho một điểm số."""
        if score >= 9.0:
            return "Xuất sắc"
        elif score >= 8.0:
            return "Rất tốt"
        elif score >= 7.0:
            return "Tốt"
        elif score >= 6.0:
            return "Khá"
        elif score >= 5.0:
            return "Trung bình"
        else:
            return "Cần cải thiện"
    
    def _identify_strengths(self, score: InterviewScore) -> List[str]:
        """Xác định điểm mạnh."""
        strengths = []
        
        scores = {
            'Cảm xúc': score.emotion_score,
            'Tập trung': score.focus_score,
            'Rõ ràng': score.clarity_score,
            'Nội dung': score.content_score
        }
        
        # Điểm >= 8 là điểm mạnh
        for name, value in scores.items():
            if value >= 8.0:
                strengths.append(f"{name} xuất sắc ({value:.1f}/10)")
        
        return strengths if strengths else ["Cần cải thiện tất cả các tiêu chí"]
    
    def _identify_weaknesses(self, score: InterviewScore) -> List[str]:
        """Xác định điểm yếu."""
        weaknesses = []
        
        scores = {
            'Cảm xúc': score.emotion_score,
            'Tập trung': score.focus_score,
            'Rõ ràng': score.clarity_score,
            'Nội dung': score.content_score
        }
        
        # Điểm < 6 là điểm yếu
        for name, value in scores.items():
            if value < 6.0:
                weaknesses.append(f"{name} cần cải thiện ({value:.1f}/10)")
        
        return weaknesses if weaknesses else ["Không có điểm yếu đáng kể"]
    
    def _generate_recommendations(self, score: InterviewScore) -> List[str]:
        """Tạo khuyến nghị cải thiện."""
        recommendations = []
        
        # Cảm xúc
        if score.emotion_score < 6.0:
            recommendations.append("Cảm xúc: Cần thể hiện thái độ tích cực, tự tin hơn")
        elif score.emotion_score < 8.0:
            recommendations.append("Cảm xúc: Tốt, có thể cải thiện thêm sự nhiệt tình")
        
        # Tập trung
        if score.focus_score < 6.0:
            recommendations.append("Tập trung: Cần duy trì sự chú ý, tránh mất tập trung")
        elif score.focus_score < 8.0:
            recommendations.append("Tập trung: Khá tốt, cần duy trì ổn định hơn")
        
        # Rõ ràng
        if score.clarity_score < 6.0:
            recommendations.append("Rõ ràng: Cần nói chậm hơn, rõ ràng hơn, giảm từ ngập ngừng")
        elif score.clarity_score < 8.0:
            recommendations.append("Rõ ràng: Tốt, có thể cải thiện tốc độ nói và giảm filler words")
        
        # Nội dung
        if score.content_score < 6.0:
            recommendations.append("Nội dung: Cần chuẩn bị kỹ hơn, thêm chi tiết và ví dụ cụ thể")
        elif score.content_score < 8.0:
            recommendations.append("Nội dung: Khá tốt, cần thêm số liệu và kết quả đo lường được")
        
        return recommendations if recommendations else ["Tiếp tục duy trì phong độ tốt!"]
    
    def generate_report(self, score: InterviewScore) -> str:
        """
        Tạo báo cáo đánh giá chi tiết.
        
        Args:
            score: InterviewScore object
            
        Returns:
            Báo cáo dạng text
        """
        report = []
        
        report.append("="*80)
        report.append("BÁO CÁO ĐÁNH GIÁ PHỎNG VẤN TỔNG HỢP")
        report.append("="*80)
        report.append("")
        
        # Điểm tổng
        report.append(f"ĐIỂM TỔNG: {score.total_score:.2f}/10 - {score.overall_rating}")
        report.append("")
        
        # Biểu đồ thanh
        bar_length = int(score.total_score)
        bar = "█" * bar_length + "░" * (10 - bar_length)
        report.append(f"[{bar}] {score.total_score:.1f}/10")
        report.append("")
        
        # Chi tiết từng tiêu chí
        report.append("-"*80)
        report.append("CHI TIẾT TỪNG TIÊU CHÍ:")
        report.append("-"*80)
        report.append("")
        
        criteria = [
            ('Cảm xúc (Emotion)', score.emotion_score, score.emotion_weight),
            ('Tập trung (Focus)', score.focus_score, score.focus_weight),
            ('Rõ ràng (Clarity)', score.clarity_score, score.clarity_weight),
            ('Nội dung (Content)', score.content_score, score.content_weight)
        ]
        
        for name, value, weight in criteria:
            contribution = value * weight
            rating = self._get_rating(value)
            bar = "█" * int(value) + "░" * (10 - int(value))
            
            report.append(f"{name}:")
            report.append(f"  Điểm: {value:.1f}/10 - {rating}")
            report.append(f"  Trọng số: {weight*100:.0f}%")
            report.append(f"  Đóng góp: {contribution:.2f} điểm")
            report.append(f"  [{bar}]")
            report.append("")
        
        # Điểm mạnh
        report.append("-"*80)
        report.append("ĐIỂM MẠNH:")
        report.append("-"*80)
        for strength in score.details['strengths']:
            report.append(f"  ✓ {strength}")
        report.append("")
        
        # Điểm yếu
        report.append("-"*80)
        report.append("ĐIỂM YẾU:")
        report.append("-"*80)
        for weakness in score.details['weaknesses']:
            report.append(f"  ✗ {weakness}")
        report.append("")
        
        # Khuyến nghị
        report.append("-"*80)
        report.append("KHUYẾN NGHỊ CẢI THIỆN:")
        report.append("-"*80)
        for i, rec in enumerate(score.details['recommendations'], 1):
            report.append(f"  {i}. {rec}")
        report.append("")
        
        # Kết luận
        report.append("="*80)
        report.append("KẾT LUẬN:")
        report.append("="*80)
        
        if score.total_score >= 8.0:
            report.append("Ứng viên có màn thể hiện xuất sắc. Đề xuất TUYỂN DỤNG.")
        elif score.total_score >= 7.0:
            report.append("Ứng viên có màn thể hiện tốt. Đề xuất TUYỂN DỤNG với điều kiện.")
        elif score.total_score >= 6.0:
            report.append("Ứng viên có màn thể hiện khá. Cần XEM XÉT thêm.")
        else:
            report.append("Ứng viên cần cải thiện nhiều. Đề xuất KHÔNG TUYỂN hoặc phỏng vấn lại.")
        
        report.append("="*80)
        
        return "\n".join(report)


# Hàm tiện ích
def quick_score(
    emotion: float,
    focus: float,
    clarity: float,
    content: float,
    position: str = 'default'
) -> InterviewScore:
    """
    Hàm tiện ích để tính điểm nhanh.
    
    Args:
        emotion: Điểm cảm xúc (0-10)
        focus: Điểm tập trung (0-10)
        clarity: Điểm rõ ràng (0-10)
        content: Điểm nội dung (0-10)
        position: Loại vị trí
        
    Returns:
        InterviewScore
    """
    scorer = OverallInterviewScorer(position_type=position)
    return scorer.calculate_score(emotion, focus, clarity, content)
