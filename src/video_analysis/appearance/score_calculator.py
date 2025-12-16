# -*- coding: utf-8 -*-
"""
Appearance Score Calculator

Calculates overall appearance scores from lighting and clothing assessments.
"""

from typing import Optional


class AppearanceScoreCalculator:
    """
    Tính toán điểm tổng hợp từ các đánh giá riêng lẻ.
    
    Weighted average:
    - Lighting: 40%
    - Clothing: 60%
    
    Interview-ready threshold: 75.0
    """
    
    LIGHTING_WEIGHT = 0.4
    CLOTHING_WEIGHT = 0.6
    INTERVIEW_READY_THRESHOLD = 75.0
    
    def calculate_overall_score(
        self,
        lighting_score: Optional[float],
        clothing_score: Optional[float]
    ) -> float:
        """
        Tính điểm tổng hợp từ lighting và clothing scores.
        
        Logic:
        - Cả hai enabled: weighted average (40% lighting, 60% clothing)
        - Chỉ lighting: lighting_score
        - Chỉ clothing: clothing_score
        - Không có gì: 0.0
        
        Args:
            lighting_score: Điểm ánh sáng (0-100) hoặc None nếu disabled
            clothing_score: Điểm quần áo (0-100) hoặc None nếu disabled
            
        Returns:
            Điểm tổng hợp (0-100)
        """
        # Normalize scores to ensure they're in [0, 100] range
        if lighting_score is not None:
            lighting_score = self._normalize_score(lighting_score)
        if clothing_score is not None:
            clothing_score = self._normalize_score(clothing_score)
        
        # Both enabled: weighted average
        if lighting_score is not None and clothing_score is not None:
            overall = (lighting_score * self.LIGHTING_WEIGHT + 
                      clothing_score * self.CLOTHING_WEIGHT)
            return self._normalize_score(overall)
        
        # Only lighting enabled
        if lighting_score is not None:
            return lighting_score
        
        # Only clothing enabled
        if clothing_score is not None:
            return clothing_score
        
        # Neither enabled
        return 0.0
    
    def is_interview_ready(self, overall_score: float) -> bool:
        """
        Kiểm tra xem ứng viên có sẵn sàng phỏng vấn không.
        
        Args:
            overall_score: Điểm tổng hợp (0-100)
            
        Returns:
            True nếu overall_score >= 75.0
        """
        return overall_score >= self.INTERVIEW_READY_THRESHOLD
    
    def _normalize_score(self, score: float) -> float:
        """
        Normalize score to [0, 100] range.
        
        Args:
            score: Score to normalize
            
        Returns:
            Normalized score in [0, 100]
        """
        return max(0.0, min(100.0, score))
