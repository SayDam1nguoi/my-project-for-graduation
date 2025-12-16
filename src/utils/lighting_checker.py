"""
Lighting Quality Checker

Kiểm tra chất lượng ánh sáng trong frame để đảm bảo điều kiện tốt
cho emotion detection.
"""

import cv2
import numpy as np
from typing import Tuple, Dict
from enum import Enum


class LightingQuality(Enum):
    """Enum cho chất lượng ánh sáng."""
    EXCELLENT = "excellent"  # Rất tốt
    GOOD = "good"           # Tốt
    FAIR = "fair"           # Chấp nhận được
    POOR = "poor"           # Kém
    VERY_POOR = "very_poor" # Rất kém


class LightingChecker:
    """
    Kiểm tra chất lượng ánh sáng trong frame.
    
    Đánh giá dựa trên:
    - Độ sáng trung bình (brightness)
    - Độ tương phản (contrast)
    - Phân bố histogram
    - Vùng quá sáng/quá tối
    """
    
    def __init__(
        self,
        min_brightness: float = 60.0,
        max_brightness: float = 200.0,
        min_contrast: float = 30.0,
        max_overexposed: float = 0.05,  # 5% pixels
        max_underexposed: float = 0.05   # 5% pixels
    ):
        """
        Initialize LightingChecker.
        
        Args:
            min_brightness: Minimum acceptable average brightness (0-255)
            max_brightness: Maximum acceptable average brightness (0-255)
            min_contrast: Minimum acceptable contrast (std deviation)
            max_overexposed: Maximum ratio of overexposed pixels
            max_underexposed: Maximum ratio of underexposed pixels
        """
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_contrast = min_contrast
        self.max_overexposed = max_overexposed
        self.max_underexposed = max_underexposed
    
    def check_lighting(
        self,
        frame: np.ndarray,
        face_region: Tuple[int, int, int, int] = None
    ) -> Tuple[LightingQuality, Dict]:
        """
        Kiểm tra chất lượng ánh sáng.
        
        Args:
            frame: Input frame (BGR)
            face_region: Optional face region (x, y, w, h) to focus on
            
        Returns:
            Tuple of (quality, details_dict)
        """
        # Chuyển sang grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Nếu có face region, focus vào đó
        if face_region is not None:
            x, y, w, h = face_region
            # Ensure coordinates are within bounds
            h_frame, w_frame = gray.shape
            x = max(0, min(x, w_frame - 1))
            y = max(0, min(y, h_frame - 1))
            w = min(w, w_frame - x)
            h = min(h, h_frame - y)
            gray = gray[y:y+h, x:x+w]
        
        # Tính các metrics
        brightness = self._calculate_brightness(gray)
        contrast = self._calculate_contrast(gray)
        overexposed_ratio = self._calculate_overexposed_ratio(gray)
        underexposed_ratio = self._calculate_underexposed_ratio(gray)
        histogram_score = self._calculate_histogram_score(gray)
        
        # Đánh giá tổng thể
        quality, score = self._evaluate_quality(
            brightness, contrast, overexposed_ratio, 
            underexposed_ratio, histogram_score
        )
        
        # Chi tiết
        details = {
            'quality': quality.value,
            'score': score,
            'brightness': brightness,
            'contrast': contrast,
            'overexposed_ratio': overexposed_ratio,
            'underexposed_ratio': underexposed_ratio,
            'histogram_score': histogram_score,
            'recommendations': self._get_recommendations(
                brightness, contrast, overexposed_ratio, underexposed_ratio
            )
        }
        
        return quality, details
    
    def _calculate_brightness(self, gray: np.ndarray) -> float:
        """Tính độ sáng trung bình."""
        return np.mean(gray)
    
    def _calculate_contrast(self, gray: np.ndarray) -> float:
        """Tính độ tương phản (standard deviation)."""
        return np.std(gray)
    
    def _calculate_overexposed_ratio(self, gray: np.ndarray) -> float:
        """Tính tỷ lệ pixels quá sáng (> 240)."""
        overexposed = np.sum(gray > 240)
        total = gray.size
        return overexposed / total if total > 0 else 0
    
    def _calculate_underexposed_ratio(self, gray: np.ndarray) -> float:
        """Tính tỷ lệ pixels quá tối (< 15)."""
        underexposed = np.sum(gray < 15)
        total = gray.size
        return underexposed / total if total > 0 else 0
    
    def _calculate_histogram_score(self, gray: np.ndarray) -> float:
        """
        Tính điểm histogram (0-100).
        
        Histogram tốt: phân bố đều, không quá tập trung ở một vùng.
        """
        # Tính histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()  # Normalize
        
        # Tính entropy (độ phân tán)
        # Entropy cao = phân bố đều = tốt
        hist_nonzero = hist[hist > 0]
        entropy = -np.sum(hist_nonzero * np.log2(hist_nonzero))
        
        # Normalize entropy to 0-100 scale
        # Max entropy for 256 bins = log2(256) = 8
        score = (entropy / 8.0) * 100
        
        return score
    
    def _evaluate_quality(
        self,
        brightness: float,
        contrast: float,
        overexposed_ratio: float,
        underexposed_ratio: float,
        histogram_score: float
    ) -> Tuple[LightingQuality, float]:
        """
        Đánh giá chất lượng tổng thể.
        
        Returns:
            Tuple of (quality, score)
        """
        score = 100.0
        
        # Kiểm tra brightness
        if brightness < self.min_brightness:
            # Quá tối
            deficit = self.min_brightness - brightness
            score -= min(deficit / 2, 30)  # Penalty up to -30
        elif brightness > self.max_brightness:
            # Quá sáng
            excess = brightness - self.max_brightness
            score -= min(excess / 2, 30)  # Penalty up to -30
        
        # Kiểm tra contrast
        if contrast < self.min_contrast:
            # Contrast thấp
            deficit = self.min_contrast - contrast
            score -= min(deficit, 20)  # Penalty up to -20
        
        # Kiểm tra overexposed
        if overexposed_ratio > self.max_overexposed:
            excess = overexposed_ratio - self.max_overexposed
            score -= min(excess * 200, 15)  # Penalty up to -15
        
        # Kiểm tra underexposed
        if underexposed_ratio > self.max_underexposed:
            excess = underexposed_ratio - self.max_underexposed
            score -= min(excess * 200, 15)  # Penalty up to -15
        
        # Bonus từ histogram score
        if histogram_score > 70:
            score += (histogram_score - 70) / 3  # Bonus up to +10
        
        # Clamp score
        score = max(0, min(100, score))
        
        # Xác định quality level
        if score >= 85:
            quality = LightingQuality.EXCELLENT
        elif score >= 70:
            quality = LightingQuality.GOOD
        elif score >= 50:
            quality = LightingQuality.FAIR
        elif score >= 30:
            quality = LightingQuality.POOR
        else:
            quality = LightingQuality.VERY_POOR
        
        return quality, score
    
    def _get_recommendations(
        self,
        brightness: float,
        contrast: float,
        overexposed_ratio: float,
        underexposed_ratio: float
    ) -> list:
        """Đưa ra khuyến nghị cải thiện."""
        recommendations = []
        
        if brightness < self.min_brightness:
            recommendations.append("Tăng ánh sáng (quá tối)")
        elif brightness > self.max_brightness:
            recommendations.append("Giảm ánh sáng (quá sáng)")
        
        if contrast < self.min_contrast:
            recommendations.append("Cải thiện độ tương phản")
        
        if overexposed_ratio > self.max_overexposed:
            recommendations.append("Giảm vùng quá sáng")
        
        if underexposed_ratio > self.max_underexposed:
            recommendations.append("Giảm vùng quá tối")
        
        if not recommendations:
            recommendations.append("Ánh sáng tốt!")
        
        return recommendations
    
    def get_quality_color(self, quality: LightingQuality) -> Tuple[int, int, int]:
        """
        Lấy màu hiển thị cho quality level.
        
        Returns:
            BGR color tuple
        """
        colors = {
            LightingQuality.EXCELLENT: (0, 255, 0),      # Green
            LightingQuality.GOOD: (0, 200, 0),           # Light green
            LightingQuality.FAIR: (0, 255, 255),         # Yellow
            LightingQuality.POOR: (0, 165, 255),         # Orange
            LightingQuality.VERY_POOR: (0, 0, 255),      # Red
        }
        return colors.get(quality, (128, 128, 128))
    
    def get_quality_text(self, quality: LightingQuality) -> str:
        """Lấy text hiển thị cho quality level."""
        texts = {
            LightingQuality.EXCELLENT: "Rat Tot",
            LightingQuality.GOOD: "Tot",
            LightingQuality.FAIR: "Chap Nhan",
            LightingQuality.POOR: "Kem",
            LightingQuality.VERY_POOR: "Rat Kem",
        }
        return texts.get(quality, "Unknown")
