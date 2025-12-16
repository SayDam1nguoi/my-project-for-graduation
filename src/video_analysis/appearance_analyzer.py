# -*- coding: utf-8 -*-
"""
Appearance Analyzer

Đánh giá ánh sáng và quần áo cho phỏng vấn/tuyển dụng.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class LightingAssessment:
    """Đánh giá ánh sáng."""
    score: float  # 0-100
    brightness: float  # 0-255
    contrast: float  # 0-100
    uniformity: float  # 0-100
    is_good: bool
    issues: list
    recommendation: str


@dataclass
class ClothingAssessment:
    """Đánh giá quần áo."""
    score: float  # 0-100
    is_professional: bool
    dominant_color: str
    color_appropriateness: float  # 0-100
    issues: list
    recommendation: str


@dataclass
class AppearanceAssessment:
    """Đánh giá tổng thể."""
    lighting: LightingAssessment
    clothing: ClothingAssessment
    overall_score: float
    is_interview_ready: bool


class AppearanceAnalyzer:
    """
    Phân tích ánh sáng và quần áo.
    
    Dùng cho:
    - Phỏng vấn online
    - Tuyển dụng
    - Đánh giá chuyên nghiệp
    """
    
    def __init__(self):
        """Khởi tạo analyzer."""
        # Lighting thresholds
        self.min_brightness = 80
        self.max_brightness = 200
        self.min_contrast = 30
        self.min_uniformity = 60
        
        # Professional colors (HSV ranges)
        self.professional_colors = {
            'white': ([0, 0, 200], [180, 30, 255]),
            'black': ([0, 0, 0], [180, 255, 50]),
            'blue': ([100, 50, 50], [130, 255, 255]),
            'gray': ([0, 0, 50], [180, 50, 200]),
            'navy': ([100, 100, 30], [130, 255, 100]),
        }
        
        # Unprofessional colors
        self.unprofessional_colors = {
            'bright_red': ([0, 150, 150], [10, 255, 255]),
            'bright_yellow': ([20, 150, 150], [30, 255, 255]),
            'bright_green': ([40, 150, 150], [80, 255, 255]),
            'pink': ([140, 50, 150], [170, 255, 255]),
        }
    
    def analyze_lighting(self, frame: np.ndarray, face_region: Optional[np.ndarray] = None) -> LightingAssessment:
        """
        Đánh giá ánh sáng.
        
        Args:
            frame: Frame BGR
            face_region: Vùng khuôn mặt (optional)
            
        Returns:
            LightingAssessment
        """
        # Use face region if available, otherwise use full frame
        region = face_region if face_region is not None else frame
        
        # Convert to grayscale
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        
        # Calculate brightness
        brightness = np.mean(gray)
        
        # Calculate contrast
        contrast = np.std(gray)
        
        # Calculate uniformity (inverse of std)
        uniformity = 100 - min(np.std(gray) / 2.55, 100)
        
        # Assess lighting
        issues = []
        
        if brightness < self.min_brightness:
            issues.append("Quá tối")
        elif brightness > self.max_brightness:
            issues.append("Quá sáng")
        
        if contrast < self.min_contrast:
            issues.append("Độ tương phản thấp")
        
        if uniformity < self.min_uniformity:
            issues.append("Ánh sáng không đều")
        
        # Calculate score
        brightness_score = self._score_brightness(brightness)
        contrast_score = self._score_contrast(contrast)
        uniformity_score = uniformity
        
        score = (brightness_score + contrast_score + uniformity_score) / 3
        is_good = score >= 70
        
        # Recommendation
        if not is_good:
            if brightness < self.min_brightness:
                recommendation = "Tăng ánh sáng (mở đèn hoặc ngồi gần cửa sổ)"
            elif brightness > self.max_brightness:
                recommendation = "Giảm ánh sáng (tránh ánh sáng trực tiếp)"
            elif uniformity < self.min_uniformity:
                recommendation = "Cải thiện ánh sáng đều (dùng đèn mềm)"
            else:
                recommendation = "Điều chỉnh ánh sáng"
        else:
            recommendation = "Ánh sáng tốt"
        
        return LightingAssessment(
            score=score,
            brightness=brightness,
            contrast=contrast,
            uniformity=uniformity,
            is_good=is_good,
            issues=issues,
            recommendation=recommendation
        )
    
    def analyze_clothing(self, frame: np.ndarray, upper_body_region: Optional[np.ndarray] = None) -> ClothingAssessment:
        """
        Đánh giá quần áo.
        
        Args:
            frame: Frame BGR
            upper_body_region: Vùng thân trên (optional)
            
        Returns:
            ClothingAssessment
        """
        # Use upper body if available, otherwise use lower half of frame
        if upper_body_region is not None:
            region = upper_body_region
        else:
            h = frame.shape[0]
            region = frame[h//3:2*h//3, :]  # Middle third
        
        # Convert to HSV
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        
        # Find dominant color
        dominant_color, color_name = self._find_dominant_color(hsv)
        
        # Check if professional
        is_professional = self._is_professional_color(hsv)
        
        # Calculate color appropriateness
        color_appropriateness = self._calculate_color_appropriateness(hsv)
        
        # Assess clothing
        issues = []
        
        if not is_professional:
            issues.append("Màu sắc không chuyên nghiệp")
        
        if color_appropriateness < 60:
            issues.append("Màu sắc quá sặc sỡ")
        
        # Calculate score
        professional_score = 100 if is_professional else 50
        score = (professional_score + color_appropriateness) / 2
        
        # Recommendation
        if score < 70:
            recommendation = "Nên mặc áo sơ mi trắng/xanh hoặc vest"
        else:
            recommendation = "Trang phục phù hợp"
        
        return ClothingAssessment(
            score=score,
            is_professional=is_professional,
            dominant_color=color_name,
            color_appropriateness=color_appropriateness,
            issues=issues,
            recommendation=recommendation
        )
    
    def analyze_appearance(self, frame: np.ndarray, face_bbox: Optional[Tuple] = None) -> AppearanceAssessment:
        """
        Đánh giá tổng thể ánh sáng và quần áo.
        
        Args:
            frame: Frame BGR
            face_bbox: Bounding box khuôn mặt (x, y, w, h)
            
        Returns:
            AppearanceAssessment
        """
        # Extract regions
        face_region = None
        upper_body_region = None
        
        if face_bbox is not None:
            x, y, w, h = face_bbox
            # Face region
            face_region = frame[y:y+h, x:x+w]
            
            # Upper body region (below face)
            body_y = y + h
            body_h = min(h * 2, frame.shape[0] - body_y)
            if body_h > 0:
                upper_body_region = frame[body_y:body_y+body_h, x:x+w]
        
        # Analyze lighting
        lighting = self.analyze_lighting(frame, face_region)
        
        # Analyze clothing
        clothing = self.analyze_clothing(frame, upper_body_region)
        
        # Overall score
        overall_score = (lighting.score * 0.4 + clothing.score * 0.6)
        is_interview_ready = overall_score >= 75
        
        return AppearanceAssessment(
            lighting=lighting,
            clothing=clothing,
            overall_score=overall_score,
            is_interview_ready=is_interview_ready
        )
    
    def _score_brightness(self, brightness: float) -> float:
        """Score brightness (0-100)."""
        if brightness < self.min_brightness:
            return (brightness / self.min_brightness) * 100
        elif brightness > self.max_brightness:
            return max(0, 100 - (brightness - self.max_brightness))
        else:
            return 100
    
    def _score_contrast(self, contrast: float) -> float:
        """Score contrast (0-100)."""
        if contrast < self.min_contrast:
            return (contrast / self.min_contrast) * 100
        else:
            return 100
    
    def _find_dominant_color(self, hsv: np.ndarray) -> Tuple[np.ndarray, str]:
        """Tìm màu dominant."""
        # Calculate histogram
        hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        dominant_hue = np.argmax(hist)
        
        # Map to color name
        if dominant_hue < 10 or dominant_hue > 170:
            color_name = "Đỏ"
        elif dominant_hue < 25:
            color_name = "Cam"
        elif dominant_hue < 35:
            color_name = "Vàng"
        elif dominant_hue < 85:
            color_name = "Xanh lá"
        elif dominant_hue < 135:
            color_name = "Xanh dương"
        else:
            color_name = "Tím/Hồng"
        
        # Check for white/black/gray
        v_mean = np.mean(hsv[:, :, 2])
        s_mean = np.mean(hsv[:, :, 1])
        
        if s_mean < 30:
            if v_mean > 200:
                color_name = "Trắng"
            elif v_mean < 50:
                color_name = "Đen"
            else:
                color_name = "Xám"
        
        return dominant_hue, color_name
    
    def _is_professional_color(self, hsv: np.ndarray) -> bool:
        """Kiểm tra màu có chuyên nghiệp không."""
        for color_name, (lower, upper) in self.professional_colors.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            ratio = np.sum(mask > 0) / mask.size
            if ratio > 0.3:  # 30% of region
                return True
        return False
    
    def _calculate_color_appropriateness(self, hsv: np.ndarray) -> float:
        """Tính độ phù hợp của màu sắc."""
        # Check for unprofessional colors
        unprofessional_ratio = 0
        for color_name, (lower, upper) in self.unprofessional_colors.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            ratio = np.sum(mask > 0) / mask.size
            unprofessional_ratio += ratio
        
        # Score (lower unprofessional ratio = higher score)
        score = max(0, 100 - (unprofessional_ratio * 200))
        return score
