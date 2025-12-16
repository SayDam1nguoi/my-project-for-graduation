# -*- coding: utf-8 -*-
"""
Lighting Assessor

Đánh giá chất lượng ánh sáng trong video phỏng vấn.
"""

import cv2
import numpy as np
import time
from typing import List, Tuple
from .models import LightingAssessment


class LightingAssessor:
    """
    Đánh giá chất lượng ánh sáng.
    
    Phân tích:
    - Độ sáng (brightness)
    - Độ tương phản (contrast)
    - Độ đều (uniformity)
    """
    
    def __init__(
        self,
        min_brightness: float = 80.0,
        max_brightness: float = 200.0,
        min_contrast: float = 30.0,
        min_uniformity: float = 60.0
    ):
        """
        Khởi tạo LightingAssessor.
        
        Args:
            min_brightness: Ngưỡng độ sáng tối thiểu (0-255)
            max_brightness: Ngưỡng độ sáng tối đa (0-255)
            min_contrast: Ngưỡng độ tương phản tối thiểu (0-100)
            min_uniformity: Ngưỡng độ đều tối thiểu (0-100)
        """
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_contrast = min_contrast
        self.min_uniformity = min_uniformity
    
    def assess(
        self,
        frame: np.ndarray,
        face_region: np.ndarray
    ) -> LightingAssessment:
        """
        Đánh giá ánh sáng.
        
        Args:
            frame: Frame BGR gốc
            face_region: Vùng khuôn mặt đã được trích xuất
            
        Returns:
            LightingAssessment với đầy đủ thông tin đánh giá
        """
        # Calculate metrics
        brightness = self.calculate_brightness(face_region)
        contrast = self.calculate_contrast(face_region)
        uniformity = self.calculate_uniformity(face_region)
        
        # Detect issues
        issues = self._detect_issues(brightness, contrast, uniformity)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(issues, brightness, contrast, uniformity)
        
        # Calculate overall score
        brightness_score = self._score_brightness(brightness)
        contrast_score = self._score_contrast(contrast)
        uniformity_score = uniformity
        
        overall_score = (brightness_score + contrast_score + uniformity_score) / 3.0
        
        # Normalize to 0-100 range
        overall_score = max(0.0, min(100.0, overall_score))
        
        # Determine if lighting is good
        is_good = overall_score >= 70.0
        
        return LightingAssessment(
            score=overall_score,
            brightness=brightness,
            contrast=contrast,
            uniformity=uniformity,
            is_good=is_good,
            issues=issues,
            recommendations=recommendations,
            timestamp=time.time()
        )
    
    def calculate_brightness(self, region: np.ndarray) -> float:
        """
        Tính độ sáng trung bình.
        
        Args:
            region: Vùng ảnh BGR
            
        Returns:
            Độ sáng (0-255)
        """
        # Convert to grayscale
        if len(region.shape) == 3:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        else:
            gray = region
        
        # Calculate mean brightness
        brightness = float(np.mean(gray))
        
        return brightness
    
    def calculate_contrast(self, region: np.ndarray) -> float:
        """
        Tính độ tương phản.
        
        Args:
            region: Vùng ảnh BGR
            
        Returns:
            Độ tương phản (0-100)
        """
        # Convert to grayscale
        if len(region.shape) == 3:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        else:
            gray = region
        
        # Calculate standard deviation as contrast measure
        std = float(np.std(gray))
        
        # Normalize to 0-100 scale (std typically ranges 0-127.5)
        contrast = min(100.0, (std / 127.5) * 100.0)
        
        return contrast
    
    def calculate_uniformity(self, region: np.ndarray) -> float:
        """
        Tính độ đều của ánh sáng.
        
        Args:
            region: Vùng ảnh BGR
            
        Returns:
            Độ đều (0-100), cao hơn = đều hơn
        """
        # Convert to grayscale
        if len(region.shape) == 3:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        else:
            gray = region
        
        # Calculate standard deviation
        std = float(np.std(gray))
        
        # Uniformity is inverse of std
        # Lower std = more uniform = higher score
        # Normalize: std of 0 = 100%, std of 127.5 = 0%
        uniformity = max(0.0, 100.0 - (std / 127.5) * 100.0)
        
        return uniformity
    
    def _detect_issues(
        self,
        brightness: float,
        contrast: float,
        uniformity: float
    ) -> List[str]:
        """
        Phát hiện các vấn đề về ánh sáng.
        
        Args:
            brightness: Độ sáng
            contrast: Độ tương phản
            uniformity: Độ đều
            
        Returns:
            Danh sách các vấn đề
        """
        issues = []
        
        if brightness < self.min_brightness:
            issues.append("Quá tối")
        elif brightness > self.max_brightness:
            issues.append("Quá sáng")
        
        if contrast < self.min_contrast:
            issues.append("Độ tương phản thấp")
        
        if uniformity < self.min_uniformity:
            issues.append("Ánh sáng không đều")
        
        return issues
    
    def _generate_recommendations(
        self,
        issues: List[str],
        brightness: float,
        contrast: float,
        uniformity: float
    ) -> List[str]:
        """
        Tạo khuyến nghị dựa trên các vấn đề phát hiện.
        
        Args:
            issues: Danh sách vấn đề
            brightness: Độ sáng
            contrast: Độ tương phản
            uniformity: Độ đều
            
        Returns:
            Danh sách khuyến nghị
        """
        recommendations = []
        
        if "Quá tối" in issues:
            recommendations.append("Tăng ánh sáng: mở đèn hoặc ngồi gần cửa sổ")
        
        if "Quá sáng" in issues:
            recommendations.append("Giảm ánh sáng: tránh ánh sáng trực tiếp từ cửa sổ hoặc đèn")
        
        if "Độ tương phản thấp" in issues:
            recommendations.append("Cải thiện độ tương phản: điều chỉnh nguồn sáng hoặc vị trí camera")
        
        if "Ánh sáng không đều" in issues:
            recommendations.append("Cải thiện độ đều: sử dụng đèn mềm hoặc nhiều nguồn sáng")
        
        # If no issues, provide positive feedback
        if not recommendations:
            recommendations.append("Ánh sáng tốt, không cần điều chỉnh")
        
        return recommendations
    
    def _score_brightness(self, brightness: float) -> float:
        """
        Tính điểm cho độ sáng.
        
        Args:
            brightness: Độ sáng (0-255)
            
        Returns:
            Điểm (0-100)
        """
        if brightness < self.min_brightness:
            # Too dark: linear scale from 0 to 100
            score = (brightness / self.min_brightness) * 100.0
        elif brightness > self.max_brightness:
            # Too bright: linear decrease
            excess = brightness - self.max_brightness
            score = max(0.0, 100.0 - excess)
        else:
            # In optimal range
            score = 100.0
        
        return max(0.0, min(100.0, score))
    
    def _score_contrast(self, contrast: float) -> float:
        """
        Tính điểm cho độ tương phản.
        
        Args:
            contrast: Độ tương phản (0-100)
            
        Returns:
            Điểm (0-100)
        """
        if contrast < self.min_contrast:
            # Low contrast: linear scale
            score = (contrast / self.min_contrast) * 100.0
        else:
            # Adequate contrast
            score = 100.0
        
        return max(0.0, min(100.0, score))
