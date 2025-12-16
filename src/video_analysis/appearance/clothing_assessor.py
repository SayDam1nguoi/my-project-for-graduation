# -*- coding: utf-8 -*-
"""
Clothing Assessor

Đánh giá trang phục chuyên nghiệp trong video phỏng vấn.
"""

import cv2
import numpy as np
import time
from typing import List, Tuple, Dict
from .models import ClothingAssessment


class ClothingAssessor:
    """
    Đánh giá trang phục chuyên nghiệp.
    
    Phân tích:
    - Màu sắc dominant
    - Tỷ lệ màu chuyên nghiệp
    - Độ phù hợp của màu sắc
    """
    
    # Professional colors in HSV ranges
    # Format: (lower_bound, upper_bound) for H, S, V
    PROFESSIONAL_COLORS = {
        'white': ([0, 0, 200], [180, 30, 255]),
        'black': ([0, 0, 0], [180, 255, 50]),
        'blue': ([100, 50, 50], [130, 255, 255]),
        'gray': ([0, 0, 50], [180, 30, 200]),
        'navy': ([100, 50, 30], [130, 255, 100])
    }
    
    # Unprofessional colors in HSV ranges
    UNPROFESSIONAL_COLORS = {
        'bright_red': ([0, 150, 150], [10, 255, 255]),
        'bright_red_wrap': ([170, 150, 150], [180, 255, 255]),  # Red wraps around in HSV
        'bright_yellow': ([20, 150, 150], [30, 255, 255]),
        'bright_green': ([40, 150, 150], [80, 255, 255]),
        'pink': ([140, 50, 150], [170, 255, 255])
    }
    
    def __init__(self):
        """Khởi tạo ClothingAssessor."""
        pass
    
    def assess(
        self,
        frame: np.ndarray,
        upper_body_region: np.ndarray,
        face_bbox: Tuple[int, int, int, int] = None
    ) -> ClothingAssessment:
        """
        Đánh giá quần áo với phân tích chi tiết.
        
        Args:
            frame: Frame BGR gốc
            upper_body_region: Vùng thân trên đã được trích xuất
            face_bbox: Bounding box của khuôn mặt (x, y, w, h) - optional
            
        Returns:
            ClothingAssessment với đầy đủ thông tin đánh giá chi tiết
        """
        # Detect dominant color
        dominant_color, dominant_hsv = self.detect_dominant_color(upper_body_region)
        
        # Calculate professional color ratio (legacy)
        professional_ratio = self.calculate_professional_ratio(upper_body_region)
        
        # Calculate color appropriateness (legacy)
        color_appropriateness = self._calculate_color_appropriateness(
            dominant_color, dominant_hsv, upper_body_region
        )
        
        # NEW: Detect clothing type
        clothing_type = self.detect_clothing_type(upper_body_region)
        
        # NEW: Assess neatness (wrinkles, stains)
        neatness_score, neatness_issues = self.assess_neatness(upper_body_region)
        
        # NEW: Detect pattern
        pattern_type = self.detect_pattern(upper_body_region)
        
        # NEW: Detect accessories
        accessories = []
        if face_bbox is not None:
            accessories = self.detect_accessories(frame, face_bbox)
        
        # Calculate sub-scores
        color_score = color_appropriateness  # Use existing color appropriateness
        type_score = self._calculate_type_score(clothing_type)
        pattern_score = self._calculate_pattern_score(pattern_type)
        accessories_score = self._calculate_accessories_score(accessories, clothing_type)
        
        # Calculate overall score with new weighted formula
        # Color: 25%, Type: 20%, Neatness: 25%, Pattern: 15%, Accessories: 15%
        overall_score = self.calculate_detailed_score(
            color_score, type_score, neatness_score, pattern_score, accessories_score
        )
        
        # Normalize to 0-100 range
        overall_score = max(0.0, min(100.0, overall_score))
        
        # Detect issues (enhanced with new criteria)
        issues = self._detect_issues_enhanced(
            dominant_color, professional_ratio, color_appropriateness,
            clothing_type, neatness_score, neatness_issues, pattern_type, accessories
        )
        
        # Generate recommendations (enhanced)
        recommendations = self._generate_recommendations_enhanced(
            issues, dominant_color, clothing_type, neatness_score, pattern_type, accessories
        )
        
        # Determine if clothing is professional
        is_professional = overall_score >= 70.0
        
        return ClothingAssessment(
            score=overall_score,
            dominant_color=dominant_color,
            clothing_type=clothing_type,
            neatness_score=neatness_score,
            pattern_type=pattern_type,
            accessories=accessories,
            color_score=color_score,
            type_score=type_score,
            pattern_score=pattern_score,
            accessories_score=accessories_score,
            professional_color_ratio=professional_ratio,
            color_appropriateness=color_appropriateness,
            is_professional=is_professional,
            issues=issues,
            recommendations=recommendations,
            timestamp=time.time()
        )
    
    def detect_dominant_color(self, region: np.ndarray) -> Tuple[str, np.ndarray]:
        """
        Phát hiện màu dominant.
        
        Args:
            region: Vùng ảnh BGR
            
        Returns:
            Tuple of (color_name, hsv_value)
        """
        # Convert to HSV
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        
        # Calculate histogram for each channel
        h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [256], [0, 256])
        
        # Find peaks
        h_peak = np.argmax(h_hist)
        s_peak = np.argmax(s_hist)
        v_peak = np.argmax(v_hist)
        
        dominant_hsv = np.array([h_peak, s_peak, v_peak])
        
        # Classify the dominant color
        color_name = self._classify_color(dominant_hsv)
        
        return color_name, dominant_hsv
    
    def calculate_professional_ratio(self, region: np.ndarray) -> float:
        """
        Tính tỷ lệ màu chuyên nghiệp.
        
        Args:
            region: Vùng ảnh BGR
            
        Returns:
            Tỷ lệ màu chuyên nghiệp (0-100)
        """
        # Convert to HSV
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        
        # Count pixels matching professional colors
        professional_pixels = 0
        total_pixels = hsv.shape[0] * hsv.shape[1]
        
        for color_name, (lower, upper) in self.PROFESSIONAL_COLORS.items():
            lower_bound = np.array(lower, dtype=np.uint8)
            upper_bound = np.array(upper, dtype=np.uint8)
            
            mask = cv2.inRange(hsv, lower_bound, upper_bound)
            professional_pixels += np.count_nonzero(mask)
        
        # Calculate ratio
        ratio = (professional_pixels / total_pixels) * 100.0
        
        return min(100.0, ratio)
    
    def _classify_color(self, hsv_value: np.ndarray) -> str:
        """
        Phân loại màu dựa trên giá trị HSV.
        
        Args:
            hsv_value: Giá trị HSV [H, S, V]
            
        Returns:
            Tên màu
        """
        h, s, v = hsv_value
        
        # Check professional colors first
        for color_name, (lower, upper) in self.PROFESSIONAL_COLORS.items():
            if (lower[0] <= h <= upper[0] and
                lower[1] <= s <= upper[1] and
                lower[2] <= v <= upper[2]):
                return color_name
        
        # Check unprofessional colors
        for color_name, (lower, upper) in self.UNPROFESSIONAL_COLORS.items():
            if (lower[0] <= h <= upper[0] and
                lower[1] <= s <= upper[1] and
                lower[2] <= v <= upper[2]):
                return color_name
        
        # Default classification based on hue
        if v < 50:
            return 'dark'
        elif s < 30:
            if v > 200:
                return 'white'
            else:
                return 'gray'
        elif 0 <= h < 15 or 165 <= h <= 180:
            return 'red'
        elif 15 <= h < 30:
            return 'orange'
        elif 30 <= h < 80:
            return 'yellow_green'
        elif 80 <= h < 140:
            return 'blue_green'
        elif 140 <= h < 165:
            return 'purple'
        else:
            return 'unknown'
    
    def _calculate_color_appropriateness(
        self,
        dominant_color: str,
        dominant_hsv: np.ndarray,
        region: np.ndarray
    ) -> float:
        """
        Tính độ phù hợp của màu sắc.
        
        Args:
            dominant_color: Tên màu dominant
            dominant_hsv: Giá trị HSV của màu dominant
            region: Vùng ảnh BGR
            
        Returns:
            Điểm độ phù hợp (0-100)
        """
        # Check if dominant color is professional
        is_professional = dominant_color in self.PROFESSIONAL_COLORS
        
        # Check if dominant color is unprofessional
        is_unprofessional = dominant_color in self.UNPROFESSIONAL_COLORS
        
        if is_professional:
            # Professional colors get high scores
            return 100.0
        elif is_unprofessional:
            # Unprofessional colors get low scores
            return 20.0
        else:
            # Neutral colors get medium scores
            # Score based on saturation and value
            h, s, v = dominant_hsv
            
            # Lower saturation = more professional (less flashy)
            saturation_score = max(0.0, 100.0 - (s / 255.0) * 100.0)
            
            # Moderate value is best
            if 80 <= v <= 180:
                value_score = 100.0
            else:
                value_score = 50.0
            
            # Average the scores
            return (saturation_score + value_score) / 2.0
    
    def _detect_issues(
        self,
        dominant_color: str,
        professional_ratio: float,
        color_appropriateness: float
    ) -> List[str]:
        """
        Phát hiện các vấn đề về trang phục.
        
        Args:
            dominant_color: Màu dominant
            professional_ratio: Tỷ lệ màu chuyên nghiệp
            color_appropriateness: Độ phù hợp màu sắc
            
        Returns:
            Danh sách các vấn đề
        """
        issues = []
        
        # Check if dominant color is unprofessional
        if dominant_color in self.UNPROFESSIONAL_COLORS:
            issues.append(f"Màu {dominant_color} không phù hợp cho phỏng vấn")
        
        # Check professional ratio
        if professional_ratio < 30.0:
            issues.append("Tỷ lệ màu chuyên nghiệp thấp")
        
        # Check color appropriateness
        if color_appropriateness < 50.0:
            issues.append("Màu sắc không phù hợp")
        
        return issues
    
    def _generate_recommendations(
        self,
        issues: List[str],
        dominant_color: str
    ) -> List[str]:
        """
        Tạo khuyến nghị dựa trên các vấn đề phát hiện.
        
        Args:
            issues: Danh sách vấn đề
            dominant_color: Màu dominant
            
        Returns:
            Danh sách khuyến nghị
        """
        recommendations = []
        
        if any("không phù hợp" in issue for issue in issues):
            recommendations.append(
                "Nên mặc áo sơ mi trắng, xanh navy, hoặc màu trung tính"
            )
        
        if "Tỷ lệ màu chuyên nghiệp thấp" in issues:
            recommendations.append(
                "Tránh quần áo có màu sắc quá sặc sỡ hoặc hoa văn phức tạp"
            )
        
        if "Màu sắc không phù hợp" in issues:
            recommendations.append(
                "Chọn màu sắc trung tính và chuyên nghiệp (đen, trắng, xanh, xám)"
            )
        
        # If no issues, provide positive feedback
        if not recommendations:
            recommendations.append("Trang phục phù hợp, không cần điều chỉnh")
        
        return recommendations
    
    # ========== NEW METHODS FOR ENHANCED ASSESSMENT ==========
    
    def detect_clothing_type(self, region: np.ndarray) -> str:
        """
        Phát hiện loại trang phục.
        
        Args:
            region: Vùng ảnh BGR của thân trên
            
        Returns:
            Loại trang phục: formal_shirt, casual_shirt, t_shirt, suit_jacket, casual_jacket
        """
        # Convert to HSV for analysis
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        
        # Analyze texture complexity using edge detection
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.count_nonzero(edges) / edges.size
        
        # Analyze color uniformity
        h_std = np.std(hsv[:, :, 0])
        s_std = np.std(hsv[:, :, 1])
        v_std = np.std(hsv[:, :, 2])
        color_variance = (h_std + s_std + v_std) / 3.0
        
        # Detect dark colors (potential suit jacket)
        v_mean = np.mean(hsv[:, :, 2])
        
        # Classification logic
        if v_mean < 80 and edge_density > 0.15:
            # Dark with structure -> likely suit jacket
            return 'suit_jacket'
        elif edge_density > 0.20:
            # High edge density -> structured jacket
            return 'casual_jacket'
        elif color_variance < 20 and edge_density < 0.10:
            # Low variance, low edges -> plain t-shirt
            return 't_shirt'
        elif edge_density > 0.12:
            # Medium structure -> formal shirt (collar, buttons)
            return 'formal_shirt'
        else:
            # Default to casual shirt
            return 'casual_shirt'
    
    def assess_neatness(self, region: np.ndarray) -> Tuple[float, List[str]]:
        """
        Đánh giá độ gọn gàng (phát hiện nhăn và vết bẩn).
        
        Args:
            region: Vùng ảnh BGR
            
        Returns:
            Tuple of (neatness_score, issues_list)
        """
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        issues = []
        
        # Wrinkle detection using texture analysis
        # Use Laplacian variance to detect texture irregularities
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_var = laplacian.var()
        
        # High variance indicates wrinkles/texture irregularities
        wrinkle_score = 100.0
        if laplacian_var > 500:
            wrinkle_penalty = min(50.0, (laplacian_var - 500) / 20.0)
            wrinkle_score -= wrinkle_penalty
            if wrinkle_penalty > 20:
                issues.append("Phát hiện nhăn nhúm")
        
        # Stain detection using color anomaly detection
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        
        # Detect dark spots (potential stains)
        v_mean = np.mean(v_channel)
        v_std = np.std(v_channel)
        
        # Find pixels significantly darker than mean
        dark_threshold = v_mean - 2 * v_std
        dark_pixels = np.sum(v_channel < dark_threshold)
        dark_ratio = dark_pixels / v_channel.size
        
        stain_score = 100.0
        if dark_ratio > 0.05:  # More than 5% dark spots
            stain_penalty = min(40.0, dark_ratio * 500)
            stain_score -= stain_penalty
            if stain_penalty > 15:
                issues.append("Phát hiện vết bẩn hoặc đốm")
        
        # Combined neatness score
        neatness_score = (wrinkle_score + stain_score) / 2.0
        neatness_score = max(0.0, min(100.0, neatness_score))
        
        return neatness_score, issues
    
    def detect_pattern(self, region: np.ndarray) -> str:
        """
        Phát hiện pattern/texture của quần áo.
        
        Args:
            region: Vùng ảnh BGR
            
        Returns:
            Pattern type: solid, striped, checkered, complex
        """
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        
        # Analyze frequency domain for pattern detection
        # Use FFT to detect periodic patterns
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        # Analyze the magnitude spectrum
        # High frequency components indicate patterns
        h, w = magnitude_spectrum.shape
        center_h, center_w = h // 2, w // 2
        
        # Exclude DC component (center)
        magnitude_spectrum[center_h-5:center_h+5, center_w-5:center_w+5] = 0
        
        # Calculate energy in different frequency bands
        high_freq_energy = np.sum(magnitude_spectrum[0:h//4, :]) + np.sum(magnitude_spectrum[3*h//4:, :])
        total_energy = np.sum(magnitude_spectrum)
        
        high_freq_ratio = high_freq_energy / (total_energy + 1e-6)
        
        # Also check color variance for pattern detection
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        h_std = np.std(hsv[:, :, 0])
        
        # Classification
        if high_freq_ratio < 0.05 and h_std < 15:
            return 'solid'
        elif high_freq_ratio < 0.15:
            # Check for stripes using horizontal/vertical analysis
            h_variance = np.var(np.mean(gray, axis=0))
            v_variance = np.var(np.mean(gray, axis=1))
            if max(h_variance, v_variance) > 100:
                return 'striped'
            else:
                return 'solid'
        elif high_freq_ratio < 0.25:
            return 'checkered'
        else:
            return 'complex'
    
    def detect_accessories(self, frame: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> List[str]:
        """
        Phát hiện phụ kiện chuyên nghiệp (cà vạt, kính).
        
        Args:
            frame: Frame BGR gốc
            face_bbox: Bounding box của khuôn mặt (x, y, w, h)
            
        Returns:
            Danh sách phụ kiện phát hiện được
        """
        accessories = []
        x, y, w, h = face_bbox
        
        # Detect tie: look in region below face
        tie_region_y = y + h
        tie_region_h = int(h * 1.5)
        tie_region_x = x + int(w * 0.3)
        tie_region_w = int(w * 0.4)
        
        # Ensure region is within frame bounds
        if tie_region_y + tie_region_h < frame.shape[0]:
            tie_region = frame[tie_region_y:tie_region_y+tie_region_h, 
                              tie_region_x:tie_region_x+tie_region_w]
            
            if self._detect_tie(tie_region):
                accessories.append('tie')
        
        # Detect glasses: look in face region (eye area)
        eye_region_y = y + int(h * 0.2)
        eye_region_h = int(h * 0.4)
        eye_region = frame[eye_region_y:eye_region_y+eye_region_h, x:x+w]
        
        if self._detect_glasses(eye_region):
            accessories.append('glasses')
        
        return accessories
    
    def _detect_tie(self, region: np.ndarray) -> bool:
        """
        Phát hiện cà vạt trong vùng ảnh.
        
        Args:
            region: Vùng ảnh BGR
            
        Returns:
            True nếu phát hiện cà vạt
        """
        if region.size == 0:
            return False
        
        # Ties typically have:
        # 1. Vertical elongated shape
        # 2. Distinct color from shirt
        # 3. Located in center of chest
        
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Look for vertical lines (tie edges)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=20, maxLineGap=10)
        
        if lines is not None:
            vertical_lines = 0
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                # Check if line is roughly vertical (70-110 degrees)
                if 70 <= angle <= 110:
                    vertical_lines += 1
            
            # If we have multiple vertical lines, likely a tie
            if vertical_lines >= 2:
                return True
        
        return False
    
    def _detect_glasses(self, region: np.ndarray) -> bool:
        """
        Phát hiện kính trong vùng mắt.
        
        Args:
            region: Vùng ảnh BGR của mắt
            
        Returns:
            True nếu phát hiện kính
        """
        if region.size == 0:
            return False
        
        # Glasses typically have:
        # 1. Horizontal lines (frame top/bottom)
        # 2. Reflections (bright spots)
        # 3. Distinct edges
        
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 100)
        
        # Look for horizontal lines (glasses frame)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, minLineLength=15, maxLineGap=5)
        
        if lines is not None:
            horizontal_lines = 0
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                # Check if line is roughly horizontal (0-20 or 160-180 degrees)
                if angle < 20 or angle > 160:
                    horizontal_lines += 1
            
            # If we have horizontal lines, likely glasses
            if horizontal_lines >= 2:
                return True
        
        # Also check for reflections (bright spots)
        _, bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        bright_pixels = np.count_nonzero(bright_mask)
        bright_ratio = bright_pixels / gray.size
        
        # Glasses often have reflections
        if bright_ratio > 0.05:
            return True
        
        return False
    
    def calculate_detailed_score(
        self,
        color_score: float,
        type_score: float,
        neatness_score: float,
        pattern_score: float,
        accessories_score: float
    ) -> float:
        """
        Tính điểm tổng hợp với trọng số chi tiết.
        
        Trọng số:
        - Color: 25%
        - Type: 20%
        - Neatness: 25%
        - Pattern: 15%
        - Accessories: 15%
        
        Args:
            color_score: Điểm màu sắc (0-100)
            type_score: Điểm loại trang phục (0-100)
            neatness_score: Điểm độ gọn gàng (0-100)
            pattern_score: Điểm pattern (0-100)
            accessories_score: Điểm phụ kiện (0-100)
            
        Returns:
            Điểm tổng hợp (0-100)
        """
        overall = (
            color_score * 0.25 +
            type_score * 0.20 +
            neatness_score * 0.25 +
            pattern_score * 0.15 +
            accessories_score * 0.15
        )
        return max(0.0, min(100.0, overall))
    
    def _calculate_type_score(self, clothing_type: str) -> float:
        """
        Tính điểm dựa trên loại trang phục.
        
        Args:
            clothing_type: Loại trang phục
            
        Returns:
            Điểm (0-100)
        """
        type_scores = {
            'suit_jacket': 100.0,
            'formal_shirt': 90.0,
            'casual_shirt': 70.0,
            'casual_jacket': 60.0,
            't_shirt': 40.0
        }
        return type_scores.get(clothing_type, 50.0)
    
    def _calculate_pattern_score(self, pattern_type: str) -> float:
        """
        Tính điểm dựa trên pattern.
        
        Args:
            pattern_type: Loại pattern
            
        Returns:
            Điểm (0-100)
        """
        pattern_scores = {
            'solid': 100.0,
            'striped': 85.0,
            'checkered': 70.0,
            'complex': 40.0
        }
        return pattern_scores.get(pattern_type, 50.0)
    
    def _calculate_accessories_score(self, accessories: List[str], clothing_type: str) -> float:
        """
        Tính điểm dựa trên phụ kiện.
        
        Args:
            accessories: Danh sách phụ kiện
            clothing_type: Loại trang phục
            
        Returns:
            Điểm (0-100)
        """
        base_score = 70.0  # Base score without accessories
        
        # Bonus for tie with formal attire
        if 'tie' in accessories:
            if clothing_type in ['formal_shirt', 'suit_jacket']:
                base_score += 30.0  # Significant bonus for tie with formal wear
            else:
                base_score += 15.0  # Smaller bonus for tie with casual wear
        
        # Small bonus for glasses (professional look)
        if 'glasses' in accessories:
            base_score += 5.0
        
        return min(100.0, base_score)
    
    def _detect_issues_enhanced(
        self,
        dominant_color: str,
        professional_ratio: float,
        color_appropriateness: float,
        clothing_type: str,
        neatness_score: float,
        neatness_issues: List[str],
        pattern_type: str,
        accessories: List[str]
    ) -> List[str]:
        """
        Phát hiện các vấn đề về trang phục (phiên bản nâng cao).
        
        Returns:
            Danh sách các vấn đề
        """
        issues = []
        
        # Color issues
        if dominant_color in self.UNPROFESSIONAL_COLORS:
            issues.append(f"Màu {dominant_color} không phù hợp cho phỏng vấn")
        if professional_ratio < 30.0:
            issues.append("Tỷ lệ màu chuyên nghiệp thấp")
        if color_appropriateness < 50.0:
            issues.append("Màu sắc không phù hợp")
        
        # Type issues
        if clothing_type == 't_shirt':
            issues.append("Áo thun không phù hợp cho phỏng vấn chính thức")
        elif clothing_type == 'casual_shirt':
            issues.append("Nên mặc áo sơ mi chính thức")
        
        # Neatness issues
        issues.extend(neatness_issues)
        if neatness_score < 60:
            issues.append("Trang phục cần gọn gàng hơn")
        
        # Pattern issues
        if pattern_type == 'complex':
            issues.append("Họa tiết quá phức tạp, nên chọn họa tiết đơn giản")
        
        # Accessories issues
        if clothing_type in ['formal_shirt', 'suit_jacket'] and 'tie' not in accessories:
            issues.append("Nên đeo cà vạt với trang phục chính thức")
        
        return issues
    
    def _generate_recommendations_enhanced(
        self,
        issues: List[str],
        dominant_color: str,
        clothing_type: str,
        neatness_score: float,
        pattern_type: str,
        accessories: List[str]
    ) -> List[str]:
        """
        Tạo khuyến nghị dựa trên các vấn đề phát hiện (phiên bản nâng cao).
        
        Returns:
            Danh sách khuyến nghị
        """
        recommendations = []
        
        # Color recommendations
        if any("không phù hợp" in issue for issue in issues):
            recommendations.append(
                "Nên mặc áo sơ mi trắng, xanh navy, hoặc màu trung tính"
            )
        
        # Type recommendations
        if clothing_type in ['t_shirt', 'casual_shirt']:
            recommendations.append(
                "Nên mặc áo sơ mi chính thức hoặc vest cho phỏng vấn"
            )
        
        # Neatness recommendations
        if neatness_score < 60:
            recommendations.append(
                "Là ủi quần áo trước khi phỏng vấn, đảm bảo sạch sẽ"
            )
        
        # Pattern recommendations
        if pattern_type in ['checkered', 'complex']:
            recommendations.append(
                "Nên chọn áo trơn hoặc họa tiết sọc đơn giản"
            )
        
        # Accessories recommendations
        if clothing_type in ['formal_shirt', 'suit_jacket'] and 'tie' not in accessories:
            recommendations.append(
                "Nên đeo cà vạt để hoàn thiện trang phục chính thức"
            )
        
        # If no issues, provide positive feedback
        if not recommendations:
            recommendations.append("Trang phục chuyên nghiệp, phù hợp cho phỏng vấn")
        
        return recommendations
