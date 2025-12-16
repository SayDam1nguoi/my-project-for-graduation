"""
Attention Detector Module

Phát hiện mức độ tập trung theo hệ thống mới (THANG ĐIỂM 0-10):

CÔNG THỨC CHÍNH:
FocusScore = (AvgScore + MinScore) / 2

Trong đó:
- AvgScore: Điểm trung bình (0-10)
- MinScore: Điểm thấp nhất (0-10) - phát hiện mất tập trung nghiêm trọng

ĐIỂM TỨC THỜI (InstantScore):
InstantScore = FTS × 40% + EGA × 40% + MS × 20%

- FTS (Face Tracking Score): 40% - góc quay đầu (yaw, pitch, roll)
- EGA (Eye Gaze Alignment): 40% - trạng thái mắt và hướng nhìn
- MS (Movement Stability): 20% - ổn định chuyển động

THANG ĐIỂM 0-10:
- 8-10: Tập trung tốt (Focused)
- 6-8: Hơi mất tập trung (Slightly Distracted)
- 4-6: Mất tập trung (Distracted)
- 0-4: Rất mất tập trung (Very Distracted)

CÁC ĐIỀU CHỈNH:
1. Điều chỉnh trọng số FTS/EGA/MS (40-40-20)
2. Thêm điểm trung vị + điểm thấp nhất → hạn chế nhiễu
3. Phân loại mức độ mất tập trung (nhẹ/vừa/nặng)
4. Xử lý mất tập trung kéo dài (continuous lapse)
5. Trọng số camera/screen theo ngữ cảnh
6. Tự nhận diện chất lượng camera để giảm bias
7. Thang điểm 0-10 thống nhất với hệ thống cảm xúc
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict
from collections import deque
import time


class AttentionLevel:
    """Mức độ tập trung (thang 0-10)."""
    FOCUSED = "focused"           # Đang tập trung (8-10)
    SLIGHTLY_DISTRACTED = "slightly_distracted"  # Hơi mất tập trung (6-8)
    DISTRACTED = "distracted"     # Mất tập trung (4-6)
    VERY_DISTRACTED = "very_distracted"  # Rất mất tập trung (0-4)


class AttentionDetector:
    """
    Phát hiện mức độ tập trung từ khuôn mặt và mắt.
    
    Đánh giá dựa trên:
    - Eye Aspect Ratio (EAR) - phát hiện mắt nhắm
    - Gaze direction - hướng nhìn
    - Head pose - hướng đầu
    - Blink frequency - tần suất chớp mắt
    """
    
    def __init__(
        self,
        ear_threshold: float = 0.15,
        gaze_threshold: float = 0.45,
        pose_threshold: float = 25.0,
        history_size: int = 30,
        alert_duration: float = 4.0
    ):
        """
        Initialize AttentionDetector.
        
        Args:
            ear_threshold: Ngưỡng EAR để phát hiện mắt nhắm (0.15 - giảm để ít nhạy cảm hơn)
            gaze_threshold: Ngưỡng gaze để phát hiện nhìn ra ngoài (0.45 - tăng mạnh để giảm false positive)
            pose_threshold: Ngưỡng góc đầu (độ) - 25° để giảm nhạy cảm
            history_size: Số frame lưu lịch sử
            alert_duration: Thời gian mất tập trung trước khi cảnh báo (giây) - tăng lên 4s
        """
        self.ear_threshold = ear_threshold
        self.gaze_threshold = gaze_threshold
        self.pose_threshold = pose_threshold
        self.history_size = history_size
        self.alert_duration = alert_duration
        
        # Dead zone - vùng coi như đang nhìn thẳng (giảm nhiễu) - TĂNG MẠNH
        self.pose_dead_zone = 15.0  # Góc < 15° = nhìn thẳng (tăng từ 10°)
        self.gaze_dead_zone = 0.25  # Gaze deviation < 0.25 = nhìn thẳng (tăng từ 0.15)
        
        # History
        self.attention_history = deque(maxlen=history_size)
        self.ear_history = deque(maxlen=history_size)
        
        # Statistics
        self.total_frames = 0
        self.focused_frames = 0
        self.distracted_frames = 0
        self.eyes_closed_frames = 0
        self.looking_away_frames = 0
        
        # Alert tracking
        self.distraction_start_time = None
        self.last_alert_time = 0
        
        # Blink detection
        self.blink_counter = 0
        self.last_blink_time = 0
    
    def calculate_ear(self, eye_landmarks: np.ndarray) -> float:
        """
        Tính Eye Aspect Ratio (EAR).
        
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        
        Args:
            eye_landmarks: 6 điểm landmarks của mắt (hoặc 1 điểm cho MTCNN)
            
        Returns:
            EAR value (0-1, thấp = mắt nhắm)
        """
        # For MTCNN (only 1 point per eye), we can't calculate EAR
        # Return a default value indicating eyes are open
        if len(eye_landmarks) < 6:
            return 0.3  # Default value (eyes open)
        
        # Vertical distances
        v1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        v2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        
        # Horizontal distance
        h = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        
        if h == 0:
            return 0.3
        
        ear = (v1 + v2) / (2.0 * h)
        return ear
    
    def extract_eye_landmarks(
        self,
        landmarks: np.ndarray,
        eye: str = "left"
    ) -> Optional[np.ndarray]:
        """
        Trích xuất landmarks của mắt.
        
        Args:
            landmarks: Full facial landmarks
            eye: "left" hoặc "right"
            
        Returns:
            6 điểm landmarks của mắt hoặc None (hoặc 1 điểm cho MTCNN)
        """
        if len(landmarks) == 5:  # MTCNN format (5 points)
            # MTCNN: [left_eye, right_eye, nose, left_mouth, right_mouth]
            if eye == "left":
                return landmarks[0:1]  # Left eye point
            else:
                return landmarks[1:2]  # Right eye point
        
        elif len(landmarks) >= 468:  # MediaPipe format
            if eye == "left":
                # Left eye landmarks in MediaPipe
                indices = [33, 160, 158, 133, 153, 144]
            else:
                # Right eye landmarks
                indices = [362, 385, 387, 263, 373, 380]
            
            if len(landmarks) > max(indices):
                return landmarks[indices]
        
        elif len(landmarks) >= 68:  # dlib format
            if eye == "left":
                # Left eye: 36-41
                return landmarks[36:42]
            else:
                # Right eye: 42-47
                return landmarks[42:48]
        
        return None
    
    def detect_eyes_closed(self, landmarks: np.ndarray) -> Tuple[bool, float]:
        """
        Phát hiện mắt nhắm.
        
        Args:
            landmarks: Facial landmarks
            
        Returns:
            Tuple of (eyes_closed, average_ear)
        """
        # For MTCNN (5 points), we can't reliably detect eye closure
        # Return default values (eyes open)
        if len(landmarks) == 5:
            return False, 0.3
        
        # Extract eye landmarks
        left_eye = self.extract_eye_landmarks(landmarks, "left")
        right_eye = self.extract_eye_landmarks(landmarks, "right")
        
        if left_eye is None or right_eye is None:
            return False, 0.3
        
        # Calculate EAR for both eyes
        left_ear = self.calculate_ear(left_eye)
        right_ear = self.calculate_ear(right_eye)
        
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Store in history
        self.ear_history.append(avg_ear)
        
        # Eyes closed if EAR below threshold
        eyes_closed = avg_ear < self.ear_threshold
        
        # Detect blink (quick close and open)
        if eyes_closed and len(self.ear_history) > 1:
            if self.ear_history[-2] >= self.ear_threshold:
                # Just closed
                current_time = time.time()
                if current_time - self.last_blink_time > 0.2:  # Min 0.2s between blinks
                    self.blink_counter += 1
                    self.last_blink_time = current_time
        
        return eyes_closed, avg_ear
    
    def estimate_gaze_direction(
        self,
        landmarks: np.ndarray,
        frame_shape: Tuple[int, int]
    ) -> Tuple[float, float]:
        """
        Ước tính hướng nhìn (đơn giản hóa).
        
        Args:
            landmarks: Facial landmarks
            frame_shape: (height, width)
            
        Returns:
            Tuple of (horizontal_ratio, vertical_ratio)
            - 0.5, 0.5 = nhìn thẳng
            - < 0.5 = nhìn trái/lên
            - > 0.5 = nhìn phải/xuống
        """
        h, w = frame_shape
        
        # For MTCNN (5 points)
        if len(landmarks) == 5:
            # Use eye positions directly
            left_eye = landmarks[0]
            right_eye = landmarks[1]
            eye_center = (left_eye + right_eye) / 2.0
            
            # Normalize to frame
            h_ratio = eye_center[0] / w if w > 0 else 0.5
            v_ratio = eye_center[1] / h if h > 0 else 0.5
            
            return h_ratio, v_ratio
        
        # Get eye centers for other formats
        left_eye = self.extract_eye_landmarks(landmarks, "left")
        right_eye = self.extract_eye_landmarks(landmarks, "right")
        
        if left_eye is None or right_eye is None:
            return 0.5, 0.5
        
        # Calculate eye centers
        left_center = np.mean(left_eye, axis=0)
        right_center = np.mean(right_eye, axis=0)
        eye_center = (left_center + right_center) / 2.0
        
        # Normalize to frame
        h_ratio = eye_center[0] / w if w > 0 else 0.5
        v_ratio = eye_center[1] / h if h > 0 else 0.5
        
        return h_ratio, v_ratio
    
    def calculate_head_pose(self, landmarks: np.ndarray) -> Tuple[float, float, float]:
        """
        Tính góc đầu (yaw, pitch, roll).
        
        Args:
            landmarks: Facial landmarks
            
        Returns:
            Tuple of (yaw, pitch, roll) in degrees
        """
        # MTCNN format (5 points)
        if len(landmarks) == 5:
            # MTCNN: [left_eye, right_eye, nose, left_mouth, right_mouth]
            left_eye = landmarks[0]
            right_eye = landmarks[1]
            nose_tip = landmarks[2]
            left_mouth = landmarks[3]
            right_mouth = landmarks[4]
        
        # MediaPipe format
        elif len(landmarks) >= 468:
            nose_tip = landmarks[1]
            left_eye = landmarks[33]
            right_eye = landmarks[263]
            left_mouth = landmarks[61]
            right_mouth = landmarks[291]
        
        # dlib format
        elif len(landmarks) >= 68:
            nose_tip = landmarks[30]
            left_eye = landmarks[36]
            right_eye = landmarks[45]
            left_mouth = landmarks[48]
            right_mouth = landmarks[54]
        
        else:
            return 0.0, 0.0, 0.0
        
        # Calculate yaw (left-right rotation)
        eye_center = (left_eye + right_eye) / 2
        nose_to_left = np.linalg.norm(nose_tip - left_eye)
        nose_to_right = np.linalg.norm(nose_tip - right_eye)
        face_width = np.linalg.norm(right_eye - left_eye)
        
        if face_width > 0:
            # ĐẢO DẤU: nose gần left = quay trái (âm), nose gần right = quay phải (dương)
            yaw_ratio = (nose_to_left - nose_to_right) / face_width
            yaw = yaw_ratio * 90  # Scale to degrees
        else:
            yaw = 0.0
        
        # Calculate pitch (up-down rotation)
        mouth_center = (left_mouth + right_mouth) / 2
        eye_to_mouth = np.linalg.norm(mouth_center - eye_center)
        nose_to_eye = np.linalg.norm(nose_tip - eye_center)
        nose_to_mouth = np.linalg.norm(nose_tip - mouth_center)
        
        if eye_to_mouth > 0:
            # Tính pitch dựa trên tỷ lệ khoảng cách
            # Nhìn lên: nose gần eyes → pitch âm
            # Nhìn xuống: nose gần mouth → pitch dương
            pitch_ratio = (nose_to_eye - nose_to_mouth) / eye_to_mouth
            pitch = pitch_ratio * 45  # Scale to degrees
        else:
            pitch = 0.0
        
        # Calculate roll (tilt)
        d_eye = right_eye - left_eye
        roll = np.degrees(np.arctan2(d_eye[1], d_eye[0]))
        
        return yaw, pitch, roll
    
    def calculate_attention_score(
        self,
        landmarks: np.ndarray,
        frame_shape: Tuple[int, int],
        face_detected: bool = True
    ) -> Tuple[float, Dict]:
        """
        Tính điểm tập trung (0-10) theo CÔNG THỨC MỚI:
        
        FocusScore = (
            FacePresenceRatio * 0.40 +
            GazeFocus        * 0.30 +
            HeadFocus        * 0.20 +
            DriftScore       * 0.10
        ) * 10
        
        4 Thành phần:
        1. Face Presence (40%): Tỷ lệ có mặt trong khung hình
        2. Gaze Focus (30%): Nhìn thẳng vs nhìn sang trái/phải/xuống
        3. Head Focus (20%): Đầu thẳng vs quay đầu/cúi đầu
        4. Drift Score (10%): Phạt khi ngó nghiêng bất thường
        
        Args:
            landmarks: Facial landmarks (có thể None nếu không phát hiện được)
            frame_shape: (height, width)
            face_detected: True nếu phát hiện được khuôn mặt, False nếu không
            
        Returns:
            Tuple of (score_0_10, details_dict)
        """
        self.total_frames += 1
        
        details = {}
        
        # ===== 1. FACE PRESENCE RATIO (40%) =====
        if not face_detected or landmarks is None or len(landmarks) == 0:
            # KHÔNG có mặt
            details['face_detected'] = False
            details['reason'] = "Khong phat hien khuon mat"
            self.distracted_frames += 1
            
            # Track face loss for drift detection
            if not hasattr(self, '_last_face_present'):
                self._last_face_present = True
            if self._last_face_present:
                # Face just disappeared - count as drift event
                if not hasattr(self, '_drift_events'):
                    self._drift_events = 0
                self._drift_events += 1
            self._last_face_present = False
            
            face_presence_score = 0.0
            gaze_focus_score = 0.0
            head_focus_score = 0.0
        else:
            # CÓ mặt
            details['face_detected'] = True
            self.focused_frames += 1
            self._last_face_present = True
            
            face_presence_score = 1.0
        
            # ===== 2. GAZE FOCUS (30%) =====
            h_ratio, v_ratio = self.estimate_gaze_direction(landmarks, frame_shape)
            details['gaze_h'] = h_ratio
            details['gaze_v'] = v_ratio
            
            # Tính độ lệch từ trung tâm (0.5, 0.5)
            h_deviation = abs(h_ratio - 0.5)
            v_deviation = abs(v_ratio - 0.5)
            
            # Ngưỡng: lệch < 0.15 = nhìn thẳng, > 0.35 = nhìn xa
            if h_deviation < 0.15 and v_deviation < 0.15:
                gaze_focus_score = 1.0  # Nhìn thẳng
                details['gaze_status'] = "center"
            elif h_deviation > 0.35 or v_deviation > 0.35:
                gaze_focus_score = 0.0  # Nhìn xa
                if h_deviation > v_deviation:
                    details['gaze_status'] = "left" if h_ratio < 0.5 else "right"
                else:
                    details['gaze_status'] = "up" if v_ratio < 0.5 else "down"
            else:
                # Trung gian: tính tỷ lệ
                max_deviation = max(h_deviation, v_deviation)
                gaze_focus_score = 1.0 - ((max_deviation - 0.15) / (0.35 - 0.15))
                gaze_focus_score = max(0.0, min(1.0, gaze_focus_score))
                details['gaze_status'] = "slight_off"
            
            # ===== 3. HEAD FOCUS (20%) =====
            yaw, pitch, roll = self.calculate_head_pose(landmarks)
            details['yaw'] = yaw
            details['pitch'] = pitch
            details['roll'] = roll
            
            # Ngưỡng: < 15° = thẳng, > 25° = quay đầu
            max_angle = max(abs(yaw), abs(pitch), abs(roll))
            
            if max_angle < 15.0:
                head_focus_score = 1.0  # Đầu thẳng
                details['head_status'] = "straight"
            elif max_angle > 25.0:
                head_focus_score = 0.0  # Quay đầu nhiều
                if abs(yaw) == max_angle:
                    details['head_status'] = "turned_left" if yaw < 0 else "turned_right"
                elif abs(pitch) == max_angle:
                    details['head_status'] = "looking_down" if pitch > 0 else "looking_up"
                else:
                    details['head_status'] = "tilted"
            else:
                # Trung gian
                head_focus_score = 1.0 - ((max_angle - 15.0) / (25.0 - 15.0))
                head_focus_score = max(0.0, min(1.0, head_focus_score))
                details['head_status'] = "slight_turn"
            
            # Tính EAR (để hiển thị)
            is_mtcnn = len(landmarks) == 5
            if not is_mtcnn:
                eyes_closed, ear = self.detect_eyes_closed(landmarks)
                details['eyes_closed'] = eyes_closed
                details['ear'] = ear
            else:
                details['eyes_closed'] = False
                details['ear'] = 0.3
        
        # ===== 4. DRIFT SCORE (10%) =====
        # Tính số lần drift trong 60 giây gần nhất
        if not hasattr(self, '_drift_events'):
            self._drift_events = 0
        
        # Cho phép tối đa 3 lần drift/phút
        max_allowed_drifts = 3
        drift_score = 1.0 - min(1.0, self._drift_events / max_allowed_drifts)
        details['drift_events'] = self._drift_events
        details['drift_score'] = drift_score
        
        # Reset drift counter mỗi 60 giây
        if not hasattr(self, '_last_drift_reset'):
            self._last_drift_reset = time.time()
        if time.time() - self._last_drift_reset > 60.0:
            self._drift_events = 0
            self._last_drift_reset = time.time()
        
        # ===== TÍNH ĐIỂM INSTANT (0-1) =====
        instant_score_normalized = (
            face_presence_score * 0.40 +
            gaze_focus_score * 0.30 +
            head_focus_score * 0.20 +
            drift_score * 0.10
        )
        
        # Chuyển sang thang 0-10
        instant_score = instant_score_normalized * 10.0
        
        details['instant_score'] = instant_score
        details['face_presence_score'] = face_presence_score
        details['gaze_focus_score'] = gaze_focus_score
        details['head_focus_score'] = head_focus_score
        
        # Xác định lý do mất tập trung (nếu có)
        if instant_score < 7.0:
            if face_presence_score < 0.5:
                details['reason'] = "Khong co mat trong khung hinh"
            elif gaze_focus_score < 0.5:
                details['reason'] = f"Nhin {details.get('gaze_status', 'away')}"
            elif head_focus_score < 0.5:
                details['reason'] = f"Dau {details.get('head_status', 'turned')}"
            else:
                details['reason'] = "Ngo nghieng qua nhieu"
        else:
            details['reason'] = "Dang tap trung"
        
        # Store instant score
        self.attention_history.append(instant_score)
        
        # ===== 5. Tính FocusScore (thang 0-10) =====
        # Làm mượt bằng cách lấy trung bình 30 frame gần nhất (~1 giây ở 30fps)
        if len(self.attention_history) >= 10:
            recent_scores = list(self.attention_history)[-30:]
            focus_score = np.mean(recent_scores)
            
            # Đếm số frame mất tập trung (score < 6.0)
            distracted_count = sum(1 for s in recent_scores if s < 6.0)
            distracted_ratio = distracted_count / len(recent_scores)
            
            # Đếm số frame liên tục mất tập trung
            continuous_lapse = 0
            for s in reversed(recent_scores):
                if s < 6.0:
                    continuous_lapse += 1
                else:
                    break
            
            details['avg_score'] = focus_score
            details['median_score'] = np.median(recent_scores)
            details['distracted_ratio'] = distracted_ratio
            details['continuous_lapse'] = continuous_lapse
            
            # Cảnh báo nếu mất tập trung kéo dài >= 15 frame (~0.5s)
            if continuous_lapse >= 15:
                details['continuous_lapse_warning'] = True
        else:
            # Chưa đủ lịch sử - dùng instant score
            focus_score = instant_score
            details['avg_score'] = focus_score
            details['median_score'] = focus_score
            details['distracted_ratio'] = 0 if instant_score >= 6.0 else 1.0
            details['continuous_lapse'] = 0 if instant_score >= 6.0 else 1
        
        # Clamp score (0-10)
        final_score = max(0, min(10, focus_score))
        details['score'] = final_score
        
        # Reason đã được set ở trên (dựa vào face_detected)
        # Không cần kiểm tra thêm
        
        return final_score, details
    
    def get_attention_level(self, score: float) -> str:
        """Lấy mức độ tập trung từ điểm (thang 0-10) - Công thức 4 thành phần."""
        if score >= 7.5:
            return AttentionLevel.FOCUSED  # Tập trung tốt
        elif score >= 6.0:
            return AttentionLevel.SLIGHTLY_DISTRACTED  # Hơi mất tập trung
        elif score >= 4.0:
            return AttentionLevel.DISTRACTED  # Mất tập trung
        else:
            return AttentionLevel.VERY_DISTRACTED  # Rất mất tập trung
    
    def should_alert(self, score: float) -> bool:
        """
        Kiểm tra có nên cảnh báo không (thang 0-10).
        
        Logic MỚI với công thức 4 thành phần:
        - Nếu tập trung tốt (score >= 7.0): TẮT cảnh báo
        - Nếu mất tập trung (score < 5.0): Bắt đầu đếm thời gian
        - Nếu mất tập trung >= 3 giây: Cảnh báo
        - Cooldown: 3 giây giữa các cảnh báo
        
        Returns:
            True nếu cần cảnh báo
        """
        current_time = time.time()
        
        # If focused (score >= 7.0), reset ALL timers and STOP alert
        if score >= 7.0:
            if self.distraction_start_time is not None:
                self.distraction_start_time = None
                # QUAN TRỌNG: Reset last_alert_time để tắt cảnh báo
                if self.last_alert_time > 0:
                    print(f"✓ Đã tập trung lại! Score: {score:.1f}/10 - TẮT cảnh báo")
                self.last_alert_time = 0
            return False
        
        # If distracted (score < 5.0), start/continue timer
        if score < 5.0:
            if self.distraction_start_time is None:
                self.distraction_start_time = current_time
                print(f"⚠️ Bắt đầu mất tập trung... Score: {score:.1f}/10")
                return False  # Chưa đủ thời gian để cảnh báo
            
            # Check if distracted for too long
            distraction_duration = current_time - self.distraction_start_time
            
            # Nếu đã mất tập trung >= 3 giây
            if distraction_duration >= 3.0:
                # Check cooldown để tránh spam
                time_since_last_alert = current_time - self.last_alert_time
                
                if time_since_last_alert > 3.0:  # Cooldown 3 giây
                    self.last_alert_time = current_time
                    print(f"🚨 CẢNH BÁO: Mất tập trung {distraction_duration:.1f}s! Score: {score:.1f}/10")
                    # KHÔNG reset distraction_start_time - để tiếp tục cảnh báo
                    return True
        else:
            # Score between 5.0 and 7.0 (slightly distracted) - monitor but don't alert
            if self.distraction_start_time is not None:
                # Reset timer nếu đã cải thiện
                print(f"ℹ️ Hơi mất tập trung nhưng chưa nghiêm trọng. Score: {score:.1f}/10")
                self.distraction_start_time = None
        
        return False
    
    def get_average_attention(self) -> float:
        """Lấy điểm tập trung trung bình (thang 0-10)."""
        if len(self.attention_history) == 0:
            return 10.0
        return np.mean(list(self.attention_history))
    
    def get_statistics(self) -> Dict:
        """Lấy thống kê."""
        if self.total_frames == 0:
            return {
                'total_frames': 0,
                'focused_rate': 0.0,
                'distracted_rate': 0.0,
                'eyes_closed_rate': 0.0,
                'looking_away_rate': 0.0,
                'average_attention': 100.0,
                'blink_count': 0
            }
        
        return {
            'total_frames': self.total_frames,
            'focused_frames': self.focused_frames,
            'distracted_frames': self.distracted_frames,
            'focused_rate': self.focused_frames / self.total_frames,
            'distracted_rate': self.distracted_frames / self.total_frames,
            'eyes_closed_rate': self.eyes_closed_frames / self.total_frames,
            'looking_away_rate': self.looking_away_frames / self.total_frames,
            'average_attention': self.get_average_attention(),
            'blink_count': self.blink_counter
        }
    
    def reset(self):
        """Reset statistics."""
        self.attention_history.clear()
        self.ear_history.clear()
        self.total_frames = 0
        self.focused_frames = 0
        self.distracted_frames = 0
        self.eyes_closed_frames = 0
        self.looking_away_frames = 0
        self.blink_counter = 0
        self.distraction_start_time = None
        self.last_alert_time = 0
