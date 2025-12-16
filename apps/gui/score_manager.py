# -*- coding: utf-8 -*-
"""
Score Manager - Quản lý điểm giữa các tab

Singleton class để chia sẻ điểm giữa:
- Tab "Nhận Diện Cảm Xúc" (Emotion + Focus)
- Tab "Chuyển Đổi Audio" (Clarity + Content)
- Tab "Tổng Hợp Điểm" (Summary)
"""

from typing import Optional, Callable
import logging

logger = logging.getLogger(__name__)


class ScoreManager:
    """
    Singleton manager để chia sẻ điểm giữa các tab.
    
    Điểm được tính theo thang 0-10.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize score manager (chỉ chạy 1 lần)."""
        if self._initialized:
            return
        
        # Điểm số (0-10)
        self.emotion_score: float = 0.0
        self.focus_score: float = 0.0
        self.clarity_score: float = 0.0
        self.content_score: float = 0.0
        
        # Metadata
        self.emotion_source: Optional[str] = None  # Nguồn điểm emotion
        self.focus_source: Optional[str] = None    # Nguồn điểm focus
        self.clarity_source: Optional[str] = None  # Nguồn điểm clarity
        self.content_source: Optional[str] = None  # Nguồn điểm content
        
        # Callbacks để notify khi có điểm mới
        self.callbacks: list[Callable] = []
        
        self._initialized = True
        logger.info("ScoreManager initialized")
    
    def set_emotion_score(self, score: float, source: str = "unknown"):
        """
        Cập nhật điểm Cảm xúc (Emotion).
        
        Args:
            score: Điểm 0-10
            source: Nguồn điểm (ví dụ: "emotion_recognition_tab")
        """
        score = max(0.0, min(10.0, score))  # Clamp 0-10
        self.emotion_score = score
        self.emotion_source = source
        logger.info(f"Emotion score updated: {score:.2f} from {source}")
        self._notify_callbacks("emotion", score)
    
    def set_focus_score(self, score: float, source: str = "unknown"):
        """
        Cập nhật điểm Tập trung (Focus).
        
        Args:
            score: Điểm 0-10
            source: Nguồn điểm
        """
        score = max(0.0, min(10.0, score))
        self.focus_score = score
        self.focus_source = source
        logger.info(f"Focus score updated: {score:.2f} from {source}")
        self._notify_callbacks("focus", score)
    
    def set_clarity_score(self, score: float, source: str = "unknown"):
        """
        Cập nhật điểm Rõ ràng (Clarity/Giọng nói).
        
        Args:
            score: Điểm 0-10
            source: Nguồn điểm
        """
        score = max(0.0, min(10.0, score))
        self.clarity_score = score
        self.clarity_source = source
        logger.info(f"Clarity score updated: {score:.2f} from {source}")
        self._notify_callbacks("clarity", score)
    
    def set_content_score(self, score: float, source: str = "unknown"):
        """
        Cập nhật điểm Nội dung (Content).
        
        Args:
            score: Điểm 0-10
            source: Nguồn điểm
        """
        score = max(0.0, min(10.0, score))
        self.content_score = score
        self.content_source = source
        logger.info(f"Content score updated: {score:.2f} from {source}")
        self._notify_callbacks("content", score)
    
    def get_all_scores(self) -> dict:
        """
        Lấy tất cả điểm.
        
        Returns:
            Dictionary chứa tất cả điểm và metadata
        """
        return {
            "emotion": {
                "score": self.emotion_score,
                "source": self.emotion_source
            },
            "focus": {
                "score": self.focus_score,
                "source": self.focus_source
            },
            "clarity": {
                "score": self.clarity_score,
                "source": self.clarity_source
            },
            "content": {
                "score": self.content_score,
                "source": self.content_source
            }
        }
    
    def calculate_total_score(
        self,
        weight_content: float = 40.0,
        weight_focus: float = 30.0,
        weight_clarity: float = 25.0,
        weight_emotion: float = 5.0
    ) -> float:
        """
        Tính điểm tổng theo công thức.
        
        Công thức mặc định:
        - Nội dung (N): 40%
        - Tập trung (T): 30%
        - Giọng nói (G): 25%
        - Cảm xúc (O): 5%
        
        Args:
            weight_content: Trọng số nội dung (%)
            weight_focus: Trọng số tập trung (%)
            weight_clarity: Trọng số giọng nói (%)
            weight_emotion: Trọng số cảm xúc (%)
        
        Returns:
            Điểm tổng (0-10)
        """
        # Kiểm tra tổng trọng số = 100%
        total_weight = weight_content + weight_focus + weight_clarity + weight_emotion
        if abs(total_weight - 100.0) > 0.01:
            logger.warning(f"Total weight is {total_weight}%, not 100%")
        
        # Tính điểm tổng
        total = (
            self.content_score * (weight_content / 100.0) +
            self.focus_score * (weight_focus / 100.0) +
            self.clarity_score * (weight_clarity / 100.0) +
            self.emotion_score * (weight_emotion / 100.0)
        )
        
        return round(total, 2)
    
    def register_callback(self, callback: Callable):
        """
        Đăng ký callback để nhận thông báo khi có điểm mới.
        
        Callback signature: callback(score_type: str, score: float)
        
        Args:
            callback: Function được gọi khi có điểm mới
        """
        if callback not in self.callbacks:
            self.callbacks.append(callback)
            logger.info(f"Callback registered: {callback.__name__}")
    
    def unregister_callback(self, callback: Callable):
        """
        Hủy đăng ký callback.
        
        Args:
            callback: Function cần hủy
        """
        if callback in self.callbacks:
            self.callbacks.remove(callback)
            logger.info(f"Callback unregistered: {callback.__name__}")
    
    def _notify_callbacks(self, score_type: str, score: float):
        """
        Thông báo cho tất cả callbacks.
        
        Args:
            score_type: Loại điểm ("emotion", "focus", "clarity", "content")
            score: Giá trị điểm
        """
        for callback in self.callbacks:
            try:
                callback(score_type, score)
            except Exception as e:
                logger.error(f"Error in callback {callback.__name__}: {e}")
    
    def reset(self):
        """Reset tất cả điểm về 0."""
        self.emotion_score = 0.0
        self.focus_score = 0.0
        self.clarity_score = 0.0
        self.content_score = 0.0
        
        self.emotion_source = None
        self.focus_source = None
        self.clarity_source = None
        self.content_source = None
        
        logger.info("All scores reset to 0")
        self._notify_callbacks("reset", 0.0)
    
    def has_all_scores(self) -> bool:
        """
        Kiểm tra xem đã có đủ 4 điểm chưa.
        
        Returns:
            True nếu tất cả điểm > 0
        """
        return all([
            self.emotion_score > 0,
            self.focus_score > 0,
            self.clarity_score > 0,
            self.content_score > 0
        ])
    
    def get_missing_scores(self) -> list[str]:
        """
        Lấy danh sách các điểm còn thiếu.
        
        Returns:
            List tên các điểm còn thiếu
        """
        missing = []
        if self.emotion_score <= 0:
            missing.append("Cảm xúc (Emotion)")
        if self.focus_score <= 0:
            missing.append("Tập trung (Focus)")
        if self.clarity_score <= 0:
            missing.append("Rõ ràng (Clarity)")
        if self.content_score <= 0:
            missing.append("Nội dung (Content)")
        return missing


# Singleton instance
_score_manager = ScoreManager()


def get_score_manager() -> ScoreManager:
    """
    Lấy singleton instance của ScoreManager.
    
    Returns:
        ScoreManager instance
    """
    return _score_manager


# Example usage
if __name__ == "__main__":
    # Test score manager
    manager = get_score_manager()
    
    # Set scores
    manager.set_emotion_score(8.5, "test")
    manager.set_focus_score(7.2, "test")
    manager.set_clarity_score(9.0, "test")
    manager.set_content_score(8.0, "test")
    
    # Get all scores
    print("All scores:", manager.get_all_scores())
    
    # Calculate total
    total = manager.calculate_total_score()
    print(f"Total score: {total}/10")
    
    # Check if has all scores
    print(f"Has all scores: {manager.has_all_scores()}")
    
    # Reset
    manager.reset()
    print(f"After reset: {manager.get_all_scores()}")
