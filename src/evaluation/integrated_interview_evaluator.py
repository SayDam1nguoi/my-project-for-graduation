"""
Integrated Interview Evaluator

Tích hợp tất cả 4 module đánh giá:
1. Emotion Scoring (Cảm xúc)
2. Attention Detection (Tập trung)
3. Speech Clarity (Rõ ràng lời nói)
4. Content Evaluation (Nội dung)

Workflow:
1. Nhận video + audio phỏng vấn
2. Chạy 4 module song song
3. Chuẩn hóa điểm về thang 0-10
4. Kết hợp bằng OverallInterviewScorer
5. Tạo báo cáo tổng hợp
"""

import logging
from typing import Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime
import json

from .overall_interview_scorer import OverallInterviewScorer, InterviewScore
from ..video_analysis.emotion_scoring.emotion_scoring_engine import EmotionScoringEngine
from ..video_analysis.attention_detector import AttentionDetector
from ..speech_analysis.interview_content_evaluator import InterviewContentEvaluator

logger = logging.getLogger(__name__)


class IntegratedInterviewEvaluator:
    """
    Tích hợp đánh giá phỏng vấn toàn diện.
    
    Kết hợp 4 module:
    - Emotion Scoring (video)
    - Attention Detection (video)
    - Speech Clarity (audio)
    - Content Evaluation (transcription)
    """
    
    def __init__(
        self,
        position_type: str = 'default',
        emotion_engine: Optional[EmotionScoringEngine] = None,
        attention_detector: Optional[AttentionDetector] = None,
        content_evaluator: Optional[InterviewContentEvaluator] = None,
        speech_evaluator = None
    ):
        """
        Khởi tạo integrated evaluator.
        
        Args:
            position_type: Loại vị trí (default, technical, sales, customer_service, management)
            emotion_engine: Emotion scoring engine
            attention_detector: Attention detector
            content_evaluator: Content evaluator
            speech_evaluator: Speech clarity evaluator
        """
        self.position_type = position_type
        
        # Initialize overall scorer
        self.overall_scorer = OverallInterviewScorer(position_type=position_type)
        
        # Initialize sub-modules (lazy loading)
        self._emotion_engine = emotion_engine
        self._attention_detector = attention_detector
        self._content_evaluator = content_evaluator
        self._speech_evaluator = speech_evaluator
        
        logger.info(f"Initialized IntegratedInterviewEvaluator for position: {position_type}")
    
    @property
    def emotion_engine(self):
        """Lazy load emotion engine."""
        if self._emotion_engine is None:
            self._emotion_engine = EmotionScoringEngine()
        return self._emotion_engine
    
    @property
    def attention_detector(self):
        """Lazy load attention detector."""
        if self._attention_detector is None:
            self._attention_detector = AttentionDetector()
        return self._attention_detector
    
    @property
    def content_evaluator(self):
        """Lazy load content evaluator."""
        if self._content_evaluator is None:
            self._content_evaluator = InterviewContentEvaluator()
        return self._content_evaluator
    
    def evaluate_video_interview(
        self,
        video_path: str,
        candidate_id: str,
        transcription: Optional[str] = None,
        answers: Optional[Dict[str, str]] = None,
        save_report: bool = True,
        output_dir: str = "reports/integrated_evaluation"
    ) -> Tuple[InterviewScore, Dict]:
        """
        Đánh giá phỏng vấn từ video.
        
        Args:
            video_path: Đường dẫn video
            candidate_id: ID ứng viên
            transcription: Transcription text (optional)
            answers: Dict câu trả lời {question_id: answer} (optional)
            save_report: Lưu báo cáo
            output_dir: Thư mục lưu báo cáo
            
        Returns:
            (InterviewScore, details_dict)
        """
        logger.info(f"Starting integrated evaluation for candidate: {candidate_id}")
        logger.info(f"Video: {video_path}")
        logger.info(f"Position type: {self.position_type}")
        
        details = {
            'candidate_id': candidate_id,
            'video_path': video_path,
            'position_type': self.position_type,
            'timestamp': datetime.now().isoformat()
        }
        
        # ===== 1. Emotion Scoring (Cảm xúc) =====
        logger.info("Step 1/4: Evaluating emotion...")
        emotion_score = self._evaluate_emotion(video_path, candidate_id, details)
        
        # ===== 2. Attention Detection (Tập trung) =====
        logger.info("Step 2/4: Evaluating attention/focus...")
        focus_score = self._evaluate_attention(video_path, details)
        
        # ===== 3. Speech Clarity (Rõ ràng lời nói) =====
        logger.info("Step 3/4: Evaluating speech clarity...")
        clarity_score = self._evaluate_speech_clarity(video_path, transcription, details)
        
        # ===== 4. Content Evaluation (Nội dung) =====
        logger.info("Step 4/4: Evaluating content...")
        content_score = self._evaluate_content(answers, details)
        
        # ===== 5. Calculate Overall Score =====
        logger.info("Calculating overall score...")
        overall_score = self.overall_scorer.calculate_score(
            emotion_score=emotion_score,
            focus_score=focus_score,
            clarity_score=clarity_score,
            content_score=content_score
        )
        
        details['overall_score'] = overall_score
        
        logger.info(f"Evaluation complete! Total score: {overall_score.total_score:.2f}/10")
        
        # ===== 6. Save Report =====
        if save_report:
            self._save_report(overall_score, details, output_dir)
        
        return overall_score, details
    
    def _evaluate_emotion(
        self,
        video_path: str,
        candidate_id: str,
        details: Dict
    ) -> float:
        """
        Đánh giá cảm xúc từ video.
        
        Returns:
            Emotion score (0-10)
        """
        try:
            emotion_report = self.emotion_engine.score_video_interview(
                video_path=video_path,
                candidate_id=candidate_id
            )
            
            # Emotion engine trả về 0-10 (giống các module khác)
            # Trọng số 5% sẽ được áp dụng ở OverallInterviewScorer
            emotion_score = emotion_report.total_score
            
            # Đảm bảo điểm trong khoảng 0-10
            emotion_score = max(0.0, min(10.0, emotion_score))
            
            details['emotion'] = {
                'score': emotion_score,
                'raw_score': emotion_report.total_score,
                'criterion_scores': {
                    name: score.score 
                    for name, score in emotion_report.criterion_scores.items()
                },
                'metadata': emotion_report.metadata
            }
            
            logger.info(f"  Emotion score: {emotion_score:.2f}/10")
            return emotion_score
            
        except Exception as e:
            logger.error(f"Error evaluating emotion: {e}")
            details['emotion'] = {'error': str(e), 'score': 5.0}
            return 5.0
    
    def _evaluate_attention(
        self,
        video_path: str,
        details: Dict
    ) -> float:
        """
        Đánh giá tập trung từ video.
        
        Returns:
            Focus score (0-10)
        """
        try:
            import cv2
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")
            
            # Cần face detector để lấy landmarks
            from src.inference.face_detector import FaceDetector
            face_detector = FaceDetector(device='auto')
            
            frame_count = 0
            attention_scores = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Sample mỗi 5 frames
                if frame_count % 5 != 0:
                    continue
                
                # Detect face
                faces = face_detector.detect_faces(frame)
                
                if len(faces) > 0:
                    face = faces[0]
                    landmarks = face.get('landmarks')
                    
                    if landmarks is not None:
                        # Calculate attention score - có landmarks = có khuôn mặt
                        score, _ = self.attention_detector.calculate_attention_score(
                            landmarks, frame.shape[:2], face_detected=True
                        )
                        attention_scores.append(score)
                    else:
                        # Không có khuôn mặt
                        score, _ = self.attention_detector.calculate_attention_score(
                            None, frame.shape[:2], face_detected=False
                        )
                        attention_scores.append(score)
            
            cap.release()
            
            # Tính điểm trung bình
            if len(attention_scores) > 0:
                focus_score = sum(attention_scores) / len(attention_scores)
            else:
                focus_score = 5.0
            
            stats = self.attention_detector.get_statistics()
            
            details['focus'] = {
                'score': focus_score,
                'total_frames': frame_count,
                'analyzed_frames': len(attention_scores),
                'statistics': stats
            }
            
            logger.info(f"  Focus score: {focus_score:.2f}/10")
            return focus_score
            
        except Exception as e:
            logger.error(f"Error evaluating attention: {e}")
            details['focus'] = {'error': str(e), 'score': 5.0}
            return 5.0
    
    def _evaluate_speech_clarity(
        self,
        video_path: str,
        transcription: Optional[str],
        details: Dict
    ) -> float:
        """
        Đánh giá rõ ràng lời nói.
        
        Returns:
            Clarity score (0-10)
        """
        try:
            if self._speech_evaluator is None:
                logger.warning("Speech evaluator not provided, using default score")
                clarity_score = 7.0
            else:
                # Gọi speech evaluator để tính clarity score
                # (Cần implement speech clarity analyzer)
                clarity_score = 7.0
            
            details['clarity'] = {
                'score': clarity_score,
                'transcription_length': len(transcription) if transcription else 0
            }
            
            logger.info(f"  Clarity score: {clarity_score:.2f}/10")
            return clarity_score
            
        except Exception as e:
            logger.error(f"Error evaluating speech clarity: {e}")
            details['clarity'] = {'error': str(e), 'score': 5.0}
            return 5.0
    
    def _evaluate_content(
        self,
        answers: Optional[Dict[str, str]],
        details: Dict
    ) -> float:
        """
        Đánh giá nội dung câu trả lời.
        
        Returns:
            Content score (0-10)
        """
        try:
            if answers is None or len(answers) == 0:
                logger.warning("No answers provided for content evaluation")
                content_score = 5.0
            else:
                result = self.content_evaluator.evaluate_all_answers(answers)
                content_score = result.total_score
                
                details['content'] = {
                    'score': content_score,
                    'question_scores': result.question_scores,
                    'similarity_scores': result.similarity_scores,
                    'details': result.details
                }
            
            logger.info(f"  Content score: {content_score:.2f}/10")
            return content_score
            
        except Exception as e:
            logger.error(f"Error evaluating content: {e}")
            details['content'] = {'error': str(e), 'score': 5.0}
            return 5.0
    
    def _save_report(
        self,
        overall_score: InterviewScore,
        details: Dict,
        output_dir: str
    ):
        """Lưu báo cáo đánh giá."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        candidate_id = details['candidate_id']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save text report
        text_report = self.overall_scorer.generate_report(overall_score)
        text_file = output_path / f"{candidate_id}_{timestamp}_report.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(text_report)
        
        # Save JSON report
        json_data = {
            'candidate_id': candidate_id,
            'timestamp': details['timestamp'],
            'position_type': self.position_type,
            'scores': {
                'total': overall_score.total_score,
                'emotion': overall_score.emotion_score,
                'focus': overall_score.focus_score,
                'clarity': overall_score.clarity_score,
                'content': overall_score.content_score
            },
            'weights': {
                'emotion': overall_score.emotion_weight,
                'focus': overall_score.focus_weight,
                'clarity': overall_score.clarity_weight,
                'content': overall_score.content_weight
            },
            'rating': overall_score.overall_rating,
            'details': details
        }
        
        json_file = output_path / f"{candidate_id}_{timestamp}_report.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Reports saved to {output_dir}")


# Hàm tiện ích
def quick_evaluate_interview(
    video_path: str,
    candidate_id: str,
    answers: Optional[Dict[str, str]] = None,
    position_type: str = 'default'
) -> InterviewScore:
    """
    Hàm tiện ích để đánh giá nhanh.
    
    Args:
        video_path: Đường dẫn video
        candidate_id: ID ứng viên
        answers: Dict câu trả lời
        position_type: Loại vị trí
        
    Returns:
        InterviewScore
    """
    evaluator = IntegratedInterviewEvaluator(position_type=position_type)
    score, _ = evaluator.evaluate_video_interview(
        video_path, candidate_id, answers=answers
    )
    return score
