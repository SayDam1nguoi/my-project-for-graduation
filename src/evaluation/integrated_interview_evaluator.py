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
        Đánh giá rõ ràng lời nói (Simplified Version).
        
        Dựa trên 3 yếu tố:
        1. ASR Confidence (60%) - Độ rõ phát âm
        2. Transcript Length (20%) - Độ dài phù hợp
        3. Speech Rate (20%) - Tốc độ nói
        
        Returns:
            Clarity score (0-10)
        """
        try:
            # Kiểm tra có transcript không
            if not transcription or len(transcription.strip()) == 0:
                logger.warning("No transcription available for clarity evaluation")
                details['clarity'] = {
                    'score': 3.0,
                    'error': 'No transcription',
                    'method': 'simplified'
                }
                return 3.0
            
            # Lấy ASR confidence từ transcriber (nếu có)
            asr_confidence = None
            try:
                from src.audio_recording.transcriber import get_transcriber
                transcriber = get_transcriber()
                if hasattr(transcriber, 'last_confidence') and transcriber.last_confidence is not None:
                    asr_confidence = transcriber.last_confidence
                    logger.info(f"  Using ASR confidence from transcriber: {asr_confidence:.2%}")
            except Exception as e:
                logger.debug(f"Could not get confidence from transcriber: {e}")
            
            # Lấy video duration để tính WPM
            video_duration = None
            try:
                import cv2
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    if fps > 0:
                        video_duration = frame_count / fps
                        logger.info(f"  Video duration: {video_duration:.1f}s")
                    cap.release()
            except Exception as e:
                logger.debug(f"Could not get video duration: {e}")
            
            # Nếu không có confidence, ước tính từ transcript quality
            if asr_confidence is None:
                words = transcription.split()
                word_count = len(words)
                
                # Ước tính confidence dựa trên độ dài transcript
                if word_count > 100:
                    asr_confidence = 0.80
                elif word_count > 50:
                    asr_confidence = 0.75
                else:
                    asr_confidence = 0.70
                
                logger.info(f"  ASR confidence estimated: {asr_confidence:.2%}")
            
            # Tính clarity score đơn giản
            clarity_score, clarity_details = self._calculate_simple_clarity(
                transcription, asr_confidence, video_duration
            )
            
            details['clarity'] = {
                'score': clarity_score,
                'method': 'simplified',
                'asr_confidence': asr_confidence,
                'video_duration': video_duration,
                'transcription_length': len(transcription),
                'word_count': clarity_details['word_count'],
                'wpm': clarity_details.get('wpm'),
                'confidence_score': clarity_details['confidence_score'],
                'length_score': clarity_details['length_score'],
                'rate_score': clarity_details['rate_score']
            }
            
            logger.info(f"  Clarity score: {clarity_score:.2f}/10 (simplified method)")
            logger.info(f"    - Confidence: {clarity_details['confidence_score']:.1f}/10 (weight: 60%)")
            logger.info(f"    - Length: {clarity_details['length_score']:.1f}/10 (weight: 20%)")
            logger.info(f"    - Rate: {clarity_details['rate_score']:.1f}/10 (weight: 20%)")
            if clarity_details.get('wpm'):
                logger.info(f"    - WPM: {clarity_details['wpm']:.1f}")
            
            return clarity_score
            
        except Exception as e:
            logger.error(f"Error evaluating speech clarity: {e}")
            details['clarity'] = {'error': str(e), 'score': 5.0}
            return 5.0
    
    def _calculate_simple_clarity(
        self,
        transcript: str,
        asr_confidence: float,
        video_duration: Optional[float] = None
    ) -> Tuple[float, Dict]:
        """
        Tính clarity score đơn giản.
        
        Args:
            transcript: Transcript text
            asr_confidence: ASR confidence (0-1)
            video_duration: Video duration in seconds (optional)
            
        Returns:
            (clarity_score, details_dict)
        """
        # 1. ASR Confidence Score (60%)
        # Whisper confidence thường cao, nên điều chỉnh thang điểm
        if asr_confidence >= 0.85:
            conf_score = 10.0
        elif asr_confidence >= 0.75:
            # Linear interpolation: 0.75→8, 0.85→10
            conf_score = 8.0 + (asr_confidence - 0.75) * 20
        elif asr_confidence >= 0.65:
            # Linear interpolation: 0.65→6, 0.75→8
            conf_score = 6.0 + (asr_confidence - 0.65) * 20
        elif asr_confidence >= 0.50:
            # Linear interpolation: 0.50→4, 0.65→6
            conf_score = 4.0 + (asr_confidence - 0.50) * 13.33
        else:
            # Linear: 0→0, 0.50→4
            conf_score = asr_confidence * 8
        
        # 2. Transcript Length Score (20%)
        # Optimal: 50-500 words (phỏng vấn ~3-5 phút)
        words = transcript.split()
        word_count = len(words)
        
        if 50 <= word_count <= 500:
            length_score = 10.0
        elif word_count < 50:
            # Quá ngắn: scale từ 0 đến 10
            length_score = (word_count / 50) * 10
        else:
            # Quá dài: giảm dần
            length_score = max(0, 10 - (word_count - 500) / 100)
        
        # 3. Speech Rate Score (20%)
        # Optimal: 120-160 WPM (words per minute)
        if video_duration and video_duration > 0:
            wpm = (word_count / video_duration) * 60
            
            if 120 <= wpm <= 160:
                rate_score = 10.0
            elif 100 <= wpm < 120:
                # Hơi chậm: 100→7, 120→10
                rate_score = 7.0 + (wpm - 100) * 0.15
            elif 160 < wpm <= 180:
                # Hơi nhanh: 160→10, 180→7
                rate_score = 10.0 - (wpm - 160) * 0.15
            elif wpm < 100:
                # Quá chậm
                rate_score = max(0, 7.0 - (100 - wpm) * 0.1)
            else:
                # Quá nhanh (>180)
                rate_score = max(0, 7.0 - (wpm - 180) * 0.1)
        else:
            # Không có duration, ước tính từ word count
            # Giả sử video ~3 phút = 180s
            # Optimal: 360-480 words
            if 360 <= word_count <= 480:
                rate_score = 10.0
            else:
                deviation = abs(word_count - 420) / 420
                rate_score = max(0, 10 - deviation * 10)
        
        # Tính tổng có trọng số
        clarity_score = (
            conf_score * 0.60 +
            length_score * 0.20 +
            rate_score * 0.20
        )
        
        # Clamp vào [0, 10]
        clarity_score = max(0.0, min(10.0, clarity_score))
        
        details = {
            'confidence_score': conf_score,
            'length_score': length_score,
            'rate_score': rate_score,
            'word_count': word_count,
            'asr_confidence': asr_confidence,
            'video_duration': video_duration,
            'wpm': (word_count / video_duration * 60) if video_duration and video_duration > 0 else None
        }
        
        return clarity_score, details
    
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
