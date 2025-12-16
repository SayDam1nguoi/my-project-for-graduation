"""
Integrated Speech Evaluator

Tích hợp đánh giá nội dung vào quy trình transcription.
Sau khi dịch WAV sang text, tự động đánh giá nội dung câu trả lời.
"""

import logging
from typing import Dict, Optional, Tuple
from pathlib import Path
import json

from .interview_content_evaluator import (
    InterviewContentEvaluator,
    ContentEvaluationResult
)

logger = logging.getLogger(__name__)


class IntegratedSpeechEvaluator:
    """
    Tích hợp transcription và content evaluation.
    """
    
    def __init__(
        self,
        transcriber=None,
        content_evaluator: Optional[InterviewContentEvaluator] = None,
        auto_evaluate: bool = True
    ):
        """
        Khởi tạo integrated evaluator.
        
        Args:
            transcriber: Speech transcriber instance (VoskTranscriber, WhisperTranscriber, etc.)
            content_evaluator: Content evaluator instance
            auto_evaluate: Tự động đánh giá sau khi transcribe
        """
        self.transcriber = transcriber
        self.content_evaluator = content_evaluator or InterviewContentEvaluator()
        self.auto_evaluate = auto_evaluate
        
        # Cache kết quả
        self.transcription_cache = {}
        self.evaluation_cache = {}
        
        logger.info("Initialized IntegratedSpeechEvaluator")
    
    def transcribe_and_evaluate(
        self,
        audio_file: str,
        question_id: str,
        save_results: bool = True,
        output_dir: str = "reports/integrated_evaluation"
    ) -> Tuple[str, Optional[ContentEvaluationResult]]:
        """
        Transcribe audio và đánh giá nội dung.
        
        Args:
            audio_file: Đường dẫn file audio
            question_id: ID câu hỏi (Q1, Q2, ...)
            save_results: Lưu kết quả ra file
            output_dir: Thư mục lưu kết quả
            
        Returns:
            (transcription_text, evaluation_result)
        """
        logger.info(f"Processing audio: {audio_file} for question: {question_id}")
        
        # Step 1: Transcribe audio
        transcription = self._transcribe_audio(audio_file)
        
        if not transcription or len(transcription.strip()) < 10:
            logger.warning(f"Transcription too short or empty: {transcription}")
            return transcription, None
        
        # Step 2: Evaluate content (nếu auto_evaluate)
        evaluation_result = None
        if self.auto_evaluate:
            evaluation_result = self._evaluate_content(question_id, transcription)
        
        # Step 3: Save results
        if save_results:
            self._save_results(
                audio_file, question_id, transcription, 
                evaluation_result, output_dir
            )
        
        return transcription, evaluation_result
    
    def transcribe_and_evaluate_multiple(
        self,
        audio_files: Dict[str, str],
        save_results: bool = True,
        output_dir: str = "reports/integrated_evaluation"
    ) -> Dict[str, Tuple[str, Optional[ContentEvaluationResult]]]:
        """
        Transcribe và đánh giá nhiều file audio.
        
        Args:
            audio_files: Dict {question_id: audio_file_path}
            save_results: Lưu kết quả
            output_dir: Thư mục lưu kết quả
            
        Returns:
            Dict {question_id: (transcription, evaluation_result)}
        """
        results = {}
        
        for question_id, audio_file in audio_files.items():
            transcription, evaluation = self.transcribe_and_evaluate(
                audio_file, question_id, save_results=False
            )
            results[question_id] = (transcription, evaluation)
        
        # Tính tổng điểm cho tất cả câu hỏi
        if self.auto_evaluate:
            total_result = self._compute_total_score(results)
            
            if save_results:
                self._save_total_results(
                    audio_files, results, total_result, output_dir
                )
        
        return results
    
    def _transcribe_audio(self, audio_file: str) -> str:
        """
        Transcribe audio file.
        
        Args:
            audio_file: Đường dẫn file audio
            
        Returns:
            Transcription text
        """
        # Check cache
        if audio_file in self.transcription_cache:
            logger.info(f"Using cached transcription for {audio_file}")
            return self.transcription_cache[audio_file]
        
        # Transcribe
        if self.transcriber is None:
            logger.warning("No transcriber provided, returning empty string")
            return ""
        
        try:
            transcription = self.transcriber.transcribe(audio_file)
            self.transcription_cache[audio_file] = transcription
            logger.info(f"Transcribed: {transcription[:100]}...")
            return transcription
        except Exception as e:
            logger.error(f"Error transcribing {audio_file}: {e}")
            return ""
    
    def _evaluate_content(
        self,
        question_id: str,
        transcription: str
    ) -> Optional[ContentEvaluationResult]:
        """
        Đánh giá nội dung transcription.
        
        Args:
            question_id: ID câu hỏi
            transcription: Text đã transcribe
            
        Returns:
            Evaluation result
        """
        try:
            score, similarity, best_match = self.content_evaluator.evaluate_answer(
                question_id, transcription
            )
            
            # Tạo result object đơn giản
            result = ContentEvaluationResult(
                total_score=score,
                question_scores={question_id: score},
                similarity_scores={question_id: similarity},
                best_matches={question_id: best_match},
                details={
                    "question_id": question_id,
                    "transcription": transcription,
                    "score": score,
                    "similarity": similarity,
                    "best_match": best_match
                }
            )
            
            logger.info(f"Evaluated {question_id}: Score={score:.1f}, Similarity={similarity:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating content: {e}")
            return None
    
    def _compute_total_score(
        self,
        results: Dict[str, Tuple[str, Optional[ContentEvaluationResult]]]
    ) -> Optional[ContentEvaluationResult]:
        """
        Tính tổng điểm cho tất cả câu hỏi.
        
        Args:
            results: Dict {question_id: (transcription, evaluation)}
            
        Returns:
            Total evaluation result
        """
        answers = {}
        for question_id, (transcription, _) in results.items():
            if transcription:
                answers[question_id] = transcription
        
        if not answers:
            return None
        
        try:
            total_result = self.content_evaluator.evaluate_all_answers(answers)
            return total_result
        except Exception as e:
            logger.error(f"Error computing total score: {e}")
            return None
    
    def _save_results(
        self,
        audio_file: str,
        question_id: str,
        transcription: str,
        evaluation: Optional[ContentEvaluationResult],
        output_dir: str
    ):
        """Lưu kết quả ra file."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Tạo filename từ audio file
        audio_name = Path(audio_file).stem
        
        # Save transcription
        transcript_file = output_path / f"{audio_name}_{question_id}_transcript.txt"
        with open(transcript_file, 'w', encoding='utf-8') as f:
            f.write(transcription)
        
        # Save evaluation
        if evaluation:
            eval_file = output_path / f"{audio_name}_{question_id}_evaluation.json"
            eval_data = {
                "audio_file": audio_file,
                "question_id": question_id,
                "transcription": transcription,
                "score": evaluation.total_score,
                "similarity": evaluation.similarity_scores.get(question_id, 0.0),
                "details": evaluation.details
            }
            
            with open(eval_file, 'w', encoding='utf-8') as f:
                json.dump(eval_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved results to {output_dir}")
    
    def _save_total_results(
        self,
        audio_files: Dict[str, str],
        results: Dict[str, Tuple[str, Optional[ContentEvaluationResult]]],
        total_result: Optional[ContentEvaluationResult],
        output_dir: str
    ):
        """Lưu kết quả tổng hợp."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Tạo report
        report_data = {
            "audio_files": audio_files,
            "results": {},
            "total_score": total_result.total_score if total_result else 0.0,
            "question_scores": total_result.question_scores if total_result else {},
            "similarity_scores": total_result.similarity_scores if total_result else {}
        }
        
        for question_id, (transcription, evaluation) in results.items():
            report_data["results"][question_id] = {
                "transcription": transcription,
                "score": evaluation.total_score if evaluation else 0.0,
                "similarity": evaluation.similarity_scores.get(question_id, 0.0) if evaluation else 0.0
            }
        
        # Save JSON report
        report_file = output_path / "total_evaluation_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        
        # Save text report
        if total_result:
            text_report = self.content_evaluator.get_feedback(total_result)
            text_file = output_path / "total_evaluation_report.txt"
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(text_report)
        
        logger.info(f"Saved total results to {output_dir}")


# Hàm tiện ích
def quick_transcribe_and_evaluate(
    audio_file: str,
    question_id: str,
    transcriber=None
) -> Tuple[str, Optional[ContentEvaluationResult]]:
    """
    Hàm tiện ích để transcribe và đánh giá nhanh.
    
    Args:
        audio_file: Đường dẫn file audio
        question_id: ID câu hỏi
        transcriber: Transcriber instance
        
    Returns:
        (transcription, evaluation_result)
    """
    evaluator = IntegratedSpeechEvaluator(transcriber=transcriber)
    return evaluator.transcribe_and_evaluate(audio_file, question_id)
