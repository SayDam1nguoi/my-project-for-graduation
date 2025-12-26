"""
Tests for Interview Content Evaluator
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from speech_analysis.interview_content_evaluator import (
    InterviewContentEvaluator,
    InterviewQuestion,
    ContentEvaluationResult,
    quick_evaluate
)


class TestInterviewContentEvaluator:
    """Test cases for InterviewContentEvaluator."""
    
    @pytest.fixture
    def evaluator(self):
        """Create evaluator instance."""
        return InterviewContentEvaluator()
    
    def test_initialization(self, evaluator):
        """Test evaluator initialization."""
        assert evaluator.model_name is not None
        assert evaluator.model is None  # Not loaded yet
        assert evaluator.sample_embeddings == {}
    
    def test_load_model(self, evaluator):
        """Test model loading."""
        evaluator._load_model()
        assert evaluator.model is not None
    
    def test_similarity_to_score(self, evaluator):
        """Test similarity to score conversion."""
        assert evaluator._similarity_to_score(0.90) == 10.0
        assert evaluator._similarity_to_score(0.80) == 8.0
        assert evaluator._similarity_to_score(0.70) == 6.0
        assert evaluator._similarity_to_score(0.55) == 4.0
        assert evaluator._similarity_to_score(0.40) == 2.0
    
    def test_prepare_sample_embeddings(self, evaluator):
        """Test preparing sample embeddings."""
        evaluator.prepare_sample_embeddings()
        
        # Check embeddings are computed
        assert len(evaluator.sample_embeddings) == 5
        assert "Q1" in evaluator.sample_embeddings
        assert "Q5" in evaluator.sample_embeddings
    
    def test_evaluate_answer_good(self, evaluator):
        """Test evaluating a good answer."""
        answer = """
        Tôi gặp một lỗi nghiêm trọng trong hệ thống CRM.
        Tôi phân tích log, tái hiện lỗi ở staging, và xác định nguyên nhân.
        Sau đó tôi đề xuất giải pháp và hệ thống ổn định lại.
        """
        
        score, similarity, best_match, details = evaluator.evaluate_answer("Q1", answer)
        
        assert 0 <= score <= 10
        assert 0 <= similarity <= 1
        assert len(best_match) > 0
        assert score >= 4  # Should be at least average
        assert isinstance(details, dict)  # Check details is returned
    
    def test_evaluate_answer_poor(self, evaluator):
        """Test evaluating a poor answer."""
        answer = "Tôi fix bug."
        
        score, similarity, best_match, details = evaluator.evaluate_answer("Q1", answer)
        
        assert 0 <= score <= 10
        assert 0 <= similarity <= 1
        assert score <= 4  # Should be low
        assert isinstance(details, dict)  # Check details is returned
    
    def test_evaluate_all_answers(self, evaluator):
        """Test evaluating all answers."""
        answers = {
            "Q1": "Tôi giải quyết bug bằng cách debug và fix code.",
            "Q2": "Tôi làm việc chăm chỉ để hoàn thành đúng hạn.",
            "Q3": "Tôi làm việc nhóm tốt và giúp đỡ mọi người."
        }
        
        result = evaluator.evaluate_all_answers(answers)
        
        assert isinstance(result, ContentEvaluationResult)
        assert 0 <= result.total_score <= 10
        assert len(result.question_scores) == 3
        assert "Q1" in result.question_scores
        assert "Q2" in result.question_scores
        assert "Q3" in result.question_scores
    
    def test_evaluate_single_question_answer(self, evaluator):
        """Test evaluating custom question."""
        question = "Bạn xử lý conflict như thế nào?"
        sample_answers = [
            "Tôi lắng nghe cả hai bên và tìm giải pháp win-win.",
            "Tôi tổ chức meeting để thảo luận."
        ]
        applicant_answer = "Tôi nói chuyện với mọi người để hiểu vấn đề."
        
        score, similarity = evaluator.evaluate_single_question_answer(
            question, applicant_answer, sample_answers
        )
        
        assert 0 <= score <= 10
        assert 0 <= similarity <= 1
    
    def test_get_feedback(self, evaluator):
        """Test feedback generation."""
        answers = {
            "Q1": "Tôi giải quyết vấn đề bằng cách phân tích và fix.",
            "Q2": "Tôi làm việc nhanh để hoàn thành đúng hạn."
        }
        
        result = evaluator.evaluate_all_answers(answers)
        feedback = evaluator.get_feedback(result)
        
        assert isinstance(feedback, str)
        assert len(feedback) > 0
        assert "KẾT QUẢ ĐÁNH GIÁ" in feedback
        assert "Tổng điểm" in feedback
    
    def test_weights_sum_to_one(self):
        """Test that question weights sum to 1.0."""
        total_weight = sum(
            q.weight for q in InterviewContentEvaluator.STANDARD_QUESTIONS
        )
        assert abs(total_weight - 1.0) < 0.01
    
    def test_quick_evaluate(self):
        """Test quick evaluate function."""
        answers = {
            "Q1": "Tôi giải quyết bug.",
            "Q2": "Tôi làm nhanh."
        }
        
        result = quick_evaluate(answers)
        
        assert isinstance(result, ContentEvaluationResult)
        assert 0 <= result.total_score <= 10


class TestInterviewQuestion:
    """Test cases for InterviewQuestion dataclass."""
    
    def test_creation(self):
        """Test creating InterviewQuestion."""
        question = InterviewQuestion(
            id="Q1",
            question="Test question?",
            sample_answers=["Answer 1", "Answer 2"],
            weight=0.5,
            category="Test"
        )
        
        assert question.id == "Q1"
        assert question.question == "Test question?"
        assert len(question.sample_answers) == 2
        assert question.weight == 0.5
        assert question.category == "Test"


class TestContentEvaluationResult:
    """Test cases for ContentEvaluationResult dataclass."""
    
    def test_creation(self):
        """Test creating ContentEvaluationResult."""
        result = ContentEvaluationResult(
            total_score=7.5,
            question_scores={"Q1": 8.0, "Q2": 6.0},
            similarity_scores={"Q1": 0.78, "Q2": 0.65},
            best_matches={"Q1": "Sample 1", "Q2": "Sample 2"},
            details={"test": "data"}
        )
        
        assert result.total_score == 7.5
        assert len(result.question_scores) == 2
        assert result.question_scores["Q1"] == 8.0
        assert result.similarity_scores["Q1"] == 0.78


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
