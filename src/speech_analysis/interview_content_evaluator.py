"""
Interview Content Evaluator

Đánh giá nội dung câu trả lời phỏng vấn bằng semantic embedding.
Sử dụng sentence-transformers để tính cosine similarity với câu mẫu.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class InterviewQuestion:
    """Câu hỏi phỏng vấn với các câu trả lời mẫu."""
    id: str
    question: str
    sample_answers: List[str]
    weight: float
    category: str


@dataclass
class ContentEvaluationResult:
    """Kết quả đánh giá nội dung."""
    total_score: float  # 0-10
    question_scores: Dict[str, float]  # {question_id: score}
    similarity_scores: Dict[str, float]  # {question_id: similarity}
    best_matches: Dict[str, str]  # {question_id: best_sample_answer}
    details: Dict[str, any]  # Thông tin chi tiết


class InterviewContentEvaluator:
    """
    Đánh giá nội dung câu trả lời phỏng vấn bằng semantic embedding.
    """
    
    # Bộ câu hỏi phỏng vấn chuẩn quốc tế
    STANDARD_QUESTIONS = [
        InterviewQuestion(
            id="Q1",
            question="Hãy kể về một lần bạn giải quyết một vấn đề khó trong công việc.",
            sample_answers=[
                "Tôi gặp một lỗi nghiêm trọng trong hệ thống CRM khiến dữ liệu khách hàng không đồng bộ. "
                "Tôi phân tích log, tái hiện lỗi ở môi trường staging, xác định nguyên nhân là do conflict ở API version. "
                "Sau đó tôi đề xuất cập nhật version và bổ sung kiểm tra hợp lệ. "
                "Kết quả hệ thống ổn định lại trong vòng 2 giờ.",
                
                "Nhóm tôi bị trễ deadline dự án 3 ngày do yêu cầu thay đổi. "
                "Tôi tách task, phân ưu tiên theo mức ảnh hưởng và trao đổi lại với khách hàng để chốt phạm vi mới. "
                "Nhờ đó team hoàn thành đúng hạn còn lại và khách hàng đánh giá cao khả năng xử lý."
            ],
            weight=0.35,
            category="Problem Solving"
        ),
        InterviewQuestion(
            id="Q2",
            question="Bạn đã từng phải làm việc dưới áp lực deadline chưa? Bạn xử lý thế nào?",
            sample_answers=[
                "Tôi từng phải hoàn thành báo cáo tài chính trong 24 giờ. "
                "Tôi lập danh sách ưu tiên, gom dữ liệu tự động bằng script, và tập trung vào các mục quan trọng nhất. "
                "Kết quả hoàn thành trước hạn 3 giờ.",
                
                "Khi đối mặt với deadline gấp, tôi chia nhỏ công việc thành các task cụ thể, "
                "ưu tiên những việc quan trọng nhất trước, và loại bỏ những việc không cần thiết. "
                "Tôi cũng thông báo rõ ràng với team về tiến độ để mọi người hỗ trợ kịp thời."
            ],
            weight=0.20,
            category="Deadline Management"
        ),
        InterviewQuestion(
            id="Q3",
            question="Kể về một lần bạn làm việc nhóm hiệu quả và vai trò của bạn.",
            sample_answers=[
                "Trong dự án website, tôi đóng vai trò kết nối giữa dev và designer. "
                "Tôi tạo quy tắc giao tiếp, cập nhật tiến độ mỗi ngày và xử lý các vướng mắc. "
                "Nhờ đó team hoàn thành đúng deadline và giảm 40% lỗi giao diện.",
                
                "Tôi làm việc trong nhóm 5 người phát triển ứng dụng mobile. "
                "Vai trò của tôi là điều phối và đảm bảo code quality. "
                "Tôi tổ chức daily standup, code review, và giải quyết conflict kỹ thuật. "
                "Dự án hoàn thành đúng hạn với chất lượng cao."
            ],
            weight=0.20,
            category="Teamwork"
        ),
        InterviewQuestion(
            id="Q4",
            question="Hãy kể về một tình huống bạn phải thuyết phục người khác.",
            sample_answers=[
                "Tôi phải thuyết phục khách hàng đồng ý thay đổi kiến trúc hệ thống. "
                "Tôi chuẩn bị tài liệu so sánh, đưa ra rủi ro và ROI cụ thể. "
                "Khách hàng đồng ý và dự án chạy ổn định hơn 30%.",
                
                "Tôi thuyết phục ban lãnh đạo đầu tư vào công nghệ mới. "
                "Tôi trình bày phân tích chi phí-lợi ích, demo sản phẩm, và so sánh với đối thủ. "
                "Cuối cùng công ty chấp thuận và năng suất tăng 25%."
            ],
            weight=0.15,
            category="Communication"
        ),
        InterviewQuestion(
            id="Q5",
            question="Thành tựu khiến bạn tự hào nhất là gì và vì sao?",
            sample_answers=[
                "Thành tựu tôi tự hào nhất là cải tiến quy trình code review giúp giảm thời gian review 50% "
                "và tăng chất lượng code. Giải pháp này sau đó được áp dụng toàn công ty.",
                
                "Tôi tự hào khi phát triển hệ thống tự động hóa giúp công ty tiết kiệm 100 giờ làm việc mỗi tháng. "
                "Hệ thống này được đánh giá cao và tôi nhận được giải thưởng nhân viên xuất sắc."
            ],
            weight=0.10,
            category="Achievement"
        )
    ]
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Khởi tạo evaluator.
        
        Args:
            model_name: Tên model sentence-transformers (hỗ trợ tiếng Việt)
        """
        self.model_name = model_name
        self.model = None
        self.sample_embeddings = {}
        
        logger.info(f"Initializing InterviewContentEvaluator with model: {model_name}")
    
    def _load_model(self):
        """Load sentence-transformers model."""
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name)
                logger.info(f"Loaded model: {self.model_name}")
            except ImportError:
                logger.error("sentence-transformers not installed. Install: pip install sentence-transformers")
                raise
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                raise
    
    def _compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Tính embedding cho danh sách text.
        
        Args:
            texts: Danh sách câu cần tính embedding
            
        Returns:
            Array embeddings shape (n, embedding_dim)
        """
        self._load_model()
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Tính cosine similarity giữa 2 vector.
        
        Args:
            vec1: Vector 1
            vec2: Vector 2
            
        Returns:
            Cosine similarity (0-1)
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _similarity_to_score(self, similarity: float) -> float:
        """
        Chuyển đổi cosine similarity sang thang điểm 0-10.
        
        Args:
            similarity: Cosine similarity (0-1)
            
        Returns:
            Score (0-10)
        """
        if similarity >= 0.85:
            return 10.0
        elif similarity >= 0.75:
            return 8.0
        elif similarity >= 0.65:
            return 6.0
        elif similarity >= 0.50:
            return 4.0
        else:
            return 2.0
    
    def prepare_sample_embeddings(self, questions: Optional[List[InterviewQuestion]] = None):
        """
        Tính trước embeddings cho các câu trả lời mẫu.
        
        Args:
            questions: Danh sách câu hỏi (mặc định dùng STANDARD_QUESTIONS)
        """
        if questions is None:
            questions = self.STANDARD_QUESTIONS
        
        logger.info("Computing embeddings for sample answers...")
        
        for question in questions:
            embeddings = self._compute_embeddings(question.sample_answers)
            self.sample_embeddings[question.id] = embeddings
        
        logger.info(f"Prepared embeddings for {len(questions)} questions")
    
    def evaluate_answer(
        self,
        question_id: str,
        applicant_answer: str,
        questions: Optional[List[InterviewQuestion]] = None
    ) -> Tuple[float, float, str]:
        """
        Đánh giá một câu trả lời.
        
        Args:
            question_id: ID câu hỏi (Q1, Q2, ...)
            applicant_answer: Câu trả lời của ứng viên
            questions: Danh sách câu hỏi (mặc định dùng STANDARD_QUESTIONS)
            
        Returns:
            (score, similarity, best_match_answer)
        """
        if questions is None:
            questions = self.STANDARD_QUESTIONS
        
        # Tìm câu hỏi
        question = next((q for q in questions if q.id == question_id), None)
        if question is None:
            logger.warning(f"Question {question_id} not found")
            return 0.0, 0.0, ""
        
        # Tính embedding cho câu trả lời ứng viên
        applicant_embedding = self._compute_embeddings([applicant_answer])[0]
        
        # Lấy embeddings mẫu
        if question_id not in self.sample_embeddings:
            self.prepare_sample_embeddings(questions)
        
        sample_embeddings = self.sample_embeddings[question_id]
        
        # Tính similarity với từng câu mẫu
        similarities = []
        for sample_emb in sample_embeddings:
            sim = self._cosine_similarity(applicant_embedding, sample_emb)
            similarities.append(sim)
        
        # Lấy similarity cao nhất
        best_similarity = max(similarities)
        best_idx = similarities.index(best_similarity)
        best_match = question.sample_answers[best_idx]
        
        # Chuyển sang điểm
        score = self._similarity_to_score(best_similarity)
        
        logger.info(f"{question_id} - Similarity: {best_similarity:.3f}, Score: {score:.1f}")
        
        return score, best_similarity, best_match

    def evaluate_all_answers(
        self,
        answers: Dict[str, str],
        questions: Optional[List[InterviewQuestion]] = None
    ) -> ContentEvaluationResult:
        """
        Đánh giá tất cả câu trả lời và tính tổng điểm có trọng số.
        
        Args:
            answers: Dict {question_id: applicant_answer}
            questions: Danh sách câu hỏi (mặc định dùng STANDARD_QUESTIONS)
            
        Returns:
            ContentEvaluationResult
        """
        if questions is None:
            questions = self.STANDARD_QUESTIONS
        
        # Chuẩn bị embeddings nếu chưa có
        if not self.sample_embeddings:
            self.prepare_sample_embeddings(questions)
        
        question_scores = {}
        similarity_scores = {}
        best_matches = {}
        
        # Đánh giá từng câu
        for question_id, answer in answers.items():
            score, similarity, best_match = self.evaluate_answer(
                question_id, answer, questions
            )
            question_scores[question_id] = score
            similarity_scores[question_id] = similarity
            best_matches[question_id] = best_match
        
        # Tính tổng điểm có trọng số
        total_score = 0.0
        total_weight = 0.0
        
        for question in questions:
            if question.id in question_scores:
                total_score += question_scores[question.id] * question.weight
                total_weight += question.weight
        
        # Chuẩn hóa về thang 0-10
        if total_weight > 0:
            total_score = total_score / total_weight
        
        # Tạo details
        details = {
            "questions": {
                q.id: {
                    "question": q.question,
                    "category": q.category,
                    "weight": q.weight,
                    "score": question_scores.get(q.id, 0.0),
                    "similarity": similarity_scores.get(q.id, 0.0),
                    "applicant_answer": answers.get(q.id, ""),
                    "best_match": best_matches.get(q.id, "")
                }
                for q in questions if q.id in answers
            },
            "total_weight": total_weight
        }
        
        result = ContentEvaluationResult(
            total_score=total_score,
            question_scores=question_scores,
            similarity_scores=similarity_scores,
            best_matches=best_matches,
            details=details
        )
        
        logger.info(f"Total Score: {total_score:.2f}/10")
        
        return result
    
    def evaluate_single_question_answer(
        self,
        question_text: str,
        applicant_answer: str,
        sample_answers: List[str]
    ) -> Tuple[float, float]:
        """
        Đánh giá một câu trả lời với câu hỏi tùy chỉnh.
        
        Args:
            question_text: Nội dung câu hỏi
            applicant_answer: Câu trả lời của ứng viên
            sample_answers: Danh sách câu trả lời mẫu
            
        Returns:
            (score, similarity)
        """
        # Tính embedding
        applicant_embedding = self._compute_embeddings([applicant_answer])[0]
        sample_embeddings = self._compute_embeddings(sample_answers)
        
        # Tính similarity
        similarities = []
        for sample_emb in sample_embeddings:
            sim = self._cosine_similarity(applicant_embedding, sample_emb)
            similarities.append(sim)
        
        best_similarity = max(similarities)
        score = self._similarity_to_score(best_similarity)
        
        return score, best_similarity
    
    def get_feedback(self, result: ContentEvaluationResult) -> str:
        """
        Tạo feedback chi tiết cho ứng viên.
        
        Args:
            result: Kết quả đánh giá
            
        Returns:
            Feedback text
        """
        feedback_lines = []
        feedback_lines.append(f"=== KẾT QUẢ ĐÁNH GIÁ NỘI DUNG PHỎNG VẤN ===\n")
        feedback_lines.append(f"Tổng điểm: {result.total_score:.2f}/10\n")
        
        # Đánh giá tổng quan
        if result.total_score >= 8.5:
            feedback_lines.append("Đánh giá: Xuất sắc! Câu trả lời rất chi tiết và chuyên nghiệp.\n")
        elif result.total_score >= 7.0:
            feedback_lines.append("Đánh giá: Tốt! Câu trả lời đạt yêu cầu.\n")
        elif result.total_score >= 5.5:
            feedback_lines.append("Đánh giá: Trung bình. Cần cải thiện độ chi tiết và cấu trúc.\n")
        else:
            feedback_lines.append("Đánh giá: Cần cải thiện nhiều. Câu trả lời chưa đủ thuyết phục.\n")
        
        feedback_lines.append("\n=== CHI TIẾT TỪNG CÂU HỎI ===\n")
        
        for q_id, q_data in result.details["questions"].items():
            feedback_lines.append(f"\n{q_id} - {q_data['category']} (Trọng số: {q_data['weight']*100:.0f}%)")
            feedback_lines.append(f"Câu hỏi: {q_data['question']}")
            feedback_lines.append(f"Điểm: {q_data['score']:.1f}/10 (Similarity: {q_data['similarity']:.3f})")
            
            if q_data['score'] >= 8:
                feedback_lines.append("✓ Câu trả lời tốt!")
            elif q_data['score'] >= 6:
                feedback_lines.append("→ Câu trả lời đạt yêu cầu, có thể cải thiện thêm.")
            else:
                feedback_lines.append("✗ Câu trả lời cần cải thiện đáng kể.")
            
            feedback_lines.append("")
        
        return "\n".join(feedback_lines)


# Hàm tiện ích để sử dụng nhanh
def quick_evaluate(
    answers: Dict[str, str],
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
) -> ContentEvaluationResult:
    """
    Hàm tiện ích để đánh giá nhanh.
    
    Args:
        answers: Dict {question_id: applicant_answer}
        model_name: Tên model
        
    Returns:
        ContentEvaluationResult
    """
    evaluator = InterviewContentEvaluator(model_name=model_name)
    return evaluator.evaluate_all_answers(answers)
