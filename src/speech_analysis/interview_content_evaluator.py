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
    
    def _check_answer_length(self, answer: str, min_length: int = 20, max_length: int = 2000) -> Tuple[bool, str]:
        """
        Kiểm tra độ dài câu trả lời (fail-safe cho ASR).
        
        Args:
            answer: Câu trả lời
            min_length: Độ dài tối thiểu (characters)
            max_length: Độ dài tối đa (characters)
            
        Returns:
            (is_valid, message)
        """
        length = len(answer.strip())
        
        if length < min_length:
            return False, f"Câu trả lời quá ngắn ({length} ký tự, tối thiểu {min_length})"
        
        if length > max_length:
            return False, f"Câu trả lời quá dài ({length} ký tự, tối đa {max_length})"
        
        return True, "OK"
    
    def _check_coverage(self, answer: str, required_keywords: List[List[str]]) -> Tuple[float, List[str]]:
        """
        Kiểm tra coverage - câu trả lời có đủ ý không.
        
        Args:
            answer: Câu trả lời
            required_keywords: List of keyword groups (phải có ít nhất 1 từ trong mỗi group)
            
        Returns:
            (coverage_score, missing_groups)
            - coverage_score: 0.0-1.0 (tỷ lệ groups được cover)
            - missing_groups: Danh sách các groups bị thiếu
        """
        answer_lower = answer.lower()
        covered_groups = 0
        missing_groups = []
        
        for i, keyword_group in enumerate(required_keywords):
            # Kiểm tra xem có ít nhất 1 keyword trong group xuất hiện không
            found = any(keyword.lower() in answer_lower for keyword in keyword_group)
            
            if found:
                covered_groups += 1
            else:
                missing_groups.append(f"Group {i+1}: {', '.join(keyword_group[:3])}")
        
        coverage_score = covered_groups / len(required_keywords) if required_keywords else 1.0
        
        return coverage_score, missing_groups
    
    def _apply_coverage_penalty(self, base_score: float, coverage_score: float, penalty: float = 0.1) -> float:
        """
        Áp dụng penalty nếu thiếu coverage.
        
        Args:
            base_score: Điểm gốc
            coverage_score: Tỷ lệ coverage (0-1)
            penalty: Penalty cho mỗi group bị thiếu
            
        Returns:
            Điểm sau khi trừ penalty
        """
        # Số groups bị thiếu
        missing_ratio = 1.0 - coverage_score
        
        # Trừ điểm
        penalty_amount = missing_ratio * penalty * 10  # penalty * 10 vì thang điểm 0-10
        
        final_score = max(0.0, base_score - penalty_amount)
        
        return final_score
    
    def _similarity_to_score(self, similarity: float, smooth: bool = True) -> float:
        """
        Chuyển đổi cosine similarity sang thang điểm 0-10 với SMOOTH INTERPOLATION.
        
        Args:
            similarity: Cosine similarity (0-1)
            smooth: Dùng nội suy mượt (True) hay nhảy cứng (False)
            
        Returns:
            Score (0-10)
        """
        if not smooth:
            # Old method - nhảy cứng
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
        
        # NEW METHOD - Smooth interpolation (nội suy tuyến tính)
        # Công thức: score = min_score + (similarity - min_sim) / (max_sim - min_sim) * (max_score - min_score)
        
        if similarity >= 0.85:
            # Range 1: 0.85-1.0 → 9.0-10.0
            return 9.0 + (similarity - 0.85) / (1.0 - 0.85) * (10.0 - 9.0)
        
        elif similarity >= 0.75:
            # Range 2: 0.75-0.85 → 7.5-9.0
            return 7.5 + (similarity - 0.75) / (0.85 - 0.75) * (9.0 - 7.5)
        
        elif similarity >= 0.65:
            # Range 3: 0.65-0.75 → 6.0-7.5
            return 6.0 + (similarity - 0.65) / (0.75 - 0.65) * (7.5 - 6.0)
        
        elif similarity >= 0.50:
            # Range 4: 0.50-0.65 → 4.0-6.0
            return 4.0 + (similarity - 0.50) / (0.65 - 0.50) * (6.0 - 4.0)
        
        elif similarity >= 0.30:
            # Range 5: 0.30-0.50 → 2.0-4.0
            return 2.0 + (similarity - 0.30) / (0.50 - 0.30) * (4.0 - 2.0)
        
        else:
            # Range 6: 0.0-0.30 → 0.0-2.0
            return 0.0 + (similarity - 0.0) / (0.30 - 0.0) * (2.0 - 0.0)
    
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
        questions: Optional[List[InterviewQuestion]] = None,
        scoring_method: str = "best_match",  # ✅ Changed default to best_match
        check_coverage: bool = True,
        coverage_penalty: float = 0.1
    ) -> Tuple[float, float, str, Dict]:
        """
        Đánh giá một câu trả lời với nhiều sample answers - VERSION 3.0.
        
        ✅ CẢI TIẾN MỚI:
        - Dùng MAX similarity thay vì average (best_match)
        - Smooth interpolation (nội suy mượt)
        - Kiểm tra coverage (đủ ý)
        - Fail-safe cho ASR (độ dài câu trả lời)
        
        Args:
            question_id: ID câu hỏi (Q1, Q2, ...)
            applicant_answer: Câu trả lời của ứng viên
            questions: Danh sách câu hỏi (mặc định dùng STANDARD_QUESTIONS)
            scoring_method: Phương pháp tính điểm:
                - "best_match": ✅ Lấy MAX similarity (KHUYẾN NGHỊ)
                - "average": Trung bình similarity
                - "weighted_average": Trung bình có trọng số
            check_coverage: Kiểm tra coverage (đủ ý)
            coverage_penalty: Penalty nếu thiếu ý (0.1 = trừ 1 điểm nếu thiếu 100%)
            
        Returns:
            (score, similarity, best_match_answer, details_dict)
            - score: Điểm 0-10 (sau khi áp dụng penalties)
            - similarity: Độ tương đồng (0-1)
            - best_match_answer: Câu trả lời mẫu giống nhất
            - details_dict: Chi tiết đánh giá
        """
        if questions is None:
            questions = self.STANDARD_QUESTIONS
        
        # Tìm câu hỏi
        question = next((q for q in questions if q.id == question_id), None)
        if question is None:
            logger.warning(f"Question {question_id} not found")
            return 0.0, 0.0, "", {}
        
        # ✅ BƯỚC 1: Kiểm tra độ dài (fail-safe cho ASR)
        is_valid_length, length_message = self._check_answer_length(applicant_answer, min_length=20, max_length=2000)
        if not is_valid_length:
            logger.warning(f"{question_id} - {length_message}")
            # Giới hạn điểm tối đa nếu câu trả lời quá ngắn/dài
            max_score_limit = 3.0 if len(applicant_answer.strip()) < 20 else 10.0
        else:
            max_score_limit = 10.0
        
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
            similarities.append(float(sim))  # Convert to Python float
        
        # Tìm best match
        best_similarity = max(similarities)
        best_idx = similarities.index(best_similarity)
        best_match = question.sample_answers[best_idx]
        
        # Tính điểm theo phương pháp được chọn
        if scoring_method == "best_match":
            # ✅ PHƯƠNG PHÁP 1: LẤY MAX SIMILARITY (KHUYẾN NGHỊ)
            # Ứng viên chỉ cần đúng 1 hướng là đạt
            final_similarity = best_similarity
            score = self._similarity_to_score(final_similarity, smooth=True)
            method_used = "best_match (MAX)"
            
        elif scoring_method == "weighted_best_match":
            # ✅ PHƯƠNG PHÁP 2: MAX(similarity × weight)
            # Lấy similarity cao nhất sau khi nhân với trọng số
            # TODO: Cần có trọng số từ config
            weighted_similarities = similarities  # Placeholder
            final_similarity = max(weighted_similarities)
            score = self._similarity_to_score(final_similarity, smooth=True)
            method_used = "weighted_best_match"
            
        elif scoring_method == "average":
            # Phương pháp 3: Trung bình tất cả similarities (không khuyến nghị)
            final_similarity = sum(similarities) / len(similarities)
            score = self._similarity_to_score(final_similarity, smooth=True)
            method_used = "average"
            
        elif scoring_method == "weighted_average":
            # Phương pháp 4: Trung bình có trọng số
            # TODO: Implement weighted average nếu có trọng số trong config
            final_similarity = sum(similarities) / len(similarities)
            score = self._similarity_to_score(final_similarity, smooth=True)
            method_used = "weighted_average (fallback to average)"
            
        else:
            # Mặc định: best_match (theo yêu cầu mới)
            final_similarity = best_similarity
            score = self._similarity_to_score(final_similarity, smooth=True)
            method_used = "best_match (default)"
        
        # ✅ BƯỚC 5: Kiểm tra coverage (đủ ý)
        coverage_score = 1.0
        missing_keywords = []
        
        if check_coverage and hasattr(question, 'required_keywords'):
            coverage_score, missing_keywords = self._check_coverage(
                applicant_answer,
                question.required_keywords if hasattr(question, 'required_keywords') else []
            )
            
            if coverage_score < 1.0:
                logger.info(f"{question_id} - Coverage: {coverage_score:.2%}, Missing: {missing_keywords}")
                # Áp dụng penalty
                score_before_penalty = score
                score = self._apply_coverage_penalty(score, coverage_score, coverage_penalty)
                logger.info(f"{question_id} - Score after coverage penalty: {score_before_penalty:.1f} → {score:.1f}")
        
        # ✅ BƯỚC 6: Áp dụng giới hạn điểm (nếu câu trả lời quá ngắn)
        if score > max_score_limit:
            logger.info(f"{question_id} - Score limited: {score:.1f} → {max_score_limit:.1f} (answer too short)")
            score = max_score_limit
        
        # Chi tiết
        details = {
            "method": method_used,
            "num_samples": len(similarities),
            "all_similarities": similarities,
            "best_similarity": best_similarity,
            "best_match_index": best_idx,
            "average_similarity": sum(similarities) / len(similarities),
            "min_similarity": min(similarities),
            "max_similarity": max(similarities),
            "final_similarity": final_similarity,
            "base_score": score,  # Điểm trước khi áp dụng penalties
            "coverage_score": coverage_score,
            "missing_keywords": missing_keywords,
            "length_check": {
                "is_valid": is_valid_length,
                "message": length_message,
                "length": len(applicant_answer.strip())
            },
            "max_score_limit": max_score_limit,
            "final_score": score,
            "smooth_interpolation": True
        }
        
        logger.info(
            f"{question_id} - Method: {method_used}, "
            f"Similarities: {[f'{s:.3f}' for s in similarities]}, "
            f"Final: {final_similarity:.3f}, "
            f"Coverage: {coverage_score:.2%}, "
            f"Score: {score:.1f}/10"
        )
        
        return score, final_similarity, best_match, details

    def evaluate_all_answers(
        self,
        answers: Dict[str, str],
        questions: Optional[List[InterviewQuestion]] = None,
        scoring_method: str = "best_match"  # ✅ Changed default to best_match
    ) -> ContentEvaluationResult:
        """
        Đánh giá tất cả câu trả lời và tính tổng điểm có trọng số.
        
        Args:
            answers: Dict {question_id: applicant_answer}
            questions: Danh sách câu hỏi (mặc định dùng STANDARD_QUESTIONS)
            scoring_method: Phương pháp tính điểm ("best_match", "average", "weighted_average")
            
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
        evaluation_details = {}
        
        # Đánh giá từng câu
        for question_id, answer in answers.items():
            score, similarity, best_match, details = self.evaluate_answer(
                question_id, answer, questions, scoring_method
            )
            question_scores[question_id] = score
            similarity_scores[question_id] = similarity
            best_matches[question_id] = best_match
            evaluation_details[question_id] = details
        
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
            "scoring_method": scoring_method,
            "questions": {
                q.id: {
                    "question": q.question,
                    "category": q.category,
                    "weight": q.weight,
                    "score": question_scores.get(q.id, 0.0),
                    "similarity": similarity_scores.get(q.id, 0.0),
                    "applicant_answer": answers.get(q.id, ""),
                    "best_match": best_matches.get(q.id, ""),
                    "evaluation_details": evaluation_details.get(q.id, {})
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
        
        logger.info(
            f"Total Score: {total_score:.2f}/10 "
            f"(Method: {scoring_method}, Questions: {len(answers)})"
        )
        
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
