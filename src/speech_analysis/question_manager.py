"""
Question Manager

Quản lý câu hỏi phỏng vấn - cho phép chọn câu hỏi và đánh giá câu trả lời.
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Question:
    """Câu hỏi phỏng vấn."""
    id: str
    question: str
    category: str
    weight: float
    sample_answers: List[str]
    description: Optional[str] = None
    tips: Optional[List[str]] = None


class QuestionManager:
    """
    Quản lý câu hỏi phỏng vấn.
    Cho phép chọn câu hỏi và đánh giá câu trả lời.
    """
    
    # Bộ câu hỏi mặc định
    DEFAULT_QUESTIONS = [
        Question(
            id="Q1",
            question="Hãy kể về một lần bạn giải quyết một vấn đề khó trong công việc.",
            category="Problem Solving",
            weight=0.35,
            sample_answers=[
                "Tôi gặp một lỗi nghiêm trọng trong hệ thống CRM khiến dữ liệu khách hàng không đồng bộ. "
                "Tôi phân tích log, tái hiện lỗi ở môi trường staging, xác định nguyên nhân là do conflict ở API version. "
                "Sau đó tôi đề xuất cập nhật version và bổ sung kiểm tra hợp lệ. "
                "Kết quả hệ thống ổn định lại trong vòng 2 giờ.",
                
                "Nhóm tôi bị trễ deadline dự án 3 ngày do yêu cầu thay đổi. "
                "Tôi tách task, phân ưu tiên theo mức ảnh hưởng và trao đổi lại với khách hàng để chốt phạm vi mới. "
                "Nhờ đó team hoàn thành đúng hạn còn lại và khách hàng đánh giá cao khả năng xử lý."
            ],
            description="Đánh giá khả năng phân tích và giải quyết vấn đề",
            tips=[
                "Mô tả vấn đề cụ thể",
                "Giải thích cách bạn phân tích",
                "Nêu giải pháp và kết quả"
            ]
        ),
        Question(
            id="Q2",
            question="Bạn đã từng phải làm việc dưới áp lực deadline chưa? Bạn xử lý thế nào?",
            category="Deadline Management",
            weight=0.20,
            sample_answers=[
                "Tôi từng phải hoàn thành báo cáo tài chính trong 24 giờ. "
                "Tôi lập danh sách ưu tiên, gom dữ liệu tự động bằng script, và tập trung vào các mục quan trọng nhất. "
                "Kết quả hoàn thành trước hạn 3 giờ.",
                
                "Khi đối mặt với deadline gấp, tôi chia nhỏ công việc thành các task cụ thể, "
                "ưu tiên những việc quan trọng nhất trước, và loại bỏ những việc không cần thiết. "
                "Tôi cũng thông báo rõ ràng với team về tiến độ để mọi người hỗ trợ kịp thời."
            ],
            description="Đánh giá khả năng quản lý thời gian và làm việc dưới áp lực",
            tips=[
                "Mô tả tình huống deadline",
                "Giải thích cách ưu tiên công việc",
                "Nêu kết quả đạt được"
            ]
        ),
        Question(
            id="Q3",
            question="Kể về một lần bạn làm việc nhóm hiệu quả và vai trò của bạn.",
            category="Teamwork",
            weight=0.20,
            sample_answers=[
                "Trong dự án website, tôi đóng vai trò kết nối giữa dev và designer. "
                "Tôi tạo quy tắc giao tiếp, cập nhật tiến độ mỗi ngày và xử lý các vướng mắc. "
                "Nhờ đó team hoàn thành đúng deadline và giảm 40% lỗi giao diện.",
                
                "Tôi làm việc trong nhóm 5 người phát triển ứng dụng mobile. "
                "Vai trò của tôi là điều phối và đảm bảo code quality. "
                "Tôi tổ chức daily standup, code review, và giải quyết conflict kỹ thuật. "
                "Dự án hoàn thành đúng hạn với chất lượng cao."
            ],
            description="Đánh giá khả năng làm việc nhóm và phối hợp",
            tips=[
                "Mô tả dự án và team",
                "Giải thích vai trò của bạn",
                "Nêu đóng góp và kết quả"
            ]
        ),
        Question(
            id="Q4",
            question="Hãy kể về một tình huống bạn phải thuyết phục người khác.",
            category="Communication",
            weight=0.15,
            sample_answers=[
                "Tôi phải thuyết phục khách hàng đồng ý thay đổi kiến trúc hệ thống. "
                "Tôi chuẩn bị tài liệu so sánh, đưa ra rủi ro và ROI cụ thể. "
                "Khách hàng đồng ý và dự án chạy ổn định hơn 30%.",
                
                "Tôi thuyết phục ban lãnh đạo đầu tư vào công nghệ mới. "
                "Tôi trình bày phân tích chi phí-lợi ích, demo sản phẩm, và so sánh với đối thủ. "
                "Cuối cùng công ty chấp thuận và năng suất tăng 25%."
            ],
            description="Đánh giá kỹ năng giao tiếp và thuyết phục",
            tips=[
                "Mô tả tình huống cần thuyết phục",
                "Giải thích cách bạn trình bày",
                "Nêu kết quả thuyết phục"
            ]
        ),
        Question(
            id="Q5",
            question="Thành tựu khiến bạn tự hào nhất là gì và vì sao?",
            category="Achievement",
            weight=0.10,
            sample_answers=[
                "Thành tựu tôi tự hào nhất là cải tiến quy trình code review giúp giảm thời gian review 50% "
                "và tăng chất lượng code. Giải pháp này sau đó được áp dụng toàn công ty.",
                
                "Tôi tự hào khi phát triển hệ thống tự động hóa giúp công ty tiết kiệm 100 giờ làm việc mỗi tháng. "
                "Hệ thống này được đánh giá cao và tôi nhận được giải thưởng nhân viên xuất sắc."
            ],
            description="Đánh giá thành tựu và động lực cá nhân",
            tips=[
                "Mô tả thành tựu cụ thể",
                "Giải thích tại sao tự hào",
                "Nêu tác động của thành tựu"
            ]
        )
    ]
    
    def __init__(self, custom_questions: Optional[List[Question]] = None):
        """
        Khởi tạo Question Manager.
        
        Args:
            custom_questions: Danh sách câu hỏi tùy chỉnh (nếu không có sẽ dùng DEFAULT_QUESTIONS)
        """
        self.questions = custom_questions if custom_questions else self.DEFAULT_QUESTIONS.copy()
        self.current_question = None
        self.question_history = []  # Lịch sử câu hỏi đã trả lời
        
        logger.info(f"Initialized QuestionManager with {len(self.questions)} questions")
    
    def get_all_questions(self) -> List[Question]:
        """Lấy tất cả câu hỏi."""
        return self.questions
    
    def get_question_by_id(self, question_id: str) -> Optional[Question]:
        """
        Lấy câu hỏi theo ID.
        
        Args:
            question_id: ID câu hỏi (Q1, Q2, ...)
            
        Returns:
            Question object hoặc None
        """
        for q in self.questions:
            if q.id == question_id:
                return q
        return None
    
    def get_questions_by_category(self, category: str) -> List[Question]:
        """
        Lấy câu hỏi theo danh mục.
        
        Args:
            category: Tên danh mục
            
        Returns:
            Danh sách câu hỏi
        """
        return [q for q in self.questions if q.category == category]
    
    def select_question(self, question_id: str) -> Optional[Question]:
        """
        Chọn câu hỏi để trả lời.
        
        Args:
            question_id: ID câu hỏi
            
        Returns:
            Question object hoặc None
        """
        question = self.get_question_by_id(question_id)
        if question:
            self.current_question = question
            logger.info(f"Selected question: {question_id} - {question.question}")
            return question
        else:
            logger.warning(f"Question {question_id} not found")
            return None
    
    def get_current_question(self) -> Optional[Question]:
        """Lấy câu hỏi hiện tại đang được chọn."""
        return self.current_question
    
    def add_to_history(self, question_id: str, answer: str, score: float, similarity: float):
        """
        Thêm câu hỏi đã trả lời vào lịch sử.
        
        Args:
            question_id: ID câu hỏi
            answer: Câu trả lời
            score: Điểm đạt được
            similarity: Độ tương đồng
        """
        self.question_history.append({
            "question_id": question_id,
            "answer": answer,
            "score": score,
            "similarity": similarity,
            "timestamp": None  # Có thể thêm timestamp nếu cần
        })
        logger.info(f"Added to history: {question_id} - Score: {score:.1f}")
    
    def get_history(self) -> List[Dict]:
        """Lấy lịch sử câu hỏi đã trả lời."""
        return self.question_history
    
    def clear_history(self):
        """Xóa lịch sử."""
        self.question_history = []
        logger.info("Cleared question history")
    
    def get_unanswered_questions(self) -> List[Question]:
        """Lấy danh sách câu hỏi chưa trả lời."""
        answered_ids = {h["question_id"] for h in self.question_history}
        return [q for q in self.questions if q.id not in answered_ids]
    
    def get_progress(self) -> Tuple[int, int]:
        """
        Lấy tiến độ trả lời câu hỏi.
        
        Returns:
            (số câu đã trả lời, tổng số câu)
        """
        return len(self.question_history), len(self.questions)
    
    def export_questions(self, filepath: str):
        """
        Export câu hỏi ra file JSON.
        
        Args:
            filepath: Đường dẫn file
        """
        data = []
        for q in self.questions:
            data.append({
                "id": q.id,
                "question": q.question,
                "category": q.category,
                "weight": q.weight,
                "sample_answers": q.sample_answers,
                "description": q.description,
                "tips": q.tips
            })
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Exported {len(data)} questions to {filepath}")
    
    def import_questions(self, filepath: str):
        """
        Import câu hỏi từ file JSON.
        
        Args:
            filepath: Đường dẫn file
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.questions = []
        for item in data:
            question = Question(
                id=item["id"],
                question=item["question"],
                category=item["category"],
                weight=item["weight"],
                sample_answers=item["sample_answers"],
                description=item.get("description"),
                tips=item.get("tips")
            )
            self.questions.append(question)
        
        logger.info(f"Imported {len(self.questions)} questions from {filepath}")
    
    def get_question_summary(self) -> str:
        """
        Tạo tóm tắt các câu hỏi.
        
        Returns:
            String tóm tắt
        """
        lines = ["=== DANH SÁCH CÂU HỎI PHỎNG VẤN ===\n"]
        
        for q in self.questions:
            lines.append(f"{q.id} - {q.category} (Trọng số: {q.weight*100:.0f}%)")
            lines.append(f"Câu hỏi: {q.question}")
            if q.description:
                lines.append(f"Mô tả: {q.description}")
            lines.append("")
        
        return "\n".join(lines)
    
    def get_question_tips(self, question_id: str) -> Optional[List[str]]:
        """
        Lấy tips cho câu hỏi.
        
        Args:
            question_id: ID câu hỏi
            
        Returns:
            Danh sách tips hoặc None
        """
        question = self.get_question_by_id(question_id)
        return question.tips if question else None


# Hàm tiện ích
def create_default_question_manager() -> QuestionManager:
    """Tạo QuestionManager với câu hỏi mặc định."""
    return QuestionManager()


def export_default_questions(filepath: str = "config/default_questions.json"):
    """Export câu hỏi mặc định ra file."""
    manager = create_default_question_manager()
    manager.export_questions(filepath)
    print(f"Exported default questions to {filepath}")
