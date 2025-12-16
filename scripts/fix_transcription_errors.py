"""
Script để sửa các lỗi transcription phổ biến trong tiếng Việt.
Sử dụng dictionary để map từ sai → từ đúng.
"""

import re
from typing import Dict, List, Tuple


class VietnameseTranscriptionFixer:
    """Sửa lỗi transcription tiếng Việt."""
    
    def __init__(self):
        """Khởi tạo với dictionary các lỗi phổ biến."""
        # Dictionary: từ sai → từ đúng
        self.word_corrections = {
            # Lỗi phổ biến với "xin"
            "sim chào": "xin chào",
            "sin chào": "xin chào",
            "xim chào": "xin chào",
            "sim": "xin",
            "sin": "xin",
            
            # Lỗi với "cảm ơn"
            "cám ơn": "cảm ơn",
            "cam ơn": "cảm ơn",
            "gam ơn": "cảm ơn",
            
            # Lỗi với "tôi"
            "toy": "tôi",
            "toi": "tôi",
            "thôi": "tôi",  # context-dependent
            
            # Lỗi với "là"
            "ra": "là",  # context-dependent
            "lá": "là",
            
            # Lỗi với "không"
            "hông": "không",
            "hong": "không",
            "khong": "không",
            
            # Lỗi với "được"
            "dược": "được",
            "đươc": "được",
            
            # Lỗi với "có"
            "go": "có",
            "gó": "có",
            
            # Lỗi với "gì"
            "zi": "gì",
            "gi": "gì",
            "dì": "gì",
            
            # Lỗi với "vậy"
            "vay": "vậy",
            "váy": "vậy",
            
            # Lỗi với "thế"
            "tế": "thế",
            "thê": "thế",
            
            # Lỗi với "này"
            "nay": "này",
            "nài": "này",
            
            # Lỗi với "đó"
            "dó": "đó",
            "đô": "đó",
            
            # Lỗi với "rồi"
            "zồi": "rồi",
            "dồi": "rồi",
            
            # Lỗi với "biết"
            "biet": "biết",
            "viết": "biết",  # context-dependent
            
            # Lỗi với "làm"
            "ram": "làm",
            "lam": "làm",
            
            # Lỗi với "như"
            "nư": "như",
            "như": "như",
            
            # Lỗi với "thì"
            "ti": "thì",
            "tì": "thì",
            
            # Lỗi với "với"
            "voi": "với",
            "vời": "với",
            
            # Lỗi với "của"
            "cua": "của",
            "gủa": "của",
            
            # Lỗi với "cho"
            "co": "cho",
            "chô": "cho",
            
            # Lỗi với "đi"
            "di": "đi",
            "dì": "đi",
            
            # Lỗi với "về"
            "vê": "về",
            "ve": "về",
            
            # Lỗi với "đến"
            "den": "đến",
            "đen": "đến",
            
            # Lỗi với "hay"
            "hai": "hay",  # context-dependent
            "hây": "hay",
            
            # Lỗi với "nhưng"
            "nưng": "nhưng",
            "nhung": "nhưng",
            
            # Lỗi với "hoặc"
            "hoac": "hoặc",
            "hoạc": "hoặc",
            
            # Lỗi với "nếu"
            "neu": "nếu",
            "nêu": "nếu",
            
            # Lỗi với "khi"
            "ki": "khi",
            "khì": "khi",
            
            # Lỗi với "đã"
            "da": "đã",
            "dã": "đã",
            
            # Lỗi với "sẽ"
            "se": "sẽ",
            "xẽ": "sẽ",
            
            # Lỗi với "phải"
            "fai": "phải",
            "phai": "phải",
            
            # Lỗi với "nữa"
            "nua": "nữa",
            "nừa": "nữa",
            
            # Lỗi với "chỉ"
            "chi": "chỉ",
            "chì": "chỉ",
            
            # Lỗi với "cũng"
            "cung": "cũng",
            "gũng": "cũng",
            
            # Lỗi với "đều"
            "deu": "đều",
            "đêu": "đều",
            
            # Lỗi với "nhiều"
            "nhieu": "nhiều",
            "nhiêu": "nhiều",
            
            # Lỗi với "người"
            "nguoi": "người",
            "ngươi": "người",
            
            # Lỗi với "thời"
            "toi": "thời",  # context-dependent
            "thoi": "thời",
            
            # Lỗi với "việc"
            "viec": "việc",
            "viêc": "việc",
            
            # Lỗi với "nước"
            "nuoc": "nước",
            "nươc": "nước",
            
            # Lỗi với "năm"
            "nam": "năm",
            "nâm": "năm",
            
            # Lỗi với "ngày"
            "ngay": "ngày",
            "ngài": "ngày",
            
            # Lỗi với "tháng"
            "thang": "tháng",
            "thâng": "tháng",
        }
        
        # Phrase corrections (ưu tiên cao hơn word corrections)
        self.phrase_corrections = {
            "sim chào": "xin chào",
            "sin chào": "xin chào",
            "xim chào": "xin chào",
            "cám ơn": "cảm ơn",
            "cam ơn": "cảm ơn",
            "gam ơn": "cảm ơn",
            "xin cám ơn": "xin cảm ơn",
            "xin cam ơn": "xin cảm ơn",
            "toy là": "tôi là",
            "toi là": "tôi là",
            "toy tên": "tôi tên",
            "toi tên": "tôi tên",
            "hông biết": "không biết",
            "hong biết": "không biết",
            "hông có": "không có",
            "hong có": "không có",
            "ra sao": "như thế nào",  # idiom
            "toy muốn": "tôi muốn",
            "toi muốn": "tôi muốn",
        }
    
    def fix_text(self, text: str) -> str:
        """
        Sửa lỗi trong text.
        
        Args:
            text: Text cần sửa
            
        Returns:
            Text đã được sửa
        """
        if not text:
            return text
        
        # Lowercase để so sánh (giữ nguyên case gốc)
        result = text
        
        # 1. Sửa phrases trước (ưu tiên cao hơn)
        for wrong_phrase, correct_phrase in self.phrase_corrections.items():
            # Case-insensitive replacement
            pattern = re.compile(re.escape(wrong_phrase), re.IGNORECASE)
            result = pattern.sub(correct_phrase, result)
        
        # 2. Sửa từng từ
        words = result.split()
        fixed_words = []
        
        for word in words:
            # Remove punctuation for comparison
            word_clean = re.sub(r'[^\w\s]', '', word.lower())
            
            # Check if word needs correction
            if word_clean in self.word_corrections:
                # Preserve punctuation
                punctuation = re.findall(r'[^\w\s]', word)
                fixed_word = self.word_corrections[word_clean]
                
                # Add punctuation back
                if punctuation:
                    fixed_word = fixed_word + ''.join(punctuation)
                
                fixed_words.append(fixed_word)
            else:
                fixed_words.append(word)
        
        result = ' '.join(fixed_words)
        
        return result
    
    def add_correction(self, wrong: str, correct: str, is_phrase: bool = False):
        """
        Thêm correction mới vào dictionary.
        
        Args:
            wrong: Từ/cụm từ sai
            correct: Từ/cụm từ đúng
            is_phrase: True nếu là cụm từ, False nếu là từ đơn
        """
        if is_phrase:
            self.phrase_corrections[wrong.lower()] = correct
        else:
            self.word_corrections[wrong.lower()] = correct
    
    def remove_correction(self, wrong: str, is_phrase: bool = False):
        """
        Xóa correction khỏi dictionary.
        
        Args:
            wrong: Từ/cụm từ sai cần xóa
            is_phrase: True nếu là cụm từ, False nếu là từ đơn
        """
        if is_phrase:
            self.phrase_corrections.pop(wrong.lower(), None)
        else:
            self.word_corrections.pop(wrong.lower(), None)
    
    def get_all_corrections(self) -> Dict[str, List[Tuple[str, str]]]:
        """
        Lấy tất cả corrections.
        
        Returns:
            Dictionary với keys 'words' và 'phrases'
        """
        return {
            'words': list(self.word_corrections.items()),
            'phrases': list(self.phrase_corrections.items()),
        }


def main():
    """Demo script."""
    fixer = VietnameseTranscriptionFixer()
    
    # Test cases
    test_texts = [
        "sim chào",
        "Sim chào, toy là Minh",
        "cám ơn bạn rất nhiều",
        "toy hông biết làm gì",
        "Xin cam ơn các bạn",
        "toi muốn di về nhà",
        "Hông có gì đâu",
        "ra sao vậy bạn",
    ]
    
    print("=" * 60)
    print("VIETNAMESE TRANSCRIPTION ERROR FIXER")
    print("=" * 60)
    print()
    
    for text in test_texts:
        fixed = fixer.fix_text(text)
        if text != fixed:
            print(f"❌ Sai:  {text}")
            print(f"✅ Đúng: {fixed}")
            print()
    
    # Show statistics
    corrections = fixer.get_all_corrections()
    print("=" * 60)
    print(f"Tổng số word corrections: {len(corrections['words'])}")
    print(f"Tổng số phrase corrections: {len(corrections['phrases'])}")
    print("=" * 60)


if __name__ == "__main__":
    main()
