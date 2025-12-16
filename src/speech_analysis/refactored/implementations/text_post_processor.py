"""Text post-processor implementation for Vietnamese transcription correction."""

import re
import unicodedata
from typing import Dict, List, Optional
from ..interfaces.processing import ITextPostProcessor, TextProcessor


class VietnameseDiacriticCorrector(TextProcessor):
    """Corrects Vietnamese diacritic marks."""
    
    def __init__(self):
        super().__init__("vietnamese_diacritic_corrector")
        
        # Common Vietnamese diacritic errors and corrections
        # Map from non-diacritic or incorrect to correct form
        self.word_corrections = {
            # Common words without diacritics -> with diacritics
            "viet": "việt",
            # "nam": "năm",  # Handled by proper noun check for "viet nam"
            "la": "là",
            "mot": "một",
            "nuoc": "nước",
            "dep": "đẹp",
            "duoc": "được",
            "thi": "thì",
            "nay": "này",
            "do": "đó",
            "nhung": "những",
            "cua": "của",
            "ma": "mà",
            "den": "đến",
            "tren": "trên",
            "duoi": "dưới",
            "noi": "nói",
            "lam": "làm",
            "viec": "việc",
            "nguoi": "người",
            "ngay": "ngày",
            "thang": "tháng",
            "hom": "hôm",
            "qua": "qua",
            "nao": "nào",
            "sao": "sao",
            "bao": "bao",
            "gio": "giờ",
            "phut": "phút",
            "giay": "giây",
            "toi": "tôi",
            "sinh": "sinh",
            "ra": "ra",
            "o": "ở",
            "anh": "anh",
            "hung": "hùng",
            "dan": "dân",
            "toc": "tộc",
            "va": "và",
            "hai": "hai",
            "biet": "biết",
            "de": "để",
            "giai": "giải",
            "quyet": "quyết",
            "van": "vấn",
            "tam": "tám",
            "chin": "chín",
            "muoi": "mười",
            
            # Tone mark corrections
            "hoà": "hòa",
            "toà": "tòa",
            "khoá": "khóa",
            "thuỷ": "thủy",
            "tuỳ": "tùy",
            "quỳ": "quý",
        }
        
        # Proper nouns (always capitalize)
        self.proper_nouns = {
            "viet nam": "Việt Nam",
            "ha noi": "Hà Nội",
            "ho chi minh": "Hồ Chí Minh",
            "sai gon": "Sài Gòn",
            "da nang": "Đà Nẵng",
        }
    
    def process(self, text: str) -> str:
        """Apply Vietnamese diacritic corrections."""
        result = text.lower()  # Normalize to lowercase first
        
        # First apply proper noun corrections (before word-level to avoid conflicts)
        for incorrect, correct in self.proper_nouns.items():
            pattern = re.compile(r'\b' + re.escape(incorrect) + r'\b', re.IGNORECASE)
            result = pattern.sub(correct, result)
        
        # Then apply word-level corrections (skip words already corrected by proper nouns)
        words = result.split()
        corrected_words = []
        
        for word in words:
            # Remove punctuation for matching
            clean_word = word.strip('.,!?;:()[]{}"\'-')
            
            # Skip if word already has diacritics (likely from proper noun correction)
            has_diacritics = any(ord(c) > 127 for c in clean_word)
            
            # Check if word needs correction
            if not has_diacritics and clean_word in self.word_corrections:
                # Preserve original punctuation
                prefix = word[:len(word) - len(word.lstrip('.,!?;:()[]{}"\'-'))]
                suffix = word[len(clean_word) + len(prefix):]
                corrected = prefix + self.word_corrections[clean_word] + suffix
                corrected_words.append(corrected)
            else:
                corrected_words.append(word)
        
        return ' '.join(corrected_words)


class WordBoundaryCorrector(TextProcessor):
    """Corrects word boundary errors in Vietnamese text."""
    
    def __init__(self):
        super().__init__("word_boundary_corrector")
        
        # Common word boundary errors
        self.split_corrections = {
            # Words that should be split
            "làmviệc": "làm việc",
            "côngnghệ": "công nghệ",
            "họctập": "học tập",
            "kinhdoanh": "kinh doanh",
            "xửlý": "xử lý",
            "pháttriển": "phát triển",
            "thựchiện": "thực hiện",
            "giảiquyết": "giải quyết",
        }
        
        self.merge_corrections = {
            # Words that should be merged
            "không thể": "không thể",  # Already correct
            "có thể": "có thể",  # Already correct
            "phải không": "phải không",  # Already correct
        }
    
    def process(self, text: str) -> str:
        """Apply word boundary corrections."""
        result = text
        
        # Apply split corrections
        for incorrect, correct in self.split_corrections.items():
            result = result.replace(incorrect, correct)
        
        # Fix multiple spaces
        result = re.sub(r'\s+', ' ', result)
        
        return result.strip()


class CommonErrorCorrector(TextProcessor):
    """Corrects common Vietnamese transcription errors."""
    
    def __init__(self):
        super().__init__("common_error_corrector")
        
        # Common misrecognitions by STT engines
        self.error_corrections = {
            # Common homophones
            "dược": "được",  # Common confusion
            "duoc": "được",
            "dc": "được",
            
            # Common abbreviations
            "k": "không",
            "ko": "không",
            "kg": "không",
            "kh": "không",
            
            # Common words
            "thi": "thì",
            "nay": "này",
            "do": "đó",
            "nhung": "những",
            "cua": "của",
            "ma": "mà",
            "den": "đến",
            "tren": "trên",
            "duoi": "dưới",
            
            # Numbers (excluding ambiguous ones like "nam" which could be part of "Việt Nam")
            "mot": "một",
            "hai": "hai",
            "ba": "ba",
            "bon": "bốn",
            # "nam": "năm",  # Ambiguous - could be number or part of proper noun
            "sau": "sáu",
            "bay": "bảy",
            "tam": "tám",
            "chin": "chín",
            "muoi": "mười",
        }
    
    def process(self, text: str) -> str:
        """Apply common error corrections."""
        result = text
        
        # Apply word-level corrections
        words = result.split()
        corrected_words = []
        
        for word in words:
            # Check if word (lowercase) needs correction
            lower_word = word.lower()
            if lower_word in self.error_corrections:
                corrected_words.append(self.error_corrections[lower_word])
            else:
                corrected_words.append(word)
        
        return ' '.join(corrected_words)


class PunctuationNormalizer(TextProcessor):
    """Normalizes punctuation in Vietnamese text."""
    
    def __init__(self):
        super().__init__("punctuation_normalizer")
    
    def process(self, text: str) -> str:
        """Normalize punctuation."""
        result = text
        
        # Remove multiple punctuation marks
        result = re.sub(r'([.!?]){2,}', r'\1', result)
        
        # Fix spacing around punctuation
        result = re.sub(r'\s+([.!?,;:])', r'\1', result)
        result = re.sub(r'([.!?,;:])\s*', r'\1 ', result)
        
        # Fix spacing around quotes
        result = re.sub(r'\s*"\s*', ' "', result)
        result = re.sub(r'\s*"\s*', '" ', result)
        
        # Remove spaces before closing punctuation
        result = re.sub(r'\s+([.!?,;:\)])', r'\1', result)
        
        # Add space after opening punctuation
        result = re.sub(r'([\(])\s*', r'\1 ', result)
        
        # Fix multiple spaces
        result = re.sub(r'\s+', ' ', result)
        
        # Capitalize first letter of sentences
        result = self._capitalize_sentences(result)
        
        return result.strip()
    
    def _capitalize_sentences(self, text: str) -> str:
        """Capitalize first letter of each sentence."""
        # Split by sentence-ending punctuation
        sentences = re.split(r'([.!?]\s+)', text)
        
        result = []
        for i, part in enumerate(sentences):
            if i % 2 == 0 and part:  # Actual sentence content
                # Capitalize first letter
                part = part[0].upper() + part[1:] if len(part) > 0 else part
            result.append(part)
        
        return ''.join(result)


class CapitalizationCorrector(TextProcessor):
    """Corrects capitalization in Vietnamese text."""
    
    def __init__(self):
        super().__init__("capitalization_corrector")
        
        # Words that should always be capitalized
        self.always_capitalize = {
            "việt nam", "hà nội", "hồ chí minh", "sài gòn", "đà nẵng",
            "hải phòng", "cần thơ", "nha trang", "huế", "vũng tàu",
            "ai", "ml", "api", "rest", "gpu", "cpu", "covid",
        }
    
    def process(self, text: str) -> str:
        """Apply capitalization corrections."""
        result = text
        
        # Capitalize proper nouns
        for word in self.always_capitalize:
            # Case-insensitive replacement
            pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
            result = pattern.sub(word.title(), result)
        
        # Capitalize first letter of text
        if result:
            result = result[0].upper() + result[1:]
        
        return result


class TextPostProcessor(ITextPostProcessor):
    """
    Text post-processor for Vietnamese transcription.
    
    Applies a pipeline of text processing steps to improve transcription quality.
    """
    
    def __init__(self):
        """Initialize post-processor with default pipeline."""
        self.processors: List[TextProcessor] = []
        self.custom_vocabulary: Dict[str, str] = {}
        
        # Add default processors in order
        self._setup_default_pipeline()
    
    def _setup_default_pipeline(self) -> None:
        """Setup default processing pipeline."""
        self.add_processor(WordBoundaryCorrector())
        self.add_processor(CommonErrorCorrector())
        self.add_processor(VietnameseDiacriticCorrector())
        self.add_processor(PunctuationNormalizer())
        self.add_processor(CapitalizationCorrector())
    
    def process(self, text: str, language: str = "vi") -> str:
        """
        Apply post-processing pipeline to text.
        
        Args:
            text: Input text to process.
            language: Language code (default: "vi").
            
        Returns:
            Processed text.
        """
        if not text or not text.strip():
            return text
        
        result = text
        
        # Apply each processor in order
        for processor in self.processors:
            result = processor.process(result)
        
        # Apply custom vocabulary if available
        if self.custom_vocabulary:
            result = self.apply_custom_vocabulary(result, self.custom_vocabulary)
        
        return result
    
    def add_processor(self, processor: TextProcessor) -> None:
        """
        Add text processor to pipeline.
        
        Args:
            processor: Text processor to add.
        """
        # Check if processor with same name already exists
        existing_names = [p.name for p in self.processors]
        if processor.name in existing_names:
            # Remove existing processor
            self.remove_processor(processor.name)
        
        self.processors.append(processor)
    
    def remove_processor(self, processor_name: str) -> None:
        """
        Remove text processor from pipeline.
        
        Args:
            processor_name: Name of processor to remove.
        """
        self.processors = [
            p for p in self.processors
            if p.name != processor_name
        ]
    
    def fix_vietnamese_errors(self, text: str) -> str:
        """
        Fix common Vietnamese transcription errors.
        
        Args:
            text: Input text with potential errors.
            
        Returns:
            Corrected text.
        """
        # Apply Vietnamese-specific corrections
        corrector = VietnameseDiacriticCorrector()
        result = corrector.process(text)
        
        # Apply common error corrections
        error_corrector = CommonErrorCorrector()
        result = error_corrector.process(result)
        
        return result
    
    def apply_custom_vocabulary(
        self,
        text: str,
        vocabulary: Dict[str, str],
    ) -> str:
        """
        Apply custom vocabulary corrections.
        
        Args:
            text: Input text.
            vocabulary: Dictionary mapping incorrect -> correct terms.
            
        Returns:
            Corrected text.
        """
        result = text
        
        # Sort by length (longest first) to handle phrases before words
        sorted_terms = sorted(vocabulary.items(), key=lambda x: len(x[0]), reverse=True)
        
        for incorrect, correct in sorted_terms:
            # Use word boundaries for whole word matching
            # Case-insensitive matching
            pattern = re.compile(r'\b' + re.escape(incorrect) + r'\b', re.IGNORECASE)
            result = pattern.sub(correct, result)
        
        return result
    
    def set_custom_vocabulary(self, vocabulary: Dict[str, str]) -> None:
        """
        Set custom vocabulary for corrections.
        
        Args:
            vocabulary: Dictionary mapping incorrect -> correct terms.
        """
        self.custom_vocabulary = vocabulary
    
    def load_custom_vocabulary_from_json(self, vocab_data: dict) -> None:
        """
        Load custom vocabulary from JSON format.
        
        Args:
            vocab_data: Dictionary containing vocabulary data in JSON format.
        """
        vocabulary = {}
        
        # Extract words
        if "words" in vocab_data:
            for word_entry in vocab_data["words"]:
                text = word_entry.get("text", "")
                if text:
                    # Map lowercase version to correct version
                    vocabulary[text.lower()] = text
        
        # Extract phrases
        if "phrases" in vocab_data:
            for phrase_entry in vocab_data["phrases"]:
                text = phrase_entry.get("text", "")
                if text:
                    # Map lowercase version to correct version
                    vocabulary[text.lower()] = text
        
        self.set_custom_vocabulary(vocabulary)
    
    def get_pipeline_info(self) -> List[Dict[str, str]]:
        """
        Get information about current processing pipeline.
        
        Returns:
            List of processor information dictionaries.
        """
        return [
            {"name": processor.name, "type": type(processor).__name__}
            for processor in self.processors
        ]
