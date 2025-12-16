"""Processing pipeline interfaces."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np

from ..models.transcription import TranscriptionSegment, TranscriptionResult
from ..models.quality import QualityReport, AudioQualityReport, TranscriptionQualityReport


class HallucinationPattern:
    """Pattern for detecting hallucinations."""
    
    def __init__(
        self,
        pattern: str,
        pattern_type: str,  # "exact", "partial", "keyword"
        language: str = "vi",
        confidence_threshold: float = 0.8,
    ):
        self.pattern = pattern
        self.pattern_type = pattern_type
        self.language = language
        self.confidence_threshold = confidence_threshold


class FilterStats:
    """Statistics about hallucination filtering."""
    
    def __init__(self):
        self.total_segments = 0
        self.filtered_segments = 0
        self.filter_reasons: Dict[str, int] = {}
    
    @property
    def filter_rate(self) -> float:
        """Get filtering rate (0-1)."""
        if self.total_segments == 0:
            return 0.0
        return self.filtered_segments / self.total_segments


class IHallucinationFilter(ABC):
    """Interface for hallucination detection and filtering."""
    
    @abstractmethod
    def is_hallucination(self, text: str, metadata: Dict[str, Any]) -> bool:
        """
        Check if text is likely hallucination.
        
        Args:
            text: Transcribed text to check.
            metadata: Additional metadata (confidence, no_speech_prob, etc.).
            
        Returns:
            True if likely hallucination, False otherwise.
        """
        pass
    
    @abstractmethod
    def filter_segments(
        self,
        segments: List[TranscriptionSegment],
    ) -> List[TranscriptionSegment]:
        """
        Filter hallucination segments from list.
        
        Args:
            segments: List of transcription segments.
            
        Returns:
            Filtered list of segments.
        """
        pass
    
    @abstractmethod
    def add_pattern(self, pattern: HallucinationPattern) -> None:
        """
        Add hallucination pattern to filter.
        
        Args:
            pattern: Hallucination pattern to add.
        """
        pass
    
    @abstractmethod
    def remove_pattern(self, pattern: str) -> None:
        """
        Remove hallucination pattern from filter.
        
        Args:
            pattern: Pattern string to remove.
        """
        pass
    
    @abstractmethod
    def get_filter_stats(self) -> FilterStats:
        """
        Get statistics about filtered content.
        
        Returns:
            Filter statistics.
        """
        pass
    
    @abstractmethod
    def reset_stats(self) -> None:
        """Reset filter statistics."""
        pass


class TextProcessor:
    """Base class for text processing steps."""
    
    def __init__(self, name: str):
        self.name = name
    
    def process(self, text: str) -> str:
        """Process text and return result."""
        raise NotImplementedError


class ITextPostProcessor(ABC):
    """Interface for text post-processing."""
    
    @abstractmethod
    def process(self, text: str, language: str = "vi") -> str:
        """
        Apply post-processing to text.
        
        Args:
            text: Input text to process.
            language: Language code.
            
        Returns:
            Processed text.
        """
        pass
    
    @abstractmethod
    def add_processor(self, processor: TextProcessor) -> None:
        """
        Add text processor to pipeline.
        
        Args:
            processor: Text processor to add.
        """
        pass
    
    @abstractmethod
    def remove_processor(self, processor_name: str) -> None:
        """
        Remove text processor from pipeline.
        
        Args:
            processor_name: Name of processor to remove.
        """
        pass
    
    @abstractmethod
    def fix_vietnamese_errors(self, text: str) -> str:
        """
        Fix common Vietnamese transcription errors.
        
        Args:
            text: Input text with potential errors.
            
        Returns:
            Corrected text.
        """
        pass
    
    @abstractmethod
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
        pass


class IQualityAnalyzer(ABC):
    """Interface for quality analysis."""
    
    @abstractmethod
    def analyze_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> AudioQualityReport:
        """
        Analyze audio quality.
        
        Args:
            audio: Audio data to analyze.
            sample_rate: Sample rate of audio.
            
        Returns:
            Audio quality report.
        """
        pass
    
    @abstractmethod
    def analyze_transcription(
        self,
        result: TranscriptionResult,
    ) -> TranscriptionQualityReport:
        """
        Analyze transcription quality.
        
        Args:
            result: Transcription result to analyze.
            
        Returns:
            Transcription quality report.
        """
        pass
    
    @abstractmethod
    def get_recommendations(
        self,
        audio_report: AudioQualityReport,
        transcription_report: TranscriptionQualityReport,
    ) -> List[str]:
        """
        Get improvement recommendations based on quality reports.
        
        Args:
            audio_report: Audio quality report.
            transcription_report: Transcription quality report.
            
        Returns:
            List of recommendations.
        """
        pass
    
    @abstractmethod
    def create_quality_report(
        self,
        audio: np.ndarray,
        sample_rate: int,
        result: TranscriptionResult,
    ) -> QualityReport:
        """
        Create comprehensive quality report.
        
        Args:
            audio: Audio data.
            sample_rate: Sample rate of audio.
            result: Transcription result.
            
        Returns:
            Complete quality report.
        """
        pass
