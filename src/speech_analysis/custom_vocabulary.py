"""
Custom Vocabulary Processor Module for Enhanced Vietnamese STT

This module provides custom vocabulary processing capabilities:
- Vocabulary file loading (JSON format)
- Vocabulary validation
- Post-processing with custom vocabulary priority
- Vietnamese diacritic preservation
- Hot-reload functionality
- Ambiguity resolution logic
"""

import json
import re
import unicodedata
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class VocabularyEntry:
    """Represents a custom vocabulary entry."""
    text: str
    priority: int = 1
    alternatives: List[str] = None
    
    def __post_init__(self):
        if self.alternatives is None:
            self.alternatives = []


class CustomVocabularyProcessor:
    """
    Custom vocabulary processor for improving STT accuracy with domain-specific terms.
    
    Provides:
    - Loading and validation of custom vocabulary from JSON files
    - Priority-based matching for custom terms
    - Vietnamese diacritic preservation
    - Hot-reload capability without service restart
    - Ambiguity resolution using custom vocabulary
    """
    
    def __init__(self, vocabulary_file: Optional[str] = None):
        """
        Initialize with optional vocabulary file.
        
        Args:
            vocabulary_file: Path to JSON vocabulary file (optional)
        """
        self.vocabulary_file = vocabulary_file
        self.words: Dict[str, VocabularyEntry] = {}
        self.phrases: Dict[str, VocabularyEntry] = {}
        
        # Normalized lookup tables for case-insensitive matching
        self._normalized_words: Dict[str, str] = {}
        self._normalized_phrases: Dict[str, str] = {}
        
        logger.info("CustomVocabularyProcessor initialized")
        
        if vocabulary_file:
            self.load_vocabulary(vocabulary_file)
    
    def load_vocabulary(self, filepath: str) -> None:
        """
        Load custom vocabulary from JSON file.
        
        Args:
            filepath: Path to vocabulary JSON file
            
        Raises:
            FileNotFoundError: If vocabulary file doesn't exist
            ValueError: If vocabulary file format is invalid
        """
        path = Path(filepath)
        
        if not path.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {filepath}")
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate and load vocabulary
            self._validate_and_load(data)
            
            self.vocabulary_file = filepath
            logger.info(
                f"Loaded vocabulary from {filepath}: "
                f"{len(self.words)} words, {len(self.phrases)} phrases"
            )
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in vocabulary file: {e}")
        except Exception as e:
            raise ValueError(f"Failed to load vocabulary: {e}")
    
    def _validate_and_load(self, data: dict) -> None:
        """
        Validate and load vocabulary data.
        
        Args:
            data: Dictionary containing vocabulary data
            
        Raises:
            ValueError: If data format is invalid
        """
        # Clear existing vocabulary
        self.words.clear()
        self.phrases.clear()
        self._normalized_words.clear()
        self._normalized_phrases.clear()
        
        # Load words
        if 'words' in data:
            if not isinstance(data['words'], list):
                raise ValueError("'words' must be a list")
            
            for item in data['words']:
                if isinstance(item, str):
                    # Simple string format
                    self.add_word(item, priority=1)
                elif isinstance(item, dict):
                    # Dictionary format with priority
                    text = item.get('text')
                    priority = item.get('priority', 1)
                    alternatives = item.get('alternatives', [])
                    
                    if not text:
                        logger.warning(f"Skipping word entry without 'text': {item}")
                        continue
                    
                    self.add_word(text, priority=priority, alternatives=alternatives)
                else:
                    logger.warning(f"Invalid word entry format: {item}")
        
        # Load phrases
        if 'phrases' in data:
            if not isinstance(data['phrases'], list):
                raise ValueError("'phrases' must be a list")
            
            for item in data['phrases']:
                if isinstance(item, str):
                    # Simple string format
                    self.add_phrase(item, priority=1)
                elif isinstance(item, dict):
                    # Dictionary format with priority
                    text = item.get('text')
                    priority = item.get('priority', 1)
                    alternatives = item.get('alternatives', [])
                    
                    if not text:
                        logger.warning(f"Skipping phrase entry without 'text': {item}")
                        continue
                    
                    self.add_phrase(text, priority=priority, alternatives=alternatives)
                else:
                    logger.warning(f"Invalid phrase entry format: {item}")
    
    def add_word(
        self, 
        word: str, 
        priority: int = 1,
        alternatives: List[str] = None
    ) -> None:
        """
        Add single word to vocabulary.
        
        Args:
            word: Word to add
            priority: Priority level (higher = more important)
            alternatives: Alternative spellings or forms
        """
        if not word or not word.strip():
            logger.warning("Attempted to add empty word")
            return
        
        word = word.strip()
        
        entry = VocabularyEntry(
            text=word,
            priority=priority,
            alternatives=alternatives or []
        )
        
        self.words[word] = entry
        
        # Add normalized version for case-insensitive lookup
        normalized = self._normalize_text(word)
        self._normalized_words[normalized] = word
        
        logger.debug(f"Added word: '{word}' with priority {priority}")
    
    def add_phrase(
        self, 
        phrase: str, 
        priority: int = 1,
        alternatives: List[str] = None
    ) -> None:
        """
        Add phrase to vocabulary.
        
        Args:
            phrase: Phrase to add
            priority: Priority level (higher = more important)
            alternatives: Alternative forms
        """
        if not phrase or not phrase.strip():
            logger.warning("Attempted to add empty phrase")
            return
        
        phrase = phrase.strip()
        
        entry = VocabularyEntry(
            text=phrase,
            priority=priority,
            alternatives=alternatives or []
        )
        
        self.phrases[phrase] = entry
        
        # Add normalized version for case-insensitive lookup
        normalized = self._normalize_text(phrase)
        self._normalized_phrases[normalized] = phrase
        
        logger.debug(f"Added phrase: '{phrase}' with priority {priority}")
    
    def process_transcription(
        self, 
        text: str, 
        alternatives: List[str] = None
    ) -> str:
        """
        Apply custom vocabulary to transcription.
        
        Args:
            text: Original transcription
            alternatives: Alternative transcriptions if available
            
        Returns:
            Processed transcription with custom vocabulary applied
        """
        if not text:
            return text
        
        # First, try to resolve ambiguity if alternatives are provided
        if alternatives:
            resolved = self._resolve_ambiguity(text, alternatives)
            if resolved != text:
                logger.debug(f"Resolved ambiguity: '{text}' -> '{resolved}'")
                text = resolved
        
        # Apply phrase replacements (longer matches first)
        text = self._apply_phrase_replacements(text)
        
        # Apply word replacements
        text = self._apply_word_replacements(text)
        
        return text
    
    def _apply_phrase_replacements(self, text: str) -> str:
        """
        Apply phrase replacements to text.
        
        Args:
            text: Input text
            
        Returns:
            Text with phrase replacements applied
        """
        # Sort phrases by length (longest first) and priority
        sorted_phrases = sorted(
            self.phrases.items(),
            key=lambda x: (len(x[0]), x[1].priority),
            reverse=True
        )
        
        result = text
        
        for phrase, entry in sorted_phrases:
            # Create case-insensitive pattern that preserves word boundaries
            pattern = r'\b' + re.escape(self._normalize_text(phrase)) + r'\b'
            
            # Find all matches
            normalized_result = self._normalize_text(result)
            matches = list(re.finditer(pattern, normalized_result, re.IGNORECASE))
            
            # Replace matches with correct form (preserving diacritics)
            for match in reversed(matches):  # Reverse to maintain indices
                start, end = match.span()
                # Replace in original text at the same position
                result = result[:start] + entry.text + result[end:]
        
        return result
    
    def _apply_word_replacements(self, text: str) -> str:
        """
        Apply word replacements to text.
        
        Args:
            text: Input text
            
        Returns:
            Text with word replacements applied
        """
        # Sort words by priority
        sorted_words = sorted(
            self.words.items(),
            key=lambda x: x[1].priority,
            reverse=True
        )
        
        # Split text into words while preserving punctuation
        words = re.findall(r'\b\w+\b|\W+', text)
        
        result_words = []
        for word in words:
            # Skip non-word tokens (punctuation, spaces)
            if not re.match(r'\w+', word):
                result_words.append(word)
                continue
            
            # Check if word matches custom vocabulary
            normalized_word = self._normalize_text(word)
            
            if normalized_word in self._normalized_words:
                # Replace with custom vocabulary form
                original_word = self._normalized_words[normalized_word]
                result_words.append(self.words[original_word].text)
            else:
                # Keep original word
                result_words.append(word)
        
        return ''.join(result_words)
    
    def _resolve_ambiguity(
        self, 
        text: str, 
        alternatives: List[str]
    ) -> str:
        """
        Resolve ambiguity using custom vocabulary.
        
        Args:
            text: Original transcription
            alternatives: Alternative transcriptions
            
        Returns:
            Best transcription based on custom vocabulary matches
        """
        # Score each alternative based on custom vocabulary matches
        candidates = [text] + alternatives
        scores = []
        
        for candidate in candidates:
            score = self._calculate_vocabulary_score(candidate)
            scores.append(score)
        
        # Return candidate with highest score
        best_idx = scores.index(max(scores))
        return candidates[best_idx]
    
    def _calculate_vocabulary_score(self, text: str) -> float:
        """
        Calculate score based on custom vocabulary matches.
        
        Args:
            text: Text to score
            
        Returns:
            Score (higher = more custom vocabulary matches)
        """
        score = 0.0
        normalized_text = self._normalize_text(text)
        
        # Check phrase matches (weighted by priority and length)
        for phrase, entry in self.phrases.items():
            normalized_phrase = self._normalize_text(phrase)
            if normalized_phrase in normalized_text:
                score += entry.priority * len(phrase.split())
        
        # Check word matches (weighted by priority)
        words = re.findall(r'\b\w+\b', normalized_text)
        for word in words:
            if word in self._normalized_words:
                original_word = self._normalized_words[word]
                score += self.words[original_word].priority
        
        return score
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for comparison while preserving Vietnamese diacritics.
        
        This normalization is for matching purposes only - we preserve the
        original diacritics in the vocabulary entries.
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text (lowercase, normalized Unicode)
        """
        # Convert to lowercase
        text = text.lower()
        
        # Normalize Unicode (NFC form - canonical composition)
        # This ensures Vietnamese diacritics are in consistent form
        text = unicodedata.normalize('NFC', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def reload_vocabulary(self) -> None:
        """
        Reload vocabulary from file without restarting.
        
        Raises:
            ValueError: If no vocabulary file was previously loaded
        """
        if not self.vocabulary_file:
            raise ValueError("No vocabulary file to reload")
        
        logger.info(f"Reloading vocabulary from {self.vocabulary_file}")
        self.load_vocabulary(self.vocabulary_file)
    
    def get_vocabulary_info(self) -> dict:
        """
        Get information about loaded vocabulary.
        
        Returns:
            Dictionary containing vocabulary statistics
        """
        return {
            'vocabulary_file': self.vocabulary_file,
            'num_words': len(self.words),
            'num_phrases': len(self.phrases),
            'total_entries': len(self.words) + len(self.phrases),
            'words': list(self.words.keys()),
            'phrases': list(self.phrases.keys())
        }
    
    def has_vocabulary(self) -> bool:
        """
        Check if any vocabulary is loaded.
        
        Returns:
            True if vocabulary is loaded, False otherwise
        """
        return len(self.words) > 0 or len(self.phrases) > 0
    
    def clear_vocabulary(self) -> None:
        """Clear all vocabulary entries."""
        self.words.clear()
        self.phrases.clear()
        self._normalized_words.clear()
        self._normalized_phrases.clear()
        logger.info("Vocabulary cleared")
