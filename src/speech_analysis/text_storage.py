"""
Text Storage Module

This module provides functionality for saving speech transcripts to files with:
- UTF-8 encoding for Vietnamese text support
- Metadata headers with session information
- Statistics section with quality metrics
- Automatic filename generation with collision handling
- Backup and recovery capabilities
- Error handling for file I/O operations

The module ensures transcripts are properly formatted and saved with all
relevant metadata for future reference.
"""

import os
import glob
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple
from pathlib import Path
import shutil

from .quality_analyzer import QualityReport
from .exceptions import FileStorageError


@dataclass
class TranscriptMetadata:
    """
    Metadata for a transcript session.
    
    Attributes:
        timestamp: When the recording started
        duration: Total duration in seconds
        word_count: Number of words in transcript
        language: Language of the transcript (e.g., "vi" for Vietnamese)
        model_name: Name of the STT model used (e.g., "whisper base")
        session_id: Optional unique identifier for the session
    """
    timestamp: datetime
    duration: float
    word_count: int
    language: str
    model_name: str
    session_id: Optional[str] = None


@dataclass
class TranscriptInfo:
    """
    Information about a saved transcript file.
    
    Attributes:
        filepath: Full path to the transcript file
        filename: Name of the file
        timestamp: When the transcript was created
        duration: Duration of the recording
        word_count: Number of words
        clarity_score: Average clarity score
        fluency_score: Average fluency score
    """
    filepath: str
    filename: str
    timestamp: datetime
    duration: float
    word_count: int
    clarity_score: float
    fluency_score: float


class TextStorage:
    """
    Manages saving and retrieving speech transcripts.
    
    This class handles:
    - Saving transcripts with UTF-8 encoding
    - Generating unique filenames
    - Creating formatted files with metadata and statistics
    - Backup and recovery operations
    - Listing saved transcripts
    """
    
    def __init__(self, output_dir: str = "transcripts"):
        """
        Initialize text storage.
        
        Args:
            output_dir: Directory where transcripts will be saved
        """
        self.output_dir = output_dir
        self._ensure_directory_exists()
    
    def _ensure_directory_exists(self) -> None:
        """Create the output directory if it doesn't exist."""
        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except Exception as e:
            raise FileStorageError(f"Cannot create directory {self.output_dir}: {e}")
    
    def generate_filename(self, timestamp: datetime, suffix: str = "") -> str:
        """
        Generate a filename for a transcript.
        
        Args:
            timestamp: Timestamp for the transcript
            suffix: Optional suffix to add (e.g., "_1", "_2" for collisions)
        
        Returns:
            Filename in format: transcript_YYYYMMDD_HHMMSS[suffix].txt
        """
        date_str = timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"transcript_{date_str}{suffix}.txt"
        return filename
    
    def _get_unique_filepath(self, base_filepath: str) -> str:
        """
        Get a unique filepath by adding a number suffix if file exists.
        
        Args:
            base_filepath: The desired filepath
        
        Returns:
            A unique filepath that doesn't exist yet
        """
        if not os.path.exists(base_filepath):
            return base_filepath
        
        # Extract directory, base name, and extension
        directory = os.path.dirname(base_filepath)
        filename = os.path.basename(base_filepath)
        name, ext = os.path.splitext(filename)
        
        # Try adding numbers until we find a unique name
        counter = 1
        while True:
            new_filename = f"{name}_{counter}{ext}"
            new_filepath = os.path.join(directory, new_filename)
            if not os.path.exists(new_filepath):
                return new_filepath
            counter += 1
    
    def save_transcript(
        self,
        text: str,
        metadata: TranscriptMetadata,
        quality_report: QualityReport,
        filepath: Optional[str] = None
    ) -> str:
        """
        Save a transcript to a file with metadata and statistics.
        
        Args:
            text: The transcript text to save
            metadata: Metadata about the recording session
            quality_report: Quality analysis report
            filepath: Optional custom filepath (if None, auto-generate)
        
        Returns:
            The filepath where the transcript was saved
        
        Raises:
            FileStorageError: If saving fails
        """
        try:
            # Generate filepath if not provided
            if filepath is None:
                filename = self.generate_filename(metadata.timestamp)
                filepath = os.path.join(self.output_dir, filename)
            
            # Ensure unique filepath
            filepath = self._get_unique_filepath(filepath)
            
            # Format the file content
            content = self._format_transcript(text, metadata, quality_report)
            
            # Save with UTF-8 encoding
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return filepath
            
        except PermissionError as e:
            raise FileStorageError(
                f"Permission denied when saving to {filepath}. "
                "Please choose a different location or check file permissions."
            ) from e
        except OSError as e:
            if "No space left on device" in str(e) or e.errno == 28:
                raise FileStorageError(
                    "Disk is full. Please free up space or choose a different location."
                ) from e
            raise FileStorageError(f"Failed to save transcript: {e}") from e
        except Exception as e:
            raise FileStorageError(f"Unexpected error saving transcript: {e}") from e
    
    def _format_transcript(
        self,
        text: str,
        metadata: TranscriptMetadata,
        quality_report: QualityReport
    ) -> str:
        """
        Format the transcript with metadata and statistics.
        
        Args:
            text: The transcript text
            metadata: Session metadata
            quality_report: Quality analysis report
        
        Returns:
            Formatted transcript content
        """
        # Format duration as HH:MM:SS
        hours = int(metadata.duration // 3600)
        minutes = int((metadata.duration % 3600) // 60)
        seconds = int(metadata.duration % 60)
        duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        # Determine quality labels
        clarity_label = self._get_quality_label(quality_report.clarity_score)
        fluency_label = self._get_quality_label(quality_report.fluency_score)
        
        # Build the formatted content
        lines = [
            "=" * 80,
            "                        TRANSCRIPT - PHIÊN GHI ÂM",
            "=" * 80,
            "",
            f"Ngày giờ:           {metadata.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Thời lượng:         {duration_str}",
            f"Số từ:              {metadata.word_count}",
            f"Clarity Score:      {quality_report.clarity_score:.0f}/100 ({clarity_label})",
            f"Fluency Score:      {quality_report.fluency_score:.0f}/100 ({fluency_label})",
            f"Model:              {metadata.model_name}",
            f"Ngôn ngữ:           {self._get_language_name(metadata.language)}",
            "",
            "=" * 80,
            "                              NỘI DUNG",
            "=" * 80,
            "",
            text.strip(),
            "",
            "=" * 80,
            "                           THỐNG KÊ CHI TIẾT",
            "=" * 80,
            "",
            f"Tốc độ nói:         {quality_report.speech_rate:.1f} âm tiết/giây",
            f"Số lần ngắt quãng:  {quality_report.pause_count}",
            f"Thời gian ngắt TB:  {quality_report.avg_pause_duration:.1f} giây",
            f"SNR:                {quality_report.snr:.1f} dB",
            "",
        ]
        
        # Add recommendations if any
        if quality_report.recommendations:
            lines.append("Khuyến nghị:")
            for rec in quality_report.recommendations:
                lines.append(f"- {rec}")
            lines.append("")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def _get_quality_label(self, score: float) -> str:
        """Get a quality label based on score."""
        if score >= 80:
            return "Tốt"
        elif score >= 60:
            return "Khá"
        elif score >= 40:
            return "Trung bình"
        else:
            return "Cần cải thiện"
    
    def _get_language_name(self, language_code: str) -> str:
        """Get language name from code."""
        language_map = {
            "vi": "Tiếng Việt",
            "en": "English",
            "zh": "中文",
            "ja": "日本語",
            "ko": "한국어",
        }
        return language_map.get(language_code, language_code)
    
    def create_backup(self, filepath: str) -> str:
        """
        Create a backup copy of a transcript file.
        
        Args:
            filepath: Path to the file to backup
        
        Returns:
            Path to the backup file
        
        Raises:
            FileStorageError: If backup creation fails
        """
        try:
            if not os.path.exists(filepath):
                raise FileStorageError(f"File not found: {filepath}")
            
            # Generate backup filename
            directory = os.path.dirname(filepath)
            filename = os.path.basename(filepath)
            name, ext = os.path.splitext(filename)
            backup_filename = f"{name}_backup{ext}"
            backup_filepath = os.path.join(directory, backup_filename)
            
            # Make it unique if backup already exists
            backup_filepath = self._get_unique_filepath(backup_filepath)
            
            # Copy the file
            shutil.copy2(filepath, backup_filepath)
            
            return backup_filepath
            
        except Exception as e:
            raise FileStorageError(f"Failed to create backup: {e}") from e
    
    def list_transcripts(self) -> List[TranscriptInfo]:
        """
        List all transcript files in the output directory.
        
        Returns:
            List of TranscriptInfo objects for each transcript file
        """
        transcripts = []
        
        try:
            # Find all .txt files in the output directory
            pattern = os.path.join(self.output_dir, "transcript_*.txt")
            files = glob.glob(pattern)
            
            for filepath in files:
                # Skip backup and temp files
                filename = os.path.basename(filepath)
                if "_backup" in filename or filename.startswith(".temp"):
                    continue
                
                try:
                    # Parse metadata from file
                    info = self._parse_transcript_file(filepath)
                    if info:
                        transcripts.append(info)
                except Exception:
                    # Skip files that can't be parsed
                    continue
            
            # Sort by timestamp (newest first)
            transcripts.sort(key=lambda x: x.timestamp, reverse=True)
            
            return transcripts
            
        except Exception as e:
            # Return empty list if directory doesn't exist or can't be read
            return []
    
    def _parse_transcript_file(self, filepath: str) -> Optional[TranscriptInfo]:
        """
        Parse a transcript file to extract metadata.
        
        Args:
            filepath: Path to the transcript file
        
        Returns:
            TranscriptInfo object or None if parsing fails
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract metadata from content
            lines = content.split('\n')
            
            timestamp = None
            duration = 0.0
            word_count = 0
            clarity_score = 0.0
            fluency_score = 0.0
            
            for line in lines:
                line = line.strip()
                
                if line.startswith("Ngày giờ:"):
                    date_str = line.split(":", 1)[1].strip()
                    timestamp = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                
                elif line.startswith("Thời lượng:"):
                    time_str = line.split(":", 1)[1].strip()
                    parts = time_str.split(":")
                    if len(parts) == 3:
                        h, m, s = map(int, parts)
                        duration = h * 3600 + m * 60 + s
                
                elif line.startswith("Số từ:"):
                    word_count = int(line.split(":")[1].strip())
                
                elif line.startswith("Clarity Score:"):
                    score_str = line.split(":")[1].split("/")[0].strip()
                    clarity_score = float(score_str)
                
                elif line.startswith("Fluency Score:"):
                    score_str = line.split(":")[1].split("/")[0].strip()
                    fluency_score = float(score_str)
            
            if timestamp is None:
                return None
            
            return TranscriptInfo(
                filepath=filepath,
                filename=os.path.basename(filepath),
                timestamp=timestamp,
                duration=duration,
                word_count=word_count,
                clarity_score=clarity_score,
                fluency_score=fluency_score
            )
            
        except Exception:
            return None
    
    def save_temp_session(
        self,
        text: str,
        metadata: TranscriptMetadata,
        quality_report: QualityReport
    ) -> str:
        """
        Save a temporary session file for auto-save/recovery.
        
        Args:
            text: The transcript text
            metadata: Session metadata
            quality_report: Quality report
        
        Returns:
            Path to the temp file
        """
        temp_filename = ".temp_session.txt"
        temp_filepath = os.path.join(self.output_dir, temp_filename)
        
        try:
            content = self._format_transcript(text, metadata, quality_report)
            with open(temp_filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return temp_filepath
        except Exception as e:
            # Silently fail for temp saves
            return ""
    
    def get_temp_sessions(self) -> List[str]:
        """
        Get list of temporary session files.
        
        Returns:
            List of paths to temp session files
        """
        pattern = os.path.join(self.output_dir, ".temp_*.txt")
        return glob.glob(pattern)
    
    def delete_temp_session(self, filepath: str) -> bool:
        """
        Delete a temporary session file.
        
        Args:
            filepath: Path to the temp file
        
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                return True
            return False
        except Exception:
            return False
