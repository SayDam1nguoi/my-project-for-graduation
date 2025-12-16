"""
Subtitle Generator module for video transcription.

This module generates subtitle files (SRT and VTT formats) from transcription results.
Handles text segmentation, timing constraints, and format validation.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from datetime import timedelta

from .timestamp_aligner import WordTimestamp, TranscriptionSegment


logger = logging.getLogger(__name__)


@dataclass
class SubtitleSegment:
    """A single subtitle segment."""
    index: int
    start: float  # seconds
    end: float  # seconds
    text: str
    speaker: Optional[str] = None


class SubtitleGenerator:
    """
    Generates subtitle files from transcription results.
    
    Supports SRT and VTT formats with configurable text segmentation
    and timing constraints.
    """
    
    def __init__(
        self,
        max_chars_per_line: int = 42,
        max_lines: int = 2,
        min_duration: float = 1.0,
        max_duration: float = 7.0
    ):
        """
        Initialize subtitle generator.
        
        Args:
            max_chars_per_line: Maximum characters per line
            max_lines: Maximum number of lines per subtitle
            min_duration: Minimum display duration in seconds
            max_duration: Maximum display duration in seconds
        """
        self.max_chars_per_line = max_chars_per_line
        self.max_lines = max_lines
        self.min_duration = min_duration
        self.max_duration = max_duration
        
        logger.info(
            f"SubtitleGenerator initialized: max_chars={max_chars_per_line}, "
            f"max_lines={max_lines}, min_duration={min_duration}s, "
            f"max_duration={max_duration}s"
        )
    
    def generate_srt(
        self,
        segments: List[TranscriptionSegment],
        output_path: str,
        include_speaker: bool = False
    ) -> None:
        """
        Generate SRT subtitle file.
        
        Args:
            segments: List of transcription segments
            output_path: Path to output SRT file
            include_speaker: Whether to include speaker labels
        """
        logger.info(f"Generating SRT file: {output_path}")
        
        # Create subtitle segments
        subtitle_segments = self._create_subtitle_segments(segments)
        
        # Generate SRT content
        srt_content = self._format_srt(subtitle_segments, include_speaker)
        
        # Write to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(srt_content)
        
        logger.info(f"SRT file generated: {len(subtitle_segments)} segments")
    
    def generate_vtt(
        self,
        segments: List[TranscriptionSegment],
        output_path: str,
        include_speaker: bool = False
    ) -> None:
        """
        Generate VTT subtitle file.
        
        Args:
            segments: List of transcription segments
            output_path: Path to output VTT file
            include_speaker: Whether to include speaker labels
        """
        logger.info(f"Generating VTT file: {output_path}")
        
        # Create subtitle segments
        subtitle_segments = self._create_subtitle_segments(segments)
        
        # Generate VTT content
        vtt_content = self._format_vtt(subtitle_segments, include_speaker)
        
        # Write to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(vtt_content)
        
        logger.info(f"VTT file generated: {len(subtitle_segments)} segments")
    
    def _create_subtitle_segments(
        self,
        transcription_segments: List[TranscriptionSegment]
    ) -> List[SubtitleSegment]:
        """
        Create subtitle segments from transcription segments.
        
        Applies text segmentation and timing constraints.
        
        Args:
            transcription_segments: List of transcription segments
            
        Returns:
            List of subtitle segments
        """
        subtitle_segments = []
        segment_index = 1
        
        for trans_segment in transcription_segments:
            # Segment the text based on length constraints
            text_segments = self._segment_text(
                trans_segment.text,
                trans_segment.words
            )
            
            for text_seg in text_segments:
                # Apply timing constraints
                duration = text_seg['end'] - text_seg['start']
                
                # Enforce minimum duration
                if duration < self.min_duration:
                    # Extend end time to meet minimum duration
                    text_seg['end'] = text_seg['start'] + self.min_duration
                
                # Enforce maximum duration
                if duration > self.max_duration:
                    # Split into multiple segments if too long
                    split_segments = self._split_long_segment(text_seg)
                    for split_seg in split_segments:
                        subtitle_segments.append(
                            SubtitleSegment(
                                index=segment_index,
                                start=split_seg['start'],
                                end=split_seg['end'],
                                text=split_seg['text'],
                                speaker=trans_segment.speaker
                            )
                        )
                        segment_index += 1
                else:
                    subtitle_segments.append(
                        SubtitleSegment(
                            index=segment_index,
                            start=text_seg['start'],
                            end=text_seg['end'],
                            text=text_seg['text'],
                            speaker=trans_segment.speaker
                        )
                    )
                    segment_index += 1
        
        return subtitle_segments
    
    def _segment_text(
        self,
        text: str,
        words: List[WordTimestamp]
    ) -> List[dict]:
        """
        Segment text based on length constraints.
        
        Args:
            text: Full text to segment
            words: Word timestamps
            
        Returns:
            List of text segments with timing
        """
        if not words:
            return []
        
        segments = []
        current_segment_words = []
        current_line_length = 0
        current_line_count = 1
        
        max_segment_chars = self.max_chars_per_line * self.max_lines
        
        for word in words:
            word_text = word.word
            word_length = len(word_text)
            
            # Check if adding this word would exceed line length
            if current_line_length + word_length + 1 > self.max_chars_per_line:
                # Need to start a new line
                if current_line_count >= self.max_lines:
                    # Need to start a new segment
                    if current_segment_words:
                        segments.append(self._create_text_segment(current_segment_words))
                        current_segment_words = []
                        current_line_count = 1
                        current_line_length = 0
                else:
                    # Start new line in same segment
                    current_line_count += 1
                    current_line_length = 0
            
            # Add word to current segment
            current_segment_words.append(word)
            current_line_length += word_length + 1  # +1 for space
        
        # Add remaining words as final segment
        if current_segment_words:
            segments.append(self._create_text_segment(current_segment_words))
        
        return segments
    
    def _create_text_segment(self, words: List[WordTimestamp]) -> dict:
        """
        Create a text segment from words.
        
        Args:
            words: List of word timestamps
            
        Returns:
            Dictionary with text, start, and end
        """
        if not words:
            return {'text': '', 'start': 0.0, 'end': 0.0}
        
        # Join words into text
        text = ' '.join(word.word for word in words)
        
        # Format text with line breaks based on max_chars_per_line
        formatted_text = self._format_text_with_line_breaks(text)
        
        return {
            'text': formatted_text,
            'start': words[0].start,
            'end': words[-1].end
        }
    
    def _format_text_with_line_breaks(self, text: str) -> str:
        """
        Format text with line breaks based on max_chars_per_line.
        
        Args:
            text: Text to format
            
        Returns:
            Formatted text with line breaks
        """
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            word_length = len(word)
            
            # Check if adding this word would exceed line length
            if current_length + word_length + len(current_line) > self.max_chars_per_line:
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                    current_length = word_length
                else:
                    # Single word exceeds line length - add it anyway
                    lines.append(word)
                    current_line = []
                    current_length = 0
            else:
                current_line.append(word)
                current_length += word_length
        
        # Add remaining words
        if current_line:
            lines.append(' '.join(current_line))
        
        # Limit to max_lines
        if len(lines) > self.max_lines:
            lines = lines[:self.max_lines]
        
        return '\n'.join(lines)
    
    def _split_long_segment(self, segment: dict) -> List[dict]:
        """
        Split a segment that exceeds max_duration.
        
        Args:
            segment: Segment dictionary
            
        Returns:
            List of split segments
        """
        duration = segment['end'] - segment['start']
        
        if duration <= self.max_duration:
            return [segment]
        
        # Calculate number of splits needed
        num_splits = int(duration / self.max_duration) + 1
        split_duration = duration / num_splits
        
        # Split text into roughly equal parts
        words = segment['text'].split()
        words_per_split = max(1, len(words) // num_splits)
        
        splits = []
        for i in range(num_splits):
            start_idx = i * words_per_split
            end_idx = start_idx + words_per_split if i < num_splits - 1 else len(words)
            
            split_words = words[start_idx:end_idx]
            split_text = ' '.join(split_words)
            
            split_start = segment['start'] + (i * split_duration)
            split_end = split_start + split_duration
            
            splits.append({
                'text': split_text,
                'start': split_start,
                'end': split_end
            })
        
        return splits
    
    def _format_srt(
        self,
        segments: List[SubtitleSegment],
        include_speaker: bool
    ) -> str:
        """
        Format segments as SRT content.
        
        Args:
            segments: List of subtitle segments
            include_speaker: Whether to include speaker labels
            
        Returns:
            SRT formatted string
        """
        srt_lines = []
        
        for segment in segments:
            # Segment index
            srt_lines.append(str(segment.index))
            
            # Timestamp line
            start_time = self._format_srt_timestamp(segment.start)
            end_time = self._format_srt_timestamp(segment.end)
            srt_lines.append(f"{start_time} --> {end_time}")
            
            # Text (with optional speaker label)
            text = segment.text
            if include_speaker and segment.speaker:
                text = f"[{segment.speaker}] {text}"
            
            srt_lines.append(text)
            
            # Blank line separator
            srt_lines.append("")
        
        return '\n'.join(srt_lines)
    
    def _format_vtt(
        self,
        segments: List[SubtitleSegment],
        include_speaker: bool
    ) -> str:
        """
        Format segments as VTT content.
        
        Args:
            segments: List of subtitle segments
            include_speaker: Whether to include speaker labels
            
        Returns:
            VTT formatted string
        """
        vtt_lines = ["WEBVTT", ""]
        
        for segment in segments:
            # Timestamp line
            start_time = self._format_vtt_timestamp(segment.start)
            end_time = self._format_vtt_timestamp(segment.end)
            vtt_lines.append(f"{start_time} --> {end_time}")
            
            # Text (with optional speaker label)
            text = segment.text
            if include_speaker and segment.speaker:
                text = f"<v {segment.speaker}>{text}</v>"
            
            vtt_lines.append(text)
            
            # Blank line separator
            vtt_lines.append("")
        
        return '\n'.join(vtt_lines)
    
    def _format_srt_timestamp(self, seconds: float) -> str:
        """
        Format timestamp for SRT format (HH:MM:SS,mmm).
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted timestamp string
        """
        td = timedelta(seconds=seconds)
        hours = int(td.total_seconds() // 3600)
        minutes = int((td.total_seconds() % 3600) // 60)
        secs = int(td.total_seconds() % 60)
        millis = int((td.total_seconds() % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    def _format_vtt_timestamp(self, seconds: float) -> str:
        """
        Format timestamp for VTT format (HH:MM:SS.mmm).
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted timestamp string
        """
        td = timedelta(seconds=seconds)
        hours = int(td.total_seconds() // 3600)
        minutes = int((td.total_seconds() % 3600) // 60)
        secs = int(td.total_seconds() % 60)
        millis = int((td.total_seconds() % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    
    def validate_srt(self, file_path: str) -> bool:
        """
        Validate SRT file format.
        
        Args:
            file_path: Path to SRT file
            
        Returns:
            True if valid SRT format
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split into subtitle blocks
            blocks = content.strip().split('\n\n')
            
            if not blocks:
                logger.error("SRT file is empty")
                return False
            
            # Validate each block
            for i, block in enumerate(blocks, 1):
                lines = block.strip().split('\n')
                
                if len(lines) < 3:
                    logger.error(f"SRT block {i} has insufficient lines")
                    return False
                
                # Check index
                try:
                    index = int(lines[0])
                    if index != i:
                        logger.warning(f"SRT block index mismatch: expected {i}, got {index}")
                except ValueError:
                    logger.error(f"SRT block {i} has invalid index: {lines[0]}")
                    return False
                
                # Check timestamp format
                timestamp_pattern = r'\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}'
                if not re.match(timestamp_pattern, lines[1]):
                    logger.error(f"SRT block {i} has invalid timestamp: {lines[1]}")
                    return False
            
            logger.info(f"SRT file validated: {len(blocks)} segments")
            return True
            
        except Exception as e:
            logger.error(f"Error validating SRT file: {e}")
            return False
    
    def validate_vtt(self, file_path: str) -> bool:
        """
        Validate VTT file format.
        
        Args:
            file_path: Path to VTT file
            
        Returns:
            True if valid VTT format
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.strip().split('\n')
            
            if not lines:
                logger.error("VTT file is empty")
                return False
            
            # Check WEBVTT header
            if not lines[0].startswith('WEBVTT'):
                logger.error("VTT file missing WEBVTT header")
                return False
            
            # Validate timestamp lines
            timestamp_pattern = r'\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}'
            timestamp_count = 0
            
            for line in lines[1:]:
                if re.match(timestamp_pattern, line):
                    timestamp_count += 1
            
            if timestamp_count == 0:
                logger.error("VTT file has no valid timestamps")
                return False
            
            logger.info(f"VTT file validated: {timestamp_count} segments")
            return True
            
        except Exception as e:
            logger.error(f"Error validating VTT file: {e}")
            return False
