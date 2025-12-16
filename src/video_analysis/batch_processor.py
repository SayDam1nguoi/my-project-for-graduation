"""
Batch Processor module for video transcription.

This module handles splitting long audio into chunks, processing them
in parallel, and merging results with accurate timestamps.
"""

import numpy as np
import logging
import time
import threading
from typing import Optional, List, Callable, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path

from ..speech_analysis.speech_to_text import (
    TranscriptionResult,
    TranscriptionSegment
)


logger = logging.getLogger(__name__)


class ProcessingStatus(Enum):
    """Status of batch processing."""
    IDLE = "idle"
    PROCESSING = "processing"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class AudioChunk:
    """Represents a chunk of audio for processing."""
    index: int
    audio: np.ndarray
    start_time: float
    end_time: float
    overlap_start: float = 0.0
    overlap_end: float = 0.0


@dataclass
class ChunkResult:
    """Result from processing a single chunk."""
    chunk_index: int
    transcription: TranscriptionResult
    start_time: float
    end_time: float
    processing_time: float


@dataclass
class ProcessingProgress:
    """Information about processing progress."""
    current_file: str
    total_files: int
    current_file_index: int
    current_chunk: int
    total_chunks: int
    percentage: float
    elapsed_time: float
    estimated_remaining: float
    status: str


@dataclass
class BatchProcessingState:
    """State for pause/resume functionality."""
    video_path: str
    total_chunks: int
    completed_chunks: List[int] = field(default_factory=list)
    chunk_results: Dict[int, ChunkResult] = field(default_factory=dict)
    start_time: float = 0.0
    paused_time: float = 0.0


class TranscriptionError(Exception):
    """Raised when transcription fails."""
    pass


class BatchProcessor:
    """
    Processes long audio files by splitting into chunks and processing in parallel.
    
    Handles progress tracking, pause/resume, and merging results.
    """
    
    def __init__(
        self,
        chunk_duration: float = 30.0,
        overlap_duration: float = 1.0,
        max_parallel: int = 2,
        sample_rate: int = 16000
    ):
        """
        Initialize batch processor.
        
        Args:
            chunk_duration: Duration of each chunk in seconds (default: 30.0)
            overlap_duration: Overlap between chunks in seconds (default: 1.0)
            overlap_duration: Overlap between chunks in seconds (default: 1.0)
            max_parallel: Maximum number of parallel chunks to process (default: 2)
            sample_rate: Audio sample rate in Hz (default: 16000)
        """
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration
        self.max_parallel = max_parallel
        self.sample_rate = sample_rate
        
        # Processing state
        self.status = ProcessingStatus.IDLE
        self.current_state: Optional[BatchProcessingState] = None
        self.pause_event = threading.Event()
        self.pause_event.set()  # Not paused initially
        
        # Thread pool for parallel processing
        self.executor: Optional[ThreadPoolExecutor] = None
        
        logger.info(
            f"BatchProcessor initialized: chunk_duration={chunk_duration}s, "
            f"overlap={overlap_duration}s, max_parallel={max_parallel}"
        )
    
    def split_audio_into_chunks(
        self,
        audio: np.ndarray,
        video_duration: Optional[float] = None
    ) -> List[AudioChunk]:
        """
        Split audio into overlapping chunks.
        
        Args:
            audio: Audio data (numpy array, float32, mono)
            video_duration: Total video duration for timestamp calculation
            
        Returns:
            List of AudioChunk objects
        """
        if len(audio) == 0:
            logger.warning("Empty audio provided to split_audio_into_chunks")
            return []
        
        # Calculate chunk size in samples
        chunk_samples = int(self.chunk_duration * self.sample_rate)
        overlap_samples = int(self.overlap_duration * self.sample_rate)
        step_samples = chunk_samples - overlap_samples
        
        chunks = []
        chunk_index = 0
        position = 0
        
        while position < len(audio):
            # Calculate chunk boundaries
            chunk_start = position
            chunk_end = min(position + chunk_samples, len(audio))
            
            # Extract chunk
            chunk_audio = audio[chunk_start:chunk_end]
            
            # Calculate time boundaries
            start_time = chunk_start / self.sample_rate
            end_time = chunk_end / self.sample_rate
            
            # Calculate overlap regions
            overlap_start = self.overlap_duration if chunk_index > 0 else 0.0
            overlap_end = self.overlap_duration if chunk_end < len(audio) else 0.0
            
            # Create chunk
            chunk = AudioChunk(
                index=chunk_index,
                audio=chunk_audio,
                start_time=start_time,
                end_time=end_time,
                overlap_start=overlap_start,
                overlap_end=overlap_end
            )
            chunks.append(chunk)
            
            # Move to next chunk
            position += step_samples
            chunk_index += 1
            
            # Stop if we've reached the end
            if chunk_end >= len(audio):
                break
        
        logger.info(
            f"Split audio into {len(chunks)} chunks: "
            f"duration={len(audio) / self.sample_rate:.2f}s, "
            f"chunk_size={self.chunk_duration}s, overlap={self.overlap_duration}s"
        )
        
        return chunks
    
    def process_chunk(
        self,
        chunk: AudioChunk,
        transcribe_fn: Callable[[np.ndarray], TranscriptionResult]
    ) -> ChunkResult:
        """
        Process a single audio chunk.
        
        Args:
            chunk: AudioChunk to process
            transcribe_fn: Function to transcribe audio
            
        Returns:
            ChunkResult with transcription and metadata
        """
        # Wait if paused
        self.pause_event.wait()
        
        start_time = time.time()
        
        try:
            # Transcribe chunk
            transcription = transcribe_fn(chunk.audio)
            
            # Adjust segment timestamps to absolute time
            adjusted_segments = []
            for segment in transcription.segments:
                adjusted_segment = TranscriptionSegment(
                    text=segment.text,
                    start_time=segment.start_time + chunk.start_time,
                    end_time=segment.end_time + chunk.start_time,
                    confidence=segment.confidence
                )
                adjusted_segments.append(adjusted_segment)
            
            # Create adjusted transcription result
            adjusted_transcription = TranscriptionResult(
                text=transcription.text,
                confidence=transcription.confidence,
                language=transcription.language,
                segments=adjusted_segments,
                processing_time=transcription.processing_time
            )
            
            processing_time = time.time() - start_time
            
            result = ChunkResult(
                chunk_index=chunk.index,
                transcription=adjusted_transcription,
                start_time=chunk.start_time,
                end_time=chunk.end_time,
                processing_time=processing_time
            )
            
            logger.debug(
                f"Chunk {chunk.index} processed: "
                f"text_length={len(transcription.text)}, "
                f"time={processing_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing chunk {chunk.index}: {str(e)}")
            raise TranscriptionError(
                f"Failed to process chunk {chunk.index}: {str(e)}"
            ) from e

    
    def process_audio(
        self,
        audio: np.ndarray,
        video_path: str,
        transcribe_fn: Callable[[np.ndarray], TranscriptionResult],
        progress_callback: Optional[Callable[[ProcessingProgress], None]] = None
    ) -> TranscriptionResult:
        """
        Process audio by splitting into chunks and transcribing.
        
        Args:
            audio: Audio data (numpy array, float32, mono)
            video_path: Path to video file (for state tracking)
            transcribe_fn: Function to transcribe audio chunks
            progress_callback: Optional callback for progress updates
            
        Returns:
            Complete TranscriptionResult with merged segments
        """
        if len(audio) == 0:
            logger.warning("Empty audio provided to process_audio")
            return TranscriptionResult(
                text="",
                confidence=0.0,
                language="vi",
                segments=[],
                processing_time=0.0
            )
        
        # Update status
        self.status = ProcessingStatus.PROCESSING
        
        # Split into chunks
        chunks = self.split_audio_into_chunks(audio)
        
        if len(chunks) == 0:
            logger.warning("No chunks created from audio")
            return TranscriptionResult(
                text="",
                confidence=0.0,
                language="vi",
                segments=[],
                processing_time=0.0
            )
        
        # Initialize state
        self.current_state = BatchProcessingState(
            video_path=video_path,
            total_chunks=len(chunks),
            start_time=time.time()
        )
        
        # Process chunks
        chunk_results = []
        start_time = time.time()
        
        try:
            # Create thread pool
            self.executor = ThreadPoolExecutor(max_workers=self.max_parallel)
            
            # Submit all chunks for processing
            futures: Dict[Future, AudioChunk] = {}
            for chunk in chunks:
                future = self.executor.submit(
                    self.process_chunk,
                    chunk,
                    transcribe_fn
                )
                futures[future] = chunk
            
            # Collect results as they complete
            completed = 0
            for future in futures:
                # Wait if paused
                self.pause_event.wait()
                
                try:
                    result = future.result()
                    chunk_results.append(result)
                    
                    # Update state
                    if self.current_state:
                        self.current_state.completed_chunks.append(result.chunk_index)
                        self.current_state.chunk_results[result.chunk_index] = result
                    
                    completed += 1
                    
                    # Report progress
                    if progress_callback:
                        elapsed = time.time() - start_time
                        percentage = (completed / len(chunks)) * 100
                        
                        # Estimate remaining time
                        if completed > 0:
                            avg_time_per_chunk = elapsed / completed
                            remaining_chunks = len(chunks) - completed
                            estimated_remaining = avg_time_per_chunk * remaining_chunks
                        else:
                            estimated_remaining = 0.0
                        
                        progress = ProcessingProgress(
                            current_file=video_path,
                            total_files=1,
                            current_file_index=0,
                            current_chunk=completed,
                            total_chunks=len(chunks),
                            percentage=percentage,
                            elapsed_time=elapsed,
                            estimated_remaining=estimated_remaining,
                            status=self.status.value
                        )
                        progress_callback(progress)
                    
                except Exception as e:
                    logger.error(f"Chunk processing failed: {str(e)}")
                    # Continue with other chunks
                    continue
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            self.executor = None
            
            # Sort results by chunk index
            chunk_results.sort(key=lambda r: r.chunk_index)
            
            # Merge results
            merged_result = self.merge_chunk_results(chunk_results)
            
            # Update status
            self.status = ProcessingStatus.COMPLETED
            
            processing_time = time.time() - start_time
            logger.info(
                f"Audio processing completed: "
                f"chunks={len(chunks)}, time={processing_time:.2f}s"
            )
            
            return merged_result
            
        except Exception as e:
            self.status = ProcessingStatus.ERROR
            logger.error(f"Audio processing failed: {str(e)}")
            raise TranscriptionError(f"Failed to process audio: {str(e)}") from e
        
        finally:
            # Cleanup
            if self.executor:
                self.executor.shutdown(wait=False)
                self.executor = None
    
    def merge_chunk_results(
        self,
        chunk_results: List[ChunkResult]
    ) -> TranscriptionResult:
        """
        Merge results from multiple chunks into a single result.
        
        Handles overlapping segments and ensures continuous timestamps.
        
        Args:
            chunk_results: List of ChunkResult objects (sorted by chunk_index)
            
        Returns:
            Merged TranscriptionResult
        """
        if not chunk_results:
            return TranscriptionResult(
                text="",
                confidence=0.0,
                language="vi",
                segments=[],
                processing_time=0.0
            )
        
        # Collect all segments
        all_segments = []
        for result in chunk_results:
            all_segments.extend(result.transcription.segments)
        
        # Remove duplicate segments in overlap regions
        merged_segments = self._remove_overlapping_segments(all_segments)
        
        # Ensure timestamps are continuous
        merged_segments = self._ensure_continuous_timestamps(merged_segments)
        
        # Combine text
        text_parts = [seg.text for seg in merged_segments if seg.text.strip()]
        full_text = " ".join(text_parts)
        
        # Calculate average confidence
        if merged_segments:
            avg_confidence = sum(seg.confidence for seg in merged_segments) / len(merged_segments)
        else:
            avg_confidence = 0.0
        
        # Get language from first result
        language = chunk_results[0].transcription.language if chunk_results else "vi"
        
        # Calculate total processing time
        total_processing_time = sum(r.processing_time for r in chunk_results)
        
        logger.info(
            f"Merged {len(chunk_results)} chunks into {len(merged_segments)} segments"
        )
        
        return TranscriptionResult(
            text=full_text,
            confidence=avg_confidence,
            language=language,
            segments=merged_segments,
            processing_time=total_processing_time
        )
    
    def _remove_overlapping_segments(
        self,
        segments: List[TranscriptionSegment]
    ) -> List[TranscriptionSegment]:
        """
        Remove duplicate segments in overlap regions.
        
        Args:
            segments: List of segments (may contain overlaps)
            
        Returns:
            List of segments without overlaps
        """
        if not segments:
            return []
        
        # Sort by start time
        sorted_segments = sorted(segments, key=lambda s: s.start_time)
        
        # Remove duplicates
        unique_segments = []
        last_end_time = -1.0
        
        for segment in sorted_segments:
            # Skip if this segment is within the overlap region of previous segment
            if segment.start_time < last_end_time - 0.1:  # 0.1s tolerance
                # Check if it's a duplicate (similar text)
                if unique_segments and self._is_similar_text(
                    segment.text,
                    unique_segments[-1].text
                ):
                    continue
            
            unique_segments.append(segment)
            last_end_time = segment.end_time
        
        return unique_segments
    
    def _is_similar_text(self, text1: str, text2: str) -> bool:
        """
        Check if two text strings are similar (for duplicate detection).
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            True if texts are similar
        """
        # Simple similarity check: same words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return False
        
        # Calculate Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        similarity = intersection / union if union > 0 else 0.0
        
        return similarity > 0.7  # 70% similarity threshold
    
    def _ensure_continuous_timestamps(
        self,
        segments: List[TranscriptionSegment]
    ) -> List[TranscriptionSegment]:
        """
        Ensure timestamps are monotonically increasing without gaps.
        
        Args:
            segments: List of segments
            
        Returns:
            List of segments with adjusted timestamps
        """
        if not segments:
            return []
        
        adjusted_segments = []
        
        for i, segment in enumerate(segments):
            if i == 0:
                # First segment - keep as is
                adjusted_segments.append(segment)
            else:
                prev_segment = adjusted_segments[-1]
                
                # Check for gap or overlap
                if segment.start_time < prev_segment.end_time:
                    # Overlap - adjust start time
                    adjusted_start = prev_segment.end_time
                    adjusted_end = max(adjusted_start + 0.1, segment.end_time)
                    
                    adjusted_segment = TranscriptionSegment(
                        text=segment.text,
                        start_time=adjusted_start,
                        end_time=adjusted_end,
                        confidence=segment.confidence
                    )
                    adjusted_segments.append(adjusted_segment)
                    
                elif segment.start_time > prev_segment.end_time + 1.0:
                    # Large gap (> 1 second) - keep as is
                    adjusted_segments.append(segment)
                    
                else:
                    # Small gap or continuous - keep as is
                    adjusted_segments.append(segment)
        
        return adjusted_segments
    
    def pause(self) -> None:
        """Pause processing."""
        if self.status == ProcessingStatus.PROCESSING:
            logger.info("Pausing batch processing")
            self.pause_event.clear()
            self.status = ProcessingStatus.PAUSED
            
            if self.current_state:
                self.current_state.paused_time = time.time()
    
    def resume(self) -> None:
        """Resume processing."""
        if self.status == ProcessingStatus.PAUSED:
            logger.info("Resuming batch processing")
            self.pause_event.set()
            self.status = ProcessingStatus.PROCESSING
            
            if self.current_state and self.current_state.paused_time > 0:
                # Adjust start time to account for pause duration
                pause_duration = time.time() - self.current_state.paused_time
                self.current_state.start_time += pause_duration
                self.current_state.paused_time = 0.0
    
    def get_progress(self) -> Optional[ProcessingProgress]:
        """
        Get current processing progress.
        
        Returns:
            ProcessingProgress object or None if not processing
        """
        if not self.current_state:
            return None
        
        elapsed = time.time() - self.current_state.start_time
        completed = len(self.current_state.completed_chunks)
        total = self.current_state.total_chunks
        
        percentage = (completed / total * 100) if total > 0 else 0.0
        
        # Estimate remaining time
        if completed > 0:
            avg_time_per_chunk = elapsed / completed
            remaining_chunks = total - completed
            estimated_remaining = avg_time_per_chunk * remaining_chunks
        else:
            estimated_remaining = 0.0
        
        return ProcessingProgress(
            current_file=self.current_state.video_path,
            total_files=1,
            current_file_index=0,
            current_chunk=completed,
            total_chunks=total,
            percentage=percentage,
            elapsed_time=elapsed,
            estimated_remaining=estimated_remaining,
            status=self.status.value
        )
    
    def process_batch(
        self,
        video_paths: List[str],
        extract_audio_fn: Callable[[str], np.ndarray],
        transcribe_fn: Callable[[np.ndarray], TranscriptionResult],
        progress_callback: Optional[Callable[[ProcessingProgress], None]] = None
    ) -> List[TranscriptionResult]:
        """
        Process multiple video files in batch.
        
        Args:
            video_paths: List of video file paths
            extract_audio_fn: Function to extract audio from video
            transcribe_fn: Function to transcribe audio
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of TranscriptionResult objects (one per video)
        """
        results = []
        total_files = len(video_paths)
        
        logger.info(f"Starting batch processing: {total_files} files")
        
        for file_index, video_path in enumerate(video_paths):
            try:
                logger.info(f"Processing file {file_index + 1}/{total_files}: {video_path}")
                
                # Extract audio
                audio = extract_audio_fn(video_path)
                
                # Process audio
                result = self.process_audio(
                    audio=audio,
                    video_path=video_path,
                    transcribe_fn=transcribe_fn,
                    progress_callback=progress_callback
                )
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to process {video_path}: {str(e)}")
                # Add error result
                results.append(TranscriptionResult(
                    text=f"[Error: {str(e)}]",
                    confidence=0.0,
                    language="vi",
                    segments=[],
                    processing_time=0.0
                ))
        
        logger.info(f"Batch processing completed: {len(results)}/{total_files} files")
        
        return results
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        if self.executor:
            self.executor.shutdown(wait=False)
            self.executor = None
        
        self.current_state = None
        self.status = ProcessingStatus.IDLE
        logger.info("BatchProcessor cleaned up")
