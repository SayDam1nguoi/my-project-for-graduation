"""Audio buffer with circular buffer management."""

import numpy as np
import threading
from collections import deque
from typing import Optional, List
from dataclasses import dataclass

from ..models.audio import AudioChunk


@dataclass
class BufferStats:
    """Statistics about the audio buffer."""
    
    current_size: int
    max_size: int
    utilization: float
    total_chunks_added: int
    total_chunks_retrieved: int
    overflow_count: int


class AudioBuffer:
    """
    Circular audio buffer with thread-safe operations.
    
    Features:
    - Fixed maximum size (circular buffer)
    - Thread-safe operations
    - Chunk-based storage
    - Statistics tracking
    """
    
    def __init__(
        self,
        max_duration: float = 30.0,
        sample_rate: int = 16000
    ):
        """
        Initialize audio buffer.
        
        Args:
            max_duration: Maximum buffer duration in seconds
            sample_rate: Sample rate of audio
        """
        self.max_duration = max_duration
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)
        
        # Circular buffer for samples
        self._buffer = deque(maxlen=self.max_samples)
        self._buffer_lock = threading.Lock()
        
        # Chunk storage
        self._chunks: List[AudioChunk] = []
        self._chunks_lock = threading.Lock()
        
        # Statistics
        self._total_chunks_added = 0
        self._total_chunks_retrieved = 0
        self._overflow_count = 0
    
    def add_chunk(self, chunk: AudioChunk) -> bool:
        """
        Add audio chunk to buffer.
        
        Args:
            chunk: AudioChunk to add
            
        Returns:
            True if added successfully, False if buffer is full
        """
        with self._buffer_lock:
            # Check if adding this chunk would overflow
            if len(self._buffer) + len(chunk.data) > self.max_samples:
                self._overflow_count += 1
            
            # Add samples to circular buffer
            self._buffer.extend(chunk.data)
            
            # Store chunk
            with self._chunks_lock:
                self._chunks.append(chunk)
                self._total_chunks_added += 1
            
            return True
    
    def get_latest_audio(self, duration: Optional[float] = None) -> np.ndarray:
        """
        Get latest audio from buffer.
        
        Args:
            duration: Duration in seconds (None = all available)
            
        Returns:
            Audio data as numpy array
        """
        with self._buffer_lock:
            if not self._buffer:
                return np.array([], dtype=np.float32)
            
            if duration is None:
                # Return all audio
                return np.array(self._buffer, dtype=np.float32)
            else:
                # Return last N seconds
                num_samples = int(duration * self.sample_rate)
                num_samples = min(num_samples, len(self._buffer))
                
                # Get last num_samples
                audio = np.array(list(self._buffer)[-num_samples:], dtype=np.float32)
                return audio
    
    def get_chunk(self, index: int) -> Optional[AudioChunk]:
        """
        Get chunk by index.
        
        Args:
            index: Chunk index
            
        Returns:
            AudioChunk if found, None otherwise
        """
        with self._chunks_lock:
            if 0 <= index < len(self._chunks):
                self._total_chunks_retrieved += 1
                return self._chunks[index]
            return None
    
    def get_all_chunks(self) -> List[AudioChunk]:
        """
        Get all stored chunks.
        
        Returns:
            List of all AudioChunks
        """
        with self._chunks_lock:
            return self._chunks.copy()
    
    def clear(self) -> None:
        """Clear the buffer."""
        with self._buffer_lock:
            self._buffer.clear()
        
        with self._chunks_lock:
            self._chunks.clear()
    
    def get_stats(self) -> BufferStats:
        """
        Get buffer statistics.
        
        Returns:
            BufferStats object
        """
        with self._buffer_lock:
            current_size = len(self._buffer)
            utilization = current_size / self.max_samples if self.max_samples > 0 else 0.0
        
        return BufferStats(
            current_size=current_size,
            max_size=self.max_samples,
            utilization=utilization,
            total_chunks_added=self._total_chunks_added,
            total_chunks_retrieved=self._total_chunks_retrieved,
            overflow_count=self._overflow_count
        )
    
    def get_duration(self) -> float:
        """
        Get current buffer duration in seconds.
        
        Returns:
            Duration in seconds
        """
        with self._buffer_lock:
            return len(self._buffer) / self.sample_rate
    
    def is_full(self) -> bool:
        """
        Check if buffer is full.
        
        Returns:
            True if buffer is at maximum capacity
        """
        with self._buffer_lock:
            return len(self._buffer) >= self.max_samples
    
    def is_empty(self) -> bool:
        """
        Check if buffer is empty.
        
        Returns:
            True if buffer is empty
        """
        with self._buffer_lock:
            return len(self._buffer) == 0
