"""
Speech Analysis Coordinator Module

This module coordinates all speech analysis components including:
- Audio capture from microphone
- Speech quality analysis
- Speech-to-text conversion
- GUI updates via callbacks

The coordinator manages the lifecycle of all components, handles threading,
and provides a unified interface for speech analysis operations.
"""

import threading
import queue
import time
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Callable, Dict, Any, List, Tuple
from enum import Enum

from .audio_capture import AudioCapture, AudioConfig
from .quality_analyzer import SpeechQualityAnalyzer, QualityReport, QualityConfig
from .speech_to_text import VoskSTTEngine, STTConfig, TranscriptionResult, create_stt_engine, SpeechToTextEngine
from .text_storage import TextStorage, TranscriptMetadata
from .exceptions import (
    MicrophoneNotFoundError,
    MicrophoneInUseError,
    AudioDriverError,
    ModelLoadError,
    OutOfMemoryError
)

# Enhanced components (optional imports)
try:
    from .audio_cleaner import AudioCleaner
    AUDIO_CLEANER_AVAILABLE = True
except ImportError:
    AUDIO_CLEANER_AVAILABLE = False

try:
    from .enhanced_vad import EnhancedVADDetector
    ENHANCED_VAD_AVAILABLE = True
except ImportError:
    ENHANCED_VAD_AVAILABLE = False

try:
    from .overlap_handler import OverlapHandler
    OVERLAP_HANDLER_AVAILABLE = True
except ImportError:
    OVERLAP_HANDLER_AVAILABLE = False

try:
    from .whisper_stt_engine import WhisperSTTEngine
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    from .openai_whisper_stt_engine import OpenAIWhisperSTTEngine
    OPENAI_WHISPER_AVAILABLE = True
except ImportError:
    OPENAI_WHISPER_AVAILABLE = False

try:
    from .custom_vocabulary import CustomVocabularyProcessor
    CUSTOM_VOCABULARY_AVAILABLE = True
except ImportError:
    CUSTOM_VOCABULARY_AVAILABLE = False

try:
    from .config import EnhancedSTTConfig
    ENHANCED_CONFIG_AVAILABLE = True
except ImportError:
    ENHANCED_CONFIG_AVAILABLE = False

try:
    from .performance_monitor import PerformanceMonitor
    PERFORMANCE_MONITOR_AVAILABLE = True
except ImportError:
    PERFORMANCE_MONITOR_AVAILABLE = False


class EventType(Enum):
    """Types of events that can be emitted by the coordinator."""
    TRANSCRIPTION = "transcription"
    QUALITY = "quality"
    ERROR = "error"
    STATUS = "status"


@dataclass
class SpeechEvent:
    """
    Base event from speech analysis.
    
    Attributes:
        event_type: Type of event (transcription, quality, error, status)
        timestamp: When the event occurred
        data: Event-specific data dictionary
    """
    event_type: EventType
    timestamp: datetime
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TranscriptionEvent:
    """
    Event for new transcription text.
    
    Attributes:
        text: The transcribed text
        confidence: Confidence score (0-1)
        timestamp: When the transcription occurred
        segment_start: Start time of the audio segment
        segment_end: End time of the audio segment
    """
    text: str
    confidence: float
    timestamp: datetime
    segment_start: float
    segment_end: float


@dataclass
class QualityEvent:
    """
    Event for quality score updates.
    
    Attributes:
        clarity_score: Clarity score (0-100)
        fluency_score: Fluency score (0-100)
        snr: Signal-to-Noise Ratio in dB
        speech_rate: Speech rate in syllables per second
        timestamp: When the quality was measured
    """
    clarity_score: float
    fluency_score: float
    snr: float
    speech_rate: float
    timestamp: datetime


@dataclass
class SessionData:
    """
    Data for a speech analysis session.
    
    Attributes:
        session_id: Unique identifier for the session
        start_time: When the session started
        end_time: When the session ended (None if still active)
        transcript: Accumulated transcript text
        quality_reports: List of quality reports during the session
        metadata: Session metadata
    """
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    transcript: str = ""
    quality_reports: List[QualityReport] = field(default_factory=list)
    metadata: Optional[TranscriptMetadata] = None


class SpeechAnalysisCoordinator:
    """
    Coordinates all speech analysis components.
    
    This class manages:
    - Audio capture lifecycle
    - Quality analysis processing
    - Speech-to-text conversion
    - Threading and synchronization
    - Event callbacks to GUI
    - Session management
    """
    
    def __init__(
        self,
        audio_config: Optional[AudioConfig] = None,
        stt_config: Optional[STTConfig] = None,
        quality_config: Optional[QualityConfig] = None,
        gui_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None
    ):
        """
        Initialize the speech analysis coordinator.
        
        Args:
            audio_config: Configuration for audio capture
            stt_config: Configuration for speech-to-text (can be STTConfig or EnhancedSTTConfig)
            quality_config: Configuration for quality analysis
            gui_callback: Callback function(event_type, data) for GUI updates
        """
        # Configurations
        self.audio_config = audio_config or AudioConfig()
        self.stt_config = stt_config or STTConfig()
        self.quality_config = quality_config or QualityConfig()
        
        # Check if enhanced mode is enabled
        self.use_enhanced = self._is_enhanced_mode()
        
        # GUI callback
        self.gui_callback = gui_callback
        
        # Components (initialized on start)
        self.audio_capture: Optional[AudioCapture] = None
        self.quality_analyzer: Optional[SpeechQualityAnalyzer] = None
        self.stt_engine: Optional[SpeechToTextEngine] = None
        self.text_storage: Optional[TextStorage] = None
        
        # Enhanced components (optional)
        self.audio_cleaner: Optional[AudioCleaner] = None
        self.vad_detector: Optional[EnhancedVADDetector] = None
        self.overlap_handler: Optional[OverlapHandler] = None
        self.vocabulary_processor: Optional[CustomVocabularyProcessor] = None
        self.performance_monitor: Optional[PerformanceMonitor] = None
        
        # Threading components
        self._audio_thread: Optional[threading.Thread] = None
        self._processing_thread: Optional[threading.Thread] = None
        self._gui_update_thread: Optional[threading.Thread] = None
        
        # Queues for inter-thread communication
        self._audio_queue: queue.Queue = queue.Queue(maxsize=100)
        self._result_queue: queue.Queue = queue.Queue(maxsize=50)
        
        # Synchronization
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        
        # State
        self._is_running = False
        self._is_paused = False
        
        # Session data
        self._current_session: Optional[SessionData] = None
        self._transcript_parts: List[str] = []
        self._latest_quality: Optional[QualityReport] = None
        
        # Auto-save
        self._last_autosave_time = time.time()
        self._autosave_interval = 30.0  # seconds
    
    def start_analysis(self) -> bool:
        """
        Start speech analysis.
        
        This initializes all components and starts the processing threads.
        
        Returns:
            True if started successfully, False otherwise
        """
        if self._is_running:
            return True  # Already running
        
        try:
            # Initialize components
            self._initialize_components()
            
            # Create new session
            self._current_session = SessionData(
                session_id=self._generate_session_id(),
                start_time=datetime.now()
            )
            
            # Clear state
            self._transcript_parts.clear()
            self._latest_quality = None
            self._stop_event.clear()
            self._pause_event.clear()
            
            # Start audio capture
            if not self.audio_capture.start_recording():
                self._emit_error("Failed to start audio recording")
                return False
            
            # Start threads
            self._start_threads()
            
            # Update state
            self._is_running = True
            self._is_paused = False
            
            # Emit status event
            self._emit_status("started", "Speech analysis started")
            
            return True
            
        except MicrophoneNotFoundError as e:
            self._emit_error(f"Microphone not found: {str(e)}")
            return False
        except MicrophoneInUseError as e:
            self._emit_error(f"Microphone in use: {str(e)}")
            return False
        except ModelLoadError as e:
            self._emit_error(f"Failed to load STT model: {str(e)}")
            return False
        except Exception as e:
            self._emit_error(f"Failed to start analysis: {str(e)}")
            return False
    
    def stop_analysis(self) -> None:
        """
        Stop speech analysis and cleanup resources.
        
        This performs a graceful shutdown of all components.
        """
        if not self._is_running:
            return
        
        try:
            # Signal stop
            self._stop_event.set()
            self._is_running = False
            
            # Stop audio capture
            if self.audio_capture:
                self.audio_capture.stop_recording()
            
            # Wait for threads to finish (with timeout)
            self._wait_for_threads()
            
            # Stop performance monitoring and log statistics
            if self.performance_monitor:
                self.performance_monitor.log_statistics()
                self.performance_monitor.stop_monitoring()
            
            # Finalize session
            if self._current_session:
                self._current_session.end_time = datetime.now()
            
            # Emit status event
            self._emit_status("stopped", "Speech analysis stopped")
            
        except Exception as e:
            self._emit_error(f"Error during stop: {str(e)}")
    
    def _is_enhanced_mode(self) -> bool:
        """Check if enhanced mode is enabled based on config."""
        # First check for explicit use_enhanced attribute (takes priority)
        if hasattr(self.stt_config, 'use_enhanced'):
            return getattr(self.stt_config, 'use_enhanced', False)
        # Fallback: Check if config is EnhancedSTTConfig instance
        if ENHANCED_CONFIG_AVAILABLE:
            from .config import EnhancedSTTConfig
            if isinstance(self.stt_config, EnhancedSTTConfig):
                return True
        return False
    
    def _initialize_components(self) -> None:
        """Initialize all speech analysis components."""
        # Audio capture
        self.audio_capture = AudioCapture(self.audio_config)
        
        # Quality analyzer
        self.quality_analyzer = SpeechQualityAnalyzer(
            sample_rate=self.audio_config.sample_rate,
            config=self.quality_config
        )
        
        # Initialize STT engine based on mode
        if self.use_enhanced:
            self._initialize_enhanced_components()
        else:
            # Legacy mode: Vosk only
            self.stt_engine = create_stt_engine(self.stt_config)
        
        # Text storage
        self.text_storage = TextStorage()
    
    def _initialize_enhanced_components(self) -> None:
        """Initialize enhanced STT components."""
        # Performance monitor
        if PERFORMANCE_MONITOR_AVAILABLE:
            cpu_limit = getattr(self.stt_config, 'cpu_limit_percent', 50.0)
            memory_limit = getattr(self.stt_config, 'max_memory_mb', 500.0)
            min_rtf = getattr(self.stt_config, 'min_real_time_factor', 1.0)
            
            self.performance_monitor = PerformanceMonitor(
                cpu_limit_percent=cpu_limit,
                memory_limit_mb=memory_limit,
                min_real_time_factor=min_rtf
            )
            self.performance_monitor.start_monitoring()
        
        # Audio cleaner
        if AUDIO_CLEANER_AVAILABLE and getattr(self.stt_config, 'enable_audio_cleaning', True):
            self.audio_cleaner = AudioCleaner(sample_rate=self.audio_config.sample_rate)
        
        # Enhanced VAD detector
        if ENHANCED_VAD_AVAILABLE and getattr(self.stt_config, 'enable_vad', True):
            vad_method = getattr(self.stt_config, 'vad_method', 'silero')
            self.vad_detector = EnhancedVADDetector(
                sample_rate=self.audio_config.sample_rate,
                method=vad_method
            )
        
        # Overlap handler
        if OVERLAP_HANDLER_AVAILABLE:
            overlap_duration = getattr(self.stt_config, 'overlap_duration', 0.5)
            max_buffer_size = getattr(self.stt_config, 'max_buffer_size', 10)
            self.overlap_handler = OverlapHandler(
                sample_rate=self.audio_config.sample_rate,
                overlap_duration=overlap_duration,
                max_buffer_size=max_buffer_size
            )
        
        # STT Engine (Whisper is default, VOSK is fallback)
        model_type = getattr(self.stt_config, 'model_type', 'whisper')
        whisper_implementation = getattr(self.stt_config, 'whisper_implementation', 'faster-whisper')
        
        if model_type == 'whisper':
            # Choose between OpenAI Whisper and faster-whisper
            if whisper_implementation == 'openai' and OPENAI_WHISPER_AVAILABLE:
                try:
                    model_size = getattr(self.stt_config, 'model_size', 'base')
                    device = getattr(self.stt_config, 'device', 'cpu')
                    download_root = getattr(self.stt_config, 'download_root', None)
                    in_memory = getattr(self.stt_config, 'in_memory', False)
                    
                    self.stt_engine = OpenAIWhisperSTTEngine(
                        config=self.stt_config,
                        model_size=model_size,
                        device=device,
                        download_root=download_root,
                        in_memory=in_memory
                    )
                except Exception as e:
                    # Fallback to VOSK
                    if getattr(self.stt_config, 'fallback_to_vosk', True):
                        self._emit_error(f"OpenAI Whisper initialization failed, falling back to VOSK: {e}")
                        self.stt_engine = create_stt_engine(self.stt_config)
                    else:
                        raise
            elif WHISPER_AVAILABLE:
                # Use faster-whisper (default)
                try:
                    model_size = getattr(self.stt_config, 'model_size', 'base')
                    model_path = getattr(self.stt_config, 'model_path', None)
                    compute_type = getattr(self.stt_config, 'compute_type', 'int8')
                    device = getattr(self.stt_config, 'device', 'cpu')
                    
                    self.stt_engine = WhisperSTTEngine(
                        config=self.stt_config,
                        model_size=model_size,
                        model_path=model_path,
                        compute_type=compute_type,
                        device=device
                    )
                except Exception as e:
                    # Fallback to VOSK
                    if getattr(self.stt_config, 'fallback_to_vosk', True):
                        self._emit_error(f"faster-whisper initialization failed, falling back to VOSK: {e}")
                        self.stt_engine = create_stt_engine(self.stt_config)
                    else:
                        raise
            else:
                # No Whisper available, use VOSK
                self._emit_error("No Whisper implementation available, using VOSK")
                self.stt_engine = create_stt_engine(self.stt_config)
        else:
            # Use VOSK
            self.stt_engine = create_stt_engine(self.stt_config)
        
        # Custom vocabulary processor
        if CUSTOM_VOCABULARY_AVAILABLE and getattr(self.stt_config, 'enable_vocabulary', False):
            vocabulary_file = getattr(self.stt_config, 'vocabulary_file', None)
            if vocabulary_file:
                self.vocabulary_processor = CustomVocabularyProcessor(vocabulary_file=vocabulary_file)
    
    def _start_threads(self) -> None:
        """Start all processing threads."""
        # Audio thread
        self._audio_thread = threading.Thread(
            target=self._audio_loop,
            name="AudioThread",
            daemon=True
        )
        self._audio_thread.start()
        
        # Processing thread
        self._processing_thread = threading.Thread(
            target=self._processing_loop,
            name="ProcessingThread",
            daemon=True
        )
        self._processing_thread.start()
        
        # GUI update thread
        self._gui_update_thread = threading.Thread(
            target=self._gui_update_loop,
            name="GUIUpdateThread",
            daemon=True
        )
        self._gui_update_thread.start()
    
    def _wait_for_threads(self, timeout: float = 2.0) -> None:
        """Wait for all threads to finish."""
        threads = [
            self._audio_thread,
            self._processing_thread,
            self._gui_update_thread
        ]
        
        for thread in threads:
            if thread and thread.is_alive():
                thread.join(timeout=timeout)
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Emit an event to the GUI callback.
        
        Args:
            event_type: Type of event
            data: Event data dictionary
        """
        if self.gui_callback:
            try:
                self.gui_callback(event_type, data)
            except Exception as e:
                print(f"Error in GUI callback: {e}")
    
    def _emit_transcription(self, text: str, timestamp: datetime, confidence: float) -> None:
        """Emit a transcription event."""
        self._emit_event("transcription", {
            "text": text,
            "timestamp": timestamp,
            "confidence": confidence
        })
    
    def _emit_quality(self, quality_report: QualityReport) -> None:
        """Emit a quality update event."""
        self._emit_event("quality", {
            "clarity": quality_report.clarity_score,
            "fluency": quality_report.fluency_score,
            "snr": quality_report.snr,
            "speech_rate": quality_report.speech_rate,
            "timestamp": quality_report.timestamp
        })
    
    def _emit_error(self, message: str) -> None:
        """Emit an error event."""
        self._emit_event("error", {
            "message": message,
            "timestamp": datetime.now()
        })
    
    def _emit_status(self, status: str, message: str) -> None:
        """Emit a status event."""
        self._emit_event("status", {
            "status": status,
            "message": message,
            "timestamp": datetime.now()
        })
    
    def _audio_loop(self) -> None:
        """
        Audio capture loop (runs in separate thread).
        
        Captures audio chunks and puts them in the audio queue.
        """
        while not self._stop_event.is_set():
            try:
                # Check if paused
                if self._is_paused:
                    time.sleep(0.1)
                    continue
                
                # Get audio chunk from capture
                audio_chunk = self.audio_capture.get_audio_chunk(timeout=0.5)
                
                if audio_chunk is not None and len(audio_chunk) > 0:
                    # Put in queue for processing
                    try:
                        self._audio_queue.put(audio_chunk, timeout=0.1)
                    except queue.Full:
                        # Queue is full, skip this chunk
                        pass
                
            except Exception as e:
                self._emit_error(f"Audio capture error: {str(e)}")
                time.sleep(0.5)  # Wait before retrying
    
    def _processing_loop(self) -> None:
        """
        Processing loop (runs in separate thread).
        
        Processes audio chunks for quality analysis and speech-to-text.
        """
        audio_buffer = []
        last_quality_update = time.time()
        last_transcription_time = time.time()
        
        while not self._stop_event.is_set():
            try:
                # Check if paused
                if self._is_paused:
                    time.sleep(0.1)
                    continue
                
                # Get audio chunk from queue
                try:
                    audio_chunk = self._audio_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                # Apply audio cleaning if enabled
                if self.use_enhanced and self.audio_cleaner:
                    audio_chunk = self.audio_cleaner.clean_audio(audio_chunk)
                
                # Add to buffer
                audio_buffer.append(audio_chunk)
                
                # Calculate buffer duration
                total_samples = sum(len(chunk) for chunk in audio_buffer)
                buffer_duration = total_samples / self.audio_config.sample_rate
                
                # Quality analysis (every 2 seconds)
                current_time = time.time()
                if current_time - last_quality_update >= self.quality_config.update_interval:
                    if buffer_duration >= 1.0:  # Need at least 1 second
                        self._process_quality_analysis(audio_buffer)
                        last_quality_update = current_time
                
                # Speech-to-text (every 2-3 seconds)
                if buffer_duration >= 2.0:
                    if self.use_enhanced:
                        self._process_transcription_enhanced(audio_buffer, current_time)
                    else:
                        self._process_transcription(audio_buffer)
                    
                    last_transcription_time = current_time
                    
                    # Keep only last 0.5 seconds for overlap
                    overlap_samples = int(0.5 * self.audio_config.sample_rate)
                    audio = np.concatenate(audio_buffer)
                    if len(audio) > overlap_samples:
                        audio_buffer = [audio[-overlap_samples:]]
                    else:
                        audio_buffer = []
                
                # Auto-save check
                self._check_autosave()
                
            except Exception as e:
                self._emit_error(f"Processing error: {str(e)}")
                time.sleep(0.5)
    
    def _gui_update_loop(self) -> None:
        """
        GUI update loop (runs in separate thread).
        
        Gets results from the result queue and emits events to GUI.
        """
        while not self._stop_event.is_set():
            try:
                # Get result from queue
                try:
                    result = self._result_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                # Process result based on type
                result_type = result.get("type")
                
                if result_type == "transcription":
                    self._emit_transcription(
                        result["text"],
                        result["timestamp"],
                        result["confidence"]
                    )
                
                elif result_type == "quality":
                    self._emit_quality(result["report"])
                
            except Exception as e:
                self._emit_error(f"GUI update error: {str(e)}")
                time.sleep(0.5)
    
    def _process_quality_analysis(self, audio_buffer: List[np.ndarray]) -> None:
        """Process quality analysis for audio buffer."""
        try:
            # Concatenate audio
            audio = np.concatenate(audio_buffer)
            
            # Analyze quality
            quality_report = self.quality_analyzer.get_quality_report(audio)
            
            # Store latest quality
            with self._lock:
                self._latest_quality = quality_report
                if self._current_session:
                    self._current_session.quality_reports.append(quality_report)
            
            # Put in result queue
            self._result_queue.put({
                "type": "quality",
                "report": quality_report
            })
            
        except Exception as e:
            self._emit_error(f"Quality analysis error: {str(e)}")
    
    def _process_transcription(self, audio_buffer: List[np.ndarray]) -> None:
        """Process speech-to-text for audio buffer (legacy mode)."""
        try:
            # Concatenate audio
            audio = np.concatenate(audio_buffer)
            
            # Transcribe
            result = self.stt_engine.transcribe_chunk(audio)
            
            if result.text and result.text.strip():
                # Add to transcript
                with self._lock:
                    self._transcript_parts.append(result.text.strip())
                    if self._current_session:
                        self._current_session.transcript = " ".join(self._transcript_parts)
                
                # Put in result queue
                self._result_queue.put({
                    "type": "transcription",
                    "text": result.text.strip(),
                    "timestamp": datetime.now(),
                    "confidence": result.confidence
                })
            
        except Exception as e:
            self._emit_error(f"Transcription error: {str(e)}")
    
    def _process_transcription_enhanced(self, audio_buffer: List[np.ndarray], current_time: float) -> None:
        """Process speech-to-text with enhanced components."""
        try:
            # Concatenate audio
            audio = np.concatenate(audio_buffer)
            
            # Apply VAD if enabled
            if self.vad_detector:
                # Detect speech segments
                speech_segments = self.vad_detector.extract_speech_segments(audio)
                
                # If no speech detected, skip processing
                if not speech_segments or len(speech_segments) == 0:
                    return
                
                # Process each speech segment
                for segment_audio in speech_segments:
                    if len(segment_audio) > 0:
                        self._transcribe_segment(segment_audio, current_time)
            else:
                # No VAD, process entire audio
                self._transcribe_segment(audio, current_time)
            
        except Exception as e:
            self._emit_error(f"Enhanced transcription error: {str(e)}")
    
    def _transcribe_segment(self, audio: np.ndarray, timestamp: float) -> None:
        """Transcribe a single audio segment."""
        try:
            # Add to overlap handler if enabled
            if self.overlap_handler:
                self.overlap_handler.add_segment(audio, timestamp)
                
                # Get next segment to process
                segment_data = self.overlap_handler.get_next_segment()
                if segment_data is None:
                    return
                
                audio, segment_timestamp = segment_data
            else:
                segment_timestamp = timestamp
            
            # Record start time for performance monitoring
            start_time = time.time()
            
            # Transcribe
            result = self.stt_engine.transcribe_chunk(audio)
            
            # Record processing time for performance monitoring
            if self.performance_monitor:
                processing_time = time.time() - start_time
                audio_duration = len(audio) / self.audio_config.sample_rate
                self.performance_monitor.record_processing(audio_duration, processing_time)
            
            if result.text and result.text.strip():
                # Apply custom vocabulary if enabled
                text = result.text.strip()
                if self.vocabulary_processor:
                    text = self.vocabulary_processor.process_transcription(text)
                
                # Add to transcript
                with self._lock:
                    self._transcript_parts.append(text)
                    if self._current_session:
                        self._current_session.transcript = " ".join(self._transcript_parts)
                
                # Put in result queue
                self._result_queue.put({
                    "type": "transcription",
                    "text": text,
                    "timestamp": datetime.now(),
                    "confidence": result.confidence
                })
            
        except Exception as e:
            self._emit_error(f"Segment transcription error: {str(e)}")
    
    def _check_autosave(self) -> None:
        """Check if auto-save is needed."""
        current_time = time.time()
        if current_time - self._last_autosave_time >= self._autosave_interval:
            self._autosave()
            self._last_autosave_time = current_time
    
    def _autosave(self) -> None:
        """Perform auto-save of current session."""
        try:
            if not self._current_session or not self._current_session.transcript:
                return
            
            # Get current data
            transcript = self.get_transcript()
            clarity, fluency = self.get_current_quality()
            
            if not transcript:
                return
            
            # Create metadata
            duration = (datetime.now() - self._current_session.start_time).total_seconds()
            word_count = len(transcript.split())
            
            metadata = TranscriptMetadata(
                timestamp=self._current_session.start_time,
                duration=duration,
                word_count=word_count,
                language=self.stt_config.language,
                model_name="vosk",
                session_id=self._current_session.session_id
            )
            
            # Create quality report
            quality_report = QualityReport(
                clarity_score=clarity,
                fluency_score=fluency,
                snr=self._latest_quality.snr if self._latest_quality else 0.0,
                speech_rate=self._latest_quality.speech_rate if self._latest_quality else 0.0,
                pause_count=self._latest_quality.pause_count if self._latest_quality else 0,
                avg_pause_duration=self._latest_quality.avg_pause_duration if self._latest_quality else 0.0,
                recommendations=[]
            )
            
            # Save temp file
            self.text_storage.save_temp_session(transcript, metadata, quality_report)
            
        except Exception as e:
            # Silently fail for auto-save
            pass

    
    # ==================== Public API Methods ====================
    
    def pause_analysis(self) -> None:
        """
        Pause speech analysis.
        
        Audio capture continues but processing is paused.
        State is maintained so analysis can be resumed.
        """
        if not self._is_running or self._is_paused:
            return
        
        self._is_paused = True
        self._pause_event.set()
        self._emit_status("paused", "Speech analysis paused")
    
    def resume_analysis(self) -> None:
        """
        Resume speech analysis after pause.
        
        Processing continues from where it was paused.
        """
        if not self._is_running or not self._is_paused:
            return
        
        self._is_paused = False
        self._pause_event.clear()
        self._emit_status("resumed", "Speech analysis resumed")
    
    def get_current_quality(self) -> Tuple[float, float]:
        """
        Get current quality scores.
        
        Returns:
            Tuple of (clarity_score, fluency_score)
        """
        with self._lock:
            if self._latest_quality:
                return (
                    self._latest_quality.clarity_score,
                    self._latest_quality.fluency_score
                )
            return (0.0, 0.0)
    
    def get_transcript(self) -> str:
        """
        Get the complete transcript text.
        
        Returns:
            Full transcript as a single string
        """
        with self._lock:
            return " ".join(self._transcript_parts)
    
    def save_session(self, filepath: Optional[str] = None) -> str:
        """
        Save the current session to a file.
        
        Args:
            filepath: Optional custom filepath (auto-generated if None)
        
        Returns:
            Path to the saved file
        
        Raises:
            FileStorageError: If saving fails
        """
        if not self._current_session:
            raise ValueError("No active session to save")
        
        # Get transcript
        transcript = self.get_transcript()
        
        if not transcript:
            raise ValueError("No transcript to save")
        
        # Calculate duration
        if self._current_session.end_time:
            duration = (self._current_session.end_time - self._current_session.start_time).total_seconds()
        else:
            duration = (datetime.now() - self._current_session.start_time).total_seconds()
        
        # Count words
        word_count = len(transcript.split())
        
        # Create metadata
        metadata = TranscriptMetadata(
            timestamp=self._current_session.start_time,
            duration=duration,
            word_count=word_count,
            language=self.stt_config.language,
            model_name="vosk",
            session_id=self._current_session.session_id
        )
        
        # Get average quality scores
        clarity, fluency = self._get_average_quality()
        
        # Create quality report
        quality_report = QualityReport(
            clarity_score=clarity,
            fluency_score=fluency,
            snr=self._latest_quality.snr if self._latest_quality else 0.0,
            speech_rate=self._latest_quality.speech_rate if self._latest_quality else 0.0,
            pause_count=self._latest_quality.pause_count if self._latest_quality else 0,
            avg_pause_duration=self._latest_quality.avg_pause_duration if self._latest_quality else 0.0,
            recommendations=self._latest_quality.recommendations if self._latest_quality else []
        )
        
        # Save to file
        saved_path = self.text_storage.save_transcript(
            transcript,
            metadata,
            quality_report,
            filepath
        )
        
        # Delete temp file if exists
        temp_files = self.text_storage.get_temp_sessions()
        for temp_file in temp_files:
            self.text_storage.delete_temp_session(temp_file)
        
        return saved_path
    
    def _get_average_quality(self) -> Tuple[float, float]:
        """
        Calculate average quality scores from all reports.
        
        Returns:
            Tuple of (avg_clarity, avg_fluency)
        """
        with self._lock:
            if not self._current_session or not self._current_session.quality_reports:
                return (0.0, 0.0)
            
            reports = self._current_session.quality_reports
            avg_clarity = sum(r.clarity_score for r in reports) / len(reports)
            avg_fluency = sum(r.fluency_score for r in reports) / len(reports)
            
            return (avg_clarity, avg_fluency)
    
    def is_running(self) -> bool:
        """
        Check if analysis is currently running.
        
        Returns:
            True if running, False otherwise
        """
        return self._is_running
    
    def is_paused(self) -> bool:
        """
        Check if analysis is currently paused.
        
        Returns:
            True if paused, False otherwise
        """
        return self._is_paused
    
    def get_session_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the current session.
        
        Returns:
            Dictionary with session information or None if no active session
        """
        if not self._current_session:
            return None
        
        duration = (datetime.now() - self._current_session.start_time).total_seconds()
        word_count = len(self.get_transcript().split())
        clarity, fluency = self.get_current_quality()
        
        return {
            "session_id": self._current_session.session_id,
            "start_time": self._current_session.start_time,
            "duration": duration,
            "word_count": word_count,
            "clarity_score": clarity,
            "fluency_score": fluency,
            "is_running": self._is_running,
            "is_paused": self._is_paused
        }
    
    def get_performance_statistics(self) -> Optional[Dict[str, Any]]:
        """
        Get performance statistics from the performance monitor.
        
        Returns:
            Dictionary with performance statistics or None if not available
        """
        if self.performance_monitor:
            return self.performance_monitor.get_performance_report()
        return None
    
    def cleanup(self) -> None:
        """
        Cleanup all resources.
        
        This should be called when the coordinator is no longer needed.
        """
        # Stop analysis if running
        if self._is_running:
            self.stop_analysis()
        
        # Cleanup components
        if self.audio_capture:
            try:
                self.audio_capture.stop_recording()
            except Exception:
                pass
        
        # Clear queues
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break
        
        while not self._result_queue.empty():
            try:
                self._result_queue.get_nowait()
            except queue.Empty:
                break
    
    def __del__(self):
        """Cleanup on deletion."""
        self.cleanup()
