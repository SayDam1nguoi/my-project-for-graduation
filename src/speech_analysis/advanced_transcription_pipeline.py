"""
Advanced Transcription Pipeline

Pipeline xử lý audio nhiều bước để đạt độ chính xác cao nhất.
Sử dụng kỹ thuật tương tự TurboScribe, AssemblyAI.
"""

import numpy as np
from typing import List, Tuple, Optional
import time


class AudioEnhancementPipeline:
    """
    Pipeline xử lý audio trước khi transcribe.
    
    Steps:
    1. Noise reduction (spectral subtraction)
    2. Volume normalization
    3. High-pass filter (remove low frequency noise)
    4. Dynamic range compression
    """
    
    def __init__(self, sample_rate: int = 16000):
        """
        Khởi tạo pipeline.
        
        Args:
            sample_rate: Sample rate của audio
        """
        self.sample_rate = sample_rate
    
    def enhance(self, audio: np.ndarray) -> np.ndarray:
        """
        Enhance audio quality.
        
        Args:
            audio: Audio data (float32, -1 to 1)
            
        Returns:
            Enhanced audio
        """
        # Convert to float if needed
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        
        # MINIMAL enhancement to avoid distortion
        # Only normalize volume, skip other processing
        audio = self._normalize_volume(audio)
        
        return audio
    
    def _reduce_noise(self, audio: np.ndarray) -> np.ndarray:
        """Reduce noise using spectral subtraction."""
        # Improved noise gate with adaptive threshold
        # Calculate noise floor from first 0.5 seconds
        noise_samples = min(int(0.5 * self.sample_rate), len(audio) // 4)
        if noise_samples > 0:
            noise_floor = np.percentile(np.abs(audio[:noise_samples]), 75)
            threshold = max(0.01, noise_floor * 1.5)
        else:
            threshold = 0.01
        
        # Apply soft noise gate
        mask = np.abs(audio) < threshold
        audio[mask] *= 0.05  # Reduce but don't eliminate
        return audio
    
    def _normalize_volume(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio volume with peak limiting."""
        # Simple peak normalization (safer than RMS)
        max_val = np.max(np.abs(audio))
        
        if max_val > 0:
            # Normalize to 0.9 to leave headroom
            audio = audio * (0.9 / max_val)
        
        return audio
    
    def _high_pass_filter(self, audio: np.ndarray, cutoff: float = 80) -> np.ndarray:
        """Apply high-pass filter."""
        # Simple first-order high-pass filter
        alpha = cutoff / (cutoff + self.sample_rate / (2 * np.pi))
        
        filtered = np.zeros_like(audio)
        filtered[0] = audio[0]
        
        for i in range(1, len(audio)):
            filtered[i] = alpha * (filtered[i-1] + audio[i] - audio[i-1])
        
        return filtered
    
    def _compress_dynamic_range(self, audio: np.ndarray, threshold: float = 0.5, ratio: float = 4.0) -> np.ndarray:
        """Apply dynamic range compression."""
        # Simple compressor
        mask = np.abs(audio) > threshold
        audio[mask] = np.sign(audio[mask]) * (threshold + (np.abs(audio[mask]) - threshold) / ratio)
        return audio


class VoiceActivityDetector:
    """
    Voice Activity Detection (VAD).
    
    Phát hiện phần nào có giọng nói, phần nào là silence.
    Chỉ transcribe phần có giọng nói để tăng độ chính xác.
    """
    
    def __init__(self, sample_rate: int = 16000):
        """
        Khởi tạo VAD.
        
        Args:
            sample_rate: Sample rate của audio
        """
        self.sample_rate = sample_rate
    
    def detect_speech_segments(self, audio: np.ndarray, min_speech_duration: float = 0.2) -> List[Tuple[float, float]]:
        """
        Phát hiện các đoạn có giọng nói với thuật toán cải tiến.
        
        Args:
            audio: Audio data
            min_speech_duration: Độ dài tối thiểu của speech segment (giây)
            
        Returns:
            List of (start_time, end_time) tuples
        """
        # Improved energy-based VAD with ZCR
        frame_length = int(0.025 * self.sample_rate)  # 25ms frames
        hop_length = int(0.010 * self.sample_rate)    # 10ms hop
        
        # Calculate energy and zero-crossing rate for each frame
        energies = []
        zcrs = []
        
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i+frame_length]
            
            # Energy
            energy = np.sum(frame ** 2)
            energies.append(energy)
            
            # Zero-crossing rate (indicates voicing)
            zcr = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * len(frame))
            zcrs.append(zcr)
        
        energies = np.array(energies)
        zcrs = np.array(zcrs)
        
        # Debug info
        print(f"[VAD] Energy stats: min={energies.min():.6f}, max={energies.max():.6f}, mean={energies.mean():.6f}")
        print(f"[VAD] ZCR stats: min={zcrs.min():.6f}, max={zcrs.max():.6f}, mean={zcrs.mean():.6f}")
        
        # Adaptive threshold based on percentiles (lowered for better detection)
        energy_threshold = np.percentile(energies, 15)  # Lowered from 30 to 15
        zcr_threshold = np.percentile(zcrs, 70)  # Raised from 50 to 70
        
        print(f"[VAD] Energy threshold: {energy_threshold:.6f}")
        print(f"[VAD] ZCR threshold: {zcr_threshold:.6f}")
        
        # Combine energy and ZCR for better detection
        # More lenient: use OR instead of AND for better speech detection
        is_speech = (energies > energy_threshold) | (zcrs < zcr_threshold)
        
        speech_ratio = np.sum(is_speech) / len(is_speech)
        print(f"[VAD] Speech frames: {np.sum(is_speech)}/{len(is_speech)} ({speech_ratio*100:.1f}%)")
        
        # Smooth with median filter
        window = 5
        is_speech_smooth = np.zeros_like(is_speech)
        for i in range(len(is_speech)):
            start_idx = max(0, i - window // 2)
            end_idx = min(len(is_speech), i + window // 2 + 1)
            is_speech_smooth[i] = np.median(is_speech[start_idx:end_idx])
        
        # Find speech segments with padding
        segments = []
        start = None
        padding = 0.3  # 300ms padding (increased for better coverage)
        
        for i, speech in enumerate(is_speech_smooth):
            time = i * hop_length / self.sample_rate
            
            if speech and start is None:
                start = max(0, time - padding)
            elif not speech and start is not None:
                end = time + padding
                duration = end - start
                if duration >= min_speech_duration:
                    segments.append((start, min(end, len(audio) / self.sample_rate)))
                start = None
        
        # Add last segment if still active
        if start is not None:
            segments.append((start, len(audio) / self.sample_rate))
        
        # Merge close segments (more aggressive merging)
        if len(segments) > 1:
            merged = [segments[0]]
            for current in segments[1:]:
                prev = merged[-1]
                if current[0] - prev[1] < 1.0:  # Gap < 1000ms (increased from 500ms)
                    merged[-1] = (prev[0], current[1])
                else:
                    merged.append(current)
            segments = merged
        
        # If no segments found but audio exists, return full audio as one segment
        if len(segments) == 0 and len(audio) > 0:
            print("[VAD] No segments detected, returning full audio")
            segments = [(0.0, len(audio) / self.sample_rate)]
        
        return segments


class AdvancedTranscriptionPipeline:
    """
    Pipeline transcription nâng cao.
    
    Kết hợp:
    - Audio enhancement
    - VAD
    - Chunking strategy
    - Multiple passes
    - Post-processing
    """
    
    def __init__(self, stt_engine, sample_rate: int = 16000):
        """
        Khởi tạo pipeline.
        
        Args:
            stt_engine: STT engine (WhisperSTTEngine)
            sample_rate: Sample rate
        """
        self.stt_engine = stt_engine
        self.sample_rate = sample_rate
        
        # Initialize components
        self.audio_enhancer = AudioEnhancementPipeline(sample_rate)
        self.vad = VoiceActivityDetector(sample_rate)
    
    def transcribe(self, audio: np.ndarray, use_vad: bool = True, use_enhancement: bool = False, chunk_length: int = 20, use_llm: bool = False) -> str:
        """
        Transcribe audio với pipeline nâng cao (TurboScribe-style).
        
        Args:
            audio: Audio data (int16 or float32)
            use_vad: Sử dụng VAD để chỉ transcribe phần có giọng nói
            use_enhancement: Sử dụng audio enhancement
            chunk_length: Độ dài mỗi chunk (giây) - TurboScribe dùng 15-30s
            use_llm: Sử dụng LLM để sửa lỗi (cần API key)
            
        Returns:
            Transcript text
        """
        print("\n[Pipeline] Bắt đầu Advanced Transcription Pipeline (TurboScribe-style)...")
        
        # Convert to float if needed
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        
        # Step 1: Audio Enhancement (TurboScribe always enhances)
        print("[Pipeline] Step 1: Audio Enhancement...")
        enhanced_audio = self.audio_enhancer.enhance(audio)
        
        # Step 2: Chunk audio into 15-30s segments (TurboScribe method)
        print(f"[Pipeline] Step 2: Chunking audio into {chunk_length}s segments...")
        audio_duration = len(enhanced_audio) / self.sample_rate
        chunk_samples = chunk_length * self.sample_rate
        
        transcripts = []
        num_chunks = int(np.ceil(audio_duration / chunk_length))
        
        print(f"[Pipeline] Total duration: {audio_duration:.1f}s, Chunks: {num_chunks}")
        
        for i in range(num_chunks):
            start_sample = i * chunk_samples
            end_sample = min((i + 1) * chunk_samples, len(enhanced_audio))
            chunk_audio = enhanced_audio[start_sample:end_sample]
            
            print(f"[Pipeline] Transcribing chunk {i+1}/{num_chunks} ({start_sample/self.sample_rate:.1f}s - {end_sample/self.sample_rate:.1f}s)...")
            
            # Convert to int16 for STT
            chunk_audio_int16 = (chunk_audio * 32768.0).astype(np.int16)
            
            # Transcribe chunk
            result = self.stt_engine.transcribe_chunk(chunk_audio_int16)
            if result.text:
                transcripts.append(result.text.strip())
        
        # Combine all chunks
        full_transcript = " ".join(transcripts)
        
        # Step 3: VAD (optional, for filtering)
        if use_vad and False:  # Disabled when using chunking
            print("[Pipeline] Step 2: Voice Activity Detection...")
            speech_segments = self.vad.detect_speech_segments(enhanced_audio)
            print(f"[Pipeline] Phát hiện {len(speech_segments)} speech segments")
            
            if len(speech_segments) == 0:
                print("[Pipeline] ⚠️ VAD không phát hiện giọng nói!")
                print("[Pipeline] Chuyển sang transcribe toàn bộ audio...")
                # Fallback to full audio transcription
                enhanced_audio_int16 = (enhanced_audio * 32768.0).astype(np.int16)
                result = self.stt_engine.transcribe_chunk(enhanced_audio_int16)
                full_transcript = result.text
                
                # Post-processing
                print("[Pipeline] Step 3: Post-processing...")
                full_transcript = self._post_process(full_transcript)
                print("[Pipeline] ✓ Pipeline hoàn tất!")
                return full_transcript
            
            # Transcribe each segment
            transcripts = []
            for i, (start, end) in enumerate(speech_segments):
                print(f"[Pipeline] Transcribing segment {i+1}/{len(speech_segments)} ({start:.1f}s - {end:.1f}s)...")
                
                start_sample = int(start * self.sample_rate)
                end_sample = int(end * self.sample_rate)
                segment_audio = enhanced_audio[start_sample:end_sample]
                
                # Convert back to int16 for STT
                segment_audio_int16 = (segment_audio * 32768.0).astype(np.int16)
                
                # Transcribe
                result = self.stt_engine.transcribe_chunk(segment_audio_int16)
                if result.text:
                    transcripts.append(result.text.strip())
            
            # Combine transcripts
            full_transcript = " ".join(transcripts)
        else:
            # Transcribe full audio
            print("[Pipeline] Step 2: Transcribing full audio...")
            enhanced_audio_int16 = (enhanced_audio * 32768.0).astype(np.int16)
            result = self.stt_engine.transcribe_chunk(enhanced_audio_int16)
            full_transcript = result.text
        
        # Step 3: Post-processing
        print("[Pipeline] Step 3: Post-processing...")
        full_transcript = self._post_process(full_transcript)
        
        # Step 4: LLM Post-processing (optional)
        if use_llm:
            print("[Pipeline] Step 4: LLM Post-processing...")
            try:
                from src.speech_analysis.llm_post_processor import create_llm_post_processor
                llm = create_llm_post_processor(provider="gemini")
                if llm.is_available():
                    full_transcript = llm.post_process(full_transcript)
                else:
                    print("[Pipeline] ⚠️ LLM not available, skipping")
            except Exception as e:
                print(f"[Pipeline] ⚠️ LLM error: {e}")
        
        print("[Pipeline] ✓ Pipeline hoàn tất!")
        return full_transcript
    
    def _post_process(self, text: str) -> str:
        """
        Post-process transcript.
        
        Args:
            text: Raw transcript
            
        Returns:
            Cleaned transcript
        """
        if not text:
            return text
        
        # Remove extra spaces
        text = " ".join(text.split())
        
        # Detect hallucination - WARNING only (don't reject)
        if self._is_likely_hallucination(text):
            print("[Pipeline] ⚠️ WARNING: Possible hallucination detected")
            print("[Pipeline] Transcript may not be accurate")
            # Don't reject - let user decide
        
        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:]
        
        # Add period at end if missing
        if text and text[-1] not in ['.', '!', '?']:
            text += '.'
        
        return text
    
    def _is_likely_hallucination(self, text: str) -> bool:
        """
        Kiểm tra xem text có phải hallucination không.
        
        Args:
            text: Text to check
            
        Returns:
            True if likely hallucination
        """
        if not text or len(text) < 10:
            return False
        
        text_lower = text.lower()
        
        # STRICT: Check for common hallucination phrases
        # These phrases almost NEVER appear in real transcripts
        strict_hallucination_phrases = [
            "hẹn gặp lại",
            "subscribe",
            "đăng ký kênh", 
            "like và share",
            "cảm ơn đã xem",
            "xin chào các bạn",
            "chào mừng các bạn",
            "video tiếp theo",
            "những video tiếp theo",
        ]
        
        # If ANY of these phrases appear, it's hallucination
        for phrase in strict_hallucination_phrases:
            if phrase in text_lower:
                print(f"[Pipeline] ⚠️ Detected hallucination phrase: '{phrase}'")
                return True
        
        # Check for repetition
        words = text_lower.split()
        if len(words) > 5:
            half = len(words) // 2
            first_half = ' '.join(words[:half])
            second_half = ' '.join(words[half:half*2])
            if first_half == second_half:
                print(f"[Pipeline] ⚠️ Detected text repetition")
                return True
        
        # Check repetition ratio
        unique_words = len(set(words))
        if unique_words < len(words) * 0.4:  # Less than 40% unique words
            print(f"[Pipeline] ⚠️ Too many repeated words ({unique_words}/{len(words)})")
            return True
        
        return False


def create_advanced_pipeline(stt_engine):
    """
    Tạo advanced transcription pipeline.
    
    Args:
        stt_engine: STT engine
        
    Returns:
        AdvancedTranscriptionPipeline instance
    """
    return AdvancedTranscriptionPipeline(stt_engine)
