"""
Speech Clarity Analyzer

Phân tích độ rõ ràng trong lời nói dựa trên 5 yếu tố:
1. Speech Rate (Tốc độ nói) - 25%
2. Filler Words (Ngập ngừng) - 25%
3. Volume Stability (Ổn định âm lượng) - 15%
4. Pitch Stability (Ổn định giọng) - 10%
5. ASR Confidence (Độ rõ phát âm) - 25%

Thang điểm: 0-10
"""

import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import re


@dataclass
class SpeechClarityResult:
    """Kết quả phân tích độ rõ ràng."""
    clarity_score: float  # 0-10
    speech_rate_score: float  # 0-10
    filler_score: float  # 0-10
    volume_stability_score: float  # 0-10
    pitch_stability_score: float  # 0-10
    asr_confidence_score: float  # 0-10
    
    # Chi tiết
    wpm: float  # Words per minute
    filler_rate: float  # Tỷ lệ filler words
    filler_count: int
    total_words: int
    volume_std: float
    pitch_std: float
    asr_confidence: float
    
    # Phân loại
    clarity_level: str  # "Rất rõ ràng", "Rõ ràng", "Tạm ổn", "Khó nghe"
    issues: List[str]  # Các vấn đề phát hiện


class SpeechClarityAnalyzer:
    """
    Phân tích độ rõ ràng trong lời nói từ file audio.
    
    Công thức:
    ClarityScore = SR×25% + FS×25% + VS×15% + PS×10% + ASR×25%
    """
    
    # Filler words tiếng Việt
    FILLER_WORDS_VI = {
        'à', 'ừ', 'ờ', 'ừm', 'ờm', 'ừa', 'ờa',
        'thì', 'là', 'kiểu', 'kiểu như', 'kiểu như là',
        'ấy', 'nhỉ', 'nhé', 'nha', 'ha',
        'uh', 'um', 'er', 'ah', 'eh'
    }
    
    # Filler words tiếng Anh
    FILLER_WORDS_EN = {
        'um', 'uh', 'er', 'ah', 'like', 'you know',
        'i mean', 'sort of', 'kind of', 'basically',
        'actually', 'literally', 'right', 'okay', 'so'
    }
    
    def __init__(
        self,
        language: str = 'vi',
        optimal_wpm_range: Tuple[float, float] = (120, 160),
        max_pause_duration: float = 1.5
    ):
        """
        Initialize Speech Clarity Analyzer.
        
        Args:
            language: 'vi' hoặc 'en'
            optimal_wpm_range: Khoảng WPM tối ưu (120-160)
            max_pause_duration: Thời gian pause tối đa (giây)
        """
        self.language = language
        self.optimal_wpm_range = optimal_wpm_range
        self.max_pause_duration = max_pause_duration
        
        # Chọn filler words theo ngôn ngữ
        if language == 'vi':
            self.filler_words = self.FILLER_WORDS_VI
        else:
            self.filler_words = self.FILLER_WORDS_EN
    
    def analyze_audio_file(
        self,
        audio_path: str,
        transcript: str,
        asr_confidence: Optional[float] = None
    ) -> SpeechClarityResult:
        """
        Phân tích độ rõ ràng từ file audio.
        
        Args:
            audio_path: Đường dẫn file WAV
            transcript: Transcript của audio
            asr_confidence: Confidence từ ASR (0-1), optional
            
        Returns:
            SpeechClarityResult
        """
        # Load audio
        y, sr = librosa.load(audio_path, sr=None)
        duration = len(y) / sr
        
        # 1. Speech Rate Score
        speech_rate_score, wpm = self._calculate_speech_rate_score(
            transcript, duration
        )
        
        # 2. Filler Score
        filler_score, filler_rate, filler_count, total_words = \
            self._calculate_filler_score(transcript)
        
        # 3. Volume Stability Score
        volume_stability_score, volume_std = \
            self._calculate_volume_stability_score(y)
        
        # 4. Pitch Stability Score
        pitch_stability_score, pitch_std = \
            self._calculate_pitch_stability_score(y, sr)
        
        # 5. ASR Confidence Score
        asr_score, asr_conf = self._calculate_asr_confidence_score(
            asr_confidence
        )
        
        # Tính điểm tổng
        clarity_score = (
            speech_rate_score * 0.25 +
            filler_score * 0.25 +
            volume_stability_score * 0.15 +
            pitch_stability_score * 0.10 +
            asr_score * 0.25
        )
        
        # Phân loại
        clarity_level = self._classify_clarity(clarity_score)
        
        # Phát hiện vấn đề
        issues = self._detect_issues(
            speech_rate_score, filler_score, volume_stability_score,
            pitch_stability_score, asr_score, wpm, filler_rate
        )
        
        return SpeechClarityResult(
            clarity_score=clarity_score,
            speech_rate_score=speech_rate_score,
            filler_score=filler_score,
            volume_stability_score=volume_stability_score,
            pitch_stability_score=pitch_stability_score,
            asr_confidence_score=asr_score,
            wpm=wpm,
            filler_rate=filler_rate,
            filler_count=filler_count,
            total_words=total_words,
            volume_std=volume_std,
            pitch_std=pitch_std,
            asr_confidence=asr_conf,
            clarity_level=clarity_level,
            issues=issues
        )
    
    def _calculate_speech_rate_score(
        self,
        transcript: str,
        duration: float
    ) -> Tuple[float, float]:
        """
        Tính Speech Rate Score (0-10).
        
        Công thức:
        - 120-160 WPM: 10 điểm
        - Ngoài khoảng: giảm dần
        
        Returns:
            (score, wpm)
        """
        # Đếm số từ
        words = transcript.split()
        word_count = len(words)
        
        # Tính WPM
        if duration > 0:
            wpm = (word_count / duration) * 60
        else:
            wpm = 0
        
        # Tính điểm
        optimal_min, optimal_max = self.optimal_wpm_range
        optimal_center = (optimal_min + optimal_max) / 2
        
        if optimal_min <= wpm <= optimal_max:
            score = 10.0
        else:
            # Giảm điểm theo khoảng cách từ center
            deviation = abs(wpm - optimal_center)
            score = max(0, 10 - deviation / 10)
        
        return score, wpm
    
    def _calculate_filler_score(
        self,
        transcript: str
    ) -> Tuple[float, float, int, int]:
        """
        Tính Filler Score (0-10).
        
        Công thức:
        - ≤2%: 10 điểm
        - ≤5%: 7 điểm
        - ≤10%: 5 điểm
        - >10%: 3 điểm
        
        Returns:
            (score, filler_rate, filler_count, total_words)
        """
        # Lowercase và tách từ
        words = transcript.lower().split()
        total_words = len(words)
        
        if total_words == 0:
            return 10.0, 0.0, 0, 0
        
        # Đếm filler words
        filler_count = 0
        for word in words:
            # Loại bỏ dấu câu
            clean_word = re.sub(r'[^\w\s]', '', word)
            if clean_word in self.filler_words:
                filler_count += 1
        
        # Tính tỷ lệ
        filler_rate = filler_count / total_words
        
        # Tính điểm
        if filler_rate <= 0.02:  # ≤2%
            score = 10.0
        elif filler_rate <= 0.05:  # ≤5%
            score = 7.0
        elif filler_rate <= 0.10:  # ≤10%
            score = 5.0
        else:  # >10%
            score = max(0, 3.0 - (filler_rate - 0.10) * 10)
        
        return score, filler_rate, filler_count, total_words
    
    def _calculate_volume_stability_score(
        self,
        y: np.ndarray
    ) -> Tuple[float, float]:
        """
        Tính Volume Stability Score (0-10).
        
        Công thức điều chỉnh (dễ hơn):
        - StdVolume ≤0.15: 10 điểm (rất ổn định)
        - StdVolume ≤0.25: 8 điểm (ổn định)
        - StdVolume ≤0.35: 6 điểm (khá)
        - StdVolume >0.35: giảm dần
        
        Returns:
            (score, volume_std)
        """
        # Tính RMS energy cho mỗi frame
        frame_length = 2048
        hop_length = 512
        
        rms = librosa.feature.rms(
            y=y,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]
        
        # Tính độ lệch chuẩn
        volume_std = np.std(rms)
        
        # Tính điểm (dễ hơn)
        if volume_std <= 0.15:
            score = 10.0
        elif volume_std <= 0.25:
            score = 8.0
        elif volume_std <= 0.35:
            score = 6.0
        else:
            # Giảm dần từ 6 xuống 0
            score = max(0, 6.0 - (volume_std - 0.35) * 10)
        
        return score, float(volume_std)
    
    def _calculate_pitch_stability_score(
        self,
        y: np.ndarray,
        sr: int
    ) -> Tuple[float, float]:
        """
        Tính Pitch Stability Score (0-10).
        
        Công thức điều chỉnh (dễ hơn):
        - CV < 0.15: 10 điểm (rất ổn định)
        - CV < 0.25: 8 điểm (ổn định)
        - CV < 0.35: 6 điểm (khá)
        - CV >= 0.35: giảm dần
        
        Returns:
            (score, pitch_std)
        """
        # Trích xuất pitch (F0)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        
        # Lọc bỏ unvoiced frames
        f0_voiced = f0[~np.isnan(f0)]
        
        if len(f0_voiced) == 0:
            # Không có voiced frames -> cho điểm trung bình
            return 7.0, 0.0
        
        # Tính độ lệch chuẩn
        pitch_std = np.std(f0_voiced)
        
        # Normalize theo mean
        pitch_mean = np.mean(f0_voiced)
        if pitch_mean > 0:
            pitch_cv = pitch_std / pitch_mean  # Coefficient of variation
        else:
            pitch_cv = 0
        
        # Tính điểm (dễ hơn)
        if pitch_cv < 0.15:
            score = 10.0
        elif pitch_cv < 0.25:
            score = 8.0
        elif pitch_cv < 0.35:
            score = 6.0
        else:
            # Giảm dần từ 6 xuống 0
            score = max(0, 6.0 - (pitch_cv - 0.35) * 10)
        
        return score, float(pitch_std)
    
    def _calculate_asr_confidence_score(
        self,
        asr_confidence: Optional[float]
    ) -> Tuple[float, float]:
        """
        Tính ASR Confidence Score (0-10).
        
        Công thức điều chỉnh để phù hợp với Whisper confidence:
        - Confidence >= 0.85: 10 điểm (rất tốt)
        - Confidence >= 0.75: 8-9 điểm (tốt)
        - Confidence >= 0.65: 6-7 điểm (khá)
        - Confidence >= 0.50: 4-5 điểm (trung bình)
        - Confidence < 0.50: 0-3 điểm (kém)
        
        Returns:
            (score, confidence)
        """
        if asr_confidence is None:
            # Nếu không có confidence, raise error
            raise ValueError(
                "ASR confidence is required for clarity analysis. "
                "Please ensure the transcriber provides confidence scores."
            )
        
        # Clamp confidence vào [0, 1]
        conf = max(0.0, min(1.0, asr_confidence))
        
        # Tính điểm theo thang mới (dễ hơn)
        if conf >= 0.85:
            score = 10.0
        elif conf >= 0.75:
            # Linear interpolation: 0.75->8, 0.85->10
            score = 8.0 + (conf - 0.75) * 20
        elif conf >= 0.65:
            # Linear interpolation: 0.65->6, 0.75->8
            score = 6.0 + (conf - 0.65) * 20
        elif conf >= 0.50:
            # Linear interpolation: 0.50->4, 0.65->6
            score = 4.0 + (conf - 0.50) * 13.33
        else:
            # Linear: 0->0, 0.50->4
            score = conf * 8
        
        # Clamp score vào [0, 10]
        score = max(0.0, min(10.0, score))
        
        return score, conf
    
    def _classify_clarity(self, score: float) -> str:
        """Phân loại độ rõ ràng."""
        if score >= 8.5:
            return "Rất rõ ràng"
        elif score >= 7.0:
            return "Rõ ràng"
        elif score >= 5.0:
            return "Tạm ổn"
        else:
            return "Khó nghe"
    
    def _detect_issues(
        self,
        sr_score: float,
        filler_score: float,
        vol_score: float,
        pitch_score: float,
        asr_score: float,
        wpm: float,
        filler_rate: float
    ) -> List[str]:
        """Phát hiện các vấn đề (chỉ báo vấn đề nghiêm trọng)."""
        issues = []
        
        # Speech rate issues - chỉ báo khi nghiêm trọng
        if sr_score < 5.0:
            if wpm < 90:
                issues.append(f"Nói quá chậm ({wpm:.0f} WPM)")
            elif wpm > 200:
                issues.append(f"Nói quá nhanh ({wpm:.0f} WPM)")
        
        # Filler issues - chỉ báo khi quá nhiều
        if filler_score < 5.0:
            if filler_rate > 0.15:
                issues.append(f"Quá nhiều từ ngập ngừng ({filler_rate*100:.1f}%)")
        
        # Volume issues - chỉ báo khi rất không ổn định
        if vol_score < 5.0:
            issues.append("Âm lượng rất không ổn định")
        
        # Pitch issues - chỉ báo khi rất không ổn định
        if pitch_score < 5.0:
            issues.append("Giọng nói rất không ổn định")
        
        # ASR issues - chỉ báo khi confidence thấp
        if asr_score < 5.0:
            issues.append("Phát âm không rõ ràng")
        
        return issues
    
    def generate_report(self, result: SpeechClarityResult) -> str:
        """Tạo báo cáo chi tiết."""
        report = f"""
╔══════════════════════════════════════════════════════════╗
║           SPEECH CLARITY ANALYSIS REPORT                 ║
╚══════════════════════════════════════════════════════════╝

📊 TỔNG QUAN:
   Điểm tổng: {result.clarity_score:.2f}/10
   Xếp loại: {result.clarity_level}

📈 CHI TIẾT CÁC YẾU TỐ:

1. Tốc độ nói (25%): {result.speech_rate_score:.2f}/10
   - WPM: {result.wpm:.1f} words/minute
   - Tối ưu: 120-160 WPM

2. Từ ngập ngừng (25%): {result.filler_score:.2f}/10
   - Tỷ lệ: {result.filler_rate*100:.2f}%
   - Số lượng: {result.filler_count}/{result.total_words} từ

3. Ổn định âm lượng (15%): {result.volume_stability_score:.2f}/10
   - Độ lệch chuẩn: {result.volume_std:.4f}

4. Ổn định giọng (10%): {result.pitch_stability_score:.2f}/10
   - Độ lệch chuẩn: {result.pitch_std:.2f} Hz

5. Độ rõ phát âm (25%): {result.asr_confidence_score:.2f}/10
   - ASR Confidence: {result.asr_confidence:.2%}

"""
        
        if result.issues:
            report += "⚠️  CÁC VẤN ĐỀ PHÁT HIỆN:\n"
            for i, issue in enumerate(result.issues, 1):
                report += f"   {i}. {issue}\n"
        else:
            report += "✅ Không phát hiện vấn đề nghiêm trọng\n"
        
        report += "\n" + "="*60 + "\n"
        
        return report
