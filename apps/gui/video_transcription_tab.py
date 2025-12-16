# -*- coding: utf-8 -*-
"""
Video Transcription Tab for GUI

Provides a user interface for video-to-text transcription with subtitle generation.
Implements Requirements 10.1, 10.2, 10.3, 10.4, 10.5
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
import threading
import time
from typing import Optional, Callable
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.video_analysis.video_transcription_coordinator import (
    VideoTranscriptionCoordinator,
    VideoTranscriptionResult,
    ProcessingProgress
)
from src.speech_analysis.whisper_stt_engine import WhisperSTTEngine
from src.speech_analysis.config import VideoTranscriptionConfig, STTConfig
from apps.gui.file_save_dialog import ask_save_file


class VideoTranscriptionTab:
    """
    GUI tab for video transcription functionality.
    
    Implements Requirements:
    - 10.1: File selection dialog for video input
    - 10.2: Real-time progress display
    - 10.3: Results display in scrollable text view
    - 10.4: Export buttons for subtitle files
    - 10.5: User-friendly error message display
    """
    
    def __init__(self, parent_frame: tk.Frame):
        """
        Initialize video transcription tab.
        
        Args:
            parent_frame: Parent frame to contain this tab
        """
        self.parent = parent_frame
        self.coordinator: Optional[VideoTranscriptionCoordinator] = None
        self.whisper_engine: Optional[WhisperSTTEngine] = None
        self.config: Optional[VideoTranscriptionConfig] = None
        
        # State variables
        self.selected_video_path: Optional[str] = None
        self.transcription_result: Optional[VideoTranscriptionResult] = None
        self.is_processing = False
        self.processing_thread: Optional[threading.Thread] = None
        
        # Create UI
        self.create_widgets()
        
        # Initialize components
        self.initialize_components()
    
    def create_widgets(self):
        """Create all UI widgets for the tab."""
        # Main container with padding
        main_container = tk.Frame(self.parent, bg='#1a1a1a')
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Configure grid
        main_container.grid_rowconfigure(0, weight=0)  # Header
        main_container.grid_rowconfigure(1, weight=0)  # File selection
        main_container.grid_rowconfigure(2, weight=0)  # Options
        main_container.grid_rowconfigure(3, weight=0)  # Progress
        main_container.grid_rowconfigure(4, weight=1)  # Results
        main_container.grid_rowconfigure(5, weight=0)  # Export buttons
        main_container.grid_columnconfigure(0, weight=1)
        
        # Header
        header_frame = tk.Frame(main_container, bg='#0d47a1', height=60)
        header_frame.grid(row=0, column=0, sticky='ew', pady=(0, 20))
        header_frame.pack_propagate(False)
        
        header_label = tk.Label(
            header_frame,
            text="📹 CHUYỂN ĐỔI VIDEO THÀNH VĂN BẢN",
            font=("Segoe UI", 18, "bold"),
            bg='#0d47a1',
            fg='#ffffff'
        )
        header_label.pack(pady=15)
        
        # File selection section (Requirement 10.1)
        file_frame = tk.Frame(main_container, bg='#252525', relief=tk.FLAT, bd=0)
        file_frame.grid(row=1, column=0, sticky='ew', pady=(0, 15))
        
        file_inner = tk.Frame(file_frame, bg='#252525')
        file_inner.pack(fill=tk.X, padx=20, pady=15)
        
        file_label = tk.Label(
            file_inner,
            text="📁 Chọn video file:",
            font=("Segoe UI", 11, "bold"),
            bg='#252525',
            fg='#90caf9'
        )
        file_label.pack(side=tk.LEFT, padx=(0, 15))
        
        self.browse_button = tk.Button(
            file_inner,
            text="📂 Chọn File...",
            font=("Segoe UI", 10, "bold"),
            bg='#455a64',
            fg='#ffffff',
            activebackground='#37474f',
            command=self.browse_video_file,
            cursor='hand2',
            relief=tk.FLAT,
            padx=20,
            pady=8
        )
        self.browse_button.pack(side=tk.LEFT, padx=10)
        
        self.file_path_label = tk.Label(
            file_inner,
            text="Chưa chọn file",
            font=("Segoe UI", 10),
            bg='#252525',
            fg='#9e9e9e',
            anchor='w'
        )
        self.file_path_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=15)
        
        # Options section
        options_frame = tk.Frame(main_container, bg='#252525', relief=tk.FLAT, bd=0)
        options_frame.grid(row=2, column=0, sticky='ew', pady=(0, 15))
        
        options_inner = tk.Frame(options_frame, bg='#252525')
        options_inner.pack(fill=tk.X, padx=20, pady=15)
        
        # Language selection
        lang_frame = tk.Frame(options_inner, bg='#252525')
        lang_frame.pack(side=tk.LEFT, padx=(0, 30))
        
        lang_label = tk.Label(
            lang_frame,
            text="🌐 Ngôn ngữ:",
            font=("Segoe UI", 10, "bold"),
            bg='#252525',
            fg='#81c784'
        )
        lang_label.pack(side=tk.LEFT, padx=(0, 10))
        
        self.language_var = tk.StringVar(value="auto")
        language_options = [
            ("Tự động phát hiện", "auto"),
            ("Tiếng Việt", "vi"),
            ("English", "en"),
            ("中文", "zh"),
            ("日本語", "ja"),
            ("한국어", "ko")
        ]
        
        self.language_combo = ttk.Combobox(
            lang_frame,
            textvariable=self.language_var,
            values=[opt[0] for opt in language_options],
            state='readonly',
            width=20,
            font=("Segoe UI", 10)
        )
        self.language_combo.pack(side=tk.LEFT)
        self.language_combo.current(0)
        
        # Store language mapping
        self.language_map = {opt[0]: opt[1] for opt in language_options}
        
        # Options checkboxes
        checkbox_frame = tk.Frame(options_inner, bg='#252525')
        checkbox_frame.pack(side=tk.LEFT, padx=30)
        
        self.diarization_var = tk.BooleanVar(value=False)
        self.diarization_check = tk.Checkbutton(
            checkbox_frame,
            text="👥 Phân biệt người nói",
            variable=self.diarization_var,
            font=("Segoe UI", 10),
            bg='#252525',
            fg='#e0e0e0',
            selectcolor='#1a1a1a',
            activebackground='#252525',
            activeforeground='#ffffff'
        )
        self.diarization_check.pack(side=tk.LEFT, padx=10)
        
        self.cache_var = tk.BooleanVar(value=True)
        self.cache_check = tk.Checkbutton(
            checkbox_frame,
            text="💾 Sử dụng cache",
            variable=self.cache_var,
            font=("Segoe UI", 10),
            bg='#252525',
            fg='#e0e0e0',
            selectcolor='#1a1a1a',
            activebackground='#252525',
            activeforeground='#ffffff'
        )
        self.cache_check.pack(side=tk.LEFT, padx=10)
        
        # Start button
        self.start_button = tk.Button(
            options_inner,
            text="▶️ BẮT ĐẦU CHUYỂN ĐỔI",
            font=("Segoe UI", 11, "bold"),
            bg='#43a047',
            fg='#ffffff',
            activebackground='#388e3c',
            command=self.start_transcription,
            cursor='hand2',
            relief=tk.FLAT,
            padx=25,
            pady=10,
            state=tk.DISABLED
        )
        self.start_button.pack(side=tk.RIGHT, padx=10)
        
        # Progress section (Requirement 10.2)
        progress_frame = tk.Frame(main_container, bg='#252525', relief=tk.FLAT, bd=0)
        progress_frame.grid(row=3, column=0, sticky='ew', pady=(0, 15))
        
        progress_inner = tk.Frame(progress_frame, bg='#252525')
        progress_inner.pack(fill=tk.X, padx=20, pady=15)
        
        self.progress_label = tk.Label(
            progress_inner,
            text="⏸️ Chưa bắt đầu",
            font=("Segoe UI", 10, "bold"),
            bg='#252525',
            fg='#ffa726',
            anchor='w'
        )
        self.progress_label.pack(fill=tk.X, pady=(0, 10))
        
        self.progress_bar = ttk.Progressbar(
            progress_inner,
            mode='determinate',
            length=400
        )
        self.progress_bar.pack(fill=tk.X, pady=(0, 5))
        
        self.progress_detail_label = tk.Label(
            progress_inner,
            text="",
            font=("Segoe UI", 9),
            bg='#252525',
            fg='#9e9e9e',
            anchor='w'
        )
        self.progress_detail_label.pack(fill=tk.X)
        
        # Results section (Requirement 10.3)
        results_frame = tk.Frame(main_container, bg='#252525', relief=tk.FLAT, bd=0)
        results_frame.grid(row=4, column=0, sticky='nsew', pady=(0, 15))
        
        results_label = tk.Label(
            results_frame,
            text="📄 Kết quả chuyển đổi:",
            font=("Segoe UI", 11, "bold"),
            bg='#252525',
            fg='#81c784'
        )
        results_label.pack(anchor='w', padx=20, pady=(15, 10))
        
        # Scrollable text view for results
        text_container = tk.Frame(results_frame, bg='#1a1a1a', relief=tk.SOLID, bd=1)
        text_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 15))
        
        self.results_text = scrolledtext.ScrolledText(
            text_container,
            font=("Consolas", 10),
            bg='#1e1e1e',
            fg='#e0e0e0',
            wrap=tk.WORD,
            state=tk.DISABLED,
            relief=tk.FLAT,
            padx=15,
            pady=15
        )
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Export buttons section (Requirement 10.4)
        export_frame = tk.Frame(main_container, bg='#252525', relief=tk.FLAT, bd=0)
        export_frame.grid(row=5, column=0, sticky='ew')
        
        export_inner = tk.Frame(export_frame, bg='#252525')
        export_inner.pack(fill=tk.X, padx=20, pady=15)
        
        export_label = tk.Label(
            export_inner,
            text="💾 Xuất kết quả:",
            font=("Segoe UI", 10, "bold"),
            bg='#252525',
            fg='#90caf9'
        )
        export_label.pack(side=tk.LEFT, padx=(0, 15))
        
        self.export_txt_button = tk.Button(
            export_inner,
            text="📄 Xuất Text",
            font=("Segoe UI", 10, "bold"),
            bg='#00897b',
            fg='#ffffff',
            activebackground='#00695c',
            command=self.export_text,
            cursor='hand2',
            relief=tk.FLAT,
            padx=20,
            pady=8,
            state=tk.DISABLED
        )
        self.export_txt_button.pack(side=tk.LEFT, padx=5)
    
    def initialize_components(self):
        """Initialize transcription components."""
        try:
            # Initialize STT configuration with correct parameters
            stt_config = STTConfig(
                language="vi",
                sample_rate=16000
            )
            
            # Initialize Whisper engine
            self.whisper_engine = WhisperSTTEngine(
                config=stt_config,
                model_size="large-v3",
                device="cpu",
                compute_type="int8"
            )
            
            # Initialize video transcription configuration
            self.config = VideoTranscriptionConfig()
            
            # Initialize coordinator
            self.coordinator = VideoTranscriptionCoordinator(
                whisper_engine=self.whisper_engine,
                config=self.config
            )
            
            self.update_status("✓ Sẵn sàng", "#4CAF50")
            
        except Exception as e:
            error_msg = f"Không thể khởi tạo: {str(e)}"
            self.show_error("Lỗi Khởi Tạo", error_msg)
            self.update_status("✗ Lỗi khởi tạo", "#f44336")
    
    def browse_video_file(self):
        """
        Open file dialog to select video file.
        Implements Requirement 10.1
        """
        file_path = filedialog.askopenfilename(
            title="Chọn video file",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.selected_video_path = file_path
            
            # Display shortened path
            path_obj = Path(file_path)
            display_name = path_obj.name
            if len(display_name) > 50:
                display_name = display_name[:47] + "..."
            
            self.file_path_label.config(
                text=display_name,
                fg='#4CAF50'
            )
            
            # Enable start button
            self.start_button.config(state=tk.NORMAL)
            
            self.update_status("✓ Đã chọn file - Sẵn sàng bắt đầu", "#4CAF50")
    
    def start_transcription(self):
        """Start video transcription process."""
        if not self.selected_video_path:
            self.show_error("Lỗi", "Vui lòng chọn file video trước!")
            return
        
        if self.is_processing:
            self.show_error("Đang xử lý", "Đang có quá trình chuyển đổi đang chạy!")
            return
        
        # Disable controls
        self.start_button.config(state=tk.DISABLED)
        self.browse_button.config(state=tk.DISABLED)
        self.language_combo.config(state=tk.DISABLED)
        self.diarization_check.config(state=tk.DISABLED)
        self.cache_check.config(state=tk.DISABLED)
        
        # Clear previous results
        self.clear_results()
        
        # Reset progress
        self.progress_bar['value'] = 0
        self.update_status("⏳ Đang xử lý...", "#2196F3")
        
        # Start processing in background thread
        self.is_processing = True
        self.processing_thread = threading.Thread(
            target=self.process_video,
            daemon=True
        )
        self.processing_thread.start()
    
    def process_video(self):
        """Process video in background thread."""
        try:
            # Get language selection
            lang_display = self.language_var.get()
            language = self.language_map.get(lang_display, "auto")
            if language == "auto":
                language = None  # Let coordinator auto-detect
            
            # Get options
            enable_diarization = self.diarization_var.get()
            
            # Update cache setting
            if self.coordinator:
                self.coordinator.config.enable_cache = self.cache_var.get()
            
            # Transcribe video with progress callback
            result = self.coordinator.transcribe_video(
                video_path=self.selected_video_path,
                language=language,
                enable_diarization=enable_diarization,
                progress_callback=self.on_progress_update
            )
            
            # Store result
            self.transcription_result = result
            
            # Update UI on main thread
            self.parent.after(0, lambda: self.on_transcription_complete(result))
            
        except Exception as e:
            # Show error on main thread
            error_msg = str(e)
            self.parent.after(0, lambda: self.on_transcription_error(error_msg))
    
    def on_progress_update(self, progress: ProcessingProgress):
        """
        Handle progress updates from coordinator.
        Implements Requirement 10.2
        """
        # Update progress bar
        percentage = progress.percentage
        self.parent.after(0, lambda: self.progress_bar.config(value=percentage))
        
        # Update progress label
        status_text = f"⏳ {progress.status} - {percentage:.1f}%"
        self.parent.after(0, lambda: self.progress_label.config(text=status_text))
        
        # Update detail label
        elapsed_min = progress.elapsed_time / 60
        remaining_min = progress.estimated_remaining / 60
        detail_text = (
            f"Chunk {progress.current_chunk}/{progress.total_chunks} | "
            f"Đã xử lý: {elapsed_min:.1f} phút | "
            f"Còn lại: {remaining_min:.1f} phút"
        )
        self.parent.after(0, lambda: self.progress_detail_label.config(text=detail_text))
    
    def on_transcription_complete(self, result: VideoTranscriptionResult):
        """
        Handle successful transcription completion.
        Implements Requirement 10.3
        """
        self.is_processing = False
        
        # Update status
        self.update_status("✓ Hoàn thành!", "#4CAF50")
        self.progress_bar['value'] = 100
        self.progress_label.config(text="✓ Hoàn thành!")
        
        # Display results
        self.display_results(result)
        
        # Enable export button
        self.export_txt_button.config(state=tk.NORMAL)
        
        # Re-enable controls
        self.start_button.config(state=tk.NORMAL)
        self.browse_button.config(state=tk.NORMAL)
        self.language_combo.config(state='readonly')
        self.diarization_check.config(state=tk.NORMAL)
        self.cache_check.config(state=tk.NORMAL)
        
        # Show completion message
        messagebox.showinfo(
            "Hoàn thành",
            f"Chuyển đổi video thành công!\n\n"
            f"Thời lượng: {result.duration:.1f}s\n"
            f"Ngôn ngữ: {result.language}\n"
            f"Độ chính xác: {result.confidence_avg*100:.1f}%\n"
            f"Thời gian xử lý: {result.processing_time:.1f}s"
        )
    
    def on_transcription_error(self, error_msg: str):
        """
        Handle transcription error.
        Implements Requirement 10.5
        """
        self.is_processing = False
        
        # Update status
        self.update_status("✗ Lỗi xử lý", "#f44336")
        self.progress_label.config(text="✗ Lỗi xử lý")
        
        # Re-enable controls
        self.start_button.config(state=tk.NORMAL)
        self.browse_button.config(state=tk.NORMAL)
        self.language_combo.config(state='readonly')
        self.diarization_check.config(state=tk.NORMAL)
        self.cache_check.config(state=tk.NORMAL)
        
        # Check if error is related to missing dependencies
        if "No video processing library available" in error_msg or "ffmpeg" in error_msg.lower():
            detailed_msg = (
                f"Không thể chuyển đổi video:\n\n{error_msg}\n\n"
                "GIẢI PHÁP:\n"
                "1. Cài đặt FFmpeg:\n"
                "   - Windows: choco install ffmpeg\n"
                "   - macOS: brew install ffmpeg\n"
                "   - Linux: sudo apt install ffmpeg\n\n"
                "2. Hoặc chạy script: scripts/install_ffmpeg_windows.ps1\n\n"
                "3. Xem hướng dẫn chi tiết: docs/VIDEO_TRANSCRIPTION_INSTALLATION.md"
            )
            self.show_error("Thiếu Thư Viện Video", detailed_msg)
        else:
            # Show regular error message
            self.show_error("Lỗi Xử Lý", f"Không thể chuyển đổi video:\n\n{error_msg}")
    
    def display_results(self, result: VideoTranscriptionResult):
        """
        Display transcription results in text view.
        Implements Requirement 10.3
        """
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        
        # FILTER HALLUCINATION - Final check before display
        filtered_full_text = self._filter_hallucination_text(result.full_text)
        
        # Header
        header = f"""
{'='*80}
VIDEO TRANSCRIPTION RESULT
{'='*80}
File: {Path(result.video_path).name}
Duration: {result.duration:.2f}s
Language: {result.language}
Average Confidence: {result.confidence_avg*100:.1f}%
Processing Time: {result.processing_time:.1f}s
{'='*80}

"""
        self.results_text.insert(tk.END, header)
        
        # Segments with timestamps - FILTERED (chỉ hiển thị phần này)
        if result.segments:
            self.results_text.insert(tk.END, "TIMESTAMPED SEGMENTS:\n")
            self.results_text.insert(tk.END, "-" * 80 + "\n")
            
            segment_count = 0
            for i, segment in enumerate(result.segments, 1):
                # FILTER each segment
                filtered_segment_text = self._filter_hallucination_text(segment.text)
                if not filtered_segment_text.strip():
                    continue  # Skip empty segments
                
                segment_count += 1
                timestamp = f"[{self.format_timestamp(segment.start)} --> {self.format_timestamp(segment.end)}]"
                speaker = f" ({segment.speaker})" if segment.speaker else ""
                
                self.results_text.insert(tk.END, f"\n{segment_count}. {timestamp}{speaker}\n")
                self.results_text.insert(tk.END, f"{filtered_segment_text}\n")
        
        # Speaker summary if available
        if result.speakers:
            self.results_text.insert(tk.END, "\n" + "="*80 + "\n")
            self.results_text.insert(tk.END, f"SPEAKERS DETECTED: {', '.join(result.speakers)}\n")
        
        self.results_text.config(state=tk.DISABLED)
    
    def _filter_hallucination_text(self, text: str) -> str:
        """
        Filter hallucination from text - GUI level final check.
        
        Args:
            text: Text to filter
            
        Returns:
            Filtered text (empty if all hallucination)
        """
        if not text or not text.strip():
            return ""
        
        text_lower = text.lower().strip()
        
        # ULTRA STRONG FILTER - Exact matches
        hallucination_patterns = [
            'hãy subscribe cho kênh',
            'subscribe cho kênh',
            'đăng ký kênh',
            'ghiền mì gõ',
            'bỏ lỡ những video',
            'bỏ lỡ video',
            'video hấp dẫn',
            'bật chuông thông báo',
            'bật chuông',
            'nhấn chuông',
            'bấm chuông',
            'like và share',
            'like share',
            'chia sẻ video',
            'theo dõi kênh',
            'cảm ơn đã xem',
            'hẹn gặp lại',
        ]
        
        # Check if text contains ANY hallucination pattern
        for pattern in hallucination_patterns:
            if pattern in text_lower:
                print(f"🚫 GUI FILTER: Removed hallucination: {text[:50]}...")
                return ""  # Return empty if hallucination detected
        
        # Check single keywords
        hallucination_keywords = [
            'subscribe',
            'subcribe',
        ]
        
        for keyword in hallucination_keywords:
            if keyword in text_lower:
                print(f"🚫 GUI FILTER: Removed keyword '{keyword}': {text[:50]}...")
                return ""
        
        return text
    
    def export_text(self):
        """
        Export transcription as plain text.
        Implements Requirement 10.4
        """
        if not self.transcription_result:
            self.show_error("Lỗi", "Chưa có kết quả để xuất!")
            return
        
        # Get default filename
        default_name = Path(self.selected_video_path).stem
        default_dir = str(Path("./transcripts").absolute())
        
        # Get save path with custom dialog
        file_path = ask_save_file(
            parent=self.parent,
            title="Lưu File Transcription",
            default_filename=default_name,
            default_dir=default_dir,
            file_extension=".txt",
            file_types=[
                ("Text files", "*.txt"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        try:
            # FILTER before export
            filtered_text = self._filter_hallucination_text(self.transcription_result.full_text)
            
            # Write filtered text to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(filtered_text)
            
            messagebox.showinfo(
                "Thành công",
                f"Đã xuất file text thành công!\n\n{file_path}"
            )
            
        except Exception as e:
            self.show_error("Lỗi Xuất File", f"Không thể xuất file:\n\n{str(e)}")
    
    def clear_results(self):
        """Clear results display."""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.config(state=tk.DISABLED)
        
        # Disable export button
        self.export_txt_button.config(state=tk.DISABLED)
    
    def update_status(self, text: str, color: str):
        """Update status label."""
        # This would update a status label if we had one
        # For now, just update progress label
        pass
    
    def show_error(self, title: str, message: str):
        """
        Show error message dialog.
        Implements Requirement 10.5
        """
        messagebox.showerror(title, message)
    
    @staticmethod
    def format_timestamp(seconds: float) -> str:
        """Format seconds as HH:MM:SS.mmm"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


# Example usage for testing
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Video Transcription Test")
    root.geometry("1000x800")
    root.configure(bg='#1a1a1a')
    
    # Create tab
    tab = VideoTranscriptionTab(root)
    
    root.mainloop()
