# -*- coding: utf-8 -*-
"""
Audio Transcription Tab for GUI (Simplified Version)

Provides a simple interface for audio-to-text transcription.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
import threading
from typing import Optional
import sys
import logging

logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.audio_recording.transcriber import get_transcriber
from src.speech_analysis.speech_clarity_analyzer import SpeechClarityAnalyzer
from src.speech_analysis.question_manager import QuestionManager, create_default_question_manager
from src.speech_analysis.interview_content_evaluator import InterviewContentEvaluator
from apps.gui.file_save_dialog import ask_save_file
from apps.gui.score_manager import get_score_manager


class AudioTranscriptionTab:
    """
    Simplified GUI tab for audio transcription.
    
    Similar to video transcription tab but for audio files.
    """
    
    def __init__(self, parent_frame: tk.Frame):
        """
        Initialize audio transcription tab.
        
        Args:
            parent_frame: Parent frame to contain this tab
        """
        self.parent = parent_frame
        self.transcriber = None
        self.clarity_analyzer = None
        
        # Question selection and content evaluation
        self.question_manager = None
        self.content_evaluator = None
        self.selected_question = None
        self.content_evaluation_result = None
        
        # Tips window reference (to prevent multiple windows)
        self.tips_window = None
        
        # State variables
        self.selected_audio_path: Optional[str] = None
        self.transcription_result: Optional[str] = None
        self.clarity_result: Optional[dict] = None
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
        main_container.grid_rowconfigure(2, weight=0)  # Start button
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
            text="🎤 CHUYỂN ĐỔI AUDIO THÀNH VĂN BẢN",
            font=("Segoe UI", 18, "bold"),
            bg='#0d47a1',
            fg='#ffffff'
        )
        header_label.pack(pady=15)
        
        # Question selection section (NEW)
        question_frame = tk.Frame(main_container, bg='#252525', relief=tk.FLAT, bd=0)
        question_frame.grid(row=1, column=0, sticky='ew', pady=(0, 15))
        
        question_inner = tk.Frame(question_frame, bg='#252525')
        question_inner.pack(fill=tk.X, padx=20, pady=15)
        
        question_label = tk.Label(
            question_inner,
            text="❓ Chọn câu hỏi phỏng vấn:",
            font=("Segoe UI", 11, "bold"),
            bg='#252525',
            fg='#90caf9'
        )
        question_label.pack(side=tk.LEFT, padx=(0, 15))
        
        # Question dropdown
        self.question_var = tk.StringVar()
        self.question_dropdown = ttk.Combobox(
            question_inner,
            textvariable=self.question_var,
            state="readonly",
            font=("Segoe UI", 10),
            width=60
        )
        self.question_dropdown.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        self.question_dropdown.bind('<<ComboboxSelected>>', self.on_question_selected)
        
        # Question info button
        self.question_info_button = tk.Button(
            question_inner,
            text="ℹ️ Tips",
            font=("Segoe UI", 9, "bold"),
            bg='#5c6bc0',
            fg='#ffffff',
            activebackground='#3f51b5',
            command=self.show_question_tips,
            cursor='hand2',
            relief=tk.FLAT,
            padx=15,
            pady=6,
            state=tk.DISABLED
        )
        self.question_info_button.pack(side=tk.LEFT, padx=10)
        
        # File selection section
        file_frame = tk.Frame(main_container, bg='#252525', relief=tk.FLAT, bd=0)
        file_frame.grid(row=2, column=0, sticky='ew', pady=(0, 15))
        
        file_inner = tk.Frame(file_frame, bg='#252525')
        file_inner.pack(fill=tk.X, padx=20, pady=15)
        
        file_label = tk.Label(
            file_inner,
            text="📁 Chọn audio file:",
            font=("Segoe UI", 11, "bold"),
            bg='#252525',
            fg='#90caf9'
        )
        file_label.pack(side=tk.LEFT, padx=(0, 15))
        
        self.browse_button = tk.Button(
            file_inner,
            text="📂 Chọn File WAV...",
            font=("Segoe UI", 10, "bold"),
            bg='#455a64',
            fg='#ffffff',
            activebackground='#37474f',
            command=self.browse_audio_file,
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
        
        # Start button section
        button_frame = tk.Frame(main_container, bg='#252525', relief=tk.FLAT, bd=0)
        button_frame.grid(row=3, column=0, sticky='ew', pady=(0, 15))
        
        button_inner = tk.Frame(button_frame, bg='#252525')
        button_inner.pack(fill=tk.X, padx=20, pady=15)
        
        self.start_button = tk.Button(
            button_inner,
            text="▶️ BẮT ĐẦU CHUYỂN ĐỔI",
            font=("Segoe UI", 12, "bold"),
            bg='#FF9800',
            fg='#ffffff',
            activebackground='#F57C00',
            command=self.start_transcription,
            cursor='hand2',
            relief=tk.FLAT,
            padx=30,
            pady=12,
            state=tk.DISABLED
        )
        self.start_button.pack(pady=5)
        
        # Progress section
        progress_frame = tk.Frame(main_container, bg='#252525', relief=tk.FLAT, bd=0)
        progress_frame.grid(row=4, column=0, sticky='ew', pady=(0, 15))
        
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
            mode='indeterminate',
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
        
        # Results section
        results_frame = tk.Frame(main_container, bg='#252525', relief=tk.FLAT, bd=0)
        results_frame.grid(row=5, column=0, sticky='nsew', pady=(0, 15))
        
        results_label = tk.Label(
            results_frame,
            text="📝 Kết quả:",
            font=("Segoe UI", 11, "bold"),
            bg='#252525',
            fg='#81c784'
        )
        results_label.pack(anchor='w', padx=20, pady=(15, 10))
        
        # Text view with scrollbar
        text_container = tk.Frame(results_frame, bg='#1a1a1a', relief=tk.SOLID, bd=1)
        text_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 15))
        
        self.results_text = scrolledtext.ScrolledText(
            text_container,
            font=("Consolas", 10),
            bg='#1e1e1e',
            fg='#e0e0e0',
            wrap=tk.WORD,
            relief=tk.FLAT,
            padx=15,
            pady=15
        )
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Export buttons section
        export_frame = tk.Frame(main_container, bg='#252525', relief=tk.FLAT, bd=0)
        export_frame.grid(row=6, column=0, sticky='ew')
        
        export_inner = tk.Frame(export_frame, bg='#252525')
        export_inner.pack(fill=tk.X, padx=20, pady=15)
        
        self.export_txt_button = tk.Button(
            export_inner,
            text="💾 Lưu Text",
            font=("Segoe UI", 10, "bold"),
            bg='#5c6bc0',
            fg='#ffffff',
            activebackground='#3f51b5',
            command=self.export_text,
            cursor='hand2',
            relief=tk.FLAT,
            padx=20,
            pady=8,
            state=tk.DISABLED
        )
        self.export_txt_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.clear_button = tk.Button(
            export_inner,
            text="🗑️ Xóa Kết Quả",
            font=("Segoe UI", 10, "bold"),
            bg='#757575',
            fg='#ffffff',
            activebackground='#616161',
            command=self.clear_results,
            cursor='hand2',
            relief=tk.FLAT,
            padx=20,
            pady=8,
            state=tk.DISABLED
        )
        self.clear_button.pack(side=tk.LEFT, padx=10)
        
        # Nút gửi điểm sang tab Tổng Hợp
        self.send_scores_button = tk.Button(
            export_inner,
            text="📊 GỬI ĐIỂM SANG TỔNG HỢP",
            font=("Segoe UI", 10, "bold"),
            bg='#FF9800',
            fg='#ffffff',
            activebackground='#F57C00',
            command=self.send_scores_to_summary,
            cursor='hand2',
            relief=tk.FLAT,
            padx=20,
            pady=8,
            state=tk.DISABLED
        )
        self.send_scores_button.pack(side=tk.LEFT, padx=10)
    
    def initialize_components(self):
        """Initialize transcriber, clarity analyzer, and content evaluator."""
        try:
            self.transcriber = get_transcriber()
            self.clarity_analyzer = SpeechClarityAnalyzer(language='vi')
            
            # Initialize question manager and content evaluator
            self.question_manager = create_default_question_manager()
            self.content_evaluator = InterviewContentEvaluator()
            
            # Load questions into dropdown
            self.load_questions()
            
            if not self.transcriber.is_available():
                self.show_error(
                    "Lỗi Khởi Tạo",
                    "Không thể khởi tạo Whisper STT engine.\n\n"
                    "Vui lòng kiểm tra:\n"
                    "• faster-whisper đã cài đặt\n"
                    "• ctranslate2 đã cài đặt\n"
                    "• Whisper model đã tải về"
                )
                self.progress_label.config(
                    text="❌ Lỗi: Transcriber không khả dụng",
                    fg='#f44336'
                )
            else:
                self.progress_label.config(
                    text="✅ Sẵn sàng - Chọn câu hỏi và file audio để bắt đầu",
                    fg='#4CAF50'
                )
                
        except Exception as e:
            self.show_error("Lỗi", f"Không thể khởi tạo transcriber:\n\n{str(e)}")
            self.progress_label.config(
                text=f"❌ Lỗi: {str(e)}",
                fg='#f44336'
            )
    
    def load_questions(self):
        """Load questions into dropdown."""
        if not self.question_manager:
            return
        
        questions = self.question_manager.get_all_questions()
        question_options = [
            f"{q.id} - {q.question[:60]}..." if len(q.question) > 60 else f"{q.id} - {q.question}"
            for q in questions
        ]
        
        self.question_dropdown['values'] = question_options
        if question_options:
            self.question_dropdown.current(0)
            self.on_question_selected(None)
    
    def on_question_selected(self, event):
        """Handle question selection."""
        if not self.question_manager:
            return
        
        selection = self.question_var.get()
        if not selection:
            return
        
        # Extract question ID
        question_id = selection.split(" - ")[0]
        
        # Select question
        self.selected_question = self.question_manager.select_question(question_id)
        
        if self.selected_question:
            self.question_info_button.config(state=tk.NORMAL)
            logger.info(f"Selected question: {question_id}")
    
    def show_question_tips(self):
        """Show tips for selected question."""
        if not self.selected_question:
            return
        
        # If tips window already exists and is visible, bring it to front
        if self.tips_window is not None and self.tips_window.winfo_exists():
            self.tips_window.lift()
            self.tips_window.focus_force()
            return
        
        # Create tips window
        tips_window = tk.Toplevel(self.parent)
        tips_window.title(f"Tips - {self.selected_question.id}")
        tips_window.geometry("600x400")
        tips_window.configure(bg='#1a1a1a')
        
        # Store reference
        self.tips_window = tips_window
        
        # Clear reference when window is closed
        def on_close():
            self.tips_window = None
            tips_window.destroy()
        
        tips_window.protocol("WM_DELETE_WINDOW", on_close)
        
        # Header
        header = tk.Label(
            tips_window,
            text=f"💡 TIPS CHO CÂU HỎI {self.selected_question.id}",
            font=("Segoe UI", 14, "bold"),
            bg='#0d47a1',
            fg='#ffffff',
            pady=15
        )
        header.pack(fill=tk.X)
        
        # Content frame
        content_frame = tk.Frame(tips_window, bg='#1a1a1a')
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Question text
        question_text = tk.Text(
            content_frame,
            font=("Segoe UI", 10),
            bg='#252525',
            fg='#e0e0e0',
            wrap=tk.WORD,
            height=15,
            relief=tk.FLAT,
            padx=15,
            pady=15
        )
        question_text.pack(fill=tk.BOTH, expand=True)
        
        # Build content
        content = f"Câu hỏi:\n{self.selected_question.question}\n\n"
        content += f"Danh mục: {self.selected_question.category}\n"
        content += f"Trọng số: {self.selected_question.weight*100:.0f}%\n\n"
        
        if self.selected_question.description:
            content += f"Mô tả:\n{self.selected_question.description}\n\n"
        
        if self.selected_question.tips:
            content += "💡 Tips để trả lời tốt:\n"
            for i, tip in enumerate(self.selected_question.tips, 1):
                content += f"  {i}. {tip}\n"
        
        question_text.insert(1.0, content)
        question_text.config(state=tk.DISABLED)
        
        # Close button
        close_btn = tk.Button(
            tips_window,
            text="Đóng",
            font=("Segoe UI", 10, "bold"),
            bg='#455a64',
            fg='#ffffff',
            command=on_close,
            cursor='hand2',
            relief=tk.FLAT,
            padx=20,
            pady=8
        )
        close_btn.pack(pady=10)
    
    def browse_audio_file(self):
        """Open file dialog to select audio file."""
        file_path = filedialog.askopenfilename(
            title="Chọn file audio",
            filetypes=[
                ("WAV files", "*.wav"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.selected_audio_path = file_path
            self.file_path_label.config(text=Path(file_path).name)
            self.start_button.config(state=tk.NORMAL)
            self.progress_label.config(
                text="✅ Đã chọn file - Nhấn 'BẮT ĐẦU CHUYỂN ĐỔI'",
                fg='#4CAF50'
            )
    
    def start_transcription(self):
        """Start audio transcription process."""
        if not self.selected_audio_path:
            self.show_error("Lỗi", "Vui lòng chọn file audio trước!")
            return
        
        if not self.transcriber or not self.transcriber.is_available():
            self.show_error("Lỗi", "Transcriber không khả dụng!")
            return
        
        if self.is_processing:
            self.show_warning("Đang Xử Lý", "Đang chuyển đổi file khác. Vui lòng đợi!")
            return
        
        # Start processing in background thread
        self.is_processing = True
        self.start_button.config(state=tk.DISABLED)
        self.browse_button.config(state=tk.DISABLED)
        self.export_txt_button.config(state=tk.DISABLED)
        self.clear_button.config(state=tk.DISABLED)
        
        # Clear previous results
        self.results_text.delete(1.0, tk.END)
        
        # Start progress bar
        self.progress_bar.start(10)
        self.progress_label.config(
            text="🔄 Đang xử lý...",
            fg='#ffa726'
        )
        
        # Run in thread
        self.processing_thread = threading.Thread(
            target=self._process_audio,
            daemon=True
        )
        self.processing_thread.start()
    
    def _process_audio(self):
        """Process audio in background thread."""
        try:
            def progress_callback(message):
                """Update progress on main thread."""
                self.parent.after(0, lambda: self.progress_detail_label.config(text=message))
            
            # Transcribe
            progress_callback("🎤 Đang chuyển đổi audio sang text...")
            result = self.transcriber.transcribe_file(
                self.selected_audio_path,
                progress_callback=progress_callback
            )
            
            # Analyze speech clarity if transcription succeeded
            clarity_result = None
            if result and self.clarity_analyzer:
                try:
                    progress_callback("📊 Đang phân tích độ rõ ràng trong lời nói...")
                    
                    # Get ASR confidence from Whisper transcriber
                    asr_confidence = self.transcriber.last_confidence
                    
                    if asr_confidence is None:
                        logger.warning("No ASR confidence available, skipping clarity analysis")
                        progress_callback("⚠️ Không có confidence từ Whisper")
                    else:
                        # Analyze clarity
                        clarity_result = self.clarity_analyzer.analyze_audio_file(
                            audio_path=self.selected_audio_path,
                            transcript=result,
                            asr_confidence=asr_confidence
                        )
                        
                        progress_callback("✅ Hoàn thành phân tích clarity!")
                    
                except Exception as e:
                    logger.error(f"Clarity analysis failed: {e}")
                    progress_callback(f"⚠️ Không thể phân tích clarity: {str(e)}")
            
            # Evaluate content if question is selected
            content_result = None
            if result and self.selected_question and self.content_evaluator:
                try:
                    progress_callback("🎯 Đang đánh giá nội dung câu trả lời...")
                    
                    score, similarity, best_match, details = self.content_evaluator.evaluate_answer(
                        self.selected_question.id,
                        result
                    )
                    
                    content_result = {
                        'question_id': self.selected_question.id,
                        'question': self.selected_question.question,
                        'category': self.selected_question.category,
                        'weight': self.selected_question.weight,
                        'score': score,
                        'similarity': similarity,
                        'best_match': best_match
                    }
                    
                    # Save to history
                    self.question_manager.add_to_history(
                        self.selected_question.id,
                        result,
                        score,
                        similarity
                    )
                    
                    progress_callback("✅ Hoàn thành đánh giá nội dung!")
                    
                except Exception as e:
                    logger.error(f"Content evaluation failed: {e}")
                    progress_callback(f"⚠️ Không thể đánh giá nội dung: {str(e)}")
            
            # Update UI on main thread
            self.parent.after(0, lambda: self.on_transcription_complete(result, clarity_result, content_result))
            
        except Exception as e:
            self.parent.after(0, lambda: self.on_transcription_error(str(e)))
    
    def on_transcription_complete(self, result: Optional[str], clarity_result: Optional[dict] = None, content_result: Optional[dict] = None):
        """Handle successful transcription completion."""
        self.is_processing = False
        self.progress_bar.stop()
        
        # Re-enable buttons
        self.start_button.config(state=tk.NORMAL)
        self.browse_button.config(state=tk.NORMAL)
        
        if result is None or not result:
            self.progress_label.config(
                text="⚠️ Không phát hiện được giọng nói trong audio",
                fg='#ffa726'
            )
            self.progress_detail_label.config(text="")
            self.show_warning(
                "Không Có Giọng Nói",
                "Không phát hiện được giọng nói trong file audio.\n\n"
                "Vui lòng kiểm tra:\n"
                "• File audio có giọng nói rõ ràng\n"
                "• Volume đủ lớn\n"
                "• Không quá nhiều tiếng ồn"
            )
            return
        
        # Store results
        self.transcription_result = result
        self.clarity_result = clarity_result
        self.content_evaluation_result = content_result
        
        # Display results
        self.display_results(result, clarity_result, content_result)
        
        # Enable export buttons
        self.export_txt_button.config(state=tk.NORMAL)
        self.clear_button.config(state=tk.NORMAL)
        self.send_scores_button.config(state=tk.NORMAL)
        
        # Update progress
        self.progress_label.config(
            text="✅ Hoàn thành!",
            fg='#4CAF50'
        )
        
        # Build progress detail with clarity and content info
        detail_text = f"Đã chuyển đổi: {len(result)} ký tự, {len(result.split())} từ"
        if clarity_result:
            clarity_score = clarity_result.clarity_score
            detail_text += f" | Độ rõ ràng: {clarity_score:.1f}/10"
        if content_result:
            content_score = content_result['score']
            detail_text += f" | Nội dung: {content_score:.1f}/10"
        self.progress_detail_label.config(text=detail_text)
        
        # Show success message with all info
        success_msg = "✅ CHUYỂN ĐỔI HOÀN TẤT!\n\n"
        success_msg += "="*50 + "\n"
        success_msg += f"📁 File: {Path(self.selected_audio_path).name}\n"
        success_msg += f"📝 Số ký tự: {len(result)}\n"
        success_msg += f"📝 Số từ: {len(result.split())}\n"
        success_msg += "="*50 + "\n\n"
        
        # Content evaluation
        if content_result:
            content_score = content_result['score']
            similarity = content_result['similarity']
            category = content_result['category']
            
            if content_score >= 8:
                level_emoji = "⭐"
                level_text = "XUẤT SẮC"
            elif content_score >= 6:
                level_emoji = "✅"
                level_text = "TỐT"
            elif content_score >= 4:
                level_emoji = "→"
                level_text = "TRUNG BÌNH"
            else:
                level_emoji = "⚠️"
                level_text = "CẦN CẢI THIỆN"
            
            success_msg += f"🎯 ĐÁNH GIÁ NỘI DUNG:\n"
            success_msg += f"   {level_emoji} Điểm: {content_score:.1f}/10 - {level_text}\n"
            success_msg += f"   📊 Similarity: {similarity:.3f}\n"
            success_msg += f"   📋 Danh mục: {category}\n\n"
        
        # Clarity analysis
        if clarity_result:
            clarity_score = clarity_result.clarity_score
            clarity_level = clarity_result.clarity_level
            
            if clarity_score >= 8:
                clarity_emoji = "✅"
            elif clarity_score >= 6:
                clarity_emoji = "→"
            else:
                clarity_emoji = "⚠️"
            
            success_msg += f"📊 ĐỘ RÕ RÀNG:\n"
            success_msg += f"   {clarity_emoji} Điểm: {clarity_score:.1f}/10 - {clarity_level}\n\n"
        
        # Overall assessment
        success_msg += "="*50 + "\n"
        if content_result and content_result['score'] >= 7 and clarity_result and clarity_result.clarity_score >= 7:
            success_msg += "🎉 Xuất sắc! Cả nội dung và độ rõ ràng đều tốt!\n"
        elif content_result and content_result['score'] >= 6:
            success_msg += "👍 Tốt! Xem chi tiết để cải thiện thêm.\n"
        else:
            success_msg += "📚 Xem chi tiết nhận xét để cải thiện.\n"
        success_msg += "="*50
        
        messagebox.showinfo("Thành Công", success_msg)
    
    def on_transcription_error(self, error_msg: str):
        """Handle transcription error."""
        self.is_processing = False
        self.progress_bar.stop()
        
        # Re-enable buttons
        self.start_button.config(state=tk.NORMAL)
        self.browse_button.config(state=tk.NORMAL)
        
        # Update progress
        self.progress_label.config(
            text="❌ Lỗi xử lý",
            fg='#f44336'
        )
        self.progress_detail_label.config(text="")
        
        # Show error
        self.show_error("Lỗi Xử Lý", f"Không thể chuyển đổi audio:\n\n{error_msg}")
    
    def display_results(self, text: str, clarity_result: Optional[dict] = None, content_result: Optional[dict] = None):
        """Display transcription results with clarity and content analysis."""
        self.results_text.delete(1.0, tk.END)
        
        # Format header
        header = f"""
{'='*80}
AUDIO TRANSCRIPTION RESULT
{'='*80}
File: {Path(self.selected_audio_path).name}
Số ký tự: {len(text)}
Số từ: {len(text.split())}
{'='*80}

"""
        
        self.results_text.insert(tk.END, header)
        
        # Add content evaluation if available
        if content_result:
            content_section = self._format_content_section(content_result)
            self.results_text.insert(tk.END, content_section)
            self.results_text.insert(tk.END, "\n")
        
        # Add clarity analysis if available
        if clarity_result:
            clarity_section = self._format_clarity_section(clarity_result)
            self.results_text.insert(tk.END, clarity_section)
            self.results_text.insert(tk.END, "\n")
        
        # Add transcription text
        self.results_text.insert(tk.END, "📝 NỘI DUNG:\n")
        self.results_text.insert(tk.END, f"{'-'*80}\n")
        self.results_text.insert(tk.END, text)
        
        # Scroll to top
        self.results_text.see(1.0)
    
    def _format_content_section(self, content_result: dict) -> str:
        """Format content evaluation results for display."""
        score = content_result['score']
        similarity = content_result['similarity']
        question = content_result['question']
        category = content_result['category']
        weight = content_result['weight']
        best_match = content_result['best_match']
        
        # Determine evaluation level
        if score >= 8:
            level = "XUẤT SẮC"
            emoji = "✅"
            color_bar = "█" * 9 + "░"
        elif score >= 6:
            level = "TỐT"
            emoji = "✓"
            color_bar = "█" * 7 + "░" * 3
        elif score >= 4:
            level = "TRUNG BÌNH"
            emoji = "→"
            color_bar = "█" * 5 + "░" * 5
        else:
            level = "CẦN CẢI THIỆN"
            emoji = "✗"
            color_bar = "█" * 2 + "░" * 8
        
        section = f"""
{'='*80}
🎯 ĐÁNH GIÁ NỘI DUNG CÂU TRẢ LỜI
{'='*80}

📋 THÔNG TIN CÂU HỎI:
{'-'*80}
Câu hỏi: {question}
Danh mục: {category}
Trọng số: {weight*100:.0f}% (trong tổng điểm phỏng vấn)

📊 KẾT QUẢ ĐÁNH GIÁ:
{'-'*80}
  {emoji} ĐIỂM SỐ: {score:.1f}/10 - {level}
  
  Thang điểm: [{color_bar}] {score:.1f}/10
  
  • Độ tương đồng với câu mẫu: {similarity:.3f}
    (Similarity càng cao = câu trả lời càng giống câu mẫu tốt)

💬 NHẬN XÉT CHI TIẾT:
{'-'*80}
"""
        
        # Detailed feedback based on score
        if score >= 9:
            section += """  ⭐⭐⭐ XUẤT SẮC ⭐⭐⭐
  
  Điểm mạnh:
  ✓ Câu trả lời rất chi tiết và cụ thể
  ✓ Cấu trúc rõ ràng, logic chặt chẽ
  ✓ Có số liệu, ví dụ thực tế thuyết phục
  ✓ Thể hiện kỹ năng chuyên môn cao
  
  Đánh giá: Câu trả lời xuất sắc! Bạn đã trả lời rất tốt cho câu hỏi này.
  Nhà tuyển dụng sẽ rất ấn tượng với câu trả lời của bạn.
"""
        elif score >= 8:
            section += """  ✅ XUẤT SẮC
  
  Điểm mạnh:
  ✓ Câu trả lời chi tiết và chuyên nghiệp
  ✓ Có cấu trúc tốt
  ✓ Nội dung liên quan trực tiếp đến câu hỏi
  
  Gợi ý cải thiện:
  → Có thể thêm một số con số cụ thể hơn
  → Làm rõ hơn về kết quả đạt được
  
  Đánh giá: Câu trả lời rất tốt! Chỉ cần điều chỉnh nhỏ để đạt điểm tối đa.
"""
        elif score >= 7:
            section += """  ✓ TỐT
  
  Điểm mạnh:
  ✓ Câu trả lời đạt yêu cầu
  ✓ Có nội dung liên quan đến câu hỏi
  ✓ Thể hiện được kinh nghiệm
  
  Gợi ý cải thiện:
  → Thêm chi tiết cụ thể hơn (số liệu, thời gian, công nghệ)
  → Cấu trúc câu trả lời theo mô hình STAR
    (Situation - Task - Action - Result)
  → Làm rõ hơn về vai trò và đóng góp của bạn
  
  Đánh giá: Câu trả lời tốt! Với một số cải thiện nhỏ sẽ đạt điểm cao hơn.
"""
        elif score >= 6:
            section += """  ✓ TỐT (Đạt yêu cầu tối thiểu)
  
  Điểm mạnh:
  ✓ Có trả lời đúng câu hỏi
  ✓ Có đề cập đến kinh nghiệm
  
  Cần cải thiện:
  ⚠ Thiếu chi tiết cụ thể
  ⚠ Chưa có cấu trúc rõ ràng
  ⚠ Chưa thể hiện rõ kết quả đạt được
  
  Gợi ý:
  → Thêm ví dụ cụ thể với số liệu
  → Mô tả rõ tình huống, hành động, và kết quả
  → Làm rõ vai trò của bạn trong tình huống đó
  → Tăng độ dài câu trả lời (50-150 từ)
  
  Đánh giá: Câu trả lời đạt yêu cầu nhưng cần cải thiện để gây ấn tượng hơn.
"""
        elif score >= 5:
            section += """  → TRUNG BÌNH
  
  Vấn đề:
  ⚠ Câu trả lời quá chung chung
  ⚠ Thiếu chi tiết và ví dụ cụ thể
  ⚠ Chưa thể hiện rõ năng lực
  
  Cần cải thiện:
  ✗ Thêm nhiều chi tiết hơn (ai, cái gì, khi nào, ở đâu, như thế nào)
  ✗ Đưa ra số liệu, kết quả đo lường được
  ✗ Cấu trúc câu trả lời theo STAR:
    • Situation: Mô tả tình huống
    • Task: Nhiệm vụ của bạn
    • Action: Hành động bạn thực hiện
    • Result: Kết quả đạt được
  
  Đánh giá: Câu trả lời cần cải thiện đáng kể. Hãy chuẩn bị kỹ hơn.
"""
        elif score >= 4:
            section += """  → TRUNG BÌNH (Dưới mức mong đợi)
  
  Vấn đề nghiêm trọng:
  ✗ Câu trả lời quá ngắn và thiếu nội dung
  ✗ Không có ví dụ cụ thể
  ✗ Không thể hiện được năng lực
  
  Cần làm:
  1. Chuẩn bị kỹ câu trả lời trước khi phỏng vấn
  2. Sử dụng mô hình STAR để cấu trúc câu trả lời
  3. Thêm chi tiết: số liệu, thời gian, công nghệ, kết quả
  4. Tăng độ dài lên ít nhất 50-100 từ
  5. Luyện tập trả lời nhiều lần
  
  Đánh giá: Câu trả lời chưa đạt yêu cầu. Cần chuẩn bị lại kỹ lưỡng.
"""
        else:
            section += """  ✗ CẦN CẢI THIỆN NHIỀU
  
  Vấn đề rất nghiêm trọng:
  ✗✗ Câu trả lời quá ngắn (< 20 từ)
  ✗✗ Không có nội dung cụ thể
  ✗✗ Không liên quan đến câu hỏi
  ✗✗ Không thể hiện được kinh nghiệm
  
  Hành động cần thiết:
  ⚠ DỪNG LẠI - Đọc lại câu hỏi kỹ
  ⚠ Chuẩn bị câu trả lời theo mô hình STAR
  ⚠ Viết ra câu trả lời trước khi nói
  ⚠ Luyện tập nhiều lần trước khi phỏng vấn thật
  
  Ví dụ cấu trúc tốt:
  • Mở đầu: "Tôi từng gặp [tình huống]..."
  • Giữa: "Tôi đã [hành động cụ thể]..."
  • Kết: "Kết quả là [số liệu/thành tựu]..."
  
  Đánh giá: Câu trả lời chưa đạt yêu cầu tối thiểu. Cần chuẩn bị lại hoàn toàn.
"""
        
        # Add similarity interpretation
        section += f"\n📈 PHÂN TÍCH ĐỘ TƯƠNG ĐỒNG:\n{'-'*80}\n"
        if similarity >= 0.85:
            section += f"  Similarity: {similarity:.3f} - RẤT CAO\n"
            section += "  → Câu trả lời của bạn rất giống với câu mẫu tốt nhất.\n"
        elif similarity >= 0.75:
            section += f"  Similarity: {similarity:.3f} - CAO\n"
            section += "  → Câu trả lời của bạn khá giống với câu mẫu tốt.\n"
        elif similarity >= 0.65:
            section += f"  Similarity: {similarity:.3f} - TRUNG BÌNH\n"
            section += "  → Câu trả lời có một số điểm tương đồng với câu mẫu.\n"
        elif similarity >= 0.50:
            section += f"  Similarity: {similarity:.3f} - THẤP\n"
            section += "  → Câu trả lời khác khá nhiều so với câu mẫu tốt.\n"
        else:
            section += f"  Similarity: {similarity:.3f} - RẤT THẤP\n"
            section += "  → Câu trả lời rất khác so với câu mẫu. Cần xem lại nội dung.\n"
        
        # Add best match example
        section += f"\n📝 CÂU TRẢ LỜI MẪU THAM KHẢO:\n{'-'*80}\n"
        section += f"{best_match}\n"
        section += f"\n💡 Gợi ý: Hãy tham khảo câu mẫu trên để cải thiện câu trả lời của bạn.\n"
        section += f"{'='*80}\n"
        
        return section
    
    def _format_clarity_section(self, clarity_result) -> str:
        """Format clarity analysis results for display."""
        # Access dataclass attributes directly
        overall_score = clarity_result.clarity_score
        clarity_level = clarity_result.clarity_level
        
        # Build clarity section
        section = f"""
📊 PHÂN TÍCH ĐỘ RÕ RÀNG TRONG LỜI NÓI
{'-'*80}
Điểm tổng thể: {overall_score:.1f}/10 - {clarity_level}

Chi tiết các yếu tố:
  • Tốc độ nói (25%):        {clarity_result.speech_rate_score:.1f}/10
    - WPM: {clarity_result.wpm:.1f} (tối ưu: 120-160)
    
  • Từ ngập ngừng (25%):     {clarity_result.filler_score:.1f}/10
    - Tỷ lệ: {clarity_result.filler_rate:.2%}
    - Số lượng: {clarity_result.filler_count}
    
  • Ổn định âm lượng (15%):  {clarity_result.volume_stability_score:.1f}/10
    - Độ lệch chuẩn: {clarity_result.volume_std:.3f}
    
  • Ổn định giọng (10%):     {clarity_result.pitch_stability_score:.1f}/10
    - Độ lệch chuẩn: {clarity_result.pitch_std:.1f} Hz
    
  • Độ tin cậy ASR (25%):    {clarity_result.asr_confidence_score:.1f}/10
    - Confidence: {clarity_result.asr_confidence:.2%}
"""
        
        # Add issues if any
        if clarity_result.issues:
            section += f"\n⚠️ Vấn đề phát hiện:\n"
            for issue in clarity_result.issues:
                section += f"  • {issue}\n"
        
        section += f"{'-'*80}\n"
        
        return section
    
    def export_text(self):
        """Export transcription with clarity and content report to text file."""
        if not self.transcription_result:
            self.show_warning("Không Có Dữ Liệu", "Chưa có kết quả để xuất!")
            return
        
        # Get default filename
        default_name = Path(self.selected_audio_path).stem + "_transcription"
        default_dir = str(Path("./transcripts").absolute())
        
        # Ask where to save with custom dialog
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
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    # Write header
                    f.write("="*80 + "\n")
                    f.write("AUDIO TRANSCRIPTION RESULT\n")
                    f.write("="*80 + "\n")
                    f.write(f"File: {Path(self.selected_audio_path).name}\n")
                    f.write(f"Số ký tự: {len(self.transcription_result)}\n")
                    f.write(f"Số từ: {len(self.transcription_result.split())}\n")
                    f.write("="*80 + "\n\n")
                    
                    # Write content evaluation if available
                    if self.content_evaluation_result:
                        content_section = self._format_content_section(self.content_evaluation_result)
                        f.write(content_section)
                        f.write("\n")
                    
                    # Write clarity analysis if available
                    if self.clarity_result:
                        clarity_section = self._format_clarity_section(self.clarity_result)
                        f.write(clarity_section)
                        f.write("\n")
                    
                    # Write transcription
                    f.write("📝 NỘI DUNG:\n")
                    f.write("-"*80 + "\n")
                    f.write(self.transcription_result)
                
                messagebox.showinfo(
                    "Thành Công",
                    f"Đã lưu file text:\n\n{file_path}"
                )
            except Exception as e:
                self.show_error("Lỗi Lưu File", f"Không thể lưu file:\n\n{str(e)}")
    
    def clear_results(self):
        """Clear results and reset."""
        self.results_text.delete(1.0, tk.END)
        self.transcription_result = None
        self.clarity_result = None
        self.content_evaluation_result = None
        self.export_txt_button.config(state=tk.DISABLED)
        self.clear_button.config(state=tk.DISABLED)
        self.send_scores_button.config(state=tk.DISABLED)
        self.progress_label.config(
            text="✅ Đã xóa - Chọn câu hỏi và file mới để tiếp tục",
            fg='#4CAF50'
        )
        self.progress_detail_label.config(text="")
    
    def send_scores_to_summary(self):
        """
        Gửi điểm Clarity và Content sang tab Tổng Hợp Điểm.
        
        Sử dụng ScoreManager để chia sẻ điểm giữa các tab.
        """
        if not self.clarity_result or not self.content_evaluation_result:
            self.show_warning(
                "Thiếu Dữ Liệu",
                "Chưa có đủ kết quả để gửi điểm!\n\n"
                "Vui lòng hoàn thành chuyển đổi audio trước."
            )
            return
        
        # Lấy điểm - clarity_result là SpeechClarityResult object
        if isinstance(self.clarity_result, dict):
            clarity_score = self.clarity_result.get('clarity_score', 0.0)
        else:
            # SpeechClarityResult object
            clarity_score = self.clarity_result.clarity_score
        
        # content_evaluation_result có thể là dict hoặc ContentEvaluationResult
        if isinstance(self.content_evaluation_result, dict):
            content_score = self.content_evaluation_result.get('score', 0.0)
        else:
            # ContentEvaluationResult object
            content_score = self.content_evaluation_result.total_score
        
        # Gửi vào ScoreManager
        score_manager = get_score_manager()
        score_manager.set_clarity_score(clarity_score, source="audio_transcription_tab")
        score_manager.set_content_score(content_score, source="audio_transcription_tab")
        
        # Thông báo thành công
        message = "✅ ĐÃ GỬI ĐIỂM THÀNH CÔNG!\n\n"
        message += f"🗣️ Rõ ràng (Clarity): {clarity_score:.2f}/10\n"
        message += f"📝 Nội dung (Content): {content_score:.2f}/10\n\n"
        message += "Điểm đã được gửi sang tab 'Tổng Hợp Điểm'.\n"
        message += "Vui lòng chuyển sang tab đó để xem và tính điểm tổng!"
        
        messagebox.showinfo("Gửi Điểm Thành Công", message)
    
    def show_error(self, title: str, message: str):
        """Show error message dialog."""
        messagebox.showerror(title, message)
    
    def show_warning(self, title: str, message: str):
        """Show warning message dialog."""
        messagebox.showwarning(title, message)
    
    def show_info(self, title: str, message: str):
        """Show info message dialog."""
        messagebox.showinfo(title, message)


# Example usage for testing
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Audio Transcription Test")
    root.geometry("1000x800")
    root.configure(bg='#1a1a1a')
    
    # Create tab
    tab = AudioTranscriptionTab(root)
    
    root.mainloop()
