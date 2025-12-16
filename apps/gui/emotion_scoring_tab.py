# -*- coding: utf-8 -*-
"""
Emotion Scoring Tab for GUI

Provides a user interface for emotion scoring in recruitment video interviews.
Implements Requirements 14.1, 14.2, 14.4
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
import threading
import time
from typing import Optional
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.video_analysis.emotion_scoring import (
    EmotionScoringEngine,
    ReportGenerator,
    ScoringConfig,
    ConversationContext,
    EmotionReport,
    load_default_config
)
from apps.gui.file_save_dialog import ask_save_file
from datetime import datetime


class EmotionScoringTab:
    """
    GUI tab for emotion scoring functionality (4-criterion mode).
    
    Implements Requirements:
    - 5.5: Display 4 criterion scores with visual indicators and weights
    - 6.1: Display all 4 criterion scores with visual indicators and weights
    - 6.2: Provide explanations for each score
    - 6.4: Allow export to PDF or JSON format
    
    4 Core Criteria:
    - Confidence (30%) - Tự tin
    - Positivity (30%) - Tích cực
    - Professionalism (25%) - Chuyên nghiệp
    - Engagement (15%) - Tương tác
    """
    
    def __init__(self, parent_frame: tk.Frame):
        """
        Initialize emotion scoring tab.
        
        Args:
            parent_frame: Parent frame to contain this tab
        """
        self.parent = parent_frame
        self.engine: Optional[EmotionScoringEngine] = None
        self.generator: Optional[ReportGenerator] = None
        self.config: Optional[ScoringConfig] = None
        
        # State variables
        self.selected_video_path: Optional[str] = None
        self.candidate_id: str = ""
        self.emotion_report: Optional[EmotionReport] = None
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
        main_container.grid_rowconfigure(2, weight=0)  # Candidate ID
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
            text="🎯 CHẤM ĐIỂM CẢM XÚC PHỎNG VẤN - 4 TIÊU CHÍ",
            font=("Segoe UI", 18, "bold"),
            bg='#0d47a1',
            fg='#ffffff'
        )
        header_label.pack(pady=15)
        
        # File selection section (Requirement 14.4 - video input)
        file_frame = tk.Frame(main_container, bg='#252525', relief=tk.FLAT, bd=0)
        file_frame.grid(row=1, column=0, sticky='ew', pady=(0, 15))
        
        file_inner = tk.Frame(file_frame, bg='#252525')
        file_inner.pack(fill=tk.X, padx=20, pady=15)
        
        file_label = tk.Label(
            file_inner,
            text="📁 Chọn video phỏng vấn:",
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
        
        # Candidate ID section
        id_frame = tk.Frame(main_container, bg='#252525', relief=tk.FLAT, bd=0)
        id_frame.grid(row=2, column=0, sticky='ew', pady=(0, 15))
        
        id_inner = tk.Frame(id_frame, bg='#252525')
        id_inner.pack(fill=tk.X, padx=20, pady=15)
        
        id_label = tk.Label(
            id_inner,
            text="👤 Mã ứng viên:",
            font=("Segoe UI", 11, "bold"),
            bg='#252525',
            fg='#81c784'
        )
        id_label.pack(side=tk.LEFT, padx=(0, 15))
        
        self.candidate_id_entry = tk.Entry(
            id_inner,
            font=("Segoe UI", 10),
            bg='#1e1e1e',
            fg='#e0e0e0',
            insertbackground='#ffffff',
            relief=tk.FLAT,
            width=30
        )
        self.candidate_id_entry.pack(side=tk.LEFT, padx=10, ipady=5)
        self.candidate_id_entry.insert(0, "UV-16")  # Default candidate ID
        
        # Bind key release event to check if ready to start
        self.candidate_id_entry.bind('<KeyRelease>', lambda e: self.check_ready_to_start())
        
        # Start button
        self.start_button = tk.Button(
            id_inner,
            text="▶️ BẮT ĐẦU PHÂN TÍCH",
            font=("Segoe UI", 11, "bold"),
            bg='#43a047',
            fg='#ffffff',
            activebackground='#388e3c',
            command=self.start_scoring,
            cursor='hand2',
            relief=tk.FLAT,
            padx=25,
            pady=10,
            state=tk.DISABLED
        )
        self.start_button.pack(side=tk.RIGHT, padx=10)
        
        # Progress section (real-time progress display)
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
        
        # Results section (Requirement 14.1, 14.2 - display scores with explanations)
        results_frame = tk.Frame(main_container, bg='#252525', relief=tk.FLAT, bd=0)
        results_frame.grid(row=4, column=0, sticky='nsew', pady=(0, 15))
        
        results_label = tk.Label(
            results_frame,
            text="📊 Kết quả chấm điểm:",
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
        
        # Export buttons section (Requirement 14.4 - export to PDF/JSON)
        export_frame = tk.Frame(main_container, bg='#252525', relief=tk.FLAT, bd=0)
        export_frame.grid(row=5, column=0, sticky='ew')
        
        export_inner = tk.Frame(export_frame, bg='#252525')
        export_inner.pack(fill=tk.X, padx=20, pady=15)
        
        export_label = tk.Label(
            export_inner,
            text="💾 Xuất báo cáo:",
            font=("Segoe UI", 10, "bold"),
            bg='#252525',
            fg='#90caf9'
        )
        export_label.pack(side=tk.LEFT, padx=(0, 15))
        
        self.export_json_button = tk.Button(
            export_inner,
            text="📄 Xuất JSON",
            font=("Segoe UI", 10, "bold"),
            bg='#00897b',
            fg='#ffffff',
            activebackground='#00695c',
            command=self.export_json,
            cursor='hand2',
            relief=tk.FLAT,
            padx=20,
            pady=8,
            state=tk.DISABLED
        )
        self.export_json_button.pack(side=tk.LEFT, padx=5)
        
        self.export_pdf_button = tk.Button(
            export_inner,
            text="📑 Xuất PDF",
            font=("Segoe UI", 10, "bold"),
            bg='#d32f2f',
            fg='#ffffff',
            activebackground='#b71c1c',
            command=self.export_pdf,
            cursor='hand2',
            relief=tk.FLAT,
            padx=20,
            pady=8,
            state=tk.DISABLED
        )
        self.export_pdf_button.pack(side=tk.LEFT, padx=5)
    
    def initialize_components(self):
        """Initialize emotion scoring components (4-criteria mode by default)."""
        try:
            # Load default configuration (4 criteria with weights)
            self.config = load_default_config()
            
            # Initialize engine with 4-criteria configuration
            self.engine = EmotionScoringEngine(
                config=self.config,
                enable_validation=True
            )
            
            # Initialize report generator
            self.generator = ReportGenerator(output_dir='reports')
            
            self.update_status("✓ Sẵn sàng (4 tiêu chí: Confidence 30%, Positivity 30%, Professionalism 25%, Engagement 15%)", "#4CAF50")
            
        except Exception as e:
            error_msg = f"Không thể khởi tạo: {str(e)}"
            self.show_error("Lỗi Khởi Tạo", error_msg)
            self.update_status("✗ Lỗi khởi tạo", "#f44336")
    
    def browse_video_file(self):
        """Open file dialog to select video file."""
        file_path = filedialog.askopenfilename(
            title="Chọn video phỏng vấn",
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
            
            # Enable start button if candidate ID is also provided
            self.check_ready_to_start()
            
            self.update_status("✓ Đã chọn file", "#4CAF50")
    
    def check_ready_to_start(self):
        """Check if ready to start scoring and enable/disable start button."""
        candidate_id = self.candidate_id_entry.get().strip()
        
        if self.selected_video_path and candidate_id:
            self.start_button.config(state=tk.NORMAL)
        else:
            self.start_button.config(state=tk.DISABLED)
    
    def start_scoring(self):
        """Start emotion scoring process."""
        if not self.selected_video_path:
            self.show_error("Lỗi", "Vui lòng chọn file video trước!")
            return
        
        candidate_id = self.candidate_id_entry.get().strip()
        if not candidate_id:
            self.show_error("Lỗi", "Vui lòng nhập mã ứng viên!")
            return
        
        if self.is_processing:
            self.show_error("Đang xử lý", "Đang có quá trình phân tích đang chạy!")
            return
        
        # Store candidate ID
        self.candidate_id = candidate_id
        
        # Disable controls
        self.start_button.config(state=tk.DISABLED)
        self.browse_button.config(state=tk.DISABLED)
        self.candidate_id_entry.config(state=tk.DISABLED)
        
        # Clear previous results
        self.clear_results()
        
        # Start progress bar
        self.progress_bar.start(10)
        self.update_status("⏳ Đang phân tích...", "#2196F3")
        
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
            # Update progress
            self.parent.after(0, lambda: self.progress_detail_label.config(
                text="Đang trích xuất dữ liệu khuôn mặt..."
            ))
            
            # Score video interview
            report = self.engine.score_video_interview(
                video_path=self.selected_video_path,
                candidate_id=self.candidate_id,
                context=ConversationContext.default()
            )
            
            # Store result
            self.emotion_report = report
            
            # Update UI on main thread
            self.parent.after(0, lambda: self.on_scoring_complete(report))
            
        except Exception as e:
            # Show error on main thread
            error_msg = str(e)
            self.parent.after(0, lambda: self.on_scoring_error(error_msg))
    
    def on_scoring_complete(self, report: EmotionReport):
        """Handle successful scoring completion."""
        self.is_processing = False
        
        # Stop progress bar
        self.progress_bar.stop()
        
        # Update status
        self.update_status("✓ Hoàn thành!", "#4CAF50")
        self.progress_label.config(text="✓ Hoàn thành!")
        self.progress_detail_label.config(text="")
        
        # Display results (Requirement 14.1, 14.2)
        self.display_results(report)
        
        # Enable export buttons
        self.export_json_button.config(state=tk.NORMAL)
        self.export_pdf_button.config(state=tk.NORMAL)
        
        # Re-enable controls
        self.start_button.config(state=tk.NORMAL)
        self.browse_button.config(state=tk.NORMAL)
        self.candidate_id_entry.config(state=tk.NORMAL)
        
        # Show completion message
        messagebox.showinfo(
            "Hoàn thành",
            f"Phân tích cảm xúc hoàn tất!\n\n"
            f"Ứng viên: {report.candidate_id}\n"
            f"Điểm tổng: {report.total_score:.2f}/10\n"
            f"Số tiêu chí: {len(report.criterion_scores)}"
        )
    
    def on_scoring_error(self, error_msg: str):
        """Handle scoring error."""
        self.is_processing = False
        
        # Stop progress bar
        self.progress_bar.stop()
        
        # Update status
        self.update_status("✗ Lỗi xử lý", "#f44336")
        self.progress_label.config(text="✗ Lỗi xử lý")
        self.progress_detail_label.config(text="")
        
        # Re-enable controls
        self.start_button.config(state=tk.NORMAL)
        self.browse_button.config(state=tk.NORMAL)
        self.candidate_id_entry.config(state=tk.NORMAL)
        
        # Show error message
        self.show_error("Lỗi Xử Lý", f"Không thể phân tích video:\n\n{error_msg}")
    
    def display_results(self, report: EmotionReport):
        """
        Display emotion scoring results (4-criterion format).
        Implements Requirements 5.5, 6.1, 6.2
        """
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        
        # Header
        header = f"""
{'='*80}
KẾT QUẢ CHẤM ĐIỂM CẢM XÚC - 4 TIÊU CHÍ
{'='*80}
Ứng viên: {report.candidate_id}
Video: {Path(report.video_path).name}
Thời gian: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}

ĐIỂM TỔNG: {report.total_score:.2f}/10

{'='*80}
CHI TIẾT 4 TIÊU CHÍ CHÍNH (VỚI TRỌNG SỐ)
{'='*80}

"""
        self.results_text.insert(tk.END, header)
        
        # 4-criterion names and weights (from config)
        criterion_info = {
            'confidence': ('TỰ TIN', 0.30, '30%'),
            'positivity': ('TÍCH CỰC', 0.30, '30%'),
            'professionalism': ('CHUYÊN NGHIỆP', 0.25, '25%'),
            'engagement': ('TƯƠNG TÁC', 0.15, '15%')
        }
        
        # Display each criterion
        for i, criterion_key in enumerate(['confidence', 'positivity', 'professionalism', 'engagement'], 1):
            if criterion_key not in report.criterion_scores:
                continue
                
            criterion_score = report.criterion_scores[criterion_key]
            score = criterion_score.score
            confidence = criterion_score.confidence
            explanation = criterion_score.explanation
            
            vn_name, weight, weight_str = criterion_info[criterion_key]
            contribution = score * weight
            
            # Visual indicator based on score
            if score >= 8.0:
                indicator = "🌟"
                color_desc = "Xuất sắc"
            elif score >= 7.0:
                indicator = "✓✓"
                color_desc = "Tốt"
            elif score >= 5.0:
                indicator = "✓"
                color_desc = "Trung bình"
            else:
                indicator = "⚠"
                color_desc = "Cần cải thiện"
            
            # Format criterion display
            criterion_text = f"""
{i}. {vn_name} {indicator}
   Điểm: {score:.2f}/10 ({color_desc})
   Trọng số: {weight_str}
   Đóng góp vào điểm tổng: {contribution:.2f}
   Độ tin cậy: {confidence*100:.0f}%
   
   Giải thích: {explanation}
"""
            self.results_text.insert(tk.END, criterion_text)
            
            # Add separator
            if i < 4:
                self.results_text.insert(tk.END, "-" * 80 + "\n")
        
        # Calculation breakdown
        if all(k in report.criterion_scores for k in ['confidence', 'positivity', 'professionalism', 'engagement']):
            conf = report.criterion_scores['confidence'].score
            pos = report.criterion_scores['positivity'].score
            prof = report.criterion_scores['professionalism'].score
            eng = report.criterion_scores['engagement'].score
            
            calc_text = f"""
{'='*80}
CÔNG THỨC TÍNH ĐIỂM (TRỌNG SỐ)
{'='*80}

Emotion Score = Tự tin × 0.325 + Tích cực × 0.275 + Chuyên nghiệp × 0.225 + Tương tác × 0.175

              = {conf:.1f} × 0.325 + {pos:.1f} × 0.275 + {prof:.1f} × 0.225 + {eng:.1f} × 0.175
              
              = {conf*0.325:.2f} + {pos*0.275:.2f} + {prof*0.225:.2f} + {eng*0.175:.2f}
              
              = {report.total_score:.2f}/10

"""
            self.results_text.insert(tk.END, calc_text)
        
        # Key moments section
        if report.key_moments:
            self.results_text.insert(tk.END, f"\n{'='*80}\n")
            self.results_text.insert(tk.END, f"CÁC KHOẢNH KHẮC QUAN TRỌNG ({len(report.key_moments)})\n")
            self.results_text.insert(tk.END, f"{'='*80}\n\n")
            
            for i, moment in enumerate(report.key_moments, 1):
                timestamp_str = self.format_timestamp(moment.timestamp)
                self.results_text.insert(tk.END, 
                    f"{i}. [{timestamp_str}] {moment.description}\n"
                    f"   Cảm xúc: {moment.emotion} (độ mạnh: {moment.intensity:.2f})\n\n"
                )
        
        # Metadata section
        self.results_text.insert(tk.END, f"\n{'='*80}\n")
        self.results_text.insert(tk.END, "THÔNG TIN BỔ SUNG\n")
        self.results_text.insert(tk.END, f"{'='*80}\n\n")
        
        if report.metadata:
            for key, value in report.metadata.items():
                self.results_text.insert(tk.END, f"{key}: {value}\n")
        
        self.results_text.config(state=tk.DISABLED)
    
    def export_json(self):
        """Export report as JSON (Requirement 14.4)."""
        if not self.emotion_report:
            self.show_error("Lỗi", "Chưa có kết quả để xuất!")
            return
        
        try:
            # Get default filename
            candidate_id = self.emotion_report.candidate_id or "Unknown"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_name = f"emotion_report_{candidate_id}_{timestamp}"
            default_dir = str(Path("./reports").absolute())
            
            # Ask where to save with custom dialog
            output_path = ask_save_file(
                parent=self.parent,
                title="Xuất Báo Cáo JSON",
                default_filename=default_name,
                default_dir=default_dir,
                file_extension=".json",
                file_types=[
                    ("JSON files", "*.json"),
                    ("All files", "*.*")
                ]
            )
            
            if not output_path:
                return
            
            # Export using generator with custom path
            json_path = self.generator.export_to_json(
                self.emotion_report,
                output_path=output_path
            )
            
            messagebox.showinfo(
                "Thành công",
                f"Đã xuất báo cáo JSON thành công!\n\n{json_path}"
            )
            
        except Exception as e:
            self.show_error("Lỗi Xuất File", f"Không thể xuất JSON:\n\n{str(e)}")
    
    def export_pdf(self):
        """Export report as PDF (Requirement 14.4)."""
        if not self.emotion_report:
            self.show_error("Lỗi", "Chưa có kết quả để xuất!")
            return
        
        try:
            # Get default filename
            candidate_id = self.emotion_report.candidate_id or "Unknown"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_name = f"emotion_report_{candidate_id}_{timestamp}"
            default_dir = str(Path("./reports").absolute())
            
            # Ask where to save with custom dialog
            output_path = ask_save_file(
                parent=self.parent,
                title="Xuất Báo Cáo PDF",
                default_filename=default_name,
                default_dir=default_dir,
                file_extension=".pdf",
                file_types=[
                    ("PDF files", "*.pdf"),
                    ("All files", "*.*")
                ]
            )
            
            if not output_path:
                return
            
            # Export using generator with custom path
            pdf_path = self.generator.export_to_pdf(
                self.emotion_report,
                output_path=output_path,
                include_visualizations=True
            )
            
            messagebox.showinfo(
                "Thành công",
                f"Đã xuất báo cáo PDF thành công!\n\n{pdf_path}"
            )
            
        except Exception as e:
            self.show_error("Lỗi Xuất File", f"Không thể xuất PDF:\n\n{str(e)}")
    
    def clear_results(self):
        """Clear results display."""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.config(state=tk.DISABLED)
        
        # Disable export buttons
        self.export_json_button.config(state=tk.DISABLED)
        self.export_pdf_button.config(state=tk.DISABLED)
    
    def update_status(self, text: str, color: str):
        """Update status label."""
        # Update progress label with status
        pass
    
    def show_error(self, title: str, message: str):
        """Show error message dialog."""
        messagebox.showerror(title, message)
    
    @staticmethod
    def format_timestamp(seconds: float) -> str:
        """Format seconds as MM:SS"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"


# Example usage for testing
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Emotion Scoring Test")
    root.geometry("1000x800")
    root.configure(bg='#1a1a1a')
    
    # Create tab
    tab = EmotionScoringTab(root)
    
    root.mainloop()
