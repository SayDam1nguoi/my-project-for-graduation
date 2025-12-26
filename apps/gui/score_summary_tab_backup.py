"""
Score Summary Tab - UI Tổng Hợp Điểm

Tab này hiển thị 4 đầu điểm và tính điểm tổng:
1. Cảm xúc (Emotion) - từ video analysis
2. Tập trung (Focus) - từ video analysis
3. Rõ ràng (Clarity) - từ speech analysis
4. Nội dung (Content) - từ speech analysis

Tính năng:
- Hiển thị 4 ô điểm (0-10)
- Tự động cập nhật từ các tab khác
- Nút tính tổng
- Xuất kết quả ra file .txt
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import datetime
from pathlib import Path
import json
from apps.gui.file_save_dialog import ask_save_file
from apps.gui.score_manager import get_score_manager


class ScoreSummaryTab:
    """Tab tổng hợp điểm phỏng vấn."""
    
    def __init__(self, parent):
        """
        Khởi tạo tab.
        
        Args:
            parent: Parent widget (notebook)
        """
        self.parent = parent
        # Sử dụng tk.Frame thay vì ttk.Frame để có thể set background color
        self.frame = tk.Frame(parent, bg="white")
        
        # Điểm số (0-10)
        self.emotion_score = tk.DoubleVar(value=0.0)
        self.focus_score = tk.DoubleVar(value=0.0)
        self.clarity_score = tk.DoubleVar(value=0.0)
        self.content_score = tk.DoubleVar(value=0.0)
        self.total_score = tk.DoubleVar(value=0.0)
        
        # Trọng số (theo công thức: N=40%, T=30%, G=25%, O=5%)
        self.weight_content = tk.DoubleVar(value=40.0)    # Nội dung (N)
        self.weight_focus = tk.DoubleVar(value=30.0)      # Tập trung (T)
        self.weight_clarity = tk.DoubleVar(value=25.0)    # Giọng nói (G)
        self.weight_emotion = tk.DoubleVar(value=5.0)     # Cảm xúc (O)
        
        # Thông tin ứng viên
        self.candidate_name = tk.StringVar(value="")
        self.candidate_id = tk.StringVar(value="")
        self.position = tk.StringVar(value="default")
        
        # Score Manager
        self.score_manager = get_score_manager()
        
        # Register callback để tự động cập nhật khi có điểm mới
        self.score_manager.register_callback(self._on_score_updated)
        
        self._create_widgets()
    
    def _create_widgets(self):
        """Tạo các widgets."""
        # Configure style for ttk widgets
        style = ttk.Style()
        style.configure("White.TFrame", background="white")
        style.configure("White.TLabel", background="white")
        style.configure("White.TLabelframe", background="white")
        style.configure("White.TLabelframe.Label", background="white")
        
        # Title
        title_label = tk.Label(
            self.frame,
            text="📊 TỔNG HỢP ĐIỂM PHỎNG VẤN",
            font=("Arial", 16, "bold"),
            bg="white",
            fg="#1976D2"
        )
        title_label.pack(pady=10)
        
        # Main container
        main_container = tk.Frame(self.frame, bg="white")
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Configure grid weights
        main_container.grid_columnconfigure(0, weight=1)
        main_container.grid_columnconfigure(1, weight=2)
        main_container.grid_columnconfigure(2, weight=1)
        
        # Left: Thông tin ứng viên
        self._create_candidate_info(main_container)
        
        # Center: 4 ô điểm
        self._create_score_boxes(main_container)
        
        # Right: Trọng số
        self._create_weights_panel(main_container)
        
        # Bottom: Điểm tổng và buttons
        self._create_bottom_panel(main_container)

    
    def _create_candidate_info(self, parent):
        """Tạo panel thông tin ứng viên."""
        info_frame = tk.LabelFrame(
            parent, 
            text="Thông Tin Ứng Viên", 
            font=("Arial", 11, "bold"),
            bg="white",
            fg="#424242",
            padx=10,
            pady=10
        )
        info_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Tên ứng viên
        tk.Label(info_frame, text="Họ tên:", bg="white").grid(row=0, column=0, sticky="w", pady=5)
        tk.Entry(info_frame, textvariable=self.candidate_name, width=25).grid(
            row=0, column=1, sticky="ew", pady=5, padx=5
        )
        
        # Mã ứng viên
        tk.Label(info_frame, text="Mã ứng viên:", bg="white").grid(row=1, column=0, sticky="w", pady=5)
        tk.Entry(info_frame, textvariable=self.candidate_id, width=25).grid(
            row=1, column=1, sticky="ew", pady=5, padx=5
        )
        
        # Vị trí
        tk.Label(info_frame, text="Vị trí:", bg="white").grid(row=2, column=0, sticky="w", pady=5)
        position_combo = ttk.Combobox(
            info_frame,
            textvariable=self.position,
            values=["default", "technical", "sales", "customer_service", "management"],
            state="readonly",
            width=22
        )
        position_combo.grid(row=2, column=1, sticky="ew", pady=5, padx=5)
        position_combo.bind("<<ComboboxSelected>>", self._on_position_changed)
    
    def _create_score_boxes(self, parent):
        """Tạo 4 ô điểm."""
        scores_frame = tk.LabelFrame(
            parent, 
            text="Điểm Đánh Giá (0-10)", 
            font=("Arial", 11, "bold"),
            bg="white",
            fg="#424242",
            padx=10,
            pady=10
        )
        scores_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        # Configure grid
        scores_frame.columnconfigure(0, weight=1)
        scores_frame.columnconfigure(1, weight=1)
        
        # 1. Cảm xúc (Emotion)
        self._create_score_box(
            scores_frame, "😊 Cảm Xúc", self.emotion_score,
            "Từ phân tích video", 0, 0, "#FF6B6B"
        )
        
        # 2. Tập trung (Focus)
        self._create_score_box(
            scores_frame, "👁️ Tập Trung", self.focus_score,
            "Từ phân tích video", 0, 1, "#4ECDC4"
        )
        
        # 3. Rõ ràng (Clarity)
        self._create_score_box(
            scores_frame, "🗣️ Rõ Ràng", self.clarity_score,
            "Từ phân tích giọng nói", 1, 0, "#95E1D3"
        )
        
        # 4. Nội dung (Content)
        self._create_score_box(
            scores_frame, "📝 Nội Dung", self.content_score,
            "Từ phân tích giọng nói", 1, 1, "#F38181"
        )
    
    def _create_score_box(self, parent, title, variable, subtitle, row, col, color):
        """Tạo một ô điểm."""
        box_frame = tk.Frame(parent, relief="solid", borderwidth=2, bg="white")
        box_frame.grid(row=row, column=col, sticky="nsew", padx=10, pady=10)
        
        # Title
        title_label = tk.Label(
            box_frame, text=title, font=("Arial", 12, "bold"), bg="white"
        )
        title_label.pack(pady=(10, 5))
        
        # Score display
        score_label = tk.Label(
            box_frame,
            textvariable=variable,
            font=("Arial", 36, "bold"),
            foreground=color,
            bg="white"
        )
        score_label.pack(pady=10)
        
        # /10
        tk.Label(box_frame, text="/10", font=("Arial", 12), bg="white").pack()
        
        # Subtitle
        tk.Label(
            box_frame, text=subtitle, font=("Arial", 9), foreground="gray", bg="white"
        ).pack(pady=(5, 10))

    
    def _create_weights_panel(self, parent):
        """Tạo panel trọng số."""
        weights_frame = tk.LabelFrame(
            parent, 
            text="Trọng Số (%)", 
            font=("Arial", 11, "bold"),
            bg="white",
            fg="#424242",
            padx=10,
            pady=10
        )
        weights_frame.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)
        
        # Content
        self._create_weight_row(weights_frame, "📝 Nội dung:", self.weight_content, 0)
        
        # Clarity
        self._create_weight_row(weights_frame, "🗣️ Rõ ràng:", self.weight_clarity, 1)
        
        # Focus
        self._create_weight_row(weights_frame, "👁️ Tập trung:", self.weight_focus, 2)
        
        # Emotion
        self._create_weight_row(weights_frame, "😊 Cảm xúc:", self.weight_emotion, 3)
        
        # Total
        tk.Frame(weights_frame, height=2, bg="#CCCCCC").grid(
            row=4, column=0, columnspan=2, sticky="ew", pady=10
        )
        
        total_label = tk.Label(weights_frame, text="Tổng:", font=("Arial", 10, "bold"), bg="white")
        total_label.grid(row=5, column=0, sticky="w", pady=5)
        
        self.total_weight_label = tk.Label(
            weights_frame, text="100%", font=("Arial", 10, "bold"), foreground="green", bg="white"
        )
        self.total_weight_label.grid(row=5, column=1, sticky="e", pady=5)
        
        # Preset buttons
        tk.Label(weights_frame, text="Preset:", font=("Arial", 9), bg="white").grid(
            row=6, column=0, columnspan=2, sticky="w", pady=(10, 5)
        )

    
    def _create_weight_row(self, parent, label, variable, row):
        """Tạo một hàng trọng số."""
        tk.Label(parent, text=label, bg="white").grid(row=row, column=0, sticky="w", pady=5)
        
        weight_spinbox = ttk.Spinbox(
            parent,
            from_=0,
            to=100,
            increment=5,
            textvariable=variable,
            width=10,
            command=self._update_total_weight
        )
        weight_spinbox.grid(row=row, column=1, sticky="e", pady=5)
        weight_spinbox.bind("<KeyRelease>", lambda e: self._update_total_weight())

    
    def _create_bottom_panel(self, parent):
        """Tạo panel dưới cùng với điểm tổng và buttons."""
        bottom_frame = tk.Frame(parent, bg="white")
        bottom_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=20)
        
        # Left: Điểm tổng
        total_frame = tk.LabelFrame(
            bottom_frame, 
            text="ĐIỂM TỔNG", 
            font=("Arial", 12, "bold"),
            bg="white",
            fg="#424242",
            padx=20,
            pady=20
        )
        total_frame.pack(side=tk.LEFT, padx=10)
        
        total_score_label = tk.Label(
            total_frame,
            textvariable=self.total_score,
            font=("Arial", 48, "bold"),
            foreground="#2ECC71",
            bg="white"
        )
        total_score_label.pack()
        
        tk.Label(total_frame, text="/10", font=("Arial", 16), bg="white").pack()
        
        self.rating_label = tk.Label(
            total_frame, text="", font=("Arial", 14, "bold"), bg="white"
        )
        self.rating_label.pack(pady=(10, 0))
        
        # Center: Quyết định tuyển dụng
        decision_frame = tk.LabelFrame(
            bottom_frame, 
            text="QUYẾT ĐỊNH", 
            font=("Arial", 12, "bold"),
            bg="white",
            fg="#424242",
            padx=20,
            pady=20
        )
        decision_frame.pack(side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True)
        
        self.decision_label = tk.Label(
            decision_frame,
            text="Chưa có quyết định",
            font=("Arial", 20, "bold"),
            fg="#757575",
            bg="white"
        )
        self.decision_label.pack(pady=10)
        
        self.decision_reason = tk.Label(
            decision_frame,
            text="",
            font=("Arial", 11),
            fg="#424242",
            bg="white",
            wraplength=300,
            justify=tk.CENTER
        )
        self.decision_reason.pack(pady=5)
        
        # Right: Buttons
        buttons_frame = tk.Frame(bottom_frame, bg="white")
        buttons_frame.pack(side=tk.LEFT, padx=10)
        
        # Nút lấy điểm từ các tab (quan trọng nhất)
        fetch_button = tk.Button(
            buttons_frame,
            text="📥 LẤY ĐIỂM TỪ CÁC TAB",
            command=self.fetch_scores_from_tabs,
            font=("Arial", 11, "bold"),
            bg="#2196F3",
            fg="white",
            activebackground="#1976D2",
            cursor="hand2",
            relief=tk.RAISED,
            bd=2,
            padx=15,
            pady=10
        )
        fetch_button.pack(fill=tk.X, pady=5)
        
        # Nút tính tổng (quan trọng thứ 2)
        calc_button = tk.Button(
            buttons_frame,
            text="🧮 TÍNH ĐIỂM TỔNG",
            command=self.calculate_total_score,
            font=("Arial", 11, "bold"),
            bg="#4CAF50",
            fg="white",
            activebackground="#388E3C",
            cursor="hand2",
            relief=tk.RAISED,
            bd=2,
            padx=15,
            pady=10
        )
        calc_button.pack(fill=tk.X, pady=5)
        
        # Nút xuất file
        export_button = tk.Button(
            buttons_frame,
            text="📄 XUẤT KẾT QUẢ (.TXT)",
            command=self.export_results,
            font=("Arial", 10),
            bg="#FF9800",
            fg="white",
            activebackground="#F57C00",
            cursor="hand2",
            relief=tk.RAISED,
            bd=2,
            padx=15,
            pady=8
        )
        export_button.pack(fill=tk.X, pady=5)
        
        # Nút lưu JSON
        save_json_button = tk.Button(
            buttons_frame,
            text="💾 LƯU JSON",
            command=self.save_json,
            font=("Arial", 10),
            bg="#9C27B0",
            fg="white",
            activebackground="#7B1FA2",
            cursor="hand2",
            relief=tk.RAISED,
            bd=2,
            padx=15,
            pady=8
        )
        save_json_button.pack(fill=tk.X, pady=5)
        
        # Nút reset
        reset_button = tk.Button(
            buttons_frame,
            text="🔄 RESET",
            command=self.reset_scores,
            font=("Arial", 10),
            bg="#607D8B",
            fg="white",
            activebackground="#455A64",
            cursor="hand2",
            relief=tk.RAISED,
            bd=2,
            padx=15,
            pady=8
        )
        reset_button.pack(fill=tk.X, pady=5)
    
    def _update_total_weight(self):
        """Cập nhật tổng trọng số."""
        total = (
            self.weight_content.get() +
            self.weight_clarity.get() +
            self.weight_focus.get() +
            self.weight_emotion.get()
        )
        
        self.total_weight_label.config(text=f"{total:.0f}%")
        
        if abs(total - 100) < 0.01:
            self.total_weight_label.config(foreground="green")
        else:
            self.total_weight_label.config(foreground="red")
    
    def _on_position_changed(self, event=None):
        """Khi thay đổi vị trí, áp dụng trọng số tương ứng."""
        position = self.position.get()
        self._apply_preset(position)

    
    def _apply_preset(self, preset_name):
        """Áp dụng trọng số preset."""
        presets = {
            "default": {
                "content": 0.40, "clarity": 0.35, "focus": 0.20, "emotion": 0.05
            },
            "technical": {
                "content": 0.45, "clarity": 0.30, "focus": 0.20, "emotion": 0.05
            },
            "sales": {
                "content": 0.35, "clarity": 0.35, "focus": 0.20, "emotion": 0.10
            },
            "customer_service": {
                "content": 0.30, "clarity": 0.40, "focus": 0.20, "emotion": 0.10
            },
            "management": {
                "content": 0.45, "clarity": 0.30, "focus": 0.20, "emotion": 0.05
            }
        }
        
        if preset_name in presets:
            weights = presets[preset_name]
            self.weight_content.set(weights["content"] * 100)
            self.weight_clarity.set(weights["clarity"] * 100)
            self.weight_focus.set(weights["focus"] * 100)
            self.weight_emotion.set(weights["emotion"] * 100)
            self._update_total_weight()
    
    def calculate_total_score(self):
        """Tính điểm tổng và hiển thị quyết định tuyển dụng."""
        # Kiểm tra trọng số
        total_weight = (
            self.weight_content.get() +
            self.weight_clarity.get() +
            self.weight_focus.get() +
            self.weight_emotion.get()
        )
        
        if abs(total_weight - 100) > 0.01:
            messagebox.showwarning(
                "Cảnh báo",
                f"Tổng trọng số phải bằng 100%!\nHiện tại: {total_weight:.1f}%"
            )
            return
        
        # Tính điểm tổng
        total = (
            self.content_score.get() * (self.weight_content.get() / 100) +
            self.clarity_score.get() * (self.weight_clarity.get() / 100) +
            self.focus_score.get() * (self.weight_focus.get() / 100) +
            self.emotion_score.get() * (self.weight_emotion.get() / 100)
        )
        
        self.total_score.set(round(total, 2))
        
        # Xác định đánh giá
        rating = self._get_rating(total)
        self.rating_label.config(text=rating)
        
        # Xác định quyết định tuyển dụng
        decision, reason, color = self._get_decision_details(total)
        self.decision_label.config(text=decision, fg=color)
        self.decision_reason.config(text=reason)
        
        # Hiển thị thông báo
        messagebox.showinfo(
            "Kết quả",
            f"Điểm tổng: {total:.2f}/10\n"
            f"Đánh giá: {rating}\n\n"
            f"Quyết định: {decision}\n"
            f"{reason}"
        )
    
    def _get_rating(self, score):
        """Lấy đánh giá từ điểm số."""
        if score >= 9.0:
            return "XUẤT SẮC ⭐⭐⭐"
        elif score >= 8.0:
            return "RẤT TỐT ⭐⭐"
        elif score >= 7.0:
            return "TỐT ⭐"
        elif score >= 6.0:
            return "KHÁ"
        elif score >= 5.0:
            return "TRUNG BÌNH"
        else:
            return "CẦN CẢI THIỆN"

    
    def export_results(self):
        """Xuất kết quả ra file .txt."""
        if self.total_score.get() == 0.0:
            messagebox.showwarning(
                "Cảnh báo",
                "Vui lòng tính điểm tổng trước khi xuất!"
            )
            return
        
        # Get default filename
        candidate_id = self.candidate_id.get() or 'Unknown'
        default_name = f"KetQua_{candidate_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        default_dir = str(Path("./reports").absolute())
        
        # Chọn file với custom dialog
        filename = ask_save_file(
            parent=self.parent,
            title="Xuất Kết Quả Phỏng Vấn",
            default_filename=default_name,
            default_dir=default_dir,
            file_extension=".txt",
            file_types=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if not filename:
            return
        
        # Tạo nội dung
        content = self._generate_report_content()
        
        # Ghi file
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            
            messagebox.showinfo(
                "Thành công",
                f"Đã xuất kết quả ra file:\n{filename}"
            )
        except Exception as e:
            messagebox.showerror(
                "Lỗi",
                f"Không thể xuất file:\n{str(e)}"
            )
    
    def _generate_report_content(self):
        """Tạo nội dung báo cáo."""
        lines = []
        lines.append("╔" + "═"*78 + "╗")
        lines.append("║" + " "*20 + "KẾT QUẢ ĐÁNH GIÁ PHỎNG VẤN" + " "*32 + "║")
        lines.append("╚" + "═"*78 + "╝")
        lines.append("")
        
        # Thông tin ứng viên
        lines.append("┌─ THÔNG TIN ỨNG VIÊN " + "─"*56 + "┐")
        lines.append(f"│  Họ tên:        {self.candidate_name.get() or 'N/A':<60}│")
        lines.append(f"│  Mã ứng viên:   {self.candidate_id.get() or 'N/A':<60}│")
        lines.append(f"│  Vị trí:        {self.position.get():<60}│")
        lines.append(f"│  Ngày đánh giá: {datetime.now().strftime('%d/%m/%Y %H:%M:%S'):<60}│")
        lines.append("└" + "─"*78 + "┘")
        lines.append("")
        
        # Điểm chi tiết
        lines.append("┌─ ĐIỂM CHI TIẾT (Thang 0-10) " + "─"*48 + "┐")
        lines.append("│" + " "*78 + "│")
        
        scores = [
            ("📝 Nội dung (Content)", self.content_score.get(), self.weight_content.get()),
            ("🗣️ Rõ ràng (Clarity)", self.clarity_score.get(), self.weight_clarity.get()),
            ("👁️ Tập trung (Focus)", self.focus_score.get(), self.weight_focus.get()),
            ("😊 Cảm xúc (Emotion)", self.emotion_score.get(), self.weight_emotion.get())
        ]
        
        for name, score, weight in scores:
            contribution = score * (weight / 100)
            lines.append(f"│  {name:<30}                                        │")
            lines.append(f"│    • Điểm:      {score:>5.2f}/10                                           │")
            lines.append(f"│    • Trọng số:  {weight:>5.0f}%                                             │")
            lines.append(f"│    • Đóng góp:  {contribution:>5.2f} điểm                                         │")
            lines.append("│" + " "*78 + "│")
        
        lines.append("└" + "─"*78 + "┘")
        lines.append("")
        
        # Điểm tổng
        total = self.total_score.get()
        rating = self._get_rating(total)
        
        lines.append("╔" + "═"*78 + "╗")
        lines.append("║" + " "*30 + "ĐIỂM TỔNG" + " "*39 + "║")
        lines.append("╠" + "═"*78 + "╣")
        lines.append(f"║  Điểm:     {total:>5.2f}/10" + " "*58 + "║")
        lines.append(f"║  Đánh giá: {rating:<60}║")
        lines.append("╚" + "═"*78 + "╝")
        lines.append("")
        
        # Kết luận
        decision, reason, _ = self._get_decision_details(total)
        
        lines.append("┌─ KẾT LUẬN " + "─"*66 + "┐")
        lines.append("│" + " "*78 + "│")
        lines.append(f"│  Quyết định: {decision:<63}│")
        lines.append("│" + " "*78 + "│")
        
        # Wrap reason text
        reason_lines = reason.split('\n')
        for reason_line in reason_lines:
            if len(reason_line) <= 74:
                lines.append(f"│  {reason_line:<76}│")
            else:
                # Split long lines
                words = reason_line.split()
                current_line = ""
                for word in words:
                    if len(current_line) + len(word) + 1 <= 74:
                        current_line += word + " "
                    else:
                        lines.append(f"│  {current_line:<76}│")
                        current_line = word + " "
                if current_line:
                    lines.append(f"│  {current_line:<76}│")
        
        lines.append("│" + " "*78 + "│")
        lines.append("└" + "─"*78 + "┘")
        lines.append("")
        
        # Chữ ký
        lines.append("─"*80)
        lines.append("Người đánh giá: ___________________    Ngày: ___/___/______")
        lines.append("")
        lines.append("Chữ ký: ___________________")
        lines.append("")
        lines.append("─"*80)
        lines.append("Hệ thống đánh giá phỏng vấn tự động - Emotion Recognition System")
        lines.append("─"*80)
        
        return "\n".join(lines)

    
    def save_json(self):
        """Lưu kết quả dạng JSON."""
        if self.total_score.get() == 0.0:
            messagebox.showwarning(
                "Cảnh báo",
                "Vui lòng tính điểm tổng trước khi lưu!"
            )
            return
        
        # Get default filename
        candidate_id = self.candidate_id.get() or 'Unknown'
        default_name = f"KetQua_{candidate_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        default_dir = str(Path("./reports").absolute())
        
        # Chọn file với custom dialog
        filename = ask_save_file(
            parent=self.parent,
            title="Lưu Kết Quả JSON",
            default_filename=default_name,
            default_dir=default_dir,
            file_extension=".json",
            file_types=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if not filename:
            return
        
        # Tạo data
        data = {
            "candidate_info": {
                "name": self.candidate_name.get(),
                "id": self.candidate_id.get(),
                "position": self.position.get(),
                "evaluation_date": datetime.now().isoformat()
            },
            "scores": {
                "emotion": self.emotion_score.get(),
                "focus": self.focus_score.get(),
                "clarity": self.clarity_score.get(),
                "content": self.content_score.get(),
                "total": self.total_score.get()
            },
            "weights": {
                "emotion": self.weight_emotion.get() / 100,
                "focus": self.weight_focus.get() / 100,
                "clarity": self.weight_clarity.get() / 100,
                "content": self.weight_content.get() / 100
            },
            "rating": self._get_rating(self.total_score.get()),
            "decision": self._get_decision(self.total_score.get())
        }
        
        # Ghi file
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            messagebox.showinfo(
                "Thành công",
                f"Đã lưu kết quả ra file JSON:\n{filename}"
            )
        except Exception as e:
            messagebox.showerror(
                "Lỗi",
                f"Không thể lưu file:\n{str(e)}"
            )
    
    def _get_decision(self, score):
        """Lấy quyết định từ điểm số."""
        if score >= 8.0:
            return "TUYỂN DỤNG"
        elif score >= 7.0:
            return "TUYỂN DỤNG CÓ ĐIỀU KIỆN"
        elif score >= 6.0:
            return "XEM XÉT THÊM"
        else:
            return "KHÔNG TUYỂN DỤNG"
    
    def _get_decision_details(self, score):
        """
        Lấy chi tiết quyết định tuyển dụng.
        
        Returns:
            (decision, reason, color)
        """
        if score >= 8.0:
            return (
                "✅ TUYỂN DỤNG",
                "Ứng viên có màn thể hiện xuất sắc/rất tốt.\nĐề xuất tuyển dụng ngay.",
                "#4CAF50"  # Green
            )
        elif score >= 7.0:
            return (
                "✅ TUYỂN DỤNG CÓ ĐIỀU KIỆN",
                "Ứng viên có màn thể hiện tốt.\nCó thể tuyển dụng với thời gian thử việc.",
                "#FF9800"  # Orange
            )
        elif score >= 6.0:
            return (
                "⚠️ CẦN XEM XÉT THÊM",
                "Ứng viên đạt mức chấp nhận được.\nCần phỏng vấn vòng 2 hoặc đánh giá kỹ hơn.",
                "#FFC107"  # Amber
            )
        else:
            return (
                "❌ KHÔNG TUYỂN DỤNG",
                "Ứng viên cần cải thiện nhiều.\nKhông phù hợp với vị trí hiện tại.",
                "#F44336"  # Red
            )
    
    def reset_scores(self):
        """Reset tất cả điểm về 0."""
        if messagebox.askyesno("Xác nhận", "Bạn có chắc muốn reset tất cả điểm?"):
            self.emotion_score.set(0.0)
            self.focus_score.set(0.0)
            self.clarity_score.set(0.0)
            self.content_score.set(0.0)
            self.total_score.set(0.0)
            self.rating_label.config(text="")
    
    # Public methods để cập nhật điểm từ các tab khác
    
    def update_emotion_score(self, score):
        """Cập nhật điểm cảm xúc từ video analysis."""
        self.emotion_score.set(round(score, 2))
    
    def update_focus_score(self, score):
        """Cập nhật điểm tập trung từ video analysis."""
        self.focus_score.set(round(score, 2))
    
    def update_clarity_score(self, score):
        """Cập nhật điểm rõ ràng từ speech analysis."""
        self.clarity_score.set(round(score, 2))
    
    def update_content_score(self, score):
        """Cập nhật điểm nội dung từ speech analysis."""
        self.content_score.set(round(score, 2))
    
    def update_video_scores(self, emotion_score, focus_score):
        """Cập nhật cả 2 điểm từ video analysis."""
        self.update_emotion_score(emotion_score)
        self.update_focus_score(focus_score)
    
    def update_speech_scores(self, clarity_score, content_score):
        """Cập nhật cả 2 điểm từ speech analysis."""
        self.update_clarity_score(clarity_score)
        self.update_content_score(content_score)
    
    def get_frame(self):
        """Lấy frame của tab."""
        return self.frame
    
    def fetch_scores_from_tabs(self):
        """
        Lấy điểm từ ScoreManager (các tab khác đã gửi điểm vào đây).
        
        Hiển thị dialog thông báo điểm nào đã có, điểm nào còn thiếu.
        """
        # Lấy tất cả điểm từ manager
        all_scores = self.score_manager.get_all_scores()
        
        # Cập nhật vào UI
        self.emotion_score.set(all_scores["emotion"]["score"])
        self.focus_score.set(all_scores["focus"]["score"])
        self.clarity_score.set(all_scores["clarity"]["score"])
        self.content_score.set(all_scores["content"]["score"])
        
        # Kiểm tra điểm nào còn thiếu
        missing = self.score_manager.get_missing_scores()
        
        if missing:
            message = "⚠️ CÒN THIẾU CÁC ĐIỂM SAU:\n\n"
            for score_name in missing:
                message += f"  • {score_name}\n"
            message += "\nVui lòng hoàn thành các bước sau:\n"
            message += "1. Tab 'Nhận Diện Cảm Xúc' → Quét khuôn mặt → Xuất điểm\n"
            message += "2. Tab 'Chuyển Đổi Audio' → Chuyển đổi → Xuất điểm\n"
            
            messagebox.showwarning("Thiếu Điểm", message)
        else:
            message = "✅ ĐÃ CÓ ĐỦ TẤT CẢ ĐIỂM!\n\n"
            message += f"📊 Cảm xúc: {all_scores['emotion']['score']:.2f}/10\n"
            message += f"🎯 Tập trung: {all_scores['focus']['score']:.2f}/10\n"
            message += f"🗣️ Rõ ràng: {all_scores['clarity']['score']:.2f}/10\n"
            message += f"📝 Nội dung: {all_scores['content']['score']:.2f}/10\n\n"
            message += "Nhấn 'TÍNH ĐIỂM TỔNG' để xem kết quả!"
            
            messagebox.showinfo("Đã Lấy Điểm", message)
    
    def _on_score_updated(self, score_type: str, score: float):
        """
        Callback khi có điểm mới từ ScoreManager.
        
        Tự động cập nhật UI khi các tab khác gửi điểm.
        
        Args:
            score_type: Loại điểm ("emotion", "focus", "clarity", "content")
            score: Giá trị điểm (0-10)
        """
        if score_type == "emotion":
            self.emotion_score.set(score)
        elif score_type == "focus":
            self.focus_score.set(score)
        elif score_type == "clarity":
            self.clarity_score.set(score)
        elif score_type == "content":
            self.content_score.set(score)
        elif score_type == "reset":
            # Reset tất cả
            self.emotion_score.set(0.0)
            self.focus_score.set(0.0)
            self.clarity_score.set(0.0)
            self.content_score.set(0.0)
            self.total_score.set(0.0)
            self.rating_label.config(text="")
