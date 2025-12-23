# -*- coding: utf-8 -*-
"""
Score Summary Tab V2 - Phiên bản đơn giản, dễ debug

Tab tổng hợp điểm với giao diện rõ ràng, dễ nhìn.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
from pathlib import Path
import json
from apps.gui.file_save_dialog import ask_save_file
from apps.gui.score_manager import get_score_manager


class ScoreSummaryTab:
    """Tab tổng hợp điểm phỏng vấn - Phiên bản 2."""
    
    def __init__(self, parent):
        """Khởi tạo tab."""
        print("[ScoreSummaryTab] Initializing...")
        self.parent = parent
        
        # Tạo frame chính với màu nền tối (dark theme)
        self.frame = tk.Frame(parent, bg="#1a1a1a")
        
        # Điểm số (dùng StringVar để format)
        self.emotion_score = tk.StringVar(value="0.0")
        self.focus_score = tk.StringVar(value="0.0")
        self.clarity_score = tk.StringVar(value="0.0")
        self.content_score = tk.StringVar(value="0.0")
        self.total_score = tk.StringVar(value="0.0")
        
        # Điểm số thực (để tính toán)
        self._emotion_score_value = 0.0
        self._focus_score_value = 0.0
        self._clarity_score_value = 0.0
        self._content_score_value = 0.0
        self._total_score_value = 0.0
        
        # Trọng số (theo công thức: N=40%, T=20%, G=35%, O=5%)
        self.weight_content = tk.DoubleVar(value=40.0)    # Nội dung (N)
        self.weight_focus = tk.DoubleVar(value=20.0)      # Tập trung (T)
        self.weight_clarity = tk.DoubleVar(value=35.0)    # Giọng nói (G)
        self.weight_emotion = tk.DoubleVar(value=5.0)     # Cảm xúc (O)
        
        # Thông tin ứng viên
        self.candidate_name = tk.StringVar(value="")
        self.candidate_id = tk.StringVar(value="")
        self.position = tk.StringVar(value="default")
        
        # Score Manager
        print("[ScoreSummaryTab] Getting ScoreManager...")
        self.score_manager = get_score_manager()
        print(f"[ScoreSummaryTab] ScoreManager ID: {id(self.score_manager)}")
        
        print("[ScoreSummaryTab] Registering callback...")
        self.score_manager.register_callback(self._on_score_updated)
        print("[ScoreSummaryTab] Callback registered!")
        
        # Lấy điểm hiện tại từ manager (nếu có)
        print("[ScoreSummaryTab] Loading existing scores...")
        all_scores = self.score_manager.get_all_scores()
        self._set_emotion_score(all_scores["emotion"]["score"])
        self._set_focus_score(all_scores["focus"]["score"])
        self._set_clarity_score(all_scores["clarity"]["score"])
        self._set_content_score(all_scores["content"]["score"])
        print(f"  Emotion: {all_scores['emotion']['score']:.1f}")
        print(f"  Focus: {all_scores['focus']['score']:.1f}")
        print(f"  Clarity: {all_scores['clarity']['score']:.1f}")
        print(f"  Content: {all_scores['content']['score']:.1f}")
        
        # Tạo UI
        print("[ScoreSummaryTab] Creating UI...")
        self._create_ui()
        print("[ScoreSummaryTab] Initialization complete!")
        
        # Bind event khi tab được hiển thị
        self.frame.bind("<Visibility>", self._on_tab_visible)
        
        # Start auto-refresh timer (check mỗi 2 giây)
        self._last_check_time = 0
        self._start_auto_refresh()
    
    def _create_ui(self):
        """Tạo giao diện."""
        # Header
        header = tk.Frame(self.frame, bg="#0d47a1", height=60)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        tk.Label(
            header,
            text="📊 TỔNG HỢP ĐIỂM PHỎNG VẤN",
            font=("Arial", 18, "bold"),
            bg="#0d47a1",
            fg="white"
        ).pack(pady=15)
        
        # Main content
        content = tk.Frame(self.frame, bg="#1a1a1a")
        content.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Row 1: Thông tin + 4 điểm + Trọng số
        row1 = tk.Frame(content, bg="#1a1a1a")
        row1.pack(fill=tk.X, pady=(0, 20))
        
        self._create_info_panel(row1)
        self._create_scores_panel(row1)
        self._create_weights_panel(row1)
        
        # Row 2: Điểm tổng + Quyết định + Buttons
        row2 = tk.Frame(content, bg="#1a1a1a")
        row2.pack(fill=tk.X)
        
        self._create_total_panel(row2)
        self._create_decision_panel(row2)
        self._create_buttons_panel(row2)
    
    def _create_info_panel(self, parent):
        """Panel thông tin ứng viên."""
        frame = tk.LabelFrame(
            parent,
            text=" Thông Tin Ứng Viên ",
            font=("Arial", 10, "bold"),
            bg="#252525",
            fg="#e0e0e0",
            relief=tk.GROOVE,
            bd=2
        )
        frame.pack(side=tk.LEFT, padx=(0, 10), fill=tk.BOTH)
        
        # Họ tên
        tk.Label(frame, text="Họ tên:", bg="#252525", fg="#e0e0e0", anchor="w").grid(
            row=0, column=0, sticky="w", padx=10, pady=5
        )
        tk.Entry(frame, textvariable=self.candidate_name, width=20, bg="#333333", fg="#ffffff", insertbackground="#ffffff").grid(
            row=0, column=1, padx=10, pady=5
        )
        
        # Mã ứng viên
        tk.Label(frame, text="Mã ứng viên:", bg="#252525", fg="#e0e0e0", anchor="w").grid(
            row=1, column=0, sticky="w", padx=10, pady=5
        )
        tk.Entry(frame, textvariable=self.candidate_id, width=20, bg="#333333", fg="#ffffff", insertbackground="#ffffff").grid(
            row=1, column=1, padx=10, pady=5
        )
        
        # Vị trí
        tk.Label(frame, text="Vị trí:", bg="#252525", fg="#e0e0e0", anchor="w").grid(
            row=2, column=0, sticky="w", padx=10, pady=5
        )
        combo = ttk.Combobox(
            frame,
            textvariable=self.position,
            values=["default", "technical", "sales", "customer_service", "management"],
            state="readonly",
            width=17
        )
        combo.grid(row=2, column=1, padx=10, pady=5)
        combo.bind("<<ComboboxSelected>>", self._on_position_changed)
    
    def _create_scores_panel(self, parent):
        """Panel 4 điểm."""
        frame = tk.LabelFrame(
            parent,
            text=" Điểm Đánh Giá (0-10) ",
            font=("Arial", 10, "bold"),
            bg="#252525",
            fg="#e0e0e0",
            relief=tk.GROOVE,
            bd=2
        )
        frame.pack(side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True)
        
        # Grid 2x2
        scores = [
            ("😊 Cảm xúc", self.emotion_score, "#FF6B6B", 0, 0),
            ("👁️ Tập trung", self.focus_score, "#4ECDC4", 0, 1),
            ("🗣️ Rõ ràng", self.clarity_score, "#95E1D3", 1, 0),
            ("📝 Nội dung", self.content_score, "#F38181", 1, 1)
        ]
        
        for title, var, color, row, col in scores:
            box = tk.Frame(frame, bg="#252525", relief=tk.SOLID, bd=1)
            box.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
            
            tk.Label(box, text=title, font=("Arial", 11, "bold"), bg="#252525", fg="#ffffff").pack(pady=5)
            tk.Label(
                box,
                textvariable=var,
                font=("Arial", 32, "bold"),
                fg=color,
                bg="#252525"
            ).pack()
            tk.Label(box, text="/10", font=("Arial", 10), bg="#252525", fg="#ffffff").pack(pady=5)
        
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_columnconfigure(1, weight=1)
    
    def _create_weights_panel(self, parent):
        """Panel trọng số."""
        frame = tk.LabelFrame(
            parent,
            text=" Trọng Số (%) ",
            font=("Arial", 10, "bold"),
            bg="#252525",
            fg="#e0e0e0",
            relief=tk.GROOVE,
            bd=2
        )
        frame.pack(side=tk.LEFT, padx=(10, 0), fill=tk.BOTH)
        
        weights = [
            ("📝 Nội dung:", self.weight_content),
            ("🗣️ Rõ ràng:", self.weight_clarity),
            ("👁️ Tập trung:", self.weight_focus),
            ("😊 Cảm xúc:", self.weight_emotion)
        ]
        
        for i, (label, var) in enumerate(weights):
            tk.Label(frame, text=label, bg="#252525", fg="#ffffff").grid(
                row=i, column=0, sticky="w", padx=10, pady=5
            )
            spinbox = ttk.Spinbox(
                frame,
                from_=0,
                to=100,
                increment=5,
                textvariable=var,
                width=8,
                command=self._update_total_weight
            )
            spinbox.grid(row=i, column=1, padx=10, pady=5)
            spinbox.bind("<KeyRelease>", lambda e: self._update_total_weight())
        
        # Tổng
        tk.Frame(frame, height=2, bg="#444444").grid(
            row=4, column=0, columnspan=2, sticky="ew", padx=10, pady=5
        )
        
        tk.Label(frame, text="Tổng:", font=("Arial", 10, "bold"), bg="#252525", fg="#ffffff").grid(
            row=5, column=0, sticky="w", padx=10, pady=5
        )
        
        self.total_weight_label = tk.Label(
            frame,
            text="100%",
            font=("Arial", 10, "bold"),
            fg="#4CAF50",
            bg="#252525"
        )
        self.total_weight_label.grid(row=5, column=1, padx=10, pady=5)

    
    def _create_total_panel(self, parent):
        """Panel điểm tổng."""
        frame = tk.LabelFrame(
            parent,
            text=" ĐIỂM TỔNG ",
            font=("Arial", 11, "bold"),
            bg="#252525",
            fg="#e0e0e0",
            relief=tk.GROOVE,
            bd=2
        )
        frame.pack(side=tk.LEFT, padx=(0, 10), ipadx=20, ipady=10)
        
        tk.Label(
            frame,
            textvariable=self.total_score,
            font=("Arial", 48, "bold"),
            fg="#2ECC71",
            bg="#252525"
        ).pack()
        
        tk.Label(frame, text="/10", font=("Arial", 14), bg="#252525", fg="#ffffff").pack()
        
        self.rating_label = tk.Label(
            frame,
            text="",
            font=("Arial", 12, "bold"),
            bg="#252525"
        )
        self.rating_label.pack(pady=5)
    
    def _create_decision_panel(self, parent):
        """Panel quyết định."""
        frame = tk.LabelFrame(
            parent,
            text=" QUYẾT ĐỊNH TUYỂN DỤNG ",
            font=("Arial", 11, "bold"),
            bg="#252525",
            fg="#e0e0e0",
            relief=tk.GROOVE,
            bd=2
        )
        frame.pack(side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True, ipadx=10, ipady=10)
        
        self.decision_label = tk.Label(
            frame,
            text="Chưa có quyết định",
            font=("Arial", 16, "bold"),
            fg="#BDBDBD",
            bg="#252525"
        )
        self.decision_label.pack(pady=10)
        
        self.decision_reason = tk.Label(
            frame,
            text="Vui lòng tính điểm tổng để xem quyết định",
            font=("Arial", 10),
            fg="#BDBDBD",
            bg="#252525",
            wraplength=300,
            justify=tk.CENTER
        )
        self.decision_reason.pack()
    
    def _create_buttons_panel(self, parent):
        """Panel buttons."""
        frame = tk.Frame(parent, bg="#1a1a1a")
        frame.pack(side=tk.LEFT, padx=(10, 0))
        
        buttons = [
            ("📥 LẤY ĐIỂM", self.fetch_scores_from_tabs, "#2196F3", "#1976D2"),
            ("🧮 TÍNH TỔNG", self.calculate_total_score, "#4CAF50", "#388E3C"),
            ("📄 XUẤT TXT", self.export_results, "#FF9800", "#F57C00"),
            ("💾 LƯU JSON", self.save_json, "#9C27B0", "#7B1FA2"),
            ("🔄 RESET", self.reset_scores, "#607D8B", "#455A64")
        ]
        
        for text, command, bg, active_bg in buttons:
            tk.Button(
                frame,
                text=text,
                command=command,
                font=("Arial", 10, "bold"),
                bg=bg,
                fg="white",
                activebackground=active_bg,
                activeforeground="white",
                relief=tk.RAISED,
                bd=2,
                padx=15,
                pady=8,
                cursor="hand2",
                width=18
            ).pack(pady=3)
    
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
            self.total_weight_label.config(fg="#4CAF50")
        else:
            self.total_weight_label.config(fg="red")
    
    def _on_position_changed(self, event=None):
        """Khi thay đổi vị trí."""
        self._apply_preset(self.position.get())
    
    def _apply_preset(self, preset_name):
        """Áp dụng trọng số preset."""
        presets = {
            "default": {"content": 40, "clarity": 35, "focus": 20, "emotion": 5},
            "technical": {"content": 45, "clarity": 30, "focus": 20, "emotion": 5},
            "sales": {"content": 35, "clarity": 35, "focus": 20, "emotion": 10},
            "customer_service": {"content": 30, "clarity": 40, "focus": 20, "emotion": 10},
            "management": {"content": 45, "clarity": 30, "focus": 20, "emotion": 5}
        }
        
        if preset_name in presets:
            w = presets[preset_name]
            self.weight_content.set(w["content"])
            self.weight_clarity.set(w["clarity"])
            self.weight_focus.set(w["focus"])
            self.weight_emotion.set(w["emotion"])
            self._update_total_weight()
    
    def fetch_scores_from_tabs(self):
        """Lấy điểm từ các tab."""
        print("\n[ScoreSummaryTab] Fetching scores from ScoreManager...")
        print(f"  ScoreManager ID: {id(self.score_manager)}")
        
        all_scores = self.score_manager.get_all_scores()
        
        print(f"  Scores in manager:")
        print(f"    Emotion: {all_scores['emotion']['score']:.1f} (from {all_scores['emotion']['source']})")
        print(f"    Focus: {all_scores['focus']['score']:.1f} (from {all_scores['focus']['source']})")
        print(f"    Clarity: {all_scores['clarity']['score']:.1f} (from {all_scores['clarity']['source']})")
        print(f"    Content: {all_scores['content']['score']:.1f} (from {all_scores['content']['source']})")
        
        self._set_emotion_score(all_scores["emotion"]["score"])
        self._set_focus_score(all_scores["focus"]["score"])
        self._set_clarity_score(all_scores["clarity"]["score"])
        self._set_content_score(all_scores["content"]["score"])
        
        print(f"  ✓ Scores loaded into UI")
        
        missing = self.score_manager.get_missing_scores()
        
        if missing:
            messagebox.showwarning(
                "Thiếu Điểm",
                f"Còn thiếu: {', '.join(missing)}\n\n"
                "Vui lòng hoàn thành các bước trước."
            )
        else:
            messagebox.showinfo(
                "Thành Công",
                f"✅ Đã lấy đủ 4 điểm!\n\n"
                f"Cảm xúc: {all_scores['emotion']['score']:.1f}\n"
                f"Tập trung: {all_scores['focus']['score']:.1f}\n"
                f"Rõ ràng: {all_scores['clarity']['score']:.1f}\n"
                f"Nội dung: {all_scores['content']['score']:.1f}"
            )
    
    def calculate_total_score(self):
        """Tính điểm tổng."""
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
        
        total = (
            self._content_score_value * (self.weight_content.get() / 100) +
            self._clarity_score_value * (self.weight_clarity.get() / 100) +
            self._focus_score_value * (self.weight_focus.get() / 100) +
            self._emotion_score_value * (self.weight_emotion.get() / 100)
        )
        
        self._set_total_score(total)
        
        # Đánh giá
        if total >= 9.0:
            rating = "XUẤT SẮC ⭐⭐⭐"
        elif total >= 8.0:
            rating = "RẤT TỐT ⭐⭐"
        elif total >= 7.0:
            rating = "TỐT ⭐"
        elif total >= 6.0:
            rating = "KHÁ"
        elif total >= 5.0:
            rating = "TRUNG BÌNH"
        else:
            rating = "CẦN CẢI THIỆN"
        
        self.rating_label.config(text=rating)
        
        # Quyết định
        if total >= 8.0:
            decision = "✅ TUYỂN DỤNG"
            reason = "Ứng viên có màn thể hiện xuất sắc/rất tốt.\nĐề xuất tuyển dụng ngay."
            color = "#4CAF50"
        elif total >= 7.0:
            decision = "✅ TUYỂN DỤNG CÓ ĐIỀU KIỆN"
            reason = "Ứng viên có màn thể hiện tốt.\nCó thể tuyển dụng với thời gian thử việc."
            color = "#FF9800"
        elif total >= 6.0:
            decision = "⚠️ CẦN XEM XÉT THÊM"
            reason = "Ứng viên đạt mức chấp nhận được.\nCần phỏng vấn vòng 2 hoặc đánh giá kỹ hơn."
            color = "#FFC107"
        else:
            decision = "❌ KHÔNG TUYỂN DỤNG"
            reason = "Ứng viên cần cải thiện nhiều.\nKhông phù hợp với vị trí hiện tại."
            color = "#F44336"
        
        self.decision_label.config(text=decision, fg=color)
        self.decision_reason.config(text=reason, fg="#e0e0e0")
        
        messagebox.showinfo(
            "Kết Quả",
            f"Điểm tổng: {total:.1f}/10\n"
            f"Đánh giá: {rating}\n\n"
            f"Quyết định: {decision}"
        )
    
    def export_results(self):
        """Xuất kết quả ra file .txt."""
        if self._total_score_value == 0.0:
            messagebox.showwarning("Cảnh báo", "Vui lòng tính điểm tổng trước!")
            return
        
        candidate_id = self.candidate_id.get() or 'Unknown'
        default_name = f"KetQua_{candidate_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        default_dir = str(Path("./reports").absolute())
        
        filename = ask_save_file(
            parent=self.parent,
            title="Xuất Kết Quả",
            default_filename=default_name,
            default_dir=default_dir,
            file_extension=".txt",
            file_types=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if not filename:
            return
        
        # Tạo nội dung (đơn giản)
        lines = []
        lines.append("="*80)
        lines.append("KẾT QUẢ ĐÁNH GIÁ PHỎNG VẤN")
        lines.append("="*80)
        lines.append("")
        lines.append(f"Họ tên: {self.candidate_name.get() or 'N/A'}")
        lines.append(f"Mã ứng viên: {self.candidate_id.get() or 'N/A'}")
        lines.append(f"Vị trí: {self.position.get()}")
        lines.append(f"Ngày: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
        lines.append("")
        lines.append("-"*80)
        lines.append("ĐIỂM CHI TIẾT:")
        lines.append("-"*80)
        lines.append(f"Nội dung:   {self._content_score_value:.1f}/10 ({self.weight_content.get():.0f}%)")
        lines.append(f"Rõ ràng:    {self._clarity_score_value:.1f}/10 ({self.weight_clarity.get():.0f}%)")
        lines.append(f"Tập trung:  {self._focus_score_value:.1f}/10 ({self.weight_focus.get():.0f}%)")
        lines.append(f"Cảm xúc:    {self._emotion_score_value:.1f}/10 ({self.weight_emotion.get():.0f}%)")
        lines.append("")
        lines.append("="*80)
        lines.append(f"ĐIỂM TỔNG: {self._total_score_value:.1f}/10")
        lines.append(f"ĐÁNH GIÁ: {self.rating_label.cget('text')}")
        lines.append(f"QUYẾT ĐỊNH: {self.decision_label.cget('text')}")
        lines.append("="*80)
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("\n".join(lines))
            messagebox.showinfo("Thành công", f"Đã xuất ra:\n{filename}")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể xuất file:\n{str(e)}")
    
    def save_json(self):
        """Lưu JSON."""
        if self._total_score_value == 0.0:
            messagebox.showwarning("Cảnh báo", "Vui lòng tính điểm tổng trước!")
            return
        
        candidate_id = self.candidate_id.get() or 'Unknown'
        default_name = f"KetQua_{candidate_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        default_dir = str(Path("./reports").absolute())
        
        filename = ask_save_file(
            parent=self.parent,
            title="Lưu JSON",
            default_filename=default_name,
            default_dir=default_dir,
            file_extension=".json",
            file_types=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if not filename:
            return
        
        data = {
            "candidate_info": {
                "name": self.candidate_name.get(),
                "id": self.candidate_id.get(),
                "position": self.position.get(),
                "date": datetime.now().isoformat()
            },
            "scores": {
                "emotion": round(self._emotion_score_value, 1),
                "focus": round(self._focus_score_value, 1),
                "clarity": round(self._clarity_score_value, 1),
                "content": round(self._content_score_value, 1),
                "total": round(self._total_score_value, 1)
            },
            "weights": {
                "emotion": self.weight_emotion.get() / 100,
                "focus": self.weight_focus.get() / 100,
                "clarity": self.weight_clarity.get() / 100,
                "content": self.weight_content.get() / 100
            }
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            messagebox.showinfo("Thành công", f"Đã lưu ra:\n{filename}")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể lưu file:\n{str(e)}")
    
    def reset_scores(self):
        """Reset điểm."""
        if messagebox.askyesno("Xác nhận", "Bạn có chắc muốn reset?"):
            self._set_emotion_score(0.0)
            self._set_focus_score(0.0)
            self._set_clarity_score(0.0)
            self._set_content_score(0.0)
            self._set_total_score(0.0)
            self.rating_label.config(text="")
            self.decision_label.config(text="Chưa có quyết định", fg="#BDBDBD")
            self.decision_reason.config(
                text="Vui lòng tính điểm tổng để xem quyết định",
                fg="#BDBDBD"
            )
    
    def _set_emotion_score(self, score: float):
        """Set emotion score với format."""
        self._emotion_score_value = score
        self.emotion_score.set(f"{score:.1f}")
    
    def _set_focus_score(self, score: float):
        """Set focus score với format."""
        self._focus_score_value = score
        self.focus_score.set(f"{score:.1f}")
    
    def _set_clarity_score(self, score: float):
        """Set clarity score với format."""
        self._clarity_score_value = score
        self.clarity_score.set(f"{score:.1f}")
    
    def _set_content_score(self, score: float):
        """Set content score với format."""
        self._content_score_value = score
        self.content_score.set(f"{score:.1f}")
    
    def _set_total_score(self, score: float):
        """Set total score với format."""
        self._total_score_value = score
        self.total_score.set(f"{score:.1f}")
    
    def _on_score_updated(self, score_type: str, score: float):
        """Callback khi có điểm mới."""
        print(f"[ScoreSummaryTab] Received score update: {score_type} = {score:.1f}")
        
        try:
            if score_type == "emotion":
                self._set_emotion_score(score)
                print(f"  → Emotion score updated to {score:.1f}")
            elif score_type == "focus":
                self._set_focus_score(score)
                print(f"  → Focus score updated to {score:.1f}")
            elif score_type == "clarity":
                self._set_clarity_score(score)
                print(f"  → Clarity score updated to {score:.1f}")
            elif score_type == "content":
                self._set_content_score(score)
                print(f"  → Content score updated to {score:.1f}")
        except Exception as e:
            print(f"  ✗ Error updating score: {e}")
    
    def _start_auto_refresh(self):
        """Bắt đầu auto-refresh timer."""
        self._auto_refresh()
    
    def _auto_refresh(self):
        """Tự động refresh điểm từ ScoreManager."""
        try:
            import time
            current_time = time.time()
            
            # Chỉ check mỗi 2 giây
            if current_time - self._last_check_time >= 2.0:
                self._last_check_time = current_time
                
                all_scores = self.score_manager.get_all_scores()
                
                # Chỉ update nếu có thay đổi
                if (all_scores["emotion"]["score"] != self._emotion_score_value or
                    all_scores["focus"]["score"] != self._focus_score_value or
                    all_scores["clarity"]["score"] != self._clarity_score_value or
                    all_scores["content"]["score"] != self._content_score_value):
                    
                    print(f"\n[ScoreSummaryTab] Auto-refresh detected changes:")
                    print(f"  Emotion: {self._emotion_score_value:.1f} → {all_scores['emotion']['score']:.1f}")
                    print(f"  Focus: {self._focus_score_value:.1f} → {all_scores['focus']['score']:.1f}")
                    print(f"  Clarity: {self._clarity_score_value:.1f} → {all_scores['clarity']['score']:.1f}")
                    print(f"  Content: {self._content_score_value:.1f} → {all_scores['content']['score']:.1f}")
                    
                    self._set_emotion_score(all_scores["emotion"]["score"])
                    self._set_focus_score(all_scores["focus"]["score"])
                    self._set_clarity_score(all_scores["clarity"]["score"])
                    self._set_content_score(all_scores["content"]["score"])
                    
                    print(f"  ✓ UI updated!")
        except Exception as e:
            print(f"[ScoreSummaryTab] Auto-refresh error: {e}")
        
        # Schedule next check (sau 1 giây)
        self.frame.after(1000, self._auto_refresh)
    
    def _on_tab_visible(self, event=None):
        """Callback khi tab được hiển thị - tự động load điểm."""
        print("\n[ScoreSummaryTab] Tab became visible - loading scores...")
        try:
            all_scores = self.score_manager.get_all_scores()
            
            self._set_emotion_score(all_scores["emotion"]["score"])
            self._set_focus_score(all_scores["focus"]["score"])
            self._set_clarity_score(all_scores["clarity"]["score"])
            self._set_content_score(all_scores["content"]["score"])
            
            print(f"  ✓ Scores loaded!")
        except Exception as e:
            print(f"  ✗ Error loading scores: {e}")
    
    def get_frame(self):
        """Lấy frame."""
        return self.frame
