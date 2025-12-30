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
        self.parent = parent
        
        # Tạo frame chính với màu nền sáng
        self.frame = tk.Frame(parent, bg="#F5F5F5")
        
        # Điểm số
        self.emotion_score = tk.DoubleVar(value=0.0)
        self.focus_score = tk.DoubleVar(value=0.0)
        self.clarity_score = tk.DoubleVar(value=0.0)
        self.content_score = tk.DoubleVar(value=0.0)
        self.total_score = tk.DoubleVar(value=0.0)
        
        # Trọng số mặc định (Content=40%, Clarity=35%, Focus=20%, Emotion=5%)
        # Phù hợp với công thức: Total = (C×40% + Cl×35% + F×20% + E×5%)
        self.weight_content = tk.DoubleVar(value=40.0)    # Nội dung (N)
        self.weight_clarity = tk.DoubleVar(value=35.0)    # Rõ ràng (G)
        self.weight_focus = tk.DoubleVar(value=20.0)      # Tập trung (T)
        self.weight_emotion = tk.DoubleVar(value=5.0)     # Cảm xúc (O)
        
        # Thông tin ứng viên
        self.candidate_name = tk.StringVar(value="")
        self.candidate_id = tk.StringVar(value="")
        self.position = tk.StringVar(value="default")
        
        # Score Manager
        self.score_manager = get_score_manager()
        self.score_manager.register_callback(self._on_score_updated)
        
        # Tạo UI
        self._create_ui()
    
    def _create_ui(self):
        """Tạo giao diện."""
        # Header
        header = tk.Frame(self.frame, bg="#1976D2", height=60)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        tk.Label(
            header,
            text="📊 TỔNG HỢP ĐIỂM PHỎNG VẤN",
            font=("Arial", 18, "bold"),
            bg="#1976D2",
            fg="white"
        ).pack(pady=15)
        
        # Main content
        content = tk.Frame(self.frame, bg="#F5F5F5")
        content.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Row 1: Thông tin + 4 điểm + Trọng số
        row1 = tk.Frame(content, bg="#F5F5F5")
        row1.pack(fill=tk.X, pady=(0, 20))
        
        self._create_info_panel(row1)
        self._create_scores_panel(row1)
        self._create_weights_panel(row1)
        
        # Row 2: Điểm tổng + Quyết định + Buttons
        row2 = tk.Frame(content, bg="#F5F5F5")
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
            bg="white",
            fg="#424242",
            relief=tk.GROOVE,
            bd=2
        )
        frame.pack(side=tk.LEFT, padx=(0, 10), fill=tk.BOTH)
        
        # Họ tên
        tk.Label(frame, text="Họ tên:", bg="white", anchor="w").grid(
            row=0, column=0, sticky="w", padx=10, pady=5
        )
        tk.Entry(frame, textvariable=self.candidate_name, width=20).grid(
            row=0, column=1, padx=10, pady=5
        )
        
        # Mã ứng viên
        tk.Label(frame, text="Mã ứng viên:", bg="white", anchor="w").grid(
            row=1, column=0, sticky="w", padx=10, pady=5
        )
        tk.Entry(frame, textvariable=self.candidate_id, width=20).grid(
            row=1, column=1, padx=10, pady=5
        )
        
        # Vị trí
        tk.Label(frame, text="Vị trí:", bg="white", anchor="w").grid(
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
            bg="white",
            fg="#424242",
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
            box = tk.Frame(frame, bg="white", relief=tk.SOLID, bd=1)
            box.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
            
            tk.Label(box, text=title, font=("Arial", 11, "bold"), bg="white").pack(pady=5)
            tk.Label(
                box,
                textvariable=var,
                font=("Arial", 32, "bold"),
                fg=color,
                bg="white"
            ).pack()
            tk.Label(box, text="/10", font=("Arial", 10), bg="white").pack(pady=5)
        
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_columnconfigure(1, weight=1)
    
    def _create_weights_panel(self, parent):
        """Panel trọng số."""
        frame = tk.LabelFrame(
            parent,
            text=" Trọng Số (%) ",
            font=("Arial", 10, "bold"),
            bg="white",
            fg="#424242",
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
            tk.Label(frame, text=label, bg="white").grid(
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
        tk.Frame(frame, height=2, bg="#CCCCCC").grid(
            row=4, column=0, columnspan=2, sticky="ew", padx=10, pady=5
        )
        
        tk.Label(frame, text="Tổng:", font=("Arial", 10, "bold"), bg="white").grid(
            row=5, column=0, sticky="w", padx=10, pady=5
        )
        
        self.total_weight_label = tk.Label(
            frame,
            text="100%",
            font=("Arial", 10, "bold"),
            fg="green",
            bg="white"
        )
        self.total_weight_label.grid(row=5, column=1, padx=10, pady=5)

    
    def _create_total_panel(self, parent):
        """Panel điểm tổng."""
        frame = tk.LabelFrame(
            parent,
            text=" ĐIỂM TỔNG ",
            font=("Arial", 11, "bold"),
            bg="white",
            fg="#424242",
            relief=tk.GROOVE,
            bd=2
        )
        frame.pack(side=tk.LEFT, padx=(0, 10), ipadx=20, ipady=10)
        
        tk.Label(
            frame,
            textvariable=self.total_score,
            font=("Arial", 48, "bold"),
            fg="#2ECC71",
            bg="white"
        ).pack()
        
        tk.Label(frame, text="/10", font=("Arial", 14), bg="white").pack()
        
        self.rating_label = tk.Label(
            frame,
            text="",
            font=("Arial", 12, "bold"),
            bg="white"
        )
        self.rating_label.pack(pady=5)
    
    def _create_decision_panel(self, parent):
        """Panel quyết định."""
        frame = tk.LabelFrame(
            parent,
            text=" QUYẾT ĐỊNH TUYỂN DỤNG ",
            font=("Arial", 11, "bold"),
            bg="white",
            fg="#424242",
            relief=tk.GROOVE,
            bd=2
        )
        frame.pack(side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True, ipadx=10, ipady=10)
        
        self.decision_label = tk.Label(
            frame,
            text="Chưa có quyết định",
            font=("Arial", 16, "bold"),
            fg="#757575",
            bg="white"
        )
        self.decision_label.pack(pady=10)
        
        self.decision_reason = tk.Label(
            frame,
            text="Vui lòng tính điểm tổng để xem quyết định",
            font=("Arial", 10),
            fg="#9E9E9E",
            bg="white",
            wraplength=300,
            justify=tk.CENTER
        )
        self.decision_reason.pack()
    
    def _create_buttons_panel(self, parent):
        """Panel buttons."""
        frame = tk.Frame(parent, bg="#F5F5F5")
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
            self.total_weight_label.config(fg="green")
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
        all_scores = self.score_manager.get_all_scores()
        
        self.emotion_score.set(all_scores["emotion"]["score"])
        self.focus_score.set(all_scores["focus"]["score"])
        self.clarity_score.set(all_scores["clarity"]["score"])
        self.content_score.set(all_scores["content"]["score"])
        
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
                f"Cảm xúc: {all_scores['emotion']['score']:.2f}\n"
                f"Tập trung: {all_scores['focus']['score']:.2f}\n"
                f"Rõ ràng: {all_scores['clarity']['score']:.2f}\n"
                f"Nội dung: {all_scores['content']['score']:.2f}"
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
            self.content_score.get() * (self.weight_content.get() / 100) +
            self.clarity_score.get() * (self.weight_clarity.get() / 100) +
            self.focus_score.get() * (self.weight_focus.get() / 100) +
            self.emotion_score.get() * (self.weight_emotion.get() / 100)
        )
        
        self.total_score.set(round(total, 2))
        
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
        self.decision_reason.config(text=reason, fg="#424242")
        
        messagebox.showinfo(
            "Kết Quả",
            f"Điểm tổng: {total:.2f}/10\n"
            f"Đánh giá: {rating}\n\n"
            f"Quyết định: {decision}"
        )
    
    def export_results(self):
        """Xuất kết quả ra file .txt."""
        if self.total_score.get() == 0.0:
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
        lines.append(f"Nội dung:   {self.content_score.get():.2f}/10 ({self.weight_content.get():.0f}%)")
        lines.append(f"Rõ ràng:    {self.clarity_score.get():.2f}/10 ({self.weight_clarity.get():.0f}%)")
        lines.append(f"Tập trung:  {self.focus_score.get():.2f}/10 ({self.weight_focus.get():.0f}%)")
        lines.append(f"Cảm xúc:    {self.emotion_score.get():.2f}/10 ({self.weight_emotion.get():.0f}%)")
        lines.append("")
        lines.append("="*80)
        lines.append(f"ĐIỂM TỔNG: {self.total_score.get():.2f}/10")
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
        if self.total_score.get() == 0.0:
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
            self.emotion_score.set(0.0)
            self.focus_score.set(0.0)
            self.clarity_score.set(0.0)
            self.content_score.set(0.0)
            self.total_score.set(0.0)
            self.rating_label.config(text="")
            self.decision_label.config(text="Chưa có quyết định", fg="#757575")
            self.decision_reason.config(
                text="Vui lòng tính điểm tổng để xem quyết định",
                fg="#9E9E9E"
            )
    
    def _on_score_updated(self, score_type: str, score: float):
        """Callback khi có điểm mới."""
        if score_type == "emotion":
            self.emotion_score.set(score)
        elif score_type == "focus":
            self.focus_score.set(score)
        elif score_type == "clarity":
            self.clarity_score.set(score)
        elif score_type == "content":
            self.content_score.set(score)
    
    def get_frame(self):
        """Lấy frame."""
        return self.frame
