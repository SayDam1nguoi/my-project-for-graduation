"""
Demo Score Summary UI

Demo ứng dụng tổng hợp điểm phỏng vấn.
"""

import tkinter as tk
from tkinter import ttk
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from apps.gui.score_summary_tab import ScoreSummaryTab


class DemoApp:
    """Demo application."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Demo - Tổng Hợp Điểm Phỏng Vấn")
        self.root.geometry("1200x800")
        
        # Create notebook
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create score summary tab
        self.score_tab = ScoreSummaryTab(self.notebook)
        self.notebook.add(self.score_tab.get_frame(), text="📊 Tổng Hợp Điểm")
        
        # Create demo controls
        self._create_demo_controls()
    
    def _create_demo_controls(self):
        """Tạo controls để demo."""
        demo_frame = ttk.LabelFrame(self.root, text="Demo Controls", padding=10)
        demo_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Simulate video analysis
        video_frame = ttk.Frame(demo_frame)
        video_frame.pack(side=tk.LEFT, padx=10)
        
        ttk.Label(video_frame, text="Giả lập Video Analysis:").pack()
        ttk.Button(
            video_frame,
            text="📹 Cập nhật điểm Video (Cảm xúc + Tập trung)",
            command=self._simulate_video_analysis
        ).pack(pady=5)
        
        # Simulate speech analysis
        speech_frame = ttk.Frame(demo_frame)
        speech_frame.pack(side=tk.LEFT, padx=10)
        
        ttk.Label(speech_frame, text="Giả lập Speech Analysis:").pack()
        ttk.Button(
            speech_frame,
            text="🎤 Cập nhật điểm Speech (Rõ ràng + Nội dung)",
            command=self._simulate_speech_analysis
        ).pack(pady=5)
        
        # Simulate all
        all_frame = ttk.Frame(demo_frame)
        all_frame.pack(side=tk.LEFT, padx=10)
        
        ttk.Label(all_frame, text="Giả lập Tất cả:").pack()
        ttk.Button(
            all_frame,
            text="🎯 Cập nhật TẤT CẢ điểm",
            command=self._simulate_all
        ).pack(pady=5)
    
    def _simulate_video_analysis(self):
        """Giả lập video analysis."""
        import random
        
        emotion_score = round(random.uniform(6.0, 9.5), 1)
        focus_score = round(random.uniform(6.5, 9.0), 1)
        
        self.score_tab.update_video_scores(emotion_score, focus_score)
        
        print(f"✅ Đã cập nhật điểm Video:")
        print(f"  Cảm xúc: {emotion_score}/10")
        print(f"  Tập trung: {focus_score}/10")
    
    def _simulate_speech_analysis(self):
        """Giả lập speech analysis."""
        import random
        
        clarity_score = round(random.uniform(6.0, 9.0), 1)
        content_score = round(random.uniform(5.5, 9.5), 1)
        
        self.score_tab.update_speech_scores(clarity_score, content_score)
        
        print(f"✅ Đã cập nhật điểm Speech:")
        print(f"  Rõ ràng: {clarity_score}/10")
        print(f"  Nội dung: {content_score}/10")
    
    def _simulate_all(self):
        """Giả lập tất cả."""
        self._simulate_video_analysis()
        self._simulate_speech_analysis()


def main():
    """Main function."""
    root = tk.Tk()
    app = DemoApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
