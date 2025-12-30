"""
Test Focus API Endpoint

Test script để kiểm tra endpoint /api/analyze-focus
"""

import requests
import sys
from pathlib import Path

API_URL = "http://localhost:8000"

def test_focus_analysis():
    """Test focus analysis endpoint."""
    
    # Find a test video
    test_videos = list(Path("api_uploads").glob("*.mp4"))
    
    if not test_videos:
        print("❌ Không tìm thấy video test trong api_uploads/")
        print("Vui lòng copy một video vào thư mục api_uploads/")
        return
    
    test_video = test_videos[0]
    print(f"📹 Testing with video: {test_video.name}")
    
    # Upload and analyze
    with open(test_video, 'rb') as f:
        files = {'file': (test_video.name, f, 'video/mp4')}
        
        print("🔄 Uploading and analyzing...")
        response = requests.post(f"{API_URL}/api/analyze-focus", files=files)
    
    if response.status_code == 200:
        result = response.json()
        
        print("\n✅ Phân tích thành công!")
        print(f"\n📊 Kết quả:")
        print(f"  - Điểm tập trung: {result['focus_score']}/10")
        print(f"  - Thời gian tập trung: {result['focused_time']}s ({result['focused_rate']}%)")
        print(f"  - Thời gian mất tập trung: {result['distracted_time']}s ({result['distracted_rate']}%)")
        print(f"  - Số lần mất tập trung: {result['distracted_count']} lần")
        print(f"  - Tổng thời gian: {result['duration']}s")
        print(f"  - Số frame phân tích: {result['analyzed_frames']}/{result['total_frames']}")
        
        if result['distraction_events']:
            print(f"\n⚠️ Các sự kiện mất tập trung (top 10):")
            for i, event in enumerate(result['distraction_events'][:10], 1):
                print(f"  {i}. Frame {event['start_frame']}-{event['end_frame']}: {event['duration']:.1f}s")
    else:
        print(f"\n❌ Lỗi: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Testing Focus Analysis API")
    print("=" * 60)
    print()
    
    test_focus_analysis()
