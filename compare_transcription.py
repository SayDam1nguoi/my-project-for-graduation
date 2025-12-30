"""
So sánh kết quả transcription giữa API và Launcher
"""

import sys
from pathlib import Path
import requests

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

def test_api_transcription(video_path):
    """Test transcription qua API"""
    print("=" * 60)
    print("TEST API TRANSCRIPTION")
    print("=" * 60)
    
    try:
        with open(video_path, 'rb') as f:
            files = {'file': f}
            response = requests.post('http://localhost:8000/api/transcribe-video', files=files)
        
        if response.status_code == 200:
            result = response.json()
            print("✓ API Success")
            print(f"Transcript: {result['transcript'][:200]}...")
            print(f"Language: {result['language']}")
            print(f"Duration: {result['duration']}s")
            print(f"Words: {result['word_count']}")
            return result['transcript']
        else:
            print(f"✗ API Error: {response.status_code}")
            print(response.text)
            return None
            
    except Exception as e:
        print(f"✗ API Error: {e}")
        return None

def test_launcher_transcription(video_path):
    """Test transcription qua Launcher code"""
    print()
    print("=" * 60)
    print("TEST LAUNCHER TRANSCRIPTION")
    print("=" * 60)
    
    try:
        from src.video_analysis.video_transcription_coordinator import VideoTranscriptionCoordinator
        from src.speech_analysis.whisper_stt_engine import WhisperSTTEngine
        from src.speech_analysis.config import WhisperSTTConfig
        
        config = WhisperSTTConfig(model_size='base', language='vi')
        whisper_engine = WhisperSTTEngine(config)
        coordinator = VideoTranscriptionCoordinator(whisper_engine=whisper_engine)
        
        result = coordinator.transcribe_video(video_path)
        
        print("✓ Launcher Success")
        print(f"Transcript: {result.full_text[:200]}...")
        print(f"Language: {result.language}")
        print(f"Duration: {result.duration}s")
        print(f"Words: {len(result.full_text.split())}")
        return result.full_text
        
    except Exception as e:
        print(f"✗ Launcher Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_results(api_transcript, launcher_transcript):
    """So sánh 2 kết quả"""
    print()
    print("=" * 60)
    print("SO SÁNH KẾT QUẢ")
    print("=" * 60)
    
    if api_transcript and launcher_transcript:
        if api_transcript == launcher_transcript:
            print("✓ GIỐNG NHAU 100%")
        else:
            print("✗ KHÁC NHAU")
            print(f"\nAPI length: {len(api_transcript)}")
            print(f"Launcher length: {len(launcher_transcript)}")
            
            # Show first difference
            for i, (a, b) in enumerate(zip(api_transcript, launcher_transcript)):
                if a != b:
                    print(f"\nFirst difference at position {i}:")
                    print(f"API: ...{api_transcript[max(0,i-20):i+20]}...")
                    print(f"Launcher: ...{launcher_transcript[max(0,i-20):i+20]}...")
                    break
    else:
        print("✗ Không thể so sánh (có lỗi)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compare_transcription.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    if not Path(video_path).exists():
        print(f"ERROR: File not found: {video_path}")
        sys.exit(1)
    
    # Test both
    api_result = test_api_transcription(video_path)
    launcher_result = test_launcher_transcription(video_path)
    
    # Compare
    compare_results(api_result, launcher_result)
