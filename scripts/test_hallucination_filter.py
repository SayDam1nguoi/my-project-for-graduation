#!/usr/bin/env python3
"""
Test Hallucination Filter

Script để test xem hallucination filter có hoạt động đúng không.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.speech_analysis.openai_whisper_stt_engine import OpenAIWhisperSTTEngine
from src.speech_analysis.speech_to_text import STTConfig


def test_hallucination_filter():
    """Test hallucination detection."""
    
    # Create a dummy engine just to test the filter
    config = STTConfig()
    
    # We don't need to load the model, just test the filter method
    class DummyEngine:
        def _is_hallucination(self, text: str) -> bool:
            """Copy of the filter method."""
            text_lower = text.lower().strip()
            
            # Exact phrase matches
            exact_hallucinations = [
                'hãy subscribe cho kênh ghiền mì gõ để không bỏ lỡ những video hấp dẫn',
                'hãy subscribe cho kênh ghiền mì gõ',
                'ghiền mì gõ để không bỏ lỡ những video hấp dẫn',
                'để không bỏ lỡ những video hấp dẫn',
                'đăng ký kênh để không bỏ lỡ video mới',
                'nhấn subscribe và bật chuông thông báo',
                'cảm ơn các bạn đã xem video',
                'cảm ơn đã theo dõi',
                'hẹn gặp lại trong video sau',
            ]
            
            for exact in exact_hallucinations:
                if exact in text_lower:
                    return True
            
            # Partial matches
            partial_hallucinations = [
                'subscribe cho kênh',
                'đăng ký kênh',
                'bỏ lỡ những video',
                'video hấp dẫn',
                'ghiền mì gõ',
                'bật chuông thông báo',
            ]
            
            partial_match_count = sum(1 for phrase in partial_hallucinations if phrase in text_lower)
            if partial_match_count >= 2:
                return True
            
            # Keyword checks
            hallucination_keywords = [
                'subscribe',
                'đăng ký kênh',
                'đăng kí kênh',
                'theo dõi kênh',
                'bấm subscribe',
                'nhấn subscribe',
                'channel',
                'like và share',
                'like share',
                'chia sẻ video',
                'bấm chuông',
                'nhấn chuông',
                'bật thông báo',
                'notification',
                'ghiền mì gõ',
                'ghiền mì',
                'cảm ơn đã xem',
                'cảm ơn các bạn đã xem',
                'hẹn gặp lại',
                'see you',
                'bye bye',
                '[âm nhạc]',
                '[music]',
                '[tiếng vỗ tay]',
                '[applause]',
            ]
            
            keyword_count = sum(1 for keyword in hallucination_keywords if keyword in text_lower)
            if keyword_count >= 2:
                return True
            
            strong_indicators = [
                'subscribe',
                'đăng ký kênh',
                'ghiền mì gõ',
                'bấm chuông',
                'nhấn subscribe',
                'like và share',
                'like share',
            ]
            
            for indicator in strong_indicators:
                if indicator in text_lower:
                    return True
            
            if len(text_lower) < 3:
                return True
            
            if all(not c.isalnum() for c in text_lower):
                return True
            
            return False
    
    engine = DummyEngine()
    
    # Test cases
    test_cases = [
        # Hallucinations (should be filtered)
        ("Hãy subscribe cho kênh Ghiền Mì Gõ Để không bỏ lỡ những video hấp dẫn", True),
        ("Hãy subscribe cho kênh Ghiền Mì Gõ", True),
        ("Ghiền Mì Gõ để không bỏ lỡ những video hấp dẫn", True),
        ("Đăng ký kênh để không bỏ lỡ video mới", True),
        ("Nhấn subscribe và bật chuông thông báo", True),
        ("Cảm ơn các bạn đã xem video", True),
        ("Subscribe cho kênh nhé", True),
        ("Đăng ký kênh", True),
        ("Bấm chuông thông báo", True),
        ("Like và share video này", True),
        
        # Valid speech (should NOT be filtered)
        ("Xin chào, tôi tên là Minh", False),
        ("Hôm nay chúng ta sẽ học về Python", False),
        ("Tôi đang làm việc tại công ty ABC", False),
        ("Bạn có thể giúp tôi không?", False),
        ("Cảm ơn bạn đã giúp đỡ", False),
        ("Hẹn gặp lại bạn vào tuần sau", False),
        ("Tôi rất vui được gặp bạn", False),
    ]
    
    print("=" * 80)
    print("HALLUCINATION FILTER TEST")
    print("=" * 80)
    print()
    
    passed = 0
    failed = 0
    
    for text, expected_filtered in test_cases:
        is_filtered = engine._is_hallucination(text)
        status = "✅ PASS" if is_filtered == expected_filtered else "❌ FAIL"
        
        if is_filtered == expected_filtered:
            passed += 1
        else:
            failed += 1
        
        print(f"{status} | Filtered: {is_filtered} | Expected: {expected_filtered}")
        print(f"       Text: {text}")
        print()
    
    print("=" * 80)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print("=" * 80)
    
    if failed == 0:
        print("✅ All tests passed! Hallucination filter is working correctly.")
        return 0
    else:
        print(f"❌ {failed} tests failed. Please review the filter logic.")
        return 1


if __name__ == "__main__":
    sys.exit(test_hallucination_filter())
