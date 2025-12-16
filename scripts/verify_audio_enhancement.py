"""
Verification script for audio enhancement integration.

This script demonstrates that audio enhancement is properly integrated
into the WhisperSTTEngine and works as expected.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from unittest.mock import MagicMock, patch


def verify_cpu_enhancement():
    """Verify that enhancement is enabled for CPU device."""
    print("=" * 60)
    print("Verifying Audio Enhancement Integration")
    print("=" * 60)
    
    # Mock whisper and torch
    mock_whisper = MagicMock()
    mock_torch = MagicMock()
    
    with patch.dict('sys.modules', {'whisper': mock_whisper, 'torch': mock_torch}):
        # Setup mocks
        mock_torch.cuda.is_available.return_value = False
        mock_model = MagicMock()
        mock_whisper.load_model.return_value = mock_model
        
        # Import after mocking
        from src.speech_analysis.speech_to_text import WhisperSTTEngine, STTConfig
        
        # Test 1: CPU device should have enhancer
        print("\n1. Testing CPU device configuration...")
        config_cpu = STTConfig(device="cpu", model_name="base")
        engine_cpu = WhisperSTTEngine(config_cpu)
        
        if engine_cpu.audio_enhancer is not None:
            print("   ✓ Audio enhancer initialized for CPU device")
        else:
            print("   ✗ Audio enhancer NOT initialized for CPU device")
            return False
        
        # Test 2: GPU device should NOT have enhancer
        print("\n2. Testing GPU device configuration...")
        mock_torch.cuda.is_available.return_value = True
        mock_model.half.return_value = mock_model
        
        config_gpu = STTConfig(device="auto", model_name="base")
        engine_gpu = WhisperSTTEngine(config_gpu)
        
        if engine_gpu.audio_enhancer is None:
            print("   ✓ Audio enhancer NOT initialized for GPU device (correct)")
        else:
            print("   ✗ Audio enhancer initialized for GPU device (incorrect)")
            return False
        
        # Test 3: Enhancement in transcription pipeline
        print("\n3. Testing enhancement in transcription pipeline...")
        mock_torch.cuda.is_available.return_value = False
        
        # Create fresh engine
        config = STTConfig(device="cpu", model_name="base")
        engine = WhisperSTTEngine(config)
        
        # Mock transcribe result
        mock_model.transcribe.return_value = {
            "text": "test transcription",
            "language": "vi",
            "segments": []
        }
        
        # Mock enhancer
        from unittest.mock import Mock
        mock_enhancer = Mock()
        enhanced_audio = np.random.randn(16000).astype(np.float32)
        mock_enhancer.enhance_audio.return_value = enhanced_audio
        engine.audio_enhancer = mock_enhancer
        
        # Create test audio
        test_audio = np.random.randint(-1000, 1000, 16000, dtype=np.int16)
        
        # Transcribe
        result = engine.transcribe_chunk(test_audio)
        
        if mock_enhancer.enhance_audio.called:
            print("   ✓ Audio enhancement called during transcription")
        else:
            print("   ✗ Audio enhancement NOT called during transcription")
            return False
        
        # Test 4: Error handling
        print("\n4. Testing error handling and fallback...")
        mock_enhancer.enhance_audio.side_effect = RuntimeError("Test error")
        
        # Should not raise error
        try:
            result = engine.transcribe_chunk(test_audio)
            print("   ✓ Graceful fallback on enhancement error")
        except Exception as e:
            print(f"   ✗ Error not handled gracefully: {e}")
            return False
        
        print("\n" + "=" * 60)
        print("All verification tests passed! ✓")
        print("=" * 60)
        return True


if __name__ == "__main__":
    success = verify_cpu_enhancement()
    exit(0 if success else 1)
