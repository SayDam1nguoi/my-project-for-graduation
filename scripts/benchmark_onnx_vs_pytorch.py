#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Benchmark ONNX vs PyTorch inference speed.

Usage:
    python scripts/benchmark_onnx_vs_pytorch.py
"""

import torch
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.emotion_classifier import EmotionClassifier
from src.inference.onnx_emotion_classifier import ONNXEmotionClassifier, is_onnx_available


def benchmark_pytorch(model_path: str, num_iterations: int = 100):
    """Benchmark PyTorch inference."""
    print("\n" + "=" * 70)
    print("PYTORCH BENCHMARK")
    print("=" * 70)
    
    # Load model
    print(f"\nLoading PyTorch model: {model_path}")
    classifier = EmotionClassifier(model_path, device='cpu')
    
    # Warm up
    print("Warming up...")
    dummy_input = torch.randn(1, 3, 224, 224)
    for _ in range(10):
        _ = classifier.predict(dummy_input)
    
    # Benchmark
    print(f"Running {num_iterations} iterations...")
    times = []
    
    for i in range(num_iterations):
        start = time.perf_counter()
        _ = classifier.predict(dummy_input)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
        
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{num_iterations}")
    
    # Results
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    fps = 1000 / avg_time
    
    print(f"\nResults:")
    print(f"  Average: {avg_time:.2f}ms")
    print(f"  Min:     {min_time:.2f}ms")
    print(f"  Max:     {max_time:.2f}ms")
    print(f"  FPS:     {fps:.1f}")
    
    return avg_time, times


def benchmark_onnx(model_path: str, num_iterations: int = 100):
    """Benchmark ONNX inference."""
    print("\n" + "=" * 70)
    print("ONNX BENCHMARK")
    print("=" * 70)
    
    if not is_onnx_available():
        print("\n⚠️  ONNX Runtime not available!")
        print("Install with: pip install onnxruntime")
        return None, None
    
    # Load model
    print(f"\nLoading ONNX model: {model_path}")
    classifier = ONNXEmotionClassifier(model_path)
    
    # Warm up
    print("Warming up...")
    dummy_input = torch.randn(1, 3, 224, 224)
    for _ in range(10):
        _ = classifier.predict(dummy_input)
    
    # Benchmark
    print(f"Running {num_iterations} iterations...")
    times = []
    
    for i in range(num_iterations):
        start = time.perf_counter()
        _ = classifier.predict(dummy_input)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
        
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{num_iterations}")
    
    # Results
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    fps = 1000 / avg_time
    
    print(f"\nResults:")
    print(f"  Average: {avg_time:.2f}ms")
    print(f"  Min:     {min_time:.2f}ms")
    print(f"  Max:     {max_time:.2f}ms")
    print(f"  FPS:     {fps:.1f}")
    
    return avg_time, times


def compare_results(pytorch_time, onnx_time):
    """Compare PyTorch and ONNX results."""
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    
    if onnx_time is None:
        print("\n⚠️  ONNX benchmark not available")
        return
    
    speedup = pytorch_time / onnx_time
    improvement = ((pytorch_time - onnx_time) / pytorch_time) * 100
    
    pytorch_fps = 1000 / pytorch_time
    onnx_fps = 1000 / onnx_time
    
    print(f"\nInference Time:")
    print(f"  PyTorch: {pytorch_time:.2f}ms")
    print(f"  ONNX:    {onnx_time:.2f}ms")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Improvement: {improvement:.1f}%")
    
    print(f"\nFPS:")
    print(f"  PyTorch: {pytorch_fps:.1f}")
    print(f"  ONNX:    {onnx_fps:.1f}")
    print(f"  Gain:    +{onnx_fps - pytorch_fps:.1f} FPS")
    
    print(f"\nVideo Processing (with caching):")
    # Assume 75% detection skip, 80% emotion cache
    pytorch_effective = pytorch_time * 0.2  # 20% of frames
    onnx_effective = onnx_time * 0.2
    
    pytorch_video_fps = 1000 / pytorch_effective
    onnx_video_fps = 1000 / onnx_effective
    
    print(f"  PyTorch: {pytorch_video_fps:.1f} FPS")
    print(f"  ONNX:    {onnx_video_fps:.1f} FPS")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    if speedup >= 1.5:
        print(f"  ✅ ONNX is {speedup:.1f}x faster - Highly recommended!")
    elif speedup >= 1.2:
        print(f"  ✅ ONNX is {speedup:.1f}x faster - Recommended")
    elif speedup >= 1.0:
        print(f"  ⚠️  ONNX is slightly faster ({speedup:.1f}x) - Optional")
    else:
        print(f"  ⚠️  PyTorch is faster - Stick with PyTorch")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark ONNX vs PyTorch')
    parser.add_argument(
        '--pytorch-model',
        type=str,
        default='models/efficientnet_b2_best.pth',
        help='Path to PyTorch model'
    )
    parser.add_argument(
        '--onnx-model',
        type=str,
        default='models/efficientnet_b2_best.onnx',
        help='Path to ONNX model'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=100,
        help='Number of iterations'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("ONNX VS PYTORCH BENCHMARK")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  PyTorch model: {args.pytorch_model}")
    print(f"  ONNX model:    {args.onnx_model}")
    print(f"  Iterations:    {args.iterations}")
    
    # Check files exist
    if not Path(args.pytorch_model).exists():
        print(f"\n❌ PyTorch model not found: {args.pytorch_model}")
        sys.exit(1)
    
    if not Path(args.onnx_model).exists():
        print(f"\n⚠️  ONNX model not found: {args.onnx_model}")
        print("Export PyTorch model to ONNX first:")
        print("  python scripts/export_to_onnx.py")
        print("\nRunning PyTorch benchmark only...")
        pytorch_time, _ = benchmark_pytorch(args.pytorch_model, args.iterations)
        sys.exit(0)
    
    # Run benchmarks
    pytorch_time, pytorch_times = benchmark_pytorch(args.pytorch_model, args.iterations)
    onnx_time, onnx_times = benchmark_onnx(args.onnx_model, args.iterations)
    
    # Compare
    compare_results(pytorch_time, onnx_time)
    
    print("\n" + "=" * 70)
    print("✅ BENCHMARK COMPLETED")
    print("=" * 70)
