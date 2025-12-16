#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Export PyTorch model to ONNX format.

Usage:
    python scripts/export_to_onnx.py
"""

import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.model_loader import ModelLoader


def export_to_onnx(
    pytorch_model_path: str,
    onnx_model_path: str = None,
    opset_version: int = 12,
    dynamic_batch: bool = True
):
    """
    Export PyTorch model to ONNX format.
    
    Args:
        pytorch_model_path: Path to PyTorch .pth file
        onnx_model_path: Path to save ONNX file (auto-generated if None)
        opset_version: ONNX opset version (12 recommended)
        dynamic_batch: Support dynamic batch size
    """
    print("=" * 70)
    print("EXPORT PYTORCH MODEL TO ONNX")
    print("=" * 70)
    
    # Auto-generate ONNX path
    if onnx_model_path is None:
        onnx_model_path = pytorch_model_path.replace('.pth', '.onnx')
    
    print(f"\nInput:  {pytorch_model_path}")
    print(f"Output: {onnx_model_path}")
    
    # Load PyTorch model
    print("\n1. Loading PyTorch model...")
    model_loader = ModelLoader(device='cpu')
    model = model_loader.load_model(pytorch_model_path)
    model.eval()
    print("✓ Model loaded successfully")
    
    # Create dummy input
    print("\n2. Creating dummy input...")
    dummy_input = torch.randn(1, 3, 224, 224)
    print(f"✓ Dummy input shape: {dummy_input.shape}")
    
    # Export to ONNX
    print("\n3. Exporting to ONNX...")
    
    # Dynamic axes for batch size
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
        verbose=False
    )
    print("✓ Export completed")
    
    # Verify ONNX model
    print("\n4. Verifying ONNX model...")
    try:
        import onnx
        onnx_model = onnx.load(onnx_model_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model is valid")
        
        # Print model info
        print(f"\n5. Model Information:")
        print(f"   Opset version: {onnx_model.opset_import[0].version}")
        print(f"   Inputs: {[inp.name for inp in onnx_model.graph.input]}")
        print(f"   Outputs: {[out.name for out in onnx_model.graph.output]}")
        
    except ImportError:
        print("⚠️  ONNX package not installed, skipping verification")
        print("   Install with: pip install onnx")
    except Exception as e:
        print(f"⚠️  Verification failed: {e}")
    
    # Test inference
    print("\n6. Testing ONNX inference...")
    try:
        import onnxruntime as ort
        
        session = ort.InferenceSession(onnx_model_path)
        
        # Test with dummy input
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        dummy_input_np = dummy_input.numpy()
        outputs = session.run([output_name], {input_name: dummy_input_np})
        
        print(f"✓ ONNX inference successful")
        print(f"   Output shape: {outputs[0].shape}")
        
        # Compare with PyTorch
        with torch.no_grad():
            pytorch_output = model(dummy_input).numpy()
        
        diff = abs(pytorch_output - outputs[0]).max()
        print(f"   Max difference: {diff:.6f}")
        
        if diff < 1e-4:
            print("✓ ONNX output matches PyTorch (excellent)")
        elif diff < 1e-3:
            print("✓ ONNX output matches PyTorch (good)")
        else:
            print(f"⚠️  Large difference detected: {diff}")
        
    except ImportError:
        print("⚠️  ONNX Runtime not installed, skipping inference test")
        print("   Install with: pip install onnxruntime")
    except Exception as e:
        print(f"⚠️  Inference test failed: {e}")
    
    # File size comparison
    print("\n7. File Size Comparison:")
    pytorch_size = Path(pytorch_model_path).stat().st_size / (1024 * 1024)
    onnx_size = Path(onnx_model_path).stat().st_size / (1024 * 1024)
    print(f"   PyTorch: {pytorch_size:.2f} MB")
    print(f"   ONNX:    {onnx_size:.2f} MB")
    print(f"   Ratio:   {onnx_size/pytorch_size:.2f}x")
    
    print("\n" + "=" * 70)
    print("✅ EXPORT COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\nONNX model saved to: {onnx_model_path}")
    print("\nNext steps:")
    print("1. Test inference: python scripts/test_onnx_inference.py")
    print("2. Benchmark: python scripts/benchmark_onnx_vs_pytorch.py")
    print("3. Run app: python launcher.py (will auto-use ONNX)")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Export PyTorch model to ONNX')
    parser.add_argument(
        '--input',
        type=str,
        default='models/efficientnet_b2_best.pth',
        help='Path to PyTorch model (.pth)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save ONNX model (.onnx)'
    )
    parser.add_argument(
        '--opset',
        type=int,
        default=12,
        help='ONNX opset version'
    )
    parser.add_argument(
        '--no-dynamic-batch',
        action='store_true',
        help='Disable dynamic batch size'
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        print("\nPlease train a model first:")
        print("  python train.py --model efficientnet_b2 --epochs 50")
        sys.exit(1)
    
    # Export
    export_to_onnx(
        pytorch_model_path=args.input,
        onnx_model_path=args.output,
        opset_version=args.opset,
        dynamic_batch=not args.no_dynamic_batch
    )
