#!/usr/bin/env python3
"""Quick GPU check script"""
import torch

print("=" * 50)
print("🔍 GPU Check")
print("=" * 50)

cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

if cuda_available:
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    print("\n✅ GPU is ready for training!")
else:
    print("\n❌ GPU not detected!")
    print("PyTorch Version:", torch.__version__)
    print("\nTo fix, reinstall PyTorch with CUDA:")
    print("pip uninstall torch torchvision torchaudio")
    print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")

print("=" * 50)
