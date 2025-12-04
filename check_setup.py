import sys
import torch

print(f"1. Python is running from: {sys.executable}")
# If this path does NOT end in .venv\Scripts\python.exe, you are still wrong!

print(f"2. PyTorch Version: {torch.__version__}")

print(f"3. CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   - GPU Device: {torch.cuda.get_device_name(0)}")
else:
    print("   - WARNING: You are running on CPU. Training will be extremely slow.")