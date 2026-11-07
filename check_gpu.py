import sys
import torch
import torchvision
print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA not detected; check driver, your torch wheel, and discrete GPU availability")
print("GPU:", torch.cuda.get_device_name(0))
print("VRAM (GiB):", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
print("GPU compute capability:", torch.cuda.get_device_capability(0))
# Real computation confirms kernels are included for the GPU architecture.
a = torch.rand((256,256), device="cuda")
b = torch.mm(a, a)
torch.cuda.synchronize()
print("GPU matrix multiplication OK:", tuple(b.shape))
