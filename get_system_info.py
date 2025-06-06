import platform
import torch
import os

print("===== SYSTEM INFO =====")
print(f"Operating System: {platform.system()} {platform.release()}")
print(f"Processor: {platform.processor()}")
print(f"Python Version: {platform.python_version()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
else:
    print("GPU: Not Available (CPU Only)")

# Optional: Get RAM
try:
    import psutil
    mem = psutil.virtual_memory()
    print(f"RAM: {mem.total / 1e9:.2f} GB")
except ImportError:
    print("Install `psutil` to display memory info.")
