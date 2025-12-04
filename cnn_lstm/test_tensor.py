import torch
from pathlib import Path

pt_file = next(Path("C:/Img10k_pt/Pop_Art").glob("*.pt"))
img = torch.load(pt_file)

print("Min:", img.min().item())
print("Max:", img.max().item())
print("Mean:", img.mean().item())
print("Std:", img.std().item())
