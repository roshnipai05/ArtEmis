import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from tqdm import tqdm


# SETTINGS
IMG_DIR = Path(r"C:\Img10k")
OUT_DIR = Path(r"C:\Img10k_pt")
OUT_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 1
NUM_WORKERS = 4   # Safe for Windows usually, set to 0 if you get Pickling errors

transform = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor(), 
])

# DATASET
class ImageFolderFlat(Dataset):
    def __init__(self, root):
        self.paths = []
        # Recursively search for images in subfolders
        for folder in root.iterdir():
            if folder.is_dir():
                for img in folder.glob("*.*"):
                    if img.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                        self.paths.append(img)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        tensor = transform(img)
        return tensor, str(p)


if __name__ == "__main__":
    print(f"Scanning images in {IMG_DIR}...")
    dataset = ImageFolderFlat(IMG_DIR)
    print(f"Found {len(dataset)} images.")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    print("Starting conversion (Normalization: [0, 1] range)...")
    
    for tensor, path_str in tqdm(loader, desc="Converting to .pt"):
        tensor = tensor.squeeze(0) # Remove batch dim (1, C, H, W) -> (C, H, W)
        jpeg_path = Path(path_str[0])

        # Maintain folder structure (e.g., Img10k_pt/Cubism/image.pt)
        relative = jpeg_path.relative_to(IMG_DIR)
        out_path = OUT_DIR / relative.with_suffix(".pt")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(tensor, out_path)

    print("DONE. All images converted to tensors.")