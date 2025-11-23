import os
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import pandas as pd

TEXT_DIR = "./text"
IMG_ROOT = r"C:\Users\91887\Documents\ArtEmis\Img3k"
SAVE_DIR = "./transformer"
PATCH = 16
EMBED_DIM = 512

os.makedirs(SAVE_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(TEXT_DIR, "captions.csv"))

transform = T.Compose([
    T.Resize((256,256)),
    T.ToTensor()
])


# ------------------------------------------------------------
# 1. Patch embedding layer
# ------------------------------------------------------------
class PatchEmbedding(torch.nn.Module):
    def __init__(self, patch=16, embed_dim=512):
        super().__init__()
        self.patch = patch
        self.proj = torch.nn.Linear(3 * patch * patch, embed_dim)
        self.cls = torch.nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, x):
        B, C, H, W = x.shape
        patches = x.unfold(2, self.patch, self.patch).unfold(3, self.patch, self.patch)
        patches = patches.permute(0,2,3,1,4,5).reshape(B, -1, 3*self.patch*self.patch)
        embed = self.proj(patches)
        cls_token = self.cls.expand(B, -1, -1)
        return torch.cat([cls_token, embed], dim=1)

patcher = PatchEmbedding(PATCH, EMBED_DIM)


# ------------------------------------------------------------
# 2. Convert all images → patch embeddings
# ------------------------------------------------------------
image_tokens = []
for idx, row in df.iterrows():
    img_path = os.path.join(IMG_ROOT, row.art_style, row.painting + ".jpg")
    if not os.path.exists(img_path):
        continue
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0)

    with torch.no_grad():
        tokens = patcher(x).squeeze(0).numpy()

    image_tokens.append(tokens)

np.save(os.path.join(SAVE_DIR, "image_tokens.npy"), image_tokens)
df.to_csv(os.path.join(SAVE_DIR, "metadata.csv"), index=False)

print("Saved transformer image tokens to:", SAVE_DIR)
