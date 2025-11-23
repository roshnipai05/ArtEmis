import os
import torch
import torchvision.transforms as T
from torchvision.models import resnet18
from PIL import Image
import pandas as pd
import numpy as np

TEXT_DIR = "./text"
IMG_ROOT = r"C:\Users\91887\Documents\ArtEmis\Img3k"
SAVE_DIR = "./cnn_lstm"
os.makedirs(SAVE_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(TEXT_DIR, "captions.csv"))

# ------------------------------------------------------------
# 1. Image preprocessing
# ------------------------------------------------------------
transform = T.Compose([
    T.Resize((256,256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

# ------------------------------------------------------------
# 2. CNN backbone (no pretraining — assignment requirement)
# ------------------------------------------------------------
model = resnet18(weights=None)
model.fc = torch.nn.Identity()  # output 512D
model.eval()


# ------------------------------------------------------------
# 3. Extract features
# ------------------------------------------------------------
def load_image(path):
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0)

features = []
paths = []

for idx, row in df.iterrows():
    img_path = os.path.join(IMG_ROOT, row.art_style, row.painting + ".jpg")
    if not os.path.exists(img_path): 
        continue
    x = load_image(img_path)
    with torch.no_grad():
        feat = model(x).squeeze().numpy()
    features.append(feat)
    paths.append(img_path)

features = np.stack(features)
np.save(os.path.join(SAVE_DIR, "cnn_features.npy"), features)

df.to_csv(os.path.join(SAVE_DIR, "metadata.csv"), index=False)

print("Saved CNN features to:", SAVE_DIR)
