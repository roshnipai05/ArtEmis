import os
import torch
import pickle
from pathlib import Path
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# CONFIGURATION 
PICKLE_PATH = r"text_cnn/df_word_encoded.pkl" 

# Path to the ORIGINAL 80k dataset (High Res JPEGs)
# changing this to where the full ArtEmis/WikiArt dataset lives
ORIGINAL_SOURCE_ROOT = r"C:\Users\91887\Documents\Downloads\wikiart\wikiart"

DEST_ROOT = r"C:\Img10k_pt_256"

# Target resolution for the new tensors
TARGET_SIZE = (256, 256)

# Transform pipeline
transform = transforms.Compose([
    transforms.Resize(TARGET_SIZE),
    transforms.ToTensor(),
])

def extract_relative_path(old_abs_path):
    """
    Extracts 'Style/Image.jpg' from the old absolute path.
    Assumes structure ends in .../Style/Image.jpg
    """
    parts = Path(old_abs_path).parts
    if len(parts) >= 2:
        return Path(parts[-2]) / parts[-1]
    return Path(parts[-1])

def regenerate_tensors():
    print(f"Loading master list from {PICKLE_PATH}...")
    
    if not os.path.exists(PICKLE_PATH):
        print("Pickle file not found!")
        return

    with open(PICKLE_PATH, "rb") as f:
        all_items = pickle.load(f)
    
    print(f"Found {len(all_items)} items in the dataset list.")
    print(f"Reading High-Res images from: {ORIGINAL_SOURCE_ROOT}")
    print(f"Saving 256x256 tensors to: {DEST_ROOT}")

    os.makedirs(DEST_ROOT, exist_ok=True)
    
    success_count = 0
    missing_count = 0
    error_count = 0

    for item in tqdm(all_items):
        # item is (original_path, caption_tokens)
        old_path = item[0]
        
        # 1. Determine Relative Path (e.g., "Fauvism/joan-miro_the-farmer.jpg")
        rel_path = extract_relative_path(old_path)
        
        # 2. Construct Current High-Res Source Path
        source_path = Path(ORIGINAL_SOURCE_ROOT) / rel_path
        
        # 3. Construct Destination Path
        dest_path = (Path(DEST_ROOT) / rel_path).with_suffix(".pt")
        
        # Ensure subfolder exists
        os.makedirs(dest_path.parent, exist_ok=True)
        
        # 4. Process
        if source_path.exists():
            try:
                # Open High-Res Image
                img = Image.open(source_path).convert("RGB")
                
                # Transform to 256x256 Tensor
                tensor = transform(img)
                
                # Save
                torch.save(tensor, dest_path)
                success_count += 1
                
            except Exception as e:
                print(f"\nError processing {rel_path}: {e}")
                error_count += 1
        else:
            missing_count += 1

    print("\n=== Processing Complete ===")
    print(f"Successfully generated: {success_count}")
    print(f"Missing source images:  {missing_count}")
    print(f"Errors:                 {error_count}")
    
    if missing_count > 0:
        print("\n[WARN] Some images were not found in the Original Source Root.")
        print("Check that 'ORIGINAL_SOURCE_ROOT' points to the folder containing style subfolders (e.g., /WikiArt/Fauvism/)")

if __name__ == "__main__":
    regenerate_tensors()