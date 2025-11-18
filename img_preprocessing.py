#!/usr/bin/env python3
"""
create_artemis_subset.py

Creates a stratified sample of images (equal across artforms) from a wikiart folder,
but only keeps images that appear in the artemis annotations table.
Resizes images to 256x256 and writes them to Documents/ArtEmis/Img3k/<ArtForm>/.

Assumptions:
- Root wikiart folder contains subfolders named by artform (e.g. 'Impressionism', 'Cubism', ...)
- Annotation file is in Documents/ArtEmis/artemis_dataset_release_v0.csv or .xlsx
- Annotation file has a column named 'paintings' listing image filenames (e.g. 'abc.jpg')
"""

import os
import sys
import random
import shutil
from pathlib import Path
from collections import defaultdict
import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

# ---------- USER CONFIG ----------
WIKIART_ROOT = Path.home() / "Documents" /"Downloads"/ "wikiart"/"wikiart" #adjust if yours is elsewhere
ANNOTATION_FILE_CSV = Path.home() / "Documents" / "ArtEmis" / "artemis_dataset_release_v0.csv"
DEST_ROOT = Path.home() / "Documents" / "ArtEmis" / "Img3k"
TARGET_TOTAL = 3000
IMAGE_SIZE = (256, 256)  # (width, height)
RANDOM_SEED = 42
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
# ---------------------------------

random.seed(RANDOM_SEED)

# def load_annotation_paintings(annotation_path: Path):
#     if not annotation_path.exists():
#         raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
#     if annotation_path.suffix.lower() in [".csv", ".txt"]:
#         df = pd.read_csv(annotation_path)
#     elif annotation_path.suffix.lower() in [".xlsx", ".xls"]:
#         df = pd.read_excel(annotation_path)
#     else:
#         raise ValueError("Unsupported annotation file type: " + str(annotation_path.suffix))
#     if "painting" not in df.columns:
#         raise KeyError("Expected 'painting' column in annotation file.")
#     painting_names = set(df["painting"].astype(str).str.strip().apply(lambda x: Path(x).name))
#     return painting_names

def load_annotation_paintings(annotation_path: Path):
    if annotation_path.suffix.lower() in [".csv", ".txt"]:
        df = pd.read_csv(annotation_path)
    else:
        df = pd.read_excel(annotation_path)
    if "painting" not in df.columns:
        raise KeyError("Expected 'painting' column in annotation file.")

    # normalize to lowercase stems
    painting_names = {Path(x).stem.lower() for x in df["painting"].astype(str)}
    return painting_names


def collect_images_by_artform(wikiart_root: Path, allowed_paintings: set):
    artform_to_images = {}
    for artform_dir in sorted(wikiart_root.iterdir()):
            
        if not artform_dir.is_dir():
            continue
        images = []
        for p in artform_dir.iterdir():
            if p.suffix.lower() in IMAGE_EXTS:
                if p.stem.lower() in allowed_paintings:
                    images.append(p)
        if images:
            artform_to_images[artform_dir.name] = images
        
        print("Extracting images from: ", artform_dir)
        print("Number of images extracted: ", len(images),"\n")
    return artform_to_images

def compute_per_class_quota(artform_counts: dict, total: int):
    n_classes = len(artform_counts)
    if n_classes == 0:
        raise ValueError("No artform classes found.")
    base = total // n_classes
    quota = {k: min(base, v) for k, v in artform_counts.items()}
    assigned = sum(quota.values())
    remaining = total - assigned
    # distribute remaining proportionally by available items
    if remaining > 0:
        # sort by remaining capacity (largest available first)
        remaining_caps = sorted([(k, artform_counts[k] - quota[k]) for k in artform_counts], key=lambda x: -x[1])
        idx = 0
        while remaining > 0:
            for k, cap in remaining_caps:
                if remaining == 0:
                    break
                if cap > 0:
                    quota[k] += 1
                    remaining -= 1
                    # update cap
                    remaining_caps = [(kk, artform_counts[kk] - quota[kk]) for kk, _ in remaining_caps]
    return quota

def safe_open_image(p: Path):
    try:
        with Image.open(p) as img:
            img.verify()  # verify integrity
        # re-open for actual processing (verify() leaves file in an unusable state sometimes)
        img = Image.open(p)
        img = img.convert("RGB")
        return img
    except (UnidentifiedImageError, OSError, ValueError) as e:
        return None

def prepare_destination(dest_root: Path):
    dest_root.mkdir(parents=True, exist_ok=True)

def process_and_save(selected_paths, dest_dir: Path, size):
    dest_dir.mkdir(parents=True, exist_ok=True)
    skipped = []
    saved = 0
    for src in tqdm(selected_paths, desc=f"Processing -> {dest_dir.name}", unit="img"):
        img = safe_open_image(src)
        if img is None:
            skipped.append(str(src))
            continue
        try:
            img = img.resize(size, resample=Image.LANCZOS)
            out_path = dest_dir / src.name
            img.save(out_path, format="JPEG", quality=92)  # JPEG high quality
            saved += 1
        except Exception as e:
            skipped.append(str(src))
    return saved, skipped

def main():
    print("Loading annotation painting list...")
    allowed_paintings = load_annotation_paintings(ANNOTATION_FILE_CSV)
    print(f"Annotation contains {len(allowed_paintings)} painting filenames.")
    print(list(allowed_paintings)[:20])
    print(f"Scanning wikiart folders at {WIKIART_ROOT} ...")
    artform_to_images = collect_images_by_artform(WIKIART_ROOT, allowed_paintings)
    if not artform_to_images:
        print("No images found matching annotation 'painting' entries. Exiting.")
        sys.exit(1)

    counts = {k: len(v) for k, v in artform_to_images.items()}
    print("Found artforms and counts (after matching annotations):")
    for k, c in counts.items():
        print(f"  {k}: {c}")

    quota = compute_per_class_quota(counts, TARGET_TOTAL)
    print("\nSampling quota per artform:")
    for k, q in quota.items():
        print(f"  {k}: {q}")

    prepare_destination(DEST_ROOT)
    overall_selected = []
    all_skipped = []
    for artform, images in artform_to_images.items():
        q = quota.get(artform, 0)
        if q <= 0:
            continue
        chosen = random.sample(images, min(q, len(images)))
        dest_dir = DEST_ROOT / artform
        saved, skipped = process_and_save(chosen, dest_dir, IMAGE_SIZE)
        overall_selected.extend([str((dest_dir / p.name)) for p in chosen if (dest_dir / p.name).exists()])
        all_skipped.extend(skipped)
        print(f"{artform}: requested {q}, saved {saved}, skipped {len(skipped)}")

    print(f"\nDone. Total selected saved: {len(overall_selected)}")
    if all_skipped:
        print(f"Skipped {len(all_skipped)} corrupted/unwritable images. Example: {all_skipped[:5]}")
    print(f"Destination: {DEST_ROOT}")

if __name__ == "__main__":
    main()
