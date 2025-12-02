#!/usr/bin/env python3
"""
create_artemis_subset.py

Creates a stratified sample of images (equal across artforms) from a WikiArt folder,
keeping only images that appear in the ArtEmis annotations table.

Selected images are resized to 256×256 and written to:
    ~/Documents/ArtEmis/Img3k/<ArtForm>/

Assumptions:
- Root wikiart folder contains subfolders named by artform (e.g. "Impressionism", "Cubism").
- Annotation file is located in:
      ~/Documents/ArtEmis/artemis_dataset_release_v0.csv  OR  .xlsx
- Annotation file contains a column named "painting" with image filenames.
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


# ----------------------------------------------------------------------
#                         USER CONFIGURATION
# ----------------------------------------------------------------------

WIKIART_ROOT = Path.home() / "Documents" / "Downloads" / "wikiart" / "wikiart"
ANNOTATION_FILE_CSV = Path.home() / "Documents" / "ArtEmis" / "artemis_dataset_release_v0.csv"
DEST_ROOT = Path.home() / "Documents" / "ArtEmis" / "Img8k"

TARGET_TOTAL = 8000
IMAGE_SIZE = (128, 128)  # width, height
RANDOM_SEED = 42
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

random.seed(RANDOM_SEED)


# ----------------------------------------------------------------------
#                              UTILITIES
# ----------------------------------------------------------------------

def load_annotation_paintings(annotation_path: Path):
    """Load annotated image names (stem only, lowercase)."""
    if annotation_path.suffix.lower() in [".csv", ".txt"]:
        df = pd.read_csv(annotation_path)
    else:
        df = pd.read_excel(annotation_path)

    if "painting" not in df.columns:
        raise KeyError("Expected 'painting' column in annotation file.")

    return {Path(x).stem.lower() for x in df["painting"].astype(str)}


def collect_images_by_artform(wikiart_root: Path, allowed_stems: set):
    """Scan wikiart directories and collect images whose stems match annotation list."""
    artform_to_images = {}

    for artform_dir in sorted(wikiart_root.iterdir()):
        if not artform_dir.is_dir():
            continue

        images = [
            p for p in artform_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTS and p.stem.lower() in allowed_stems
        ]

        if images:
            artform_to_images[artform_dir.name] = images
            print(f"Extracting images from: {artform_dir}")
            print(f"Number of images extracted: {len(images)}\n")

    return artform_to_images


def compute_per_class_quota(artform_counts: dict, total: int):
    """Compute how many images to sample from each artform."""
    n = len(artform_counts)
    if n == 0:
        raise ValueError("No artform classes found.")

    base = total // n
    quota = {k: min(base, v) for k, v in artform_counts.items()}

    # distribute leftover based on remaining capacity
    remaining = total - sum(quota.values())
    if remaining > 0:
        remaining_caps = sorted(
            [(k, artform_counts[k] - quota[k]) for k in artform_counts],
            key=lambda x: -x[1]
        )
        while remaining > 0:
            for k, cap in remaining_caps:
                if remaining == 0:
                    break
                if cap > 0:
                    quota[k] += 1
                    remaining -= 1
            remaining_caps = [(k, artform_counts[k] - quota[k]) for k, _ in remaining_caps]

    return quota


def safe_open_image(p: Path):
    """Open and validate an image safely."""
    try:
        with Image.open(p) as img:
            img.verify()
        img = Image.open(p).convert("RGB")
        return img
    except (UnidentifiedImageError, OSError, ValueError):
        return None


def prepare_destination(dest_root: Path):
    dest_root.mkdir(parents=True, exist_ok=True)


def process_and_save(selected_paths, dest_dir: Path, size):
    dest_dir.mkdir(parents=True, exist_ok=True)
    skipped = []
    saved = 0

    for src in tqdm(selected_paths, desc=f"Processing → {dest_dir.name}", unit="img"):
        img = safe_open_image(src)
        if img is None:
            skipped.append(str(src))
            continue

        try:
            img = img.resize(size, resample=Image.LANCZOS)
            img.save(dest_dir / src.name, format="JPEG", quality=92)
            saved += 1
        except Exception:
            skipped.append(str(src))

    return saved, skipped


# ----------------------------------------------------------------------
#                                 MAIN
# ----------------------------------------------------------------------

def main():
    print("Loading annotation painting list...")
    allowed_paintings = load_annotation_paintings(ANNOTATION_FILE_CSV)
    print(f"Annotation contains {len(allowed_paintings)} painting filenames.")
    print(list(allowed_paintings)[:20])

    print(f"\nScanning WikiArt folders at {WIKIART_ROOT} ...")
    artform_to_images = collect_images_by_artform(WIKIART_ROOT, allowed_paintings)

    if not artform_to_images:
        print("No images found matching annotation entries. Exiting.")
        sys.exit(1)

    counts = {k: len(v) for k, v in artform_to_images.items()}
    print("\nFound artforms and counts (after matching annotations):")
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
        overall_selected.extend([
            str(dest_dir / p.name) for p in chosen if (dest_dir / p.name).exists()
        ])
        all_skipped.extend(skipped)

        print(f"{artform}: requested {q}, saved {saved}, skipped {len(skipped)}")

    print(f"\nDone. Total selected saved: {len(overall_selected)}")
    if all_skipped:
        print(f"Skipped {len(all_skipped)} corrupted/unwritable images. Example: {all_skipped[:5]}")
    print(f"Destination: {DEST_ROOT}")


if __name__ == "__main__":
    main()
