import os
import re
import json
import pickle
from pathlib import Path
from collections import Counter
import pandas as pd

# ===============================================================
# FIXED PATHS FROM USER
# ===============================================================
CSV_PATH = Path(r"C:\Users\91887\Documents\ArtEmis\artemis_dataset_release_v0.csv")
IMG_ROOT = Path(r"C:\Users\91887\Documents\ArtEmis\Img10k")
SAVE_DIR = Path(r"C:\Users\91887\Documents\ArtEmis\text_cnn")

VOCAB_SIZE = 8000          # total vocab size including special tokens
MAX_LEN = 30               # max seq length incl. <start> and <end>
RECOMPUTE = True           # write output files automatically

# special tokens
PAD_IDX = 0
UNK_IDX = 1
START_IDX = 2
END_IDX = 3

# ===============================================================
# 1) Text cleaning
# ===============================================================
def clean_text(text):
    if pd.isna(text):
        return []
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.split()

# ===============================================================
# 2) Load CSV
# ===============================================================
if not CSV_PATH.exists():
    raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")

print("Loading metadata CSV:", CSV_PATH)
df = pd.read_csv(CSV_PATH)

if "art_style" not in df.columns or "painting" not in df.columns or "utterance" not in df.columns:
    raise ValueError(f"CSV file must contain 'art_style', 'painting', and 'utterance' columns.\nColumns found: {df.columns}")

# ===============================================================
# 3) Identify valid image paths inside Img10k (subfolders = art styles)
# ===============================================================
print("Scanning image root:", IMG_ROOT)

valid_images = set()
for art_style_dir in IMG_ROOT.iterdir():
    if not art_style_dir.is_dir():
        continue
    art_style = art_style_dir.name.lower()
    for img_file in art_style_dir.iterdir():
        if img_file.is_file():
            valid_images.add((art_style, img_file.stem.lower()))

print(f"Detected {len(valid_images)} image entries in Img10k.")

# Filter df rows that match existing images
orig_len = len(df)
df = df[df.apply(lambda r: (str(r.art_style).lower(), str(r.painting).lower()) in valid_images, axis=1)]
df = df.reset_index(drop=True)
print(f"Annotations after filtering: {len(df)} (from original {orig_len})")

# ===============================================================
# 4) Tokenize utterances
# ===============================================================
print("Tokenizing...")
df["tokens"] = df["utterance"].apply(clean_text)

all_tokens = [tok for toks in df["tokens"] for tok in toks]
counter = Counter(all_tokens)
print("Total tokens:", len(all_tokens))
print("Unique words:", len(counter))

# ===============================================================
# 5) Build Vocabulary
# ===============================================================
MOST_COMMON_COUNT = VOCAB_SIZE - 4   # minus special tokens

most_common = counter.most_common(MOST_COMMON_COUNT)
print(f"Selecting top {MOST_COMMON_COUNT} words for vocab.")

vocab = {
    "<pad>": PAD_IDX,
    "<unk>": UNK_IDX,
    "<start>": START_IDX,
    "<end>": END_IDX
}

for idx, (word, _) in enumerate(most_common, start=4):
    vocab[word] = idx

rev_vocab = {idx: word for word, idx in vocab.items()}

print("Vocabulary built. Total size:", len(vocab))

# ===============================================================
# 6) Encode (NO padding here)
# ===============================================================
def encode(tokens):
    seq = ["<start>"] + tokens + ["<end>"]
    if len(seq) > MAX_LEN:
        seq = seq[:MAX_LEN]
        seq[-1] = "<end>"
    return [vocab.get(tok, UNK_IDX) for tok in seq]

print("Encoding all captions...")

entries = []  # list of (image_abs_path, token_id_list)
missing_count = 0

for idx, row in df.iterrows():
    art_style = str(row.art_style).lower()
    painting = str(row.painting).lower()

    folder = IMG_ROOT / art_style
    image_path = None

    if folder.exists():
        for f in folder.iterdir():
            if f.is_file() and f.stem.lower() == painting:
                image_path = f.resolve()
                break

    if image_path is None:
        missing_count += 1
        continue

    token_ids = encode(row["tokens"])
    entries.append((str(image_path), token_ids))

print(f"Total encoded entries: {len(entries)} (Missing images: {missing_count})")

# ===============================================================
# 7) Save outputs
# ===============================================================
os.makedirs(SAVE_DIR, exist_ok=True)

vocab_file = SAVE_DIR / "vocab.json"
rev_vocab_file = SAVE_DIR / "rev_vocab.json"
pkl_file = SAVE_DIR / "df_word_encoded.pkl"

if RECOMPUTE:
    with open(vocab_file, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    with open(rev_vocab_file, "w", encoding="utf-8") as f:
        json.dump({int(k): v for k, v in rev_vocab.items()}, f, ensure_ascii=False, indent=2)

    with open(pkl_file, "wb") as f:
        pickle.dump(entries, f)

    print("Saved:")
    print(" -", vocab_file)
    print(" -", rev_vocab_file)
    print(" -", pkl_file)
else:
    print("RECOMPUTE = False, files not saved.")

# ===============================================================
# 8) Sanity checks
# ===============================================================
print("\nSample entries:")
for i in range(min(5, len(entries))):
    print(entries[i])

max_id = max(max(ids) for _, ids in entries)
print("\nMax token ID seen:", max_id)
print("Expected max (VOCAB_SIZE - 1):", VOCAB_SIZE - 1)
print("Done.")
