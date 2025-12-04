import os
import re
import json
import pickle
from pathlib import Path
from collections import Counter
import pandas as pd

# ===============================================================
# FIXED PATHS
# ===============================================================
CSV_PATH = Path(r"C:\Users\91887\Documents\ArtEmis\artemis_dataset_release_v0.csv")
IMG_ROOT = Path(r"C:\Img10k")
SAVE_DIR = Path(r"C:\Users\91887\Documents\ArtEmis\text_cnn")

# REDUCED VOCAB SIZE AS DISCUSSED
VOCAB_SIZE = 3500
MAX_LEN = 30
RECOMPUTE = True

# Token IDs
PAD_IDX = 0
UNK_IDX = 1
START_IDX = 2
END_IDX = 3

# ===============================================================
# 1) Clean text
# ===============================================================
def clean_text(text):
    if pd.isna(text):
        return []
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.split()

# ===============================================================
# 2) Load Artemis CSV
# ===============================================================
print("\nLoading CSV...")
df = pd.read_csv(CSV_PATH)

required = {"art_style", "painting", "utterance"}
if not required.issubset(df.columns):
    raise ValueError(f"CSV missing required columns: {required}")

# ===============================================================
# 3) Scan Img10k and build FAST lookup
# ===============================================================
print("\nIndexing image folders...")
image_lookup = {} 

for art_style_dir in IMG_ROOT.iterdir():
    if not art_style_dir.is_dir():
        continue
    style_name = art_style_dir.name.lower()
    for img_file in art_style_dir.iterdir():
        if img_file.is_file():
            image_lookup[(style_name, img_file.stem.lower())] = str(img_file.resolve())

print(f"Indexed {len(image_lookup)} images.")

# ===============================================================
# 4) Filter DataFrame
# ===============================================================
print("\nFiltering annotations...")
df["art_key"] = df["art_style"].astype(str).str.lower()
df["paint_key"] = df["painting"].astype(str).str.lower()
df["pair"] = list(zip(df["art_key"], df["paint_key"]))

df = df[df["pair"].isin(image_lookup.keys())]
df = df.reset_index(drop=True)
print(f"Remaining annotations: {len(df)}")

# ===============================================================
# 5) Tokenize
# ===============================================================
print("\nTokenizing text...")
df["tokens"] = df["utterance"].apply(clean_text)

all_tokens = [tok for toks in df["tokens"] for tok in toks]
counter = Counter(all_tokens)

print(f"Total tokens: {len(all_tokens)}")
print(f"Unique words: {len(counter)}")

# ===============================================================
# 6) Build Vocabulary
# ===============================================================
print("\nBuilding vocabulary...")

# Reserve spots for specials
MOST_COMMON_COUNT = VOCAB_SIZE - 4
most_common = counter.most_common(MOST_COMMON_COUNT)

vocab = {
    "<pad>": PAD_IDX,
    "<unk>": UNK_IDX,
    "<start>": START_IDX,
    "<end>": END_IDX
}

# Add common words
for idx, (word, _) in enumerate(most_common, start=4):
    vocab[word] = idx

# Reverse vocab (ID -> Word)
rev_vocab = {idx: word for word, idx in vocab.items()}

print("Vocabulary size:", len(vocab))

# ===============================================================
# 7) Encode Captions
# ===============================================================
print("\nEncoding captions...")

def encode(tokens):
    seq = ["<start>"] + tokens + ["<end>"]
    if len(seq) > MAX_LEN:
        seq = seq[:MAX_LEN]
        seq[-1] = "<end>"
    return [vocab.get(tok, UNK_IDX) for tok in seq]

entries = []
for idx, row in df.iterrows():
    key = row["pair"]
    if key not in image_lookup:
        continue 
    image_path = image_lookup[key]
    token_ids = encode(row["tokens"])
    entries.append((image_path, token_ids))

print(f"Final encoded pairs: {len(entries)}")

# ===============================================================
# 8) SAFETY CHECK (Coverage Analysis)
# ===============================================================
total_tokens_count = sum(counter.values())
kept_tokens_count = 0

for word, count in counter.items():
    if word in vocab:
        kept_tokens_count += count

unk_count = total_tokens_count - kept_tokens_count
unk_percentage = (unk_count / total_tokens_count) * 100

print(f"\n=== SAFETY CHECK ===")
print(f"Total Tokens in dataset: {total_tokens_count}")
print(f"Tokens converted to <unk>: {unk_count}")
print(f"Percentage of <unk>: {unk_percentage:.2f}%")

if unk_percentage > 5.0:
    print("WARNING: <unk> is becoming too frequent (>5%). Consider increasing VOCAB_SIZE.")
else:
    print("SAFE: <unk> is a small fraction of the data.")

# ===============================================================
# 9) Save Outputs
# ===============================================================
os.makedirs(SAVE_DIR, exist_ok=True)

pkl_file = SAVE_DIR / "df_word_encoded.pkl"
vocab_file = SAVE_DIR / "vocab.json"
rev_vocab_file = SAVE_DIR / "rev_vocab.json"

if RECOMPUTE:
    # 1. Save Dataset
    with open(pkl_file, "wb") as f:
        pickle.dump(entries, f)
    print(f"\nSaved dataset: {pkl_file}")

    # 2. Save Vocab (Word -> ID)
    with open(vocab_file, "w") as f:
        json.dump(vocab, f)
    print(f"Saved vocab: {vocab_file}")

    # 3. Save Reverse Vocab (ID -> Word)
    # Json keys must be strings, so we convert int keys to string
    # When loading back, remember to convert keys back to int!
    with open(rev_vocab_file, "w") as f:
        json.dump(rev_vocab, f)
    print(f"Saved reverse vocab: {rev_vocab_file}")

else:
    print("\nRECOMPUTE = False → Did not save output.")

print("\nDone.")