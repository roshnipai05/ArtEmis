import os
import re
import json
import pandas as pd
from collections import Counter
from pathlib import Path
import argparse

recompute = False

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="./", help="Folder containing Img3k dataset and Artemis original CSV")

args = parser.parse_args()

DATA_DIR = Path(args.data_dir)
CSV_PATH = DATA_DIR / "artemis_dataset_release_v0.csv"
IMG_ROOT = DATA_DIR / "Img15k"
VOCAB_SIZE = 12000
MAX_LEN = 30
SAVE_DIR = "./text_15k"
OLD_VOCAB_PATH = "./text_8k/vocab.json"

os.makedirs(SAVE_DIR, exist_ok=True)


# ------------------------------------------------------------
# 1. Basic text cleaning
# ------------------------------------------------------------
def clean_text(text):
    text = text.lower().strip()
    # 1. Replace all non-alphanumeric/non-space characters with a SINGLE SPACE.
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    
    # 2. OPTIONAL: Compress multiple spaces into one space (for robustness)
    text = re.sub(r"\s+", " ", text)
    
    # 3. Split by space
    return text.split()


# ------------------------------------------------------------
# 2. Filter to only images in Img3k
# ------------------------------------------------------------
df = pd.read_csv(CSV_PATH)

valid_images = set()
for art_style in os.listdir(IMG_ROOT):
    folder = os.path.join(IMG_ROOT, art_style)
    if not os.path.isdir(folder):
        continue
    for img in os.listdir(folder):
        name, _ = os.path.splitext(img)
        valid_images.add((art_style.lower(), name.lower()))

df = df[
    df.apply(lambda r: (str(r.art_style).lower(), str(r.painting).lower()) in valid_images, axis=1)
].reset_index(drop=True)

print("Filtered annotations:", len(df))


# ------------------------------------------------------------
# 3. Tokenization
# ------------------------------------------------------------
df["tokens"] = df["utterance"].astype(str).apply(clean_text)

all_tokens = [t for toks in df["tokens"] for t in toks]
counter = Counter(all_tokens)
most_common = counter.most_common(VOCAB_SIZE - 4)

vocab = {
    "<pad>": 0,
    "<unk>": 1,
    "<start>": 2,
    "<end>" : 3
}

for i, (w, _) in enumerate(most_common, start=4):
    vocab[w] = i

rev_vocab = {i:w for w,i in vocab.items()}

with open(OLD_VOCAB_PATH, "r") as f:
    old_vocab = json.load(f)   # word → index
old_word_set = set(old_vocab.keys())
print("Old vocab size:", len(old_word_set))

new_words = set()
for tok in all_tokens:
    new_words.add(tok)
    
print("Unique new dataset words:", len(new_words))
new_oov = new_words - old_word_set
print("New words not in old vocab:", len(new_oov))
# ------------------------------------------------------------
# 4. Encode + pad
# ------------------------------------------------------------
def encode(tokens):
    seq = ["<start>"] + tokens + ["<end>"]
    if len(seq) >MAX_LEN:
        seq = seq[:MAX_LEN]
        seq[-1] = "<end>"
    return [vocab.get(t, vocab["<unk>"]) for t in seq]

def pad(seq):
    seq = seq[:MAX_LEN]
    return seq + [0] * (MAX_LEN - len(seq))

df["encoded"] = df["tokens"].apply(encode)
df["padded"] = df["encoded"].apply(pad)


# ------------------------------------------------------------
# 5. Train/val/test split by painting
# ------------------------------------------------------------
unique_paintings = df[["art_style", "painting"]].drop_duplicates()
train = unique_paintings.sample(frac=0.7, random_state=42)
remaining = unique_paintings.drop(train.index)
val = remaining.sample(frac=0.5, random_state=42)
test = remaining.drop(val.index)

def assign_split(row):
    key = (row.art_style, row.painting)
    if key in set(map(tuple, train.values)):
        return "train"
    if key in set(map(tuple, val.values)):
        return "val"
    return "test"

df["split"] = df.apply(assign_split, axis=1)

# ------------------------------------------------------------
# 6. Save text preprocessing output
# ------------------------------------------------------------

if recompute:
    df.to_pickle(os.path.join(SAVE_DIR, "df_word_encoded.pkl"))
    with open(os.path.join(SAVE_DIR, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    with open(os.path.join(SAVE_DIR, "rev_vocab.json"), "w", encoding="utf-8") as f:
        json.dump(rev_vocab, f, ensure_ascii=False, indent=2)

    print("Saved text preprocessing to:", SAVE_DIR)