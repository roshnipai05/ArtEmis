import os
import pandas as pd
from pathlib import Path
import argparse
from tokenizers import Tokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="./", help="Folder containing Img3k dataset and Artemis original CSV")

args = parser.parse_args()

DATA_DIR = Path(args.data_dir)
CSV_PATH = DATA_DIR / "artemis_dataset_release_v0.csv"
IMG_ROOT = DATA_DIR / "Img15k"
VOCAB_SIZE = 8000
MAX_LEN = 30
SAVE_DIR = Path("./text_transformers")

os.makedirs(SAVE_DIR, exist_ok=True)

df = pd.read_pickle(SAVE_DIR / "df_img15k.pkl")

try:
    tokenizer = Tokenizer.from_file(str(SAVE_DIR / "bpe-tokenizer.json"))
    print(f"Tokenizer loaded from: ./bpe-tokenizer.json")
    print(f"Vocab Size: {tokenizer.get_vocab_size()}")

except Exception as e:
    print(f"Failed to load tokenizer due to: {e}")

# ------------------------------------------------------------
# 1. Encode our captions
# ------------------------------------------------------------
texts = df["utterance"].astype(str).tolist()
BATCH = 1000
all_tokens = []
all_ids = []
all_attention = []

for i in range(0, len(texts), BATCH):
    batch = texts[i:i+BATCH]
    encs = tokenizer.encode_batch(batch)
    for enc in encs:
        all_tokens.append(enc.tokens)
        all_ids.append(enc.ids)
        all_attention.append(enc.attention_mask)

df["tokens"] = all_tokens
df["padding"] = all_ids
df["attention_mask"] = all_attention
# ------------------------------------------------------------
# 5. Train/val/test split by painting
# ------------------------------------------------------------
# 1. Compute unique (style, painting) pairs
unique_paintings = df[["art_style", "painting"]].drop_duplicates()

# 2. Sample train/val/test indices at painting level
train = unique_paintings.sample(frac=0.7, random_state=42)
remaining = unique_paintings.drop(train.index)
val = remaining.sample(frac=0.5, random_state=42)
test = remaining.drop(val.index)

# 3. Convert pairs to a single key column (vectorized)
df["key"] = list(zip(df["art_style"], df["painting"]))
train_set = set(zip(train["art_style"], train["painting"]))
val_set   = set(zip(val["art_style"], val["painting"]))
# test set = everything else implicitly

# 4. Vectorized assignment (no apply!)
df["split"] = "test"
df.loc[df["key"].isin(train_set), "split"] = "train"
df.loc[df["key"].isin(val_set),   "split"] = "val"

# 5. Remove helper column
df.drop(columns=["key"], inplace=True)


# ------------------------------------------------------------
# 6. Save text preprocessing output
# ------------------------------------------------------------
df.to_pickle(os.path.join(SAVE_DIR, "df_subword_encoded.pkl"))
print("Saved text preprocessing dataframe to:", SAVE_DIR, "folder")
