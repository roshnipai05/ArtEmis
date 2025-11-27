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
IMG_ROOT = DATA_DIR / "Img3k"
VOCAB_SIZE = 8000
MAX_LEN = 30
SAVE_DIR = Path("./text_transformers")

os.makedirs(SAVE_DIR, exist_ok=True)

df = pd.read_pickle(SAVE_DIR / "df_img3k.pkl")

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
encodings = tokenizer.encode_batch(texts)

df["tokens"] = [enc.tokens for enc in encodings]
df["padding"] = [enc.ids for enc in encodings]

try:
    df["attention_mask"] = [enc.attention_mask for enc in encodings]
except Exception:
    pad_id = tokenizer.token_to_id("<pad>")
    df["attention_mask"] = [1 if enc.ids!=pad_id else 0 for enc in encodings]

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
df.to_pickle(os.path.join(SAVE_DIR, "df_subword_encoded.pkl"))
print("Saved text preprocessing dataframe to:", SAVE_DIR, "folder")
