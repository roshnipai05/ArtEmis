import os
import pandas as pd
from pathlib import Path
import argparse
from tokenizers import Tokenizer, pre_tokenizers, models, trainers
from tokenizers.normalizers import Sequence, NFKC, Lowercase
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="./", help="Folder containing Img3k dataset and Artemis original CSV")

args = parser.parse_args()

DATA_DIR = Path(args.data_dir)
CSV_PATH = DATA_DIR / "artemis_dataset_release_v0.csv"
IMG_ROOT = DATA_DIR / "Img3k"
VOCAB_SIZE = 8000
MAX_LEN = 30
SAVE_DIR = Path("./text")

os.makedirs(SAVE_DIR, exist_ok=True)

# ------------------------------------------------------------
# 1. Filter to only images in Img3k
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
df.to_pickle(SAVE_DIR / "df_img3k.pkl")
print("IMG3k DF saved to:", SAVE_DIR)


# ------------------------------------------------------------
# 2. Tokenization Pipeline using HuggingFace
# ------------------------------------------------------------
# Creating instance of tokenizer class that using Byte-Pair encoding
tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

# Cleaning the text and lowercasing it using a normalizer
tokenizer.normalizer = Sequence([NFKC(), Lowercase()])

# Setting up pre-tokenizer, i.e. how our model will split captions
tokenizer.pre_tokenizer = Whitespace()

# Creating a trainer which will learn how to break down words into efficient subword tokens based on frequency, etc.
trainer = trainers.BpeTrainer(vocab_size=8000, special_tokens=["<unk", "<pad>", "<start>", "<end>"])

# Training our tokenizer on the set of all captions
tokenizer.train_from_iterator(df["utterance"], trainer=trainer)

# Adding start and end tokens to captions
tokenizer.post_processor = TemplateProcessing(
    single= "<start> $A <end>", 
    special_tokens=[
        ("<start>", tokenizer.token_to_id("<start>")),
        ("<end>", tokenizer.token_to_id("<end>")),
    ]
) 

#Enable truncation and padding for our captions
tokenizer.enable_truncation(max_length=MAX_LEN)
tokenizer.enable_padding(length=MAX_LEN, pad_id=tokenizer.token_to_id("<pad>"), pad_token="<pad>")

# Save tokenizer
tokenizer_path = str(SAVE_DIR / "bpe-tokenizer.json")
tokenizer.save(tokenizer_path)
print(f"Tokenizer saved to: {tokenizer_path}")
