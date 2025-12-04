import json
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------
# 1. Load vocab + captions
# ----------------------------------------------------
CAPTIONS_PATH = "text_cnn/df_word_encoded.pkl"   # list of (img_path, token_ids)
VOCAB_JSON = "text_cnn/vocab.json"               # stoi
REV_VOCAB_JSON = "text_cnn/rev_vocab.json"       # itos

with open(VOCAB_JSON, "r") as f:
    stoi = json.load(f)

with open(REV_VOCAB_JSON, "r") as f:
    itos = json.load(f)      # itos is expected to be dict of string indices

df = pd.read_pickle(CAPTIONS_PATH)

# ----------------------------------------------------
# 2. Count token frequencies
# ----------------------------------------------------
counter = Counter()

for img_path, caption_ids in df:
    for tok in caption_ids:
        counter[tok] += 1

# Remove special tokens if present
SPECIAL_TOKENS = {0, 1, 2}   # pad, start, end (adjust if needed)
for st in SPECIAL_TOKENS:
    counter.pop(st, None)

sorted_freqs = counter.most_common()
vocab_size = len(sorted_freqs)

print(f"\nTotal vocabulary size (excluding specials): {vocab_size}")

# ----------------------------------------------------
# 3. Helper to print rank information
# ----------------------------------------------------
def print_rank_info(rank):
    if rank > vocab_size:
        print(f"Rank {rank} is out of range (vocab size = {vocab_size}).")
        return
    
    token_id, freq = sorted_freqs[rank - 1]  # rank is 1-based

    token_id_str = str(token_id)
    if token_id_str in itos:
        word = itos[token_id_str]
    else:
        word = f"<unk:{token_id}>"

    print(f"Rank {rank}: word='{word}', token_id={token_id}, frequency={freq}")

# ----------------------------------------------------
# 4. Print the 4k and 8k ranks
# ----------------------------------------------------
print_rank_info(3000)
print_rank_info(4000)
print_rank_info(7990)
print_rank_info(7992)
print_rank_info(7994)
print_rank_info(7996)
print_rank_info(7997)


def count_words_above_rank(sorted_freqs, rank_threshold):
    """
    Returns the number of unique words whose rank is ABOVE rank_threshold.
    Rank is 1-based.
    
    Example:
        rank_threshold = 3000
        → counts ranks 3001, 3002, ..., vocab_size
    """
    vocab_size = len(sorted_freqs)

    if rank_threshold >= vocab_size:
        return 0

    # Words ranked ABOVE threshold → indices rank_threshold..end
    return vocab_size - rank_threshold

num_words_above_3000 = count_words_above_rank(sorted_freqs, 3000)
print("Number of unique words with rank > 3000:", num_words_above_3000)



# ----------------------------------------------------
# 5. Plot the rank-frequency curve
# ----------------------------------------------------
ranks = np.arange(1, vocab_size + 1)
freqs = np.array([f for (_, f) in sorted_freqs])

plt.figure(figsize=(12, 6))
plt.plot(ranks, freqs, linewidth=1)

# Log-log scale (Zipf curve)
plt.xscale("log")
plt.yscale("log")

# Mark rank 4000 and 8000
if vocab_size >= 4000:
    plt.axvline(4000, color="red", linestyle="--", alpha=0.7, label="Rank 4000")
if vocab_size >= 7997:
    plt.axvline(7997, color="green", linestyle="--", alpha=0.7, label="Rank 7990")

plt.title(f"Token Frequency Distribution (Vocab Size: {vocab_size})", fontsize=14)
plt.xlabel("Rank (log scale)", fontsize=12)
plt.ylabel("Frequency (log scale)", fontsize=12)
plt.grid(True, which="both", ls="--", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()



