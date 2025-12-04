import os
import json
import numpy as np
from pathlib import Path
from gensim.models import KeyedVectors
import pandas as pd

# ============================================================
# FIXED USER PATHS
# ============================================================
VOCAB_PATH = Path(r"C:\Users\91887\Documents\ArtEmis\text_cnn\vocab.json")
CAPTION_PKL = Path(r"C:\Users\91887\Documents\ArtEmis\text_cnn\df_word_encoded.pkl")
EMBED_SAVE_DIR = Path(r"C:\Users\91887\Documents\ArtEmis\cnn_lstm\updated_embeddings")

# Pretrained embedding files YOU must already have
FASTTEXT_BIN = r"C:\Users\91887\Documents\ArtEmis\pretrained_embeds\fasttext.vec"
GLOVE_W2V = r"C:\Users\91887\Documents\ArtEmis\pretrained_embeds\glove_w2v.txt"

os.makedirs(EMBED_SAVE_DIR, exist_ok=True)

# ============================================================
# 1. Load CNN vocabulary (word → index)
# ============================================================
with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    cnn_vocab = json.load(f)

vocab_size = len(cnn_vocab)
print(f"[INFO] Loaded CNN vocab of size: {vocab_size}")

# Reverse vocab: idx → word
id2word = {idx: word for word, idx in cnn_vocab.items()}

# ============================================================
# 2. FastText embeddings
# ============================================================
def build_fasttext_embeddings(model_path, vocab, out_path, embedding_dim=300):
    print("\n[INFO] Loading FastText model...")
    wv = KeyedVectors.load_word2vec_format(model_path)  # .vec format

    print("[INFO] Building embedding matrix...")
    emb_matrix = np.random.normal(0, 1, (len(vocab), embedding_dim)).astype("float32")

    found = 0
    for word, idx in vocab.items():
        if word in wv:
            emb_matrix[idx] = wv[word]
            found += 1

    coverage = found / len(vocab)
    print(f"[INFO] FastText coverage = {coverage*100:.2f}%")

    np.save(out_path, emb_matrix)
    print(f"[INFO] Saved FastText embeddings → {out_path}")

    return emb_matrix


# ============================================================
# 3. GloVe embeddings (converted to word2vec format)
# ============================================================
def build_glove_embeddings(model_path, vocab, out_path, embedding_dim=300):
    print("\n[INFO] Loading GloVe (word2vec format)...")
    wv = KeyedVectors.load_word2vec_format(model_path)

    print("[INFO] Building embedding matrix...")
    emb_matrix = np.random.normal(0, 1, (len(vocab), embedding_dim)).astype("float32")

    found = 0
    for word, idx in vocab.items():
        if word in wv:
            emb_matrix[idx] = wv[word]
            found += 1

    coverage = found / len(vocab)
    print(f"[INFO] GloVe coverage = {coverage*100:.2f}%")

    np.save(out_path, emb_matrix)
    print(f"[INFO] Saved GloVe embeddings → {out_path}")

    return emb_matrix


# ============================================================
# 4. Main — Build ALL embeddings
# ============================================================
if __name__ == "__main__":

    # 1. FASTTEXT
    fasttext_out = EMBED_SAVE_DIR / "fasttext_matrix.npy"
    build_fasttext_embeddings(
        model_path=FASTTEXT_BIN,
        vocab=cnn_vocab,
        out_path=fasttext_out
    )

    # 2. GLOVE
    glove_out = EMBED_SAVE_DIR / "glove_matrix.npy"
    build_glove_embeddings(
        model_path=GLOVE_W2V,
        vocab=cnn_vocab,
        out_path=glove_out
    )

    print("\n[INFO] Completed building ALL embedding matrices.")
    print("[INFO] Saved in:", EMBED_SAVE_DIR)
