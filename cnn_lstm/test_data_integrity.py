import os
import pickle
import numpy as np

# ---------------------------
# CONFIG — EDIT THESE PATHS
# ---------------------------
VOCAB_SIZE = 8000           # or your actual vocab size
UNK_IDX = 3                  # update if different
START_IDX = 1
END_IDX = 2

EMBEDDING_DIR = r"C:\Users\91887\Documents\ArtEmis\embedding_matrices"
CAPTION_PKL = r"C:\Users\91887\Documents\ArtEmis\text_cnn\df_word_encoded.pkl"

# ---------------------------------------
# 1) TEST EMBEDDING MATRICES (.npy files)
# ---------------------------------------
print("=== TESTING WORD EMBEDDING MATRICES ===")

if not os.path.isdir(EMBEDDING_DIR):
    print(f"Embedding folder does not exist: {EMBEDDING_DIR}")
else:
    files = [f for f in os.listdir(EMBEDDING_DIR) if f.endswith(".npy")]
    
    if len(files) == 0:
        print("❌ No .npy embedding files found.")
    else:
        for fname in files:
            path = os.path.join(EMBEDDING_DIR, fname)
            print(f"\nChecking: {path}")

            try:
                mat = np.load(path)
                print("Loaded successfully.")

                # Check dimensionality
                if mat.ndim != 2:
                    print(f"❌ Not a matrix. ndim={mat.ndim}")
                else:
                    vocab, dim = mat.shape
                    print(f"Shape: {mat.shape}")

                    # Check vocab size
                    if vocab != VOCAB_SIZE:
                        print(f"⚠️ WARNING: Expected vocab size {VOCAB_SIZE}, got {vocab}")
                    else:
                        print("Vocab size matches.")

                    # Check numeric values
                    if not np.isfinite(mat).all():
                        print("❌ Matrix contains NaN or Inf values.")
                    else:
                        print("Numeric values OK.")

            except Exception as e:
                print(f"❌ Error loading {fname}: {e}")


# ---------------------------------------
# 2) TEST CAPTION IDS PICKLE FILE
# ---------------------------------------
print("\n=== TESTING CAPTION IDS PICKLE FILE ===")

if not os.path.isfile(CAPTION_PKL):
    print(f"❌ Caption pickle file not found: {CAPTION_PKL}")
else:
    try:
        with open(CAPTION_PKL, "rb") as f:
            data = pickle.load(f)

        print(f"Loaded file. Total entries = {len(data)}")

        # ----------- CHECK FIRST 5 EXAMPLES -----------
        print("\nFirst 5 samples:")
        for i in range(min(5, len(data))):
            print(data[i])

        # ----------- VALIDATION CHECKS -----------
        print("\nRunning validation checks...")

        for idx, item in enumerate(data):
            if not (isinstance(item, tuple) or isinstance(item, list)):
                raise ValueError(f"Entry {idx} is not a tuple/list.")

            if len(item) != 2:
                raise ValueError(f"Entry {idx} does not have 2 elements.")

            img_path, token_ids = item

            # Check image path
            if not isinstance(img_path, str):
                raise ValueError(f"Entry {idx}: image path is not a string.")

            # Check token list
            if not isinstance(token_ids, list):
                raise ValueError(f"Entry {idx}: token_ids is not a list.")

            for t in token_ids:
                if not isinstance(t, int):
                    raise ValueError(f"Entry {idx}: token ID {t} is not an integer.")

                if not (0 <= t < VOCAB_SIZE):
                    raise ValueError(
                        f"Entry {idx}: token ID {t} is out of range [0, {VOCAB_SIZE})"
                    )

            # Optional check: start/end tokens
            if token_ids[0] != START_IDX:
                print(f"⚠️ WARNING: Entry {idx} does not start with <start> token.")
            if token_ids[-1] != END_IDX:
                print(f"⚠️ WARNING: Entry {idx} does not end with <end> token.")

        print("\n✅ Caption IDs file is correctly formatted.")

    except Exception as e:
        print(f"❌ ERROR: {e}")
