import os
import json
import numpy as np
import logging 
import pandas as pd
from pathlib import Path
from gensim.models import KeyedVectors
from tqdm import tqdm 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# LOGGING SETUP
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', 
    level=logging.INFO
)

# FIXED PATHS
BASE_DIR = Path(r"C:\Users\91887\Documents\ArtEmis")
CSV_PATH = BASE_DIR / "artemis_dataset_release_v0.csv"
VOCAB_PATH = BASE_DIR / "text_cnn" / "vocab.json"

# Embeddings Sources
GLOVE_TXT = Path(r"C:\Users\91887\Documents\Downloads\glove.6B\glove.6B.300d.txt")
FASTTEXT_VEC = Path(r"C:\Users\91887\Documents\Downloads\cc.en.300.vec\cc.en.300.vec")

# Output
EMBED_SAVE_DIR = BASE_DIR / "cnn_lstm" / "updated_embeddings"
os.makedirs(EMBED_SAVE_DIR, exist_ok=True)

EMBED_DIM = 300   

# HELPER: Custom GloVe Converter
def convert_glove_to_w2v_with_progress(glove_input_file, w2v_output_file):
    glove_input_file = str(glove_input_file)
    w2v_output_file = str(w2v_output_file)
    
    print(f"[INFO] Counting lines in {glove_input_file}...")
    try:
        num_lines = sum(1 for _ in open(glove_input_file, 'r', encoding='utf-8', errors='ignore'))
    except Exception as e:
        print(f"[ERROR] Could not read GloVe file: {e}")
        return False
    
    print(f"[INFO] Detected {num_lines} vectors. Converting...")
    
    with open(glove_input_file, 'r', encoding='utf-8', errors='ignore') as fin, \
         open(w2v_output_file, 'w', encoding='utf-8') as fout:
        
        first_line = fin.readline()
        dim = len(first_line.strip().split()) - 1
        fin.seek(0)
        
        fout.write(f"{num_lines} {dim}\n")
        
        for line in tqdm(fin, total=num_lines, unit="vec", desc="Converting GloVe"):
            fout.write(line)
            
    print(f"[INFO] Conversion complete: {w2v_output_file}")
    return True

# 1. Load Vocab
print("-" * 50)
print("LOADING VOCAB")
print("-" * 50)

with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    vocab = json.load(f)

VOCAB_SIZE = len(vocab)
print(f"[INFO] Loaded vocab size = {VOCAB_SIZE}")

# 2. EMBEDDING BUILDER GENERIC FUNCTION
def build_embedding_matrix(wv, vocab, out_path, dim=300):
    if wv is None: return 0, 0

    print(f"\n[INFO] Extracting embeddings for {len(vocab)} words...")
    emb = np.random.normal(0, 0.6, (len(vocab), dim)).astype("float32")
    found = 0
    
    for word, idx in tqdm(vocab.items(), desc="Matching Words"):
        if word in wv:
            emb[idx] = wv[word]
            found += 1
        elif word.lower() in wv:
            emb[idx] = wv[word.lower()]
            found += 1
            
    coverage = (found / len(vocab)) * 100
    print(f"[INFO] Coverage: {found}/{len(vocab)} ({coverage:.2f}%)")
    np.save(out_path, emb)
    print(f"[INFO] Saved matrix to {out_path}")
    return found, coverage

# 3. TF-IDF EMBEDDING GENERATION (SVD / LSA)
def build_tfidf_embeddings(csv_path, vocab, out_path, dim=300):
    print("\n" + "-"*30)
    print("GENERATING TF-IDF EMBEDDINGS")
    print("-"*30)
    
    # 1. Load Texts
    print("[INFO] Loading dataset for TF-IDF calculation...")
    df = pd.read_csv(csv_path)
    corpus = df['utterance'].astype(str).tolist()
    
    # 2. TfidfVectorizer with OUR vocab
    # We pass 'vocabulary=vocab' to enforce our specific ID mapping
    print("[INFO] Vectorizing corpus...")
    vectorizer = TfidfVectorizer(vocabulary=vocab, token_pattern=r"(?u)\b\w+\b")
    X = vectorizer.fit_transform(corpus)
    
    # X shape is (n_documents, n_vocab). 
    # We want word vectors, so we need (n_vocab, n_documents) -> reduced to (n_vocab, dim)
    print(f"[INFO] Term-Document Matrix shape: {X.T.shape}")
    
    # 3. Dimensionality Reduction (LSA)
    print(f"[INFO] Reducing dimensions to {dim} using TruncatedSVD...")
    svd = TruncatedSVD(n_components=dim, random_state=42)
    # Transpose X so rows are words
    word_vectors = svd.fit_transform(X.T) 
    
    # 4. Save
    # Normalize vectors to unit length (optional but recommended for embeddings)
    from sklearn.preprocessing import normalize
    word_vectors = normalize(word_vectors, norm='l2', axis=1)
    
    np.save(out_path, word_vectors.astype("float32"))
    print(f"[INFO] Saved TF-IDF matrix to {out_path}")
    
    # Coverage for TF-IDF is always 100% of the vocab found in corpus
    return len(vocab), 100.0

# EXECUTION & REPORTING
stats = []

#  1. GLOVE 
if GLOVE_TXT.exists():
    try:
        w2v_path = GLOVE_TXT.with_suffix(".w2v")
        kv_path = GLOVE_TXT.with_suffix(".kv")
        
        if not kv_path.exists():
            if not w2v_path.exists():
                 convert_glove_to_w2v_with_progress(GLOVE_TXT, w2v_path)
            KeyedVectors.load_word2vec_format(str(w2v_path), binary=False).save(str(kv_path))
            
        glove_wv = KeyedVectors.load(str(kv_path), mmap='r')
        out_path = EMBED_SAVE_DIR / "glove_matrix.npy"
        found, cov = build_embedding_matrix(glove_wv, vocab, out_path, EMBED_DIM)
        stats.append(["GloVe", VOCAB_SIZE, EMBED_DIM, f"{cov:.2f}%"])
        del glove_wv
    except Exception as e:
        print(f"[ERROR] GloVe failed: {e}")

# 2. FASTTEXT 
if FASTTEXT_VEC.exists():
    try:
        kv_path = FASTTEXT_VEC.with_suffix(".kv")
        if not kv_path.exists():
            KeyedVectors.load_word2vec_format(str(FASTTEXT_VEC), binary=False).save(str(kv_path))
        
        ft_wv = KeyedVectors.load(str(kv_path), mmap='r')
        out_path = EMBED_SAVE_DIR / "fasttext_matrix.npy"
        found, cov = build_embedding_matrix(ft_wv, vocab, out_path, EMBED_DIM)
        stats.append(["FastText", VOCAB_SIZE, EMBED_DIM, f"{cov:.2f}%"])
        del ft_wv
    except Exception as e:
        print(f"[ERROR] FastText failed: {e}")

# 3. TF-IDF 
try:
    out_path = EMBED_SAVE_DIR / "tfidf_matrix.npy"
    found, cov = build_tfidf_embeddings(CSV_PATH, vocab, out_path, EMBED_DIM)
    stats.append(["TF-IDF (SVD)", VOCAB_SIZE, EMBED_DIM, f"{cov:.2f}%"])
except Exception as e:
    print(f"[ERROR] TF-IDF failed: {e}")

# FINAL REPORT
print("\n" + "="*60)
print(f"{'EMBEDDING TYPE':<20} | {'VOCAB':<10} | {'DIM':<5} | {'COVERAGE':<10}")
print("-" * 60)
for row in stats:
    print(f"{row[0]:<20} | {row[1]:<10} | {row[2]:<5} | {row[3]:<10}")
print("="*60)