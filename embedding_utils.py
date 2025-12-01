import numpy as np
import json, os
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from tokenizers import Tokenizer
from pathlib import Path
import pandas as pd

#variable to compute and store all embedding matrices from scratch
compute_matrices = True

TOKENIZER_PATH = "./text_transformers/bpe-tokenizer.json"
VOCAB_PATH = Path("./text_cnn/vocab.json")
DF_DIR = "./text_transformers/df_subword_encoded.pkl"

try:
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    print(f"Tokenizer loaded from: ./bpe-tokenizer.json")
    print(f"Vocab Size: {tokenizer.get_vocab_size()}")

except Exception as e:
    print(f"Failed to load tokenizer due to: {e}")

# -----------------------------------------
# Load vocab
# -----------------------------------------

transformer_vocab = tokenizer.get_vocab()
i2w_sw = {i:w for w,i in transformer_vocab.items()}

# -----------------------------------------
# GloVe / Word2Vec
# -----------------------------------------
def load_word2vec(model_path, vocab, out_path, embedding_dim=300):
    wv = KeyedVectors.load(model_path)
    emb = np.random.normal(0, 1, (len(vocab), embedding_dim)).astype(np.float32)
    found = 0
    for word, idx in vocab.items():
        if word in wv:
            emb[idx] = wv[word]
            found += 1
    print("Coverage:", found / len(vocab))
    np.save(out_path, emb)
    print(f"Matrix saved to {out_path}")
    return emb

def load_fasttxt(model_path, vocab, out_path, embedding_dim=300):
    wv = KeyedVectors.load_word2vec_format(model_path)
    emb = np.random.normal(0, 1, (len(vocab), embedding_dim)).astype(np.float32)
    found = 0
    for word, idx in vocab.items():
        if word in wv:
            emb[idx] = wv[word]
            found += 1
    print("Coverage:", found / len(vocab))
    np.save(out_path, emb)
    print(f"Matrix saved to {out_path}")
    return emb


# -----------------------------------------
# TF-IDF embeddings
# -----------------------------------------
def build_tfidf_embeddings(captions, vocab, out_path, out_dim=128):
    vectorizer = TfidfVectorizer(vocabulary=vocab)
    tfidf_matrix = vectorizer.fit_transform(captions)

    svd = TruncatedSVD(n_components=out_dim)
    reduced = svd.fit_transform(tfidf_matrix)    

    np.save(out_path, reduced)
    return reduced

json_string = VOCAB_PATH.read_text()
cnn_vocab = json.loads(json_string)
id2wrd = {i:w for w,i in cnn_vocab.items()}
print(f"Vocab Loaded with Length: {len(transformer_vocab)}")

fasttxt_path = "./pretrained_embeds/fasttext.vec"
fasttxt_out_path = "./embedding_matrices/fasttext_matrix_bpe.npy"

w2vec_path = "./pretrained_embeds/word2vec.model"
w2vec_out_path = "./embedding_matrices/word2vec_matrix.npy"

tfidf_out_path = "./embedding_matrices/tfidf_matrix.npy"
df = pd.read_pickle(DF_DIR)
captions = df["utterance"]

if compute_matrices:
    '''
    load_word2vec(model_path=w2vec_path,
                vocab=cnn_vocab,
                    out_path=w2vec_out_path)
    '''
    load_fasttxt(model_path=fasttxt_path,
                vocab=transformer_vocab,
                out_path=fasttxt_out_path)
    '''
    build_tfidf_embeddings(captions=captions,
                           vocab=cnn_vocab,
                           out_path=tfidf_out_path)
    '''

with open(os.path.join("./text_transformers", "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(transformer_vocab, f, ensure_ascii=False, indent=2)