import numpy as np
import json
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from tokenizers import Tokenizer
from pathlib import Path

TOKENIZER_PATH = "./text_transformers/bpe-tokenizer.json"
VOCAB_PATH = Path("./text_cnn/vocab.json")
'''
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
'''
json_string = VOCAB_PATH.read_text(encoding='latin-1')
cnn_vocab = json.loads(json_string)
i2w_w = {i:w for w,i in cnn_vocab.items()}
print(f"Vocab Loaded with Length: {len(cnn_vocab)}")

# -----------------------------------------
# GloVe / Word2Vec
# -----------------------------------------
def load_pretrained(model_path, vocab, out_path, embedding_dim=300):
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
    pca = PCA(n_components=out_dim)
    reduced = pca.fit_transform(tfidf_matrix.toarray())
    np.save(out_path, reduced)
    return reduced

w2vec_path = "./pretrained_embeds/word2vec.model"
w2vec_out_path = "./embedding_matrices/word2vec_matrix.npy"

load_pretrained(model_path=w2vec_path,
                vocab=cnn_vocab ,
                out_path=w2vec_out_path)