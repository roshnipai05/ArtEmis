import numpy as np
import json
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

VOCAB_FILE = "./processed_artemis/text/vocab.json"

# -----------------------------------------
# Load vocab
# -----------------------------------------
vocab = json.load(open(VOCAB_FILE))
idx2word = {i:w for w,i in vocab.items()}


# -----------------------------------------
# GloVe / Word2Vec
# -----------------------------------------
def load_pretrained(path, embedding_dim=300):
    wv = KeyedVectors.load_word2vec_format(path, binary=False)
    emb = np.random.normal(0, 1, (len(vocab), embedding_dim))
    found = 0
    for word, idx in vocab.items():
        if word in wv:
            emb[idx] = wv[word]
            found += 1
    print("Coverage:", found / len(vocab))
    return emb


# -----------------------------------------
# TF-IDF embeddings
# -----------------------------------------
def build_tfidf_embeddings(captions, out_dim=128):
    vectorizer = TfidfVectorizer(vocabulary=vocab)
    tfidf_matrix = vectorizer.fit_transform(captions)
    pca = PCA(n_components=out_dim)
    reduced = pca.fit_transform(tfidf_matrix.toarray())
    return reduced
