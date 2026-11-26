"""BERT-based keyword extraction (KeyBERT-style): candidate n-grams are ranked by
cosine similarity to the document embedding, then re-ranked with MMR for diversity."""
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer


@lru_cache(maxsize=4)
def get_encoder(model_name: str):
    return SentenceTransformer(model_name)


def extract_keywords(text, encoder, top_n=5, ngram_range=(1, 2), diversity=0.5, candidate_pool=30):
    try:
        vectorizer = CountVectorizer(ngram_range=tuple(ngram_range), stop_words="english")
        vectorizer.fit([text])
    except ValueError:  # empty vocabulary, e.g. only stop words
        return []

    candidates = np.array(vectorizer.get_feature_names_out())
    doc_emb = encoder.encode([text], normalize_embeddings=True)
    cand_emb = encoder.encode(list(candidates), normalize_embeddings=True)

    doc_sim = (cand_emb @ doc_emb.T).ravel()
    pool = np.argsort(doc_sim)[::-1][: min(candidate_pool, len(candidates))]
    pool_emb, pool_sim = cand_emb[pool], doc_sim[pool]
    cand_sim = pool_emb @ pool_emb.T

    selected = [0]  # pool is sorted by relevance, so index 0 is the best candidate
    remaining = list(range(1, len(pool)))
    k = min(top_n, len(pool))
    while len(selected) < k and remaining:
        def mmr(i):
            redundancy = max(cand_sim[i, s] for s in selected)
            return (1 - diversity) * pool_sim[i] - diversity * redundancy

        best = max(remaining, key=mmr)
        selected.append(best)
        remaining.remove(best)

    return [str(candidates[pool[i]]) for i in selected]
