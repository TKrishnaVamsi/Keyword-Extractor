import hashlib

import numpy as np

from src.extractor import extract_keywords
from src.train import f1


class FakeEncoder:
    """Deterministic stand-in for the BERT encoder so tests run without downloading a model."""

    def _vec(self, text):
        seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        return np.random.default_rng(seed).normal(size=64)

    def encode(self, texts, normalize_embeddings=True, **kwargs):
        vecs = np.stack([self._vec(t) for t in texts])
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def test_extract_respects_top_n():
    text = "Machine learning models need clean data. Data pipelines feed machine learning systems."
    kws = extract_keywords(text, FakeEncoder(), top_n=3, ngram_range=(1, 2), diversity=0.5)
    assert 1 <= len(kws) <= 3
    assert all(isinstance(k, str) for k in kws)


def test_all_stopwords_returns_empty():
    assert extract_keywords("the and of to", FakeEncoder()) == []


def test_f1_score():
    assert f1(["a", "b"], ["a", "c"]) == 0.5
    assert f1([], ["a"]) == 0.0
