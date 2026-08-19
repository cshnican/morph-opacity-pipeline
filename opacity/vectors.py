"""Meaning vectors: GloVe lookup, or planted synthetic vectors."""

from __future__ import annotations

import gzip
import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache"

# Largest n where a dense n×n float32 similarity matrix is comfortable on this
# machine (~48 GB RAM): 30k × 30k × 4 bytes ≈ 3.6 GB, plus a 6k × 30k rank fold.
PAIRWISE_MAX_N = 30_000

# ~66MB gzipped word2vec-format GloVe 50d (gensim-data mirror).
GLOVE_URL = (
    "https://github.com/piskvorky/gensim-data/releases/download/"
    "glove-wiki-gigaword-50/glove-wiki-gigaword-50.gz"
)
GLOVE_DIM = 50


def _open_glove_stream(url: str = GLOVE_URL):
    req = urllib.request.Request(url, headers={"User-Agent": "morph-opacity-pipeline"})
    raw = urllib.request.urlopen(req, timeout=300)
    if url.endswith(".gz"):
        return gzip.GzipFile(fileobj=raw)
    return raw


def load_glove_for_vocab(
    vocab: list[str],
    cache_path: Path | None = None,
    url: str = GLOVE_URL,
    dim: int = GLOVE_DIM,
) -> dict[str, np.ndarray]:
    """
    Return {word: vector} for the intersection of `vocab` and GloVe.
    Caches the filtered subset so later runs don't re-download.
    """
    cache_path = cache_path or (CACHE_DIR / "glove_sample.npz")
    vocab_set = set(w.lower() for w in vocab)

    stored: dict[str, np.ndarray] = {}
    if cache_path.exists():
        blob = np.load(cache_path, allow_pickle=True)
        words = blob["words"].tolist()
        vecs = blob["vectors"]
        stored = {w: vecs[i] for i, w in enumerate(words)}
        missing = vocab_set - set(stored)
        if not missing:
            return {w: stored[w] for w in vocab_set if w in stored}
    else:
        missing = set(vocab_set)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    found: dict[str, np.ndarray] = {}
    with _open_glove_stream(url) as fh:
        text = io.TextIOWrapper(fh, encoding="utf-8")
        first = True
        for line in text:
            parts = line.rstrip().split(" ")
            if first and len(parts) == 2 and parts[0].isdigit():
                first = False
                continue
            first = False
            if not parts:
                continue
            w = parts[0]
            if w in missing and len(parts) > dim:
                found[w] = np.asarray(parts[1 : 1 + dim], dtype=np.float32)
                if len(found) == len(missing):
                    break

    stored.update(found)
    if not any(w in stored for w in vocab_set):
        raise RuntimeError("GloVe download succeeded but matched 0 vocabulary items")

    words = np.array(list(stored.keys()))
    vectors = np.stack([stored[w] for w in words])
    np.savez_compressed(cache_path, words=words, vectors=vectors)
    return {w: stored[w] for w in vocab_set if w in stored}


def attach_vectors(df: pd.DataFrame, vectors: dict[str, np.ndarray]) -> tuple[pd.DataFrame, np.ndarray]:
    """Keep rows with a vector; return (filtered_df, matrix aligned to df)."""
    keep = df["word"].map(lambda w: w in vectors)
    out = df.loc[keep].reset_index(drop=True)
    mat = np.stack([vectors[w] for w in out["word"]])
    out = out.copy()
    out["vec_norm"] = np.linalg.norm(mat, axis=1)
    mat = mat / (out["vec_norm"].to_numpy()[:, None] + 1e-8)
    return out, mat


def all_but_the_top(mat: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Remove the mean and leading PCs (Mu & Viswanath 2018).

    Those directions in GloVe track frequency / hubness. Scoring form→meaning
    in the residual subspace stops frequent words from looking transparent
    just because they sit near the center of the space.
    """
    if n_components <= 0:
        return mat
    x = np.asarray(mat, dtype=np.float64)
    x = x - x.mean(axis=0)
    n, d = x.shape
    n_components = int(min(n_components, max(d - 1, 0), max(n - 1, 0)))
    if n_components <= 0:
        return mat
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    pcs = vt[:n_components]
    x = x - (x @ pcs.T) @ pcs
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    return x.astype(mat.dtype, copy=False)


def knn_mean_cosine(mat: np.ndarray, k: int = 5) -> np.ndarray:
    """Mean cosine to the k nearest other words (hubness / typicality)."""
    x = np.asarray(mat, dtype=np.float32)
    sim = x @ x.T
    np.fill_diagonal(sim, -np.inf)
    k = int(min(k, mat.shape[0] - 1))
    if k <= 0:
        return np.zeros(mat.shape[0])
    nn = np.partition(sim, -k, axis=1)[:, -k:]
    return nn.mean(axis=1)
