"""Meaning vectors: GloVe lookup, or planted synthetic vectors."""

from __future__ import annotations

import gzip
import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache"

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

    if cache_path.exists():
        blob = np.load(cache_path, allow_pickle=True)
        words = blob["words"].tolist()
        vecs = blob["vectors"]
        cached = {w: vecs[i] for i, w in enumerate(words) if w in vocab_set}
        missing = vocab_set - set(cached)
        if not missing:
            return cached
        # cache is stale / incomplete — fall through and rebuild

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
            if w in vocab_set and len(parts) > dim:
                found[w] = np.asarray(parts[1 : 1 + dim], dtype=np.float32)
                if len(found) == len(vocab_set):
                    break

    if not found:
        raise RuntimeError("GloVe download succeeded but matched 0 vocabulary items")

    words = np.array(list(found.keys()))
    vectors = np.stack([found[w] for w in words])
    np.savez_compressed(cache_path, words=words, vectors=vectors)
    return found


def attach_vectors(df: pd.DataFrame, vectors: dict[str, np.ndarray]) -> tuple[pd.DataFrame, np.ndarray]:
    """Keep rows with a vector; return (filtered_df, matrix aligned to df)."""
    keep = df["word"].map(lambda w: w in vectors)
    out = df.loc[keep].reset_index(drop=True)
    mat = np.stack([vectors[w] for w in out["word"]])
    mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)
    return out, mat
