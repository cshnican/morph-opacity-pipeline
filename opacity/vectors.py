"""Meaning vectors: GloVe lookup."""

from __future__ import annotations  # postponed annotation evaluation

import gzip  # decompress the .gz GloVe dump on the fly
import io    # wrap the byte stream as decoded text
import urllib.request  # stream the embedding file over HTTP(S)
from pathlib import Path  # filesystem paths

import numpy as np  # array math + .npz cache I/O
import pandas as pd  # DataFrame handling in attach_vectors

CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache"  # where filtered embeddings are cached

# Largest n where a dense n×n float32 similarity matrix is comfortable on this
# machine (~48 GB RAM): 30k × 30k × 4 bytes ≈ 3.6 GB, plus a 6k × 30k rank fold.
PAIRWISE_MAX_N = 30_000  # cap for hubness / rank (O(n^2) memory)

# ~66MB gzipped word2vec-format GloVe 50d (gensim-data mirror).
GLOVE_URL = (  # English embedding download URL
    "https://github.com/piskvorky/gensim-data/releases/download/"
    "glove-wiki-gigaword-50/glove-wiki-gigaword-50.gz"
)
GLOVE_DIM = 50  # English GloVe dimensionality
GLOVE_EL_URL = (  # Greek embedding download URL (CC100 300d, DFKI)
    "https://huggingface.co/DFKI/glove-el-cc100/resolve/main/el_embeddings.txt"
)
GLOVE_EL_DIM = 300  # Greek GloVe dimensionality
GLOVE_EL_CACHE = CACHE_DIR / "glove_el_subtlex.npz"  # dedicated cache file for Greek vectors


def _open_glove_stream(url: str = GLOVE_URL, timeout: int = 300):
    """Open a (possibly gzipped) embedding file as a readable byte stream."""
    req = urllib.request.Request(url, headers={"User-Agent": "morph-opacity-pipeline"})  # UA header
    raw = urllib.request.urlopen(req, timeout=timeout)  # open the network stream
    if url.endswith(".gz"):                # English dump is gzipped
        return gzip.GzipFile(fileobj=raw)  # transparently decompress
    return raw                             # Greek dump is plain text


def load_glove_for_vocab(
    vocab: list[str],                 # the words we want vectors for
    cache_path: Path | None = None,   # where to read/write the filtered cache
    url: str = GLOVE_URL,             # embedding source URL
    dim: int = GLOVE_DIM,             # expected vector dimensionality
    timeout: int = 300,              # per-read socket timeout (seconds)
    progress_every: int = 0,          # print progress every N scanned rows (0 = silent)
    rescan_missing: bool = True,      # if False, don't re-stream the dump for words absent from cache
) -> dict[str, np.ndarray]:
    """
    Return {word: vector} for the intersection of `vocab` and GloVe.
    Matching is case-folded. Caches the filtered subset so later runs
    don't re-download.

    The cache also records which vocab words were previously *searched for*
    (`attempted`), so words known to be absent from the dump are not
    re-streamed on every run. If `rescan_missing` is False and a cache
    exists, leftover vocab items are treated as absent outright.
    """
    cache_path = cache_path or (CACHE_DIR / "glove_sample.npz")  # default English cache location
    vocab_set = {str(w).casefold() for w in vocab}  # case-folded set of requested words

    stored: dict[str, np.ndarray] = {}  # word→vector loaded from cache
    attempted: set[str] = set()         # words previously searched for (found or confirmed absent)
    if cache_path.exists():             # reuse a prior cache when available
        blob = np.load(cache_path, allow_pickle=True)  # load the .npz
        words = blob["words"].tolist()                 # cached word list
        vecs = blob["vectors"]                         # cached vectors aligned to `words`
        stored = {str(w).casefold(): vecs[i] for i, w in enumerate(words)}  # rebuild lookup
        if "attempted" in blob:                        # older caches may lack this key
            attempted = {str(w) for w in blob["attempted"].tolist()}  # previously-searched words
        missing = vocab_set - set(stored) - attempted  # only truly-new words need a scan
        if not missing:  # everything is either cached or known-absent → no streaming needed
            return {w: stored[w] for w in vocab_set if w in stored}
        if not rescan_missing:  # caller opted out of streaming for leftovers (e.g. huge Greek dump)
            print(
                f"using embedding cache ({len(stored):,} vectors); "
                f"skipping rescan of {len(missing):,} missing types"
            )
            return {w: stored[w] for w in vocab_set if w in stored}
    else:
        missing = set(vocab_set)  # no cache yet → every word is missing

    CACHE_DIR.mkdir(parents=True, exist_ok=True)  # ensure the cache directory exists
    found: dict[str, np.ndarray] = {}  # newly found word→vector this pass
    n_lines = 0                         # count of embedding rows scanned (for progress)
    with _open_glove_stream(url, timeout=timeout) as fh:  # open the (decompressed) stream
        text = io.TextIOWrapper(fh, encoding="utf-8", errors="replace")  # decode bytes to text
        first = True                    # flag to detect a possible "N D" header line
        for line in text:               # iterate embedding rows
            n_lines += 1
            parts = line.rstrip().split(" ")  # split "word v1 v2 ... vd"
            if first and len(parts) == 2 and parts[0].isdigit():  # word2vec header "count dim"
                first = False
                continue                # skip the header
            first = False
            if not parts:               # blank line guard
                continue
            w = parts[0].casefold()     # the token, case-folded to match vocab_set
            if w in missing and len(parts) > dim:  # a word we want, with enough numeric fields
                found[w] = np.asarray(parts[1 : 1 + dim], dtype=np.float32)  # parse its vector
                if len(found) == len(missing):  # got everything we were still looking for
                    break                        # stop early (don't read the rest of the dump)
            if progress_every and n_lines % progress_every == 0:  # periodic progress print
                print(f"  scanned {n_lines:,} embedding rows, found {len(found)}/{len(missing)}")

    stored.update(found)  # merge newly found vectors into the cache dict
    if not any(w in stored for w in vocab_set):  # sanity: at least one requested word must have a vector
        raise RuntimeError("embedding download succeeded but matched 0 vocabulary items")

    attempted |= vocab_set  # we have now searched for every current vocab word (found or not)
    words = np.array(list(stored.keys()))                     # cache word order
    vectors = np.stack([stored[w] for w in words])            # matching vectors
    np.savez_compressed(  # persist vectors AND the attempted set so absent words aren't rescanned
        cache_path,
        words=words,
        vectors=vectors,
        attempted=np.array(sorted(attempted)),
    )
    return {w: stored[w] for w in vocab_set if w in stored}  # only return the requested intersection


def attach_vectors(df: pd.DataFrame, vectors: dict[str, np.ndarray]) -> tuple[pd.DataFrame, np.ndarray]:
    """Keep rows with a vector; return (filtered_df, matrix aligned to df)."""
    keep = df["word"].map(lambda w: w in vectors)  # boolean mask: which words have a GloVe vector
    out = df.loc[keep].reset_index(drop=True)      # drop the rest; reindex 0..m-1 (aligns with mat rows)
    mat = np.stack([vectors[w] for w in out["word"]])  # stack the kept vectors into an (m x d) matrix
    out = out.copy()                                    # avoid SettingWithCopy warnings
    out["vec_norm"] = np.linalg.norm(mat, axis=1)       # record each vector's raw L2 norm
    mat = mat / (out["vec_norm"].to_numpy()[:, None] + 1e-8)  # L2-normalize every row (+eps avoids /0)
    return out, mat  # aligned DataFrame and normalized matrix


def all_but_the_top(mat: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Remove the mean and leading PCs (Mu & Viswanath 2018).

    Those directions in GloVe track frequency / hubness. Scoring form→meaning
    in the residual subspace stops frequent words from looking transparent
    just because they sit near the center of the space.
    """
    if n_components <= 0:  # nothing to remove
        return mat
    x = np.asarray(mat, dtype=np.float64)  # work in double precision for a stable SVD
    x = x - x.mean(axis=0)                 # step 1: center by the global mean
    n, d = x.shape                         # n words, d dims
    n_components = int(min(n_components, max(d - 1, 0), max(n - 1, 0)))  # clamp to a valid rank
    if n_components <= 0:                   # degenerate shapes → nothing to remove
        return mat
    _, _, vt = np.linalg.svd(x, full_matrices=False)  # SVD; rows of vt are principal directions
    pcs = vt[:n_components]                # top-k principal components
    x = x - (x @ pcs.T) @ pcs             # step 2: subtract the projection onto those PCs
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)  # re-normalize rows
    return x.astype(mat.dtype, copy=False)  # cast back to the input dtype (e.g. float32)


def knn_mean_cosine(mat: np.ndarray, k: int = 5) -> np.ndarray:
    """Mean cosine to the k nearest other words (hubness / typicality)."""
    x = np.asarray(mat, dtype=np.float32)  # rows assumed already L2-normalized upstream
    sim = x @ x.T                          # full pairwise cosine matrix (n x n)
    np.fill_diagonal(sim, -np.inf)         # exclude self-similarity from the neighbor search
    k = int(min(k, mat.shape[0] - 1))      # can't have more neighbors than other words
    if k <= 0:                             # trivial lexicon (0 or 1 word)
        return np.zeros(mat.shape[0])
    nn = np.partition(sim, -k, axis=1)[:, -k:]  # per row, grab the k largest similarities (unordered)
    return nn.mean(axis=1)                 # mean cosine to the k nearest neighbors = hubness
