"""Cross-validated form→meaning residuals = per-word opacity."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from opacity.model import make_form_model, row_cosine


def cross_validated_opacity(
    words: list[str] | np.ndarray | pd.Series,
    vectors: np.ndarray,
    n_splits: int = 5,
    ngram_range: tuple[int, int] = (2, 5),
    alpha: float = 8.0,
    seed: int = 0,
    whiten_d: int = 0,
) -> pd.DataFrame:
    """
    For each word, train g(form)→meaning on the *other* folds, then score
    how well the held-out meaning is predicted.

        cosine         = cos(g(form_w), v_w)
        cosine_null    = cos(mean(v_train), v_w)     # form-blind baseline
        opacity_raw    = 1 - cosine
        opacity        = 1 - (cosine - cosine_null)  # form must beat centroid
        rank_frac      = fraction of lexicon meanings closer to g(form_w)

    Held-out scoring is required: otherwise g could memorize laptop as a
    whole string and look spuriously transparent.

    Raw cosine is inflated for frequent GloVe hubs (they sit near the
    centroid, so a shrunk predictor matches them for free). Adjusted
    opacity asks whether form recovers *this* meaning beyond that.

    If `whiten_d` > 0, each fold drops that many leading PCs estimated on
    *training* vectors only (all-but-the-top), then scores in that subspace.
    """
    words = np.asarray(list(words))
    n = len(words)
    if n < n_splits:
        raise ValueError(f"need at least {n_splits} words, got {n}")

    pred = np.zeros_like(vectors)
    null_pred = np.zeros_like(vectors)
    targets = np.zeros_like(vectors)
    rank_frac = np.zeros(n)
    fold_id = np.full(n, -1, dtype=int)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for k, (tr, te) in enumerate(kf.split(words)):
        v_tr, v_all = _fold_space(vectors, tr, whiten_d)
        model = make_form_model(ngram_range=ngram_range, alpha=alpha)
        model.fit(words[tr], v_tr)
        pred[te] = model.predict(words[te])
        null_pred[te] = v_tr.mean(axis=0)
        targets[te] = v_all[te]
        rank_frac[te] = _retrieval_rank_frac(pred[te], v_all[te], v_all)
        fold_id[te] = k

    cosine = np.clip(row_cosine(pred, targets), -1.0, 1.0)
    cosine_null = np.clip(row_cosine(null_pred, targets), -1.0, 1.0)
    return pd.DataFrame(
        {
            "word": words,
            "cosine": cosine,
            "cosine_null": cosine_null,
            "opacity_raw": 1.0 - cosine,
            "opacity": 1.0 - (cosine - cosine_null),
            "rank_frac": rank_frac,
            "fold": fold_id,
        }
    )


def _fold_space(
    vectors: np.ndarray, train_idx: np.ndarray, whiten_d: int
) -> tuple[np.ndarray, np.ndarray]:
    """Project all vectors with PCs / mean fit on the training fold only."""
    if whiten_d <= 0:
        return vectors[train_idx], vectors
    mu = vectors[train_idx].mean(axis=0)
    centered = vectors - mu
    n_components = int(min(whiten_d, centered.shape[1] - 1, len(train_idx) - 1))
    _, _, vt = np.linalg.svd(centered[train_idx], full_matrices=False)
    pcs = vt[:n_components]
    projected = centered - (centered @ pcs.T) @ pcs
    projected = projected / (np.linalg.norm(projected, axis=1, keepdims=True) + 1e-8)
    return projected[train_idx], projected


def _retrieval_rank_frac(
    pred: np.ndarray, true_vec: np.ndarray, candidates: np.ndarray
) -> np.ndarray:
    """Share of candidate meanings closer to g(form) than the true vector."""
    pred_n = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-8)
    cand_n = candidates / (np.linalg.norm(candidates, axis=1, keepdims=True) + 1e-8)
    true_n = true_vec / (np.linalg.norm(true_vec, axis=1, keepdims=True) + 1e-8)
    sim = pred_n @ cand_n.T
    true = np.einsum("ij,ij->i", pred_n, true_n)
    n = candidates.shape[0]
    n_better = (sim > true[:, None]).sum(axis=1)
    return n_better / max(n - 1, 1)


def score_query_words(
    queries: list[str] | np.ndarray | pd.Series,
    train_words: list[str] | np.ndarray | pd.Series,
    train_vectors: np.ndarray,
    query_vectors: dict[str, np.ndarray],
    ngram_range: tuple[int, int] = (2, 5),
    alpha: float = 8.0,
    whiten_d: int = 2,
) -> pd.DataFrame:
    """Opacity for arbitrary words using a fixed training lexicon.

    Each query is scored by a model trained on the lexicon *without* that
    word (if it is in the lexicon). Words never seen in training are the
    intended case: g cannot memorize the whole string.
    """
    train_words = np.asarray([str(w).lower() for w in train_words])
    queries = [str(w).strip().lower() for w in queries]
    rows = []
    for word in queries:
        if word not in query_vectors:
            rows.append(
                {
                    "word": word,
                    "in_lexicon": word in set(train_words),
                    "in_glove": False,
                    "held_out": False,
                    "cosine": np.nan,
                    "cosine_null": np.nan,
                    "opacity_raw": np.nan,
                    "opacity": np.nan,
                    "rank_frac": np.nan,
                    "error": "not in GloVe",
                }
            )
            continue
        mask = train_words != word
        if mask.sum() < 10:
            raise ValueError("need at least 10 training words after holding out the query")
        q = np.asarray(query_vectors[word], dtype=train_vectors.dtype)
        q = q / (np.linalg.norm(q) + 1e-8)
        stacked = np.vstack([train_vectors[mask], q[None, :]])
        tr = np.arange(mask.sum())
        v_tr, v_all = _fold_space(stacked, tr, whiten_d)
        model = make_form_model(ngram_range=ngram_range, alpha=alpha)
        model.fit(train_words[mask], v_tr)
        pred = model.predict(np.array([word]))
        centroid = v_tr.mean(axis=0, keepdims=True)
        target = v_all[-1:]
        cosine = float(np.clip(row_cosine(pred, target)[0], -1.0, 1.0))
        cosine_null = float(np.clip(row_cosine(centroid, target)[0], -1.0, 1.0))
        rank = float(_retrieval_rank_frac(pred, target, v_all)[0])
        rows.append(
            {
                "word": word,
                "in_lexicon": bool((train_words == word).any()),
                "in_glove": True,
                "held_out": True,
                "cosine": cosine,
                "cosine_null": cosine_null,
                "opacity_raw": 1.0 - cosine,
                "opacity": 1.0 - (cosine - cosine_null),
                "rank_frac": rank,
                "error": "",
            }
        )
    return pd.DataFrame(rows)
