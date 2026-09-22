"""Cross-validated form→meaning residuals = per-word transparency."""

from __future__ import annotations  # postponed annotation evaluation (union types in hints)

import numpy as np  # array math
import pandas as pd  # returns results as a DataFrame
from sklearn.model_selection import KFold  # splits words into train/held-out folds

from opacity.model import make_form_model, row_cosine  # the char-ngram→Ridge model and cosine helper
from opacity.vectors import PAIRWISE_MAX_N  # size cap above which the O(n^2) rank step is skipped


def cross_validated_opacity(
    words: list[str] | np.ndarray | pd.Series,  # the word strings, aligned row-for-row with `vectors`
    vectors: np.ndarray,                         # meaning vectors (n x d), already L2-normalized
    n_splits: int = 5,                           # number of CV folds
    ngram_range: tuple[int, int] = (2, 5),       # char n-gram span passed to the form model
    alpha: float = 8.0,                          # Ridge strength passed to the form model
    seed: int = 0,                               # RNG seed for the fold shuffle (reproducible)
    whiten_d: int = 0,                           # #leading PCs to drop per fold (all-but-the-top); 0 disables
    min_df: int | None = None,                   # min document freq for n-grams; auto-chosen if None
    compute_rank: bool | None = None,            # whether to compute retrieval rank; auto if None
) -> pd.DataFrame:
    """
    For each word, train g(form)→meaning on the *other* folds, then score
    how well the held-out meaning is predicted.

        cosine         = cos(g(form_w), v_w)
        cosine_null    = cos(mean(v_train), v_w)   # form-blind baseline
        transparency   = cosine − cosine_null      # form must beat centroid
        rank_frac      = fraction of lexicon meanings closer to g(form_w)

    Held-out scoring is required: otherwise g could memorize laptop as a
    whole string and look spuriously transparent.

    Raw cosine is inflated for frequent GloVe hubs (they sit near the
    centroid, so a shrunk predictor matches them for free). Transparency
    asks whether form recovers *this* meaning beyond that.

    If `whiten_d` > 0, each fold drops that many leading PCs estimated on
    *training* vectors only (all-but-the-top), then scores in that subspace.

    `rank_frac` needs a large similarity matrix, so it is skipped when n exceeds
    `PAIRWISE_MAX_N` unless `compute_rank=True`. `min_df` defaults to 2 for
    large lexicons.
    """
    words = np.asarray(list(words))  # normalize input to a 1-D numpy array of strings
    n = len(words)                   # number of words / rows
    if n < n_splits:                 # cannot make K folds with fewer than K items
        raise ValueError(f"need at least {n_splits} words, got {n}")
    if min_df is None:               # auto: require n-grams in >=2 words for big lexicons, else 1
        min_df = 2 if n >= 3000 else 1
    if compute_rank is None:         # auto: only build the n x n rank matrix when n is small enough
        compute_rank = n <= PAIRWISE_MAX_N

    pred = np.zeros_like(vectors)       # will hold g(form_w) for each word (filled per fold)
    null_pred = np.zeros_like(vectors)  # will hold the train-fold centroid broadcast to held-out rows
    targets = np.zeros_like(vectors)    # will hold the (possibly whitened) true meaning vectors
    rank_frac = np.full(n, np.nan) if compute_rank else None  # per-word retrieval rank, or None if skipped
    fold_id = np.full(n, -1, dtype=int)  # which fold each word was held out in (for bookkeeping)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)  # shuffled K-fold splitter

    for k, (tr, te) in enumerate(kf.split(words)):  # tr = train indices, te = held-out indices for fold k
        if n >= 5000:  # progress print only for large runs
            print(f"  fold {k + 1}/{n_splits} (n_train={len(tr)}, min_df={min_df})")
        v_tr, v_all = _fold_space(vectors, tr, whiten_d)  # project vectors using PCs/mean fit on train only
        model = make_form_model(ngram_range=ngram_range, alpha=alpha, min_df=min_df)  # fresh model per fold
        model.fit(words[tr], v_tr)             # learn form→meaning on the training words only
        pred[te] = model.predict(words[te])    # predict meaning for held-out words from their spelling
        null_pred[te] = v_tr.mean(axis=0)      # form-blind baseline = mean of training meaning vectors
        targets[te] = v_all[te]                # store the held-out words' true (projected) vectors
        if compute_rank:                       # optional retrieval metric against all words
            rank_frac[te] = _retrieval_rank_frac(pred[te], v_all[te], v_all)
        fold_id[te] = k                        # record fold membership

    cosine = np.clip(row_cosine(pred, targets), -1.0, 1.0)        # cos(prediction, truth), clamped to [-1,1]
    cosine_null = np.clip(row_cosine(null_pred, targets), -1.0, 1.0)  # cos(centroid, truth), clamped
    out = {
        "word": words,                        # word column
        "cosine": cosine,                     # form-based similarity
        "cosine_null": cosine_null,           # form-blind baseline similarity
        "transparency": cosine - cosine_null, # the per-word transparency score (the paper's delta)
        "fold": fold_id,                      # fold index the word was scored in
    }
    if compute_rank:                          # attach rank only when it was computed
        out["rank_frac"] = rank_frac
    return pd.DataFrame(out)                   # one row per word


def _fold_space(
    vectors: np.ndarray, train_idx: np.ndarray, whiten_d: int
) -> tuple[np.ndarray, np.ndarray]:
    """Project all vectors with PCs / mean fit on the training fold only."""
    if whiten_d <= 0:                       # no whitening requested
        return vectors[train_idx], vectors  # return raw train rows and raw all-rows unchanged
    mu = vectors[train_idx].mean(axis=0)    # training-fold mean (all-but-the-top step 1: center)
    centered = vectors - mu                 # center every row by the TRAIN mean (no leakage)
    n_components = int(min(whiten_d, centered.shape[1] - 1, len(train_idx) - 1))  # safe #PCs to remove
    _, _, vt = np.linalg.svd(centered[train_idx], full_matrices=False)  # SVD of TRAIN rows; vt = right singular vecs
    pcs = vt[:n_components]                  # the top principal directions (fit on train only)
    projected = centered - (centered @ pcs.T) @ pcs  # remove the top-PC components from every row
    projected = projected / (np.linalg.norm(projected, axis=1, keepdims=True) + 1e-8)  # re-normalize rows
    return projected[train_idx], projected   # (train rows, all rows) in the whitened space


def _retrieval_rank_frac(
    pred: np.ndarray, true_vec: np.ndarray, candidates: np.ndarray
) -> np.ndarray:
    """Share of candidate meanings closer to g(form) than the true vector."""
    pred_n = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-8)  # unit-normalize predictions
    cand_n = candidates / (np.linalg.norm(candidates, axis=1, keepdims=True) + 1e-8)  # unit-normalize candidates
    true_n = true_vec / (np.linalg.norm(true_vec, axis=1, keepdims=True) + 1e-8)  # unit-normalize true vectors
    sim = pred_n @ cand_n.T                  # similarity of each prediction to every candidate (rows x N)
    true = np.einsum("ij,ij->i", pred_n, true_n)  # similarity of each prediction to its own true vector
    n = candidates.shape[0]                  # total number of candidates
    n_better = (sim > true[:, None]).sum(axis=1)  # count candidates ranked above the true meaning
    return n_better / max(n - 1, 1)          # convert to a fraction (exclude the item itself)


def score_query_words(
    queries: list[str] | np.ndarray | pd.Series,       # arbitrary words to score
    train_words: list[str] | np.ndarray | pd.Series,   # fixed training lexicon words
    train_vectors: np.ndarray,                          # training lexicon meaning vectors (L2-normalized)
    query_vectors: dict[str, np.ndarray],               # word→GloVe vector lookup for the queries
    ngram_range: tuple[int, int] = (2, 5),              # char n-gram span
    alpha: float = 8.0,                                 # Ridge strength
    whiten_d: int = 2,                                  # #leading PCs to drop (all-but-the-top)
) -> pd.DataFrame:
    """Transparency for arbitrary words using a fixed training lexicon.

    Each query is scored by a model trained on the lexicon *without* that
    word (if it is in the lexicon). Words never seen in training are the
    intended case: g cannot memorize the whole string.
    """
    train_words = np.asarray([str(w).lower() for w in train_words])  # normalize training words to lowercase
    queries = [str(w).strip().lower() for w in queries]              # normalize query words too
    rows = []                                                        # accumulate one result dict per query
    for word in queries:
        if word not in query_vectors:  # no GloVe vector → cannot score; emit a NaN row with an error note
            rows.append(
                {
                    "word": word,
                    "in_lexicon": word in set(train_words),
                    "in_glove": False,
                    "held_out": False,
                    "cosine": np.nan,
                    "cosine_null": np.nan,
                    "transparency": np.nan,
                    "rank_frac": np.nan,
                    "error": "not in GloVe",
                }
            )
            continue
        mask = train_words != word  # hold the query out of training if it happens to be in the lexicon
        if mask.sum() < 10:         # need a minimum training set to fit anything meaningful
            raise ValueError("need at least 10 training words after holding out the query")
        q = np.asarray(query_vectors[word], dtype=train_vectors.dtype)  # the query's meaning vector
        q = q / (np.linalg.norm(q) + 1e-8)  # L2-normalize it to match the training vectors
        stacked = np.vstack([train_vectors[mask], q[None, :]])  # append the query as the last row
        tr = np.arange(mask.sum())  # training indices = every row except the appended query (the last one)
        v_tr, v_all = _fold_space(stacked, tr, whiten_d)  # whiten using train rows only; project all incl. query
        min_df = 2 if mask.sum() >= 3000 else 1  # same min_df heuristic as the CV path
        model = make_form_model(ngram_range=ngram_range, alpha=alpha, min_df=min_df)  # fresh model
        model.fit(train_words[mask], v_tr)  # fit form→meaning on training words only
        pred = model.predict(np.array([word]))  # predict the query's meaning from its spelling
        centroid = v_tr.mean(axis=0, keepdims=True)  # form-blind baseline = training centroid
        target = v_all[-1:]  # the query's true (projected) vector is the last row
        cosine = float(np.clip(row_cosine(pred, target)[0], -1.0, 1.0))       # form similarity
        cosine_null = float(np.clip(row_cosine(centroid, target)[0], -1.0, 1.0))  # baseline similarity
        rank = float(_retrieval_rank_frac(pred, target, v_all)[0])  # retrieval rank against train+query
        rows.append(
            {
                "word": word,
                "in_lexicon": bool((train_words == word).any()),  # was the query part of the lexicon?
                "in_glove": True,
                "held_out": True,
                "cosine": cosine,
                "cosine_null": cosine_null,
                "transparency": cosine - cosine_null,  # per-word transparency
                "rank_frac": rank,
                "error": "",
            }
        )
    return pd.DataFrame(rows)  # one row per query word
