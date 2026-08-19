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
) -> pd.DataFrame:
    """
    For each word, train g(form)→meaning on the *other* folds, then score
    how well the held-out meaning is predicted.

        cosine     = cos(g(form_w), v_w)
        opacity    = 1 - cosine      (high = meaning not recoverable from form)

    Held-out scoring is required: otherwise g could memorize laptop as a
    whole string and look spuriously transparent.
    """
    words = np.asarray(list(words))
    n = len(words)
    if n < n_splits:
        raise ValueError(f"need at least {n_splits} words, got {n}")

    pred = np.zeros_like(vectors)
    fold_id = np.full(n, -1, dtype=int)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for k, (tr, te) in enumerate(kf.split(words)):
        model = make_form_model(ngram_range=ngram_range, alpha=alpha)
        model.fit(words[tr], vectors[tr])
        pred[te] = model.predict(words[te])
        fold_id[te] = k

    cosine = row_cosine(pred, vectors)
    cosine = np.clip(cosine, -1.0, 1.0)
    return pd.DataFrame(
        {
            "word": words,
            "cosine": cosine,
            "opacity": 1.0 - cosine,
            "fold": fold_id,
        }
    )
