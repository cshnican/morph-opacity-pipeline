"""Character n-gram encoder → linear map into meaning space.

Bound morphemes never get their own vectors. The model sees the whole
character sequence and learns whatever regularities (-ed, cardio-, -gram)
are present in the training words. 1, 2, or 5 morphemes are all just strings.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline


def make_form_model(
    ngram_range: tuple[int, int] = (2, 5),
    alpha: float = 8.0,
    min_df: int = 1,
) -> Pipeline:
    return Pipeline(
        [
            (
                "chars",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=ngram_range,
                    min_df=min_df,
                    lowercase=True,
                ),
            ),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )


def row_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return (a * b).sum(axis=1)
