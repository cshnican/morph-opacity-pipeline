"""Character n-gram encoder → linear map into meaning space.

Bound morphemes never get their own vectors. The model sees the whole
character sequence and learns whatever regularities (-ed, cardio-, -gram)
are present in the training words. 1, 2, or 5 morphemes are all just strings.
"""

from __future__ import annotations  # allow `tuple[int, int]` style hints on older Pythons

import numpy as np  # array math for the cosine helper
from sklearn.linear_model import Ridge  # L2-penalized linear regression (form → meaning)
from sklearn.feature_extraction.text import TfidfVectorizer  # turns strings into char n-gram tf-idf features
from sklearn.pipeline import Pipeline  # chains the vectorizer and the regressor into one estimator


def make_form_model(
    ngram_range: tuple[int, int] = (2, 5),  # use character n-grams of length 2 through 5
    alpha: float = 8.0,                      # Ridge regularization strength (heuristic, not tuned)
    min_df: int = 1,                         # drop n-grams occurring in fewer than this many training words
) -> Pipeline:
    # Build the two-stage estimator: char tf-idf features → Ridge regression onto meaning vectors.
    return Pipeline(
        [
            (
                "chars",  # stage name for the feature extractor
                TfidfVectorizer(
                    analyzer="char",         # tokens are character n-grams, not word tokens
                    ngram_range=ngram_range, # span of n-gram lengths to extract
                    min_df=min_df,           # ignore ultra-rare n-grams (fit on the training set only)
                    lowercase=True,          # fold case before extracting n-grams
                ),
            ),
            ("ridge", Ridge(alpha=alpha)),   # multi-output Ridge: predicts the full meaning vector
        ]
    )


def row_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between corresponding rows of `a` and `b`."""
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)  # L2-normalize each row of a (+eps avoids /0)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)  # L2-normalize each row of b
    return (a * b).sum(axis=1)  # elementwise product then row-sum == per-row dot of unit vectors == cosine
