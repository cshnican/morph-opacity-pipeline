"""Frequency ~ transparency regressions."""

from __future__ import annotations

import pandas as pd
import statsmodels.formula.api as smf

_DROP = [
    "class",
    "is_monomorph",
    "opacity",
    "opacity_raw",
    "transparency_raw",
]


def ensure_transparency(df: pd.DataFrame) -> pd.DataFrame:
    """Transparency is form cosine minus the training-centroid baseline."""
    out = df.copy()
    if "transparency" not in out.columns:
        if {"cosine", "cosine_null"} <= set(out.columns):
            out["transparency"] = out["cosine"] - out["cosine_null"]
        elif "opacity" in out.columns:
            out["transparency"] = 1.0 - out["opacity"]
    drop = [c for c in _DROP if c in out.columns]
    if drop:
        out = out.drop(columns=drop)
    return out


def join_scores(lexicon: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    return ensure_transparency(lexicon.merge(scores, on="word", how="inner"))


def fit_models(df: pd.DataFrame) -> dict:
    """Overall transparency ~ frequency, with length and hubness covariates."""
    df = ensure_transparency(df)
    models = {}
    models["transparency_freq"] = smf.ols("transparency ~ zipf_freq", data=df).fit()
    models["transparency_freq_len"] = smf.ols(
        "transparency ~ zipf_freq + length", data=df
    ).fit()
    if "hubness" in df.columns:
        models["transparency_freq_len_hub"] = smf.ols(
            "transparency ~ zipf_freq + length + hubness", data=df
        ).fit()
    return models


def model_table(models: dict) -> pd.DataFrame:
    rows = []
    for name, fit in models.items():
        for term, coef in fit.params.items():
            rows.append(
                {
                    "model": name,
                    "term": term,
                    "coef": float(coef),
                    "se": float(fit.bse[term]),
                    "p": float(fit.pvalues[term]),
                    "n": int(fit.nobs),
                }
            )
    return pd.DataFrame(rows)
