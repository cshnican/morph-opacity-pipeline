"""Frequency ~ transparency regressions."""

from __future__ import annotations  # postponed annotation evaluation

import pandas as pd  # DataFrame handling
import statsmodels.formula.api as smf  # OLS via R-style formulas

_DROP = [  # legacy columns from earlier pipeline versions; removed if present so they don't leak in
    "is_monomorph",
    "opacity",
    "opacity_raw",
    "transparency_raw",
]


def ensure_transparency(df: pd.DataFrame) -> pd.DataFrame:
    """Transparency is form cosine minus the training-centroid baseline."""
    out = df.copy()  # never mutate the caller's frame
    if "transparency" not in out.columns:  # compute it if a raw scores frame was passed
        if {"cosine", "cosine_null"} <= set(out.columns):  # preferred source columns
            out["transparency"] = out["cosine"] - out["cosine_null"]  # delta = form − baseline
        elif "opacity" in out.columns:  # backward-compat with an old "opacity" convention
            out["transparency"] = 1.0 - out["opacity"]
    drop = [c for c in _DROP if c in out.columns]  # which legacy columns are present
    if drop:
        out = out.drop(columns=drop)  # remove them
    return out


def join_scores(lexicon: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """Inner-join lexicon metadata with per-word scores on `word`."""
    return ensure_transparency(lexicon.merge(scores, on="word", how="inner"))  # keep words present in both


def fit_models(df: pd.DataFrame) -> dict:
    """Overall transparency ~ frequency, with a hubness covariate."""
    df = ensure_transparency(df)  # guarantee the transparency column exists
    models = {}
    models["transparency_freq"] = smf.ols("transparency ~ zipf_freq", data=df).fit()  # baseline model
    if "hubness" in df.columns:  # only when hubness was computed (n <= PAIRWISE_MAX_N)
        models["transparency_freq_hub"] = smf.ols(
            "transparency ~ zipf_freq + hubness", data=df  # add hubness as a covariate
        ).fit()
    return models  # dict of name → fitted OLS results


def model_table(models: dict) -> pd.DataFrame:
    """Flatten fitted models into a tidy coefficient table."""
    rows = []
    for name, fit in models.items():          # each model
        for term, coef in fit.params.items(): # each term (Intercept, zipf_freq, hubness, ...)
            rows.append(
                {
                    "model": name,             # model identifier
                    "term": term,              # coefficient name
                    "coef": float(coef),       # point estimate
                    "se": float(fit.bse[term]),      # standard error
                    "p": float(fit.pvalues[term]),   # two-sided p-value
                    "n": int(fit.nobs),               # number of observations used
                }
            )
    return pd.DataFrame(rows)  # one row per (model, term)
