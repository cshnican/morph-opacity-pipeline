"""Frequency ~ opacity regressions, including the two-stage hurdle."""

from __future__ import annotations

import pandas as pd
import statsmodels.formula.api as smf


def join_scores(lexicon: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    return lexicon.merge(scores, on="word", how="inner")


def summarize_by_class(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("class")
        .agg(
            n=("word", "size"),
            mean_opacity=("opacity", "mean"),
            mean_cosine=("cosine", "mean"),
            mean_zipf=("zipf_freq", "mean"),
            mean_length=("length", "mean"),
        )
        .sort_values("mean_opacity", ascending=False)
        .reset_index()
    )


def fit_models(df: pd.DataFrame) -> dict:
    """
    Stage 0: overall opacity ~ frequency, with and without length.
    Stage 1 (hurdle): P(monomorph) ~ frequency.
    Stage 2 (hurdle): among multimorphs, opacity ~ frequency + length.
    """
    models = {}
    models["opacity_freq"] = smf.ols("opacity ~ zipf_freq", data=df).fit()
    models["opacity_freq_len"] = smf.ols(
        "opacity ~ zipf_freq + length", data=df
    ).fit()
    models["monomorph_freq"] = smf.logit(
        "is_monomorph ~ zipf_freq", data=df.assign(is_monomorph=df["is_monomorph"].astype(int))
    ).fit(disp=False)

    multi = df.loc[~df["is_monomorph"]]
    if len(multi) >= 20:
        models["opacity_multi_freq_len"] = smf.ols(
            "opacity ~ zipf_freq + length", data=multi
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
