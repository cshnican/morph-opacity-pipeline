"""Frequency ~ opacity regressions, including the two-stage hurdle."""

from __future__ import annotations

import pandas as pd
import statsmodels.formula.api as smf


def join_scores(lexicon: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    return lexicon.merge(scores, on="word", how="inner")


def summarize_by_class(df: pd.DataFrame) -> pd.DataFrame:
    aggs = dict(
        n=("word", "size"),
        mean_opacity=("opacity", "mean"),
        mean_opacity_raw=("opacity_raw", "mean") if "opacity_raw" in df.columns else ("opacity", "mean"),
        mean_cosine=("cosine", "mean"),
        mean_zipf=("zipf_freq", "mean"),
        mean_length=("length", "mean"),
    )
    if "rank_frac" in df.columns:
        aggs["mean_rank_frac"] = ("rank_frac", "mean")
    if "hubness" in df.columns:
        aggs["mean_hubness"] = ("hubness", "mean")
    return (
        df.groupby("class")
        .agg(**aggs)
        .sort_values("mean_opacity", ascending=False)
        .reset_index()
    )


def fit_models(df: pd.DataFrame) -> dict:
    """
    Stage 0: overall opacity ~ frequency, with and without length.
    Stage 1 (hurdle): P(monomorph) ~ frequency.
    Stage 2 (hurdle): among multimorphs, opacity ~ frequency + length.

    `opacity` is the centroid-adjusted measure. Raw cosine opacity is also
    fit (opacity_raw_*) so the hubness confound is visible. When hubness is
    present it is added as a covariate rather than residualizing on frequency.
    """
    models = {}
    models["opacity_freq"] = smf.ols("opacity ~ zipf_freq", data=df).fit()
    models["opacity_freq_len"] = smf.ols(
        "opacity ~ zipf_freq + length", data=df
    ).fit()

    labeled = df
    if "is_monomorph" in df.columns:
        labeled = df.loc[df["is_monomorph"].notna()].copy()
    multi = pd.DataFrame()
    if len(labeled) >= 50 and labeled["is_monomorph"].nunique() == 2:
        models["monomorph_freq"] = smf.logit(
            "is_monomorph ~ zipf_freq",
            data=labeled.assign(is_monomorph=labeled["is_monomorph"].astype(int)),
        ).fit(disp=False)
        multi = labeled.loc[~labeled["is_monomorph"].astype(bool)]
        if len(multi) >= 20:
            models["opacity_multi_freq_len"] = smf.ols(
                "opacity ~ zipf_freq + length", data=multi
            ).fit()

    if "opacity_raw" in df.columns:
        models["opacity_raw_freq"] = smf.ols("opacity_raw ~ zipf_freq", data=df).fit()
        models["opacity_raw_freq_len"] = smf.ols(
            "opacity_raw ~ zipf_freq + length", data=df
        ).fit()
        if len(multi) >= 20:
            models["opacity_raw_multi_freq_len"] = smf.ols(
                "opacity_raw ~ zipf_freq + length", data=multi
            ).fit()

    if "hubness" in df.columns:
        models["opacity_freq_len_hub"] = smf.ols(
            "opacity ~ zipf_freq + length + hubness", data=df
        ).fit()
        if len(multi) >= 20:
            models["opacity_multi_freq_len_hub"] = smf.ols(
                "opacity ~ zipf_freq + length + hubness", data=multi
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
