"""Figures for the transparency pipeline."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Fixed colors so Noun/Verb/etc. stay the same across figures.
_POS_COLORS = {
    "Noun": "#4C78A8",
    "Verb": "#F58518",
    "Adjective": "#54A24B",
    "Adverb": "#E45756",
    "Name": "#B279A2",
    "Pronoun": "#FF9DA6",
    "Preposition": "#9D755D",
    "Determiner": "#BAB0AC",
    "Conjunction": "#72B7B2",
    "Article": "#8C6D31",
    "Number": "#E4C72F",
    "Interjection": "#D67195",
    "Letter": "#1F9E89",
    "Unclassified": "#7F7F7F",
    "unknown": "#C7C7C7",
    "Other": "#C7C7C7",
}


def plot_transparency_vs_freq(
    df: pd.DataFrame,
    out: Path,
    title: str,
    y: str = "transparency",
) -> None:
    large = len(df) > 3000
    fig, ax = plt.subplots(figsize=(8, 5.2))
    if "pos" in df.columns:
        labels = df["pos"].fillna("unknown").astype(str)
        counts = labels.value_counts()
        if large:
            rare = set(counts[counts < 50].index)
            if rare:
                labels = labels.where(~labels.isin(rare), "Other")
                counts = labels.value_counts()
        cmap = plt.colormaps["tab20"]
        extra_i = 0
        for i, name in enumerate(counts.index):
            sub = df.loc[labels == name]
            if name in _POS_COLORS:
                c = _POS_COLORS[name]
            else:
                c = cmap(extra_i % 20)
                extra_i += 1
            ax.scatter(
                sub["zipf_freq"],
                sub[y],
                s=6 if large else 18,
                alpha=0.18 if large else 0.65,
                color=c,
                label=f"{name} (n={len(sub)})",
                zorder=i + 1,
            )
        ncol = 2 if len(counts) > 6 else 1
        ax.legend(frameon=False, fontsize=8, ncol=ncol, loc="best")
    else:
        ax.scatter(
            df["zipf_freq"],
            df[y],
            s=6 if large else 18,
            alpha=0.18 if large else 0.65,
            color="#4C78A8",
            label=f"n={len(df)}",
        )
        ax.legend(frameon=False, fontsize=8, loc="best")
    ax.set_xlabel("frequency (Zipf)")
    ax.set_ylabel("transparency  (cos_form − cos_centroid)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
