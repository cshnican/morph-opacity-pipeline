"""Figures for the opacity pipeline."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_YLABEL = {
    "opacity": "adjusted opacity  (1 − (cos_form − cos_centroid))",
    "opacity_raw": "raw opacity  (1 − form→meaning cosine)",
}


def plot_opacity_vs_freq(
    df: pd.DataFrame,
    out: Path,
    title: str,
    y: str = "opacity",
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for cls, sub in df.groupby("class"):
        ax.scatter(
            sub["zipf_freq"],
            sub[y],
            s=18,
            alpha=0.65,
            label=f"{cls} (n={len(sub)})",
        )
    ax.set_xlabel("frequency (Zipf)")
    ax.set_ylabel(_YLABEL.get(y, y))
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_opacity_by_class(
    df: pd.DataFrame,
    out: Path,
    title: str,
    y: str = "opacity",
) -> None:
    order = (
        df.groupby("class")[y].mean().sort_values(ascending=False).index.tolist()
    )
    data = [df.loc[df["class"] == c, y].to_numpy() for c in order]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.boxplot(data, tick_labels=order, showfliers=False)
    ax.set_ylabel(_YLABEL.get(y, y))
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
