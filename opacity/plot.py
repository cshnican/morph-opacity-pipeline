"""Figures for the transparency pipeline."""

from __future__ import annotations  # postponed annotation evaluation

from pathlib import Path  # output file paths

import numpy as np  # array math for the fit grid
import statsmodels.api as sm  # OLS + confidence intervals for the trend line
import matplotlib

matplotlib.use("Agg")  # non-interactive backend (write PNGs without a display)
import matplotlib.pyplot as plt  # plotting API


def _add_lm_smooth(ax, x, y) -> None:
    """OLS line + 95% CI ribbon, like ggplot2::geom_smooth(method='lm')."""
    x = np.asarray(x, dtype=float)  # coerce x to float array
    y = np.asarray(y, dtype=float)  # coerce y to float array
    ok = np.isfinite(x) & np.isfinite(y)  # keep only rows where both are finite
    x, y = x[ok], y[ok]
    if len(x) < 3:  # need a few points to fit a line + CI
        return
    fit = sm.OLS(y, sm.add_constant(x)).fit()  # y = b0 + b1*x with an intercept column
    grid = np.linspace(x.min(), x.max(), 200)  # dense x grid for a smooth line
    pred = fit.get_prediction(sm.add_constant(grid)).summary_frame(alpha=0.05)  # mean + 95% CI on the grid
    ax.fill_between(  # shaded 95% confidence band
        grid,
        pred["mean_ci_lower"],
        pred["mean_ci_upper"],
        color="0.2",
        alpha=0.18,
        zorder=20,
        linewidth=0,
    )
    slope = float(fit.params[1])  # x coefficient; params is a plain ndarray here (array API, not formula)
    ax.plot(  # the fitted regression line
        grid,
        pred["mean"],
        color="0.15",
        lw=2.0,
        zorder=21,
        label=f"lm (slope={slope:.3f})",  # show the slope in the legend
    )


def _scatter_vs_freq(ax, df, y: str = "transparency") -> None:
    """Draw one transparency-vs-frequency scatter with an OLS trend onto `ax`."""
    large = len(df) > 3000  # shrink markers / lower alpha for dense clouds
    ax.scatter(
        df["zipf_freq"],           # x = Zipf frequency
        df[y],                     # y = transparency (or another score column)
        s=6 if large else 18,      # marker size
        alpha=0.18 if large else 0.65,  # transparency of points
        color="#4C78A8",
        label=f"n={len(df)}",      # sample size in the legend
    )
    _add_lm_smooth(ax, df["zipf_freq"], df[y])  # overlay the OLS line + CI
    ax.legend(frameon=False, fontsize=8, loc="best")  # legend (n + slope)
    ax.grid(alpha=0.3)  # light grid


def plot_transparency_vs_freq(
    df,
    out: Path,
    title: str,
    y: str = "transparency",
) -> None:
    """Single-panel transparency-vs-frequency figure saved to `out`."""
    fig, ax = plt.subplots(figsize=(8, 5.2))  # one axes
    _scatter_vs_freq(ax, df, y=y)             # draw the scatter + trend
    ax.set_xlabel("frequency (Zipf)")
    ax.set_ylabel("transparency  (cos_form − cos_centroid)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=140)  # write PNG
    plt.close(fig)             # free the figure


def plot_transparency_vs_freq_grid(
    panels: list[tuple[object, str]],
    out: Path,
    y: str = "transparency",
) -> None:
    """Scatter grid. `panels` is [(df, title), ...] in reading order."""
    n = len(panels)  # number of sub-panels
    if n <= 0:
        raise ValueError("need at least one panel")
    if n == 4:                       # 4 panels → 2x2
        nrows, ncols = 2, 2
        figsize = (10.5, 8.6)
    else:                            # otherwise a single row of n panels
        nrows, ncols = 1, n
        figsize = (4.8 * n, 4.6)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=True)  # shared y for comparability
    axes_list = np.atleast_1d(axes).ravel()  # flatten to a 1-D list even when there is a single axes
    for ax, (df, title) in zip(axes_list, panels):  # one panel per (df, title)
        _scatter_vs_freq(ax, df, y=y)
        ax.set_title(title)
        ax.set_xlabel("frequency (Zipf)")
    axes_list[0].set_ylabel("transparency  (cos_form − cos_centroid)")  # y-label on the first panel
    if nrows == 2:  # for the 2x2 layout also label the first panel of the second row
        axes_list[ncols].set_ylabel("transparency  (cos_form − cos_centroid)")
    fig.tight_layout()
    fig.savefig(out, dpi=140)  # write PNG
    plt.close(fig)
