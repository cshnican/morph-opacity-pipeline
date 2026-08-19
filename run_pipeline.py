"""Run the morphological-opacity pipeline.

  python run_pipeline.py --mode demo     # planted lexicon (always works)
  python run_pipeline.py --mode glove    # English nouns + GloVe (downloads once)
  python run_pipeline.py --mode both     # default
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from opacity.analyze import fit_models, join_scores, model_table, summarize_by_class
from opacity.lexicon import load_sample_nouns
from opacity.plot import plot_opacity_by_class, plot_opacity_vs_freq
from opacity.scores import cross_validated_opacity
from opacity.synthetic import make_synthetic_lexicon
from opacity.vectors import all_but_the_top, attach_vectors, knn_mean_cosine, load_glove_for_vocab

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
FIG = ROOT / "figures"


def _write_outputs(tag: str, df: pd.DataFrame, title_stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    scores_path = OUT / f"{tag}_scores.csv"
    df.to_csv(scores_path, index=False)

    by_class = summarize_by_class(df)
    by_class.to_csv(OUT / f"{tag}_by_class.csv", index=False)

    models = fit_models(df)
    table = model_table(models)
    table.to_csv(OUT / f"{tag}_regression.csv", index=False)

    plot_opacity_vs_freq(df, FIG / f"{tag}_opacity_vs_freq.png", f"{title_stem}: adjusted opacity vs frequency")
    plot_opacity_by_class(df, FIG / f"{tag}_opacity_by_class.png", f"{title_stem}: adjusted opacity by class")
    if "opacity_raw" in df.columns:
        plot_opacity_vs_freq(
            df,
            FIG / f"{tag}_opacity_raw_vs_freq.png",
            f"{title_stem}: raw opacity vs frequency",
            y="opacity_raw",
        )

    print(f"\n=== {tag} ===")
    print(f"n = {len(df)} words")
    print(by_class.to_string(index=False, float_format=lambda x: f"{x:6.3f}"))
    print("\nregressions (term zipf_freq is the frequency slope):")
    freq_rows = table[table["term"] == "zipf_freq"]
    print(freq_rows.to_string(index=False, float_format=lambda x: f"{x: .4f}"))
    print(f"\nwrote tables → {OUT}/{tag}_*")
    print(f"wrote figures → {FIG}/{tag}_*")


def _with_hubness(lexicon: pd.DataFrame, mat) -> pd.DataFrame:
    out = lexicon.copy()
    out["hubness"] = knn_mean_cosine(mat)
    return out


def run_demo(seed: int = 0) -> pd.DataFrame:
    lexicon, vectors = make_synthetic_lexicon(seed=seed)
    lexicon = _with_hubness(lexicon, vectors)
    scores = cross_validated_opacity(lexicon["word"], vectors, n_splits=5, seed=seed)
    df = join_scores(lexicon, scores)
    _write_outputs("demo", df, "Planted lexicon")
    return df


def run_glove(
    seed: int = 0,
    whiten_d: int = 2,
    lexicon_path: Path | None = None,
    tag: str = "glove",
    title_stem: str = "English nouns (GloVe)",
) -> pd.DataFrame:
    lexicon = load_sample_nouns(lexicon_path)
    print(f"sample lexicon: {len(lexicon)} unique nouns ({lexicon_path or 'data/sample_nouns.csv'})")
    vectors = load_glove_for_vocab(lexicon["word"].tolist())
    lexicon, mat = attach_vectors(lexicon, vectors)
    print(f"in GloVe: {len(lexicon)} nouns")
    if whiten_d:
        print(f"all-but-the-top: dropping {whiten_d} leading PC(s) (fit on train folds)")
        lexicon = _with_hubness(lexicon, all_but_the_top(mat, n_components=whiten_d))
    else:
        lexicon = _with_hubness(lexicon, mat)
    scores = cross_validated_opacity(
        lexicon["word"], mat, n_splits=5, seed=seed, whiten_d=whiten_d
    )
    df = join_scores(lexicon, scores)
    _write_outputs(tag, df, title_stem)
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("demo", "glove", "both"), default="both")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--whiten-d",
        type=int,
        default=2,
        help="GloVe: drop this many leading embedding PCs (0 disables)",
    )
    p.add_argument(
        "--lexicon",
        type=Path,
        default=None,
        help="CSV of hand-classed nouns (default: data/sample_nouns.csv)",
    )
    p.add_argument(
        "--tag",
        default="glove",
        help="Output prefix for GloVe tables/figures (default: glove)",
    )
    args = p.parse_args()

    if args.mode in ("demo", "both"):
        run_demo(seed=args.seed)
    if args.mode in ("glove", "both"):
        title = "English nouns (GloVe)"
        if args.lexicon is not None:
            title = f"English nouns (GloVe, {args.lexicon.name})"
        run_glove(
            seed=args.seed,
            whiten_d=args.whiten_d,
            lexicon_path=args.lexicon,
            tag=args.tag,
            title_stem=title,
        )


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
