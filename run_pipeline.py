"""Run the morphological-opacity pipeline.

  python run_pipeline.py --mode demo     # planted lexicon (always works)
  python run_pipeline.py --mode morpholex  # MorphoLex nouns ∩ GloVe
  python run_pipeline.py --mode subtlex  # SUBTLEX-US ∩ GloVe (downloads once)
  python run_pipeline.py --mode ladec    # LADEC compounds ∩ GloVe
  python run_pipeline.py --mode subtlex-gr  # SUBTLEX-GR ∩ Greek GloVe
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from opacity.analyze import fit_models, join_scores, model_table, summarize_by_class
from opacity.lexicon import (
    SUBTLEX_GLOVE,
    SUBTLEX_GR_GLOVE,
    SUBTLEX_NOUNS_GLOVE,
    load_ladec_compounds,
    load_morpholex_nouns,
    load_word_csv,
    load_subtlex_gr,
    load_subtlex_us,
)
from opacity.plot import plot_opacity_by_class, plot_opacity_vs_freq
from opacity.scores import cross_validated_opacity
from opacity.synthetic import make_synthetic_lexicon
from opacity.vectors import (
    GLOVE_EL_CACHE,
    GLOVE_EL_DIM,
    GLOVE_EL_URL,
    PAIRWISE_MAX_N,
    all_but_the_top,
    attach_vectors,
    knn_mean_cosine,
    load_glove_for_vocab,
)

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
    if len(lexicon) > PAIRWISE_MAX_N:
        print(f"skipping hubness (n={len(lexicon)} > {PAIRWISE_MAX_N})")
        return lexicon
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
    tag: str = "glove_morpholex",
    title_stem: str = "MorphoLex nouns (GloVe)",
) -> pd.DataFrame:
    if lexicon_path is None:
        raise ValueError("run_glove requires lexicon_path (an external word CSV)")
    lexicon = load_word_csv(lexicon_path)
    print(f"lexicon: {len(lexicon)} unique types ({lexicon_path})")
    vectors = load_glove_for_vocab(lexicon["word"].tolist())
    lexicon, mat = attach_vectors(lexicon, vectors)
    print(f"in GloVe: {len(lexicon)} types")
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


def run_morpholex(seed: int = 0, whiten_d: int = 2, tag: str = "glove_morpholex") -> pd.DataFrame:
    lexicon = load_morpholex_nouns()
    print(f"MorphoLex nouns: {len(lexicon)} unique types")
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
    _write_outputs(tag, df, "MorphoLex nouns (GloVe)")
    return df


def run_ladec(seed: int = 0, whiten_d: int = 2, tag: str = "glove_ladec") -> pd.DataFrame:
    lexicon = load_ladec_compounds()
    print(f"LADEC compounds: {len(lexicon)} unique types")
    vectors = load_glove_for_vocab(lexicon["word"].tolist())
    lexicon, mat = attach_vectors(lexicon, vectors)
    print(f"in GloVe: {len(lexicon)} compounds")
    if whiten_d:
        print(f"all-but-the-top: dropping {whiten_d} leading PC(s) (fit on train folds)")
        lexicon = _with_hubness(lexicon, all_but_the_top(mat, n_components=whiten_d))
    else:
        lexicon = _with_hubness(lexicon, mat)
    scores = cross_validated_opacity(
        lexicon["word"], mat, n_splits=5, seed=seed, whiten_d=whiten_d
    )
    df = join_scores(lexicon, scores)
    _write_outputs(tag, df, "LADEC compounds (GloVe)")
    return df


def _subsample_aligned(
    lexicon: pd.DataFrame, mat: np.ndarray, n: int, seed: int
) -> tuple[pd.DataFrame, np.ndarray]:
    sampled = lexicon.sample(n=n, random_state=seed).sort_values("word")
    mat = mat[sampled.index.to_numpy()]
    return sampled.reset_index(drop=True), mat


def run_subtlex(
    seed: int = 0,
    whiten_d: int = 2,
    min_zipf: float = 0.0,
    max_words: int | None = None,
    pos: str | None = None,
    tag: str = "glove_subtlex",
) -> pd.DataFrame:
    if max_words is None:
        max_words = PAIRWISE_MAX_N
    lexicon = load_subtlex_us(min_zipf=min_zipf, pos=pos)
    label = f"{pos}s" if pos else "types"
    print(f"SUBTLEX-US: {len(lexicon)} alphabetic {label} (min_zipf={min_zipf})")
    print("intersecting with GloVe (streams the dump on the first miss)…")
    vectors = load_glove_for_vocab(lexicon["word"].tolist())
    lexicon, mat = attach_vectors(lexicon, vectors)
    n_pool = len(lexicon)
    print(f"in GloVe: {n_pool} {label}")
    if max_words is not None and n_pool > max_words:
        print(f"subsample {max_words}/{n_pool} (seed={seed})")
        lexicon, mat = _subsample_aligned(lexicon, mat, max_words, seed)
    out_csv = SUBTLEX_NOUNS_GLOVE if pos else SUBTLEX_GLOVE
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    save_cols = [c for c in ["word", "zipf_freq", "length", "class", "pos", "n_morphemes", "note"] if c in lexicon.columns]
    lexicon[save_cols].to_csv(out_csv, index=False)
    meta = {
        "seed": seed,
        "n": int(len(lexicon)),
        "n_pool": int(n_pool),
        "pos": pos,
        "min_zipf": min_zipf,
        "pairwise_max_n": PAIRWISE_MAX_N,
        "lexicon": str(out_csv),
    }
    meta_path = out_csv.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote GloVe-filtered lexicon → {out_csv}")
    print(f"wrote subsample metadata → {meta_path} {meta}")
    if whiten_d:
        print(f"all-but-the-top: dropping {whiten_d} leading PC(s) (fit on train folds)")
        lexicon = _with_hubness(lexicon, all_but_the_top(mat, n_components=whiten_d))
    else:
        lexicon = _with_hubness(lexicon, mat)
    scores = cross_validated_opacity(
        lexicon["word"], mat, n_splits=5, seed=seed, whiten_d=whiten_d
    )
    df = join_scores(lexicon, scores)
    title = "SUBTLEX-US nouns ∩ GloVe" if pos else "SUBTLEX-US ∩ GloVe"
    _write_outputs(tag, df, title)
    return df


def run_subtlex_gr(
    seed: int = 0,
    whiten_d: int = 2,
    min_zipf: float = 0.0,
    max_words: int | None = None,
    tag: str = "glove_subtlex_gr",
) -> pd.DataFrame:
    if max_words is None:
        max_words = PAIRWISE_MAX_N
    lexicon = load_subtlex_gr(min_zipf=min_zipf)
    print(f"SUBTLEX-GR: {len(lexicon)} Greek-letter types (min_zipf={min_zipf})")
    print("intersecting with Greek GloVe (CC100 300d; streams ~9GB on the first miss)…")
    vectors = load_glove_for_vocab(
        lexicon["word"].tolist(),
        cache_path=GLOVE_EL_CACHE,
        url=GLOVE_EL_URL,
        dim=GLOVE_EL_DIM,
        timeout=7200,
        progress_every=200_000,
    )
    lexicon, mat = attach_vectors(lexicon, vectors)
    n_pool = len(lexicon)
    print(f"in Greek GloVe: {n_pool} types")
    if max_words is not None and n_pool > max_words:
        print(f"subsample {max_words}/{n_pool} (seed={seed})")
        lexicon, mat = _subsample_aligned(lexicon, mat, max_words, seed)
    SUBTLEX_GR_GLOVE.parent.mkdir(parents=True, exist_ok=True)
    save_cols = [c for c in ["word", "zipf_freq", "length", "class", "n_morphemes", "note"] if c in lexicon.columns]
    lexicon[save_cols].to_csv(SUBTLEX_GR_GLOVE, index=False)
    meta = {
        "seed": seed,
        "n": int(len(lexicon)),
        "n_pool": int(n_pool),
        "min_zipf": min_zipf,
        "pairwise_max_n": PAIRWISE_MAX_N,
        "embeddings": "DFKI/glove-el-cc100",
        "lexicon": str(SUBTLEX_GR_GLOVE),
    }
    meta_path = SUBTLEX_GR_GLOVE.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote GloVe-filtered lexicon → {SUBTLEX_GR_GLOVE}")
    if whiten_d:
        print(f"all-but-the-top: dropping {whiten_d} leading PC(s) (fit on train folds)")
        lexicon = _with_hubness(lexicon, all_but_the_top(mat, n_components=whiten_d))
    else:
        lexicon = _with_hubness(lexicon, mat)
    scores = cross_validated_opacity(
        lexicon["word"], mat, n_splits=5, seed=seed, whiten_d=whiten_d
    )
    df = join_scores(lexicon, scores)
    _write_outputs(tag, df, "SUBTLEX-GR ∩ Greek GloVe")
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("demo", "morpholex", "glove", "subtlex", "subtlex-gr", "ladec", "both"), default="both")
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
        help="External word CSV with a 'word' column (required for --mode glove)",
    )
    p.add_argument(
        "--tag",
        default="glove",
        help="Output prefix for GloVe tables/figures (default: glove)",
    )
    p.add_argument(
        "--min-zipf",
        type=float,
        default=0.0,
        help="SUBTLEX: drop types below this Zipf (default: 0, i.e. GloVe filter only)",
    )
    p.add_argument(
        "--max-words",
        type=int,
        default=None,
        help=(
            "SUBTLEX: after the GloVe intersect, randomly subsample to this many "
            f"types (seed from --seed; default {PAIRWISE_MAX_N} so rank/hubness fit)."
        ),
    )
    p.add_argument(
        "--pos",
        default=None,
        help="SUBTLEX: keep this dominant POS only (e.g. Noun)",
    )
    args = p.parse_args()

    if args.mode in ("demo", "both"):
        run_demo(seed=args.seed)
    if args.mode in ("morpholex", "both"):
        run_morpholex(seed=args.seed, whiten_d=args.whiten_d)
    if args.mode == "glove":
        if args.lexicon is None:
            raise SystemExit("--mode glove requires --lexicon (an external word CSV)")
        tag = args.tag if args.tag != "glove" else args.lexicon.stem
        run_glove(
            seed=args.seed,
            whiten_d=args.whiten_d,
            lexicon_path=args.lexicon,
            tag=tag,
            title_stem=f"{args.lexicon.name} (GloVe)",
        )
    if args.mode == "subtlex":
        pos = args.pos
        tag = args.tag if args.tag != "glove" else "glove_subtlex"
        run_subtlex(
            seed=args.seed,
            whiten_d=args.whiten_d,
            min_zipf=args.min_zipf,
            max_words=args.max_words,
            pos=pos,
            tag=tag,
        )
    if args.mode == "ladec":
        run_ladec(seed=args.seed, whiten_d=args.whiten_d)
    if args.mode == "subtlex-gr":
        tag = args.tag if args.tag != "glove" else "glove_subtlex_gr"
        run_subtlex_gr(
            seed=args.seed,
            whiten_d=args.whiten_d,
            min_zipf=args.min_zipf,
            max_words=args.max_words,
            tag=tag,
        )


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
