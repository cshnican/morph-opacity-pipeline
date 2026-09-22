"""Run the morphological-transparency pipeline.

  python run_pipeline.py --mode morpholex  # MorphoLex ∩ GloVe
  python run_pipeline.py --mode subtlex  # SUBTLEX-US ∩ GloVe (downloads once)
  python run_pipeline.py --mode ladec    # LADEC compounds ∩ GloVe
  python run_pipeline.py --mode subtlex-gr  # SUBTLEX-GR ∩ Greek GloVe
  python run_pipeline.py --mode all      # all four + English/Greek figures
"""

from __future__ import annotations  # postponed annotation evaluation

import argparse  # CLI argument parsing
import json      # write subsample metadata sidecars
import os        # chdir to the repo root before running
from pathlib import Path  # filesystem paths

import numpy as np  # typing + array ops in helpers
import pandas as pd  # DataFrames + reading score CSVs for the combined figures

from opacity.analyze import ensure_transparency, fit_models, join_scores, model_table  # scoring→stats
from opacity.lexicon import (  # lexicon loaders and output-path constants
    SUBTLEX_GLOVE,
    SUBTLEX_GR_GLOVE,
    SUBTLEX_NOUNS_GLOVE,
    load_ladec_compounds,
    load_morpholex,
    load_word_csv,
    load_subtlex_gr,
    load_subtlex_us,
)
from opacity.plot import plot_transparency_vs_freq, plot_transparency_vs_freq_grid  # figures
from opacity.scores import cross_validated_opacity  # the held-out transparency scorer
from opacity.vectors import (  # embedding + geometry utilities
    GLOVE_EL_CACHE,
    GLOVE_EL_DIM,
    GLOVE_EL_URL,
    PAIRWISE_MAX_N,
    all_but_the_top,
    attach_vectors,
    knn_mean_cosine,
    load_glove_for_vocab,
)

ROOT = Path(__file__).resolve().parent  # repo root (this file's directory)
OUT = ROOT / "outputs"                  # where score/regression CSVs go
FIG = ROOT / "figures"                  # where PNGs go


def _write_outputs(tag: str, df: pd.DataFrame, title_stem: str) -> None:
    """Persist per-word scores, the regression table, and the single-lexicon figure."""
    df = ensure_transparency(df)  # make sure the transparency column exists
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    scores_path = OUT / f"{tag}_scores.csv"  # per-word scores file
    df.to_csv(scores_path, index=False)

    models = fit_models(df)              # transparency ~ freq (+ hubness)
    table = model_table(models)          # tidy coefficient table
    table.to_csv(OUT / f"{tag}_regression.csv", index=False)

    plot_transparency_vs_freq(  # single-panel scatter for this lexicon
        df, FIG / f"{tag}_transparency_vs_freq.png", f"{title_stem}: transparency vs frequency"
    )

    print(f"\n=== {tag} ===")                              # console summary header
    print(f"n = {len(df)} words")                          # sample size
    print(f"mean transparency = {df['transparency'].mean():.3f}")  # mean delta
    print("\nregressions (term zipf_freq is the frequency slope):")
    freq_rows = table[table["term"] == "zipf_freq"]        # show only the frequency slope rows
    print(freq_rows.to_string(index=False, float_format=lambda x: f"{x: .4f}"))
    print(f"\nwrote tables → {OUT}/{tag}_*")
    print(f"wrote figures → {FIG}/{tag}_*")


# Which score files feed each combined figure, and their panel labels.
ENGLISH_PANELS = (
    ("glove_ladec", "a) LADEC"),
    ("glove_morpholex", "b) MorphoLex"),
    ("glove_subtlex", "c) SUBTLEX-US"),
)
GREEK_PANELS = (
    ("glove_subtlex_gr", "SUBTLEX-GR"),
)


def _panels_from_tags(tags) -> list:
    """Load (DataFrame, title) pairs from each tag's score CSV."""
    panels = []
    for tag, title in tags:
        path = OUT / f"{tag}_scores.csv"
        if not path.exists():  # the lexicon must have been scored first
            raise FileNotFoundError(f"missing {path}; run that lexicon first")
        panels.append((pd.read_csv(path), title))
    return panels


def write_combined_figures() -> None:
    """Write the English (3-panel) and Greek (1-panel) transparency-vs-frequency figures."""
    FIG.mkdir(parents=True, exist_ok=True)
    en_out = FIG / "transparency_vs_freq_english.png"
    plot_transparency_vs_freq_grid(_panels_from_tags(ENGLISH_PANELS), en_out)  # LADEC/MorphoLex/SUBTLEX-US
    print(f"wrote English figure → {en_out}")
    gr_out = FIG / "transparency_vs_freq_greek.png"
    plot_transparency_vs_freq_grid(_panels_from_tags(GREEK_PANELS), gr_out)    # SUBTLEX-GR
    print(f"wrote Greek figure → {gr_out}")


def _with_hubness(lexicon: pd.DataFrame, mat) -> pd.DataFrame:
    """Attach a hubness column, unless the lexicon is too large for the n×n matrix."""
    if len(lexicon) > PAIRWISE_MAX_N:  # O(n^2) memory guard
        print(f"skipping hubness (n={len(lexicon)} > {PAIRWISE_MAX_N})")
        return lexicon
    out = lexicon.copy()
    out["hubness"] = knn_mean_cosine(mat)  # mean cosine to 5 nearest neighbors
    return out


def run_glove(
    seed: int = 0,                   # CV shuffle seed
    whiten_d: int = 2,               # #leading PCs to drop (all-but-the-top)
    lexicon_path: Path | None = None,  # external word CSV to score
    tag: str = "glove_morpholex",    # output prefix
    title_stem: str = "MorphoLex (GloVe)",  # figure title stem
) -> pd.DataFrame:
    """Score an arbitrary external word list (via --mode glove --lexicon ...)."""
    if lexicon_path is None:
        raise ValueError("run_glove requires lexicon_path (an external word CSV)")
    lexicon = load_word_csv(lexicon_path)  # load words + features
    print(f"lexicon: {len(lexicon)} unique types ({lexicon_path})")
    vectors = load_glove_for_vocab(lexicon["word"].tolist())  # fetch/lookup GloVe vectors
    lexicon, mat = attach_vectors(lexicon, vectors)           # keep words with a vector; build matrix
    print(f"in GloVe: {len(lexicon)} types")
    if whiten_d:  # hubness is measured in the (globally) whitened space when whitening is on
        print(f"all-but-the-top: dropping {whiten_d} leading PC(s) (fit on train folds)")
        lexicon = _with_hubness(lexicon, all_but_the_top(mat, n_components=whiten_d))
    else:
        lexicon = _with_hubness(lexicon, mat)
    scores = cross_validated_opacity(  # held-out transparency (per-fold whitening happens inside)
        lexicon["word"], mat, n_splits=5, seed=seed, whiten_d=whiten_d
    )
    df = join_scores(lexicon, scores)  # merge metadata with scores
    _write_outputs(tag, df, title_stem)
    return df


def run_morpholex(
    seed: int = 0,
    whiten_d: int = 2,
    tag: str = "glove_morpholex",
    max_words: int | None = None,  # subsample cap after the GloVe intersect (defaults to PAIRWISE_MAX_N)
) -> pd.DataFrame:
    """Score MorphoLex ∩ GloVe (subsampled to 30k so hubness fits in memory)."""
    if max_words is None:
        max_words = PAIRWISE_MAX_N
    lexicon = load_morpholex()  # build/load the MorphoLex working list
    print(f"MorphoLex: {len(lexicon)} unique types")
    vectors = load_glove_for_vocab(lexicon["word"].tolist())
    lexicon, mat = attach_vectors(lexicon, vectors)
    n_pool = len(lexicon)  # size before subsampling
    print(f"in GloVe: {n_pool} types")
    if max_words is not None and n_pool > max_words:  # subsample so the n×n hubness matrix is feasible
        print(f"subsample {max_words}/{n_pool} (seed={seed})")
        lexicon, mat = _subsample_aligned(lexicon, mat, max_words, seed)
    if whiten_d:
        print(f"all-but-the-top: dropping {whiten_d} leading PC(s) (fit on train folds)")
        lexicon = _with_hubness(lexicon, all_but_the_top(mat, n_components=whiten_d))
    else:
        lexicon = _with_hubness(lexicon, mat)
    scores = cross_validated_opacity(
        lexicon["word"], mat, n_splits=5, seed=seed, whiten_d=whiten_d
    )
    df = join_scores(lexicon, scores)
    _write_outputs(tag, df, "MorphoLex (GloVe)")
    return df


def run_ladec(seed: int = 0, whiten_d: int = 2, tag: str = "glove_ladec") -> pd.DataFrame:
    """Score LADEC compounds ∩ GloVe (rebuilt from source to enforce native-Zipf filtering)."""
    lexicon = load_ladec_compounds(rebuild=True)  # rebuild so the native-Zipf-only filter is applied
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
    """Randomly keep `n` rows, keeping the DataFrame and matrix row-aligned."""
    sampled = lexicon.sample(n=n, random_state=seed).sort_values("word")  # reproducible subsample, sorted
    mat = mat[sampled.index.to_numpy()]  # select matching matrix rows (index == positional here)
    return sampled.reset_index(drop=True), mat  # reindex 0..n-1 so it stays aligned with mat


def run_subtlex(
    seed: int = 0,
    whiten_d: int = 2,
    min_zipf: float = 0.0,           # frequency floor
    max_words: int | None = None,    # subsample cap after the GloVe intersect
    pos: str | None = None,          # optional dominant-POS filter
    tag: str = "glove_subtlex",
) -> pd.DataFrame:
    """Score SUBTLEX-US ∩ GloVe (subsampled to 30k)."""
    if max_words is None:
        max_words = PAIRWISE_MAX_N
    lexicon = load_subtlex_us(min_zipf=min_zipf, pos=pos)  # load SUBTLEX-US types
    label = f"{pos}s" if pos else "types"  # console label
    print(f"SUBTLEX-US: {len(lexicon)} alphabetic {label} (min_zipf={min_zipf})")
    print("intersecting with GloVe (streams the dump on the first miss)…")
    vectors = load_glove_for_vocab(lexicon["word"].tolist())
    lexicon, mat = attach_vectors(lexicon, vectors)
    n_pool = len(lexicon)
    print(f"in GloVe: {n_pool} {label}")
    if max_words is not None and n_pool > max_words:  # subsample for memory
        print(f"subsample {max_words}/{n_pool} (seed={seed})")
        lexicon, mat = _subsample_aligned(lexicon, mat, max_words, seed)
    out_csv = SUBTLEX_NOUNS_GLOVE if pos else SUBTLEX_GLOVE  # where to save the filtered list
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    save_cols = [c for c in ["word", "zipf_freq", "length", "pos", "n_morphemes", "note"] if c in lexicon.columns]
    lexicon[save_cols].to_csv(out_csv, index=False)  # persist the subsample (used by score_word.py)
    meta = {  # metadata sidecar documenting the subsample
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
    """Score SUBTLEX-GR ∩ Greek GloVe (subsampled to 30k)."""
    if max_words is None:
        max_words = PAIRWISE_MAX_N
    lexicon = load_subtlex_gr(min_zipf=min_zipf)  # load Greek types
    print(f"SUBTLEX-GR: {len(lexicon)} Greek-letter types (min_zipf={min_zipf})")
    print("intersecting with Greek GloVe (CC100 300d; uses cache if present)…")
    vectors = load_glove_for_vocab(  # Greek embeddings: huge, so don't re-stream for cache-misses
        lexicon["word"].tolist(),
        cache_path=GLOVE_EL_CACHE,   # dedicated Greek cache
        url=GLOVE_EL_URL,
        dim=GLOVE_EL_DIM,
        timeout=7200,                # long timeout for the ~9GB file
        progress_every=200_000,      # periodic progress prints during a stream
        rescan_missing=False,        # rely on the cache; skip re-streaming absent words
    )
    lexicon, mat = attach_vectors(lexicon, vectors)
    n_pool = len(lexicon)
    print(f"in Greek GloVe: {n_pool} types")
    if max_words is not None and n_pool > max_words:  # subsample for memory
        print(f"subsample {max_words}/{n_pool} (seed={seed})")
        lexicon, mat = _subsample_aligned(lexicon, mat, max_words, seed)
    SUBTLEX_GR_GLOVE.parent.mkdir(parents=True, exist_ok=True)
    save_cols = [c for c in ["word", "zipf_freq", "length", "n_morphemes", "note"] if c in lexicon.columns]
    lexicon[save_cols].to_csv(SUBTLEX_GR_GLOVE, index=False)  # persist the filtered subsample
    meta = {  # metadata sidecar
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
    """Parse CLI flags and dispatch to the requested lexicon runner(s)."""
    p = argparse.ArgumentParser(description=__doc__)  # program description = module docstring
    p.add_argument(
        "--mode",
        choices=("morpholex", "glove", "subtlex", "subtlex-gr", "ladec", "all"),  # which lexicon(s) to run
        default="morpholex",
    )
    p.add_argument("--seed", type=int, default=0)  # CV shuffle + subsample seed
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
    args = p.parse_args()  # parse the command line

    if args.mode == "morpholex":  # MorphoLex only
        run_morpholex(seed=args.seed, whiten_d=args.whiten_d)
    if args.mode == "glove":      # arbitrary external lexicon
        if args.lexicon is None:
            raise SystemExit("--mode glove requires --lexicon (an external word CSV)")
        tag = args.tag if args.tag != "glove" else args.lexicon.stem  # derive tag from filename if default
        run_glove(
            seed=args.seed,
            whiten_d=args.whiten_d,
            lexicon_path=args.lexicon,
            tag=tag,
            title_stem=f"{args.lexicon.name} (GloVe)",
        )
    if args.mode == "subtlex":    # SUBTLEX-US only
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
    if args.mode == "ladec":      # LADEC only
        run_ladec(seed=args.seed, whiten_d=args.whiten_d)
    if args.mode == "subtlex-gr":  # Greek only
        tag = args.tag if args.tag != "glove" else "glove_subtlex_gr"
        run_subtlex_gr(
            seed=args.seed,
            whiten_d=args.whiten_d,
            min_zipf=args.min_zipf,
            max_words=args.max_words,
            tag=tag,
        )
    if args.mode == "all":        # run all four, then draw both combined figures
        run_ladec(seed=args.seed, whiten_d=args.whiten_d)
        run_morpholex(seed=args.seed, whiten_d=args.whiten_d)
        run_subtlex(
            seed=args.seed,
            whiten_d=args.whiten_d,
            min_zipf=args.min_zipf,
            max_words=args.max_words,
            pos=args.pos,
        )
        run_subtlex_gr(
            seed=args.seed,
            whiten_d=args.whiten_d,
            min_zipf=args.min_zipf,
            max_words=args.max_words,
        )
        write_combined_figures()  # English 3-panel + Greek 1-panel


if __name__ == "__main__":
    os.chdir(ROOT)  # run relative to the repo root so data/ and outputs/ resolve
    main()
