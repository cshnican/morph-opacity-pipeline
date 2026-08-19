"""Score morphological opacity for one or more words.

  python score_word.py electrocardiogram
  python score_word.py dog laptop teacher
  python score_word.py hologram --lexicon data/morpholex_nouns.csv

Trains g(form)→meaning on the lexicon without the query word, then:

  opacity = 1 − (cos(g(form), v) − cos(centroid, v))

High opacity ≈ spelling did not beat a form-blind guess. Below 1 means
form recovered this meaning better than the training-set average.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from opacity.lexicon import load_sample_nouns
from opacity.scores import score_query_words
from opacity.vectors import attach_vectors, load_glove_for_vocab

ROOT = Path(__file__).resolve().parent


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("words", nargs="+", help="word(s) to score (letters only)")
    p.add_argument(
        "--lexicon",
        type=Path,
        default=ROOT / "data" / "sample_nouns.csv",
        help="training noun CSV with a 'word' column",
    )
    p.add_argument(
        "--whiten-d",
        type=int,
        default=2,
        help="drop this many leading GloVe PCs (0 disables)",
    )
    args = p.parse_args()

    queries = [w.strip().lower() for w in args.words]
    bad = [w for w in queries if not w.isalpha()]
    if bad:
        sys.exit(f"only a-z words are supported; got {bad}")

    lexicon = load_sample_nouns(args.lexicon)
    vocab = list(dict.fromkeys(lexicon["word"].tolist() + queries))
    print(f"training lexicon: {len(lexicon)} nouns from {args.lexicon}", file=sys.stderr)
    print("loading GloVe (cached after the first download)…", file=sys.stderr)
    glove = load_glove_for_vocab(vocab)
    train_df, train_mat = attach_vectors(lexicon, glove)
    print(f"in GloVe: {len(train_df)} training nouns", file=sys.stderr)

    scores = score_query_words(
        queries,
        train_df["word"].to_numpy(),
        train_mat,
        glove,
        whiten_d=args.whiten_d,
    )
    show = scores[
        [
            "word",
            "opacity",
            "opacity_raw",
            "cosine",
            "cosine_null",
            "rank_frac",
            "in_lexicon",
            "error",
        ]
    ]
    with pd.option_context("display.max_columns", None, "display.width", 120):
        print(show.to_string(index=False, float_format=lambda x: f"{x: .4f}"))


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
