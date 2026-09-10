"""Score morphological transparency for one or more words.

  python score_word.py electrocardiogram
  python score_word.py dog laptop teacher
  python score_word.py hologram --lexicon data/morpholex_nouns.csv
  python score_word.py hologram --lexicon data/subtlex_glove.csv

Trains g(form)→meaning on the lexicon without the query word, then:

  transparency = cos(g(form), v) − cos(centroid, v)

High transparency ≈ spelling beat a form-blind guess. Below 0 means
form recovered this meaning worse than the training-set average.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from opacity.lexicon import load_word_csv
from opacity.scores import score_query_words
from opacity.vectors import attach_vectors, load_glove_for_vocab

ROOT = Path(__file__).resolve().parent


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("words", nargs="+", help="word(s) to score (letters only)")
    p.add_argument(
        "--lexicon",
        type=Path,
        default=ROOT / "data" / "morpholex_nouns.csv",
        help="training CSV with a 'word' column (default: MorphoLex nouns)",
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

    lexicon = load_word_csv(args.lexicon)
    vocab = list(dict.fromkeys(lexicon["word"].tolist() + queries))
    print(f"training lexicon: {len(lexicon)} words from {args.lexicon}", file=sys.stderr)
    print("loading GloVe (cached after the first download)…", file=sys.stderr)
    glove = load_glove_for_vocab(vocab)
    train_df, train_mat = attach_vectors(lexicon, glove)
    print(f"in GloVe: {len(train_df)} training words", file=sys.stderr)

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
            "transparency",
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
