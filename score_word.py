"""Score morphological transparency for one or more words.

  python score_word.py electrocardiogram
  python score_word.py dog laptop teacher
  python score_word.py hologram --lexicon data/morpholex_words.csv
  python score_word.py hologram --lexicon data/subtlex_glove.csv

Trains g(form)→meaning on the lexicon without the query word, then:

  transparency = cos(g(form), v) − cos(centroid, v)

High transparency ≈ spelling beat a form-blind guess. Below 0 means
form recovered this meaning worse than the training-set average.
"""

from __future__ import annotations  # postponed annotation evaluation

import argparse  # CLI parsing
import os        # chdir to repo root
import sys       # stderr messages + exit codes
from pathlib import Path  # paths

import pandas as pd  # display the results table

from opacity.lexicon import load_word_csv       # load the training lexicon
from opacity.scores import score_query_words     # score arbitrary query words
from opacity.vectors import attach_vectors, load_glove_for_vocab  # GloVe lookup + matrix build

ROOT = Path(__file__).resolve().parent  # repo root


def main() -> None:
    """Parse args, train on the lexicon (minus the queries), and print each word's transparency."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("words", nargs="+", help="word(s) to score (letters only)")  # one or more query words
    p.add_argument(
        "--lexicon",
        type=Path,
        default=ROOT / "data" / "morpholex_words.csv",  # default training lexicon
        help="training CSV with a 'word' column (default: MorphoLex)",
    )
    p.add_argument(
        "--whiten-d",
        type=int,
        default=2,  # drop 2 leading PCs (all-but-the-top), matching the pipeline default
        help="drop this many leading GloVe PCs (0 disables)",
    )
    args = p.parse_args()

    queries = [w.strip().lower() for w in args.words]  # normalize query words
    bad = [w for w in queries if not w.isalpha()]      # only a–z queries are supported
    if bad:
        sys.exit(f"only a-z words are supported; got {bad}")

    lexicon = load_word_csv(args.lexicon)  # load the training lexicon + features
    vocab = list(dict.fromkeys(lexicon["word"].tolist() + queries))  # dedup union of lexicon + queries
    print(f"training lexicon: {len(lexicon)} words from {args.lexicon}", file=sys.stderr)
    print("loading GloVe (cached after the first download)…", file=sys.stderr)
    glove = load_glove_for_vocab(vocab)               # fetch vectors for lexicon + queries
    train_df, train_mat = attach_vectors(lexicon, glove)  # keep lexicon words with vectors; build matrix
    print(f"in GloVe: {len(train_df)} training words", file=sys.stderr)

    scores = score_query_words(  # score each query, holding it out of training if present
        queries,
        train_df["word"].to_numpy(),
        train_mat,
        glove,
        whiten_d=args.whiten_d,
    )
    show = scores[  # columns to display
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
    with pd.option_context("display.max_columns", None, "display.width", 120):  # don't truncate output
        print(show.to_string(index=False, float_format=lambda x: f"{x: .4f}"))


if __name__ == "__main__":
    os.chdir(ROOT)  # resolve data/ paths relative to the repo root
    main()
