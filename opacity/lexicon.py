"""Load the noun sample and attach frequency / length features."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from wordfreq import zipf_frequency

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_sample_nouns(path: Path | None = None) -> pd.DataFrame:
    path = path or (DATA_DIR / "sample_nouns.csv")
    df = pd.read_csv(path)
    df["word"] = df["word"].str.strip().str.lower()
    df = df.drop_duplicates("word").reset_index(drop=True)
    df = df[df["word"].str.fullmatch(r"[a-z]+")].copy()
    df["n_morphemes"] = df["n_morphemes"].astype(int)
    df["is_monomorph"] = df["n_morphemes"] == 1
    df["length"] = df["word"].str.len()
    df["zipf_freq"] = [zipf_frequency(w, "en") for w in df["word"]]
    df["log_freq"] = df["zipf_freq"]  # Zipf is already log10(freq) + constant
    return df.reset_index(drop=True)
