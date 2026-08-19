"""Planted-opacity lexicon used to check that residuals recover transparency."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_lexicon(
    n_mono: int = 80,
    n_opaque: int = 80,
    n_trans2: int = 100,
    n_trans3: int = 40,
    dim: int = 32,
    seed: int = 0,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Build words from a private morpheme inventory.

    Transparent multimorphs: meaning = sum of morpheme vectors (+ tiny noise).
    Opaque multimorphs and monomorphs: meaning is an independent random vector
    (form does not predict meaning). Frequency is planted so frequent items
    are more often opaque — the pattern the real analysis is looking for.
    """
    rng = np.random.default_rng(seed)

    morphs = _unique_morphs(rng, n=60, min_len=3, max_len=4)
    morph_vec = rng.normal(size=(len(morphs), dim))
    morph_vec /= np.linalg.norm(morph_vec, axis=1, keepdims=True) + 1e-8
    morph_ix = {m: i for i, m in enumerate(morphs)}

    rows = []
    vectors = []

    def take_morphs(k: int) -> list[str]:
        return [str(m) for m in rng.choice(morphs, size=k, replace=False)]

    for _ in range(n_mono):
        m = take_morphs(1)[0]
        meaning = rng.normal(size=dim)
        rows.append((m, 1, "monomorph", "planted arbitrary"))
        vectors.append(meaning)

    for _ in range(n_opaque):
        ms = take_morphs(2)
        meaning = rng.normal(size=dim)
        rows.append(("".join(ms), 2, "opaque_multi", "planted lexicalized"))
        vectors.append(meaning)

    for _ in range(n_trans2):
        ms = take_morphs(2)
        meaning = morph_vec[morph_ix[ms[0]]] + morph_vec[morph_ix[ms[1]]]
        meaning = meaning + 0.05 * rng.normal(size=dim)
        rows.append(("".join(ms), 2, "transparent_multi", "planted compositional"))
        vectors.append(meaning)

    for _ in range(n_trans3):
        ms = take_morphs(3)
        meaning = sum(morph_vec[morph_ix[m]] for m in ms)
        meaning = meaning + 0.05 * rng.normal(size=dim)
        rows.append(("".join(ms), 3, "transparent_multi", "planted compositional"))
        vectors.append(meaning)

    df = pd.DataFrame(rows, columns=["word", "n_morphemes", "class", "note"])
    df = df.drop_duplicates("word").reset_index(drop=True)
    df["is_monomorph"] = df["n_morphemes"] == 1
    df["length"] = df["word"].str.len()

    # Plant frequency: opaque / monomorphemic items are used more.
    base = {"monomorph": 5.5, "opaque_multi": 4.8, "transparent_multi": 3.2}
    df["zipf_freq"] = [
        base[c] + float(rng.normal(0, 0.35)) for c in df["class"]
    ]
    df["log_freq"] = df["zipf_freq"]

    mat = np.stack(vectors[: len(df)])
    # drop vectors for any dropped duplicate rows by rebuilding from surviving order
    # (duplicates were rare; rebuild aligned matrix from the row list before drop
    # using the surviving words)
    word_to_vec = {r[0]: v for r, v in zip(rows, vectors)}
    mat = np.stack([word_to_vec[w] for w in df["word"]])
    mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)
    return df, mat


def _unique_morphs(rng: np.random.Generator, n: int, min_len: int, max_len: int) -> list[str]:
    alphabet = np.array(list("abcdefghijklmnopqrstuvwxyz"))
    seen: set[str] = set()
    out: list[str] = []
    while len(out) < n:
        k = int(rng.integers(min_len, max_len + 1))
        m = "".join(rng.choice(alphabet, size=k))
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out
