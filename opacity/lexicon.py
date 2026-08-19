"""Load noun lexicons and attach frequency / length features."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from wordfreq import zipf_frequency

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MORPHOLEX_XLSX = DATA_DIR / "external" / "MorphoLEX_en.xlsx"
MORPHOLEX_NOUNS = DATA_DIR / "morpholex_nouns.csv"


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


def build_morpholex_nouns(
    xlsx: Path | None = None,
    out: Path | None = None,
    min_zipf: float = 2.5,
    max_per_class: int = 2500,
    seed: int = 0,
) -> pd.DataFrame:
    """Nouns from MorphoLex-en (Sánchez-Gutiérrez et al. 2018), not hand-coded.

    Sheets are Prefix-Root-Suffix signatures. Mapping onto this pipeline:

    - 0-1-0 (one root, no affixes) → monomorph
    - 1 root + prefixes/suffixes → transparent_multi (derivational)
    - 2+ roots → opaque_multi (compounds; MorphoLex has no transparency ratings)

    Inflected forms are dropped when the stem is also in the noun list.
    Large classes are subsampled so the Ridge CV stays tractable.
    """
    from openpyxl import load_workbook

    xlsx = xlsx or MORPHOLEX_XLSX
    out = out or MORPHOLEX_NOUNS
    if not xlsx.exists():
        raise FileNotFoundError(
            f"MorphoLex workbook not found at {xlsx}. "
            "Download MorphoLEX_en.xlsx from https://github.com/hugomailhot/MorphoLex-en"
        )

    wb = load_workbook(xlsx, read_only=True, data_only=True)
    sheets = [s for s in wb.sheetnames if s[0].isdigit() and s != "0-0-0"]
    rows = []
    for name in sheets:
        n_pref, n_root, n_suff = map(int, name.split("-"))
        ws = wb[name]
        it = ws.iter_rows(values_only=True)
        header = list(next(it))
        idx = {h: i for i, h in enumerate(header) if h}
        for r in it:
            if r is None or r[idx["Word"]] is None:
                continue
            pos = str(r[idx["POS"]] or "")
            if "NN" not in pos:
                continue
            orig = str(r[idx["Word"]]).strip()
            word = orig.lower()
            if orig.isupper() or not word.isalpha() or len(word) < 3:
                continue
            if orig.istitle() and zipf_frequency(word, "en") < 4.0:
                continue
            nm = r[idx["Nmorph"]]
            rows.append(
                {
                    "word": word,
                    "n_morphemes": int(nm) if nm is not None else n_pref + n_root + n_suff,
                    "n_pref": n_pref,
                    "n_root": n_root,
                    "n_suff": n_suff,
                    "prs": str(r[idx["PRS_signature"]]),
                    "note": str(r[idx["MorphoLexSegm"]]) if "MorphoLexSegm" in idx else name,
                }
            )
    wb.close()

    df = pd.DataFrame(rows).drop_duplicates("word")
    stems = set(df["word"])
    inflected = set()
    for w in df["word"]:
        dropped = False
        for suf in ("s", "es", "ed", "ing"):
            if w.endswith(suf) and w[: -len(suf)] in stems:
                inflected.add(w)
                dropped = True
                break
        if not dropped and w.endswith("ing") and (w[:-3] + "e") in stems:
            inflected.add(w)
    df = df.loc[~df["word"].isin(inflected)].copy()

    def morpho_class(r) -> str:
        if r.n_root == 1 and r.n_pref == 0 and r.n_suff == 0:
            return "monomorph"
        if r.n_root >= 2:
            return "opaque_multi"
        return "transparent_multi"

    df["class"] = df.apply(morpho_class, axis=1)
    df["zipf_freq"] = [zipf_frequency(w, "en") for w in df["word"]]
    df = df.loc[df["zipf_freq"] >= min_zipf].copy()

    parts = []
    for _, sub in df.groupby("class"):
        if len(sub) > max_per_class:
            sub = sub.sample(n=max_per_class, random_state=seed)
        parts.append(sub)
    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values("word").reset_index(drop=True)

    keep = ["word", "n_morphemes", "class", "note", "n_pref", "n_root", "n_suff", "prs"]
    out.parent.mkdir(parents=True, exist_ok=True)
    df[keep].to_csv(out, index=False)
    return df


def load_morpholex_nouns(path: Path | None = None, rebuild: bool = False) -> pd.DataFrame:
    path = path or MORPHOLEX_NOUNS
    if rebuild or not path.exists():
        build_morpholex_nouns(out=path)
    return load_sample_nouns(path)
