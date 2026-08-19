"""Load lexicons and attach frequency / length features."""

from __future__ import annotations

import math
import urllib.request
from pathlib import Path

import pandas as pd
from wordfreq import zipf_frequency

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MORPHOLEX_XLSX = DATA_DIR / "external" / "MorphoLEX_en.xlsx"
MORPHOLEX_NOUNS = DATA_DIR / "morpholex_nouns.csv"
SUBTLEX_PATH = DATA_DIR / "external" / "SUBTLEXus74286.txt"
SUBTLEX_GLOVE = DATA_DIR / "subtlex_glove.csv"
SUBTLEX_NOUNS_GLOVE = DATA_DIR / "subtlex_nouns_glove.csv"
SUBTLEX_URL = (
    "https://raw.githubusercontent.com/cltl/python-for-text-analysis/master/"
    "Data/SUBTLEX-US/SUBTLEXus74286wordstextversion.txt"
)
SUBTLEX_POS_PATH = DATA_DIR / "external" / "SUBTLEX-US_POS_Zipf.xlsx"
SUBTLEX_POS_URL = "https://osf.io/download/55d4847a8c5e4a5fe4a6c8d2/"


def load_sample_nouns(path: Path | None = None) -> pd.DataFrame:
    path = path or (DATA_DIR / "sample_nouns.csv")
    df = pd.read_csv(path)
    df["word"] = df["word"].str.strip().str.lower()
    df = df.drop_duplicates("word").reset_index(drop=True)
    df = df[df["word"].str.fullmatch(r"[a-z]+")].copy()
    if "n_morphemes" in df.columns:
        df["n_morphemes"] = pd.to_numeric(df["n_morphemes"], errors="coerce")
        df["is_monomorph"] = pd.Series(pd.NA, index=df.index, dtype="boolean")
        known = df["n_morphemes"].notna()
        df.loc[known, "is_monomorph"] = df.loc[known, "n_morphemes"] == 1
    else:
        df["n_morphemes"] = pd.NA
        df["is_monomorph"] = pd.Series(pd.NA, index=df.index, dtype="boolean")
    if "class" not in df.columns:
        df["class"] = "unlabeled"
    else:
        df["class"] = df["class"].fillna("unlabeled")
    df["length"] = df["word"].str.len()
    if "zipf_freq" not in df.columns:
        df["zipf_freq"] = [zipf_frequency(w, "en") for w in df["word"]]
    df["log_freq"] = df["zipf_freq"]  # Zipf is already log10(freq) + constant
    df = attach_subtlex_pos(df)
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


def download_subtlex(path: Path | None = None, url: str = SUBTLEX_URL) -> Path:
    path = path or SUBTLEX_PATH
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "morph-opacity-pipeline"})
    with urllib.request.urlopen(req, timeout=120) as src, path.open("wb") as dst:
        dst.write(src.read())
    return path


def download_subtlex_pos(path: Path | None = None, url: str = SUBTLEX_POS_URL) -> Path:
    path = path or SUBTLEX_POS_PATH
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "morph-opacity-pipeline"})
    with urllib.request.urlopen(req, timeout=180) as src, path.open("wb") as dst:
        dst.write(src.read())
    return path


def load_subtlex_pos_table(path: Path | None = None) -> pd.DataFrame:
    """Dominant CLAWS POS per SUBTLEX-US type (Brysbaert, New & Keuleers 2012)."""
    cache = DATA_DIR / "cache" / "subtlex_pos.csv"
    if cache.exists():
        return pd.read_csv(cache)
    xlsx = download_subtlex_pos(path)
    raw = pd.read_excel(xlsx, sheet_name=0, usecols=["Word", "Dom_PoS_SUBTLEX"])
    raw = raw.rename(columns={"Word": "word", "Dom_PoS_SUBTLEX": "pos"})
    raw["word"] = raw["word"].astype(str).str.strip().str.lower()
    raw = raw.loc[raw["word"].str.fullmatch(r"[a-z]+")].drop_duplicates("word")
    raw["pos"] = raw["pos"].where(raw["pos"].notna(), "unknown").astype(str)
    raw.loc[raw["pos"].isin(["nan", "#N/A", "None", "NaN"]), "pos"] = "unknown"
    cache.parent.mkdir(parents=True, exist_ok=True)
    raw[["word", "pos"]].to_csv(cache, index=False)
    return raw[["word", "pos"]]


def attach_subtlex_pos(df: pd.DataFrame) -> pd.DataFrame:
    pos = load_subtlex_pos_table()
    out = df.merge(pos, on="word", how="left", suffixes=("", "_subtlex"))
    if "pos_subtlex" in out.columns:
        out["pos"] = out["pos"].combine_first(out["pos_subtlex"])
        out = out.drop(columns=["pos_subtlex"])
    out["pos"] = out["pos"].fillna("unknown")
    return out


def load_subtlex_us(
    path: Path | None = None,
    min_zipf: float = 0.0,
    morpholex: Path | None = None,
    pos: str | None = None,
) -> pd.DataFrame:
    """SUBTLEX-US types (Brysbaert & New 2009), letters only.

    If `pos` is set (e.g. \"Noun\"), load the POS+Zipf spreadsheet (Brysbaert,
    New & Keuleers 2012) and keep rows whose dominant CLAWS tag matches.
    Zipf is the official SUBTLEX Zipf-value when POS is used, otherwise
    log10(SUBTLWF)+3. Morphology is unlabeled unless the word is in MorphoLex.
    """
    if pos:
        xlsx = download_subtlex_pos()
        raw = pd.read_excel(xlsx, sheet_name=0)
        raw = raw.rename(columns={"Word": "word", "Zipf-value": "zipf_freq"})
        raw["word"] = raw["word"].astype(str).str.strip().str.lower()
        dom = raw["Dom_PoS_SUBTLEX"].astype(str)
        df = raw.loc[dom.str.fullmatch(pos, case=False)].copy()
        df = df.loc[df["word"].str.fullmatch(r"[a-z]+")].drop_duplicates("word")
        df["zipf_freq"] = pd.to_numeric(df["zipf_freq"], errors="coerce")
        df["note"] = f"subtlex-us {pos.lower()}"
    else:
        path = download_subtlex(path)
        raw = pd.read_csv(path, sep="\t")
        raw = raw.rename(columns={"Word": "word"})
        raw["word"] = raw["word"].astype(str).str.strip().str.lower()
        df = raw.loc[raw["word"].str.fullmatch(r"[a-z]+")].copy()
        df = df.drop_duplicates("word")
        df["zipf_freq"] = df["SUBTLWF"].map(lambda x: math.log10(max(float(x), 1e-12)) + 3.0)
        df["note"] = "subtlex-us"
    df = df.loc[df["zipf_freq"].notna() & (df["zipf_freq"] >= min_zipf)].copy()
    df = df.sort_values(["zipf_freq", "word"], ascending=[False, True])
    df["length"] = df["word"].str.len()
    df["class"] = "unlabeled"
    df["n_morphemes"] = pd.NA

    ml_path = morpholex if morpholex is not None else MORPHOLEX_NOUNS
    if ml_path.exists():
        ml = pd.read_csv(ml_path, usecols=lambda c: c in {"word", "class", "n_morphemes", "note"})
        ml["word"] = ml["word"].str.strip().str.lower()
        ml = ml.drop_duplicates("word")
        df = df.merge(ml, on="word", how="left", suffixes=("", "_ml"))
        df["class"] = df["class_ml"].fillna(df["class"])
        df["n_morphemes"] = df["n_morphemes_ml"].combine_first(df["n_morphemes"])
        if "note_ml" in df.columns:
            df["note"] = df["note_ml"].fillna(df["note"])
        drop = [c for c in df.columns if c.endswith("_ml")]
        df = df.drop(columns=drop)

    df["n_morphemes"] = pd.to_numeric(df["n_morphemes"], errors="coerce")
    df["is_monomorph"] = pd.Series(pd.NA, index=df.index, dtype="boolean")
    known = df["n_morphemes"].notna()
    df.loc[known, "is_monomorph"] = df.loc[known, "n_morphemes"] == 1
    df["log_freq"] = df["zipf_freq"]
    df = attach_subtlex_pos(df)
    keep = [
        "word",
        "zipf_freq",
        "length",
        "class",
        "pos",
        "n_morphemes",
        "is_monomorph",
        "note",
        "log_freq",
    ]
    return df[keep].reset_index(drop=True)
