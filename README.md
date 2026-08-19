# Morphological opacity via form→meaning residuals

Per-word **opacity** = how poorly a word's meaning can be predicted from its
written form. That unifies the two sources of morphological opacity:

1. **Monomorphs** (*dog*) — no independently meaningful sub-parts, so the
   form→meaning map has nothing to work with.
2. **Lexicalized multimorphs** (*laptop*) — parts exist at the form level,
   but the meaning is not recoverable from them.

Transparent multimorphs (*electrocardiogram*, *teacher*) sit at the other
end: a learner who has seen *electro-*, *cardio-*, *-gram* in other words
can predict the meaning from the string.

The pipeline **never builds morpheme vectors** and never sums parts. A single
map `g(form) → meaning` is trained on character n-grams. Bound morphemes
(*-ed*, *cardio-*) are just character sequences the model can pick up if they
are predictive across words. 1, 2, or 5 morphemes are all just strings.

Held-out scoring is required. If `g` were trained on the target word itself it
could memorize *laptop* as a whole string and look spuriously transparent.

```
opacity_raw(w) = 1 − cos( g(form_w), v_w )                          # g trained without w
opacity(w)     = 1 − ( cos(g(form_w), v_w) − cos(mean(v_¬w), v_w) ) # form must beat centroid
```

Raw cosine is inflated for frequent GloVe hubs: they sit near the center of the
space, so a shrunk predictor matches them even when spelling is arbitrary
(*life*, *time*). Adjusted opacity asks whether form recovers *this* meaning
beyond a form-blind guess (the fold's training centroid). On GloVe, leading
principal components — which track frequency — are also dropped before scoring
(all-but-the-top; Mu & Viswanath 2018). The planted `demo` lexicon has no hub
geometry, so raw and adjusted rankings agree.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python run_pipeline.py --mode demo    # planted lexicon; no download
python run_pipeline.py --mode glove   # English nouns + GloVe 50d (downloaded once)
python run_pipeline.py --mode both    # default
python run_pipeline.py --mode glove --whiten-d 0   # skip all-but-the-top
python run_pipeline.py --mode glove --lexicon data/sample_nouns_alt.csv --tag glove_alt
python run_pipeline.py --mode glove --lexicon data/morpholex_nouns.csv --tag glove_morpholex
```

Outputs:

- tables in `outputs/`
- figures in `figures/`

| file | what |
|---|---|
| `outputs/{tag}_scores.csv` | per-word cosine, centroid baseline, raw/adjusted opacity, retrieval rank |
| `outputs/{tag}_by_class.csv` | mean opacity by morphological class |
| `outputs/{tag}_regression.csv` | OLS / logit coefficients (adjusted + raw) |
| `figures/{tag}_opacity_vs_freq.png` | adjusted opacity scatter |
| `figures/{tag}_opacity_by_class.png` | adjusted opacity boxplot |
| `figures/{tag}_opacity_raw_vs_freq.png` | raw cosine opacity (diagnostic) |

`demo` plants the pattern (transparent multimorphs = sum of morpheme vectors;
monomorphs and lexicalized multimorphs = random meanings; frequency higher
for opaque items). The residuals recover that ranking — planted transparent
words get much lower opacity than monomorphs / lexicalized multimorphs —
which checks the measurement before you trust it on English.

`glove` uses a noun CSV plus
[wordfreq](https://github.com/rspeer/wordfreq) Zipf frequencies and GloVe
wiki-gigaword 50d meaning vectors (downloaded once, then cached under
`data/cache/`). Default is the hand-classed `data/sample_nouns.csv`.
`data/sample_nouns_alt.csv` is a disjoint hand set. `data/morpholex_nouns.csv`
is built from [MorphoLex-en](https://github.com/hugomailhot/MorphoLex-en)
(Sánchez-Gutiérrez et al. 2018): simplex → monomorph, affixed →
transparent_multi, 2+ roots → opaque_multi (compounds; MorphoLex has no
human transparency ratings).

On English, **Stage 1 holds** (more frequent nouns are more often
monomorphemic). After the centroid adjustment and all-but-the-top,
morphological class ranks as planted: monomorphs most opaque, transparent
multimorphs least, and the frequency slope is positive (including among
multimorphs only). Raw cosine-to-GloVe without those corrections went the
other way because hubs look form-predictable for free. Split-half vector
stability is still not estimated (GloVe is a single dump).

## The two-stage (hurdle) analysis

Matching the two sources of opacity:

- **Stage 1 (logit):** `P(monomorphemic) ~ frequency`
  — frequent nouns should be more likely to be unimorphemic.
- **Stage 2 (OLS, multimorphs only):** `opacity ~ frequency + length`
  — among words that *have* parts, frequent ones should be more opaque.
- **Overall OLS:** `opacity ~ frequency` and `opacity ~ frequency + length`
  — length is a *mediator* (frequent → short → fewer morphemes), so both
  specs are reported. Hubness (mean cosine to 5 nearest neighbors) is added
  as a covariate in extra specs; that is a typicality control, not a
  residualization on frequency itself.

The coefficient of interest is `zipf_freq`: positive means more frequent →
more opaque, which is the downstream prediction of the form–meaning cost model.

## What this is not

- Not a morpheme parser. `n_morphemes` / `class` on the English sample are
  coarse hand labels for slicing the results, not inputs to `g`.
- Not an LLM-as-judge transparency rating (easy to add later; validate
  against LADEC / Libben norms if you do).
- Vector reliability is not split-half estimated (GloVe is a single dump).
  Adjusted opacity and all-but-the-top remove hubness / frequency-PC
  confounds; they do not replace a second independent embedding.

## Layout

```
opacity/lexicon.py     sample noun list + Zipf frequencies
opacity/vectors.py     GloVe download / cache / all-but-the-top / hubness
opacity/synthetic.py   planted-opacity lexicon
opacity/model.py       character n-gram → Ridge → meaning
opacity/scores.py      K-fold held-out residuals
opacity/analyze.py     hurdle + overall regressions
opacity/plot.py        figures
run_pipeline.py        CLI
```
