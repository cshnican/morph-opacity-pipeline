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
opacity(w) = 1 − cos( g(form_w), v_w )     # g trained without w
```

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
```

Outputs:

- tables in `outputs/`
- figures in `figures/`

| file | what |
|---|---|
| `outputs/{tag}_scores.csv` | per-word cosine / opacity, frequency, n_morphemes, class |
| `outputs/{tag}_by_class.csv` | mean opacity by morphological class |
| `outputs/{tag}_regression.csv` | OLS / logit coefficients |
| `figures/{tag}_opacity_vs_freq.png` | scatter |
| `figures/{tag}_opacity_by_class.png` | boxplot |

`demo` plants the pattern (transparent multimorphs = sum of morpheme vectors;
monomorphs and lexicalized multimorphs = random meanings; frequency higher
for opaque items). The residuals recover that ranking — planted transparent
words get much lower opacity than monomorphs / lexicalized multimorphs —
which checks the measurement before you trust it on English.

`glove` uses `data/sample_nouns.csv` (hand-classed English nouns) plus
[wordfreq](https://github.com/rspeer/wordfreq) Zipf frequencies and GloVe
wiki-gigaword 50d meaning vectors (downloaded once, then cached under
`data/cache/`).

On the first English pass, **Stage 1 holds** (more frequent nouns are more
often monomorphemic) but the residual-opacity slope goes the other way:
frequent words look *more* form-predictable. That is the vector-reliability
/ length confound discussed in the design — GloVe vectors of rare long
scientific words are noisier, and a held-out char-ngram model also has
little evidence for their combining forms. Treat the English opacity slope
as a measurement diagnostic, not a test of the theory, until split-half
vector stability and a morpheme-aware split are in place.

## The two-stage (hurdle) analysis

Matching the two sources of opacity:

- **Stage 1 (logit):** `P(monomorphemic) ~ frequency`
  — frequent nouns should be more likely to be unimorphemic.
- **Stage 2 (OLS, multimorphs only):** `opacity ~ frequency + length`
  — among words that *have* parts, frequent ones should be more opaque.
- **Overall OLS:** `opacity ~ frequency` and `opacity ~ frequency + length`
  — length is a *mediator* (frequent → short → fewer morphemes), so both
  specs are reported.

The coefficient of interest is `zipf_freq`: positive means more frequent →
more opaque, which is the downstream prediction of the form–meaning cost model.

## What this is not

- Not a morpheme parser. `n_morphemes` / `class` on the English sample are
  coarse hand labels for slicing the results, not inputs to `g`.
- Not an LLM-as-judge transparency rating (easy to add later; validate
  against LADEC / Libben norms if you do).
- Vector reliability is not split-half estimated (GloVe is a single dump).
  Frequency itself is a rough reliability proxy; treat the English slope as
  suggestive until you control split-half stability and polysemy.

## Layout

```
opacity/lexicon.py     sample noun list + Zipf frequencies
opacity/vectors.py     GloVe download / cache / lookup
opacity/synthetic.py   planted-opacity lexicon
opacity/model.py       character n-gram → Ridge → meaning
opacity/scores.py      K-fold held-out residuals
opacity/analyze.py     hurdle + overall regressions
opacity/plot.py        figures
run_pipeline.py        CLI
```
