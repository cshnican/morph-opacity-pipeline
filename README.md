# Morphological transparency via form→meaning residuals

Per-word **transparency** = how well a word's meaning can be predicted from
its written form, relative to a form-blind baseline. Low transparency covers
both monomorphemic words (*dog*) and lexicalized multimorphs (*laptop*): in
neither case is the meaning recoverable from the string. Transparent
multimorphs (*electrocardiogram*, *teacher*) sit at the other end.

The pipeline **never builds morpheme vectors** and never sums parts. A single
map `g(form) → meaning` is trained on character n-grams. Bound morphemes
(*-ed*, *cardio-*) are just character sequences the model can pick up if they
are predictive across words.

Held-out scoring is required. If `g` were trained on the target word itself it
could memorize *laptop* as a whole string and look spuriously transparent.

```
transparency(w) = cos(g(form_w), v_w) − cos(mean(v_¬w), v_w)
```

`g` is trained without `w`. The centroid term asks whether form recovers
*this* meaning beyond a form-blind guess (the fold's training mean). On GloVe,
leading principal components — which track frequency — are also dropped before
scoring (all-but-the-top; Mu & Viswanath 2018).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python run_pipeline.py --mode morpholex # MorphoLex ∩ GloVe 50d (default)
python run_pipeline.py --mode morpholex --whiten-d 0
python run_pipeline.py --mode subtlex   # SUBTLEX ∩ GloVe, subsample 30k (seed=0)
python run_pipeline.py --mode subtlex-gr  # SUBTLEX-GR ∩ Greek GloVe 300d
python run_pipeline.py --mode ladec     # LADEC compounds ∩ GloVe
python run_pipeline.py --mode all       # all four lexicons + English/Greek figures
python score_word.py dog --lexicon data/morpholex_words.csv
python score_word.py dog --lexicon data/subtlex_glove.csv
```

Outputs:

- tables in `outputs/`
- figures in `figures/`

| file | what |
|---|---|
| `outputs/{tag}_scores.csv` | per-word cosine, centroid baseline, transparency, retrieval rank |
| `outputs/{tag}_regression.csv` | OLS coefficients (`transparency ~ frequency` and covariates) |
| `figures/{tag}_transparency_vs_freq.png` | transparency vs Zipf |
| `figures/transparency_vs_freq_english.png` | English (LADEC, MorphoLex, SUBTLEX-US) |
| `figures/transparency_vs_freq_greek.png` | Greek (SUBTLEX-GR) |

`morpholex` uses all alphabetic types from [MorphoLex-en](https://github.com/hugomailhot/MorphoLex-en)
(Sánchez-Gutiérrez et al. 2018) plus
[wordfreq](https://github.com/rspeer/wordfreq) Zipf frequencies and GloVe
wiki-gigaword 50d meaning vectors (downloaded once, then cached under
`data/cache/`). Inflected forms are dropped when the stem is also in the
list. After the GloVe intersect the list is randomly subsampled to 30,000
(`--seed`) so rank and hubness fit in memory. `data/morpholex_words.csv` is
the full working list.

`subtlex` trains `g` on [SUBTLEX-US](https://www.ugent.be/pp/experimentele-psychologie/en/research/documents/subtlexus)
(Brysbaert & New 2009): alphabetic types, intersected with GloVe so every
training row has a meaning vector, then randomly subsampled to 30,000
(`--seed`, written to `data/subtlex_glove.json`) so rank and hubness fit in
memory (~48 GB; n×n float32). Zipf is `log10(SUBTLWF)+3`. The subsampled list
is written to `data/subtlex_glove.csv` for `score_word.py`.

`subtlex-gr` is the same pipeline on Modern Greek: [SUBTLEX-GR](https://www.bcbl.eu/databases/subtlex-gr/)
(Dimitropoulou, Duñabeitia, Avilés, Corral & Carreiras 2010) types
intersected with [Greek GloVe](https://huggingface.co/DFKI/glove-el-cc100)
(CC100 300d; Gurgurov, Korencič & Fischer 2024), then subsampled to 30,000.
Zipf is `log10(SUBTLEX_WF)+3`. The first run streams the ~9 GB embedding file
and caches the intersection under `data/cache/glove_el_subtlex.npz`.

`ladec` trains on [LADEC](https://doi.org/10.7939/r3-dyqx-9b36) closed compounds
(Gagné, Spalding & Schmidtke 2019; `correctParse=yes`, letters only, ∩ GloVe).
Every item is a two-part `c1+c2` concatenation; they are scored as one group.
Zipf is LADEC's native SUBTLEX `Zipfvalue`; items without that value are dropped.

The coefficient of interest is `zipf_freq` in `transparency ~ frequency`
(with and without a hubness covariate). Negative means more frequent → less
transparent. Hubness (mean cosine to 5 nearest neighbors) is a typicality
control, not a residualization on frequency itself.

## What this is not

- Not a morpheme parser. Character n-grams are the only form features.
- Not an LLM-as-judge transparency rating (easy to add later; validate
  against LADEC / Libben norms if you do).
- Vector reliability is not split-half estimated (GloVe is a single dump).
  The centroid baseline and all-but-the-top remove hubness / frequency-PC
  confounds; they do not replace a second independent embedding.

## Layout

```
opacity/lexicon.py     MorphoLex / SUBTLEX-US / LADEC lists + Zipf frequencies
opacity/vectors.py     GloVe download / cache / all-but-the-top / hubness
opacity/model.py       character n-gram → Ridge → meaning
opacity/scores.py      K-fold held-out residuals
opacity/analyze.py     frequency regressions
opacity/plot.py        figures
run_pipeline.py        CLI
```
