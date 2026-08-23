# Copywriting features of a headline as predictors of user engagement — pipeline code

Code for a master's thesis that asks whether *content-level* copywriting
characteristics of a news headline — an information gap, an explicit
conflict, a violated expectation — add predictive power over publication
metadata, and whether the same characteristics work the same way on
different platforms.

**This repository contains source code only.** No data, no annotation
labels, no experiment configurations, no results, no thesis text. Read
the "What is not here" section before trying to reproduce anything: some
stages can be run from scratch, and some cannot.

---

## What the study found

Measured on 5 055 annotated headlines (2 000 train / 3 055 test),
temporal split, bootstrap confidence intervals.

| Question | Answer |
|---|---|
| Is the annotation reliable enough to build on? | Cohen's weighted kappa 0.296–0.789 against a human coder, 0.445–0.805 between two independent runs. Seven of eight features pass the 0.40 threshold fixed before annotation. |
| Do copywriting features add accuracy over metadata? | Facebook +0.0506 [+0.0322; +0.0692] Spearman rho. Within-source: Facebook +0.1120, Google+ +0.0574. |
| Is the contribution independent of formal text statistics? | Facebook +0.0563, Google+ +0.0419. Formal statistics on this corpus do not help and significantly hurt on two platforms out of three. |
| Do the same techniques work across platforms? | Facebook and Google+ agree marginally (Kendall tau 0.7143 against a 0.70 threshold). LinkedIn differs from both: explicit conflict drops from rank 2 to rank 5, scale of consequences rises from 5 to 2, source authority from 7 to 3. |

These numbers are reported here for context. They are **not** reproducible
from this repository alone — see below.

---

## What is not here, and why it matters

| Missing | Why | Consequence |
|---|---|---|
| `data/raw/` | The dataset is not redistributed. | Download it yourself, see below. |
| `annotations/` | Rubric, annotator prompt and 5 055 labels. | The labels were produced by a language model reading headlines against a fixed rubric inside an interactive session, in batches. **There is no script that regenerates them.** |
| `experiments/*/config.json` | One config per experiment. | `src/experiment.py` needs a config file to run. |
| `docs/`, `reports/` | Thesis chapters, figures, summary tables. | `src/build_thesis.py` and `src/build_slides.py` cannot run. |

The annotation is the part that a reader cannot re-derive from code. It
is a procedure, not a function: the rubric and the written instruction
define it, and re-running it with a different model or a different
prompt produces different labels. That is precisely why the thesis
measures the annotation's reliability twice — against a human coder and
against a second independent run — rather than assuming it.

---

## What can be run from this repository

### Requires only the raw dataset

Download "News Popularity in Multiple Social Media Platforms" (UCI ML
Repository, ID 432) by Moniz and Torgo and place `News_Final.csv` at
`data/raw/Dataset/News_Final.csv` relative to the repository root.

```bash
python -m src.data.load        # -> data/interim/news_clean.parquet
python -m src.data.split       # -> data/interim/split.json
python -m src.audit_phase0     # dataset integrity report
python -m src.audit_truncation # truncated-headline confound
python -m src.data.sample pilot   # 400 headlines, seed 42
python -m src.data.sample manual  # 200 headlines, seed 43
```

This reproduces the working sample of 81 634 rows with all three targets
present, the temporal split at 2016-05-17, and two of the three
annotation samples.

### Requires annotation labels in addition

```bash
python -m src.agreement --a ... --b ... --out ...   # human vs model
python -m src.stability                             # run 1 vs run 2
python -m src.build_features                        # feature matrices + leakage check
python -m src.experiment --config experiments/<exp>/config.json
python -m src.compare --question Q1 --model lightgbm ...
python -m src.shap_importance --config ...
python -m src.tables
python -m src.figures_phase5
python -m src.figures_phase6
```

### Requires the full thesis working tree

```bash
python -m src.build_thesis   # assembles the .docx from chapter drafts
python -m src.build_slides   # assembles the defence deck
```

Both read Ukrainian-language drafts and the university template, neither
of which is in this repository.

---

## Environment

Python 3.12.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`python-docx`, `python-pptx` and `lxml` are needed only by the two
document-assembly scripts. The experiment pipeline does not import them.

---

## Layout

```
src/
  data/       loading, typing, the fixed temporal split, sample construction
  features/   three feature levels behind one interface, plus assembly
  models/     constant baseline, ridge, gradient boosting as pipelines
  eval/       metrics and bootstrap confidence intervals
  experiment.py       runs one experiment from a config file
  compare.py          paired-bootstrap gain between two configurations
  agreement.py        agreement between two codings
  stability.py        stability between two annotation runs
  build_features.py   feature matrices and the leakage check
  shap_importance.py  feature importance and cross-platform comparison
  tables.py, figures*.py    summary tables and figures
  build_thesis.py, build_slides.py    document assembly
```

---

## Method notes that explain the code

**Temporal split, not random.** Training is the earlier period.
Duplicate-headline groups that straddle the boundary are dropped from the
test side only. The boundary is fixed once and read from a file so later
stages cannot drift.

**Nothing that learns from data is stored between stages.** Feature
matrices hold only row-wise quantities. One-hot categories and the
TF-IDF vocabulary are fitted inside the fold. `build_features.py`
quantifies why: it reports how many vocabulary terms would enter training
if the vectoriser were fitted outside the fold.

**Paired comparisons are guarded.** `compare.py` refuses to compute a
gain if two runs were evaluated on different test rows; identity is
checked through a SHA-256 digest of the row identifiers stored in each
results file.

**Two metrics, not one.** Overall Spearman rho largely measures how well
the model learned a publication's baseline level — the source feature
alone explains 59.69% of the Facebook target variance. A within-source
correlation is computed alongside, inside each of 50 publications with at
least 30 test rows.

**The comparison model was fixed before the runs, and the data
contradicted its justification.** Gradient boosting was chosen because
the features are ordinal and act non-linearly; it turned out worse than
ridge regression on every configuration and platform. It was not switched
afterwards, and both sets of numbers are reported. `models/zoo.py` still
builds both.

Seeds: 42 everywhere except sample construction, where they are 42
(pilot), 43 (human reference), 44 (annotation queue) and 45 (stability
sample). Library versions are recorded in every experiment config.

---

## Licence

CC BY 4.0, see `LICENSE`.

The underlying dataset is not redistributed. University templates and the
department's submission procedure are not part of this repository.
