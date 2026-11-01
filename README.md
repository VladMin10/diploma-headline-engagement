# Content characteristics of a headline as predictors of user engagement — pipeline code

Code for a master's thesis (Lviv Polytechnic National University,
specialty 122 "Computer Science", 2026). The thesis asks whether *content
characteristics* of a news headline — an information gap, an explicit
conflict, a violated expectation and four more — add predictive power over
publication metadata, whether they can be replaced by automatic text
representations, and whether the same characteristics work the same way
on Facebook, Google+ and LinkedIn.

**This repository contains source code only.** No data, no annotation
labels, no experiment configurations, no results, no thesis text. Read
[What is not here](#what-is-not-here) before trying to run anything: some
stages run from the raw dataset alone, the rest need inputs that are not
distributed.

---

## The seven content characteristics

Each headline is rated on an ordinal scale 0–3 by a large language model
following a written rubric. The feature extractor `src/features/copywriting.py`
reads the ratings under these column names:

| No. | Characteristic | Column |
|---|---|---|
| 1 | Information gap | `gap` |
| 2 | Scale of consequences for the reader | `scale` |
| 3 | Concreteness of predicate | `concrete` |
| 4 | Explicitness of conflict | `conflict` |
| 5 | Authority weight | `authority` |
| 7 | Expectation violation | `violation` |
| 9 | Prospective uncertainty | `stakes` |

Numbers follow the rubric and are never reused: candidate 8 was dropped
after the pilot, candidate 6 (specificity of the target audience) after
the reliability check, because its agreement with a human coder was below
the threshold fixed before annotation.

---

## Two task formulations

**Regression.** Target: number of reactions on a platform, `log1p`
transformed. Temporal split at 17 May 2016; duplicate-headline groups that
straddle the boundary are dropped from the test side. Metric: Spearman's
rank correlation, overall and within outlet. Feature levels: T1
publication metadata, T2 the seven ratings, T3 formal text statistics.
Gains are paired bootstrap differences with 95% confidence intervals.

**Classification.** A headline is positive if its reactions exceed the
75th percentile of its own outlet on that platform (20% and 33% as
robustness checks). Grouped stratified cross-validation, 5 folds × 4
repeats, grouped by normalised headline text. Four models: logistic
regression, random forest, LightGBM, CatBoost. Seven metrics, including F1
at 0.5 and at a tuned threshold, ROC-AUC and PR-AUC. A difference counts
as confirmed only when the Wilcoxon signed-rank test, the Nadeau–Bengio
corrected resampled t-test and a bootstrap confidence interval agree.

---

## What each module does

```
src/
  data/
    load.py                  read and clean the raw UCI files, working sample
    split.py                 fix and apply the temporal split
    sample.py                build the pilot, human-reference, annotation
                             and stability samples
    target_classification.py classification label: 75th percentile within
                             outlet, with fallback rules
    target_global.py         auxiliary label with a platform-wide threshold
    cv_protocol.py           grouped stratified folds and their checks
    embeddings.py            sentence-transformers vectors of headlines
  features/
    base.py                  common extractor interface
    meta.py                  T1: publication metadata
    copywriting.py           T2: the seven ratings, and which of them pass
                             the reliability threshold
    text.py                  T3: formal text statistics, TF-IDF inside the fold
    derived.py               derived features of the classification
                             formulation (profile features, interactions)
    assemble.py              feature matrix from chosen levels
  models/zoo.py              constant baseline, ridge, gradient boosting
                             as pipelines (regression)
  eval/
    metrics.py               Spearman (overall and within outlet), MAE in
                             log scale, ROC-AUC for non-zero reactions
    bootstrap.py             bootstrap confidence intervals and paired
                             differences
  experiment.py              run one regression experiment from a config file
  compare.py                 paired-bootstrap gain between two configurations
  shap_importance.py         SHAP importance by platform, Kendall's tau
  agreement.py               weighted Cohen's kappa between two codings
  stability.py               stability between two annotation runs
  build_features.py          feature matrices and the leakage check
  audit_phase0.py, audit_truncation.py, audit_pilot.py, audit_t3_frequency.py
                             data audits
  tables.py, figures*.py     summary tables and figures (regression)
  experiments/               classification experiments and their statistics
    ablation_f0_f3.py, ablation_stats.py, ablation_report.py, probe_f0_f3.py
    text_vs_semantic.py, text_vs_semantic_stats.py, text_vs_semantic_report.py
    embeddings_check.py, embeddings_stats.py
    platform_comparison.py, global_threshold_check.py
    interaction_form.py, interaction_form_stats.py
    fuzzy_strength.py, fuzzy_strength_stats.py
    metadata_baseline.py, metadata_baseline_stats.py
  report/
    predictions.py           predicted probabilities for the reports below
    practical_lift.py        precision at top k percent against the base rate
    descriptive_rates.py     top-quarter share by rating level
    examples.py              headlines the model rates strong and weak
    kafedra_summary.py       summary over the department's experiment list
    figures_classification.py  figures of the classification formulation
```

Code comments and printed reports are in Ukrainian, the language of the
thesis. Comments refer to decision records (`ADR-…`), experiment cards
and `docs/…` files of the thesis working tree; these are not part of this
repository.

---

## Experiments

| ID | Question | Code |
|---|---|---|
| EXP-001 | Do publication metadata predict engagement better than a constant? | `src.experiment` |
| EXP-002 | Is the annotation reliable: agreement with a human coder and stability between two runs? | `src.agreement`, `src.stability` |
| EXP-003 | Baseline on the annotated subsample | `src.experiment` |
| EXP-004 | Do the seven characteristics add accuracy over metadata? (regression) | `src.experiment`, `src.compare` |
| EXP-005b | Do formal text statistics add accuracy over metadata? (control) | `src.experiment`, `src.compare` |
| EXP-005 | Do the characteristics add accuracy over metadata and formal statistics? | `src.experiment`, `src.compare` |
| EXP-006 | Do the same characteristics matter on the three platforms? (regression) | `src.shap_importance` |
| EXP-007 | Does derived feature engineering improve F1 over the seven ratings? | `src.experiments.ablation_f0_f3`, `.ablation_stats`, `.ablation_report`, `.probe_f0_f3` |
| EXP-008 | Can text vectors (TF-IDF, sentence embeddings) replace the seven ratings? | `src.experiments.text_vs_semantic`, `.text_vs_semantic_stats`, `.text_vs_semantic_report`; `src.data.embeddings`, `src.experiments.embeddings_check`, `.embeddings_stats` |
| EXP-009 | Does the platform ranking hold in the classification formulation? | `src.experiments.platform_comparison` |
| EXP-010 | Does a platform-wide label threshold explain the differences between formulations? | `src.data.target_global`, `src.experiments.global_threshold_check` |
| EXP-011 | Pairwise interactions: product versus minimum | `src.experiments.interaction_form`, `.interaction_form_stats` |
| EXP-012 | Fuzzy representation of rating strength | `src.experiments.fuzzy_strength`, `.fuzzy_strength_stats` |
| EXP-013 | Do the seven ratings add accuracy over metadata? (classification) | `src.experiments.metadata_baseline`, `.metadata_baseline_stats` |

EXP-001 to EXP-006 take their parameters from a `config.json`; EXP-007 to
EXP-013 keep them in the scripts.

---

## What is not here

| Missing | What it is | Consequence |
|---|---|---|
| `data/raw/` | The UCI dataset. | Download it yourself, see [Data](#data). |
| `annotations/` | Rubric, annotator instruction, ratings of 5,055 headlines, reliability samples. | Everything that reads T2 cannot run: feature matrices with T2, EXP-002 to EXP-013. |
| `experiments/` | Experiment cards, `config.json` files, reliability results. | `src.experiment` and `src.shap_importance` need a config file; `src/features/copywriting.py` reads the reliability results to decide which ratings enter T2. |
| `reports/` | Result tables and figures. | Scripts recreate them when they run. |

The ratings were produced by a language model reading headlines against
a fixed rubric inside an interactive session, in batches. There is no
script that regenerates them: the rubric and the written instruction
define the procedure, and re-running it with a different model or prompt
gives different ratings. That is why the thesis measures their
reliability twice, against a human coder and between two independent
runs, rather than assuming it.

---

## Data

"News Popularity in Multiple Social Media Platforms", UCI Machine Learning
Repository, ID 432, by Nuno Moniz and Luís Torgo, licensed under
CC BY 4.0:
<https://archive.ics.uci.edu/dataset/432/news+popularity+in+multiple+social+media+platforms>.
It is not redistributed here. Download the archive and place its `Dataset`
folder at `data/raw/Dataset/`, so that `data/raw/Dataset/News_Final.csv`
exists.

---

## How to run

Python 3.12.13; versions are pinned in `requirements.txt`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Requires only the raw dataset

```bash
python -m src.data.load           # working sample -> data/interim/news_clean.parquet
python -m src.data.split          # temporal split -> data/interim/split.json
python -m src.audit_phase0        # dataset integrity report
python -m src.audit_truncation    # truncated-headline confound
python -m src.data.sample pilot   # 400 headlines, seed 42
python -m src.data.sample manual  # 200 headlines, seed 43
```

### Requires annotation inputs in addition

Reliability (EXP-002):

```bash
python -m src.agreement --a <human coding>.csv --b <model coding>.csv \
    --out experiments/EXP-002_annotation-agreement/
python -m src.stability
```

Features and regression experiments:

```bash
python -m src.build_features      # T1, T2, T3 matrices and the leakage check
python -m src.experiment --config experiments/<EXP>/config.json
python -m src.compare --question Q1 --model lightgbm \
    --a experiments/<with T2>/results.json --variant-a T1_T2 \
    --b experiments/<without T2>/results.json --variant-b T1 \
    --out experiments/<EXP>/delta_lightgbm.json
python -m src.shap_importance --config experiments/<EXP-006>/config.json
python -m src.tables
python -m src.figures_phase5
python -m src.figures_phase6
```

Classification:

```bash
python -m src.data.target_classification   # label
python -m src.data.cv_protocol             # 20 grouped splits
python -m src.experiments.ablation_f0_f3   # then the other modules of the
                                           # Experiments table in order
python -m src.report.predictions
python -m src.report.practical_lift
```

The sentence-embedding comparison needs the last block of
`requirements.txt` and downloads `sentence-transformers/all-MiniLM-L6-v2`
on first use.

Random seed 42 everywhere except sample construction (42 pilot, 43 human
reference, 44 annotation queue, 45 stability sample).

---

## Licence

CC BY 4.0, see `LICENSE`. The dataset keeps its own CC BY 4.0 licence and
attribution to its authors.
