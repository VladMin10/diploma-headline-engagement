"""Розширення E5: sentence embeddings проти семи змістових ознак.

Запуск:
    .venv/bin/python -m src.experiments.embeddings_check

Протокол — підрозділ «Розширення E5» у картці
`experiments/EXP-008_text-vs-semantic/hypothesis.md`, записаний до цього коду.

Два нові варіанти представлення: самі embeddings (384 виміри) і embeddings
разом із сімома змістовими оцінками (391). Уже пораховані F₀ і TF-IDF
беруться з наявних результатів і не перераховуються.

Ті самі двадцять розбиттів, основний поріг мітки, логістична регресія,
підбір C на внутрішній validation-підвибірці. Масштабування вчиться лише
на навчальній частині фолда — embeddings є щільними ознаками, і правило
для них те саме, що для решти щільних.

Векторизація виконана поза фолдом (`src/data/embeddings.py`) і витоку
не створює: модель попередньо навчена і на цих даних не донавчається.

**Відтворення з нуля — два кроки в такому порядку:**

    .venv/bin/python -m src.data.embeddings          # приблизно 43 с
    .venv/bin/python -m src.experiments.embeddings_check   # приблизно 2 хв

Перший крок обов'язковий: `data/processed/embeddings.parquet` не лежить
у версіонуванні. Він важить 10,6 МБ, є похідним артефактом, а не джерелом,
і відтворюється детерміновано — модель зафіксована, донавчання немає.
Метадані прогону векторизації зберігаються поруч
у `embeddings_meta.json`, і цей файл версіонується.

Крім семи метрик рахується влучність на верхніх 10, 20 і 25 відсотках,
щоб нові варіанти можна було поставити поруч із таблицею `practical_lift.md`.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from src.experiments.ablation_f0_f3 import PLATFORMS
from src.experiments.text_vs_semantic import C_GRID, THRESHOLD_GRID, best_threshold
from src.features.derived import F0

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features_derived.parquet"
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "processed" / "embeddings.parquet"
TARGETS_PATH = PROJECT_ROOT / "data" / "processed" / "targets_classification.parquet"
FOLDS_PATH = PROJECT_ROOT / "data" / "processed" / "cv_folds.parquet"
CLEAN_PATH = PROJECT_ROOT / "data" / "interim" / "news_clean.parquet"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "embeddings_results.parquet"

RANDOM_STATE = 42
PERCENTILE = "25"
PLATFORM_ORDER = ["Facebook", "GooglePlus", "LinkedIn"]
INNER_SPLITS = 4
SHARES = (0.10, 0.20, 0.25)
VARIANTS = ("EMB", "EMB+F0")


def columns_for(variant: str, embedding_columns: list[str]) -> list[str]:
    if variant == "EMB":
        return embedding_columns
    if variant == "EMB+F0":
        return embedding_columns + list(F0)
    raise ValueError(variant)


def fit_scores(value: float, x_train, y_train, x_eval):
    scaler = StandardScaler().fit(x_train)
    model = LogisticRegression(C=value, max_iter=1000, random_state=RANDOM_STATE)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(scaler.transform(x_train), y_train)
    return (
        model.predict_proba(scaler.transform(x_train))[:, 1],
        model.predict_proba(scaler.transform(x_eval))[:, 1],
    )


def tune(x_train: np.ndarray, y_train: np.ndarray, groups: np.ndarray) -> float:
    """Підбір C на внутрішній validation-підвибірці з групуванням."""
    splitter = StratifiedGroupKFold(
        n_splits=INNER_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    inner, validation = next(splitter.split(x_train, y_train, groups))
    best, best_score = C_GRID[0], -1.0
    for value in C_GRID:
        inner_scores, val_scores = fit_scores(
            value, x_train[inner], y_train[inner], x_train[validation]
        )
        threshold = best_threshold(y_train[inner], inner_scores)
        score = f1_score(
            y_train[validation],
            (val_scores >= threshold).astype(int),
            zero_division=0,
        )
        if score > best_score:
            best, best_score = value, score
    return best


def precision_at_k(y: np.ndarray, proba: np.ndarray, share: float) -> float:
    k = max(1, int(round(len(y) * share)))
    order = np.argsort(-proba, kind="stable")
    return float(y[order[:k]].mean())


def metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    tuned = (scores >= threshold).astype(int)
    half = (scores >= 0.5).astype(int)
    row = {
        "F1_підібраний": f1_score(y_true, tuned, zero_division=0),
        "F1_05": f1_score(y_true, half, zero_division=0),
        "Accuracy": accuracy_score(y_true, tuned),
        "Precision": precision_score(y_true, tuned, zero_division=0),
        "Recall": recall_score(y_true, tuned, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, scores),
        "PR_AUC": average_precision_score(y_true, scores),
    }
    for share in SHARES:
        row[f"precision@{int(share * 100)}"] = precision_at_k(y_true, scores, share)
    return row


def main() -> None:
    features = pd.read_parquet(FEATURES_PATH)
    embeddings = pd.read_parquet(EMBEDDINGS_PATH)
    targets = pd.read_parquet(TARGETS_PATH)
    clean = pd.read_parquet(CLEAN_PATH, columns=["IDLink", "title_norm"])
    clean["IDLink"] = clean["IDLink"].astype("int64")

    embedding_columns = [c for c in embeddings.columns if c != "IDLink"]
    frame = (
        features.merge(embeddings, on="IDLink", validate="one_to_one")
        .merge(targets, on="IDLink", validate="one_to_one")
        .merge(clean, on="IDLink", validate="one_to_one")
        .set_index("IDLink")
    )
    folds = pd.read_parquet(FOLDS_PATH)

    started = time.perf_counter()
    records = []
    for platform in PLATFORM_ORDER:
        target = f"y_{PLATFORMS[platform]}_{PERCENTILE}"
        part = folds[
            (folds["platform"] == platform) & (folds["percentile"] == PERCENTILE)
        ]
        splits = sorted(
            part[["repeat", "fold"]].drop_duplicates().itertuples(index=False, name=None)
        )
        for variant in VARIANTS:
            columns = columns_for(variant, embedding_columns)
            for repeat, fold in splits:
                current = part[(part["repeat"] == repeat) & (part["fold"] == fold)]
                train_ids = current.loc[current["part"] == "train", "IDLink"].to_numpy()
                test_ids = current.loc[current["part"] == "test", "IDLink"].to_numpy()
                train, test = frame.loc[train_ids], frame.loc[test_ids]

                x_train = train[columns].to_numpy(dtype=np.float64)
                x_test = test[columns].to_numpy(dtype=np.float64)
                y_train = train[target].to_numpy(dtype=np.int8)
                y_test = test[target].to_numpy(dtype=np.int8)
                groups = train["title_norm"].to_numpy()

                value = tune(x_train, y_train, groups)
                train_scores, test_scores = fit_scores(value, x_train, y_train, x_test)
                threshold = best_threshold(y_train, train_scores)

                row = {
                    "платформа": platform,
                    "варіант": variant,
                    "модель": "LogisticRegression",
                    "repeat": int(repeat),
                    "fold": int(fold),
                    "C": value,
                    "поріг": threshold,
                    "n_ознак": len(columns),
                    "базова_частка": float(y_test.mean()),
                }
                row.update(metrics(y_test, test_scores, threshold))
                records.append(row)
            print(f"  {platform} / {variant} — готово", flush=True)

    results = pd.DataFrame(records)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(OUT_PATH, index=False)
    print(f"прогін: {(time.perf_counter() - started) / 60:.1f} хв")
    print(f"результати: {OUT_PATH.relative_to(PROJECT_ROOT)} — {len(results)} рядків")


if __name__ == "__main__":
    main()
