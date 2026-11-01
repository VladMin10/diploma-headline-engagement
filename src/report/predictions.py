"""Прогнозовані ймовірності для практичного шару звітів.

Запуск:
    .venv/bin/python -m src.report.predictions [--force]

Новим експериментом це не є: ті самі моделі, ті самі фолди і той самий підбір
гіперпараметрів, що в EXP-007 і EXP-008. Різниця в тому, що тут зберігаються
самі ймовірності, а не лише метрики, бо влучність на верхніх k відсотках,
приклади заголовків і ROC-криві рахуються з ймовірностей.

Наявні експерименти не чіпаються: їхні файли тільки читаються.

П'ять варіантів. Чотири моделі на наборі F₀ потрібні для ROC-кривих, яких
прямо вимагає розділ 8 документа кафедри. П'ятий, логістична регресія
на TF-IDF, потрібен для того, щоб влучність семи ознак було з чим порівняти.
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from src.experiments.ablation_f0_f3 import PLATFORMS, build_model, tune
from src.experiments.text_vs_semantic import C_GRID, TFIDF_PARAMS, best_threshold
from src.features.derived import F0

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features_derived.parquet"
TARGETS_PATH = PROJECT_ROOT / "data" / "processed" / "targets_classification.parquet"
FOLDS_PATH = PROJECT_ROOT / "data" / "processed" / "cv_folds.parquet"
CLEAN_PATH = PROJECT_ROOT / "data" / "interim" / "news_clean.parquet"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "report_predictions.parquet"

RANDOM_STATE = 42
PERCENTILE = "25"
PLATFORM_ORDER = ["Facebook", "GooglePlus", "LinkedIn"]
# (назва варіанта, модель, чи це TF-IDF)
VARIANTS = [
    ("LogisticRegression_F0", "LogisticRegression", False),
    ("RandomForest_F0", "RandomForest", False),
    ("LightGBM_F0", "LightGBM", False),
    ("CatBoost_F0", "CatBoost", False),
    ("LogisticRegression_TFIDF", "LogisticRegression", True),
]
INNER_SPLITS = 4


def tfidf_matrices(train: pd.DataFrame, test: pd.DataFrame):
    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    return (
        vectorizer.fit_transform(train["Title"]),
        vectorizer.transform(test["Title"]),
    )


def tune_tfidf(train: pd.DataFrame, y_train: np.ndarray, groups: np.ndarray) -> float:
    """Підбір C для TF-IDF, як у EXP-008: вектаризатор теж усередині підбору."""
    splitter = StratifiedGroupKFold(
        n_splits=INNER_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    inner, validation = next(
        splitter.split(np.zeros((len(train), 1)), y_train, groups)
    )
    x_inner, x_val = tfidf_matrices(train.iloc[inner], train.iloc[validation])

    best, best_score = C_GRID[0], -1.0
    for value in C_GRID:
        model = LogisticRegression(C=value, max_iter=1000, random_state=RANDOM_STATE)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(x_inner, y_train[inner])
        inner_scores = model.predict_proba(x_inner)[:, 1]
        val_scores = model.predict_proba(x_val)[:, 1]
        threshold = best_threshold(y_train[inner], inner_scores)
        from sklearn.metrics import f1_score

        score = f1_score(
            y_train[validation],
            (val_scores >= threshold).astype(int),
            zero_division=0,
        )
        if score > best_score:
            best, best_score = value, score
    return best


def run_split(train: pd.DataFrame, test: pd.DataFrame, target: str, variant) -> np.ndarray:
    """Ймовірності позитивного класу на перевірочній частині."""
    name, model_name, is_tfidf = variant
    y_train = train[target].to_numpy(dtype=np.int8)
    groups = train["title_norm"].to_numpy()

    if is_tfidf:
        value = tune_tfidf(train, y_train, groups)
        x_train, x_test = tfidf_matrices(train, test)
        model = LogisticRegression(C=value, max_iter=1000, random_state=RANDOM_STATE)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(x_train, y_train)
        return model.predict_proba(x_test)[:, 1]

    x_train = train[F0].to_numpy(dtype=np.float64)
    x_test = test[F0].to_numpy(dtype=np.float64)
    value = tune(model_name, x_train, y_train, groups)
    if model_name == "LogisticRegression":
        scaler = StandardScaler().fit(x_train)
        x_train, x_test = scaler.transform(x_train), scaler.transform(x_test)
    model = build_model(model_name, value)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(x_train, y_train)
    return model.predict_proba(x_test)[:, 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if OUT_PATH.exists() and not args.force:
        raise SystemExit(
            f"{OUT_PATH.relative_to(PROJECT_ROOT)} уже існує; "
            "запустіть з --force, щоб перерахувати."
        )

    features = pd.read_parquet(FEATURES_PATH)
    targets = pd.read_parquet(TARGETS_PATH)
    clean = pd.read_parquet(CLEAN_PATH, columns=["IDLink", "Title", "title_norm"])
    clean["IDLink"] = clean["IDLink"].astype("int64")
    frame = (
        features.merge(targets, on="IDLink", validate="one_to_one")
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
            for repeat, fold in splits:
                current = part[(part["repeat"] == repeat) & (part["fold"] == fold)]
                train_ids = current.loc[current["part"] == "train", "IDLink"].to_numpy()
                test_ids = current.loc[current["part"] == "test", "IDLink"].to_numpy()
                train, test = frame.loc[train_ids], frame.loc[test_ids]
                proba = run_split(train, test, target, variant)
                records.append(
                    pd.DataFrame(
                        {
                            "платформа": platform,
                            "варіант": variant[0],
                            "repeat": np.int8(repeat),
                            "fold": np.int8(fold),
                            "IDLink": test_ids,
                            "y": test[target].to_numpy(dtype=np.int8),
                            "ймовірність": proba,
                        }
                    )
                )
            print(f"  {platform} / {variant[0]} — готово", flush=True)

    predictions = pd.concat(records, ignore_index=True)
    for column in ("платформа", "варіант"):
        predictions[column] = predictions[column].astype("category")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(OUT_PATH, index=False)
    print(f"прогнози: {OUT_PATH.relative_to(PROJECT_ROOT)} — "
          f"{len(predictions)} рядків, {(time.perf_counter() - started) / 60:.1f} хв")


if __name__ == "__main__":
    main()
