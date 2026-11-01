"""EXP-013: базова лінія на метаданих у класифікаційній лінії.

Запуск:
    .venv/bin/python -m src.experiments.metadata_baseline

Протокол — `experiments/EXP-013_metadata-baseline/hypothesis.md`, складений
і закомічений до написання цього коду.

Три набори на тих самих двадцяти розбиттях: M — метадані публікації в складі
рівня T1 регресійної лінії; F₀ — сім змістових характеристик; M+F₀ — разом.
Два порівняння: F₀ проти M і M+F₀ проти M. Друге є прямою відповіддю на Q1
у класифікаційній постановці.

**Умови взяті з EXP-007.** Побудова моделей, сітка гіперпараметрів, сітка
порогів, кількість внутрішніх фолдів і набір метрик імпортуються
з `ablation_f0_f3`. Підбір гіперпараметра і підбір порога повторюють ту саму
логіку, але не можуть бути імпортовані дослівно: тут ознаки кодуються
всередині фолда, і кодувальник мусить навчатися на тій самій частині, що
й модель.

**Кодування, зафіксоване карткою до прогону.** CatBoost і LightGBM отримують
`source` і `topic` як справжні категорії. Логістична регресія і випадковий
ліс отримують one-hot, навчений лише на навчальній частині фолда, з тим самим
`min_frequency=30`, що в `src/features/meta.py`. Цілі коди замість one-hot
не використовуються: вони нав'язали б 50 виданням довільний порядок.

**Статистична перевірка тут не робиться.** Прогін дає матрицю метрик без
усереднення; правило ADR-006 застосовує до неї окремий крок.

**Стійкість до переривання.** Результат кожної комбінації лягає окремим
файлом у `data/interim/metadata_parts/`.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.experiments.ablation_f0_f3 import (
    GRIDS,
    INNER_SPLITS,
    MODELS,
    PLATFORMS,
    RANDOM_STATE,
    best_threshold,
    build_model,
    load_inputs,
    metrics,
)
from src.features.derived import F0

PROJECT_ROOT = Path(__file__).resolve().parents[2]
META_PATH = PROJECT_ROOT / "data" / "processed" / "features_T1.parquet"
PARTS_DIR = PROJECT_ROOT / "data" / "interim" / "metadata_parts"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "metadata_baseline_results.parquet"

PERCENTILE = "25"

# Склад рівня T1 регресійної лінії, узятий без змін.
CATEGORICAL = ["source", "topic"]
TIME = ["hour", "weekday", "month"]
META = CATEGORICAL + TIME

# Той самий поріг рідкості, що в src/features/meta.py регресійної лінії.
SOURCE_MIN_FREQUENCY = 30

VARIANTS = {
    "M": (True, False),
    "F0": (False, True),
    "M+F0": (True, True),
}

# Моделі з власною підтримкою категорій. Решті подається one-hot.
NATIVE_CATEGORICAL = {"LightGBM", "CatBoost"}


def design(model_name: str, variant: str, train: pd.DataFrame,
           evaluate: pd.DataFrame):
    """Матриці ознак для однієї пари частин фолда.

    Кодувальник навчається лише на `train`: перевірочна частина в жодному
    вигляді в нього не потрапляє.
    """
    with_meta, with_features = VARIANTS[variant]

    if not with_meta:
        columns = list(F0)
        return (
            train[columns].to_numpy(dtype=np.float64),
            evaluate[columns].to_numpy(dtype=np.float64),
            [],
        )

    if model_name in NATIVE_CATEGORICAL:
        columns = META + (list(F0) if with_features else [])
        left, right = train[columns].copy(), evaluate[columns].copy()
        for name in CATEGORICAL:
            categories = pd.Index(sorted(left[name].astype(str).unique()))
            left[name] = pd.Categorical(left[name].astype(str), categories=categories)
            right[name] = pd.Categorical(
                right[name].astype(str), categories=categories
            )
            if model_name == "CatBoost":
                # CatBoost працює з рядками і сам обробляє небачені значення.
                left[name] = left[name].astype(str)
                right[name] = right[name].astype(str)
        indices = [columns.index(name) for name in CATEGORICAL]
        return left, right, indices

    encoder = OneHotEncoder(
        min_frequency=SOURCE_MIN_FREQUENCY,
        handle_unknown="infrequent_if_exist",
        sparse_output=False,
        dtype=np.float64,
    ).fit(train[META].astype(str))
    left = encoder.transform(train[META].astype(str))
    right = encoder.transform(evaluate[META].astype(str))
    if with_features:
        left = np.hstack([left, train[list(F0)].to_numpy(dtype=np.float64)])
        right = np.hstack([right, evaluate[list(F0)].to_numpy(dtype=np.float64)])
    return left, right, []


def fit_score(model_name: str, value, x_train, y_train, x_eval,
              cat_indices: list[int]):
    """Навчання і оцінки ймовірності на навчальній та оцінюваній частинах."""
    model = build_model(model_name, value)

    if model_name == "LogisticRegression":
        scaler = StandardScaler().fit(x_train)
        x_train, x_eval = scaler.transform(x_train), scaler.transform(x_eval)

    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        if model_name == "CatBoost" and cat_indices:
            model.fit(x_train, y_train, cat_features=cat_indices)
        else:
            model.fit(x_train, y_train)

    return model.predict_proba(x_train)[:, 1], model.predict_proba(x_eval)[:, 1]


def take(frame, rows: np.ndarray):
    """Рядки за позиціями — однаково для DataFrame і масиву."""
    return frame.iloc[rows] if isinstance(frame, pd.DataFrame) else frame[rows]


def tune(model_name: str, variant: str, train: pd.DataFrame, y_train: np.ndarray,
         groups: np.ndarray):
    """Підбір параметра на внутрішній validation-підвибірці.

    Кодувальник навчається всередині підбору заново, лише на внутрішній
    навчальній частині: інакше в підбір потрапляв би склад категорій
    валідаційної частини.
    """
    _, values = GRIDS[model_name]
    splitter = StratifiedGroupKFold(
        n_splits=INNER_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    inner, validation = next(
        splitter.split(np.zeros(len(train)), y_train, groups)
    )
    inner_frame, validation_frame = train.iloc[inner], train.iloc[validation]
    x_inner, x_validation, cat_indices = design(
        model_name, variant, inner_frame, validation_frame
    )

    best, best_value = values[0], -1.0
    for value in values:
        inner_scores, validation_scores = fit_score(
            model_name, value, x_inner, y_train[inner], x_validation, cat_indices
        )
        threshold = best_threshold(y_train[inner], inner_scores)
        score = f1_score(
            y_train[validation],
            (validation_scores >= threshold).astype(int),
            zero_division=0,
        )
        if score > best_value:
            best, best_value = value, score
    return best


def run_combo(frame: pd.DataFrame, folds: pd.DataFrame, platform: str,
              variant: str, model_name: str) -> pd.DataFrame:
    """Двадцять розбиттів однієї комбінації платформи, набору і моделі."""
    target = f"y_{PLATFORMS[platform]}_{PERCENTILE}"
    indexed = frame.set_index("IDLink")

    part = folds[
        (folds["platform"] == platform) & (folds["percentile"] == PERCENTILE)
    ]
    splits = sorted(
        part[["repeat", "fold"]].drop_duplicates().itertuples(index=False, name=None)
    )

    records = []
    for repeat, fold in splits:
        current = part[(part["repeat"] == repeat) & (part["fold"] == fold)]
        train_ids = current.loc[current["part"] == "train", "IDLink"].to_numpy()
        test_ids = current.loc[current["part"] == "test", "IDLink"].to_numpy()

        train, test = indexed.loc[train_ids], indexed.loc[test_ids]
        y_train = train[target].to_numpy(dtype=np.int8)
        y_test = test[target].to_numpy(dtype=np.int8)
        groups = train["title_norm"].to_numpy()

        value = tune(model_name, variant, train, y_train, groups)
        x_train, x_test, cat_indices = design(model_name, variant, train, test)
        train_scores, test_scores = fit_score(
            model_name, value, x_train, y_train, x_test, cat_indices
        )
        threshold = best_threshold(y_train, train_scores)

        row = {
            "платформа": platform,
            "перцентиль": PERCENTILE,
            "варіант": variant,
            "модель": model_name,
            "repeat": int(repeat),
            "fold": int(fold),
            "параметр": GRIDS[model_name][0],
            "значення_параметра": str(value),
            "поріг": threshold,
            "n_ознак": int(x_train.shape[1]),
            "n_train": int(len(train_ids)),
            "n_test": int(len(test_ids)),
        }
        row.update(metrics(y_test, test_scores, threshold))
        records.append(row)

    return pd.DataFrame(records)


def part_path(platform: str, variant: str, model_name: str) -> Path:
    safe = variant.replace("+", "_plus_")
    return PARTS_DIR / f"{PLATFORMS[platform]}_{safe}_{model_name}.parquet"


def combos() -> list[tuple[str, str, str]]:
    return [
        (platform, variant, model_name)
        for platform in PLATFORMS
        for variant in VARIANTS
        for model_name in MODELS
    ]


def load_with_meta() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ознаки, мітка і заголовки з EXP-007 плюс метадані рівня T1."""
    frame, folds = load_inputs()
    meta = pd.read_parquet(META_PATH, columns=META)
    meta.index = meta.index.astype("int64")
    frame = frame.merge(
        meta, left_on="IDLink", right_index=True, how="left", validate="one_to_one"
    )
    missing = frame[META].isna().any(axis=1).sum()
    if missing:
        raise ValueError(f"метадані відсутні для {missing} анотованих рядків")
    return frame, folds


def main() -> None:
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    frame, folds = load_with_meta()
    todo = [combo for combo in combos() if not part_path(*combo).exists()]
    print(f"комбінацій усього: {len(combos())}, лишилося: {len(todo)}")

    started = time.perf_counter()
    for index, combo in enumerate(todo, start=1):
        combo_started = time.perf_counter()
        result = run_combo(frame, folds, *combo)
        result.to_parquet(part_path(*combo), index=False)
        print(
            f"[{index}/{len(todo)}] {combo} — "
            f"{time.perf_counter() - combo_started:.1f} с",
            flush=True,
        )
    if todo:
        print(f"прогін: {(time.perf_counter() - started) / 60:.1f} хв")

    parts = [pd.read_parquet(part_path(*combo)) for combo in combos()]
    results = pd.concat(parts, ignore_index=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(OUT_PATH, index=False)
    print(f"зведено: {OUT_PATH.relative_to(PROJECT_ROOT)} — {len(results)} рядків")


if __name__ == "__main__":
    main()
