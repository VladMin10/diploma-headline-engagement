"""EXP-011: спосіб формування парних взаємодій (E6).

Запуск:
    .venv/bin/python -m src.experiments.interaction_form

Протокол — `experiments/EXP-011_interaction-form/hypothesis.md`, складений
і закомічений до написання цього коду.

Порівнюються два варіанти набору F₂, що відрізняються лише формулою
взаємодії: `prod_a_b = a · b` проти `min_a_b = min(a, b)`. Сім базових оцінок
і склад п'яти пар в обох варіантах однакові.

Сітка: три платформи × два варіанти × чотири моделі × двадцять розбиттів.
Разом 480 навчань підсумкових моделей і втричі більше на підбір
гіперпараметрів. Пороги мітки 20% і 33% не прогоняються — підстава
в розділі 3 картки.

**Умови взяті з EXP-007, а не переписані.** Побудова моделей, підбір
гіперпараметрів, підбір порога і набір метрик імпортуються
з `ablation_f0_f3`: так однаковість умов гарантується кодом, а не звіркою
двох копій.

**Статистична перевірка тут не робиться.** Прогін дає матрицю метрик без
усереднення; правило ADR-006 застосовує до неї окремий крок.

**Стійкість до переривання.** Результат кожної комбінації лягає окремим
файлом у `data/interim/interaction_parts/`; повторний запуск пропускає
пораховане.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.ablation_f0_f3 import (
    GRIDS,
    MODELS,
    PLATFORMS,
    best_threshold,
    fit_score,
    load_inputs,
    metrics,
    tune,
)
from src.features.derived import F0, INTERACTION, INTERACTION_MIN

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARTS_DIR = PROJECT_ROOT / "data" / "interim" / "interaction_parts"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "interaction_form_results.parquet"

PERCENTILE = "25"

# Два варіанти одного набору. Базові оцінки й склад пар однакові,
# відрізняється лише формула взаємодії.
VARIANTS = {
    "product": F0 + INTERACTION,
    "min": F0 + INTERACTION_MIN,
}


def run_combo(
    frame: pd.DataFrame,
    folds: pd.DataFrame,
    platform: str,
    variant: str,
    model_name: str,
) -> pd.DataFrame:
    """Двадцять розбиттів однієї комбінації платформи, варіанта і моделі."""
    target = f"y_{PLATFORMS[platform]}_{PERCENTILE}"
    columns = VARIANTS[variant]
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
        x_train = train[columns].to_numpy(dtype=np.float64)
        x_test = test[columns].to_numpy(dtype=np.float64)
        y_train = train[target].to_numpy(dtype=np.int8)
        y_test = test[target].to_numpy(dtype=np.int8)
        groups = train["title_norm"].to_numpy()

        value = tune(model_name, x_train, y_train, groups)
        train_scores, test_scores = fit_score(
            model_name, value, x_train, y_train, x_test
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
            "n_ознак": len(columns),
            "n_train": int(len(train_ids)),
            "n_test": int(len(test_ids)),
        }
        row.update(metrics(y_test, test_scores, threshold))
        records.append(row)

    return pd.DataFrame(records)


def part_path(platform: str, variant: str, model_name: str) -> Path:
    return PARTS_DIR / f"{PLATFORMS[platform]}_{variant}_{model_name}.parquet"


def combos() -> list[tuple[str, str, str]]:
    return [
        (platform, variant, model_name)
        for platform in PLATFORMS
        for variant in VARIANTS
        for model_name in MODELS
    ]


def main() -> None:
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    frame, folds = load_inputs()
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
