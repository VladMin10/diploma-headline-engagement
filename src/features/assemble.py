"""Складання матриці ознак із заданих рівнів (docs/architecture.md розділ 4).

Модуль не знає, що всередині кожного рівня. Він знає лише те, що всі
екстрактори мають однаковий інтерфейс, тому будь-яку їх комбінацію можна
скласти в один перетворювач. Різниця між EXP-003, EXP-004 і EXP-005
зводиться до переліку рівнів, поданого сюди.

Об'єднання зібране як `FeatureUnion`, тобто лишається частиною
`sklearn.Pipeline`. Це означає, що `fit` кожного рівня викликається
всередині фолда, а не на всіх даних.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.pipeline import FeatureUnion

from src.features.base import FeatureExtractor
from src.features.copywriting import LABELS, CopywritingFeatures, load_labels
from src.features.meta import MetaFeatures
from src.features.text import TextFeatures

LEVELS = ("T1", "T2", "T3")

BUILDERS = {
    "T1": MetaFeatures,
    "T2": CopywritingFeatures,
    "T3": TextFeatures,
}


def build_extractors(
    levels: list[str], options: dict | None = None
) -> list[tuple[str, FeatureExtractor]]:
    """Екстрактори для перелічених рівнів, у порядку T1, T2, T3."""
    options = options or {}
    unknown = set(levels) - set(LEVELS)
    if unknown:
        raise ValueError(f"невідомі рівні ознак: {sorted(unknown)}")
    return [
        (level, BUILDERS[level](**options.get(level, {})))
        for level in LEVELS
        if level in levels
    ]


def build_feature_union(levels: list[str], options: dict | None = None) -> FeatureUnion:
    return FeatureUnion(build_extractors(levels, options))


def annotated_ids(labels_path: Path = LABELS) -> pd.Index:
    return pd.Index(load_labels(labels_path).index, name="IDLink")


def select_rows(frame: pd.DataFrame, levels: list[str]) -> pd.DataFrame:
    """Рядки, придатні для заданої комбінації рівнів.

    Рівні T1 і T3 доступні для всіх 81 634 новин, рівень T2 — лише для
    5 055 анотованих. Порівняння моделей має сенс лише на однаковій
    множині рядків, тому щойно в комбінацію входить T2, вся комбінація
    звужується до анотованих новин. Звуження виконується тут, а не
    всередині пайплайна: це відбір даних, а не перетворення ознак.

    Порядок операцій має значення. Спершу `apply_split` з `src.data.split`,
    потім звуження. `apply_split` вилучає з тесту групи повторюваних
    заголовків, представлені по обидва боки межі, і шукає їх серед усіх
    новин. Якщо спершу звузити дані до анотованих, частина таких груп
    стане невидимою, бо їхній навчальний бік у вибірку не потрапив.
    """
    if "T2" not in levels:
        return frame
    return frame[frame["IDLink"].astype("int64").isin(annotated_ids())].copy()
