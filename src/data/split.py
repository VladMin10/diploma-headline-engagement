"""Фіксація та застосування часового спліту (ADR-002).

Межа спліту узгоджена з користувачем 6 серпня 2026 року і після запису
у data/interim/split.json не змінюється до кінця роботи (CLAUDE.md розділ 8).

Схема має дві частини. Перша — поділ за датою публікації: рядки, датовані
раніше за межу, утворюють train, решта — test. Друга — обробка груп
повторюваних заголовків. Одну й ту саму новину публікують різні джерела,
і 22,09% рядків набору належать до таких груп. Якщо група перетинає межу,
її члени з тестового боку видаляються з test: копірайтингові ознаки для
однакового тексту збігаються цілком, тож модель отримала б у тесті
підказку, яку вже бачила в навчанні.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLIT_PATH = PROJECT_ROOT / "data" / "interim" / "split.json"

# Узгоджено з користувачем 6 серпня 2026 року: приблизно 80% рядків
# у train. Межа належить тестовій частині: train — строго раніше.
BOUNDARY = "2016-05-17 00:00:00"


@dataclass(frozen=True)
class Split:
    boundary: pd.Timestamp
    train_mask: np.ndarray
    test_mask: np.ndarray
    dropped_mask: np.ndarray  # рядки тестового періоду, вилучені як повтори


def compute_split(df: pd.DataFrame, boundary: str = BOUNDARY) -> Split:
    """Обчислює маски train / test за датою публікації та повторами.

    Вимагає колонок PublishDate і title_norm (див. src/data/load.py).
    Функція детермінована: за тими самими даними і межею дає той самий
    результат, тому перелік вилучених рядків у split.json не зберігається.
    """
    ts = pd.Timestamp(boundary)
    is_train = (df["PublishDate"] < ts).to_numpy()
    is_after = ~is_train

    # Групи повторюваних заголовків, представлені по обидва боки межі.
    titles_train = set(df.loc[is_train, "title_norm"])
    crosses = df["title_norm"].isin(titles_train).to_numpy() & is_after

    return Split(
        boundary=ts,
        train_mask=is_train,
        test_mask=is_after & ~crosses,
        dropped_mask=crosses,
    )


def split_summary(df: pd.DataFrame, split: Split) -> dict:
    """Числа, що описують спліт. Ідуть у split.json і в картку даних."""
    n = len(df)
    n_train = int(split.train_mask.sum())
    n_test = int(split.test_mask.sum())
    n_dropped = int(split.dropped_mask.sum())

    train_dates = df.loc[split.train_mask, "PublishDate"]
    test_dates = df.loc[split.test_mask, "PublishDate"]

    # Скільки груп повторів фактично перетинає межу — критерій перевірки
    # з ADR-002, розділ «Як перевірити, що рішення було правильним».
    crossing_groups = df.loc[split.dropped_mask, "title_norm"].nunique()

    return {
        "boundary": str(split.boundary),
        "boundary_rule": "train: PublishDate < boundary; test: PublishDate >= boundary",
        "n_rows_total": n,
        "n_train": n_train,
        "n_test": n_test,
        "n_test_dropped_as_duplicates": n_dropped,
        "share_train": round(n_train / n, 4),
        "share_test": round(n_test / n, 4),
        "n_crossing_duplicate_groups": int(crossing_groups),
        "share_dropped_of_test_period": round(n_dropped / (n_dropped + n_test), 4),
        "train_date_min": str(train_dates.min()),
        "train_date_max": str(train_dates.max()),
        "test_date_min": str(test_dates.min()),
        "test_date_max": str(test_dates.max()),
    }


def write_split(summary: dict, path: Path | str = SPLIT_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return path


def load_split(path: Path | str = SPLIT_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def apply_split(df: pd.DataFrame, path: Path | str = SPLIT_PATH):
    """Повертає (train, test) за зафіксованою межею.

    Читає межу з файлу, а не з константи модуля: після фіксації джерелом
    істини є split.json.
    """
    boundary = load_split(path)["boundary"]
    split = compute_split(df, boundary)
    return df.loc[split.train_mask].copy(), df.loc[split.test_mask].copy()
