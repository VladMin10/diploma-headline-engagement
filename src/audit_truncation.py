"""Аудит обірваних заголовків.

Запуск:  .venv/bin/python -m src.audit_truncation

Частина заголовків у наборі обірвана трикрапкою на етапі збору даних.
Скрипт міряє масштаб явища, його зв'язок з виданням і з цільовими
змінними. Числа йдуть у docs/truncation_confound.md і в текст роботи.

Результат: data/interim/truncation_audit.json
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.data.load import (
    CLEAN_PATH,
    PROJECT_ROOT,
    TARGETS,
    is_truncated,
    load_raw,
)
from src.data.split import compute_split, load_split

OUT_PATH = PROJECT_ROOT / "data" / "interim" / "truncation_audit.json"


def describe_group(df: pd.DataFrame, mask: np.ndarray) -> dict:
    subset = df.loc[mask]
    words = subset["Title"].fillna("").astype(str).str.split().str.len()
    out = {
        "n": int(len(subset)),
        "share": round(float(mask.mean()), 4),
        "words_mean": round(float(words.mean()), 2),
        "words_median": float(words.median()),
        "words_max": int(words.max()),
    }
    for target in TARGETS:
        values = subset[f"{target}_value"]
        out[target] = {
            "median": float(values.median()),
            "mean": round(float(values.mean()), 2),
            "share_zero": round(float((values == 0).mean()), 4),
        }
    return out


def main() -> None:
    df = pd.read_parquet(CLEAN_PATH)
    truncated = is_truncated(df["Title"]).to_numpy()

    raw = load_raw()
    raw_truncated = is_truncated(raw["Title"])

    by_source = (
        df.assign(truncated=truncated)
        .groupby(df["Source"].fillna("__відсутнє__").astype(str))
        .agg(share_truncated=("truncated", "mean"), n=("truncated", "size"))
    )
    large = by_source[by_source["n"] >= 300].sort_values("share_truncated")

    by_topic = (
        df.assign(truncated=truncated)
        .groupby("Topic")
        .agg(share_truncated=("truncated", "mean"), n=("truncated", "size"))
        .sort_values("share_truncated")
    )

    kept = df.loc[~truncated]
    split_full = compute_split(df, load_split()["boundary"])
    split_kept = compute_split(kept, load_split()["boundary"])

    test_sources = (
        kept.loc[split_kept.test_mask, "Source"].fillna("__відсутнє__").astype(str).value_counts()
    )
    qualifying = test_sources[test_sources >= 30]

    payload = {
        "generated_by": "src/audit_truncation.py",
        "definition": "Title після видалення кінцевих пробілів закінчується на '...' або '…'",
        "raw": {
            "n_rows": int(len(raw)),
            "n_truncated": int(raw_truncated.sum()),
            "share_truncated": round(float(raw_truncated.mean()), 4),
        },
        "working_sample": {
            "n_rows": int(len(df)),
            "truncated": describe_group(df, truncated),
            "not_truncated": describe_group(df, ~truncated),
        },
        "by_source_large": {
            name: {"share_truncated": round(float(row.share_truncated), 4), "n": int(row.n)}
            for name, row in large.iterrows()
        },
        "by_topic": {
            name: {"share_truncated": round(float(row.share_truncated), 4), "n": int(row.n)}
            for name, row in by_topic.iterrows()
        },
        "split_before": {
            "n_rows": int(len(df)),
            "n_train": int(split_full.train_mask.sum()),
            "n_test": int(split_full.test_mask.sum()),
        },
        "split_after": {
            "n_rows": int(len(kept)),
            "n_train": int(split_kept.train_mask.sum()),
            "n_test": int(split_kept.test_mask.sum()),
            "n_test_dropped_as_duplicates": int(split_kept.dropped_mask.sum()),
            "n_sources_ge30_in_test": int(len(qualifying)),
            "n_rows_covered_by_those_sources": int(qualifying.sum()),
            "share_of_test_covered": round(float(qualifying.sum() / split_kept.test_mask.sum()), 4),
        },
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    w = payload["working_sample"]
    print(f"Обірваних у сирому файлі: {payload['raw']['n_truncated']} "
          f"({payload['raw']['share_truncated']*100:.2f}%)")
    print(f"У робочій вибірці: {w['truncated']['n']} ({w['truncated']['share']*100:.1f}%)")
    print(f"Медіана Facebook: обірвані {w['truncated']['Facebook']['median']:.0f}, "
          f"необірвані {w['not_truncated']['Facebook']['median']:.0f}")
    print("\nЧастка обірваних по великих джерелах:")
    for name, row in large.iterrows():
        print(f"  {name:24s} {row.share_truncated*100:5.1f}%  (n={int(row.n)})")
    print(f"\nПісля виключення: {payload['split_after']['n_rows']} рядків, "
          f"train {payload['split_after']['n_train']}, test {payload['split_after']['n_test']}")
    print("Записано:", OUT_PATH.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
