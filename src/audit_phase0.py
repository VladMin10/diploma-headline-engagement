"""Фаза 0: аудит даних, фіксація спліту, розподіли цільових змінних.

Запуск:  .venv/bin/python -m src.audit_phase0

Скрипт нічого не вирішує самостійно. Межа спліту береться з константи
BOUNDARY у src/data/split.py, узгодженої з користувачем. Результати:
  data/interim/news_clean.parquet
  data/interim/split.json
  data/interim/audit_phase0.json   — усі числа для docs/data_card.md
  reports/figures/fig01_target_distributions.{png,pdf}
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data.load import (  # noqa: E402
    CLEAN_PATH,
    MISSING_CODE,
    PROJECT_ROOT,
    RAW_PATH,
    TARGETS,
    add_derived,
    load_raw,
    write_clean,
)
from src.data.split import (  # noqa: E402
    BOUNDARY,
    compute_split,
    split_summary,
    write_split,
)

FIGURES = PROJECT_ROOT / "reports" / "figures"
AUDIT_PATH = PROJECT_ROOT / "data" / "interim" / "audit_phase0.json"

PLATFORM_LABELS = {
    "Facebook": "Facebook",
    "GooglePlus": "Google+",
    "LinkedIn": "LinkedIn",
}


def describe_targets(df: pd.DataFrame, columns: list[str]) -> dict:
    out = {}
    for col in columns:
        series = df[f"{col}_value"].dropna()
        out[col] = {
            "n_measured": int(series.size),
            "n_zero": int((series == 0).sum()),
            "share_zero": round(float((series == 0).mean()), 4),
            "mean": round(float(series.mean()), 2),
            "std": round(float(series.std()), 2),
            "min": float(series.min()),
            "p25": float(series.quantile(0.25)),
            "median": float(series.median()),
            "p75": float(series.quantile(0.75)),
            "p90": float(series.quantile(0.90)),
            "p99": float(series.quantile(0.99)),
            "max": float(series.max()),
            "mean_log1p": round(float(np.log1p(series).mean()), 4),
            "std_log1p": round(float(np.log1p(series).std()), 4),
        }
    return out


def audit_raw(raw: pd.DataFrame) -> dict:
    """Цілісність повного набору до будь-якої фільтрації."""
    df = add_derived(raw)

    missing_pattern = (
        df[TARGETS]
        .eq(MISSING_CODE)
        .value_counts()
        .rename_axis(TARGETS)
        .reset_index(name="n")
    )
    patterns = [
        {
            "Facebook_missing": bool(row.Facebook),
            "GooglePlus_missing": bool(row.GooglePlus),
            "LinkedIn_missing": bool(row.LinkedIn),
            "n": int(row.n),
        }
        for row in missing_pattern.itertuples()
    ]

    title_counts = df["title_norm"].value_counts()
    dup_groups = title_counts[title_counts > 1]

    by_month = (
        df["PublishDate"].dt.to_period("M").astype(str).value_counts().sort_index()
    )

    n = len(df)
    return {
        "n_rows": n,
        "n_columns": int(raw.shape[1]),
        "columns": list(raw.columns),
        "n_unique_idlink": int(df["IDLink"].nunique()),
        "date_min": str(df["PublishDate"].min()),
        "date_max": str(df["PublishDate"].max()),
        "n_before_2015_11": int((df["PublishDate"] < pd.Timestamp("2015-11-01")).sum()),
        "share_before_2015_11": round(
            float((df["PublishDate"] < pd.Timestamp("2015-11-01")).mean()), 4
        ),
        "rows_by_month": {k: int(v) for k, v in by_month.items()},
        "nulls_by_column": {
            c: int(raw[c].isna().sum()) for c in raw.columns if raw[c].isna().any()
        },
        "n_empty_title": int((df["title_norm"] == "").sum()),
        "topics": {k: int(v) for k, v in df["Topic"].value_counts().items()},
        "n_sources": int(df["Source"].nunique()),
        "n_sources_single_item": int((df["Source"].value_counts() == 1).sum()),
        "top_sources": {
            k: int(v) for k, v in df["Source"].value_counts().head(10).items()
        },
        "n_duplicate_title_groups": int(dup_groups.size),
        "n_rows_in_duplicate_groups": int(dup_groups.sum()),
        "share_rows_in_duplicate_groups": round(float(dup_groups.sum() / n), 4),
        "largest_duplicate_group": int(dup_groups.max()) if dup_groups.size else 0,
        "missing_target_counts": {
            c: int((df[c] == MISSING_CODE).sum()) for c in TARGETS
        },
        "missing_target_shares": {
            c: round(float((df[c] == MISSING_CODE).mean()), 4) for c in TARGETS
        },
        "missing_patterns": patterns,
        "targets_full_set": describe_targets(df, TARGETS),
    }


def plot_target_distributions(df: pd.DataFrame) -> list[Path]:
    """Гістограми log1p для трьох платформ.

    Заголовок усередині рисунка не ставиться: він дається в тексті статті
    (CLAUDE.md розділ 6, пункт 8).
    """
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)

    for ax, col in zip(axes, TARGETS):
        values = df[f"{col}_log1p"].dropna()
        ax.hist(values, bins=60, color="#4C72B0", edgecolor="white", linewidth=0.3)
        share_zero = float((values == 0).mean())
        ax.set_xlabel(f"{PLATFORM_LABELS[col]}\nlog(1 + популярність)")
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.text(
            0.97,
            0.95,
            f"нулів {share_zero * 100:.1f}%".replace(".", ","),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
        )
    axes[0].set_ylabel("Кількість новин")
    fig.tight_layout()

    paths = []
    for ext in ("png", "pdf"):
        path = FIGURES / f"fig01_target_distributions.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def plot_missingness_by_month(raw: pd.DataFrame, boundary: str) -> list[Path]:
    """Розподіл рядків по місяцях і частка неповних вимірювань.

    Рисунок документує аномалію, виявлену у Фазі 0: підсумкова популярність
    міряється через дві доби після публікації, а збір даних обірвався
    на початку липня 2016 року, тому пізні новини не встигли дозміритися.
    """
    FIGURES.mkdir(parents=True, exist_ok=True)
    df = raw.copy()
    df["month"] = df["PublishDate"].dt.to_period("M").astype(str)
    df["incomplete"] = df[TARGETS].eq(MISSING_CODE).any(axis=1)
    grouped = df.groupby("month").agg(n=("IDLink", "size"), miss=("incomplete", "sum"))
    grouped = grouped.loc[grouped.index >= "2015-11"]
    share = grouped["miss"] / grouped["n"] * 100

    fig, ax = plt.subplots(figsize=(9, 3.4))
    x = np.arange(len(grouped))
    ax.bar(x, grouped["n"], color="#B0B7C3", label="Усього новин")
    ax.bar(x, grouped["miss"], color="#C44E52", label="З неповним вимірюванням")
    ax.set_xticks(x)
    ax.set_xticklabels(grouped.index, rotation=45, ha="right")
    ax.set_ylabel("Кількість новин")
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)

    ts = pd.Timestamp(boundary)
    boundary_month = str(ts.to_period("M"))
    if boundary_month in grouped.index:
        # Межа проходить усередині місяця, тому позиція обчислюється
        # пропорційно до дня, а не ставиться на початок стовпчика.
        within = (ts.day - 1) / ts.days_in_month
        pos = list(grouped.index).index(boundary_month) - 0.5 + within
        ax.axvline(pos, color="#4C72B0", linestyle="--", linewidth=1.2)
        ax.text(
            pos + 0.1,
            grouped["n"].max() * 0.95,
            f"межа спліту\n{ts.date()}",
            color="#4C72B0",
            fontsize=9,
            va="top",
        )

    for xi, value in zip(x, share):
        if value >= 15:
            ax.text(
                xi,
                grouped["n"].iloc[xi] * 1.02,
                f"{value:.0f}%",
                ha="center",
                fontsize=8,
                color="#C44E52",
            )
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()

    paths = []
    for ext in ("png", "pdf"):
        path = FIGURES / f"fig02_missingness_by_month.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def main() -> None:
    raw_untouched = load_raw(RAW_PATH, drop_exact_duplicates=False)
    raw = load_raw(RAW_PATH)
    n_exact_duplicates = len(raw_untouched) - len(raw)
    print(
        f"Сирий набір: {len(raw_untouched)} рядків, {raw.shape[1]} колонок; "
        f"повних дублікатів рядка вилучено: {n_exact_duplicates}"
    )

    raw_audit = audit_raw(raw)

    df = add_derived(raw)
    complete = np.ones(len(df), dtype=bool)
    for col in TARGETS:
        complete &= (df[col] != MISSING_CODE).to_numpy()
    clean = df.loc[complete].reset_index(drop=True)
    print(
        f"Повні спостереження: {len(clean)} рядків "
        f"({len(clean) / len(df) * 100:.2f}%)"
    )
    write_clean(clean, CLEAN_PATH)

    split = compute_split(clean, BOUNDARY)
    summary = split_summary(clean, split)
    summary["agreed_with_user"] = "2026-08-06"
    summary["source"] = "ADR-002; підтверджено користувачем 6 серпня 2026 року"
    write_split(summary)
    print(
        f"Спліт {summary['boundary']}: train {summary['n_train']}, "
        f"test {summary['n_test']}, вилучено як повтори "
        f"{summary['n_test_dropped_as_duplicates']}"
    )

    figures = plot_target_distributions(clean)
    figures += plot_missingness_by_month(df, BOUNDARY)
    print("Рисунки:", ", ".join(p.name for p in figures))

    audit = {
        "generated_by": "src/audit_phase0.py",
        "raw_file": str(RAW_PATH.relative_to(PROJECT_ROOT)),
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "n_rows_in_file": int(len(raw_untouched)),
        "n_exact_duplicate_rows_dropped": int(n_exact_duplicates),
        "raw": raw_audit,
        "complete_cases": {
            "n_rows": int(len(clean)),
            "share_of_raw": round(float(len(clean) / len(df)), 4),
            "targets": describe_targets(clean, TARGETS),
        },
        "split": summary,
        "targets_train": None,
        "targets_test": None,
    }
    audit["targets_train"] = describe_targets(clean.loc[split.train_mask], TARGETS)
    audit["targets_test"] = describe_targets(clean.loc[split.test_mask], TARGETS)

    with open(AUDIT_PATH, "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    print("Аудит записано:", AUDIT_PATH.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    sys.exit(main())
