"""Влучність на верхніх k відсотках: практичний вимір якості.

Запуск:
    .venv/bin/python -m src.report.practical_lift

Рахується з `data/processed/report_predictions.parquet`; нічого не навчається.

Метрика відповідає на питання, яке ставить копірайтер, а не дослідник. Не «яка
площа під кривою», а «якщо я візьму десяту частину заголовків, що їх модель
вважає найкращими, яка частка з них справді потрапить у верхню чверть свого
видання». Базовим рівнем є частка позитивного класу: стільки дала б випадкова
вибірка тієї самої величини.

Влучність рахується окремо на кожному з двадцяти розбиттів і усереднюється,
а не рахується на злитих прогнозах. Друге дало б одне число без розкиду,
і не було б видно, наскільки воно стійке.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "report_predictions.parquet"
REPORT_PATH = PROJECT_ROOT / "reports" / "tables" / "practical_lift.md"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "practical_lift.parquet"

PLATFORM_ORDER = ["Facebook", "GooglePlus", "LinkedIn"]
VARIANTS = {
    "LogisticRegression_F0": "сім змістових ознак",
    "LogisticRegression_TFIDF": "TF-IDF",
}
SHARES = (0.10, 0.20, 0.25)


def precision_at_k(y: np.ndarray, proba: np.ndarray, share: float) -> tuple[float, float]:
    """Влучність серед верхніх k і серед нижніх k відсотків."""
    k = max(1, int(round(len(y) * share)))
    order = np.argsort(-proba, kind="stable")
    top = y[order[:k]].mean()
    bottom = y[order[-k:]].mean()
    return float(top), float(bottom)


def compute(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for platform in PLATFORM_ORDER:
        for variant in VARIANTS:
            block = predictions[
                (predictions["платформа"] == platform)
                & (predictions["варіант"] == variant)
            ]
            for share in SHARES:
                per_split = []
                for _, split in block.groupby(["repeat", "fold"], observed=True):
                    y = split["y"].to_numpy(dtype=np.int8)
                    proba = split["ймовірність"].to_numpy(dtype=np.float64)
                    top, bottom = precision_at_k(y, proba, share)
                    per_split.append((top, bottom, float(y.mean())))
                top_values = np.array([v[0] for v in per_split])
                bottom_values = np.array([v[1] for v in per_split])
                base = float(np.mean([v[2] for v in per_split]))
                rows.append(
                    {
                        "платформа": platform,
                        "варіант": variant,
                        "k": share,
                        "precision_топ": float(top_values.mean()),
                        "мінімум": float(top_values.min()),
                        "максимум": float(top_values.max()),
                        "lift": float(top_values.mean()) / base,
                        "precision_низ": float(bottom_values.mean()),
                        "базова_частка": base,
                        "n_розбиттів": int(len(per_split)),
                    }
                )
    return pd.DataFrame(rows)


def _num(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def report_lines(table: pd.DataFrame) -> list[str]:
    lines = [
        "# Влучність на верхніх k відсотках",
        "",
        "Згенеровано `src/report/practical_lift.py` з "
        "`data/processed/report_predictions.parquet`.",
        "Нічого не навчається: рахується з уже збережених ймовірностей.",
        "",
        "Метрика відповідає на питання копірайтера: якщо взяти k відсотків "
        "заголовків",
        "із найвищою прогнозованою ймовірністю, яка частка з них справді "
        "потрапляє",
        "у верхню чверть свого видання. Базовим рівнем є частка позитивного "
        "класу —",
        "стільки дала б випадкова вибірка тієї самої величини. Lift — відношення",
        "влучності до базового рівня.",
        "",
        "Влучність рахується окремо на кожному з двадцяти розбиттів "
        "і усереднюється;",
        "мінімум і максимум показують розкид по розбиттях.",
        "",
        "---",
        "",
    ]
    for variant, title in VARIANTS.items():
        lines += [
            f"## {title}",
            "",
            "| Платформа | k | Влучність зверху | Мінімум | Максимум | Lift | Влучність знизу | Базовий рівень |",
            "|---|---|---|---|---|---|---|---|",
        ]
        block = table[table["варіант"] == variant]
        for row in block.itertuples(index=False):
            lines.append(
                f"| {row.платформа} | {int(row.k * 100)}% | "
                f"**{_num(row.precision_топ)}** | {_num(row.мінімум)} | "
                f"{_num(row.максимум)} | **{_num(row.lift, 2)}** | "
                f"{_num(row.precision_низ)} | {_num(row.базова_частка)} |"
            )
        lines.append("")

    lines += [
        "---",
        "",
        "## Два представлення поруч",
        "",
        "| Платформа | k | Сім ознак | TF-IDF | Різниця | Базовий рівень |",
        "|---|---|---|---|---|---|",
    ]
    for platform in PLATFORM_ORDER:
        for share in SHARES:
            own = table[
                (table["платформа"] == platform)
                & (table["k"] == share)
            ].set_index("варіант")
            semantic = float(own.loc["LogisticRegression_F0", "precision_топ"])
            text = float(own.loc["LogisticRegression_TFIDF", "precision_топ"])
            base = float(own.loc["LogisticRegression_F0", "базова_частка"])
            lines.append(
                f"| {platform} | {int(share * 100)}% | {_num(semantic)} | "
                f"{_num(text)} | {semantic - text:+.4f}".replace(".", ",")
                + f" | {_num(base)} |"
            )
    lines.append("")
    return lines


def main() -> None:
    predictions = pd.read_parquet(PREDICTIONS_PATH)
    table = compute(predictions)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(OUT_PATH, index=False)

    lines = report_lines(table)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"звіт: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
