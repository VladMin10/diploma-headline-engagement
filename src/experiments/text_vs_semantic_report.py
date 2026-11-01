"""Таблиці EXP-008 з матриці метрик E5.

Запуск:
    .venv/bin/python -m src.experiments.text_vs_semantic_report

Читає `data/processed/text_vs_semantic_results.parquet` і будує
`reports/tables/text_vs_semantic.md`. Навчання немає: усереднення робиться
при побудові таблиць, щоб їх можна було перебудувати без прогону.

**Висновків тут немає, тільки числа.** Статистична перевірка винесена
в окремий крок протоколом EXP-008.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "text_vs_semantic_results.parquet"
)
TARGETS_PATH = PROJECT_ROOT / "data" / "processed" / "targets_classification.parquet"
REPORT_PATH = PROJECT_ROOT / "reports" / "tables" / "text_vs_semantic.md"

PERCENTILE = "25"
PLATFORM_ORDER = ["Facebook", "GooglePlus", "LinkedIn"]
PLATFORM_SHORT = {"Facebook": "fb", "GooglePlus": "gp", "LinkedIn": "li"}
VARIANT_ORDER = ["TFIDF", "F0", "F3", "TFIDF+F0", "TFIDF+F3"]
BASE_VARIANT = "TFIDF"
# Порівняння з картки: головне і два додаткові.
COMPARISONS = [("TFIDF+F0", "головне"), ("TFIDF+F3", "додаткове"), ("F0", "додаткове")]
METRICS = [
    "F1_підібраний",
    "F1_05",
    "Accuracy",
    "Precision",
    "Recall",
    "ROC_AUC",
    "PR_AUC",
]


def _num(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def _signed(value: float, digits: int = 4) -> str:
    return f"{value:+.{digits}f}".replace(".", ",")


def reference_levels() -> pd.DataFrame:
    """F1 тривіального класифікатора і базовий рівень PR-AUC."""
    targets = pd.read_parquet(TARGETS_PATH)
    rows = []
    for platform in PLATFORM_ORDER:
        share = float(targets[f"y_{PLATFORM_SHORT[platform]}_{PERCENTILE}"].mean())
        rows.append(
            {
                "платформа": platform,
                "частка": share,
                "F1_тривіальний": 2 * share / (1 + share),
            }
        )
    return pd.DataFrame(rows).set_index("платформа")


def paired(results: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Парні різниці варіантів проти TF-IDF на однакових розбиттях."""
    rows = []
    for platform in PLATFORM_ORDER:
        block = results[results["платформа"] == platform]
        base = block[block["варіант"] == BASE_VARIANT].set_index(
            ["repeat", "fold"]
        )[metric]
        for variant, kind in COMPARISONS:
            arm = block[block["варіант"] == variant].set_index(["repeat", "fold"])[metric]
            diff = (arm - base).sort_index()
            rows.append(
                {
                    "платформа": platform,
                    "порівняння": f"{variant} − {BASE_VARIANT}",
                    "тип": kind,
                    "середня": float(diff.mean()),
                    "мінімум": float(diff.min()),
                    "максимум": float(diff.max()),
                    "додатних": int((diff > 0).sum()),
                    "n": int(len(diff)),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    results = pd.read_parquet(RESULTS_PATH)
    levels = reference_levels()
    means = results.groupby(["платформа", "варіант"])[METRICS].mean()

    lines = [
        "# EXP-008: змістові ознаки проти текстового представлення (E5)",
        "",
        "Згенеровано `src/experiments/text_vs_semantic_report.py` з "
        "`data/processed/text_vs_semantic_results.parquet`.",
        "Числа не редагуються вручну.",
        "",
        f"Сітка: {results['платформа'].nunique()} платформи × "
        f"{results['варіант'].nunique()} варіанти представлення × "
        f"{results.groupby('платформа')['варіант'].count().iloc[0] // results['варіант'].nunique()} "
        "розбиттів. "
        f"Рядків: {len(results)}. Поріг мітки — верхні {PERCENTILE}%, "
        "модель одна: логістична регресія.",
        "",
        "**Висновків тут немає, тільки числа.** Статистична перевірка винесена",
        "в окремий крок протоколом EXP-008.",
        "",
        "---",
        "",
        "## Довідкові рівні",
        "",
        "| Платформа | Частка позитивного класу | F1 тривіального | Базовий PR-AUC |",
        "|---|---|---|---|",
    ]
    for platform in PLATFORM_ORDER:
        row = levels.loc[platform]
        lines.append(
            f"| {platform} | {_num(float(row['частка']))} | "
            f"{_num(float(row['F1_тривіальний']))} | {_num(float(row['частка']))} |"
        )

    lines += [
        "",
        "F1 тривіального класифікатора — це 2p / (1 + p) для класифікатора, "
        "що відносить",
        "усі рядки до позитивного класу. Базовий рівень PR-AUC дорівнює частці "
        "позитивного",
        "класу, базовий рівень ROC-AUC — 0,5.",
        "",
        "---",
        "",
        "## Середні значення всіх семи метрик",
        "",
        "Середнє по двадцяти розбиттях.",
        "",
        "| Платформа | Варіант | F1 підібр. | F1 при 0,5 | Accuracy | Precision | Recall | ROC-AUC | PR-AUC |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for platform in PLATFORM_ORDER:
        for variant in VARIANT_ORDER:
            row = means.loc[(platform, variant)]
            cells = [_num(float(row[m])) for m in METRICS]
            lines.append(f"| {platform} | {variant} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "---",
        "",
        "## Парні різниці за головною метрикою",
        "",
        "F1 при підібраному порозі. Різниці обчислені парно на однакових "
        "розбиттях.",
        "",
        "| Платформа | Порівняння | Тип | Середня різниця | Мінімум | Максимум | Додатних із 20 |",
        "|---|---|---|---|---|---|---|",
    ]
    diffs = paired(results, "F1_підібраний")
    for row in diffs.itertuples(index=False):
        lines.append(
            f"| {row.платформа} | {row.порівняння} | {row.тип} | "
            f"{_signed(row.середня)} | {_signed(row.мінімум)} | "
            f"{_signed(row.максимум)} | {row.додатних} |"
        )

    for metric, title in (("ROC_AUC", "ROC-AUC"), ("PR_AUC", "PR-AUC")):
        lines += [
            "",
            f"### Ті самі різниці за {title}",
            "",
            "| Платформа | Порівняння | Середня різниця | Додатних із 20 |",
            "|---|---|---|---|",
        ]
        for row in paired(results, metric).itertuples(index=False):
            lines.append(
                f"| {row.платформа} | {row.порівняння} | "
                f"{_signed(row.середня)} | {row.додатних} |"
            )

    tfidf = results[results["розмір_словника"] > 0]
    lines += [
        "",
        "---",
        "",
        "## Словник TF-IDF",
        "",
        "Вектаризатор навчається лише на навчальній частині фолда, тому розмір "
        "словника",
        "змінюється від розбиття до розбиття.",
        "",
        "| Платформа | Середній розмір | Мінімум | Максимум | Ознак у TF-IDF+F₀ |",
        "|---|---|---|---|---|",
    ]
    for platform in PLATFORM_ORDER:
        block = tfidf[tfidf["платформа"] == platform]
        combined = block[block["варіант"] == "TFIDF+F0"]["n_ознак"]
        lines.append(
            f"| {platform} | {_num(float(block['розмір_словника'].mean()), 1)} | "
            f"{int(block['розмір_словника'].min())} | "
            f"{int(block['розмір_словника'].max())} | "
            f"{_num(float(combined.mean()), 1)} |"
        )

    lines += [
        "",
        f"Стеля `max_features` дорівнює {5000}; фактичний словник її "
        "не досягає,",
        "тобто обмежує не стеля, а `min_df`.",
        "",
        "---",
        "",
        "## Підібране C по варіантах",
        "",
        "Скільки разів кожне значення сітки обрано з двадцяти розбиттів "
        "на платформу.",
        "",
        "| Платформа | Варіант | Розподіл вибору C | Медіана |",
        "|---|---|---|---|",
    ]
    for platform in PLATFORM_ORDER:
        for variant in VARIANT_ORDER:
            block = results[
                (results["платформа"] == platform) & (results["варіант"] == variant)
            ]["C"]
            counts = Counter(block)
            spread = "; ".join(
                f"{_num(value, 2)} — {counts[value]}"
                for value in sorted(counts)
            )
            lines.append(
                f"| {platform} | {variant} | {spread} | "
                f"{_num(float(block.median()), 2)} |"
            )
    lines.append("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"звіт: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
