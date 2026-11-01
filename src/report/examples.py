"""Приклади заголовків: що модель вважає сильним і слабким.

Запуск:
    .venv/bin/python -m src.report.examples

Рахується з `data/processed/report_predictions.parquet`; нічого не навчається.

Розбиття зафіксоване: повтор 0, фолд 0, платформа Facebook, модель —
логістична регресія на семи змістових ознаках. Вибір першого розбиття
навмисний і зроблений до перегляду заголовків: інакше добір «показового»
фолда був би добором прикладів під результат.

Друга частина — пари заголовків з одного видання і однієї теми з різними
мітками. Вони потрібні для того, щоб різниця в оцінках не пояснювалася
різницею видань чи тем: обидва чинники зафіксовані.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.features.copywriting import FEATURE_ORDER, load_labels

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "report_predictions.parquet"
CLEAN_PATH = PROJECT_ROOT / "data" / "interim" / "news_clean.parquet"
TARGETS_PATH = PROJECT_ROOT / "data" / "processed" / "targets_classification.parquet"
REPORT_PATH = PROJECT_ROOT / "reports" / "tables" / "examples.md"

PLATFORM = "Facebook"
VARIANT = "LogisticRegression_F0"
REPEAT, FOLD = 0, 0
TOP_N = 10
N_PAIRS = 5
SHORT = {"Facebook": "fb", "GooglePlus": "gp", "LinkedIn": "li"}
PERCENTILE = "25"

HEADS = {
    "gap": "прог",
    "scale": "масш",
    "concrete": "конк",
    "conflict": "конф",
    "authority": "авто",
    "violation": "поруш",
    "stakes": "невиз",
}


def load_context() -> pd.DataFrame:
    clean = pd.read_parquet(
        CLEAN_PATH, columns=["IDLink", "Title", "Source", "Topic", "Facebook_value"]
    )
    clean["IDLink"] = clean["IDLink"].astype("int64")
    labels = load_labels().reset_index()
    targets = pd.read_parquet(TARGETS_PATH)[["IDLink", f"y_{SHORT[PLATFORM]}_{PERCENTILE}"]]
    return labels.merge(clean, on="IDLink").merge(targets, on="IDLink")


def _clip(text: str, width: int = 70) -> str:
    text = str(text).replace("|", "\\|").strip()
    return text if len(text) <= width else text[: width - 1] + "…"


def _num(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def extremes(predictions: pd.DataFrame, context: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    split = predictions[
        (predictions["платформа"] == PLATFORM)
        & (predictions["варіант"] == VARIANT)
        & (predictions["repeat"] == REPEAT)
        & (predictions["fold"] == FOLD)
    ].merge(context, on="IDLink")
    ordered = split.sort_values("ймовірність", ascending=False)
    return ordered.head(TOP_N), ordered.tail(TOP_N).iloc[::-1]


def pairs(context: pd.DataFrame) -> pd.DataFrame:
    """Пари з одного видання і однієї теми з різними мітками.

    Серед таких пар беруться ті, де різниця в кількості реакцій найбільша:
    вони найнаочніше показують, що мітка не є випадковою.
    """
    column = f"y_{SHORT[PLATFORM]}_{PERCENTILE}"
    rows = []
    for (source, topic), block in context.groupby(["Source", "Topic"]):
        positive = block[block[column] == 1]
        negative = block[block[column] == 0]
        if positive.empty or negative.empty:
            continue
        best = positive.loc[positive["Facebook_value"].idxmax()]
        worst = negative.loc[negative["Facebook_value"].idxmin()]
        rows.append(
            {
                "Source": source,
                "Topic": topic,
                "різниця_реакцій": float(best["Facebook_value"] - worst["Facebook_value"]),
                "верх": best,
                "низ": worst,
            }
        )
    frame = pd.DataFrame(rows).sort_values("різниця_реакцій", ascending=False)
    return frame.head(N_PAIRS)


def table_lines(block: pd.DataFrame, title: str) -> list[str]:
    heads = " | ".join(HEADS[f] for f in FEATURE_ORDER)
    lines = [
        f"### {title}",
        "",
        f"| Заголовок | {heads} | Мітка | Реакції | Ймовірність |",
        "|---" * (len(FEATURE_ORDER) + 4) + "|",
    ]
    column = f"y_{SHORT[PLATFORM]}_{PERCENTILE}"
    for row in block.itertuples(index=False):
        values = " | ".join(str(int(getattr(row, f))) for f in FEATURE_ORDER)
        label = "так" if getattr(row, column) == 1 else "ні"
        lines.append(
            f"| {_clip(row.Title)} | {values} | {label} | "
            f"{int(row.Facebook_value)} | {_num(row.ймовірність)} |"
        )
    lines.append("")
    return lines


def main() -> None:
    predictions = pd.read_parquet(PREDICTIONS_PATH)
    context = load_context()
    top, bottom = extremes(predictions, context)

    accuracy_top = float(top[f"y_{SHORT[PLATFORM]}_{PERCENTILE}"].mean())
    accuracy_bottom = float(bottom[f"y_{SHORT[PLATFORM]}_{PERCENTILE}"].mean())

    lines = [
        "# Приклади заголовків",
        "",
        "Згенеровано `src/report/examples.py` з "
        "`data/processed/report_predictions.parquet`.",
        "Нічого не навчається.",
        "",
        f"Платформа {PLATFORM}, мітка «верхня чверть свого видання», модель —"
        " логістична регресія",
        f"на семи змістових ознаках. Розбиття зафіксоване: повтор {REPEAT}, "
        f"фолд {FOLD}.",
        "Вибір першого розбиття зроблений до перегляду заголовків: інакше добір",
        "показового фолда був би добором прикладів під результат.",
        "",
        "Скорочення колонок: прог — прогалина в інформації, масш — масштаб "
        "наслідків,",
        "конк — конкретність предиката, конф — явність конфлікту, авто — вага "
        "авторитету,",
        "поруш — порушення очікування, невиз — проспективна невизначеність.",
        "",
        "---",
        "",
        "## Краї прогнозу",
        "",
        f"Серед десяти з найвищою ймовірністю справді в позитивному класі "
        f"{int(top[f'y_{SHORT[PLATFORM]}_{PERCENTILE}'].sum())} з {TOP_N} "
        f"({_num(accuracy_top)}); серед десяти з найнижчою — "
        f"{int(bottom[f'y_{SHORT[PLATFORM]}_{PERCENTILE}'].sum())} з {TOP_N} "
        f"({_num(accuracy_bottom)}).",
        "",
    ]
    lines += table_lines(top, "Десять із найвищою прогнозованою ймовірністю")
    lines += table_lines(bottom, "Десять із найнижчою прогнозованою ймовірністю")

    lines += [
        "---",
        "",
        "## Пари з одного видання і однієї теми",
        "",
        "Видання і тема зафіксовані всередині пари, тому різниця в мітці "
        "не пояснюється",
        "ані рівнем видання, ані темою. Пари відсортовані за різницею "
        "в кількості реакцій.",
        "",
    ]
    heads = " | ".join(HEADS[f] for f in FEATURE_ORDER)
    for index, row in enumerate(pairs(context).itertuples(index=False), start=1):
        lines += [
            f"### Пара {index}: {row.Source}, тема {row.Topic}",
            "",
            f"| Заголовок | {heads} | Мітка | Реакції |",
            "|---" * (len(FEATURE_ORDER) + 3) + "|",
        ]
        for record, label in ((row.верх, "так"), (row.низ, "ні")):
            values = " | ".join(str(int(record[f])) for f in FEATURE_ORDER)
            lines.append(
                f"| {_clip(record['Title'])} | {values} | {label} | "
                f"{int(record['Facebook_value'])} |"
            )
        lines.append("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"звіт: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
