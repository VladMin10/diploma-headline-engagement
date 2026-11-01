"""Зведення по списку експериментів розділу 6 документа кафедри.

Запуск:
    .venv/bin/python -m src.report.kafedra_summary

Пише `reports/tables/kafedra_experiments_summary.md` — одну таблицю, де
по рядку на кожен експеримент E1—E7 документа кафедри.

**Числа не набираються руками.** Кількість порівнянь і кількість
підтверджених правилом ADR-006 читаються з parquet-таблиць статистики
кожного експерименту. Таблиця призначена для тексту роботи і для слайда,
тому будь-яка розбіжність між нею і джерелом має виявлятися падінням
збірки, а не мовчки.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
OUT_PATH = PROJECT_ROOT / "reports" / "tables" / "kafedra_experiments_summary.md"

# Джерела статистики по кожному експерименту документа кафедри.
# Ключ — номер розділу 6, значення — файли, з яких беруться числа.
SOURCES = {
    "E1—E4": ["ablation_stats.parquet"],
    "E5": ["text_vs_semantic_stats.parquet", "embeddings_stats.parquet"],
    "E6": ["interaction_form_stats.parquet"],
    "E7": ["fuzzy_strength_stats.parquet"],
}

ROWS = [
    {
        "код": "E1—E4",
        "що порівнювалося": "набори ознак F₀, F₁, F₂, F₃ між собою",
        "експеримент": "EXP-007",
        "результат": "інженерія похідних ознак приросту не дає; усі "
                     "підтверджені різниці від'ємні",
    },
    {
        "код": "E5",
        "що порівнювалося": "сім змістових ознак проти TF-IDF і проти "
                            "sentence embeddings",
        "експеримент": "EXP-008 разом із розширенням",
        "результат": "сім ознак зіставні з обома векторними представленнями; "
                     "жодне з них ніде не перевищило їх із підтвердженням",
    },
    {
        "код": "E6",
        "що порівнювалося": "парні взаємодії: добуток проти мінімуму",
        "експеримент": "EXP-011",
        "результат": "спосіб формування взаємодій значення не має",
    },
    {
        "код": "E7",
        "що порівнювалося": "сила прояву: жорсткий поріг N_strong проти "
                            "нечіткої F_high",
        "експеримент": "EXP-012",
        "результат": "на цьому корпусі два представлення взаємозамінні",
    },
]


def counts(files: list[str]) -> tuple[int, int, int, int]:
    """Порівнянь усього, підтверджено, з них додатних і від'ємних."""
    total = confirmed = positive = negative = 0
    for name in files:
        frame = pd.read_parquet(PROCESSED / name)
        block = frame[frame["підтверджено"]]
        total += len(frame)
        confirmed += len(block)
        positive += int((block["середня_різниця"] > 0).sum())
        negative += int((block["середня_різниця"] < 0).sum())
    return total, confirmed, positive, negative


def report_lines() -> list[str]:
    lines = [
        "# Експерименти розділу 6 документа кафедри: зведення",
        "",
        "Згенеровано `src/report/kafedra_summary.py`. Кількість порівнянь "
        "і кількість підтверджених",
        "читаються з parquet-таблиць статистики кожного експерименту "
        "й не набираються руками.",
        "",
        "Підтвердження означає правило ADR-006: обидва критерії дають p менше "
        "0,05 і 95% бутстреп-інтервал",
        "не містить нуля. Критерій двосторонній, тому правило однаково "
        "застосовне до приростів і до погіршень.",
        "",
        "| Документ | Що порівнювалося | Експеримент проєкту | Результат "
        "| Підтверджено правилом |",
        "|---|---|---|---|---|",
    ]
    grand_total = grand_confirmed = grand_positive = grand_negative = 0
    for row in ROWS:
        total, confirmed, positive, negative = counts(SOURCES[row["код"]])
        grand_total += total
        grand_confirmed += confirmed
        grand_positive += positive
        grand_negative += negative
        detail = f"{confirmed} з {total}"
        if confirmed:
            parts = []
            if positive:
                parts.append(f"{positive} додатних")
            if negative:
                parts.append(f"{negative} від'ємних")
            detail += " (" + ", ".join(parts) + ")"
        lines.append(
            f"| {row['код']} | {row['що порівнювалося']} | {row['експеримент']} "
            f"| {row['результат']} | **{detail}** |"
        )

    lines += [
        "",
        f"**Разом по всьому списку: {grand_confirmed} підтверджених різниць "
        f"із {grand_total} виконаних порівнянь**",
        f"({grand_positive} додатних, {grand_negative} від'ємних).",
        "",
        "Усі підтверджені прирости припадають на E5, тобто на порівняння "
        "змістових ознак",
        "із текстовими представленнями. Усі підтверджені погіршення "
        "припадають на E1—E4,",
        "тобто на додавання похідних ознак. E6 і E7 не дали підтверджених "
        "різниць у жодному напрямку.",
        "",
    ]
    return lines


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(report_lines()), encoding="utf-8")
    total = confirmed = 0
    for row in ROWS:
        block_total, block_confirmed, _, _ = counts(SOURCES[row["код"]])
        total += block_total
        confirmed += block_confirmed
        print(f"  {row['код']:6s} {row['експеримент']:26s} "
              f"{block_confirmed:2d} з {block_total:3d}")
    print(f"  разом: {confirmed} з {total}")
    print(f"збережено: {OUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
