"""Допоміжна мітка з глобальним порогом (ADR-005, четверте доповнення).

Запуск:
    .venv/bin/python -m src.data.target_global [--force]

Мітка потрібна для одного експерименту — EXP-010, перевірки гіпотези про те,
чи пояснює поріг усередині видання розходження впорядкувань між лініями.
У жодному порівнянні наборів ознак і в жодній відповіді на Q1, Q2 чи Q3
вона участі не бере.

**Основна мітка не змінюється.** `targets_classification.parquet` тільки
читається — для підрахунку того, скільки рядків змінили мітку. Поріг усередині
видання лишається незворотним, як фіксує розділ «Наслідки» ADR-005.

Правила, зафіксовані четвертим доповненням ADR-005 до цього обчислення:
y дорівнює одиниці, якщо кількість реакцій строго більша за 75-й перцентиль
по всій платформі; перцентиль рахується один раз на всіх 5 055 анотованих
рядках; запасні правила 1 і 2 не застосовуються, бо глобальний поріг
виродитися не може; рядки з відсутнім вимірюванням виключаються за ADR-003.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.copywriting import load_labels

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEAN_PATH = PROJECT_ROOT / "data" / "interim" / "news_clean.parquet"
MAIN_TARGETS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "targets_classification.parquet"
)
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "targets_global.parquet"
META_PATH = PROJECT_ROOT / "data" / "processed" / "targets_global_meta.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "tables" / "target_global.md"

PLATFORMS = {
    "Facebook": ("fb", "Facebook_value"),
    "GooglePlus": ("gp", "GooglePlus_value"),
    "LinkedIn": ("li", "LinkedIn_value"),
}
PERCENTILE = 75.0
MAIN_SHARE = "25"


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "недоступний"


def build() -> tuple[pd.DataFrame, dict]:
    """Мітка і числа для звіту."""
    frame = pd.read_parquet(CLEAN_PATH)
    frame["IDLink"] = frame["IDLink"].astype("int64")
    annotated = frame[frame["IDLink"].isin(load_labels().index)].copy()
    annotated = annotated.sort_values("IDLink").reset_index(drop=True)

    main = pd.read_parquet(MAIN_TARGETS_PATH)
    targets = pd.DataFrame({"IDLink": annotated["IDLink"].to_numpy()})
    summary = {}

    for platform, (short, column) in PLATFORMS.items():
        values = annotated[column].to_numpy(dtype=np.float64)
        present = ~np.isnan(values)
        observed = values[present]
        threshold = float(np.percentile(observed, PERCENTILE))

        y = np.full(values.size, np.nan, dtype=np.float64)
        # Строга нерівність, як і в основній мітці: через однакові значення
        # фактична частка буде меншою за чверть, і поріг під неї
        # не підганяється (ADR-005, запасне правило 3).
        y[present] = (values[present] > threshold).astype(np.float64)
        targets[f"y_{short}_global"] = y

        merged = targets[["IDLink", f"y_{short}_global"]].merge(
            main[["IDLink", f"y_{short}_{MAIN_SHARE}"]], on="IDLink"
        )
        old = merged[f"y_{short}_{MAIN_SHARE}"]
        new = merged[f"y_{short}_global"]
        both = old.notna() & new.notna()
        summary[platform] = {
            "поріг": threshold,
            "n_з_вимірюванням": int(present.sum()),
            "n_виключено_NaN": int((~present).sum()),
            "частка_позитивного_класу": round(float(np.nanmean(y)), 4),
            "n_позитивного_класу": int(np.nansum(y)),
            "змінили_мітку_усього": int((old[both] != new[both]).sum()),
            "стали_позитивними": int(((old[both] == 0) & (new[both] == 1)).sum()),
            "стали_негативними": int(((old[both] == 1) & (new[both] == 0)).sum()),
            "n_порівняних": int(both.sum()),
        }

    return targets, summary


def _num(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def report_lines(summary: dict, n_rows: int) -> list[str]:
    lines = [
        "# Допоміжна мітка з глобальним порогом",
        "",
        "Згенеровано `src/data/target_global.py` за четвертим доповненням ADR-005.",
        "",
        "Мітка потрібна для одного експерименту — EXP-010. Основна мітка "
        "не змінюється",
        "і тут тільки читається. У порівняннях наборів ознак і у відповідях",
        "на Q1, Q2 і Q3 допоміжна мітка участі не бере.",
        "",
        f"Правило: y = 1, якщо кількість реакцій строго більша за "
        f"{int(PERCENTILE)}-й перцентиль по всій платформі.",
        f"Перцентиль обчислено один раз на всіх {n_rows} анотованих рядках.",
        "",
        "### Частка позитивного класу і поріг",
        "",
        "| Платформа | Поріг | З вимірюванням | Виключено за NaN | Частка позитивного класу | Рядків у класі |",
        "|---|---|---|---|---|---|",
    ]
    for platform, block in summary.items():
        lines.append(
            f"| {platform} | {_num(block['поріг'], 1)} | "
            f"{block['n_з_вимірюванням']} | {block['n_виключено_NaN']} | "
            f"{_num(block['частка_позитивного_класу'])} | "
            f"{block['n_позитивного_класу']} |"
        )

    lines += [
        "",
        "### Скільки рядків змінили мітку порівняно з основною",
        "",
        "Основна мітка будується за порогом усередині видання, допоміжна — "
        "за глобальним.",
        "Це число показує, наскільки сильно нормалізація за виданням змінює те,",
        "що вважається успіхом.",
        "",
        "| Платформа | Змінили мітку | Стали позитивними | Стали негативними | Частка змінених |",
        "|---|---|---|---|---|",
    ]
    for platform, block in summary.items():
        share = block["змінили_мітку_усього"] / block["n_порівняних"]
        lines.append(
            f"| {platform} | {block['змінили_мітку_усього']} | "
            f"{block['стали_позитивними']} | {block['стали_негативними']} | "
            f"{_num(share)} |"
        )
    lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="перезаписати мітку")
    args = parser.parse_args()

    if OUT_PATH.exists() and not args.force:
        raise SystemExit(
            f"{OUT_PATH.relative_to(PROJECT_ROOT)} уже існує.\n"
            "Перегенерація без --force заборонена: мітка, на якій щось "
            "пораховано, мінятися не може."
        )

    targets, summary = build()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    targets.to_parquet(OUT_PATH, index=False)

    META_PATH.write_text(
        json.dumps(
            {
                "джерело": "ADR-005, четверте доповнення від 16 вересня 2026 року",
                "створено": datetime.now().isoformat(timespec="seconds"),
                "git_commit": git_commit(),
                "перцентиль": PERCENTILE,
                "правило": (
                    "y = 1, якщо кількість реакцій строго більша за перцентиль "
                    "по всій платформі; запасні правила 1 і 2 не застосовуються"
                ),
                "платформи": summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = report_lines(summary, len(targets))
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"мітка: {OUT_PATH.relative_to(PROJECT_ROOT)} — {len(targets)} рядків")
    print(f"пороги: {META_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
