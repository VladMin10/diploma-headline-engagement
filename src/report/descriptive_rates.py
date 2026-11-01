"""Описова статистика: частка потрапляння у верхню чверть за рівнями оцінок.

Запуск:
    .venv/bin/python -m src.report.descriptive_rates

Жодного навчання: це підрахунок часток по даних. Основна мітка, верхні 25%
усередині видання, три платформи.

Сенс таблиці в тому, що вона показує силу кожної характеристики **без моделі**.
Модель могла б знайти зв'язок, якого немає, або не знайти того, що є; проста
частка від такого підозри вільна.

**Довірчі інтервали — за Вілсоном.** Нормальне наближення для часток
непридатне там, де частка близька до нуля або одиниці, а тут такі клітинки
трапляються: рівень 3 ознаки `authority` не трапився жодного разу, рівень 3
ознаки `stakes` має лише двадцять два заголовки на весь корпус. Інтервал
Вілсона не виходить за межі від нуля до одиниці й не вироджується на малих
групах.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.features.copywriting import FEATURE_ORDER, load_labels

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TARGETS_PATH = PROJECT_ROOT / "data" / "processed" / "targets_classification.parquet"
REPORT_PATH = PROJECT_ROOT / "reports" / "tables" / "descriptive_rates.md"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "descriptive_rates.parquet"

PLATFORMS = {"Facebook": "fb", "GooglePlus": "gp", "LinkedIn": "li"}
PERCENTILE = "25"
LEVELS = (0, 1, 2, 3)
Z = 1.96  # 95% інтервал

LABELS = {
    "gap": "прогалина в інформації",
    "scale": "масштаб наслідків",
    "concrete": "конкретність предиката",
    "conflict": "явність конфлікту",
    "authority": "вага авторитету",
    "violation": "порушення очікування",
    "stakes": "проспективна невизначеність",
}


def wilson(successes: int, total: int, z: float = Z) -> tuple[float, float, float]:
    """Частка і 95% інтервал Вілсона. Повертає (частка, низ, верх)."""
    if total == 0:
        return float("nan"), float("nan"), float("nan")
    p = successes / total
    denominator = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denominator
    half = (z / denominator) * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2))
    return p, max(0.0, centre - half), min(1.0, centre + half)


def load_frame() -> pd.DataFrame:
    labels = load_labels().reset_index()
    targets = pd.read_parquet(TARGETS_PATH)
    return labels.merge(targets, on="IDLink", validate="one_to_one")


def rates_by_level(frame: pd.DataFrame) -> pd.DataFrame:
    """Частка позитивного класу для кожного рівня кожної ознаки."""
    rows = []
    for platform, short in PLATFORMS.items():
        column = f"y_{short}_{PERCENTILE}"
        for feature in FEATURE_ORDER:
            for level in LEVELS:
                block = frame[(frame[feature] == level) & frame[column].notna()]
                total = int(len(block))
                successes = int(block[column].sum()) if total else 0
                share, low, high = wilson(successes, total)
                rows.append(
                    {
                        "платформа": platform,
                        "ознака": feature,
                        "мітка": LABELS[feature],
                        "рівень": level,
                        "n": total,
                        "n_позитивних": successes,
                        "частка": share,
                        "ДІ_низ": low,
                        "ДІ_верх": high,
                    }
                )
    return pd.DataFrame(rows)


def strength(rates: pd.DataFrame) -> pd.DataFrame:
    """Різниця частки між найвищим наявним рівнем і рівнем 0."""
    rows = []
    for platform in PLATFORMS:
        block = rates[(rates["платформа"] == platform) & (rates["n"] > 0)]
        for feature in FEATURE_ORDER:
            own = block[block["ознака"] == feature].sort_values("рівень")
            base = own[own["рівень"] == 0].iloc[0]
            top = own.iloc[-1]
            rows.append(
                {
                    "платформа": platform,
                    "ознака": feature,
                    "мітка": LABELS[feature],
                    "найвищий_рівень": int(top["рівень"]),
                    "частка_0": float(base["частка"]),
                    "частка_вищий": float(top["частка"]),
                    "різниця": float(top["частка"] - base["частка"]),
                    "n_0": int(base["n"]),
                    "n_вищий": int(top["n"]),
                    "інтервали_перетинаються": bool(
                        top["ДІ_низ"] <= base["ДІ_верх"]
                        and base["ДІ_низ"] <= top["ДІ_верх"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def rates_by_total(frame: pd.DataFrame) -> pd.DataFrame:
    """Частка позитивного класу за квартилями суми семи оцінок."""
    total = frame[list(FEATURE_ORDER)].sum(axis=1)
    # Квартилі суми. Через дискретність суми межі можуть збігатися,
    # тому групи виходять нерівними за обсягом; це властивість даних.
    quartiles = pd.qcut(total, 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
    rows = []
    for platform, short in PLATFORMS.items():
        column = f"y_{short}_{PERCENTILE}"
        for quartile in quartiles.cat.categories:
            mask = (quartiles == quartile) & frame[column].notna()
            block = frame[mask]
            n = int(len(block))
            successes = int(block[column].sum())
            share, low, high = wilson(successes, n)
            rows.append(
                {
                    "платформа": platform,
                    "квартиль": str(quartile),
                    "сума_мінімум": int(total[mask].min()),
                    "сума_максимум": int(total[mask].max()),
                    "n": n,
                    "частка": share,
                    "ДІ_низ": low,
                    "ДІ_верх": high,
                }
            )
    return pd.DataFrame(rows)


def _num(value: float, digits: int = 4) -> str:
    if np.isnan(value):
        return "—"
    return f"{value:.{digits}f}".replace(".", ",")


def _signed(value: float, digits: int = 4) -> str:
    return f"{value:+.{digits}f}".replace(".", ",")


def report_lines(
    rates: pd.DataFrame, power: pd.DataFrame, totals: pd.DataFrame, frame: pd.DataFrame
) -> list[str]:
    base = {
        platform: float(frame[f"y_{short}_{PERCENTILE}"].mean())
        for platform, short in PLATFORMS.items()
    }

    lines = [
        "# Описова статистика: частка потрапляння у верхню чверть",
        "",
        "Згенеровано `src/report/descriptive_rates.py`. Жодного навчання: це "
        "підрахунок",
        "часток по даних. Основна мітка, верхні 25% усередині видання, "
        f"{len(frame)} анотованих заголовків.",
        "",
        "Довірчі інтервали — **за Вілсоном**, 95%. Нормальне наближення тут "
        "непридатне:",
        "частина клітинок має малий обсяг або частку біля нуля, а інтервал "
        "Вілсона",
        "не виходить за межі від нуля до одиниці й не вироджується на малих "
        "групах.",
        "",
        "Базова частка позитивного класу: "
        + "; ".join(f"{p} {_num(v)}" for p, v in base.items())
        + ".",
        "",
        "---",
        "",
        "## Сила ознак: різниця між найвищим рівнем і рівнем 0",
        "",
        "Ознаки відсортовані за різницею. Стовпчик «інтервали» показує, чи "
        "перетинаються",
        "довірчі інтервали двох часток: якщо так, різниця може бути наслідком "
        "випадковості.",
        "",
    ]
    for platform in PLATFORMS:
        block = power[power["платформа"] == platform].sort_values(
            "різниця", ascending=False
        )
        lines += [
            f"### {platform}",
            "",
            "| Ознака | Найвищий рівень | Частка при 0 | Частка при найвищому | Різниця | n при 0 | n при найвищому | Інтервали |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for row in block.itertuples(index=False):
            overlap = "перетинаються" if row.інтервали_перетинаються else "не перетинаються"
            lines.append(
                f"| {row.мітка} | {row.найвищий_рівень} | {_num(row.частка_0)} | "
                f"{_num(row.частка_вищий)} | {_signed(row.різниця)} | {row.n_0} | "
                f"{row.n_вищий} | {overlap} |"
            )
        lines.append("")

    lines += [
        "---",
        "",
        "## Частка за рівнями кожної ознаки",
        "",
        "Рівні, яких у корпусі немає, позначені прочерком.",
        "",
    ]
    for platform in PLATFORMS:
        block = rates[rates["платформа"] == platform]
        lines += [
            f"### {platform}",
            "",
            "| Ознака | Рівень | n | Частка | 95% ДІ за Вілсоном |",
            "|---|---|---|---|---|",
        ]
        for row in block.itertuples(index=False):
            if row.n == 0:
                lines.append(f"| {row.мітка} | {row.рівень} | 0 | — | — |")
                continue
            lines.append(
                f"| {row.мітка} | {row.рівень} | {row.n} | {_num(row.частка)} | "
                f"[{_num(row.ДІ_низ)}; {_num(row.ДІ_верх)}] |"
            )
        lines.append("")

    lines += [
        "---",
        "",
        "## Частка за квартилями суми семи оцінок",
        "",
        "Сума семи оцінок дискретна, тому межі квартилів збігаються і групи "
        "виходять",
        "нерівними за обсягом. Це властивість даних, а не помилка розбиття.",
        "",
        "| Платформа | Квартиль | Сума оцінок | n | Частка | 95% ДІ |",
        "|---|---|---|---|---|---|",
    ]
    for row in totals.itertuples(index=False):
        span = (
            f"{row.сума_мінімум}"
            if row.сума_мінімум == row.сума_максимум
            else f"{row.сума_мінімум}–{row.сума_максимум}"
        )
        lines.append(
            f"| {row.платформа} | {row.квартиль} | {span} | {row.n} | "
            f"{_num(row.частка)} | [{_num(row.ДІ_низ)}; {_num(row.ДІ_верх)}] |"
        )
    lines.append("")
    return lines


def main() -> None:
    frame = load_frame()
    rates = rates_by_level(frame)
    power = strength(rates)
    totals = rates_by_total(frame)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rates.to_parquet(OUT_PATH, index=False)

    lines = report_lines(rates, power, totals, frame)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"таблиця: {OUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"звіт: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
