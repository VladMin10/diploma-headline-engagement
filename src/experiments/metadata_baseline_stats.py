"""EXP-013: статистична перевірка базової лінії на метаданих.

Запуск:
    .venv/bin/python -m src.experiments.metadata_baseline_stats

Нічого не навчає: усе рахується з
`data/processed/metadata_baseline_results.parquet`, збереженого без
усереднення.

Два порівняння на тих самих двадцяти парних значеннях:

    F0 − M        чи прогнозує зміст заголовка краще за метадані
    M+F0 − M      чи додають характеристики точності понад метадані;
                  це пряма відповідь на Q1 у класифікаційній постановці

Головна величина — F1 при підібраному на навчальній частині порозі.
Три способи перевірки і правило згоди імпортуються з `ablation_stats`,
щоб правило було тим самим кодом, а не його копією. Критерій двосторонній.

Окрім F1 рахуються різниці за ROC-AUC і PR-AUC — для перевірки узгодженості
знаків.

**Висновків тут немає.** Модуль дає числа і результат застосування правила.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.ablation_stats import (
    ALPHA,
    BOOTSTRAP_ITERATIONS,
    MODEL_ORDER,
    PLATFORM_ORDER,
    bootstrap_interval,
    corrected_t_test,
    wilcoxon_p,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = PROJECT_ROOT / "data" / "processed" / "metadata_baseline_results.parquet"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "metadata_baseline_stats.parquet"
REPORT_PATH = PROJECT_ROOT / "reports" / "tables" / "metadata_baseline.md"

METRIC = "F1_підібраний"
EXTRA_METRICS = ["ROC_AUC", "PR_AUC"]
BASE = "M"
COMPARISONS = [
    ("F0", "F0 − M", "чи прогнозує зміст краще за метадані"),
    ("M+F0", "M+F0 − M", "Q1 у класифікаційній постановці"),
]
VARIANT_ORDER = ["M", "F0", "M+F0"]
TITLE_UA = {"Facebook": "Facebook", "GooglePlus": "Google+", "LinkedIn": "LinkedIn"}


def _num(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def _signed(value: float, digits: int = 4) -> str:
    return f"{value:+.{digits}f}".replace(".", ",")


def _p(value: float) -> str:
    return "< 0,0001" if value < 0.0001 else f"{value:.4f}".replace(".", ",")


def paired(results: pd.DataFrame, platform: str, model: str, arm: str,
           metric: str) -> np.ndarray:
    """Двадцять парних різниць набору проти M на однакових розбиттях."""
    block = results[
        (results["платформа"] == platform) & (results["модель"] == model)
    ]
    base = block[block["варіант"] == BASE].set_index(["repeat", "fold"])[metric]
    other = block[block["варіант"] == arm].set_index(["repeat", "fold"])[metric]
    return (other - base).sort_index().to_numpy(dtype=np.float64)


def compute(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for platform in PLATFORM_ORDER:
        for model in MODEL_ORDER:
            for arm, label, purpose in COMPARISONS:
                diff = paired(results, platform, model, arm, METRIC)
                p_wilcoxon = wilcoxon_p(diff)
                t_value, p_corrected = corrected_t_test(diff)
                low, high = bootstrap_interval(diff)
                row = {
                    "платформа": platform,
                    "модель": model,
                    "порівняння": label,
                    "призначення": purpose,
                    "n": int(diff.size),
                    "середня_різниця": float(diff.mean()),
                    "додатних": int((diff > 0).sum()),
                    "p_Вілкоксона": p_wilcoxon,
                    "t_поправлений": t_value,
                    "p_поправлений": p_corrected,
                    "ДІ_низ": low,
                    "ДІ_верх": high,
                    "інтервал_містить_нуль": bool(low <= 0 <= high),
                    "підтверджено": bool(
                        p_wilcoxon < ALPHA and p_corrected < ALPHA
                        and not (low <= 0 <= high)
                    ),
                }
                for metric in EXTRA_METRICS:
                    row[f"різниця_{metric}"] = float(
                        paired(results, platform, model, arm, metric).mean()
                    )
                signs = {np.sign(row["середня_різниця"])} | {
                    np.sign(row[f"різниця_{metric}"]) for metric in EXTRA_METRICS
                }
                row["знаки_збігаються"] = bool(len(signs) == 1)
                rows.append(row)
    return pd.DataFrame(rows)


def means_table(results: pd.DataFrame) -> list[str]:
    grouped = results.groupby(["платформа", "модель", "варіант"])[
        [METRIC, "ROC_AUC", "PR_AUC", "n_ознак"]
    ].mean()
    lines = [
        "| Платформа | Модель | Набір | Ознак | F1 підібраний | ROC-AUC | PR-AUC |",
        "|---|---|---|---|---|---|---|",
    ]
    for platform in PLATFORM_ORDER:
        for model in MODEL_ORDER:
            for variant in VARIANT_ORDER:
                row = grouped.loc[(platform, model, variant)]
                lines.append(
                    f"| {TITLE_UA[platform]} | {model} | {variant} "
                    f"| {int(round(row['n_ознак']))} | {_num(row[METRIC])} "
                    f"| {_num(row['ROC_AUC'])} | {_num(row['PR_AUC'])} |"
                )
    return lines


def report_lines(results: pd.DataFrame, stats_frame: pd.DataFrame) -> list[str]:
    confirmed = stats_frame[stats_frame["підтверджено"]]
    lines = [
        "# EXP-013: базова лінія на метаданих у класифікаційній лінії",
        "",
        "Згенеровано `src/experiments/metadata_baseline_stats.py` "
        "з `data/processed/metadata_baseline_results.parquet`.",
        "Навчання тут немає: усе рахується з уже збережених метрик.",
        "",
        "Три набори на тих самих двадцяти розбиттях: **M** — метадані "
        "публікації в складі рівня T1",
        "регресійної лінії (видання, тема, година, день тижня, місяць); "
        "**F₀** — сім змістових",
        "характеристик; **M+F₀** — разом.",
        "",
        "Головна величина — F1 при підібраному на навчальній частині порозі. "
        "Три способи на тих самих",
        "двадцяти парних значеннях: критерій Вілкоксона зі знаковими рангами, "
        "поправлений t-критерій",
        f"Nadeau—Bengio і 95% бутстреп-інтервал ({BOOTSTRAP_ITERATIONS} "
        "ітерацій). Критерій двосторонній.",
        "",
        f"**Правило ADR-006:** різниця підтверджена, якщо обидва p менші "
        f"за {_num(ALPHA, 2)} і інтервал не містить нуля.",
        "",
        "**Висновків тут немає.** Наведені числа і результат застосування "
        "правила.",
        "",
        "---",
        "",
        "## Середні метрики трьох наборів",
        "",
        "Середнє по двадцяти розбиттях, основний поріг мітки — верхні 25% "
        "усередині видання.",
        "Стовпчик «Ознак» — середня розмірність поданої моделі матриці: "
        "для логістичної регресії",
        "і випадкового лісу метадані подаються one-hot, для LightGBM "
        "і CatBoost — категоріями.",
        "",
    ]
    lines += means_table(results)
    lines += [
        "",
        "---",
        "",
        "## Парні різниці",
        "",
        "| Платформа | Модель | Порівняння | Середня різниця | Додатних із 20 "
        "| p Вілкоксона | p поправлений | 95% ДІ | Правило ADR-006 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in stats_frame.itertuples(index=False):
        verdict = "**підтверджено**" if row.підтверджено else "ні"
        lines.append(
            f"| {TITLE_UA[row.платформа]} | {row.модель} | {row.порівняння} "
            f"| {_signed(row.середня_різниця)} | {row.додатних} "
            f"| {_p(row.p_Вілкоксона)} | {_p(row.p_поправлений)} "
            f"| [{_signed(row.ДІ_низ)}; {_signed(row.ДІ_верх)}] | {verdict} |"
        )

    lines += ["", "### Зведення по двох порівняннях", "",
              "| Порівняння | Підтверджено | З них додатних | З них від'ємних |",
              "|---|---|---|---|"]
    for _, label, _purpose in COMPARISONS:
        block = stats_frame[stats_frame["порівняння"] == label]
        good = block[block["підтверджено"]]
        lines.append(
            f"| {label} | {len(good)} з {len(block)} "
            f"| {int((good['середня_різниця'] > 0).sum())} "
            f"| {int((good['середня_різниця'] < 0).sum())} |"
        )
    lines += [
        "",
        f"Підтверджено {len(confirmed)} різниць із {len(stats_frame)}.",
        "",
        "---",
        "",
        "## Узгодженість знаків",
        "",
        "| Платформа | Модель | Порівняння | ΔF1 | ΔROC-AUC | ΔPR-AUC | Знаки |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in stats_frame.itertuples(index=False):
        mark = "збігаються" if row.знаки_збігаються else "**розходяться**"
        lines.append(
            f"| {TITLE_UA[row.платформа]} | {row.модель} | {row.порівняння} "
            f"| {_signed(row.середня_різниця)} | {_signed(row.різниця_ROC_AUC)} "
            f"| {_signed(row.різниця_PR_AUC)} | {mark} |"
        )
    agree = int(stats_frame["знаки_збігаються"].sum())
    lines += [
        "",
        f"Знаки збігаються в {agree} комбінаціях із {len(stats_frame)}.",
        "",
    ]
    return lines


def main() -> None:
    results = pd.read_parquet(RESULTS_PATH)
    stats_frame = compute(results)
    stats_frame.to_parquet(OUT_PATH, index=False)
    REPORT_PATH.write_text(
        "\n".join(report_lines(results, stats_frame)), encoding="utf-8"
    )
    print(f"рядків у матриці метрик: {len(results)}")
    for _, label, _purpose in COMPARISONS:
        block = stats_frame[stats_frame["порівняння"] == label]
        good = block[block["підтверджено"]]
        print(f"  {label:12s} підтверджено {len(good)} з {len(block)}; "
              f"середня різниця по дванадцяти комбінаціях "
              f"{block['середня_різниця'].mean():+.4f}")
    print(f"збережено: {OUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"звіт: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
