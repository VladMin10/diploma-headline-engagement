"""EXP-011: статистична перевірка E6 за правилом ADR-006.

Запуск:
    .venv/bin/python -m src.experiments.interaction_form_stats

Нічого не навчає: усе рахується з
`data/processed/interaction_form_results.parquet`, збереженого без усереднення.

Головна величина — парна різниця F1 при підібраному на навчальній частині
порозі, у напрямку **min − product**. Три способи на тих самих двадцяти
парних значеннях, як у EXP-007: критерій Вілкоксона зі знаковими рангами,
поправлений t-критерій Nadeau—Bengio і 95% перцентильний бутстреп-інтервал.
Критерій двосторонній. Функції перевірки імпортуються з `ablation_stats`,
щоб правило було тим самим кодом, а не його копією.

Окрім F1 рахуються різниці за ROC-AUC і PR-AUC — для перевірки узгодженості
знаків, як у EXP-008.

Окремо рахується, на скількох рядках корпусу два варіанти дають різні
значення хоча б в одній із п'яти пар. Це верхня межа того, де різниця між
варіантами взагалі можлива.

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
from src.features.derived import PAIRS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = PROJECT_ROOT / "data" / "processed" / "interaction_form_results.parquet"
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features_derived.parquet"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "interaction_form_stats.parquet"
REPORT_PATH = PROJECT_ROOT / "reports" / "tables" / "interaction_form.md"

METRIC = "F1_підібраний"
EXTRA_METRICS = ["ROC_AUC", "PR_AUC"]
TITLE_UA = {"Facebook": "Facebook", "GooglePlus": "Google+", "LinkedIn": "LinkedIn"}


def _num(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def _signed(value: float, digits: int = 4) -> str:
    return f"{value:+.{digits}f}".replace(".", ",")


def _p(value: float) -> str:
    return "< 0,0001" if value < 0.0001 else f"{value:.4f}".replace(".", ",")


def paired(results: pd.DataFrame, platform: str, model: str,
           metric: str) -> np.ndarray:
    """Двадцять парних різниць min − product на однакових розбиттях."""
    block = results[
        (results["платформа"] == platform) & (results["модель"] == model)
    ]
    product = block[block["варіант"] == "product"].set_index(["repeat", "fold"])[metric]
    minimum = block[block["варіант"] == "min"].set_index(["repeat", "fold"])[metric]
    return (minimum - product).sort_index().to_numpy(dtype=np.float64)


def differing_rows() -> tuple[int, int, dict[str, float]]:
    """Скільки рядків корпусу дають різні значення хоч в одній із п'яти пар."""
    frame = pd.read_parquet(FEATURES_PATH)
    per_pair = {}
    any_diff = np.zeros(len(frame), dtype=bool)
    for left, right in PAIRS:
        diff = (frame[f"prod_{left}_{right}"] != frame[f"min_{left}_{right}"])
        per_pair[f"{left} × {right}"] = float(diff.mean())
        any_diff |= diff.to_numpy()
    return int(any_diff.sum()), int(len(frame)), per_pair


def compute(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for platform in PLATFORM_ORDER:
        for model in MODEL_ORDER:
            diff = paired(results, platform, model, METRIC)
            p_wilcoxon = wilcoxon_p(diff)
            t_value, p_corrected = corrected_t_test(diff)
            low, high = bootstrap_interval(diff)
            row = {
                "платформа": platform,
                "модель": model,
                "порівняння": "min − product",
                "n": int(diff.size),
                "середня_різниця": float(diff.mean()),
                "додатних": int((diff > 0).sum()),
                "нульових": int((diff == 0).sum()),
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
                    paired(results, platform, model, metric).mean()
                )
            signs = {np.sign(row["середня_різниця"])} | {
                np.sign(row[f"різниця_{metric}"]) for metric in EXTRA_METRICS
            }
            row["знаки_збігаються"] = bool(len(signs) == 1)
            rows.append(row)
    return pd.DataFrame(rows)


def means_table(results: pd.DataFrame) -> list[str]:
    grouped = results.groupby(["платформа", "модель", "варіант"])[
        [METRIC] + EXTRA_METRICS
    ].mean()
    lines = [
        "| Платформа | Модель | F1 product | F1 min | ROC-AUC product "
        "| ROC-AUC min | PR-AUC product | PR-AUC min |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for platform in PLATFORM_ORDER:
        for model in MODEL_ORDER:
            product = grouped.loc[(platform, model, "product")]
            minimum = grouped.loc[(platform, model, "min")]
            lines.append(
                f"| {TITLE_UA[platform]} | {model} "
                f"| {_num(product[METRIC])} | {_num(minimum[METRIC])} "
                f"| {_num(product['ROC_AUC'])} | {_num(minimum['ROC_AUC'])} "
                f"| {_num(product['PR_AUC'])} | {_num(minimum['PR_AUC'])} |"
            )
    return lines


def report_lines(results: pd.DataFrame, stats_frame: pd.DataFrame) -> list[str]:
    confirmed = stats_frame[stats_frame["підтверджено"]]
    rows_diff, rows_total, per_pair = differing_rows()

    lines = [
        "# EXP-011: спосіб формування парних взаємодій (E6)",
        "",
        "Згенеровано `src/experiments/interaction_form_stats.py` "
        "з `data/processed/interaction_form_results.parquet`.",
        "Навчання тут немає: усе рахується з уже збережених метрик.",
        "",
        "Порівнюються два варіанти набору F₂, що відрізняються лише формулою "
        "взаємодії:",
        "`prod_a_b = a · b` проти `min_a_b = min(a, b)`. Сім базових оцінок "
        "і склад п'яти пар однакові.",
        "",
        "Головна величина — F1 при підібраному на навчальній частині порозі, "
        "у напрямку **min − product**.",
        "Три способи на тих самих двадцяти парних значеннях: критерій "
        "Вілкоксона зі знаковими рангами,",
        "поправлений t-критерій Nadeau—Bengio і 95% бутстреп-інтервал "
        f"({BOOTSTRAP_ITERATIONS} ітерацій).",
        "Критерій двосторонній.",
        "",
        f"**Правило ADR-006:** різниця підтверджена, якщо обидва p менші "
        f"за {_num(ALPHA, 2)} і інтервал не містить нуля.",
        "",
        "**Висновків тут немає.** Наведені числа і результат застосування "
        "правила.",
        "",
        "---",
        "",
        "## На скількох рядках варіанти взагалі різняться",
        "",
        f"Два варіанти дають різні значення хоча б в одній із п'яти пар "
        f"у **{rows_diff} рядках з {rows_total}**, тобто в "
        f"{_num(rows_diff / rows_total)} корпусу.",
        "Це верхня межа того, де різниця між варіантами взагалі можлива: "
        "на решті рядків",
        f"({rows_total - rows_diff}) обидва набори ознак тотожні.",
        "",
        "| Пара | Частка рядків із різними значеннями |",
        "|---|---|",
    ]
    for pair, share in per_pair.items():
        lines.append(f"| {pair} | {_num(share)} |")
    lines += [
        "",
        "Нулі в обох варіантах стоять у тих самих рядках: добуток і мінімум "
        "двох невід'ємних",
        "цілих дорівнюють нулю за тієї самої умови. Різниця можлива лише там, "
        "де обидві оцінки",
        "ненульові й не рівні між собою.",
        "",
        "---",
        "",
        "## Середні метрики обох варіантів",
        "",
        "Середнє по двадцяти розбиттях, основний поріг мітки — верхні 25% "
        "усередині видання.",
        "",
    ]
    lines += means_table(results)
    lines += [
        "",
        "---",
        "",
        "## Парні різниці min − product",
        "",
        "| Платформа | Модель | Середня різниця | Додатних із 20 | Нульових "
        "| p Вілкоксона | p поправлений | 95% ДІ | Правило ADR-006 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in stats_frame.itertuples(index=False):
        verdict = "**підтверджено**" if row.підтверджено else "ні"
        lines.append(
            f"| {TITLE_UA[row.платформа]} | {row.модель} "
            f"| {_signed(row.середня_різниця)} | {row.додатних} | {row.нульових} "
            f"| {_p(row.p_Вілкоксона)} | {_p(row.p_поправлений)} "
            f"| [{_signed(row.ДІ_низ)}; {_signed(row.ДІ_верх)}] | {verdict} |"
        )
    lines += [
        "",
        f"Підтверджено {len(confirmed)} різниць із {len(stats_frame)}.",
        "",
        "---",
        "",
        "## Узгодженість знаків",
        "",
        "| Платформа | Модель | ΔF1 | ΔROC-AUC | ΔPR-AUC | Знаки |",
        "|---|---|---|---|---|---|",
    ]
    for row in stats_frame.itertuples(index=False):
        mark = "збігаються" if row.знаки_збігаються else "**розходяться**"
        lines.append(
            f"| {TITLE_UA[row.платформа]} | {row.модель} "
            f"| {_signed(row.середня_різниця)} "
            f"| {_signed(row.різниця_ROC_AUC)} | {_signed(row.різниця_PR_AUC)} "
            f"| {mark} |"
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
    confirmed = int(stats_frame["підтверджено"].sum())
    rows_diff, rows_total, _ = differing_rows()
    print(f"рядків у матриці метрик: {len(results)}")
    print(f"комбінацій перевірено: {len(stats_frame)}; підтверджено: {confirmed}")
    print(f"варіанти різняться на {rows_diff} рядках з {rows_total}")
    print(f"збережено: {OUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"звіт: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
