"""EXP-012: статистична перевірка E7 за правилом ADR-006.

Запуск:
    .venv/bin/python -m src.experiments.fuzzy_strength_stats

Нічого не навчає: усе рахується з
`data/processed/fuzzy_strength_results.parquet`, збереженого без усереднення.

Головна величина — парна різниця F1 при підібраному на навчальній частині
порозі, у напрямку **F_high − N_strong**. Три способи на тих самих двадцяти
парних значеннях: критерій Вілкоксона зі знаковими рангами, поправлений
t-критерій Nadeau—Bengio і 95% перцентильний бутстреп-інтервал. Критерій
двосторонній. Функції перевірки імпортуються з `ablation_stats`, щоб правило
було тим самим кодом, а не його копією.

Окрім F1 рахуються різниці за ROC-AUC і PR-AUC — для перевірки узгодженості
знаків.

Окремо рахуються дві властивості корпусу, зафіксовані в картці до прогону:
скільки рядків мають хоча б одну оцінку рівня 3 і на скількох рядках
виконується точна рівність `N_strong = 2 · F_high`. Перше число є верхньою
межею того, де дві ознаки взагалі можуть розрізнятися нетривіально.

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
from src.features.derived import FEATURE_ORDER

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = PROJECT_ROOT / "data" / "processed" / "fuzzy_strength_results.parquet"
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features_derived.parquet"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "fuzzy_strength_stats.parquet"
REPORT_PATH = PROJECT_ROOT / "reports" / "tables" / "fuzzy_strength.md"

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
    """Двадцять парних різниць F_high − N_strong на однакових розбиттях."""
    block = results[
        (results["платформа"] == platform) & (results["модель"] == model)
    ]
    hard = block[block["варіант"] == "N_strong"].set_index(["repeat", "fold"])[metric]
    fuzzy = block[block["варіант"] == "F_high"].set_index(["repeat", "fold"])[metric]
    return (fuzzy - hard).sort_index().to_numpy(dtype=np.float64)


def corpus_facts() -> dict:
    """Дві властивості корпусу, названі в картці до прогону."""
    frame = pd.read_parquet(FEATURES_PATH)
    base = frame[list(FEATURE_ORDER)]
    has_three = (base == 3).any(axis=1)
    exact = frame["N_strong"] == 2 * frame["F_high"]
    per_feature = {
        name: (int((frame[name] == 3).sum()), int((frame[name] == 2).sum()))
        for name in FEATURE_ORDER
    }
    return {
        "усього": int(len(frame)),
        "з_трійкою": int(has_three.sum()),
        "частка_з_трійкою": float(has_three.mean()),
        "точна_рівність": int(exact.sum()),
        "по_ознаках": per_feature,
    }


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
                "порівняння": "F_high − N_strong",
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
        "| Платформа | Модель | F1 N_strong | F1 F_high | ROC-AUC N_strong "
        "| ROC-AUC F_high | PR-AUC N_strong | PR-AUC F_high |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for platform in PLATFORM_ORDER:
        for model in MODEL_ORDER:
            hard = grouped.loc[(platform, model, "N_strong")]
            fuzzy = grouped.loc[(platform, model, "F_high")]
            lines.append(
                f"| {TITLE_UA[platform]} | {model} "
                f"| {_num(hard[METRIC])} | {_num(fuzzy[METRIC])} "
                f"| {_num(hard['ROC_AUC'])} | {_num(fuzzy['ROC_AUC'])} "
                f"| {_num(hard['PR_AUC'])} | {_num(fuzzy['PR_AUC'])} |"
            )
    return lines


def report_lines(results: pd.DataFrame, stats_frame: pd.DataFrame) -> list[str]:
    confirmed = stats_frame[stats_frame["підтверджено"]]
    facts = corpus_facts()

    lines = [
        "# EXP-012: нечітке представлення сили прояву (E7)",
        "",
        "Згенеровано `src/experiments/fuzzy_strength_stats.py` "
        "з `data/processed/fuzzy_strength_results.parquet`.",
        "Навчання тут немає: усе рахується з уже збережених метрик.",
        "",
        "Порівнюються два варіанти набору F₁, що відрізняються рівно однією "
        "ознакою:",
        "жорсткий поріг `N_strong` проти нечіткої `F_high`. Сім базових оцінок "
        "і решта восьми",
        "профільних ознак однакові.",
        "",
        "Головна величина — F1 при підібраному на навчальній частині порозі, "
        "у напрямку **F_high − N_strong**.",
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
        "## Де дві ознаки взагалі можуть розрізнятися",
        "",
        f"Хоча б одну оцінку рівня 3 має **{facts['з_трійкою']} рядків "
        f"з {facts['усього']}**, тобто {_num(facts['частка_з_трійкою'])} корпусу.",
        "Це верхня межа того, де дві ознаки можуть розрізнятися нетривіально.",
        "",
        f"На решті {facts['усього'] - facts['з_трійкою']} рядків виконується "
        "точна рівність `N_strong = 2 · F_high`:",
        "якщо трійок немає, нечітка ознака дорівнює половині кількості двійок. "
        "Рівність",
        f"перевірено на даних — вона виконується рівно на "
        f"{facts['точна_рівність']} рядках.",
        "Для дерев монотонне перетворення однієї ознаки не змінює нічого; "
        "для логістичної",
        "регресії зі стандартизацією сталий множник не змінює нічого теж.",
        "",
        "Розподіл рівнів 2 і 3 по семи базових ознаках:",
        "",
        "| Ознака | Рівень 3 | Рівень 2 |",
        "|---|---|---|",
    ]
    for name, (three, two) in facts["по_ознаках"].items():
        mark = f"**{three}**" if three == 0 else str(three)
        lines.append(f"| {name} | {mark} | {two} |")
    lines += [
        "",
        "Ознака `authority` не має рівня 3 жодного разу.",
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
        "## Парні різниці F_high − N_strong",
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
    facts = corpus_facts()
    print(f"рядків у матриці метрик: {len(results)}")
    print(f"комбінацій перевірено: {len(stats_frame)}; "
          f"підтверджено: {int(stats_frame['підтверджено'].sum())}")
    print(f"рядків з хоча б однією оцінкою 3: {facts['з_трійкою']} "
          f"з {facts['усього']}")
    print(f"збережено: {OUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"звіт: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
