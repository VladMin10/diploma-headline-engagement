"""Статистична перевірка EXP-007 за правилом ADR-006.

Запуск:
    .venv/bin/python -m src.experiments.ablation_stats

Нічого не навчає: усе рахується з `data/processed/ablation_results.parquet`.
Матриця метрик збережена без усереднення саме для того, щоб цей крок можна
було виконати без повторного прогону.

**Три способи на тих самих двадцяти парних значеннях** (ADR-006, розділ
«Рішення», і п'яте доповнення).

Перший — критерій Вілкоксона зі знаковими рангами, двосторонній. Його
рекомендує розділ 9 документа кафедри як практичний варіант для невеликої
кількості парних результатів по фолдах.

Другий — поправлений t-критерій для повторюваної крос-валідації
(Nadeau, Bengio, 2003; позиція 20 в `docs/references.md`). Звичайний
t-критерій припускає незалежність спостережень, а двадцять різниць
не є незалежними: чотири повтори проганяють ті самі 5 055 рядків у різних
перестановках, тому навчальні частини розбиттів перекриваються. Поправка
збільшує оцінку дисперсії середньої різниці на відношення розміру
перевірочної частини до навчальної:

    t = mean(d) / sqrt((1/n + n_test/n_train) · var(d)),   df = n − 1

При п'яти фолдах перевірочна частина дорівнює чверті навчальної, тому
відношення становить 1/4. Без поправки множник був би 1/n, тобто 1/20;
з поправкою він стає 1/20 + 1/4, тобто в шість разів більший, і критерій
відповідно консервативніший.

Третій — 95% перцентильний бутстреп-інтервал середньої різниці,
1 000 ітерацій (CLAUDE.md розділ 6, пункт 7).

**Правило рішення.** Приріст називається підтвердженим лише тоді, коли
обидва критерії дають p менше 0,05 **і** довірчий інтервал не містить нуля.
Критерій двосторонній (перше доповнення ADR-006), тому правило однаково
застосовне і до додатних, і до від'ємних різниць.

**Висновків тут немає.** Модуль дає числа і результат застосування правила.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = PROJECT_ROOT / "data" / "processed" / "ablation_results.parquet"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "ablation_stats.parquet"
REPORT_PATH = PROJECT_ROOT / "reports" / "tables" / "ablation_stats.md"

METRIC = "F1_підібраний"
MAIN_PERCENTILE = "25"
PLATFORM_ORDER = ["Facebook", "GooglePlus", "LinkedIn"]
MODEL_ORDER = ["LogisticRegression", "RandomForest", "LightGBM", "CatBoost"]
PERCENTILE_ORDER = ["25", "20", "33"]
COMPARISONS = ["F1", "F2", "F3"]

ALPHA = 0.05
BOOTSTRAP_ITERATIONS = 1000
RANDOM_STATE = 42

# Відношення розміру перевірочної частини до навчальної при п'яти фолдах:
# перевірочна частина дорівнює одній п'ятій, навчальна — чотирьом п'ятим.
TEST_TRAIN_RATIO = 1 / 4


def corrected_t_test(differences: np.ndarray) -> tuple[float, float]:
    """Поправлений t-критерій Nadeau—Bengio. Повертає t і двостороннє p.

    Поправка потрібна саме через перекриття навчальних частин: без неї
    оцінка дисперсії середньої різниці занижена, і p виходить оптимістичним.
    """
    n = differences.size
    mean = float(differences.mean())
    variance = float(differences.var(ddof=1))
    if variance == 0:
        return (np.inf if mean != 0 else 0.0), (0.0 if mean != 0 else 1.0)
    denominator = np.sqrt((1 / n + TEST_TRAIN_RATIO) * variance)
    t_value = mean / denominator
    p_value = 2 * stats.t.sf(abs(t_value), df=n - 1)
    return float(t_value), float(p_value)


def wilcoxon_p(differences: np.ndarray) -> float:
    """Двостороннє p критерію Вілкоксона зі знаковими рангами.

    Якщо всі різниці дорівнюють нулю, критерій не визначений; повертається
    одиниця, тобто відсутність підстав відхилити нульову гіпотезу.
    """
    if np.all(differences == 0):
        return 1.0
    return float(stats.wilcoxon(differences, alternative="two-sided").pvalue)


def bootstrap_interval(differences: np.ndarray) -> tuple[float, float]:
    """95% перцентильний інтервал середньої різниці."""
    generator = np.random.default_rng(RANDOM_STATE)
    n = differences.size
    indices = generator.integers(0, n, size=(BOOTSTRAP_ITERATIONS, n))
    means = differences[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def paired_differences(
    results: pd.DataFrame,
    platform: str,
    model: str,
    percentile: str,
    arm: str,
) -> np.ndarray:
    """Двадцять парних різниць набору проти F₀ на однакових розбиттях."""
    block = results[
        (results["платформа"] == platform)
        & (results["модель"] == model)
        & (results["перцентиль"] == percentile)
    ]
    base = block[block["набір"] == "F0"].set_index(["repeat", "fold"])[METRIC]
    other = block[block["набір"] == arm].set_index(["repeat", "fold"])[METRIC]
    return (other - base).sort_index().to_numpy(dtype=np.float64)


def compute(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for percentile in PERCENTILE_ORDER:
        for platform in PLATFORM_ORDER:
            for model in MODEL_ORDER:
                for arm in COMPARISONS:
                    diff = paired_differences(
                        results, platform, model, percentile, arm
                    )
                    p_wilcoxon = wilcoxon_p(diff)
                    t_value, p_corrected = corrected_t_test(diff)
                    low, high = bootstrap_interval(diff)
                    confirmed = (
                        p_wilcoxon < ALPHA
                        and p_corrected < ALPHA
                        and not (low <= 0 <= high)
                    )
                    rows.append(
                        {
                            "перцентиль": percentile,
                            "платформа": platform,
                            "модель": model,
                            "порівняння": f"{arm} − F0",
                            "n": int(diff.size),
                            "середня_різниця": float(diff.mean()),
                            "p_Вілкоксона": p_wilcoxon,
                            "t_поправлений": t_value,
                            "p_поправлений": p_corrected,
                            "ДІ_низ": low,
                            "ДІ_верх": high,
                            "інтервал_містить_нуль": bool(low <= 0 <= high),
                            "підтверджено": bool(confirmed),
                        }
                    )
    return pd.DataFrame(rows)


def _num(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def _signed(value: float, digits: int = 4) -> str:
    return f"{value:+.{digits}f}".replace(".", ",")


def _p(value: float) -> str:
    if value < 0.0001:
        return "< 0,0001"
    return f"{value:.4f}".replace(".", ",")


def table(stats_frame: pd.DataFrame, percentile: str) -> list[str]:
    subset = stats_frame[stats_frame["перцентиль"] == percentile]
    lines = [
        "| Платформа | Модель | Порівняння | Середня різниця | p Вілкоксона | p поправлений | 95% ДІ | Правило ADR-006 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in subset.itertuples(index=False):
        verdict = "підтверджено" if row.підтверджено else "ні"
        lines.append(
            f"| {row.платформа} | {row.модель} | {row.порівняння} | "
            f"{_signed(row.середня_різниця)} | {_p(row.p_Вілкоксона)} | "
            f"{_p(row.p_поправлений)} | "
            f"[{_signed(row.ДІ_низ)}; {_signed(row.ДІ_верх)}] | {verdict} |"
        )
    return lines


def disagreements(stats_frame: pd.DataFrame) -> pd.DataFrame:
    """Комбінації, де два критерії дають різні відповіді при рівні 0,05."""
    wilcoxon_yes = stats_frame["p_Вілкоксона"] < ALPHA
    corrected_yes = stats_frame["p_поправлений"] < ALPHA
    return stats_frame[wilcoxon_yes != corrected_yes]


def report_lines(stats_frame: pd.DataFrame) -> list[str]:
    confirmed = stats_frame[stats_frame["підтверджено"]]
    positive = confirmed[confirmed["середня_різниця"] > 0]
    negative = confirmed[confirmed["середня_різниця"] < 0]
    diverge = disagreements(stats_frame)

    lines = [
        "# EXP-007: статистична перевірка абляції",
        "",
        "Згенеровано `src/experiments/ablation_stats.py` з "
        "`data/processed/ablation_results.parquet`.",
        "Навчання тут немає: усе рахується з уже збережених метрик.",
        "",
        "Головна величина — F1 при підібраному на навчальній частині порозі "
        "(п'яте доповнення ADR-006).",
        "Три способи на тих самих двадцяти парних значеннях: критерій "
        "Вілкоксона зі знаковими рангами,",
        "поправлений t-критерій Nadeau—Bengio для повторюваної крос-валідації "
        "і 95% бутстреп-інтервал",
        f"({BOOTSTRAP_ITERATIONS} ітерацій). Критерій двосторонній.",
        "",
        f"**Правило ADR-006:** різниця підтверджена, якщо обидва p менші "
        f"за {_num(ALPHA, 2)} і інтервал не містить нуля.",
        "",
        "**Висновків тут немає.** Наведені числа і результат застосування "
        "правила.",
        "",
        "---",
        "",
        "## Зведення",
        "",
        f"Комбінацій усього: {len(stats_frame)} "
        f"(3 платформи × 4 моделі × 3 пороги мітки × 3 порівняння).",
        "",
        "| Результат правила | Комбінацій |",
        "|---|---|",
        f"| Підтверджено, різниця додатна | **{len(positive)}** |",
        f"| Підтверджено, різниця від'ємна | **{len(negative)}** |",
        f"| Не підтверджено | {len(stats_frame) - len(confirmed)} |",
        "",
    ]

    if len(positive):
        lines += ["Підтверджені додатні різниці:", ""]
        for row in positive.itertuples(index=False):
            lines.append(
                f"- верхні {row.перцентиль}%, {row.платформа}, {row.модель}, "
                f"{row.порівняння}: {_signed(row.середня_різниця)}"
            )
        lines.append("")
    else:
        lines += ["Підтверджених додатних різниць немає.", ""]

    lines += [
        f"### Узгодженість двох критеріїв",
        "",
        f"Критерії дають різні відповіді при рівні {_num(ALPHA, 2)} "
        f"у **{len(diverge)}** комбінаціях із {len(stats_frame)}.",
        "",
    ]
    if len(diverge):
        lines += [
            "| Перцентиль | Платформа | Модель | Порівняння | Середня різниця | p Вілкоксона | p поправлений |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in diverge.itertuples(index=False):
            lines.append(
                f"| верхні {row.перцентиль}% | {row.платформа} | {row.модель} | "
                f"{row.порівняння} | {_signed(row.середня_різниця)} | "
                f"{_p(row.p_Вілкоксона)} | {_p(row.p_поправлений)} |"
            )
        only_wilcoxon = int(
            ((diverge["p_Вілкоксона"] < ALPHA) & (diverge["p_поправлений"] >= ALPHA)).sum()
        )
        only_corrected = len(diverge) - only_wilcoxon
        lines += [
            "",
            f"З них у {only_wilcoxon} комбінаціях Вілкоксон відхиляє нульову "
            f"гіпотезу, а поправлений критерій ні;",
            f"у {only_corrected} — навпаки. Поправка на перекриття навчальних "
            "частин збільшує оцінку",
            "дисперсії в шість разів, тому поправлений критерій "
            "консервативніший за побудовою.",
            "Правило ADR-006 вимагає згоди обох, тому такі комбінації "
            "підтвердженими не вважаються.",
            "",
            f"Усього значущих за Вілкоксоном: "
            f"{int((stats_frame['p_Вілкоксона'] < ALPHA).sum())} із "
            f"{len(stats_frame)}; за поправленим критерієм: "
            f"{int((stats_frame['p_поправлений'] < ALPHA).sum())}.",
            "",
        ]

    for percentile in PERCENTILE_ORDER:
        mark = " (основний)" if percentile == MAIN_PERCENTILE else ""
        lines += [
            "---",
            "",
            f"## Поріг мітки: верхні {percentile}%{mark}",
            "",
        ]
        lines += table(stats_frame, percentile)
        lines.append("")

    return lines


def main() -> None:
    results = pd.read_parquet(RESULTS_PATH)
    stats_frame = compute(results)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    stats_frame.to_parquet(OUT_PATH, index=False)

    lines = report_lines(stats_frame)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"\nтаблиця: {OUT_PATH.relative_to(PROJECT_ROOT)} — "
          f"{len(stats_frame)} рядків")
    print(f"звіт: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
