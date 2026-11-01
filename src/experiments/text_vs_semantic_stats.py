"""Статистична перевірка EXP-008 за правилом ADR-006.

Запуск:
    .venv/bin/python -m src.experiments.text_vs_semantic_stats

Нічого не навчає: усе рахується з
`data/processed/text_vs_semantic_results.parquet`.

Три способи ті самі, що в EXP-007, і на тих самих двадцяти парних значеннях:
критерій Вілкоксона зі знаковими рангами, поправлений t-критерій
Nadeau—Bengio для повторюваної крос-валідації з відношенням перевірочної
частини до навчальної 1/4, і 95% перцентильний бутстреп-інтервал
на 1 000 ітерацій. Критерій двосторонній.

**Правило ADR-006.** Різниця підтверджена, якщо обидва p менші за 0,05
і довірчий інтервал не містить нуля.

**Чотири порівняння.** Три зафіксовані в картці до прогону і беруть за базу
чистий TF-IDF. Четверте, комбінація проти самого F₀, додане після перегляду
середніх і до цієї перевірки; воно позначене як розвідувальне і головного
не замінює. Підстава і статус описані в картці EXP-008, підрозділ
«Доповнення до переліку порівнянь».

**Висновків тут немає.** Модуль дає числа і результат застосування правила.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "text_vs_semantic_results.parquet"
)
TARGETS_PATH = PROJECT_ROOT / "data" / "processed" / "targets_classification.parquet"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "text_vs_semantic_stats.parquet"
REPORT_PATH = PROJECT_ROOT / "reports" / "tables" / "text_vs_semantic_stats.md"

PERCENTILE = "25"
PLATFORM_ORDER = ["Facebook", "GooglePlus", "LinkedIn"]
PLATFORM_SHORT = {"Facebook": "fb", "GooglePlus": "gp", "LinkedIn": "li"}
METRIC = "F1_підібраний"

# (варіант, база, позначка статусу)
COMPARISONS = [
    ("TFIDF+F0", "TFIDF", "головне, зафіксоване до прогону"),
    ("TFIDF+F3", "TFIDF", "додаткове, зафіксоване до прогону"),
    ("F0", "TFIDF", "додаткове, зафіксоване до прогону"),
    ("TFIDF+F0", "F0", "розвідувальне, додане після середніх"),
]
SIGN_METRICS = [METRIC, "ROC_AUC", "PR_AUC"]

ALPHA = 0.05
BOOTSTRAP_ITERATIONS = 1000
RANDOM_STATE = 42
TEST_TRAIN_RATIO = 1 / 4


def corrected_t_test(differences: np.ndarray) -> tuple[float, float]:
    """Поправлений t-критерій Nadeau—Bengio. Повертає t і двостороннє p."""
    n = differences.size
    mean = float(differences.mean())
    variance = float(differences.var(ddof=1))
    if variance == 0:
        return (np.inf if mean != 0 else 0.0), (0.0 if mean != 0 else 1.0)
    denominator = np.sqrt((1 / n + TEST_TRAIN_RATIO) * variance)
    t_value = mean / denominator
    return float(t_value), float(2 * stats.t.sf(abs(t_value), df=n - 1))


def wilcoxon_p(differences: np.ndarray) -> float:
    if np.all(differences == 0):
        return 1.0
    return float(stats.wilcoxon(differences, alternative="two-sided").pvalue)


def bootstrap_interval(differences: np.ndarray) -> tuple[float, float]:
    generator = np.random.default_rng(RANDOM_STATE)
    n = differences.size
    indices = generator.integers(0, n, size=(BOOTSTRAP_ITERATIONS, n))
    means = differences[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def paired(
    results: pd.DataFrame, platform: str, arm: str, base: str, metric: str
) -> np.ndarray:
    block = results[results["платформа"] == platform]
    left = block[block["варіант"] == arm].set_index(["repeat", "fold"])[metric]
    right = block[block["варіант"] == base].set_index(["repeat", "fold"])[metric]
    return (left - right).sort_index().to_numpy(dtype=np.float64)


def compute(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for platform in PLATFORM_ORDER:
        for arm, base, kind in COMPARISONS:
            diff = paired(results, platform, arm, base, METRIC)
            p_wilcoxon = wilcoxon_p(diff)
            t_value, p_corrected = corrected_t_test(diff)
            low, high = bootstrap_interval(diff)
            confirmed = (
                p_wilcoxon < ALPHA
                and p_corrected < ALPHA
                and not (low <= 0 <= high)
            )
            row = {
                "платформа": platform,
                "порівняння": f"{arm} − {base}",
                "статус": kind,
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
            for metric in ("ROC_AUC", "PR_AUC"):
                other = paired(results, platform, arm, base, metric)
                row[f"різниця_{metric}"] = float(other.mean())
            signs = {row["середня_різниця"] > 0, row["різниця_ROC_AUC"] > 0,
                     row["різниця_PR_AUC"] > 0}
            row["знаки_збігаються"] = bool(len(signs) == 1)
            rows.append(row)
    return pd.DataFrame(rows)


def _num(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def _signed(value: float, digits: int = 4) -> str:
    return f"{value:+.{digits}f}".replace(".", ",")


def _p(value: float) -> str:
    return "< 0,0001" if value < 0.0001 else f"{value:.4f}".replace(".", ",")


def representation_table(results: pd.DataFrame) -> list[str]:
    """F₀ проти TF-IDF за трьома метриками з позначкою переможця."""
    means = results.groupby(["платформа", "варіант"])[
        [METRIC, "ROC_AUC", "PR_AUC"]
    ].mean()
    lines = [
        "| Платформа | Метрика | F₀ | TF-IDF | Виграє |",
        "|---|---|---|---|---|",
    ]
    titles = {METRIC: "F1 підібраний", "ROC_AUC": "ROC-AUC", "PR_AUC": "PR-AUC"}
    for platform in PLATFORM_ORDER:
        for metric, title in titles.items():
            f0 = float(means.loc[(platform, "F0"), metric])
            tfidf = float(means.loc[(platform, "TFIDF"), metric])
            winner = "F₀" if f0 > tfidf else ("TF-IDF" if tfidf > f0 else "порівну")
            lines.append(
                f"| {platform} | {title} | {_num(f0)} | {_num(tfidf)} | {winner} |"
            )
    return lines


def recall_table(results: pd.DataFrame) -> list[str]:
    """Recall і accuracy F₀ поруч з тривіальним рівнем і часткою класу."""
    targets = pd.read_parquet(TARGETS_PATH)
    means = results.groupby(["платформа", "варіант"])[
        [METRIC, "Recall", "Accuracy", "Precision"]
    ].mean()
    lines = [
        "| Платформа | Частка позитивного класу | F1 тривіального | F1 F₀ | Recall F₀ | Precision F₀ | Accuracy F₀ |",
        "|---|---|---|---|---|---|---|",
    ]
    for platform in PLATFORM_ORDER:
        share = float(targets[f"y_{PLATFORM_SHORT[platform]}_{PERCENTILE}"].mean())
        trivial = 2 * share / (1 + share)
        row = means.loc[(platform, "F0")]
        lines.append(
            f"| {platform} | {_num(share)} | {_num(trivial)} | "
            f"{_num(float(row[METRIC]))} | {_num(float(row['Recall']))} | "
            f"{_num(float(row['Precision']))} | {_num(float(row['Accuracy']))} |"
        )
    return lines


def report_lines(stats_frame: pd.DataFrame, results: pd.DataFrame) -> list[str]:
    confirmed = stats_frame[stats_frame["підтверджено"]]
    fixed = stats_frame[stats_frame["статус"].str.startswith(("головне", "додаткове"))]
    exploratory = stats_frame[stats_frame["статус"].str.startswith("розвідувальне")]

    lines = [
        "# EXP-008: статистична перевірка E5",
        "",
        "Згенеровано `src/experiments/text_vs_semantic_stats.py` з "
        "`data/processed/text_vs_semantic_results.parquet`.",
        "Навчання немає: усе рахується з уже збережених метрик.",
        "",
        "Головна величина — F1 при підібраному на навчальній частині порозі.",
        "Три способи на тих самих двадцяти парних значеннях: критерій "
        "Вілкоксона зі знаковими рангами,",
        "поправлений t-критерій Nadeau—Bengio і 95% бутстреп-інтервал "
        f"({BOOTSTRAP_ITERATIONS} ітерацій).",
        "Критерій двосторонній.",
        "",
        f"**Правило ADR-006:** різниця підтверджена, якщо обидва p менші за "
        f"{_num(ALPHA, 2)} і інтервал не містить нуля.",
        "",
        "**Висновків тут немає.** Наведені числа і результат застосування "
        "правила.",
        "",
        "---",
        "",
        "## Зведення",
        "",
        f"| Результат правила | Комбінацій із {len(stats_frame)} |",
        "|---|---|",
        f"| Підтверджено, різниця додатна | "
        f"**{int((confirmed['середня_різниця'] > 0).sum())}** |",
        f"| Підтверджено, різниця від'ємна | "
        f"**{int((confirmed['середня_різниця'] < 0).sum())}** |",
        f"| Не підтверджено | {len(stats_frame) - len(confirmed)} |",
        "",
        f"Серед порівнянь, зафіксованих до прогону: підтверджено "
        f"{int(fixed['підтверджено'].sum())} із {len(fixed)}.",
        f"Серед розвідувальних: підтверджено "
        f"{int(exploratory['підтверджено'].sum())} із {len(exploratory)}.",
        "",
        "---",
        "",
        "## Усі дванадцять комбінацій",
        "",
        "| Платформа | Порівняння | Статус | Середня різниця | p Вілкоксона | p поправлений | 95% ДІ | Правило ADR-006 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in stats_frame.itertuples(index=False):
        verdict = "**підтверджено**" if row.підтверджено else "ні"
        lines.append(
            f"| {row.платформа} | {row.порівняння} | {row.статус} | "
            f"{_signed(row.середня_різниця)} | {_p(row.p_Вілкоксона)} | "
            f"{_p(row.p_поправлений)} | "
            f"[{_signed(row.ДІ_низ)}; {_signed(row.ДІ_верх)}] | {verdict} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Узгодженість знаків",
        "",
        "П'яте доповнення ADR-006 вимагає перевіряти, чи має різниця F1 той "
        "самий знак,",
        "що й зміни безпорогових метрик.",
        "",
        "| Платформа | Порівняння | ΔF1 | ΔROC-AUC | ΔPR-AUC | Знаки |",
        "|---|---|---|---|---|---|",
    ]
    for row in stats_frame.itertuples(index=False):
        verdict = "збігаються" if row.знаки_збігаються else "**розходяться**"
        lines.append(
            f"| {row.платформа} | {row.порівняння} | "
            f"{_signed(row.середня_різниця)} | "
            f"{_signed(getattr(row, 'різниця_ROC_AUC'))} | "
            f"{_signed(getattr(row, 'різниця_PR_AUC'))} | {verdict} |"
        )
    lines += [
        "",
        f"Знаки збігаються в {int(stats_frame['знаки_збігаються'].sum())} "
        f"комбінаціях із {len(stats_frame)}.",
        "",
        "---",
        "",
        "## F₀ проти TF-IDF за трьома метриками",
        "",
        "Середні по двадцяти розбиттях.",
        "",
    ]
    lines += representation_table(results)

    lines += [
        "",
        "---",
        "",
        "## Recall, precision і accuracy набору F₀",
        "",
        "Поруч з часткою позитивного класу і F1 тривіального класифікатора, "
        "що дорівнює 2p / (1 + p).",
        "",
    ]
    lines += recall_table(results)
    lines.append("")
    return lines


def main() -> None:
    results = pd.read_parquet(RESULTS_PATH)
    stats_frame = compute(results)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    stats_frame.to_parquet(OUT_PATH, index=False)

    lines = report_lines(stats_frame, results)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"\nтаблиця: {OUT_PATH.relative_to(PROJECT_ROOT)} — "
          f"{len(stats_frame)} рядків")
    print(f"звіт: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
