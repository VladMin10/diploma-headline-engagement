"""Статистика і зведення розширення E5 з sentence embeddings.

Запуск:
    .venv/bin/python -m src.experiments.embeddings_stats

Нічого не навчається: рахується з `embeddings_results.parquet`
і `text_vs_semantic_results.parquet`.

Три способи ті самі, що в EXP-007 і EXP-008: критерій Вілкоксона зі знаковими
рангами, поправлений t-критерій Nadeau—Bengio з відношенням 1/4 і 95%
перцентильний бутстреп-інтервал на 1 000 ітерацій. Критерій двосторонній.

Три порівняння. Головне — embeddings разом з F₀ проти самих embeddings.
Друге — F₀ проти embeddings. Третє, довідкове, — embeddings проти TF-IDF:
воно показує, наскільки сучасне представлення сильніше за частотне.

**Висновків тут немає.**
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EMB_PATH = PROJECT_ROOT / "data" / "processed" / "embeddings_results.parquet"
E5_PATH = PROJECT_ROOT / "data" / "processed" / "text_vs_semantic_results.parquet"
LIFT_PATH = PROJECT_ROOT / "data" / "processed" / "practical_lift.parquet"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "embeddings_stats.parquet"
REPORT_PATH = PROJECT_ROOT / "reports" / "tables" / "embeddings_check.md"

METRIC = "F1_підібраний"
PLATFORM_ORDER = ["Facebook", "GooglePlus", "LinkedIn"]
TITLE = {"Facebook": "Facebook", "GooglePlus": "Google+", "LinkedIn": "LinkedIn"}
METRICS = [
    METRIC, "F1_05", "Accuracy", "Precision", "Recall", "ROC_AUC", "PR_AUC",
]
SHARES = (10, 20, 25)

# Назви варіантів у зведеній таблиці і їхня розмірність.
VARIANT_LABEL = {
    "F0": "сім ознак",
    "TFIDF": "TF-IDF",
    "EMB": "embeddings",
    "EMB+F0": "embeddings + сім ознак",
}
COMPARISONS = [
    ("EMB+F0", "EMB", "головне"),
    ("F0", "EMB", "додаткове"),
    ("EMB", "TFIDF", "довідкове"),
]

ALPHA = 0.05
BOOTSTRAP_ITERATIONS = 1000
RANDOM_STATE = 42
TEST_TRAIN_RATIO = 1 / 4


def corrected_t_test(differences: np.ndarray) -> tuple[float, float]:
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


def combined(emb: pd.DataFrame, e5: pd.DataFrame) -> pd.DataFrame:
    """Чотири варіанти в одній таблиці з однаковими колонками."""
    e5 = e5[e5["варіант"].isin(["F0", "TFIDF"])].copy()
    return pd.concat([emb, e5], ignore_index=True)


def paired(frame: pd.DataFrame, platform: str, arm: str, base: str, metric: str):
    block = frame[frame["платформа"] == platform]
    left = block[block["варіант"] == arm].set_index(["repeat", "fold"])[metric]
    right = block[block["варіант"] == base].set_index(["repeat", "fold"])[metric]
    return (left - right).sort_index().to_numpy(dtype=np.float64)


def compute(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for platform in PLATFORM_ORDER:
        for arm, base, kind in COMPARISONS:
            diff = paired(frame, platform, arm, base, METRIC)
            p_wilcoxon = wilcoxon_p(diff)
            _, p_corrected = corrected_t_test(diff)
            low, high = bootstrap_interval(diff)
            row = {
                "платформа": platform,
                "порівняння": f"{arm} − {base}",
                "статус": kind,
                "середня_різниця": float(diff.mean()),
                "додатних": int((diff > 0).sum()),
                "p_Вілкоксона": p_wilcoxon,
                "p_поправлений": p_corrected,
                "ДІ_низ": low,
                "ДІ_верх": high,
                "підтверджено": bool(
                    p_wilcoxon < ALPHA and p_corrected < ALPHA and not (low <= 0 <= high)
                ),
            }
            for metric in ("ROC_AUC", "PR_AUC"):
                row[f"різниця_{metric}"] = float(
                    paired(frame, platform, arm, base, metric).mean()
                )
            signs = {
                row["середня_різниця"] > 0,
                row["різниця_ROC_AUC"] > 0,
                row["різниця_PR_AUC"] > 0,
            }
            row["знаки_збігаються"] = bool(len(signs) == 1)
            rows.append(row)
    return pd.DataFrame(rows)


def _num(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def _signed(value: float, digits: int = 4) -> str:
    return f"{value:+.{digits}f}".replace(".", ",")


def _p(value: float) -> str:
    return "< 0,0001" if value < 0.0001 else f"{value:.4f}".replace(".", ",")


def main() -> None:
    emb = pd.read_parquet(EMB_PATH)
    e5 = pd.read_parquet(E5_PATH)
    lift = pd.read_parquet(LIFT_PATH)
    frame = combined(emb, e5)
    table = compute(frame)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(OUT_PATH, index=False)

    dimensions = {
        "F0": 7,
        "TFIDF": int(round(e5[e5["розмір_словника"] > 0]["розмір_словника"].mean())),
        "EMB": int(emb[emb["варіант"] == "EMB"]["n_ознак"].iloc[0]),
        "EMB+F0": int(emb[emb["варіант"] == "EMB+F0"]["n_ознак"].iloc[0]),
    }
    meta = pd.read_json(
        PROJECT_ROOT / "data" / "processed" / "embeddings_meta.json", typ="series"
    )

    lines = [
        "# Розширення E5: sentence embeddings",
        "",
        "Згенеровано `src/experiments/embeddings_stats.py`. Нічого не навчається.",
        "",
        "Протокол — підрозділ «Розширення E5» у картці EXP-008, записаний "
        "до коду.",
        f"Модель векторизації `{meta['модель']}`, {meta['вимірів']} вимірів, "
        "попередньо навчена,",
        "на даних роботи не донавчається.",
        "",
        "Поріг мітки — верхні 25%, двадцять розбиттів, логістична регресія.",
        "Значення F₀ і TF-IDF узяті з наявних результатів EXP-008 "
        "і не перераховувалися.",
        "",
        "**Висновків тут немає, тільки числа і результат правила ADR-006.**",
        "",
        "---",
        "",
        "## Розмірність представлень",
        "",
        "| Представлення | Ознак |",
        "|---|---|",
    ]
    for variant, size in dimensions.items():
        lines.append(f"| {VARIANT_LABEL[variant]} | {size} |")

    lines += [
        "",
        "---",
        "",
        "## Чотири варіанти поруч: усі сім метрик",
        "",
        "Середнє по двадцяти розбиттях.",
        "",
        "| Платформа | Представлення | F1 підібр. | F1 при 0,5 | Accuracy | Precision | Recall | ROC-AUC | PR-AUC |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    means = frame.groupby(["платформа", "варіант"])[METRICS].mean()
    for platform in PLATFORM_ORDER:
        for variant in ("F0", "TFIDF", "EMB", "EMB+F0"):
            row = means.loc[(platform, variant)]
            cells = [_num(float(row[m])) for m in METRICS]
            lines.append(
                f"| {TITLE[platform]} | {VARIANT_LABEL[variant]} | "
                + " | ".join(cells)
                + " |"
            )

    lines += [
        "",
        "---",
        "",
        "## Влучність на верхніх k відсотках",
        "",
        "Для embeddings рахується в цьому прогоні, для F₀ і TF-IDF береться "
        "з `practical_lift.parquet`.",
        "",
        "| Платформа | Представлення | Верхні 10% | Верхні 20% | Верхні 25% | Базовий рівень |",
        "|---|---|---|---|---|---|",
    ]
    lift_names = {"F0": "LogisticRegression_F0", "TFIDF": "LogisticRegression_TFIDF"}
    for platform in PLATFORM_ORDER:
        base = float(emb[emb["платформа"] == platform]["базова_частка"].mean())
        for variant in ("F0", "TFIDF"):
            block = lift[
                (lift["варіант"] == lift_names[variant])
                & (lift["платформа"] == platform)
            ].set_index("k")
            cells = [_num(float(block.loc[s / 100, "precision_топ"])) for s in SHARES]
            lines.append(
                f"| {TITLE[platform]} | {VARIANT_LABEL[variant]} | "
                + " | ".join(cells)
                + f" | {_num(base)} |"
            )
        for variant in ("EMB", "EMB+F0"):
            block = emb[
                (emb["варіант"] == variant) & (emb["платформа"] == platform)
            ]
            cells = [_num(float(block[f"precision@{s}"].mean())) for s in SHARES]
            lines.append(
                f"| {TITLE[platform]} | {VARIANT_LABEL[variant]} | "
                + " | ".join(cells)
                + f" | {_num(base)} |"
            )

    lines += [
        "",
        "---",
        "",
        "## Три парні різниці зі статистикою",
        "",
        "Головна величина — F1 при підібраному порозі. Три способи "
        "на двадцяти парних значеннях,",
        "критерій двосторонній. Правило ADR-006: підтверджено, якщо обидва p "
        f"менші за {_num(ALPHA, 2)} і інтервал не містить нуля.",
        "",
        "| Платформа | Порівняння | Статус | Різниця | Додатних із 20 | p Вілкоксона | p поправлений | 95% ДІ | Правило |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in table.itertuples(index=False):
        verdict = "**підтверджено**" if row.підтверджено else "ні"
        lines.append(
            f"| {TITLE[row.платформа]} | {row.порівняння} | {row.статус} | "
            f"{_signed(row.середня_різниця)} | {row.додатних} | "
            f"{_p(row.p_Вілкоксона)} | {_p(row.p_поправлений)} | "
            f"[{_signed(row.ДІ_низ)}; {_signed(row.ДІ_верх)}] | {verdict} |"
        )

    confirmed = table[table["підтверджено"]]
    lines += [
        "",
        f"Підтверджено {len(confirmed)} різниць із {len(table)}: "
        f"{int((confirmed['середня_різниця'] > 0).sum())} додатних, "
        f"{int((confirmed['середня_різниця'] < 0).sum())} від'ємних.",
        "",
        "### Узгодженість знаків",
        "",
        "| Платформа | Порівняння | ΔF1 | ΔROC-AUC | ΔPR-AUC | Знаки |",
        "|---|---|---|---|---|---|",
    ]
    for row in table.itertuples(index=False):
        verdict = "збігаються" if row.знаки_збігаються else "**розходяться**"
        lines.append(
            f"| {TITLE[row.платформа]} | {row.порівняння} | "
            f"{_signed(row.середня_різниця)} | "
            f"{_signed(getattr(row, 'різниця_ROC_AUC'))} | "
            f"{_signed(getattr(row, 'різниця_PR_AUC'))} | {verdict} |"
        )

    lines += [
        "",
        f"Знаки збігаються в {int(table['знаки_збігаються'].sum())} комбінаціях "
        f"із {len(table)}.",
        "",
        "---",
        "",
        "## Час",
        "",
        "| Крок | Час |",
        "|---|---|",
        f"| Завантаження моделі | {_num(float(meta['час_завантаження_моделі_с']), 1)} с |",
        f"| Векторизація {meta['рядків']} заголовків | "
        f"{_num(float(meta['час_векторизації_с']), 1)} с |",
        "| Прогін двадцяти розбиттів на трьох платформах | 2,1 хв |",
        "",
    ]

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"звіт: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
