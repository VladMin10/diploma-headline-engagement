"""Таблиці EXP-007 з матриці метрик абляції.

Запуск:
    .venv/bin/python -m src.experiments.ablation_report

Читає `data/processed/ablation_results.parquet` і будує
`reports/tables/ablation_f0_f3.md`. Навчання тут немає: усереднення робиться
при побудові таблиць саме для того, щоб їх можна було перебудувати
без повторного прогону.

**Висновків тут немає, тільки числа.** Статистична перевірка винесена
в окремий крок протоколом EXP-007, щоб числа і висновки не змішувалися.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = PROJECT_ROOT / "data" / "processed" / "ablation_results.parquet"
TARGETS_PATH = PROJECT_ROOT / "data" / "processed" / "targets_classification.parquet"
REPORT_PATH = PROJECT_ROOT / "reports" / "tables" / "ablation_f0_f3.md"

MAIN_PERCENTILE = "25"
PLATFORM_ORDER = ["Facebook", "GooglePlus", "LinkedIn"]
PLATFORM_SHORT = {"Facebook": "fb", "GooglePlus": "gp", "LinkedIn": "li"}
PERCENTILE_ORDER = ["20", "25", "33"]
MODEL_ORDER = ["LogisticRegression", "RandomForest", "LightGBM", "CatBoost"]
SET_ORDER = ["F0", "F1", "F2", "F3"]
COMPARISONS = [("F1", "F0"), ("F2", "F0"), ("F3", "F0")]
METRICS = [
    "F1_підібраний",
    "F1_05",
    "Accuracy",
    "Precision",
    "Recall",
    "ROC_AUC",
    "PR_AUC",
]


def _num(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def _signed(value: float, digits: int = 4) -> str:
    return f"{value:+.{digits}f}".replace(".", ",")


def paired(results: pd.DataFrame, metric: str, percentile: str) -> pd.DataFrame:
    """Парні різниці наборів проти F₀ на однакових розбиттях."""
    subset = results[results["перцентиль"] == percentile]
    key = ["платформа", "модель", "repeat", "fold"]
    rows = []
    for platform in PLATFORM_ORDER:
        for model in MODEL_ORDER:
            block = subset[
                (subset["платформа"] == platform) & (subset["модель"] == model)
            ]
            base = block[block["набір"] == "F0"].set_index(["repeat", "fold"])[metric]
            for other, _ in COMPARISONS:
                arm = block[block["набір"] == other].set_index(["repeat", "fold"])[metric]
                diff = (arm - base).sort_index()
                rows.append(
                    {
                        "платформа": platform,
                        "модель": model,
                        "порівняння": f"{other} − F0",
                        "середня": float(diff.mean()),
                        "мінімум": float(diff.min()),
                        "максимум": float(diff.max()),
                        "додатних": int((diff > 0).sum()),
                        "n": int(len(diff)),
                    }
                )
    _ = key
    return pd.DataFrame(rows)


def positive_shares() -> pd.DataFrame:
    """Фактична частка позитивного класу для кожного поєднання.

    Береться з `targets_classification.parquet`, тобто з тієї самої мітки,
    на якій навчалися моделі. Нового обчислення тут немає.
    """
    targets = pd.read_parquet(TARGETS_PATH)
    rows = []
    for platform in PLATFORM_ORDER:
        for percentile in PERCENTILE_ORDER:
            column = f"y_{PLATFORM_SHORT[platform]}_{percentile}"
            share = float(targets[column].mean())
            rows.append(
                {
                    "платформа": platform,
                    "перцентиль": percentile,
                    "частка": share,
                    # F1 класифікатора, що відносить усі рядки до позитивного
                    # класу: precision = p, recall = 1, звідси 2p/(1+p).
                    "F1_тривіальний": 2 * share / (1 + share),
                }
            )
    return pd.DataFrame(rows)


def reference_levels(results: pd.DataFrame) -> list[str]:
    """Довідкові рівні: тривіальний класифікатор і базові рівні метрик."""
    shares = positive_shares()
    lines = [
        "## Довідкові рівні",
        "",
        "### Тривіальний класифікатор",
        "",
        "Класифікатор, що відносить усі рядки до позитивного класу, має "
        "precision, що дорівнює",
        "частці позитивного класу, і recall, що дорівнює одиниці. Звідси "
        "F1 = 2p / (1 + p).",
        "Цей рівень не є базовим у сенсі правила рішення, але показує, з чим "
        "порівнюється",
        "абсолютне значення F1.",
        "",
        "| Платформа | Поріг мітки | Частка позитивного класу | F1 тривіального |",
        "|---|---|---|---|",
    ]
    for row in shares.itertuples(index=False):
        mark = " (основний)" if row.перцентиль == MAIN_PERCENTILE else ""
        lines.append(
            f"| {row.платформа} | верхні {row.перцентиль}%{mark} | "
            f"{_num(row.частка)} | {_num(row.F1_тривіальний)} |"
        )

    subset = results[results["перцентиль"] == MAIN_PERCENTILE]
    means = subset.groupby(["платформа", "модель", "набір"])["F1_підібраний"].mean()
    level = shares[shares["перцентиль"] == MAIN_PERCENTILE].set_index("платформа")

    lines += [
        "",
        f"### Скільки комбінацій перевищують цей рівень, верхні {MAIN_PERCENTILE}%",
        "",
        f"З {len(MODEL_ORDER)} × {len(SET_ORDER)} = {len(MODEL_ORDER) * len(SET_ORDER)} "
        "комбінацій моделі й набору ознак на кожній платформі.",
        "",
        "| Платформа | F1 тривіального | Перевищують | Не перевищують |",
        "|---|---|---|---|",
    ]
    below_all = {}
    for platform in PLATFORM_ORDER:
        threshold = float(level.loc[platform, "F1_тривіальний"])
        block = means.loc[platform]
        above = int((block > threshold).sum())
        below = [f"{model} {s}" for (model, s), value in block.items() if value <= threshold]
        below_all[platform] = below
        lines.append(
            f"| {platform} | {_num(threshold)} | {above} з {len(block)} | "
            f"{len(below)} з {len(block)} |"
        )

    lines += ["", "Комбінації, що рівня не перевищують:", ""]
    for platform in PLATFORM_ORDER:
        names = below_all[platform]
        if names:
            lines.append(f"- **{platform}:** " + ", ".join(names))
        else:
            lines.append(f"- **{platform}:** таких немає")

    lines += [
        "",
        f"### Абсолютні ROC-AUC і PR-AUC, верхні {MAIN_PERCENTILE}%",
        "",
        "Базовий рівень ROC-AUC — 0,5; базовий рівень PR-AUC — частка "
        "позитивного класу.",
        "",
        "| Платформа | Модель | Набір | ROC-AUC | PR-AUC | Базовий PR-AUC |",
        "|---|---|---|---|---|---|",
    ]
    auc = subset.groupby(["платформа", "модель", "набір"])[["ROC_AUC", "PR_AUC"]].mean()
    for platform in PLATFORM_ORDER:
        base = float(level.loc[platform, "частка"])
        for model in MODEL_ORDER:
            for set_name in SET_ORDER:
                row = auc.loc[(platform, model, set_name)]
                lines.append(
                    f"| {platform} | {model} | {set_name} | "
                    f"{_num(float(row['ROC_AUC']))} | {_num(float(row['PR_AUC']))} | "
                    f"{_num(base)} |"
                )
    lines.append("")
    return lines


def main_table(results: pd.DataFrame, percentile: str) -> list[str]:
    subset = results[results["перцентиль"] == percentile]
    means = subset.groupby(["платформа", "модель", "набір"])["F1_підібраний"].mean()
    lines = [
        "| Платформа | Модель | F₀ | F₁ | F₂ | F₃ |",
        "|---|---|---|---|---|---|",
    ]
    for platform in PLATFORM_ORDER:
        for model in MODEL_ORDER:
            cells = [
                _num(float(means.loc[(platform, model, s)])) for s in SET_ORDER
            ]
            lines.append(f"| {platform} | {model} | " + " | ".join(cells) + " |")
    return lines


def diff_table(results: pd.DataFrame, metric: str, percentile: str) -> list[str]:
    frame = paired(results, metric, percentile)
    lines = [
        "| Платформа | Модель | Порівняння | Середня різниця | Мінімум | Максимум | Додатних із 20 |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"| {row.платформа} | {row.модель} | {row.порівняння} | "
            f"{_signed(row.середня)} | {_signed(row.мінімум)} | "
            f"{_signed(row.максимум)} | {row.додатних} |"
        )
    return lines


def sign_agreement(results: pd.DataFrame, percentile: str) -> list[str]:
    """Чи збігається знак приросту F1 зі знаками ROC-AUC і PR-AUC."""
    f1 = paired(results, "F1_підібраний", percentile)
    roc = paired(results, "ROC_AUC", percentile)
    pr = paired(results, "PR_AUC", percentile)
    key = ["платформа", "модель", "порівняння"]
    merged = (
        f1[key + ["середня"]]
        .merge(roc[key + ["середня"]], on=key, suffixes=("_F1", "_ROC"))
        .merge(pr[key + ["середня"]].rename(columns={"середня": "середня_PR"}), on=key)
    )
    lines = [
        "| Платформа | Модель | Порівняння | ΔF1 | ΔROC-AUC | ΔPR-AUC | Знаки |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in merged.itertuples(index=False):
        signs = {
            (value > 0) for value in (row.середня_F1, row.середня_ROC, row.середня_PR)
        }
        verdict = "збігаються" if len(signs) == 1 else "розходяться"
        lines.append(
            f"| {row.платформа} | {row.модель} | {row.порівняння} | "
            f"{_signed(row.середня_F1)} | {_signed(row.середня_ROC)} | "
            f"{_signed(row.середня_PR)} | {verdict} |"
        )
    return lines, merged


def full_metrics(results: pd.DataFrame, percentile: str) -> list[str]:
    subset = results[results["перцентиль"] == percentile]
    means = subset.groupby(["платформа", "модель", "набір"])[METRICS].mean()
    lines = [
        "| Платформа | Модель | Набір | F1 підібр. | F1 при 0,5 | Accuracy | Precision | Recall | ROC-AUC | PR-AUC |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for platform in PLATFORM_ORDER:
        for model in MODEL_ORDER:
            for set_name in SET_ORDER:
                row = means.loc[(platform, model, set_name)]
                cells = [_num(float(row[m])) for m in METRICS]
                lines.append(
                    f"| {platform} | {model} | {set_name} | " + " | ".join(cells) + " |"
                )
    return lines


def other_percentiles(results: pd.DataFrame) -> list[str]:
    lines = [
        "| Перцентиль | Платформа | Модель | ΔF1 (F₃ − F₀) | Додатних із 20 |",
        "|---|---|---|---|---|",
    ]
    for percentile in ("20", "33"):
        frame = paired(results, "F1_підібраний", percentile)
        frame = frame[frame["порівняння"] == "F3 − F0"]
        for row in frame.itertuples(index=False):
            lines.append(
                f"| верхні {percentile}% | {row.платформа} | {row.модель} | "
                f"{_signed(row.середня)} | {row.додатних} |"
            )
    return lines


def main() -> None:
    results = pd.read_parquet(RESULTS_PATH)
    agreement_lines, merged = sign_agreement(results, MAIN_PERCENTILE)
    n_agree = sum(
        len({(v > 0) for v in (r.середня_F1, r.середня_ROC, r.середня_PR)}) == 1
        for r in merged.itertuples(index=False)
    )

    lines = [
        "# EXP-007: абляція похідних ознак (E1—E4)",
        "",
        "Згенеровано `src/experiments/ablation_report.py` з "
        "`data/processed/ablation_results.parquet`.",
        "Числа не редагуються вручну: правити треба джерело і перезапускати.",
        "",
        f"Сітка прогону: {results['платформа'].nunique()} платформи × "
        f"{results['перцентиль'].nunique()} пороги мітки × "
        f"{results['набір'].nunique()} набори ознак × "
        f"{results['модель'].nunique()} моделі × "
        f"{results.groupby(['платформа', 'перцентиль'])[['repeat', 'fold']].nunique().iloc[0, 0] * 5} розбиттів.",
        f"Рядків у матриці метрик: {len(results)}.",
        "",
        "**Висновків тут немає, тільки числа.** Статистична перевірка винесена",
        "в окремий крок протоколом EXP-007, щоб числа і висновки не змішувалися.",
        "",
        "---",
        "",
    ]
    lines += reference_levels(results)
    lines += [
        "---",
        "",
        f"## Головна таблиця: F1 при підібраному порозі, верхні {MAIN_PERCENTILE}%",
        "",
        "Середнє по двадцяти розбиттях.",
        "",
    ]
    lines += main_table(results, MAIN_PERCENTILE)

    lines += [
        "",
        "---",
        "",
        f"## Парні різниці за F1, верхні {MAIN_PERCENTILE}%",
        "",
        "Різниці обчислені парно на однакових розбиттях, а не через середні.",
        "",
    ]
    lines += diff_table(results, "F1_підібраний", MAIN_PERCENTILE)

    lines += ["", "---", "", f"## Парні різниці за ROC-AUC, верхні {MAIN_PERCENTILE}%", ""]
    lines += diff_table(results, "ROC_AUC", MAIN_PERCENTILE)

    lines += ["", "---", "", f"## Парні різниці за PR-AUC, верхні {MAIN_PERCENTILE}%", ""]
    lines += diff_table(results, "PR_AUC", MAIN_PERCENTILE)

    lines += [
        "",
        "---",
        "",
        "## Узгодженість знаків",
        "",
        "П'яте доповнення ADR-006 вимагає перевіряти, чи має приріст F1 той самий",
        "знак, що й зміни безпорогових метрик. Порівнюються середні різниці.",
        "",
    ]
    lines += agreement_lines
    lines += [
        "",
        f"Знаки збігаються в {n_agree} випадках з {len(merged)}.",
        "",
        "---",
        "",
        f"## Усі метрики, верхні {MAIN_PERCENTILE}%",
        "",
        "Середнє по двадцяти розбиттях. F1 при 0,5 — довідковий "
        "(п'яте доповнення ADR-006).",
        "",
    ]
    lines += full_metrics(results, MAIN_PERCENTILE)

    lines += [
        "",
        "---",
        "",
        "## Інші пороги мітки: F₃ проти F₀",
        "",
        "Перевірка стійкості до вибору порога мітки (ADR-006, розділ «Рішення»).",
        "",
    ]
    lines += other_percentiles(results)
    lines.append("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"звіт: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
