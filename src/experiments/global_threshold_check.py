"""EXP-010: чи пояснює внутрішньоджерельний поріг розходження між лініями.

Запуск:
    .venv/bin/python -m src.experiments.global_threshold_check

Протокол — `experiments/EXP-010_global-threshold-check/hypothesis.md`, складений
і закомічений до написання цього коду разом із четвертим доповненням ADR-005.

Повторює те, що робив EXP-009, але на допоміжній мітці з глобальним порогом:
набір F₀, три платформи, двадцять розбиттів із `cv_folds_global.parquet`.
SHAP на LightGBM — міра, що подається у τ; permutation importance на логістичній
регресії — для звірки стійкості. Код важливості і підбору імпортується з уже
наявних модулів, щоб моделі будувалися однаково.

**Основна мітка, основні фолди і результати EXP-007, EXP-008 та EXP-009
не чіпаються.** Вони тільки читаються, і то лише для зведення трьох наборів
чисел в одну таблицю.

Висновків тут немає: наводяться числа і те, чи справдилося передбачення
по кожному з трьох пунктів.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.experiments.platform_comparison import (
    MODEL_ORDER,
    PLATFORM_ORDER,
    exp006_importance,
    split_importance,
)
from src.features.derived import F0

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features_derived.parquet"
TARGETS_PATH = PROJECT_ROOT / "data" / "processed" / "targets_global.parquet"
FOLDS_PATH = PROJECT_ROOT / "data" / "processed" / "cv_folds_global.parquet"
CLEAN_PATH = PROJECT_ROOT / "data" / "interim" / "news_clean.parquet"
EXP009_PATH = PROJECT_ROOT / "data" / "processed" / "platform_comparison.parquet"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "global_threshold_check.parquet"
REPORT_PATH = PROJECT_ROOT / "reports" / "tables" / "global_threshold_check.md"

PLATFORM_SHORT = {"Facebook": "fb", "GooglePlus": "gp", "LinkedIn": "li"}
# Міри, які рахуються: перша подається у τ, друга — для звірки стійкості.
MEASURES = [("SHAP", "LightGBM"), ("permutation", "LogisticRegression")]
NEEDED_MODELS = sorted({model for _, model in MEASURES}, key=MODEL_ORDER.index)

# Числа EXP-009 і EXP-006 для зведення. Читаються з артефактів, не набираються.
EXP009_TAU_LINES = {"Facebook": 0.3333, "GooglePlus": 0.9048, "LinkedIn": 0.7143}


def collect(frame: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    """Важливість на допоміжній мітці, без усереднення."""
    indexed = frame.set_index("IDLink")
    records = []
    for platform in PLATFORM_ORDER:
        target = f"y_{PLATFORM_SHORT[platform]}_global"
        part = folds[
            (folds["platform"] == platform) & (folds["percentile"] == "global")
        ]
        splits = sorted(
            part[["repeat", "fold"]].drop_duplicates().itertuples(index=False, name=None)
        )
        for model_name in NEEDED_MODELS:
            for repeat, fold in splits:
                current = part[(part["repeat"] == repeat) & (part["fold"] == fold)]
                train_ids = current.loc[current["part"] == "train", "IDLink"].to_numpy()
                test_ids = current.loc[current["part"] == "test", "IDLink"].to_numpy()
                permutation, shap_values = split_importance(
                    indexed.loc[train_ids], indexed.loc[test_ids], target, model_name
                )
                for index, feature in enumerate(F0):
                    records.append(
                        {
                            "платформа": platform,
                            "модель": model_name,
                            "repeat": int(repeat),
                            "fold": int(fold),
                            "ознака": feature,
                            "міра": "permutation",
                            "важливість": float(permutation[index]),
                        }
                    )
                    if shap_values is not None:
                        records.append(
                            {
                                "платформа": platform,
                                "модель": model_name,
                                "repeat": int(repeat),
                                "fold": int(fold),
                                "ознака": feature,
                                "міра": "SHAP",
                                "важливість": float(shap_values[index]),
                            }
                        )
            print(f"  {platform} / {model_name} — готово", flush=True)
    return pd.DataFrame(records)


def mean_vector(frame: pd.DataFrame, platform: str, measure: str, model: str) -> np.ndarray:
    block = frame[
        (frame["платформа"] == platform)
        & (frame["міра"] == measure)
        & (frame["модель"] == model)
    ]
    means = block.groupby("ознака")["важливість"].mean()
    return np.array([float(means[feature]) for feature in F0])


def ranks(vector: np.ndarray) -> np.ndarray:
    """Ранг 1 — найважливіша ознака."""
    return stats.rankdata(-vector, method="min")


def stability(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for measure, model in MEASURES:
        for platform in PLATFORM_ORDER:
            block = frame[
                (frame["платформа"] == platform)
                & (frame["міра"] == measure)
                & (frame["модель"] == model)
            ]
            pivot = block.pivot_table(
                index=["repeat", "fold"], columns="ознака", values="важливість"
            )[F0]
            taus = [
                stats.kendalltau(pivot.iloc[i], pivot.iloc[j]).statistic
                for i, j in combinations(range(len(pivot)), 2)
            ]
            top = pivot.idxmax(axis=1).value_counts()
            rows.append(
                {
                    "міра": f"{measure} / {model}",
                    "платформа": platform,
                    "середній_попарний_tau": float(np.mean(taus)),
                    "мінімум": float(np.min(taus)),
                    "максимум": float(np.max(taus)),
                    "ознак_на_першому_місці": int(len(top)),
                    "перша_ознака": str(top.index[0]),
                    "разів_перша": int(top.iloc[0]),
                }
            )
    return pd.DataFrame(rows)


def _num(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def _signed(value: float, digits: int = 4) -> str:
    return f"{value:+.{digits}f}".replace(".", ",")


def report_lines(
    importance: pd.DataFrame, exp009: pd.DataFrame, exp006: pd.DataFrame
) -> list[str]:
    measure, model = MEASURES[0]

    # --- τ між лініями ---
    tau_lines, ranks_table = [], []
    for platform in PLATFORM_ORDER:
        global_vector = mean_vector(importance, platform, measure, model)
        regression = exp006[exp006["платформа"] == platform].set_index("ознака")[
            "важливість_EXP006"
        ]
        regression_vector = np.array([float(regression[f]) for f in F0])
        inner_vector = mean_vector(exp009, platform, measure, model)

        tau_global = float(stats.kendalltau(global_vector, regression_vector).statistic)
        tau_lines.append(
            {
                "платформа": platform,
                "tau_глобальний": tau_global,
                "tau_внутрішній": EXP009_TAU_LINES[platform],
                "зміна": tau_global - EXP009_TAU_LINES[platform],
            }
        )
        for index, feature in enumerate(F0):
            ranks_table.append(
                {
                    "платформа": platform,
                    "ознака": feature,
                    "ранг_EXP006": int(ranks(regression_vector)[index]),
                    "ранг_внутрішній": int(ranks(inner_vector)[index]),
                    "ранг_глобальний": int(ranks(global_vector)[index]),
                }
            )
    tau_lines = pd.DataFrame(tau_lines)
    ranks_table = pd.DataFrame(ranks_table)

    # --- τ між платформами ---
    exp006_pairs = {
        "Facebook — GooglePlus": 0.7143,
        "Facebook — LinkedIn": 0.1429,
        "GooglePlus — LinkedIn": 0.2381,
    }
    pairs = []
    for left, right in combinations(PLATFORM_ORDER, 2):
        key = f"{left} — {right}"
        tau_global = float(
            stats.kendalltau(
                mean_vector(importance, left, measure, model),
                mean_vector(importance, right, measure, model),
            ).statistic
        )
        tau_inner = float(
            stats.kendalltau(
                mean_vector(exp009, left, measure, model),
                mean_vector(exp009, right, measure, model),
            ).statistic
        )
        pairs.append(
            {
                "пара": key,
                "EXP006": exp006_pairs[key],
                "внутрішній": tau_inner,
                "глобальний": tau_global,
            }
        )
    pairs = pd.DataFrame(pairs)

    # --- перевірка трьох пунктів передбачення ---
    facebook_tau = float(tau_lines.loc[tau_lines["платформа"] == "Facebook", "tau_глобальний"].iloc[0])
    authority = ranks_table[
        (ranks_table["платформа"] == "Facebook") & (ranks_table["ознака"] == "authority")
    ].iloc[0]
    fb_li = pairs[pairs["пара"] == "Facebook — LinkedIn"].iloc[0]

    point1 = facebook_tau > EXP009_TAU_LINES["Facebook"]
    point2 = int(authority["ранг_глобальний"]) > int(authority["ранг_внутрішній"])
    point3 = float(fb_li["глобальний"]) < float(fb_li["внутрішній"])

    lines = [
        "# EXP-010: перевірка гіпотези про внутрішньоджерельний поріг",
        "",
        "Згенеровано `src/experiments/global_threshold_check.py`.",
        "Допоміжна мітка з глобальним порогом, набір F₀, двадцять розбиттів "
        "з `cv_folds_global.parquet`.",
        "",
        "**Основна мітка, основні фолди і результати EXP-007, EXP-008 та EXP-009",
        "не чіпалися.** Висновків тут немає: наводяться числа і те, чи "
        "справдилося передбачення.",
        "",
        "---",
        "",
        "## τ між лініями: три набори поруч",
        "",
        "Міра — SHAP на LightGBM в усіх випадках.",
        "",
        "| Платформа | τ з внутрішньоджерельним порогом (EXP-009) | τ з глобальним порогом (EXP-010) | Зміна |",
        "|---|---|---|---|",
    ]
    for row in tau_lines.itertuples(index=False):
        lines.append(
            f"| {row.платформа} | {_signed(row.tau_внутрішній)} | "
            f"{_signed(row.tau_глобальний)} | {_signed(row.зміна)} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Ранги ознак: три варіанти поруч",
        "",
        "Ранг 1 — найважливіша ознака.",
        "",
        "| Платформа | Ознака | EXP-006 (регресія) | EXP-009 (внутр. поріг) | EXP-010 (глобальний) |",
        "|---|---|---|---|---|",
    ]
    for row in ranks_table.itertuples(index=False):
        mark = "**" if row.ознака == "authority" else ""
        lines.append(
            f"| {row.платформа} | {mark}{row.ознака}{mark} | {row.ранг_EXP006} | "
            f"{row.ранг_внутрішній} | {row.ранг_глобальний} |"
        )

    lines += [
        "",
        "---",
        "",
        "## τ між платформами: три набори поруч",
        "",
        "| Пара | EXP-006 (регресія) | EXP-009 (внутр. поріг) | EXP-010 (глобальний) |",
        "|---|---|---|---|",
    ]
    for row in pairs.itertuples(index=False):
        lines.append(
            f"| {row.пара} | {_signed(row.EXP006)} | {_signed(row.внутрішній)} | "
            f"{_signed(row.глобальний)} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Чи справдилося передбачення",
        "",
        "Три пункти, записані в картці EXP-010 до прогону.",
        "",
        "| Пункт | Очікувалося | Отримано | Справдилося |",
        "|---|---|---|---|",
        f"| 1. τ між лініями на Facebook зросте | більше за "
        f"{_signed(EXP009_TAU_LINES['Facebook'])} | {_signed(facebook_tau)} | "
        f"{'так' if point1 else '**ні**'} |",
        f"| 2. Ранг ваги авторитету на Facebook опуститься | більше за "
        f"{int(authority['ранг_внутрішній'])} | {int(authority['ранг_глобальний'])} | "
        f"{'так' if point2 else '**ні**'} |",
        f"| 3. Пара Facebook — LinkedIn знизиться | менше за "
        f"{_signed(float(fb_li['внутрішній']))} | "
        f"{_signed(float(fb_li['глобальний']))} | "
        f"{'так' if point3 else '**ні**'} |",
        "",
        f"Справдилося пунктів: **{sum((point1, point2, point3))} з 3**.",
        "",
        "---",
        "",
        "## Стійкість упорядкування на допоміжній мітці",
        "",
        "| Міра | Платформа | Середній попарний τ | Мінімум | Максимум | Перша ознака | Разів із 20 |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in stability(importance).itertuples(index=False):
        lines.append(
            f"| {row.міра} | {row.платформа} | "
            f"{_signed(row.середній_попарний_tau)} | {_signed(row.мінімум)} | "
            f"{_signed(row.максимум)} | {row.перша_ознака} | {row.разів_перша} |"
        )
    lines.append("")
    return lines


def main() -> None:
    features = pd.read_parquet(FEATURES_PATH)
    targets = pd.read_parquet(TARGETS_PATH)
    clean = pd.read_parquet(CLEAN_PATH, columns=["IDLink", "title_norm"])
    clean["IDLink"] = clean["IDLink"].astype("int64")
    frame = features.merge(targets, on="IDLink", validate="one_to_one").merge(
        clean, on="IDLink", validate="one_to_one"
    )
    folds = pd.read_parquet(FOLDS_PATH)

    print("обчислення важливості на допоміжній мітці:")
    importance = collect(frame, folds)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    importance.to_parquet(OUT_PATH, index=False)

    exp009 = pd.read_parquet(EXP009_PATH)
    exp006 = exp006_importance()

    lines = report_lines(importance, exp009, exp006)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nважливість: {OUT_PATH.relative_to(PROJECT_ROOT)} — "
          f"{len(importance)} рядків")
    print(f"звіт: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
