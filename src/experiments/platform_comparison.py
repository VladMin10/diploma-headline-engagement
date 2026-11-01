"""EXP-009: порівняння платформ і перевірка ADR-005.

Запуск:
    .venv/bin/python -m src.experiments.platform_comparison

Протокол — `experiments/EXP-009_platform-comparison/hypothesis.md`, складений
і закомічений до написання цього коду.

Набір ознак F₀, поріг мітки основний, двадцять розбиттів з `cv_folds.parquet`.
Моделі й підбір гіперпараметрів імпортуються з `ablation_f0_f3.py`, щоб вони
будувалися однаково і різниця між експериментами не виникла з реалізації.

Дві міри важливості за другим доповненням ADR-006. Permutation importance
на всіх чотирьох моделях — основна міра подання; метрикою для перестановки
береться ROC-AUC, бо вона не залежить від порога. SHAP на LightGBM — єдина
міра, що подається у смуги τ другого доповнення ADR-005, бо EXP-006 рахував
саме SHAP на LightGBM.

Стійкість упорядкування вимірюється окремо, і це головне діагностичне число
прогону: слабкість LightGBM у цій лінії відома з EXP-007, тому нестійкість
очікувана і має бути зміряна, а не переглянута постфактум.

Висновків тут немає, тільки числа.
"""

from __future__ import annotations

import json
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from scipy import stats
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler

from src.experiments.ablation_f0_f3 import (
    PLATFORMS,
    build_model,
    tune,
)
from src.features.copywriting import FEATURE_ORDER
from src.features.derived import F0

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features_derived.parquet"
TARGETS_PATH = PROJECT_ROOT / "data" / "processed" / "targets_classification.parquet"
FOLDS_PATH = PROJECT_ROOT / "data" / "processed" / "cv_folds.parquet"
CLEAN_PATH = PROJECT_ROOT / "data" / "interim" / "news_clean.parquet"
EXP006_PATH = (
    PROJECT_ROOT / "experiments" / "EXP-006_platform-differences" / "results.json"
)
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "platform_comparison.parquet"
REPORT_PATH = PROJECT_ROOT / "reports" / "tables" / "platform_comparison.md"

RANDOM_STATE = 42
PERCENTILE = "25"
PLATFORM_ORDER = ["Facebook", "GooglePlus", "LinkedIn"]
MODEL_ORDER = ["LogisticRegression", "RandomForest", "LightGBM", "CatBoost"]
PERMUTATION_REPEATS = 5

# Смуги τ з другого доповнення ADR-005, зафіксовані до прогону.
TAU_HIGH = 0.7
TAU_LOW = 0.3

# Міра, що подається у смуги τ, і друга міра для порівняння стійкості.
TAU_SOURCE = ("SHAP", "LightGBM")
STABILITY_SOURCES = [("SHAP", "LightGBM"), ("permutation", "LogisticRegression")]


def exp006_importance() -> pd.DataFrame:
    """Впорядкування семи ознак з регресійної лінії.

    Джерело — `results.json` EXP-006, поле `важливість[платформа]["T2"]`.
    Похідна таблиця `reports/tables/phase6_importance.csv` містить ті самі
    числа, але будується з цього ж файлу, тому береться первинний артефакт.
    """
    data = json.loads(EXP006_PATH.read_text(encoding="utf-8"))
    rows = []
    for platform in PLATFORM_ORDER:
        block = data["важливість"][platform]["T2"]
        for feature in FEATURE_ORDER:
            rows.append(
                {
                    "платформа": platform,
                    "ознака": feature,
                    "важливість_EXP006": float(block[feature]["mean_abs_shap"]),
                }
            )
    return pd.DataFrame(rows)


def split_importance(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    model_name: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Важливість ознак на одному розбитті.

    Повертає permutation importance для будь-якої моделі і середнє абсолютне
    значення SHAP для LightGBM; для решти моделей друге значення порожнє.
    """
    x_train = train[F0].to_numpy(dtype=np.float64)
    x_test = test[F0].to_numpy(dtype=np.float64)
    y_train = train[target].to_numpy(dtype=np.int8)
    y_test = test[target].to_numpy(dtype=np.int8)
    groups = train["title_norm"].to_numpy()

    value = tune(model_name, x_train, y_train, groups)

    if model_name == "LogisticRegression":
        scaler = StandardScaler().fit(x_train)
        x_train_used = scaler.transform(x_train)
        x_test_used = scaler.transform(x_test)
    else:
        x_train_used, x_test_used = x_train, x_test

    model = build_model(model_name, value)
    model.fit(x_train_used, y_train)

    # Перестановка на перевірочній частині фолда. Метрика ROC-AUC:
    # від порога не залежить, тому не тягне в важливість варіацію порога.
    result = permutation_importance(
        model,
        x_test_used,
        y_test,
        scoring="roc_auc",
        n_repeats=PERMUTATION_REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    permutation = result.importances_mean

    shap_values = None
    if model_name == "LightGBM":
        explainer = shap.TreeExplainer(model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            values = explainer.shap_values(x_test_used)
        if isinstance(values, list):
            values = values[1] if len(values) > 1 else values[0]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, -1]
        shap_values = np.abs(values).mean(axis=0)

    return permutation, shap_values


def collect(frame: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    """Важливість по всіх платформах, моделях і розбиттях, без усереднення."""
    indexed = frame.set_index("IDLink")
    records = []
    for platform in PLATFORM_ORDER:
        target = f"y_{PLATFORMS[platform]}_{PERCENTILE}"
        part = folds[
            (folds["platform"] == platform) & (folds["percentile"] == PERCENTILE)
        ]
        splits = sorted(
            part[["repeat", "fold"]].drop_duplicates().itertuples(index=False, name=None)
        )
        for model_name in MODEL_ORDER:
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


def ordering(values: pd.Series) -> np.ndarray:
    """Вектор важливості у порядку FEATURE_ORDER."""
    return np.array([float(values[feature]) for feature in F0])


def tau_between_lines(importance: pd.DataFrame, exp006: pd.DataFrame) -> pd.DataFrame:
    """Кендалл τ між лініями. Міра — SHAP на LightGBM (ADR-006, доповнення 2)."""
    measure, model = TAU_SOURCE
    rows = []
    for platform in PLATFORM_ORDER:
        block = importance[
            (importance["платформа"] == platform)
            & (importance["міра"] == measure)
            & (importance["модель"] == model)
        ]
        classification = block.groupby("ознака")["важливість"].mean()
        regression = exp006[exp006["платформа"] == platform].set_index("ознака")[
            "важливість_EXP006"
        ]
        tau, p_value = stats.kendalltau(
            ordering(classification), ordering(regression)
        )
        if tau >= TAU_HIGH:
            band = "узгоджені"
        elif tau >= TAU_LOW:
            band = "часткова узгодженість"
        else:
            band = "розходяться"
        rows.append(
            {
                "платформа": platform,
                "tau": float(tau),
                "p_value": float(p_value),
                "смуга": band,
            }
        )
    return pd.DataFrame(rows)


def tau_between_platforms(importance: pd.DataFrame) -> pd.DataFrame:
    """Кендалл τ між платформами всередині класифікаційної лінії."""
    measure, model = TAU_SOURCE
    data = json.loads(EXP006_PATH.read_text(encoding="utf-8"))["kendall_tau"]
    rows = []
    for left, right in combinations(PLATFORM_ORDER, 2):
        vectors = []
        for platform in (left, right):
            block = importance[
                (importance["платформа"] == platform)
                & (importance["міра"] == measure)
                & (importance["модель"] == model)
            ]
            vectors.append(ordering(block.groupby("ознака")["важливість"].mean()))
        tau, p_value = stats.kendalltau(*vectors)
        key = f"{left}|{right}"
        rows.append(
            {
                "пара": f"{left} — {right}",
                "tau_класифікаційна": float(tau),
                "p_класифікаційна": float(p_value),
                "tau_регресійна": float(data[key]["tau"]),
                "ДІ_регресійна": f"[{data[key]['ci_low']:+.4f}; {data[key]['ci_high']:+.4f}]",
            }
        )
    return pd.DataFrame(rows)


def stability(importance: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Три таблиці стійкості впорядкування по двадцяти розбиттях."""
    summary, leaders, spread = [], [], []
    for measure, model in STABILITY_SOURCES:
        for platform in PLATFORM_ORDER:
            block = importance[
                (importance["платформа"] == platform)
                & (importance["міра"] == measure)
                & (importance["модель"] == model)
            ]
            pivot = block.pivot_table(
                index=["repeat", "fold"], columns="ознака", values="важливість"
            )[F0]
            # Ранг 1 — найважливіша ознака.
            ranks = pivot.rank(axis=1, ascending=False, method="min")

            taus = [
                stats.kendalltau(pivot.iloc[i], pivot.iloc[j]).statistic
                for i, j in combinations(range(len(pivot)), 2)
            ]
            top = pivot.idxmax(axis=1).value_counts()
            summary.append(
                {
                    "міра": f"{measure} / {model}",
                    "платформа": platform,
                    "середній_попарний_tau": float(np.mean(taus)),
                    "мінімум_tau": float(np.min(taus)),
                    "максимум_tau": float(np.max(taus)),
                    "n_пар": len(taus),
                    "ознак_на_першому_місці": int(len(top)),
                }
            )
            for feature, count in top.items():
                leaders.append(
                    {
                        "міра": f"{measure} / {model}",
                        "платформа": platform,
                        "ознака": feature,
                        "разів_перша": int(count),
                        "частка": round(float(count) / len(pivot), 4),
                    }
                )
            for feature in F0:
                spread.append(
                    {
                        "міра": f"{measure} / {model}",
                        "платформа": platform,
                        "ознака": feature,
                        "ранг_мінімум": int(ranks[feature].min()),
                        "ранг_максимум": int(ranks[feature].max()),
                        "ранг_медіана": float(ranks[feature].median()),
                    }
                )
    return pd.DataFrame(summary), pd.DataFrame(leaders), pd.DataFrame(spread)


def _num(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def _signed(value: float, digits: int = 4) -> str:
    return f"{value:+.{digits}f}".replace(".", ",")


def report_lines(
    importance: pd.DataFrame,
    exp006: pd.DataFrame,
    lines_tau: pd.DataFrame,
    platforms_tau: pd.DataFrame,
    summary: pd.DataFrame,
    leaders: pd.DataFrame,
    spread: pd.DataFrame,
) -> list[str]:
    out = [
        "# EXP-009: порівняння платформ і перевірка ADR-005",
        "",
        "Згенеровано `src/experiments/platform_comparison.py`.",
        "Набір ознак F₀, поріг мітки верхні 25%, двадцять розбиттів.",
        "",
        "**Висновків тут немає, тільки числа.**",
        "",
        "---",
        "",
        "## Permutation importance: платформа × модель × ознака",
        "",
        "Середнє по двадцяти розбиттях. Метрика перестановки — ROC-AUC.",
        "",
        "| Платформа | Модель | " + " | ".join(F0) + " |",
        "|---" * (len(F0) + 2) + "|",
    ]
    perm = importance[importance["міра"] == "permutation"]
    means = perm.groupby(["платформа", "модель", "ознака"])["важливість"].mean()
    for platform in PLATFORM_ORDER:
        for model in MODEL_ORDER:
            cells = [_signed(float(means.loc[(platform, model, f)])) for f in F0]
            out.append(f"| {platform} | {model} | " + " | ".join(cells) + " |")

    out += [
        "",
        "---",
        "",
        "## SHAP на LightGBM: платформа × ознака",
        "",
        "Середнє абсолютне значення на перевірочній частині, усереднене "
        "по двадцяти розбиттях.",
        "Поруч — важливість тих самих ознак у EXP-006 регресійної лінії.",
        "",
        "| Платформа | Ознака | SHAP класифікаційна | SHAP EXP-006 | Ранг класиф. | Ранг EXP-006 |",
        "|---|---|---|---|---|---|",
    ]
    shap_block = importance[
        (importance["міра"] == "SHAP") & (importance["модель"] == "LightGBM")
    ]
    shap_means = shap_block.groupby(["платформа", "ознака"])["важливість"].mean()
    for platform in PLATFORM_ORDER:
        classification = shap_means.loc[platform]
        regression = exp006[exp006["платформа"] == platform].set_index("ознака")[
            "важливість_EXP006"
        ]
        rank_c = classification.rank(ascending=False, method="min")
        rank_r = regression.rank(ascending=False, method="min")
        for feature in F0:
            out.append(
                f"| {platform} | {feature} | {_num(float(classification[feature]), 6)} | "
                f"{_num(float(regression[feature]), 6)} | "
                f"{int(rank_c[feature])} | {int(rank_r[feature])} |"
            )

    out += [
        "",
        "---",
        "",
        "## τ між лініями і смуга",
        "",
        f"Міра — SHAP на LightGBM в обох лініях. Смуги другого доповнення "
        f"ADR-005: не нижче {_num(TAU_HIGH, 1)} — узгоджені; "
        f"від {_num(TAU_LOW, 1)} до {_num(TAU_HIGH, 1)} — часткова "
        f"узгодженість; нижче {_num(TAU_LOW, 1)} — розходяться.",
        "",
        "| Платформа | τ | p | Смуга |",
        "|---|---|---|---|",
    ]
    for row in lines_tau.itertuples(index=False):
        out.append(
            f"| {row.платформа} | {_signed(row.tau)} | {_num(row.p_value)} | "
            f"**{row.смуга}** |"
        )

    minimum = lines_tau["tau"].min()
    if minimum < TAU_LOW:
        verdict = "розходяться"
    elif lines_tau["tau"].min() >= TAU_HIGH:
        verdict = "узгоджені"
    else:
        verdict = "часткова узгодженість"
    out += [
        "",
        f"Спрацювала смуга за найгіршою платформою: **{verdict}** "
        f"(мінімальна τ дорівнює {_signed(float(minimum))}).",
        "",
        "---",
        "",
        "## τ між платформами: дві лінії поруч",
        "",
        "| Пара | τ класифікаційна | p | τ регресійна (EXP-006) | 95% ДІ регресійної |",
        "|---|---|---|---|---|",
    ]
    for row in platforms_tau.itertuples(index=False):
        out.append(
            f"| {row.пара} | {_signed(row.tau_класифікаційна)} | "
            f"{_num(row.p_класифікаційна)} | {_signed(row.tau_регресійна)} | "
            f"{row.ДІ_регресійна} |"
        )

    out += [
        "",
        "---",
        "",
        "## Стійкість упорядкування",
        "",
        "Упорядкування семи ознак обчислене окремо на кожному з двадцяти "
        "розбиттів.",
        "Середній попарний τ береться по всіх 190 парах розбиттів.",
        "",
        "| Міра | Платформа | Середній попарний τ | Мінімум | Максимум | Ознак на першому місці |",
        "|---|---|---|---|---|---|",
    ]
    for row in summary.itertuples(index=False):
        out.append(
            f"| {row.міра} | {row.платформа} | "
            f"{_signed(row.середній_попарний_tau)} | {_signed(row.мінімум_tau)} | "
            f"{_signed(row.максимум_tau)} | {row.ознак_на_першому_місці} |"
        )

    out += [
        "",
        "### Хто опинявся на першому місці",
        "",
        "| Міра | Платформа | Ознака | Разів із 20 | Частка |",
        "|---|---|---|---|---|",
    ]
    for row in leaders.itertuples(index=False):
        out.append(
            f"| {row.міра} | {row.платформа} | {row.ознака} | "
            f"{row.разів_перша} | {_num(row.частка)} |"
        )

    out += [
        "",
        "### Розкид рангу кожної ознаки",
        "",
        "Ранг 1 — найважливіша ознака.",
        "",
        "| Міра | Платформа | Ознака | Мінімум | Медіана | Максимум |",
        "|---|---|---|---|---|---|",
    ]
    for row in spread.itertuples(index=False):
        out.append(
            f"| {row.міра} | {row.платформа} | {row.ознака} | "
            f"{row.ранг_мінімум} | {_num(row.ранг_медіана, 1)} | "
            f"{row.ранг_максимум} |"
        )
    out.append("")
    return out


def main() -> None:
    features = pd.read_parquet(FEATURES_PATH)
    targets = pd.read_parquet(TARGETS_PATH)
    clean = pd.read_parquet(CLEAN_PATH, columns=["IDLink", "title_norm"])
    clean["IDLink"] = clean["IDLink"].astype("int64")
    frame = features.merge(targets, on="IDLink", validate="one_to_one").merge(
        clean, on="IDLink", validate="one_to_one"
    )
    folds = pd.read_parquet(FOLDS_PATH)

    print("обчислення важливості:")
    importance = collect(frame, folds)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    importance.to_parquet(OUT_PATH, index=False)

    exp006 = exp006_importance()
    lines_tau = tau_between_lines(importance, exp006)
    platforms_tau = tau_between_platforms(importance)
    summary, leaders, spread = stability(importance)

    lines = report_lines(
        importance, exp006, lines_tau, platforms_tau, summary, leaders, spread
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nважливість: {OUT_PATH.relative_to(PROJECT_ROOT)} — "
          f"{len(importance)} рядків")
    print(f"звіт: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
