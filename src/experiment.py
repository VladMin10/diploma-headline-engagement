"""Прогін одного експерименту за файлом конфігурації.

Запуск:
    .venv/bin/python -m src.experiment --config experiments/EXP-001_.../config.json

Скрипт читає конфігурацію, збирає потрібні рівні ознак, навчає моделі,
рахує метрики, будує довірчі інтервали і записує results.json поруч
із конфігурацією. Числа мають єдине походження — цей запуск.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform as py_platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.model_selection import TimeSeriesSplit

from src.data.load import CLEAN_PATH, PROJECT_ROOT, TARGETS
from src.data.split import SPLIT_PATH, compute_split, load_split
from src.eval.bootstrap import (
    bootstrap_difference,
    bootstrap_indices,
    bootstrap_metrics,
    bootstrap_within_source,
    stratified_bootstrap_indices,
)
from src.eval.metrics import spearman
from src.features.assemble import select_rows
from src.models.zoo import build_pipeline


def collect_versions() -> dict[str, str]:
    return {
        "python": py_platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
        "scipy": scipy.__version__,
    }


def variant_spec(value, config: dict) -> tuple[list[str], bool]:
    """Опис варіанта: які рівні ознак і чи ввімкнено місяць.

    Два формати заради сумісності. EXP-001 задавав варіанти як
    `{"T1": true, "T1_no_month": false}`, де значення — прапорець місяця,
    а рівень один і той самий. Фаза 5 порівнює комбінації рівнів, тому
    варіант задається словником `{"levels": [...], "include_month": ...}`.
    Старий формат читається без змін, щоб числа EXP-001 відтворювалися.
    """
    if isinstance(value, bool):
        return list(config["feature_levels"]), value
    levels = list(value["levels"])
    return levels, bool(value.get("include_month", False))


def select_alpha(
    train: pd.DataFrame,
    target_col: str,
    alphas: list[float],
    n_splits: int,
    include_month: bool,
    level: str | list[str],
) -> dict:
    """Підбір alpha крос-валідацією з розширенням вікна всередині train.

    Навчальне вікно нарощується, валідаційне зсувається вперед у часі —
    схема відтворює в малому масштабі структуру основного спліту (ADR-002).
    Тестова вибірка в підборі участі не бере.

    Критерій — Spearman ρ, головна метрика роботи.
    """
    ordered = train.sort_values("PublishDate").reset_index(drop=True)
    y = ordered[target_col].to_numpy()
    splitter = TimeSeriesSplit(n_splits=n_splits)

    per_alpha = {}
    for alpha in alphas:
        scores = []
        for fit_idx, val_idx in splitter.split(ordered):
            pipe = build_pipeline("ridge", level=level, include_month=include_month, alpha=alpha)
            pipe.fit(ordered.iloc[fit_idx], y[fit_idx])
            pred = pipe.predict(ordered.iloc[val_idx])
            scores.append(spearman(y[val_idx], pred))
        per_alpha[alpha] = {
            "per_fold": [round(float(s), 6) for s in scores],
            "mean": float(np.nanmean(scores)),
        }

    best = max(per_alpha, key=lambda a: per_alpha[a]["mean"])
    return {
        "alpha_selected": best,
        "cv_spearman_mean": per_alpha[best]["mean"],
        "cv_spearman_per_fold": per_alpha[best]["per_fold"],
        "grid": {str(a): per_alpha[a] for a in alphas},
        "n_splits": n_splits,
    }


def source_groups(test: pd.DataFrame) -> np.ndarray:
    """Мітки джерела для внутрішньоджерельної метрики."""
    return test["Source"].fillna("__відсутнє__").astype(str).to_numpy()


def run_variant(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str,
    config: dict,
    include_month: bool,
    levels: list[str] | None = None,
) -> dict:
    """Одна комбінація рівнів ознак × одна платформа: підбір, навчання, оцінка."""
    level = levels if levels is not None else config["feature_levels"][0]
    y_train = train[target_col].to_numpy()
    y_test = test[target_col].to_numpy()

    cv = select_alpha(
        train,
        target_col,
        [float(a) for a in config["ridge_alphas"]],
        int(config["cv_splits"]),
        include_month,
        level,
    )

    predictions = {}
    fitted = {}
    for model_name in config["models"]:
        alpha = cv["alpha_selected"] if model_name == "ridge" else None
        pipe = build_pipeline(model_name, level=level, include_month=include_month, alpha=alpha)
        pipe.fit(train, y_train)
        predictions[model_name] = pipe.predict(test)
        fitted[model_name] = pipe

    indices = bootstrap_indices(
        len(y_test),
        n_iterations=int(config["bootstrap_iters"]),
        random_state=int(config["random_state"]),
    )

    test_metrics = {
        name: bootstrap_metrics(y_test, pred, indices)
        for name, pred in predictions.items()
    }

    # Внутрішньоджерельна кореляція. Зафіксована як обов'язкова метрика
    # 7 серпня 2026 року, до запуску Фази 5: загальна ρ значною мірою
    # відображає різницю між виданнями, тоді як практичне питання роботи
    # стоїть при фіксованому виданні.
    within = {}
    if config.get("within_source_metric", True):
        groups = source_groups(test)
        strat_indices = stratified_bootstrap_indices(
            groups,
            n_iterations=int(config["bootstrap_iters"]),
            random_state=int(config["random_state"]),
        )
        for name, pred in predictions.items():
            within[name] = bootstrap_within_source(y_test, pred, groups, strat_indices)

    comparisons = {}
    if "ridge" in predictions and "median" in predictions:
        comparisons["ridge_minus_median"] = bootstrap_difference(
            y_test, predictions["ridge"], predictions["median"], indices
        )

    # Перевірка ADR-002: розрив між крос-валідацією і тестом.
    test_rho = test_metrics.get("ridge", {}).get("spearman", {})
    ci_width = float("nan")
    if test_rho:
        ci_width = test_rho["ci_high"] - test_rho["ci_low"]
    gap = float("nan")
    if test_rho and np.isfinite(cv["cv_spearman_mean"]):
        gap = cv["cv_spearman_mean"] - test_rho["value"]

    ridge_step = fitted["ridge"].named_steps["features"] if "ridge" in fitted else None
    n_features = (
        int(len(ridge_step.get_feature_names_out())) if ridge_step is not None else None
    )

    # Порівняння останнього фолда крос-валідації з тестом. Середнє по фолдах
    # занижене конструктивно: у схемі з розширенням вікна перші фолди
    # навчаються на малій частині train. Останній фолд має обсяг навчання,
    # найближчий до фінального, тому саме він зіставний з тестом.
    last_fold = cv["cv_spearman_per_fold"][-1] if cv["cv_spearman_per_fold"] else float("nan")
    gap_last = (
        last_fold - test_rho["value"] if test_rho and np.isfinite(last_fold) else float("nan")
    )

    return {
        "predictions": {k: v.tolist() for k, v in predictions.items()},
        "include_month": include_month,
        "n_features": n_features,
        "cv": cv,
        "test": test_metrics,
        "test_within_source": within,
        "comparisons": comparisons,
        "drift_check": {
            "cv_spearman_mean": cv["cv_spearman_mean"],
            "test_spearman": test_rho.get("value"),
            "gap_cv_minus_test": gap,
            "test_ci_width": ci_width,
            "gap_within_two_ci_widths": bool(
                np.isfinite(gap) and np.isfinite(ci_width) and abs(gap) <= 2 * ci_width
            ),
            "last_fold_spearman": last_fold,
            "gap_last_fold_minus_test": gap_last,
            "gap_last_fold_within_two_ci_widths": bool(
                np.isfinite(gap_last)
                and np.isfinite(ci_width)
                and abs(gap_last) <= 2 * ci_width
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    df = pd.read_parquet(CLEAN_PATH)
    split_info = load_split(SPLIT_PATH)
    split = compute_split(df, split_info["boundary"])
    train = df.loc[split.train_mask].copy()
    test = df.loc[split.test_mask].copy()

    specs = {
        name: variant_spec(value, config)
        for name, value in config["variants"].items()
    }

    # Звуження до анотованої підвибірки. Робиться після спліту, бо
    # `compute_split` шукає групи повторюваних заголовків серед усіх новин
    # (див. assemble.select_rows). Звуження застосовується до всіх варіантів
    # одразу, навіть до тих, що обходяться без T2: Δρ обчислюється парно
    # на тих самих рядках, інакше різниця міряла б ще й зміну вибірки.
    # Прапорець потрібен окремо від наявності T2. Базові лінії Фази 5,
    # EXP-003 (T1) і EXP-005b (T1+T3), рівня T2 не містять, але мусять
    # бути оцінені на тих самих рядках, що EXP-004 і EXP-005, інакше
    # Δρ між ними міряло б ще й зміну вибірки.
    uses_t2 = any("T2" in levels for levels, _ in specs.values()) or config.get(
        "restrict_to_annotated", False
    )
    if uses_t2:
        train = select_rows(train, ["T2"])
        test = select_rows(test, ["T2"])
    print(
        f"train {len(train)}, test {len(test)}"
        + (" (анотована підвибірка)" if uses_t2 else "")
    )

    comparison_model = config.get("comparison_model", "ridge")
    comparison_pairs = [tuple(pair) for pair in config.get("comparisons", [])]
    if not comparison_pairs and len(specs) == 2:
        comparison_pairs = [tuple(specs)]

    results = {}
    for target in config["platforms"]:
        target_col = f"{target}_log1p"
        results[target] = {}
        for variant, (levels, include_month) in specs.items():
            print(f"  {target} / {variant} ({'+'.join(levels)}) ...", flush=True)
            results[target][variant] = run_variant(
                train, test, target_col, config, include_month, levels
            )
            for model_name in config["models"]:
                if model_name == "median":
                    continue
                rho = results[target][variant]["test"][model_name]["spearman"]
                print(
                    f"    {model_name:9s} Spearman ρ = {rho['value']:.4f} "
                    f"[{rho['ci_low']:.4f}; {rho['ci_high']:.4f}]"
                )

        # Порівняння варіантів між собою на однакових бутстреп-вибірках
        # і на однакових рядках тесту.
        y_test = test[target_col].to_numpy()
        indices = bootstrap_indices(
            len(y_test),
            n_iterations=int(config["bootstrap_iters"]),
            random_state=int(config["random_state"]),
        )
        for a, b in comparison_pairs:
            key = f"comparison_{a}_minus_{b}"
            results[target][key] = bootstrap_difference(
                y_test,
                np.asarray(results[target][a]["predictions"][comparison_model]),
                np.asarray(results[target][b]["predictions"][comparison_model]),
                indices,
            )
            d = results[target][key]["spearman"]
            print(
                f"    Δρ ({a} − {b}) = {d['difference']:+.4f} "
                f"[{d['ci_low']:+.4f}; {d['ci_high']:+.4f}], "
                f"виключає нуль: {d['excludes_zero']}"
            )

        # Прогнози потрібні для порівнянь між експериментами. У EXP-001
        # вони не зберігалися, бо обидва варіанти жили в одному прогоні.
        # Фаза 5 порівнює конфігурації з різних папок, тому прогнози
        # зберігаються, а тотожність рядків тесту перевіряється за хешем.
        if config.get("save_predictions", False):
            results[target]["y_test"] = y_test.tolist()
        else:
            for variant in specs:
                results[target][variant].pop("predictions", None)

    config["versions"] = collect_versions()
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    out = {
        "exp_id": config["exp_id"],
        "generated_by": "src/experiment.py",
        "config_file": str(config_path.relative_to(PROJECT_ROOT)),
        "versions": config["versions"],
        "data": {
            "clean_file": str(CLEAN_PATH.relative_to(PROJECT_ROOT)),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "boundary": split_info["boundary"],
            "restricted_to_annotated": bool(uses_t2),
            # Відбиток рядків тесту. Порівняння Δρ між експериментами
            # має сенс лише тоді, коли обидва оцінені на тих самих рядках
            # у тому самому порядку; `src/compare.py` звіряє цей хеш.
            "test_ids_sha256": hashlib.sha256(
                ",".join(str(int(i)) for i in test["IDLink"]).encode()
            ).hexdigest(),
        },
        "results": results,
    }
    results_path = config_path.parent / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("Записано:", results_path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
