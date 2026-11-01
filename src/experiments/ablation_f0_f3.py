"""EXP-007: абляція похідних ознак класифікаційної лінії (E1—E4).

Запуск:
    .venv/bin/python -m src.experiments.ablation_f0_f3 [--estimate-only]

Протокол — `experiments/EXP-007_ablation-derived-features/hypothesis.md`,
складений і закомічений до написання цього коду.

Сітка: три платформи × три пороги мітки × чотири набори ознак × чотири моделі
× двадцять розбиттів. Разом 2 880 навчань підсумкових моделей і втричі більше
навчань на підбір гіперпараметрів.

**Статистична перевірка тут не робиться.** Прогін дає матрицю метрик без
усереднення; правило рішення п'ятого доповнення ADR-006 застосовує до неї
окремий крок. Поділ навмисний, щоб числа і висновки не змішувалися.

**Стійкість до переривання.** Результат кожної комбінації платформи,
перцентиля, набору і моделі одразу лягає окремим файлом
у `data/interim/ablation_parts/`. Повторний запуск пропускає ті комбінації,
для яких файл уже є, тому прогін можна перервати й продовжити. Зведений
`data/processed/ablation_results.parquet` збирається з частин наприкінці.

**Що куди не потрапляє.** `cv_folds.parquet` і `features_derived.parquet`
тільки читаються. Склад наборів ознак береться з `src/features/derived.py`
і тут не змінюється.
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from src.features.derived import F0, F1, F2, F3

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features_derived.parquet"
TARGETS_PATH = PROJECT_ROOT / "data" / "processed" / "targets_classification.parquet"
FOLDS_PATH = PROJECT_ROOT / "data" / "processed" / "cv_folds.parquet"
CLEAN_PATH = PROJECT_ROOT / "data" / "interim" / "news_clean.parquet"
PARTS_DIR = PROJECT_ROOT / "data" / "interim" / "ablation_parts"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "ablation_results.parquet"

RANDOM_STATE = 42

PLATFORMS = {"Facebook": "fb", "GooglePlus": "gp", "LinkedIn": "li"}
PERCENTILES = ("25", "20", "33")
SETS = {"F0": F0, "F1": F1, "F2": F2, "F3": F3}

# Сітка зафіксована в картці EXP-007 до прогону: по одному параметру
# складності на модель, по три значення.
GRIDS = {
    "LogisticRegression": ("C", (0.1, 1.0, 10.0)),
    "RandomForest": ("max_depth", (None, 8, 16)),
    "LightGBM": ("num_leaves", (15, 31, 63)),
    "CatBoost": ("depth", (4, 6, 8)),
}
MODELS = tuple(GRIDS)

# Сітка порогів класифікації. Поріг не є гіперпараметром моделі: він
# застосовується до вже готових оцінок ймовірності.
THRESHOLD_GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)

# Внутрішня validation-підвибірка: один фолд з чотирьох, тобто чверть
# навчальної частини. Групування за заголовком зберігається і тут.
INNER_SPLITS = 4


def build_model(name: str, value):
    """Модель з одним підібраним параметром; решта — за замовчуванням."""
    if name == "LogisticRegression":
        return LogisticRegression(C=value, max_iter=1000, random_state=RANDOM_STATE)
    if name == "RandomForest":
        return RandomForestClassifier(
            max_depth=value, n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        )
    if name == "LightGBM":
        return LGBMClassifier(num_leaves=value, random_state=RANDOM_STATE, verbose=-1)
    if name == "CatBoost":
        return CatBoostClassifier(
            depth=value,
            iterations=200,
            verbose=0,
            random_seed=RANDOM_STATE,
            allow_writing_files=False,
        )
    raise ValueError(f"невідома модель: {name}")


def fit_score(
    name: str,
    value,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Навчання і оцінки ймовірності на навчальній та оцінюваній частинах.

    Масштабування для логістичної регресії вчиться лише на навчальній
    частині поданого поділу — і при підборі гіперпараметра, і при
    підсумковому навчанні (ADR-002, третій рівень схеми).
    """
    if name == "LogisticRegression":
        scaler = StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        x_eval = scaler.transform(x_eval)

    model = build_model(name, value)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(x_train, y_train)

    return model.predict_proba(x_train)[:, 1], model.predict_proba(x_eval)[:, 1]


def best_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Поріг, що максимізує F1 на поданій частині.

    Викликається лише на навчальних рядках: перевірочна частина фолда
    в підборі порога участі не бере (п'яте доповнення ADR-006).
    """
    best, best_value = 0.5, -1.0
    for threshold in THRESHOLD_GRID:
        value = f1_score(y_true, (scores >= threshold).astype(int), zero_division=0)
        if value > best_value:
            best, best_value = float(threshold), value
    return best


def tune(
    name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
) -> object:
    """Підбір параметра на внутрішній validation-підвибірці.

    Підвибірка виділяється `StratifiedGroupKFold`, тому групування
    за заголовком діє і всередині навчальної частини. Критерій підбору —
    той самий, що й головна метрика роботи: F1 на валідаційній частині
    при порозі, підібраному на внутрішній навчальній.
    """
    _, values = GRIDS[name]
    splitter = StratifiedGroupKFold(
        n_splits=INNER_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    inner_train, inner_val = next(
        splitter.split(x_train, y_train, groups)
    )

    best, best_value = values[0], -1.0
    for value in values:
        train_scores, val_scores = fit_score(
            name,
            value,
            x_train[inner_train],
            y_train[inner_train],
            x_train[inner_val],
        )
        threshold = best_threshold(y_train[inner_train], train_scores)
        score = f1_score(
            y_train[inner_val],
            (val_scores >= threshold).astype(int),
            zero_division=0,
        )
        if score > best_value:
            best, best_value = value, score
    return best


def metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Сім метрик на перевірочній частині фолда."""
    tuned = (scores >= threshold).astype(int)
    half = (scores >= 0.5).astype(int)
    return {
        "F1_підібраний": f1_score(y_true, tuned, zero_division=0),
        "F1_05": f1_score(y_true, half, zero_division=0),
        "Accuracy": accuracy_score(y_true, tuned),
        "Precision": precision_score(y_true, tuned, zero_division=0),
        "Recall": recall_score(y_true, tuned, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, scores),
        "PR_AUC": average_precision_score(y_true, scores),
    }


def run_combo(
    frame: pd.DataFrame,
    folds: pd.DataFrame,
    platform: str,
    percentile: str,
    set_name: str,
    model_name: str,
) -> pd.DataFrame:
    """Двадцять розбиттів однієї комбінації."""
    target = f"y_{PLATFORMS[platform]}_{percentile}"
    columns = SETS[set_name]
    indexed = frame.set_index("IDLink")

    part = folds[
        (folds["platform"] == platform) & (folds["percentile"] == percentile)
    ]
    splits = sorted(
        part[["repeat", "fold"]].drop_duplicates().itertuples(index=False, name=None)
    )

    records = []
    for repeat, fold in splits:
        current = part[(part["repeat"] == repeat) & (part["fold"] == fold)]
        train_ids = current.loc[current["part"] == "train", "IDLink"].to_numpy()
        test_ids = current.loc[current["part"] == "test", "IDLink"].to_numpy()

        train = indexed.loc[train_ids]
        test = indexed.loc[test_ids]
        x_train = train[columns].to_numpy(dtype=np.float64)
        x_test = test[columns].to_numpy(dtype=np.float64)
        y_train = train[target].to_numpy(dtype=np.int8)
        y_test = test[target].to_numpy(dtype=np.int8)
        groups = train["title_norm"].to_numpy()

        value = tune(model_name, x_train, y_train, groups)
        train_scores, test_scores = fit_score(
            model_name, value, x_train, y_train, x_test
        )
        threshold = best_threshold(y_train, train_scores)

        row = {
            "платформа": platform,
            "перцентиль": percentile,
            "набір": set_name,
            "модель": model_name,
            "repeat": int(repeat),
            "fold": int(fold),
            "параметр": GRIDS[model_name][0],
            "значення_параметра": str(value),
            "поріг": threshold,
            "n_train": int(len(train_ids)),
            "n_test": int(len(test_ids)),
        }
        row.update(metrics(y_test, test_scores, threshold))
        records.append(row)

    return pd.DataFrame(records)


def part_path(platform: str, percentile: str, set_name: str, model_name: str) -> Path:
    return PARTS_DIR / f"{PLATFORMS[platform]}_{percentile}_{set_name}_{model_name}.parquet"


def combos() -> list[tuple[str, str, str, str]]:
    return [
        (platform, percentile, set_name, model_name)
        for platform in PLATFORMS
        for percentile in PERCENTILES
        for set_name in SETS
        for model_name in MODELS
    ]


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    features = pd.read_parquet(FEATURES_PATH)
    targets = pd.read_parquet(TARGETS_PATH)
    clean = pd.read_parquet(CLEAN_PATH, columns=["IDLink", "title_norm"])
    clean["IDLink"] = clean["IDLink"].astype("int64")

    frame = features.merge(targets, on="IDLink", validate="one_to_one")
    frame = frame.merge(clean, on="IDLink", validate="one_to_one")
    folds = pd.read_parquet(FOLDS_PATH)
    return frame, folds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="порахувати одну комбінацію, оцінити час і вийти",
    )
    args = parser.parse_args()

    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    frame, folds = load_inputs()
    todo = [c for c in combos() if not part_path(*c).exists()]
    done = len(combos()) - len(todo)

    print(f"комбінацій усього: {len(combos())}, уже пораховано: {done}, "
          f"лишилося: {len(todo)}")
    if not todo:
        print("усі комбінації пораховані")
    else:
        # Оцінка часу за однією комбінацією. Береться найдорожча з тих,
        # що лишилися, щоб оцінка не була оптимістичною.
        probe = max(todo, key=lambda c: (c[3] == "LightGBM", c[2] == "F3"))
        started = time.perf_counter()
        result = run_combo(frame, folds, *probe)
        elapsed = time.perf_counter() - started
        result.to_parquet(part_path(*probe), index=False)
        todo.remove(probe)

        print(f"пробна комбінація {probe}: {elapsed:.1f} с на 20 розбиттів")
        print(f"оцінка часу на решту {len(todo)} комбінацій: "
              f"{elapsed * len(todo) / 60:.0f} хв (верхня межа)")

        if args.estimate_only:
            print("зупинка за --estimate-only")
            return

        run_started = time.perf_counter()
        for index, combo in enumerate(todo, start=1):
            combo_started = time.perf_counter()
            result = run_combo(frame, folds, *combo)
            result.to_parquet(part_path(*combo), index=False)
            print(
                f"[{index}/{len(todo)}] {combo} — "
                f"{time.perf_counter() - combo_started:.1f} с",
                flush=True,
            )
        print(f"прогін: {(time.perf_counter() - run_started) / 60:.1f} хв")

    parts = [pd.read_parquet(part_path(*c)) for c in combos()]
    results = pd.concat(parts, ignore_index=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(OUT_PATH, index=False)
    print(f"зведено: {OUT_PATH.relative_to(PROJECT_ROOT)} — {len(results)} рядків")


if __name__ == "__main__":
    main()
