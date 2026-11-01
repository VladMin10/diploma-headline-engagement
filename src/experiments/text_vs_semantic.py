"""EXP-008: змістові ознаки проти текстового представлення (E5).

Запуск:
    .venv/bin/python -m src.experiments.text_vs_semantic [--estimate-only]

Протокол — `experiments/EXP-008_text-vs-semantic/hypothesis.md`, складений
і закомічений до написання цього коду.

Сітка: три платформи × п'ять варіантів представлення × двадцять розбиттів,
разом 300 навчань підсумкових моделей і вчетверо більше на підбір `C`.
Поріг мітки один, основний: EXP-007 показав стійкість картини до його вибору.

**Витік через словник.** `TfidfVectorizer` навчається лише на навчальній
частині кожного фолда. Слова, що трапляються тільки в перевірочній частині,
у словник не потрапляють і для моделі не існують — так само, як у реальному
застосуванні модель не знає слів із майбутніх заголовків. Фіт вектаризатора
на повному наборі був би витоком того самого роду, що й фіт масштабування
(ADR-002, третій рівень схеми).

**Статистична перевірка тут не робиться** — окремим кроком, як у EXP-007.

**Стійкість до переривання.** Результат кожної комбінації платформи
і варіанта одразу лягає окремим файлом у `data/interim/e5_parts/`; повторний
запуск пропускає пораховане.
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
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

from src.features.derived import F0, F3

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features_derived.parquet"
TARGETS_PATH = PROJECT_ROOT / "data" / "processed" / "targets_classification.parquet"
FOLDS_PATH = PROJECT_ROOT / "data" / "processed" / "cv_folds.parquet"
CLEAN_PATH = PROJECT_ROOT / "data" / "interim" / "news_clean.parquet"
PARTS_DIR = PROJECT_ROOT / "data" / "interim" / "e5_parts"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "text_vs_semantic_results.parquet"

RANDOM_STATE = 42
PERCENTILE = "25"
PLATFORMS = {"Facebook": "fb", "GooglePlus": "gp", "LinkedIn": "li"}

# П'ять варіантів представлення, зафіксовані в картці EXP-008.
# Значення: (чи входить TF-IDF, перелік щільних колонок).
VARIANTS = {
    "TFIDF": (True, None),
    "F0": (False, F0),
    "F3": (False, F3),
    "TFIDF+F0": (True, F0),
    "TFIDF+F3": (True, F3),
}
VARIANT_ORDER = list(VARIANTS)

# Параметри TF-IDF зафіксовані в картці до прогону і тут не змінюються.
TFIDF_PARAMS = dict(
    ngram_range=(1, 2),
    min_df=3,
    max_features=5000,
    sublinear_tf=True,
    lowercase=True,
    stop_words=None,
)

# Сітка ширша за EXP-007: розмірність TF-IDF на три порядки більша.
C_GRID = (0.01, 0.1, 1.0, 10.0)
THRESHOLD_GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)
INNER_SPLITS = 4


def build_matrices(
    train: pd.DataFrame, evaluate: pd.DataFrame, variant: str
) -> tuple:
    """Матриці ознак варіанта. Усе, що вчиться, вчиться на `train`.

    Повертає також розмір словника: він потрібен у звіті й залежить
    від навчальної частини, тому змінюється від фолда до фолда.
    """
    use_tfidf, dense_columns = VARIANTS[variant]
    blocks_train, blocks_eval = [], []
    vocabulary = 0

    if use_tfidf:
        vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
        blocks_train.append(vectorizer.fit_transform(train["Title"]))
        blocks_eval.append(vectorizer.transform(evaluate["Title"]))
        vocabulary = len(vectorizer.vocabulary_)

    if dense_columns is not None:
        dense_train = train[dense_columns].to_numpy(dtype=np.float64)
        dense_eval = evaluate[dense_columns].to_numpy(dtype=np.float64)
        scaler = StandardScaler().fit(dense_train)
        blocks_train.append(sparse.csr_matrix(scaler.transform(dense_train)))
        blocks_eval.append(sparse.csr_matrix(scaler.transform(dense_eval)))

    x_train = sparse.hstack(blocks_train, format="csr")
    x_eval = sparse.hstack(blocks_eval, format="csr")
    return x_train, x_eval, vocabulary


def fit_score(value: float, x_train, y_train: np.ndarray, x_eval):
    """Навчання логістичної регресії і оцінки ймовірності."""
    model = LogisticRegression(C=value, max_iter=1000, random_state=RANDOM_STATE)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(x_train, y_train)
    return model.predict_proba(x_train)[:, 1], model.predict_proba(x_eval)[:, 1]


def best_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Поріг, що максимізує F1 на поданій частині."""
    best, best_value = 0.5, -1.0
    for threshold in THRESHOLD_GRID:
        value = f1_score(y_true, (scores >= threshold).astype(int), zero_division=0)
        if value > best_value:
            best, best_value = float(threshold), value
    return best


def tune(
    train: pd.DataFrame, y_train: np.ndarray, groups: np.ndarray, variant: str
) -> float:
    """Підбір `C` на внутрішній validation-підвибірці.

    Вектаризатор і масштабування всередині підбору теж навчаються лише
    на внутрішній навчальній частині: інакше підбір бачив би валідаційні
    рядки через словник.
    """
    splitter = StratifiedGroupKFold(
        n_splits=INNER_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    inner_train, inner_val = next(
        splitter.split(np.zeros((len(train), 1)), y_train, groups)
    )
    x_inner, x_val, _ = build_matrices(
        train.iloc[inner_train], train.iloc[inner_val], variant
    )

    best, best_value = C_GRID[0], -1.0
    for value in C_GRID:
        inner_scores, val_scores = fit_score(
            value, x_inner, y_train[inner_train], x_val
        )
        threshold = best_threshold(y_train[inner_train], inner_scores)
        score = f1_score(
            y_train[inner_val],
            (val_scores >= threshold).astype(int),
            zero_division=0,
        )
        if score > best_value:
            best, best_value = value, score
    return best


def metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
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
    frame: pd.DataFrame, folds: pd.DataFrame, platform: str, variant: str
) -> pd.DataFrame:
    """Двадцять розбиттів однієї комбінації платформи і варіанта."""
    target = f"y_{PLATFORMS[platform]}_{PERCENTILE}"
    indexed = frame.set_index("IDLink")
    part = folds[
        (folds["platform"] == platform) & (folds["percentile"] == PERCENTILE)
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
        y_train = train[target].to_numpy(dtype=np.int8)
        y_test = test[target].to_numpy(dtype=np.int8)
        groups = train["title_norm"].to_numpy()

        value = tune(train, y_train, groups, variant)
        x_train, x_test, vocabulary = build_matrices(train, test, variant)
        train_scores, test_scores = fit_score(value, x_train, y_train, x_test)
        threshold = best_threshold(y_train, train_scores)

        row = {
            "платформа": platform,
            "перцентиль": PERCENTILE,
            "варіант": variant,
            "модель": "LogisticRegression",
            "repeat": int(repeat),
            "fold": int(fold),
            "C": value,
            "поріг": threshold,
            "розмір_словника": int(vocabulary),
            "n_ознак": int(x_train.shape[1]),
            "n_train": int(len(train_ids)),
            "n_test": int(len(test_ids)),
        }
        row.update(metrics(y_test, test_scores, threshold))
        records.append(row)

    return pd.DataFrame(records)


def part_path(platform: str, variant: str) -> Path:
    return PARTS_DIR / f"{PLATFORMS[platform]}_{variant.replace('+', '_')}.parquet"


def combos() -> list[tuple[str, str]]:
    return [(platform, variant) for platform in PLATFORMS for variant in VARIANT_ORDER]


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    features = pd.read_parquet(FEATURES_PATH)
    targets = pd.read_parquet(TARGETS_PATH)
    clean = pd.read_parquet(CLEAN_PATH, columns=["IDLink", "Title", "title_norm"])
    clean["IDLink"] = clean["IDLink"].astype("int64")

    frame = features.merge(targets, on="IDLink", validate="one_to_one")
    frame = frame.merge(clean, on="IDLink", validate="one_to_one")
    folds = pd.read_parquet(FOLDS_PATH)
    return frame, folds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimate-only", action="store_true")
    args = parser.parse_args()

    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    frame, folds = load_inputs()
    todo = [c for c in combos() if not part_path(*c).exists()]
    done = len(combos()) - len(todo)
    print(f"комбінацій усього: {len(combos())}, уже пораховано: {done}, "
          f"лишилося: {len(todo)}")

    if todo:
        # Найдорожча з тих, що лишилися: комбінований варіант з TF-IDF.
        probe = max(todo, key=lambda c: (c[1].startswith("TFIDF"), len(c[1])))
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
