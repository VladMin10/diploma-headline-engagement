"""Діагностична перевірка F₀ проти F₃ на Facebook.

Запуск:
    .venv/bin/python -m src.experiments.probe_f0_f3

**Це діагностика, а не експеримент.** Картка не заводиться, в реєстрі
експериментів перевірка не з'являється, числа з неї в текст роботи
не потрапляють. Гіперпараметри лишаються дефолтними, підбір не робиться,
тому отримані числа не є результатом і не можуть ним стати. Правила
перевірки і правило тлумачення ΔF1 записані в `docs/JOURNAL.md` за 16 вересня
2026 року **до** запуску.

Мета одна: з'ясувати до написання коду E1—E4, чи дає інженерія похідних ознак
приріст узагалі і на яких моделях.

Обсяг навмисно вузький. Одна платформа — Facebook. Один поріг мітки — основний,
верхні 25%. Два набори ознак з країв абляції — F₀ і F₃. Дві моделі з чотирьох —
LightGBM і логістична регресія, тобто ансамблева і лінійна. Усі двадцять
розбиттів з `cv_folds.parquet`, бо саме вони є одиницями парного порівняння
в цій лінії.

Два протокольні рішення, оголошені в журналі і чинні для всієї лінії.
Поріг класифікації: основний 0,5, за ним рахується F1 для правила рішення
ADR-006; поріг, підібраний на навчальній частині фолда за максимумом F1,
наводиться довідково. Ваги класів не застосовуються, бо дисбаланс помірний,
а ROC-AUC і PR-AUC від порога і ваг не залежать.

Єдине відхилення від дефолтів, і воно не є підбором: логістичній регресії
задано `max_iter=1000`. Дефолтні 100 ітерацій lbfgs на цих даних до збіжності
не доходять, а незбіжна модель дала б число, яке нічого не вимірює. Це вимога
збіжності оптимізатора, а не вибір гіперпараметра за якістю. `random_state`
скрізь 42 (CLAUDE.md розділ 6, пункт 4).
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.features.derived import F0, F3

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features_derived.parquet"
TARGETS_PATH = PROJECT_ROOT / "data" / "processed" / "targets_classification.parquet"
FOLDS_PATH = PROJECT_ROOT / "data" / "processed" / "cv_folds.parquet"
REPORT_PATH = PROJECT_ROOT / "reports" / "tables" / "probe_f0_f3.md"

PLATFORM = "Facebook"
PERCENTILE = "25"
TARGET_COLUMN = "y_fb_25"
RANDOM_STATE = 42

SETS = {"F0": F0, "F3": F3}
MODELS = ("LightGBM", "LogisticRegression")

# Сітка для підбору порога на навчальній частині. Не гіперпараметр моделі:
# поріг застосовується до вже готових оцінок ймовірності.
THRESHOLD_GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ознаки з міткою і перелік розбиттів для Facebook, верхні 25%."""
    features = pd.read_parquet(FEATURES_PATH)
    targets = pd.read_parquet(TARGETS_PATH)[["IDLink", TARGET_COLUMN]]
    frame = features.merge(targets, on="IDLink", validate="one_to_one")

    folds = pd.read_parquet(FOLDS_PATH)
    folds = folds[
        (folds["platform"] == PLATFORM) & (folds["percentile"] == PERCENTILE)
    ]
    return frame, folds


def build_model(name: str):
    """Модель з дефолтними гіперпараметрами.

    Задані лише `random_state` для відтворюваності, `verbose` для тиші
    і `max_iter` для збіжності lbfgs. Жоден з трьох не підбирається
    за якістю і гіперпараметром у сенсі протоколу не є.
    """
    if name == "LightGBM":
        return LGBMClassifier(random_state=RANDOM_STATE, verbose=-1)
    if name == "LogisticRegression":
        return LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    raise ValueError(f"невідома модель: {name}")


def best_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Поріг, що максимізує F1 на навчальній частині фолда.

    Перевірочна частина в підборі участі не бере: підбір іде на тих самих
    рядках, на яких модель навчалася. Число наводиться довідково і в правило
    рішення ADR-006 не входить.
    """
    best, best_f1 = 0.5, -1.0
    for threshold in THRESHOLD_GRID:
        value = f1_score(y_true, (scores >= threshold).astype(int), zero_division=0)
        if value > best_f1:
            best, best_f1 = float(threshold), value
    return best


def run_split(
    frame: pd.DataFrame,
    train_ids: np.ndarray,
    test_ids: np.ndarray,
    model_name: str,
    set_name: str,
) -> dict:
    """Одна комбінація модель × набір на одному розбитті."""
    columns = SETS[set_name]
    indexed = frame.set_index("IDLink")
    train = indexed.loc[train_ids]
    test = indexed.loc[test_ids]

    x_train = train[columns].to_numpy(dtype=np.float64)
    x_test = test[columns].to_numpy(dtype=np.float64)
    y_train = train[TARGET_COLUMN].to_numpy(dtype=np.int8)
    y_test = test[TARGET_COLUMN].to_numpy(dtype=np.int8)

    if model_name == "LogisticRegression":
        # Масштабування вчиться лише на навчальній частині фолда
        # (ADR-002, третій рівень схеми; розділ 7 документа кафедри).
        scaler = StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

    model = build_model(model_name)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(x_train, y_train)

    train_scores = model.predict_proba(x_train)[:, 1]
    test_scores = model.predict_proba(x_test)[:, 1]
    tuned = best_threshold(y_train, train_scores)

    return {
        "модель": model_name,
        "набір": set_name,
        "F1_0.5": f1_score(y_test, (test_scores >= 0.5).astype(int), zero_division=0),
        "F1_підібраний": f1_score(
            y_test, (test_scores >= tuned).astype(int), zero_division=0
        ),
        "поріг_підібраний": tuned,
        "ROC_AUC": roc_auc_score(y_test, test_scores),
        "PR_AUC": average_precision_score(y_test, test_scores),
    }


def run(frame: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    """Усі двадцять розбиттів × дві моделі × два набори."""
    records = []
    splits = sorted(
        folds[["repeat", "fold"]].drop_duplicates().itertuples(index=False, name=None)
    )
    for repeat, fold in splits:
        part = folds[(folds["repeat"] == repeat) & (folds["fold"] == fold)]
        train_ids = part.loc[part["part"] == "train", "IDLink"].to_numpy()
        test_ids = part.loc[part["part"] == "test", "IDLink"].to_numpy()
        for model_name in MODELS:
            for set_name in SETS:
                row = run_split(frame, train_ids, test_ids, model_name, set_name)
                row["repeat"] = int(repeat)
                row["fold"] = int(fold)
                records.append(row)
    return pd.DataFrame(records)


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Середні по двадцяти розбиттях для кожної комбінації."""
    metrics = ["F1_0.5", "F1_підібраний", "ROC_AUC", "PR_AUC", "поріг_підібраний"]
    return results.groupby(["модель", "набір"], as_index=False)[metrics].mean()


def paired_deltas(results: pd.DataFrame) -> pd.DataFrame:
    """Парні різниці F₃ мінус F₀ на однакових розбиттях."""
    rows = []
    for model_name in MODELS:
        subset = results[results["модель"] == model_name]
        f0 = subset[subset["набір"] == "F0"].set_index(["repeat", "fold"])
        f3 = subset[subset["набір"] == "F3"].set_index(["repeat", "fold"])
        for metric in ("F1_0.5", "F1_підібраний", "ROC_AUC", "PR_AUC"):
            diff = (f3[metric] - f0[metric]).sort_index()
            rows.append(
                {
                    "модель": model_name,
                    "метрика": metric,
                    "середня_різниця": float(diff.mean()),
                    "мінімум": float(diff.min()),
                    "максимум": float(diff.max()),
                    "додатних_з_20": int((diff > 0).sum()),
                }
            )
    return pd.DataFrame(rows)


def _num(value: float, digits: int = 4) -> str:
    """Число з комою як десятковим роздільником (CLAUDE.md розділ 5)."""
    text = f"{value:+.{digits}f}" if value < 0 else f"{value:.{digits}f}"
    return text.replace(".", ",")


def _signed(value: float, digits: int = 4) -> str:
    return f"{value:+.{digits}f}".replace(".", ",")


def report_lines(
    summary: pd.DataFrame,
    deltas: pd.DataFrame,
    baseline: float,
    n_splits: int,
    n_rows: int,
    elapsed: float,
) -> list[str]:
    lines = [
        "# Діагностична перевірка: F₀ проти F₃ на Facebook",
        "",
        "Згенеровано `src/experiments/probe_f0_f3.py`.",
        "",
        "**Це діагностика, а не експеримент.** Картка не заводиться, в реєстрі",
        "перевірка не з'являється, числа в текст роботи не потрапляють.",
        "Гіперпараметри дефолтні, підбір не робиться, тому числа не є",
        "результатом і не можуть ним стати. Правила і правило тлумачення ΔF1",
        "записані в `docs/JOURNAL.md` за 16 вересня 2026 року до запуску.",
        "",
        f"Платформа {PLATFORM}, поріг мітки — верхні {PERCENTILE}%, "
        f"{n_rows} рядків, {n_splits} розбиттів.",
        f"Базовий рівень PR-AUC — фактична частка позитивного класу "
        f"{_num(baseline)}; для ROC-AUC — 0,5.",
        "",
        "### Середні по двадцяти розбиттях",
        "",
        "| Модель | Набір | F1 при 0,5 | F1 при підібраному | Підібраний поріг | ROC-AUC | PR-AUC |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.модель} | {row.набір} | {_num(getattr(row, '_2'))} | "
            f"{_num(row.F1_підібраний)} | {_num(row.поріг_підібраний, 2)} | "
            f"{_num(row.ROC_AUC)} | {_num(row.PR_AUC)} |"
        )

    lines += [
        "",
        "### Парні різниці F₃ мінус F₀ на однакових розбиттях",
        "",
        "| Модель | Метрика | Середня різниця | Мінімум | Максимум | Додатних із 20 |",
        "|---|---|---|---|---|---|",
    ]
    for row in deltas.itertuples(index=False):
        lines.append(
            f"| {row.модель} | {row.метрика} | {_signed(row.середня_різниця)} | "
            f"{_signed(row.мінімум)} | {_signed(row.максимум)} | "
            f"{row.додатних_з_20} |"
        )

    lines += [
        "",
        f"Час прогону: {elapsed:.1f} с.",
        "",
        "Різниці обчислені парно: F₃ і F₀ на кожному розбитті порівнюються",
        "між собою, а не через середні. Жодного статистичного критерію тут",
        "не застосовано: це діагностика, і правило рішення ADR-006 до неї",
        "не застосовується.",
        "",
    ]
    return lines


def main() -> None:
    started = time.perf_counter()
    frame, folds = load_inputs()
    results = run(frame, folds)
    elapsed = time.perf_counter() - started

    summary = summarise(results)
    deltas = paired_deltas(results)
    baseline = float(frame[TARGET_COLUMN].mean())
    n_splits = int(folds[["repeat", "fold"]].drop_duplicates().shape[0])

    lines = report_lines(
        summary, deltas, baseline, n_splits, len(frame), elapsed
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"звіт: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
