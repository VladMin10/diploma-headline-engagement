"""Три метрики роботи (ADR-003).

Усі три обчислюються на тих самих прогнозах тієї самої моделі. Функції
отримують два масиви — факт і прогноз — і нічого не знають ані про модель,
ані про ознаки.

Прогноз і факт подаються в логарифмічній шкалі, тобто як log1p
від популярності.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

METRIC_NAMES = ["spearman", "mae_log", "auc_nonzero"]


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Рангова кореляція Спірмена з поправкою на зв'язані ранги.

    Головна метрика роботи. Не визначена, якщо прогноз не має варіації,
    як у константної моделі: у такому разі повертається NaN, а не нуль.
    Нуль означав би «зв'язку немає», тоді як насправді величина
    не обчислюється.
    """
    if np.all(y_pred == y_pred[0]) or np.all(y_true == y_true[0]):
        return float("nan")
    return float(spearmanr(y_true, y_pred).statistic)


def mae_log(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def auc_nonzero(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """ROC AUC для події «популярність більша за нуль».

    Прогноз моделі використовується як ранжувальна оцінка. Окрема модель
    не навчається, поріг не вводиться, тому простору для підганяння немає.
    Оскільки цільова змінна подана як log1p, подія «більше нуля»
    відповідає ненульовому значенню log1p.
    """
    labels = (y_true > 0).astype(int)
    if labels.min() == labels.max():
        return float("nan")
    if np.all(y_pred == y_pred[0]):
        return float("nan")
    return float(roc_auc_score(labels, y_pred))


# Мінімальна кількість рядків джерела в тестовій вибірці, за якої
# для нього обчислюється внутрішньоджерельна кореляція. Поріг зафіксовано
# 7 серпня 2026 року за покриттям, до обчислення будь-яких значень ρ:
# при 30 рядках кваліфікуються 82 джерела, що охоплюють 42,9% тесту.
SOURCE_MIN_TEST_COUNT = 30


def spearman_within_source(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    min_count: int = SOURCE_MIN_TEST_COUNT,
) -> dict[str, float]:
    """Spearman ρ усередині кожного великого джерела, усереднена по джерелах.

    Навіщо потрібна. Загальна рангова кореляція значною мірою відображає
    різницю між виданнями: медіана популярності у Facebook коливається
    від 5 у WinBeta до 173,5 у New York Times (EXP-001, notes.md).
    Модель, що вивчила лише рівень видання, отримує високу загальну ρ,
    нічого не знаючи про заголовок. Внутрішньоджерельна кореляція фіксує
    видання і показує, чи вміє модель ранжувати матеріали всередині нього.
    Це ближче до практичного питання роботи: копірайтер пише для свого
    видання, а не обирає видання.

    Джерело з кількістю рядків нижче порога не бере участі. Джерело,
    у якому цільова змінна не має варіації, дає невизначену ρ і
    виключається; кількість таких джерел повертається окремим числом,
    щоб мовчазного відсіву не було.

    Основне число — незважене середнє по джерелах: кожне видання
    важить однаково. Зважене за кількістю рядків подається додатково.
    """
    order = np.argsort(groups, kind="stable")
    g_sorted = groups[order]
    yt_sorted = y_true[order]
    yp_sorted = y_pred[order]

    boundaries = np.flatnonzero(np.r_[True, g_sorted[1:] != g_sorted[:-1], True])

    rhos: list[float] = []
    weights: list[int] = []
    n_degenerate = 0
    n_too_small = 0

    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        size = stop - start
        if size < min_count:
            n_too_small += 1
            continue
        value = spearman(yt_sorted[start:stop], yp_sorted[start:stop])
        if not np.isfinite(value):
            n_degenerate += 1
            continue
        rhos.append(value)
        weights.append(size)

    if not rhos:
        return {
            "mean": float("nan"),
            "weighted_mean": float("nan"),
            "n_sources": 0,
            "n_sources_degenerate": n_degenerate,
            "n_sources_below_threshold": n_too_small,
            "n_rows_covered": 0,
        }

    rho_array = np.asarray(rhos)
    weight_array = np.asarray(weights, dtype=float)
    return {
        "mean": float(rho_array.mean()),
        "weighted_mean": float(np.average(rho_array, weights=weight_array)),
        "n_sources": len(rhos),
        "n_sources_degenerate": n_degenerate,
        "n_sources_below_threshold": n_too_small,
        "n_rows_covered": int(weight_array.sum()),
    }


def compute_all(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "spearman": spearman(y_true, y_pred),
        "mae_log": mae_log(y_true, y_pred),
        "auc_nonzero": auc_nonzero(y_true, y_pred),
    }
