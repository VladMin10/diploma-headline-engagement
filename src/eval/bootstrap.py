"""Довірчі інтервали бутстрепом (CLAUDE.md розділ 6, пункт 7).

Різниця між моделями без довірчого інтервалу нічого не означає, тому
інтервали будуються для кожної метрики і для кожного порівняння моделей.

Ключова деталь — порівняння двох моделей виконується на однакових
бутстреп-вибірках. Якщо будувати інтервали для кожної моделі окремо
і дивитися, чи перекриваються вони, висновок виходить надто
консервативним: похибка, спільна для обох моделей, врахується двічі.
Інтервал для різниці позбавлений цієї вади.
"""

from __future__ import annotations

import numpy as np

from src.eval.metrics import compute_all

N_ITERATIONS = 1000
RANDOM_STATE = 42


def _percentile_ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan"), float("nan")
    lo = float(np.percentile(finite, 100 * alpha / 2))
    hi = float(np.percentile(finite, 100 * (1 - alpha / 2)))
    return lo, hi


def bootstrap_indices(
    n: int, n_iterations: int = N_ITERATIONS, random_state: int = RANDOM_STATE
) -> np.ndarray:
    """Матриця індексів (n_iterations, n) для вибірки з поверненням.

    Індекси генеруються один раз і використовуються для всіх моделей,
    тому порівняння моделей відбувається на тих самих псевдовибірках.
    """
    rng = np.random.default_rng(random_state)
    return rng.integers(0, n, size=(n_iterations, n))


def bootstrap_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    indices: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Точкова оцінка і 95% перцентильний інтервал для кожної метрики."""
    point = compute_all(y_true, y_pred)
    draws = {name: np.empty(len(indices)) for name in point}

    for i, idx in enumerate(indices):
        sample = compute_all(y_true[idx], y_pred[idx])
        for name, value in sample.items():
            draws[name][i] = value

    out = {}
    for name, value in point.items():
        lo, hi = _percentile_ci(draws[name])
        out[name] = {
            "value": value,
            "ci_low": lo,
            "ci_high": hi,
            "n_valid_draws": int(np.isfinite(draws[name]).sum()),
        }
    return out


def stratified_bootstrap_indices(
    groups: np.ndarray,
    n_iterations: int = N_ITERATIONS,
    random_state: int = RANDOM_STATE,
) -> np.ndarray:
    """Індекси для бутстрепу з розшаруванням за джерелом.

    Рядки перевибираються всередині кожного джерела, а розмір джерела
    зберігається. Це потрібно для внутрішньоджерельної метрики: за звичайного
    бутстрепу склад великих джерел коливався б від ітерації до ітерації,
    і набір джерел, що проходять поріг, змінювався б разом з ним.
    Тоді інтервал відображав би не точність оцінки, а хиткість самого
    переліку джерел.
    """
    rng = np.random.default_rng(random_state)
    order = np.argsort(groups, kind="stable")
    g_sorted = groups[order]
    boundaries = np.flatnonzero(np.r_[True, g_sorted[1:] != g_sorted[:-1], True])

    out = np.empty((n_iterations, len(groups)), dtype=np.int64)
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        block = order[start:stop]
        picks = rng.integers(0, len(block), size=(n_iterations, len(block)))
        out[:, start:stop] = block[picks]
    return out


def bootstrap_within_source(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    indices: np.ndarray,
    min_count: int = None,
) -> dict[str, float]:
    """Точкова оцінка і 95% інтервал для внутрішньоджерельної кореляції."""
    from src.eval.metrics import SOURCE_MIN_TEST_COUNT, spearman_within_source

    if min_count is None:
        min_count = SOURCE_MIN_TEST_COUNT

    point = spearman_within_source(y_true, y_pred, groups, min_count)
    draws = np.empty(len(indices))
    draws_weighted = np.empty(len(indices))
    for i, idx in enumerate(indices):
        sample = spearman_within_source(y_true[idx], y_pred[idx], groups[idx], min_count)
        draws[i] = sample["mean"]
        draws_weighted[i] = sample["weighted_mean"]

    lo, hi = _percentile_ci(draws)
    lo_w, hi_w = _percentile_ci(draws_weighted)
    return {
        "value": point["mean"],
        "ci_low": lo,
        "ci_high": hi,
        "weighted_value": point["weighted_mean"],
        "weighted_ci_low": lo_w,
        "weighted_ci_high": hi_w,
        "n_sources": point["n_sources"],
        "n_sources_degenerate": point["n_sources_degenerate"],
        "n_sources_below_threshold": point["n_sources_below_threshold"],
        "n_rows_covered": point["n_rows_covered"],
        "min_count": min_count,
    }


def bootstrap_within_source_difference(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    groups: np.ndarray,
    indices: np.ndarray,
    min_count: int = None,
) -> dict[str, float]:
    """Інтервал для різниці внутрішньоджерельних кореляцій: A мінус B."""
    from src.eval.metrics import SOURCE_MIN_TEST_COUNT, spearman_within_source

    if min_count is None:
        min_count = SOURCE_MIN_TEST_COUNT

    point_a = spearman_within_source(y_true, y_pred_a, groups, min_count)
    point_b = spearman_within_source(y_true, y_pred_b, groups, min_count)

    draws = np.empty(len(indices))
    for i, idx in enumerate(indices):
        yt, g = y_true[idx], groups[idx]
        a = spearman_within_source(yt, y_pred_a[idx], g, min_count)
        b = spearman_within_source(yt, y_pred_b[idx], g, min_count)
        draws[i] = a["mean"] - b["mean"]

    lo, hi = _percentile_ci(draws)
    diff = point_a["mean"] - point_b["mean"]
    return {
        "difference": diff,
        "ci_low": lo,
        "ci_high": hi,
        "excludes_zero": bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0)),
        "min_count": min_count,
    }


def bootstrap_difference(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    indices: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Інтервал для різниці метрик двох моделей: A мінус B.

    Обидві моделі оцінюються на однакових бутстреп-вибірках.
    """
    point_a = compute_all(y_true, y_pred_a)
    point_b = compute_all(y_true, y_pred_b)
    draws = {name: np.empty(len(indices)) for name in point_a}

    for i, idx in enumerate(indices):
        yt = y_true[idx]
        sample_a = compute_all(yt, y_pred_a[idx])
        sample_b = compute_all(yt, y_pred_b[idx])
        for name in draws:
            draws[name][i] = sample_a[name] - sample_b[name]

    out = {}
    for name in point_a:
        lo, hi = _percentile_ci(draws[name])
        diff = point_a[name] - point_b[name]
        out[name] = {
            "difference": diff,
            "ci_low": lo,
            "ci_high": hi,
            "excludes_zero": bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0)),
            "n_valid_draws": int(np.isfinite(draws[name]).sum()),
        }
    return out
