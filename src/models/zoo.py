"""Моделі, зібрані як пайплайни (docs/architecture.md розділ 6).

Модель ніколи не навчається на голій матриці ознак. Побудова ознак
і сама модель становлять одне ціле, тому при навчанні на фолді всі кроки
підготовки навчаються теж на цьому фолді.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline

from src.features.meta import MetaFeatures

RANDOM_STATE = 42
RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]


class MedianBaseline(BaseEstimator, RegressorMixin):
    """Константний прогноз, що дорівнює медіані навчальної вибірки.

    Медіана, а не середнє: цільові змінні скошені, і медіана дає нижчу
    середню абсолютну помилку. Ця модель є нулем відліку для EXP-001.
    Spearman ρ для неї не визначений, оскільки прогноз не має варіації.
    """

    def fit(self, X, y):
        self.constant_ = float(np.median(y))
        return self

    def predict(self, X):
        n = X.shape[0]
        return np.full(n, self.constant_, dtype=float)


def build_features(levels: str | list[str], include_month: bool = True):
    """Крок побудови ознак для одного рівня або комбінації рівнів.

    Приймає як рядок «T1» (сумісність з EXP-001), так і перелік
    ["T1", "T2", "T3"]. Складання комбінації робить `assemble.py`,
    який нічого не знає про моделі.
    """
    from src.features.assemble import build_feature_union

    if isinstance(levels, str):
        levels = [levels]
    return build_feature_union(levels, {"T1": {"include_month": include_month}})


def build_model(name: str, alpha: float | None = None, random_state: int = RANDOM_STATE):
    if name == "median":
        return MedianBaseline()
    if name == "ridge":
        from sklearn.linear_model import Ridge

        return Ridge(alpha=alpha if alpha is not None else 1.0, random_state=random_state)
    if name == "lightgbm":
        from lightgbm import LGBMRegressor

        # Параметри зафіксовані до Фази 5 і не підбираються за результатом
        # на тесті. Обмеження складності узгоджене з обсягом навчальної
        # частини анотованої підвибірки: 2 000 рядків.
        return LGBMRegressor(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=30,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
    raise ValueError(f"Невідома модель: {name}")


def build_pipeline(
    model_name: str,
    level: str | list[str] = "T1",
    include_month: bool = True,
    alpha: float | None = None,
) -> Pipeline:
    return Pipeline(
        [
            ("features", build_features(level, include_month=include_month)),
            ("model", build_model(model_name, alpha=alpha)),
        ]
    )
