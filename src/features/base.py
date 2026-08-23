"""Спільний інтерфейс екстрактора ознак (docs/architecture.md розділ 4).

Кожен рівень ознак — T1, T2, T3 — реалізує цей інтерфейс. Однаковість
потрібна для того, щоб складання матриці з довільної комбінації рівнів
не залежало від того, що всередині кожного.

Механізм захисту від витоку вбудований в інтерфейс двома способами.
Перший: метод transform не отримує цільової змінної, тому використати її
неможливо навіть помилково. Другий: усе, що екстрактор дізнається з даних,
він дізнається у fit, а fit викликається лише на навчальній частині.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Колонки, недоступні до публікації заголовка. Жоден екстрактор не має
# права їх читати; перевірка виконується у Фазі 4.
FORBIDDEN_COLUMNS = frozenset(
    {
        "Facebook",
        "GooglePlus",
        "LinkedIn",
        "Facebook_value",
        "GooglePlus_value",
        "LinkedIn_value",
        "Facebook_log1p",
        "GooglePlus_log1p",
        "LinkedIn_log1p",
    }
)


class FeatureExtractor(BaseEstimator, TransformerMixin, ABC):
    """Базовий клас рівня ознак.

    Нащадок зобов'язаний оголосити required_columns — перелік колонок
    вихідних даних, які він читає. Перелік перевіряється проти
    FORBIDDEN_COLUMNS при створенні об'єкта, тож екстрактор, що звертається
    до цільової змінної, не може бути навіть інстанційований.
    """

    level: str = ""

    @property
    @abstractmethod
    def required_columns(self) -> list[str]:
        """Колонки вихідних даних, потрібні цьому екстрактору."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    def check_no_leakage(self) -> None:
        """Перевіряє, що екстрактор не читає післяпублікаційних колонок."""
        used = set(self.required_columns)
        forbidden = used & FORBIDDEN_COLUMNS
        if forbidden:
            raise ValueError(
                f"{type(self).__name__} звертається до колонок, недоступних "
                f"до публікації: {sorted(forbidden)}"
            )

    @abstractmethod
    def fit(self, X: pd.DataFrame, y=None) -> "FeatureExtractor":
        """Вивчає з навчальних даних усе, що потрібно для transform."""

    @abstractmethod
    def transform(self, X: pd.DataFrame):
        """Перетворює дані. Нічого нового не вивчає."""

    @abstractmethod
    def get_feature_names_out(self, input_features=None):
        """Назви створених ознак. Потрібні для підписів на рисунках SHAP."""
