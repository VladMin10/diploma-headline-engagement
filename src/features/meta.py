"""Рівень ознак T1 — метадані публікації.

Джерело, тема, година, день тижня, місяць. Це те, що відоме без жодного
аналізу тексту заголовка.

Кодування джерела заслуговує окремої уваги. У наборі 5 756 різних джерел,
з них 2 042 трапляються рівно один раз (docs/data_card.md розділ 4).
Часовий спліт гарантує появу в тесті джерел, не бачених у навчанні.
Тому джерела з частотою нижче порога і джерела, відсутні в train,
зводяться в одну категорію: змістовно «джерело рідкісне» і «джерело
невідоме» для моделі однакові.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from src.features.base import FeatureExtractor

# Поріг частоти джерела в навчальній вибірці. Обраний до прогону
# і не підбирається за результатом на тесті (EXP-001, розділ 3).
SOURCE_MIN_FREQUENCY = 30

MISSING_SOURCE = "__відсутнє__"


class MetaFeatures(FeatureExtractor):
    """Рівень T1.

    Ознака «місяць» за замовчуванням вимкнена. Рішення ухвалене
    користувачем 7 серпня 2026 року за результатом EXP-001: приріст
    Spearman ρ від її включення становив 0,0003 для Facebook, 0,0012
    для Google+ і 0,0002 для LinkedIn. Підстави для виключення, окрім
    нульового внеску: під часовим сплітом червня і липня в навчальній
    вибірці немає, а частка відсіяних рядків у цих місяцях різко вища,
    тож ознака здатна підхоплювати слід обриву збору даних замість
    сезонності.
    """

    level = "T1"

    def __init__(
        self,
        include_month: bool = False,
        source_min_frequency: int = SOURCE_MIN_FREQUENCY,
    ):
        self.include_month = include_month
        self.source_min_frequency = source_min_frequency

    @property
    def required_columns(self) -> list[str]:
        return ["Source", "Topic", "PublishDate"]

    def _build_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        """Сирі колонки -> категоріальні поля, готові до кодування."""
        dates = X["PublishDate"]
        frame = pd.DataFrame(
            {
                "source": X["Source"].fillna(MISSING_SOURCE).astype(str),
                "topic": X["Topic"].astype(str),
                "hour": dates.dt.hour.astype(str),
                "weekday": dates.dt.dayofweek.astype(str),
            },
            index=X.index,
        )
        if self.include_month:
            frame["month"] = dates.dt.month.astype(str)
        return frame

    def fit(self, X: pd.DataFrame, y=None) -> "MetaFeatures":
        self.check_no_leakage()
        frame = self._build_frame(X)

        # min_frequency застосовується до всіх полів, але змістовно
        # потрібен лише для джерела: тема, година, день тижня і місяць
        # мають небагато значень і всі вони частіші за поріг.
        self.encoder_ = OneHotEncoder(
            min_frequency=self.source_min_frequency,
            handle_unknown="infrequent_if_exist",
            sparse_output=True,
            dtype=np.float64,
        )
        self.encoder_.fit(frame)
        self.columns_ = list(frame.columns)
        return self

    def transform(self, X: pd.DataFrame):
        frame = self._build_frame(X)
        return self.encoder_.transform(frame[self.columns_])

    def get_feature_names_out(self, input_features=None):
        return self.encoder_.get_feature_names_out(self.columns_)
