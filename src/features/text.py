"""Рівень ознак T3 — формальні статистики тексту.

Контрольна група для питання Q2: сюди входить те, що рахується
автоматично, без анотації і без розуміння змісту. Якщо приріст від T2
не перевищує приросту від T3, копірайтингова таксономія не окуповує
праці анотації.

Склад рівня за `docs/architecture.md` розділ 4: довжина заголовка
в словах і символах, наявність цифри, наявність знака питання та знака
оклику, частка слів з великої літери, дві наявні в даних оцінки
тональності та векторне представлення тексту.

Поділ на дві частини принциповий для захисту від витоку. Статистики
рахуються порядково і не залежать від інших рядків, тому їх можна
обчислити один раз і покласти у `data/processed/features_T3.parquet`.
Векторне представлення навчається на даних: словник TF-IDF залежить
від того, які тексти бачив `fit`. Тому воно живе всередині екстрактора
і фітиться у фолді, а не зберігається файлом.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from src.features.base import FeatureExtractor

STATISTIC_NAMES = [
    "len_chars",
    "len_words",
    "has_digit",
    "has_question",
    "has_exclamation",
    "upper_word_share",
    "sentiment_title",
    "sentiment_headline",
]

WORD = re.compile(r"\S+")
DIGIT = re.compile(r"\d")
# Слово «з великої літери» — таке, що починається з великої і має
# принаймні одну літеру. Абревіатури й слова в капслоку теж рахуються:
# рубрика T3 міряє форму, а не значення.
CAPITALISED = re.compile(r"^[^\W\d_]*[A-ZА-ЯЇІЄҐ]")


def title_statistics(titles: pd.Series) -> pd.DataFrame:
    """Порядкові статистики заголовка. Від інших рядків не залежать."""
    text = titles.fillna("").astype(str)
    words = text.map(lambda s: WORD.findall(s))
    word_counts = words.map(len)

    upper_share = [
        (sum(1 for w in ws if CAPITALISED.match(w)) / len(ws)) if ws else 0.0
        for ws in words
    ]

    return pd.DataFrame(
        {
            "len_chars": text.str.len().astype(np.float64),
            "len_words": word_counts.astype(np.float64),
            "has_digit": text.str.contains(DIGIT).astype(np.float64),
            "has_question": text.str.contains("?", regex=False).astype(np.float64),
            "has_exclamation": text.str.contains("!", regex=False).astype(np.float64),
            "upper_word_share": np.asarray(upper_share, dtype=np.float64),
        },
        index=titles.index,
    )


class TextFeatures(FeatureExtractor):
    """Рівень T3.

    Параметр `include_vectors` вмикає TF-IDF поверх статистик. Вимкнений
    за замовчуванням: питання Q2 порівнює таксономію з формальними
    статистиками, а векторне представлення додає до T3 інший клас
    інформації і робить порівняння неоднозначним. Вмикається окремим
    експериментом, а не мовчки.
    """

    level = "T3"

    def __init__(
        self,
        include_vectors: bool = False,
        max_features: int = 2000,
        min_df: int = 5,
        ngram_range: tuple[int, int] = (1, 2),
    ):
        self.include_vectors = include_vectors
        self.max_features = max_features
        self.min_df = min_df
        self.ngram_range = ngram_range

    @property
    def required_columns(self) -> list[str]:
        return ["Title", "SentimentTitle", "SentimentHeadline"]

    def _statistics(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = title_statistics(X["Title"])
        frame["sentiment_title"] = X["SentimentTitle"].astype(np.float64).to_numpy()
        frame["sentiment_headline"] = (
            X["SentimentHeadline"].astype(np.float64).to_numpy()
        )
        return frame[STATISTIC_NAMES]

    def fit(self, X: pd.DataFrame, y=None) -> "TextFeatures":
        self.check_no_leakage()
        self.statistic_names_ = list(STATISTIC_NAMES)
        if self.include_vectors:
            # Словник будується лише з текстів, поданих у fit, тобто
            # з навчальної частини фолда. Тестові тексти на словник
            # не впливають (CLAUDE.md розділ 6, пункт 2).
            self.vectoriser_ = TfidfVectorizer(
                max_features=self.max_features,
                min_df=self.min_df,
                ngram_range=self.ngram_range,
                lowercase=True,
            )
            self.vectoriser_.fit(X["Title"].fillna("").astype(str))
        return self

    def transform(self, X: pd.DataFrame):
        statistics = self._statistics(X).to_numpy(dtype=np.float64)
        if not self.include_vectors:
            return statistics
        vectors = self.vectoriser_.transform(X["Title"].fillna("").astype(str))
        return sparse.hstack([sparse.csr_matrix(statistics), vectors], format="csr")

    def get_feature_names_out(self, input_features=None):
        names = [f"T3__{name}" for name in self.statistic_names_]
        if self.include_vectors:
            names += [
                f"T3__tfidf__{token}"
                for token in self.vectoriser_.get_feature_names_out()
            ]
        return np.array(names, dtype=object)
