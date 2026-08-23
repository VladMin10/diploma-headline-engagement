"""Фаза 4: побудова матриць ознак і перевірка на витік.

Запуск:
    .venv/bin/python -m src.build_features

Пише три файли в `data/processed/` і звіт перевірки на витік.

Що саме потрапляє у файл. Матриця містить тільки те, що обчислюється
порядково і не залежить від інших рядків: категоріальні поля метаданих
у сирому вигляді, мітки анотації, статистики тексту. Усе, що вчиться
на даних — перелік категорій для one-hot, словник TF-IDF — у файл
не потрапляє, бо фітиться всередині фолда (CLAUDE.md розділ 6, пункт 2).
Інакше файл ознак сам став би каналом витоку: категорії, побудовані
на всіх 81 634 рядках, несли б у навчання інформацію про тестовий період.

Тому матриця T1 містить не готові стовпці one-hot, а поля, з яких
`MetaFeatures` їх будує у фолді. Матриця T3 містить статистики, але
не векторне представлення.
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from src.data.split import apply_split
from src.features.assemble import annotated_ids, build_extractors
from src.features.base import FORBIDDEN_COLUMNS
from src.features.copywriting import LABELS, load_labels, resolve_included_features
from src.features.text import STATISTIC_NAMES, title_statistics

CLEAN = Path("data/interim/news_clean.parquet")
SPLIT = Path("data/interim/split.json")
OUT_DIR = Path("data/processed")
REPORT = OUT_DIR / "leakage_check.json"

# Поля метаданих у сирому вигляді. One-hot будується у фолді.
META_FIELDS = ["source", "topic", "hour", "weekday", "month"]


def build_t1(frame: pd.DataFrame) -> pd.DataFrame:
    dates = frame["PublishDate"]
    return pd.DataFrame(
        {
            "source": frame["Source"].fillna("__відсутнє__").astype(str),
            "topic": frame["Topic"].astype(str),
            "hour": dates.dt.hour.astype("int16"),
            "weekday": dates.dt.dayofweek.astype("int16"),
            "month": dates.dt.month.astype("int16"),
        },
        index=frame.index,
    )


def build_t2(ids: pd.Series, features: list[str]) -> pd.DataFrame:
    labels = load_labels(LABELS, features)
    return labels.reindex(ids.astype("int64")).dropna(how="all")


def build_t3(frame: pd.DataFrame) -> pd.DataFrame:
    stats = title_statistics(frame["Title"])
    stats["sentiment_title"] = frame["SentimentTitle"].astype(np.float64).to_numpy()
    stats["sentiment_headline"] = (
        frame["SentimentHeadline"].astype(np.float64).to_numpy()
    )
    return stats[STATISTIC_NAMES]


def audit_extractors() -> list[dict]:
    """Прохід усіма екстракторами: які колонки читає кожен.

    Четвертий рівень захисту від витоку за `docs/architecture.md`
    розділ 6. Перевірка виконується на самих об'єктах, а не на файлах,
    тому ловить намір читати заборонену колонку навіть тоді, коли
    в поточних даних її немає.
    """
    rows = []
    for level, extractor in build_extractors(list("T1 T2 T3".split())):
        required = list(extractor.required_columns)
        forbidden = sorted(set(required) & FORBIDDEN_COLUMNS)
        # Викидає ValueError, якщо екстрактор звертається до цільової
        # змінної; тут виклик прямий, щоб зафіксувати результат у звіті.
        extractor.check_no_leakage()
        rows.append(
            {
                "рівень": level,
                "клас": type(extractor).__name__,
                "читає_колонки": required,
                "перетин_із_забороненими": forbidden,
                "перевірку_пройдено": not forbidden,
            }
        )
    return rows


def audit_vocabulary_leak(train: pd.DataFrame, everything: pd.DataFrame) -> dict:
    """Скільки термінів TF-IDF з'явилося б у словнику через витік.

    Перевірка робить абстрактну вимогу «фітити у фолді» перевіряним
    числом. Словник будується двічі: на навчальній частині і на всіх
    даних. Терміни, які є лише в другому, — це та інформація про
    тестовий період, яку модель отримала б, якби векторизатор
    фітився поза фолдом.
    """
    from src.features.text import TextFeatures

    on_train = TextFeatures(include_vectors=True).fit(train)
    on_all = TextFeatures(include_vectors=True).fit(everything)
    vocab_train = set(on_train.vectoriser_.get_feature_names_out())
    vocab_all = set(on_all.vectoriser_.get_feature_names_out())
    only_all = sorted(vocab_all - vocab_train)
    return {
        "термінів_у_словнику_train": len(vocab_train),
        "термінів_у_словнику_на_всіх_даних": len(vocab_all),
        "потрапило_б_лише_через_витік": len(only_all),
        "приклади": only_all[:15],
        "висновок": (
            "розміри словників збігаються, склад — ні; фіт векторизатора "
            "поза фолдом переніс би в навчання терміни тестового періоду"
        ),
    }


def audit_matrix(name: str, frame: pd.DataFrame) -> dict:
    """Перевірка готової матриці: чи не просочилася цільова колонка."""
    columns = list(frame.columns)
    forbidden = sorted(set(columns) & FORBIDDEN_COLUMNS)
    return {
        "файл": f"{name}.parquet",
        "рядків": int(len(frame)),
        "колонок": len(columns),
        "колонки": columns,
        "перетин_із_забороненими": forbidden,
        "перевірку_пройдено": not forbidden,
        "частка_пропусків": {
            column: round(float(frame[column].isna().mean()), 6) for column in columns
        },
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clean = pd.read_parquet(CLEAN)
    clean["IDLink"] = clean["IDLink"].astype("int64")
    split = json.loads(SPLIT.read_text(encoding="utf-8"))

    inclusion = resolve_included_features()
    included = inclusion["включено"]

    matrices = {
        "features_T1": build_t1(clean).set_index(clean["IDLink"]),
        "features_T2": build_t2(clean["IDLink"], included),
        "features_T3": build_t3(clean).set_index(clean["IDLink"]),
    }
    for name, frame in matrices.items():
        frame.index.name = "IDLink"
        frame.to_parquet(OUT_DIR / f"{name}.parquet")

    annotated = annotated_ids()
    boundary = pd.Timestamp(split["boundary"])
    in_train = clean.set_index("IDLink").loc[annotated, "PublishDate"] < boundary

    report = {
        "фаза": "Фаза 4 — побудова матриць ознак і перевірка на витік",
        "джерело_даних": str(CLEAN),
        "спліт": {
            "файл": str(SPLIT),
            "межа": split["boundary"],
            "правило": split["boundary_rule"],
        },
        "склад_T2": inclusion,
        "покриття": {
            "усього_новин": int(len(clean)),
            "анотовано": int(len(annotated)),
            "анотовано_train": int(in_train.sum()),
            "анотовано_test": int((~in_train).sum()),
            "примітка": (
                "рівні T1 і T3 доступні для всіх новин, T2 — лише для "
                "анотованих; комбінація з T2 звужує множину рядків до "
                "анотованих, звуження робить select_rows з assemble.py"
            ),
        },
        "що_не_потрапляє_у_файл": (
            "перелік категорій для one-hot і словник TF-IDF: вони вчаться "
            "на даних і фітяться всередині фолда, інакше файл ознак сам "
            "став би каналом витоку"
        ),
        "перевірка_екстракторів": audit_extractors(),
        "перевірка_словника_TFIDF": audit_vocabulary_leak(
            apply_split(clean)[0], clean
        ),
        "перевірка_матриць": [
            audit_matrix(name, frame) for name, frame in matrices.items()
        ],
        "версії": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
        },
    }
    report["перевірку_пройдено"] = all(
        row["перевірку_пройдено"]
        for row in report["перевірка_екстракторів"] + report["перевірка_матриць"]
    )
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("Матриці ознак:")
    for name, frame in matrices.items():
        print(f"  {name:14s} {len(frame):>6d} рядків × {len(frame.columns)} колонок")
    print(f"\nСклад T2: {', '.join(included)}")
    if inclusion["виключено"]:
        print(f"Виключено з T2: {', '.join(inclusion['виключено'])}")
    print(f"Із застереженням: {', '.join(inclusion['із_застереженням'])}")
    print(
        f"\nПокриття T2: {len(annotated)} новин "
        f"({int(in_train.sum())} train, {int((~in_train).sum())} test)"
    )
    vocab = report["перевірка_словника_TFIDF"]
    print(
        f"Словник TF-IDF: {vocab['потрапило_б_лише_через_витік']} з "
        f"{vocab['термінів_у_словнику_на_всіх_даних']} термінів потрапили б "
        "у навчання лише через витік"
    )
    print("Перевірка на витік:", "пройдено" if report["перевірку_пройдено"] else "ПРОВАЛЕНО")
    print("Звіт:", REPORT)


if __name__ == "__main__":
    main()
