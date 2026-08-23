"""Зчитування та очищення сирого набору UCI 432.

Модуль відповідає за перший етап конвеєра (docs/architecture.md розділ 2):
сирий CSV -> типізована таблиця -> data/interim/news_clean.parquet.

Рішення, реалізовані тут, обґрунтовані в ADR-003:
  - значення -1 у цільовій змінній означає відсутність вимірювання,
    а не нульову популярність, тому воно перетворюється на NaN;
  - для порівняння платформ (питання Q3) використовуються лише рядки
    з наявними вимірюваннями всіх трьох платформ.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "Dataset" / "News_Final.csv"
CLEAN_PATH = PROJECT_ROOT / "data" / "interim" / "news_clean.parquet"

TARGETS = ["Facebook", "GooglePlus", "LinkedIn"]
MISSING_CODE = -1

# Колонки, відомі до публікації заголовка. Будь-яка ознака будується
# лише з них (CLAUDE.md розділ 6, пункт 3).
PRE_PUBLICATION_COLUMNS = [
    "IDLink",
    "Title",
    "Headline",
    "Source",
    "Topic",
    "PublishDate",
    "SentimentTitle",
    "SentimentHeadline",
]

_WHITESPACE = re.compile(r"\s+")


def normalize_title(title: str) -> str:
    """Нормалізація заголовка для пошуку повторів (ADR-002).

    Регістр і зайві пробіли не є змістовною відмінністю: та сама новина,
    передрукована іншим джерелом, часто відрізняється лише ними.
    """
    if title is None or (isinstance(title, float) and np.isnan(title)):
        return ""
    return _WHITESPACE.sub(" ", str(title)).strip().lower()


def is_truncated(titles: pd.Series) -> pd.Series:
    """Чи обірваний заголовок трикрапкою на етапі збору даних.

    Обрив є артефактом збору, а не авторським прийомом, і створює
    інформаційну прогалину механічно. Він також невипадковий: частка
    обірваних коливається від 5,7% у Bloomberg до 47,8% у Business Insider,
    а медіана популярності у Facebook становить 17 проти 6 у необірваних.
    Обґрунтування виключення — docs/truncation_confound.md.
    """
    return titles.fillna("").astype(str).str.rstrip().str.endswith(("...", "…"))


def load_raw(path: Path | str = RAW_PATH, drop_exact_duplicates: bool = True) -> pd.DataFrame:
    """Сирий набір з розібраною датою.

    За drop_exact_duplicates=True відкидаються рядки, що збігаються з іншим
    рядком у всіх одинадцяти колонках, разом з ідентифікатором IDLink.
    Такий рядок є технічним повтором запису, а не другою публікацією:
    друга публікація мала б інший IDLink. У наборі їх чотири.
    """
    df = pd.read_csv(path)
    df["PublishDate"] = pd.to_datetime(df["PublishDate"], format="%Y-%m-%d %H:%M:%S")
    for col in TARGETS:
        df[col] = pd.to_numeric(df[col], errors="raise").astype("int64")
    for col in ("Title", "Headline", "Source", "Topic"):
        df[col] = df[col].astype("string")
    if drop_exact_duplicates:
        df = df.drop_duplicates().reset_index(drop=True)
    return df


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Похідні колонки, потрібні на кількох етапах конвеєра.

    Цільові змінні тут перетворюються на NaN у місцях відсутнього
    вимірювання і дублюються у логарифмічній шкалі. Логарифм цільової
    змінної не є ознакою і в матриці ознак не потрапляє.
    """
    out = df.copy()
    out["title_norm"] = out["Title"].map(normalize_title)

    for col in TARGETS:
        measured = out[col] != MISSING_CODE
        value = out[col].where(measured).astype("float64")
        out[f"{col}_value"] = value
        out[f"{col}_log1p"] = np.log1p(value)
    return out


def load_clean(
    path: Path | str = RAW_PATH,
    complete_cases_only: bool = True,
) -> pd.DataFrame:
    """Очищений набір, готовий до фіксації спліту та побудови ознак.

    complete_cases_only=True залишає рядки з наявними вимірюваннями всіх
    трьох платформ. Це рішення ADR-003: без спільної множини рядків
    різницю у важливості ознак між платформами можна було б пояснити
    різним складом вибірок, і питання Q3 втратило б сенс.
    """
    df = add_derived(load_raw(path))
    if complete_cases_only:
        measured = np.ones(len(df), dtype=bool)
        for col in TARGETS:
            measured &= (df[col] != MISSING_CODE).to_numpy()
        df = df.loc[measured].reset_index(drop=True)
    return df


def write_clean(df: pd.DataFrame, path: Path | str = CLEAN_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def main() -> None:
    frame = load_clean()
    path = write_clean(frame)
    print(f"Робоча вибірка: {len(frame)} рядків -> {path}")


if __name__ == "__main__":
    main()
