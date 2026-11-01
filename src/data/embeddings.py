"""Векторизація заголовків моделлю sentence-transformers.

Запуск:
    .venv/bin/python -m src.data.embeddings [--force]

Розширення E5, зафіксоване в картці EXP-008 до написання цього коду.

**Модель попередньо навчена і на даних роботи не донавчається.** Її ваги
зафіксовані до початку роботи, тому векторизація виконується один раз поза
фолдом і витоку це не створює: модель нічого не дізнається з цих даних
і не може перенести відомості з перевірочної частини в навчальну.

Цим вона відрізняється від TF-IDF. Там словник будується з тексту навчальної
частини, тобто вчиться на даних, і тому вектаризатор фітиться всередині
кожного фолда (ADR-002, третій рівень схеми). Тут фітити нічого.

Модель `all-MiniLM-L6-v2`, 384 виміри. Підстава вибору записана в картці
EXP-008: попередньо навчена, легка, працює на процесорі, стандартний базовий
вибір для коротких англомовних текстів.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.copywriting import load_labels

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEAN_PATH = PROJECT_ROOT / "data" / "interim" / "news_clean.parquet"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "embeddings.parquet"
META_PATH = PROJECT_ROOT / "data" / "processed" / "embeddings_meta.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DIMENSIONS = 384
BATCH_SIZE = 64


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if OUT_PATH.exists() and not args.force:
        raise SystemExit(
            f"{OUT_PATH.relative_to(PROJECT_ROOT)} уже існує; "
            "запустіть з --force, щоб перерахувати."
        )

    frame = pd.read_parquet(CLEAN_PATH, columns=["IDLink", "Title"])
    frame["IDLink"] = frame["IDLink"].astype("int64")
    frame = frame[frame["IDLink"].isin(load_labels().index)]
    frame = frame.sort_values("IDLink").reset_index(drop=True)
    print(f"заголовків: {len(frame)}")

    from sentence_transformers import SentenceTransformer

    started = time.perf_counter()
    model = SentenceTransformer(MODEL_NAME)
    loaded = time.perf_counter() - started
    print(f"модель завантажено за {loaded:.1f} с")

    encoded_started = time.perf_counter()
    vectors = model.encode(
        frame["Title"].tolist(),
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    elapsed = time.perf_counter() - encoded_started
    print(f"векторизація: {elapsed:.1f} с, форма {vectors.shape}")

    if vectors.shape[1] != DIMENSIONS:
        raise SystemExit(
            f"очікувалося {DIMENSIONS} вимірів, отримано {vectors.shape[1]}"
        )

    table = pd.DataFrame(
        vectors.astype(np.float32),
        columns=[f"emb_{index:03d}" for index in range(DIMENSIONS)],
    )
    table.insert(0, "IDLink", frame["IDLink"].to_numpy())
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(OUT_PATH, index=False)

    META_PATH.write_text(
        json.dumps(
            {
                "модель": MODEL_NAME,
                "вимірів": DIMENSIONS,
                "рядків": int(len(table)),
                "донавчання": "не виконувалося; ваги моделі зафіксовані",
                "витік": (
                    "векторизація поза фолдом витоку не створює, бо модель "
                    "нічого не дізнається з цих даних"
                ),
                "створено": datetime.now().isoformat(timespec="seconds"),
                "час_завантаження_моделі_с": round(loaded, 1),
                "час_векторизації_с": round(elapsed, 1),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"збережено: {OUT_PATH.relative_to(PROJECT_ROOT)} — "
          f"{len(table)} рядків, {DIMENSIONS} вимірів")


if __name__ == "__main__":
    main()
