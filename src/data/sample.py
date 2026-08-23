"""Побудова вибірок для анотації.

Пілотна вибірка (Фаза 2) — випадкова з train, без стратифікації.
Умова користувача від 7 серпня 2026 року: вибірка випадкова, seed
зафіксований, «цікаві» заголовки не відбираються. Відсутність
стратифікації навмисна: пілот міряє частоту прийомів у корпусі
такою, якою вона є, а стратифікація за темою її б спотворила.

Основна вибірка (Фаза 3) буде стратифікованою — там завдання інше.
"""

from __future__ import annotations

import argparse

from pathlib import Path

import pandas as pd

from src.data.load import CLEAN_PATH, PROJECT_ROOT, is_truncated
from src.data.split import compute_split, load_split

ANNOTATIONS = PROJECT_ROOT / "annotations"
PILOT_PATH = ANNOTATIONS / "pilot_sample_400.csv"

# Службові рубрики без копірайтингу: щоденні дайджести, датовані заголовки,
# заглушки стрічки. Розмічати їх за кодбуком безглуздо — усі дев'ять ознак
# дорівнювали б нулю, а нульовий рівень роздувся б штучно.
SERVICE_TITLE_PATTERNS = [
    r"^\s*business highlights\s*$",
    r"^\s*(mon|tues|wednes|thurs|fri|satur|sun)day,\s*\d",
    r"^\s*also in (the )?news\s*$",
    r"^\s*news (digest|roundup|summary|briefs?)\s*$",
    r"^\s*(morning|evening|daily|weekly) (briefing|roundup|digest|report)\s*$",
    r"^\s*\W*\s*$",
]


def is_service_title(titles):
    """Чи є заголовок службовою рубрикою, а не журналістським текстом."""
    import re as _re

    rx = _re.compile("|".join(SERVICE_TITLE_PATTERNS), _re.IGNORECASE)
    return titles.fillna("").astype(str).map(lambda t: bool(rx.search(t)))

PILOT_SIZE = 400
PILOT_SEED = 42


def build_pilot_sample(
    size: int = PILOT_SIZE,
    seed: int = PILOT_SEED,
    path: Path = PILOT_PATH,
) -> pd.DataFrame:
    df = pd.read_parquet(CLEAN_PATH)
    # Обірвані заголовки виключено з Фази 5 рішенням користувача
    # від 7 серпня 2026 року, обґрунтування — docs/truncation_confound.md.
    df = df.loc[~is_truncated(df["Title"])].reset_index(drop=True)
    split = compute_split(df, load_split()["boundary"])
    train = df.loc[split.train_mask]

    sample = train.sample(n=size, random_state=seed).sort_values("IDLink")
    out = sample[["IDLink", "Title", "Source", "Topic", "PublishDate"]].copy()
    out["PublishDate"] = out["PublishDate"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out["service_title"] = is_service_title(out["Title"]).astype(int)

    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


MANUAL_PATH = ANNOTATIONS / "manual_sample_200.csv"
MANUAL_SIZE = 200
MANUAL_SEED = 43

# Порядок полів збігається з promptv1.txt і з pilot_labels.csv,
# за винятком виключеної ознаки 8.
FEATURE_COLUMNS = [
    "gap",
    "scale",
    "concrete",
    "conflict",
    "authority",
    "audience",
    "violation",
    "stakes",
]


def build_manual_sample(
    size: int = MANUAL_SIZE,
    seed: int = MANUAL_SEED,
    path: Path = MANUAL_PATH,
    pilot_path: Path = PILOT_PATH,
) -> pd.DataFrame:
    """Вибірка для ручної розмітки користувачем (Q0, каппа Коена).

    Заголовки пілоту виключаються: на них уже існує автоматична розмітка,
    і незалежність двох оцінювачів була б порушена.

    Вибірка випадкова, без стратифікації. Стратифікація за очікуваним
    рівнем ознаки спотворила б каппу, оскільки та чутлива до поширеності
    категорій. Ціна цього рішення — рідкісні рівні будуть представлені
    кількома випадками, і каппа для них матиме широкий інтервал.

    Порядок рядків перемішано, щоб втома оцінювача не збігалася
    з порядком тем чи джерел.
    """
    df = pd.read_parquet(CLEAN_PATH)
    df = df.loc[~is_truncated(df["Title"])].reset_index(drop=True)
    split = compute_split(df, load_split()["boundary"])
    train = df.loc[split.train_mask]

    pilot_ids = set(pd.read_csv(pilot_path)["IDLink"].astype(int))
    available = train.loc[~train["IDLink"].astype(int).isin(pilot_ids)]

    sample = available.sample(n=size, random_state=seed).reset_index(drop=True)
    out = sample[["IDLink", "Title", "Source", "Topic", "PublishDate"]].copy()
    out["IDLink"] = out["IDLink"].astype(int)
    out["PublishDate"] = out["PublishDate"].dt.strftime("%Y-%m-%d %H:%M:%S")
    for column in FEATURE_COLUMNS:
        out[column] = ""

    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


QUEUE_PATH = ANNOTATIONS / "queue.csv"
STABILITY_PATH = ANNOTATIONS / "stability_sample_300_blank.csv"
STABILITY_SIZE_PER_STRATUM = 100
STABILITY_SEED = 45

# Сім ознак, що лишилися в T2 після виключення ознаки 8 (частота, Фаза 2)
# і ознаки 6 (узгодженість, EXP-002). Порядок збігається з prompt_v1.txt
# і з llm_labels_run1.csv.
RUN_FEATURE_COLUMNS = [
    "gap",
    "scale",
    "concrete",
    "conflict",
    "authority",
    "violation",
    "stakes",
]

# Межі режимів розміру пачки в першому прогоні Фази 3, за позицією рядка
# в queue.csv (нумерація з одиниці, обидві межі включно). Відновлено
# з історії комітів і progress.json; зафіксовано тут, щоб страти
# не довелося перевизначати заднім числом.
BATCH_REGIMES = {
    25: (1, 550),
    100: (551, 1750),
    200: (1751, 5055),
}


def build_stability_sample(
    per_stratum: int = STABILITY_SIZE_PER_STRATUM,
    seed: int = STABILITY_SEED,
    path: Path = STABILITY_PATH,
    queue_path: Path = QUEUE_PATH,
) -> pd.DataFrame:
    """Вибірка для повторного прогону анотатора (EXP-002, частина друга).

    Стратифікована за режимом розміру пачки першого прогону, по `per_stratum`
    рядків на режим. Рішення користувача від 7 серпня 2026 року. Мета
    стратифікації — перевірити, чи погіршило якість розмітки збільшення
    пачки з 25 до 200 рядків.

    Наслідок зафіксовано до запуску: вибірка не є випадковою вибіркою
    з черги, тому загальна каппа стабільності подається як середнє,
    зважене за часткою режимів у черзі, а не як просте середнє по стратах.

    Файл записується з порожніми полями оцінок. Мітки першого прогону
    в нього не потрапляють: порівняння виконує скрипт після запису run2.
    """
    queue = pd.read_csv(queue_path)
    queue["position"] = range(1, len(queue) + 1)

    parts = []
    for batch_size, (lo, hi) in sorted(BATCH_REGIMES.items()):
        stratum = queue.loc[queue["position"].between(lo, hi)]
        if len(stratum) < per_stratum:
            raise ValueError(
                f"страта {batch_size}: {len(stratum)} рядків, потрібно {per_stratum}"
            )
        picked = stratum.sample(n=per_stratum, random_state=seed + batch_size)
        picked = picked.assign(batch_regime=batch_size)
        parts.append(picked)

    sample = pd.concat(parts)
    # Перемішування: інакше порядок рядків видавав би межі страт,
    # а разом з ними і позицію рядка в черзі першого прогону.
    sample = sample.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    columns = ["IDLink", "Title"]
    for optional in ("Source", "Topic", "PublishDate"):
        if optional in sample.columns:
            columns.append(optional)
    out = sample[columns].copy()
    out["IDLink"] = out["IDLink"].astype(int)
    for column in RUN_FEATURE_COLUMNS:
        out[column] = ""

    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)

    # Ключ страт зберігається окремо. У файлі розмітки його немає навмисно:
    # знання режиму під час розмітки могло б на неї вплинути.
    key = sample[["IDLink", "position", "batch_regime"]].copy()
    key["IDLink"] = key["IDLink"].astype(int)
    key.to_csv(path.with_name("stability_sample_300_strata.csv"), index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Побудова вибірок для анотації. Кожна вибірка вже "
                    "збудована і закомічена; команда відтворює її з того "
                    "самого зерна."
    )
    parser.add_argument("sample", choices=["pilot", "manual", "stability"])
    args = parser.parse_args()

    builder = {
        "pilot": build_pilot_sample,
        "manual": build_manual_sample,
        "stability": build_stability_sample,
    }[args.sample]
    result = builder()
    print(f"{args.sample}: {len(result)} заголовків")


if __name__ == "__main__":
    main()
