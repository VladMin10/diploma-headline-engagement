"""Цільова мітка класифікаційної лінії (ADR-005).

Запуск:
    .venv/bin/python -m src.data.target_classification

Друга постановка задачі, ухвалена ADR-005, працює з бінарною міткою:
заголовок належить до верхньої чверті за кількістю реакцій серед заголовків
свого видання на цій платформі. Поріг береться всередині видання, а не по
платформі загалом, бо за EXP-001 приблизно чотири п'ятих прогностичної сили
походить з різниці між виданнями, і загальний поріг відокремлював би переважно
великі видання від малих.

Модуль нічого не навчає і нічого не підбирає. Він рахує пороги, застосовує
три запасні правила ADR-005 і записує готову мітку. Це підготовка даних,
як `src/data/split.py`, а не експеримент.

**Правило 4 першого доповнення ADR-005.** Поріг і перевірки запасних правил
обчислюються один раз на всіх 5 055 анотованих рядках і далі не
перераховуються: ані по фолдах крос-валідації, ані окремо по train і test.
Поріг визначає цільову мітку, а не ознаку, тому модель його не спостерігає
і витоку з ознак у ціль виникнути не може. Перерахунок усередині фолда мав би
гірший наслідок: мітка змінювалася б від розбиття до розбиття, і парне
порівняння наборів ознак на однакових фолдах втратило б підставу. Ціна цього
рішення названа в ADR-005 і винесена в обмеження роботи.

Три пороги замість одного. Основний — верхні 25%, тобто 75-й перцентиль.
Верхні 20% і 33% рахуються поряд як перевірка стійкості і подаються в роботі
незалежно від того, який з них виявиться вигіднішим для гіпотези (ADR-005,
розділ «Рішення», другий захід проти підганяння).
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.assemble import annotated_ids

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEAN_PATH = PROJECT_ROOT / "data" / "interim" / "news_clean.parquet"
TARGETS_PATH = PROJECT_ROOT / "data" / "processed" / "targets_classification.parquet"
THRESHOLDS_PATH = PROJECT_ROOT / "data" / "processed" / "classification_thresholds.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "tables" / "target_classification.md"

# Скорочення платформ у назвах колонок мітки: y_fb_25 і так далі.
PLATFORMS = {
    "Facebook": ("fb", "Facebook_value"),
    "GooglePlus": ("gp", "GooglePlus_value"),
    "LinkedIn": ("li", "LinkedIn_value"),
}

# Ключ — частка верхнього класу у відсотках, значення — перцентиль порога.
# Основний варіант ADR-005 — верхні 25%; два інші є перевіркою стійкості.
TOP_SHARES = {"25": 75.0, "20": 80.0, "33": 67.0}
MAIN_SHARE = "25"

# Запасне правило 1: видання з меншою кількістю рядків власного порога
# не отримує. Число 30 у роботі не нове — воно вже вживається як поріг
# частоти джерела при кодуванні ознаки Source (src/features/meta.py)
# і як мінімум тестових рядків для внутрішньоджерельної метрики
# (src/eval/metrics.py).
MIN_ROWS_PER_SOURCE = 30

# Запасне правило 2: поріг у верхню чверть вироджується, якщо всередині
# видання нулів більше за цю частку. Величина не залежить від перцентиля:
# так зафіксовано ADR-005 до першого прогону.
ZERO_SHARE_LIMIT = 0.75

MODE_OWN = "власний поріг"
MODE_GLOBAL = "загальний поріг"
MODE_NONZERO = "ненульова мітка"
MODES = (MODE_OWN, MODE_GLOBAL, MODE_NONZERO)


def git_commit() -> str:
    """Коміт, на якому виконувався розрахунок.

    Потрібен, щоб мітку можна було відтворити точно тим кодом, який її дав.
    Якщо git недоступний, записується позначка, а не вигадане значення.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "недоступний"


def load_annotated(clean_path: Path = CLEAN_PATH) -> pd.DataFrame:
    """Очищений набір, звужений до анотованих новин.

    Звуження робиться за тим самим переліком, що й `select_rows`
    у `src/features/assemble.py`: джерелом істини є файл міток анотації.
    """
    frame = pd.read_parquet(clean_path)
    frame["IDLink"] = frame["IDLink"].astype("int64")
    ids = annotated_ids()
    subset = frame[frame["IDLink"].isin(ids)].copy()
    subset = subset.sort_values("IDLink").reset_index(drop=True)
    return subset


def source_mode(values: np.ndarray) -> str:
    """Режим видання за запасними правилами 1 і 2.

    Режим залежить лише від платформи і видання, не від перцентиля:
    ADR-005 фіксує обидва правила однаковими для всіх трьох порогів.
    """
    if values.size < MIN_ROWS_PER_SOURCE:
        return MODE_GLOBAL
    if float((values == 0).mean()) > ZERO_SHARE_LIMIT:
        return MODE_NONZERO
    return MODE_OWN


def build_platform(values: pd.Series, sources: pd.Series) -> dict:
    """Пороги, режими і мітки однієї платформи.

    Рядки з відсутнім вимірюванням, тобто зі значенням −1 у сирих даних
    і NaN після `src/data/load.py`, у задачі для цієї платформи участі
    не беруть (правило ADR-003, перенесене в ADR-005 без змін). Вони
    отримують NaN у мітці, а не нуль: заповнення нулем змішало б
    «реакції не було» з «реакцію не виміряли» і створило б хибні нулі
    цільової змінної.
    """
    present = values.notna().to_numpy()
    observed = values.to_numpy(dtype=np.float64)[present]
    observed_sources = sources.to_numpy()[present]

    global_thresholds = {
        share: float(np.percentile(observed, q)) for share, q in TOP_SHARES.items()
    }

    by_source: dict[str, dict] = {}
    for name in pd.unique(observed_sources):
        own = observed[observed_sources == name]
        mode = source_mode(own)
        if mode == MODE_OWN:
            thresholds = {
                share: float(np.percentile(own, q)) for share, q in TOP_SHARES.items()
            }
        elif mode == MODE_GLOBAL:
            thresholds = dict(global_thresholds)
        else:
            thresholds = {share: None for share in TOP_SHARES}
        by_source[str(name)] = {
            "режим": mode,
            "n": int(own.size),
            "частка_нулів": round(float((own == 0).mean()), 4),
            "пороги": thresholds,
        }

    labels = {}
    for share in TOP_SHARES:
        y = np.full(values.size, np.nan, dtype=np.float64)
        for position, (value, name) in enumerate(zip(values.to_numpy(), sources)):
            if not present[position]:
                continue
            record = by_source[str(name)]
            if record["режим"] == MODE_NONZERO:
                y[position] = 1.0 if value > 0 else 0.0
            else:
                # Строга нерівність. Через однакові значення поблизу порога
                # фактична частка позитивного класу буде меншою за названу;
                # поріг під неї не підганяється (запасне правило 3).
                y[position] = 1.0 if value > record["пороги"][share] else 0.0
        labels[share] = y

    return {
        "present": present,
        "global_thresholds": global_thresholds,
        "by_source": by_source,
        "labels": labels,
        "degenerate": degenerate_sources(labels, sources.to_numpy(), present),
    }


def degenerate_sources(
    labels: dict[str, np.ndarray], sources: np.ndarray, present: np.ndarray
) -> dict[str, dict]:
    """Видання, усередині яких мітка не має варіації.

    Поріг у верхню чверть може збігтися з максимумом видання, якщо значень
    поблизу нього багато однакових. Строга нерівність тоді не дає жодного
    позитивного рядка, і всередині такого видання класифікація вироджується.
    Це не помилка правила, а наслідок дискретності лічильників реакцій,
    але число таких видань потрібне при читанні результатів: модель на них
    не вчиться нічого, а внутрішньоджерельна структура задачі там зникає.
    """
    result: dict[str, dict] = {}
    for share, y in labels.items():
        names = sources[present]
        values = y[present]
        no_positive: list[str] = []
        no_negative: list[str] = []
        for name in pd.unique(names):
            own = values[names == name]
            if own.max() == 0:
                no_positive.append(str(name))
            elif own.min() == 1:
                no_negative.append(str(name))
        rows = int(sum((names == name).sum() for name in no_positive + no_negative))
        result[share] = {
            "видань_без_позитивного_класу": len(no_positive),
            "видань_без_негативного_класу": len(no_negative),
            "рядків": rows,
            "перелік": sorted(no_positive + no_negative),
        }
    return result


def platform_summary(platform: str, values: pd.Series, built: dict) -> dict:
    """Числа, що описують мітку однієї платформи. Ідуть у звіт."""
    present = built["present"]
    observed = values.to_numpy(dtype=np.float64)[present]
    n_total = int(values.size)
    n_present = int(present.sum())

    modes = [record["режим"] for record in built["by_source"].values()]
    rows_nonzero_mode = sum(
        record["n"]
        for record in built["by_source"].values()
        if record["режим"] == MODE_NONZERO
    )

    return {
        "платформа": platform,
        "n_анотованих": n_total,
        "n_з_вимірюванням": n_present,
        "n_виключено_NaN": n_total - n_present,
        "частка_нулів": round(float((observed == 0).mean()), 4),
        "загальні_пороги": built["global_thresholds"],
        "частка_позитивного_класу": {
            share: round(float(np.nanmean(y)), 4) for share, y in built["labels"].items()
        },
        "n_позитивного_класу": {
            share: int(np.nansum(y)) for share, y in built["labels"].items()
        },
        "видань_усього": len(built["by_source"]),
        "видань_за_режимами": {mode: modes.count(mode) for mode in MODES},
        "рядків_у_режимі_ненульова_мітка": int(rows_nonzero_mode),
        "вироджені_видання": built["degenerate"],
    }


def build_all(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict, list[dict]]:
    """Мітки, довідник порогів і зведення по трьох платформах."""
    targets = pd.DataFrame({"IDLink": frame["IDLink"].to_numpy()})
    thresholds: dict[str, dict] = {}
    summaries: list[dict] = []

    for platform, (short, column) in PLATFORMS.items():
        built = build_platform(frame[column], frame["Source"])
        for share, y in built["labels"].items():
            targets[f"y_{short}_{share}"] = y

        summary = platform_summary(platform, frame[column], built)
        summaries.append(summary)
        thresholds[platform] = {
            "колонка": column,
            "скорочення": short,
            "n_анотованих": summary["n_анотованих"],
            "n_з_вимірюванням": summary["n_з_вимірюванням"],
            "n_виключено_NaN": summary["n_виключено_NaN"],
            "частка_нулів": summary["частка_нулів"],
            "загальні_пороги": built["global_thresholds"],
            "вироджені_видання": built["degenerate"],
            "видання": built["by_source"],
        }

    return targets, thresholds, summaries


def thresholds_document(thresholds: dict) -> dict:
    """Довідник, якого достатньо, щоб мітку більше ніколи не перераховувати."""
    return {
        "джерело": "ADR-005, розділ «Рішення», і правило 4 першого доповнення",
        "створено": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "обчислено_на": {
            "файл": str(CLEAN_PATH.relative_to(PROJECT_ROOT)),
            "множина_рядків": "усі анотовані новини, без поділу на train і test",
        },
        "правила": {
            "основний_поріг": "верхні 25%, тобто 75-й перцентиль усередині видання",
            "перевірка_стійкості": "верхні 20% і 33%, тобто 80-й і 67-й перцентилі",
            "запасне_правило_1": (
                f"видання з менш ніж {MIN_ROWS_PER_SOURCE} рядками з наявним "
                "вимірюванням отримує загальний поріг платформи"
            ),
            "запасне_правило_2": (
                f"якщо частка нулів усередині видання перевищує {ZERO_SHARE_LIMIT}, "
                "поріг вироджується і мітка означає «реакцій більше нуля»; "
                "частка не залежить від перцентиля"
            ),
            "запасне_правило_3": (
                "поріг під рівно названу частку не підганяється; мітка ставиться "
                "за строгою нерівністю, фактична частка записується як є "
                "і слугує базовим рівнем для PR-AUC"
            ),
            "відсутні_вимірювання": (
                "рядок зі значенням −1 у сирих даних, тобто NaN після load.py, "
                "у задачі для цієї платформи участі не бере і має NaN у мітці; "
                "заповнення нулем неприпустиме (ADR-003)"
            ),
            "перцентиль": "numpy.percentile, лінійна інтерполяція",
        },
        "платформи": thresholds,
    }


def report_lines(summaries: list[dict]) -> list[str]:
    """Звіт у markdown. Ті самі числа, що друкуються в консоль."""
    lines = [
        "# Цільова мітка класифікаційної лінії",
        "",
        "Згенеровано `src/data/target_classification.py` за правилами ADR-005.",
        "Числа не редагуються вручну: правити треба джерело і перезапускати.",
        "",
        "Поріг обчислено один раз на всіх анотованих рядках і далі",
        "не перераховується (правило 4 першого доповнення ADR-005).",
        "",
        "### Обсяг і виключення",
        "",
        "| Платформа | Анотованих рядків | З вимірюванням | Виключено за NaN | Частка нулів |",
        "|---|---|---|---|---|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['платформа']} | {s['n_анотованих']} | {s['n_з_вимірюванням']} | "
            f"{s['n_виключено_NaN']} | {_num(s['частка_нулів'])} |"
        )

    lines += [
        "",
        "Частка нулів рахується серед рядків з наявним вимірюванням.",
        "",
        "### Фактична частка позитивного класу",
        "",
        "Базовий рівень для PR-AUC (ADR-005, запасне правило 3). Названа частка",
        "і фактична розходяться через однакові значення поблизу порога; поріг",
        "під названу частку не підганяється.",
        "",
        "| Платформа | Верхні 20% | Верхні 25% (основний) | Верхні 33% |",
        "|---|---|---|---|",
    ]
    for s in summaries:
        share = s["частка_позитивного_класу"]
        count = s["n_позитивного_класу"]
        lines.append(
            f"| {s['платформа']} | {_num(share['20'])} ({count['20']}) | "
            f"{_num(share['25'])} ({count['25']}) | {_num(share['33'])} ({count['33']}) |"
        )

    lines += [
        "",
        "У дужках — кількість рядків позитивного класу.",
        "",
        "### Режими видань",
        "",
        "| Платформа | Видань усього | Власний поріг | Загальний поріг | Ненульова мітка | Рядків у режимі «ненульова мітка» |",
        "|---|---|---|---|---|---|",
    ]
    for s in summaries:
        by_mode = s["видань_за_режимами"]
        lines.append(
            f"| {s['платформа']} | {s['видань_усього']} | {by_mode[MODE_OWN]} | "
            f"{by_mode[MODE_GLOBAL]} | {by_mode[MODE_NONZERO]} | "
            f"{s['рядків_у_режимі_ненульова_мітка']} |"
        )

    lines += [
        "",
        "Останній стовпчик іде в обмеження роботи: для цих рядків мітка означає",
        "«реакцій більше нуля», а не «верхня чверть видання», тому зміст",
        "цільової змінної між виданнями неоднорідний (ADR-005, запасне правило 2).",
        "",
        "### Видання без варіації мітки",
        "",
        "Поріг у верхню чверть може збігтися з максимумом видання, якщо значень",
        "поблизу нього багато однакових. Строга нерівність тоді не дає жодного",
        "позитивного рядка. Всередині такого видання задача вироджується.",
        "",
        "| Платформа | Верхні 20% | Верхні 25% (основний) | Верхні 33% |",
        "|---|---|---|---|",
    ]
    for s in summaries:
        d = s["вироджені_видання"]
        lines.append(
            "| {} | {} ({}) | {} ({}) | {} ({}) |".format(
                s["платформа"],
                d["20"]["видань_без_позитивного_класу"]
                + d["20"]["видань_без_негативного_класу"],
                d["20"]["рядків"],
                d["25"]["видань_без_позитивного_класу"]
                + d["25"]["видань_без_негативного_класу"],
                d["25"]["рядків"],
                d["33"]["видань_без_позитивного_класу"]
                + d["33"]["видань_без_негативного_класу"],
                d["33"]["рядків"],
            )
        )

    lines += [
        "",
        "У дужках — кількість рядків у таких виданнях. Перелік видань —",
        "у `data/processed/classification_thresholds.json`.",
        "",
        "### Загальні пороги платформи",
        "",
        "Застосовуються до видань у режимі «загальний поріг».",
        "",
        "| Платформа | Верхні 20% | Верхні 25% | Верхні 33% |",
        "|---|---|---|---|",
    ]
    for s in summaries:
        t = s["загальні_пороги"]
        lines.append(
            f"| {s['платформа']} | {_num(t['20'])} | {_num(t['25'])} | {_num(t['33'])} |"
        )

    lines.append("")
    return lines


def _num(value: float) -> str:
    """Число з комою як десятковим роздільником (CLAUDE.md розділ 5)."""
    return f"{value:.4f}".rstrip("0").rstrip(".").replace(".", ",") or "0"


def main() -> None:
    frame = load_annotated()
    targets, thresholds, summaries = build_all(frame)

    TARGETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    targets.to_parquet(TARGETS_PATH, index=False)

    document = thresholds_document(thresholds)
    THRESHOLDS_PATH.write_text(
        json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = report_lines(summaries)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"мітка: {TARGETS_PATH.relative_to(PROJECT_ROOT)} — {len(targets)} рядків")
    print(f"пороги: {THRESHOLDS_PATH.relative_to(PROJECT_ROOT)}")
    print(f"звіт: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
