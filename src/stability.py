"""Стабільність анотатора між прогонами: EXP-002, частина друга.

Запуск:
    .venv/bin/python -m src.stability

Числове ядро — `weighted_kappa` і `bootstrap_kappa_ci` з `src.agreement`,
той самий код, що дав числа першої частини. Тут додано те, чого перша
частина не потребувала: стратифікацію за режимом розміру пачки, зважене
середнє за часткою режимів у черзі, явне виключення нечислових міток
і паралельний розрахунок на вибірці без дублікатів заголовків.

Три рішення, зафіксовані до розрахунку (запит користувача від 7 серпня
2026 року, після завершення розмітки run2 і до порівняння з run1):

1. Основною подається каппа на вибірці без дублікатів заголовків, повна
   наводиться поруч. Дублікати визначаються нормалізацією тексту
   і мірою подібності difflib з порогом DUPLICATE_THRESHOLD. Поріг
   зафіксовано до розрахунку каппи і не переглядався.
2. Рядки з нечисловою міткою (`unclear`) виключаються з розрахунку каппи
   по відповідній ознаці явно, з переліком IDLink у результаті.
3. Рівні шкали, не використані жодним прогоном, фіксуються по кожній
   ознаці: каппа по таких ознаках спирається на коротшу шкалу, ніж
   передбачає рубрика.
"""

from __future__ import annotations

import difflib
import html
import json
import platform
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from src.agreement import bootstrap_kappa_ci, weighted_kappa

FEATURES = {
    "gap": ("1. Прогалина в інформації", 3),
    "scale": ("2. Масштаб наслідків", 3),
    "concrete": ("3. Конкретність предиката", 3),
    "conflict": ("4. Явність конфлікту", 3),
    "authority": ("5. Вага авторитету", 3),
    "violation": ("7. Порушення очікування", 3),
    "stakes": ("9. Проспективна невизначеність", 3),
}

# Частки режимів розміру пачки в черзі, зафіксовані протоколом:
# рядки 1–550 розмічені пачками по 25, 551–1750 — по 100, 1751–5055 — по 200.
REGIME_WEIGHTS = {25: 550, 100: 1200, 200: 3305}

DUPLICATE_THRESHOLD = 0.80
BOOTSTRAP_ITERATIONS = 2000
RANDOM_STATE = 42

SAMPLE = Path("annotations/stability_sample_300_blank.csv")
STRATA = Path("annotations/stability_sample_300_strata.csv")
RUN1 = Path("annotations/llm_labels_run1.csv")
RUN2 = Path("annotations/llm_labels_run2.csv")
AGREEMENT = Path("experiments/EXP-002_annotation-agreement/agreement.json")
OUT = Path("experiments/EXP-002_annotation-agreement/stability.json")


def normalise_title(title: str) -> str:
    """Нормалізація заголовка для пошуку дублікатів.

    Знімає службовий префікс рубрики до символу «|», HTML-сутності,
    регістр і пунктуацію. Змістовних слів не чіпає.
    """
    text = html.unescape(str(title))
    if "|" in text:
        text = text.split("|")[-1]
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def find_duplicate_groups(titles: pd.Series) -> list[list[int]]:
    """Групи позицій, чиї заголовки збігаються повністю або майже дослівно.

    Порівняння попарне, послідовне: рядок приєднується до першої групи,
    з представником якої його подібність не нижча за поріг. Порядок
    рядків у вибірці зафіксовано, тому результат детермінований.
    """
    normalised = [normalise_title(t) for t in titles]
    groups: list[list[int]] = []
    for i, text in enumerate(normalised):
        for group in groups:
            ratio = difflib.SequenceMatcher(None, normalised[group[0]], text).ratio()
            if ratio >= DUPLICATE_THRESHOLD:
                group.append(i)
                break
        else:
            groups.append([i])
    return [g for g in groups if len(g) > 1]


def feature_stats(a: np.ndarray, b: np.ndarray, max_level: int) -> dict:
    delta = b - a
    matrix = np.zeros((max_level + 1, max_level + 1), dtype=int)
    for x, y in zip(a, b):
        matrix[x, y] += 1
    return {
        "n": int(len(a)),
        "kappa_linear": round(weighted_kappa(a, b, max_level, "linear"), 4),
        "kappa_linear_ci": bootstrap_kappa_ci(
            a, b, max_level, "linear", BOOTSTRAP_ITERATIONS, RANDOM_STATE
        ),
        "kappa_quadratic": round(weighted_kappa(a, b, max_level, "quadratic"), 4),
        "kappa_quadratic_ci": bootstrap_kappa_ci(
            a, b, max_level, "quadratic", BOOTSTRAP_ITERATIONS, RANDOM_STATE
        ),
        "точний збіг": round(float((a == b).mean()), 4),
        "у межах 1 рівня": round(float((np.abs(delta) <= 1).mean()), 4),
        "середній зсув run2 мінус run1": round(float(delta.mean()), 4),
        "середнє run1": round(float(a.mean()), 3),
        "середнє run2": round(float(b.mean()), 3),
        "матриця": matrix.tolist(),
    }


def analyse_set(
    frame: pd.DataFrame, label: str, unclear_ids: dict[str, list[int]]
) -> dict:
    """Каппа по кожній ознаці, загалом і окремо по стратах режиму пачки."""
    result = {"label": label, "n": int(len(frame)), "features": {}}

    for key, (name, max_level) in FEATURES.items():
        mask = frame[f"{key}_1"].notna() & frame[f"{key}_2"].notna()
        dropped = sorted(int(x) for x in frame.loc[~mask, "IDLink"])
        a = frame.loc[mask, f"{key}_1"].to_numpy(dtype=int)
        b = frame.loc[mask, f"{key}_2"].to_numpy(dtype=int)

        entry = {"ознака": name, **feature_stats(a, b, max_level)}
        entry["виключено_нечислових"] = len(dropped)
        entry["виключено_IDLink"] = dropped
        entry["невикористані_рівні_run1"] = [
            lvl for lvl in range(max_level + 1) if lvl not in set(a.tolist())
        ]
        entry["невикористані_рівні_run2"] = [
            lvl for lvl in range(max_level + 1) if lvl not in set(b.tolist())
        ]

        strata = {}
        for regime in sorted(REGIME_WEIGHTS):
            part = frame.loc[mask & (frame["batch_regime"] == regime)]
            if part.empty:
                continue
            strata[str(regime)] = {
                "частка_режиму_в_черзі": REGIME_WEIGHTS[regime],
                **feature_stats(
                    part[f"{key}_1"].to_numpy(dtype=int),
                    part[f"{key}_2"].to_numpy(dtype=int),
                    max_level,
                ),
            }
        entry["страти"] = strata

        total = sum(REGIME_WEIGHTS[int(r)] for r in strata)
        entry["kappa_linear_зважена_за_чергою"] = round(
            sum(
                strata[r]["kappa_linear"] * REGIME_WEIGHTS[int(r)] for r in strata
            )
            / total,
            4,
        )
        result["features"][key] = entry

    return result


def check_hypothesis(analysis: dict) -> dict:
    """Гіпотеза 1а: стабільність між прогонами вища за узгодженість з людиною.

    Каппа першої частини береться з `agreement.json` тієї самої папки.
    Загальною каппою стабільності протокол визначив зважене середнє
    за часткою режимів у черзі, тому порівняння ведеться по ній;
    пулована каппа наводиться поруч.
    """
    part_one = {
        row["код"]: row["kappa_linear"]
        for row in json.load(open(AGREEMENT, encoding="utf-8"))["features"]
    }
    rows = {}
    for key in FEATURES:
        entry = analysis["features"][key]
        weighted = entry["kappa_linear_зважена_за_чергою"]
        rows[key] = {
            "kappa_людина_частина1": part_one[key],
            "kappa_стабільності_зважена": weighted,
            "kappa_стабільності_пулована": entry["kappa_linear"],
            "різниця_зважена_мінус_людина": round(weighted - part_one[key], 4),
            "гіпотеза_підтверджена": bool(weighted > part_one[key]),
        }
    return {
        "формулювання": (
            "повторний прогін відтворює мітки першого прогону на рівні, "
            "вищому за узгодженість з людиною"
        ),
        "джерело_каппи_частини_1": str(AGREEMENT),
        "підтверджено_ознак": sum(r["гіпотеза_підтверджена"] for r in rows.values()),
        "усього_ознак": len(rows),
        "за_ознаками": rows,
    }


def check_batch_rule(analysis: dict) -> dict:
    """Правило страт: чи погіршив розмір пачки 200 якість розмітки.

    Протокол: «якщо каппа страти пачка 200 нижча за каппу страти пачка 25
    більш ніж на ширину довірчого інтервалу». Який саме з двох інтервалів
    береться за міру, протокол не вказує, тому обидва читання наводяться
    поруч і жодне не подається як єдине.
    """
    rows = {}
    for key in FEATURES:
        strata = analysis["features"][key]["страти"]
        k25, k200 = strata["25"]["kappa_linear"], strata["200"]["kappa_linear"]
        w25 = strata["25"]["kappa_linear_ci"]
        w200 = strata["200"]["kappa_linear_ci"]
        width25 = round(w25["ci_high"] - w25["ci_low"], 4)
        width200 = round(w200["ci_high"] - w200["ci_low"], 4)
        diff = round(k25 - k200, 4)
        rows[key] = {
            "kappa_пачка_25": k25,
            "kappa_пачка_100": strata["100"]["kappa_linear"],
            "kappa_пачка_200": k200,
            "різниця_25_мінус_200": diff,
            "ширина_ДІ_страти_25": width25,
            "ширина_ДІ_страти_200": width200,
            "погіршення_за_ДІ_страти_25": bool(diff > width25),
            "погіршення_за_ДІ_страти_200": bool(diff > width200),
        }
    return {
        "неоднозначність_правила": (
            "протокол не вказує, ширина якого з двох довірчих інтервалів "
            "є мірою порівняння; за ДІ страти 25 правило спрацьовує, "
            "за ДІ страти 200 — ні"
        ),
        "спрацювало_за_ДІ_25": sorted(
            k for k, v in rows.items() if v["погіршення_за_ДІ_страти_25"]
        ),
        "спрацювало_за_ДІ_200": sorted(
            k for k, v in rows.items() if v["погіршення_за_ДІ_страти_200"]
        ),
        "за_ознаками": rows,
    }


def scale_limits(analysis: dict) -> dict:
    """Рівні шкали, не використані жодним прогоном.

    Каппа по такій ознаці спирається на коротшу шкалу, ніж передбачає
    рубрика, і це обмеження фіксується разом з числом.
    """
    rows = {}
    for key, (name, max_level) in FEATURES.items():
        entry = analysis["features"][key]
        unused = sorted(
            set(entry["невикористані_рівні_run1"])
            & set(entry["невикористані_рівні_run2"])
        )
        counts_1 = [sum(row) for row in entry["матриця"]]
        counts_2 = [sum(col) for col in zip(*entry["матриця"])]
        rows[key] = {
            "ознака": name,
            "рівнів_за_рубрикою": max_level + 1,
            "не_використано_жодним_прогоном": unused,
            "фактична_довжина_шкали": max_level + 1 - len(unused),
            "частоти_run1": counts_1,
            "частоти_run2": counts_2,
        }
    return {
        "правило": (
            "рівень вважається невикористаним, якщо його не поставив "
            "жоден із двох прогонів; каппа по такій ознаці фактично "
            "спирається на коротшу шкалу"
        ),
        "за_ознаками": rows,
    }


def main() -> None:
    sample = pd.read_csv(SAMPLE, dtype={"IDLink": int})
    strata = pd.read_csv(STRATA, dtype={"IDLink": int, "batch_regime": int})
    run1 = pd.read_csv(RUN1, dtype={"IDLink": int}).set_index("IDLink")
    run2 = pd.read_csv(RUN2, dtype={"IDLink": int}).set_index("IDLink")

    frame = sample[["IDLink", "Title"]].copy()
    frame = frame.merge(strata[["IDLink", "batch_regime"]], on="IDLink", how="left")
    for key in FEATURES:
        # to_numeric з errors="coerce" перетворює 'unclear' на NaN; такі рядки
        # виключаються з каппи по цій ознаці й перелічуються в результаті.
        frame[f"{key}_1"] = pd.to_numeric(
            run1.loc[frame["IDLink"], key].to_numpy(), errors="coerce"
        )
        frame[f"{key}_2"] = pd.to_numeric(
            run2.loc[frame["IDLink"], key].to_numpy(), errors="coerce"
        )

    unclear = {}
    for key in FEATURES:
        bad = frame.loc[
            frame[f"{key}_1"].isna() | frame[f"{key}_2"].isna(), "IDLink"
        ]
        if len(bad):
            unclear[key] = sorted(int(x) for x in bad)

    groups = find_duplicate_groups(frame["Title"])
    drop_positions = sorted(pos for group in groups for pos in group[1:])
    deduped = frame.drop(index=frame.index[drop_positions]).reset_index(drop=True)

    duplicates = [
        {
            "залишено": {
                "IDLink": int(frame["IDLink"].iloc[group[0]]),
                "Title": str(frame["Title"].iloc[group[0]]),
            },
            "виключено": [
                {
                    "IDLink": int(frame["IDLink"].iloc[pos]),
                    "Title": str(frame["Title"].iloc[pos]),
                    "подібність": round(
                        difflib.SequenceMatcher(
                            None,
                            normalise_title(frame["Title"].iloc[group[0]]),
                            normalise_title(frame["Title"].iloc[pos]),
                        ).ratio(),
                        4,
                    ),
                }
                for pos in group[1:]
            ],
        }
        for group in groups
    ]

    main_analysis = analyse_set(deduped, "без дублікатів заголовків", unclear)
    full_analysis = analyse_set(frame, "усі 300 рядків", unclear)

    result = {
        "експеримент": "EXP-002, частина друга — стабільність між прогонами",
        "config": {
            "run1": str(RUN1),
            "run2": str(RUN2),
            "вибірка": str(SAMPLE),
            "страти": str(STRATA),
            "seed_вибірки": 45,
            "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
            "random_state": RANDOM_STATE,
            "поріг_подібності_дублікатів": DUPLICATE_THRESHOLD,
            "частки_режимів_у_черзі": {str(k): v for k, v in REGIME_WEIGHTS.items()},
            "версії": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "pandas": pd.__version__,
                "numpy": np.__version__,
                "scikit-learn": sklearn.__version__,
            },
        },
        "дублікати": {
            "правило": (
                "нормалізація заголовка (зняття префікса рубрики до «|», "
                "HTML-сутностей, регістру, пунктуації) і подібність difflib "
                f"не нижча за {DUPLICATE_THRESHOLD}; з групи залишається "
                "перший за порядком вибірки рядок"
            ),
            "груп": len(groups),
            "виключено_рядків": len(drop_positions),
            "групи": duplicates,
        },
        "нечислові_мітки": {
            "правило": (
                "рядок з міткою unclear виключається з розрахунку каппи "
                "по відповідній ознаці; на інші ознаки не впливає"
            ),
            "за_ознаками": unclear,
        },
        "основний_розрахунок": main_analysis,
        "повний_розрахунок": full_analysis,
        "перевірка_гіпотези": check_hypothesis(main_analysis),
        "правило_страт": check_batch_rule(main_analysis),
        "обмеження_шкали": scale_limits(main_analysis),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    main_set = result["основний_розрахунок"]
    full_set = result["повний_розрахунок"]
    print(
        f"Основний розрахунок: n = {main_set['n']} "
        f"(виключено дублікатів: {len(drop_positions)}); "
        f"повний: n = {full_set['n']}\n"
    )
    header = (
        f"{'ознака':32s} {'κ лін. без дубл.':>24s} {'κ лін. усі 300':>24s} "
        f"{'зсув':>7s}"
    )
    print(header)
    print("-" * len(header))
    for key, (name, _) in FEATURES.items():
        m, fl = main_set["features"][key], full_set["features"][key]
        mci, fci = m["kappa_linear_ci"], fl["kappa_linear_ci"]
        left = f"{m['kappa_linear']:.3f} [{mci['ci_low']:.3f}; {mci['ci_high']:.3f}]"
        right = f"{fl['kappa_linear']:.3f} [{fci['ci_low']:.3f}; {fci['ci_high']:.3f}]"
        shift = m["середній зсув run2 мінус run1"]
        print(f"{name:32s} {left:>24s} {right:>24s} {shift:+7.3f}")
    print("-" * len(header))
    print("\nКаппа лінійна по стратах розміру пачки (основний розрахунок):")
    print(f"{'ознака':32s} {'пачка 25':>10s} {'пачка 100':>10s} {'пачка 200':>10s}")
    for key, (name, _) in FEATURES.items():
        st = main_set["features"][key]["страти"]
        cells = "".join(f"{st[r]['kappa_linear']:>10.3f}" for r in ("25", "100", "200"))
        print(f"{name:32s}{cells}")
    print("\nЗаписано:", OUT)


if __name__ == "__main__":
    main()
