"""Протокол крос-валідації класифікаційної лінії (ADR-005).

Запуск:
    .venv/bin/python -m src.data.cv_protocol [--force]

Схема — `StratifiedGroupKFold`, п'ять фолдів, чотири повтори з різними
`random_state`, разом двадцять розбиттів на кожне поєднання платформи
і перцентиля.

**Групування за текстом заголовка.** Група — нормалізований заголовок,
колонка `title_norm` з `src/data/load.py`. У повному корпусі до груп
повторюваних заголовків належать 22,09% рядків (ADR-002): ту саму новину
публікують різні джерела. На анотованій підвибірці частка вдвічі нижча —
точне число дає звіт, — бо вибірка Фази 5 складається з 50 великих видань.
Без групування один заголовок потрапив би і в навчальну, і
в перевірочну частину фолда, а копірайтингові ознаки для однакового тексту
збігаються цілком, тож модель отримала б у перевірці вже бачену підказку.
Це той самий ризик, який ADR-002 закривав видаленням груп, що перетинають
межу часового спліту; тут він закривається групуванням усередині фолда.

**Чому двадцять розбиттів, а не п'ять.** Головна величина цієї лінії —
парне порівняння наборів ознак на однакових розбиттях. При n = 5 критерій
Вілкоксона має мінімальне можливе двостороннє p = 0,0625 і не може дати
значущості в принципі, скільки б не був великий ефект. Чотири повтори
знімають це обмеження.

**Чому потрібен shuffle.** `StratifiedGroupKFold` за замовчуванням
детермінований: без `shuffle=True` усі чотири повтори дали б однакові фолди,
і повторення було б порожньою роботою. Тому `shuffle=True` і різний
`random_state` у кожному повторі: 42, 43, 44, 45. Число 42 вже вживається
в проєкті як базове (CLAUDE.md розділ 6, пункт 4).

**Чому дев'ять окремих наборів фолдів.** Стратифікація йде за цільовою
міткою, а мітка своя для кожного поєднання платформи і перцентиля. Усі
порівняння відбуваються всередині одного поєднання, тому фолди будуються
за тією самою міткою, з якою потім працюватимуть: так частка позитивного
класу по фолдах рівніша.

**Файл генерується один раз.** Перегенерація без явного `--force` заборонена:
розбиття, на яких уже щось пораховано, мінятися не можуть, інакше парне
порівняння втратить підставу.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEAN_PATH = PROJECT_ROOT / "data" / "interim" / "news_clean.parquet"
TARGETS_PATH = PROJECT_ROOT / "data" / "processed" / "targets_classification.parquet"
FOLDS_PATH = PROJECT_ROOT / "data" / "processed" / "cv_folds.parquet"
REPORT_PATH = PROJECT_ROOT / "reports" / "tables" / "cv_protocol.md"

PLATFORMS = {"Facebook": "fb", "GooglePlus": "gp", "LinkedIn": "li"}
TOP_SHARES = ("25", "20", "33")
MAIN_SHARE = "25"

# Допоміжна мітка з глобальним порогом (ADR-005, четверте доповнення) має
# власні фолди в окремому файлі: основні не зачіпаються. Стратифікація йде
# за іншою міткою, тому розбиття закономірно інші.
GLOBAL_TARGETS_PATH = PROJECT_ROOT / "data" / "processed" / "targets_global.parquet"
GLOBAL_FOLDS_PATH = PROJECT_ROOT / "data" / "processed" / "cv_folds_global.parquet"
GLOBAL_REPORT_PATH = PROJECT_ROOT / "reports" / "tables" / "cv_protocol_global.md"

MODES = {
    "main": {
        "targets": TARGETS_PATH,
        "folds": FOLDS_PATH,
        "report": REPORT_PATH,
        "shares": TOP_SHARES,
        "main_share": MAIN_SHARE,
        "опис": "основна мітка, три пороги",
    },
    "global": {
        "targets": GLOBAL_TARGETS_PATH,
        "folds": GLOBAL_FOLDS_PATH,
        "report": GLOBAL_REPORT_PATH,
        "shares": ("global",),
        "main_share": "global",
        "опис": "допоміжна мітка з глобальним порогом",
    },
}

N_SPLITS = 5
RANDOM_STATES = (42, 43, 44, 45)

# Поріг, за яким розмах частки позитивного класу по фолдах вважається
# завеликим. Величина обрана до запуску: 5 процентних пунктів на частці
# близько 25% означає відхилення у п'яту частину базового рівня.
SPREAD_ALERT_PP = 5.0


class ProtocolError(RuntimeError):
    """Перевірка протоколу не пройдена. Фолди не записуються."""


def git_commit() -> str:
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


def load_inputs(targets_path: Path = TARGETS_PATH) -> pd.DataFrame:
    """Мітки разом з групою заголовка і виданням.

    Файл мітки тільки читається: мітка зафіксована правилом 4 першого
    доповнення ADR-005 і тут не перераховується.
    """
    targets = pd.read_parquet(targets_path)
    clean = pd.read_parquet(CLEAN_PATH, columns=["IDLink", "title_norm", "Source"])
    clean["IDLink"] = clean["IDLink"].astype("int64")
    frame = targets.merge(clean, on="IDLink", how="left", validate="one_to_one")
    missing = int(frame["title_norm"].isna().sum())
    if missing:
        raise ProtocolError(
            f"{missing} анотованих рядків не знайшли відповідника "
            f"в {CLEAN_PATH.name}; групування було б неповним"
        )
    return frame


def duplicate_groups(frame: pd.DataFrame) -> dict:
    """Групи повторюваних заголовків серед анотованих рядків."""
    sizes = frame["title_norm"].value_counts()
    repeated = sizes[sizes > 1]
    largest = repeated.index[0] if len(repeated) else None
    return {
        "n_рядків": int(len(frame)),
        "n_груп": int(len(sizes)),
        "n_груп_з_повторами": int(len(repeated)),
        "n_рядків_у_повторах": int(repeated.sum()) if len(repeated) else 0,
        "частка_рядків_у_повторах": round(
            float(repeated.sum() / len(frame)) if len(repeated) else 0.0, 4
        ),
        "найбільша_група": int(sizes.iloc[0]),
        "найбільша_група_заголовок": str(largest) if largest is not None else None,
        "груп_від_50_рядків": int((sizes >= 50).sum()),
    }


def build_folds(frame: pd.DataFrame, shares: tuple[str, ...] = TOP_SHARES) -> pd.DataFrame:
    """Двадцять розбиттів на кожне з дев'яти поєднань платформи і перцентиля.

    Рядок з NaN у мітці для цієї платформи у фолди не входить узагалі:
    відсутнє вимірювання не є ані позитивним, ані негативним класом
    (правило ADR-003, перенесене в ADR-005).
    """
    records = []
    for platform, short in PLATFORMS.items():
        for share in shares:
            column = f"y_{short}_{share}"
            subset = frame[frame[column].notna()]
            y = subset[column].to_numpy(dtype=np.int8)
            groups = subset["title_norm"].to_numpy()
            ids = subset["IDLink"].to_numpy()

            for repeat, seed in enumerate(RANDOM_STATES):
                splitter = StratifiedGroupKFold(
                    n_splits=N_SPLITS, shuffle=True, random_state=seed
                )
                for fold, (train_idx, test_idx) in enumerate(
                    splitter.split(ids.reshape(-1, 1), y, groups)
                ):
                    for part, index in (("train", train_idx), ("test", test_idx)):
                        records.append(
                            pd.DataFrame(
                                {
                                    "IDLink": ids[index],
                                    "platform": platform,
                                    "percentile": share,
                                    "repeat": np.int8(repeat),
                                    "fold": np.int8(fold),
                                    "part": part,
                                }
                            )
                        )

    folds = pd.concat(records, ignore_index=True)
    for column in ("platform", "percentile", "part"):
        folds[column] = folds[column].astype("category")
    return folds


def check_no_group_leak(folds: pd.DataFrame, frame: pd.DataFrame) -> dict:
    """Перевірка 1. Жодна група заголовка не лежить по обидва боки фолда.

    Це головна перевірка протоколу. Витік через повторюваний заголовок
    метрикою не виявляється: він робить числа кращими, а не гіршими.
    """
    titles = frame.set_index("IDLink")["title_norm"]
    marked = folds.assign(title_norm=folds["IDLink"].map(titles))
    key = ["platform", "percentile", "repeat", "fold", "title_norm"]
    parts = marked.groupby(key, observed=True)["part"].nunique()
    offenders = parts[parts > 1]

    if len(offenders):
        first = offenders.index[0]
        example = marked[
            (marked["platform"] == first[0])
            & (marked["percentile"] == first[1])
            & (marked["repeat"] == first[2])
            & (marked["fold"] == first[3])
            & (marked["title_norm"] == first[4])
        ]
        raise ProtocolError(
            "витік через групу заголовка: група присутня і в train, і в test.\n"
            f"перше порушення: платформа {first[0]}, перцентиль {first[1]}, "
            f"повтор {first[2]}, фолд {first[3]}\n"
            f"заголовок: {first[4]!r}\n"
            f"{example.to_string(index=False)}\n"
            f"усього порушень: {len(offenders)}"
        )

    n_splits = int(
        folds.groupby(["platform", "percentile", "repeat", "fold"], observed=True).ngroups
    )
    return {"пройдено": True, "перевірено_розбиттів": n_splits}


def check_test_once_per_repeat(folds: pd.DataFrame) -> dict:
    """Перевірка 2. У межах повтору кожен рядок буває в test рівно раз."""
    test = folds[folds["part"] == "test"]
    counts = test.groupby(
        ["platform", "percentile", "repeat", "IDLink"], observed=True
    ).size()
    offenders = counts[counts != 1]
    if len(offenders):
        raise ProtocolError(
            "рядок потрапив у test не рівно один раз у межах повтору.\n"
            f"перші порушення:\n{offenders.head(10).to_string()}"
        )
    return {"пройдено": True, "перевірено_рядків": int(len(counts))}


def check_nan_excluded(
    folds: pd.DataFrame, frame: pd.DataFrame, shares: tuple[str, ...] = TOP_SHARES
) -> dict:
    """Перевірка 3. Рядки з NaN у мітці у фолди не потрапляють."""
    excluded_total = 0
    for platform, short in PLATFORMS.items():
        for share in shares:
            column = f"y_{short}_{share}"
            missing = set(frame.loc[frame[column].isna(), "IDLink"])
            excluded_total += len(missing)
            if not missing:
                continue
            present = folds[
                (folds["platform"] == platform) & (folds["percentile"] == share)
            ]
            intruders = sorted(set(present["IDLink"]) & missing)
            if intruders:
                raise ProtocolError(
                    f"рядки з NaN у мітці {column} потрапили у фолди: "
                    f"{intruders[:10]} (усього {len(intruders)})"
                )
    return {"пройдено": True, "рядків_з_NaN_у_мітці": excluded_total}


def fold_statistics(
    folds: pd.DataFrame, frame: pd.DataFrame, shares: tuple[str, ...] = TOP_SHARES
) -> list[dict]:
    """Частка позитивного класу і розміри частин по кожному поєднанню."""
    stats = []
    for platform, short in PLATFORMS.items():
        for share in shares:
            column = f"y_{short}_{share}"
            labels = frame.set_index("IDLink")[column]
            subset = folds[
                (folds["platform"] == platform) & (folds["percentile"] == share)
            ]
            test = subset[subset["part"] == "test"].assign(
                y=lambda d: d["IDLink"].map(labels)
            )
            per_fold = test.groupby(["repeat", "fold"], observed=True)["y"].agg(
                ["mean", "size"]
            )
            train_sizes = (
                subset[subset["part"] == "train"]
                .groupby(["repeat", "fold"], observed=True)
                .size()
            )
            fold_shares = per_fold["mean"]
            stats.append(
                {
                    "платформа": platform,
                    "перцентиль": share,
                    "n_розбиттів": int(len(per_fold)),
                    "частка_середня": round(float(fold_shares.mean()), 4),
                    "частка_мінімум": round(float(fold_shares.min()), 4),
                    "частка_максимум": round(float(fold_shares.max()), 4),
                    "розмах_пп": round(
                        float((fold_shares.max() - fold_shares.min()) * 100), 2
                    ),
                    "test_середній": round(float(per_fold["size"].mean()), 1),
                    "test_мінімум": int(per_fold["size"].min()),
                    "test_максимум": int(per_fold["size"].max()),
                    "train_середній": round(float(train_sizes.mean()), 1),
                }
            )
    return stats


def _num(value: float, digits: int = 4) -> str:
    """Число з комою як десятковим роздільником (CLAUDE.md розділ 5)."""
    return f"{value:.{digits}f}".replace(".", ",")


def share_label(share: str) -> str:
    """Підпис порога мітки в таблицях."""
    return "глобальний поріг" if share == "global" else f"верхні {share}%"


def report_lines(
    groups: dict, stats: list[dict], checks: dict, main_share: str = MAIN_SHARE
) -> list[str]:
    alerts = [s for s in stats if s["розмах_пп"] > SPREAD_ALERT_PP]

    lines = [
        "# Протокол крос-валідації класифікаційної лінії",
        "",
        "Згенеровано `src/data/cv_protocol.py`. Числа не редагуються вручну.",
        "",
        "Схема: `StratifiedGroupKFold`, 5 фолдів, 4 повтори з `random_state` "
        "42, 43, 44, 45,",
        "разом 20 розбиттів на кожне поєднання платформи і перцентиля.",
        "Група — нормалізований заголовок `title_norm`. Стратифікація — "
        "за цільовою міткою.",
        "",
        "### Групи повторюваних заголовків серед анотованих рядків",
        "",
        f"- рядків усього: {groups['n_рядків']}",
        f"- різних заголовків: {groups['n_груп']}",
        f"- груп із більш ніж одним рядком: {groups['n_груп_з_повторами']}",
        f"- рядків у таких групах: {groups['n_рядків_у_повторах']} "
        f"({_num(groups['частка_рядків_у_повторах'])} від усіх)",
        f"- найбільша група: {groups['найбільша_група']} рядків",
        f"- груп від 50 рядків: {groups['груп_від_50_рядків']}",
        "",
        "Саме через ці групи схема потребує групування: без нього однаковий "
        "заголовок",
        "опинився б по обидва боки фолда, а копірайтингові ознаки для нього "
        "збігаються цілком.",
        "",
        "### Частка позитивного класу по фолдах",
        "",
        "| Платформа | Перцентиль | Середня | Мінімум | Максимум | Розмах, пп |",
        "|---|---|---|---|---|---|",
    ]
    for s in stats:
        main = " (основний)" if s["перцентиль"] == main_share and main_share != "global" else ""
        lines.append(
            f"| {s['платформа']} | {share_label(s['перцентиль'])}{main} | "
            f"{_num(s['частка_середня'])} | {_num(s['частка_мінімум'])} | "
            f"{_num(s['частка_максимум'])} | {_num(s['розмах_пп'], 2)} |"
        )

    lines += [
        "",
        f"Мінімум і максимум беруться по 20 розбиттях кожного поєднання. "
        f"Порогом уваги обрано {_num(SPREAD_ALERT_PP, 1)} процентного пункту:",
        "розмах вище нього означав би, що стратифікація погано уживається "
        "з групуванням.",
        "",
    ]
    if alerts:
        lines.append("**Поєднання з розмахом вище порога:**")
        lines.append("")
        for s in alerts:
            lines.append(
                f"- {s['платформа']}, {share_label(s['перцентиль'])}: "
                f"{_num(s['розмах_пп'], 2)} пп"
            )
        lines.append("")
    else:
        lines += [
            "Жодне поєднання порога не перевищило: стратифікація з групуванням "
            "уживається.",
            "",
        ]

    lines += [
        "### Розміри частин у фолді",
        "",
        "| Платформа | Перцентиль | Train, середній | Test, середній | Test, мінімум | Test, максимум |",
        "|---|---|---|---|---|---|",
    ]
    for s in stats:
        lines.append(
            f"| {s['платформа']} | {share_label(s['перцентиль'])} | "
            f"{_num(s['train_середній'], 1)} | {_num(s['test_середній'], 1)} | "
            f"{s['test_мінімум']} | {s['test_максимум']} |"
        )

    lines += [
        "",
        "Розміри фолдів нерівні, бо групи повторюваних заголовків не діляться.",
        "",
        "### Перевірки протоколу",
        "",
        "| Перевірка | Результат |",
        "|---|---|",
        "| Жодна група заголовка не лежить по обидва боки фолда | "
        f"пройдено, {checks['витік']['перевірено_розбиттів']} розбиттів |",
        "| Кожен рядок у test рівно раз у межах повтору | "
        f"пройдено, {checks['test_раз']['перевірено_рядків']} перевірок |",
        "| Рядки з NaN у мітці у фолди не входять | "
        f"пройдено, рядків з NaN у мітці: {checks['nan']['рядків_з_NaN_у_мітці']} |",
        "",
        "Перевірки виконуються до запису файлу. Якщо будь-яка падає, фолди",
        "не записуються взагалі.",
        "",
    ]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="перезаписати наявний файл фолдів",
    )
    parser.add_argument(
        "--mode",
        choices=tuple(MODES),
        default="main",
        help="main — основна мітка; global — допоміжна з глобальним порогом",
    )
    args = parser.parse_args()

    mode = MODES[args.mode]
    targets_path = mode["targets"]
    folds_path = mode["folds"]
    report_path = mode["report"]
    shares = mode["shares"]
    print(f"режим: {args.mode} — {mode['опис']}")

    if folds_path.exists() and not args.force:
        raise SystemExit(
            f"{folds_path.relative_to(PROJECT_ROOT)} уже існує.\n"
            "Перегенерація заборонена: розбиття, на яких уже щось пораховано, "
            "мінятися не можуть,\n"
            "інакше парне порівняння наборів ознак втратить підставу.\n"
            "Якщо перегенерація справді потрібна, запустіть з --force."
        )

    frame = load_inputs(targets_path)
    groups = duplicate_groups(frame)
    folds = build_folds(frame, shares)

    checks = {
        "витік": check_no_group_leak(folds, frame),
        "test_раз": check_test_once_per_repeat(folds),
        "nan": check_nan_excluded(folds, frame, shares),
    }
    stats = fold_statistics(folds, frame, shares)

    folds_path.parent.mkdir(parents=True, exist_ok=True)
    folds.to_parquet(folds_path, index=False)

    lines = report_lines(groups, stats, checks, mode["main_share"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(
        json.dumps(
            {
                "створено": datetime.now().isoformat(timespec="seconds"),
                "git_commit": git_commit(),
                "режим": args.mode,
                "файл": str(folds_path.relative_to(PROJECT_ROOT)),
                "рядків": int(len(folds)),
                "розбиттів": checks["витік"]["перевірено_розбиттів"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
