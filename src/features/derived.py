"""Похідні семантичні ознаки класифікаційної лінії (ADR-006).

Запуск:
    .venv/bin/python -m src.features.derived

Модуль реалізує інженерію ознак з розділу 2 документа кафедри у складі,
зафіксованому третім доповненням ADR-006 від 16 вересня 2026 року. Перелік
похідних і склад наборів F₀—F₃ зафіксовані **до** обчислення самих ознак
і за результатами діагностики не переглядаються: розділ 2.5 документа
застерігає не додавати всі 21 парну комбінацію без обґрунтування, а 2.6
вимагає зафіксувати групування до експерименту.

Модуль нічого не навчає. Усі похідні є детермінованими функціями семи оцінок
одного заголовка, тому рахуються порядково і від інших рядків не залежать.
Це означає, що обчислити їх один раз наперед можна без витоку: жодна з них
не використовує ані цільової змінної, ані статистик по вибірці.

Масштабування тут не робиться. Воно вчиться на даних, тому належить пайплайну
і фітиться всередині фолда (ADR-002, третій рівень схеми; розділ 7 документа
кафедри).

Джерело базових оцінок — `annotations/llm_labels_run1.csv` через
`src.features.copywriting.load_labels`. Обрано саме його, а не
`data/processed/features_T2.parquet`: файл міток є єдиним джерелом істини
для рівня T2, саме з нього будується і сам parquet, і мітка `unclear`
перетворюється на NaN у тому самому місці, що й у регресійній лінії.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.features.copywriting import FEATURE_ORDER, load_labels

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TARGETS_PATH = PROJECT_ROOT / "data" / "processed" / "targets_classification.parquet"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "features_derived.parquet"
REPORT_PATH = PROJECT_ROOT / "reports" / "tables" / "features_derived.md"

# Захист від ділення на нуль у C. Вироджений випадок S_all = 0 означений
# окремо як нуль, тому мала величина тут безпечна (ADR-006, третє доповнення).
EPS = 1e-9

# Адитивне згладжування знаменника R. Четверте доповнення ADR-006: eps = 1e-9
# на 218 рядках з нульовим знаменником не згладжувало ділення, а множило
# чисельник на мільярд. З одиницею діапазон R стає від 0 до 9.
R_SMOOTHING = 1.0

# Групи для співвідношення R. Склад зафіксований третім доповненням ADR-006
# за механізмами Розділу 1 роботи, а не за важливістю ознак з EXP-006.
INFORMATIONAL = ["gap", "concrete", "stakes"]
SOCIAL_EMOTIONAL = ["scale", "conflict", "violation"]

# П'ять пар з розділу 2.5 документа кафедри. Перелік узятий звідти,
# а не складений заново.
PAIRS = [
    ("gap", "stakes"),
    ("conflict", "scale"),
    ("authority", "violation"),
    ("gap", "violation"),
    ("concrete", "scale"),
]

PROFILE = ["S_all", "N_active", "N_strong", "M", "N_max", "C", "s2", "H", "R"]
INTERACTION = [f"prod_{a}_{b}" for a, b in PAIRS]
INTERACTION_MIN = [f"min_{a}_{b}" for a, b in PAIRS]

# Набори ознак абляції. Задані явними списками, а не збираються на льоту:
# склад зафіксований до прогону і має читатися з коду без обчислень.
F0 = list(FEATURE_ORDER)
F1 = F0 + PROFILE
F2 = F0 + INTERACTION
F3 = F0 + PROFILE + INTERACTION
FEATURE_SETS = {"F0": F0, "F1": F1, "F2": F2, "F3": F3}

# Поза наборами. D — категоріальний індекс, рахується для опису корпусу;
# F_high і min-взаємодії належать експериментам E7 і E6.
OUTSIDE_SETS = ["D", "F_high"] + INTERACTION_MIN

# Правило про порожні профілі, оголошене до підрахунку частки.
EMPTY_PROFILE_LIMIT = 0.02


def membership_high(values: np.ndarray) -> np.ndarray:
    """Нечітка належність μ_high з розділу 2.7 документа кафедри.

    Кусковий запис документа: 0 при x ≤ 1, 0,5 при x = 2, 1 при x = 3.
    """
    out = np.zeros_like(values, dtype=np.float64)
    out[values == 2] = 0.5
    out[values == 3] = 1.0
    return out


def build(labels: pd.DataFrame) -> pd.DataFrame:
    """Похідні ознаки для всіх заголовків.

    Кожна похідна є функцією семи оцінок одного рядка. Порядок колонок
    у `labels` задає `FEATURE_ORDER`, тому індекси в D читаються однозначно.
    """
    x = labels[FEATURE_ORDER].to_numpy(dtype=np.float64)

    s_all = x.sum(axis=1)
    n_active = (x > 0).sum(axis=1).astype(np.float64)
    n_strong = (x >= 2).sum(axis=1).astype(np.float64)
    m = x.max(axis=1)
    n_max = (x == m[:, None]).sum(axis=1).astype(np.float64)
    c = m / (s_all + EPS)
    s2 = x.var(axis=1, ddof=1)

    # Ентропія нормованого профілю. Підсумовування йде лише по ненульових
    # p_i за звичайною угодою 0·ln0 = 0, тому eps усередині логарифма
    # не потрібен (ADR-006, четверте доповнення). Для порожнього профілю
    # нормування неможливе, і H дорівнює нулю за правилом третього доповнення.
    empty = s_all == 0
    safe_total = np.where(empty, 1.0, s_all)
    p = x / safe_total[:, None]
    terms = np.zeros_like(p)
    positive = p > 0
    terms[positive] = p[positive] * np.log(p[positive])
    h = -terms.sum(axis=1)
    h[empty] = 0.0
    c[empty] = 0.0

    informational = labels[INFORMATIONAL].to_numpy(dtype=np.float64).sum(axis=1)
    social = labels[SOCIAL_EMOTIONAL].to_numpy(dtype=np.float64).sum(axis=1)
    r = informational / (social + R_SMOOTHING)

    frame = pd.DataFrame({"IDLink": labels.index.to_numpy()})
    for name in FEATURE_ORDER:
        frame[name] = labels[name].to_numpy(dtype=np.float64)

    frame["S_all"] = s_all
    frame["N_active"] = n_active
    frame["N_strong"] = n_strong
    frame["M"] = m
    frame["N_max"] = n_max
    frame["C"] = c
    frame["s2"] = s2
    frame["H"] = h
    frame["R"] = r

    for a, b in PAIRS:
        left = labels[a].to_numpy(dtype=np.float64)
        right = labels[b].to_numpy(dtype=np.float64)
        frame[f"prod_{a}_{b}"] = left * right
        frame[f"min_{a}_{b}"] = np.minimum(left, right)

    # Індекс домінуючої ознаки. При збігу максимумів береться перший
    # за порядком FEATURE_ORDER; величина категоріальна і в модель
    # не подається (ADR-006, третє доповнення).
    frame["D"] = x.argmax(axis=1).astype(np.float64)
    frame["F_high"] = membership_high(x).sum(axis=1)
    frame["empty_profile"] = empty.astype(np.float64)

    return frame


def empty_profile_share(frame: pd.DataFrame) -> dict:
    """Частка порожніх профілів, загальна і по кожній платформі.

    Платформи мають власні множини рядків, бо рядок без вимірювання
    в задачі для цієї платформи участі не бере. Правило ADR-006
    формулює поріг «хоча б на одній платформі», тому частка рахується
    для кожної окремо.
    """
    shares = {"усі_анотовані": round(float(frame["empty_profile"].mean()), 4)}
    if TARGETS_PATH.exists():
        targets = pd.read_parquet(TARGETS_PATH)
        merged = frame[["IDLink", "empty_profile"]].merge(targets, on="IDLink")
        for platform, short in (("Facebook", "fb"), ("GooglePlus", "gp"), ("LinkedIn", "li")):
            eligible = merged[merged[f"y_{short}_25"].notna()]
            shares[platform] = round(float(eligible["empty_profile"].mean()), 4)
    return shares


def degenerate_denominators(frame: pd.DataFrame) -> dict:
    """Рядки, у яких сума соціально-емоційних оцінок дорівнює нулю.

    До четвертого доповнення ADR-006 знаменник R згладжувався величиною
    eps = 1e-9, і на цих рядках ділення перетворювалося на множення
    чисельника на мільярд. Після переходу на адитивне згладжування одиницею
    вони перестали бути викидами. Числа лишаються у звіті, щоб виправлення
    мало видимий слід, а не зникло безслідно.
    """
    social = frame[SOCIAL_EMOTIONAL].sum(axis=1)
    info = frame[INFORMATIONAL].sum(axis=1)
    zero = social == 0
    normal = frame.loc[~zero, "R"]
    return {
        "n_рядків": int(zero.sum()),
        "частка": round(float(zero.mean()), 4),
        "n_з_нульовим_чисельником": int((zero & (info == 0)).sum()),
        "R_максимум_на_решті": round(float(normal.max()), 4),
        "R_медіана_на_решті": round(float(normal.median()), 4),
        "R_мінімум_на_вироджених": round(float(frame.loc[zero, "R"].min()), 4),
        "R_медіана_на_вироджених": round(float(frame.loc[zero, "R"].median()), 4),
        "R_максимум_на_вироджених": round(float(frame.loc[zero, "R"].max()), 4),
    }


def describe(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Мінімум, медіана, максимум і частка нулів для кожної похідної."""
    rows = []
    for name in columns:
        values = frame[name].to_numpy(dtype=np.float64)
        rows.append(
            {
                "ознака": name,
                "мінімум": float(values.min()),
                "медіана": float(np.median(values)),
                "максимум": float(values.max()),
                "частка_нулів": float((values == 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def modal_share(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Частка найчастішого значення. Показує майже-константні ознаки."""
    return pd.Series(
        {name: float(frame[name].value_counts(normalize=True).iloc[0]) for name in columns}
    )


def _num(value: float, digits: int = 4) -> str:
    """Число з комою як десятковим роздільником (CLAUDE.md розділ 5)."""
    return f"{value:.{digits}f}".replace(".", ",")


def report_lines(
    frame: pd.DataFrame,
    shares: dict,
    stats: pd.DataFrame,
    corr: pd.DataFrame,
    denom: dict,
    strong: list[tuple],
    moderate: list[tuple],
    modal: pd.Series,
    derived_columns: list[str],
) -> list[str]:
    n = len(frame)
    lines = [
        "# Похідні семантичні ознаки: діагностика",
        "",
        "Згенеровано `src/features/derived.py`. Числа не редагуються вручну.",
        "",
        "Склад похідних і наборів F₀—F₃ зафіксований третім доповненням ADR-006",
        "**до** цього обчислення. Діагностика описує ознаки, але складу не змінює:",
        "жодна ознака за її результатами не відкидається.",
        "",
        "### Порожні профілі",
        "",
        "| Множина рядків | Частка заголовків з S_all = 0 |",
        "|---|---|",
    ]
    for key, value in shares.items():
        lines.append(f"| {key} | {_num(value)} |")

    over = [k for k, v in shares.items() if k != "усі_анотовані" and v > EMPTY_PROFILE_LIMIT]
    lines += [
        "",
        f"Поріг правила ADR-006 — {_num(EMPTY_PROFILE_LIMIT, 2)}, оголошений "
        "до цього підрахунку.",
    ]
    if over:
        lines.append(
            "Поріг перевищено на платформах: " + ", ".join(over) + ". "
            "Ознака `empty_profile` додається в усі набори F₀—F₃."
        )
    else:
        lines.append(
            "Поріг не перевищено на жодній платформі, тому ознака `empty_profile` "
            "в набори не додається. Колонка обчислена і збережена у файлі ознак "
            "для довідки."
        )

    lines += ["", "### Розподіл значень семи базових оцінок", "",
              "| Ознака | 0 | 1 | 2 | 3 |", "|---|---|---|---|---|"]
    for name in FEATURE_ORDER:
        counts = frame[name].value_counts(normalize=True)
        cells = [
            _num(float(counts.get(float(level), 0.0)) * 100, 2) for level in range(4)
        ]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "У відсотках. Рівень 3 ознаки `authority` не трапився жодного разу — "
        "властивість прогону анотації, зафіксована в ADR-006 і в обмеженнях.",
        "",
        "Дрібна властивість формули H: eps стоїть усередині логарифма, тому для "
        "профілю",
        "з єдиною ненульовою оцінкою ентропія дорівнює −ln(1+eps), тобто близько "
        "−1e-9.",
        "Це від'ємне значення порядку похибки заокруглення, не змістовне.",
        "",
        "### Описова статистика похідних",
        "",
        "| Ознака | Мінімум | Медіана | Максимум | Частка нулів |",
        "|---|---|---|---|---|",
    ]
    for row in stats.itertuples(index=False):
        lines.append(
            f"| {row.ознака} | {_num(row.мінімум)} | {_num(row.медіана)} | "
            f"{_num(row.максимум)} | {_num(row.частка_нулів)} |"
        )

    lines += [
        "",
        "### Знаменник R: рядки з нульовою сумою соціально-емоційних оцінок",
        "",
        f"Рядків, у яких сума `scale`, `conflict` і `violation` дорівнює нулю: "
        f"**{denom['n_рядків']}** ({_num(denom['частка'])} від усіх), "
        f"з них {denom['n_з_нульовим_чисельником']} мають нульовий і чисельник.",
        "",
        "Знаменник згладжується адитивно одиницею (ADR-006, четверте доповнення),",
        "тому ці рядки викидами не є.",
        "",
        "| Рядки | Мінімум R | Медіана R | Максимум R |",
        "|---|---|---|---|",
        f"| з нульовим знаменником | {_num(denom['R_мінімум_на_вироджених'])} | "
        f"{_num(denom['R_медіана_на_вироджених'])} | "
        f"{_num(denom['R_максимум_на_вироджених'])} |",
        f"| решта корпусу | — | {_num(denom['R_медіана_на_решті'])} | "
        f"{_num(denom['R_максимум_на_решті'])} |",
        "",
        "До четвертого доповнення знаменник згладжувався величиною 1e-9, і на цих",
        "рядках R сягала 6e9 при медіані 0,75 на решті корпусу. Числа лишені тут,",
        "щоб виправлення мало видимий слід.",
        "",
        "### Кореляції Спірмена між похідними",
        "",
        f"Обчислено на {len(derived_columns)} похідних; ознака D у матрицю "
        "не входить,",
        "бо є номінальним індексом і рангова кореляція для неї не визначена.",
        "",
        "**Пари з |ρ| понад 0,95:**",
        "",
    ]
    if strong:
        lines += ["| Пара | ρ |", "|---|---|"]
        for a, b, v in strong:
            lines.append(f"| {a} — {b} | {_num(v)} |")
    else:
        lines.append("Таких пар немає.")

    lines += ["", "**Пари з |ρ| від 0,80 до 0,95:**", ""]
    if moderate:
        lines += ["| Пара | ρ |", "|---|---|"]
        for a, b, v in moderate:
            lines.append(f"| {a} — {b} | {_num(v)} |")
    else:
        lines.append("Таких пар немає.")

    near_const = modal[modal > 0.95]
    lines += [
        "",
        "### Майже-константні похідні",
        "",
        f"Похідних, що приймають одне значення більш ніж у 95% з {n} рядків: "
        f"**{len(near_const)}**.",
        "",
    ]
    if len(near_const):
        lines += ["| Ознака | Частка модального значення |", "|---|---|"]
        for name, value in near_const.sort_values(ascending=False).items():
            lines.append(f"| {name} | {_num(value)} |")
        lines += [
            "",
            "Такі ознаки лишаються в наборах: склад зафіксований до обчислення,",
            "і відкидання за результатом діагностики було б добором за результатом.",
        ]

    lines += [
        "",
        "### Дві властивості корпусу для тексту роботи",
        "",
        "**Розрідженість парних взаємодій.** Взаємодія відмінна від нуля лише "
        "тоді,",
        "коли обидві її ознаки ненульові, а `gap` дорівнює нулю в 73,89% "
        "заголовків.",
        "Звідси висока частка нулів у чотирьох парах із п'яти.",
        "",
        "| Пара | Частка нулів |",
        "|---|---|",
    ]
    for a, b in PAIRS:
        share = float((frame[f"prod_{a}_{b}"] == 0).mean())
        lines.append(f"| {a} × {b} | {_num(share)} |")

    authority = frame["authority"].value_counts(normalize=True)
    all_zero = float(
        (frame[[f"prod_{a}_{b}" for a, b in PAIRS]] == 0).all(axis=1).mean()
    )
    lines += [
        "",
        f"У {_num(all_zero)} заголовків усі п'ять взаємодій дорівнюють нулю "
        "одночасно:",
        "для них F₂ і F₀ подають моделі однакову інформацію, а п'ять доданих "
        "колонок",
        "є константним нулем. Приріст від F₂ порівняно з F₀ може виникнути лише "
        "на решті",
        f"корпусу, тобто на {_num(1 - all_zero)} рядків.",
        "",
        "**Фактична тризначність `authority`.** Рівень 3, «високостатусне "
        "джерело",
        "виносить остаточний вирок», не трапився в анотації жодного разу.",
        "",
        "| Рівень | 0 | 1 | 2 | 3 |",
        "|---|---|---|---|---|",
        "| Частка | "
        + " | ".join(
            _num(float(authority.get(float(level), 0.0)), 4) for level in range(4)
        )
        + " |",
        "",
        "Ознака працює як тризначна, причому рівень 1, розмита атрибуція, "
        "трапляється",
        "втричі рідше за рівень 2. Практично вона близька до бінарної: джерело "
        "назване",
        "конкретно чи ні. Це звужує її внесок у похідні M, N_max і F_high "
        "і зафіксоване",
        "в третьому доповненні ADR-006 та в обмеженнях роботи.",
        "",
        "### Склад наборів",
        "",
        "| Набір | Ознак | Склад |",
        "|---|---|---|",
        f"| F₀ | {len(F0)} | сім базових оцінок |",
        f"| F₁ | {len(F1)} | F₀ + дев'ять профільних |",
        f"| F₂ | {len(F2)} | F₀ + п'ять product-взаємодій |",
        f"| F₃ | {len(F3)} | F₀ + профільні + взаємодії |",
        "",
        "Поза наборами обчислені й збережені: " + ", ".join(f"`{c}`" for c in OUTSIDE_SETS)
        + ", `empty_profile`.",
        "",
    ]
    return lines


def main() -> None:
    labels = load_labels()
    frame = build(labels)

    derived_columns = PROFILE + INTERACTION + INTERACTION_MIN + ["F_high"]
    shares = empty_profile_share(frame)
    stats = describe(frame, derived_columns + ["empty_profile"])
    corr = frame[derived_columns].corr(method="spearman")

    strong, moderate = [], []
    names = list(corr.columns)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            value = float(corr.loc[a, b])
            if np.isnan(value):
                continue
            if abs(value) > 0.95:
                strong.append((a, b, value))
            elif abs(value) > 0.80:
                moderate.append((a, b, value))
    strong.sort(key=lambda t: -abs(t[2]))
    moderate.sort(key=lambda t: -abs(t[2]))

    modal = modal_share(frame, derived_columns)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUT_PATH, index=False)

    denom = degenerate_denominators(frame)
    lines = report_lines(frame, shares, stats, corr, denom, strong, moderate,
                         modal, derived_columns)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"\nознаки: {OUT_PATH.relative_to(PROJECT_ROOT)} — {len(frame)} рядків, "
          f"{len(frame.columns)} колонок")
    print(f"звіт: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
