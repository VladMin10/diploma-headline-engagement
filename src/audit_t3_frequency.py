"""Частота регулярочних ознак рівня T3 на повному наборі.

Запуск:  .venv/bin/python -m src.audit_t3_frequency

Мета — відсіяти прийоми, надто рідкісні для моделювання. Поріг 2%
заголовків узгоджено з користувачем 7 серпня 2026 року: ознака, що
трапляється рідше, дає надто мало спостережень, щоб її вага в моделі
відрізнялася від шуму, і водночас витрачає ступінь свободи.

Рахується на колонці Title повного набору (93 235 рядків після вилучення
технічних дублікатів), а не на робочій вибірці: частота прийому є
властивістю мови заголовків, а не наслідком фільтра повних спостережень.
Додатково наводиться частота на train, щоб було видно, чи вистачить
спостережень для навчання.

Результат: data/interim/t3_frequency.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from src.data.load import PROJECT_ROOT, add_derived, load_raw
from src.data.split import compute_split, load_split

OUT_PATH = PROJECT_ROOT / "data" / "interim" / "t3_frequency.json"

# Поріг придатності, узгоджений до обчислення частот.
MIN_SHARE = 0.02

# Словник високозбудливих слів для ознаки, перенесеної користувачем
# з T2 до T3 (кодбук, ознака 6). Підстава конструкта — Berger і Milkman
# (2012): вирішальним є рівень збудження емоції, а не її валентність.
# Список охоплює чотири активуючі стани: гнів, тривогу, подив, захват.
# Він свідомо короткий і складається зі слів, що трапляються в новинному
# заголовку; повнота не заявляється, і це обмеження підлягає опису.
AROUSAL_WORDS = {
    # гнів і обурення
    "attack", "attacks", "slam", "slams", "slammed", "blast", "blasts",
    "fury", "furious", "outrage", "outraged", "rage", "backlash", "clash",
    "clashes", "war", "battle", "fight", "fights", "blame", "blames",
    "accuse", "accuses", "accused", "condemn", "condemns", "protest",
    "protests", "threat", "threats", "threaten", "threatens",
    # тривога і страх
    "fear", "fears", "panic", "crisis", "danger", "dangerous", "warn",
    "warns", "warning", "alert", "risk", "risks", "collapse", "crash",
    "crashes", "plunge", "plunges", "slump", "emergency", "urgent",
    "scandal", "shock", "shocked", "shocking",
    # подив і несподіванка
    "surprise", "surprises", "surprising", "stun", "stuns", "stunning",
    "unexpected", "sudden", "suddenly", "reveal", "reveals", "revealed",
    "secret", "secrets", "mystery",
    # захват і піднесення
    "amazing", "incredible", "extraordinary", "stunning", "soar", "soars",
    "surge", "surges", "boom", "record", "historic", "triumph", "victory",
    "breakthrough",
}

PATTERNS: dict[str, dict] = {
    "has_digit": {
        "opis": "будь-яка цифра в заголовку",
        "regex": r"\d",
    },
    "starts_with_number": {
        "opis": "заголовок починається з числівника (список на кшталт «5 things…»)",
        "regex": r"^\s*\d+\b",
    },
    "has_question_mark": {
        "opis": "знак питання",
        "regex": r"\?",
    },
    "has_exclamation_mark": {
        "opis": "знак оклику",
        "regex": r"!",
    },
    "has_colon": {
        "opis": "двокрапка — типовий роздільник «тема: уточнення»",
        "regex": r":",
    },
    "has_quotes": {
        "opis": "лапки — пряма цитата в заголовку",
        "regex": r"[\"'“”‘’]",
    },
    "has_allcaps_word": {
        "opis": "слово з двох і більше великих літер поспіль",
        "regex": r"\b[A-Z]{2,}\b",
    },
    "has_second_person": {
        "opis": "звертання до читача займенником другої особи",
        "regex": r"\b(you|your|yours|yourself)\b",
        "flags": re.IGNORECASE,
    },
    "has_superlative": {
        "opis": "найвищий ступінь порівняння",
        "regex": r"\b(best|worst|biggest|largest|smallest|greatest|top|most|least)\b",
        "flags": re.IGNORECASE,
    },
    "has_comparative": {
        "opis": "порівняльна конструкція",
        "regex": r"\b(better|worse|bigger|larger|faster|slower|stronger|weaker|than)\b",
        "flags": re.IGNORECASE,
    },
    "has_parentheses": {
        "opis": "дужки",
        "regex": r"[()\[\]]",
    },
    "has_dash": {
        "opis": "тире або дефіс як роздільник частин заголовка",
        "regex": r"\s[-–—]\s",
    },
    "has_modal_should": {
        "opis": "модальність поради або обов'язку",
        "regex": r"\b(should|must|need to|have to)\b",
        "flags": re.IGNORECASE,
    },
    "has_negation": {
        "opis": "заперечення",
        "regex": r"\b(not|no|never|nothing|without|isn't|don't|won't|can't)\b",
        "flags": re.IGNORECASE,
    },
    "has_here_is_why": {
        "opis": "відкладена причина: «here's why», «this is why»",
        "regex": r"\b(here'?s why|this is why|here'?s how|that'?s why)\b",
        "flags": re.IGNORECASE,
    },
    "has_arousal_word": {
        "opis": "слово зі словника високозбудливої лексики (кодбук, ознака 6)",
        "regex": r"\b(" + "|".join(sorted(AROUSAL_WORDS)) + r")\b",
        "flags": re.IGNORECASE,
    },
}


def compute_shares(titles: pd.Series) -> dict[str, dict]:
    out = {}
    for name, spec in PATTERNS.items():
        rx = re.compile(spec["regex"], spec.get("flags", 0))
        hits = titles.map(lambda t: bool(rx.search(t)))
        n_hits = int(hits.sum())
        out[name] = {
            "opis": spec["opis"],
            "n": n_hits,
            "share": round(float(n_hits / len(titles)), 4),
        }
    return out


def main() -> None:
    raw = load_raw()
    df = add_derived(raw)
    titles_full = df["Title"].fillna("").astype(str)

    split = compute_split(df, load_split()["boundary"]) if False else None  # noqa: F841
    # Спліт обчислюється на робочій вибірці, тому train для довідки
    # береться простим порівнянням дати з тією самою межею.
    boundary = pd.Timestamp(load_split()["boundary"])
    titles_train = df.loc[df["PublishDate"] < boundary, "Title"].fillna("").astype(str)

    full = compute_shares(titles_full)
    train = compute_shares(titles_train)

    print(f"Повний набір: {len(titles_full)} заголовків, train: {len(titles_train)}\n")
    print(f"{'ознака':26s} {'повний':>10s} {'train':>10s}  {'придатна':>9s}")
    print("-" * 62)
    kept, dropped = [], []
    for name in PATTERNS:
        share = full[name]["share"]
        ok = share >= MIN_SHARE
        (kept if ok else dropped).append(name)
        print(
            f"{name:26s} {share*100:9.2f}% {train[name]['share']*100:9.2f}%  "
            f"{'так' if ok else 'НІ':>9s}"
        )
    print("-" * 62)
    print(f"Придатних: {len(kept)}, відсіяно за порогом {MIN_SHARE:.0%}: {len(dropped)}")
    if dropped:
        print("Відсіяно:", ", ".join(dropped))

    payload = {
        "generated_by": "src/audit_t3_frequency.py",
        "n_titles_full": int(len(titles_full)),
        "n_titles_train": int(len(titles_train)),
        "min_share_threshold": MIN_SHARE,
        "arousal_dictionary_size": len(AROUSAL_WORDS),
        "arousal_dictionary": sorted(AROUSAL_WORDS),
        "full": full,
        "train": train,
        "kept": kept,
        "dropped": dropped,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("Записано:", OUT_PATH.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
