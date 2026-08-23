"""Зведені таблиці Фази 5 для статті.

Запуск:
    .venv/bin/python -m src.tables

Пише CSV і markdown у `reports/tables/`. Жодне число тут не набирається
руками: усі беруться з `results.json` і `delta_*.json` відповідних
експериментів. Це вимога розділу 2 CLAUDE.md — кожне число має походження.

Таблиці подають обидві метрики і обидві моделі. Рішення користувача
від 8 серпня 2026 року: загальна ρ значною мірою вимірює рівень видання,
внутрішньоджерельна відповідає практичному питанню роботи, і жодну
з них не можна подавати замість другої. LightGBM позначається як основна,
бо була зафіксована такою до прогону; Ridge наводиться поруч.
"""

from __future__ import annotations

import csv
from pathlib import Path

import json

EXP = {
    "T1": Path("experiments/EXP-003_baseline-annotated"),
    "T1+T2": Path("experiments/EXP-004_copywriting-gain"),
    "T1+T3": Path("experiments/EXP-005b_text-control"),
    "T1+T2+T3": Path("experiments/EXP-005_full-stack"),
}
VARIANT = {"T1": "T1", "T1+T2": "T1_T2", "T1+T3": "T1_T3", "T1+T2+T3": "T1_T2_T3"}
PLATFORMS = {"Facebook": "Facebook", "GooglePlus": "Google+", "LinkedIn": "LinkedIn"}
MODELS = {"lightgbm": "LightGBM (основна)", "ridge": "Ridge"}
OUT = Path("reports/tables")
EXP006 = Path("experiments/EXP-006_platform-differences")
FEATURE_ORDER = ["gap", "scale", "concrete", "conflict", "authority", "violation", "stakes"]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def uk(value: float, digits: int = 4, sign: bool = False) -> str:
    """Число з комою як десятковим роздільником (CLAUDE.md розділ 5)."""
    text = f"{value:+.{digits}f}" if sign else f"{value:.{digits}f}"
    return text.replace(".", ",").replace("-", "−")


def interval(low: float, high: float, sign: bool = False) -> str:
    return f"[{uk(low, sign=sign)}; {uk(high, sign=sign)}]"


def table_levels() -> list[dict]:
    """Spearman ρ по конфігураціях ознак: обидві метрики, обидві моделі."""
    rows = []
    for config, folder in EXP.items():
        results = load(folder / "results.json")["results"]
        for platform, label in PLATFORMS.items():
            entry = results[platform][VARIANT[config]]
            for model, model_label in MODELS.items():
                overall = entry["test"][model]["spearman"]
                within = entry["test_within_source"][model]
                rows.append(
                    {
                        "конфігурація": config,
                        "платформа": label,
                        "модель": model_label,
                        "rho_загальна": overall["value"],
                        "rho_загальна_ДІ_низ": overall["ci_low"],
                        "rho_загальна_ДІ_верх": overall["ci_high"],
                        "rho_внутрішньоджерельна": within["value"],
                        "rho_внутрішньоджерельна_ДІ_низ": within["ci_low"],
                        "rho_внутрішньоджерельна_ДІ_верх": within["ci_high"],
                    }
                )
    return rows


def table_deltas() -> list[dict]:
    """Прирости Δρ для Q1 і Q2: обидві метрики, обидві моделі."""
    questions = [
        ("Q1", "T1+T2 − T1", EXP["T1+T2"]),
        ("Q2", "T1+T2+T3 − (T1+T3)", EXP["T1+T2+T3"]),
    ]
    rows = []
    for question, contrast, folder in questions:
        for model, model_label in MODELS.items():
            data = load(folder / f"delta_{model}.json")["за_платформами"]
            for platform, label in PLATFORMS.items():
                entry = data[platform]
                within = entry["within_source"]
                rows.append(
                    {
                        "питання": question,
                        "порівняння": contrast,
                        "платформа": label,
                        "модель": model_label,
                        "delta_загальна": entry["delta_rho"],
                        "delta_загальна_ДІ_низ": entry["ci_low"],
                        "delta_загальна_ДІ_верх": entry["ci_high"],
                        "відповідь_загальна": entry["відповідь"],
                        "delta_внутрішньоджерельна": within["delta_rho"],
                        "delta_внутрішньоджерельна_ДІ_низ": within["ci_low"],
                        "delta_внутрішньоджерельна_ДІ_верх": within["ci_high"],
                        "відповідь_внутрішньоджерельна": within["відповідь"],
                    }
                )
    return rows


def table_importance() -> list[dict]:
    """Важливість копірайтингових ознак по платформах (EXP-006)."""
    data = load(EXP006 / "results.json")["важливість"]
    rows = []
    for name in FEATURE_ORDER:
        row = {"ознака": data["Facebook"]["T2"][name]["мітка"]}
        for platform, label in PLATFORMS.items():
            entry = data[platform]["T2"][name]
            order = sorted(
                FEATURE_ORDER,
                key=lambda k: -data[platform]["T2"][k]["mean_abs_shap"],
            )
            row[f"{label}_важливість"] = entry["mean_abs_shap"]
            row[f"{label}_ранг"] = order.index(name) + 1
        entry = data["Facebook"]["T2"][name]
        row["kappa_людина"] = entry["kappa_людина"]
        row["kappa_стабільність"] = entry["kappa_стабільність"]
        row["потребує_застереження"] = "так" if entry["застереження"] else "ні"
        rows.append(row)
    return rows


def table_tau() -> list[dict]:
    """Кендалл τ між платформами і умова, за якої вона читається."""
    data = load(EXP006 / "results.json")
    rows = []
    for key, entry in data["kendall_tau"].items():
        a, b = key.split("|")
        rows.append(
            {
                "пара": f"{PLATFORMS[a]} — {PLATFORMS[b]}",
                "tau": entry["tau"],
                "ДІ_низ": entry["ci_low"],
                "ДІ_верх": entry["ci_high"],
                "висновок": entry["висновок"],
            }
        )
    return rows


def markdown_phase6(importance: list[dict], tau: list[dict]) -> str:
    lines = [
        "### Кендалл τ між упорядкуваннями ознак",
        "",
        "| Пара платформ | τ | 95% ДІ | Висновок за правилом |",
        "|---|---|---|---|",
    ]
    for row in tau:
        lines.append(
            f"| {row['пара']} | {uk(row['tau'], sign=True)} | "
            f"{interval(row['ДІ_низ'], row['ДІ_верх'], sign=True)} | {row['висновок']} |"
        )
    lines += [
        "",
        "### Важливість копірайтингових ознак, середнє |SHAP|",
        "",
        "| Ознака | Facebook | Google+ | LinkedIn | κ людина | κ стабільність | Застереження |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in importance:
        cells = " | ".join(
            f"{uk(row[f'{label}_важливість'])} ({row[f'{label}_ранг']})"
            for label in PLATFORMS.values()
        )
        lines.append(
            f"| {row['ознака']} | {cells} | {uk(row['kappa_людина'], 3)} | "
            f"{uk(row['kappa_стабільність'], 3)} | {row['потребує_застереження']} |"
        )
    lines.append("")
    lines.append("У дужках — ранг ознаки за важливістю всередині платформи.")
    return "\n".join(lines)


def write_csv(rows: list[dict], name: str) -> Path:
    path = OUT / f"{name}.csv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def markdown_deltas(rows: list[dict]) -> str:
    """Головна таблиця статті: приріст від T2 по обох метриках і моделях."""
    lines = [
        "### Приріст Spearman ρ від копірайтингових ознак",
        "",
        "| Питання | Платформа | Модель | Δρ загальна | 95% ДІ | Δρ внутрішньоджерельна | 95% ДІ |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['питання']} | {row['платформа']} | {row['модель']} | "
            f"{uk(row['delta_загальна'], sign=True)} | "
            f"{interval(row['delta_загальна_ДІ_низ'], row['delta_загальна_ДІ_верх'], sign=True)} | "
            f"{uk(row['delta_внутрішньоджерельна'], sign=True)} | "
            f"{interval(row['delta_внутрішньоджерельна_ДІ_низ'], row['delta_внутрішньоджерельна_ДІ_верх'], sign=True)} |"
        )
    lines += [
        "",
        "Q1 порівнює T1+T2 з T1, Q2 порівнює T1+T2+T3 з T1+T3. Відповідь «так» "
        "за зафіксованим до запуску правилом означає, що нижня межа інтервалу "
        "строго більша за нуль.",
    ]
    return "\n".join(lines)


def markdown_levels(rows: list[dict]) -> str:
    lines = [
        "### Spearman ρ на тесті по конфігураціях ознак",
        "",
        "| Конфігурація | Платформа | Модель | ρ загальна | 95% ДІ | ρ внутрішньоджерельна | 95% ДІ |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['конфігурація']} | {row['платформа']} | {row['модель']} | "
            f"{uk(row['rho_загальна'])} | "
            f"{interval(row['rho_загальна_ДІ_низ'], row['rho_загальна_ДІ_верх'])} | "
            f"{uk(row['rho_внутрішньоджерельна'])} | "
            f"{interval(row['rho_внутрішньоджерельна_ДІ_низ'], row['rho_внутрішньоджерельна_ДІ_верх'])} |"
        )
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    levels, deltas = table_levels(), table_deltas()
    importance, tau = table_importance(), table_tau()

    paths = [
        write_csv(levels, "phase5_levels"),
        write_csv(deltas, "phase5_deltas"),
        write_csv(importance, "phase6_importance"),
        write_csv(tau, "phase6_tau"),
    ]
    markdown = (
        "# Таблиці Фази 5\n\n"
        "Згенеровано `src/tables.py` з `results.json` і `delta_*.json`.\n"
        "Числа не редагуються вручну: правити треба джерело і перезапускати.\n\n"
        + markdown_deltas(deltas)
        + "\n\n---\n\n"
        + markdown_levels(levels)
        + "\n"
    )
    path = OUT / "phase5.md"
    path.write_text(markdown, encoding="utf-8")
    paths.append(path)

    path = OUT / "phase6.md"
    path.write_text(
        "# Таблиці Фази 6\n\nЗгенеровано `src/tables.py` з `results.json` "
        "експерименту EXP-006.\n\n" + markdown_phase6(importance, tau) + "\n",
        encoding="utf-8",
    )
    paths.append(path)

    for item in paths:
        print("Записано:", item)


if __name__ == "__main__":
    main()
