"""Рисунки EXP-002, частина друга: стабільність анотатора між прогонами.

Запуск:
    .venv/bin/python -m src.figures_stability

Стиль і збереження — `src.figures.save`: PNG 300 dpi і PDF, підписи
українською, заголовок усередині рисунка не ставиться.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.figures import COLOR, COLOR_ACCENT, save  # noqa: E402

EXP_DIR = Path("experiments/EXP-002_annotation-agreement")
STABILITY = EXP_DIR / "stability.json"

SHORT_LABELS = {
    "gap": "прогалина",
    "scale": "масштаб",
    "concrete": "конкретність",
    "conflict": "конфлікт",
    "authority": "авторитет",
    "violation": "порушення\nочікування",
    "stakes": "проспективна\nневизначеність",
}


def plot_stability_vs_agreement(data: dict, stem: str) -> list[Path]:
    """Каппа стабільності проти каппи узгодженості з людиною.

    Порівняння відповідає гіпотезі 1а: розбіжність між прогонами моделі
    містить лише власну нестабільність моделі, розбіжність з людиною —
    ще й різне розуміння визначень, тому стабільність очікувалася вищою.
    """
    features = list(SHORT_LABELS)
    check = data["перевірка_гіпотези"]["за_ознаками"]
    main = data["основний_розрахунок"]["features"]

    human = [check[k]["kappa_людина_частина1"] for k in features]
    pooled = [main[k]["kappa_linear"] for k in features]
    low = [main[k]["kappa_linear_ci"]["ci_low"] for k in features]
    high = [main[k]["kappa_linear_ci"]["ci_high"] for k in features]
    weighted = [check[k]["kappa_стабільності_зважена"] for k in features]

    x = np.arange(len(features))
    fig, ax = plt.subplots(figsize=(9, 4.4))

    ax.axhspan(0, 0.40, color="#000000", alpha=0.05)
    ax.axhline(0.40, color="#666666", lw=0.9, ls="--")
    ax.axhline(0.60, color="#666666", lw=0.9, ls=":")

    ax.errorbar(
        x - 0.12,
        pooled,
        yerr=[np.array(pooled) - np.array(low), np.array(high) - np.array(pooled)],
        fmt="o",
        color=COLOR,
        capsize=3,
        lw=1.2,
        markersize=6,
        label="стабільність run1–run2, пулована, 95% ДІ",
    )
    ax.plot(
        x - 0.12,
        weighted,
        marker="_",
        ls="none",
        color=COLOR,
        markersize=14,
        markeredgewidth=2,
        label="стабільність, зважена за часткою режимів у черзі",
    )
    ax.plot(
        x + 0.12,
        human,
        marker="s",
        ls="none",
        color=COLOR_ACCENT,
        markersize=6,
        label="узгодженість з людиною, частина перша",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_LABELS[k] for k in features], fontsize=9)
    ax.set_ylabel("зважена каппа Коена, лінійні ваги")
    ax.set_ylim(0, 1)
    ax.text(len(features) - 0.45, 0.405, "поріг 0,40", fontsize=8, color="#666666")
    ax.text(len(features) - 0.45, 0.605, "поріг 0,60", fontsize=8, color="#666666")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    return save(fig, stem, EXP_DIR / "figures")


def plot_strata(data: dict, stem: str) -> list[Path]:
    """Каппа стабільності по стратах розміру пачки першого прогону.

    Питання страти зафіксоване протоколом: чи погіршило збільшення пачки
    з 25 до 200 рядків якість розмітки.
    """
    features = list(SHORT_LABELS)
    main = data["основний_розрахунок"]["features"]
    regimes = ["25", "100", "200"]
    colors = [COLOR, "#8C8C8C", COLOR_ACCENT]

    x = np.arange(len(features))
    width = 0.26
    fig, ax = plt.subplots(figsize=(9, 4.4))

    for i, (regime, color) in enumerate(zip(regimes, colors)):
        values = [main[k]["страти"][regime]["kappa_linear"] for k in features]
        ax.bar(
            x + (i - 1) * width,
            values,
            width,
            color=color,
            label=f"пачка {regime} рядків",
        )

    ax.axhline(0.40, color="#666666", lw=0.9, ls="--")
    ax.axhline(0.60, color="#666666", lw=0.9, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_LABELS[k] for k in features], fontsize=9)
    ax.set_ylabel("зважена каппа Коена, лінійні ваги")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    return save(fig, stem, EXP_DIR / "figures")


def main() -> None:
    data = json.load(open(STABILITY, encoding="utf-8"))
    for paths in (
        plot_stability_vs_agreement(data, "fig07_exp002_stability"),
        plot_strata(data, "fig08_exp002_stability_strata"),
    ):
        for path in paths:
            print("Записано:", path)


if __name__ == "__main__":
    main()
