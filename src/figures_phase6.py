"""Рисунки EXP-006: важливість копірайтингових ознак по платформах.

Запуск:
    .venv/bin/python -m src.figures_phase6
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.features.copywriting import FEATURE_ORDER  # noqa: E402
from src.figures import COLOR, COLOR_ACCENT, save  # noqa: E402

EXP = Path("experiments/EXP-006_platform-differences")
PLATFORMS = ["Facebook", "GooglePlus", "LinkedIn"]
LABELS = {"Facebook": "Facebook", "GooglePlus": "Google+", "LinkedIn": "LinkedIn"}
SHORT = {
    "gap": "прогалина",
    "scale": "масштаб",
    "concrete": "конкретність",
    "conflict": "конфлікт",
    "authority": "авторитет",
    "violation": "порушення\nочікування",
    "stakes": "проспективна\nневизначеність",
}


def plot_importance(data: dict, stem: str) -> list[Path]:
    """Важливість семи ознак T2 по платформах, з масштабом Source для порівняння.

    Пунктир показує сукупну важливість Source. Він потрібен у самому
    рисунку, а не в підписі: без нього стовпчики T2 читалися б як великі,
    хоча найбільший з них становить від 15% до 33% від Source.
    """
    # Спільна шкала X обов'язкова: без неї стовпчики Google+ виглядали б
    # такими самими, як у Facebook, хоча вони вдвічі менші.
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2), sharey=True, sharex=True)

    for ax, platform in zip(axes, PLATFORMS):
        importance = data["важливість"][platform]["T2"]
        values = [importance[name]["mean_abs_shap"] for name in FEATURE_ORDER]
        caveat = [importance[name]["застереження"] for name in FEATURE_ORDER]
        colours = [COLOR_ACCENT if c else COLOR for c in caveat]
        y = np.arange(len(FEATURE_ORDER))

        ax.barh(y, values, color=colours)
        source = data["важливість"][platform]["T1"]["T1:source"]
        ax.axvline(source, color="#444444", lw=1.0, ls="--")
        ax.annotate(
            f"Source {source:.2f}".replace(".", ","),
            xy=(source, -0.55),
            xytext=(-4, 0),
            textcoords="offset points",
            fontsize=8,
            color="#444444",
            ha="right",
            va="center",
        )
        ax.set_yticks(y)
        ax.set_yticklabels([SHORT[name] for name in FEATURE_ORDER], fontsize=8.5)
        ax.set_xlabel(LABELS[platform], fontsize=9.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ","))
        )

    axes[0].set_ylim(len(FEATURE_ORDER) - 0.4, -1.0)
    axes[0].set_ylabel("середнє |SHAP| на тестовій вибірці")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLOR),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_ACCENT),
    ]
    fig.legend(
        handles,
        ["надійність без застережень", "потребує застереження (EXP-002)"],
        frameon=False,
        fontsize=8.5,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.06),
    )
    return save(fig, stem, EXP / "figures")


def plot_tau(data: dict, stem: str) -> list[Path]:
    """Кендалл τ між платформами з довірчими інтервалами і порогами."""
    pairs = list(data["kendall_tau"])
    values = [data["kendall_tau"][p]["tau"] for p in pairs]
    low = [data["kendall_tau"][p]["ci_low"] for p in pairs]
    high = [data["kendall_tau"][p]["ci_high"] for p in pairs]
    labels = [
        " — ".join(LABELS[part] for part in pair.split("|")) for pair in pairs
    ]

    fig, ax = plt.subplots(figsize=(7, 3.4))
    x = np.arange(len(pairs))
    ax.errorbar(
        x,
        values,
        yerr=[np.array(values) - np.array(low), np.array(high) - np.array(values)],
        fmt="o",
        color=COLOR,
        capsize=4,
        lw=1.4,
        markersize=7,
    )
    ax.axhline(data["пороги"]["tau_подібність"], color="#666666", lw=0.9, ls="--")
    ax.axhline(data["пороги"]["tau_часткова_подібність"], color="#666666", lw=0.9, ls=":")
    ax.text(len(pairs) - 0.45, 0.71, "поріг 0,70", fontsize=8, color="#666666")
    ax.text(len(pairs) - 0.45, 0.41, "поріг 0,40", fontsize=8, color="#666666")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Кендалл τ")
    ax.set_ylim(-0.1, 1.0)
    ax.spines[["top", "right"]].set_visible(False)
    return save(fig, stem, EXP / "figures")


def main() -> None:
    data = json.loads((EXP / "results.json").read_text(encoding="utf-8"))
    for paths in (
        plot_importance(data, "fig11_exp006_importance"),
        plot_tau(data, "fig12_exp006_tau"),
    ):
        for path in paths:
            print("Записано:", path)


if __name__ == "__main__":
    main()
