"""Рисунки класифікаційної лінії для презентації.

Запуск:
    .venv/bin/python -m src.report.figures_classification

Нічого не навчається: усе будується з уже збережених
`descriptive_rates.parquet`, `report_predictions.parquet`
і `ablation_results.parquet`.

Формат за розділом 6 CLAUDE.md, пункт 8: PNG 300 dpi і PDF, підписи
українською, заголовок усередині рисунка не ставиться — він дається в тексті.
Шрифт великий, рамки зняті: рисунки йдуть у презентацію, де їх дивляться
з відстані.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RATES_PATH = PROJECT_ROOT / "data" / "processed" / "descriptive_rates.parquet"
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "report_predictions.parquet"
LIFT_PATH = PROJECT_ROOT / "data" / "processed" / "practical_lift.parquet"
ABLATION_PATH = PROJECT_ROOT / "data" / "processed" / "ablation_results.parquet"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

PLATFORM_ORDER = ["Facebook", "GooglePlus", "LinkedIn"]
PLATFORM_TITLE = {"Facebook": "Facebook", "GooglePlus": "Google+", "LinkedIn": "LinkedIn"}
SHORT = {"Facebook": "fb", "GooglePlus": "gp", "LinkedIn": "li"}
PERCENTILE = "25"

# Палітра: один насичений колір на серію, сірий для базових рівнів.
BLUE, ORANGE, GREEN, PURPLE = "#2F6FB5", "#D1741F", "#3E8A5A", "#7B4FA3"
GREY = "#8A8A8A"

plt.rcParams.update(
    {
        "font.size": 15,
        "axes.titlesize": 17,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "-",
        "figure.autolayout": False,
    }
)


def save(fig: plt.Figure, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES_DIR / f"{name}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}.png і {name}.pdf")


def figure_levels(rates: pd.DataFrame) -> list[str]:
    """4.1. Частка потрапляння у верхню чверть за рівнями трьох найсильніших ознак."""
    # Найсильніші три ознаки за різницею між найвищим рівнем і нулем,
    # усередненою по трьох платформах. Відбір рахується тут, а не задається.
    strength = []
    for feature, block in rates[rates["n"] > 0].groupby("ознака"):
        diffs = []
        for platform in PLATFORM_ORDER:
            own = block[block["платформа"] == platform].sort_values("рівень")
            diffs.append(float(own.iloc[-1]["частка"] - own.iloc[0]["частка"]))
        strength.append((feature, float(np.mean(diffs)), block["мітка"].iloc[0]))
    strength.sort(key=lambda item: -item[1])
    chosen = strength[:3]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharey=True)
    colours = [BLUE, ORANGE, GREEN]
    width = 0.26
    for axis, platform in zip(axes, PLATFORM_ORDER):
        block = rates[(rates["платформа"] == platform) & (rates["n"] > 0)]
        base = block["n_позитивних"].sum() / block["n"].sum()
        for index, (feature, _, label) in enumerate(chosen):
            own = block[block["ознака"] == feature].sort_values("рівень")
            positions = own["рівень"].to_numpy() + (index - 1) * width
            heights = own["частка"].to_numpy()
            errors = np.vstack(
                [
                    heights - own["ДІ_низ"].to_numpy(),
                    own["ДІ_верх"].to_numpy() - heights,
                ]
            )
            axis.bar(
                positions,
                heights,
                width=width,
                color=colours[index],
                label=label if platform == PLATFORM_ORDER[0] else None,
            )
            axis.errorbar(
                positions, heights, yerr=errors, fmt="none",
                ecolor="#333333", elinewidth=1.2, capsize=3,
            )
        axis.axhline(base, color=GREY, linestyle="--", linewidth=1.6)
        axis.set_title(PLATFORM_TITLE[platform])
        axis.set_xlabel("рівень оцінки")
        axis.set_xticks([0, 1, 2, 3])
    axes[0].set_ylabel("частка у верхній чверті")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=GREY, linestyle="--", linewidth=1.6))
    labels.append("базовий рівень платформи")
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.09))
    save(fig, "fig13_rates_by_level")
    return [label for _, _, label in chosen]


def figure_lift(lift: pd.DataFrame) -> None:
    """4.2. Влучність на верхніх k відсотках проти базового рівня."""
    block = lift[lift["варіант"] == "LogisticRegression_F0"]
    fig, axis = plt.subplots(figsize=(11, 5.6))
    shares = sorted(block["k"].unique())
    width = 0.22
    positions = np.arange(len(PLATFORM_ORDER))
    colours = [BLUE, ORANGE, GREEN]
    for index, share in enumerate(shares):
        heights = [
            float(block[(block["платформа"] == p) & (block["k"] == share)]["precision_топ"].iloc[0])
            for p in PLATFORM_ORDER
        ]
        axis.bar(
            positions + (index - 1) * width,
            heights,
            width=width,
            color=colours[index],
            label=f"верхні {int(share * 100)}%",
        )
    for index, platform in enumerate(PLATFORM_ORDER):
        base = float(block[block["платформа"] == platform]["базова_частка"].iloc[0])
        axis.hlines(base, index - 0.42, index + 0.42, color=GREY,
                    linestyle="--", linewidth=1.8)
    axis.set_xticks(positions)
    axis.set_xticklabels([PLATFORM_TITLE[p] for p in PLATFORM_ORDER])
    axis.set_ylabel("частка справді у верхній чверті")
    handles, labels = axis.get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=GREY, linestyle="--", linewidth=1.8))
    labels.append("базовий рівень")
    axis.legend(handles, labels, frameon=False, ncol=2)
    save(fig, "fig14_precision_at_k")


def figure_roc(predictions: pd.DataFrame) -> None:
    """4.3. ROC-криві чотирьох моделей, усереднені по двадцяти розбиттях."""
    variants = {
        "LogisticRegression_F0": ("Логістична регресія", BLUE),
        "RandomForest_F0": ("Випадковий ліс", ORANGE),
        "LightGBM_F0": ("LightGBM", GREEN),
        "CatBoost_F0": ("CatBoost", PURPLE),
    }
    grid = np.linspace(0, 1, 201)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), sharey=True)
    for axis, platform in zip(axes, PLATFORM_ORDER):
        for variant, (label, colour) in variants.items():
            block = predictions[
                (predictions["платформа"] == platform)
                & (predictions["варіант"] == variant)
            ]
            curves = []
            for _, split in block.groupby(["repeat", "fold"], observed=True):
                fpr, tpr, _ = roc_curve(split["y"], split["ймовірність"])
                curves.append(np.interp(grid, fpr, tpr))
            axis.plot(grid, np.mean(curves, axis=0), color=colour,
                      linewidth=2.2, label=label)
        axis.plot([0, 1], [0, 1], color=GREY, linestyle="--", linewidth=1.4)
        axis.set_title(PLATFORM_TITLE[platform])
        axis.set_xlabel("частка хибнопозитивних")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
    axes[0].set_ylabel("частка істиннопозитивних")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=GREY, linestyle="--", linewidth=1.4))
    labels.append("випадковий рівень")
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.07))
    save(fig, "fig15_roc_curves")


def figure_ablation(ablation: pd.DataFrame) -> None:
    """4.4. Середній F1 для чотирьох наборів ознак на Facebook."""
    block = ablation[
        (ablation["платформа"] == "Facebook") & (ablation["перцентиль"] == PERCENTILE)
    ]
    models = ["LogisticRegression", "RandomForest", "LightGBM", "CatBoost"]
    titles = ["Логістична регресія", "Випадковий ліс", "LightGBM", "CatBoost"]
    sets = ["F0", "F1", "F2", "F3"]
    labels = ["F₀", "F₁", "F₂", "F₃"]
    colours = [BLUE, ORANGE, GREEN, PURPLE]

    fig, axis = plt.subplots(figsize=(12, 5.8))
    width = 0.2
    positions = np.arange(len(models))
    for index, (set_name, label) in enumerate(zip(sets, labels)):
        heights, errors = [], []
        for model in models:
            values = block[
                (block["модель"] == model) & (block["набір"] == set_name)
            ]["F1_підібраний"].to_numpy()
            mean = float(values.mean())
            # 95% інтервал середнього по двадцяти розбиттях.
            half = 1.96 * float(values.std(ddof=1)) / np.sqrt(len(values))
            heights.append(mean)
            errors.append(half)
        axis.bar(positions + (index - 1.5) * width, heights, width=width,
                 color=colours[index], label=label)
        axis.errorbar(positions + (index - 1.5) * width, heights, yerr=errors,
                      fmt="none", ecolor="#333333", elinewidth=1.2, capsize=3)
    axis.set_xticks(positions)
    axis.set_xticklabels(titles)
    axis.set_ylabel("F1 при підібраному порозі")
    # Легенда виноситься під вісь: усередині вона перекривала стовпчики.
    axis.legend(
        frameon=False, ncol=4, title="набір ознак",
        loc="upper center", bbox_to_anchor=(0.5, -0.12),
    )
    save(fig, "fig16_ablation_f1")


def main() -> None:
    rates = pd.read_parquet(RATES_PATH)
    predictions = pd.read_parquet(PREDICTIONS_PATH)
    lift = pd.read_parquet(LIFT_PATH)
    ablation = pd.read_parquet(ABLATION_PATH)

    print("рисунки:")
    chosen = figure_levels(rates)
    figure_lift(lift)
    figure_roc(predictions)
    figure_ablation(ablation)
    print("три найсильніші ознаки на рисунку 13:", ", ".join(chosen))


if __name__ == "__main__":
    main()
