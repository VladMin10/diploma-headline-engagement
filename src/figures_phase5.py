"""Рисунки Фази 5: приріст від копірайтингових ознак (Q1 і Q2).

Запуск:
    .venv/bin/python -m src.figures_phase5

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

EXP = {
    "T1": Path("experiments/EXP-003_baseline-annotated"),
    "T1_T2": Path("experiments/EXP-004_copywriting-gain"),
    "T1_T3": Path("experiments/EXP-005b_text-control"),
    "T1_T2_T3": Path("experiments/EXP-005_full-stack"),
}
OUT_DIR = EXP["T1_T2"] / "figures"
PLATFORMS = ["Facebook", "GooglePlus", "LinkedIn"]
LABELS = {"Facebook": "Facebook", "GooglePlus": "Google+", "LinkedIn": "LinkedIn"}
CONFIG_LABELS = {
    "T1": "T1\nметадані",
    "T1_T2": "T1+T2\n+ копірайтинг",
    "T1_T3": "T1+T3\n+ статистики",
    "T1_T2_T3": "T1+T2+T3\nповний набір",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def plot_deltas(stem: str) -> list[Path]:
    """Δρ по платформах для Q1 і Q2, обидві моделі, із 95% ДІ.

    Нуль показано лінією: правило рішення, зафіксоване до запуску,
    зводиться до того, чи перетинає інтервал цю лінію.
    """
    panels = [
        ("Q1: T1+T2 проти T1", EXP["T1_T2"], "delta"),
        ("Q2: T1+T2+T3 проти T1+T3", EXP["T1_T2_T3"], "delta"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)

    for ax, (title, folder, prefix) in zip(axes, panels):
        data = {
            model: load(folder / f"{prefix}_{model}.json")["за_платформами"]
            for model in ("lightgbm", "ridge")
        }
        x = np.arange(len(PLATFORMS))
        for offset, (model, colour, marker) in enumerate(
            [("lightgbm", COLOR, "o"), ("ridge", COLOR_ACCENT, "s")]
        ):
            values = [data[model][p]["delta_rho"] for p in PLATFORMS]
            low = [data[model][p]["ci_low"] for p in PLATFORMS]
            high = [data[model][p]["ci_high"] for p in PLATFORMS]
            ax.errorbar(
                x + (offset - 0.5) * 0.22,
                values,
                yerr=[
                    np.array(values) - np.array(low),
                    np.array(high) - np.array(values),
                ],
                fmt=marker,
                color=colour,
                capsize=3,
                lw=1.2,
                markersize=6,
                label="LightGBM (основна)" if model == "lightgbm" else "Ridge",
            )
        ax.axhline(0, color="#444444", lw=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[p] for p in PLATFORMS], fontsize=9)
        ax.set_xlabel(title, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("приріст Spearman ρ, Δρ")
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper right")
    return save(fig, stem, OUT_DIR)


def plot_configurations(stem: str) -> list[Path]:
    """Spearman ρ по чотирьох конфігураціях ознак, LightGBM.

    Рисунок показує те, чого не видно з самих Δρ: рівень T3 знижує
    якість прогнозу відносно T1, тому базова лінія Q2 слабша за базову
    лінію Q1.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    configs = list(CONFIG_LABELS)

    for ax, (metric, ylabel) in zip(
        axes,
        [("test", "Spearman ρ на тесті"), ("test_within_source", "внутрішньоджерельна ρ")],
    ):
        x = np.arange(len(configs))
        width = 0.26
        for i, platform in enumerate(PLATFORMS):
            values, lows, highs = [], [], []
            for config in configs:
                results = load(EXP[config] / "results.json")["results"][platform][config]
                entry = results[metric]["lightgbm"]
                entry = entry["spearman"] if metric == "test" else entry
                values.append(entry["value"])
                lows.append(entry["ci_low"])
                highs.append(entry["ci_high"])
            ax.bar(
                x + (i - 1) * width,
                values,
                width,
                color=[COLOR, "#8C8C8C", COLOR_ACCENT][i],
                label=LABELS[platform],
                yerr=[
                    np.array(values) - np.array(lows),
                    np.array(highs) - np.array(values),
                ],
                capsize=2,
                error_kw={"lw": 0.8},
            )
        ax.set_xticks(x)
        ax.set_xticklabels([CONFIG_LABELS[c] for c in configs], fontsize=8)
        ax.set_ylabel(ylabel)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].legend(frameon=False, fontsize=8.5, ncol=3, loc="upper left")
    return save(fig, stem, OUT_DIR)


def main() -> None:
    for paths in (
        plot_deltas("fig09_phase5_delta_rho"),
        plot_configurations("fig10_phase5_configurations"),
    ):
        for path in paths:
            print("Записано:", path)


if __name__ == "__main__":
    main()
