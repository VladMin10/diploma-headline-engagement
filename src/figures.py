"""Рисунки для статті: єдиний стиль, PNG 300 dpi і PDF.

Підписи українською, заголовок усередині рисунка не ставиться —
він дається в тексті статті (CLAUDE.md розділ 6, пункт 8).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES = PROJECT_ROOT / "reports" / "figures"

PLATFORM_LABELS = {
    "Facebook": "Facebook",
    "GooglePlus": "Google+",
    "LinkedIn": "LinkedIn",
}
COLOR = "#4C72B0"
COLOR_ACCENT = "#C44E52"


def save(fig, stem: str, out_dir: Path = FIGURES) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def plot_cv_vs_test(results_path: Path, stem: str, out_dir: Path = FIGURES) -> list[Path]:
    """Spearman ρ по фолдах крос-валідації і на відкладеному тесті.

    Рисунок пояснює, чому середнє по фолдах не є коректною точкою
    порівняння з тестом: у схемі з розширенням вікна перші фолди
    навчаються на малій частині train, тож середнє занижене конструктивно.
    """
    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)

    platforms = [p for p in PLATFORM_LABELS if p in data["results"]]
    fig, axes = plt.subplots(1, len(platforms), figsize=(11, 3.4), sharey=True)
    if len(platforms) == 1:
        axes = [axes]

    for ax, platform in zip(axes, platforms):
        variant = data["results"][platform]["T1"]
        folds = variant["cv"]["cv_spearman_per_fold"]
        test = variant["test"]["ridge"]["spearman"]
        x = np.arange(1, len(folds) + 1)

        ax.plot(x, folds, marker="o", color=COLOR, linewidth=1.4, label="фолди CV")
        ax.axhline(test["value"], color=COLOR_ACCENT, linestyle="--", linewidth=1.4,
                   label="відкладений тест")
        ax.fill_between(
            [x[0] - 0.3, x[-1] + 0.3],
            test["ci_low"],
            test["ci_high"],
            color=COLOR_ACCENT,
            alpha=0.15,
            linewidth=0,
        )
        ax.set_xticks(x)
        ax.set_xlabel(f"{PLATFORM_LABELS[platform]}\nномер фолда")
        ax.set_xlim(x[0] - 0.3, x[-1] + 0.3)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Spearman ρ")
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    return save(fig, stem, out_dir)


def plot_baseline_metrics(results_path: Path, stem: str, out_dir: Path = FIGURES) -> list[Path]:
    """Три метрики базової лінії по платформах з довірчими інтервалами."""
    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)

    platforms = [p for p in PLATFORM_LABELS if p in data["results"]]
    metrics = [
        ("spearman", "Spearman ρ", ("ridge",)),
        ("auc_nonzero", "ROC AUC для «популярність > 0»", ("ridge",)),
        ("mae_log", "MAE у логарифмічній шкалі", ("median", "ridge")),
    ]
    model_labels = {"median": "константа", "ridge": "Ridge"}
    model_colors = {"median": "#B0B7C3", "ridge": COLOR}

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, (metric, label, models) in zip(axes, metrics):
        width = 0.8 / len(models)
        for j, model in enumerate(models):
            xs, values, lows, highs = [], [], [], []
            for i, platform in enumerate(platforms):
                m = data["results"][platform]["T1"]["test"][model][metric]
                if m["value"] != m["value"]:
                    continue
                xs.append(i + (j - (len(models) - 1) / 2) * width)
                values.append(m["value"])
                lows.append(m["value"] - m["ci_low"])
                highs.append(m["ci_high"] - m["value"])
            ax.bar(xs, values, width=width * 0.9, color=model_colors[model],
                   label=model_labels[model])
            ax.errorbar(xs, values, yerr=[lows, highs], fmt="none",
                        ecolor="#333333", elinewidth=1, capsize=3)
        ax.set_xticks(range(len(platforms)))
        ax.set_xticklabels([PLATFORM_LABELS[p] for p in platforms])
        ax.set_xlabel(label)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        ax.set_axisbelow(True)
        if len(models) > 1:
            ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return save(fig, stem, out_dir)


def main() -> None:
    results = Path("experiments/EXP-001_baseline-metadata/results.json")
    for paths in (
        plot_cv_vs_test(results, "fig03_exp001_cv_vs_test"),
        plot_baseline_metrics(results, "fig04_exp001_baseline_metrics"),
    ):
        for path in paths:
            print("Записано:", path)


if __name__ == "__main__":
    main()
