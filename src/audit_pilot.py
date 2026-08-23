"""Частоти копірайтингових ознак на пілотній вибірці.

Запуск:  .venv/bin/python -m src.audit_pilot

Порогове правило зафіксоване в кодбуку до розмітки: ознака, що має
рівень 2 або 3 менш ніж у 3% заголовків, непридатна для моделювання;
її визначення переробляється або ознака замінюється.

Результат: data/interim/pilot_frequency.json
           reports/tables/pilot_frequency.csv і .md
           reports/figures/fig05_pilot_levels.{png,pdf}
"""

from __future__ import annotations

import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data.load import PROJECT_ROOT  # noqa: E402
from src.figures import save  # noqa: E402

LABELS_PATH = PROJECT_ROOT / "annotations" / "pilot_labels.csv"
SAMPLE_PATH = PROJECT_ROOT / "annotations" / "pilot_sample_400.csv"
OUT_JSON = PROJECT_ROOT / "data" / "interim" / "pilot_frequency.json"
TABLES = PROJECT_ROOT / "reports" / "tables"
FIGURES = PROJECT_ROOT / "reports" / "figures"

MIN_SHARE_HIGH = 0.03  # поріг з кодбуку: рівень 2 або 3

FEATURES = {
    "gap": "1. Прогалина в інформації",
    "scale": "2. Масштаб наслідків",
    "concrete": "3. Конкретність предиката",
    "conflict": "4. Явність конфлікту",
    "authority": "5. Вага авторитету",
    "audience": "6. Специфічність адресної групи",
    "violation": "7. Порушення очікування",
    "personal": "8. Персоналізація через людину",
    "stakes": "9. Проспективна невизначеність",
}


def main() -> None:
    labels = pd.read_csv(LABELS_PATH)
    n = len(labels)

    rows = []
    for key, name in FEATURES.items():
        counts = labels[key].value_counts().reindex([0, 1, 2, 3], fill_value=0)
        shares = counts / n
        high = float(shares[2] + shares[3])
        rows.append(
            {
                "ознака": name,
                "код": key,
                "рівень 0": int(counts[0]),
                "рівень 1": int(counts[1]),
                "рівень 2": int(counts[2]),
                "рівень 3": int(counts[3]),
                "частка 0": round(float(shares[0]), 4),
                "частка 1": round(float(shares[1]), 4),
                "частка 2": round(float(shares[2]), 4),
                "частка 3": round(float(shares[3]), 4),
                "частка 2 або 3": round(high, 4),
                "придатна": bool(high >= MIN_SHARE_HIGH),
                "середнє": round(float(labels[key].mean()), 3),
                "ст. відхилення": round(float(labels[key].std()), 3),
                "рівнів використано": int((counts > 0).sum()),
            }
        )
    table = pd.DataFrame(rows)

    TABLES.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES / "pilot_frequency.csv", index=False)

    # Кореляції між ознаками: показують, чи не вимірюють дві ознаки те саме.
    corr = labels[list(FEATURES)].corr(method="spearman").round(3)
    corr.to_csv(TABLES / "pilot_feature_correlations.csv")

    pairs = []
    keys = list(FEATURES)
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            pairs.append({"a": a, "b": b, "rho": float(corr.loc[a, b])})
    pairs.sort(key=lambda p: -abs(p["rho"]))

    payload = {
        "generated_by": "src/audit_pilot.py",
        "n_annotated": n,
        "sample": "annotations/pilot_sample_400.csv",
        "labels": "annotations/pilot_labels.csv",
        "codebook": "annotations/codebook_draft.md",
        "seed": 42,
        "min_share_high_threshold": MIN_SHARE_HIGH,
        "features": rows,
        "top_correlations": pairs[:10],
        "failed": [r["код"] for r in rows if not r["придатна"]],
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # Markdown-таблиця для тексту роботи.
    lines = [
        "# Частоти копірайтингових ознак на пілотній вибірці",
        "",
        f"Вибірка: {n} заголовків, випадкова з train, seed 42, обірвані виключено.",
        f"Поріг придатності: рівень 2 або 3 щонайменше в {MIN_SHARE_HIGH:.0%} заголовків.",
        "",
        "| Ознака | Рів. 0 | Рів. 1 | Рів. 2 | Рів. 3 | Рів. 2–3 | Середнє | Придатна |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['ознака']} | {r['частка 0']*100:.1f}% | {r['частка 1']*100:.1f}% | "
            f"{r['частка 2']*100:.1f}% | {r['частка 3']*100:.1f}% | "
            f"**{r['частка 2 або 3']*100:.1f}%** | {r['середнє']:.2f} | "
            f"{'так' if r['придатна'] else '**НІ**'} |"
        )
    (TABLES / "pilot_frequency.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Рисунок: розподіл рівнів по ознаках.
    fig, ax = plt.subplots(figsize=(10, 4.2))
    colors = ["#D9DDE3", "#9FB2C9", "#4C72B0", "#C44E52"]
    bottom = np.zeros(len(FEATURES))
    x = np.arange(len(FEATURES))
    for level in range(4):
        values = np.array([r[f"частка {level}"] * 100 for r in rows])
        ax.bar(x, values, bottom=bottom, color=colors[level], label=f"рівень {level}")
        bottom += values
    ax.axhline(100 - MIN_SHARE_HIGH * 100, color="black", linestyle=":", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([r["ознака"].split(". ")[1] for r in rows], rotation=30, ha="right")
    ax.set_ylabel("Частка заголовків, %")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=9, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.01))
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "fig05_pilot_levels", FIGURES)

    print(f"Пілотна вибірка: {n} заголовків\n")
    print(f"{'ознака':32s} {'0':>7s} {'1':>7s} {'2':>7s} {'3':>7s} {'2–3':>8s}  придатна")
    print("-" * 78)
    for r in rows:
        print(
            f"{r['ознака']:32s} {r['частка 0']*100:6.1f}% {r['частка 1']*100:6.1f}% "
            f"{r['частка 2']*100:6.1f}% {r['частка 3']*100:6.1f}% "
            f"{r['частка 2 або 3']*100:7.1f}%  {'так' if r['придатна'] else 'НІ'}"
        )
    print("-" * 78)
    if payload["failed"]:
        print("Не пройшли поріг:", ", ".join(payload["failed"]))
    else:
        print("Усі ознаки пройшли поріг.")
    print("\nНайсильніші кореляції між ознаками:")
    for p in pairs[:5]:
        print(f"  {p['a']:10s} — {p['b']:10s} ρ = {p['rho']:+.3f}")
    print("\nЗаписано:", OUT_JSON.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
