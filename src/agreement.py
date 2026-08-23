"""Узгодженість двох розміток: зважена каппа Коена та матриці розбіжностей.

Запуск:
    .venv/bin/python -m src.agreement --a annotations/manual_sample_200_scored.csv \
        --b annotations/manual_sample_200_claude.csv --out experiments/EXP-002_.../

Шкали ознак порядкові, тому каппа зважена. Основними прийнято лінійні ваги,
рішення користувача від 7 серпня 2026 року: за скошених розподілів і короткої
шкали квадратичні ваги завищують згоду, бо майже не карають за розбіжність
на один рівень, а саме такі розбіжності тут переважають. Квадратичні
наводяться поруч, щоб звіт не був вибірковим.

Каппа чутлива до поширеності категорій. Тому поряд наводяться частка точних
збігів, частка розбіжностей у межах одного рівня та середній зсув, який
показує, чи один оцінювач систематично вищий за іншого.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = {
    "gap": ("1. Прогалина в інформації", 3),
    "scale": ("2. Масштаб наслідків", 3),
    "concrete": ("3. Конкретність предиката", 3),
    "conflict": ("4. Явність конфлікту", 3),
    "authority": ("5. Вага авторитету", 3),
    "audience": ("6. Специфічність адресної групи", 2),
    "violation": ("7. Порушення очікування", 3),
    "stakes": ("9. Проспективна невизначеність", 3),
}


def weighted_kappa(a: np.ndarray, b: np.ndarray, max_level: int, weights: str) -> float:
    """Зважена каппа Коена для порядкової шкали 0..max_level.

    Повертає NaN, якщо обидва оцінювачі використали лише одну категорію:
    у такому разі очікувана згода дорівнює спостережуваній і каппа
    не визначена.
    """
    levels = np.arange(max_level + 1)
    n = len(a)

    observed = np.zeros((len(levels), len(levels)))
    for x, y in zip(a, b):
        observed[x, y] += 1
    observed /= n

    hist_a = np.array([(a == level).mean() for level in levels])
    hist_b = np.array([(b == level).mean() for level in levels])
    expected = np.outer(hist_a, hist_b)

    diff = np.abs(levels[:, None] - levels[None, :]).astype(float)
    if weights == "quadratic":
        w = (diff / max_level) ** 2
    elif weights == "linear":
        w = diff / max_level
    else:
        raise ValueError(weights)

    denominator = float((w * expected).sum())
    if denominator == 0:
        return float("nan")
    return float(1 - (w * observed).sum() / denominator)


def bootstrap_kappa_ci(
    a: np.ndarray,
    b: np.ndarray,
    max_level: int,
    weights: str,
    n_iterations: int = 2000,
    random_state: int = 42,
) -> dict:
    """95% перцентильний інтервал для каппи бутстрепом по заголовках.

    Додано 7 серпня 2026 року, після основного розрахунку, на вимогу
    користувача. Точкові оцінки не перераховувалися і не змінилися;
    жодне рішення щодо ознак від інтервалу не залежить.

    Ітерації, у яких оцінювач використав єдину категорію, дають невизначену
    каппу і відкидаються; їхня кількість повертається окремим числом.
    """
    rng = np.random.default_rng(random_state)
    n = len(a)
    draws = np.empty(n_iterations)
    for i in range(n_iterations):
        idx = rng.integers(0, n, size=n)
        draws[i] = weighted_kappa(a[idx], b[idx], max_level, weights)

    finite = draws[np.isfinite(draws)]
    if finite.size == 0:
        return {"ci_low": float("nan"), "ci_high": float("nan"), "n_valid": 0}
    return {
        "ci_low": round(float(np.percentile(finite, 2.5)), 4),
        "ci_high": round(float(np.percentile(finite, 97.5)), 4),
        "n_valid": int(finite.size),
        "n_iterations": n_iterations,
    }


def confusion(a: np.ndarray, b: np.ndarray, max_level: int) -> list[list[int]]:
    matrix = np.zeros((max_level + 1, max_level + 1), dtype=int)
    for x, y in zip(a, b):
        matrix[x, y] += 1
    return matrix.tolist()


def analyse(frame_a: pd.DataFrame, frame_b: pd.DataFrame, titles: pd.Series) -> dict:
    rows = []
    disagreements = {}

    for key, (name, max_level) in FEATURES.items():
        a = frame_a[key].to_numpy()
        b = frame_b[key].to_numpy()
        delta = b - a

        rows.append(
            {
                "код": key,
                "ознака": name,
                "kappa_linear": round(weighted_kappa(a, b, max_level, "linear"), 4),
                "kappa_linear_ci": bootstrap_kappa_ci(a, b, max_level, "linear"),
                "kappa_quadratic": round(weighted_kappa(a, b, max_level, "quadratic"), 4),
                "kappa_quadratic_ci": bootstrap_kappa_ci(a, b, max_level, "quadratic"),
                "точний збіг": round(float((a == b).mean()), 4),
                "у межах 1 рівня": round(float((np.abs(delta) <= 1).mean()), 4),
                "середній зсув": round(float(delta.mean()), 4),
                "середнє A": round(float(a.mean()), 3),
                "середнє B": round(float(b.mean()), 3),
                "рівнів у A": int(len(np.unique(a))),
                "рівнів у B": int(len(np.unique(b))),
                "матриця": confusion(a, b, max_level),
            }
        )

        order = np.argsort(-np.abs(delta))
        worst = [
            {
                "IDLink": int(frame_a["IDLink"].iloc[i]),
                "Title": str(titles.iloc[i]),
                "A": int(a[i]),
                "B": int(b[i]),
            }
            for i in order[:12]
            if abs(delta[i]) >= 2
        ]
        disagreements[key] = worst

    return {"features": rows, "disagreements": disagreements}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, help="розмітка оцінювача A (людина)")
    parser.add_argument("--b", required=True, help="розмітка оцінювача B (модель)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--label-a", default="людина")
    parser.add_argument("--label-b", default="модель")
    args = parser.parse_args()

    frame_a = pd.read_csv(args.a)
    frame_b = pd.read_csv(args.b)
    frame_a["IDLink"] = frame_a["IDLink"].astype(int)
    frame_b["IDLink"] = frame_b["IDLink"].astype(int)

    if not (frame_a["IDLink"].values == frame_b["IDLink"].values).all():
        frame_b = frame_b.set_index("IDLink").loc[frame_a["IDLink"]].reset_index()

    titles = frame_a["Title"] if "Title" in frame_a else frame_b["Title"]

    result = analyse(frame_a, frame_b, titles)
    result["n"] = int(len(frame_a))
    result["file_a"] = args.a
    result["file_b"] = args.b
    result["label_a"] = args.label_a
    result["label_b"] = args.label_b

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "agreement.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"n = {result['n']}, A = {args.label_a}, B = {args.label_b}\n")
    header = (
        f"{'ознака':32s} {'κ лін (осн.)':>22s} {'κ квадр':>22s} {'зсув':>7s}"
    )
    print(header)
    print("-" * len(header))
    for row in result["features"]:
        cl, cq = row["kappa_linear_ci"], row["kappa_quadratic_ci"]
        lin = f"{row['kappa_linear']:.3f} [{cl['ci_low']:.3f}; {cl['ci_high']:.3f}]"
        quad = f"{row['kappa_quadratic']:.3f} [{cq['ci_low']:.3f}; {cq['ci_high']:.3f}]"
        print(f"{row['ознака']:32s} {lin:>22s} {quad:>22s} {row['середній зсув']:+7.3f}")
    print("-" * len(header))
    print("Записано:", out_dir / "agreement.json")


if __name__ == "__main__":
    main()
