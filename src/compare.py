"""Приріст Δρ між двома конфігураціями ознак (питання Q1 і Q2).

Запуск:
    .venv/bin/python -m src.compare \
        --a experiments/EXP-004_.../results.json --variant-a T1_T2 \
        --b experiments/EXP-003_.../results.json --variant-b T1 \
        --out experiments/EXP-004_.../delta.json

Порівняння виконується парно: та сама бутстреп-вибірка застосовується
до обох наборів прогнозів. Інакше довірчий інтервал різниці був би
ширшим за реальний, бо включав би незалежну варіацію двох оцінок.

Тотожність рядків тесту перевіряється за хешем `test_ids_sha256`
з `results.json`. Якщо хеші різні, порівняння не виконується: різниця
міряла б ще й зміну вибірки. Це не формальність — рівень T2 існує лише
для анотованих новин, тому переплутати повну вибірку з анотованою легко.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.eval.bootstrap import (
    bootstrap_difference,
    bootstrap_indices,
    bootstrap_within_source_difference,
    stratified_bootstrap_indices,
)

PLATFORMS = ["Facebook", "GooglePlus", "LinkedIn"]


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_source_groups(results: dict) -> np.ndarray:
    """Мітки джерела для тестових рядків прогону.

    Відновлюються з даних, а не зберігаються в `results.json`. Коректність
    відновлення підтверджує звірка з хешем `test_ids_sha256`: якщо порядок
    чи склад рядків розійшовся б, хеш не збігся б.
    """
    import hashlib

    import pandas as pd

    from src.data.load import CLEAN_PATH
    from src.data.split import SPLIT_PATH, compute_split, load_split
    from src.features.assemble import select_rows

    frame = pd.read_parquet(CLEAN_PATH)
    frame["IDLink"] = frame["IDLink"].astype("int64")
    split = compute_split(frame, load_split(SPLIT_PATH)["boundary"])
    test = frame.loc[split.test_mask]
    if results["data"].get("restricted_to_annotated"):
        test = select_rows(test, ["T2"])

    digest = hashlib.sha256(
        ",".join(str(int(i)) for i in test["IDLink"]).encode()
    ).hexdigest()
    if digest != results["data"]["test_ids_sha256"]:
        raise ValueError(
            "відновлені рядки тесту не збігаються з тими, на яких виконано "
            f"{results['exp_id']}; внутрішньоджерельна різниця не рахується"
        )
    return test["Source"].fillna("__відсутнє__").astype(str).to_numpy()


def compare(
    results_a: dict,
    variant_a: str,
    results_b: dict,
    variant_b: str,
    model: str,
    n_iterations: int,
    random_state: int,
) -> dict:
    hash_a = results_a["data"]["test_ids_sha256"]
    hash_b = results_b["data"]["test_ids_sha256"]
    if hash_a != hash_b:
        raise ValueError(
            "рядки тесту не збігаються: "
            f"{results_a['exp_id']} має хеш {hash_a[:12]}, "
            f"{results_b['exp_id']} — {hash_b[:12]}. "
            "Парне порівняння Δρ на різних вибірках не має сенсу."
        )

    groups = test_source_groups(results_a)

    out = {}
    for platform in PLATFORMS:
        if platform not in results_a["results"] or platform not in results_b["results"]:
            continue
        a = results_a["results"][platform][variant_a]
        b = results_b["results"][platform][variant_b]
        for name, entry in (("A", a), ("B", b)):
            if "predictions" not in entry:
                raise ValueError(
                    f"у {name} немає збережених прогнозів: додайте "
                    '"save_predictions": true у config.json і перезапустіть'
                )
        pred_a = np.asarray(a["predictions"][model])
        pred_b = np.asarray(b["predictions"][model])
        y_true = np.asarray(results_a["results"][platform]["y_test"])
        y_other = np.asarray(results_b["results"][platform]["y_test"])
        if not np.array_equal(y_true, y_other):
            raise ValueError(
                f"{platform}: цільові вектори двох прогонів не збігаються "
                "попри однаковий хеш рядків — порівняння скасовано"
            )

        indices = bootstrap_indices(
            len(y_true), n_iterations=n_iterations, random_state=random_state
        )
        delta = bootstrap_difference(y_true, pred_a, pred_b, indices)
        rho = delta["spearman"]

        # Внутрішньоджерельна різниця. Метрика зафіксована як обов'язкова
        # 7 серпня 2026 року: загальна ρ значною мірою вимірює рівень
        # видання, а практичне питання роботи стоїть при фіксованому виданні.
        strat = stratified_bootstrap_indices(
            groups, n_iterations=n_iterations, random_state=random_state
        )
        within = bootstrap_within_source_difference(
            y_true, pred_a, pred_b, groups, strat
        )

        out[platform] = {
            "rho_A": a["test"][model]["spearman"]["value"],
            "rho_B": b["test"][model]["spearman"]["value"],
            "delta_rho": rho["difference"],
            "ci_low": rho["ci_low"],
            "ci_high": rho["ci_high"],
            # Правило рішення зафіксоване в docs/research_questions.md
            # до запуску: «так», якщо нижня межа строго більша за нуль.
            "відповідь": "так" if rho["ci_low"] > 0 else "ні",
            "within_source": {
                "rho_A": a["test_within_source"][model]["value"],
                "rho_B": b["test_within_source"][model]["value"],
                "delta_rho": within["difference"],
                "ci_low": within["ci_low"],
                "ci_high": within["ci_high"],
                "відповідь": "так" if within["ci_low"] > 0 else "ні",
            },
            "повністю": delta,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, help="results.json конфігурації з ознаками")
    parser.add_argument("--variant-a", required=True)
    parser.add_argument("--b", required=True, help="results.json конфігурації без них")
    parser.add_argument("--variant-b", required=True)
    parser.add_argument("--model", default="lightgbm")
    parser.add_argument("--out", required=True)
    parser.add_argument("--question", default="", help="Q1 або Q2")
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    results_a, results_b = load(Path(args.a)), load(Path(args.b))
    per_platform = compare(
        results_a,
        args.variant_a,
        results_b,
        args.variant_b,
        args.model,
        args.bootstrap_iters,
        args.random_state,
    )

    out = {
        "питання": args.question,
        "модель": args.model,
        "A": {"exp_id": results_a["exp_id"], "варіант": args.variant_a},
        "B": {"exp_id": results_b["exp_id"], "варіант": args.variant_b},
        "n_test": results_a["data"]["n_test"],
        "test_ids_sha256": results_a["data"]["test_ids_sha256"],
        "bootstrap_iters": args.bootstrap_iters,
        "random_state": args.random_state,
        "правило_рішення": (
            "відповідь «так», якщо нижня межа 95% ДІ Δρ строго більша "
            "за нуль (docs/research_questions.md, зафіксовано 6 серпня 2026)"
        ),
        "за_платформами": per_platform,
    }
    Path(args.out).write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"{args.question}  {args.variant_a} − {args.variant_b}  ({args.model})\n")
    print("Загальна Spearman ρ")
    print(f"{'платформа':12s} {'ρ(A)':>8s} {'ρ(B)':>8s} {'Δρ':>9s} {'95% ДІ':>20s}  відповідь")
    for platform, row in per_platform.items():
        ci = f"[{row['ci_low']:+.4f}; {row['ci_high']:+.4f}]"
        print(
            f"{platform:12s} {row['rho_A']:8.4f} {row['rho_B']:8.4f} "
            f"{row['delta_rho']:+9.4f} {ci:>20s}  {row['відповідь']}"
        )
    print("\nВнутрішньоджерельна Spearman ρ")
    print(f"{'платформа':12s} {'ρ(A)':>8s} {'ρ(B)':>8s} {'Δρ':>9s} {'95% ДІ':>20s}  відповідь")
    for platform, row in per_platform.items():
        w = row["within_source"]
        ci = f"[{w['ci_low']:+.4f}; {w['ci_high']:+.4f}]"
        print(
            f"{platform:12s} {w['rho_A']:8.4f} {w['rho_B']:8.4f} "
            f"{w['delta_rho']:+9.4f} {ci:>20s}  {w['відповідь']}"
        )
    print("\nЗаписано:", args.out)


if __name__ == "__main__":
    main()
