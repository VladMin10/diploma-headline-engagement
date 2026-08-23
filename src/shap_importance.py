"""EXP-006: важливість копірайтингових ознак по платформах (питання Q3).

Запуск:
    .venv/bin/python -m src.shap_importance --config experiments/EXP-006_.../config.json

Конфігурація ознак T1 + T2, модель LightGBM, значення SHAP обчислюються
точним TreeSHAP. Ridge у цьому експерименті не бере участі: для лінійної
моделі важливість зводиться до коефіцієнтів на розкид ознаки, і два
показники різної природи в одну таблицю τ не зводяться.

Правило рішення і умова читання τ зафіксовані в картці до прогону
і в коді не перевизначаються: пороги подано константами нижче
з посиланням на джерело.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform as py_platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import shap
import sklearn
from scipy.stats import kendalltau

from src.data.load import CLEAN_PATH, PROJECT_ROOT
from src.data.split import SPLIT_PATH, compute_split, load_split
from src.features.assemble import build_feature_union, select_rows
from src.features.copywriting import FEATURE_ORDER
from src.models.zoo import build_model

PLATFORMS = ["Facebook", "GooglePlus", "LinkedIn"]

# Пороги з docs/research_questions.md (6 серпня 2026) і з картки EXP-006
# (доповнення 8 серпня 2026). У коді не переглядаються.
TAU_SIMILAR = 0.7
TAU_PARTIAL = 0.4
MIN_IMPORTANCE_RATIO = 0.1

FEATURE_LABELS = {
    "gap": "прогалина в інформації",
    "scale": "масштаб наслідків",
    "concrete": "конкретність предиката",
    "conflict": "явність конфлікту",
    "authority": "вага авторитету",
    "violation": "порушення очікування",
    "stakes": "проспективна невизначеність",
}

# Ознаки, чия надійність за EXP-002 потребує застереження. Перелік
# виводиться з результатів, а не задається тут: див. annotate_caveats().
CAVEAT_SOURCE = "experiments/EXP-002_annotation-agreement"


def annotate_caveats() -> dict:
    """Які ознаки супроводжуються застереженням про шум розмітки.

    Ознака вважається надійною без застережень лише тоді, коли проходить
    0,60 і за узгодженістю з людиною, і за стабільністю між прогонами.
    """
    base = Path(CAVEAT_SOURCE)
    agreement = {
        row["код"]: row["kappa_linear"]
        for row in json.loads((base / "agreement.json").read_text(encoding="utf-8"))[
            "features"
        ]
    }
    stability = {
        key: row["kappa_стабільності_зважена"]
        for key, row in json.loads(
            (base / "stability.json").read_text(encoding="utf-8")
        )["перевірка_гіпотези"]["за_ознаками"].items()
    }
    return {
        name: {
            "kappa_людина": agreement[name],
            "kappa_стабільність": stability[name],
            "застереження": bool(min(agreement[name], stability[name]) < 0.60),
        }
        for name in FEATURE_ORDER
    }


def group_of(name: str) -> str:
    """Група ознаки за назвою стовпця після кодування.

    Source розгорнутий у десятки стовпців one-hot, тому важливість
    поля дорівнює сумі по його стовпцях. Порівнювати окремий стовпець
    з ознакою T2 означало б порівнювати з часткою.
    """
    if name.startswith("T2__"):
        return name.rsplit("__", 1)[-1]
    body = name.split("__", 1)[1]
    for field in ("source", "topic", "hour", "weekday", "month"):
        if body.startswith(field):
            return f"T1:{field}"
    return "T1:інше"


def mean_abs_shap(values: np.ndarray, groups: np.ndarray, rows: np.ndarray) -> dict:
    """Середнє абсолютне SHAP по підмножині рядків, згорнуте по групах."""
    per_feature = np.abs(values[rows]).mean(axis=0)
    out = {}
    for group in np.unique(groups):
        out[str(group)] = float(per_feature[groups == group].sum())
    return out


def bootstrap_tau(
    values: dict[str, np.ndarray],
    groups: np.ndarray,
    n_rows: int,
    n_iterations: int,
    random_state: int,
) -> dict:
    """Довірчі інтервали для τ бутстрепом по тестових рядках.

    Модель не перенавчається: перераховуються лише середні абсолютні
    значення SHAP на вибраних рядках і побудоване на них упорядкування.
    Отже інтервал враховує варіацію через вибірку тесту, але не через
    навчання. Обмеження назване в картці.
    """
    rng = np.random.default_rng(random_state)
    pairs = [(a, b) for i, a in enumerate(PLATFORMS) for b in PLATFORMS[i + 1 :]]
    draws = {f"{a}|{b}": np.empty(n_iterations) for a, b in pairs}

    for i in range(n_iterations):
        rows = rng.integers(0, n_rows, size=n_rows)
        ranks = {}
        for platform in PLATFORMS:
            importance = mean_abs_shap(values[platform], groups, rows)
            ranks[platform] = [importance[name] for name in FEATURE_ORDER]
        for a, b in pairs:
            draws[f"{a}|{b}"][i] = kendalltau(ranks[a], ranks[b]).statistic

    return {
        key: {
            "ci_low": round(float(np.nanpercentile(values_, 2.5)), 4),
            "ci_high": round(float(np.nanpercentile(values_, 97.5)), 4),
            "n_iterations": n_iterations,
        }
        for key, values_ in draws.items()
    }


def read_tau(tau: float) -> str:
    if tau >= TAU_SIMILAR:
        return "прийоми працюють подібно"
    if tau >= TAU_PARTIAL:
        return "часткова подібність; висновок робиться з таблиці важливості"
    return "набори дієвих прийомів різні"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))

    frame = pd.read_parquet(CLEAN_PATH)
    frame["IDLink"] = frame["IDLink"].astype("int64")
    split_info = load_split(SPLIT_PATH)
    split = compute_split(frame, split_info["boundary"])
    train = select_rows(frame.loc[split.train_mask].copy(), ["T2"])
    test = select_rows(frame.loc[split.test_mask].copy(), ["T2"])
    print(f"train {len(train)}, test {len(test)} (анотована підвибірка)")

    union = build_feature_union(["T1", "T2"], {"T1": {"include_month": False}})
    union.fit(train)
    x_train = union.transform(train)
    x_test = union.transform(test)
    names = list(union.get_feature_names_out())
    groups = np.array([group_of(name) for name in names])

    dense_train = x_train.toarray() if hasattr(x_train, "toarray") else x_train
    dense_test = x_test.toarray() if hasattr(x_test, "toarray") else x_test
    all_rows = np.arange(len(test))

    values, importance, ratios = {}, {}, {}
    for platform in PLATFORMS:
        y_train = train[f"{platform}_log1p"].to_numpy()
        model = build_model("lightgbm", random_state=int(config["random_state"]))
        model.fit(dense_train, y_train)

        explainer = shap.TreeExplainer(model)
        values[platform] = np.asarray(explainer.shap_values(dense_test))
        importance[platform] = mean_abs_shap(values[platform], groups, all_rows)

        max_t2 = max(importance[platform][name] for name in FEATURE_ORDER)
        source = importance[platform].get("T1:source", 0.0)
        ratios[platform] = {
            "max_важливість_T2": round(max_t2, 6),
            "сукупна_важливість_Source": round(source, 6),
            "R": round(max_t2 / source, 4) if source else float("inf"),
            "τ_читається": bool(source and (max_t2 / source) >= MIN_IMPORTANCE_RATIO),
        }
        print(
            f"  {platform:12s} R = {ratios[platform]['R']:.4f} "
            f"({'τ читається' if ratios[platform]['τ_читається'] else 'τ НЕ читається'})"
        )

    pairs = [(a, b) for i, a in enumerate(PLATFORMS) for b in PLATFORMS[i + 1 :]]
    intervals = bootstrap_tau(
        values,
        groups,
        len(test),
        int(config["bootstrap_iters"]),
        int(config["random_state"]),
    )

    tau_table = {}
    for a, b in pairs:
        key = f"{a}|{b}"
        stat = kendalltau(
            [importance[a][name] for name in FEATURE_ORDER],
            [importance[b][name] for name in FEATURE_ORDER],
        )
        readable = ratios[a]["τ_читається"] and ratios[b]["τ_читається"]
        tau_table[key] = {
            "tau": round(float(stat.statistic), 4),
            "p_value": round(float(stat.pvalue), 4),
            **intervals[key],
            "обидві_платформи_читаються": readable,
            "висновок": (
                read_tau(stat.statistic)
                if readable
                else "не інтерпретується: внесок T2 замалий на тлі Source"
            ),
        }

    caveats = annotate_caveats()
    out = {
        "exp_id": config["exp_id"],
        "generated_by": "src/shap_importance.py",
        "config_file": str(config_path.relative_to(PROJECT_ROOT)),
        "data": {
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "boundary": split_info["boundary"],
            "test_ids_sha256": hashlib.sha256(
                ",".join(str(int(i)) for i in test["IDLink"]).encode()
            ).hexdigest(),
        },
        "пороги": {
            "tau_подібність": TAU_SIMILAR,
            "tau_часткова_подібність": TAU_PARTIAL,
            "мінімальне_відношення_важливості": MIN_IMPORTANCE_RATIO,
            "джерело": (
                "docs/research_questions.md від 6 серпня 2026 року; "
                "доповнення про зону 0,4–0,7 і про умову читання τ — "
                "картка EXP-006, рішення користувача від 8 серпня 2026 року"
            ),
        },
        "умова_читання_tau": ratios,
        "kendall_tau": tau_table,
        "важливість": {
            platform: {
                "T2": {
                    name: {
                        "мітка": FEATURE_LABELS[name],
                        "mean_abs_shap": round(importance[platform][name], 6),
                        **caveats[name],
                    }
                    for name in FEATURE_ORDER
                },
                "T1": {
                    key: round(value, 6)
                    for key, value in sorted(importance[platform].items())
                    if key.startswith("T1:")
                },
            }
            for platform in PLATFORMS
        },
        "версії": {
            "python": py_platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
            "scipy": scipy.__version__,
            "shap": shap.__version__,
        },
    }

    results_path = config_path.parent / "results.json"
    results_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nКендалл τ між платформами (упорядкування семи ознак T2)")
    for key, row in tau_table.items():
        print(
            f"  {key:24s} τ = {row['tau']:+.4f} "
            f"[{row['ci_low']:+.4f}; {row['ci_high']:+.4f}]  {row['висновок']}"
        )
    print("\nЗаписано:", results_path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
