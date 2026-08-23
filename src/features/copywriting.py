"""Рівень ознак T2 — копірайтингові характеристики заголовка.

Екстрактор нічого не обчислює з тексту. Він приєднує готові мітки
анотації Фази 3 за ідентифікатором новини і перетворює їх на числа
(docs/architecture.md розділ 4).

Перелік ознак, що входять у матрицю, не задається в коді вручну.
Він виводиться з результатів EXP-002: ознака входить, якщо проходить
поріг каппи 0,40 і в першій частині (узгодженість з людиною), і в другій
(стабільність між прогонами). Поріг зафіксовано в `docs/research_questions.md`
до запуску експериментів.

Наслідок для Фази 5: мітки є лише для 5 055 заголовків черги анотації,
тому будь-яке порівняння з рівнем T2 йде на цій підмножині, а не на всіх
81 634 рядках. Матриця T2 містить рядки лише для анотованих новин;
перетин рівнів робить `assemble.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.base import FeatureExtractor

LABELS = Path("annotations/llm_labels_run1.csv")
AGREEMENT = Path("experiments/EXP-002_annotation-agreement/agreement.json")
STABILITY = Path("experiments/EXP-002_annotation-agreement/stability.json")

KAPPA_THRESHOLD = 0.40

# Порядок ознак у рубриці. Номери 6 і 8 не використовуються:
# ознака 8 виключена за результатом пілоту (частота), ознака 6 —
# за результатом EXP-002, частина перша (каппа 0,296).
FEATURE_ORDER = [
    "gap",
    "scale",
    "concrete",
    "conflict",
    "authority",
    "violation",
    "stakes",
]


def resolve_included_features(
    agreement_path: Path = AGREEMENT,
    stability_path: Path = STABILITY,
    threshold: float = KAPPA_THRESHOLD,
) -> dict:
    """Які ознаки входять у T2 і чому.

    Ознака входить, якщо лінійна каппа не нижча за поріг в обох частинах
    EXP-002. Для другої частини береться загальна каппа стабільності,
    тобто зважене середнє за часткою режимів розміру пачки в черзі,
    як визначає протокол.

    Ознака зі смугою 0,40–0,60 входить, але позначається як така, що
    потребує застереження про шум розмітки у Фазі 6.
    """
    agreement = {
        row["код"]: row["kappa_linear"]
        for row in json.loads(agreement_path.read_text(encoding="utf-8"))["features"]
    }
    stability = {
        key: row["kappa_стабільності_зважена"]
        for key, row in json.loads(stability_path.read_text(encoding="utf-8"))[
            "перевірка_гіпотези"
        ]["за_ознаками"].items()
    }

    decisions = {}
    for name in FEATURE_ORDER:
        k_human = agreement.get(name)
        k_stable = stability.get(name)
        included = (
            k_human is not None
            and k_stable is not None
            and k_human >= threshold
            and k_stable >= threshold
        )
        decisions[name] = {
            "kappa_людина": k_human,
            "kappa_стабільність": k_stable,
            "входить": bool(included),
            "потребує_застереження": bool(
                included and min(k_human, k_stable) < 0.60
            ),
        }

    return {
        "поріг": threshold,
        "правило": (
            "ознака входить, якщо лінійна каппа не нижча за поріг в обох "
            "частинах EXP-002; для другої частини береться загальна каппа "
            "стабільності, зважена за часткою режимів пачки в черзі"
        ),
        "джерела": {"частина_1": str(agreement_path), "частина_2": str(stability_path)},
        "включено": [n for n, d in decisions.items() if d["входить"]],
        "виключено": [n for n, d in decisions.items() if not d["входить"]],
        "із_застереженням": [
            n for n, d in decisions.items() if d["потребує_застереження"]
        ],
        "за_ознаками": decisions,
    }


def load_labels(path: Path = LABELS, features: list[str] | None = None) -> pd.DataFrame:
    """Мітки анотації, індексовані за IDLink, у числовому вигляді.

    Мітка `unclear`, передбачена промптом для заголовків, які неможливо
    оцінити, стає NaN. Рядок не відкидається: інші ознаки того самого
    заголовка залишаються придатними.
    """
    features = features or FEATURE_ORDER
    frame = pd.read_csv(path)
    frame["IDLink"] = frame["IDLink"].astype("int64")
    numeric = pd.DataFrame(index=frame["IDLink"].to_numpy())
    for name in features:
        numeric[name] = pd.to_numeric(frame[name], errors="coerce").to_numpy()
    numeric.index.name = "IDLink"
    return numeric


class CopywritingFeatures(FeatureExtractor):
    """Рівень T2.

    Екстрактор нічого не вчить із даних: перелік ознак і самі мітки
    відомі до будь-якого фолда, а fit лише перевіряє доступність міток.
    Ознака, яка не пройшла поріг каппи, у матрицю не потрапляє.
    """

    level = "T2"

    def __init__(self, labels_path: Path | str = LABELS, features: list[str] | None = None):
        self.labels_path = labels_path
        self.features = features

    @property
    def required_columns(self) -> list[str]:
        return ["IDLink"]

    def _resolve_features(self) -> list[str]:
        if self.features is not None:
            return list(self.features)
        return resolve_included_features()["включено"]

    def fit(self, X: pd.DataFrame, y=None) -> "CopywritingFeatures":
        self.check_no_leakage()
        self.features_ = self._resolve_features()
        self.labels_ = load_labels(Path(self.labels_path), self.features_)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        ids = X["IDLink"].astype("int64").to_numpy()
        return self.labels_.reindex(ids).to_numpy(dtype=np.float64)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"T2__{name}" for name in self.features_], dtype=object)
