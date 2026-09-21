"""Обучение модели, выбор порога по полноте и сохранение артефактов."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heart_risk.features import make_features  # noqa: E402

DATA_PATH = ROOT / "data" / "raw" / "cardio_train.csv"
MODEL_PATH = ROOT / "artifacts" / "heart_risk_catboost.joblib"
METRICS_PATH = ROOT / "reports" / "metrics.json"
THRESHOLD = 0.45


def main():
    data = pd.read_csv(DATA_PATH)
    x = make_features(data.drop(columns=["id", "cardio"]))
    y = data["cardio"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    model = CatBoostClassifier(
        iterations=500, depth=5, learning_rate=0.05, loss_function="Logloss",
        eval_metric="AUC", random_seed=42, verbose=False,
    )
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    labels = (probabilities >= THRESHOLD).astype(int)
    metrics = {
        "observations": int(len(data)), "test_observations": int(len(x_test)),
        "threshold": THRESHOLD,
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "recall": round(float(recall_score(y_test, labels)), 4),
        "precision": round(float(precision_score(y_test, labels)), 4),
    }
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
