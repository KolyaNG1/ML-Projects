"""Полный эксперимент: сравнение моделей, подбор CatBoost и выбор порога."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heart_risk.features import CATEGORICAL_FEATURES, clean_training_data, make_features  # noqa: E402

DATA_PATH = ROOT / "data" / "raw" / "cardio_train.csv"
MODEL_PATH = ROOT / "artifacts" / "heart_risk_catboost.joblib"
REPORTS = ROOT / "reports"
RANDOM_STATE = 42
TARGET_RECALL = 0.80


def evaluate(y_true, probabilities, threshold=0.5):
    labels = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, labels).ravel()
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "recall": float(recall_score(y_true, labels)),
        "precision": float(precision_score(y_true, labels, zero_division=0)),
        "f1": float(f1_score(y_true, labels)),
        "accuracy": float(accuracy_score(y_true, labels)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def compare_models(x_train, y_train, x_valid, y_valid):
    models = {
        "Логистическая регрессия": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
        ),
        "Дерево решений": DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=20, random_state=RANDOM_STATE
        ),
        "Случайный лес": RandomForestClassifier(
            n_estimators=250, max_depth=12, min_samples_leaf=5,
            n_jobs=-1, random_state=RANDOM_STATE,
        ),
        "CatBoost (базовый)": CatBoostClassifier(
            iterations=400, depth=6, learning_rate=0.05,
            cat_features=CATEGORICAL_FEATURES, loss_function="Logloss",
            eval_metric="AUC", random_seed=RANDOM_STATE, verbose=False,
            allow_writing_files=False,
        ),
    }
    rows = []
    for name, model in models.items():
        started = time.perf_counter()
        model.fit(x_train, y_train)
        metrics = evaluate(y_valid, model.predict_proba(x_valid)[:, 1])
        rows.append({"model": name, "fit_seconds": time.perf_counter() - started, **metrics})
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False)


def tune_catboost(x_train, y_train):
    estimator = CatBoostClassifier(
        loss_function="Logloss", eval_metric="AUC", random_seed=RANDOM_STATE, verbose=False,
        allow_writing_files=False, thread_count=2,
    )
    search_space = {
        "iterations": [300, 500, 700],
        "depth": [4, 5, 6, 7],
        "learning_rate": [0.03, 0.05, 0.08, 0.1],
        "l2_leaf_reg": [3, 5, 7, 9],
        "random_strength": [0.5, 1.0, 2.0],
        "bagging_temperature": [0.0, 0.5, 1.0],
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=search_space,
        n_iter=12,
        scoring={"roc_auc": "roc_auc", "recall": "recall", "precision": "precision"},
        refit="roc_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=1,
        return_train_score=False,
        verbose=1,
    )
    search.fit(x_train, y_train, cat_features=CATEGORICAL_FEATURES)
    columns = [
        "rank_test_roc_auc", "mean_test_roc_auc", "std_test_roc_auc",
        "mean_test_recall", "mean_test_precision", "mean_fit_time", "params",
    ]
    results = pd.DataFrame(search.cv_results_)[columns].sort_values("rank_test_roc_auc")
    results["params"] = results["params"].map(
        lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True)
    )
    return search.best_params_, results


def choose_threshold(y_valid, probabilities):
    rows = [
        evaluate(y_valid, probabilities, threshold)
        for threshold in np.arange(0.20, 0.701, 0.01)
    ]
    table = pd.DataFrame(rows)
    eligible = table[table["recall"] >= TARGET_RECALL]
    if eligible.empty:
        selected = table.sort_values(["recall", "precision"], ascending=False).iloc[0]
    else:
        selected = eligible.sort_values(["precision", "threshold"], ascending=False).iloc[0]
    return float(selected["threshold"]), table


def main():
    REPORTS.mkdir(exist_ok=True)
    raw = pd.read_csv(DATA_PATH)
    data = clean_training_data(raw)
    x = make_features(data.drop(columns=["id", "cardio"]))
    y = data["cardio"]

    x_dev, x_test, y_dev, y_test = train_test_split(
        x, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_dev, y_dev, test_size=0.25, random_state=RANDOM_STATE, stratify=y_dev
    )

    comparison = compare_models(x_train, y_train, x_valid, y_valid)
    comparison.to_csv(REPORTS / "model-comparison.csv", index=False, encoding="utf-8-sig")

    best_params, search_results = tune_catboost(x_train, y_train)
    search_results.to_csv(REPORTS / "hyperparameter-search.csv", index=False, encoding="utf-8-sig")

    validation_model = CatBoostClassifier(
        **best_params, cat_features=CATEGORICAL_FEATURES,
        loss_function="Logloss", eval_metric="AUC", random_seed=RANDOM_STATE,
        verbose=False, allow_writing_files=False,
    )
    validation_model.fit(x_train, y_train)
    valid_probabilities = validation_model.predict_proba(x_valid)[:, 1]
    threshold, threshold_table = choose_threshold(y_valid, valid_probabilities)
    threshold_table.to_csv(REPORTS / "threshold-search.csv", index=False, encoding="utf-8-sig")

    final_model = CatBoostClassifier(
        **best_params, cat_features=CATEGORICAL_FEATURES,
        loss_function="Logloss", eval_metric="AUC", random_seed=RANDOM_STATE,
        verbose=False, allow_writing_files=False,
    )
    final_model.fit(x_dev, y_dev)
    test_probabilities = final_model.predict_proba(x_test)[:, 1]
    test_metrics = evaluate(y_test, test_probabilities, threshold)

    summary = {
        "raw_observations": int(len(raw)),
        "clean_observations": int(len(data)),
        "removed_observations": int(len(raw) - len(data)),
        "train_observations": int(len(x_train)),
        "validation_observations": int(len(x_valid)),
        "test_observations": int(len(x_test)),
        "target_recall": TARGET_RECALL,
        "selected_threshold": threshold,
        "best_cv_roc_auc": float(search_results.iloc[0]["mean_test_roc_auc"]),
        "best_params": best_params,
        "test": test_metrics,
    }
    normalized_summary = json.loads(json.dumps(summary, default=float))
    (REPORTS / "metrics.json").write_text(
        json.dumps(normalized_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    joblib.dump(
        {
            "model": final_model,
            "threshold": threshold,
            "metrics": test_metrics,
            "best_params": best_params,
        },
        MODEL_PATH,
    )
    print(json.dumps(normalized_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
