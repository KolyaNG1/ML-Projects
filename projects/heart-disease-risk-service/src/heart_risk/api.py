"""FastAPI-приложение для оценки риска ССЗ."""

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from heart_risk.features import make_features
from heart_risk.schemas import PatientQuestionnaire


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "artifacts" / "heart_risk_catboost.joblib"
RISK_THRESHOLD = 0.45

app = FastAPI(title="Heart Risk API", version="0.1.0")
_model = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(MODEL_PATH)
        _model = joblib.load(MODEL_PATH)
    return _model


@app.get("/health")
def health():
    return {"status": "ok", "model_ready": MODEL_PATH.exists()}


@app.post("/score")
def score(questionnaire: PatientQuestionnaire):
    try:
        raw = pd.DataFrame([questionnaire.model_dump()])
        probability = float(get_model().predict_proba(make_features(raw))[0, 1])
    except FileNotFoundError:
        raise HTTPException(503, "Модель не найдена. Запустите scripts/train.py")

    return {
        "risk_probability": round(probability, 4),
        "risk_level": "high" if probability >= RISK_THRESHOLD else "low",
        "threshold": RISK_THRESHOLD,
        "medical_disclaimer": "Результат носит информационный характер и не является диагнозом.",
    }
