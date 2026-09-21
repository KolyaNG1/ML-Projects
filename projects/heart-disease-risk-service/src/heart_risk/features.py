"""Единое преобразование анкетных данных для обучения и сервиса."""

from __future__ import annotations

import pandas as pd


RAW_FEATURES = [
    "age", "gender", "height", "weight", "ap_hi", "ap_lo",
    "cholesterol", "gluc", "smoke", "alco", "active",
]
MODEL_FEATURES = RAW_FEATURES + ["bmi", "age_years"]
DEFAULT_MEDICAL_VALUES = {"ap_hi": 120, "ap_lo": 80, "cholesterol": 1, "gluc": 1}
CATEGORICAL_FEATURES = ["gender", "cholesterol", "gluc", "smoke", "alco", "active"]


def clean_training_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Удаляет дубликаты и физиологически неправдоподобные наблюдения."""
    result = frame.drop_duplicates().copy()
    plausible = (
        result["height"].between(120, 220)
        & result["weight"].between(35, 200)
        & result["ap_hi"].between(80, 250)
        & result["ap_lo"].between(40, 180)
        & (result["ap_hi"] > result["ap_lo"])
    )
    return result.loc[plausible].reset_index(drop=True)


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Заполняет неизвестные расширенные поля и создаёт признаки для модели.

    Значения по умолчанию применяются только для короткой анкеты. В боевом
    сервисе их следует заменить отдельной моделью или явно показывать это
    пользователю в результате.
    """
    result = frame.copy()
    for column, value in DEFAULT_MEDICAL_VALUES.items():
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(value)

    result[CATEGORICAL_FEATURES] = result[CATEGORICAL_FEATURES].astype(int)

    result["bmi"] = result["weight"] / (result["height"] / 100) ** 2
    result["age_years"] = result["age"] / 365.25
    return result.loc[:, MODEL_FEATURES]
