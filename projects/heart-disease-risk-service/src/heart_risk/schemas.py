"""Схема входных данных API."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class PatientQuestionnaire(BaseModel):
    """Ответы анкеты. Пол: 1 — женщина, 2 — мужчина, как в наборе данных."""

    age: int = Field(ge=18 * 365, le=100 * 366, description="Возраст в днях")
    gender: Literal[1, 2]
    height: int = Field(ge=100, le=250, description="Рост в сантиметрах")
    weight: float = Field(ge=30, le=250, description="Вес в килограммах")
    smoke: int = Field(ge=0, le=1)
    alco: int = Field(ge=0, le=1)
    active: int = Field(ge=0, le=1)
    ap_hi: int | None = Field(default=None, ge=70, le=260)
    ap_lo: int | None = Field(default=None, ge=40, le=180)
    cholesterol: int | None = Field(default=None, ge=1, le=3)
    gluc: int | None = Field(default=None, ge=1, le=3)

    @model_validator(mode="after")
    def pressure_is_consistent(self):
        if self.ap_hi is not None and self.ap_lo is not None and self.ap_hi <= self.ap_lo:
            raise ValueError("Верхнее давление должно быть больше нижнего")
        return self
