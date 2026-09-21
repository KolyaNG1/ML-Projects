"""Строит графики для README по исходному набору и обученной модели."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heart_risk.features import make_features  # noqa: E402

DATA_PATH = ROOT / "data" / "raw" / "cardio_train.csv"
MODEL_PATH = ROOT / "artifacts" / "heart_risk_catboost.joblib"
OUTPUT = ROOT / "reports" / "figures"

RUSSIAN_LABELS = {
    "age_years": "Возраст, лет", "bmi": "Индекс массы тела", "ap_hi": "Верхнее давление, мм рт. ст.",
    "ap_lo": "Нижнее давление, мм рт. ст.", "cholesterol": "Холестерин", "gluc": "Глюкоза",
    "active": "Физическая активность", "smoke": "Курение", "alco": "Алкоголь",
    "height": "Рост", "weight": "Вес", "gender": "Пол", "age": "Возраст, дни",
}


def save_class_balance(data: pd.DataFrame):
    counts = data["cardio"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(["Нет ССЗ", "Есть ССЗ"], counts.values, color=["#6BAED6", "#E6550D"], width=0.6)
    ax.set_title("Баланс целевого признака")
    ax.set_ylabel("Количество анкет")
    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, count + 700, f"{count:,}".replace(",", " "), ha="center")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT / "class-balance.png", dpi=180)
    plt.close(fig)


def save_distributions(data: pd.DataFrame):
    prepared = make_features(data.drop(columns=["id", "cardio"]))
    prepared["cardio"] = data["cardio"].map({0: "Нет ССЗ", 1: "Есть ССЗ"})
    # Для читаемости графика отсеиваем только экстремумы давления; модель учится на полном наборе.
    prepared = prepared.query("70 <= ap_hi <= 240 and 40 <= ap_lo <= 160 and 15 <= bmi <= 55")
    sns.set_theme(style="whitegrid", font_scale=0.9)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    specs = [("age_years", 25), ("bmi", 28), ("ap_hi", 34)]
    for ax, (column, bins) in zip(axes, specs):
        sns.histplot(data=prepared, x=column, hue="cardio", bins=bins, stat="density", common_norm=False,
                     element="step", fill=True, alpha=0.25, palette={"Нет ССЗ": "#3182BD", "Есть ССЗ": "#E6550D"}, ax=ax)
        ax.set_xlabel(RUSSIAN_LABELS[column])
        ax.set_ylabel("Плотность")
        ax.set_title(RUSSIAN_LABELS[column])
    fig.suptitle("Распределения ключевых признаков по классу", y=1.03, fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT / "feature-distributions.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_importance():
    model = joblib.load(MODEL_PATH)
    importance = pd.Series(model.feature_importances_, index=model.feature_names_).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5.4))
    colors = ["#9ECAE1"] * len(importance)
    colors[-3:] = ["#3182BD", "#2171B5", "#08519C"]
    ax.barh([RUSSIAN_LABELS.get(key, key) for key in importance.index], importance.values, color=colors)
    ax.set_title("Важность признаков в CatBoost")
    ax.set_xlabel("Вклад в модель")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT / "feature-importance.png", dpi=180)
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA_PATH)
    save_class_balance(data)
    save_distributions(data)
    save_importance()


if __name__ == "__main__":
    main()
