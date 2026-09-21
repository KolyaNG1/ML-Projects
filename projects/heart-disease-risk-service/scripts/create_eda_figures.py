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

from heart_risk.features import clean_training_data, make_features  # noqa: E402

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
    data = clean_training_data(data)
    prepared = make_features(data.drop(columns=["id", "cardio"]))
    prepared["cardio"] = data["cardio"].map({0: "Нет ССЗ", 1: "Есть ССЗ"})
    # Для читаемости дополнительно ограничиваем только область, показанную на графике.
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
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"] if isinstance(artifact, dict) else artifact
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


def save_model_comparison():
    table = pd.read_csv(ROOT / "reports" / "model-comparison.csv")
    table = table.sort_values("roc_auc")
    fig, ax = plt.subplots(figsize=(9, 5.2))
    y = range(len(table))
    width = 0.24
    ax.barh([value - width for value in y], table["roc_auc"], height=width, label="ROC-AUC", color="#08519C")
    ax.barh(y, table["recall"], height=width, label="Полнота", color="#E6550D")
    ax.barh([value + width for value in y], table["precision"], height=width, label="Точность", color="#31A354")
    ax.set_yticks(list(y), table["model"])
    ax.set_xlim(0.55, 0.83)
    ax.set_xlabel("Значение метрики на валидации")
    ax.set_title("Сравнение базовых алгоритмов при пороге 0,50")
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT / "model-comparison.png", dpi=180)
    plt.close(fig)


def save_threshold_tradeoff():
    table = pd.read_csv(ROOT / "reports" / "threshold-search.csv")
    artifact = joblib.load(MODEL_PATH)
    selected = float(artifact["threshold"])
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(table["threshold"], table["recall"], label="Полнота", linewidth=2.5, color="#E6550D")
    ax.plot(table["threshold"], table["precision"], label="Точность", linewidth=2.5, color="#31A354")
    ax.plot(table["threshold"], table["f1"], label="F1", linewidth=2, color="#3182BD")
    ax.axvline(selected, color="#252525", linestyle="--", label=f"Выбранный порог: {selected:.2f}")
    ax.axhline(0.80, color="#969696", linestyle=":", label="Целевая полнота: 0,80")
    ax.set_xlabel("Порог отнесения к повышенному риску")
    ax.set_ylabel("Значение метрики")
    ax.set_ylim(0.5, 0.95)
    ax.set_title("Выбор рабочего порога на валидационной выборке")
    ax.legend(ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT / "threshold-tradeoff.png", dpi=180)
    plt.close(fig)


def save_confusion_matrix():
    artifact = joblib.load(MODEL_PATH)
    metrics = artifact["metrics"]
    matrix = [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]]
    fig, ax = plt.subplots(figsize=(6.3, 5.2))
    sns.heatmap(
        matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
        xticklabels=["Низкий риск", "Повышенный риск"],
        yticklabels=["Нет ССЗ", "Есть ССЗ"],
    )
    ax.set_xlabel("Прогноз")
    ax.set_ylabel("Фактический класс")
    ax.set_title("Матрица ошибок на отложенном тесте")
    fig.tight_layout()
    fig.savefig(OUTPUT / "confusion-matrix.png", dpi=180)
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA_PATH)
    save_class_balance(data)
    save_distributions(data)
    save_importance()
    save_model_comparison()
    save_threshold_tradeoff()
    save_confusion_matrix()


if __name__ == "__main__":
    main()
