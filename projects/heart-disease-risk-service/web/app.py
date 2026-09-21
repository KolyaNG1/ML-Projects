"""Минималистичная анкета Streamlit."""

from pathlib import Path
import os
import sys

import requests
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/score")

st.set_page_config(page_title="Проверка риска ССЗ", page_icon="❤️")
st.title("Проверка риска ССЗ ❤️")
st.caption("Быстрая информационная оценка по данным анкеты. Не заменяет консультацию врача.")

with st.form("risk_questionnaire"):
    st.subheader("Основная анкета")
    age_years = st.number_input("Возраст, лет", min_value=18, max_value=100, value=45)
    gender_label = st.selectbox("Пол", ["Женщина", "Мужчина"])
    height = st.number_input("Рост, см", min_value=100, max_value=250, value=170)
    weight = st.number_input("Вес, кг", min_value=30.0, max_value=250.0, value=70.0)
    smoke = st.checkbox("Курю")
    alco = st.checkbox("Употребляю алкоголь")
    active = st.checkbox("Регулярно занимаюсь физической активностью", value=True)

    with st.expander("Подробнее: медицинские показатели (необязательно)"):
        st.caption("Если поля оставить пустыми, сервис применит нейтральные значения.")
        ap_hi = st.number_input("Верхнее давление, мм рт. ст.", min_value=70, max_value=260, value=None)
        ap_lo = st.number_input("Нижнее давление, мм рт. ст.", min_value=40, max_value=180, value=None)
        levels = ["Не знаю", "Норма", "Выше нормы", "Значительно выше нормы"]
        cholesterol = st.selectbox("Холестерин", levels)
        gluc = st.selectbox("Глюкоза", levels)

    submitted = st.form_submit_button("Оценить риск")

if submitted:
    levels = {"Не знаю": None, "Норма": 1, "Выше нормы": 2, "Значительно выше нормы": 3}
    payload = {
        "age": int(age_years * 365.25), "gender": 1 if gender_label == "Женщина" else 2,
        "height": height, "weight": weight, "smoke": int(smoke), "alco": int(alco),
        "active": int(active), "ap_hi": ap_hi, "ap_lo": ap_lo,
        "cholesterol": levels[cholesterol], "gluc": levels[gluc],
    }
    try:
        result = requests.post(API_URL, json=payload, timeout=10).json()
        probability = result["risk_probability"]
        if result["risk_level"] == "high":
            st.error(f"Повышенный риск: {probability:.1%}")
            st.info("Рекомендуем записаться на консультацию к врачу-кардиологу.")
        else:
            st.success(f"Низкий риск: {probability:.1%}")
            st.info("Продолжайте следить за здоровьем и проходить профилактические осмотры.")
        st.caption(result["medical_disclaimer"])
    except requests.RequestException:
        st.error("Сервис временно недоступен. Проверьте, что API запущен.")
