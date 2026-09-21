"""Минималистичная анкета Streamlit."""

from pathlib import Path
import os
import sys

import requests
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/score")

st.set_page_config(page_title="Проверка риска ССЗ", page_icon="❤️", layout="centered")
st.markdown(
    """
    <style>
      .stApp { background: linear-gradient(180deg, #f8fbff 0%, #ffffff 42%); }
      .block-container { max-width: 900px; padding-top: 2rem; padding-bottom: 4rem; }
      .hero { padding: 1.5rem 1.7rem; border-radius: 24px; color: white;
              background: linear-gradient(125deg, #0f4c81 0%, #1976a3 52%, #2a9d8f 100%);
              box-shadow: 0 14px 35px rgba(15,76,129,.18); margin-bottom: 1.25rem; }
      .hero h1 { margin: 0 0 .45rem 0; font-size: 2.15rem; }
      .hero p { margin: 0; opacity: .92; font-size: 1.02rem; max-width: 680px; }
      .trust-row { display:flex; gap:.55rem; flex-wrap:wrap; margin-top:1rem; }
      .trust-pill { background:rgba(255,255,255,.16); padding:.34rem .68rem;
                    border-radius:999px; font-size:.82rem; }
      [data-testid="stForm"] { background:white; border:1px solid #e3edf5; border-radius:22px;
                               padding:1.2rem 1.35rem 1.35rem; box-shadow:0 10px 28px rgba(35,70,100,.07); }
      .section-title { color:#163b56; font-weight:700; font-size:1.12rem; margin:.35rem 0 .7rem; }
      .result-card { border-radius:20px; padding:1.25rem 1.4rem; margin-top:1rem; }
      .result-card h2 { margin:0 0 .35rem; font-size:1.45rem; }
      .result-card p { margin:.25rem 0; }
      .risk-low { background:#eaf8f3; border:1px solid #96d5bd; color:#155d46; }
      .risk-high { background:#fff1ef; border:1px solid #f0aaa2; color:#8c2720; }
      .footer-note { color:#617383; font-size:.82rem; text-align:center; margin-top:1.25rem; }
      .stButton > button, [data-testid="stFormSubmitButton"] button { border-radius:12px; font-weight:700; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>❤️ Проверьте риск заболеваний сердца</h1>
      <p>Ответьте на несколько вопросов и получите мгновенную информационную оценку. Это займёт около двух минут.</p>
      <div class="trust-row">
        <span class="trust-pill">✓ Бесплатно</span>
        <span class="trust-pill">✓ Без регистрации</span>
        <span class="trust-pill">✓ Результат сразу</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.form("risk_questionnaire"):
    st.markdown('<div class="section-title">Основная информация</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        age_years = st.number_input("Возраст, лет", min_value=18, max_value=100, value=45)
        height = st.number_input("Рост, см", min_value=120, max_value=220, value=170)
    with col2:
        gender_label = st.selectbox("Пол", ["Женщина", "Мужчина"])
        weight = st.number_input("Вес, кг", min_value=35.0, max_value=200.0, value=70.0)

    st.markdown('<div class="section-title">Образ жизни</div>', unsafe_allow_html=True)
    lifestyle1, lifestyle2, lifestyle3 = st.columns(3)
    with lifestyle1:
        smoke = st.checkbox("Курю")
    with lifestyle2:
        alco = st.checkbox("Употребляю алкоголь")
    with lifestyle3:
        active = st.checkbox("Физически активен", value=True)

    with st.expander("Медицинские показатели · необязательно"):
        st.caption("Если показатели неизвестны, сервис применит нейтральные значения и отметит ограничение оценки.")
        med1, med2 = st.columns(2)
        with med1:
            ap_hi = st.number_input("Верхнее давление", min_value=80, max_value=250, value=None, placeholder="например, 120")
            levels = ["Не знаю", "Норма", "Выше нормы", "Значительно выше нормы"]
            cholesterol = st.selectbox("Холестерин", levels)
        with med2:
            ap_lo = st.number_input("Нижнее давление", min_value=40, max_value=180, value=None, placeholder="например, 80")
            gluc = st.selectbox("Глюкоза", levels)

    submitted = st.form_submit_button("Получить оценку риска", use_container_width=True)

if submitted:
    level_codes = {"Не знаю": None, "Норма": 1, "Выше нормы": 2, "Значительно выше нормы": 3}
    payload = {
        "age": int(age_years * 365.25),
        "gender": 1 if gender_label == "Женщина" else 2,
        "height": height,
        "weight": weight,
        "smoke": int(smoke),
        "alco": int(alco),
        "active": int(active),
        "ap_hi": ap_hi,
        "ap_lo": ap_lo,
        "cholesterol": level_codes[cholesterol],
        "gluc": level_codes[gluc],
    }
    try:
        response = requests.post(API_URL, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        probability = float(result["risk_probability"])
        st.progress(probability, text=f"Расчётная вероятность риска: {probability:.1%}")
        if result["risk_level"] == "high":
            st.markdown(
                f"""<div class="result-card risk-high"><h2>Повышенный риск · {probability:.1%}</h2>
                <p>Рекомендуем не откладывать профилактическую консультацию с врачом-кардиологом.</p>
                <p><b>Результат не является диагнозом.</b></p></div>""",
                unsafe_allow_html=True,
            )
            st.button("Записаться на консультацию", type="primary", use_container_width=True)
        else:
            st.markdown(
                f"""<div class="result-card risk-low"><h2>Низкий риск · {probability:.1%}</h2>
                <p>Продолжайте следить за здоровьем и проходить регулярные профилактические осмотры.</p>
                <p><b>Результат не является диагнозом.</b></p></div>""",
                unsafe_allow_html=True,
            )
    except requests.RequestException as error:
        detail = ""
        if getattr(error, "response", None) is not None:
            try:
                detail = error.response.json().get("detail", "")
            except requests.JSONDecodeError:
                detail = ""
        st.error(detail or "Сервис временно недоступен. Проверьте, что API запущен.")

st.markdown(
    '<div class="footer-note">Учебный сервис · Не является медицинским изделием · При жалобах обратитесь к врачу</div>',
    unsafe_allow_html=True,
)
