# Streamlit-демонстрация: загрузка изображения, Top-3 предсказания,
# уверенность, история запросов и краткая статистика.
# Запуск из корня:  streamlit run app.py

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

from dataset import get_val_transforms
from models import build_model

# ---------- Настройки ----------
CKPT_PATH = "models/densenet121_best.pth"   # лучшая модель
INPUT_SIZE = 224
HISTORY_PATH = Path("logs/app_history.json")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(page_title="Распознавание дорожных знаков", layout="wide")


@st.cache_resource
def load_model():
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model = build_model(ckpt["model_name"], ckpt["num_classes"], pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE).eval()
    return model, ckpt["class_names"]


def load_history():
    if HISTORY_PATH.exists():
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_history(history):
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


@torch.no_grad()
def predict(model, class_names, image: Image.Image, top_k: int = 3):
    tf = get_val_transforms(INPUT_SIZE)
    x = tf(image).unsqueeze(0).to(DEVICE)
    logits = model(x)
    probs = F.softmax(logits, dim=1)[0].cpu().numpy()
    top_idx = np.argsort(probs)[::-1][:top_k]
    return [(class_names[i], float(probs[i])) for i in top_idx]


st.title("🚦 Распознавание дорожных знаков")
st.caption("GTSRB · 43 класса · Top-3 предсказания")

model, class_names = load_model()

tab_predict, tab_history = st.tabs(["🔍 Распознавание", "📊 История и статистика"])

with tab_predict:
    col1, col2 = st.columns([1, 1])
    with col1:
        uploaded = st.file_uploader("Загрузите изображение знака",
                                    type=["jpg", "jpeg", "png", "ppm"])
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Исходное изображение", use_container_width=True)

    with col2:
        if uploaded:
            results = predict(model, class_names, image, top_k=3)
            st.subheader("Top-3 предсказания")
            for i, (cls, prob) in enumerate(results, 1):
                st.write(f"**{i}. {cls}** — {prob:.2%}")
                st.progress(prob)

            st.subheader("Краткая статистика")
            st.write(f"Уверенность топ-1: **{results[0][1]:.2%}**")
            if results[0][1] < 0.5:
                st.warning("⚠️ Низкая уверенность модели. Рекомендуется ручная проверка.")

            history = load_history()
            history.append({
                "timestamp": datetime.now().isoformat(),
                "top1_class": results[0][0],
                "top1_prob": results[0][1],
                "top3": [{"class": c, "prob": p} for c, p in results],
            })
            save_history(history)

with tab_history:
    history = load_history()
    if not history:
        st.info("История пуста. Обработайте хотя бы одно изображение.")
    else:
        st.metric("Всего обработано изображений", len(history))
        df = pd.DataFrame([
            {"Время": h["timestamp"], "Класс (top-1)": h["top1_class"],
             "Уверенность": round(h["top1_prob"], 4)}
            for h in history
        ])
        st.dataframe(df, use_container_width=True)

        st.subheader("Распределение предсказанных классов")
        st.bar_chart(df["Класс (top-1)"].value_counts())

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Скачать историю CSV", csv, "history.csv", "text/csv")