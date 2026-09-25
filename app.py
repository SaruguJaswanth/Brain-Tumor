import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Brain Tumor Detection Dashboard",
    page_icon="🧠",
    layout="wide"
)

MODEL_PATH = r"C:\Users\gowthami\PycharmProjects\Brain Tumor\.venv\New_BrainTumorClassification_2025\ModelFiles\brain_tumor_cnn.h5"
IMG_SIZE = 224

ACC_IMG = r"C:\Users\gowthami\PycharmProjects\Brain Tumor\.venv\New_BrainTumorClassification_2025\CReports\accuracy.png"
LOSS_IMG = r"C:\Users\gowthami\PycharmProjects\Brain Tumor\.venv\New_BrainTumorClassification_2025\CReports\loss.png"

# 🔴 CLASS LABELS (MATCH TRAINING)
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_cnn_model():
    return load_model(MODEL_PATH)

model = load_cnn_model()

# ---------------- STYLES ----------------
st.markdown("""
<style>
.title {
    font-size:38px;
    font-weight:bold;
    color:#00c6ff;
}
.card {
    background-color:#1e2228;
    padding:20px;
    border-radius:12px;
}
.good {
    background-color:#0f5132;
    padding:15px;
    border-radius:10px;
    color:white;
    font-size:20px;
}
.bad {
    background-color:#842029;
    padding:15px;
    border-radius:10px;
    color:white;
    font-size:20px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<div class='title'>🧠 Brain Tumor Detection & Model Performance</div>", unsafe_allow_html=True)
st.write("CNN-based MRI Multi-Class Classification System")

st.markdown("---")

# ---------------- LAYOUT ----------------
col1, col2 = st.columns([1, 1])

# ================= LEFT: PREDICTION =================
with col1:
    st.markdown("### 🔍 Tumor Prediction")
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload Brain MRI Image",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded MRI Image", use_container_width=True)

        img = np.array(image.convert("RGB"))
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        if st.button("Predict"):
            with st.spinner("Analyzing image..."):
                preds = model.predict(img)[0]

            class_id = np.argmax(preds)
            confidence = preds[class_id] * 100
            label = CLASS_NAMES[class_id]

            if label == "notumor":
                st.markdown("<div class='good'>✅ No Tumor Detected</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='bad'>⚠ Tumor Type: {label.capitalize()}</div>", unsafe_allow_html=True)

            st.write(f"Confidence: **{confidence:.2f}%**")

    st.markdown("</div>", unsafe_allow_html=True)

# ================= RIGHT: GRAPHS =================
with col2:
    st.markdown("### 📊 Model Training Performance")
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    if os.path.exists(ACC_IMG):
        st.subheader("Accuracy Curve")
        st.image(ACC_IMG, use_container_width=True)

    if os.path.exists(LOSS_IMG):
        st.subheader("Loss Curve")
        st.image(LOSS_IMG, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<center><b>Final Year Project – Brain Tumor Detection using CNN</b></center>",
    unsafe_allow_html=True
)
