# ===============================
# MCC Diagnostic App (Enhanced UI)
# ===============================

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from fpdf import FPDF
from datetime import datetime

# -------------------------------
# CONFIG
# -------------------------------
IMG_SIZE = 224
THRESHOLD = 0.35  # MCC-safe ROC threshold

# Class semantics (LOCKED)
IDX_TO_CLASS = {0: "MCC", 1: "Non-MCC"}

# -------------------------------
# PAGE SETTINGS
# -------------------------------
st.set_page_config(
    page_title="MCC Diagnostic App",
    page_icon="🩺",
    layout="centered"
)

# -------------------------------
# HEADER
# -------------------------------
st.title("🩺 MCC Diagnostic App")
st.subheader("AI-Assisted Merkel Cell Carcinoma Risk Assessment")

st.markdown("""
⚠️ **Academic Research Prototype – NOT for Clinical Diagnosis**  
This system is developed strictly for **research and educational purposes**.  
Final medical decisions must always be made by certified healthcare professionals.
""")

st.divider()

# -------------------------------
# ABOUT MCC
# -------------------------------
st.header("📘 About Merkel Cell Carcinoma (MCC)")
st.markdown("""
Merkel Cell Carcinoma (MCC) is a **rare but highly aggressive skin cancer**
originating from neuroendocrine cells of the skin.  
It often presents as a **painless, fast-growing lesion** on sun-exposed areas
and requires **early detection** due to its high metastatic potential.
""")

st.divider()

# -------------------------------
# LOAD MODEL (KERAS 3 SAFE)
# -------------------------------
@st.cache_resource
def load_model():
    model_dir = "mcc_model_savedmodel"
    if not os.path.exists(model_dir):
        st.error("❌ Model folder not found.")
        return None
    return tf.keras.layers.TFSMLayer(
        model_dir,
        call_endpoint="serving_default"
    )

model = load_model()
if model is None:
    st.stop()

# -------------------------------
# IMAGE UPLOAD
# -------------------------------
st.header("📤 Upload Skin Lesion Image")

uploaded_file = st.file_uploader(
    "Supported formats: JPG, JPEG, PNG",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # -------------------------------
    # PREPROCESS
    # -------------------------------
    img_resized = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    # -------------------------------
    # FEATURE EXTRACTION (SAFE)
    # -------------------------------
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lesion_area = 0
    perimeter = 0
    circularity = 0

    if contours:
        c = max(contours, key=cv2.contourArea)
        lesion_area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        if perimeter != 0:
            circularity = (4 * np.pi * lesion_area) / (perimeter ** 2)

    st.divider()

    # -------------------------------
    # EXTRACTED FEATURES
    # -------------------------------
    st.header("🧬 Extracted Image Features")
    col1, col2, col3 = st.columns(3)

    col1.metric("Lesion Area", f"{lesion_area:.0f} px²")
    col2.metric("Perimeter", f"{perimeter:.2f} px")
    col3.metric("Circularity", f"{circularity:.3f}")

    st.caption("⚠️ Feature values are **approximate** and used for visualization only.")

    st.divider()

    # -------------------------------
    # PREDICTION
    # -------------------------------
    with st.spinner("Analyzing lesion..."):
        output = model(img_array)
        prob_non_mcc = float(list(output.values())[0][0][0])

    prob_mcc = 1.0 - prob_non_mcc

    # -------------------------------
    # RISK INTERPRETATION
    # -------------------------------
    if prob_mcc >= 0.65:
        risk = "HIGH"
        color = "🚨"
    elif prob_mcc >= THRESHOLD:
        risk = "MEDIUM"
        color = "⚠️"
    else:
        risk = "LOW"
        color = "✅"

    st.header("📊 MCC Prediction Scores")

    col1, col2 = st.columns(2)
    col1.metric("MCC Probability", f"{prob_mcc * 100:.2f}%")
    col2.metric("Benign Probability", f"{prob_non_mcc * 100:.2f}%")

    st.markdown(f"### {color} **Risk Level: {risk}**")

    if risk == "HIGH":
        st.error("Immediate dermatological evaluation is strongly advised.")
    elif risk == "MEDIUM":
        st.warning("Clinical review recommended.")
    else:
        st.success("Routine monitoring suggested.")

    st.divider()

    # -------------------------------
    # PDF MEDICAL REPORT
    # -------------------------------
    st.header("📄 Generate Medical Report")

    def generate_pdf():
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        pdf.cell(0, 10, "MCC Diagnostic Report (Research Prototype)", ln=True)
        pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
        pdf.ln(5)

        pdf.multi_cell(0, 8, "Disclaimer: This report is generated by an academic AI system "
                              "and is NOT a clinical diagnosis.\n")

        pdf.ln(5)
        pdf.cell(0, 8, f"MCC Probability: {prob_mcc:.4f}", ln=True)
        pdf.cell(0, 8, f"Benign Probability: {prob_non_mcc:.4f}", ln=True)
        pdf.cell(0, 8, f"Risk Level: {risk}", ln=True)

        pdf.ln(5)
        pdf.cell(0, 8, "Extracted Image Features:", ln=True)
        pdf.cell(0, 8, f"Lesion Area: {lesion_area:.0f}", ln=True)
        pdf.cell(0, 8, f"Perimeter: {perimeter:.2f}", ln=True)
        pdf.cell(0, 8, f"Circularity: {circularity:.3f}", ln=True)

        return pdf.output(dest="S").encode("latin-1")

    pdf_bytes = generate_pdf()

    st.download_button(
        label="📥 Download Medical Report (PDF)",
        data=pdf_bytes,
        file_name="MCC_Diagnostic_Report.pdf",
        mime="application/pdf"
    )

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.caption("© MSc Research Prototype · AI-Assisted MCC Risk Assessment")
