# ==========================================================
# MCC Diagnostic App – Enhanced Academic UI (Streamlit)
# ==========================================================

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

# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------
IMG_SIZE = 224
THRESHOLD = 0.35  # MCC-safe ROC threshold

# Class semantics (LOCKED from training)
IDX_TO_CLASS = {0: "MCC", 1: "Non-MCC"}

# ----------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------
st.set_page_config(
    page_title="MCC Diagnostic System",
    page_icon="🩺",
    layout="centered"
)

# ----------------------------------------------------------
# HEADER
# ----------------------------------------------------------
st.title("🩺 MCC Diagnostic System")
st.subheader("AI-Assisted Skin Lesion Risk Assessment")

st.markdown("""
⚠️ **Academic Research Prototype – NOT a Medical Device**

This application demonstrates **deep learning–based skin lesion analysis**
for **Merkel Cell Carcinoma (MCC)** as part of an **academic MSc research project**.
""")

st.divider()

# ----------------------------------------------------------
# ABOUT MCC
# ----------------------------------------------------------
st.header("📘 About Merkel Cell Carcinoma (MCC)")
st.markdown("""
Merkel Cell Carcinoma (MCC) is a **rare but highly aggressive neuroendocrine skin cancer**.

**Key characteristics:**
- Rapid growth
- Often painless
- Appears on sun-exposed areas
- High risk of early metastasis

➡️ **Early detection is critical for patient survival.**
""")

# ----------------------------------------------------------
# ABOUT AI & ViT
# ----------------------------------------------------------
st.divider()
st.header("🧠 AI & Vision Transformer (ViT) in Skin Cancer Analysis")

st.markdown("""
Modern dermatological AI systems rely on **deep learning models**
to identify subtle visual patterns in skin lesions.

### 🔬 Vision Transformer (ViT)
Vision Transformers treat an image as a **sequence of patches**, similar to words in NLP.

**Advantages over CNNs:**
- Captures **global context**
- Learns long-range dependencies
- Less inductive bias than convolutions

**Typical ViT Pipeline:**
1. Image → fixed-size patches  
2. Linear embedding + positional encoding  
3. Transformer encoder blocks  
4. Classification head  

📌 *This project combines CNN efficiency with Transformer-inspired reasoning concepts.*
""")

# ----------------------------------------------------------
# LOAD MODEL (KERAS 3 SAFE – TFSMLayer)
# ----------------------------------------------------------
@st.cache_resource
def load_model():
    model_dir = "mcc_model_savedmodel"
    if not os.path.exists(model_dir):
        st.error("❌ Model folder not found in repository.")
        return None

    return tf.keras.layers.TFSMLayer(
        model_dir,
        call_endpoint="serving_default"
    )

model = load_model()
if model is None:
    st.stop()

# ----------------------------------------------------------
# IMAGE UPLOAD
# ----------------------------------------------------------
st.divider()
st.header("📤 Upload Skin Lesion Image")

uploaded_file = st.file_uploader(
    "Accepted formats: JPG, JPEG, PNG",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ------------------------------------------------------
    # PREPROCESS
    # ------------------------------------------------------
    img_resized = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    # ------------------------------------------------------
    # SIMPLE FEATURE VISUALIZATION (NOT USED FOR MODEL)
    # ------------------------------------------------------
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lesion_area, perimeter, circularity = 0, 0, 0
    if contours:
        c = max(contours, key=cv2.contourArea)
        lesion_area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        if perimeter != 0:
            circularity = (4 * np.pi * lesion_area) / (perimeter ** 2)

    st.divider()
    st.subheader("🧬 Extracted Visual Indicators (Illustrative)")

    c1, c2, c3 = st.columns(3)
    c1.metric("Area", f"{lesion_area:.0f} px²")
    c2.metric("Perimeter", f"{perimeter:.2f} px")
    c3.metric("Circularity", f"{circularity:.3f}")

    st.caption("These features are **for visualization only** and not directly used by the model.")

    # ------------------------------------------------------
    # PREDICTION
    # ------------------------------------------------------
    st.divider()
    st.subheader("📊 Model Prediction")

    with st.spinner("Analyzing lesion using deep learning model..."):
        output = model(img_array)
        prob_non_mcc = float(list(output.values())[0][0][0])

    prob_mcc = 1.0 - prob_non_mcc

    # ------------------------------------------------------
    # RISK INTERPRETATION
    # ------------------------------------------------------
    if prob_mcc >= 0.65:
        risk = "HIGH"
        icon = "🚨"
        st.error("High-risk MCC probability detected.")
    elif prob_mcc >= THRESHOLD:
        risk = "MEDIUM"
        icon = "⚠️"
        st.warning("Moderate MCC risk detected.")
    else:
        risk = "LOW"
        icon = "✅"
        st.success("Low MCC risk detected.")

    col1, col2 = st.columns(2)
    col1.metric("MCC Probability", f"{prob_mcc * 100:.2f}%")
    col2.metric("Benign Probability", f"{prob_non_mcc * 100:.2f}%")

    st.markdown(f"### {icon} **Overall Risk Level: {risk}**")

    # ------------------------------------------------------
    # PDF REPORT
    # ------------------------------------------------------
    st.divider()
    st.subheader("📄 Generate Academic Diagnostic Report")

    def generate_pdf():
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        pdf.cell(0, 10, "MCC Diagnostic Report (Academic Prototype)", ln=True)
        pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
        pdf.ln(5)

        pdf.multi_cell(
            0, 8,
            "DISCLAIMER:\n"
            "This report is generated by an academic AI system.\n"
            "It is NOT intended for clinical diagnosis.\n"
        )

        pdf.ln(5)
        pdf.cell(0, 8, f"MCC Probability: {prob_mcc:.4f}", ln=True)
        pdf.cell(0, 8, f"Benign Probability: {prob_non_mcc:.4f}", ln=True)
        pdf.cell(0, 8, f"Risk Level: {risk}", ln=True)

        pdf.ln(5)
        pdf.cell(0, 8, "Visual Indicators:", ln=True)
        pdf.cell(0, 8, f"Area: {lesion_area:.0f}", ln=True)
        pdf.cell(0, 8, f"Perimeter: {perimeter:.2f}", ln=True)
        pdf.cell(0, 8, f"Circularity: {circularity:.3f}", ln=True)

        return pdf.output(dest="S").encode("latin-1")

    st.download_button(
        "📥 Download PDF Report",
        data=generate_pdf(),
        file_name="MCC_AI_Report.pdf",
        mime="application/pdf"
    )

# ----------------------------------------------------------
# FOOTER
# ----------------------------------------------------------
st.markdown("---")
st.caption(
    "© MSc Research Prototype | AI-Assisted MCC Detection | "
    "CNN + Transformer-Inspired Design"
)
