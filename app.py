# ===============================
# MCC Diagnostic App (Streamlit)
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
THRESHOLD = 0.35   # ROC-optimized MCC-safe threshold

# Model semantics (LOCKED)
# Sigmoid output = P(non_mcc)
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
⚠️ **Academic Research Prototype – NOT for Clinical Use**  
This system is intended strictly for **research and educational purposes**.  
Final diagnosis must always be performed by a certified dermatologist.
""")

st.divider()

# -------------------------------
# ABOUT MCC
# -------------------------------
st.header("📘 About Merkel Cell Carcinoma (MCC)")
st.markdown("""
Merkel Cell Carcinoma (MCC) is a **rare but aggressive neuroendocrine skin cancer**.
It typically presents as a **rapidly growing, painless lesion** on sun-exposed skin.

Early detection is critical due to:
- High metastatic potential
- Rapid progression
- Poor prognosis if untreated
""")

st.divider()

# -------------------------------
# ABOUT AI / ViT
# -------------------------------
st.header("🧠 AI & Vision Transformer (ViT) Perspective")
st.markdown("""
This system uses **deep learning–based visual representation learning**.

### Key Concepts:
- **CNN backbone** extracts local texture patterns
- **Transformer-inspired attention** models global lesion structure
- **Sigmoid probability output** estimates malignancy risk
- **ROC-calibrated threshold** prioritizes cancer safety

> The model learns *where to look* and *what patterns matter most* in skin lesions.
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

    # Keras 3 inference-safe loader
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
    # BASIC VISUAL FEATURES (UI ONLY)
    # -------------------------------
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    lesion_area = perimeter = circularity = 0.0

    if contours:
        c = max(contours, key=cv2.contourArea)
        lesion_area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        if perimeter > 0:
            circularity = (4 * np.pi * lesion_area) / (perimeter ** 2)

    st.divider()

    # -------------------------------
    # DISPLAY FEATURES
    # -------------------------------
    st.header("🧬 Extracted Visual Indicators")
    c1, c2, c3 = st.columns(3)

    c1.metric("Lesion Area", f"{lesion_area:.0f} px²")
    c2.metric("Perimeter", f"{perimeter:.2f} px")
    c3.metric("Circularity", f"{circularity:.3f}")

    st.caption("⚠️ Feature values are approximate and for visualization only.")

    st.divider()

    # -------------------------------
    # PREDICTION
    # -------------------------------
    with st.spinner("Analyzing lesion..."):
        output = model(img_array)

        # SavedModel returns dict
        prob_non_mcc = float(list(output.values())[0][0][0])

    prob_mcc = 1.0 - prob_non_mcc

    # -------------------------------
    # RISK INTERPRETATION
    # -------------------------------
    if prob_mcc >= 0.65:
        risk = "HIGH"
        icon = "🚨"
    elif prob_mcc >= THRESHOLD:
        risk = "MEDIUM"
        icon = "⚠️"
    else:
        risk = "LOW"
        icon = "✅"

    st.header("📊 Prediction Results")

    r1, r2 = st.columns(2)
    r1.metric("MCC Probability", f"{prob_mcc * 100:.2f}%")
    r2.metric("Benign Probability", f"{prob_non_mcc * 100:.2f}%")

    st.markdown(f"### {icon} **Risk Level: {risk}**")

    if risk == "HIGH":
        st.error("Immediate dermatological evaluation is strongly advised.")
    elif risk == "MEDIUM":
        st.warning("Clinical review is recommended.")
    else:
        st.success("Routine monitoring suggested.")

    st.divider()

    # -------------------------------
    # PDF REPORT
    # -------------------------------
    st.header("📄 Generate Diagnostic Report")

    def generate_pdf():
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "MCC Diagnostic Report", ln=True, align="C")

        pdf.set_font("Arial", "", 11)
        pdf.cell(0, 8, "AI-Assisted Skin Lesion Risk Assessment", ln=True, align="C")
        pdf.ln(4)

        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(6)

        pdf.set_font("Arial", "", 10)
        pdf.cell(0, 8, f"Date: {datetime.now().strftime('%d %b %Y | %H:%M')}", ln=True)
        pdf.cell(0, 8, "System: Academic Research Prototype", ln=True)
        pdf.ln(4)

        pdf.set_font("Arial", "B", 10)
        pdf.cell(0, 8, "DISCLAIMER", ln=True)

        pdf.set_font("Arial", "", 9)
        pdf.multi_cell(
            0, 6,
            "This AI-generated report is NOT a clinical diagnosis. "
            "It must not be used for medical decision-making."
        )
        pdf.ln(4)

        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Prediction Summary", ln=True)

        pdf.set_font("Arial", "", 10)
        pdf.cell(90, 8, "MCC Probability:", 0, 0)
        pdf.cell(0, 8, f"{prob_mcc * 100:.2f} %", ln=True)

        pdf.cell(90, 8, "Benign Probability:", 0, 0)
        pdf.cell(0, 8, f"{prob_non_mcc * 100:.2f} %", ln=True)

        pdf.cell(90, 8, "Risk Level:", 0, 0)
        pdf.set_font("Arial", "B", 10)
        pdf.cell(0, 8, risk, ln=True)

        pdf.ln(4)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Extracted Visual Indicators", ln=True)

        pdf.set_font("Arial", "", 10)
        pdf.cell(90, 8, "Lesion Area:", 0, 0)
        pdf.cell(0, 8, f"{lesion_area:.0f}", ln=True)

        pdf.cell(90, 8, "Perimeter:", 0, 0)
        pdf.cell(0, 8, f"{perimeter:.2f}", ln=True)

        pdf.cell(90, 8, "Circularity:", 0, 0)
        pdf.cell(0, 8, f"{circularity:.3f}", ln=True)

        pdf.ln(5)
        pdf.set_font("Arial", "I", 8)
        pdf.multi_cell(
            0, 5,
            "Generated by MCC Diagnostic System\n"
            "MSc Research Prototype – AI-Assisted Skin Cancer Analysis"
        )

        return pdf.output(dest="S").encode("latin-1")

    pdf_bytes = generate_pdf()

    st.download_button(
        "📥 Download Diagnostic Report (PDF)",
        data=pdf_bytes,
        file_name="MCC_Diagnostic_Report.pdf",
        mime="application/pdf"
    )

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.caption("© MSc Research Prototype · AI-Assisted MCC Risk Assessment")
