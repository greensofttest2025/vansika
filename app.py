# ===============================
# Streamlit App - MCC Detection
# ===============================

import streamlit as st
import tensorflow as tf
import numpy as np

import os
from PIL import Image
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# -------------------------------
# CONFIG
# -------------------------------
IMG_SIZE = 224
THRESHOLD = 0.35   # cancer-safe threshold (from ROC)

# 🔒 CLASS MAPPING (SINGLE SOURCE OF TRUTH)
# From training: {'mcc': 0, 'non_mcc': 1}
IDX_TO_CLASS = {0: "mcc", 1: "non_mcc"}

# -------------------------------
# PAGE SETTINGS
# -------------------------------
st.set_page_config(
    page_title="MCC Risk Assessment",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 MCC Risk Assessment (Academic Prototype)")
st.markdown("""
⚠️ **For academic research only**  
This tool is **NOT a medical device**.  
Always consult a dermatologist for diagnosis.
""")

# -------------------------------
# LOAD MODEL (ONCE)
# -------------------------------
@st.cache_resource
def load_model():
    model_path = "mcc_model_savedmodel"   # folder
    if not os.path.exists(model_path):
        st.error("❌ Model folder not found.")
        return None
    return tf.keras.models.load_model(model_path)

model = load_model()
if model is None:
    st.stop()


# -------------------------------
# IMAGE UPLOAD
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload a skin lesion image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file and model:

    # Show image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    with st.spinner("Analyzing image..."):
        prob = float(model.predict(img_array, verbose=0)[0][0])

    # -------------------------------
    # CORRECT CLASS INTERPRETATION
    # -------------------------------
    predicted_index = 1 if prob >= THRESHOLD else 0
    predicted_class = IDX_TO_CLASS[predicted_index]

    if predicted_class == "mcc":
        label = "🚨 MCC (Suspicious for Cancer)"
        confidence = (1 - prob) * 100
        st.error(label)
        st.metric("Confidence", f"{confidence:.2f}%")
        st.warning("HIGH RISK – Consult dermatologist immediately")
    else:
        label = "✅ Non-MCC (Likely Benign)"
        confidence = prob * 100
        st.success(label)
        st.metric("Confidence", f"{confidence:.2f}%")
        st.info("LOW RISK – Routine monitoring suggested")

    # Technical details
    with st.expander("🔍 Technical Details"):
        st.write(f"Raw sigmoid output: {prob:.4f}")
        st.write("Sigmoid = P(class=1) = P(non_mcc)")
        st.write(f"Decision threshold: {THRESHOLD}")

# -------------------------------
# FOOTER DISCLAIMER
# -------------------------------
st.markdown("---")
st.caption("""
This system is an **academic research prototype**.  
It is **not approved for clinical use**.
""")
