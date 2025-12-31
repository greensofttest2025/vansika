# ===============================
# Streamlit App – MCC Detection
# ===============================

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -------------------------------
# CONFIG
# -------------------------------
IMG_SIZE = 224
THRESHOLD = 0.35   # MCC-safe ROC threshold

# From training: {'mcc': 0, 'non_mcc': 1}
IDX_TO_CLASS = {
    0: "MCC",
    1: "Non-MCC"
}

# -------------------------------
# PAGE SETTINGS
# -------------------------------
st.set_page_config(
    page_title="MCC Risk Assessment",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 MCC Risk Assessment")
st.markdown("""
⚠️ **Academic Research Prototype**  
This tool is **NOT a medical device**.  
Always consult a dermatologist.
""")

# -------------------------------
# LOAD MODEL (KERAS 3 SAFE)
# -------------------------------
@st.cache_resource
def load_model():
    model_dir = "mcc_model_savedmodel"

    if not os.path.exists(model_dir):
        st.error("❌ Model folder not found.")
        return None

    # Load SavedModel as inference-only layer
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
uploaded_file = st.file_uploader(
    "Upload a skin lesion image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    # Predict
    with st.spinner("Analyzing image..."):
        output = model(img_array)

        # TFSMLayer returns a dict → extract tensor
        prob_non_mcc = float(list(output.values())[0][0][0])

    # -------------------------------
    # CORRECT CLASS LOGIC (FIXED)
    # -------------------------------
    prob_mcc = 1.0 - prob_non_mcc

    if prob_mcc >= THRESHOLD:
        st.error("🚨 **MCC (Suspicious for Cancer)**")
        st.metric("Confidence", f"{prob_mcc * 100:.2f}%")
        st.warning("HIGH RISK – Consult dermatologist immediately")
    else:
        st.success("✅ **Non-MCC (Likely Benign)**")
        st.metric("Confidence", f"{prob_non_mcc * 100:.2f}%")
        st.info("LOW RISK – Routine monitoring suggested")

    # -------------------------------
    # TECH DETAILS
    # -------------------------------
    with st.expander("🔍 Technical Details"):
        st.write(f"Sigmoid output (P Non-MCC): {prob_non_mcc:.4f}")
        st.write(f"MCC probability: {prob_mcc:.4f}")
        st.write(f"Decision threshold: {THRESHOLD}")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.caption("Academic research prototype · Not approved for clinical use")
