# ===============================
# Streamlit App - MCC Detection
# Compatible with Python 3.13 + TF 2.20 + Keras 3
# ===============================

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from keras.layers import TFSMLayer
from keras.models import Sequential

# -------------------------------
# CONFIG
# -------------------------------
IMG_SIZE = 224
THRESHOLD = 0.35  # cancer-safe threshold

IDX_TO_CLASS = {
    0: "mcc",
    1: "non_mcc"
}

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
# LOAD MODEL (KERAS 3 SAFE)
# -------------------------------
@st.cache_resource
def load_model():
    model_dir = "mcc_model_savedmodel"

    if not os.path.exists(model_dir):
        st.error("❌ Model folder not found.")
        return None

    try:
        tfsm_layer = TFSMLayer(
            model_dir,
            call_endpoint="serving_default"
        )

        model = Sequential([tfsm_layer])
        return model

    except Exception as e:
        st.error(f"❌ Model load failed: {e}")
        return None


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
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    with st.spinner("Analyzing image..."):
        prob = float(model(img_array, training=False).numpy()[0][0])

    predicted_index = 1 if prob >= THRESHOLD else 0
    predicted_class = IDX_TO_CLASS[predicted_index]

    if predicted_class == "mcc":
        st.error("🚨 MCC (Suspicious for Cancer)")
        st.metric("Confidence", f"{(1 - prob) * 100:.2f}%")
        st.warning("HIGH RISK – Consult dermatologist immediately")
    else:
        st.success("✅ Non-MCC (Likely Benign)")
        st.metric("Confidence", f"{prob * 100:.2f}%")
        st.info("LOW RISK – Routine monitoring suggested")

    with st.expander("🔍 Technical Details"):
        st.write(f"Raw model output: {prob:.4f}")
        st.write("Output = P(non_mcc)")
        st.write(f"Decision threshold: {THRESHOLD}")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.caption("""
This system is an **academic research prototype**.  
It is **not approved for clinical use**.
""")
