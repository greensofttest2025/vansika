# ================================
# Streamlit MCC Detection App
# ================================

import os

# --- MUST be before tensorflow import ---
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from keras.layers import TFSMLayer

# ================================
# App Config
# ================================

st.set_page_config(
    page_title="MCC Detection",
    layout="centered"
)

st.title("🩺 MCC Skin Cancer Detection")
st.write("Upload a skin lesion image to classify **MCC / Non-MCC**")

# ================================
# Constants
# ================================

MODEL_PATH = "mcc_model_savedmodel"
IMG_SIZE = 224
THRESHOLD = 0.5

IDX_TO_CLASS = {
    0: "Non-MCC",
    1: "MCC"
}

# ================================
# Load Model (Keras 3 Compatible)
# ================================

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model folder not found.")
        return None

    try:
        model = TFSMLayer(
            MODEL_PATH,
            call_endpoint="serving_default"
        )
        return model
    except Exception as e:
        st.error(f"❌ Model load failed: {e}")
        return None


model = load_model()
if model is None:
    st.stop()

# ================================
# Image Preprocessing
# ================================

def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = np.array(image)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0
    image = np.expand_dims(image, axis=0).astype(np.float32)
    return image

# ================================
# UI
# ================================

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_array = preprocess_image(image)

    # ================================
    # Prediction
    # ================================
    with st.spinner("🔬 Analyzing image..."):
        outputs = model(img_array, training=False)

        # TFSMLayer returns dict → extract tensor
        prob = float(list(outputs.values())[0].numpy()[0][0])

    predicted_index = 1 if prob >= THRESHOLD else 0
    predicted_class = IDX_TO_CLASS[predicted_index]

    # ================================
    # Results
    # ================================
    st.markdown("---")
    st.subheader("🧪 Prediction Result")

    if predicted_class == "MCC":
        st.error(f"⚠️ **MCC Detected**\n\nConfidence: **{prob*100:.2f}%**")
    else:
        st.success(f"✅ **Non-MCC**\n\nConfidence: **{(1-prob)*100:.2f}%**")


