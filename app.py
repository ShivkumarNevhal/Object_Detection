import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

st.set_page_config(page_title="Leaf Detection 🌿", layout="centered")
st.title("🌿 Leaf Detection System")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "png"])

if uploaded_file:
    input_img = Image.open(uploaded_file)
    st.image(input_img, caption="Input Image", use_column_width=True)

    if st.button("🔍 Detect"):

        img_np = np.array(input_img)

        # Resize (critical)
        img_np = cv2.resize(img_np, (640, 640))

        # Normalize lighting (IMPORTANT FIX)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(img_np)
        l = cv2.equalizeHist(l)
        img_np = cv2.merge((l, a, b))
        img_np = cv2.cvtColor(img_np, cv2.COLOR_LAB2RGB)

        # Run model with high confidence
        results = model(img_np, conf=0.7)

        result_img = results[0].plot()
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

        st.image(result_img, caption="Detection Result")

        # -------------------------------
        # TAKE ONLY BEST PREDICTION
        # -------------------------------
        st.subheader("Prediction")

        if len(results[0].boxes) == 0:
            st.error("No leaf detected ❌")

        else:
            # Get highest confidence box
            best_box = max(results[0].boxes, key=lambda x: float(x.conf[0]))

            conf = float(best_box.conf[0])
            cls = int(best_box.cls[0])

            if conf > 0.7:
                st.success(f"{model.names[cls]} ({conf:.2f})")
            else:
                st.warning("Prediction not confident ⚠️")
