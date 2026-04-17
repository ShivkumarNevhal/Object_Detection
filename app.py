import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Leaf Detection 🌿",
    layout="centered"
)

st.title("🌿 Leaf Detection System")

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")   # keep model in root folder

model = load_model()

# -------------------------------
# Upload Image
# -------------------------------
uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "png"])

if uploaded_file:
    input_img = Image.open(uploaded_file)

    st.image(input_img, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Detect"):

        # Convert image
        img_np = np.array(input_img)

        # Resize (VERY IMPORTANT)
        img_np = cv2.resize(img_np, (640, 640))

        # Reduce noise
        img_np = cv2.GaussianBlur(img_np, (5, 5), 0)

        # Prediction (higher confidence)
        results = model(img_np, conf=0.6)

        # Result image
        result_img = results[0].plot()
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

        st.image(result_img, caption="Detection Result", use_column_width=True)

        # Results
        st.subheader("Prediction")

        found = False
        for box in results[0].boxes:
            conf = float(box.conf[0])

            if conf > 0.6:
                cls = int(box.cls[0])
                st.success(f"{model.names[cls]} ({conf:.2f})")
                found = True

        if not found:
            st.warning("No clear detection 🚫")
