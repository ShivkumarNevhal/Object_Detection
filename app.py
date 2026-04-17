import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Leaf Detection AI 🌿",
    page_icon="🌿",
    layout="wide"
)

# -------------------------------
# UI
# -------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(rgba(0,0,0,0.4), rgba(0,0,0,0.5)),
                url("https://www.shaip.com/wp-content/uploads/2022/01/Blog_Most-Sought-Open-Source-Datasets-for-Computer-Vision.jpg");
    background-size: cover;
    background-position: center;
}
.main-box {
    background: rgba(0, 0, 0, 0.55);
    backdrop-filter: blur(12px);
    padding: 25px;
    border-radius: 20px;
}
h1 { color: #a5d6a7; text-align: center; }
h2, h3 { color: #c8e6c9; }
label, p { color: #eee !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🌿 Leaf Detection AI System</h1>", unsafe_allow_html=True)

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# -------------------------------
# Layout
# -------------------------------
col1, col2 = st.columns(2)

# -------------------------------
# LEFT SIDE (Input)
# -------------------------------
with col1:
    st.markdown('<div class="main-box">', unsafe_allow_html=True)

    # Upload
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader("Choose image", type=["jpg", "png"])

    if uploaded_file:
        st.session_state["input_image"] = Image.open(uploaded_file)

    # -------------------------------
    # Camera + Capture (Stable Version)
    # -------------------------------
    st.subheader("📷 Camera")

    cam_on = st.toggle("Turn Camera ON / OFF")

    if cam_on:
        camera_image = st.camera_input("Take a picture")

        if camera_image:
            st.session_state["input_image"] = Image.open(camera_image)
            st.success("Image Captured ✅")
    else:
        st.info("Camera is OFF")

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# RIGHT SIDE (Detection)
# -------------------------------
with col2:
    st.markdown('<div class="main-box">', unsafe_allow_html=True)

    st.subheader("🧠 Detection Result")

    input_img = st.session_state.get("input_image", None)

    if input_img:

       if st.button("🔍 Run Detection"):

         # Convert image (KEEP RGB)
         img_np = np.array(input_img)

        # Resize to model size (VERY IMPORTANT)
        img_np = cv2.resize(img_np, (640, 640))

        # Run detection with higher confidence
        results = model(img_np, conf=0.6)
 
        # Get plotted result
        result_img = results[0].plot()
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

    # -------------------------------
    # SHOW CLEAN OUTPUT
    # -------------------------------
        colA, colB = st.columns(2)

        with colA:
           st.image(input_img, caption="Input Image")

        with colB:
           st.image(result_img, caption="Detected Image")

    # -------------------------------
    # FILTER RESULTS (VERY IMPORTANT)
    # -------------------------------
        st.subheader("Results")

        valid = False

        for box in results[0].boxes:
            conf = float(box.conf[0])

        # Only show strong detections
             if conf > 0.6:
               cls = int(box.cls[0])
               st.write(f"{model.names[cls]} → {conf:.2f}")
               valid = True

        if not valid:
             st.warning("No confident detection 🚫")

    st.markdown('</div>', unsafe_allow_html=True)
