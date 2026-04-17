import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

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
# LEFT SIDE (Upload + Camera)
# -------------------------------
with col1:
    st.markdown('<div class="main-box">', unsafe_allow_html=True)

    # Upload
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader("Choose image", type=["jpg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img)

    # -------------------------------
    # Camera Input (Cloud Compatible)
    # -------------------------------
    st.subheader("📷 Capture from Camera")

    camera_image = st.camera_input("Take a photo")

    if camera_image:
        cam_img = Image.open(camera_image)
        st.image(cam_img)
        st.session_state["camera_image"] = cam_img

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# RIGHT SIDE (Detection)
# -------------------------------
with col2:
    st.markdown('<div class="main-box">', unsafe_allow_html=True)

    st.subheader("🧠 Detection Result")

    input_img = None

    if uploaded_file:
        input_img = Image.open(uploaded_file)

    elif "camera_image" in st.session_state:
        input_img = st.session_state["camera_image"]

    if input_img:
        st.image(input_img, caption="Input Image")

        if st.button("🔍 Run Detection"):
            img_np = np.array(input_img)

            results = model(img_np, conf=0.4)

            result_img = results[0].plot()
            st.image(result_img, caption="Detected")

            st.subheader("Results")

            if len(results[0].boxes) == 0:
                st.warning("No objects detected")
            else:
                for box in results[0].boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    st.write(f"{model.names[cls]} → {conf:.2f}")

    st.markdown('</div>', unsafe_allow_html=True)
