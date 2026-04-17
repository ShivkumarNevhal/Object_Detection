import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

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
# YOLO Live Processor
# -------------------------------
class YOLOProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model(img, conf=0.4)
        annotated = results[0].plot()
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

# -------------------------------
# LEFT SIDE (Upload + Live Camera)
# -------------------------------
with col1:
    st.markdown('<div class="main-box">', unsafe_allow_html=True)

    # Upload
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader("Choose image", type=["jpg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img)
        st.session_state["upload_image"] = img

    # -------------------------------
    # Live Camera (REAL-TIME)
    # -------------------------------
    st.subheader("🎥 Live Detection Camera")

    ctx = webrtc_streamer(
        key="leaf-detect",
        video_processor_factory=YOLOProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# RIGHT SIDE (Detection Result)
# -------------------------------
with col2:
    st.markdown('<div class="main-box">', unsafe_allow_html=True)

    st.subheader("🧠 Detection Result")

    input_img = None

    if "upload_image" in st.session_state:
        input_img = st.session_state["upload_image"]

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
