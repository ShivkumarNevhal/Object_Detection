import numpy as np
import cv2
print("NumPy version:", np.__version__)
print("CV2 loaded successfully")
import streamlit as st
from ultralytics import YOLO
from PIL import Image

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
# Deficiency Info
# -------------------------------
deficiency_info = {
    "Nitrogen Deficiency": {
        "symptoms": "Yellowing of older leaves, slow growth.",
        "treatment": "Apply urea or compost. Use 1-2% urea foliar spray."
    },
    "Boron Deficiency": {
        "symptoms": "Deformed leaves, brittle tissues, poor growth.",
        "treatment": "Apply borax in small quantity or 0.1–0.2% boric acid spray."
    },
    "Potassium Deficiency": {
        "symptoms": "Yellow/brown leaf edges (leaf burn), weak stems.",
        "treatment": "Use muriate of potash (KCl) or wood ash."
    },
    "Magnesium Deficiency": {
        "symptoms": "Yellowing between veins (interveinal chlorosis) in older leaves.",
        "treatment": "Apply magnesium sulfate (Epsom salt) spray (1–2%)."
    },
    "Healthy": {
        "symptoms": "Leaves are green and healthy with no deficiency signs.",
        "treatment": "No treatment needed. Maintain proper care."
    }
}

# -------------------------------
# Layout
# -------------------------------
col1, col2 = st.columns(2)

# -------------------------------
# LEFT SIDE (Input)
# -------------------------------
with col1:
    st.markdown('<div class="main-box">', unsafe_allow_html=True)

    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader("Choose image", type=["jpg", "png"])

    if uploaded_file:
        st.session_state["input_image"] = Image.open(uploaded_file)

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

    if input_img is not None:

        if st.button("🔍 Run Detection"):

            # -------------------------------
            # STANDARD INPUT PIPELINE
            # -------------------------------
            img_np = np.array(input_img)
            img_np = img_np[:, :, ::-1]

            img_np = cv2.resize(img_np, (640, 640))
            img_np = cv2.GaussianBlur(img_np, (5, 5), 0)

            # -------------------------------
            # Run Model
            # -------------------------------
            results = model(img_np, conf=0.6)

            result_img = results[0].plot()
            result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

            # -------------------------------
            # Show Images
            # -------------------------------
            st.subheader("📊 Before vs After Detection")

            colA, colB = st.columns(2)

            with colA:
                st.image(input_img, caption="Input Image")

            with colB:
                st.image(result_img, caption="Detected Image")

            # -------------------------------
            # Show Results
            # -------------------------------
            st.subheader("Results")

            boxes = results[0].boxes

            if boxes is None or len(boxes) == 0:
                st.warning("No objects detected 🚫")
            else:
                best_box = max(boxes, key=lambda x: float(x.conf[0]))

                conf = float(best_box.conf[0])
                cls = int(best_box.cls[0])

                if conf > 0.6:
                    detected_class = model.names[cls]

                    st.success(f"{detected_class} → {conf:.2f}")

                    # -------------------------------
                    # Symptoms & Treatment
                    # -------------------------------
                    info = deficiency_info.get(detected_class, None)

                    if info:
                        st.markdown("### 🌿 Symptoms")
                        st.write(info["symptoms"])

                        st.markdown("### 💊 Treatment")
                        st.write(info["treatment"])
                    else:
                        st.info("No additional information available.")
                else:
                    st.warning("Low confidence detection ⚠️")

    else:
        st.info("Upload or capture an image first 👆")

    st.markdown('</div>', unsafe_allow_html=True)
