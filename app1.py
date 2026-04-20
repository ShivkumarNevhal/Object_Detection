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

st.markdown("<h1>🌿 AI-Powered Palm Leaf Nutrient Deficiency Detection System</h1>", unsafe_allow_html=True)

# -------------------------------
# 🌐 Language Selector (NEW)
# -------------------------------
language = st.selectbox(
    "🌐 Select Language",
    ["en", "mr", "te"],
    format_func=lambda x: {
        "en": "English",
        "mr": "Marathi",
        "te": "Telugu"
    }[x]
)

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# -------------------------------
# Deficiency Info (FIXED)
# -------------------------------
deficiency_info = {

    "nitrogen": {
        "symptoms": {
            "en": "Yellowing of older leaves, slow growth, reduced leaf size.",
            "mr": "जुन्या पानांचे पिवळे पडणे, वाढ मंद होणे, पानांचा आकार कमी होणे",
            "te": "పాత ఆకులు పసుపు రంగులో మారడం, వృద్ధి నెమ్మదిగా ఉండడం, ఆకుల పరిమాణం తగ్గడం"
        },
        "treatment": {
            "en": """### 🌿 Treatment Steps
- Apply urea or compost to improve nitrogen levels in the soil.
- Use 1–2% urea foliar spray (10–20 g per liter of water) for quick recovery.
- Spray evenly on both sides of leaves.
- Apply during early morning or late evening to avoid leaf burn.
- Repeat every 10–15 days if symptoms persist.
- Add organic manure to improve long-term soil fertility.
- Ensure proper irrigation after fertilizer application.
- Avoid excessive urea use to prevent soil imbalance.
""",
            "mr": """### 🌿 उपचार पायऱ्या
- मातीतील नायट्रोजन वाढवण्यासाठी युरिया किंवा कंपोस्ट वापरा.
- जलद परिणामांसाठी 1–2% युरिया फवारणी करा.
- पानांच्या दोन्ही बाजूंवर फवारणी करा.
- सकाळी किंवा संध्याकाळी फवारणी करा.
- 10–15 दिवसांनी पुन्हा करा.
- सेंद्रिय खत वापरा.
- योग्य प्रमाणात पाणी द्या.
- अति वापर टाळा.
""",
            "te": """### 🌿 చికిత్స దశలు
- నేలలో నైట్రోజన్ పెంచడానికి యూరియా లేదా కంపోస్ట్ ఉపయోగించండి.
- త్వరిత ఫలితాల కోసం 1–2% యూరియా స్ప్రే చేయండి.
- ఆకుల రెండు వైపులా స్ప్రే చేయండి.
- ఉదయం లేదా సాయంత్రం చేయండి.
- 10–15 రోజులకు మళ్లీ చేయండి.
- ఆర్గానిక్ ఎరువు జోడించండి.
- సరైన నీరు ఇవ్వండి.
- అధిక వినియోగం నివారించండి.
"""
        }
    },

    "boron": {
        "symptoms": {
            "en": "Deformed leaves, brittle tissues, poor root and shoot growth.",
            "mr": "वाकडी पाने, नाजूक ऊती, मुळांची व वाढ कमी",
            "te": "వంకర ఆకులు, బలహీన కణజాలం, వేర్లు మరియు పెరుగుదల తగ్గడం"
        },
        "treatment": {
            "en": """### 🌿 Treatment Steps
- Apply borax in small quantity to soil (very low dose required).
- Use 0.1–0.2% boric acid foliar spray (1–2 g per liter of water).
- Spray evenly on leaves.
- Apply during cool hours (morning/evening).
- Repeat every 10–15 days if needed.
- Avoid overapplication as boron toxicity can damage plants.
""",
            "mr": """### 🌿 उपचार पायऱ्या
- कमी प्रमाणात बोरेक्स वापरा.
- 0.1–0.2% बोरिक अॅसिड फवारणी करा.
- पानांवर समप्रमाणात फवारणी करा.
- सकाळी/संध्याकाळी करा.
- 10–15 दिवसांनी पुन्हा करा.
- अति वापर टाळा.
""",
            "te": """### 🌿 చికిత్స దశలు
- తక్కువ పరిమాణంలో బోరాక్స్ ఉపయోగించండి.
- 0.1–0.2% బోరిక్ ఆమ్ల స్ప్రే చేయండి.
- ఆకులపై సమానంగా స్ప్రే చేయండి.
- ఉదయం/సాయంత్రం చేయండి.
- 10–15 రోజులకు మళ్లీ చేయండి.
- అధిక వినియోగం నివారించండి.
"""
        }
    },

    "kalium": {
        "symptoms": {
            "en": "Yellow/brown leaf edges (leaf burn), weak stems, reduced resistance to stress.",
            "mr": "पानांच्या कडा पिवळ्या/तपकिरी, खोड कमकुवत, ताण सहनशक्ती कमी",
            "te": "ఆకుల అంచులు పసుపు/గోధుమ, కాండం బలహీనంగా, ఒత్తిడి నిరోధకత తగ్గడం"
        },
        "treatment": {
            "en": """### 🌿 Treatment Steps
- Apply muriate of potash (KCl) to soil to increase potassium levels.
- Use wood ash as a natural potassium source.
- Mix fertilizer properly into soil near plant roots.
- Water immediately after application.
- Avoid direct contact with roots.
- Repeat as required depending on deficiency.
- Do not overapply as excess potassium affects nutrient balance.
""",
            "mr": """### 🌿 उपचार पायऱ्या
- मातीमध्ये पोटॅश (KCl) वापरा.
- पर्याय म्हणून राख वापरा.
- मातीमध्ये चांगले मिसळा.
- लगेच पाणी द्या.
- मुळांशी थेट संपर्क टाळा.
- गरजेनुसार पुन्हा करा.
- अति वापर टाळा.
""",
            "te": """### 🌿 చికిత్స దశలు
- నేలలో పొటాష్ (KCl) ఉపయోగించండి.
- చెక్క బూడిదను ఉపయోగించండి.
- నేలలో బాగా కలపండి.
- వెంటనే నీరు పోయండి.
- వేర్లకు తగలకుండా చూడండి.
- అవసరమైతే మళ్లీ చేయండి.
- అధిక వినియోగం నివారించండి.
"""
        }
    },

    "magnesium": {
        "symptoms": {
            "en": "Yellowing between veins (interveinal chlorosis) in older leaves.",
            "mr": "जुन्या पानांमध्ये शिरांदरम्यान पिवळेपणा",
            "te": "పాత ఆకులలో నరాల మధ్య పసుపు రంగు మారడం"
        },
        "treatment": {
            "en": """### 🌿 Treatment Steps
- Apply magnesium sulfate (Epsom salt) as 1–2% foliar spray.
- Mix 10–20 g per liter of water.
- Spray evenly on both sides of leaves.
- Apply during morning or evening.
- Repeat every 10–15 days if needed.
- Can also apply to soil around plant base.
- Water after application.
- Avoid overuse as it may affect calcium/potassium balance.
""",
            "mr": """### 🌿 उपचार पायऱ्या
- एप्सम सॉल्ट 1–2% फवारणी करा.
- पाण्यात मिसळा.
- पानांवर फवारणी करा.
- सकाळी/संध्याकाळी वापरा.
- 10–15 दिवसांनी पुन्हा करा.
- मातीमध्येही वापरू शकता.
- पाणी द्या.
- अति वापर टाळा.
""",
            "te": """### 🌿 చికిత్స దశలు
- ఎప్సమ్ సాల్ట్ 1–2% స్ప్రే చేయండి.
- నీటిలో కలపండి.
- ఆకులపై స్ప్రే చేయండి.
- ఉదయం/సాయంత్రం చేయండి.
- 10–15 రోజులకు మళ్లీ చేయండి.
- నేలలో కూడా ఉపయోగించవచ్చు.
- నీరు పోయండి.
- అధిక వినియోగం నివారించండి.
"""
        }
    },

    "healthy": {
        "symptoms": {
            "en": "Leaves are green, well-developed, and free from any deficiency symptoms.",
            "mr": "पाने हिरवी, निरोगी आणि कोणतीही कमतरता नाही",
            "te": "ఆకులు పచ్చగా, ఆరోగ్యంగా ఉన్నాయి, ఎలాంటి లోపాలు లేవు"
        },
        "treatment": {
            "en": """### 🌿 Plant Status: Healthy
- No treatment required.
- Maintain regular watering.
- Ensure proper sunlight exposure.
- Apply balanced fertilizers periodically.
- Monitor plant health regularly.
- Keep surroundings clean to prevent pests.
""",
            "mr": """### 🌿 वनस्पती स्थिती: निरोगी
- कोणतीही उपचार गरज नाही.
- नियमित पाणी द्या.
- सूर्यप्रकाश द्या.
- संतुलित खत वापरा.
- नियमित निरीक्षण करा.
- परिसर स्वच्छ ठेवा.
""",
            "te": """### 🌿 మొక్క స్థితి: ఆరోగ్యకరం
- చికిత్స అవసరం లేదు.
- క్రమంగా నీరు ఇవ్వండి.
- సూర్యకాంతి ఇవ్వండి.
- సమతుల ఎరువులు ఉపయోగించండి.
- పరిశీలించండి.
- పరిసరాలను శుభ్రంగా ఉంచండి.
"""
        }
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

            img_np = np.array(input_img)
            img_np = img_np[:, :, ::-1]
            img_np = cv2.resize(img_np, (640, 640))
            img_np = cv2.GaussianBlur(img_np, (5, 5), 0)

            results = model(img_np, conf=0.6)

            result_img = results[0].plot()
            result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

            st.subheader("📊 Before vs After Detection")

            colA, colB = st.columns(2)

            with colA:
                st.image(input_img, caption="Input Image")

            with colB:
                st.image(result_img, caption="Detected Image")

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

                    info = deficiency_info.get(detected_class)

                    if info:
                        st.markdown("### 🌿 Symptoms")
                        st.write(info["symptoms"].get(language, info["symptoms"]["en"]))

                        st.markdown("### 💊 Treatment")
                        st.markdown(info["treatment"].get(language, info["treatment"]["en"]))
                    else:
                        st.info("No additional information available.")
                else:
                    st.warning("Low confidence detection ⚠️")

     else:
        st.info("Upload or capture an image first 👆")

    # -------------------------------
    # ⚠️ Disclaimer (ADD HERE)
    # -------------------------------
    st.markdown("---")

    if language == "en":
        st.caption("⚠️ This model may make mistakes. Please verify important information.")
    elif language == "mr":
        st.caption("⚠️ हे मॉडेल चुका करू शकते. कृपया महत्त्वाची माहिती तपासा.")
    elif language == "te":
        st.caption("⚠️ ఈ మోడల్ తప్పులు చేయవచ్చు. ముఖ్యమైన సమాచారాన్ని దయచేసి ధృవీకరించండి.")

    st.markdown('</div>', unsafe_allow_html=True)
