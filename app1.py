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
# Language Selection (NEW)
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
# Deficiency Info (FIXED STRUCTURE)
# -------------------------------
deficiency_info = {
    "nitrogen": {
        "symptoms": {
            "en": "Yellowing of older leaves, slow growth.",
            "mr": "जुन्या पानांचे पिवळे पडणे, वाढ मंद होणे",
            "te": "పాత ఆకులు పసుపు రంగులో మారడం, వృద్ధి నెమ్మదిగా ఉండడం"
        },
        "treatment": {
            "en": """### 🌿 Treatment Steps
- Apply urea or compost to improve nitrogen levels in the soil.
- Use a 1–2% urea foliar spray.
- Spray evenly on leaves.
- Apply during early morning or late evening.
- Repeat every 10–15 days if needed.
- Add compost for long-term fertility.
- Ensure proper irrigation.
- Avoid overuse of urea.
""",
            "mr": """### 🌿 उपचार पायऱ्या
- मातीतील नायट्रोजन वाढवण्यासाठी युरिया वापरा.
- 1–2% फवारणी करा.
- पानांवर समप्रमाणात फवारणी करा.
- सकाळी किंवा संध्याकाळी वापरा.
- 10–15 दिवसांनी पुन्हा करा.
- सेंद्रिय खत वापरा.
- पाणी द्या.
- अति वापर टाळा.
""",
            "te": """### 🌿 చికిత్స దశలు
- నేలలో నైట్రోజన్ పెంచడానికి యూరియా ఉపయోగించండి.
- 1–2% స్ప్రే చేయండి.
- ఆకులపై సమానంగా స్ప్రే చేయండి.
- ఉదయం/సాయంత్రం ఉపయోగించండి.
- 10–15 రోజులకు మళ్లీ చేయండి.
- కంపోస్ట్ జోడించండి.
- నీరు ఇవ్వండి.
- అధిక వినియోగం నివారించండి.
"""
        }
    },

    "boron": {
        "symptoms": {
            "en": "Deformed leaves, brittle tissues, poor growth.",
            "mr": "वाकडी पाने, नाजूक ऊती, वाढ कमी होणे",
            "te": "వంకరగా ఉన్న ఆకులు, బలహీన కణజాలం, వృద్ధి తగ్గడం"
        },
        "treatment": {
            "en": """### 🌿 Treatment Steps
- Apply borax in small quantity.
- Use 0.1–0.2% boric acid spray.
- Spray evenly on leaves.
- Apply during morning/evening.
- Repeat if needed.
- Avoid overuse.
""",
            "mr": """### 🌿 उपचार पायऱ्या
- कमी प्रमाणात बोरेक्स वापरा.
- 0.1–0.2% फवारणी करा.
- पानांवर स्प्रे करा.
- सकाळी/संध्याकाळी करा.
- पुन्हा करा.
- अति वापर टाळा.
""",
            "te": """### 🌿 చికిత్స దశలు
- తక్కువ పరిమాణంలో బోరాక్స్ ఉపయోగించండి.
- 0.1–0.2% స్ప్రే చేయండి.
- ఆకులపై స్ప్రే చేయండి.
- ఉదయం/సాయంత్రం చేయండి.
- మళ్లీ చేయండి.
- అధిక వినియోగం నివారించండి.
"""
        }
    },

    "kalium": {
        "symptoms": {
            "en": "Yellow/brown edges, weak stems.",
            "mr": "पानांच्या कडा पिवळ्या, खोड कमकुवत",
            "te": "ఆకుల అంచులు పసుపు, కాండం బలహీనంగా"
        },
        "treatment": {
            "en": """### 🌿 Treatment Steps
- Apply potash fertilizer.
- Use wood ash if needed.
- Mix in soil.
- Water properly.
- Avoid excess.
""",
            "mr": """### 🌿 उपचार पायऱ्या
- पोटॅश खत वापरा.
- लाकडाची राख वापरा.
- मातीमध्ये मिसळा.
- पाणी द्या.
- अति वापर टाळा.
""",
            "te": """### 🌿 చికిత్స దశలు
- పొటాష్ ఎరువు ఉపయోగించండి.
- చెక్క బూడిద ఉపయోగించండి.
- నేలలో కలపండి.
- నీరు పోయండి.
- అధిక వినియోగం నివారించండి.
"""
        }
    },

    "magnesium": {
        "symptoms": {
            "en": "Yellowing between veins.",
            "mr": "शिरांदरम्यान पिवळेपणा",
            "te": "నరాల మధ్య పసుపు రంగు"
        },
        "treatment": {
            "en": """### 🌿 Treatment Steps
- Apply Epsom salt spray (1–2%).
- Mix in water.
- Spray evenly.
- Repeat every 10–15 days.
""",
            "mr": """### 🌿 उपचार पायऱ्या
- एप्सम सॉल्ट फवारणी करा.
- पाण्यात मिसळा.
- स्प्रे करा.
- पुन्हा करा.
""",
            "te": """### 🌿 చికిత్స దశలు
- ఎప్సమ్ సాల్ట్ స్ప్రే చేయండి.
- నీటిలో కలపండి.
- స్ప్రే చేయండి.
- మళ్లీ చేయండి.
"""
        }
    },

    "healthy": {
        "symptoms": {
            "en": "Healthy green leaves.",
            "mr": "निरोगी हिरवी पाने",
            "te": "ఆరోగ్యకరమైన పచ్చని ఆకులు"
        },
        "treatment": {
            "en": """### 🌿 Healthy Plant
- No treatment needed.
- Maintain care.
""",
            "mr": """### 🌿 निरोगी वनस्पती
- उपचार गरज नाही.
- काळजी घ्या.
""",
            "te": """### 🌿 ఆరోగ్యకరమైన మొక్క
- చికిత్స అవసరం లేదు.
- సంరక్షణ చేయండి.
"""
        }
    }
}

# -------------------------------
# Layout
# -------------------------------
col1, col2 = st.columns(2)

# LEFT SIDE
with col1:
    st.markdown('<div class="main-box">', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png"])
    if uploaded_file:
        st.session_state["input_image"] = Image.open(uploaded_file)

    st.markdown('</div>', unsafe_allow_html=True)

# RIGHT SIDE
with col2:
    st.markdown('<div class="main-box">', unsafe_allow_html=True)

    input_img = st.session_state.get("input_image", None)

    if input_img:
        if st.button("Run Detection"):

            img_np = np.array(input_img)
            img_np = cv2.resize(img_np, (640, 640))

            results = model(img_np)
            boxes = results[0].boxes

            if boxes:
                cls = int(boxes[0].cls[0])
                detected_class = model.names[cls]

                st.success(detected_class)

                info = deficiency_info.get(detected_class)

                if info:
                    st.markdown("### 🌿 Symptoms")
                    st.write(info["symptoms"].get(language))

                    st.markdown("### 💊 Treatment")
                    st.markdown(info["treatment"].get(language))

    st.markdown('</div>', unsafe_allow_html=True)
