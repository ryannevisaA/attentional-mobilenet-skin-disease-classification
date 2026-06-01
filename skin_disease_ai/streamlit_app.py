import os
import math
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import streamlit as st
import tensorflow as tf
import keras
from keras import layers, models
import numpy as np
from PIL import Image
from datetime import datetime
from fpdf import FPDF
import tempfile
import keras
st.write("Keras version:", keras.__version__)
st.stop()

st.set_page_config(
    page_title="DermAI — Skin Disease Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

* { font-family: 'DM Sans', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1117 50%, #0a0e1a 100%);
    min-height: 100vh;
}

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem !important; }

/* Hero header */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #00d4ff, #0099cc, #00ffaa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin: 0;
    letter-spacing: -1px;
}

.hero-subtitle {
    text-align: center;
    color: #4a5568;
    font-size: 0.95rem;
    font-weight: 300;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 2.5rem;
}

/* Stats bar */
.stats-bar {
    display: flex;
    justify-content: center;
    gap: 3rem;
    margin: 1rem 0 2.5rem;
    padding: 1rem;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
}

.stat-item {
    text-align: center;
}

.stat-number {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #00d4ff;
}

.stat-label {
    font-size: 0.72rem;
    color: #4a5568;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Upload card */
.upload-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(0,212,255,0.15);
    border-radius: 20px;
    padding: 1.5rem;
    height: 100%;
}

.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.8rem;
    font-weight: 600;
    color: #00d4ff;
    text-transform: uppercase;
    letter-spacing: 3px;
    margin-bottom: 1rem;
}

/* Result card */
.result-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(0,212,255,0.15);
    border-radius: 20px;
    padding: 1.5rem;
}

/* Disease badge */
.disease-badge {
    display: inline-block;
    background: linear-gradient(135deg, rgba(0,212,255,0.15), rgba(0,255,170,0.1));
    border: 1px solid rgba(0,212,255,0.4);
    border-radius: 50px;
    padding: 0.5rem 1.5rem;
    font-family: 'Syne', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: #00d4ff;
    margin: 0.5rem 0;
}

.unknown-badge {
    display: inline-block;
    background: linear-gradient(135deg, rgba(255,107,107,0.15), rgba(255,50,50,0.1));
    border: 1px solid rgba(255,107,107,0.4);
    border-radius: 50px;
    padding: 0.5rem 1.5rem;
    font-family: 'Syne', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: #ff6b6b;
    margin: 0.5rem 0;
}

/* Confidence meter */
.conf-row {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin: 0.4rem 0;
}

.conf-name {
    font-size: 0.78rem;
    color: #6b7280;
    width: 150px;
    flex-shrink: 0;
}

.conf-bar-bg {
    flex: 1;
    height: 6px;
    background: rgba(255,255,255,0.06);
    border-radius: 10px;
    overflow: hidden;
}

.conf-bar-fill {
    height: 100%;
    border-radius: 10px;
    background: linear-gradient(90deg, #00d4ff, #00ffaa);
    transition: width 0.5s ease;
}

.conf-bar-fill-unknown {
    height: 100%;
    border-radius: 10px;
    background: linear-gradient(90deg, #ff6b6b, #ff9900);
}

.conf-pct {
    font-size: 0.78rem;
    font-weight: 600;
    color: #9ca3af;
    width: 45px;
    text-align: right;
    flex-shrink: 0;
}

/* Confidence value display */
.big-conf {
    font-family: 'Syne', sans-serif;
    font-size: 2.5rem;
    font-weight: 800;
    color: #00ff88;
}

.conf-label {
    font-size: 0.75rem;
    color: #4a5568;
    text-transform: uppercase;
    letter-spacing: 2px;
}

/* Desc text */
.desc-text {
    font-size: 0.88rem;
    color: #6b7280;
    line-height: 1.7;
    padding: 0.8rem;
    background: rgba(255,255,255,0.02);
    border-left: 2px solid rgba(0,212,255,0.3);
    border-radius: 0 8px 8px 0;
    margin: 0.8rem 0;
}

/* Prescription */
.rx-item {
    display: flex;
    gap: 0.8rem;
    padding: 0.6rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    font-size: 0.85rem;
    color: #9ca3af;
    line-height: 1.5;
}

.rx-num {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    color: #00d4ff;
    min-width: 24px;
    font-size: 0.9rem;
}

/* Disclaimer */
.disclaimer {
    font-size: 0.72rem;
    color: #374151;
    font-style: italic;
    padding: 0.8rem;
    background: rgba(255,255,255,0.02);
    border-radius: 8px;
    margin-top: 1rem;
    line-height: 1.6;
}

/* Warning */
.warning-box {
    background: rgba(255,107,107,0.08);
    border: 1px solid rgba(255,107,107,0.25);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    color: #ff6b6b;
    font-size: 0.88rem;
    margin: 0.5rem 0;
}

/* Divider */
.divider {
    height: 1px;
    background: rgba(255,255,255,0.06);
    margin: 1.2rem 0;
}

/* Section label */
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.72rem;
    color: #4a5568;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 0.6rem;
}

/* Override streamlit button */
.stButton > button {
    background: linear-gradient(135deg, #00d4ff, #0099cc) !important;
    color: #0a0e1a !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 2rem !important;
    letter-spacing: 1px !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(0,212,255,0.3) !important;
}

/* Download button */
.stDownloadButton > button {
    background: linear-gradient(135deg, #00ff88, #00cc6a) !important;
    color: #0a0e1a !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    width: 100% !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: rgba(0,212,255,0.03) !important;
    border: 2px dashed rgba(0,212,255,0.2) !important;
    border-radius: 16px !important;
    padding: 1rem !important;
}

/* Spinner */
.stSpinner { color: #00d4ff !important; }

/* Warning suppress */
div[data-baseweb="notification"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────
CLASS_NAMES = {
    'akiec': 'Actinic Keratosis',
    'bcc':   'Basal Cell Carcinoma',
    'bkl':   'Benign Keratosis',
    'df':    'Dermatofibroma',
    'mel':   'Melanoma',
    'nv':    'Melanocytic Nevus',
    'vasc':  'Vascular Lesion'
}

CLASS_INFO = {
    'akiec': 'A rough, scaly patch caused by years of sun exposure. Can develop into skin cancer.',
    'bcc':   'Most common form of skin cancer. Appears as a waxy bump or flat lesion.',
    'bkl':   'Non-cancerous skin growth. Usually brown, black or pale in color.',
    'df':    'Common benign fibrous nodule usually found on the legs.',
    'mel':   'Most dangerous form of skin cancer. Develops from pigment cells.',
    'nv':    'Common mole. Usually harmless but monitor for changes.',
    'vasc':  'Lesion caused by abnormal blood vessels. Usually benign.'
}

CLASS_PRESCRIPTION = {
    'akiec': ["Apply prescribed topical treatments (5-fluorouracil or imiquimod cream).", "Use broad-spectrum SPF 50+ sunscreen daily.", "Avoid direct sun exposure between 10am-4pm.", "Wear protective clothing and wide-brimmed hats.", "Schedule cryotherapy or laser treatment with dermatologist.", "Follow up with dermatologist every 3-6 months.", "Do not scratch or pick the affected area."],
    'bcc':   ["Consult a dermatologist or oncologist immediately.", "Surgical excision is the most common treatment.", "Mohs surgery may be recommended for facial lesions.", "Avoid all UV radiation exposure.", "Apply SPF 50+ sunscreen even on cloudy days.", "Regular skin checks every 3 months.", "Do not delay treatment - early removal is critical."],
    'bkl':   ["Usually benign - no urgent treatment required.", "Cryotherapy can remove unsightly growths.", "Avoid scratching to prevent infection.", "Monitor for changes in size, color, or shape.", "Apply moisturizer to keep skin hydrated.", "Annual dermatology checkup recommended.", "Consult doctor if growth changes rapidly."],
    'df':    ["Generally harmless - treatment optional.", "Surgical excision if causing discomfort.", "Avoid picking or irritating the nodule.", "Apply gentle moisturizer to the area.", "Monitor for any rapid size changes.", "Consult dermatologist if pain develops.", "Annual skin examination recommended."],
    'mel':   ["URGENT - Consult oncologist immediately!", "Surgical excision required as soon as possible.", "Sentinel lymph node biopsy may be needed.", "Avoid any sun exposure on affected area.", "Immunotherapy or targeted therapy may be prescribed.", "Regular full-body skin checks every month.", "Inform family members - melanoma can be hereditary."],
    'nv':    ["Common mole - usually harmless.", "Monitor using ABCDE rule monthly.", "A=Asymmetry, B=Border, C=Color, D=Diameter, E=Evolving.", "Use SPF 50+ sunscreen on moles exposed to sun.", "Do not shave or irritate the mole.", "Consult dermatologist if mole changes in 4 weeks.", "Annual dermatology checkup recommended."],
    'vasc':  ["Consult dermatologist for proper diagnosis.", "Laser therapy is most effective treatment.", "Avoid trauma or pressure to the lesion.", "Apply gentle moisturizer around the area.", "Monitor for bleeding or rapid growth.", "Sclerotherapy may be recommended.", "Follow up every 6 months with specialist."]
}

CLASS_KEYS = list(CLASS_NAMES.keys())

# ── FIXED thresholds ──────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.55
ENTROPY_THRESHOLD = 0.90
base_dir = os.path.dirname(os.path.abspath(__file__))

# ── SE Block (custom layer) ───────────────────────────────────
class SEBlock(keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.filters = filters
        self.gap = keras.layers.GlobalAveragePooling2D()
        self.reshape = keras.layers.Reshape((1, 1, filters))
        self.dense1 = keras.layers.Dense(filters // 16, activation='relu')
        self.dense2 = keras.layers.Dense(filters, activation='sigmoid')
        self.multiply = keras.layers.Multiply()

    def call(self, x):
        se = self.gap(x)
        se = self.reshape(se)
        se = self.dense1(se)
        se = self.dense2(se)
        return self.multiply([x, se])

    def get_config(self):
        config = super().get_config()
        config.update({'filters': self.filters})
        return config

def attention_block(x):
    filters = x.shape[-1]
    se = keras.layers.GlobalAveragePooling2D()(x)
    se = keras.layers.Reshape((1, 1, filters))(se)
    se = keras.layers.Dense(filters // 16, activation='relu')(se)
    se = keras.layers.Dense(filters, activation='sigmoid')(se)
    return keras.layers.Multiply()([x, se])

# ── Model ─────────────────────────────────────────────────────
@st.cache_resource
@st.cache_resource
def load_model():
    m = keras.models.load_model(
        os.path.join(base_dir, 'best_model_clean.keras'),
        compile=False
    )
    m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return m

def safe(text):
    return (text.replace('\u2014', '-').replace('\u2013', '-')
            .replace('\u2018', "'").replace('\u2019', "'")
            .encode('latin-1', 'ignore').decode('latin-1'))

# ── PDF ───────────────────────────────────────────────────────
def generate_pdf(disease_name, confidence, description, prescriptions, image, is_unknown):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_fill_color(10, 14, 26)
    pdf.rect(0, 0, 210, 45, 'F')
    pdf.set_text_color(0, 212, 255)
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_xy(10, 8)
    pdf.cell(0, 12, safe("DermAI — Skin Disease Report"), ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(100, 100, 130)
    pdf.set_xy(10, 23)
    pdf.cell(0, 7, safe("Attentional-MobileNet | MobileNetV2 + SE Attention"), ln=True)
    pdf.set_xy(10, 33)
    pdf.cell(0, 6, safe(f"Generated: {datetime.now().strftime('%d %B %Y, %I:%M %p')}"), ln=True)
    pdf.set_y(52)
    try:
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            image.save(tmp.name)
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(50, 50, 50)
            pdf.set_x(10)
            pdf.cell(0, 8, safe("Analyzed Image:"), ln=True)
            pdf.image(tmp.name, x=10, y=pdf.get_y(), w=65, h=60)
            pdf.set_y(pdf.get_y() + 65)
            os.unlink(tmp.name)
    except Exception:
        pass
    pdf.set_draw_color(220, 220, 230)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 15)
    if is_unknown:
        pdf.set_text_color(220, 50, 50)
        pdf.set_x(10)
        pdf.cell(0, 10, safe("Diagnosis: Unknown / Unrecognized Condition"), ln=True)
    else:
        pdf.set_text_color(0, 140, 90)
        pdf.set_x(10)
        pdf.cell(0, 10, safe(f"Diagnosis: {disease_name}"), ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(80, 80, 80)
    pdf.set_x(10)
    pdf.cell(0, 8, safe(f"Confidence Score: {confidence:.1f}%"), ln=True)
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 30, 30)
    pdf.set_x(10)
    pdf.cell(0, 8, safe("About this Condition:"), ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(80, 80, 80)
    pdf.set_x(10)
    pdf.multi_cell(0, 7, safe(description))
    pdf.ln(3)
    pdf.set_draw_color(220, 220, 230)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(0, 100, 180)
    pdf.set_x(10)
    pdf.cell(0, 9, safe("Medical Recommendations:"), ln=True)
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(50, 50, 50)
    for i, tip in enumerate(prescriptions, 1):
        pdf.set_x(10)
        pdf.multi_cell(0, 7, safe(f"{i}. {tip}"))
    pdf.ln(5)
    pdf.set_draw_color(220, 220, 230)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(160, 160, 160)
    pdf.set_x(10)
    pdf.multi_cell(0, 5, safe("DISCLAIMER: This report is for educational screening only. Not a substitute for professional medical advice. Always consult a qualified dermatologist."))
    return bytes(pdf.output())

# ── Skin Validator ────────────────────────────────────────────
def is_valid_skin_image(pil_img):
    small = pil_img.resize((50, 50)).convert('RGB')
    pixels = np.array(small).reshape(-1, 3).astype(float)
    r, g, b = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    skin_mask = (r > 30) & (g > 20) & (b > 10) & ((r + g + b) > 80)
    skin_ratio = np.sum(skin_mask) / len(pixels)
    color_std = np.std(pixels)
    w, h = pil_img.size
    aspect_ratio = max(w, h) / min(w, h)
    is_clearly_not_skin = skin_ratio < 0.10
    is_colorful_poster = color_std > 75 and skin_ratio < 0.20
    is_very_portrait = aspect_ratio > 2.0 and skin_ratio < 0.25
    return not (is_clearly_not_skin or is_colorful_poster or is_very_portrait)

# ── Main ──────────────────────────────────────────────────────
def main():
    st.markdown('<h1 class="hero-title">🔬 DermAI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Attentional-MobileNet · Dermoscopic Skin Disease Classifier</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="stats-bar">
        <div class="stat-item">
            <div class="stat-number">77.64%</div>
            <div class="stat-label">Test Accuracy</div>
        </div>
        <div class="stat-item">
            <div class="stat-number">0.900</div>
            <div class="stat-label">Macro AUC</div>
        </div>
        <div class="stat-item">
            <div class="stat-number">10,015</div>
            <div class="stat-label">Training Images</div>
        </div>
        <div class="stat-item">
            <div class="stat-number">7</div>
            <div class="stat-label">Disease Classes</div>
        </div>
        <div class="stat-item">
            <div class="stat-number">3.2M</div>
            <div class="stat-label">Parameters</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Loading AI model..."):
        model = load_model()

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<p class="section-title">📁 Upload Image</p>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose a dermoscopic skin image",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            label_visibility="collapsed"
        )

        if uploaded_file:
            pil_img = Image.open(uploaded_file).convert('RGB')
            st.image(pil_img, use_column_width=True)
            st.markdown("""
            <div style="margin-top:1rem; padding:0.8rem; background:rgba(0,212,255,0.05); border:1px solid rgba(0,212,255,0.15); border-radius:10px;">
                <p style="font-size:0.75rem; color:#4a5568; margin:0; text-transform:uppercase; letter-spacing:1px;">Supported Diseases</p>
                <p style="font-size:0.82rem; color:#6b7280; margin:0.4rem 0 0;">Melanocytic Nevus · Melanoma · Benign Keratosis · Basal Cell Carcinoma · Actinic Keratosis · Vascular Lesion · Dermatofibroma</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align:center; padding:4rem 2rem; background:rgba(0,212,255,0.02); border:2px dashed rgba(0,212,255,0.1); border-radius:16px; color:#374151;">
                <div style="font-size:3rem; margin-bottom:1rem;">🔬</div>
                <p style="font-family:'Syne',sans-serif; font-size:1rem; color:#4a5568;">Upload a dermoscopic image</p>
                <p style="font-size:0.8rem; color:#374151;">JPG, PNG, BMP supported</p>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown('<p class="section-title">📊 Analysis Results</p>', unsafe_allow_html=True)

        if not uploaded_file:
            st.markdown("""
            <div style="text-align:center; padding:4rem 2rem; color:#374151;">
                <div style="font-size:2.5rem; margin-bottom:1rem;">⬅️</div>
                <p style="font-size:0.9rem; color:#4a5568;">Upload an image to see results</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            with st.spinner("Analyzing..."):
                if not is_valid_skin_image(pil_img):
                    st.markdown('<div class="warning-box">⚠️ <strong>Not a Valid Dermoscopic Image</strong><br><span style="font-size:0.82rem; opacity:0.8;">Please upload a close-up dermoscopic skin lesion image.</span></div>', unsafe_allow_html=True)
                    result = {
                        'disease': 'Not a Dermoscopic Image', 'confidence': 0,
                        'description': 'The uploaded image does not appear to be a dermoscopic skin lesion image.',
                        'prescriptions': ['Please upload a proper dermoscopic skin image.', 'Dermoscopic images are close-up photos of skin lesions.', 'Use images from a dermatoscope or medical camera.', 'Ensure the image shows only the skin lesion area.', 'Image should be well-lit and in focus.', 'Consult a dermatologist for proper image capture.', 'Normal photos or non-skin images are not supported.'],
                        'is_unknown': True
                    }
                else:
                    img_resized = pil_img.resize((224, 224))
                    arr = np.expand_dims(np.array(img_resized) / 255.0, axis=0)
                    preds = model.predict(arr, verbose=0)[0]
                    max_conf = float(np.max(preds))
                    pred_idx = int(np.argmax(preds))
                    pred_key = CLASS_KEYS[pred_idx]
                    entropy = -sum(p * math.log(p + 1e-9) for p in preds)
                    normalized_entropy = entropy / math.log(7)
                    is_unknown = (max_conf < CONFIDENCE_THRESHOLD) or (normalized_entropy > ENTROPY_THRESHOLD)

                    if is_unknown:
                        st.markdown(f'<div class="unknown-badge">⚠️ Unknown Disease</div>', unsafe_allow_html=True)
                        st.markdown(f'<div><span class="big-conf" style="color:#ff6b6b">{max_conf*100:.1f}%</span><br><span class="conf-label">Confidence (Uncertain)</span></div>', unsafe_allow_html=True)
                        st.markdown('<div class="desc-text">This condition does not confidently match any of the 7 known disease classes. Please consult a dermatologist for proper diagnosis.</div>', unsafe_allow_html=True)
                        result = {
                            'disease': 'Unknown Condition', 'confidence': max_conf * 100,
                            'description': 'The AI model could not confidently identify this skin condition. It may be an uncommon disease not present in the training dataset.',
                            'prescriptions': ['Consult a certified dermatologist immediately.', 'Do not self-medicate or ignore the condition.', 'Take clear photos of the affected area over time.', 'Note any changes in size, color, or texture.', 'Mention any symptoms like itching, pain, or bleeding.', 'A biopsy may be needed for accurate diagnosis.', 'Seek a second medical opinion if needed.'],
                            'is_unknown': True
                        }
                    else:
                        name = CLASS_NAMES[pred_key]
                        info = CLASS_INFO[pred_key]
                        st.markdown(f'<div class="disease-badge">✅ {name}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div><span class="big-conf">{max_conf*100:.1f}%</span><br><span class="conf-label">Confidence Score</span></div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="desc-text">{info}</div>', unsafe_allow_html=True)
                        result = {
                            'disease': name, 'confidence': max_conf * 100,
                            'description': info, 'prescriptions': CLASS_PRESCRIPTION[pred_key],
                            'is_unknown': False
                        }

                    # Confidence bars
                    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                    st.markdown('<p class="section-label">Confidence per class</p>', unsafe_allow_html=True)

                    bars_html = ""
                    for i, key in enumerate(CLASS_KEYS):
                        pct = float(preds[i]) * 100
                        fill_class = "conf-bar-fill" if key == pred_key and not is_unknown else "conf-bar-fill-unknown" if is_unknown and key == pred_key else "conf-bar-fill"
                        opacity = "1" if key == pred_key else "0.4"
                        bars_html += f"""
                        <div class="conf-row" style="opacity:{opacity}">
                            <span class="conf-name">{CLASS_NAMES[key]}</span>
                            <div class="conf-bar-bg"><div class="{fill_class}" style="width:{min(pct,100):.1f}%"></div></div>
                            <span class="conf-pct">{pct:.1f}%</span>
                        </div>"""
                    st.markdown(bars_html, unsafe_allow_html=True)

            # Prescriptions
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown('<p class="section-label">💊 Medical Recommendations</p>', unsafe_allow_html=True)

            rx_html = ""
            for i, tip in enumerate(result['prescriptions'], 1):
                rx_html += f'<div class="rx-item"><span class="rx-num">{i:02d}</span><span>{tip}</span></div>'
            st.markdown(rx_html, unsafe_allow_html=True)

            st.markdown('<div class="disclaimer">⚕️ DISCLAIMER: This report is generated by an AI system for educational and screening purposes only. It is NOT a substitute for professional medical advice. Always consult a qualified dermatologist.</div>', unsafe_allow_html=True)

            # PDF Download
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            if st.button("📄 Generate Medical Report", use_container_width=True):
                with st.spinner("Generating PDF..."):
                    pdf_bytes = generate_pdf(
                        disease_name=result['disease'],
                        confidence=result['confidence'],
                        description=result['description'],
                        prescriptions=result['prescriptions'],
                        image=pil_img,
                        is_unknown=result['is_unknown']
                    )
                st.download_button(
                    label="⬇️ Download Report PDF",
                    data=pdf_bytes,
                    file_name=f"dermAI_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

if __name__ == "__main__":
    main()