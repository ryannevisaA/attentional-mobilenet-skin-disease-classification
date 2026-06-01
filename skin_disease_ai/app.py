import os
import math
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tkinter as tk
from tkinter import filedialog, messagebox
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import numpy as np
from PIL import Image, ImageTk
from datetime import datetime
from fpdf import FPDF

# ── Model Setup ──────────────────────────────────────────────
base_dir = os.path.dirname(os.path.abspath(__file__))

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
    'akiec': [
        "Apply prescribed topical treatments (5-fluorouracil or imiquimod cream).",
        "Use broad-spectrum SPF 50+ sunscreen daily.",
        "Avoid direct sun exposure between 10am-4pm.",
        "Wear protective clothing and wide-brimmed hats.",
        "Schedule cryotherapy or laser treatment with dermatologist.",
        "Follow up with dermatologist every 3-6 months.",
        "Do not scratch or pick the affected area."
    ],
    'bcc': [
        "Consult a dermatologist or oncologist immediately.",
        "Surgical excision is the most common treatment.",
        "Mohs surgery may be recommended for facial lesions.",
        "Avoid all UV radiation exposure.",
        "Apply SPF 50+ sunscreen even on cloudy days.",
        "Regular skin checks every 3 months.",
        "Do not delay treatment - early removal is critical."
    ],
    'bkl': [
        "Usually benign - no urgent treatment required.",
        "Cryotherapy can remove unsightly growths.",
        "Avoid scratching to prevent infection.",
        "Monitor for changes in size, color, or shape.",
        "Apply moisturizer to keep skin hydrated.",
        "Annual dermatology checkup recommended.",
        "Consult doctor if growth changes rapidly."
    ],
    'df': [
        "Generally harmless - treatment optional.",
        "Surgical excision if causing discomfort.",
        "Avoid picking or irritating the nodule.",
        "Apply gentle moisturizer to the area.",
        "Monitor for any rapid size changes.",
        "Consult dermatologist if pain develops.",
        "Annual skin examination recommended."
    ],
    'mel': [
        "URGENT - Consult oncologist immediately!",
        "Surgical excision required as soon as possible.",
        "Sentinel lymph node biopsy may be needed.",
        "Avoid any sun exposure on affected area.",
        "Immunotherapy or targeted therapy may be prescribed.",
        "Regular full-body skin checks every month.",
        "Inform family members - melanoma can be hereditary."
    ],
    'nv': [
        "Common mole - usually harmless.",
        "Monitor using ABCDE rule monthly.",
        "A=Asymmetry, B=Border, C=Color, D=Diameter, E=Evolving.",
        "Use SPF 50+ sunscreen on moles exposed to sun.",
        "Do not shave or irritate the mole.",
        "Consult dermatologist if mole changes in 4 weeks.",
        "Annual dermatology checkup recommended."
    ],
    'vasc': [
        "Consult dermatologist for proper diagnosis.",
        "Laser therapy is most effective treatment.",
        "Avoid trauma or pressure to the lesion.",
        "Apply gentle moisturizer around the area.",
        "Monitor for bleeding or rapid growth.",
        "Sclerotherapy may be recommended.",
        "Follow up every 6 months with specialist."
    ]
}

CONFIDENCE_THRESHOLD = 0.80
ENTROPY_THRESHOLD = 0.75

# ── Attention Block ──────────────────────────────────────────
def attention_block(x):
    filters = x.shape[-1]
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Reshape((1, 1, filters))(se)
    se = layers.Dense(filters // 16, activation='relu')(se)
    se = layers.Dense(filters, activation='sigmoid')(se)
    return layers.Multiply()([x, se])

# ── Load Model ───────────────────────────────────────────────
print("Loading model...")
model = tf.keras.models.load_model(
    os.path.join(base_dir, 'best_model.keras'),
    custom_objects={'attention_block': attention_block}
)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("Model loaded!")

CLASS_KEYS = list(CLASS_NAMES.keys())

# ── Safe text for PDF ────────────────────────────────────────
def safe(text):
    return (text
            .replace('\u2014', '-')
            .replace('\u2013', '-')
            .replace('\u2018', "'")
            .replace('\u2019', "'")
            .encode('latin-1', 'ignore')
            .decode('latin-1'))

# ── PDF Report Generator ─────────────────────────────────────
def generate_pdf(disease_name, confidence, description, prescriptions, image_path, is_unknown):
    pdf = FPDF()
    pdf.add_page()

    # Header background
    pdf.set_fill_color(15, 17, 23)
    pdf.rect(0, 0, 210, 40, 'F')

    # Title
    pdf.set_text_color(0, 212, 255)
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_xy(10, 8)
    pdf.cell(0, 10, safe("Skin Disease Classification Report"), ln=True)

    # Subtitle
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(150, 150, 150)
    pdf.set_xy(10, 22)
    pdf.cell(0, 8, safe("Powered by Attentional-MobileNet (MobileNetV2 + SE Attention)"), ln=True)

    # Date
    pdf.set_text_color(100, 100, 100)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_xy(10, 32)
    pdf.cell(0, 6, safe(f"Generated: {datetime.now().strftime('%d %B %Y, %I:%M %p')}"), ln=True)

    pdf.set_y(48)

    # Analyzed image
    try:
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(50, 50, 50)
        pdf.set_x(10)
        pdf.cell(0, 8, safe("Analyzed Image:"), ln=True)
        pdf.image(image_path, x=10, y=pdf.get_y(), w=60, h=55)
        pdf.set_y(pdf.get_y() + 62)
    except Exception:
        pass

    # Divider
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)

    # Diagnosis
    pdf.set_font("Helvetica", "B", 14)
    if is_unknown:
        pdf.set_text_color(220, 50, 50)
        pdf.set_x(10)
        pdf.cell(0, 10, safe("Diagnosis: Unknown / Unrecognized Condition"), ln=True)
    else:
        pdf.set_text_color(0, 150, 100)
        pdf.set_x(10)
        pdf.cell(0, 10, safe(f"Diagnosis: {disease_name}"), ln=True)

    # Confidence
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(80, 80, 80)
    pdf.set_x(10)
    pdf.cell(0, 8, safe(f"Confidence Score: {confidence:.1f}%"), ln=True)
    pdf.ln(3)

    # About
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 30, 30)
    pdf.set_x(10)
    pdf.cell(0, 8, safe("About this Condition:"), ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(80, 80, 80)
    pdf.set_x(10)
    pdf.multi_cell(0, 7, safe(description))
    pdf.ln(3)

    # Divider
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)

    # Prescription
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(0, 100, 180)
    pdf.set_x(10)
    pdf.cell(0, 9, safe("Medical Recommendations & Prescription:"), ln=True)
    pdf.ln(2)

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(50, 50, 50)
    for i, tip in enumerate(prescriptions, 1):
        pdf.set_x(10)
        pdf.multi_cell(0, 7, safe(f"{i}. {tip}"))

    pdf.ln(5)

    # Disclaimer
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(150, 150, 150)
    pdf.set_x(10)
    pdf.multi_cell(0, 5, safe(
        "DISCLAIMER: This report is generated by an AI system for educational and "
        "screening purposes only. It is NOT a substitute for professional medical "
        "advice, diagnosis, or treatment. Always consult a qualified dermatologist "
        "or physician for medical decisions."))

    # Save dialog
    save_path = filedialog.asksaveasfilename(
        defaultextension=".pdf",
        filetypes=[("PDF Files", "*.pdf")],
        initialfile=f"skin_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    )
    if save_path:
        pdf.output(save_path)
        messagebox.showinfo("Report Saved", f"Report saved!\n{save_path}")

# ── Main App ─────────────────────────────────────────────────
class SkinDiseaseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Attentional-MobileNet Skin Disease Classifier")
        self.root.geometry("950x720")
        self.root.configure(bg="#0f1117")
        self.root.resizable(True, True)
        self.current_image_path = None
        self.current_result = None
        self.build_ui()

    def build_ui(self):
        # Title
        title_frame = tk.Frame(self.root, bg="#0f1117")
        title_frame.pack(pady=15)

        tk.Label(title_frame,
                 text="Skin Disease Classifier",
                 font=("Helvetica", 24, "bold"),
                 fg="#00d4ff", bg="#0f1117").pack()

        tk.Label(title_frame,
                 text="Powered by Attentional-MobileNet (MobileNetV2 + SE Attention)",
                 font=("Helvetica", 10),
                 fg="#888888", bg="#0f1117").pack()

        # Main container
        main_frame = tk.Frame(self.root, bg="#0f1117")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=30)

        # Left panel
        left = tk.Frame(main_frame, bg="#1a1d27", relief=tk.RIDGE, bd=2)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10), pady=10)

        tk.Label(left, text="Input Image",
                 font=("Helvetica", 12, "bold"),
                 fg="#ffffff", bg="#1a1d27").pack(pady=10)

        self.image_label = tk.Label(left,
                                    text="No image selected\n\nClick Upload Image\nto get started",
                                    font=("Helvetica", 11),
                                    fg="#555555", bg="#1a1d27",
                                    width=30, height=15)
        self.image_label.pack(pady=10, padx=10)

        tk.Button(left,
                  text="Upload Image",
                  font=("Helvetica", 12, "bold"),
                  bg="#00d4ff", fg="#0f1117",
                  relief=tk.FLAT, cursor="hand2",
                  padx=20, pady=8,
                  command=self.upload_image).pack(pady=8)

        self.download_btn = tk.Button(left,
                  text="Download Report",
                  font=("Helvetica", 12, "bold"),
                  bg="#00ff88", fg="#0f1117",
                  relief=tk.FLAT, cursor="hand2",
                  padx=20, pady=8,
                  state=tk.DISABLED,
                  command=self.download_report)
        self.download_btn.pack(pady=8)

        # Right panel
        right = tk.Frame(main_frame, bg="#1a1d27", relief=tk.RIDGE, bd=2)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0), pady=10)

        tk.Label(right, text="Prediction Results",
                 font=("Helvetica", 12, "bold"),
                 fg="#ffffff", bg="#1a1d27").pack(pady=10)

        self.result_label = tk.Label(right, text="--",
                                     font=("Helvetica", 18, "bold"),
                                     fg="#00d4ff", bg="#1a1d27",
                                     wraplength=300)
        self.result_label.pack(pady=5)

        self.confidence_label = tk.Label(right, text="",
                                         font=("Helvetica", 13),
                                         fg="#ffffff", bg="#1a1d27")
        self.confidence_label.pack()

        self.desc_label = tk.Label(right, text="",
                                   font=("Helvetica", 10),
                                   fg="#aaaaaa", bg="#1a1d27",
                                   wraplength=280, justify=tk.CENTER)
        self.desc_label.pack(pady=8, padx=15)

        tk.Frame(right, bg="#333333", height=1).pack(fill=tk.X, padx=15, pady=4)

        tk.Label(right, text="Confidence per Class",
                 font=("Helvetica", 10, "bold"),
                 fg="#888888", bg="#1a1d27").pack(pady=4)

        self.bar_frame = tk.Frame(right, bg="#1a1d27")
        self.bar_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=4)

        self.bars = {}
        self.bar_labels = {}

        for key, name in CLASS_NAMES.items():
            row = tk.Frame(self.bar_frame, bg="#1a1d27")
            row.pack(fill=tk.X, pady=2)

            tk.Label(row, text=name[:20],
                     font=("Helvetica", 8),
                     fg="#aaaaaa", bg="#1a1d27",
                     width=20, anchor='w').pack(side=tk.LEFT)

            bar_bg = tk.Frame(row, bg="#2a2d3a", height=14, width=150)
            bar_bg.pack(side=tk.LEFT, padx=5)
            bar_bg.pack_propagate(False)

            bar_fill = tk.Frame(bar_bg, bg="#00d4ff", height=14, width=0)
            bar_fill.place(x=0, y=0, relheight=1)

            pct_label = tk.Label(row, text="0%",
                                 font=("Helvetica", 8),
                                 fg="#888888", bg="#1a1d27", width=5)
            pct_label.pack(side=tk.LEFT)

            self.bars[key] = bar_fill
            self.bar_labels[key] = pct_label

        self.status = tk.Label(self.root,
                               text="Ready -- Upload a skin image to classify",
                               font=("Helvetica", 9),
                               fg="#555555", bg="#0a0d14",
                               anchor='w', padx=10)
        self.status.pack(fill=tk.X, side=tk.BOTTOM, ipady=4)

    def upload_image(self):
        path = filedialog.askopenfilename(
            title="Select Skin Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not path:
            return

        self.current_image_path = path
        self.download_btn.config(state=tk.DISABLED)
        self.status.config(text="Analyzing image...")
        self.root.update()

        img = Image.open(path).resize((280, 250))
        photo = ImageTk.PhotoImage(img)
        self.image_label.config(image=photo, text="")
        self.image_label.image = photo

        self.predict(path)

    def predict(self, path):
        # Skin validator
        pil_img = Image.open(path).convert('RGB')
        pil_small = pil_img.resize((50, 50))
        pixels = np.array(pil_small).reshape(-1, 3).astype(float)

        r, g, b = pixels[:, 0], pixels[:, 1], pixels[:, 2]
        skin_mask = (
            (r > 40) & (g > 25) & (b > 15) &
            (r > b) &
            ((r + g + b) > 100)
        )
        skin_ratio = np.sum(skin_mask) / len(pixels)
        color_std = np.std(pixels)

        # Check image dimensions
        pil_check = Image.open(path)
        w, h = pil_check.size
        aspect_ratio = max(w, h) / min(w, h)

        # Non-skin checks
        is_portrait = aspect_ratio > 1.3 and skin_ratio < 0.40
        is_scene_photo = color_std > 55 and skin_ratio < 0.40
        is_not_skin = skin_ratio < 0.20 or color_std < 10

        if is_not_skin or is_portrait or is_scene_photo:
            self.result_label.config(
                text="Not a Dermoscopic Image", fg="#ff6b6b")
            self.confidence_label.config(
                text="This is not a skin/dermoscopic image",
                fg="#ff6b6b")
            self.desc_label.config(
                text="Please upload a proper dermoscopic skin lesion image. Photos of objects, people, posters or non-skin items are not supported.",
                fg="#ff9999")
            self.status.config(
                text="Invalid input -- Not a dermoscopic image")
            for key in CLASS_KEYS:
                self.bars[key].config(width=0)
                self.bar_labels[key].config(text="0%")
            self.current_result = {
                'disease': 'Not a Dermoscopic Image',
                'confidence': 0,
                'description': 'The uploaded image does not appear to be a dermoscopic skin lesion image.',
                'prescriptions': [
                    'Please upload a proper dermoscopic skin image.',
                    'Dermoscopic images are close-up photos of skin lesions.',
                    'Use images from a dermatoscope or medical camera.',
                    'Normal photos, posters or objects are not supported.',
                    'Consult a dermatologist for proper image capture.',
                    'Ensure the image shows only the skin lesion area.',
                    'Image should be well-lit and focused on the lesion.'
                ],
                'is_unknown': True
            }
            self.download_btn.config(state=tk.NORMAL)
            return

        # Model prediction
        img = tf.keras.preprocessing.image.load_img(path, target_size=(224, 224))
        arr = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)

        preds = model.predict(arr, verbose=0)[0]
        max_conf = float(np.max(preds))
        pred_idx = int(np.argmax(preds))
        pred_key = CLASS_KEYS[pred_idx]

        # Entropy check
        entropy = -sum(p * math.log(p + 1e-9) for p in preds)
        max_entropy = math.log(7)
        normalized_entropy = entropy / max_entropy
        is_unknown = (max_conf < CONFIDENCE_THRESHOLD) or (normalized_entropy > ENTROPY_THRESHOLD)

        # Update bars
        for i, key in enumerate(CLASS_KEYS):
            pct = float(preds[i]) * 100
            width = max(0, min(150, int((pct / 100) * 150)))
            color = "#00d4ff" if key == pred_key else "#3a3d4a"
            self.bars[key].config(width=width, bg=color)
            self.bar_labels[key].config(text=f"{pct:.1f}%")

        if is_unknown:
            self.result_label.config(
                text="Unknown / Unrecognized Disease", fg="#ff6b6b")
            self.confidence_label.config(
                text=f"Confidence: {max_conf*100:.1f}% (Uncertain)",
                fg="#ff6b6b")
            self.desc_label.config(
                text="This does not confidently match any known disease. Please consult a dermatologist.",
                fg="#ff9999")
            self.status.config(text="Result: Unknown disease detected")
            self.current_result = {
                'disease': 'Unknown Condition',
                'confidence': max_conf * 100,
                'description': 'The AI model could not confidently identify this skin condition. It may be an uncommon disease not present in the training dataset.',
                'prescriptions': [
                    'Consult a certified dermatologist immediately.',
                    'Do not self-medicate or ignore the condition.',
                    'Take clear photos of the affected area over time.',
                    'Note any changes in size, color, or texture.',
                    'Mention any symptoms like itching, pain, or bleeding.',
                    'A biopsy may be needed for accurate diagnosis.',
                    'Seek a second medical opinion if needed.'
                ],
                'is_unknown': True
            }
        else:
            name = CLASS_NAMES[pred_key]
            info = CLASS_INFO[pred_key]
            self.result_label.config(text=f"Detected: {name}", fg="#00d4ff")
            self.confidence_label.config(
                text=f"Confidence: {max_conf*100:.1f}%", fg="#00ff88")
            self.desc_label.config(text=info, fg="#aaaaaa")
            self.status.config(
                text=f"Result: {name} detected with {max_conf*100:.1f}% confidence")
            self.current_result = {
                'disease': name,
                'confidence': max_conf * 100,
                'description': info,
                'prescriptions': CLASS_PRESCRIPTION[pred_key],
                'is_unknown': False
            }

        self.download_btn.config(state=tk.NORMAL)

    def download_report(self):
        if not self.current_result:
            return
        generate_pdf(
            disease_name=self.current_result['disease'],
            confidence=self.current_result['confidence'],
            description=self.current_result['description'],
            prescriptions=self.current_result['prescriptions'],
            image_path=self.current_image_path,
            is_unknown=self.current_result['is_unknown']
        )

# ── Run ──────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app = SkinDiseaseApp(root)
    root.mainloop()