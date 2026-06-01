# 🔬 DermAI — Attentional-MobileNet Skin Disease Classifier

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://attentional-mobilenet-skin-disease-classification-aznpoig5my2h.streamlit.app/)

A deep learning web application for dermoscopic skin disease classification using **MobileNetV2 with Squeeze-and-Excitation (SE) Attention**, trained on the HAM10000 dataset.

---

## 🚀 Live Demo

**[👉 Try the App](https://attentional-mobilenet-skin-disease-classification-aznpoig5my2h.streamlit.app/)**

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Test Accuracy | 77.64% |
| Macro AUC | 0.900 |
| Training Images | 10,015 |
| Parameters | 3.2M |

---

## 🧬 Supported Disease Classes

| Code | Disease |
|------|---------|
| `akiec` | Actinic Keratosis |
| `bcc` | Basal Cell Carcinoma |
| `bkl` | Benign Keratosis |
| `df` | Dermatofibroma |
| `mel` | Melanoma |
| `nv` | Melanocytic Nevus |
| `vasc` | Vascular Lesion |

---

## 🏗️ Model Architecture

```
Input (224×224×3)
    ↓
MobileNetV2 Backbone (pretrained, ImageNet)
    ↓
SE Attention Block (Squeeze-and-Excitation)
    ↓
Global Average Pooling
    ↓
Dense (256) + Dropout
    ↓
Softmax Output (7 classes)
```

- **Backbone:** MobileNetV2 (1.00, 224×224)
- **Attention:** Squeeze-and-Excitation block (ratio 1/16)
- **Dataset:** HAM10000 (Human Against Machine with 10000 training images)

---

## ✨ App Features

- Upload dermoscopic skin lesion images (JPG, PNG, BMP)
- Real-time classification with confidence scores per class
- Unknown/uncertain condition detection using entropy thresholding
- Skin image validator (rejects non-dermoscopic images)
- Downloadable PDF medical report with recommendations
- Clean dark-themed UI built with Streamlit

---

## 🗂️ Project Structure

```
skin_disease_ai/
├── streamlit_app.py        # Main Streamlit application
├── train.py                # Model training script
├── model.py                # Model architecture definition
├── preprocess.py           # Data preprocessing
├── evaluate.py             # Model evaluation
├── gradcam.py              # Grad-CAM visualization
├── compare_models.py       # Model comparison
├── final_weights.weights.h5 # Trained model weights
├── HAM10000_metadata.csv   # Dataset metadata
├── requirements.txt        # Python dependencies
└── .python-version         # Python version pin (3.11)
```

---

## ⚙️ Run Locally

```bash
# Clone the repo
git clone https://github.com/ryannevisaA/attentional-mobilenet-skin-disease-classification.git
cd attentional-mobilenet-skin-disease-classification

# Install dependencies
pip install -r skin_disease_ai/requirements.txt

# Run the app
streamlit run skin_disease_ai/streamlit_app.py
```

---

## 📦 Dependencies

- TensorFlow 2.21.0
- Keras 3.13.2
- Streamlit
- NumPy, Pandas, Pillow
- Matplotlib, Scikit-learn
- OpenCV, fpdf2

---

## ⚕️ Disclaimer

This application is for **educational and screening purposes only**. It is not a substitute for professional medical advice. Always consult a qualified dermatologist for diagnosis and treatment.

---

## 👨‍💻 Author

**ryannevisaA**
- GitHub: [@ryannevisaA](https://github.com/ryannevisaA)

---

*Built with MobileNetV2 + SE Attention | Trained on HAM10000 | Deployed on Streamlit Cloud*
