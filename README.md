# TeleTibb 🩺

A non-contact blood pressure and heart rate monitoring system using computer vision and machine learning. TeleTibb analyzes facial video from a standard webcam to extract photoplethysmography (PPG) signals and predict systolic/diastolic blood pressure — no physical sensor required.

---

## How It Works

1. **Face Detection** — Haar Cascade detects and crops the facial region from webcam frames
2. **Signal Extraction** — Three rPPG algorithms extract pulse signals from subtle RGB color changes in the face:
   - **POS (Wang et al.)** — plane-orthogonal-to-skin method
   - **CHROME (De Haan et al.)** — chrominance-based extraction
   - **ICA (Poh et al.)** — Independent Component Analysis
3. **Feature Engineering** — Statistical (mean, median, variance, skewness, kurtosis), frequency domain (Welch PSD), and wavelet transform features are computed per signal
4. **Prediction** — Ensemble of Random Forest + Gradient Boosting models predicts systolic and diastolic BP alongside heart rate. Age and gender are used as demographic features.

---

## Project Structure

TeleTibb/
├── Teletibb-main.py # Main Streamlit application
├── best_rf_sys_model.pkl # Trained systolic BP model
├── best_rf_dia_model.pkl # Trained diastolic BP model
├── haarcascade_frontalface_default.xml # Face detection classifier
├── DataLoader.csv # Dataset reference file
├── preprocessing.ipynb # Face extraction & frame preprocessing
├── Dataloader_with_batches.ipynb # Batch data loading pipeline
├── Webcam_to_ICA.ipynb # Webcam to ICA signal extraction
└── Final_Model.ipynb # Model training & evaluation

---

## Tech Stack

| Category | Libraries |
|---|---|
| UI | Streamlit |
| Computer Vision | OpenCV |
| Signal Processing | SciPy, PyWavelets |
| Machine Learning | scikit-learn |
| Deep Learning utils | TensorFlow / Keras |
| Data | NumPy, Pandas |

---

## Getting Started

### Prerequisites

- Python 3.8+
- Webcam

### Installation

```bash
git clone https://github.com/MairAhmed/TeleTibb.git
cd TeleTibb
pip install streamlit opencv-python scikit-learn scipy pywavelets tensorflow numpy pandas

Run
python -m streamlit run Teletibb-main.py


Open the browser at http://localhost:8501. Grant webcam access when prompted. The app captures ~900 frames (~30 seconds at 30 fps) before making a prediction.

Input Requirements
When prompted, enter:

Age
Gender
The webcam feed will then be processed automatically.

Model Training
The models were trained using:

80/20 train-test split
GridSearchCV with 5-fold cross-validation
Hyperparameter tuning over estimators (50–200), max depth (3–15), and learning rates (0.05–0.2)
Evaluation metric: Mean Absolute Error (MAE) on held-out test set
See Final_Model.ipynb for the full training pipeline.

References
Wang, W. et al. — Algorithmic Principles of Remote PPG (POS method)
De Haan, G. & Jeanne, V. — Robust Pulse Rate From Chrominance-Based rPPG (CHROME method)
Poh, M. Z. et al. — Non-contact, automated cardiac pulse measurements using video imaging and blind source separation (ICA method)
Disclaimer
TeleTibb is a research/educational project and is not a medical device. Do not use it for clinical diagnosis or medical decision-making.

