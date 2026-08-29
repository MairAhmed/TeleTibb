# TeleTibb

TeleTibb is a research prototype for estimating heart rate and blood pressure
from facial video. It combines computer vision, remote photoplethysmography
(rPPG), signal processing, and pre-trained machine-learning models in a
Streamlit interface.

> [!WARNING]
> TeleTibb is an educational and research project, not a medical device. Its
> output must not be used for diagnosis, treatment, or other medical decisions.

## How it works

1. **Capture:** OpenCV reads frames from the computer's default webcam.
2. **Face detection:** A Haar cascade locates a face in each frame. The first
   detected face is cropped and resized to `224 × 224`.
3. **Signal extraction:** The application derives three rPPG signals from the
   facial RGB values:
   - POS (Plane-Orthogonal-to-Skin)
   - CHROM (chrominance-based rPPG)
   - ICA (Independent Component Analysis)
4. **Feature engineering:** The signals are filtered, normalized, padded, and
   converted into statistical, frequency-domain, and wavelet features.
5. **Prediction:** Pre-trained Random Forest models estimate heart rate,
   systolic blood pressure, and diastolic blood pressure using the extracted
   features together with age and gender.

The application collects **900 frames containing a detected face** before
running the prediction pipeline. At the configured sample rate of 30 FPS, this
is approximately 30 seconds of usable face footage.

## Repository structure

```text
TeleTibb/
├── Teletibb-main.py
│   └── Streamlit app, webcam capture, rPPG extraction, and inference
├── DataLoader.csv
│   └── Processed training data and physiological signal outputs
├── preprocessing.ipynb
│   └── Video preprocessing and facial-frame extraction experiments
├── Dataloader_with_batches.ipynb
│   └── Dataset loading, batched video processing, and rPPG generation
├── Webcam_to_ICA.ipynb
│   └── Webcam/rPPG experiments and blood-pressure model comparisons
├── Final_Model.ipynb
│   └── Final feature-engineering, training, tuning, and evaluation workflow
├── random_forest_model.pkl
│   └── Trained heart-rate regressor used by the app
├── best_rf_sys_model.pkl
│   └── Trained systolic blood-pressure regressor used by the app
├── best_rf_dia_model.pkl
│   └── Trained diastolic blood-pressure regressor used by the app
├── haarcascade_frontalface_default.xml
│   └── Haar cascade artifact retained for face-detection experiments
└── van_Putten_Improving_Systolic_Blood_Pressure_Prediction_...pdf
    └── Reference paper included with the project
```

The application currently uses OpenCV's bundled frontal-face Haar cascade at
runtime. The XML file in the repository is retained as a project artifact.

## Tech stack

| Area | Main libraries |
| --- | --- |
| User interface | Streamlit |
| Video and face processing | OpenCV, Pillow, scikit-image |
| Numerical and data processing | NumPy, Pandas, SciPy |
| Signal processing | SciPy Signal, PyWavelets |
| Machine learning | scikit-learn, joblib |
| Sequence preprocessing | TensorFlow/Keras |
| Research notebooks | Jupyter, Matplotlib, PyTorch |

## Getting started

### Requirements

- Python 3 and `pip`
- A webcam available as the computer's default OpenCV camera (`VideoCapture(0)`)
- A desktop environment capable of opening the Streamlit application

The webcam is opened by the Python process, not through the browser's media
permissions. Run the app on the same computer that has the camera attached.

### Installation

Clone the repository and create an isolated environment:

```bash
git clone https://github.com/MairAhmed/TeleTibb.git
cd TeleTibb

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the packages imported by the Streamlit application:

```bash
python -m pip install \
  streamlit streamlit-webrtc \
  opencv-python pillow scikit-image matplotlib \
  numpy pandas scipy pywavelets \
  scikit-learn==1.3.0 joblib tensorflow
```

The model artifacts were serialized with scikit-learn 1.3.0, so matching that
version avoids model-persistence compatibility warnings.

### Run the application

Run this command from the repository root so the app can find the three
committed model files:

```bash
python -m streamlit run Teletibb-main.py
```

Streamlit will print the local application URL, normally
`http://localhost:8501`.

### Take a measurement

1. Enter an age.
2. Select a gender.
3. Position one face clearly in front of the webcam.
4. Select **Start** and remain still while 900 detected-face frames are
   collected.
5. Wait for the estimated heart rate, systolic pressure, and diastolic
   pressure to appear.

Even lighting, limited motion, and a consistently visible face will produce a
cleaner input signal. Capture duration may exceed 30 seconds when frames do not
contain a detectable face.

## Data and model development

`DataLoader.csv` contains demographic information, raw/derived physiological
signals, blood-pressure targets, and heart-rate values. The primary columns
include:

- participant ID, gender, and age;
- PPG, ICA, CHROM, and POS signal sequences;
- systolic and diastolic blood pressure;
- signal-specific and ground-truth heart rates.

The notebooks represent stages of the research workflow rather than a single
automated pipeline:

| Notebook | Purpose |
| --- | --- |
| `preprocessing.ipynb` | Detect, crop, resize, and collect facial video frames |
| `Dataloader_with_batches.ipynb` | Load recordings in batches and generate ICA, CHROM, and POS outputs |
| `Webcam_to_ICA.ipynb` | Explore real-time capture, rPPG methods, and candidate regressors |
| `Final_Model.ipynb` | Prepare features, tune regressors with grid search, and evaluate BP predictions |

The committed `.pkl` files are the artifacts consumed by
`Teletibb-main.py`; retraining is not required to run the application.

## Current limitations

- Predictions depend heavily on lighting, movement, camera quality, and face
  detection.
- The application assumes a 30 FPS sample rate and a single primary face.
- Webcam capture uses the first local camera and does not currently expose a
  camera selector.
- Model artifacts are committed directly to the repository and no reproducible
  dependency lock file is currently provided.
- The notebooks may reference experiment-specific datasets, paths, or optional
  packages that are not required by the Streamlit app.

## Research references

- Wang, W. et al. *Algorithmic Principles of Remote PPG.*
- de Haan, G. and Jeanne, V. *Robust Pulse Rate From Chrominance-Based rPPG.*
- Poh, M.-Z. et al. *Non-contact, automated cardiac pulse measurements using
  video imaging and blind source separation.*
- van Putten, L. D. and Bamford, K. E. *Improving Systolic Blood Pressure
  Prediction From Remote Photoplethysmography Using a Stacked Ensemble
  Regressor.*
