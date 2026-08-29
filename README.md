# TeleTibb

TeleTibb is a research prototype that estimates heart rate and blood pressure
from facial video using remote photoplethysmography (rPPG), signal processing,
and participant-grouped machine-learning models.

> [!WARNING]
> TeleTibb is not a medical device. Its estimates must not be used for
> diagnosis, treatment, medication decisions, or emergency assessment.

## Accuracy-focused pipeline

1. **Quality-controlled capture:** OpenCV reads the local webcam for 30 seconds.
   The app tracks one smoothed face box, samples forehead and cheek regions,
   rejects low-quality frames, and records a timestamp for every accepted
   sample.
2. **Time correction:** Accepted RGB samples are resampled onto a uniform time
   grid using the measured frame rate rather than an assumed 30 FPS.
3. **rPPG extraction:** POS, CHROM, and ICA generate independent pulse signals
   from correctly ordered RGB values.
4. **Heart rate:** A quality-weighted consensus combines the dominant
   pulse-band frequency from all three methods. The measurement is rejected
   when the methods disagree.
5. **Blood pressure:** A shared feature pipeline extracts temporal,
   frequency-domain, pulse morphology, signal agreement, age, and gender
   features. The same Python functions are used during training and inference.
6. **Uncertainty:** The interface displays signal quality and
   participant-holdout error ranges instead of presenting bare point estimates.

The app requires at least 300 accepted frames. Poor lighting, blur, motion,
camera timing gaps, or weak rPPG agreement produce a retry result rather than a
prediction.

## Repository structure

```text
TeleTibb/
├── Teletibb-main.py
│   └── Streamlit interface, quality-controlled capture, and inference
├── teletibb_signal.py
│   └── Shared rPPG, heart-rate, preprocessing, and feature pipeline
├── train_models.py
│   └── Reproducible participant-grouped BP training and evaluation
├── DataLoader.csv
│   └── Processed training signals, demographics, and reference values
├── bp_systolic_model.pkl
│   └── Versioned systolic model bundle used by the app
├── bp_diastolic_model.pkl
│   └── Versioned diastolic model bundle used by the app
├── model_metrics.json
│   └── Grouped cross-validation, holdout metrics, and feature schema
├── requirements.txt
│   └── Pinned runtime and training dependencies
├── requirements-dev.txt
│   └── Test and lint dependencies
├── tests/
│   └── Signal, feature, split, and model-bundle regression tests
├── preprocessing.ipynb
├── Dataloader_with_batches.ipynb
├── Webcam_to_ICA.ipynb
├── Final_Model.ipynb
│   └── Historical research notebooks
├── haarcascade_frontalface_default.xml
└── van_Putten_Improving_Systolic_Blood_Pressure_Prediction_...pdf
```

`train_models.py` is now the authoritative training workflow. The notebooks are
retained as research history and are not used to produce deployed artifacts.

## Installation

Requirements:

- Python 3.10
- `pip`
- a webcam available to OpenCV as `VideoCapture(0)`
- a local desktop session

```bash
git clone https://github.com/MairAhmed/TeleTibb.git
cd TeleTibb

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The webcam is opened by the Python process, not through browser media
permissions. Run Streamlit on the computer that has the camera attached.

## Run the application

```bash
python -m streamlit run Teletibb-main.py
```

Then:

1. enter an age and select the demographic value used by the research model;
2. select **Start measurement**;
3. keep one face centered and still in even lighting for 30 seconds;
4. review the estimate, quality score, uncertainty, and diagnostics.

## Reproduce the models

The training command safely parses the committed signals, constructs the same
54-feature schema used by the app, removes participant leakage, and regenerates
both model bundles and the metrics report:

```bash
python train_models.py
```

For a faster development run with fewer trees:

```bash
python train_models.py --quick
```

Training uses:

- a 79-participant development partition;
- grouped five-fold cross-validation keyed by `GUID`;
- an untouched 20-participant holdout partition;
- mean and demographics-only baselines;
- Ridge, Random Forest, Extra Trees, and Gradient Boosting candidates;
- model selection by grouped validation MAE;
- final refitting on all available rows after holdout evaluation.

No participant appears in both development and holdout data. Feature order and
model schema are checked by the app; dependency version, validation metadata,
and uncertainty are stored with each model bundle.

## Current participant-holdout results

These results are from the committed `model_metrics.json`, not the historical
row-level notebook split:

| Target | Selected model | Grouped CV MAE | Holdout MAE | Holdout RMSE | Bias | 90th-percentile absolute error |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Systolic | Extra Trees | 15.88 mmHg | 15.63 mmHg | 19.73 mmHg | +3.38 mmHg | 32.34 mmHg |
| Diastolic | Random Forest | 8.97 mmHg | 8.04 mmHg | 11.60 mmHg | -1.34 mmHg | 20.34 mmHg |

These results show that the project is not accurate enough for clinical use.
They are intentionally reported because participant-grouped evaluation is more
trustworthy than selecting a model from a random row split.

## Automated checks

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
python -m ruff check .
```

## Data and validation limitations

- `DataLoader.csv` contains only 201 rows from 99 participants.
- Most participants have repeated rows with the same BP target, making
  participant grouping essential.
- The raw source videos and synchronized reference acquisition workflow are not
  included. Historical signals therefore cannot be regenerated with the new
  RGB-correct, tracked-ROI capture pipeline.
- The deployed capture pipeline and historical processed signals still have an
  unavoidable acquisition-domain mismatch until data is recollected or raw
  videos are reprocessed.
- The data does not include enough structured device, lighting, movement, skin
  tone, session, or signal-quality metadata for comprehensive subgroup
  validation.
- The current holdout contains only 20 participants, so its uncertainty
  estimates are themselves imprecise.

The next accuracy improvement requires synchronized video and validated cuff
references from more independent participants, multiple sessions and BP states,
and varied cameras and capture conditions. New data should be processed by
`teletibb_signal.py` and evaluated with the grouped workflow before replacing
the committed models.

## Research references

- Wang, W. et al. *Algorithmic Principles of Remote PPG.*
- de Haan, G. and Jeanne, V. *Robust Pulse Rate From Chrominance-Based rPPG.*
- Poh, M.-Z. et al. *Non-contact, automated cardiac pulse measurements using
  video imaging and blind source separation.*
- van Putten, L. D. and Bamford, K. E. *Improving Systolic Blood Pressure
  Prediction From Remote Photoplethysmography Using a Stacked Ensemble
  Regressor.*
