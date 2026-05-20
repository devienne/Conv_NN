# SeismicCNN — ConvNetQuake Reproduction

Reproduction of [**Perol, Gharbi & Denolle (2018)**](https://www.science.org/doi/10.1126/sciadv.1700578) — a convolutional neural network for earthquake detection from seismic records, applied to induced seismicity in central Oklahoma US.

> Devienne, J. A. P. M. "Convolutional neural network for earthquake detection."
> arXiv. doi: [10.48550/arXiv.2304.08328](https://doi.org/10.48550/arXiv.2304.08328)

---

## Overview

The [2011–2015 Oklahoma induced seismicity sequence](https://earthquake.usgs.gov/research/induced/)
produced an exponential growth of seismic recordings. Most classical detection methods target
moderate/large events and miss low-magnitude earthquakes buried in noise.
ConvNetQuake addresses this with a lightweight CNN (~22 k parameters) that classifies
20-second seismic windows as **earthquake** or **noise**.

This repository reproduces the detection pipeline end-to-end:

| Step | Description |
|---|---|
| Data ingestion | Download waveforms (IRIS FDSNWS) and the OGS earthquake catalog |
| Preprocessing | P-wave travel-time correction, monthly normalisation |
| Windowing | Slice daily MiniSEED files into labelled 20-second windows |
| Augmentation | Gaussian noise applied to train earthquake windows to balance classes |
| Training | 8 × Conv1D → Flatten → Dense(1, sigmoid), trained with Adam + class weights |
| API | FastAPI REST interface for querying the catalog, dataset, and model |
| Dashboard | Streamlit app reproducing all exploration plots from the analysis notebook |

**Study area:** 35.7–36.0°N, 97.2–97.6°W (central Oklahoma)  
**Station:** OK027 (HHE / HHN / HHZ, 100 Hz)  
**Period:** February 2014 – February 2015

---

## Project Structure

```
Conv_NN/
├── configs/
│   ├── experiment_convnetquake.yaml  # hyperparameters and study-area settings
│   └── local.yaml                    # machine-specific overrides (gitignored)
│
├── data/                             # all data artifacts (gitignored)
│   ├── raw/
│   │   ├── mseed/                    # downloaded .mseed files (~1 file per day)
│   │   └── catalogs/                 # historical.csv, Benz_catalog.csv
│   ├── interim/                      # historical_corrected.csv (travel-time corrected)
│   └── processed/
│       └── windows/
│           ├── train/
│           │   ├── earthquake/       # labelled 20-second CSVs (3 × 2000)
│           │   └── noise/
│           └── test/
│               ├── earthquake/
│               └── noise/
│
├── models/                           # saved Keras model (gitignored)
│   └── best_model.keras
│
├── notebooks/
│   └── exploration.ipynb             # full exploratory analysis
│
├── reports/
│   ├── figures/                      # dataset_check.png, training_history.png
│   └── model_comparisons/
│
├── scripts/
│   ├── run_pipeline.py               # end-to-end data pipeline
│   ├── train.py                      # model training
│   └── backfill_db.py                # one-time DB population from existing files
│
├── src/
│   └── seismic_cnn/
│       ├── config.py
│       ├── ingest/                   # waveform + catalog download, travel-time correction
│       ├── preprocessing.py
│       ├── windowing.py
│       ├── augmentation.py
│       ├── datasets.py               # tf.data pipeline
│       ├── models/
│       │   └── convnetquake.py       # model architecture
│       ├── training/
│       │   └── train.py              # training loop
│       ├── evaluation/
│       │   └── metrics.py
│       ├── inference/
│       │   └── predict.py
│       ├── api/                      # FastAPI application
│       │   ├── main.py
│       │   └── routes/
│       │       ├── data.py
│       │       └── model.py
│       ├── dashboard/
│       │   └── app.py                # Streamlit dashboard
│       └── db/
│           ├── schema.sql
│           └── database.py
│
├── tests/
├── pyproject.toml
└── Makefile
```

---

## Setup

**Requirements:** Python 3.9+

```bash
# 1. Clone the repository
git clone https://github.com/devienne/Conv_NN.git
cd Conv_NN

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 3. Install the package and all dependencies
pip install -e .
```

---

## Configuration

Two YAML files control the project.

**`configs/experiment_convnetquake.yaml`** — committed, controls everything scientific:

```yaml
# Study area
lat_min: 35.7  lat_max: 36.0
lon_min: -97.6  lon_max: -97.2

# Station
station:    OK027
channel:    "HH?"
start_date: "2014-02-15"
end_date:   "2015-02-15"

# Processing
window_s:           20      # window length in seconds
max_noise_per_file: 150     # noise windows per daily file
train_ratio:        0.70    # first 70% of dates → training

# Model
n_channels:  3
n_samples:   2000           # 20 s × 100 Hz
batch_size:  32
epochs:      10
```

**`configs/local.yaml`** — gitignored, use it to override paths on your machine if needed.
Create it only if you need to point to data on a different drive:

```yaml
# Example — only needed if your data lives somewhere else
project_root: /path/to/Conv_NN
```

---

## Execution Order

Run each step in sequence. Steps that have already completed can be skipped — the pipeline
detects existing files and caches results in the database.

### Step 1 — Run the data pipeline

Downloads waveforms and the earthquake catalog, corrects travel times, slices 20-second
windows, and augments the training set.

```bash
python scripts/run_pipeline.py --config configs/experiment_convnetquake.yaml
```

Expected runtime: 30–90 minutes depending on network speed and CPU count.
The pipeline uses all available CPU cores minus three for the windowing step.

### Step 2 — Populate the database

If the pipeline was interrupted or run before the database was initialised, run the backfill
once to register all existing files in SQLite:

```bash
python scripts/backfill_db.py --config configs/experiment_convnetquake.yaml
```

Safe to run multiple times — duplicates are ignored.

### Step 3 — Train the model

```bash
python scripts/train.py --config configs/experiment_convnetquake.yaml
```

Trains for up to 10 epochs with early stopping (patience = 5) on `val_auc`.
The best checkpoint is saved to `models/best_model.keras`.
Training metrics are logged to `data/project.db` and a history plot is saved to
`reports/figures/training_history.png`.

### Step 4 — Start the API

```bash
.venv\Scripts\uvicorn seismic_cnn.api.main:app --reload --host 127.0.0.1 --port 8000
```

Interactive docs: **http://127.0.0.1:8000/docs**

### Step 5 — Start the dashboard

Open a second terminal:

```bash
.venv\Scripts\streamlit run src/seismic_cnn/dashboard/app.py
```

Dashboard: **http://localhost:8501**

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/data/events` | Earthquake events in a date range (`?start=&end=`) |
| `GET` | `/data/windows` | Paginated window registry (`?label=&split=&limit=`) |
| `GET` | `/data/windows/summary` | Window counts by split / label / augmented |
| `GET` | `/data/windows/monthly` | Window counts per calendar month |
| `GET` | `/data/db/stats` | Row counts for all database tables |
| `POST` | `/model/predict` | Classify a waveform (JSON body: `waveform`, `threshold`) |
| `GET` | `/model/predict/sample` | Classify a random window from disk (`?label=&split=`) |
| `GET` | `/model/runs` | All training runs ordered by val_auc |

---

## Dashboard Pages

| Page | Content |
|------|---------|
| **Earthquake Catalog** | Study-area scatter plot · K-means cluster selection (elbow + silhouette) · clustered map with decision regions |
| **Raw Waveform Explorer** | Pick any day by date or randomly · full-day raw waveforms · preprocessing step breakdown (normalise → detrend → demean) |
| **Window Viewer** | Random earthquake vs noise side-by-side · augmentation overview · Gaussian noise sigma sweep · live model prediction |
| **Dataset Overview** | Class balance bar chart · monthly window timeline (both from SQLite) |
| **Model Performance** | AUC / F1 / Precision / Recall metric tiles · all training runs table · training history plot |
| **Live Prediction** | Upload any window CSV → waveform plot → model classification |

---

## Model Architecture

ConvNetQuake as described in Perol et al. (2018):

```
Input  (2000, 3)          20 s × 100 Hz × 3 channels
  │
  8 × Conv1D(32, kernel=3, stride=2, activation=relu)
  │
  Flatten  →  192 features
  │
  Dense(1, sigmoid)       output = P(earthquake)
```

Total trainable parameters: ~22 000.
Decision threshold: 0.5 (configurable per request via the API).

---

## Database Schema

SQLite database at `data/project.db`.

| Table | Description |
|-------|-------------|
| `events` | Earthquake catalog (origin time, magnitude, lat/lon, source) |
| `windows` | Registry of every window CSV (station, times, label, split, filepath) |
| `monthly_stats` | Cached normalisation statistics per station / month / channel |
| `model_runs` | Training history (hyperparameters, val_auc, val_f1, …) |

---

## Makefile shortcuts

```bash
make pipeline   # run_pipeline.py
make train      # train.py
make api        # start uvicorn
make dashboard  # start streamlit
make test       # pytest tests/
make lint       # ruff check src/
```
