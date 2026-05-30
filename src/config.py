"""
config.py
Central configuration: paths, feature lists, model parameters.
Import this module everywhere instead of hard-coding values.
"""

from pathlib import Path

# ── Project root ──────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

# ── Data paths ────────────────────────────────────────────
DATA_RAW        = ROOT / "data" / "raw"
DATA_PROCESSED  = ROOT / "data" / "processed"

ALGERIAN_RAW    = DATA_RAW / "Algerian_forest_fires_dataset_UPDATE.csv"
ALGERIAN_CLEAN  = DATA_PROCESSED / "algerian_clean.csv"
US_FIRES_RAW    = DATA_RAW / "us_wildfires_sample.csv"   # sampled subset

# ── Model artefacts ───────────────────────────────────────
MODELS_DIR      = ROOT / "models"
RF_MODEL_PATH   = MODELS_DIR / "random_forest.pkl"
LR_MODEL_PATH   = MODELS_DIR / "logistic_regression.pkl"
SCALER_PATH     = MODELS_DIR / "scaler.pkl"

# ── Feature definitions ───────────────────────────────────
# All numeric features used for training
FEATURE_COLS = [
    "Temperature",  # °C
    "RH",           # Relative Humidity (%)
    "Ws",           # Wind speed (km/h)
    "Rain",         # Daily rainfall (mm)
    "FFMC",         # Fine Fuel Moisture Code
    "DMC",          # Duff Moisture Code
    "DC",           # Drought Code
    "ISI",          # Initial Spread Index
    "BUI",          # Build-Up Index
    "FWI",          # Fire Weather Index
]

TARGET_COL = "Classes"  # 1 = fire, 0 = not fire

# ── Model hyperparameters ────────────────────────────────
RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 10,
    "min_samples_split": 5,
    "random_state": 42,
    "n_jobs": -1,
}

LR_PARAMS = {
    "max_iter": 1000,
    "random_state": 42,
    "C": 1.0,
}

TEST_SIZE    = 0.2
RANDOM_STATE = 42

# ── NLP ───────────────────────────────────────────────────
OPENAI_MODEL = "gpt-4o-mini"
MAX_TOKENS   = 512