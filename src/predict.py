"""
predict.py
Inference module: load saved model artefacts and make predictions.
This is the only module the Gradio app imports for ML logic.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from src.config import (
    RF_MODEL_PATH, LR_MODEL_PATH, SCALER_PATH, FEATURE_COLS
)


# ── Lazy-loaded singletons ────────────────────────────────
_rf_model  = None
_lr_model  = None
_scaler    = None


def _load_artefacts():
    global _rf_model, _lr_model, _scaler
    if _rf_model is None:
        _rf_model = joblib.load(RF_MODEL_PATH)
    if _lr_model is None:
        _lr_model = joblib.load(LR_MODEL_PATH)
    if _scaler is None:
        _scaler = joblib.load(SCALER_PATH)


# ── Public API ────────────────────────────────────────────

def predict(input_values: dict, model: str = "Random Forest") -> dict:
    """
    Run inference for one set of weather/FWI readings.

    Parameters
    ----------
    input_values : dict
        Keys must match FEATURE_COLS.
        E.g. {"Temperature": 34, "RH": 40, "Ws": 15, "Rain": 0,
               "FFMC": 88.0, "DMC": 26.0, "DC": 100.0,
               "ISI": 5.5, "BUI": 30.0, "FWI": 12.0}
    model : str
        "Random Forest" (default) or "Logistic Regression"

    Returns
    -------
    dict with keys:
        prediction   : int   (1 = fire risk, 0 = no fire risk)
        probability  : float (probability of fire)
        label        : str   ("Fire Risk" / "No Fire Risk")
        top_features : list  of (feature_name, importance) tuples
    """
    _load_artefacts()

    X = pd.DataFrame([input_values])[FEATURE_COLS]

    if model == "Logistic Regression":
        X_input = _scaler.transform(X)
        clf = _lr_model
    else:
        X_input = X
        clf = _rf_model

    pred   = int(clf.predict(X_input)[0])
    prob   = float(clf.predict_proba(X_input)[0][1])
    label  = "Fire Risk" if pred == 1 else "No Fire Risk"

    # Feature importances (RF only; use |coef| for LR)
    if model == "Random Forest":
        importances = clf.feature_importances_
    else:
        importances = np.abs(_lr_model.coef_[0])

    top_features = sorted(
        zip(FEATURE_COLS, importances),
        key=lambda x: x[1], reverse=True
    )[:3]

    return {
        "prediction":   pred,
        "probability":  round(prob, 4),
        "label":        label,
        "top_features": top_features,
    }
