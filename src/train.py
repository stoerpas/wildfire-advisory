"""
train.py
Train and compare Random Forest vs Logistic Regression on the
cleaned Algerian dataset.  Saves the best model + scaler to models/.

Run with:
    python -m src.train
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, roc_auc_score,
    confusion_matrix, RocCurveDisplay, roc_curve
)

from src.config import (
    ALGERIAN_CLEAN, FEATURE_COLS, TARGET_COL,
    RF_PARAMS, LR_PARAMS, TEST_SIZE, RANDOM_STATE,
    RF_MODEL_PATH, LR_MODEL_PATH, SCALER_PATH, MODELS_DIR, ROOT
)
from src.preprocess import load_algerian, save_clean, load_clean

FIGURES = ROOT / "notebooks" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


# ── Data preparation ──────────────────────────────────────

def get_data():
    if not ALGERIAN_CLEAN.exists():
        df = load_algerian()
        save_clean(df)
    else:
        df = load_clean()

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_test, y_train, y_test, X, y


# ── Training ──────────────────────────────────────────────

def train_models(X_train, y_train):
    """Fit both models; return fitted dict + scaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    models = {
        "Random Forest":      RandomForestClassifier(**RF_PARAMS),
        "Logistic Regression": LogisticRegression(**LR_PARAMS),
    }
    fitted = {}
    for name, model in models.items():
        X_fit = X_train_scaled if name == "Logistic Regression" else X_train
        model.fit(X_fit, y_train)
        fitted[name] = model
        print(f"  ✔ Trained {name}")

    return fitted, scaler


# ── Cross-validation ──────────────────────────────────────

def cross_validate(fitted, scaler, X, y):
    """5-fold stratified CV for both models."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    X_scaled = scaler.transform(X)
    results = {}

    print("\nCross-validation (5-fold stratified):")
    for name, model in fitted.items():
        X_cv = X_scaled if name == "Logistic Regression" else X
        scores = cross_val_score(model, X_cv, y, cv=cv,
                                 scoring="roc_auc", n_jobs=-1)
        results[name] = scores
        print(f"  {name:22s}: AUC {scores.mean():.4f} ± {scores.std():.4f}")

    return results


# ── Evaluation ────────────────────────────────────────────

def evaluate(fitted, scaler, X_test, y_test):
    """Full evaluation: metrics table + print classification reports."""
    X_test_scaled = scaler.transform(X_test)
    rows = []

    for name, model in fitted.items():
        X_eval = X_test_scaled if name == "Logistic Regression" else X_test
        y_pred = model.predict(X_eval)
        y_prob = model.predict_proba(X_eval)[:, 1]

        rows.append({
            "Model":    name,
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "ROC-AUC":  round(roc_auc_score(y_test, y_prob), 4),
            "F1 (fire)": round(
                float(classification_report(y_test, y_pred,
                      target_names=["not fire","fire"],
                      output_dict=True)["fire"]["f1-score"]), 4),
        })

        print(f"\n{'─'*50}\n  {name}\n{'─'*50}")
        print(classification_report(
            y_test, y_pred, target_names=["not fire", "fire"]
        ))

    comparison = pd.DataFrame(rows)
    print("\nModel Comparison:")
    print(comparison.to_string(index=False))
    return comparison


# ── Figure A: Confusion matrices ──────────────────────────

def plot_confusion_matrices(fitted, scaler, X_test, y_test):
    X_test_scaled = scaler.transform(X_test)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, (name, model) in zip(axes, fitted.items()):
        X_eval = X_test_scaled if name == "Logistic Regression" else X_test
        y_pred = model.predict(X_eval)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Not Fire", "Fire"],
                    yticklabels=["Not Fire", "Fire"],
                    ax=ax, cbar=False,
                    annot_kws={"size": 14, "weight": "bold"})
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.suptitle("Confusion Matrices (Test Set)", fontsize=13)
    plt.tight_layout()
    out = FIGURES / "08_confusion_matrices.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"✔ Figure saved: {out.name}")


# ── Figure B: ROC curves ──────────────────────────────────

def plot_roc_curves(fitted, scaler, X_test, y_test):
    X_test_scaled = scaler.transform(X_test)
    fig, ax = plt.subplots(figsize=(7, 5))

    colors = {"Random Forest": "#E85C4C", "Logistic Regression": "#4C9BE8"}
    for name, model in fitted.items():
        X_eval = X_test_scaled if name == "Logistic Regression" else X_test
        y_prob = model.predict_proba(X_eval)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, label=f"{name}  (AUC = {auc:.3f})",
                color=colors[name], lw=2)

    ax.plot([0,1],[0,1], "k--", lw=1, label="Random baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Test Set")
    ax.legend(loc="lower right")
    plt.tight_layout()
    out = FIGURES / "09_roc_curves.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"✔ Figure saved: {out.name}")


# ── Figure C: Feature importances ────────────────────────

def plot_feature_importances(fitted, scaler):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Random Forest — Gini importances
    rf = fitted["Random Forest"]
    rf_imp = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values()
    rf_imp.plot(kind="barh", ax=axes[0], color="#E85C4C", edgecolor="white")
    axes[0].set_title("Random Forest — Feature Importances")
    axes[0].set_xlabel("Gini Importance")

    # Logistic Regression — absolute coefficients (scaled)
    lr = fitted["Logistic Regression"]
    lr_imp = pd.Series(np.abs(lr.coef_[0]), index=FEATURE_COLS).sort_values()
    lr_imp.plot(kind="barh", ax=axes[1], color="#4C9BE8", edgecolor="white")
    axes[1].set_title("Logistic Regression — |Coefficients|")
    axes[1].set_xlabel("|Coefficient|")

    plt.suptitle("Feature Importance Comparison", fontsize=13)
    plt.tight_layout()
    out = FIGURES / "10_feature_importances.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"✔ Figure saved: {out.name}")


# ── Figure D: CV score distributions ─────────────────────

def plot_cv_scores(cv_results):
    fig, ax = plt.subplots(figsize=(7, 4))
    names  = list(cv_results.keys())
    scores = [cv_results[n] for n in names]
    colors = ["#E85C4C", "#4C9BE8"]

    bp = ax.boxplot(scores, patch_artist=True,
                    medianprops={"color": "black", "linewidth": 2},
                    widths=0.4)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # overlay individual points
    for i, (name, sc) in enumerate(cv_results.items(), start=1):
        ax.scatter([i]*len(sc), sc, color="black", s=30, zorder=5, alpha=0.8)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(names)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("5-Fold Stratified CV — ROC-AUC Distribution")
    ax.set_ylim(0.85, 1.02)
    plt.tight_layout()
    out = FIGURES / "11_cv_scores.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"✔ Figure saved: {out.name}")


# ── Persistence ───────────────────────────────────────────

def save_artefacts(fitted, scaler):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted["Random Forest"],       RF_MODEL_PATH)
    joblib.dump(fitted["Logistic Regression"], LR_MODEL_PATH)
    joblib.dump(scaler,                        SCALER_PATH)
    print(f"\nSaved artefacts → {MODELS_DIR}")
    for p in [RF_MODEL_PATH, LR_MODEL_PATH, SCALER_PATH]:
        print(f"  {p.name}  ({p.stat().st_size/1024:.1f} KB)")


# ── CLI entry point ───────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  WildfireAdvisor — Model Training")
    print("=" * 55)

    print("\nLoading data …")
    X_train, X_test, y_train, y_test, X_all, y_all = get_data()
    print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")

    print("\nTraining models …")
    fitted, scaler = train_models(X_train, y_train)

    print("\nCross-validating …")
    cv_results = cross_validate(fitted, scaler, X_all, y_all)

    print("\nEvaluating on held-out test set …")
    comparison = evaluate(fitted, scaler, X_test, y_test)

    print("\nGenerating figures …")
    plot_confusion_matrices(fitted, scaler, X_test, y_test)
    plot_roc_curves(fitted, scaler, X_test, y_test)
    plot_feature_importances(fitted, scaler)
    plot_cv_scores(cv_results)

    save_artefacts(fitted, scaler)

    print("\n" + "=" * 55)
    print("  Training complete.")
    print("=" * 55)
