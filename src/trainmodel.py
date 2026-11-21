"""Model training script for Random Forest Machine Failure Classifier."""

import json
from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.data_loader import load_data, FEATURE_COLS
from src.preprocessing import prepare_training_data
from src.evaluate import compute_metrics

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODELS_DIR / "random_forest_model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"


def train_model(
    df: pd.DataFrame = None,
    force_retrain: bool = False
) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    """
    Train Random Forest model using stratified sampling and class imbalance weighting.
    Saves trained artifact and metadata to disk.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    if not force_retrain and MODEL_PATH.exists() and METADATA_PATH.exists():
        print(f"Loading cached model from {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return model, metadata

    if df is None:
        df = load_data()

    print("Splitting dataset (80/20 stratified)...")
    X_train, X_test, y_train, y_test = prepare_training_data(df)

    print("Training Random Forest Classifier (balanced weights, n_estimators=150)...")
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=14,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    print("Evaluating held-out test performance...")
    metrics = compute_metrics(model, X_test, y_test)

    metadata = {
        "model_name": "Random Forest Classifier",
        "algorithm": "RandomForestClassifier",
        "class_weight": "balanced_subsample",
        "n_estimators": 150,
        "features": FEATURE_COLS,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "metrics": metrics
    }

    # Save artifacts
    joblib.dump(model, MODEL_PATH)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Model saved to {MODEL_PATH}")
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test F1-Score: {metrics['f1']:.4f}")
    print(f"Test PR-AUC:   {metrics['pr_auc']:.4f}")
    print(f"Test ROC-AUC:  {metrics['roc_auc']:.4f}")

    return model, metadata


if __name__ == "__main__":
    train_model(force_retrain=True)
