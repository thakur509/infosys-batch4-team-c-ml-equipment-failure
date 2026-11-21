"""Multiclass Failure-Type Model Training (Task T38 & T39).

Predicts specific AI4I 2020 failure modes:
0: No Failure
1: Heat Dissipation Failure (HDF)
2: Power Failure (PWF)
3: Overstrain Failure (OSF)
4: Tool Wear Failure (TWF)
5: Random Failure (RNF)
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from src.data_loader import load_data, FEATURE_COLS, add_derived_features

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MULTICLASS_MODEL_PATH = MODELS_DIR / "failure_type_model.joblib"
MULTICLASS_METADATA_PATH = MODELS_DIR / "multiclass_metadata.json"

FAILURE_CLASS_NAMES = [
    "No Failure",
    "Heat Dissipation (HDF)",
    "Power Failure (PWF)",
    "Overstrain (OSF)",
    "Tool Wear (TWF)",
    "Random Failure (RNF)"
]


def create_multiclass_target(df: pd.DataFrame) -> pd.Series:
    """Create a single multiclass target column from the binary indicator columns."""
    target = np.zeros(len(df), dtype=int)
    
    # Priority assignment based on physics severity
    # HDF: 1, PWF: 2, OSF: 3, TWF: 4, RNF: 5
    if "HDF" in df.columns:
        target[df["HDF"] == 1] = 1
    if "PWF" in df.columns:
        target[df["PWF"] == 1] = 2
    if "OSF" in df.columns:
        target[df["OSF"] == 1] = 3
    if "TWF" in df.columns:
        target[df["TWF"] == 1] = 4
    if "RNF" in df.columns:
        target[df["RNF"] == 1] = 5
        
    return pd.Series(target, index=df.index, name="Failure_Type")


def train_multiclass_model(
    df: pd.DataFrame = None,
    force_retrain: bool = False
) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    """Train and evaluate 6-class specific failure-type Random Forest model."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    if not force_retrain and MULTICLASS_MODEL_PATH.exists() and MULTICLASS_METADATA_PATH.exists():
        model = joblib.load(MULTICLASS_MODEL_PATH)
        with open(MULTICLASS_METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return model, metadata

    if df is None:
        df = load_data()

    data = add_derived_features(df)
    y_multi = create_multiclass_target(data)
    X = data[FEATURE_COLS]

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_multi,
        test_size=0.2,
        random_state=42,
        stratify=y_multi
    )

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=14,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    clf_report = classification_report(
        y_test,
        y_pred,
        target_names=FAILURE_CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    # Format table for T38 report
    class_table = []
    for idx, name in enumerate(FAILURE_CLASS_NAMES):
        entry = clf_report.get(name, {})
        class_table.append({
            "Failure Class": name,
            "Precision": f"{entry.get('precision', 0.0) * 100:.2f}%",
            "Recall": f"{entry.get('recall', 0.0) * 100:.2f}%",
            "F1-Score": f"{entry.get('f1-score', 0.0) * 100:.2f}%",
            "Test Support": int(entry.get("support", 0))
        })

    # Summary rows
    class_table.append({
        "Failure Class": "macro avg",
        "Precision": f"{clf_report['macro avg']['precision'] * 100:.2f}%",
        "Recall": f"{clf_report['macro avg']['recall'] * 100:.2f}%",
        "F1-Score": f"{clf_report['macro avg']['f1-score'] * 100:.2f}%",
        "Test Support": int(clf_report['macro avg']['support'])
    })
    class_table.append({
        "Failure Class": "weighted avg",
        "Precision": f"{clf_report['weighted avg']['precision'] * 100:.2f}%",
        "Recall": f"{clf_report['weighted avg']['recall'] * 100:.2f}%",
        "F1-Score": f"{clf_report['weighted avg']['f1-score'] * 100:.2f}%",
        "Test Support": int(clf_report['weighted avg']['support'])
    })

    metadata = {
        "model_name": "Failure-Type Multiclass Classifier",
        "accuracy": float(acc),
        "weighted_f1": float(weighted_f1),
        "macro_f1": float(macro_f1),
        "class_table": class_table,
        "classification_report": clf_report,
        "classes": FAILURE_CLASS_NAMES
    }

    joblib.dump(model, MULTICLASS_MODEL_PATH)
    with open(MULTICLASS_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return model, metadata


def predict_failure_mode(
    air_temp: float,
    process_temp: float,
    rotational_speed: float,
    torque: float,
    tool_wear: float,
    product_type: str = "M"
) -> Tuple[str, float, Dict[str, float]]:
    """Predict specific failure mode for an individual telemetry reading."""
    from src.preprocessing import prepare_single_sample
    if not MULTICLASS_MODEL_PATH.exists():
        train_multiclass_model()
    model = joblib.load(MULTICLASS_MODEL_PATH)
    
    sample = prepare_single_sample(
        air_temp=air_temp,
        process_temp=process_temp,
        rotational_speed=rotational_speed,
        torque=torque,
        tool_wear=tool_wear,
        product_type=product_type
    )
    
    probs = model.predict_proba(sample)[0]
    best_idx = int(np.argmax(probs))
    conf = float(probs[best_idx])
    
    class_probs = {FAILURE_CLASS_NAMES[i]: float(probs[i]) for i in range(len(probs))}
    return FAILURE_CLASS_NAMES[best_idx], conf, class_probs


if __name__ == "__main__":
    train_multiclass_model(force_retrain=True)

