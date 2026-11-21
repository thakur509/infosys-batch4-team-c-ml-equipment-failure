"""Prediction service for machine failure inference and prescriptive recommendations."""

from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd

from src.preprocessing import prepare_single_sample, add_derived_features
from src.data_loader import FEATURE_COLS
from src.trainmodel import train_model, MODEL_PATH


_CACHED_MODEL = None


def get_model():
    """Load and cache the trained Random Forest model in memory."""
    global _CACHED_MODEL
    if _CACHED_MODEL is None:
        if not MODEL_PATH.exists():
            _CACHED_MODEL, _ = train_model()
        else:
            _CACHED_MODEL = joblib.load(MODEL_PATH)
    return _CACHED_MODEL


def evaluate_sensor_diagnostics(
    air_temp: float,
    process_temp: float,
    rotational_speed: float,
    torque: float,
    tool_wear: float
) -> Dict[str, Any]:
    """Analyze sensor inputs against physical engineering thresholds."""
    temp_diff = process_temp - air_temp
    power_kw = (torque * rotational_speed * 2 * np.pi) / 60000.0
    overstrain = tool_wear * torque

    diagnostics = []
    root_causes = []

    # 1. Heat Dissipation risk rule: Temp_diff < 8.6 K and speed < 1380 rpm
    if temp_diff < 8.6:
        diagnostics.append(f"Low thermal delta ({temp_diff:.1f} K < 8.6 K target)")
        if rotational_speed < 1380:
            root_causes.append("Heat Dissipation Risk (HDF): Insufficient cooling convection at low spindle velocity.")

    # 2. Power Failure risk rule: Power < 3.5 kW or Power > 9.0 kW
    if power_kw < 3.5:
        diagnostics.append(f"Sub-nominal motor power ({power_kw:.2f} kW < 3.5 kW)")
        root_causes.append("Power Failure Risk (PWF): Motor stalling or under-voltage condition.")
    elif power_kw > 9.0:
        diagnostics.append(f"Excessive motor power draw ({power_kw:.2f} kW > 9.0 kW)")
        root_causes.append("Power Failure Risk (PWF): Overloaded spindle motor exceeding rated electrical capacity.")

    # 3. Tool Wear threshold (industrial AI4I benchmark is 200 - 240 min)
    if tool_wear >= 200:
        diagnostics.append(f"Critical tool wear duration ({tool_wear:.0f} min ≥ 200 min limit)")
        root_causes.append("Tool Wear Failure (TWF): Cutting tool edge reached end-of-life flank wear.")
    elif tool_wear >= 160:
        diagnostics.append(f"Elevated tool wear ({tool_wear:.0f} min)")

    # 4. Overstrain threshold (Tool wear * Torque product)
    if overstrain >= 11000:
        diagnostics.append(f"Critical overstrain index ({overstrain:.0f} min·Nm ≥ 11,000 threshold)")
        root_causes.append("Overstrain Failure (OSF): Mechanical overload on worn tooling.")
    elif overstrain >= 9000:
        diagnostics.append(f"High mechanical strain product ({overstrain:.0f} min·Nm)")

    # 5. Spindle Speed extremes
    if rotational_speed < 1250:
        diagnostics.append(f"Abnormally low RPM ({rotational_speed:.0f} RPM)")
    elif rotational_speed > 2500:
        diagnostics.append(f"High-frequency rotation ({rotational_speed:.0f} RPM)")

    # 6. Torque extremes
    if torque > 65.0:
        diagnostics.append(f"Extreme mechanical torque ({torque:.1f} Nm)")

    return {
        "temp_diff": temp_diff,
        "power_kw": power_kw,
        "overstrain": overstrain,
        "diagnostics": diagnostics,
        "root_causes": root_causes
    }


def predict_machine_failure(
    air_temp: float,
    process_temp: float,
    rotational_speed: float,
    torque: float,
    tool_wear: float
) -> Dict[str, Any]:
    """
    Run inference using the trained Random Forest model.
    Returns prediction status, probability, risk level, physical metrics, and prescription.
    """
    model = get_model()
    X_single = prepare_single_sample(air_temp, process_temp, rotational_speed, torque, tool_wear)
    
    pred_label = int(model.predict(X_single)[0])
    probabilities = model.predict_proba(X_single)[0]
    fail_prob = float(probabilities[1])
    
    # Risk categorization
    if fail_prob >= 0.75:
        risk_level = "Critical"
    elif fail_prob >= 0.45:
        risk_level = "High"
    elif fail_prob >= 0.20:
        risk_level = "Moderate"
    else:
        risk_level = "Low"

    # Physics-based diagnostic assessment
    diag = evaluate_sensor_diagnostics(air_temp, process_temp, rotational_speed, torque, tool_wear)
    
    # Recommendation synthesis
    if pred_label == 1 or fail_prob >= 0.45:
        if diag["root_causes"]:
            rec = " • ".join(diag["root_causes"])
            rec += " Schedule immediate maintenance inspection before resuming spindle cycle."
        else:
            rec = "Elevated multi-sensor risk detected. Inspect mechanical bearings, check spindle alignment, and verify lubrication."
    elif risk_level == "Moderate":
        rec = "Machine is in caution zone. Monitor torque stability and inspect tooling at next shift change."
    else:
        rec = "Operating within normal parameters. Machine health is optimal for standard production run."

    # 0-100 Continuous Health Score
    health_score = round(max(0.0, min(100.0, (1.0 - fail_prob) * 100.0)), 1)

    # Multiclass specific mode prediction (Task T38)
    specific_mode = "No Failure"
    try:
        from src.train_multiclass import MULTICLASS_MODEL_PATH, FAILURE_CLASS_NAMES
        if MULTICLASS_MODEL_PATH.exists():
            multi_model = joblib.load(MULTICLASS_MODEL_PATH)
            mode_idx = int(multi_model.predict(X_single)[0])
            specific_mode = FAILURE_CLASS_NAMES[mode_idx]
    except Exception:
        specific_mode = diag["root_causes"][0].split(":")[0] if diag["root_causes"] else ("Normal" if pred_label == 0 else "Mechanical Fault")

    return {
        "predicted_failure": pred_label,
        "status_label": "FAILURE RISK DETECTED" if pred_label == 1 else "NORMAL CONDITION",
        "failure_probability": fail_prob,
        "probability_pct": fail_prob * 100.0,
        "health_score": health_score,
        "specific_mode": specific_mode,
        "risk_level": risk_level,
        "recommendation": rec,
        "temp_diff": diag["temp_diff"],
        "power_kw": diag["power_kw"],
        "overstrain": diag["overstrain"],
        "diagnostic_flags": diag["diagnostics"]
    }


def batch_predict(df: pd.DataFrame) -> pd.DataFrame:
    """Run batch prediction over a dataframe of sensor values."""
    model = get_model()
    data = add_derived_features(df)
    X = data[FEATURE_COLS]
    
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    
    results = df.copy()
    results["Predicted_Failure"] = preds
    results["Failure_Probability"] = np.round(probs * 100.0, 1)
    
    conditions = [
        probs >= 0.75,
        probs >= 0.45,
        probs >= 0.20
    ]
    choices = ["Critical", "High", "Moderate"]
    results["Risk_Level"] = np.select(conditions, choices, default="Low")
    
    return results
