"""Anomaly detection engine using statistical IQR thresholds and Isolation Forest."""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import streamlit as st

from src.data_loader import RAW_NUMERIC_FEATURES, load_data


def compute_iqr_thresholds(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Compute 1.5x and 2.5x IQR boundaries for all numeric sensors."""
    thresholds = {}
    for col in RAW_NUMERIC_FEATURES:
        q1 = float(df[col].quantile(0.25))
        q3 = float(df[col].quantile(0.75))
        iqr = q3 - q1
        thresholds[col] = {
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_warning": q1 - 1.5 * iqr,
            "upper_warning": q3 + 1.5 * iqr,
            "lower_critical": q1 - 2.5 * iqr,
            "upper_critical": q3 + 2.5 * iqr,
        }
    return thresholds


@st.cache_data(show_spinner=False)
def detect_anomalies(df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Detect abnormal sensor readings across the AI4I dataset using a combination
    of domain IQR bounds and an Isolation Forest model.
    """
    if df is None:
        df = load_data()

    data = df.copy()
    thresholds = compute_iqr_thresholds(data)

    # 1. Fit Isolation Forest on scaled sensor measurements
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=0.04,
        random_state=42
    )
    iso_preds = iso_forest.fit_predict(data[RAW_NUMERIC_FEATURES])
    iso_scores = iso_forest.decision_function(data[RAW_NUMERIC_FEATURES])

    # 2. Check individual sensor breach rules
    anomaly_counts = np.zeros(len(data), dtype=int)
    critical_counts = np.zeros(len(data), dtype=int)
    abnormal_sensor_lists = [[] for _ in range(len(data))]

    for col in RAW_NUMERIC_FEATURES:
        t = thresholds[col]
        val = data[col].values
        
        # Warning breach
        warn_mask = (val < t["lower_warning"]) | (val > t["upper_warning"])
        crit_mask = (val < t["lower_critical"]) | (val > t["upper_critical"])
        
        for idx in np.where(warn_mask)[0]:
            anomaly_counts[idx] += 1
            abnormal_sensor_lists[idx].append(col)
            
        for idx in np.where(crit_mask)[0]:
            critical_counts[idx] += 1

    # 3. Categorize Anomaly Level
    # Critical: severe breach (>2.5 IQR), or multiple sensor breaches, or high iso score outlier
    levels = []
    for i in range(len(data)):
        if critical_counts[i] > 0 or anomaly_counts[i] >= 2 or (iso_preds[i] == -1 and anomaly_counts[i] >= 1):
            levels.append("Critical")
        elif anomaly_counts[i] >= 1 or iso_preds[i] == -1:
            levels.append("Warning")
        else:
            levels.append("Healthy")

    data["Anomaly_Level"] = levels
    data["Anomaly_Count"] = anomaly_counts
    data["Abnormal_Sensors"] = [", ".join(sensors) if sensors else "None" for sensors in abnormal_sensor_lists]
    data["Anomaly_Score"] = np.round(iso_scores, 4)

    return data
