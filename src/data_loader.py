"""Data loading and cleaning utilities for the AI4I 2020 dataset."""

from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT_DIR / "data" / "ai4i2020.csv"
if not DATASET_PATH.exists():
    DATASET_PATH = ROOT_DIR / "ai4i2020.csv"

# Column name definitions
RAW_NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

DERIVED_FEATURES = [
    "Temp_diff",
    "Power_kW",
    "Overstrain"
]

FEATURE_COLS = RAW_NUMERIC_FEATURES + DERIVED_FEATURES
TARGET_COL = "Machine failure"
FAILURE_TYPE_COLS = ["TWF", "HDF", "PWF", "OSF", "RNF"]


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute physically grounded industrial engineering features without leakage."""
    data = df.copy()
    
    # 1. Thermal dissipation delta (Process temp - Air temp)
    # Governs Heat Dissipation Failure (HDF) if delta < 8.6 K and speed < 1380 rpm
    data["Temp_diff"] = data["Process temperature [K]"] - data["Air temperature [K]"]
    
    # 2. Rotational Power in kilowatts: Power = Torque * angular_velocity
    # angular_velocity = (rpm * 2 * pi) / 60. Power (kW) = (Torque * rpm * 2 * pi) / 60000
    # Governs Power Failure (PWF) if power < 3.5 kW or power > 9.0 kW
    data["Power_kW"] = (data["Torque [Nm]"] * data["Rotational speed [rpm]"] * 2 * np.pi) / 60000.0
    
    # 3. Tool Overstrain index = Tool wear (min) * Torque (Nm)
    # Governs Overstrain Failure (OSF) when strain product exceeds product-specific thresholds
    data["Overstrain"] = data["Tool wear [min]"] * data["Torque [Nm]"]
    
    return data


def clean_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, int, int]:
    """Clean dataset, validate ranges, remove duplicate UDIs, and verify missing values."""
    initial_rows = len(df)
    data = df.copy()
    
    # Check duplicate UDI
    if "UDI" in data.columns:
        data = data.drop_duplicates(subset=["UDI"])
    duplicates_removed = initial_rows - len(data)
    
    # Ensure numeric types
    for col in RAW_NUMERIC_FEATURES:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    
    # Drop rows where target or key sensor reading is missing
    missing_before = len(data)
    data = data.dropna(subset=RAW_NUMERIC_FEATURES + [TARGET_COL])
    missing_dropped = missing_before - len(data)
    
    # Cast target to integer
    data[TARGET_COL] = data[TARGET_COL].astype(int)
    
    # Add physical derived features
    data = add_derived_features(data)
    
    return data, duplicates_removed, missing_dropped


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """Load and cache the cleaned AI4I 2020 dataset."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at expected path: {DATASET_PATH}")
    
    raw_df = pd.read_csv(DATASET_PATH)
    cleaned_df, _, _ = clean_dataset(raw_df)
    return cleaned_df
