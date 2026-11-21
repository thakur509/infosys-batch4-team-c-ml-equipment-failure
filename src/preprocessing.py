"""Preprocessing pipeline for training, evaluation, and inference."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, List

from src.data_loader import (
    FEATURE_COLS,
    RAW_NUMERIC_FEATURES,
    TARGET_COL,
    add_derived_features
)


def prepare_training_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare stratified train/test split.
    Uses strictly non-leakage sensor features + domain-engineered physical indicators.
    """
    data = add_derived_features(df)
    
    X = data[FEATURE_COLS]
    y = data[TARGET_COL]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    return X_train, X_test, y_train, y_test


def prepare_single_sample(
    air_temp: float,
    process_temp: float,
    rotational_speed: float,
    torque: float,
    tool_wear: float,
    product_type: str = "M",
    **kwargs
) -> pd.DataFrame:
    """
    Convert single manual sensor readings into model-ready feature DataFrame,
    automatically calculating derived physical features.
    """
    row = {
        "Air temperature [K]": [float(air_temp)],
        "Process temperature [K]": [float(process_temp)],
        "Rotational speed [rpm]": [float(rotational_speed)],
        "Torque [Nm]": [float(torque)],
        "Tool wear [min]": [float(tool_wear)]
    }
    df_single = pd.DataFrame(row)
    df_feat = add_derived_features(df_single)
    return df_feat[FEATURE_COLS]
