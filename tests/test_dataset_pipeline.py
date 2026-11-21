import pandas as pd
from src.data_loader import load_data, FEATURE_COLS, TARGET_COL

def test_load_data():
    df = load_data()
    assert len(df) == 10000
    assert TARGET_COL in df.columns
    for f in FEATURE_COLS:
        assert f in df.columns

def test_no_null_values():
    df = load_data()
    assert df[FEATURE_COLS].isnull().sum().sum() == 0
