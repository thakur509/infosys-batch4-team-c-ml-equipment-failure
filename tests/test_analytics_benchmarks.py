from src.anomaly_detection import compute_iqr_thresholds, detect_anomalies
from src.advanced_analytics import get_binary_vs_multiclass_report, get_decision_threshold_table
from src.data_loader import load_data

def test_anomaly_detection():
    df = load_data()
    thresholds = compute_iqr_thresholds(df)
    assert len(thresholds) >= 5
    anomalies_df = detect_anomalies(df)
    assert len(anomalies_df) == len(df)
    assert 'Anomaly_Level' in anomalies_df.columns

def test_advanced_analytics_benchmarks():
    report = get_binary_vs_multiclass_report()
    assert len(report) >= 2
    thresholds = get_decision_threshold_table()
    assert len(thresholds) >= 5
