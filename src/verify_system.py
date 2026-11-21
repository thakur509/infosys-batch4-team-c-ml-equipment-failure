"""Automated verification suite testing end-to-end functionality of all modules."""

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

def run_tests():
    print("==================================================")
    print("RUNNING PREDICTIVE MAINTENANCE SYSTEM VERIFICATION")
    print("==================================================")
    
    # 1. Test Data Ingestion
    print("\n[1/6] Testing Data Loader...")
    from src.data_loader import load_data, FEATURE_COLS, TARGET_COL, RAW_NUMERIC_FEATURES
    df = load_data()
    assert len(df) == 10000, f"Expected 10,000 rows, got {len(df)}"
    for col in FEATURE_COLS:
        assert col in df.columns, f"Missing feature column: {col}"
    assert TARGET_COL in df.columns, f"Missing target column: {TARGET_COL}"
    print(f"[PASS] Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"[PASS] Features verified: {FEATURE_COLS}")
    
    # 2. Test Model & Inference
    print("\n[2/6] Testing Model Inference & Prescriptions...")
    from src.predict import predict_machine_failure, batch_predict
    
    # Normal input
    res_normal = predict_machine_failure(
        air_temp=298.5,
        process_temp=308.8,
        rotational_speed=1520,
        torque=40.0,
        tool_wear=45
    )
    assert "status_label" in res_normal
    assert "failure_probability" in res_normal
    assert "risk_level" in res_normal
    print(f"[PASS] Normal input test: Status={res_normal['status_label']}, Prob={res_normal['probability_pct']:.1f}%, Risk={res_normal['risk_level']}")
    
    # Critical overstrain input
    res_crit = predict_machine_failure(
        air_temp=303.0,
        process_temp=309.0,
        rotational_speed=1250,
        torque=68.0,
        tool_wear=230
    )
    assert res_crit["predicted_failure"] == 1 or res_crit["failure_probability"] > 0.4
    print(f"[PASS] High-stress input test: Status={res_crit['status_label']}, Prob={res_crit['probability_pct']:.1f}%, Risk={res_crit['risk_level']}")
    print(f"[PASS] Prescriptive recommendation: {res_crit['recommendation'][:80]}...")
    
    # Batch predict test
    batch_res = batch_predict(df.head(20))
    assert "Predicted_Failure" in batch_res.columns
    assert "Failure_Probability" in batch_res.columns
    assert "Risk_Level" in batch_res.columns
    print(f"[PASS] Batch prediction verified over 20 records")
    
    # 3. Test Anomaly Detection
    print("\n[3/6] Testing Anomaly Detection Engine...")
    from src.anomaly_detection import detect_anomalies
    anom_df = detect_anomalies(df)
    assert "Anomaly_Level" in anom_df.columns
    assert "Anomaly_Count" in anom_df.columns
    assert "Abnormal_Sensors" in anom_df.columns
    levels = anom_df["Anomaly_Level"].value_counts().to_dict()
    print(f"[PASS] Anomaly levels breakdown: {levels}")
    
    # 4. Test Database Operations
    print("\n[4/6] Testing SQLite Database Persistence...")
    from src.db_manager import (
        init_db,
        insert_prediction,
        get_predictions,
        insert_alert,
        update_alert_status,
        get_alerts,
        get_alert_counts,
        insert_maintenance,
        update_maintenance,
        delete_maintenance,
        get_maintenance_records,
        get_maintenance_counts
    )
    
    init_db()
    
    # Prediction DB Test
    p_id = insert_prediction(
        air_temp=298.5,
        process_temp=308.8,
        rotational_speed=1500,
        torque=42.0,
        tool_wear=50,
        predicted_failure=0,
        failure_probability=0.03,
        risk_level="Low",
        recommendation="Test Normal",
        equipment_id="TEST-MACHINE-01"
    )
    assert p_id > 0
    p_df = get_predictions(limit=10)
    assert not p_df.empty
    print(f"[PASS] Predictions DB verified: Logged ID #{p_id}")
    
    # Alerts DB Test
    a_id = insert_alert(
        equipment_id="TEST-ALERT-01",
        severity="Critical",
        condition="Simulated Power Failure",
        notes="Automated unit test alert"
    )
    assert a_id > 0
    update_alert_status(a_id, "Acknowledged")
    a_df = get_alerts()
    ack_row = a_df[a_df["id"] == a_id].iloc[0]
    assert ack_row["status"] == "Acknowledged"
    update_alert_status(a_id, "Resolved")
    a_df2 = get_alerts()
    res_row = a_df2[a_df2["id"] == a_id].iloc[0]
    assert res_row["status"] == "Resolved"
    print(f"[PASS] Alerts lifecycle verified: New -> Acknowledged -> Resolved (ID #{a_id})")
    
    # Maintenance DB Test
    m_id = insert_maintenance(
        equipment_id="TEST-MAINT-01",
        maintenance_type="Preventive Maintenance",
        description="Test maintenance task",
        priority="High",
        assigned_technician="Alex Kumar",
        scheduled_date="2026-12-01",
        status="Pending"
    )
    assert m_id > 0
    update_maintenance(
        record_id=m_id,
        maintenance_type="Preventive Maintenance",
        description="Updated test task",
        priority="Critical",
        assigned_technician="Sarah Jenkins",
        scheduled_date="2026-12-02",
        status="In Progress",
        notes="Test in progress"
    )
    m_df = get_maintenance_records()
    m_row = m_df[m_df["id"] == m_id].iloc[0]
    assert m_row["status"] == "In Progress"
    assert m_row["priority"] == "Critical"
    delete_maintenance(m_id)
    m_df_after = get_maintenance_records()
    assert m_id not in m_df_after["id"].values
    print(f"[PASS] Maintenance CRUD verified: Insert -> Update -> Delete (ID #{m_id})")
    
    # 5. Test Evaluation Metrics
    print("\n[5/8] Testing Evaluation Module & Metadata...")
    from src.trainmodel import METADATA_PATH
    assert METADATA_PATH.exists()
    import json
    with open(METADATA_PATH, "r") as f:
        meta = json.load(f)
    print(f"[PASS] Model metadata verified: Accuracy={meta['metrics']['accuracy']:.2%}, PR-AUC={meta['metrics']['pr_auc']:.2%}, F1={meta['metrics']['f1']:.2%}")

    # 6. Test Multiclass Model & Diagnostics (T38)
    print("\n[6/8] Testing Multiclass Failure-Type Model (T38)...")
    from src.train_multiclass import MULTICLASS_MODEL_PATH, MULTICLASS_METADATA_PATH, predict_failure_mode
    assert MULTICLASS_MODEL_PATH.exists(), "Multiclass model file missing"
    with open(MULTICLASS_METADATA_PATH, "r") as f:
        m_meta = json.load(f)
    print(f"[PASS] Multiclass model metadata: Accuracy={m_meta['accuracy']:.2%}, Weighted F1={m_meta['weighted_f1']:.2%}")
    test_mode, test_conf, _ = predict_failure_mode(
        air_temp=303.0, process_temp=309.0, rotational_speed=1250, torque=68.0, tool_wear=230, product_type="M"
    )
    print(f"[PASS] Diagnostic mode test: Predicted={test_mode}, Confidence={test_conf:.1%}")

    # 7. Test Advanced Analytics Suite (T39-T45) & Registered Asset CRUD
    print("\n[7/8] Testing Advanced Analytics & Asset CRUD (T39-T45)...")
    from src.advanced_analytics import (
        get_binary_vs_multiclass_report,
        get_isolation_forest_benchmark,
        get_decision_threshold_table,
        get_plant_generalization_report,
        get_streaming_simulation_data,
        get_alert_escalation_matrix
    )
    bvm = get_binary_vs_multiclass_report()
    assert len(bvm) == 3, "Binary vs multiclass rows mismatch"
    if_bench = get_isolation_forest_benchmark()
    assert len(if_bench) == 5, "Isolation Forest contamination benchmark mismatch"
    thresh_table = get_decision_threshold_table()
    assert len(thresh_table) >= 8, "Threshold table missing rows"
    plants = get_plant_generalization_report()
    assert len(plants) == 4, "Plant report missing plants"
    stream_sim = get_streaming_simulation_data()
    assert len(stream_sim) == 20, "Stream simulation ticks mismatch"
    esc_matrix = get_alert_escalation_matrix()
    assert len(esc_matrix) == 4, "Escalation matrix missing rows"
    print(f"[PASS] Advanced Analytics (T39–T45) verified successfully")

    # Asset CRUD Test
    from src.db_manager import insert_equipment, get_all_registered_equipment, update_equipment, delete_equipment
    test_eq_id = "VERIFY-CNC-99"
    insert_equipment(equipment_id=test_eq_id, product_type="H", location="Plant A (Main)", torque=52.0, tool_wear=40.0)
    all_eq = get_all_registered_equipment()
    assert test_eq_id in all_eq["equipment_id"].values
    update_equipment(equipment_id=test_eq_id, tool_wear=95.0, trajectory="Deteriorating", status="Warning")
    all_eq_updated = get_all_registered_equipment()
    eq_row = all_eq_updated[all_eq_updated["equipment_id"] == test_eq_id].iloc[0]
    assert eq_row["status"] == "Warning"
    assert eq_row["trajectory"] == "Deteriorating"
    delete_equipment(test_eq_id)
    all_eq_final = get_all_registered_equipment()
    assert test_eq_id not in all_eq_final["equipment_id"].values
    print(f"[PASS] Asset Registry CRUD verified: Register -> Update -> Decommission ({test_eq_id})")

    # 8. Syntax validation for all page files
    print("\n[8/8] Verifying Syntax of all Page Scripts...")
    page_files = [
        "app.py",
        "pages/1_Overview.py",
        "pages/2_Failure_Prediction.py",
        "pages/3_Equipment_Monitoring.py",
        "pages/4_Anomaly_Detection.py",
        "pages/5_Equipment_Alerts.py",
        "pages/6_Maintenance.py",
        "pages/7_Trends_Reports.py",
        "pages/8_Advanced_ML.py",
        "pages/9_Live_Monitoring.py"
    ]
    import py_compile
    for p in page_files:
        path = ROOT / p
        assert path.exists(), f"Page file does not exist: {path}"
        py_compile.compile(str(path), doraise=True)
        print(f"[OK] Compiled {p} with zero syntax errors")
        
    print("\n==================================================")
    print("ALL 8 VERIFICATION TEST SUITES PASSED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    run_tests()
