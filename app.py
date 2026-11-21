"""Main entrypoint for Predictive Maintenance & Machine Failure Dashboard."""

import streamlit as st
from pathlib import Path

from src.data_loader import load_data
from src.trainmodel import train_model, MODEL_PATH
from src.train_multiclass import train_multiclass_model, MULTICLASS_MODEL_PATH
from src.db_manager import init_db, seed_initial_data

# Cold start initialization
try:
    df = load_data()
    seed_initial_data(df)
    if not MODEL_PATH.exists():
        train_model(df)
    if not MULTICLASS_MODEL_PATH.exists():
        train_multiclass_model(df)
except Exception as e:
    print(f"Startup initialization notice: {e}")

# Modern Streamlit Navigation (Streamlit 1.36+)
pages = [
    st.Page("src/pages/1_Overview.py", title="Overview", icon="📊", default=True),
    st.Page("src/pages/2_Failure_Prediction.py", title="Failure Prediction", icon="🔮"),
    st.Page("src/pages/3_Equipment_Monitoring.py", title="Equipment Monitoring", icon="⚙️"),
    st.Page("src/pages/4_Anomaly_Detection.py", title="Anomaly Detection", icon="🔍"),
    st.Page("src/pages/5_Equipment_Alerts.py", title="Equipment Alerts", icon="🚨"),
    st.Page("src/pages/6_Maintenance.py", title="Maintenance", icon="🛠️"),
    st.Page("src/pages/7_Trends_Reports.py", title="Trends & Reports", icon="📈"),
    st.Page("src/pages/8_Advanced_ML.py", title="Advanced ML (T38–T45)", icon="⚡"),
    st.Page("src/pages/9_Live_Monitoring.py", title="Live Machine Health", icon="📡"),
]

pg = st.navigation(pages)
pg.run()
