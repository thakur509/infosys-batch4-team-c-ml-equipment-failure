import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.styles import apply_styles, apply_chart_theme, render_hero, render_sidebar_status
from src.data_loader import load_data, RAW_NUMERIC_FEATURES
from src.anomaly_detection import detect_anomalies, compute_iqr_thresholds

st.set_page_config(
    page_title="Anomaly Detection | Predictive Maintenance",
    page_icon="🔍",
    layout="wide"
)
apply_styles()
render_sidebar_status()

# Page Hero
render_hero(
    title="Sensor Anomaly & Outlier Detection Engine",
    subtitle="Detect abnormal sensor telemetry outside nominal operating thresholds. Combines 1.5× & 2.5× IQR statistical boundary breaches with multi-variate Isolation Forest scoring to identify equipment deviations before mechanical breakdown occurs.",
    tag="Statistical Process Control & Outlier Engine"
)

# Load data with detected anomalies
df = load_data()
anomalies_df = detect_anomalies(df)
thresholds = compute_iqr_thresholds(df)

# Anomaly Summary Metrics
total_records = len(anomalies_df)
crit_anom = int((anomalies_df["Anomaly_Level"] == "Critical").sum())
warn_anom = int((anomalies_df["Anomaly_Level"] == "Warning").sum())
healthy_readings = total_records - (crit_anom + warn_anom)
flagged_total = crit_anom + warn_anom

kpi_cols = st.columns(4)
with kpi_cols[0]:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Flagged Readings</div>
        <div class="metric-value" style="color:#60A5FA;">{flagged_total:,}</div>
        <div class="metric-subtext">{flagged_total/total_records*100:.1f}% of fleet records</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_cols[1]:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Critical Anomalies</div>
        <div class="metric-value" style="color:#F87171;">{crit_anom:,}</div>
        <div class="metric-subtext">Severe sensor threshold breach</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_cols[2]:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Warning Anomalies</div>
        <div class="metric-value" style="color:#FBBF24;">{warn_anom:,}</div>
        <div class="metric-subtext">Moderate deviation from norm</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_cols[3]:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Healthy Readings</div>
        <div class="metric-value" style="color:#34D399;">{healthy_readings:,}</div>
        <div class="metric-subtext">{healthy_readings/total_records*100:.1f}% within normal bounds</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

# Search, Filter & CSV Export Section
st.markdown('<div class="content-card">', unsafe_allow_html=True)
c_tab, c_search, c_download = st.columns([2, 1.5, 1])

with c_tab:
    selected_tier = st.radio(
        "Filter by Severity",
        ["All Records", "Critical", "Warning", "Healthy"],
        horizontal=True
    )

with c_search:
    query = st.text_input("Quick search by Equipment ID", placeholder="e.g. M14860 or L47230")

# Apply filters
view_df = anomalies_df.copy()
if selected_tier != "All Records":
    view_df = view_df[view_df["Anomaly_Level"] == selected_tier]

if query:
    view_df = view_df[view_df["Product ID"].astype(str).str.contains(query.strip(), case=False, na=False)]

with c_download:
    st.markdown("<div style='padding-top:1.75rem;'></div>", unsafe_allow_html=True)
    csv_out = view_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"📥 Export ({len(view_df):,}) CSV",
        data=csv_out,
        file_name=f"anomaly_readings_{selected_tier.lower()}.csv",
        mime="text/csv",
        use_container_width=True
    )

st.markdown('</div>', unsafe_allow_html=True)

# Anomaly Table
st.markdown('<div class="content-card">', unsafe_allow_html=True)
st.markdown(f'<div class="card-header-title">Anomaly Telemetry Records ({len(view_df):,} matching)</div>', unsafe_allow_html=True)
st.markdown('<div class="card-header-desc">Sensor readings flagged for exceeding IQR limits or triggering multivariate isolation trees:</div>', unsafe_allow_html=True)

table_columns = [
    "Product ID", "Type", "Anomaly_Level", "Anomaly_Count", "Abnormal_Sensors",
    "Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]",
    "Torque [Nm]", "Tool wear [min]"
]
table_rename = {
    "Product ID": "Equipment ID",
    "Type": "Type",
    "Anomaly_Level": "Anomaly Level",
    "Anomaly_Count": "Abnormal Count",
    "Abnormal_Sensors": "Breached Sensor Parameters",
    "Air temperature [K]": "Air Temp (K)",
    "Process temperature [K]": "Proc Temp (K)",
    "Rotational speed [rpm]": "Speed (RPM)",
    "Torque [Nm]": "Torque (Nm)",
    "Tool wear [min]": "Tool Wear (min)"
}

display_anomaly_table = view_df[table_columns].rename(columns=table_rename)
st.dataframe(
    display_anomaly_table.head(200),
    use_container_width=True,
    hide_index=True,
    height=480
)
st.caption("Showing first 200 records. Use search or download the full CSV export above.")
st.markdown('</div>', unsafe_allow_html=True)

# Visualizations: Multi-Sensor Outlier Distribution with dynamic axis controls
st.markdown('<div class="content-card">', unsafe_allow_html=True)
st.markdown('<div class="card-header-title">Sensor Operating Envelopes & Anomaly Visualization</div>', unsafe_allow_html=True)
st.markdown('<div class="card-header-desc">Interactive bivariate scatter plot with selectable axes and color-coded anomaly classifications:</div>', unsafe_allow_html=True)

# Axis selector controls
ax_col1, ax_col2 = st.columns(2)
with ax_col1:
    x_sensor = st.selectbox("X-Axis Telemetry Parameter", RAW_NUMERIC_FEATURES, index=2)  # Rotational speed
with ax_col2:
    y_sensor = st.selectbox("Y-Axis Telemetry Parameter", RAW_NUMERIC_FEATURES, index=3)  # Torque

v1, v2 = st.columns([1.5, 1])

with v1:
    sample_size = min(2500, len(anomalies_df))
    sampled_anom = anomalies_df.sample(sample_size, random_state=42)
    fig_scatter = px.scatter(
        sampled_anom,
        x=x_sensor,
        y=y_sensor,
        color="Anomaly_Level",
        color_discrete_map={"Healthy": "#10B981", "Warning": "#F59E0B", "Critical": "#EF4444"},
        hover_data=["Product ID", "Abnormal_Sensors", "Tool wear [min]"],
        title=f"{x_sensor.split(' [')[0]} vs. {y_sensor.split(' [')[0]} Operating Envelope"
    )
    apply_chart_theme(fig_scatter, height=380)
    fig_scatter.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with v2:
    # Breached Sensors Frequency Bar Chart
    breached_counts = {}
    for col in RAW_NUMERIC_FEATURES:
        t = thresholds[col]
        count = int(((df[col] < t["lower_warning"]) | (df[col] > t["upper_warning"])).sum())
        clean_name = col.split(" [")[0]
        breached_counts[clean_name] = count
        
    sensor_bar_df = pd.DataFrame(list(breached_counts.items()), columns=["Sensor", "Outlier Readings"]).sort_values("Outlier Readings", ascending=True)
    
    fig_sensor_bar = px.bar(
        sensor_bar_df,
        x="Outlier Readings",
        y="Sensor",
        orientation="h",
        color="Outlier Readings",
        color_continuous_scale="Reds",
        text="Outlier Readings",
        title="Frequency of Outlier Readings by Sensor"
    )
    apply_chart_theme(fig_sensor_bar, height=380)
    fig_sensor_bar.update_layout(
        coloraxis_showscale=False,
        yaxis_title=""
    )
    st.plotly_chart(fig_sensor_bar, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)
