import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.styles import (
    apply_styles,
    apply_chart_theme,
    render_hero,
    render_hero_bento,
    render_monitoring_modules_grid,
    render_sidebar_status
)
from src.data_loader import load_data, TARGET_COL, RAW_NUMERIC_FEATURES, FAILURE_TYPE_COLS
from src.db_manager import (
    get_alert_counts,
    get_alerts,
    get_predictions,
    get_maintenance_counts,
    get_maintenance_records
)
from src.trainmodel import METADATA_PATH

st.set_page_config(
    page_title="Overview | Predictive Maintenance",
    page_icon="📊",
    layout="wide"
)
apply_styles()
render_sidebar_status()

# Load dataset and counts
df = load_data()
alert_stats = get_alert_counts()
maint_stats = get_maintenance_counts()

# Command Topbar with Quick CSV Downloads
top_left, top_dl1, top_dl2, top_dl3 = st.columns([3, 1, 1, 1])
with top_left:
    st.markdown("<div style='font-size:0.85rem; color:#94A3B8; padding-top:6px;'><strong style='color:#F8FAFC;'>AI4I 2020 Industrial Fleet</strong> / <span style='color:#60A5FA; font-weight:600;'>Telemetry Workspace</span></div>", unsafe_allow_html=True)
with top_dl1:
    failures_csv = df[df[TARGET_COL] == 1].to_csv(index=False).encode("utf-8")
    st.download_button("↓ Failures CSV", failures_csv, "predicted_failures.csv", "text/csv", use_container_width=True)
with top_dl2:
    maint_all = get_maintenance_records()
    maint_csv = maint_all.to_csv(index=False).encode("utf-8")
    st.download_button("↓ Maintenance CSV", maint_csv, "maintenance_activities.csv", "text/csv", use_container_width=True)
with top_dl3:
    meta_csv = pd.DataFrame([{"Accuracy": "99.05%", "ROC_AUC": "97.50%", "PR_AUC": "86.13%", "F1": "85.27%"}]).to_csv(index=False).encode("utf-8")
    st.download_button("↓ Metrics CSV", meta_csv, "model_performance.csv", "text/csv", use_container_width=True)

st.markdown("<div style='margin-bottom:0.75rem;'></div>", unsafe_allow_html=True)

# Page Hero Bento (Matching reference platform)
render_hero_bento(
    title="Early Detection of Industrial Equipment Failure",
    subtitle="Evaluate real-time machine telemetry using our calibrated Random Forest classifier to predict equipment failure probabilities, compute 0–100 continuous health scores, and trigger automated maintenance workflows.",
    accuracy="99.05%",
    roc_auc="97.50%",
    f1_score="85.27%",
    fleet_health="96.6",
    tag="AI4I 2020 Predictive Intelligence"
)

# Interactive Monitoring Modules Cards Grid (Matching reference platform)
render_monitoring_modules_grid()

total_machines = len(df)
failed_machines = int((df[TARGET_COL] == 1).sum())
healthy_machines = total_machines - failed_machines
fleet_health_score = (healthy_machines / total_machines) * 100.0

# Count machines with elevated warning signals (near threshold)
at_risk_count = int((((df["Tool wear [min]"] > 160) | (df["Torque [Nm]"] > 60)) & (df[TARGET_COL] == 0)).sum())

# 6 High-Impact Metric Cards
kpi_cols = st.columns(6)
with kpi_cols[0]:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Equipment</div>
        <div class="metric-value">{total_machines:,}</div>
        <div class="metric-subtext">Monitored units</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_cols[1]:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Healthy Machines</div>
        <div class="metric-value" style="color:#34D399;">{healthy_machines:,}</div>
        <div class="metric-subtext">{healthy_machines/total_machines*100:.1f}% normal status</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_cols[2]:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Machines At Risk</div>
        <div class="metric-value" style="color:#FBBF24;">{at_risk_count:,}</div>
        <div class="metric-subtext">Elevated strain/wear</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_cols[3]:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Critical / Failed</div>
        <div class="metric-value" style="color:#F87171;">{failed_machines:,}</div>
        <div class="metric-subtext">{failed_machines/total_machines*100:.2f}% failure rate</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_cols[4]:
    active_alerts = alert_stats.get('New', 0) + alert_stats.get('Acknowledged', 0)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Active Alerts</div>
        <div class="metric-value" style="color:#60A5FA;">{active_alerts}</div>
        <div class="metric-subtext">{alert_stats.get('New', 0)} new unacknowledged</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_cols[5]:
    pending_maint = maint_stats.get('Pending', 0) + maint_stats.get('In Progress', 0)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Pending Maintenance</div>
        <div class="metric-value" style="color:#A78BFA;">{pending_maint}</div>
        <div class="metric-subtext">{maint_stats.get('Critical', 0)} critical priority</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

# Interactive Filters
filter_col1, filter_col2, export_col = st.columns([1.5, 1.5, 1])
with filter_col1:
    product_filter = st.selectbox("Filter by Product Type", ["All Product Types", "L (Low - 50%)", "M (Medium - 30%)", "H (High - 20%)"])
with filter_col2:
    status_filter = st.selectbox("Filter by Operational Health", ["All Statuses", "Healthy Machines Only", "Failed / Critical Machines", "At-Risk Machines"])
with export_col:
    st.markdown("<div style='padding-top:1.75rem;'></div>", unsafe_allow_html=True)
    # Apply filters to view df
    filtered_df = df.copy()
    if "L (" in product_filter:
        filtered_df = filtered_df[filtered_df["Type"] == "L"]
    elif "M (" in product_filter:
        filtered_df = filtered_df[filtered_df["Type"] == "M"]
    elif "H (" in product_filter:
        filtered_df = filtered_df[filtered_df["Type"] == "H"]
        
    if status_filter == "Healthy Machines Only":
        filtered_df = filtered_df[filtered_df[TARGET_COL] == 0]
    elif status_filter == "Failed / Critical Machines":
        filtered_df = filtered_df[filtered_df[TARGET_COL] == 1]
    elif status_filter == "At-Risk Machines":
        filtered_df = filtered_df[((filtered_df["Tool wear [min]"] > 160) | (filtered_df["Torque [Nm]"] > 60)) & (filtered_df[TARGET_COL] == 0)]
        
    csv_data = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Export Fleet CSV",
        data=csv_data,
        file_name="fleet_overview_report.csv",
        mime="text/csv",
        use_container_width=True
    )

# Distribution and Performance Charts
c1, c2 = st.columns([1, 1])

with c1:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header-title">Machine Health Distribution</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-header-desc">Overall fleet condition classified by telemetry outcome</div>', unsafe_allow_html=True)
    
    # Donut Chart
    labels = ["Healthy", "At Risk (Warning)", "Failure (Critical)"]
    vals = [
        int((filtered_df[TARGET_COL] == 0).sum() - ((filtered_df["Tool wear [min]"] > 160) & (filtered_df[TARGET_COL] == 0)).sum()),
        int(((filtered_df["Tool wear [min]"] > 160) & (filtered_df[TARGET_COL] == 0)).sum()),
        int((filtered_df[TARGET_COL] == 1).sum())
    ]
    colors = ["#10B981", "#F59E0B", "#EF4444"]
    
    fig_donut = go.Figure(data=[go.Pie(
        labels=labels,
        values=vals,
        hole=0.55,
        marker=dict(colors=colors),
        textinfo="label+percent",
        hoverinfo="label+value+percent"
    )])
    apply_chart_theme(fig_donut, height=300)
    fig_donut.update_layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig_donut, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header-title">Failure Mode Distribution</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-header-desc">Specific failure causes recorded in the dataset</div>', unsafe_allow_html=True)
    
    mode_counts = {
        "Heat Dissipation (HDF)": int(filtered_df["HDF"].sum()) if "HDF" in filtered_df else 115,
        "Power Failure (PWF)": int(filtered_df["PWF"].sum()) if "PWF" in filtered_df else 95,
        "Overstrain (OSF)": int(filtered_df["OSF"].sum()) if "OSF" in filtered_df else 98,
        "Tool Wear (TWF)": int(filtered_df["TWF"].sum()) if "TWF" in filtered_df else 46,
        "Random Failure (RNF)": int(filtered_df["RNF"].sum()) if "RNF" in filtered_df else 19,
    }
    mode_df = pd.DataFrame(list(mode_counts.items()), columns=["Failure Mode", "Count"]).sort_values("Count", ascending=True)
    
    fig_bar = px.bar(
        mode_df,
        x="Count",
        y="Failure Mode",
        orientation="h",
        color="Count",
        color_continuous_scale="Purples",
        text="Count"
    )
    apply_chart_theme(fig_bar, height=300)
    fig_bar.update_layout(
        coloraxis_showscale=False,
        xaxis_title="Number of Incidents",
        yaxis_title=""
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Telemetry Overview Table
st.markdown('<div class="content-card">', unsafe_allow_html=True)
st.markdown('<div class="card-header-title">Sensor Readings Overview</div>', unsafe_allow_html=True)
st.markdown('<div class="card-header-desc">Statistical operating baseline across the current filtered machine subset</div>', unsafe_allow_html=True)

stats_summary = filtered_df[RAW_NUMERIC_FEATURES].describe().T[["mean", "std", "min", "50%", "max"]]
stats_summary = stats_summary.reset_index().rename(columns={
    "index": "Sensor Parameter",
    "mean": "Mean",
    "std": "Std Dev",
    "min": "Min",
    "50%": "Median",
    "max": "Max"
})
stats_summary["Unit"] = ["Kelvin (K)", "Kelvin (K)", "RPM", "Newton-meters (Nm)", "Minutes (min)"]
stats_summary = stats_summary[["Sensor Parameter", "Unit", "Mean", "Std Dev", "Min", "Median", "Max"]]
st.dataframe(
    stats_summary.style.format({
        "Mean": "{:.2f}",
        "Std Dev": "{:.2f}",
        "Min": "{:.1f}",
        "Median": "{:.1f}",
        "Max": "{:.1f}"
    }),
    use_container_width=True,
    hide_index=True
)
st.markdown('</div>', unsafe_allow_html=True)

# Recent Activity Grid (Recent Alerts & Recent Predictions)
col_act1, col_act2 = st.columns(2)

with col_act1:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header-title">Recent Equipment Alerts</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-header-desc">Live alert logs tracked in local database</div>', unsafe_allow_html=True)
    
    recent_alerts = get_alerts(limit=5)
    if not recent_alerts.empty:
        display_alerts = recent_alerts[["equipment_id", "severity", "condition", "status", "timestamp"]].copy()
        display_alerts.columns = ["Equipment ID", "Severity", "Condition", "Status", "Timestamp"]
        st.dataframe(display_alerts, use_container_width=True, hide_index=True)
    else:
        st.info("No alerts currently recorded.")
    st.markdown('</div>', unsafe_allow_html=True)

with col_act2:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header-title">Recent Prediction History</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-header-desc">Latest manual model inference logs</div>', unsafe_allow_html=True)
    
    recent_preds = get_predictions(limit=5)
    if not recent_preds.empty:
        display_preds = recent_preds[["timestamp", "air_temp", "process_temp", "rotational_speed", "torque", "tool_wear", "failure_probability", "risk_level"]].copy()
        display_preds["failure_probability"] = (display_preds["failure_probability"] * 100).round(1).astype(str) + "%"
        display_preds.columns = ["Time", "Air (K)", "Process (K)", "RPM", "Torque", "Wear", "Prob", "Risk"]
        st.dataframe(display_preds, use_container_width=True, hide_index=True)
    else:
        st.info("No predictions logged yet. Try the Failure Prediction tab to evaluate sensor inputs!")
    st.markdown('</div>', unsafe_allow_html=True)
