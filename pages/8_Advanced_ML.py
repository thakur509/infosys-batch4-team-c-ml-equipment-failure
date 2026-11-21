import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.styles import apply_styles, apply_chart_theme, render_hero, render_sidebar_status
from src.train_multiclass import MULTICLASS_METADATA_PATH, train_multiclass_model
from src.advanced_analytics import (
    get_binary_vs_multiclass_report,
    get_isolation_forest_benchmark,
    get_feature_attribution_report,
    get_decision_threshold_table,
    get_plant_generalization_report,
    get_streaming_simulation_data,
    get_alert_escalation_matrix
)

st.set_page_config(
    page_title="Advanced ML Tasks (T38–T45) | Predictive Maintenance",
    page_icon="⚡",
    layout="wide"
)
apply_styles()
render_sidebar_status()

# Page Hero
render_hero(
    title="Advanced Machine Learning Tasks (T38–T45)",
    subtitle="Interactive telemetry intelligence suite: Specific Failure-Type Classification, Binary vs. Multiclass Benchmarks, Isolation Forest Anomaly Evaluation, Feature Explainability, Decision Threshold Sensitivity, Multi-Plant Generalization, and 20-Tick Streaming Pipeline Simulation.",
    tag="Deep ML Research & Model Diagnostics"
)

# Load multiclass metadata
multi_meta = {}
if MULTICLASS_METADATA_PATH.exists():
    with open(MULTICLASS_METADATA_PATH, "r", encoding="utf-8") as f:
        multi_meta = json.load(f)
else:
    _, multi_meta = train_multiclass_model()

# Advanced Tabs matching reference platform
t38_tab, t39_tab, t40_tab, t41_tab, t42_tab, t43_tab, t44_tab, t45_tab = st.tabs([
    "T38 Failure-Type Model",
    "T39 Binary vs Multiclass",
    "T40 Isolation Forest",
    "T41 Explainability",
    "T42 Threshold PR",
    "T43 Generalization",
    "T44 Streaming Pipeline",
    "T45 Alert Escalation"
])

# ==========================================
# T38: Specific Failure-Type Multiclass Model
# ==========================================
with t38_tab:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header-title">T38 · Specific Failure-Type Multiclass Model (<code>models/failure_type_model.joblib</code>)</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="card-header-desc">Predicts specific AI4I 2020 failure classes (<code>No Failure</code>, <code>HDF</code>, <code>OSF</code>, <code>PWF</code>, <code>TWF</code>, <code>RNF</code>) instead of only binary overall failure. Overall Multiclass Accuracy: <strong>{multi_meta.get("accuracy", 0.9810):.2%}</strong> · Weighted F1: <strong>{multi_meta.get("weighted_f1", 0.9826):.2%}</strong>.</div>', unsafe_allow_html=True)

    class_table = multi_meta.get("class_table", [])
    if class_table:
        class_df = pd.DataFrame(class_table)
        st.dataframe(class_df, use_container_width=True, hide_index=True)

    # Bar chart of class F1-scores
    chart_data = [row for row in class_table if not row["Failure Class"].endswith("avg")]
    if chart_data:
        cf1_df = pd.DataFrame(chart_data)
        cf1_df["F1_Num"] = cf1_df["F1-Score"].str.rstrip("%").astype(float)
        fig_cf1 = px.bar(
            cf1_df,
            x="Failure Class",
            y="F1_Num",
            color="F1_Num",
            color_continuous_scale="Purples",
            text="F1-Score",
            title="Per-Class F1-Score Performance"
        )
        apply_chart_theme(fig_cf1, height=320)
        fig_cf1.update_layout(coloraxis_showscale=False, yaxis_title="F1-Score (%)")
        st.plotly_chart(fig_cf1, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# T39: Binary vs Multiclass Comparison
# ==========================================
with t39_tab:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header-title">T39 · Binary vs. Multiclass Classification Comparison Report</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-header-desc">Side-by-side comparative analysis on the 2,000 held-out test observations comparing binary failure screening with 6-class specific failure-type diagnosis:</div>', unsafe_allow_html=True)

    comparison_data = get_binary_vs_multiclass_report()
    comp_df = pd.DataFrame(comparison_data)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # Comparison metrics chart
    st.markdown("<hr style='margin:1.2rem 0; border:none; border-top:1px solid #1F2937;'>", unsafe_allow_html=True)
    c_m1, c_m2 = st.columns(2)
    with c_m1:
        st.markdown("<div style='font-weight:700; font-size:0.95rem; margin-bottom:0.4rem; color:#F8FAFC;'>Macro Precision vs Recall Comparison</div>", unsafe_allow_html=True)
        bar_data = pd.DataFrame({
            "Model": ["Binary Screening", "Multiclass Diagnosis", "Collapsed Multiclass"],
            "Precision": [96.23, 61.46, 70.13],
            "Recall": [75.00, 65.08, 79.41]
        }).melt(id_vars="Model", var_name="Metric", value_name="Score (%)")
        fig_pr = px.bar(bar_data, x="Model", y="Score (%)", color="Metric", barmode="group", color_discrete_map={"Precision": "#3B82F6", "Recall": "#10B981"})
        apply_chart_theme(fig_pr, height=300)
        st.plotly_chart(fig_pr, use_container_width=True)
    with c_m2:
        st.markdown("<div style='font-weight:700; font-size:0.95rem; margin-bottom:0.4rem; color:#F8FAFC;'>Architectural Recommendation</div>", unsafe_allow_html=True)
        st.markdown("""
        <div style="background:rgba(30, 41, 59, 0.7); border:1px solid #334155; border-radius:10px; padding:1.2rem; font-size:0.88rem; line-height:1.6; color:#CBD5E1;">
            <strong style="color:#F8FAFC;">Two-Stage Cascade Architecture:</strong><br>
            • <strong>Stage 1 (Binary Screening):</strong> High-precision triage (96.2% precision) to immediately flag failing units with minimal false alarms.<br>
            • <strong>Stage 2 (Multiclass Root Cause):</strong> Dispatches specific engineering prescriptions (HDF convective check, PWF motor check, OSF tool replacement).
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# T40: Isolation Forest Anomaly Detection
# ==========================================
with t40_tab:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header-title">T40 · Isolation Forest Anomaly Detection Benchmarking</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-header-desc">Compares unsupervised multivariate <code>IsolationForest</code> across contamination rates ($c \\in [0.034, 0.10]$) against univariate IQR rule-based bounds on the 2,000 test records (68 true failures):</div>', unsafe_allow_html=True)

    iso_data = get_isolation_forest_benchmark()
    iso_df = pd.DataFrame(iso_data)
    st.dataframe(iso_df, use_container_width=True, hide_index=True)

    # Chart showing captured failures vs contamination
    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
    iso_chart_df = pd.DataFrame({
        "Method": ["Univariate IQR", "Isolation Forest (3.4%)", "Isolation Forest (5.0%)", "Isolation Forest (8.0%)", "Isolation Forest (10.0%)"],
        "Recall (%)": [19.12, 23.53, 33.82, 50.00, 55.88],
        "Failures Caught": [13, 16, 23, 34, 38]
    })
    fig_iso = px.line(
        iso_chart_df,
        x="Method",
        y="Recall (%)",
        markers=True,
        text="Failures Caught",
        title="Unsupervised True Failure Capture Rate Across Contamination Regimes"
    )
    apply_chart_theme(fig_iso, height=300)
    fig_iso.update_layout(yaxis_title="Failure Recall (%)")
    st.plotly_chart(fig_iso, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# T41: Explainability Report
# ==========================================
with t41_tab:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header-title">T41 · Explainability Report: Exact TreeSHAP & Feature Importance</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-header-desc">Global ranking of engineered physical features computed across test telemetry:</div>', unsafe_allow_html=True)

    shap_data = get_feature_attribution_report()
    shap_df = pd.DataFrame(shap_data)
    st.dataframe(shap_df, use_container_width=True, hide_index=True)

    # Horizontal bar of SHAP share
    shap_chart_df = pd.DataFrame({
        "Feature": [r["Feature"] for r in shap_data],
        "Impact Share (%)": [float(r["SHAP Impact Share"].rstrip("%")) for r in shap_data]
    }).sort_values("Impact Share (%)", ascending=True)

    fig_shap = px.bar(
        shap_chart_df,
        x="Impact Share (%)",
        y="Feature",
        orientation="h",
        color="Impact Share (%)",
        color_continuous_scale="Teal",
        text="Impact Share (%)",
        title="Relative Contribution to Failure Decision Boundary"
    )
    apply_chart_theme(fig_shap, height=340)
    fig_shap.update_layout(coloraxis_showscale=False, yaxis_title="")
    st.plotly_chart(fig_shap, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# T42: Decision Threshold Sensitivity
# ==========================================
with t42_tab:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header-title">T42 · Decision Threshold & Precision-Recall Comparison (PR-AUC: 88.24%)</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-header-desc">Examines how Precision, Recall, F1-Score, F2-Score, False Positives, and Missed Failures change across decision thresholds $t \\in [0.10, 0.90]$:</div>', unsafe_allow_html=True)

    # Interactive slider for threshold tuning
    thresh_val = st.slider("Select Operating Threshold (t)", min_value=0.10, max_value=0.90, value=0.50, step=0.05)
    
    t_table = get_decision_threshold_table()
    t_df = pd.DataFrame(t_table)
    st.dataframe(t_df, use_container_width=True, hide_index=True)

    # Precision-Recall trade-off curve
    pr_curve_df = pd.DataFrame({
        "Threshold": [r["Threshold (t)"] for r in t_table],
        "Precision": [float(r["Precision"].rstrip("%")) for r in t_table],
        "Recall": [float(r["Recall"].rstrip("%")) for r in t_table],
        "F1": [float(r["F1-Score"].rstrip("%")) for r in t_table]
    })
    
    fig_thresh = go.Figure()
    fig_thresh.add_trace(go.Scatter(x=pr_curve_df["Threshold"], y=pr_curve_df["Precision"], mode="lines+markers", name="Precision", line=dict(color="#3B82F6", width=2)))
    fig_thresh.add_trace(go.Scatter(x=pr_curve_df["Threshold"], y=pr_curve_df["Recall"], mode="lines+markers", name="Recall", line=dict(color="#10B981", width=2)))
    fig_thresh.add_trace(go.Scatter(x=pr_curve_df["Threshold"], y=pr_curve_df["F1"], mode="lines+markers", name="F1-Score", line=dict(color="#8B5CF6", width=3)))
    fig_thresh.add_vline(x=thresh_val, line_dash="dash", line_color="#EF4444", annotation_text=f"Current: t={thresh_val:.2f}", annotation_font_color="#F8FAFC")
    
    apply_chart_theme(fig_thresh, height=340, title="Precision, Recall, and F1-Score as a Function of Decision Threshold (t)")
    fig_thresh.update_layout(
        xaxis_title="Decision Threshold (t)",
        yaxis_title="Percentage (%)",
        legend=dict(orientation="h", y=-0.22)
    )
    st.plotly_chart(fig_thresh, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# T43: Cross-Dataset Generalization
# ==========================================
with t43_tab:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header-title">T43 · Cross-Dataset Model Generalization Report</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-header-desc">Evaluates the production model without retraining across three external plant regimes (1,500 stress samples: High-Ambient $+3.5\\text{ K}$ shift, Heavy-Load torque shift, and Noisy legacy IoT sensors):</div>', unsafe_allow_html=True)

    gen_data = get_plant_generalization_report()
    gen_df = pd.DataFrame(gen_data)
    st.dataframe(gen_df, use_container_width=True, hide_index=True)

    # Bar chart of generalization accuracy
    fig_gen = px.bar(
        gen_df,
        x="Dataset / Plant Domain",
        y=[float(r["Accuracy"].rstrip("%")) for r in gen_data],
        color="Dataset / Plant Domain",
        text=[r["Accuracy"] for r in gen_data],
        title="Accuracy Retention Across Plant Distribution Shifts"
    )
    apply_chart_theme(fig_gen, height=300)
    fig_gen.update_layout(yaxis_title="Accuracy (%)", showlegend=False)
    st.plotly_chart(fig_gen, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# T44: Streaming Telemetry Pipeline Simulator
# ==========================================
with t44_tab:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header-title">T44 · Simulated Real-Time Sensor-Monitoring Pipeline Prototype (<code>SIM-CNC-01</code>)</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-header-desc">20-tick streaming telemetry simulation progressing from Normal Operation → Early Degradation → Critical Failure Stress → Post-Maintenance Reset:</div>', unsafe_allow_html=True)

    stream_records = get_streaming_simulation_data()
    sim_tick = st.slider("Telemetry Stream Tick Position", min_value=1, max_value=20, value=12, step=1)
    current_tick_data = stream_records[sim_tick - 1]
    
    # Display tick status cards
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.metric("Current Stream Phase", current_tick_data["Phase"])
    with s2:
        st.metric("Spindle Torque / Speed", f"{current_tick_data['Torque']} Nm", f"{current_tick_data['Speed']} RPM")
    with s3:
        st.metric("Tool Wear", f"{current_tick_data['Wear']} min", "Limit < 200 min")
    with s4:
        st.metric("Health Score Index", f"{current_tick_data['Health']} / 100", f"Risk: {current_tick_data['Risk']}")

    # Streaming trajectory curve
    tick_df = pd.DataFrame(stream_records)
    tick_df["Prob_Num"] = tick_df["Prob"].str.rstrip("%").astype(float)
    
    fig_stream = go.Figure()
    fig_stream.add_trace(go.Scatter(x=tick_df["Tick"], y=tick_df["Health"], mode="lines+markers", name="Health Score (0-100)", line=dict(color="#10B981", width=3)))
    fig_stream.add_trace(go.Scatter(x=tick_df["Tick"], y=tick_df["Prob_Num"], mode="lines+markers", name="Failure Probability (%)", line=dict(color="#EF4444", width=3)))
    fig_stream.add_vline(x=f"#{sim_tick}", line_dash="dash", line_color="#6366F1", annotation_text=f"Current: #{sim_tick}", annotation_font_color="#F8FAFC")
    
    apply_chart_theme(fig_stream, height=320, title="SIM-CNC-01 Continuous Health & Risk Trajectory over 20 Ticks")
    fig_stream.update_layout(
        xaxis_title="Telemetry Cycle Tick",
        yaxis_title="Index / Percentage",
        legend=dict(orientation="h", y=-0.22)
    )
    st.plotly_chart(fig_stream, use_container_width=True)

    # Complete 20-tick table
    with st.expander("📋 View Complete 20-Tick Streaming Log"):
        st.dataframe(pd.DataFrame(stream_records), use_container_width=True, hide_index=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# T45: Automated Alert Escalation Matrix
# ==========================================
with t45_tab:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header-title">T45 · Automated Alert Dispatch & SLA Escalation Matrix</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-header-desc">Standard Operating Procedures (SOP) and automated routing matrix for industrial shop-floor integration:</div>', unsafe_allow_html=True)

    esc_data = get_alert_escalation_matrix()
    esc_df = pd.DataFrame(esc_data)
    st.dataframe(esc_df, use_container_width=True, hide_index=True)

    st.markdown("""
    <div style="background:rgba(30, 41, 59, 0.7); border-left:4px solid #6366F1; padding:1rem; border-radius:8px; margin-top:1rem; font-size:0.88rem; line-height:1.55; color:#CBD5E1;">
        <strong style="color:#F8FAFC;">Enterprise Integration Hook:</strong><br>
        When an alert status transitions to <code>New</code> in the Equipment Alerts registry with severity <code>Critical</code>, our system invokes the maintenance database dispatcher to automatically generate an urgent work order and notify on-duty technicians.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
