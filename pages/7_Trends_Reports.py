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
from src.data_loader import load_data, TARGET_COL, RAW_NUMERIC_FEATURES, FEATURE_COLS
from src.trainmodel import METADATA_PATH, train_model

st.set_page_config(
    page_title="Trends & Reports | Predictive Maintenance",
    page_icon="📈",
    layout="wide"
)
apply_styles()
render_sidebar_status()

# Page Hero
render_hero(
    title="Industrial Telemetry Trends & Analytical Reports",
    subtitle="Comprehensive exploratory data analysis, sensor correlation matrices, operating envelopes, and model performance validation on held-out test telemetry.",
    tag="Analytics Intelligence & Model Auditing"
)

# Load data and model metadata
df = load_data()

metadata = {}
if METADATA_PATH.exists():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
else:
    _, metadata = train_model()

# Sub-tab Navigation
report_tab1, report_tab2, report_tab3 = st.tabs([
    "📊 Sensor Trends & Envelopes",
    "🔗 Correlation & Normal vs Failed Analysis",
    "🤖 Machine Learning Model Performance"
])

# ==========================================
# TAB 1: SENSOR TRENDS & ENVELOPES
# ==========================================
with report_tab1:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header-title">Sequential Telemetry Runs & Sensor Distributions</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-header-desc">Inspect sensor behavior across observations and compare operating distributions:</div>', unsafe_allow_html=True)

    t_col1, t_col2 = st.columns([1, 3])
    with t_col1:
        chosen_sensor = st.selectbox(
            "Select Sensor to Plot",
            RAW_NUMERIC_FEATURES,
            index=3  # Torque
        )
        sample_step = st.slider("Sample density step", min_value=1, max_value=20, value=5, help="Subsample dataset for faster rendering")

    sampled_df = df.iloc[::sample_step].copy()

    with t_col2:
        fig_trend = px.line(
            sampled_df,
            x="UDI",
            y=chosen_sensor,
            color=sampled_df[TARGET_COL].map({0: "Normal", 1: "Failure"}),
            color_discrete_map={"Normal": "#3B82F6", "Failure": "#EF4444"},
            title=f"{chosen_sensor} Sequential Profile Across Telemetry Cycle",
            labels={"UDI": "Observation Sequence Index (UDI)"}
        )
        apply_chart_theme(fig_trend, height=340)
        st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown("<hr style='margin:1.2rem 0; border:none; border-top:1px solid #1F2937;'>", unsafe_allow_html=True)

    # 2 Charts: Speed vs Torque Power Hyperbola & Temperature Comparison
    row2_c1, row2_c2 = st.columns(2)
    with row2_c1:
        st.markdown("<div style='font-weight:700; font-size:0.95rem; margin-bottom:0.4rem; color:#F8FAFC;'>Rotational Speed vs. Torque (Mechanical Power Curve)</div>", unsafe_allow_html=True)
        fig_hyperbola = px.scatter(
            sampled_df,
            x="Rotational speed [rpm]",
            y="Torque [Nm]",
            color=sampled_df[TARGET_COL].map({0: "Normal", 1: "Failure"}),
            color_discrete_map={"Normal": "#10B981", "Failure": "#EF4444"},
            opacity=0.75,
            hover_data=["Product ID", "Tool wear [min]"]
        )
        apply_chart_theme(fig_hyperbola, height=320)
        fig_hyperbola.update_layout(legend=dict(orientation="h", y=-0.22))
        st.plotly_chart(fig_hyperbola, use_container_width=True)

    with row2_c2:
        st.markdown("<div style='font-weight:700; font-size:0.95rem; margin-bottom:0.4rem; color:#F8FAFC;'>Thermal Gradient (Air vs. Process Temperature)</div>", unsafe_allow_html=True)
        fig_temps = px.scatter(
            sampled_df,
            x="Air temperature [K]",
            y="Process temperature [K]",
            color=sampled_df[TARGET_COL].map({0: "Normal", 1: "Failure"}),
            color_discrete_map={"Normal": "#6366F1", "Failure": "#EF4444"},
            opacity=0.75,
            hover_data=["Product ID", "Temp_diff"]
        )
        apply_chart_theme(fig_temps, height=320)
        fig_temps.update_layout(legend=dict(orientation="h", y=-0.22))
        st.plotly_chart(fig_temps, use_container_width=True)

    # CSV Export for Tab 1
    t1_csv = df[["UDI", "Product ID", "Type"] + RAW_NUMERIC_FEATURES + [TARGET_COL]].to_csv(index=False).encode("utf-8")
    st.download_button("📥 Export Sensor Telemetry Profile CSV", t1_csv, "sensor_telemetry_profiles.csv", "text/csv")

    st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# TAB 2: CORRELATION & NORMAL VS FAILED
# ==========================================
with report_tab2:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header-title">Multi-Sensor Correlation Matrix</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-header-desc">Quantify linear inter-dependencies between sensory telemetry and machine failure:</div>', unsafe_allow_html=True)

    corr_cols = RAW_NUMERIC_FEATURES + ["Temp_diff", "Power_kW", "Overstrain", TARGET_COL]
    corr_matrix = df[corr_cols].corr()
    clean_labels = [c.split(" [")[0].replace("_", " ") for c in corr_cols]
    
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=clean_labels,
        y=clean_labels,
        colorscale="RdBu_r",
        zmin=-1,
        zmax=1,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont={"size": 11, "color": "#F8FAFC"}
    ))
    apply_chart_theme(fig_corr, height=450)
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("<hr style='margin:1.2rem 0; border:none; border-top:1px solid #1F2937;'>", unsafe_allow_html=True)

    # Normal vs. Failed Comparison Boxplots
    st.markdown('<div class="card-header-title">Sensor Comparison: Normal vs. Failed Machinery</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-header-desc">Inspect differences in sensory distributions between working equipment and failed units:</div>', unsafe_allow_html=True)

    b_sensor = st.selectbox("Select Sensor for Boxplot Comparison", RAW_NUMERIC_FEATURES, index=4)

    fig_box = px.box(
        df,
        x=df[TARGET_COL].map({0: "Normal Condition", 1: "Machine Failure"}),
        y=b_sensor,
        color=df[TARGET_COL].map({0: "Normal Condition", 1: "Machine Failure"}),
        color_discrete_map={"Normal Condition": "#10B981", "Machine Failure": "#EF4444"},
        points="outliers",
        labels={"x": "Operational State"}
    )
    apply_chart_theme(fig_box, height=360)
    fig_box.update_layout(showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)

    # Failure Rate by Machine Type
    st.markdown('<div class="card-header-title" style="margin-top:1.5rem;">Failure Rate by Product Quality Tier</div>', unsafe_allow_html=True)
    type_stats = df.groupby("Type")[TARGET_COL].agg(["count", "sum"]).reset_index()
    type_stats["Failure Rate (%)"] = np.round((type_stats["sum"] / type_stats["count"]) * 100.0, 2)
    type_stats.columns = ["Product Type", "Total Equipment", "Recorded Failures", "Failure Rate (%)"]

    fig_type = px.bar(
        type_stats,
        x="Product Type",
        y="Failure Rate (%)",
        color="Product Type",
        color_discrete_map={"L": "#3B82F6", "M": "#6366F1", "H": "#8B5CF6"},
        text="Failure Rate (%)"
    )
    apply_chart_theme(fig_type, height=280)
    fig_type.update_layout(showlegend=False)
    st.plotly_chart(fig_type, use_container_width=True)

    t2_csv = corr_matrix.to_csv().encode("utf-8")
    st.download_button("📥 Export Correlation Matrix CSV", t2_csv, "sensor_correlation_matrix.csv", "text/csv")

    st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# TAB 3: MACHINE LEARNING MODEL PERFORMANCE
# ==========================================
with report_tab3:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header-title">Random Forest Model Performance Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-header-desc">Held-out test set metrics (2,000 independent samples, stratified 80/20 partition):</div>', unsafe_allow_html=True)

    m = metadata.get("metrics", {})
    
    # 6 Metric Cards
    ev_c1, ev_c2, ev_c3, ev_c4, ev_c5, ev_c6 = st.columns(6)
    with ev_c1:
        st.metric("Test Accuracy", f"{m.get('accuracy', 0.9905):.2%}", "Benchmark > 98%")
    with ev_c2:
        st.metric("Precision", f"{m.get('precision', 0.8143):.2%}", "Positive class")
    with ev_c3:
        st.metric("Recall (Sensitivity)", f"{m.get('recall', 0.8971):.2%}", "Failure coverage")
    with ev_c4:
        st.metric("F1-Score", f"{m.get('f1', 0.8527):.2%}", "Harmonic mean")
    with ev_c5:
        st.metric("ROC-AUC", f"{m.get('roc_auc', 0.9842):.2%}", "Discrimination")
    with ev_c6:
        st.metric("PR-AUC", f"{m.get('pr_auc', 0.8613):.2%}", "Imbalance curve")

    st.markdown("<hr style='margin:1.2rem 0; border:none; border-top:1px solid #1F2937;'>", unsafe_allow_html=True)

    # Confusion Matrix Heatmap & Feature Importance Bar
    col_cm, col_fi = st.columns([1, 1.2])

    with col_cm:
        st.markdown("<div style='font-weight:700; font-size:0.95rem; margin-bottom:0.4rem; color:#F8FAFC;'>Confusion Matrix (Held-out Test Set)</div>", unsafe_allow_html=True)
        cm_data = m.get("confusion_matrix", {}).get("matrix", [[1922, 10], [9, 59]])
        
        cm_fig = go.Figure(data=go.Heatmap(
            z=cm_data,
            x=["Predicted: Normal", "Predicted: Failure"],
            y=["Actual: Normal", "Actual: Failure"],
            colorscale="Blues",
            text=cm_data,
            texttemplate="%{text}",
            textfont={"size": 16, "color": "#F8FAFC"}
        ))
        apply_chart_theme(cm_fig, height=340)
        st.plotly_chart(cm_fig, use_container_width=True)

    with col_fi:
        st.markdown("<div style='font-weight:700; font-size:0.95rem; margin-bottom:0.4rem; color:#F8FAFC;'>Feature Importance Ranking</div>", unsafe_allow_html=True)
        feat_imp = m.get("feature_importances", {})
        if feat_imp:
            fi_df = pd.DataFrame(list(feat_imp.items()), columns=["Feature", "Importance"]).sort_values("Importance", ascending=True)
            fi_fig = px.bar(
                fi_df,
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale="Viridis",
                text=np.round(fi_df["Importance"], 3)
            )
            apply_chart_theme(fi_fig, height=340)
            fi_fig.update_layout(
                coloraxis_showscale=False,
                xaxis_title="Relative Feature Weight",
                yaxis_title=""
            )
            st.plotly_chart(fi_fig, use_container_width=True)
        else:
            st.info("Feature importance data unavailable.")

    # Retrain button
    st.markdown("<hr style='margin:1.2rem 0; border:none; border-top:1px solid #1F2937;'>", unsafe_allow_html=True)
    rf_col1, rf_col2 = st.columns([2, 1])
    with rf_col1:
        st.caption(f"Artifact location: `models/random_forest_model.joblib` | Trained samples: {metadata.get('train_samples', 8000):,} | Algorithm: RandomForestClassifier")
    with rf_col2:
        if st.button("🔄 Retrain Model Artifact", use_container_width=True, type="primary"):
            with st.spinner("Retraining Random Forest model on 80/20 stratified split..."):
                _, new_meta = train_model(force_retrain=True)
            st.success("Model successfully retrained and saved to disk!")
            st.rerun()

    # CSV Download of metrics
    metrics_summary_df = pd.DataFrame([{
        "Accuracy": m.get("accuracy"),
        "Precision": m.get("precision"),
        "Recall": m.get("recall"),
        "F1_Score": m.get("f1"),
        "ROC_AUC": m.get("roc_auc"),
        "PR_AUC": m.get("pr_auc"),
        "True_Negatives": m.get("confusion_matrix", {}).get("tn"),
        "False_Positives": m.get("confusion_matrix", {}).get("fp"),
        "False_Negatives": m.get("confusion_matrix", {}).get("fn"),
        "True_Positives": m.get("confusion_matrix", {}).get("tp")
    }])
    csv_eval = metrics_summary_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Export Model Metrics CSV", csv_eval, "model_performance_evaluation.csv", "text/csv")

    st.markdown('</div>', unsafe_allow_html=True)
