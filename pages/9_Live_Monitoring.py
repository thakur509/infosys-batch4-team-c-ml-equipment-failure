import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datetime import datetime, timedelta
import random
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.styles import apply_styles, apply_chart_theme, render_hero, render_sidebar_status
from src.predict import predict_machine_failure
from src.db_manager import insert_prediction

st.set_page_config(
    page_title="Machine Health Monitoring | Predictive Maintenance",
    page_icon="⚙️",
    layout="wide"
)
apply_styles()
render_sidebar_status()

# Page Title Header matching user reference image
st.markdown("""
<div style="display:flex; align-items:center; gap:12px; margin-bottom:1.5rem;">
    <span style="font-size:2rem;">⚙️</span>
    <div>
        <h1 style="font-size:1.85rem; font-weight:800; color:#F8FAFC; margin:0; letter-spacing:-0.02em;">Machine Health Monitoring Dashboard</h1>
        <p style="font-size:0.9rem; color:#94A3B8; margin:0.2rem 0 0 0;">Real-time continuous telemetry streaming, automated anomaly detection, and sensor simulation engine.</p>
    </div>
</div>
""", unsafe_allow_html=True)


# Initialize default streaming session data if not already created
def generate_baseline_stream():
    now = datetime.now()
    base_time = now - timedelta(hours=48)
    records = []
    
    # 48 historical ticks
    for i in range(48):
        t = base_time + timedelta(hours=i)
        
        # Nominal baseline with natural industrial variance
        temp_c = round(random.gauss(72.0, 3.5), 2)
        vibration = round(max(0.2, random.gauss(0.95, 0.18)), 2)
        pressure = round(random.gauss(52.0, 3.8), 2)
        speed = int(random.gauss(1520, 45))
        torque = round(pressure * 0.85, 1)  # pressure correlated with spindle torque
        wear = int(min(240, 20 + i * 3.5 + random.randint(0, 5)))
        
        # Inject few realistic anomaly points as shown in user reference
        is_anomaly = False
        if i in [14, 26, 39]:
            if i == 14:
                temp_c = 58.2  # Sudden thermal drop
                is_anomaly = True
            elif i == 26:
                temp_c = 84.6  # Thermal spike
                is_anomaly = True
            elif i == 39:
                temp_c = 56.4  # Cooling fault
                is_anomaly = True
        
        # Calculate real model risk
        proc_k = temp_c + 273.15
        air_k = proc_k - 9.5
        pred_res = predict_machine_failure(
            air_temp=air_k,
            process_temp=proc_k,
            rotational_speed=speed,
            torque=torque,
            tool_wear=wear
        )
        fail_risk = round(pred_res["probability_pct"], 1)
        
        records.append({
            "Timestamp": t.strftime("%Y-%m-%d %H:%M"),
            "Temperature (°C)": temp_c,
            "Vibration (mm/s)": vibration,
            "Pressure (PSI)": pressure,
            "Rotational Speed (RPM)": speed,
            "Torque (Nm)": torque,
            "Tool Wear (min)": wear,
            "Failure Risk (%)": fail_risk,
            "Health Score": pred_res["health_score"],
            "Is_Anomaly": is_anomaly
        })
    return records


if "live_monitoring_stream" not in st.session_state:
    st.session_state.live_monitoring_stream = generate_baseline_stream()

stream_data = st.session_state.live_monitoring_stream
df_stream = pd.DataFrame(stream_data)
latest_point = stream_data[-1]

# ==========================================
# 1. TOP METRIC CARDS (Matching Reference)
# ==========================================
m_col1, m_col2, m_col3, m_col4 = st.columns(4)

curr_temp = latest_point["Temperature (°C)"]
curr_vib = latest_point["Vibration (mm/s)"]
curr_press = latest_point["Pressure (PSI)"]
curr_risk = latest_point["Failure Risk (%)"]

temp_status = "Critical" if curr_temp > 85 or curr_temp < 60 else ("Warning" if curr_temp > 80 else "Normal")
temp_color = "#34D399" if temp_status == "Normal" else ("#FBBF24" if temp_status == "Warning" else "#F87171")

vib_status = "Critical" if curr_vib > 2.0 else ("Warning" if curr_vib > 1.5 else "Normal")
vib_color = "#34D399" if vib_status == "Normal" else ("#FBBF24" if vib_status == "Warning" else "#F87171")

press_status = "Critical" if curr_press > 70 or curr_press < 35 else "Normal"
press_color = "#34D399" if press_status == "Normal" else "#F87171"

risk_status = "Safe" if curr_risk < 25.0 else ("Moderate" if curr_risk < 50.0 else "Critical")
risk_color = "#34D399" if risk_status == "Safe" else ("#FBBF24" if risk_status == "Moderate" else "#F87171")

with m_col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Temperature (°C)</div>
        <div class="metric-value">{curr_temp:.2f}</div>
        <div class="metric-subtext" style="color:{temp_color}; font-weight:600;">● {temp_status}</div>
    </div>
    """, unsafe_allow_html=True)

with m_col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Vibration (mm/s)</div>
        <div class="metric-value">{curr_vib:.2f}</div>
        <div class="metric-subtext" style="color:{vib_color}; font-weight:600;">● {vib_status}</div>
    </div>
    """, unsafe_allow_html=True)

with m_col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Pressure (PSI)</div>
        <div class="metric-value">{curr_press:.2f}</div>
        <div class="metric-subtext" style="color:{press_color}; font-weight:600;">● {press_status}</div>
    </div>
    """, unsafe_allow_html=True)

with m_col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Failure Risk</div>
        <div class="metric-value">{curr_risk:.1f}%</div>
        <div class="metric-subtext" style="color:{risk_color}; font-weight:600;">● {risk_status}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

# ==========================================
# 2. ALERTS & STATUS SECTION
# ==========================================
st.markdown('<div class="content-card">', unsafe_allow_html=True)
st.markdown('<div class="card-header-title">Alerts & Status</div>', unsafe_allow_html=True)

# Determine fleet health banner
if curr_risk > 75.0 or latest_point.get("Is_Anomaly", False):
    st.markdown("""
    <div style="background:rgba(239, 68, 68, 0.15); border:1px solid rgba(239, 68, 68, 0.4); border-radius:8px; padding:0.9rem 1.2rem; display:flex; align-items:center; gap:10px; color:#F87171;">
        <span style="font-size:1.2rem;">🚨</span>
        <span style="font-weight:700; font-size:0.92rem;">System Health Status: Critical Alert - Sensor threshold breached or elevated failure risk detected!</span>
    </div>
    """, unsafe_allow_html=True)
elif curr_risk > 40.0:
    st.markdown("""
    <div style="background:rgba(245, 158, 11, 0.15); border:1px solid rgba(245, 158, 11, 0.4); border-radius:8px; padding:0.9rem 1.2rem; display:flex; align-items:center; gap:10px; color:#FBBF24;">
        <span style="font-size:1.2rem;">⚠️</span>
        <span style="font-weight:700; font-size:0.92rem;">System Health Status: Warning - High mechanical stress accumulating. Schedule inspection.</span>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="background:rgba(16, 185, 129, 0.12); border:1px solid rgba(16, 185, 129, 0.35); border-radius:8px; padding:0.9rem 1.2rem; display:flex; align-items:center; gap:10px; color:#34D399;">
        <span style="font-size:1.2rem;">✅</span>
        <span style="font-weight:700; font-size:0.92rem;">System Health Status: Healthy - All operational sensors running within calibrated envelopes.</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# 3. MACHINE PERFORMANCE OVER TIME
# ==========================================
st.markdown('<div class="content-card">', unsafe_allow_html=True)
st.markdown('<div class="card-header-title">Machine Performance Over Time</div>', unsafe_allow_html=True)

metric_options = [
    "Temperature (°C)",
    "Vibration (mm/s)",
    "Pressure (PSI)",
    "Rotational Speed (RPM)",
    "Torque (Nm)",
    "Tool Wear (min)",
    "Failure Risk (%)"
]
selected_metric = st.selectbox("Select Metric to Visualize", metric_options, index=0)

# Build interactive line chart with red anomaly markers
st.markdown(f"<div style='font-size:0.88rem; font-weight:700; color:#CBD5E1; margin:0.8rem 0 0.4rem 0;'>{selected_metric.split(' (')[0]} Trend Analysis</div>", unsafe_allow_html=True)

fig_trend = go.Figure()

# Main trend line
fig_trend.add_trace(go.Scatter(
    x=df_stream["Timestamp"],
    y=df_stream[selected_metric],
    mode="lines",
    name=selected_metric,
    line=dict(color="#38BDF8", width=2.2),
    hoverinfo="x+y"
))

# Highlight anomaly markers with red dots (matching reference image)
anomalies_subset = df_stream[df_stream["Is_Anomaly"] == True]
if not anomalies_subset.empty:
    fig_trend.add_trace(go.Scatter(
        x=anomalies_subset["Timestamp"],
        y=anomalies_subset[selected_metric],
        mode="markers",
        name="Anomalies",
        marker=dict(color="#EF4444", size=9, symbol="circle", line=dict(color="#FFFFFF", width=1.5)),
        hoverinfo="x+y+name"
    ))

apply_chart_theme(fig_trend, height=350)
fig_trend.update_layout(
    xaxis=dict(title="Timestamp", tickangle=-30),
    yaxis=dict(title=selected_metric),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig_trend, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# 4. ADD CURRENT MEASUREMENT (SIMULATION)
# ==========================================
st.markdown('<div class="content-card">', unsafe_allow_html=True)
st.markdown('<div class="card-header-title">Add Current Measurement (Simulation)</div>', unsafe_allow_html=True)
st.markdown('<div class="card-header-desc">Simulate sensor telemetry injection to test real-time inference, anomaly detection, and live trend plotting:</div>', unsafe_allow_html=True)

with st.form("add_measurement_form"):
    in_col1, in_col2, in_col3 = st.columns(3)
    
    with in_col1:
        in_temp = st.number_input("Temperature [°C]", min_value=30.0, max_value=120.0, value=75.0, step=0.5, format="%.2f")
        in_speed = st.number_input("Spindle Speed [RPM]", min_value=800, max_value=3200, value=1520, step=10)
        
    with in_col2:
        in_vib = st.number_input("Vibration [mm/s]", min_value=0.1, max_value=8.0, value=1.20, step=0.05, format="%.2f")
        in_wear = st.number_input("Tool Wear [min]", min_value=0, max_value=300, value=int(latest_point["Tool Wear (min)"] + 5), step=5)
        
    with in_col3:
        in_press = st.number_input("Pressure [PSI]", min_value=10.0, max_value=120.0, value=50.0, step=0.5, format="%.2f")
        in_torque = st.number_input("Torque [Nm]", min_value=10.0, max_value=90.0, value=round(in_press * 0.85, 1), step=0.5)

    btn_c1, btn_c2, btn_c3 = st.columns([1.5, 1, 1])
    with btn_c1:
        record_btn = st.form_submit_button("Record Measurement", use_container_width=True, type="primary")

if record_btn:
    # 1. Evaluate prediction with real Random Forest model
    proc_k = in_temp + 273.15
    air_k = proc_k - 9.5
    pred_res = predict_machine_failure(
        air_temp=air_k,
        process_temp=proc_k,
        rotational_speed=in_speed,
        torque=in_torque,
        tool_wear=in_wear
    )
    fail_prob = round(pred_res["probability_pct"], 1)
    
    # Check if anomalous
    is_anom = in_temp > 85.0 or in_temp < 60.0 or in_vib > 2.2 or in_press > 75.0 or fail_prob > 50.0
    
    now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_entry = {
        "Timestamp": now_ts,
        "Temperature (°C)": round(in_temp, 2),
        "Vibration (mm/s)": round(in_vib, 2),
        "Pressure (PSI)": round(in_press, 2),
        "Rotational Speed (RPM)": int(in_speed),
        "Torque (Nm)": round(in_torque, 1),
        "Tool Wear (min)": int(in_wear),
        "Failure Risk (%)": fail_prob,
        "Health Score": pred_res["health_score"],
        "Is_Anomaly": is_anom
    }
    
    # Store to SQLite
    insert_prediction(
        air_temp=air_k,
        process_temp=proc_k,
        rotational_speed=in_speed,
        torque=in_torque,
        tool_wear=in_wear,
        predicted_failure=pred_res["predicted_failure"],
        failure_probability=pred_res["failure_probability"],
        risk_level=pred_res["risk_level"],
        recommendation=pred_res["recommendation"],
        equipment_id="SIM-LIVE-01",
        product_type="M"
    )
    
    # Append to session state and refresh
    st.session_state.live_monitoring_stream.append(new_entry)
    st.success(f"Measurement recorded at {now_ts}! Real model failure probability: {fail_prob}%")
    st.rerun()

# Quick Helper Actions (Simulate tick, Reset)
col_sim1, col_sim2 = st.columns(2)
with col_sim1:
    if st.button("🎲 Simulate Next Live Stream Tick", use_container_width=True):
        last_t = datetime.strptime(latest_point["Timestamp"], "%Y-%m-%d %H:%M" if len(latest_point["Timestamp"]) <= 16 else "%Y-%m-%d %H:%M:%S")
        next_t = (last_t + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M")
        
        sim_temp = round(random.gauss(73.5, 3.2), 2)
        sim_vib = round(max(0.3, random.gauss(1.0, 0.2)), 2)
        sim_press = round(random.gauss(53.0, 3.5), 2)
        sim_speed = int(random.gauss(1510, 40))
        sim_torque = round(sim_press * 0.85, 1)
        sim_wear = int(min(260, latest_point["Tool Wear (min)"] + random.randint(2, 6)))
        
        proc_k = sim_temp + 273.15
        air_k = proc_k - 9.5
        pred_res = predict_machine_failure(
            air_temp=air_k,
            process_temp=proc_k,
            rotational_speed=sim_speed,
            torque=sim_torque,
            tool_wear=sim_wear
        )
        is_anom = random.random() < 0.12 or sim_temp > 84 or sim_temp < 58
        
        st.session_state.live_monitoring_stream.append({
            "Timestamp": next_t,
            "Temperature (°C)": sim_temp,
            "Vibration (mm/s)": sim_vib,
            "Pressure (PSI)": sim_press,
            "Rotational Speed (RPM)": sim_speed,
            "Torque (Nm)": sim_torque,
            "Tool Wear (min)": sim_wear,
            "Failure Risk (%)": round(pred_res["probability_pct"], 1),
            "Health Score": pred_res["health_score"],
            "Is_Anomaly": is_anom
        })
        st.rerun()

with col_sim2:
    if st.button("🔄 Reset Stream to Baseline", use_container_width=True):
        st.session_state.live_monitoring_stream = generate_baseline_stream()
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)
