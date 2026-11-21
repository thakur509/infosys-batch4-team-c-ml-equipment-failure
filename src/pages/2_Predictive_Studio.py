import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datetime import datetime
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.styles import apply_styles, apply_chart_theme, render_hero, render_sidebar_status
from src.predict import predict_machine_failure
from src.db_manager import insert_prediction, get_predictions, insert_maintenance
from src.data_loader import load_data

st.set_page_config(
    page_title="Failure Prediction | Predictive Maintenance",
    page_icon="🔮",
    layout="wide"
)
apply_styles()
render_sidebar_status()

# Page Hero
render_hero(
    title="Equipment Failure Prediction Studio",
    subtitle="Evaluate real-time machine telemetry using our calibrated Random Forest classifier. Enter operational sensor readings in Celsius or Kelvin to assess failure likelihood, physical stress indices, and prescriptive maintenance actions.",
    tag="Machine Learning Inference Engine"
)

# Load dataset for registered equipment prefill
fleet_raw = load_data()

# Registered equipment prefill selector
st.markdown('<div class="content-card">', unsafe_allow_html=True)
pref_c1, pref_c2 = st.columns([2.2, 1.2])

with pref_c1:
    st.markdown('<div class="card-header-title">Equipment Telemetry Source & Presets</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-header-desc">Select an active machine to auto-fill its readings, or choose a diagnostic scenario preset:</div>', unsafe_allow_html=True)

with pref_c2:
    temp_unit = st.radio("Temperature Display Unit", ["Celsius [°C]", "Kelvin [K]"], horizontal=True, key="temp_unit_toggle")
    is_celsius = temp_unit.startswith("Celsius")

reg_options = ["Manual Custom Telemetry"] + [
    f"{row['Product ID']} · Type {row['Type']} (Speed: {int(row['Rotational speed [rpm]']):,} RPM, Torque: {row['Torque [Nm]']:.1f} Nm)"
    for _, row in fleet_raw.head(40).iterrows()
]
selected_reg = st.selectbox("Choose Registered Fleet Unit to Populate", reg_options, key="select_reg_unit")

# Diagnostic Presets
preset_cols = st.columns(5)
preset_selected = None

if preset_cols[0].button("🟢 Normal Operation", use_container_width=True):
    preset_selected = {"air_k": 298.5, "proc_k": 308.8, "speed": 1520, "torque": 40.0, "wear": 45, "eq_id": "PRESET-NORMAL", "ptype": "M"}
if preset_cols[1].button("🔴 Heat Dissipation (HDF)", use_container_width=True):
    preset_selected = {"air_k": 303.0, "proc_k": 309.5, "speed": 1310, "torque": 52.0, "wear": 90, "eq_id": "PRESET-HDF", "ptype": "L"}
if preset_cols[2].button("🔴 Power Failure (PWF)", use_container_width=True):
    preset_selected = {"air_k": 298.9, "proc_k": 309.1, "speed": 2860, "torque": 4.5, "wear": 140, "eq_id": "PRESET-PWF", "ptype": "L"}
if preset_cols[3].button("🔴 Overstrain (OSF)", use_container_width=True):
    preset_selected = {"air_k": 298.4, "proc_k": 308.2, "speed": 1340, "torque": 62.0, "wear": 215, "eq_id": "PRESET-OSF", "ptype": "L"}
if preset_cols[4].button("🔴 Tool Wear Failure", use_container_width=True):
    preset_selected = {"air_k": 298.8, "proc_k": 308.9, "speed": 1450, "torque": 42.0, "wear": 225, "eq_id": "PRESET-TWF", "ptype": "M"}

# If registered machine chosen from dropdown, populate
if selected_reg != "Manual Custom Telemetry" and not preset_selected:
    chosen_pid = selected_reg.split(" · ")[0]
    matched_row = fleet_raw[fleet_raw["Product ID"] == chosen_pid]
    if not matched_row.empty:
        r = matched_row.iloc[0]
        preset_selected = {
            "air_k": float(r["Air temperature [K]"]),
            "proc_k": float(r["Process temperature [K]"]),
            "speed": int(r["Rotational speed [rpm]"]),
            "torque": float(r["Torque [Nm]"]),
            "wear": int(r["Tool wear [min]"]),
            "eq_id": str(r["Product ID"]),
            "ptype": str(r["Type"])
        }

# Maintain persistent form telemetry in Kelvin internally
if "stored_telemetry" not in st.session_state:
    st.session_state.stored_telemetry = {
        "air_k": 298.5,
        "proc_k": 308.8,
        "speed": 1520,
        "torque": 40.0,
        "wear": 45,
        "eq_id": "EQ-47201",
        "ptype": "M"
    }

if preset_selected:
    st.session_state.stored_telemetry.update(preset_selected)
    st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# Main Form and Output Layout
form_col, result_col = st.columns([1.15, 1.25])

current_telemetry = st.session_state.stored_telemetry

with form_col:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header-title">Sensor Reading Input Form</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="card-header-desc">Enter operating readings ({("Celsius [°C]" if is_celsius else "Kelvin [K]")}):</div>', unsafe_allow_html=True)
    
    with st.form("prediction_form"):
        f_c1, f_c2 = st.columns(2)
        with f_c1:
            eq_id = st.text_input(
                "Equipment ID",
                value=current_telemetry.get("eq_id", "EQ-47201"),
                help="Unique machine serial or inventory tag"
            )
        with f_c2:
            ptype = st.selectbox(
                "Product Type",
                ["M (Medium Quality)", "L (Low Quality)", "H (High Quality)"],
                index=0 if current_telemetry.get("ptype", "M") == "M" else (1 if current_telemetry.get("ptype", "M") == "L" else 2)
            )

        # Dynamic Dual Unit Inputs
        if is_celsius:
            # Display and accept Celsius: nominal ~20°C to 35°C (K - 273.15)
            air_val_c = round(current_telemetry["air_k"] - 273.15, 1)
            proc_val_c = round(current_telemetry["proc_k"] - 273.15, 1)
            
            air_input = st.number_input(
                "Air Temperature [°C]",
                min_value=10.0,
                max_value=45.0,
                value=float(air_val_c),
                step=0.1,
                format="%.1f",
                help="Ambient air temperature in Celsius (nominal ~25°C - 30°C)"
            )
            process_input = st.number_input(
                "Process Temperature [°C]",
                min_value=20.0,
                max_value=55.0,
                value=float(proc_val_c),
                step=0.1,
                format="%.1f",
                help="Internal process temperature in Celsius (nominal ~35°C - 42°C)"
            )
        else:
            air_input = st.number_input(
                "Air Temperature [K]",
                min_value=285.0,
                max_value=315.0,
                value=float(current_telemetry["air_k"]),
                step=0.1,
                format="%.1f",
                help="Ambient air temperature in Kelvin (nominal ~298 K - 304 K)"
            )
            process_input = st.number_input(
                "Process Temperature [K]",
                min_value=295.0,
                max_value=325.0,
                value=float(current_telemetry["proc_k"]),
                step=0.1,
                format="%.1f",
                help="Internal machine process temperature in Kelvin (nominal ~308 K - 315 K)"
            )

        rot_speed = st.number_input(
            "Rotational Speed [RPM]",
            min_value=1000,
            max_value=3000,
            value=int(current_telemetry["speed"]),
            step=10,
            help="Spindle rotational velocity in revolutions per minute (nominal ~1300-1800 RPM)"
        )

        torque = st.number_input(
            "Torque [Nm]",
            min_value=1.0,
            max_value=90.0,
            value=float(current_telemetry["torque"]),
            step=0.5,
            format="%.1f",
            help="Spindle torque output in Newton-meters (nominal ~30-55 Nm)"
        )

        tool_wear = st.number_input(
            "Tool Wear [min]",
            min_value=0,
            max_value=300,
            value=int(current_telemetry["wear"]),
            step=5,
            help="Cumulative cutting tool usage duration in minutes (critical threshold >200 min)"
        )

        submitted = st.form_submit_button("⚡ Predict Failure Likelihood", use_container_width=True, type="primary")
    st.markdown('</div>', unsafe_allow_html=True)

# If submitted, convert to Kelvin and run inference
if submitted:
    # Convert input to Kelvin for the ML model
    air_k = float(air_input + 273.15 if is_celsius else air_input)
    proc_k = float(process_input + 273.15 if is_celsius else process_input)
    
    st.session_state.stored_telemetry = {
        "air_k": air_k,
        "proc_k": proc_k,
        "speed": rot_speed,
        "torque": torque,
        "wear": tool_wear,
        "eq_id": eq_id,
        "ptype": ptype[0]
    }
    
    # Run Inference
    res = predict_machine_failure(
        air_temp=air_k,
        process_temp=proc_k,
        rotational_speed=rot_speed,
        torque=torque,
        tool_wear=tool_wear
    )
    
    # Store to SQLite
    insert_prediction(
        air_temp=air_k,
        process_temp=proc_k,
        rotational_speed=rot_speed,
        torque=torque,
        tool_wear=tool_wear,
        predicted_failure=res["predicted_failure"],
        failure_probability=res["failure_probability"],
        risk_level=res["risk_level"],
        recommendation=res["recommendation"],
        equipment_id=eq_id,
        product_type=ptype[0]
    )
    
    st.session_state.last_prediction = res
    st.session_state.last_pred_inputs = {
        "eq_id": eq_id,
        "ptype": ptype[0],
        "air_k": air_k,
        "proc_k": proc_k,
        "air_c": round(air_k - 273.15, 1),
        "proc_c": round(proc_k - 273.15, 1),
        "speed": rot_speed,
        "torque": torque,
        "wear": tool_wear
    }

# Display results
with result_col:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header-title">Diagnostic Prediction Result</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-header-desc">Inference generated by calibrated Random Forest & Multiclass Failure Mode Classifier:</div>', unsafe_allow_html=True)

    if "last_prediction" in st.session_state:
        res = st.session_state.last_prediction
        p_inputs = st.session_state.last_pred_inputs
        
        is_failure = res["predicted_failure"] == 1
        prob_pct = res["probability_pct"]
        risk = res["risk_level"]
        
        # Status Box (Obsidian & Slate theme)
        if is_failure or risk in ["High", "Critical"]:
            status_bg = "rgba(239, 68, 68, 0.12)"
            status_border = "rgba(239, 68, 68, 0.4)"
            status_text = "#F87171"
            status_icon = "⚠️"
            status_title = "FAILURE RISK DETECTED"
        elif risk == "Moderate":
            status_bg = "rgba(245, 158, 11, 0.12)"
            status_border = "rgba(245, 158, 11, 0.4)"
            status_text = "#FBBF24"
            status_icon = "⚡"
            status_title = "MODERATE WEAR / ELEVATED RISK"
        else:
            status_bg = "rgba(16, 185, 129, 0.12)"
            status_border = "rgba(16, 185, 129, 0.4)"
            status_text = "#34D399"
            status_icon = "✅"
            status_title = "NOMINAL OPERATING CONDITION"

        st.markdown(f"""
        <div style="background:{status_bg}; border:1.5px solid {status_border}; border-radius:12px; padding:1.2rem; text-align:center; margin-bottom:1.2rem;">
            <div style="font-size:1.8rem; margin-bottom:0.2rem;">{status_icon}</div>
            <div style="font-size:1.35rem; font-weight:800; color:{status_text}; letter-spacing:0.02em;">{status_title}</div>
            <div style="font-size:0.92rem; color:#CBD5E1; margin-top:0.3rem;">
                Target: <strong>{p_inputs['eq_id']}</strong> (Type {p_inputs['ptype']}) &bull; Risk Level: <strong style="color:{status_text};">{risk}</strong>
            </div>
            <div style="margin-top:0.6rem; display:flex; justify-content:center; gap:10px;">
                <span class="badge" style="background:rgba(99, 102, 241, 0.2); color:#A5B4FC; border:1px solid rgba(129, 140, 248, 0.35); font-size:0.8rem; padding:4px 10px;">
                    Root Cause: {res.get('specific_mode', 'Normal')}
                </span>
                <span class="badge" style="background:rgba(16, 185, 129, 0.2); color:#34D399; border:1px solid rgba(16, 185, 129, 0.35); font-size:0.8rem; padding:4px 10px;">
                    Health Score: {res.get('health_score', 95.0)} / 100
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Plotly Gauge Chart for failure probability
        gauge_color = "#10B981" if prob_pct < 20 else ("#3B82F6" if prob_pct < 45 else ("#F59E0B" if prob_pct < 75 else "#EF4444"))
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_pct,
            domain={'x': [0, 1], 'y': [0, 1]},
            number={'suffix': "%", 'font': {'size': 36, 'family': 'JetBrains Mono', 'color': '#F8FAFC'}},
            title={'text': "Predicted Failure Likelihood", 'font': {'size': 13, 'color': '#94A3B8'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
                'bar': {'color': gauge_color, 'thickness': 0.26},
                'bgcolor': "rgba(30, 41, 59, 0.5)",
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 20], 'color': 'rgba(16, 185, 129, 0.15)'},
                    {'range': [20, 45], 'color': 'rgba(59, 130, 246, 0.15)'},
                    {'range': [45, 75], 'color': 'rgba(245, 158, 11, 0.15)'},
                    {'range': [75, 100], 'color': 'rgba(239, 68, 68, 0.15)'}
                ],
                'threshold': {
                    'line': {'color': "#EF4444", 'width': 3},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        apply_chart_theme(fig_gauge, height=230)
        fig_gauge.update_layout(margin=dict(t=25, b=10, l=25, r=25))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Derived Physics & Thermal Metrics
        m_c1, m_c2, m_c3, m_c4 = st.columns(4)
        with m_c1:
            st.metric("Health Score", f"{res.get('health_score', 95.0)}", "0–100 Continuous")
        with m_c2:
            st.metric("Thermal Delta (ΔT)", f"{res['temp_diff']:.1f} K", "Limit ≥ 8.6 K")
        with m_c3:
            st.metric("Spindle Power", f"{res['power_kw']:.2f} kW", "Safe: 3.5–9.0 kW")
        with m_c4:
            st.metric("Overstrain Index", f"{res['overstrain']:,.0f}", "Limit < 11,000")

        # Dual Unit Summary Note
        st.caption(f"Operating Temperatures: Air = {p_inputs['air_k']:.1f} K ({p_inputs['air_c']:.1f} °C) &bull; Process = {p_inputs['proc_k']:.1f} K ({p_inputs['proc_c']:.1f} °C)")

        # Prescriptive Recommendation
        st.markdown(f"""
        <div style="background:rgba(30, 41, 59, 0.7); border:1px solid #334155; border-radius:10px; padding:1rem; margin-top:0.75rem;">
            <strong style="font-size:0.9rem; color:#F8FAFC;">💡 Prescriptive Engineering Action:</strong>
            <p style="margin-top:0.35rem; margin-bottom:0; font-size:0.86rem; color:#CBD5E1; line-height:1.5;">{res['recommendation']}</p>
        </div>
        """, unsafe_allow_html=True)

        # Action: Quick Schedule Maintenance if at risk
        st.markdown("<div style='margin-top:0.75rem;'></div>", unsafe_allow_html=True)
        if is_failure or risk in ["Moderate", "High", "Critical"]:
            if st.button("🛠️ Log Work Order for this Equipment", use_container_width=True, type="primary"):
                m_id = insert_maintenance(
                    equipment_id=p_inputs["eq_id"],
                    maintenance_type="Corrective Repair" if is_failure else "Preventive Inspection",
                    description=f"Auto-logged from Prediction Studio. Prob: {prob_pct:.1f}%. Mode: {res.get('specific_mode', 'Fault')}. Rec: {res['recommendation'][:90]}...",
                    priority="Critical" if risk == "Critical" else ("High" if risk == "High" else "Medium"),
                    assigned_technician="Alex Kumar",
                    scheduled_date=datetime.now().strftime("%Y-%m-%d"),
                    status="Pending",
                    notes=f"Inputs: Air={p_inputs['air_k']}K, Proc={p_inputs['proc_k']}K, Speed={p_inputs['speed']}RPM, Torque={p_inputs['torque']}Nm, Wear={p_inputs['wear']}min"
                )
                st.success(f"Work Order #{m_id} created successfully in the Maintenance registry!")
    else:
        st.info("👈 Enter sensor readings on the left and click **Predict Failure Likelihood** to see live model diagnostics.")
        
    st.markdown('</div>', unsafe_allow_html=True)

# Historical Predictions Log
st.markdown('<div class="content-card">', unsafe_allow_html=True)
hist_header_col1, hist_header_col2 = st.columns([3, 1])
with hist_header_col1:
    st.markdown('<div class="card-header-title">Prediction History Audit Log</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-header-desc">Persistent record of past model inferences stored in SQLite database</div>', unsafe_allow_html=True)
with hist_header_col2:
    preds_df = get_predictions(limit=500)
    if not preds_df.empty:
        csv_preds = preds_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export Predictions CSV",
            data=csv_preds,
            file_name="failure_prediction_history.csv",
            mime="text/csv",
            use_container_width=True
        )

if not preds_df.empty:
    display_history = preds_df.copy()
    display_history["failure_probability"] = (display_history["failure_probability"] * 100).round(1).astype(str) + "%"
    display_history["predicted_failure"] = display_history["predicted_failure"].map({0: "Normal", 1: "Failure"})
    
    cols_rename = {
        "id": "ID",
        "timestamp": "Timestamp",
        "equipment_id": "Equipment ID",
        "product_type": "Type",
        "air_temp": "Air Temp [K]",
        "process_temp": "Process Temp [K]",
        "rotational_speed": "Speed [RPM]",
        "torque": "Torque [Nm]",
        "tool_wear": "Tool Wear [min]",
        "predicted_failure": "Status",
        "failure_probability": "Probability",
        "risk_level": "Risk Level"
    }
    display_cols = [c for c in cols_rename.keys() if c in display_history.columns]
    history_view = display_history[display_cols].rename(columns=cols_rename)
    st.dataframe(history_view, use_container_width=True, hide_index=True)
else:
    st.caption("No prediction history recorded yet.")

st.markdown('</div>', unsafe_allow_html=True)
