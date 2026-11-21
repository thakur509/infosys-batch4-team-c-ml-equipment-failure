import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datetime import datetime
import streamlit as st
import pandas as pd

from src.styles import apply_styles, render_hero, render_sidebar_status
from src.db_manager import (
    get_alerts,
    get_alert_counts,
    update_alert_status,
    insert_alert
)

st.set_page_config(
    page_title="Equipment Alerts | Predictive Maintenance",
    page_icon="🚨",
    layout="wide"
)
apply_styles()
render_sidebar_status()

# Page Hero
render_hero(
    title="Equipment Alerts & Fault Management",
    subtitle="Review critical equipment warnings, investigate anomalous operating conditions, and manage alert lifecycle (New → Acknowledged → Resolved). All status transitions are immediately persisted in the SQLite audit ledger.",
    tag="Active Alert Dispatch & Remediation"
)

# Fetch live counts from SQLite
alert_counts = get_alert_counts()

# Summary KPI Cards
k_c1, k_c2, k_c3, k_c4 = st.columns(4)
with k_c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">New Alerts</div>
        <div class="metric-value" style="color:#F87171;">{alert_counts.get('New', 0):,}</div>
        <div class="metric-subtext">Requires immediate operator review</div>
    </div>
    """, unsafe_allow_html=True)

with k_c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Acknowledged Alerts</div>
        <div class="metric-value" style="color:#FBBF24;">{alert_counts.get('Acknowledged', 0):,}</div>
        <div class="metric-subtext">Investigation active in progress</div>
    </div>
    """, unsafe_allow_html=True)

with k_c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Resolved Alerts</div>
        <div class="metric-value" style="color:#34D399;">{alert_counts.get('Resolved', 0):,}</div>
        <div class="metric-subtext">Remediation confirmed cleared</div>
    </div>
    """, unsafe_allow_html=True)

with k_c4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total System Alerts</div>
        <div class="metric-value" style="color:#60A5FA;">{alert_counts.get('Total', 0):,}</div>
        <div class="metric-subtext">Logged in audit registry</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

# Filters and Search Bar
st.markdown('<div class="content-card">', unsafe_allow_html=True)
f1, f2, f3, f4 = st.columns([1.5, 1.2, 1.2, 1])

with f1:
    status_tab = st.radio(
        "Alert Queue Filter",
        ["All Alerts", "New", "Acknowledged", "Resolved"],
        horizontal=True
    )

with f2:
    severity_choice = st.selectbox("Severity Filter", ["All Severities", "Critical", "Warning", "Informational"])

with f3:
    search_alert = st.text_input("Search Alert / Machine ID", placeholder="e.g. M14860 or Tool Wear")

# Query Alerts from SQLite
filter_param = None if status_tab == "All Alerts" else status_tab
alerts_df = get_alerts(status_filter=filter_param, limit=1000)

if severity_choice != "All Severities":
    alerts_df = alerts_df[alerts_df["severity"] == severity_choice]

if search_alert:
    alerts_df = alerts_df[
        alerts_df["equipment_id"].astype(str).str.contains(search_alert.strip(), case=False, na=False) |
        alerts_df["condition"].astype(str).str.contains(search_alert.strip(), case=False, na=False)
    ]

with f4:
    st.markdown("<div style='padding-top:1.75rem;'></div>", unsafe_allow_html=True)
    csv_alert_data = alerts_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"📥 Export Alerts CSV ({len(alerts_df)})",
        data=csv_alert_data,
        file_name=f"equipment_alerts_{status_tab.lower().replace(' ', '_')}.csv",
        mime="text/csv",
        use_container_width=True
    )

st.markdown('</div>', unsafe_allow_html=True)

# Main Grid: Alert Table & Interactive Action Panel
table_area, action_area = st.columns([1.6, 1.1])

with table_area:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="card-header-title">Alert Roster ({len(alerts_df):,} Events)</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-header-desc">Active alert conditions recorded from sensor telemetry thresholds:</div>', unsafe_allow_html=True)

    if not alerts_df.empty:
        table_view = alerts_df[["id", "equipment_id", "severity", "condition", "timestamp", "status"]].copy()
        table_view.columns = ["Alert ID", "Equipment ID", "Severity", "Alert Condition", "Logged Timestamp", "Status"]
        st.dataframe(
            table_view,
            use_container_width=True,
            hide_index=True,
            height=460
        )
    else:
        st.info("No alerts match the selected filter criteria.")

    st.markdown('</div>', unsafe_allow_html=True)

with action_area:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header-title">Alert Operations & Lifecycle</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-header-desc">Select an active alert to transition status or inspect diagnostic notes:</div>', unsafe_allow_html=True)

    if not alerts_df.empty:
        alert_options = {
            f"#{row['id']} - {row['equipment_id']} ({row['condition']}) [{row['status']}]": row["id"]
            for _, row in alerts_df.head(60).iterrows()
        }
        
        selected_option = st.selectbox("Select Alert to Manage", list(alert_options.keys()))
        selected_id = alert_options[selected_option]
        selected_row = alerts_df[alerts_df["id"] == selected_id].iloc[0]

        # Display Selected Alert Meta with Dark Theme
        curr_status = selected_row["status"]
        curr_sev = selected_row["severity"]
        
        sev_color = "#F87171" if curr_sev == "Critical" else ("#FBBF24" if curr_sev == "Warning" else "#60A5FA")
        st.markdown(f"""
        <div style="background:rgba(30, 41, 59, 0.7); border:1px solid #334155; border-radius:10px; padding:1rem; margin-bottom:1rem;">
            <div style="font-size:0.78rem; color:#94A3B8;">ALERT #{selected_row['id']}</div>
            <div style="font-size:1.1rem; font-weight:700; color:#F8FAFC; margin:0.2rem 0;">{selected_row['equipment_id']} &bull; {selected_row['condition']}</div>
            <div style="font-size:0.85rem; color:#CBD5E1;">
                Severity: <strong style="color:{sev_color};">{curr_sev}</strong> &bull; Status: <strong>{curr_status}</strong>
            </div>
            <div style="font-size:0.78rem; color:#64748B; margin-top:0.3rem;">Logged: {selected_row['timestamp']}</div>
        </div>
        """, unsafe_allow_html=True)

        if selected_row.get("notes"):
            st.markdown(f"<div style='font-size:0.82rem; color:#94A3B8; margin-bottom:0.75rem;'><strong>Diagnostic Notes:</strong> {selected_row['notes']}</div>", unsafe_allow_html=True)

        # Action Buttons
        act_col1, act_col2 = st.columns(2)
        with act_col1:
            if curr_status == "New":
                if st.button("👁️ Acknowledge", use_container_width=True, type="primary"):
                    update_alert_status(selected_id, "Acknowledged")
                    st.success(f"Alert #{selected_id} marked Acknowledged!")
                    st.rerun()
            else:
                st.button("👁️ Acknowledged", disabled=True, use_container_width=True)

        with act_col2:
            if curr_status != "Resolved":
                if st.button("✅ Resolve Alert", use_container_width=True):
                    update_alert_status(selected_id, "Resolved", notes="Resolved by operator action.")
                    st.success(f"Alert #{selected_id} marked Resolved!")
                    st.rerun()
            else:
                st.button("✅ Resolved", disabled=True, use_container_width=True)

        # Telemetry detail expander
        with st.expander("🔍 Telemetry Details"):
            st.json(dict(selected_row))

    else:
        st.info("No alerts available in the current view.")

    st.markdown('</div>', unsafe_allow_html=True)

# Create New Alert Expander
with st.expander("➕ Log New Manual Equipment Alert"):
    with st.form("new_alert_form"):
        na_col1, na_col2 = st.columns(2)
        with na_col1:
            new_eq_id = st.text_input("Equipment ID", value="M14860")
            new_sev = st.selectbox("Severity Level", ["Critical", "Warning", "Informational"])
        with na_col2:
            new_cond = st.selectbox("Alert Condition", [
                "Abnormal Temperature Gradient",
                "High Torque Output",
                "Abnormal Rotational Speed",
                "Tool Wear Failure (TWF)",
                "Heat Dissipation Failure (HDF)",
                "Power Failure (PWF)",
                "Overstrain Failure (OSF)",
                "Mechanical Vibration Anomaly"
            ])
            new_notes = st.text_input("Operational Notes", value="Logged by shift supervisor")
            
        na_submit = st.form_submit_button("Submit Alert to System", use_container_width=True, type="primary")
        if na_submit:
            inserted_id = insert_alert(
                equipment_id=new_eq_id,
                severity=new_sev,
                condition=new_cond,
                notes=new_notes
            )
            st.success(f"New alert #{inserted_id} created and logged to database!")
            st.rerun()
