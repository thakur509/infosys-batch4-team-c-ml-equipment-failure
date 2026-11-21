import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from src.styles import apply_styles, apply_chart_theme, render_hero, render_sidebar_status
from src.data_loader import load_data, TARGET_COL, RAW_NUMERIC_FEATURES
from src.db_manager import (
    insert_alert, insert_maintenance,
    get_all_registered_equipment, insert_equipment, update_equipment, delete_equipment
)
from src.predict import evaluate_sensor_diagnostics

st.set_page_config(
    page_title="Equipment Monitoring | Predictive Maintenance",
    page_icon="⚙️",
    layout="wide"
)
apply_styles()
render_sidebar_status()

# Page Hero
render_hero(
    title="Real-Time Equipment Telemetry & Asset Management",
    subtitle="Inspect fleet-wide machinery health, track machine trajectory dynamics (Stable, Deteriorating, Improving), drill down into sensor stresses, and manage registered physical assets with full CRUD capability.",
    tag="Asset Registry & Telemetry Feed"
)

# Load dataset
df = load_data()

# Precompute machine status, failure risk, and trajectory
@st.cache_data(show_spinner=False)
def get_monitored_fleet(data: pd.DataFrame) -> pd.DataFrame:
    fleet = data.copy()
    
    # Calculate operational risk and status
    is_failed = fleet[TARGET_COL] == 1
    is_at_risk = ((fleet["Tool wear [min]"] > 160) | (fleet["Torque [Nm]"] > 60) | (fleet["Temp_diff"] < 8.6)) & (~is_failed)
    
    statuses = np.where(is_failed, "Critical", np.where(is_at_risk, "Warning", "Healthy"))
    fleet["Health_Status"] = statuses
    
    # Risk percentage estimate
    risk_scores = np.where(is_failed, 95.0, np.where(is_at_risk, 48.0, 5.0))
    risk_scores = np.clip(risk_scores + (fleet["Tool wear [min]"] / 300.0) * 15.0, 2.0, 99.0)
    fleet["Failure_Risk"] = np.round(risk_scores, 1)
    fleet["Health_Score"] = np.round(100.0 - fleet["Failure_Risk"], 1)
    
    # Trajectory classification matching reference platform
    trajectories = np.where(
        (fleet["Tool wear [min]"] > 180) | (fleet["Torque [Nm]"] > 62) | is_failed,
        "Deteriorating",
        np.where(fleet["Tool wear [min]"] < 35, "Improving", "Stable")
    )
    fleet["Trajectory"] = trajectories
    
    return fleet

fleet_df = get_monitored_fleet(df)

# Tabs
tab_telemetry, tab_trajectories, tab_registry = st.tabs([
    "📊 Fleet Telemetry & Inspection",
    "📈 Trajectory Tracking & History",
    "⚙️ Registered Asset Registry (CRUD)"
])

# ==========================================
# TAB 1: FLEET TELEMETRY & DRILLDOWN
# ==========================================
with tab_telemetry:
    # Search and Filters Bar
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    c_search, c_status, c_type, c_traj, c_exp = st.columns([1.5, 1.1, 1.0, 1.1, 1.0])

    with c_search:
        search_query = st.text_input("🔍 Search Machine", placeholder="e.g. M14860 or L47181", key="search_query_input")

    with c_status:
        selected_status = st.selectbox("Status Filter", ["All Statuses", "Healthy", "Warning", "Critical"], key="status_filter_sel")

    with c_type:
        selected_type = st.selectbox("Product Type", ["All Types", "L (Low)", "M (Medium)", "H (High)"], key="type_filter_sel")

    with c_traj:
        selected_traj = st.selectbox("Trajectory", ["All Trajectories", "Stable", "Deteriorating", "Improving"], key="traj_filter_sel")

    # Filter dataframe
    filtered_fleet = fleet_df.copy()
    if search_query:
        filtered_fleet = filtered_fleet[
            filtered_fleet["Product ID"].astype(str).str.contains(search_query.strip(), case=False, na=False) |
            filtered_fleet["UDI"].astype(str).str.contains(search_query.strip(), na=False)
        ]

    if selected_status != "All Statuses":
        filtered_fleet = filtered_fleet[filtered_fleet["Health_Status"] == selected_status]

    if selected_type != "All Types":
        filtered_fleet = filtered_fleet[filtered_fleet["Type"] == selected_type[0]]

    if selected_traj != "All Trajectories":
        filtered_fleet = filtered_fleet[filtered_fleet["Trajectory"] == selected_traj]

    with c_exp:
        st.markdown("<div style='padding-top:1.75rem;'></div>", unsafe_allow_html=True)
        export_csv = filtered_fleet.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export CSV",
            data=export_csv,
            file_name="monitored_equipment.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # Grid Layout: Table on left / Machine Detailed View on right
    table_col, detail_col = st.columns([1.45, 1.15])

    with table_col:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="card-header-title">Equipment Registry ({len(filtered_fleet):,} Machines)</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-header-desc">Continuous Health Score (0–100) and live trajectory classification:</div>', unsafe_allow_html=True)
        
        cols_to_show = [
            "Product ID", "Type", "Air temperature [K]", "Process temperature [K]",
            "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]", "Health_Score", "Health_Status", "Trajectory"
        ]
        
        display_table = filtered_fleet[cols_to_show].copy()
        display_table["Rotational speed [rpm]"] = display_table["Rotational speed [rpm]"].astype(int)
        display_table["Tool wear [min]"] = display_table["Tool wear [min]"].astype(int)
        display_table.columns = [
            "Equipment ID", "Type", "Air Temp (K)", "Proc Temp (K)",
            "Speed (RPM)", "Torque (Nm)", "Wear (min)", "Health Score", "Status", "Trajectory"
        ]
        
        st.dataframe(
            display_table.head(150),
            use_container_width=True,
            hide_index=True,
            height=540
        )
        st.caption(f"Showing up to 150 matching rows out of {len(filtered_fleet):,}. Use search above to locate specific machines.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Machine Detailed View
    with detail_col:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header-title">Machine Telemetry Drilldown</div>', unsafe_allow_html=True)
        
        available_ids = filtered_fleet["Product ID"].astype(str).tolist()
        if available_ids:
            selected_machine_id = st.selectbox(
                "Select Equipment ID for Inspection",
                available_ids[:100],
                index=0,
                key="drilldown_select"
            )
            
            machine_row = fleet_df[fleet_df["Product ID"] == selected_machine_id].iloc[0]
            
            m_status = machine_row["Health_Status"]
            m_traj = machine_row["Trajectory"]
            badge_cls = "badge-critical" if m_status == "Critical" else ("badge-warning" if m_status == "Warning" else "badge-healthy")
            traj_cls = "badge-critical" if m_traj == "Deteriorating" else ("badge-healthy" if m_traj == "Improving" else "badge-neutral")
            
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem; padding-bottom:0.75rem; border-bottom:1px solid #1F2937;">
                <div>
                    <span style="font-size:1.3rem; font-weight:800; color:#F8FAFC;">{machine_row['Product ID']}</span>
                    <span style="color:#94A3B8; font-size:0.85rem; margin-left:8px;">(Type {machine_row['Type']})</span>
                </div>
                <div>
                    <span class="badge {badge_cls}">{m_status.upper()}</span>
                    <span class="badge {traj_cls}" style="margin-left:4px;">{m_traj.upper()}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Sensor values grid
            s_c1, s_c2 = st.columns(2)
            with s_c1:
                st.metric("Air Temperature", f"{machine_row['Air temperature [K]']:.1f} K", f"{machine_row['Air temperature [K]']-273.15:.1f} °C")
                st.metric("Rotational Speed", f"{int(machine_row['Rotational speed [rpm]']):,} RPM")
                st.metric("Tool Wear", f"{int(machine_row['Tool wear [min]'])} min")
            with s_c2:
                st.metric("Process Temperature", f"{machine_row['Process temperature [K]']:.1f} K", f"{machine_row['Process temperature [K]']-273.15:.1f} °C")
                st.metric("Spindle Torque", f"{machine_row['Torque [Nm]']:.1f} Nm")
                st.metric("Health Score", f"{machine_row['Health_Score']}/100")
                
            # Physical stress metrics
            p_temp_diff = machine_row["Temp_diff"]
            p_power = machine_row["Power_kW"]
            p_strain = machine_row["Overstrain"]
            
            st.markdown("<hr style='margin:1rem 0; border:none; border-top:1px solid #1F2937;'>", unsafe_allow_html=True)
            st.markdown("<div style='font-size:0.85rem; font-weight:700; color:#94A3B8; margin-bottom:0.5rem;'>PHYSICAL STRESS INDICES</div>", unsafe_allow_html=True)
            
            d_c1, d_c2, d_c3 = st.columns(3)
            with d_c1:
                st.markdown(f"<div style='font-size:0.75rem; color:#64748B;'>Thermal Delta</div><div style='font-weight:700; color:#F8FAFC;'>{p_temp_diff:.1f} K</div>", unsafe_allow_html=True)
            with d_c2:
                st.markdown(f"<div style='font-size:0.75rem; color:#64748B;'>Motor Power</div><div style='font-weight:700; color:#F8FAFC;'>{p_power:.2f} kW</div>", unsafe_allow_html=True)
            with d_c3:
                st.markdown(f"<div style='font-size:0.75rem; color:#64748B;'>Overstrain</div><div style='font-weight:700; color:#F8FAFC;'>{p_strain:,.0f}</div>", unsafe_allow_html=True)

            # Radar Comparison vs Fleet Normal Mean
            fleet_means = df[RAW_NUMERIC_FEATURES].mean()
            curr_vals = [
                machine_row["Air temperature [K]"] / fleet_means["Air temperature [K]"],
                machine_row["Process temperature [K]"] / fleet_means["Process temperature [K]"],
                machine_row["Rotational speed [rpm]"] / fleet_means["Rotational speed [rpm]"],
                machine_row["Torque [Nm]"] / fleet_means["Torque [Nm]"],
                machine_row["Tool wear [min]"] / max(fleet_means["Tool wear [min]"], 1)
            ]
            
            categories = ["Air Temp", "Proc Temp", "Speed", "Torque", "Tool Wear"]
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=[1.0, 1.0, 1.0, 1.0, 1.0],
                theta=categories,
                fill='toself',
                name='Fleet Baseline (1.0)',
                line=dict(color='#64748B', dash='dot')
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=curr_vals,
                theta=categories,
                fill='toself',
                name=f'{selected_machine_id}',
                line=dict(color='#60A5FA', width=2)
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, max(2.0, max(curr_vals) * 1.1)], gridcolor="#1F2937", linecolor="#374151"),
                    angularaxis=dict(gridcolor="#1F2937", linecolor="#374151", tickfont=dict(color="#CBD5E1")),
                    bgcolor="rgba(17,24,39,0.5)"
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, sans-serif", color="#CBD5E1"),
                showlegend=True,
                height=260,
                margin=dict(t=20, b=20, l=40, r=40),
                legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5, bgcolor="rgba(17,24,39,0.7)", bordercolor="#1F2937")
            )
            st.plotly_chart(fig_radar, use_container_width=True)

            # Quick Actions
            act_c1, act_c2 = st.columns(2)
            with act_c1:
                if st.button("🚨 Raise System Alert", use_container_width=True, key="raise_alert_btn"):
                    new_a_id = insert_alert(
                        equipment_id=selected_machine_id,
                        severity="Critical" if m_status == "Critical" else "Warning",
                        condition=f"Manual operator alert for {selected_machine_id}",
                        product_type=machine_row["Type"],
                        notes=f"Raised via monitoring view. Wear: {machine_row['Tool wear [min]']}min, Torque: {machine_row['Torque [Nm]']}Nm"
                    )
                    st.success(f"Alert #{new_a_id} logged to Equipment Alerts registry!")
            with act_c2:
                if st.button("🛠️ Log Work Order", use_container_width=True, key="log_maint_btn", type="primary"):
                    new_m_id = insert_maintenance(
                        equipment_id=selected_machine_id,
                        maintenance_type="Preventive Inspection" if m_status == "Healthy" else "Corrective Repair",
                        description=f"Maintenance logged for {selected_machine_id}. Current wear: {machine_row['Tool wear [min]']} min.",
                        priority="High" if m_status in ["Warning", "Critical"] else "Low",
                        assigned_technician="Alex Kumar",
                        scheduled_date=datetime.now().strftime("%Y-%m-%d"),
                        status="Pending"
                    )
                    st.success(f"Work order #{new_m_id} created in Maintenance module!")
        else:
            st.info("No machines match the selected filter criteria.")

        st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# TAB 2: TRAJECTORY TRACKING & HISTORY
# ==========================================
with tab_trajectories:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header-title">Fleet Health Trajectory Dynamics (Reference /monitoring/history)</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-header-desc">Tracks machines categorized by health trajectory over operating cycles: <strong>Stable</strong> (nominal envelope), <strong>Deteriorating</strong> (rapid tool wear or elevated torque), and <strong>Improving</strong> (post-maintenance or fresh tooling).</div>', unsafe_allow_html=True)

    traj_counts = fleet_df["Trajectory"].value_counts()
    tot_machines = len(fleet_df)

    # 4 Metric Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Monitored Fleet", f"{tot_machines:,}")
    with m2:
        stable_cnt = traj_counts.get("Stable", 0)
        st.metric("Stable Trajectory", f"{stable_cnt:,}", f"{stable_cnt / tot_machines:.1%}")
    with m3:
        det_cnt = traj_counts.get("Deteriorating", 0)
        st.metric("Deteriorating Trajectory", f"{det_cnt:,}", f"-{det_cnt / tot_machines:.1%}", delta_color="inverse")
    with m4:
        imp_cnt = traj_counts.get("Improving", 0)
        st.metric("Improving Trajectory", f"{imp_cnt:,}", f"+{imp_cnt / tot_machines:.1%}")

    # Charts: Donut + Health Score Histogram
    ch_col1, ch_col2 = st.columns([1, 1.4])
    with ch_col1:
        fig_donut = px.pie(
            values=traj_counts.values,
            names=traj_counts.index,
            hole=0.6,
            color=traj_counts.index,
            color_discrete_map={"Stable": "#10B981", "Deteriorating": "#EF4444", "Improving": "#3B82F6"},
            title="Trajectory Distribution Ratio"
        )
        apply_chart_theme(fig_donut, height=320)
        fig_donut.update_layout(margin=dict(t=35, b=10, l=10, r=10))
        st.plotly_chart(fig_donut, use_container_width=True)

    with ch_col2:
        fig_hist = px.histogram(
            fleet_df,
            x="Health_Score",
            color="Trajectory",
            nbins=30,
            color_discrete_map={"Stable": "#10B981", "Deteriorating": "#EF4444", "Improving": "#3B82F6"},
            title="Continuous Fleet Health Score Distribution (0–100)"
        )
        apply_chart_theme(fig_hist, height=320)
        fig_hist.update_layout(xaxis_title="Health Score (0-100)", yaxis_title="Number of Machines")
        st.plotly_chart(fig_hist, use_container_width=True)

    # Deteriorating Assets Priority Watchlist
    st.markdown('<div class="card-header-title" style="margin-top:1.5rem;">⚠️ Deteriorating Equipment Priority Watchlist</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-header-desc">Machines displaying deteriorating operational trajectories requiring immediate inspection scheduling:</div>', unsafe_allow_html=True)

    deteriorating_df = fleet_df[fleet_df["Trajectory"] == "Deteriorating"].sort_values("Failure_Risk", ascending=False)
    
    det_display = deteriorating_df[[
        "Product ID", "Type", "Air temperature [K]", "Process temperature [K]",
        "Torque [Nm]", "Tool wear [min]", "Failure_Risk", "Health_Score", "Health_Status"
    ]].head(50).copy()
    det_display.columns = [
        "Equipment ID", "Type", "Air Temp (K)", "Proc Temp (K)",
        "Torque (Nm)", "Wear (min)", "Failure Risk (%)", "Health Score", "Status"
    ]
    st.dataframe(det_display, use_container_width=True, hide_index=True, height=350)
    st.caption("Displaying top 50 highest-risk deteriorating machinery.")

    st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# TAB 3: REGISTERED ASSET MANAGEMENT (CRUD)
# ==========================================
with tab_registry:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header-title">Physical Asset Registry & Lifecycle Management (Reference /equipment)</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-header-desc">Manage plant machinery inventory, register new machines with baseline telemetry, edit operating profiles, and remove decommissioned units. Persisted directly to <code>database/maintenance.db</code>.</div>', unsafe_allow_html=True)

    # Load custom equipment from SQLite
    custom_assets = get_all_registered_equipment()

    # Asset summary metrics
    a_m1, a_m2, a_m3, a_m4 = st.columns(4)
    with a_m1:
        st.metric("Total Registered Assets", len(custom_assets))
    with a_m2:
        healthy_cnt = len(custom_assets[custom_assets["status"] == "Healthy"]) if not custom_assets.empty else 0
        st.metric("Operational Assets", healthy_cnt)
    with a_m3:
        warn_cnt = len(custom_assets[custom_assets["status"] == "Warning"]) if not custom_assets.empty else 0
        st.metric("Warning / At-Risk", warn_cnt)
    with a_m4:
        crit_cnt = len(custom_assets[custom_assets["status"] == "Critical"]) if not custom_assets.empty else 0
        st.metric("Critical / Down", crit_cnt)

    # Expandable Form: Register New Equipment
    with st.expander("➕ Register New Equipment Asset", expanded=False):
        with st.form("register_asset_form", clear_on_submit=True):
            r_c1, r_c2, r_c3 = st.columns(3)
            with r_c1:
                new_id = st.text_input("Equipment / Asset ID", placeholder="e.g. CNC-GAMMA-09")
                new_type = st.selectbox("Product Type", ["L", "M", "H"], index=1)
                new_loc = st.selectbox("Plant Location", [
                    "Plant A (Main)", "Plant B (Assembly)", "Plant C (High-Load)", "Plant D (Cold Storage)"
                ])
            with r_c2:
                new_air = st.number_input("Air Temp [K]", 290.0, 310.0, 300.0, 0.1)
                new_proc = st.number_input("Process Temp [K]", 300.0, 320.0, 310.0, 0.1)
                new_speed = st.number_input("Speed [RPM]", 1000.0, 3000.0, 1500.0, 10.0)
            with r_c3:
                new_torque = st.number_input("Torque [Nm]", 10.0, 90.0, 42.0, 0.5)
                new_wear = st.number_input("Tool Wear [min]", 0.0, 300.0, 25.0, 1.0)
                new_traj = st.selectbox("Initial Trajectory", ["Stable", "Improving", "Deteriorating"])

            reg_submit = st.form_submit_button("Submit & Register Asset", use_container_width=True, type="primary")
            if reg_submit:
                if not new_id.strip():
                    st.error("Please provide a valid Equipment ID.")
                else:
                    try:
                        # Infer status
                        inferred_status = "Critical" if new_wear > 200 or new_torque > 65 else ("Warning" if new_wear > 150 or new_torque > 55 else "Healthy")
                        insert_equipment(
                            equipment_id=new_id.strip(),
                            product_type=new_type,
                            location=new_loc,
                            air_temp=new_air,
                            process_temp=new_proc,
                            rotational_speed=new_speed,
                            torque=new_torque,
                            tool_wear=new_wear,
                            trajectory=new_traj,
                            status=inferred_status
                        )
                        st.success(f"Equipment asset '{new_id.strip()}' registered successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to register asset: {e}")

    # Display Assets Table
    if not custom_assets.empty:
        st.markdown('<div class="card-header-title" style="margin-top:1.25rem;">Active Registered Assets</div>', unsafe_allow_html=True)
        
        display_assets = custom_assets[[
            "equipment_id", "product_type", "location", "air_temp", "process_temp",
            "rotational_speed", "torque", "tool_wear", "trajectory", "status", "created_at"
        ]].copy()
        display_assets.columns = [
            "Asset ID", "Type", "Plant Location", "Air Temp (K)", "Proc Temp (K)",
            "Speed (RPM)", "Torque (Nm)", "Wear (min)", "Trajectory", "Status", "Registered At"
        ]
        st.dataframe(display_assets, use_container_width=True, hide_index=True)

        # Asset Actions: Edit / Delete
        st.markdown("<hr style='margin:1.25rem 0; border:none; border-top:1px solid #1F2937;'>", unsafe_allow_html=True)
        act_col1, act_col2 = st.columns(2)

        with act_col1:
            st.markdown("<strong>📝 Update Asset Parameters</strong>", unsafe_allow_html=True)
            selected_edit_id = st.selectbox("Select Asset to Modify", custom_assets["equipment_id"].tolist(), key="edit_asset_sel")
            curr_item = custom_assets[custom_assets["equipment_id"] == selected_edit_id].iloc[0]

            with st.form("edit_asset_form"):
                e_loc = st.selectbox("Location", [
                    "Plant A (Main)", "Plant B (Assembly)", "Plant C (High-Load)", "Plant D (Cold Storage)"
                ], index=["Plant A (Main)", "Plant B (Assembly)", "Plant C (High-Load)", "Plant D (Cold Storage)"].index(curr_item["location"]) if curr_item["location"] in ["Plant A (Main)", "Plant B (Assembly)", "Plant C (High-Load)", "Plant D (Cold Storage)"] else 0)
                e_torque = st.number_input("Torque [Nm]", 10.0, 90.0, float(curr_item["torque"]), 0.5)
                e_wear = st.number_input("Tool Wear [min]", 0.0, 300.0, float(curr_item["tool_wear"]), 1.0)
                e_status = st.selectbox("Status", ["Healthy", "Warning", "Critical"], index=["Healthy", "Warning", "Critical"].index(curr_item["status"]))
                e_traj = st.selectbox("Trajectory", ["Stable", "Improving", "Deteriorating"], index=["Stable", "Improving", "Deteriorating"].index(curr_item["trajectory"]))
                
                if st.form_submit_button("Save Changes", use_container_width=True, type="primary"):
                    update_equipment(
                        equipment_id=selected_edit_id,
                        location=e_loc,
                        torque=e_torque,
                        tool_wear=e_wear,
                        status=e_status,
                        trajectory=e_traj
                    )
                    st.success(f"Asset '{selected_edit_id}' updated successfully!")
                    st.rerun()

        with act_col2:
            st.markdown("<strong>🗑️ Decommission Asset</strong>", unsafe_allow_html=True)
            selected_del_id = st.selectbox("Select Asset to Decommission", custom_assets["equipment_id"].tolist(), key="del_asset_sel")
            st.caption(f"Permanently removes asset '{selected_del_id}' from the registered database inventory.")
            
            if st.button("⚠️ Decommission & Delete Asset", type="secondary", use_container_width=True):
                delete_equipment(selected_del_id)
                st.warning(f"Asset '{selected_del_id}' has been decommissioned and removed.")
                st.rerun()
    else:
        st.info("No registered custom assets found. Use the form above to register your first equipment asset.")

    st.markdown('</div>', unsafe_allow_html=True)
