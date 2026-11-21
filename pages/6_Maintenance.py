import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datetime import datetime, date
import streamlit as st
import pandas as pd

from src.styles import apply_styles, render_hero, render_sidebar_status
from src.db_manager import (
    get_maintenance_records,
    get_maintenance_counts,
    insert_maintenance,
    update_maintenance,
    delete_maintenance
)

st.set_page_config(
    page_title="Maintenance | Predictive Maintenance",
    page_icon="🛠️",
    layout="wide"
)
apply_styles()
render_sidebar_status()

# Page Hero
render_hero(
    title="Predictive Maintenance & Work Order Management",
    subtitle="Coordinate machine work orders, assign certified technicians, manage scheduled preventative servicing, and track corrective maintenance lifecycles. All records persist in the SQLite enterprise database.",
    tag="Work Order Dispatch & Asset Servicing"
)

# Live KPI Counts from SQLite
m_counts = get_maintenance_counts()

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Pending Work Orders</div>
        <div class="metric-value" style="color:#FBBF24;">{m_counts.get('Pending', 0):,}</div>
        <div class="metric-subtext">Awaiting servicing dispatch</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">In Progress</div>
        <div class="metric-value" style="color:#60A5FA;">{m_counts.get('In Progress', 0):,}</div>
        <div class="metric-subtext">Active shop-floor technician</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Completed Servicing</div>
        <div class="metric-value" style="color:#34D399;">{m_counts.get('Completed', 0):,}</div>
        <div class="metric-subtext">Verified operational clearance</div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Critical Priority Tasks</div>
        <div class="metric-value" style="color:#F87171;">{m_counts.get('Critical', 0):,}</div>
        <div class="metric-subtext">High breakdown urgency</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

# Search, Filters & Export
st.markdown('<div class="content-card">', unsafe_allow_html=True)
c_filter1, c_filter2, c_search, c_btn = st.columns([1.5, 1.2, 1.5, 1])

with c_filter1:
    status_select = st.radio(
        "Status Filter",
        ["All", "Pending", "In Progress", "Completed"],
        horizontal=True
    )

with c_filter2:
    priority_select = st.selectbox("Priority Filter", ["All Priorities", "Critical", "High", "Medium", "Low"])

with c_search:
    search_term = st.text_input("Search Machine ID / Technician", placeholder="e.g. M14860 or Alex")

# Query records
maint_df = get_maintenance_records(status_filter=None if status_select == "All" else status_select)

if priority_select != "All Priorities":
    maint_df = maint_df[maint_df["priority"] == priority_select]

if search_term:
    maint_df = maint_df[
        maint_df["equipment_id"].astype(str).str.contains(search_term.strip(), case=False, na=False) |
        maint_df["assigned_technician"].astype(str).str.contains(search_term.strip(), case=False, na=False) |
        maint_df["description"].astype(str).str.contains(search_term.strip(), case=False, na=False)
    ]

with c_btn:
    st.markdown("<div style='padding-top:1.75rem;'></div>", unsafe_allow_html=True)
    csv_maint = maint_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"📥 Export ({len(maint_df)}) CSV",
        data=csv_maint,
        file_name="maintenance_records.csv",
        mime="text/csv",
        use_container_width=True
    )

st.markdown('</div>', unsafe_allow_html=True)

# Records Table
st.markdown('<div class="content-card">', unsafe_allow_html=True)
st.markdown(f'<div class="card-header-title">Work Order Registry ({len(maint_df):,} Records)</div>', unsafe_allow_html=True)
st.markdown('<div class="card-header-desc">Maintenance history and scheduled work orders:</div>', unsafe_allow_html=True)

if not maint_df.empty:
    cols_display = [
        "id", "equipment_id", "maintenance_type", "priority",
        "assigned_technician", "scheduled_date", "status", "description"
    ]
    view_table = maint_df[cols_display].rename(columns={
        "id": "WO ID",
        "equipment_id": "Equipment ID",
        "maintenance_type": "Maintenance Type",
        "priority": "Priority",
        "assigned_technician": "Technician",
        "scheduled_date": "Scheduled Date",
        "status": "Status",
        "description": "Task Description"
    })
    st.dataframe(view_table, use_container_width=True, hide_index=True, height=360)
else:
    st.info("No maintenance work orders match the current filters.")

st.markdown('</div>', unsafe_allow_html=True)

# Two-column action tabs: Add New Record vs. Update / Delete Existing
action_tab1, action_tab2 = st.tabs(["➕ Add New Work Order", "✏️ Edit / Complete / Delete Record"])

with action_tab1:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header-title">Create Maintenance Work Order</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-header-desc">Assign a new task to scheduled service queues:</div>', unsafe_allow_html=True)
    
    with st.form("create_maint_form"):
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            in_eq_id = st.text_input("Target Equipment ID", value="M14860")
            in_type = st.selectbox("Maintenance Classification", [
                "Preventive Maintenance",
                "Corrective Repair",
                "Component Overhaul",
                "Emergency Inspection",
                "Tool Replacement"
            ])
            in_priority = st.selectbox("Priority Level", ["Medium", "Low", "High", "Critical"])
            
        with col_m2:
            in_tech = st.selectbox("Assign Certified Technician", [
                "Alex Kumar (Lead Mechanist)",
                "Sarah Jenkins (Vibration Specialist)",
                "Carlos Mendoza (Electrical Tech)",
                "Priya Sharma (Thermal Systems)",
                "David Chen (Hydraulics)"
            ])
            in_date = st.date_input("Scheduled Date", value=date.today())
            in_status = st.selectbox("Initial Status", ["Pending", "In Progress", "Completed"])

        in_desc = st.text_area("Task Description & Scope", placeholder="Describe the maintenance instructions, symptoms, or components to inspect...")
        in_notes = st.text_input("Operational Notes", placeholder="e.g. Requires shut down of Line 4 for 2 hours")

        add_submit = st.form_submit_button("Submit Work Order", use_container_width=True, type="primary")
        if add_submit:
            if not in_desc.strip():
                st.error("Please provide a task description.")
            else:
                new_id = insert_maintenance(
                    equipment_id=in_eq_id.strip(),
                    maintenance_type=in_type,
                    description=in_desc.strip(),
                    priority=in_priority,
                    assigned_technician=in_tech.split(" (")[0],
                    scheduled_date=str(in_date),
                    status=in_status,
                    notes=in_notes.strip()
                )
                st.success(f"Work Order #{new_id} created successfully!")
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

with action_tab2:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header-title">Update or Delete Existing Record</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-header-desc">Select an active work order to modify technician, status, or remove:</div>', unsafe_allow_html=True)
    
    all_records = get_maintenance_records()
    if not all_records.empty:
        wo_map = {
            f"WO #{row['id']} - {row['equipment_id']} ({row['maintenance_type']}) [{row['status']}]": row["id"]
            for _, row in all_records.iterrows()
        }
        selected_wo_label = st.selectbox("Select Work Order", list(wo_map.keys()))
        selected_wo_id = wo_map[selected_wo_label]
        target_row = all_records[all_records["id"] == selected_wo_id].iloc[0]

        with st.form("edit_maint_form"):
            e_c1, e_c2 = st.columns(2)
            with e_c1:
                e_type = st.selectbox("Maintenance Type", [
                    "Preventive Maintenance", "Corrective Repair", "Component Overhaul",
                    "Emergency Inspection", "Tool Replacement"
                ], index=["Preventive Maintenance", "Corrective Repair", "Component Overhaul", "Emergency Inspection", "Tool Replacement"].index(target_row["maintenance_type"]) if target_row["maintenance_type"] in ["Preventive Maintenance", "Corrective Repair", "Component Overhaul", "Emergency Inspection", "Tool Replacement"] else 0)
                e_priority = st.selectbox("Priority", ["Critical", "High", "Medium", "Low"], index=["Critical", "High", "Medium", "Low"].index(target_row["priority"]) if target_row["priority"] in ["Critical", "High", "Medium", "Low"] else 2)
                e_status = st.selectbox("Status", ["Pending", "In Progress", "Completed"], index=["Pending", "In Progress", "Completed"].index(target_row["status"]) if target_row["status"] in ["Pending", "In Progress", "Completed"] else 0)

            with e_c2:
                e_tech = st.text_input("Technician", value=target_row["assigned_technician"])
                e_date = st.text_input("Scheduled Date (YYYY-MM-DD)", value=target_row["scheduled_date"])
                e_notes = st.text_input("Notes", value=target_row["notes"] or "")

            e_desc = st.text_area("Description", value=target_row["description"])

            save_btn = st.form_submit_button("💾 Save Changes", use_container_width=True, type="primary")

        if save_btn:
            update_maintenance(
                record_id=selected_wo_id,
                maintenance_type=e_type,
                description=e_desc,
                priority=e_priority,
                assigned_technician=e_tech,
                scheduled_date=e_date,
                status=e_status,
                notes=e_notes
            )
            st.success(f"Work Order #{selected_wo_id} successfully updated!")
            st.rerun()

        # Mark completed shortcut
        if target_row["status"] != "Completed":
            if st.button("Mark as Completed", use_container_width=True):
                update_maintenance(
                    record_id=selected_wo_id,
                    maintenance_type=target_row["maintenance_type"],
                    description=target_row["description"],
                    priority=target_row["priority"],
                    assigned_technician=target_row["assigned_technician"],
                    scheduled_date=target_row["scheduled_date"],
                    status="Completed",
                    notes=f"Completed on {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                )
                st.success(f"Work order #{selected_wo_id} marked as Completed!")
                st.rerun()

        # Delete with confirmation
        st.markdown("<hr style='margin:1.2rem 0; border:none; border-top:1px solid #1F2937;'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.85rem; font-weight:700; color:#F87171;'>Destructive Action:</div>", unsafe_allow_html=True)
        del_confirm = st.checkbox(f"Confirm deletion of Work Order #{selected_wo_id}")
        if del_confirm:
            if st.button("🗑️ Permanently Delete Record", type="secondary", use_container_width=True):
                delete_maintenance(selected_wo_id)
                st.warning(f"Work Order #{selected_wo_id} deleted.")
                st.rerun()

    else:
        st.info("No work orders available.")

    st.markdown('</div>', unsafe_allow_html=True)
