"""Database manager for SQLite persistence of predictions, alerts, and maintenance records."""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DB_DIR = ROOT_DIR / "data"
DB_PATH = DB_DIR / "maintenance.db"
# Fallback check if old database folder exists
if not DB_PATH.exists() and (ROOT_DIR / "database" / "maintenance.db").exists():
    DB_PATH = ROOT_DIR / "database" / "maintenance.db"


def get_connection() -> sqlite3.Connection:
    """Create and return a thread-safe connection to the SQLite database."""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize database tables and constraints if they do not exist."""
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # Predictions history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                equipment_id TEXT,
                product_type TEXT DEFAULT 'M',
                air_temp REAL NOT NULL,
                process_temp REAL NOT NULL,
                rotational_speed REAL NOT NULL,
                torque REAL NOT NULL,
                tool_wear REAL NOT NULL,
                predicted_failure INTEGER NOT NULL,
                failure_probability REAL NOT NULL,
                risk_level TEXT NOT NULL,
                recommendation TEXT
            )
        """)
        
        # Equipment alerts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                equipment_id TEXT NOT NULL,
                product_type TEXT DEFAULT 'M',
                severity TEXT NOT NULL,
                condition TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'New',
                acknowledged_at TEXT,
                resolved_at TEXT,
                notes TEXT
            )
        """)
        
        # Maintenance records table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS maintenance_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                equipment_id TEXT NOT NULL,
                maintenance_type TEXT NOT NULL,
                description TEXT NOT NULL,
                priority TEXT NOT NULL,
                assigned_technician TEXT NOT NULL,
                scheduled_date TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'Pending',
                notes TEXT,
                created_at TEXT NOT NULL
            )
        """)
        # Registered custom equipment table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS registered_equipment (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                equipment_id TEXT UNIQUE NOT NULL,
                product_type TEXT NOT NULL DEFAULT 'M',
                location TEXT DEFAULT 'Plant A (Main)',
                air_temp REAL DEFAULT 300.0,
                process_temp REAL DEFAULT 310.0,
                rotational_speed REAL DEFAULT 1500.0,
                torque REAL DEFAULT 40.0,
                tool_wear REAL DEFAULT 30.0,
                trajectory TEXT DEFAULT 'Stable',
                status TEXT DEFAULT 'Healthy',
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()


# ==========================================
# PREDICTIONS OPERATIONS
# ==========================================

def insert_prediction(
    air_temp: float,
    process_temp: float,
    rotational_speed: float,
    torque: float,
    tool_wear: float,
    predicted_failure: int,
    failure_probability: float,
    risk_level: str,
    recommendation: str,
    equipment_id: str = "MANUAL-INPUT",
    product_type: str = "M"
) -> int:
    """Insert a manual or simulated prediction record."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO predictions (
                timestamp, equipment_id, product_type, air_temp, process_temp,
                rotational_speed, torque, tool_wear, predicted_failure,
                failure_probability, risk_level, recommendation
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp, equipment_id, product_type, air_temp, process_temp,
            rotational_speed, torque, tool_wear, predicted_failure,
            failure_probability, risk_level, recommendation
        ))
        conn.commit()
        return cursor.lastrowid


def get_predictions(limit: int = 200) -> pd.DataFrame:
    """Retrieve historical predictions as a pandas DataFrame."""
    with get_connection() as conn:
        query = f"SELECT * FROM predictions ORDER BY id DESC LIMIT {limit}"
        return pd.read_sql_query(query, conn)


# ==========================================
# ALERTS OPERATIONS
# ==========================================

def insert_alert(
    equipment_id: str,
    severity: str,
    condition: str,
    product_type: str = "M",
    timestamp: Optional[str] = None,
    status: str = "New",
    notes: str = ""
) -> int:
    """Insert a new alert into the alerts table."""
    if not timestamp:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO alerts (
                equipment_id, product_type, severity, condition,
                timestamp, status, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (equipment_id, product_type, severity, condition, timestamp, status, notes))
        conn.commit()
        return cursor.lastrowid


def update_alert_status(alert_id: int, new_status: str, notes: Optional[str] = None):
    """Update alert status (New -> Acknowledged / Resolved) and set timestamps."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_connection() as conn:
        cursor = conn.cursor()
        if new_status == "Acknowledged":
            cursor.execute("""
                UPDATE alerts 
                SET status = ?, acknowledged_at = ?
                WHERE id = ?
            """, (new_status, now_str, alert_id))
        elif new_status == "Resolved":
            cursor.execute("""
                UPDATE alerts 
                SET status = ?, resolved_at = ?
                WHERE id = ?
            """, (new_status, now_str, alert_id))
        else:
            cursor.execute("""
                UPDATE alerts SET status = ? WHERE id = ?
            """, (new_status, alert_id))
        
        if notes:
            cursor.execute("UPDATE alerts SET notes = ? WHERE id = ?", (notes, alert_id))
            
        conn.commit()


def get_alerts(status_filter: Optional[str] = None, limit: int = 500) -> pd.DataFrame:
    """Retrieve alerts with optional status filtering."""
    with get_connection() as conn:
        if status_filter and status_filter != "All":
            query = f"SELECT * FROM alerts WHERE status = ? ORDER BY id DESC LIMIT {limit}"
            return pd.read_sql_query(query, conn, params=(status_filter,))
        else:
            query = f"SELECT * FROM alerts ORDER BY id DESC LIMIT {limit}"
            return pd.read_sql_query(query, conn)


def get_alert_counts() -> Dict[str, int]:
    """Return count of alerts by status."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT status, COUNT(*) FROM alerts GROUP BY status")
        rows = cursor.fetchall()
        counts = {"New": 0, "Acknowledged": 0, "Resolved": 0, "Total": 0}
        total = 0
        for status, cnt in rows:
            if status in counts:
                counts[status] = cnt
            total += cnt
        counts["Total"] = total
        return counts


# ==========================================
# MAINTENANCE RECORDS OPERATIONS
# ==========================================

def insert_maintenance(
    equipment_id: str,
    maintenance_type: str,
    description: str,
    priority: str,
    assigned_technician: str,
    scheduled_date: str,
    status: str = "Pending",
    notes: str = ""
) -> int:
    """Insert a new maintenance task."""
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO maintenance_records (
                equipment_id, maintenance_type, description, priority,
                assigned_technician, scheduled_date, status, notes, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            equipment_id, maintenance_type, description, priority,
            assigned_technician, scheduled_date, status, notes, created_at
        ))
        conn.commit()
        return cursor.lastrowid


def update_maintenance(
    record_id: int,
    maintenance_type: str,
    description: str,
    priority: str,
    assigned_technician: str,
    scheduled_date: str,
    status: str,
    notes: str
):
    """Update an existing maintenance record."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE maintenance_records
            SET maintenance_type = ?, description = ?, priority = ?,
                assigned_technician = ?, scheduled_date = ?, status = ?, notes = ?
            WHERE id = ?
        """, (
            maintenance_type, description, priority, assigned_technician,
            scheduled_date, status, notes, record_id
        ))
        conn.commit()


def delete_maintenance(record_id: int):
    """Delete a maintenance record by ID."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM maintenance_records WHERE id = ?", (record_id,))
        conn.commit()


def get_maintenance_records(status_filter: Optional[str] = None) -> pd.DataFrame:
    """Retrieve maintenance records as DataFrame."""
    with get_connection() as conn:
        if status_filter and status_filter != "All":
            query = "SELECT * FROM maintenance_records WHERE status = ? ORDER BY id DESC"
            return pd.read_sql_query(query, conn, params=(status_filter,))
        else:
            query = "SELECT * FROM maintenance_records ORDER BY id DESC"
            return pd.read_sql_query(query, conn)


def get_maintenance_counts() -> Dict[str, int]:
    """Return count of maintenance tasks by status & priority."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT status, COUNT(*) FROM maintenance_records GROUP BY status")
        status_rows = cursor.fetchall()
        cursor.execute("SELECT COUNT(*) FROM maintenance_records WHERE priority = 'Critical' AND status != 'Completed'")
        critical_cnt = cursor.fetchone()[0]
        
        counts = {
            "Pending": 0,
            "In Progress": 0,
            "Completed": 0,
            "Critical": critical_cnt,
            "Total": 0
        }
        total = 0
        for status, cnt in status_rows:
            if status in counts:
                counts[status] = cnt
            total += cnt
        counts["Total"] = total
        return counts


# ==========================================
# DATABASE SEEDING
# ==========================================

def seed_initial_data(df: pd.DataFrame):
    """Seed initial alerts and maintenance records from the actual dataset if empty."""
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM alerts")
        alert_count = cursor.fetchone()[0]
        
        if alert_count == 0 and df is not None and not df.empty:
            # Seed alerts based on actual failure modes in the dataset
            alerts_to_insert = []
            
            # 1. Real failure cases from dataset
            failures = df[df["Machine failure"] == 1]
            for _, row in failures.head(50).iterrows():
                equip_id = str(row["Product ID"])
                ptype = str(row["Type"])
                
                # Determine dominant failure mode
                if row.get("TWF", 0) == 1:
                    condition = "Tool Wear Failure (TWF)"
                    severity = "Critical"
                elif row.get("HDF", 0) == 1:
                    condition = "Heat Dissipation Failure (HDF)"
                    severity = "Critical"
                elif row.get("PWF", 0) == 1:
                    condition = "Power Failure (PWF)"
                    severity = "Critical"
                elif row.get("OSF", 0) == 1:
                    condition = "Overstrain Failure (OSF)"
                    severity = "Critical"
                elif row.get("RNF", 0) == 1:
                    condition = "Random Failure (RNF)"
                    severity = "Warning"
                else:
                    condition = "Abnormal Mechanical Strain"
                    severity = "Critical"
                
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                alerts_to_insert.append((equip_id, ptype, severity, condition, ts, "New", "Auto-generated from telemetry fault label"))
            
            # 2. Add some high torque or high speed warning alerts
            extreme_torque = df[(df["Torque [Nm]"] > 65) & (df["Machine failure"] == 0)].head(15)
            for _, row in extreme_torque.iterrows():
                equip_id = str(row["Product ID"])
                ptype = str(row["Type"])
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                alerts_to_insert.append((equip_id, ptype, "Warning", "High Torque Warning (>65 Nm)", ts, "New", "Operating near motor threshold"))
                
            cursor.executemany("""
                INSERT INTO alerts (equipment_id, product_type, severity, condition, timestamp, status, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, alerts_to_insert)
            conn.commit()

        # Seed initial maintenance records if empty
        cursor.execute("SELECT COUNT(*) FROM maintenance_records")
        maint_count = cursor.fetchone()[0]
        
        if maint_count == 0:
            sample_maint = [
                ("M14860", "Preventive Maintenance", "Quarterly spindle lubrication and bearing check", "Medium", "Alex Kumar", "2026-11-25", "Pending", "Routine PM schedule"),
                ("L47230", "Corrective Repair", "Tool wear exceeded 200 min threshold. Replace cutting bit.", "Critical", "Sarah Jenkins", "2026-11-22", "In Progress", "High vibration detected"),
                ("M15032", "Component Overhaul", "Motor power fluctuations observed during peak torque cycle", "High", "Carlos Mendoza", "2026-11-28", "Pending", "Awaiting replacement stator"),
                ("H29424", "Emergency Inspection", "Thermal dissipation check due to ambient delta breach", "Critical", "Priya Sharma", "2026-11-21", "In Progress", "Heat sensor recalibration needed"),
                ("L47348", "Preventive Maintenance", "Torque calibration and drive belt alignment", "Low", "Alex Kumar", "2026-11-30", "Pending", "Scheduled maintenance"),
                ("M14926", "Inspection", "Routine hydraulic pressure test", "Low", "David Chen", "2026-11-18", "Completed", "All pressure seals intact"),
            ]
            cursor.executemany("""
                INSERT INTO maintenance_records (
                    equipment_id, maintenance_type, description, priority,
                    assigned_technician, scheduled_date, status, notes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """, sample_maint)
            conn.commit()

        # Seed initial registered equipment if empty
        cursor.execute("SELECT COUNT(*) FROM registered_equipment")
        equip_count = cursor.fetchone()[0]
        if equip_count == 0:
            sample_equipment = [
                ("CNC-ALPHA-01", "H", "Plant A (Main)", 298.2, 308.7, 1502.0, 42.5, 35.0, "Stable", "Healthy"),
                ("CNC-BETA-02", "M", "Plant A (Main)", 301.5, 310.8, 1420.0, 58.2, 195.0, "Deteriorating", "Warning"),
                ("ROBOTIC-ARM-03", "L", "Plant B (Assembly)", 299.1, 309.2, 1610.0, 38.0, 22.0, "Improving", "Healthy"),
                ("LATHE-EXT-04", "H", "Plant C (High-Load)", 303.4, 312.1, 1315.0, 68.4, 215.0, "Deteriorating", "Critical"),
                ("MILL-5AXIS-05", "M", "Plant A (Main)", 297.8, 308.1, 1540.0, 39.1, 85.0, "Stable", "Healthy"),
                ("PRESS-HYD-06", "L", "Plant D (Cold Storage)", 295.4, 305.2, 1490.0, 44.0, 110.0, "Stable", "Healthy"),
                ("GRINDER-PREC-07", "M", "Plant B (Assembly)", 300.2, 310.0, 1720.0, 29.5, 15.0, "Improving", "Healthy"),
                ("DRILL-RAD-08", "H", "Plant C (High-Load)", 302.8, 311.9, 1380.0, 61.0, 185.0, "Deteriorating", "Warning"),
            ]
            cursor.executemany("""
                INSERT INTO registered_equipment (
                    equipment_id, product_type, location, air_temp, process_temp,
                    rotational_speed, torque, tool_wear, trajectory, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """, sample_equipment)
            conn.commit()


# ==========================================
# REGISTERED ASSET OPERATIONS (CRUD)
# ==========================================

def insert_equipment(
    equipment_id: str,
    product_type: str = "M",
    location: str = "Plant A (Main)",
    air_temp: float = 300.0,
    process_temp: float = 310.0,
    rotational_speed: float = 1500.0,
    torque: float = 40.0,
    tool_wear: float = 30.0,
    trajectory: str = "Stable",
    status: str = "Healthy"
) -> int:
    """Register a new machine asset in the database."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO registered_equipment (
                equipment_id, product_type, location, air_temp, process_temp,
                rotational_speed, torque, tool_wear, trajectory, status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
        """, (
            equipment_id, product_type, location, air_temp, process_temp,
            rotational_speed, torque, tool_wear, trajectory, status
        ))
        conn.commit()
        return cursor.lastrowid


def get_all_registered_equipment() -> pd.DataFrame:
    """Retrieve all registered custom equipment assets."""
    with get_connection() as conn:
        df = pd.read_sql_query("""
            SELECT * FROM registered_equipment ORDER BY id DESC
        """, conn)
    return df


def update_equipment(
    equipment_id: str,
    location: Optional[str] = None,
    air_temp: Optional[float] = None,
    process_temp: Optional[float] = None,
    rotational_speed: Optional[float] = None,
    torque: Optional[float] = None,
    tool_wear: Optional[float] = None,
    trajectory: Optional[str] = None,
    status: Optional[str] = None
) -> bool:
    """Update telemetry and operational parameters for a registered machine."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM registered_equipment WHERE equipment_id = ?", (equipment_id,))
        row = cursor.fetchone()
        if not row:
            return False
            
        cursor.execute("""
            UPDATE registered_equipment
            SET location = COALESCE(?, location),
                air_temp = COALESCE(?, air_temp),
                process_temp = COALESCE(?, process_temp),
                rotational_speed = COALESCE(?, rotational_speed),
                torque = COALESCE(?, torque),
                tool_wear = COALESCE(?, tool_wear),
                trajectory = COALESCE(?, trajectory),
                status = COALESCE(?, status)
            WHERE equipment_id = ?
        """, (
            location, air_temp, process_temp, rotational_speed, torque,
            tool_wear, trajectory, status, equipment_id
        ))
        conn.commit()
        return True


def delete_equipment(equipment_id: str) -> bool:
    """Remove a registered equipment asset from the database."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM registered_equipment WHERE equipment_id = ?", (equipment_id,))
        conn.commit()
        return cursor.rowcount > 0
