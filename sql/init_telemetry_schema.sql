-- Predictive Maintenance Database Schema (SQLite)
-- Schema Definition for Industrial Telemetry, Anomaly Detection & Work Order Management

CREATE TABLE IF NOT EXISTS equipment (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    equipment_id TEXT UNIQUE NOT NULL,
    product_type TEXT CHECK(product_type IN ('L', 'M', 'H')) NOT NULL,
    location TEXT DEFAULT 'Plant A (Main)',
    installed_date DATE DEFAULT (DATE('now')),
    torque REAL DEFAULT 40.0,
    tool_wear REAL DEFAULT 0.0,
    status TEXT DEFAULT 'Healthy',
    trajectory TEXT DEFAULT 'Stable',
    last_inspected TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    air_temp REAL NOT NULL,
    process_temp REAL NOT NULL,
    rotational_speed REAL NOT NULL,
    torque REAL NOT NULL,
    tool_wear REAL NOT NULL,
    product_type TEXT DEFAULT 'M',
    prediction INTEGER NOT NULL,
    probability REAL NOT NULL,
    risk_level TEXT NOT NULL,
    failure_mode TEXT DEFAULT 'None',
    temp_diff REAL,
    power_kw REAL,
    overstrain REAL,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    machine_id TEXT NOT NULL,
    severity TEXT CHECK(severity IN ('Critical', 'Warning', 'Info')) NOT NULL,
    message TEXT NOT NULL,
    status TEXT CHECK(status IN ('New', 'Acknowledged', 'Resolved')) DEFAULT 'New',
    resolved_at TIMESTAMP,
    acknowledged_by TEXT,
    failure_type TEXT
);

CREATE TABLE IF NOT EXISTS maintenance_schedule (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    machine_id TEXT NOT NULL,
    maintenance_type TEXT NOT NULL,
    description TEXT,
    priority TEXT CHECK(priority IN ('Low', 'Medium', 'High', 'Critical')) DEFAULT 'Medium',
    scheduled_date DATE NOT NULL,
    assigned_technician TEXT,
    status TEXT CHECK(status IN ('Pending', 'In Progress', 'Completed')) DEFAULT 'Pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status);
CREATE INDEX IF NOT EXISTS idx_alerts_machine ON alerts(machine_id);
CREATE INDEX IF NOT EXISTS idx_predictions_time ON predictions(timestamp);
CREATE INDEX IF NOT EXISTS idx_maintenance_status ON maintenance_schedule(status);
CREATE INDEX IF NOT EXISTS idx_equipment_id ON equipment(equipment_id);
