-- Team C Initial Operational Seed Data for SQLite Telemetry DB
INSERT OR IGNORE INTO equipment (equipment_id, product_type, location, torque, tool_wear, status, trajectory)
VALUES 
('TC-MILL-101', 'M', 'Plant 1 (CNC Milling)', 42.5, 35.0, 'Healthy', 'Stable'),
('TC-LATHE-202', 'L', 'Plant 1 (Spindle Turning)', 58.2, 195.0, 'Warning', 'Deteriorating'),
('TC-ROBOT-303', 'H', 'Plant 2 (6-Axis Robotic Arm)', 38.0, 15.0, 'Healthy', 'Stable'),
('TC-PRESS-404', 'M', 'Plant 1 (Heavy Hydraulic Press)', 65.4, 218.0, 'Critical', 'Deteriorating'),
('TC-MOLD-505', 'H', 'Plant 2 (Injection Molding)', 34.1, 80.0, 'Healthy', 'Improving');

INSERT OR IGNORE INTO alerts (machine_id, severity, message, status, failure_type)
VALUES
('TC-LATHE-202', 'Warning', 'Tool wear limit approaching warning threshold (>190 min)', 'New', 'TWF'),
('TC-PRESS-404', 'Critical', 'Spindle overstrain product exceeded 14,000 min*Nm', 'Acknowledged', 'OSF'),
('TC-MILL-101', 'Info', 'Routine 500-hour telemetry calibration scheduled', 'Resolved', 'None');

INSERT OR IGNORE INTO maintenance_schedule (machine_id, maintenance_type, description, priority, scheduled_date, assigned_technician, status)
VALUES
('TC-PRESS-404', 'Corrective Maintenance', 'Emergency tool replacement and hydraulic pressure bleed', 'Critical', DATE('now', '+1 day'), 'Marcus Vance', 'Pending'),
('TC-LATHE-202', 'Preventive Maintenance', 'Inspect spindle bearings and replace cutting insert', 'High', DATE('now', '+3 days'), 'Sarah Jenkins', 'In Progress'),
('TC-ROBOT-303', 'Routine Inspection', 'Six-axis joint lubrication and backlash measurement', 'Low', DATE('now', '+7 days'), 'Alex Chen', 'Pending');
