-- Initial Seed Data for Predictive Maintenance SQLite DB
INSERT OR IGNORE INTO equipment (equipment_id, product_type, location, torque, tool_wear, status, trajectory)
VALUES 
('CNC-MILL-01', 'M', 'Plant A (Machining)', 42.5, 35.0, 'Healthy', 'Stable'),
('CNC-LATHE-04', 'L', 'Plant A (Turning)', 58.2, 195.0, 'Warning', 'Deteriorating'),
('ROBOT-ARM-09', 'H', 'Plant B (Assembly)', 38.0, 15.0, 'Healthy', 'Stable'),
('PRESS-HYD-12', 'M', 'Plant A (Press Shop)', 65.4, 218.0, 'Critical', 'Deteriorating'),
('INJ-MOLD-03', 'H', 'Plant B (Plastics)', 34.1, 80.0, 'Healthy', 'Improving');

INSERT OR IGNORE INTO alerts (machine_id, severity, message, status, failure_type)
VALUES
('CNC-LATHE-04', 'Warning', 'Tool wear limit approaching warning threshold (>190 min)', 'New', 'TWF'),
('PRESS-HYD-12', 'Critical', 'Spindle overstrain product exceeded 14,000 min*Nm', 'Acknowledged', 'OSF'),
('CNC-MILL-01', 'Info', 'Routine 500-hour telemetry calibration scheduled', 'Resolved', 'None');

INSERT OR IGNORE INTO maintenance_schedule (machine_id, maintenance_type, description, priority, scheduled_date, assigned_technician, status)
VALUES
('PRESS-HYD-12', 'Corrective Maintenance', 'Emergency tool replacement and hydraulic pressure bleed', 'Critical', DATE('now', '+1 day'), 'Marcus Vance', 'Pending'),
('CNC-LATHE-04', 'Preventive Maintenance', 'Inspect spindle bearings and replace cutting insert', 'High', DATE('now', '+3 days'), 'Sarah Jenkins', 'In Progress'),
('ROBOT-ARM-09', 'Routine Inspection', 'Six-axis joint lubrication and backlash measurement', 'Low', DATE('now', '+7 days'), 'Alex Chen', 'Pending');
