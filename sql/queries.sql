-- Analytical & Monitoring Queries for SCADA Telemetry & Maintenance

-- 1. Fleet Health Summary
SELECT 
    COUNT(*) as total_units,
    SUM(CASE WHEN status = 'Healthy' THEN 1 ELSE 0 END) as healthy_units,
    SUM(CASE WHEN status = 'Warning' THEN 1 ELSE 0 END) as warning_units,
    SUM(CASE WHEN status = 'Critical' THEN 1 ELSE 0 END) as critical_units,
    ROUND(AVG(torque), 2) as fleet_avg_torque,
    ROUND(AVG(tool_wear), 2) as fleet_avg_tool_wear
FROM equipment;

-- 2. Active Alert Distribution by Severity
SELECT 
    severity, 
    COUNT(*) as alert_count 
FROM alerts 
WHERE status != 'Resolved' 
GROUP BY severity 
ORDER BY CASE severity 
    WHEN 'Critical' THEN 1 
    WHEN 'Warning' THEN 2 
    WHEN 'Info' THEN 3 
END;

-- 3. Maintenance Work Order Backlog by Technician
SELECT 
    assigned_technician, 
    COUNT(*) as total_assigned,
    SUM(CASE WHEN priority = 'Critical' THEN 1 ELSE 0 END) as critical_orders,
    SUM(CASE WHEN status = 'Pending' THEN 1 ELSE 0 END) as pending_orders
FROM maintenance_schedule
WHERE status != 'Completed'
GROUP BY assigned_technician
ORDER BY total_assigned DESC;

-- 4. High Risk Recent Predictions Audit Log
SELECT 
    id, timestamp, probability, risk_level, failure_mode, notes
FROM predictions
WHERE probability >= 0.50 OR risk_level IN ('High', 'Critical')
ORDER BY timestamp DESC
LIMIT 50;
