from src.db_manager import (
    insert_prediction,
    insert_alert,
    update_alert_status,
    insert_maintenance,
    update_maintenance,
    delete_maintenance
)

def test_database_crud():
    # Alert lifecycle
    alert_id = insert_alert(
        equipment_id='TEST-UNIT-01',
        severity='Critical',
        condition='Emergency test alert: High Spindle Overstrain',
        notes='Automated unit test'
    )
    assert alert_id > 0
    update_alert_status(alert_id, 'Acknowledged', 'Technician A')
    update_alert_status(alert_id, 'Resolved')

    # Maintenance CRUD
    m_id = insert_maintenance(
        equipment_id='TEST-UNIT-01',
        maintenance_type='Inspection',
        description='Test order',
        priority='Low',
        assigned_technician='Tester',
        scheduled_date='2026-12-15'
    )
    assert m_id > 0
    delete_maintenance(m_id)
