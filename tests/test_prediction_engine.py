from src.predict import predict_machine_failure
from src.train_multiclass import predict_failure_mode

def test_predict_nominal():
    res = predict_machine_failure(
        air_temp=298.1,
        process_temp=308.2,
        rotational_speed=1500,
        torque=40.0,
        tool_wear=30
    )
    assert res['predicted_failure'] == 0
    assert res['failure_probability'] < 0.50
    assert res['risk_level'] in ['Low', 'Moderate']

def test_predict_critical_stress():
    res = predict_machine_failure(
        air_temp=304.5,
        process_temp=308.6,
        rotational_speed=1180,
        torque=72.0,
        tool_wear=235
    )
    assert res['failure_probability'] > 0.50
    assert res['risk_level'] in ['High', 'Critical']

def test_multiclass_diagnostic():
    mode, conf, _ = predict_failure_mode(
        air_temp=303.0,
        process_temp=309.0,
        rotational_speed=1250,
        torque=68.0,
        tool_wear=230,
        product_type='M'
    )
    assert isinstance(mode, str)
    assert 0.0 <= conf <= 1.0
