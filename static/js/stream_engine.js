// Telemetry Simulation & Visual Helper Functions
console.log('Telemetry OS Industrial Engine Initialized');

function computeStressIndices(airK, procK, rpm, torque, toolWear) {
  const tempDiff = procK - airK;
  const powerKw = (torque * rpm * 2 * Math.PI) / 60000;
  const overstrain = toolWear * torque;
  return { tempDiff, powerKw, overstrain };
}
