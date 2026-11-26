# Predictive Maintenance System Architecture

```text
[ Physical Machinery / Sensors ]
              │
              ▼
[ Industrial Telemetry Ingestion ] (Dual-Unit °C / K Normalization)
              │
              ├──▶ [ Anomaly Detection Engine ] (IQR & Isolation Forest)
              │             │
              │             ▼
              │      [ SCADA Alert State Machine ] (New -> Ack -> Resolved)
              │
              ├──▶ [ Feature Engineering Layer ] (Thermal Delta, kW Power, Overstrain)
              │             │
              │             ▼
              │      [ ML Inference Service ]
              │             ├── Binary Failure Prediction (Random Forest)
              │             └── 6-Class Root-Cause Classifier (T38)
              │                     │
              │                     ▼
              │      [ Prescriptive Action & Work Order Dispatch ]
              │
              ▼
[ SQLite Persistence Engine ] (Equipment Registry, Telemetry, Audit Logs)
              │
              ▼
[ Obsidian / Slate Interactive Dashboard ] (Streamlit Multi-Page UI)
              │
              ▼
[ Live Cloud Deployment ] (Render Web Service)
  URL: https://infosys-batch4-team-c-ml-equipment.onrender.com/
```
