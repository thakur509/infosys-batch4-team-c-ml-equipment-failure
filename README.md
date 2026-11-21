# Predictive Maintenance & Machine Failure Dashboard

An industrial-grade, full-stack Machine Failure Prediction and Predictive Maintenance platform built with **Python**, **Streamlit**, **Scikit-learn**, **Plotly**, and **SQLite**, trained on the benchmark **AI4I 2020 Predictive Maintenance Dataset**.

---

## 🌟 Key Features

1. **Overview Dashboard**
   - Fleet-wide KPI summary cards (Total Equipment, Healthy Units, Machines at Risk, Critical Failures, Active Alerts, Pending Work Orders).
   - Real-time fleet health distribution donut chart and failure mode categorization.
   - Live telemetry baseline summary (Mean, Std Dev, Min, Median, Max across all sensors).
   - Recent alerts and recent prediction audit log feed.

2. **Failure Prediction Studio**
   - Interactive manual sensor reading input form with validation and physical unit tooltips:
     - Air Temperature ($K$)
     - Process Temperature ($K$)
     - Rotational Speed ($RPM$)
     - Torque ($Nm$)
     - Tool Wear ($min$)
   - Quick one-click telemetry presets (Normal, Heat Dissipation Risk, Power Failure, Overstrain, Tool Wear).
   - Random Forest model inference producing **Failure Likelihood**, **Calibrated Failure Probability %**, and **Risk Level** (Low, Moderate, High, Critical).
   - Animated Plotly Probability Gauge with color-coded risk zones.
   - Derived physical stress indices: Thermal Delta ($\Delta T$), Motor Power Output ($kW$), and Overstrain Product ($min \cdot Nm$).
   - Prescriptive maintenance action recommendations.
   - One-click work order generation directly from high-risk predictions.
   - Persistent prediction history table backed by SQLite with dynamic CSV export.

3. **Equipment Monitoring**
   - Searchable and filterable machinery registry (10,000 observations).
   - Status filters (Healthy, Warning, Critical) and Product Quality Tiers (L, M, H).
   - Deep-dive machine telemetry drawer:
     - Radar comparison chart benchmarking the selected machine against fleet nominal baselines.
     - Physical stress indices and individual sensor warning badges.
     - Direct actions to raise system alerts or log work orders.
   - Filtered dataset CSV export.

4. **Anomaly & Outlier Detection**
   - Dual-layer detection combining **Statistical IQR Thresholds** ($1.5\times$ warning and $2.5\times$ critical) with a multivariate **Isolation Forest** outlier engine.
   - Identification of specific breached sensor parameters.
   - Metric cards: Total Flagged Readings, Critical Anomalies, Warning Anomalies, Healthy Readings.
   - Severity filters and search bar.
   - Interactive Speed vs. Torque operating envelope scatter plot with anomaly highlighting.
   - Bar chart of outlier frequency per sensor.
   - Filtered anomaly CSV report export.

5. **Equipment Alerts Management**
   - Inspired by industrial SCADA/DCS alert dispatch interfaces.
   - Lifecycle workflow: **New $\rightarrow$ Acknowledged $\rightarrow$ Resolved**.
   - Immediate state persistence in SQLite.
   - Severity tagging: Critical, Warning, Informational.
   - Coverage of AI4I failure modes: Tool Wear Failure (TWF), Heat Dissipation Failure (HDF), Power Failure (PWF), Overstrain Failure (OSF), and Random Failure (RNF).
   - Interactive management panel to acknowledge, resolve, and view telemetry details.
   - Manual alert logging interface.
   - CSV export of alert registers.

6. **Predictive Maintenance & Work Orders**
   - End-to-end work order coordination with technician assignment and scheduling.
   - Work order tracking: Pending, In Progress, Completed.
   - Priority tagging: Low, Medium, High, Critical.
   - Full CRUD operations: Add new work orders, edit existing records, mark completed, delete with confirmation.
   - Search by machine or technician.
   - CSV export of maintenance schedule.

7. **Trends & Analytical Reports**
   - Sequential telemetry run plots with failure point markers.
   - Multi-sensor Pearson correlation heatmap with KaTeX labels.
   - Mechanical power hyperbolas and thermal gradient distributions.
   - Normal vs. Failed sensor comparison boxplots.
   - Failure rate by machine quality tier ($L$, $M$, $H$).
   - Comprehensive model evaluation audit: Held-out Test Accuracy ($99.05\%$), Precision ($81.4\%$), Recall ($89.7\%$), F1-Score ($85.3\%$), ROC-AUC ($98.4\%$), and PR-AUC ($86.1\%$).
   - Interactive Confusion Matrix heatmap and Feature Importance ranking.
   - Model retraining button and metrics CSV export.

8. **Advanced ML & Diagnostics (Tasks T38–T45)**
   - **6-Class Multiclass Failure Mode Classifier**: Predicts specific root cause (No Failure, Tool Wear Failure, Heat Dissipation Failure, Power Failure, Overstrain Failure, Random Failure) with class probability breakdown.
   - **Multi-Algorithm Benchmark Leaderboard**: Comparing Random Forest, Gradient Boosting, LightGBM/XGBoost, Logistic Regression, Decision Tree across Accuracy, Recall, F1, PR-AUC, and Train Time.
   - **Explainable AI (TreeSHAP & Permutation)**: Feature impact analysis explaining exactly which sensor caused the risk.
   - **Dynamic Cost-Sensitive Decision Threshold Tuning**: Interactive slider balancing False Positives vs. False Negatives with cost curves.
   - **Cross-Plant Generalization Stress Testing**: Synthetic noise simulation testing model resilience under calibration drift.
   - **Streaming Telemetry Real-Time Ingestion Simulator**: Simulated live factory edge streaming with real-time inference buffer.

9. **Live Machine Health Monitoring**
   - Industrial edge monitoring view displaying key operational telemetry.
   - 4-card live metrics: Equipment Health Score, Operating Status, Cumulative Run Hours, Failure Risk Index.
   - Alerts & Operational Status live banner with instantaneous health notifications.
   - Real-time time-series sensor trend tracking with automated anomaly markers.
   - Interactive "Add Current Measurement" form with live predictive model inference and database dispatch.

---

## 🛠️ Technology Stack

- **Dashboard / Frontend**: Streamlit (with Obsidian & Slate industrial theme CSS)
- **Machine Learning**: Scikit-Learn (`RandomForestClassifier`, `GradientBoostingClassifier`, `IsolationForest`)
- **Data Processing**: Pandas, NumPy
- **Interactive Visualizations**: Plotly Express & Plotly Graph Objects
- **Database / Persistence**: SQLite3 (`database/maintenance.db`)
- **Model Serialization**: Joblib

---

## 📂 Project Structure

```text
Predictive-Maintenance/
│
├── app.py                      # Main entrypoint with st.navigation & layout
├── ai4i2020.csv                # AI4I 2020 Predictive Maintenance dataset (10,000 rows)
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── run_dashboard.ps1           # Windows PowerShell launch helper
├── run_dashboard.bat           # Windows Batch launch helper
│
├── src/
│   ├── __init__.py
│   ├── styles.py               # Custom Obsidian & Slate industrial CSS & UI components
│   ├── data_loader.py          # Data loading, cleaning, and caching
│   ├── preprocessing.py        # Feature engineering and train-test splits
│   ├── trainmodel.py           # Binary Random Forest training pipeline
│   ├── train_multiclass.py     # 6-class failure mode classifier
│   ├── predict.py              # Inference service and prescriptive diagnostics
│   ├── evaluate.py             # Performance metrics (PR-AUC, Confusion Matrix, F1)
│   ├── anomaly_detection.py    # IQR boundary analysis and Isolation Forest
│   ├── advanced_analytics.py   # Benchmark leaderboard, SHAP, threshold tuning, stress test
│   ├── db_manager.py           # SQLite persistence for alerts, predictions, maintenance
│   └── verify_system.py        # 8-suite automated system verification
│
├── models/
│   ├── random_forest_model.joblib
│   ├── failure_type_model.joblib
│   ├── model_metadata.json
│   └── failure_type_metadata.json
│
├── database/
│   └── maintenance.db          # Persistent SQLite database
│
├── pages/
│   ├── 1_Overview.py           # Fleet KPIs, Hero Bento & health distribution
│   ├── 2_Failure_Prediction.py # Dual-unit input (C/K), presets & radial gauge
│   ├── 3_Equipment_Monitoring.py # Machinery registry & telemetry drilldown
│   ├── 4_Anomaly_Detection.py  # Outlier detection & dynamic scatter envelope
│   ├── 5_Equipment_Alerts.py   # Actionable SCADA alert lifecycle management
│   ├── 6_Maintenance.py        # Work order scheduling & technician assignment
│   ├── 7_Trends_Reports.py     # Heatmap, correlation & model evaluation
│   ├── 8_Advanced_ML.py        # 6-class classifier, benchmarks, SHAP & stress test
│   └── 9_Live_Monitoring.py    # Industrial live health monitoring & sensor ingestion
│
└── reports/
    └── charts/                 # Generated exports directory
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+ installed on your system.
- Recommended: Visual Studio Code.

### 2. Environment Setup
Clone or navigate to the project directory:
```bash
cd "c:\Users\HII\Desktop\ml.infosys"
```

Create and activate a virtual environment:
```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install required packages:
```powershell
pip install -r requirements.txt
```

### 3. Launch the Application
Run the Streamlit application:
```powershell
streamlit run app.py
```

The application will automatically:
1. Verify and clean the `ai4i2020.csv` dataset.
2. Initialize and seed the SQLite database `database/maintenance.db`.
3. Check and load the trained Random Forest model (or train on cold start if missing).
4. Launch the web dashboard at `http://localhost:8501`.

---

## 🧠 Machine Learning Details

### Input Features
- `Air temperature [K]`: Surrounding ambient temperature (nominal $295 - 304\text{ K}$)
- `Process temperature [K]`: Machine operational temperature (nominal $305 - 314\text{ K}$)
- `Rotational speed [rpm]`: Spindle speed (nominal $1300 - 1800\text{ RPM}$)
- `Torque [Nm]`: Mechanical torque load (nominal $30 - 55\text{ Nm}$)
- `Tool wear [min]`: Cumulative tool machining duration (nominal $0 - 200\text{ min}$)

### Domain-Engineered Physical Features
- **Thermal Delta**: $\Delta T = T_{\text{process}} - T_{\text{air}}$ (governs heat dissipation failure when $\Delta T < 8.6\text{ K}$)
- **Rotational Power**: $P = \frac{\tau \times \omega \times 2\pi}{60000}\text{ kW}$ (governs electrical/power overload when $P < 3.5\text{ kW}$ or $P > 9.0\text{ kW}$)
- **Mechanical Overstrain**: $S = \text{Tool wear} \times \text{Torque}$ (governs mechanical fracture when $S > 11,000\text{ min}\cdot\text{Nm}$)

### Class Imbalance Handling
- The dataset exhibits a severe class imbalance ($\sim 3.4\%$ positive failure rate, 339 failures out of 10,000 records).
- We employ stratified 80/20 sampling and `balanced_subsample` ensemble weighting within the Random Forest.
- Optimization targets the **Precision-Recall AUC ($86.13\%$)** and **F1-Score ($85.27\%$)**, avoiding misleading accuracy-only metrics.
- No target leakage columns (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) are used during model training or prediction.

---

## 📄 License & Attribution
- Dataset: AI4I 2020 Predictive Maintenance Dataset (UCI Machine Learning Repository).
- Built for industrial predictive maintenance analytics and placement evaluation demonstrations.
