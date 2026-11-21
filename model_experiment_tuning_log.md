# Team C Model Optimization & Hyperparameter Tuning Log

## Project: Predictive Maintenance & Equipment Health Intelligence
**Team Attribution**: Infosys Springboard Batch 4 — Team C  
**Target Dataset**: AI4I 2020 Predictive Maintenance Benchmark (10,000 industrial run-to-failure records)  
**Primary Metric**: Precision-Recall AUC (PR-AUC) & F1-Score (prioritizing failure recall over unweighted accuracy)

---

## Model Benchmark & Tuning Matrix

| Experiment ID | Architecture & Pipeline | Hyperparameter Configuration | Accuracy | Precision | Recall | F1-Score | PR-AUC | ROC-AUC | Outcome |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TC-EXP-01** | Standard Logistic Regression | `C=1.0, penalty='l2', max_iter=500, solver='lbfgs'` | 96.80% | 53.00% | 58.82% | 55.76% | 51.20% | 84.10% | Linear Baseline |
| **TC-EXP-02** | Decision Tree Classifier | `criterion='gini', max_depth=10, min_samples_split=12` | 97.40% | 61.20% | 67.65% | 64.26% | 58.90% | 82.50% | High Overfit |
| **TC-EXP-03** | Random Forest (Unweighted) | `n_estimators=100, max_depth=None, min_samples_leaf=1` | 98.45% | 88.20% | 63.24% | 73.65% | 79.10% | 96.20% | Poor Minority Recall |
| **TC-EXP-04** | Random Forest (`balanced`) | `n_estimators=100, class_weight='balanced'` | 98.80% | 78.40% | 86.76% | 82.37% | 84.20% | 97.90% | Strong Improvement |
| **TC-EXP-05** | **Random Forest (`balanced_subsample`)** | `n_estimators=100, class_weight='balanced_subsample', random_state=42` | **99.05%** | **81.43%** | **89.71%** | **85.27%** | **86.13%** | **98.42%** | **Primary Binary Model** |
| **TC-EXP-06** | Gradient Boosting (XGBoost) | `n_estimators=150, learning_rate=0.08, max_depth=5` | 98.75% | 79.10% | 85.30% | 82.08% | 83.50% | 97.80% | High Precision Alternative |
| **TC-EXP-07** | **6-Class Multiclass Diagnostic RF** | `n_estimators=100, class_weight='balanced_subsample'` | **98.10%** | **98.30%** | **98.10%** | **98.26%** | - | - | **Root-Cause Diagnostic Engine** |

---

## Team C Engineering Synthesis

### 1. Imbalance Mitigation via Subsample Weighting (TC-EXP-05)
Standard tree splits heavily favor the 96.6% majority non-failure class. By computing `balanced_subsample` weights per bootstrap replicate, we penalize false negatives dynamically, boosting critical failure recall from 63.24% to **89.71%** without generating excessive false alarms.

### 2. Multi-Class Root Cause Resolution (TC-EXP-07)
Rather than just predicting a generic failure flag, the 6-class model predicts the specific physical failure mechanism:
- **Overstrain Failure (OSF)**: $F_1 = 92.5\%$ — Triggered when $\text{Tool Wear} \times \text{Torque} > 11,000\,\text{min}\cdot\text{Nm}$
- **Heat Dissipation Failure (HDF)**: $F_1 = 94.1\%$ — Triggered when $\Delta T < 8.6\,\text{K}$ at low rotational speeds
- **Power Failure (PWF)**: $F_1 = 96.0\%$ — Triggered when spindle mechanical power falls outside $3.5 - 9.0\,\text{kW}$
- **Tool Wear Failure (TWF)**: $F_1 = 78.8\%$ — Gradual insert degradation exceeding structural limits
- **Random Failure (RNF)**: $F_1 = 35.0\%$ — Spontaneous stochastic occurrences modeled independently

