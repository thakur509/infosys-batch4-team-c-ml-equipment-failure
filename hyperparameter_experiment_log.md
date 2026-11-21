# Hyperparameter Tuning & Model Experimentation Log

## Project: Machine Learning System for Early Detection of Equipment Failure
**Internship**: Infosys Springboard Batch 4 - Team C  
**Benchmark Dataset**: AI4I 2020 Predictive Maintenance (10,000 instances, 3.39% positive class incidence)

---

## Experiment Summary Table

| Exp ID | Model Architecture | Hyperparameters | Accuracy | Precision | Recall | F1-Score | PR-AUC | ROC-AUC | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **EXP-01** | Logistic Regression | `C=1.0, penalty='l2', max_iter=500` | 96.80% | 53.00% | 58.82% | 55.76% | 51.20% | 84.10% | Baseline |
| **EXP-02** | Decision Tree (CART) | `criterion='gini', max_depth=12, min_samples_split=10` | 97.40% | 61.20% | 67.65% | 64.26% | 58.90% | 82.50% | High Overfit |
| **EXP-03** | Random Forest (Unweighted) | `n_estimators=100, max_depth=None` | 98.45% | 88.20% | 63.24% | 73.65% | 79.10% | 96.20% | Poor Minority Recall |
| **EXP-04** | Random Forest (`balanced`) | `n_estimators=100, class_weight='balanced'` | 98.80% | 78.40% | 86.76% | 82.37% | 84.20% | 97.90% | Strong Improvement |
| **EXP-05** | **Random Forest (`balanced_subsample`)** | `n_estimators=100, max_depth=None, class_weight='balanced_subsample', random_state=42` | **99.05%** | **81.43%** | **89.71%** | **85.27%** | **86.13%** | **98.42%** | **Selected Production Model** |
| **EXP-06** | Gradient Boosting (XGBoost) | `n_estimators=150, learning_rate=0.08, max_depth=5` | 98.75% | 79.10% | 85.30% | 82.08% | 83.50% | 97.80% | Runner-up |
| **EXP-07** | Multiclass 6-Class RF (T38) | `n_estimators=100, class_weight='balanced_subsample'` | 98.10% | 98.30% | 98.10% | 98.26% (W) | - | - | **Selected Diagnostic Model** |

---

## Detailed Run Notes

### Run EXP-01: Baseline Logistic Regression
- Standard scaler applied to 5 raw features + 3 engineered physical features.
- Failure recall was insufficient (58.82%) due to severe non-linear operating envelopes between rotational speed and torque.

### Run EXP-03 vs EXP-05: Class Imbalance Strategy
- Without class weighting, Random Forest achieved misleading 98.45% accuracy while missing over 36% of actual machinery failures.
- Employing `class_weight='balanced_subsample'` recalibrated tree splitting probabilities, jumping held-out failure Recall to **89.71%** and PR-AUC to **86.13%**.

### Run EXP-07: Multiclass Root-Cause Model (Task T38)
- Successfully isolates failure modes:
  - Overstrain Failure (OSF): F1 = 92.5%
  - Heat Dissipation Failure (HDF): F1 = 94.1%
  - Power Failure (PWF): F1 = 96.0%
  - Tool Wear Failure (TWF): F1 = 78.8%
  - Random Failure (RNF): F1 = 35.0% (inherently stochastic background noise)
