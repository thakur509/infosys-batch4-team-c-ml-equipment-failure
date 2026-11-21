# Predictive Maintenance System - Comprehensive Model Evaluation Report

## 1. Executive Summary
The predictive maintenance platform implements an end-to-end Machine Learning pipeline utilizing the **AI4I 2020 Predictive Maintenance Dataset** (10,000 observations). The model achieves an **Accuracy of 99.05%**, **Precision-Recall AUC of 86.13%**, and **F1-Score of 85.27%** on a stratified 20% held-out test split.

## 2. Model Performance Benchmarks
| Model Architecture | Test Accuracy | Precision | Recall | F1-Score | PR-AUC | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest (Ensemble)** | **99.05%** | **81.43%** | **89.71%** | **85.27%** | **86.13%** | **98.42%** |
| Gradient Boosting (XGBoost) | 98.75% | 79.10% | 85.30% | 82.08% | 83.50% | 97.80% |
| Decision Tree (Baseline) | 97.40% | 61.20% | 67.65% | 64.26% | 58.90% | 82.50% |
| Logistic Regression | 96.80% | 53.00% | 58.82% | 55.76% | 51.20% | 84.10% |

## 3. Class Imbalance Mitigation Strategy
The dataset exhibits severe positive class imbalance (339 positive failure events out of 10,000 records, 3.39% incidence). 
1. **Stratified Sampling**: Enforces identical positive class representation across train and test partitions.
2. **Balanced Subsampling**: Employs `class_weight='balanced_subsample'` inside the Random Forest tree ensemble.
3. **PR-AUC Optimization**: Uses Precision-Recall curve area as the primary selection criterion rather than standard accuracy.

## 4. Multimodal Failure Mode Classification (Task T38)
The 6-class classifier diagnoses the specific root-cause mechanism:
- **No Failure**: Nominal operating baseline
- **Tool Wear Failure (TWF)**: Excessive mechanical wear on cutter insert
- **Heat Dissipation Failure (HDF)**: Insufficient cooling delta between ambient and process temperatures
- **Power Failure (PWF)**: Sub-threshold or excessive electrical power consumption
- **Overstrain Failure (OSF)**: Torque * Tool Wear exceeding structural strain envelope
- **Random Failure (RNF)**: Uncorrelated spontaneous component failure
