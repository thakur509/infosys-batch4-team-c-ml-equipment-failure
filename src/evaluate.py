"""Evaluation metrics computation for the Random Forest failure classifier."""

from typing import Any, Dict
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)


def compute_metrics(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """Compute comprehensive evaluation metrics on held-out test data."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "pr_auc": float(average_precision_score(y_test, y_prob)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "matrix": cm.tolist()
        },
        "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    }
    
    # Feature importances if available
    if hasattr(model, "feature_importances_"):
        importances = dict(zip(X_test.columns.tolist(), [float(v) for v in model.feature_importances_]))
        # Sort descending
        metrics["feature_importances"] = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))
    else:
        metrics["feature_importances"] = {}
        
    return metrics
