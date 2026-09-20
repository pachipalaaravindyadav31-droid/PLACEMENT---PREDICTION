"""
PlacementPredict - Model Evaluation Module
Computes full classification and regression performance metrics.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def evaluate_classifier(pipeline, X_test: pd.DataFrame, y_test: pd.Series, model_name: str = "") -> Dict[str, Any]:
    """Evaluates a classification model on test data."""
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    metrics = {
        "model_name": model_name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)) if y_prob is not None else None,
        "log_loss": float(log_loss(y_test, y_prob)) if y_prob is not None else None,
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        }
    }
    return metrics


def evaluate_regressor(pipeline, X_test: pd.DataFrame, y_test: pd.Series, model_name: str = "") -> Dict[str, Any]:
    """Evaluates a continuous regression model (e.g. Salary Package)."""
    # Filter only positive / placed samples for realistic salary estimation
    mask = y_test > 0
    X_eval = X_test[mask]
    y_eval = y_test[mask]
    
    y_pred = pipeline.predict(X_eval)
    
    mse = float(mean_squared_error(y_eval, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_eval, y_pred))
    r2 = float(r2_score(y_eval, y_pred))
    
    return {
        "model_name": model_name,
        "mae_lpa": round(mae, 3),
        "rmse_lpa": round(rmse, 3),
        "r2_score": round(r2, 4),
        "sample_count": int(len(y_eval))
    }
