"""
PlacementPredict - Linear Models at Depth (Module 2)
Implements:
1. Ridge Logistic Regression (L2 regularisation, handles multicollinearity)
2. Lasso Logistic Regression (L1 regularisation, sparsity & feature selection)
3. ElasticNet Logistic Regression (L1 + L2 blend)
4. Ridge/Linear Regression for continuous Salary Package prediction
5. Mathematical interpretation: Weights, Odds Ratios (exp(w)), and Feature Impacts
"""

from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.metrics import classification_report, roc_auc_score, mean_squared_error, r2_score

from src.preprocessor import build_preprocessor, get_feature_names


def create_logistic_pipeline(
    penalty: str = 'l2',
    C: float = 1.0,
    l1_ratio: float = 0.5,
    random_state: int = 42
) -> Pipeline:
    """
    Creates an end-to-end Pipeline: Preprocessor (with Scaling) -> Logistic Regression.
    
    Args:
        penalty: 'l2' (Ridge), 'l1' (Lasso), or 'elasticnet'
        C: Inverse regularization strength (1 / lambda)
        l1_ratio: Mixing parameter for ElasticNet (0.0 = L2, 1.0 = L1)
    """
    preprocessor = build_preprocessor(include_scaling=True)
    
    if penalty == 'l1':
        clf = LogisticRegression(
            penalty='l1',
            solver='saga',
            C=C,
            max_iter=1000,
            random_state=random_state
        )
    elif penalty == 'elasticnet':
        clf = LogisticRegression(
            penalty='elasticnet',
            solver='saga',
            l1_ratio=l1_ratio,
            C=C,
            max_iter=1000,
            random_state=random_state
        )
    else:  # default Ridge L2
        clf = LogisticRegression(
            penalty='l2',
            solver='lbfgs',
            C=C,
            max_iter=1000,
            random_state=random_state
        )
        
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])
    
    return pipeline


def create_salary_regressor_pipeline(alpha: float = 1.0, random_state: int = 42) -> Pipeline:
    """
    Creates an end-to-end Pipeline for predicting continuous Salary Package (CTC in LPA).
    Uses Ridge Regression to handle correlated academic predictors.
    """
    preprocessor = build_preprocessor(include_scaling=True)
    regressor = Ridge(alpha=alpha, random_state=random_state)
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', regressor)
    ])
    
    return pipeline


def extract_linear_interpretability(fitted_pipeline: Pipeline) -> Dict[str, Any]:
    """
    Extracts weights, odds ratios, and non-zero feature selection indicators from a trained linear pipeline.
    
    Odds Ratio = exp(weight)
    - OR > 1: Increases odds of placement
    - OR < 1: Decreases odds of placement
    - Weight == 0: Sparsity induced by Lasso (L1)
    """
    preprocessor = fitted_pipeline.named_steps['preprocessor']
    clf = fitted_pipeline.named_steps['classifier']
    
    feature_names = get_feature_names(preprocessor)
    weights = clf.coef_[0]
    intercept = float(clf.intercept_[0])
    
    # Pair features with weights and odds ratios
    impacts = []
    for name, w in zip(feature_names, weights):
        odds_ratio = float(np.exp(w))
        impacts.append({
            "feature": name,
            "weight": float(w),
            "odds_ratio": odds_ratio,
            "abs_weight": abs(float(w)),
            "is_zeroed_by_lasso": bool(abs(w) < 1e-5),
            "direction": "Positive" if w > 0 else "Negative" if w < 0 else "Neutral"
        })
        
    # Sort by absolute magnitude of impact
    impacts.sort(key=lambda x: x["abs_weight"], reverse=True)
    
    top_positive = [f for f in impacts if f["weight"] > 0][:8]
    top_negative = [f for f in impacts if f["weight"] < 0][:8]
    
    zeroed_features_count = sum(1 for f in impacts if f["is_zeroed_by_lasso"])
    
    return {
        "intercept": intercept,
        "total_features": len(impacts),
        "zeroed_features_count": zeroed_features_count,
        "top_positive_drivers": top_positive,
        "top_negative_drivers": top_negative,
        "all_impacts": impacts
    }
