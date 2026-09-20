"""
PlacementPredict - Tree-Based Models & Interpretability (Module 3)
Implements:
1. Decision Trees (Gini, Entropy, and Cost-Complexity Pruning / CCP)
2. Random Forest (Bagging, feature subsampling, Out-Of-Bag / OOB score)
3. Gradient Boosted Trees (LightGBM & XGBoost industrial workhorses)
4. Feature Importance (MDI / Gini Importance & Permutation Importance)
5. Individual prediction impact attribution
"""

from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import lightgbm as lgb
import xgboost as xgb
from sklearn.inspection import permutation_importance

from src.preprocessor import build_preprocessor, get_feature_names


def create_decision_tree_pipeline(
    criterion: str = 'gini',
    max_depth: Optional[int] = 10,
    min_samples_split: int = 20,
    min_samples_leaf: int = 10,
    ccp_alpha: float = 0.0,
    random_state: int = 42
) -> Pipeline:
    """Decision Tree pipeline with pre-pruning parameters and CCP alpha."""
    preprocessor = build_preprocessor(include_scaling=False)
    clf = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        ccp_alpha=ccp_alpha,
        random_state=random_state
    )
    return Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])


def create_random_forest_pipeline(
    n_estimators: int = 150,
    max_depth: Optional[int] = 15,
    min_samples_split: int = 10,
    max_features: str = 'sqrt',
    random_state: int = 42
) -> Pipeline:
    """Random Forest ensemble pipeline with OOB score computation."""
    preprocessor = build_preprocessor(include_scaling=False)
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        max_features=max_features,
        oob_score=True,
        n_jobs=-1,
        random_state=random_state
    )
    return Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])


def create_lightgbm_pipeline(
    n_estimators: int = 150,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    max_depth: int = -1,
    random_state: int = 42
) -> Pipeline:
    """LightGBM Classifier: fast histogram-based splitting and leaf-wise tree growth."""
    preprocessor = build_preprocessor(include_scaling=False)
    clf = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1
    )
    return Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])


def create_xgboost_pipeline(
    n_estimators: int = 150,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_state: int = 42
) -> Pipeline:
    """XGBoost Classifier with regularisation and column subsampling."""
    preprocessor = build_preprocessor(include_scaling=False)
    clf = xgb.XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        eval_metric='logloss',
        random_state=random_state,
        n_jobs=-1
    )
    return Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])


def extract_tree_feature_importance(
    fitted_pipeline: Pipeline,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    compute_permutation: bool = False
) -> Dict[str, Any]:
    """
    Extracts Gini / MDI feature importances and optional Permutation Importances.
    """
    preprocessor = fitted_pipeline.named_steps['preprocessor']
    clf = fitted_pipeline.named_steps['classifier']
    feature_names = get_feature_names(preprocessor)
    
    # 1. Gini / MDI importance
    if hasattr(clf, 'feature_importances_'):
        importances = clf.feature_importances_
    else:
        importances = np.zeros(len(feature_names))
        
    mdi_list = []
    for name, score in zip(feature_names, importances):
        mdi_list.append({
            "feature": name,
            "importance": float(score)
        })
    mdi_list.sort(key=lambda x: x["importance"], reverse=True)
    
    result = {
        "mdi_importance": mdi_list,
        "top_features": mdi_list[:10]
    }
    
    # 2. Permutation Importance (Model-Agnostic)
    if compute_permutation and X_val is not None and y_val is not None:
        perm_res = permutation_importance(
            fitted_pipeline, X_val, y_val,
            n_repeats=5,
            random_state=42,
            n_jobs=-1
        )
        perm_list = []
        for name, mean_score, std_score in zip(X_val.columns, perm_res.importances_mean, perm_res.importances_std):
            perm_list.append({
                "feature": name,
                "importance_mean": float(mean_score),
                "importance_std": float(std_score)
            })
        perm_list.sort(key=lambda x: x["importance_mean"], reverse=True)
        result["permutation_importance"] = perm_list
        
    return result
