"""
PlacementPredict - Model Training & Artifact Packaging Orchestrator
Executes training for both M2 (Linear Models) and M3 (Tree-Based Models),
evaluates performance, generates interpretability data, and packages production artifacts.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import json
import joblib
import time
import pandas as pd
import numpy as np

from src.data_loader import prepare_datasets
from src.linear_models import (
    create_logistic_pipeline,
    create_salary_regressor_pipeline,
    extract_linear_interpretability
)
from src.tree_models import (
    create_decision_tree_pipeline,
    create_random_forest_pipeline,
    create_lightgbm_pipeline,
    create_xgboost_pipeline,
    extract_tree_feature_importance
)
from src.evaluator import evaluate_classifier, evaluate_regressor


def run_full_training_lifecycle(artifacts_dir: str = "artifacts"):
    """Runs data ingestion, preprocessing, multi-model training, evaluation, and artifact export."""
    os.makedirs(artifacts_dir, exist_ok=True)
    print("=" * 60)
    print("[*] STARTING PLACEMENTPREDICT ML LIFECYCLE (M1, M2, M3)")
    print("=" * 60)

    # 1. Load & Partition Data
    print("\n[Stage 1/5] Ingesting & Partitioning 50k Dataset...")
    data = prepare_datasets(filter_anomalies=True)
    X_train, y_clf_train, y_reg_train = data["X_train"], data["y_clf_train"], data["y_reg_train"]
    X_val, y_clf_val, y_reg_val = data["X_val"], data["y_clf_val"], data["y_reg_val"]
    X_test, y_clf_test, y_reg_test = data["X_test"], data["y_clf_test"], data["y_reg_test"]
    stats = data["stats"]
    print(f"[+] Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"[+] Placement Rate: {stats['placement_rate']*100:.1f}%, Filtered Anomalies: {stats['anomalies_count']}")

    # 2. Module 2: Train Linear Models
    print("\n[Stage 2/5] Training Module 2 Linear Models...")
    
    # 2a. Ridge Logistic Regression (L2)
    print("  -> Training Ridge (L2) Logistic Regression...")
    t0 = time.time()
    ridge_clf = create_logistic_pipeline(penalty='l2', C=1.0)
    ridge_clf.fit(X_train, y_clf_train)
    ridge_time = time.time() - t0
    ridge_metrics = evaluate_classifier(ridge_clf, X_test, y_clf_test, "Ridge (L2) Logistic Regression")
    ridge_metrics["train_time_sec"] = round(ridge_time, 3)

    # 2b. Lasso Logistic Regression (L1)
    print("  -> Training Lasso (L1) Logistic Regression (Sparsity check)...")
    lasso_clf = create_logistic_pipeline(penalty='l1', C=0.5)
    lasso_clf.fit(X_train, y_clf_train)
    lasso_metrics = evaluate_classifier(lasso_clf, X_test, y_clf_test, "Lasso (L1) Logistic Regression")

    # 2c. ElasticNet Logistic Regression
    print("  -> Training ElasticNet Logistic Regression...")
    elastic_clf = create_logistic_pipeline(penalty='elasticnet', C=0.5, l1_ratio=0.5)
    elastic_clf.fit(X_train, y_clf_train)
    elastic_metrics = evaluate_classifier(elastic_clf, X_test, y_clf_test, "ElasticNet Logistic Regression")

    # 2d. Ridge Salary Regressor (for continuous CTC in LPA)
    print("  -> Training Ridge Salary Regressor...")
    salary_mask_train = (y_clf_train == 1) & (y_reg_train > 0)
    salary_reg = create_salary_regressor_pipeline(alpha=1.0)
    salary_reg.fit(X_train[salary_mask_train], y_reg_train[salary_mask_train])
    salary_metrics = evaluate_regressor(salary_reg, X_test, y_reg_test, "Ridge Salary Regressor")
    print(f"  [+] Salary Regressor R2: {salary_metrics['r2_score']}, MAE: {salary_metrics['mae_lpa']} LPA")

    # 3. Module 3: Train Tree-Based Models
    print("\n[Stage 3/5] Training Module 3 Tree-Based Models...")

    # 3a. Decision Tree Classifier
    print("  -> Training Decision Tree (Cost Complexity Pruned)...")
    dt_clf = create_decision_tree_pipeline(max_depth=8, min_samples_leaf=15)
    dt_clf.fit(X_train, y_clf_train)
    dt_metrics = evaluate_classifier(dt_clf, X_test, y_clf_test, "Decision Tree (Pruned)")

    # 3b. Random Forest Classifier
    print("  -> Training Random Forest (Bagging + OOB Score)...")
    t0 = time.time()
    rf_clf = create_random_forest_pipeline(n_estimators=100, max_depth=12)
    rf_clf.fit(X_train, y_clf_train)
    rf_time = time.time() - t0
    rf_metrics = evaluate_classifier(rf_clf, X_test, y_clf_test, "Random Forest")
    rf_metrics["train_time_sec"] = round(rf_time, 3)
    rf_metrics["oob_score"] = round(float(rf_clf.named_steps['classifier'].oob_score_), 4)
    print(f"  [+] Random Forest OOB Score: {rf_metrics['oob_score']}")

    # 3c. LightGBM Classifier
    print("  -> Training LightGBM (Histogram-based Boosting)...")
    t0 = time.time()
    lgb_clf = create_lightgbm_pipeline(n_estimators=150, learning_rate=0.05)
    lgb_clf.fit(X_train, y_clf_train)
    lgb_time = time.time() - t0
    lgb_metrics = evaluate_classifier(lgb_clf, X_test, y_clf_test, "LightGBM")
    lgb_metrics["train_time_sec"] = round(lgb_time, 3)

    # 3d. XGBoost Classifier
    print("  -> Training XGBoost (Gradient Boosting)...")
    t0 = time.time()
    xgb_clf = create_xgboost_pipeline(n_estimators=150, learning_rate=0.05)
    xgb_clf.fit(X_train, y_clf_train)
    xgb_time = time.time() - t0
    xgb_metrics = evaluate_classifier(xgb_clf, X_test, y_clf_test, "XGBoost")
    xgb_metrics["train_time_sec"] = round(xgb_time, 3)

    # 4. Interpretability & Feature Importance Extraction
    print("\n[Stage 4/5] Extracting Interpretability Insights (Linear Weights & Tree MDI)...")
    linear_interpret = extract_linear_interpretability(ridge_clf)
    tree_interpret = extract_tree_feature_importance(lgb_clf)

    # 5. Packaging Artifacts
    print("\n[Stage 5/5] Packaging & Exporting Production Artifacts...")
    
    # Save Pipeline models
    joblib.dump(ridge_clf, os.path.join(artifacts_dir, "linear_pipeline.joblib"))
    joblib.dump(lasso_clf, os.path.join(artifacts_dir, "lasso_pipeline.joblib"))
    joblib.dump(elastic_clf, os.path.join(artifacts_dir, "elasticnet_pipeline.joblib"))
    joblib.dump(dt_clf, os.path.join(artifacts_dir, "dt_pipeline.joblib"))
    joblib.dump(rf_clf, os.path.join(artifacts_dir, "rf_pipeline.joblib"))
    joblib.dump(lgb_clf, os.path.join(artifacts_dir, "tree_pipeline.joblib"))
    joblib.dump(lgb_clf, os.path.join(artifacts_dir, "lgbm_pipeline.joblib"))
    joblib.dump(xgb_clf, os.path.join(artifacts_dir, "xgboost_pipeline.joblib"))
    joblib.dump(salary_reg, os.path.join(artifacts_dir, "salary_regressor.joblib"))
    
    # Aggregate Metrics & Benchmarks
    all_metrics = {
        "models": {
            "ridge_logistic": ridge_metrics,
            "lasso_logistic": lasso_metrics,
            "elasticnet_logistic": elastic_metrics,
            "decision_tree": dt_metrics,
            "random_forest": rf_metrics,
            "lightgbm": lgb_metrics,
            "xgboost": xgb_metrics
        },
        "salary_regressor": salary_metrics,
        "dataset_stats": stats
    }
    
    with open(os.path.join(artifacts_dir, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    with open(os.path.join(artifacts_dir, "linear_interpretability.json"), "w") as f:
        json.dump(linear_interpret, f, indent=2)

    with open(os.path.join(artifacts_dir, "feature_importance.json"), "w") as f:
        json.dump(tree_interpret, f, indent=2)
        
    # Also save training baseline distributions for drift detection
    baselines = {}
    for col in data["feature_types"]["numerical"]:
        baselines[col] = {
            "mean": float(X_train[col].mean(skipna=True)),
            "std": float(X_train[col].std(skipna=True)),
            "median": float(X_train[col].median(skipna=True))
        }
    with open(os.path.join(artifacts_dir, "baselines.json"), "w") as f:
        json.dump(baselines, f, indent=2)

    print("=" * 60)
    print("[SUCCESS] TRAINING COMPLETED & ALL ARTIFACTS PACKAGED!")
    print(f"[+] Linear Ridge Accuracy: {ridge_metrics['accuracy']*100:.2f}%, ROC-AUC: {ridge_metrics['roc_auc']:.4f}")
    print(f"[+] LightGBM Accuracy:     {lgb_metrics['accuracy']*100:.2f}%, ROC-AUC: {lgb_metrics['roc_auc']:.4f}")
    print(f"[+] XGBoost Accuracy:      {xgb_metrics['accuracy']*100:.2f}%, ROC-AUC: {xgb_metrics['roc_auc']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    run_full_training_lifecycle()
