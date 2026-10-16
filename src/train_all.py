"""
PlacementPredict - Complete ML Lifecycle Orchestrator (Modules 1 through 6)
Executes:
- M1: Ingestion, cleaning, leakage-free 70/15/15 stratified partitioning
- M2: Linear models (Ridge L2, Lasso L1, ElasticNet, Ridge Salary Regressor)
- M3: Tree-based ensembles (Decision Tree with CCP, Random Forest OOB, LightGBM, XGBoost)
- M4: Unsupervised learning (K-Means Elbow/Silhouette, Student Personas, PCA, DBSCAN, Isolation Forest)
- M5: Calibration & Evaluation (Platt scaling, Isotonic regression, Brier score, Stratified & Nested CV, McNemar's test, Paired bootstrap)
- M6: Production ML Engineering (Model manifest, checksums, drift baselines, feature store contracts)
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

from src.data_loader import prepare_datasets, load_raw_data
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
from src.unsupervised_models import (
    fit_student_persona_clusters,
    compute_kmeans_elbow_and_silhouette,
    perform_pca_decomposition,
    perform_dbscan_analysis,
    perform_anomaly_detection_isolation_forest,
    ClusterFeatureAugmenter
)
from src.calibration_and_evaluation import (
    compute_probability_calibration,
    perform_mcnemar_test,
    compute_stratified_and_nested_cv,
    compute_threshold_tuning_curve,
    compute_paired_bootstrap_regression
)
from src.mlops_production import (
    ProductionDriftDetector,
    generate_production_manifest
)


def run_full_training_lifecycle(artifacts_dir: str = "artifacts"):
    """Runs data ingestion, preprocessing, multi-model training, evaluation, and artifact export across M1-M6."""
    os.makedirs(artifacts_dir, exist_ok=True)
    print("=" * 70)
    print("⚡ STARTING PLACEMENTPREDICT PRODUCTION ML LIFECYCLE (M1 - M6)")
    print("=" * 70)

    # ---------------------------------------------------------
    # STAGE 1: M1 - Ingestion & Splitting
    # ---------------------------------------------------------
    print("\n[Stage 1/7] Ingesting & Partitioning 50k Dataset (M1)...")
    data = prepare_datasets(filter_anomalies=True)
    X_train, y_clf_train, y_reg_train = data["X_train"], data["y_clf_train"], data["y_reg_train"]
    X_val, y_clf_val, y_reg_val = data["X_val"], data["y_clf_val"], data["y_reg_val"]
    X_test, y_clf_test, y_reg_test = data["X_test"], data["y_clf_test"], data["y_reg_test"]
    stats = data["stats"]
    print(f"  [+] Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")
    print(f"  [+] Placement Rate: {stats['placement_rate']*100:.1f}%, Anomalies Filtered: {stats['anomalies_count']}")

    # ---------------------------------------------------------
    # STAGE 2: M2 - Linear Models at Depth
    # ---------------------------------------------------------
    print("\n[Stage 2/7] Training Linear Models at Depth (M2)...")
    
    # 2a. Ridge Logistic Regression (L2)
    print("  -> Fitting Ridge (L2) Logistic Regression...")
    t0 = time.time()
    ridge_clf = create_logistic_pipeline(penalty='l2', C=1.0)
    ridge_clf.fit(X_train, y_clf_train)
    ridge_time = time.time() - t0
    ridge_metrics = evaluate_classifier(ridge_clf, X_test, y_clf_test, "Ridge (L2) Logistic Regression")
    ridge_metrics["train_time_sec"] = round(ridge_time, 3)

    # 2b. Lasso Logistic Regression (L1)
    print("  -> Fitting Lasso (L1) Logistic Regression (Sparsity check)...")
    lasso_clf = create_logistic_pipeline(penalty='l1', C=0.5)
    lasso_clf.fit(X_train, y_clf_train)
    lasso_metrics = evaluate_classifier(lasso_clf, X_test, y_clf_test, "Lasso (L1) Logistic Regression")

    # 2c. ElasticNet Logistic Regression
    print("  -> Fitting ElasticNet Logistic Regression...")
    elastic_clf = create_logistic_pipeline(penalty='elasticnet', C=0.5, l1_ratio=0.5)
    elastic_clf.fit(X_train, y_clf_train)
    elastic_metrics = evaluate_classifier(elastic_clf, X_test, y_clf_test, "ElasticNet Logistic Regression")

    # 2d. Ridge Salary Regressor (for continuous CTC in LPA)
    print("  -> Fitting Ridge Salary Regressor...")
    salary_mask_train = (y_clf_train == 1) & (y_reg_train > 0)
    salary_reg = create_salary_regressor_pipeline(alpha=1.0)
    salary_reg.fit(X_train[salary_mask_train], y_reg_train[salary_mask_train])
    salary_metrics = evaluate_regressor(salary_reg, X_test, y_reg_test, "Ridge Salary Regressor")
    print(f"  [+] Salary Regressor R2: {salary_metrics['r2_score']}, MAE: ₹{salary_metrics['mae_lpa']} LPA")

    # ---------------------------------------------------------
    # STAGE 3: M3 - Tree-Based Models & Ensembles
    # ---------------------------------------------------------
    print("\n[Stage 3/7] Training Tree-Based Models & Ensembles (M3)...")

    # 3a. Decision Tree Classifier (Pruned)
    print("  -> Fitting Decision Tree (Cost Complexity Pruned)...")
    dt_clf = create_decision_tree_pipeline(max_depth=8, min_samples_leaf=15)
    dt_clf.fit(X_train, y_clf_train)
    dt_metrics = evaluate_classifier(dt_clf, X_test, y_clf_test, "Decision Tree (Pruned)")

    # 3b. Random Forest Classifier
    print("  -> Fitting Random Forest (Bagging + OOB Score)...")
    t0 = time.time()
    rf_clf = create_random_forest_pipeline(n_estimators=100, max_depth=12)
    rf_clf.fit(X_train, y_clf_train)
    rf_time = time.time() - t0
    rf_metrics = evaluate_classifier(rf_clf, X_test, y_clf_test, "Random Forest")
    rf_metrics["train_time_sec"] = round(rf_time, 3)
    rf_metrics["oob_score"] = round(float(rf_clf.named_steps['classifier'].oob_score_), 4)
    print(f"  [+] Random Forest OOB Score: {rf_metrics['oob_score']*100:.2f}%")

    # 3c. LightGBM Classifier
    print("  -> Fitting LightGBM (Histogram-based Boosting)...")
    t0 = time.time()
    lgb_clf = create_lightgbm_pipeline(n_estimators=150, learning_rate=0.05)
    lgb_clf.fit(X_train, y_clf_train)
    lgb_time = time.time() - t0
    lgb_metrics = evaluate_classifier(lgb_clf, X_test, y_clf_test, "LightGBM")
    lgb_metrics["train_time_sec"] = round(lgb_time, 3)

    # 3d. XGBoost Classifier
    print("  -> Fitting XGBoost (Gradient Boosting)...")
    t0 = time.time()
    xgb_clf = create_xgboost_pipeline(n_estimators=150, learning_rate=0.05)
    xgb_clf.fit(X_train, y_clf_train)
    xgb_time = time.time() - t0
    xgb_metrics = evaluate_classifier(xgb_clf, X_test, y_clf_test, "XGBoost")
    xgb_metrics["train_time_sec"] = round(xgb_time, 3)

    # ---------------------------------------------------------
    # STAGE 4: M4 - Unsupervised Learning & Clustering
    # ---------------------------------------------------------
    print("\n[Stage 4/7] Running Unsupervised Learning Workflows (M4)...")
    raw_df = load_raw_data()
    clean_df = raw_df[raw_df['IsAnomaly'] == 0] if 'IsAnomaly' in raw_df.columns else raw_df

    # 4a. K-Means Elbow & Silhouette Analysis
    print("  -> Evaluating K-Means Elbow & Silhouette Across K=2..8...")
    kmeans_eval = compute_kmeans_elbow_and_silhouette(clean_df)

    # 4b. Student Persona Clustering (K=4)
    print("  -> Discovering Student Persona Clusters (K=4)...")
    persona_clusters = fit_student_persona_clusters(clean_df, n_clusters=4)

    # 4c. PCA Decomposition (Explained Variance & Biplot)
    print("  -> Computing PCA Dimensionality Reduction & Standardization Pitfalls...")
    pca_results = perform_pca_decomposition(clean_df, n_components=6)

    # 4d. DBSCAN Clustering & Outlier Extraction
    print("  -> Computing DBSCAN Density-Based Cluster Separation...")
    dbscan_results = perform_dbscan_analysis(clean_df, eps=1.2, min_samples=15)

    # 4e. Isolation Forest Anomaly Detection
    print("  -> Computing Isolation Forest Anomaly Scores...")
    iso_results = perform_anomaly_detection_isolation_forest(clean_df, contamination=0.03)

    # ---------------------------------------------------------
    # STAGE 5: M5 - Calibration, Selection & Hypothesis Testing
    # ---------------------------------------------------------
    print("\n[Stage 5/7] Model Evaluation, Selection, and Calibration (M5)...")

    # 5a. Probability Calibration (Platt Sigmoid vs Isotonic Regression)
    print("  -> Fitting Probability Calibrators (Platt Scaling vs Isotonic)...")
    calibration_results = compute_probability_calibration(
        base_pipeline=lgb_clf,
        X_train=X_train,
        y_train=y_clf_train,
        X_test=X_test,
        y_test=y_clf_test,
        n_bins=10
    )
    print(f"  [+] Uncalibrated Brier Score: {calibration_results['brier_scores']['uncalibrated']} -> Platt Brier: {calibration_results['brier_scores']['platt_sigmoid']}")

    # 5b. Statistical Significance: McNemar's Test for Classifiers
    print("  -> Running McNemar's Hypothesis Test (Ridge Logistic vs LightGBM)...")
    ridge_preds = ridge_clf.predict(X_test)
    lgb_preds = lgb_clf.predict(X_test)
    mcnemar_results = perform_mcnemar_test(
        ridge_preds, lgb_preds, y_clf_test.values,
        model_a_name="Ridge Logistic (M2)",
        model_b_name="LightGBM (M3)"
    )
    print(f"  [+] McNemar Chi2: {mcnemar_results['chi2_statistic']}, p-value: {mcnemar_results['p_value']} (Significant: {mcnemar_results['is_statistically_significant']})")

    # 5c. Stratified K-Fold & Nested Cross-Validation
    print("  -> Running Stratified 5-Fold & Nested CV Generalization Analysis...")
    cv_results = compute_stratified_and_nested_cv(ridge_clf, X_train, y_clf_train, n_splits=5)
    print(f"  [+] Stratified 5-Fold Mean Accuracy: {cv_results['stratified_cv_mean_accuracy']}% (+/- {cv_results['stratified_cv_std_accuracy']}%)")

    # 5d. Threshold Tuning Diagnostics
    print("  -> Evaluating Decision Threshold Tuning Curve (0.05 - 0.95)...")
    threshold_results = compute_threshold_tuning_curve(lgb_clf, X_test, y_clf_test)

    # 5e. Paired Bootstrap for Salary Regressor
    print("  -> Running Paired Bootstrap Test for Salary Regression...")
    dt_salary_pred = salary_reg.predict(X_test)
    paired_bootstrap_results = compute_paired_bootstrap_regression(
        y_clf_test.values, dt_salary_pred, dt_salary_pred, n_bootstraps=500
    )

    # ---------------------------------------------------------
    # STAGE 6: M6 - ML Engineering & Production MLOps
    # ---------------------------------------------------------
    print("\n[Stage 6/7] Production MLOps, Packaging & Drift Baselines (M6)...")

    # 6a. Extract Feature Drift Baselines & Quantiles
    drift_detector = ProductionDriftDetector()
    drift_detector.fit_baselines(X_train)
    drift_report = drift_detector.evaluate_drift_report(X_test)
    print(f"  [+] Production Drift Engine Status: {drift_report['system_status']}")

    # ---------------------------------------------------------
    # STAGE 7: Serializing Artifacts & Manifest
    # ---------------------------------------------------------
    print("\n[Stage 7/7] Serializing Artifacts & Generating Production Manifest...")

    # Save fitted pipelines
    joblib.dump(ridge_clf, os.path.join(artifacts_dir, "linear_pipeline.joblib"))
    joblib.dump(lasso_clf, os.path.join(artifacts_dir, "lasso_pipeline.joblib"))
    joblib.dump(elastic_clf, os.path.join(artifacts_dir, "elasticnet_pipeline.joblib"))
    joblib.dump(dt_clf, os.path.join(artifacts_dir, "dt_pipeline.joblib"))
    joblib.dump(rf_clf, os.path.join(artifacts_dir, "rf_pipeline.joblib"))
    joblib.dump(lgb_clf, os.path.join(artifacts_dir, "tree_pipeline.joblib"))
    joblib.dump(lgb_clf, os.path.join(artifacts_dir, "lgbm_pipeline.joblib"))
    joblib.dump(xgb_clf, os.path.join(artifacts_dir, "xgboost_pipeline.joblib"))
    joblib.dump(salary_reg, os.path.join(artifacts_dir, "salary_regressor.joblib"))

    # Interpretability
    linear_interpret = extract_linear_interpretability(ridge_clf)
    tree_interpret = extract_tree_feature_importance(lgb_clf)

    with open(os.path.join(artifacts_dir, "linear_interpretability.json"), "w", encoding="utf-8") as f:
        json.dump(linear_interpret, f, indent=2)

    with open(os.path.join(artifacts_dir, "feature_importance.json"), "w", encoding="utf-8") as f:
        json.dump(tree_interpret, f, indent=2)

    # M4, M5, M6 JSON artifacts
    with open(os.path.join(artifacts_dir, "unsupervised_clustering.json"), "w", encoding="utf-8") as f:
        json.dump({
            "kmeans_elbow": kmeans_eval,
            "personas": persona_clusters,
            "pca": pca_results,
            "dbscan": dbscan_results,
            "isolation_forest": iso_results
        }, f, indent=2)

    with open(os.path.join(artifacts_dir, "calibration_and_evaluation.json"), "w", encoding="utf-8") as f:
        json.dump({
            "calibration": calibration_results,
            "mcnemar": mcnemar_results,
            "cross_validation": cv_results,
            "threshold_tuning": threshold_results,
            "paired_bootstrap": paired_bootstrap_results
        }, f, indent=2)

    # Save aggregate metrics
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
    with open(os.path.join(artifacts_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    # Baseline numerical statistics for live drift comparison
    baselines = {}
    for col in data["feature_types"]["numerical"]:
        baselines[col] = {
            "mean": float(X_train[col].mean(skipna=True)),
            "std": float(X_train[col].std(skipna=True)),
            "median": float(X_train[col].median(skipna=True))
        }
    with open(os.path.join(artifacts_dir, "baselines.json"), "w", encoding="utf-8") as f:
        json.dump(baselines, f, indent=2)

    # Generate MLOps Production Manifest
    generate_production_manifest(artifacts_dir, all_metrics)

    print("=" * 70)
    print("[SUCCESS] ALL ML LIFECYCLE STAGES (M1-M6) COMPLETED & PACKAGED!")
    print(f"  [+] Champion LightGBM Accuracy: {lgb_metrics['accuracy']*100:.2f}% | ROC-AUC: {lgb_metrics['roc_auc']:.4f}")
    print(f"  [+] Linear Ridge Accuracy:      {ridge_metrics['accuracy']*100:.2f}% | ROC-AUC: {ridge_metrics['roc_auc']:.4f}")
    print(f"  [+] Calibrated Brier Score:     {calibration_results['brier_scores']['platt_sigmoid']}")
    print(f"  [+] Production Manifest:        artifacts/model_manifest.json")
    print("=" * 70)


if __name__ == "__main__":
    run_full_training_lifecycle()
