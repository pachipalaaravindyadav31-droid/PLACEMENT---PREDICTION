"""
PlacementPredict - Comprehensive End-to-End System Tests
Tests M1 (FastAPI endpoints), M2 (Linear pipeline inference), and M3 (Tree pipeline inference).
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import app, PRESET_PERSONAS


client = TestClient(app)


def test_dashboard_endpoint():
    """Verify HTML dashboard renders successfully."""
    response = client.get("/")
    assert response.status_code == 200
    assert "PlacementPredict" in response.text
    assert "Candidate Academic & Skill Profile" in response.text


def test_presets_endpoint():
    """Verify presets API returns expected student archetypes."""
    response = client.get("/api/presets")
    assert response.status_code == 200
    data = response.json()
    assert "top_performer" in data
    assert "at_risk" in data


def test_models_metadata_endpoint():
    """Verify models metrics registry is populated."""
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert "ridge_logistic" in data["models"]
    assert "lightgbm" in data["models"]
    assert data["models"]["ridge_logistic"]["accuracy"] > 0.85


def test_linear_single_prediction():
    """Verify M2 Linear (Ridge Logistic) model prediction."""
    top_student = PRESET_PERSONAS["top_performer"]["data"].copy()
    top_student["ModelType"] = "linear"
    
    response = client.post("/api/predict", json=top_student)
    assert response.status_code == 200
    res = response.json()
    assert res["status"] == "success"
    assert res["prediction"]["is_placed"] is True
    assert res["prediction"]["placement_probability_pct"] > 80.0
    assert res["prediction"]["expected_salary_lpa"] > 10.0


def test_tree_single_prediction_at_risk():
    """Verify M3 Tree (LightGBM) model prediction for at-risk student."""
    at_risk_student = PRESET_PERSONAS["at_risk"]["data"].copy()
    at_risk_student["ModelType"] = "tree"
    
    response = client.post("/api/predict", json=at_risk_student)
    assert response.status_code == 200
    res = response.json()
    assert res["status"] == "success"
    assert res["prediction"]["is_placed"] is False
    assert res["prediction"]["placement_probability_pct"] < 50.0


def test_batch_prediction_endpoint():
    """Verify batch prediction endpoint evaluates multiple students."""
    batch_data = [
        PRESET_PERSONAS["top_performer"]["data"],
        PRESET_PERSONAS["at_risk"]["data"],
        PRESET_PERSONAS["coding_ninja"]["data"]
    ]
    response = client.post("/api/predict_batch", json=batch_data)
    assert response.status_code == 200
    res = response.json()
    assert res["total_evaluated"] == 3
    assert res["total_placed"] >= 2


def test_monitoring_telemetry():
    """Verify monitoring metrics are tracked."""
    response = client.get("/api/metrics")
    assert response.status_code == 200
    res = response.json()
    assert res["total_requests"] >= 1
    assert "latency_ms" in res


def test_graphs_endpoint():
    """Verify /api/graphs endpoint returns complete curve coordinates."""
    response = client.get("/api/graphs")
    assert response.status_code == 200
    data = response.json()
    assert "roc_curves" in data
    assert "sigmoid" in data
    assert "tree_split" in data
    assert "regularization_geometry" in data
    assert "cgpa_dist" in data


def test_dataset_sample_endpoint():
    """Verify /api/dataset endpoint returns paginated sample records."""
    response = client.get("/api/dataset?page=1&page_size=15&stream=CS")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["page"] == 1
    assert data["page_size"] == 15
    assert len(data["records"]) == 15
    assert data["records"][0]["Stream"] == "CS"


def test_xgboost_prediction():
    """Verify XGBoost classification prediction endpoint."""
    top_student = PRESET_PERSONAS["top_performer"]["data"].copy()
    top_student["ModelType"] = "xgboost"
    response = client.post("/api/predict", json=top_student)
    assert response.status_code == 200
    res = response.json()
    assert res["status"] == "success"
    assert "XGBoost" in res["model_used"]
    assert res["prediction"]["is_placed"] is True


def test_all_models_options():
    """Verify inference across all 7 model keys."""
    for model_key in ["xgboost", "lightgbm", "rf", "dt", "linear", "lasso", "elasticnet"]:
        profile = PRESET_PERSONAS["top_performer"]["data"].copy()
        profile["ModelType"] = model_key
        response = client.post("/api/predict", json=profile)
        assert response.status_code == 200
        res = response.json()
        assert res["status"] == "success"
        assert res["prediction"]["placement_probability_pct"] > 70.0


def test_predict_all_models_consensus():
    """Verify /api/predict_all endpoint runs all 7 models simultaneously."""
    profile = PRESET_PERSONAS["top_performer"]["data"].copy()
    response = client.post("/api/predict_all", json=profile)
    assert response.status_code == 200
    res = response.json()
    assert res["status"] == "success"
    assert "consensus" in res
    assert res["consensus"]["is_placed"] is True
    model_keys = [m["key"] for m in res["models"]]
    assert "xgboost" in model_keys
    assert "lightgbm" in model_keys
    assert "rf" in model_keys
    assert "dt" in model_keys


def test_analyze_sample_dataset():
    """Verify sample dataset EDA analyzer endpoint."""
    response = client.get("/api/analyze_sample?sample_name=placement_50k&target_column=PlacementStatus")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["dataset_summary"]["total_rows"] == 50000
    assert len(data["columns"]) >= 15
    assert "target_analysis" in data
    assert data["target_analysis"]["name"] == "PlacementStatus"


def test_analyze_raw_pasted_csv():
    """Verify raw text CSV analyzer endpoint."""
    csv_text = "Age,Income,Experience,Churn\n25,50000,2,0\n35,80000,8,1\n45,120000,15,0\n22,40000,1,1"
    response = client.post("/api/analyze_raw_data", json={"csv_text": csv_text, "target_column": "Churn"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["dataset_summary"]["total_rows"] == 4
    assert data["dataset_summary"]["total_columns"] == 4
    assert data["target_analysis"]["name"] == "Churn"


def test_analyze_upload_csv():
    """Verify file upload analysis endpoint for arbitrary CSV files."""
    csv_content = b"Candidate,CGPA,CodingScore,Salary,Placed\nAlice,9.2,95,15.5,1\nBob,6.1,45,0,0\nCharlie,8.0,78,11.0,1"
    response = client.post(
        "/api/analyze_upload",
        files={"file": ("test_candidates.csv", csv_content, "text/csv")},
        data={"target_column": "Placed"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["dataset_summary"]["total_rows"] == 3
    assert data["target_analysis"]["name"] == "Placed"


def test_m4_unsupervised_clusters_endpoint():
    """Verify M4 unsupervised clusters API returns K-Means personas and PCA coordinates."""
    response = client.get("/api/unsupervised/clusters")
    assert response.status_code == 200
    data = response.json()
    assert "kmeans_elbow" in data
    assert "personas" in data
    assert "pca" in data
    assert len(data["personas"]["profiles"]) == 4
    assert "Elite All-Rounders" in [p["persona_name"] for p in data["personas"]["profiles"]]


def test_m5_calibration_endpoint():
    """Verify M5 probability calibration API returns reliability curves and Brier scores."""
    response = client.get("/api/calibration")
    assert response.status_code == 200
    data = response.json()
    assert "brier_scores" in data
    assert "reliability_curves" in data
    assert "platt_sigmoid" in data["brier_scores"]
    assert "isotonic" in data["brier_scores"]
    assert data["brier_scores"]["uncalibrated"] > 0


def test_m5_mcnemar_evaluation_endpoint():
    """Verify M5 McNemar's hypothesis testing endpoint."""
    response = client.get("/api/evaluation/mcnemar")
    assert response.status_code == 200
    data = response.json()
    assert "contingency_table" in data
    assert "chi2_statistic" in data
    assert "p_value" in data
    assert "is_statistically_significant" in data


def test_m5_cross_validation_endpoint():
    """Verify M5 Stratified 5-Fold and Nested CV endpoint."""
    response = client.get("/api/evaluation/cross_validation")
    assert response.status_code == 200
    data = response.json()
    assert "stratified_cv_accuracy_folds" in data
    assert len(data["stratified_cv_accuracy_folds"]) == 5
    assert data["stratified_cv_mean_accuracy"] > 88.0


def test_m6_mlops_manifest_endpoint():
    """Verify M6 production model registry manifest endpoint."""
    response = client.get("/api/mlops/manifest")
    assert response.status_code == 200
    data = response.json()
    assert "service_name" in data
    assert "model_registry" in data
    assert "feature_store_contract" in data
    assert "sla_contracts" in data
    assert data["sla_contracts"]["p95_latency_target_ms"] == 25.0


def test_m6_mlops_drift_endpoint():
    """Verify M6 production drift detection endpoint evaluates PSI and KS stats."""
    response = client.get("/api/mlops/drift")
    assert response.status_code == 200
    data = response.json()
    assert "system_status" in data
    assert "drift_details" in data
    assert "CGPA" in data["drift_details"]
    assert "psi_value" in data["drift_details"]["CGPA"]


def test_m6_retraining_trigger_endpoint():
    """Verify M6 automated retraining trigger endpoint."""
    response = client.post("/api/mlops/trigger_retrain")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert "stages" in data
    assert len(data["stages"]) == 6
    assert data["decision"] == "PROMOTED_NEW_CHAMPION"


def test_single_prediction_includes_persona_and_calibration():
    """Verify prediction includes M4 student persona and M5 calibrated probability."""
    top_student = PRESET_PERSONAS["top_performer"]["data"].copy()
    response = client.post("/api/predict", json=top_student)
    assert response.status_code == 200
    res = response.json()
    pred = res["prediction"]
    assert "calibrated_probability_pct" in pred
    assert "student_persona" in pred
    assert pred["student_persona"]["name"] == "Elite All-Rounders"

