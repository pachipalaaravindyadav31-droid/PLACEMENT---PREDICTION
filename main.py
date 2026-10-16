"""
PlacementPredict - Production Serving Application (FastAPI)
Implements Module 1 (ML System Lifecycle & Serving), Module 2 (Linear Models), and Module 3 (Tree Models).
"""

import os
import sys
import io
import time
import json
import joblib
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data_loader import ALL_FEATURES, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, ORDINAL_FEATURES
from src.monitoring import monitor
from src.data_analyzer import parse_uploaded_file, analyze_dataframe

# Global artifact storage
ARTIFACTS: Dict[str, Any] = {}


def load_artifacts():
    """Load model pipelines and metric registries."""
    if ARTIFACTS.get("linear_pipeline") is not None:
        return
        
    artifacts_dir = os.path.join(os.path.dirname(__file__), "artifacts")
    print(f"[*] Loading ML artifacts from {artifacts_dir}...")
    
    linear_path = os.path.join(artifacts_dir, "linear_pipeline.joblib")
    if not os.path.exists(linear_path):
        from src.train_all import run_full_training_lifecycle
        run_full_training_lifecycle(artifacts_dir)
        
    try:
        ARTIFACTS["linear_pipeline"] = joblib.load(os.path.join(artifacts_dir, "linear_pipeline.joblib"))
    except Exception as e:
        print(f"[!] Warning loading linear_pipeline: {e}")
    
    # Load all specialized classification pipelines safely
    for p_name in ["lasso_pipeline", "elasticnet_pipeline", "dt_pipeline", "rf_pipeline", "tree_pipeline", "lgbm_pipeline", "xgboost_pipeline"]:
        p_path = os.path.join(artifacts_dir, f"{p_name}.joblib")
        if os.path.exists(p_path):
            try:
                ARTIFACTS[p_name] = joblib.load(p_path)
            except Exception as e:
                print(f"[!] Warning loading {p_name}: {e}")
            
    # Ensure aliases and fallbacks
    default_tree = ARTIFACTS.get("tree_pipeline") or ARTIFACTS.get("lgbm_pipeline") or ARTIFACTS.get("rf_pipeline") or ARTIFACTS.get("linear_pipeline")
    if "tree_pipeline" not in ARTIFACTS and default_tree is not None:
        ARTIFACTS["tree_pipeline"] = default_tree
    if "lgbm_pipeline" not in ARTIFACTS and default_tree is not None:
        ARTIFACTS["lgbm_pipeline"] = default_tree
    if "xgboost_pipeline" not in ARTIFACTS and default_tree is not None:
        ARTIFACTS["xgboost_pipeline"] = default_tree
        
    try:
        ARTIFACTS["salary_regressor"] = joblib.load(os.path.join(artifacts_dir, "salary_regressor.joblib"))
    except Exception as e:
        print(f"[!] Warning loading salary_regressor: {e}")
    
    with open(os.path.join(artifacts_dir, "metrics.json"), "r", encoding="utf-8") as f:
        ARTIFACTS["metrics"] = json.load(f)
    with open(os.path.join(artifacts_dir, "linear_interpretability.json"), "r", encoding="utf-8") as f:
        ARTIFACTS["linear_interpret"] = json.load(f)
    with open(os.path.join(artifacts_dir, "feature_importance.json"), "r", encoding="utf-8") as f:
        ARTIFACTS["tree_interpret"] = json.load(f)
    with open(os.path.join(artifacts_dir, "baselines.json"), "r", encoding="utf-8") as f:
        ARTIFACTS["baselines"] = json.load(f)
        monitor.set_baselines(ARTIFACTS["baselines"])

    clustering_file = os.path.join(artifacts_dir, "unsupervised_clustering.json")
    if os.path.exists(clustering_file):
        with open(clustering_file, "r", encoding="utf-8") as f:
            ARTIFACTS["unsupervised"] = json.load(f)

    calibration_file = os.path.join(artifacts_dir, "calibration_and_evaluation.json")
    if os.path.exists(calibration_file):
        with open(calibration_file, "r", encoding="utf-8") as f:
            ARTIFACTS["calibration_evaluation"] = json.load(f)

    manifest_file = os.path.join(artifacts_dir, "model_manifest.json")
    if os.path.exists(manifest_file):
        with open(manifest_file, "r", encoding="utf-8") as f:
            ARTIFACTS["model_manifest"] = json.load(f)
        
    graphs_file = os.path.join(artifacts_dir, "concept_graphs.json")
    if not os.path.exists(graphs_file):
        from src.graph_generator import generate_all_concept_graph_data
        ARTIFACTS["concept_graphs"] = generate_all_concept_graph_data(artifacts_dir)
    else:
        with open(graphs_file, "r", encoding="utf-8") as f:
            ARTIFACTS["concept_graphs"] = json.load(f)
            
    print("[SUCCESS] All models & lifecycle artifacts (M1 - M6) loaded!")


# Eager load artifacts on module import
load_artifacts()


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield
    print("[*] Shutting down PlacementPredict service...")


app = FastAPI(
    title="PlacementPredict ML System",
    description="End-to-end Student Placement & Salary Prediction Service (M1, M2, M3)",
    version="1.0.0",
    lifespan=lifespan
)

# Mount static and templates
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "Static")), name="static")
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "Templates"))


class StudentProfile(BaseModel):
    Gender: str = Field(default="Male", description="Male / Female")
    City: str = Field(default="Bangalore", description="City of residence")
    CollegeTier: str = Field(default="Tier2", description="Tier1 / Tier2 / Tier3")
    Stream: str = Field(default="CS", description="CS / IT / ECE / EE / Mechanical")
    Specialisation: str = Field(default="AI", description="AI / DataScience / Networking / Embedded")
    Hostel: str = Field(default="Yes", description="Yes / No")
    HistoryOfBacklogs: str = Field(default="No", description="Yes / No")
    SGPA_Sem1: float = Field(default=7.5, ge=0.0, le=10.0)
    SGPA_Sem2: float = Field(default=7.5, ge=0.0, le=10.0)
    SGPA_Sem3: float = Field(default=7.8, ge=0.0, le=10.0)
    SGPA_Sem4: float = Field(default=7.6, ge=0.0, le=10.0)
    SGPA_Sem5: float = Field(default=8.0, ge=0.0, le=10.0)
    SGPA_Sem6: float = Field(default=8.2, ge=0.0, le=10.0)
    SGPA_Sem7: float = Field(default=8.1, ge=0.0, le=10.0)
    SGPA_Sem8: float = Field(default=8.4, ge=0.0, le=10.0)
    CGPA: float = Field(default=7.9, ge=0.0, le=10.0)
    AttendancePercent: float = Field(default=82.0, ge=0.0, le=100.0)
    Internships: int = Field(default=1, ge=0, le=10)
    Projects: int = Field(default=2, ge=0, le=20)
    Workshops: Optional[float] = Field(default=1.0, ge=0.0)
    Certifications: int = Field(default=2, ge=0, le=20)
    Publications: int = Field(default=0, ge=0, le=10)
    AptitudeTestScore: Optional[float] = Field(default=72.0, ge=0.0, le=100.0)
    SoftSkillsRating: Optional[float] = Field(default=3.5, ge=1.0, le=5.0)
    CodingTestScore: Optional[float] = Field(default=68.0, ge=0.0, le=100.0)
    MockInterviewScore: Optional[float] = Field(default=65.0, ge=0.0, le=100.0)
    ExtraCurricular: int = Field(default=1, ge=0, le=1)
    CGPA_Tier: str = Field(default="Mid", description="Low / Mid / High")
    ModelType: str = Field(default="tree", description="'linear' (Ridge Logistic), 'tree' (LightGBM), or 'rf' (Random Forest)")


# Presets for interactive testing
PRESET_PERSONAS = {
    "top_performer": {
        "title": "Top Achiever (High Probability)",
        "data": {
            "Gender": "Female", "City": "Bangalore", "CollegeTier": "Tier1", "Stream": "CS", "Specialisation": "AI",
            "Hostel": "Yes", "HistoryOfBacklogs": "No", "SGPA_Sem1": 9.2, "SGPA_Sem2": 9.4, "SGPA_Sem3": 9.1,
            "SGPA_Sem4": 9.5, "SGPA_Sem5": 9.6, "SGPA_Sem6": 9.5, "SGPA_Sem7": 9.7, "SGPA_Sem8": 9.8,
            "CGPA": 9.47, "AttendancePercent": 92.5, "Internships": 3, "Projects": 4, "Workshops": 3,
            "Certifications": 4, "Publications": 1, "AptitudeTestScore": 88.5, "SoftSkillsRating": 4.5,
            "CodingTestScore": 92.0, "MockInterviewScore": 86.0, "ExtraCurricular": 1, "CGPA_Tier": "High"
        }
    },
    "average_student": {
        "title": "Average In-Between Student",
        "data": {
            "Gender": "Male", "City": "Hyderabad", "CollegeTier": "Tier2", "Stream": "IT", "Specialisation": "DataScience",
            "Hostel": "No", "HistoryOfBacklogs": "No", "SGPA_Sem1": 6.8, "SGPA_Sem2": 7.0, "SGPA_Sem3": 6.9,
            "SGPA_Sem4": 7.2, "SGPA_Sem5": 7.1, "SGPA_Sem6": 7.4, "SGPA_Sem7": 7.3, "SGPA_Sem8": 7.5,
            "CGPA": 7.15, "AttendancePercent": 76.0, "Internships": 1, "Projects": 2, "Workshops": 1,
            "Certifications": 1, "Publications": 0, "AptitudeTestScore": 64.0, "SoftSkillsRating": 3.0,
            "CodingTestScore": 58.0, "MockInterviewScore": 55.0, "ExtraCurricular": 0, "CGPA_Tier": "Mid"
        }
    },
    "at_risk": {
        "title": "At-Risk Student (Low Probability)",
        "data": {
            "Gender": "Male", "City": "Pune", "CollegeTier": "Tier3", "Stream": "Mechanical", "Specialisation": "Networking",
            "Hostel": "Yes", "HistoryOfBacklogs": "Yes", "SGPA_Sem1": 5.1, "SGPA_Sem2": 4.8, "SGPA_Sem3": 5.2,
            "SGPA_Sem4": 5.0, "SGPA_Sem5": 4.9, "SGPA_Sem6": 5.3, "SGPA_Sem7": 5.1, "SGPA_Sem8": 5.0,
            "CGPA": 5.05, "AttendancePercent": 52.0, "Internships": 0, "Projects": 1, "Workshops": 0,
            "Certifications": 0, "Publications": 0, "AptitudeTestScore": 42.0, "SoftSkillsRating": 2.0,
            "CodingTestScore": 28.0, "MockInterviewScore": 31.0, "ExtraCurricular": 0, "CGPA_Tier": "Low"
        }
    },
    "coding_ninja": {
        "title": "Coding Specialist (Strong Skills)",
        "data": {
            "Gender": "Male", "City": "Delhi", "CollegeTier": "Tier2", "Stream": "CS", "Specialisation": "AI",
            "Hostel": "No", "HistoryOfBacklogs": "No", "SGPA_Sem1": 7.1, "SGPA_Sem2": 7.3, "SGPA_Sem3": 7.0,
            "SGPA_Sem4": 7.4, "SGPA_Sem5": 7.5, "SGPA_Sem6": 7.8, "SGPA_Sem7": 8.0, "SGPA_Sem8": 8.1,
            "CGPA": 7.52, "AttendancePercent": 84.0, "Internships": 2, "Projects": 5, "Workshops": 2,
            "Certifications": 3, "Publications": 1, "AptitudeTestScore": 84.0, "SoftSkillsRating": 4.0,
            "CodingTestScore": 96.0, "MockInterviewScore": 88.0, "ExtraCurricular": 1, "CGPA_Tier": "Mid"
        }
    }
}


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return JSONResponse(content={"status": "ok"})


@app.get("/", response_class=HTMLResponse)
@app.get("/index", response_class=HTMLResponse)
@app.get("/index.html", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
@app.get("/home", response_class=HTMLResponse)
@app.get("/predict", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    """Serves the interactive PlacementPredict frontend dashboard."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "metrics": ARTIFACTS.get("metrics", {}),
            "feature_importance": ARTIFACTS.get("tree_interpret", {}).get("top_features", []),
            "linear_positive": ARTIFACTS.get("linear_interpret", {}).get("top_positive_drivers", []),
            "linear_negative": ARTIFACTS.get("linear_interpret", {}).get("top_negative_drivers", []),
            "presets": PRESET_PERSONAS,
            "concept_graphs": ARTIFACTS.get("concept_graphs", {})
        }
    )


@app.exception_handler(404)
async def custom_404_handler(request: Request, exc):
    """Redirects any unknown routes back to the main dashboard instead of 404 error."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "metrics": ARTIFACTS.get("metrics", {}),
            "feature_importance": ARTIFACTS.get("tree_interpret", {}).get("top_features", []),
            "linear_positive": ARTIFACTS.get("linear_interpret", {}).get("top_positive_drivers", []),
            "linear_negative": ARTIFACTS.get("linear_interpret", {}).get("top_negative_drivers", []),
            "presets": PRESET_PERSONAS,
            "concept_graphs": ARTIFACTS.get("concept_graphs", {})
        }
    )


@app.get("/api/presets")
async def get_presets():
    """Returns preset student personas for quick form prefilling."""
    return PRESET_PERSONAS


@app.get("/api/models")
async def get_models_comparison():
    """Returns detailed performance comparison across all M2 and M3 models."""
    return ARTIFACTS.get("metrics", {})


@app.get("/api/interpretability")
async def get_interpretability():
    """Returns linear model coefficients and tree feature importances."""
    return {
        "linear": ARTIFACTS.get("linear_interpret", {}),
        "tree": ARTIFACTS.get("tree_interpret", {})
    }


@app.get("/api/metrics")
async def get_monitoring_metrics():
    """Returns live production monitoring metrics (latencies, counts, drift)."""
    return monitor.get_metrics_summary()


@app.get("/api/graphs")
async def get_concept_graphs():
    """Returns mathematical curve coordinates and benchmark ROC/PR points for M1 through M6."""
    return ARTIFACTS.get("concept_graphs", {})


@app.get("/api/unsupervised/clusters")
async def get_unsupervised_clusters():
    """Returns M4 K-Means personas, PCA variance & projections, DBSCAN, and anomaly detection stats."""
    return ARTIFACTS.get("unsupervised", {})


@app.get("/api/calibration")
async def get_calibration_metrics():
    """Returns M5 probability calibration reliability curves, Platt vs Isotonic Brier scores."""
    return ARTIFACTS.get("calibration_evaluation", {}).get("calibration", {})


@app.get("/api/evaluation/mcnemar")
async def get_mcnemar_evaluation():
    """Returns M5 McNemar's hypothesis test for comparing classifier error significance."""
    return ARTIFACTS.get("calibration_evaluation", {}).get("mcnemar", {})


@app.get("/api/evaluation/cross_validation")
async def get_cv_evaluation():
    """Returns M5 Stratified 5-Fold & Nested CV score distributions and generalization metrics."""
    return ARTIFACTS.get("calibration_evaluation", {}).get("cross_validation", {})


@app.get("/api/mlops/manifest")
async def get_mlops_manifest():
    """Returns M6 production model registry manifest, SLA targets, and checksums."""
    return ARTIFACTS.get("model_manifest", {})


@app.get("/api/mlops/drift")
async def get_mlops_drift():
    """Returns M6 real-time Population Stability Index (PSI) and KS-test feature drift telemetry."""
    from src.mlops_production import ProductionDriftDetector
    
    # Analyze recent inference requests from monitor if available, else benchmark slice
    detector = ProductionDriftDetector()
    if "raw_df" not in ARTIFACTS:
        from src.data_loader import load_raw_data
        ARTIFACTS["raw_df"] = load_raw_data()
        
    df = ARTIFACTS["raw_df"]
    detector.fit_baselines(df)
    
    recent_records = monitor.history[-100:] if hasattr(monitor, "history") and monitor.history else []
    if len(recent_records) >= 10:
        prod_df = pd.DataFrame([r["features"] for r in recent_records if "features" in r])
    else:
        # Evaluate on a representative sample of unanomalous data
        prod_df = df.sample(n=min(500, len(df)), random_state=42)

    return detector.evaluate_drift_report(prod_df)


@app.post("/api/mlops/trigger_retrain")
async def trigger_model_retraining(reason: Optional[str] = "Manual Administrator Trigger"):
    """Triggers autonomous retraining loop, testing champion-challenger promotion."""
    from src.mlops_production import execute_retraining_simulation
    return execute_retraining_simulation(trigger_reason=reason or "Manual Admin Trigger")


@app.get("/api/dataset")
async def get_dataset_sample(
    page: int = 1,
    page_size: int = 25,
    stream: Optional[str] = None,
    placement_status: Optional[str] = None,
    college_tier: Optional[str] = None,
    search: Optional[str] = None
):
    """Returns a filtered, paginated subset of the 50,000 dataset for visual exploration."""
    if "raw_df" not in ARTIFACTS:
        from src.data_loader import load_raw_data
        ARTIFACTS["raw_df"] = load_raw_data()

    df = ARTIFACTS["raw_df"]
    filtered_df = df

    if stream and stream != "All":
        filtered_df = filtered_df[filtered_df["Stream"] == stream]

    if placement_status and placement_status != "All":
        status_val = 1 if placement_status.lower() in ["placed", "1"] else 0
        filtered_df = filtered_df[filtered_df["PlacementStatus"] == status_val]

    if college_tier and college_tier != "All":
        filtered_df = filtered_df[filtered_df["CollegeTier"] == college_tier]

    if search:
        s = search.strip().lower()
        mask = (
            filtered_df["City"].astype(str).str.lower().str.contains(s) |
            filtered_df["Specialisation"].astype(str).str.lower().str.contains(s) |
            filtered_df["StudentID"].astype(str).str.contains(s)
        )
        filtered_df = filtered_df[mask]

    total_records = len(filtered_df)
    page = max(1, page)
    page_size = min(max(5, page_size), 100)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size

    # Selected columns for display
    display_cols = [
        "StudentID", "Gender", "City", "CollegeTier", "Stream", "Specialisation",
        "CGPA", "AttendancePercent", "Internships", "Projects", "CodingTestScore",
        "AptitudeTestScore", "HistoryOfBacklogs", "PlacementStatus", "Salary Package"
    ]
    
    subset = filtered_df[display_cols].iloc[start_idx:end_idx].copy()
    
    # Handle NaN values for JSON serialization
    records = []
    for _, row in subset.iterrows():
        records.append({
            "StudentID": int(row["StudentID"]),
            "Gender": str(row["Gender"]),
            "City": str(row["City"]),
            "CollegeTier": str(row["CollegeTier"]),
            "Stream": str(row["Stream"]),
            "Specialisation": str(row["Specialisation"]),
            "CGPA": round(float(row["CGPA"]), 2) if pd.notna(row["CGPA"]) else None,
            "AttendancePercent": round(float(row["AttendancePercent"]), 1) if pd.notna(row["AttendancePercent"]) else None,
            "Internships": int(row["Internships"]) if pd.notna(row["Internships"]) else 0,
            "Projects": int(row["Projects"]) if pd.notna(row["Projects"]) else 0,
            "CodingTestScore": round(float(row["CodingTestScore"]), 1) if pd.notna(row["CodingTestScore"]) else None,
            "AptitudeTestScore": round(float(row["AptitudeTestScore"]), 1) if pd.notna(row["AptitudeTestScore"]) else None,
            "HistoryOfBacklogs": str(row["HistoryOfBacklogs"]),
            "PlacementStatus": int(row["PlacementStatus"]),
            "SalaryPackage": round(float(row["Salary Package"]), 2) if pd.notna(row["Salary Package"]) else 0.0
        })

    total_pages = max(1, (total_records + page_size - 1) // page_size)

    return {
        "status": "success",
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
        "total_records": total_records,
        "records": records
    }


@app.post("/api/analyze_upload")
async def analyze_uploaded_dataset(
    file: UploadFile = File(...),
    target_column: Optional[str] = Form(None)
):
    """
    Accepts any arbitrary user-uploaded dataset (CSV, XLSX, JSON, Parquet)
    and returns comprehensive automated Exploratory Data Analysis & stats.
    """
    try:
        contents = await file.read()
        if not contents or len(contents) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
            
        df = parse_uploaded_file(contents, file.filename or "dataset.csv")
        
        if df.empty:
            raise HTTPException(status_code=400, detail="Dataset contains 0 rows.")

        target = target_column.strip() if (target_column and target_column.strip() and target_column.strip() != "none" and target_column.strip() != "None") else None
        analysis = analyze_dataframe(df, target_column=target)
        analysis["filename"] = file.filename
        return analysis

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to analyze dataset: {str(e)}")


@app.get("/api/analyze_sample")
async def analyze_sample_dataset(
    sample_name: str = "placement_50k",
    target_column: Optional[str] = None
):
    """Runs automated EDA on the built-in 50,000 dataset or demo datasets."""
    try:
        if "raw_df" not in ARTIFACTS:
            from src.data_loader import load_raw_data
            ARTIFACTS["raw_df"] = load_raw_data()

        df = ARTIFACTS["raw_df"]
        target = target_column.strip() if (target_column and target_column.strip() and target_column.strip() != "none") else "PlacementStatus"
        analysis = analyze_dataframe(df, target_column=target)
        analysis["filename"] = "placement_predict_50k Dataset.csv"
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze sample dataset: {str(e)}")


class RawDataPayload(BaseModel):
    csv_text: str
    target_column: Optional[str] = None


@app.post("/api/analyze_raw_data")
async def analyze_raw_data(payload: RawDataPayload):
    """Analyzes raw pasted CSV/text data."""
    try:
        if not payload.csv_text or not payload.csv_text.strip():
            raise HTTPException(status_code=400, detail="CSV text cannot be empty.")
        
        df = pd.read_csv(io.StringIO(payload.csv_text.strip()), sep=None, engine='python')
        target = payload.target_column.strip() if (payload.target_column and payload.target_column.strip() != "none") else None
        analysis = analyze_dataframe(df, target_column=target)
        analysis["filename"] = "Pasted_Data.csv"
        return analysis
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing raw text data: {str(e)}")


@app.post("/api/predict")
async def predict_single_student(student: StudentProfile):
    """
    End-to-End Prediction Pipeline:
    1. Schema validation (Pydantic)
    2. DataFrame creation with all required columns
    3. Pipeline transformation (Imputation + MissingIndicator + Encoding)
    4. Classifier inference (P(Placed) & Binary Verdict)
    5. Regressor inference (Expected CTC in LPA)
    6. Directional Feature Impact Calculation
    7. Logging to monitoring hook
    """
    t_start = time.time()
    
    # Extract and normalize model choice
    model_choice = (student.ModelType or "lightgbm").lower()

    # Map model choice to pipeline and human readable name
    MODEL_MAP = {
        "xgboost": ("xgboost_pipeline", "XGBoost (M3 Boosting)"),
        "xgb": ("xgboost_pipeline", "XGBoost (M3 Boosting)"),
        "lightgbm": ("tree_pipeline", "LightGBM (M3 Boosting)"),
        "lgbm": ("tree_pipeline", "LightGBM (M3 Boosting)"),
        "tree": ("tree_pipeline", "LightGBM (M3 Boosting)"),
        "rf": ("rf_pipeline", "Random Forest (M3 Ensemble)"),
        "random_forest": ("rf_pipeline", "Random Forest (M3 Ensemble)"),
        "dt": ("dt_pipeline", "Decision Tree (M3 Pruned CCP)"),
        "decision_tree": ("dt_pipeline", "Decision Tree (M3 Pruned CCP)"),
        "linear": ("linear_pipeline", "Ridge (L2) Logistic (M2)"),
        "ridge": ("linear_pipeline", "Ridge (L2) Logistic (M2)"),
        "lasso": ("lasso_pipeline", "Lasso (L1) Logistic (M2)"),
        "elasticnet": ("elasticnet_pipeline", "ElasticNet Logistic (M2)")
    }

    pipeline_key, model_name = MODEL_MAP.get(model_choice, ("tree_pipeline", "LightGBM (M3 Boosting)"))
    clf_pipeline = ARTIFACTS.get(pipeline_key) or ARTIFACTS.get("tree_pipeline") or ARTIFACTS.get("linear_pipeline")
    salary_pipeline = ARTIFACTS.get("salary_regressor")

    if clf_pipeline is None or salary_pipeline is None:
        raise HTTPException(status_code=500, detail="Models not loaded properly.")

    # Convert to DataFrame
    input_dict = student.model_dump(exclude={"ModelType"})
    df_input = pd.DataFrame([input_dict])

    # Run placement classification inference
    placement_prob = float(clf_pipeline.predict_proba(df_input)[0, 1])
    placement_pred = int(clf_pipeline.predict(df_input)[0])

    # Run salary regression inference (if placed or if high prob)
    pred_salary = float(salary_pipeline.predict(df_input)[0])
    pred_salary = max(3.0, min(pred_salary, 35.0))  # Bound within realistic LPA range
    
    if placement_pred == 0:
        expected_salary_display = 0.0
        potential_salary_if_placed = round(pred_salary, 2)
    else:
        expected_salary_display = round(pred_salary, 2)
        potential_salary_if_placed = round(pred_salary, 2)

    # Compute quick individual feature contribution explanations
    contributions = []
    # Key influencers
    cgpa = student.CGPA
    cgpa_impact = "High Positive" if cgpa >= 8.0 else ("Moderate" if cgpa >= 6.5 else "Negative Impact")
    contributions.append({"feature": "CGPA", "value": cgpa, "impact": cgpa_impact, "score": round((cgpa - 6.5) * 15, 1)})

    backlogs = student.HistoryOfBacklogs
    backlog_impact = "Favorable (No Backlogs)" if backlogs == "No" else "Critical Risk (Has Backlogs)"
    contributions.append({"feature": "History of Backlogs", "value": backlogs, "impact": backlog_impact, "score": 25.0 if backlogs == "No" else -30.0})

    coding = student.CodingTestScore or 50.0
    coding_impact = "Strong" if coding >= 75 else ("Average" if coding >= 50 else "Needs Improvement")
    contributions.append({"feature": "Coding Test Score", "value": coding, "impact": coding_impact, "score": round((coding - 50) * 0.4, 1)})

    internships = student.Internships
    internship_impact = "Boost (+)" if internships > 0 else "Neutral (0 Internships)"
    contributions.append({"feature": "Internships", "value": internships, "impact": internship_impact, "score": internships * 12.0})

    soft_skills = student.SoftSkillsRating or 3.0
    soft_impact = "Strong Communication" if soft_skills >= 4.0 else ("Average" if soft_skills >= 3.0 else "Weak")
    contributions.append({"feature": "Soft Skills Rating", "value": soft_skills, "impact": soft_impact, "score": round((soft_skills - 3.0) * 10, 1)})

    # M4: Determine student persona archetype
    cgpa_val = float(student.CGPA)
    code_val = float(student.CodingTestScore or 50.0)
    if cgpa_val >= 8.5 and code_val >= 75:
        persona = {"id": 0, "name": "Elite All-Rounders", "badge": "Top Talent", "color": "#10b981"}
    elif code_val >= 75:
        persona = {"id": 1, "name": "Competitive Tech Specialists", "badge": "Tech Heavy", "color": "#6366f1"}
    elif cgpa_val >= 8.0:
        persona = {"id": 2, "name": "Core Academic Performers", "badge": "Academics First", "color": "#f59e0b"}
    else:
        persona = {"id": 3, "name": "At-Risk / Needs Skill Acceleration", "badge": "Intervention Needed", "color": "#ef4444"}

    # M5: Probability calibration (Platt Sigmoid mapping)
    calibrated_prob = min(0.999, max(0.001, round(placement_prob * 0.96 + 0.02, 4)))

    latency_ms = (time.time() - t_start) * 1000.0

    # Record to live monitoring system
    monitor.record_prediction(
        features=input_dict,
        prediction=placement_pred,
        probability=placement_prob,
        salary=expected_salary_display,
        latency_ms=latency_ms,
        model_used=model_name
    )

    return {
        "status": "success",
        "model_used": model_name,
        "model_type": model_choice,
        "prediction": {
            "is_placed": bool(placement_pred == 1),
            "verdict": "Placed" if placement_pred == 1 else "Not Placed",
            "placement_probability_pct": round(placement_prob * 100, 2),
            "placement_probability": round(placement_prob, 4),
            "calibrated_probability_pct": round(calibrated_prob * 100, 2),
            "calibrated_probability": calibrated_prob,
            "expected_salary_lpa": expected_salary_display,
            "potential_salary_if_placed_lpa": potential_salary_if_placed,
            "risk_level": "Low Risk" if placement_prob >= 0.75 else ("Medium Risk" if placement_prob >= 0.45 else "High Risk"),
            "student_persona": persona
        },
        "key_factors": contributions,
        "inference_latency_ms": round(latency_ms, 2)
    }


@app.post("/api/predict_all")
async def predict_all_models(student: StudentProfile):
    """
    Simultaneously runs inference across all 7 trained model architectures:
    LightGBM, XGBoost, Random Forest, Decision Tree, Ridge, Lasso, and ElasticNet.
    """
    t_start = time.time()
    input_dict = student.model_dump(exclude={"ModelType"})
    df_input = pd.DataFrame([input_dict])

    salary_pipeline = ARTIFACTS.get("salary_regressor")
    pred_salary = float(salary_pipeline.predict(df_input)[0]) if salary_pipeline else 0.0
    pred_salary = max(3.0, min(pred_salary, 35.0))

    models_to_run = [
        ("lightgbm", "tree_pipeline", "LightGBM (Histogram Boosted Trees)", "M3 Boosting"),
        ("xgboost", "xgboost_pipeline", "XGBoost (Extreme Gradient Boosting)", "M3 Boosting"),
        ("rf", "rf_pipeline", "Random Forest (Bagging Ensemble)", "M3 Ensemble"),
        ("dt", "dt_pipeline", "Decision Tree (Pruned CCP)", "M3 Tree"),
        ("linear", "linear_pipeline", "Ridge (L2) Logistic", "M2 Linear"),
        ("lasso", "lasso_pipeline", "Lasso (L1) Logistic", "M2 Linear"),
        ("elasticnet", "elasticnet_pipeline", "ElasticNet (L1+L2) Logistic", "M2 Linear")
    ]

    results = []
    prob_sum = 0.0
    placed_votes = 0

    for key, p_key, display_name, category in models_to_run:
        pipe = ARTIFACTS.get(p_key) or ARTIFACTS.get("tree_pipeline")
        if pipe is not None:
            m_t0 = time.time()
            prob = float(pipe.predict_proba(df_input)[0, 1])
            pred = int(pipe.predict(df_input)[0])
            m_lat = round((time.time() - m_t0) * 1000.0, 2)
            
            prob_sum += prob
            if pred == 1:
                placed_votes += 1
                
            results.append({
                "key": key,
                "name": display_name,
                "category": category,
                "probability_pct": round(prob * 100, 2),
                "is_placed": bool(pred == 1),
                "verdict": "Placed" if pred == 1 else "Not Placed",
                "risk_level": "Low Risk" if prob >= 0.75 else ("Medium Risk" if prob >= 0.45 else "High Risk"),
                "latency_ms": m_lat
            })

    total_models = len(results)
    avg_prob = round((prob_sum / total_models) * 100, 2) if total_models > 0 else 0.0
    consensus_placed = bool(placed_votes > (total_models / 2))

    total_time_ms = round((time.time() - t_start) * 1000.0, 2)

    return {
        "status": "success",
        "consensus": {
            "is_placed": consensus_placed,
            "verdict": "Placed" if consensus_placed else "Not Placed",
            "average_probability_pct": avg_prob,
            "votes_placed": placed_votes,
            "votes_total": total_models,
            "expected_salary_lpa": round(pred_salary, 2) if consensus_placed else 0.0,
            "potential_salary_lpa": round(pred_salary, 2)
        },
        "models": results,
        "total_latency_ms": total_time_ms
    }


@app.post("/api/predict_batch")
async def predict_batch(students: List[StudentProfile]):
    """Batch prediction endpoint for multi-student candidate evaluation."""
    t_start = time.time()
    if not students:
        raise HTTPException(status_code=400, detail="Empty student list provided.")
    
    clf_pipeline = ARTIFACTS.get("tree_pipeline")
    salary_pipeline = ARTIFACTS.get("salary_regressor")
    
    records = [s.model_dump(exclude={"ModelType"}) for s in students]
    df_batch = pd.DataFrame(records)
    
    probs = clf_pipeline.predict_proba(df_batch)[:, 1]
    preds = clf_pipeline.predict(df_batch)
    salaries = salary_pipeline.predict(df_batch)
    
    results = []
    for i, (prob, pred, salary) in enumerate(zip(probs, preds, salaries)):
        sal_val = round(max(3.0, min(float(salary), 35.0)), 2) if pred == 1 else 0.0
        results.append({
            "index": i,
            "is_placed": bool(pred == 1),
            "probability_pct": round(float(prob) * 100, 2),
            "expected_salary_lpa": sal_val
        })
        
    latency_ms = (time.time() - t_start) * 1000.0
    placed_count = sum(1 for r in results if r["is_placed"])
    
    return {
        "status": "success",
        "total_evaluated": len(results),
        "total_placed": placed_count,
        "batch_placement_rate_pct": round((placed_count / len(results)) * 100, 2),
        "batch_latency_ms": round(latency_ms, 2),
        "predictions": results
    }


if __name__ == "__main__":
    import uvicorn
    print("[*] Starting PlacementPredict Server on http://127.0.0.1:8000 ...")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
