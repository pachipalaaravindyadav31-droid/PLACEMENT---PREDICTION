# ⚡ PlacementPredict: Production ML System (M1, M2, M3)

An end-to-end Machine Learning system for **Student Placement Probability and Salary (CTC) Package Prediction**, built from a 50,000-candidate dataset. It covers the full lifecycle from raw data ingestion and leak-free preprocessing to linear mathematical modeling, tree ensemble training, FastAPI production serving, and real-time telemetry monitoring.

---

## 🏗️ System Architecture & Concept Mapping

```
PLACEMENT-PREDICTION/
├── Data/
│   └── placement_predict_50k Dataset.csv     # 50k candidate records (academic, coding, soft skills)
├── artifacts/                                # Serialized production artifacts & metrics
│   ├── linear_pipeline.joblib                # Fitted Ridge Logistic Pipeline (with Preprocessor)
│   ├── tree_pipeline.joblib                  # Fitted LightGBM Pipeline (with Preprocessor)
│   ├── rf_pipeline.joblib                    # Fitted Random Forest Pipeline (with OOB)
│   ├── salary_regressor.joblib               # Fitted Ridge Salary Package Regressor
│   ├── metrics.json                          # Full benchmark comparison matrix
│   ├── linear_interpretability.json          # Weights, log-odds & Odds Ratios
│   └── feature_importance.json               # Gini / MDI feature importances
├── src/
│   ├── data_loader.py                        # Ingestion, anomaly filtering, 70/15/15 stratified splits
│   ├── preprocessor.py                       # ColumnTransformer (MissingIndicator, Imputer, Scaler, One-Hot)
│   ├── linear_models.py                      # Module 2: Ridge (L2), Lasso (L1), ElasticNet, Odds Ratios
│   ├── tree_models.py                        # Module 3: Decision Trees (CCP), Random Forest, LightGBM, XGBoost
│   ├── evaluator.py                          # Multi-metric evaluation (Accuracy, F1, ROC-AUC, RMSE, MAE)
│   ├── monitoring.py                         # Module 1: Live latency & drift telemetry hook
│   └── train_all.py                          # Full training lifecycle & artifact packaging
├── Static/
│   ├── style.css                             # Modern responsive dark-slate dashboard design
│   └── app.js                                # Real-time inference client & cross-model comparison
├── Templates/
│   └── index.html                            # Interactive web portal with live gauges & factor waterfall
├── tests/
│   └── test_system.py                        # Automated pytest suite for API & model inferences
├── main.py                                   # FastAPI production serving application
└── README.md
```

---

## 📚 Concepts Implemented

### 🔹 Module 1 (M1): PlacementPredict as a System — The ML Lifecycle
* **End-to-End Pipeline**: Raw data $\to$ cleaning & anomaly removal $\to$ feature engineering $\to$ training $\to$ serialization $\to$ FastAPI endpoint $\to$ live telemetry monitoring.
* **Zero Training-Serving Skew**: Preprocessing transforms (median imputation with `MissingIndicator`, `StandardScaler`, and `OneHotEncoder`) are bundled directly within `sklearn.pipeline.Pipeline` objects.
* **Supervised Separation**: Binary classification (`PlacementStatus`) and continuous regression (`Salary Package`).
* **Production REST API**: Includes single prediction (`/api/predict`), batch predictions (`/api/predict_batch`), models comparison (`/api/models`), and telemetry stats (`/api/metrics`).

### 🔹 Module 2 (M2): Supervised Learning — Linear Models at Depth
* **Ridge Logistic Regression ($L_2$)**: Shrinks coefficients smoothly and handles collinearity across semester SGPAs.
* **Lasso Logistic Regression ($L_1$)**: Evaluates geometric sparsity and automated feature selection.
* **ElasticNet Logistic Regression**: Blends $L_1$ and $L_2$ penalties via `l1_ratio`.
* **Ridge Salary Regressor**: Accurately estimates CTC packages with an $R^2$ of **0.879** and MAE of **₹2.01 LPA**.
* **Odds Ratio Interpretation**: Direct calculation of $\text{Odds Ratio} = e^{w_j}$ reveals that **CGPA** ($14.36\times$), **No Backlogs** ($4.16\times$), **Internships** ($3.47\times$), and **Soft Skills** ($3.05\times$) are the highest positive placement drivers.

### 🔹 Module 3 (M3): Supervised Learning — Tree-Based Models
* **Decision Trees with CCP**: Implements Cost-Complexity Pruning to balance tree depth and leaf samples.
* **Random Forests**: Bagging ensemble with out-of-bag validation score (**$92.01\%$ OOB**).
* **Boosting Workhorses (LightGBM & XGBoost)**: Histogram-based binning and leaf-wise growth achieving **$92.18\%$ test accuracy** and **$0.980$ ROC-AUC**.
* **MDI Feature Importance**: Ranks continuous and categorical importance across ensemble splits.

---

## 📊 Benchmark Results on 7,238 Unseen Test Samples

| Model Architecture | Module | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Log-Loss | Training Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ridge Logistic ($L_2$)** | M2 Linear | **92.33%** | 0.933 | 0.951 | **0.942** | **0.9810** | 0.164 | 0.40s |
| **Lasso Logistic ($L_1$)** | M2 Linear | **92.39%** | 0.934 | 0.951 | **0.943** | **0.9810** | 0.164 | 1.82s |
| **ElasticNet Logistic** | M2 Linear | **92.37%** | 0.934 | 0.951 | **0.943** | **0.9810** | 0.164 | 1.65s |
| **Decision Tree (Pruned)** | M3 Tree | **91.64%** | 0.932 | 0.941 | **0.937** | 0.9720 | 0.240 | 0.12s |
| **Random Forest** | M3 Ensemble | **92.24%** | 0.927 | 0.957 | **0.942** | 0.9782 | 0.179 | 1.29s |
| **LightGBM** | M3 Boosting | **92.11%** | 0.932 | 0.950 | **0.941** | **0.9792** | 0.172 | 0.55s |
| **XGBoost** | M3 Boosting | **92.18%** | 0.931 | 0.952 | **0.941** | **0.9798** | 0.169 | 2.78s |

---

## 🚀 How to Run the Application

### 1. Run Automated Training & Generate Artifacts
```bash
python src/train_all.py
```

### 2. Run Automated System Tests
```bash
pytest tests/test_system.py -v
```

### 3. Launch the FastAPI Serving Dashboard
```bash
python main.py
```
Open your browser at **`http://127.0.0.1:8000`** to access the live interactive application.
