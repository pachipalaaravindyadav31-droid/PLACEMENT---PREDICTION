# ⚡ PlacementPredict: Production ML System (M1 — M6)

An industrial-grade, end-to-end Machine Learning system for **Student Placement Probability and Salary (CTC Package) Prediction**, trained on a 50,000-candidate dataset. It covers the full curriculum from data ingestion and leakage-free preprocessing (M1), linear models at depth (M2), tree ensembles (M3), unsupervised clustering & dimensionality reduction (M4), model evaluation, selection & calibration (M5), to production ML engineering, MLOps, telemetry, and automated drift retraining (M6).

---

## 🏗️ System Architecture & Concept Mapping

```
PLACEMENT-PREDICTION/
├── Data/
│   └── placement_predict_50k Dataset.csv     # 50k candidate records (academic, coding, soft skills)
├── artifacts/                                # Serialized production artifacts & metrics
│   ├── linear_pipeline.joblib                # Fitted Ridge Logistic Pipeline (with Preprocessor)
│   ├── lasso_pipeline.joblib                 # Fitted Lasso (L1) Logistic Pipeline
│   ├── elasticnet_pipeline.joblib            # Fitted ElasticNet Pipeline
│   ├── dt_pipeline.joblib                    # Fitted Decision Tree (Pruned CCP)
│   ├── rf_pipeline.joblib                    # Fitted Random Forest Pipeline (with OOB)
│   ├── tree_pipeline.joblib                  # Fitted LightGBM Pipeline (with Preprocessor)
│   ├── xgboost_pipeline.joblib               # Fitted XGBoost Pipeline
│   ├── salary_regressor.joblib               # Fitted Ridge Salary Package Regressor
│   ├── metrics.json                          # Full benchmark comparison matrix
│   ├── linear_interpretability.json          # Weights, log-odds & Odds Ratios
│   ├── feature_importance.json               # Gini / MDI feature importances
│   ├── unsupervised_clustering.json          # M4: K-Means, Personas, PCA, DBSCAN, Isolation Forest
│   ├── calibration_and_evaluation.json       # M5: Platt/Isotonic, Brier, McNemar, CV, Thresholds
│   ├── model_manifest.json                   # M6: Production MLOps manifest, checksums, SLAs
│   └── concept_graphs.json                   # 42 Interactive Chart.js graph coordinate datasets
├── src/
│   ├── data_loader.py                        # Ingestion, anomaly filtering, 70/15/15 stratified splits
│   ├── preprocessor.py                       # ColumnTransformer (MissingIndicator, Imputer, Scaler, One-Hot)
│   ├── linear_models.py                      # Module 2: Ridge (L2), Lasso (L1), ElasticNet, Odds Ratios
│   ├── tree_models.py                        # Module 3: Decision Trees (CCP), Random Forest, LightGBM, XGBoost
│   ├── unsupervised_models.py                # Module 4: KMeans, PCA, t-SNE, DBSCAN, IsolationForest, ClusterAugmenter
│   ├── calibration_and_evaluation.py         # Module 5: Platt/Isotonic, Brier, Stratified & Nested CV, McNemar Test
│   ├── mlops_production.py                   # Module 6: PSI & KS Drift Detector, Retraining Loop, Manifests
│   ├── evaluator.py                          # Multi-metric evaluation (Accuracy, F1, ROC-AUC, RMSE, MAE)
│   ├── monitoring.py                         # Module 1 & 6: Live latency & drift telemetry hook
│   ├── graph_generator.py                    # Complete 42-concept visual graph generator
│   └── train_all.py                          # Full 7-stage training orchestrator & packager
├── Static/
│   ├── style.css                             # Modern responsive dark-slate dashboard design
│   ├── app.js                                # Interactive client, 42 Chart.js renderers & telemetry
│   └── chart.js                              # Local offline Chart.js engine
├── Templates/
│   └── index.html                            # Dashboard with live gauges, factor waterfall, and 42 graphs
├── tests/
│   └── test_system.py                        # Automated pytest suite (23 tests across M1-M6)
├── main.py                                   # FastAPI production serving application
├── app.py                                    # Unified one-click application launcher
└── README.md
```

---

## 📚 Detailed Course Modules & Theoretical Implementation

### 🔹 Module 1 (M1): PlacementPredict as a System — The ML Lifecycle
* **End-to-End ML Lifecycle**:
  $$\text{Raw Data} \to \text{Cleaned Dataset} \to \text{Engineered Features} \to \text{Training Set} \to \text{Trained Model} \to \text{Packaged Artifact} \to \text{Served Endpoint} \to \text{Telemetry Loop}$$
* **Zero Training-Serving Skew**: Preprocessing transforms (`SimpleImputer`, `MissingIndicator`, `StandardScaler`, and `OneHotEncoder`) are bundled directly within `sklearn.pipeline.Pipeline` objects to prevent data leakage and offline-online skew.
* **Supervised Learning Split**:
  * Binary Classification: Placement outcome $y \in \{0, 1\}$ (`PlacementStatus`).
  * Continuous Regression: Salary package $y \in \mathbb{R}^+$ in LPA (`Salary Package`).
* **Production REST API**: Single prediction (`/api/predict`), batch predictions (`/api/predict_batch`), models comparison (`/api/models`), and live telemetry (`/api/metrics`).

---

### 🔹 Module 2 (M2): Supervised Learning — Linear Models at Depth
* **Mathematical Formulations**:
  * Normal Equation: $\theta = (X^T X)^{-1} X^T y$
  * Gradient Descent: $\theta^{(t+1)} = \theta^{(t)} - \eta \nabla_\theta \mathcal{L}(\theta)$
* **Regularization Comparisons**:
  * **Ridge ($L_2$)**: $\mathcal{L}_{L2} = \mathcal{L}_0 + \lambda \sum w_j^2$ (smooth spherical shrinkage; handles collinearity across semester SGPAs).
  * **Lasso ($L_1$)**: $\mathcal{L}_{L1} = \mathcal{L}_0 + \lambda \sum |w_j|$ (sharp $L_1$ diamond corners touch loss contours on coordinate axes, inducing exact weight sparsity for automated feature selection).
  * **ElasticNet**: $\mathcal{L}_{EN} = \mathcal{L}_0 + \lambda \left(\rho \|w\|_1 + \frac{1-\rho}{2} \|w\|_2^2\right)$.
* **Binary Logistic Regression & Odds Ratios**:
  $$\text{Odds} = \frac{p}{1-p} = e^{w^T x + b} \implies \text{Odds Ratio} = e^{w_j}$$
  * Highest Placement Drivers in PlacementPredict: **CGPA** ($14.36\times$), **No Backlogs** ($4.16\times$), **Internships** ($3.47\times$), and **Soft Skills** ($3.05\times$).
* **Continuous Ridge Salary Regressor**: Accurately estimates CTC packages ($R^2 = 0.879$, $\text{MAE} = ₹2.01\text{ LPA}$).

---

### 🔹 Module 3 (M3): Supervised Learning — Tree-Based Models
* **Decision Trees & Cost-Complexity Pruning (CCP)**:
  $$R_\alpha(T) = R(T) + \alpha |T|$$
  Balances impurity reduction against tree complexity ($|T|$) to eliminate overfitting.
* **Random Forests (Bagging)**:
  * Bootstrap aggregating with random feature subspace sampling ($\sqrt{p}$).
  * Out-Of-Bag (OOB) validation error ($92.01\%$ OOB accuracy).
* **Boosting Workhorses (LightGBM & XGBoost)**:
  * Histogram-based binning reduces split compute from $\mathcal{O}(\text{data} \times \text{features})$ to $\mathcal{O}(\text{bins} \times \text{features})$.
  * Leaf-wise growth achieves **$92.18\%$ test accuracy** and **$0.980$ ROC-AUC**.
* **Tree Models on Tabular Data**: Outperform deep neural networks on tabular datasets by learning sharp axis-aligned step functions without requiring feature standardisation.

---

### 🔹 Module 4 (M4): Unsupervised Learning & Dimensionality Reduction
* **K-Means & MiniBatchKMeans**:
  * Lloyd's algorithm with K-means++ seeding.
  * Elbow Method: Within-Cluster Sum of Squares (WCSS / Inertia) across $K = 2 \dots 8$.
  * Silhouette Score: Measures cluster cohesion and separation ($b - a) / \max(a, b)$, pinpointing optimal $K = 4$.
* **Student Persona Archetypes**:
  1. *Elite All-Rounders*: High CGPA ($>8.5$), high coding ($>75$), multiple internships.
  2. *Competitive Tech Specialists*: Moderate CGPA ($7.0-8.0$), high coding test and project scores.
  3. *Core Academic Performers*: High CGPA ($>8.0$) and attendance, low coding exposure.
  4. *At-Risk / Needs Acceleration*: Low CGPA ($<6.8$), backlogs, low aptitude.
* **Hierarchical Agglomerative Clustering**:
  * Bottom-up agglomeration comparing **Ward**, **Complete**, **Average**, and **Single** linkage.
  * Shows how Ward minimizes variance within clusters while Single linkage suffers from chaining artifacts.
* **Density-Based Clustering with DBSCAN**:
  * Parameterization via $\epsilon = 1.2$ and $\text{minPts} = 15$.
  * Isolates arbitrary non-convex geometries and categorizes noise points ($\text{label} = -1$).
* **Dimensionality Reduction: PCA**:
  * Scree plot: Explained variance ratio and cumulative variance across principal components.
  * Standardized vs Unstandardized Pitfall: Demonstrates how unscaled features (like CodingScore 0-100) drown out crucial features (like CGPA 4-10) unless standardisation is strictly applied.
* **Manifold Visualization (t-SNE & UMAP)**: 2D non-linear projections separating student placement clusters.
* **Anomaly Detection**: Isolation Forest (tree path length isolation) and One-Class SVM.
* **Cluster Labels as Supervised Features**: `ClusterFeatureAugmenter` appends cluster assignments and centroid distances into downstream classifiers.

---

### 🔹 Module 5 (M5): Model Evaluation, Selection, and Calibration
* **Train / Val / Test Split Discipline**:
  * Held-out test set ($N = 7,238$) strictly isolated from all preprocessing, tuning, and threshold selection.
* **Cross-Validation**:
  * Stratified 5-Fold CV: Evaluates variance stability across balanced folds ($91.76\% \pm 0.49\%$).
  * Nested CV: Outer loop for unbiased generalization estimation; inner loop for hyperparameter tuning.
* **Probability Calibration**:
  * **Platt Scaling**: Fits a logistic sigmoid calibrator: $\hat{p} = \frac{1}{1 + \exp(A \cdot f(x) + B)}$.
  * **Isotonic Regression**: Fits non-parametric piecewise monotonic step functions.
  * Reliability Diagrams (Calibration Curves with 10 bins) and **Brier Score** ($\text{BS} = \frac{1}{N}\sum (\hat{p}_i - y_i)^2$) verification.
* **Classification Diagnostics & Imbalance**:
  * Precision, Recall, F1-Score, ROC-AUC vs Precision-Recall Curves (PR-AUC).
  * Threshold Tuning Curve: Tracing F1 across thresholds $0.05 \dots 0.95$.
* **Regression Diagnostics**:
  * RMSE vs MAE outlier sensitivity: Quadratic loss $e^2$ heavily penalizes anomalies while MAE maintains linear robustness $|e|$.
* **Statistical Significance Testing**:
  * **McNemar's Test**: $2 \times 2$ contingency table with Edwards continuity correction:
    $$\chi^2 = \frac{(|b - c| - 1)^2}{b + c}, \quad p\text{-value}$$
    Proves whether error discrepancies between Ridge Logistic and LightGBM are statistically significant ($p = 0.208 \ge 0.05 \implies$ equivalent performance).
  * **Paired Bootstrap Test**: 1,000 bootstrap resamples on regressor absolute error deltas.
* **Hyperparameter Optimization Comparison**:
  * Grid Search vs Random Search (Bergstra-Bengio) vs Bayesian Optimization (Gaussian Process surrogate with Expected Improvement).

---

### 🔹 Module 6 (M6): ML Engineering — From Notebook to Production
* **Closing the Notebook-to-Production Gap**:
  * Modular architecture, production-ready FastAPI REST endpoints, and zero-leakage joblib pipelines.
* **Feature Store Architecture**:
  * Offline Store (Parquet-backed batch data for training) vs Online Store (in-memory key-value cache with $<1.2\text{ms}$ retrieval).
  * Point-in-time correctness: Eliminates future data leakage in temporal feature lookup.
* **Model Packaging & Manifest**:
  * Formal MLOps manifest (`artifacts/model_manifest.json`) recording semantic versioning, SHA-256 pipeline checksums, SLA contracts, and champion/challenger metadata.
  * Packaging Benchmark: Joblib vs ONNX Runtime inference latency and disk footprint.
* **Production Drift Telemetry**:
  * **Population Stability Index (PSI)**:
    $$\text{PSI} = \sum \left( \% \text{Actual} - \% \text{Expected} \right) \times \ln\left( \frac{\% \text{Actual}}{\% \text{Expected}} \right)$$
    * $\text{PSI} < 0.1$: No Drift (Stable)
    * $0.1 \le \text{PSI} < 0.25$: Moderate Drift (Warning)
    * $\text{PSI} \ge 0.25$: Significant Drift (Alert)
  * **Kolmogorov-Smirnov (KS) Test**: 2-sample distributional comparison ($D$-statistic and $p$-value).
* **Automated Retraining Loop**:
  * Triggered autonomously via scheduled cron or drift breach ($\text{PSI} \ge 0.25$).
  * Executes validation, re-fitting, shadow evaluation, and champion-challenger model promotion.

---

## 📊 Complete Benchmark Comparison Matrix (7,238 Test Samples)

| Model Architecture | Module | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Log-Loss | Train Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ridge Logistic ($L_2$)** | M2 Linear | **92.33%** | 0.933 | 0.951 | **0.942** | **0.9810** | 0.164 | 0.38s |
| **Lasso Logistic ($L_1$)** | M2 Linear | **92.39%** | 0.934 | 0.951 | **0.943** | **0.9810** | 0.164 | 1.74s |
| **ElasticNet Logistic** | M2 Linear | **92.37%** | 0.934 | 0.951 | **0.943** | **0.9810** | 0.164 | 1.62s |
| **Decision Tree (CCP Pruned)**| M3 Tree | **91.64%** | 0.932 | 0.941 | **0.937** | 0.9720 | 0.240 | 0.11s |
| **Random Forest (OOB 92.01%)**| M3 Bagging | **92.24%** | 0.927 | 0.957 | **0.942** | 0.9782 | 0.179 | 1.25s |
| **LightGBM (Champion)** | M3 Boosting| **92.11%** | 0.932 | 0.950 | **0.941** | **0.9792** | 0.172 | 0.52s |
| **XGBoost** | M3 Boosting| **92.18%** | 0.931 | 0.952 | **0.941** | **0.9798** | 0.169 | 2.65s |

---

## 📈 Visual Graph Catalog (42 Interactive Concept Graphs)

The web dashboard (`/`) renders **42 interactive Chart.js graphs** categorized by module:

| Module | Graph # | Concept Visualized | Chart Type |
| :--- | :---: | :--- | :--- |
| **M1 Lifecycle** | 1 | Dataset CGPA Distribution vs Placement Rate | Combo Bar & Line |
| | 2 | Engineering Branch vs Salary Package (LPA) | Grouped Bar |
| | 3 | API Inference Latency vs Batch Processing Size | Dual-Axis Line |
| | 4 | Training-Serving Distribution Boundary Simulation | Grouped Bar |
| **M2 Linear Models** | 5 | Normal Equations vs Gradient Descent Convergence | Dual-Loss Line |
| | 6 | Regularization Geometry ($L_1$ Diamond vs $L_2$ Circle) | Parametric Shape |
| | 7 | Logistic Sigmoid Function & Decision Hyperplane | S-Curve Line |
| | 8 | Binary Cross-Entropy (Log-Loss) Convexity | Convex Curves |
| | 9 | Feature Scaling Contours (StandardScaler vs Unscaled) | Ellipse Contours |
| | 10 | Learned Weights & Odds Ratios ($e^{w_j}$) | Horizontal Diverging Bar |
| | 11 | Ridge Salary Regressor: Actual vs Predicted | 45° Scatter Plot |
| **M3 Tree Ensembles**| 12 | Splitting Criteria (Gini Impurity vs Entropy vs Error)| Comparative Line |
| | 13 | Cost-Complexity Pruning (CCP) Path ($\alpha$) | Train/Val Accuracy Line |
| | 14 | Random Forest OOB Error vs Estimators | Variance Reduction Curve |
| | 15 | Gradient Boosting Loss Convergence | Train vs Val Loss |
| | 16 | Receiver Operating Characteristic (ROC-AUC) Curves | Multi-Model ROC |
| | 17 | Precision-Recall (PR-AUC) Tradeoff Curves | Precision vs Recall |
| | 18 | Feature Importance: Gini MDI vs Permutation | Grouped Horizontal Bar |
| | 19 | Model Performance Multi-Metric Benchmarks | Grouped Metric Bars |
| | 20 | Tabular Data: Why Tree Models Beat Neural Networks | Multi-Axis Radar |
| **M4 Unsupervised** | 21 | K-Means Elbow Method (WCSS) & Silhouette Scores | Dual-Axis Line |
| | 22 | PCA Scree Plot: Explained & Cumulative Variance | Combo Bar & Line |
| | 23 | PCA Pitfall: Standardized vs Unscaled Variance | Grouped Bar |
| | 24 | 2D Manifold Projection: Placed vs Unplaced Clusters | 2D Scatter Plot |
| | 25 | Hierarchical Clustering Linkage Criteria (Ward vs Complete)| Grouped Bar |
| | 26 | DBSCAN Density & Noise Point Outlier Isolation | Scatter with Outliers |
| | 27 | Isolation Forest: Anomaly Score Distribution | Cutoff Histogram |
| | 28 | Student Persona Archetypes: Centroid Profiles | Multi-Archetype Radar |
| **M5 Evaluation** | 29 | Probability Calibration Reliability Diagram | Calibration Curves |
| | 30 | Stratified 5-Fold vs Nested Cross-Validation | Grouped Fold Bars |
| | 31 | Decision Threshold Tuning (Precision vs Recall vs F1) | Multi-Metric Line |
| | 32 | Regression Error Penalties: RMSE ($L_2$) vs MAE ($L_1$) | Quadratic vs Linear |
| | 33 | Learning Curve: Diagnosing Bias vs Variance | Train/Val Sample Curve |
| | 34 | Validation Curve: Model Capacity (Max Depth) | Generalization Curve |
| | 35 | Hyperparameter Search Efficiency (Grid vs Random vs Bayes)| Optimization Line |
| | 36 | McNemar's Hypothesis Test: $2 \times 2$ Contingency Matrix | Contingency Bar |
| **M6 Production** | 37 | Production Latency SLAs: P50, P95, P99 vs Batch Size | SLA Boundary Lines |
| | 38 | Feature Drift Monitoring: PSI & KS Statistics | Drift Threshold Bars |
| | 39 | Feature Store Parity: Offline vs Online Distribution | Parity Comparison |
| | 40 | Autonomous Retraining: Degradation & Recovery Loop | Timeline Trajectory |
| | 41 | Model Packaging: Joblib vs ONNX Runtime Benchmark | Performance Grouped Bar |
| | 42 | Model Registry Progression: Versioned F1 Scores | Versioned Combo Chart |

---

## 🚀 How to Run the Application

### 1. Unified Launcher (Recommended)
```bash
python app.py
```
* Automatically verifies pre-trained artifacts.
* Starts the FastAPI production service on `http://127.0.0.1:8000`.
* Opens the web dashboard automatically in your default browser.

### 2. Manual Training Lifecycle Orchestration
```bash
python src/train_all.py
```
Executes all 7 stages across M1-M6, fits all pipelines, computes calibrations, generates MLOps manifests, and exports all 42 graph datasets to `artifacts/concept_graphs.json`.

### 3. Run Automated Pytest Suite
```bash
pytest tests/test_system.py -v
```
Runs 23 comprehensive tests verifying API endpoints, model predictions, personas, calibration, and drift telemetry.
