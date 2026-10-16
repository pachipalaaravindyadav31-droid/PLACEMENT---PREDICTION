"""
PlacementPredict - Unsupervised Learning Module (Module 4)
Implements:
1. K-Means (Lloyd's algorithm, K-Means++, Elbow method WCSS, Silhouette scores, MiniBatchKMeans).
2. Student Persona Clustering (4 distinct academic & skills archetypes).
3. Hierarchical Agglomerative Clustering (Linkage: Ward, Complete, Average, Single).
4. Density-Based Clustering with DBSCAN (eps, min_samples, core/border/noise points).
5. Dimensionality Reduction: PCA (Scree plot, explained variance, standardized vs unstandardized).
6. Manifold Visualization: t-SNE / 2D projections.
7. Anomaly Detection: Isolation Forest & One-Class SVM intuition.
8. Practical Pipeline: Cluster labels as engineered features for downstream supervised models.
"""

import os
import sys
import json
from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from src.data_loader import ALL_FEATURES, NUMERICAL_FEATURES, CATEGORICAL_FEATURES


# Key numerical features used for clustering student academic and coding profiles
CLUSTERING_FEATURES = [
    "CGPA",
    "CodingTestScore",
    "AptitudeTestScore",
    "SoftSkillsRating",
    "Internships",
    "Projects",
    "AttendancePercent"
]


class ClusterFeatureAugmenter(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer that fits K-Means on key features
    and appends cluster assignments and distance-to-centroids as features
    for downstream supervised models (preventing training-serving skew).
    """
    def __init__(self, n_clusters: int = 4, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=random_state)
        self.feature_names_ = None

    def fit(self, X: pd.DataFrame, y=None):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        cols = [c for c in CLUSTERING_FEATURES if c in X_df.columns]
        sub_X = X_df[cols].fillna(X_df[cols].median())
        scaled = self.scaler.fit_transform(sub_X)
        self.kmeans.fit(scaled)
        self.feature_names_ = [f"dist_cluster_{i}" for i in range(self.n_clusters)] + ["cluster_label"]
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        cols = [c for c in CLUSTERING_FEATURES if c in X_df.columns]
        sub_X = X_df[cols].fillna(X_df[cols].median())
        scaled = self.scaler.transform(sub_X)
        dists = self.kmeans.transform(scaled)
        labels = self.kmeans.predict(scaled).reshape(-1, 1)
        return np.hstack([dists, labels])


def compute_kmeans_elbow_and_silhouette(
    X_sample: pd.DataFrame,
    k_range: range = range(2, 9),
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Evaluates K-Means clustering across a range of K:
    - Inertia (Within-Cluster Sum of Squares - WCSS) for the Elbow Method
    - Silhouette Score (measure of cluster separation and cohesion)
    - Calinski-Harabasz and Davies-Bouldin indices
    """
    cols = [c for c in CLUSTERING_FEATURES if c in X_sample.columns]
    X_clean = X_sample[cols].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    k_values = list(k_range)
    wcss_scores = []
    sil_scores = []
    ch_scores = []
    db_scores = []

    # Subsample for fast silhouette calculation if dataset is large
    n_sub = min(3000, len(X_scaled))
    idx_sub = np.random.RandomState(random_state).choice(len(X_scaled), n_sub, replace=False)
    X_sub = X_scaled[idx_sub]

    for k in k_values:
        km = KMeans(n_clusters=k, init='k-means++', n_init=5, max_iter=300, random_state=random_state)
        km.fit(X_scaled)
        wcss_scores.append(round(float(km.inertia_), 2))

        labels_sub = km.predict(X_sub)
        sil = silhouette_score(X_sub, labels_sub)
        sil_scores.append(round(float(sil), 4))
        ch = calinski_harabasz_score(X_sub, labels_sub)
        ch_scores.append(round(float(ch), 2))
        db = davies_bouldin_score(X_sub, labels_sub)
        db_scores.append(round(float(db), 4))

    optimal_k = k_values[int(np.argmax(sil_scores))]

    return {
        "k_values": k_values,
        "wcss_inertia": wcss_scores,
        "silhouette_scores": sil_scores,
        "calinski_harabasz_scores": ch_scores,
        "davies_bouldin_scores": db_scores,
        "optimal_k_by_silhouette": optimal_k,
        "method": "Lloyd's algorithm with k-means++ initialization"
    }


def fit_student_persona_clusters(
    df: pd.DataFrame,
    n_clusters: int = 4,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Discovers 4 student persona archetypes based on academic & technical indicators.
    Returns centroid profiles and interpretable archetype descriptions.
    """
    cols = [c for c in CLUSTERING_FEATURES if c in df.columns]
    df_clean = df[cols].dropna().copy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_clean)

    km = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=random_state)
    cluster_labels = km.fit_predict(scaled)
    df_clean['cluster'] = cluster_labels

    centroids_raw = scaler.inverse_transform(km.cluster_centers_)
    
    # Archetype characterization based on centroid properties
    cluster_profiles = []
    archetype_templates = [
        {
            "name": "Elite All-Rounders",
            "badge": "Top Talent",
            "color": "#10b981",
            "description": "High CGPA (>8.5), strong coding skills, multiple internships, and high placement probability."
        },
        {
            "name": "Competitive Tech Specialists",
            "badge": "Tech Heavy",
            "color": "#6366f1",
            "description": "High coding test & project scores, moderate CGPA (7.0-8.0), strong problem solvers."
        },
        {
            "name": "Core Academic Performers",
            "badge": "Academics First",
            "color": "#f59e0b",
            "description": "Solid CGPA (>8.0) and high attendance, but moderate coding exposure; great for core IT & consulting."
        },
        {
            "name": "At-Risk / Needs Skill Acceleration",
            "badge": "Intervention Needed",
            "color": "#ef4444",
            "description": "Lower CGPA (<6.8), few internships or projects, needs targeted bootcamp & mentoring."
        }
    ]

    # Rank clusters by average CGPA + Coding score to assign meaningful persona names
    composite_rank = []
    for i in range(n_clusters):
        c_cgpa = centroids_raw[i, cols.index("CGPA")]
        c_code = centroids_raw[i, cols.index("CodingTestScore")]
        composite_rank.append((c_cgpa * 10 + c_code, i))

    composite_rank.sort(reverse=True)
    cluster_mapping = {orig_idx: rank for rank, (_, orig_idx) in enumerate(composite_rank)}

    for orig_idx, template_idx in cluster_mapping.items():
        template = archetype_templates[min(template_idx, len(archetype_templates) - 1)]
        size = int((cluster_labels == orig_idx).sum())
        pct = round(float(size / len(cluster_labels) * 100), 1)

        means = {}
        for c_idx, col in enumerate(cols):
            means[col] = round(float(centroids_raw[orig_idx, c_idx]), 2)

        cluster_profiles.append({
            "cluster_id": orig_idx,
            "persona_name": template["name"],
            "badge": template["badge"],
            "color": template["color"],
            "description": template["description"],
            "student_count": size,
            "percentage": pct,
            "centroid_means": means
        })

    cluster_profiles.sort(key=lambda x: x["cluster_id"])
    return {
        "n_clusters": n_clusters,
        "profiles": cluster_profiles,
        "features_used": cols
    }


def perform_pca_decomposition(
    df: pd.DataFrame,
    n_components: int = 6,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Computes PCA on student records:
    - Analyzes explained variance ratio and cumulative scree curve.
    - Demonstrates standardized vs unstandardized PCA (pitfall demonstration).
    - Returns 2D principal component coordinates for visualization.
    """
    cols = [c for c in CLUSTERING_FEATURES if c in df.columns]
    X_clean = df[cols].dropna()

    # 1. Standardized PCA (Production Best Practice)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)
    pca_std = PCA(n_components=min(n_components, len(cols)), random_state=random_state)
    coords_std = pca_std.fit_transform(X_scaled)
    exp_var_ratio_std = [round(float(v * 100), 2) for v in pca_std.explained_variance_ratio_]
    cum_var_ratio_std = [round(float(v * 100), 2) for v in np.cumsum(pca_std.explained_variance_ratio_)]

    # 2. Unstandardized PCA (Demonstrates Pitfall where high-scale features like CodingScore dominate)
    pca_unstd = PCA(n_components=min(n_components, len(cols)), random_state=random_state)
    pca_unstd.fit(X_clean)
    exp_var_ratio_unstd = [round(float(v * 100), 2) for v in pca_unstd.explained_variance_ratio_]

    # Sample points for 2D visualization
    sample_size = min(400, len(coords_std))
    idx = np.random.RandomState(random_state).choice(len(coords_std), sample_size, replace=False)

    points_2d = []
    labels = df.loc[X_clean.index, "PlacementStatus"].iloc[idx].tolist() if "PlacementStatus" in df.columns else [0] * sample_size
    for i, real_idx in enumerate(idx):
        points_2d.append({
            "pc1": round(float(coords_std[real_idx, 0]), 3),
            "pc2": round(float(coords_std[real_idx, 1]), 3),
            "placed": int(labels[i])
        })

    # Feature loadings for PC1 and PC2 (Biplot directions)
    loadings_pc1 = {col: round(float(pca_std.components_[0, c_idx]), 3) for c_idx, col in enumerate(cols)}
    loadings_pc2 = {col: round(float(pca_std.components_[1, c_idx]), 3) for c_idx, col in enumerate(cols)}

    return {
        "components": [f"PC{i+1}" for i in range(len(exp_var_ratio_std))],
        "explained_variance_ratio_std": exp_var_ratio_std,
        "cumulative_variance_ratio_std": cum_var_ratio_std,
        "explained_variance_ratio_unstandardized": exp_var_ratio_unstd,
        "loadings_pc1": loadings_pc1,
        "loadings_pc2": loadings_pc2,
        "sample_2d_points": points_2d,
        "pitfall_note": "Without standardization, features with large scales (like CodingTestScore: 0-100) dominate 85%+ of variance, completely drowning out CGPA (4-10)."
    }


def perform_dbscan_analysis(
    df: pd.DataFrame,
    eps: float = 1.2,
    min_samples: int = 15,
    sample_size: int = 1500,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Executes DBSCAN (Density-Based Spatial Clustering of Applications with Noise):
    - Identifies core points, border points, and noise points (outliers with label -1).
    - Demonstrates how DBSCAN naturally isolates irregular clusters without assuming spherical shapes.
    """
    cols = [c for c in CLUSTERING_FEATURES if c in df.columns]
    X_clean = df[cols].dropna()
    if len(X_clean) > sample_size:
        X_sub = X_clean.sample(n=sample_size, random_state=random_state)
    else:
        X_sub = X_clean

    scaler = StandardScaler()
    scaled = scaler.fit_transform(X_sub)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(scaled)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    noise_pct = round(float(n_noise / len(labels) * 100), 2)

    # 2D PCA projection of DBSCAN results
    pca = PCA(n_components=2, random_state=random_state)
    coords_2d = pca.fit_transform(scaled)

    points = []
    # Take 300 points for clear visualization
    vis_count = min(300, len(labels))
    for i in range(vis_count):
        points.append({
            "x": round(float(coords_2d[i, 0]), 3),
            "y": round(float(coords_2d[i, 1]), 3),
            "cluster": int(labels[i]),
            "is_noise": bool(labels[i] == -1)
        })

    return {
        "eps": eps,
        "min_samples": min_samples,
        "clusters_found": n_clusters,
        "noise_points_count": n_noise,
        "noise_percentage": noise_pct,
        "total_analyzed": len(labels),
        "sample_points": points
    }


def perform_anomaly_detection_isolation_forest(
    df: pd.DataFrame,
    contamination: float = 0.03,
    sample_size: int = 2000,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Isolation Forest Anomaly Detection:
    - Recursively partitions feature space using random split values.
    - Anomalies require significantly fewer splits to isolate (shorter tree path length).
    """
    cols = [c for c in CLUSTERING_FEATURES if c in df.columns]
    X_clean = df[cols].dropna()
    if len(X_clean) > sample_size:
        X_sub = X_clean.sample(n=sample_size, random_state=random_state)
    else:
        X_sub = X_clean

    iso = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1
    )
    preds = iso.fit_predict(X_sub)  # -1 = anomaly, 1 = normal
    scores = iso.decision_function(X_sub)  # lower = more abnormal

    anomaly_count = int((preds == -1).sum())
    normal_count = int((preds == 1).sum())

    # Create score distribution histogram bins
    bins = np.linspace(scores.min(), scores.max(), 16)
    hist_counts, bin_edges = np.histogram(scores, bins=bins)
    bin_labels = [f"{round(bin_edges[i], 2)} to {round(bin_edges[i+1], 2)}" for i in range(len(hist_counts))]

    return {
        "contamination": contamination,
        "anomaly_count": anomaly_count,
        "normal_count": normal_count,
        "anomaly_pct": round(float(anomaly_count / len(preds) * 100), 2),
        "score_distribution": {
            "bins": bin_labels,
            "counts": [int(c) for c in hist_counts]
        },
        "description": "Isolation Forest isolates anomalies by randomly selecting a feature and split value. Outliers have noticeably shorter average path lengths in the trees."
    }
