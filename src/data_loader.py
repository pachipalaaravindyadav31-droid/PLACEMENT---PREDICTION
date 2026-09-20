"""
PlacementPredict - Data Ingestion & Split Module (M1 & M2/M3)
Handles dataset loading, anomaly isolation, feature/target separation, and reproducible stratified splitting.
"""

from typing import Tuple, Optional, Dict, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


CATEGORICAL_FEATURES = [
    'Gender',
    'City',
    'Stream',
    'Specialisation',
    'Hostel',
    'HistoryOfBacklogs'
]

ORDINAL_FEATURES = [
    'CollegeTier',
    'CGPA_Tier'
]

NUMERICAL_FEATURES = [
    'SGPA_Sem1',
    'SGPA_Sem2',
    'SGPA_Sem3',
    'SGPA_Sem4',
    'SGPA_Sem5',
    'SGPA_Sem6',
    'SGPA_Sem7',
    'SGPA_Sem8',
    'CGPA',
    'AttendancePercent',
    'Internships',
    'Projects',
    'Workshops',
    'Certifications',
    'Publications',
    'AptitudeTestScore',
    'SoftSkillsRating',
    'CodingTestScore',
    'MockInterviewScore',
    'ExtraCurricular'
]

ALL_FEATURES = CATEGORICAL_FEATURES + ORDINAL_FEATURES + NUMERICAL_FEATURES
CLASSIFICATION_TARGET = 'PlacementStatus'
REGRESSION_TARGET = 'Salary Package'


def load_raw_data(data_path: str = "Data/placement_predict_50k Dataset.csv") -> pd.DataFrame:
    """Loads raw dataset from CSV."""
    df = pd.read_csv(data_path)
    return df


def prepare_datasets(
    data_path: str = "Data/placement_predict_50k Dataset.csv",
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    filter_anomalies: bool = True
) -> Dict[str, Any]:
    """
    Ingests and partitions the dataset into stratified Train, Validation, and Test sets.
    
    Returns:
        Dictionary containing:
        - X_train, y_clf_train, y_reg_train
        - X_val, y_clf_val, y_reg_val
        - X_test, y_clf_test, y_reg_test
        - feature_names: Dict of feature lists by type
        - raw_stats: Basic dataset descriptive statistics
    """
    df = load_raw_data(data_path)
    
    if filter_anomalies and 'IsAnomaly' in df.columns:
        # Filter severe recording anomalies for cleaner training distribution
        df_clean = df[df['IsAnomaly'] == 0].copy()
    else:
        df_clean = df.copy()

    X = df_clean[ALL_FEATURES].copy()
    y_clf = df_clean[CLASSIFICATION_TARGET].copy()
    y_reg = df_clean[REGRESSION_TARGET].copy()

    # Split train+val vs test
    combined_test_val_size = test_size + val_size
    X_train, X_temp, y_clf_train, y_clf_temp, y_reg_train, y_reg_temp = train_test_split(
        X, y_clf, y_reg,
        test_size=combined_test_val_size,
        stratify=y_clf,
        random_state=random_state
    )

    # Split val vs test
    val_ratio_in_temp = val_size / combined_test_val_size
    X_val, X_test, y_clf_val, y_clf_test, y_reg_val, y_reg_test = train_test_split(
        X_temp, y_clf_temp, y_reg_temp,
        test_size=(1.0 - val_ratio_in_temp),
        stratify=y_clf_temp,
        random_state=random_state
    )

    stats = {
        "total_records": len(df),
        "clean_records": len(df_clean),
        "anomalies_count": int(df['IsAnomaly'].sum()) if 'IsAnomaly' in df.columns else 0,
        "train_count": len(X_train),
        "val_count": len(X_val),
        "test_count": len(X_test),
        "placement_rate": float(y_clf.mean()),
        "avg_salary_placed": float(y_reg[y_clf == 1].mean())
    }

    return {
        "X_train": X_train, "y_clf_train": y_clf_train, "y_reg_train": y_reg_train,
        "X_val": X_val, "y_clf_val": y_clf_val, "y_reg_val": y_reg_val,
        "X_test": X_test, "y_clf_test": y_clf_test, "y_reg_test": y_reg_test,
        "feature_types": {
            "categorical": CATEGORICAL_FEATURES,
            "ordinal": ORDINAL_FEATURES,
            "numerical": NUMERICAL_FEATURES,
            "all": ALL_FEATURES
        },
        "stats": stats
    }


if __name__ == "__main__":
    data = prepare_datasets()
    print("Dataset loaded successfully!")
    print("Stats:", data["stats"])
