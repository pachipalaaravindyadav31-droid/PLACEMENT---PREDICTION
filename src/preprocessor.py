"""
PlacementPredict - Preprocessing Pipeline Module (M1 & M2/M3)
Implements scikit-learn ColumnTransformer with MissingIndicator, Imputation,
StandardScaler, One-Hot Encoding, and Ordinal Encoding to guarantee zero training-serving skew.
"""

from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

from src.data_loader import (
    CATEGORICAL_FEATURES,
    ORDINAL_FEATURES,
    NUMERICAL_FEATURES,
    ALL_FEATURES
)

# Explicit ordinal category orderings
ORDINAL_CATEGORIES = [
    ['Tier3', 'Tier2', 'Tier1'],  # CollegeTier: Tier3 < Tier2 < Tier1
    ['Low', 'Mid', 'High']         # CGPA_Tier: Low < Mid < High
]


def build_preprocessor(include_scaling: bool = True) -> ColumnTransformer:
    """
    Constructs a robust ColumnTransformer pipeline.
    
    Args:
        include_scaling: If True, applies StandardScaler to numerical features (needed for M2 Linear models).
                        If False, skips scaling (appropriate for M3 Tree models).
                        
    Returns:
        Fitted or unfitted scikit-learn ColumnTransformer.
    """
    # 1. Numerical Pipeline: Median Imputation + MissingIndicator + (Optional) StandardScaler
    num_steps = [
        ('imputer', SimpleImputer(strategy='median', add_indicator=True))
    ]
    if include_scaling:
        num_steps.append(('scaler', StandardScaler()))
    
    numerical_pipeline = Pipeline(steps=num_steps)

    # 2. Categorical Pipeline: Most frequent imputation + OneHotEncoder
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # 3. Ordinal Pipeline: Most frequent imputation + OrdinalEncoder with predefined order
    ordinal_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(
            categories=ORDINAL_CATEGORIES,
            handle_unknown='use_encoded_value',
            unknown_value=-1
        ))
    ])

    # Compose into full ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, NUMERICAL_FEATURES),
            ('cat', categorical_pipeline, CATEGORICAL_FEATURES),
            ('ord', ordinal_pipeline, ORDINAL_FEATURES)
        ],
        remainder='drop',
        verbose_feature_names_out=True
    )
    
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """Extracts human-readable transformed feature names from a fitted ColumnTransformer."""
    feature_names = []
    
    # 1. Numerical features and their missing indicators
    num_transformer = preprocessor.named_transformers_['num']
    imputer = num_transformer.named_steps['imputer']
    
    feature_names.extend(NUMERICAL_FEATURES)
    if hasattr(imputer, 'indicator_') and imputer.indicator_ is not None:
        missing_indices = imputer.indicator_.features_
        for idx in missing_indices:
            feature_names.append(f"{NUMERICAL_FEATURES[idx]}_is_missing")
            
    # 2. Categorical one-hot feature names
    cat_transformer = preprocessor.named_transformers_['cat']
    encoder = cat_transformer.named_steps['encoder']
    cat_encoded_names = encoder.get_feature_names_out(CATEGORICAL_FEATURES)
    feature_names.extend(list(cat_encoded_names))
    
    # 3. Ordinal features
    feature_names.extend(ORDINAL_FEATURES)
    
    return feature_names
