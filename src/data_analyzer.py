"""
PlacementPredict - Universal Dataset Analyzer & Automated EDA Engine
Provides full-featured exploratory data analysis on any arbitrary CSV, Excel, or JSON dataset.
"""

from typing import Dict, Any, List, Optional, Tuple
import io
import pandas as pd
import numpy as np


def parse_uploaded_file(file_contents: bytes, filename: str) -> pd.DataFrame:
    """Parses raw bytes into a pandas DataFrame based on file extension."""
    filename_lower = filename.lower()
    
    if filename_lower.endswith('.csv') or filename_lower.endswith('.txt'):
        try:
            return pd.read_csv(io.BytesIO(file_contents))
        except UnicodeDecodeError:
            return pd.read_csv(io.BytesIO(file_contents), encoding='latin1')
        except Exception:
            # Try delimiter autodetection
            return pd.read_csv(io.BytesIO(file_contents), sep=None, engine='python')
            
    elif filename_lower.endswith('.xlsx') or filename_lower.endswith('.xls'):
        return pd.read_excel(io.BytesIO(file_contents))
        
    elif filename_lower.endswith('.json'):
        return pd.read_json(io.BytesIO(file_contents))
        
    elif filename_lower.endswith('.parquet'):
        return pd.read_parquet(io.BytesIO(file_contents))
        
    else:
        # Default try CSV
        try:
            return pd.read_csv(io.BytesIO(file_contents))
        except Exception as e:
            raise ValueError(f"Unsupported file format '{filename}'. Please upload CSV, Excel, JSON, or Parquet. Error: {e}")


def analyze_dataframe(df: pd.DataFrame, target_column: Optional[str] = None, max_rows: int = 50000) -> Dict[str, Any]:
    """
    Performs comprehensive automated EDA on any pandas DataFrame.
    """
    total_original_rows = len(df)
    
    # Subsample if extremely large to maintain sub-second response times
    if total_original_rows > max_rows:
        df_sample = df.sample(n=max_rows, random_state=42)
        was_sampled = True
    else:
        df_sample = df
        was_sampled = False

    total_rows = len(df)
    total_cols = len(df.columns)
    total_cells = total_rows * total_cols
    total_missing_cells = int(df.isna().sum().sum())
    missing_pct = round((total_missing_cells / total_cells * 100.0), 2) if total_cells > 0 else 0.0
    
    duplicate_rows = int(df.duplicated().sum())
    duplicate_pct = round((duplicate_rows / total_rows * 100.0), 2) if total_rows > 0 else 0.0
    memory_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)

    # Inferred column types
    numerical_cols = []
    categorical_cols = []
    datetime_cols = []
    boolean_cols = []
    constant_cols = []
    id_like_cols = []

    columns_profile = []
    alerts = []

    for col in df.columns:
        series = df[col]
        n_missing = int(series.isna().sum())
        n_unique = int(series.nunique(dropna=True))
        col_missing_pct = round((n_missing / total_rows * 100.0), 2) if total_rows > 0 else 0.0
        unique_ratio = (n_unique / total_rows) if total_rows > 0 else 0.0

        # Type detection
        if n_unique <= 1:
            inferred_type = "constant"
            constant_cols.append(col)
            alerts.append({"type": "warning", "column": col, "message": f"Column '{col}' has only {n_unique} unique value (zero variance)."})
        elif pd.api.types.is_bool_dtype(series) or (n_unique == 2 and set(series.dropna().unique()).issubset({0, 1, '0', '1', True, False, 'True', 'False', 'yes', 'no', 'Yes', 'No'})):
            inferred_type = "boolean"
            boolean_cols.append(col)
        elif pd.api.types.is_datetime64_any_dtype(series):
            inferred_type = "datetime"
            datetime_cols.append(col)
        elif pd.api.types.is_numeric_dtype(series):
            if unique_ratio > 0.95 and total_rows > 50 and ('id' in str(col).lower() or 'index' in str(col).lower()):
                inferred_type = "id_like"
                id_like_cols.append(col)
            else:
                inferred_type = "numerical"
                numerical_cols.append(col)
        else:
            # Try parsing datetime if object
            if 'date' in str(col).lower() or 'time' in str(col).lower():
                try:
                    pd.to_datetime(series.dropna().iloc[:20])
                    inferred_type = "datetime"
                    datetime_cols.append(col)
                except Exception:
                    if unique_ratio > 0.95 and total_rows > 50:
                        inferred_type = "id_like"
                        id_like_cols.append(col)
                    else:
                        inferred_type = "categorical"
                        categorical_cols.append(col)
            elif unique_ratio > 0.95 and total_rows > 50:
                inferred_type = "id_like"
                id_like_cols.append(col)
            else:
                inferred_type = "categorical"
                categorical_cols.append(col)

        # High missingness alert
        if col_missing_pct >= 20.0:
            alerts.append({"type": "danger", "column": col, "message": f"High missing values: '{col}' is missing {col_missing_pct}% of records."})

        # High cardinality alert
        if inferred_type == "categorical" and n_unique > 50:
            alerts.append({"type": "info", "column": col, "message": f"High cardinality: Categorical '{col}' has {n_unique} distinct categories."})

        col_meta: Dict[str, Any] = {
            "name": str(col),
            "dtype": str(series.dtype),
            "inferred_type": inferred_type,
            "missing_count": n_missing,
            "missing_pct": col_missing_pct,
            "unique_count": n_unique,
            "sample_values": [str(x) for x in series.dropna().iloc[:5].tolist()]
        }

        # Stats for Numerical
        if inferred_type in ["numerical", "id_like"] and pd.api.types.is_numeric_dtype(series):
            valid_nums = series.dropna()
            if len(valid_nums) > 0:
                q25 = float(valid_nums.quantile(0.25))
                q75 = float(valid_nums.quantile(0.75))
                iqr = q75 - q25
                lower_bound = q25 - 1.5 * iqr
                upper_bound = q75 + 1.5 * iqr
                outliers = int(((valid_nums < lower_bound) | (valid_nums > upper_bound)).sum())
                outlier_pct = round((outliers / len(valid_nums) * 100.0), 2)

                col_meta["stats"] = {
                    "min": round(float(valid_nums.min()), 3),
                    "q25": round(q25, 3),
                    "median": round(float(valid_nums.median()), 3),
                    "mean": round(float(valid_nums.mean()), 3),
                    "q75": round(q75, 3),
                    "max": round(float(valid_nums.max()), 3),
                    "std": round(float(valid_nums.std(ddof=1) if len(valid_nums) > 1 else 0.0), 3),
                    "skewness": round(float(valid_nums.skew() if len(valid_nums) > 2 else 0.0), 3),
                    "outliers_count": outliers,
                    "outliers_pct": outlier_pct
                }

                if outlier_pct >= 5.0:
                    alerts.append({"type": "warning", "column": col, "message": f"Outlier anomaly: '{col}' contains {outliers} outlier points ({outlier_pct}% of values)."})

                # 10-bin histogram for fast UI sparkline / chart
                try:
                    counts, bin_edges = np.histogram(valid_nums, bins=10)
                    col_meta["histogram"] = {
                        "labels": [f"{round(bin_edges[i], 2)}-{round(bin_edges[i+1], 2)}" for i in range(len(counts))],
                        "counts": [int(c) for c in counts]
                    }
                except Exception:
                    col_meta["histogram"] = None

        # Top values for Categorical / Boolean / Constant
        if inferred_type in ["categorical", "boolean", "constant"]:
            vc = series.value_counts(dropna=True).head(10)
            col_meta["top_categories"] = [
                {"value": str(k), "count": int(v), "pct": round(v / len(series.dropna()) * 100.0, 1)}
                for k, v in vc.items()
            ]

        columns_profile.append(col_meta)

    # Correlation Matrix (Numerical only)
    corr_matrix = {}
    top_correlations = []
    if len(numerical_cols) >= 2:
        try:
            num_df = df[numerical_cols].dropna()
            if len(num_df) > 5:
                corr_df = num_df.corr(method='pearson').round(3)
                corr_matrix = {
                    "columns": list(corr_df.columns),
                    "matrix": corr_df.values.tolist()
                }

                # Extract notable pairs
                for i in range(len(corr_df.columns)):
                    for j in range(i + 1, len(corr_df.columns)):
                        c1, c2 = corr_df.columns[i], corr_df.columns[j]
                        val = float(corr_df.iloc[i, j])
                        if not np.isnan(val) and abs(val) >= 0.4:
                            top_correlations.append({
                                "feature_a": c1,
                                "feature_b": c2,
                                "correlation": val,
                                "strength": "Strong Positive" if val >= 0.7 else ("Moderate Positive" if val >= 0.4 else ("Strong Negative" if val <= -0.7 else "Moderate Negative"))
                            })
                            if abs(val) >= 0.85:
                                alerts.append({"type": "info", "column": f"{c1} & {c2}", "message": f"Multicollinearity: '{c1}' and '{c2}' have a high correlation of {val}."})

                top_correlations = sorted(top_correlations, key=lambda x: abs(x["correlation"]), reverse=True)[:10]
        except Exception as e:
            print(f"Error computing correlation: {e}")

    # Missing values bar chart dataset
    missing_chart = {
        "columns": [c["name"] for c in columns_profile if c["missing_count"] > 0],
        "percentages": [c["missing_pct"] for c in columns_profile if c["missing_count"] > 0],
        "counts": [c["missing_count"] for c in columns_profile if c["missing_count"] > 0]
    }

    # Target variable deep dive analysis (if provided or auto-detected)
    target_analysis = None
    if target_column and target_column in df.columns:
        target_series = df[target_column].dropna()
        n_unique_target = target_series.nunique()
        
        # Decide if classification or regression
        is_classification = (n_unique_target <= 10) or not pd.api.types.is_numeric_dtype(target_series)
        
        target_info: Dict[str, Any] = {
            "name": target_column,
            "type": "classification" if is_classification else "regression",
            "unique_classes_count": int(n_unique_target),
            "non_null_records": len(target_series)
        }
        
        if is_classification:
            vc = target_series.value_counts()
            target_info["classes"] = [
                {"label": str(k), "count": int(v), "percentage": round(v / len(target_series) * 100.0, 2)}
                for k, v in vc.items()
            ]
            if len(vc) == 2:
                imbalance = max(vc.iloc[0], vc.iloc[1]) / min(vc.iloc[0], vc.iloc[1]) if min(vc.iloc[0], vc.iloc[1]) > 0 else 999.0
                target_info["imbalance_ratio"] = f"{round(imbalance, 2)}:1"
                if imbalance >= 4.0:
                    alerts.append({"type": "warning", "column": target_column, "message": f"Class Imbalance: Target '{target_column}' is imbalanced ({round(imbalance, 1)}:1 ratio)."})
        else:
            target_info["stats"] = {
                "mean": round(float(target_series.mean()), 3),
                "median": round(float(target_series.median()), 3),
                "std": round(float(target_series.std()), 3),
                "min": round(float(target_series.min()), 3),
                "max": round(float(target_series.max()), 3)
            }

        # Compute feature correlations / associations with target
        feature_impacts = []
        for col in numerical_cols:
            if col != target_column:
                try:
                    if pd.api.types.is_numeric_dtype(df[target_column]):
                        corr = float(df[[col, target_column]].dropna().corr().iloc[0, 1])
                        if not np.isnan(corr):
                            feature_impacts.append({
                                "feature": col,
                                "metric_type": "Pearson r",
                                "score": round(corr, 3),
                                "abs_score": abs(corr),
                                "direction": "Positive" if corr > 0 else "Negative"
                            })
                except Exception:
                    pass

        feature_impacts = sorted(feature_impacts, key=lambda x: x["abs_score"], reverse=True)
        target_info["top_features"] = feature_impacts[:12]
        target_analysis = target_info

    # Sample records for preview table (first 25 rows)
    preview_df = df.head(25).copy()
    preview_records = []
    for _, row in preview_df.iterrows():
        r = {}
        for col in preview_df.columns:
            val = row[col]
            if pd.isna(val):
                r[col] = None
            elif isinstance(val, (int, np.integer)):
                r[col] = int(val)
            elif isinstance(val, (float, np.floating)):
                r[col] = round(float(val), 3)
            else:
                r[col] = str(val)
        preview_records.append(r)

    return {
        "status": "success",
        "dataset_summary": {
            "total_rows": total_rows,
            "total_columns": total_cols,
            "memory_mb": memory_mb,
            "duplicate_rows": duplicate_rows,
            "duplicate_percentage": duplicate_pct,
            "total_missing_cells": total_missing_cells,
            "missing_percentage": missing_pct,
            "was_sampled": was_sampled
        },
        "column_types_summary": {
            "numerical": len(numerical_cols),
            "categorical": len(categorical_cols),
            "boolean": len(boolean_cols),
            "datetime": len(datetime_cols),
            "id_like": len(id_like_cols),
            "constant": len(constant_cols)
        },
        "columns": columns_profile,
        "missing_chart": missing_chart,
        "correlation": {
            "matrix": corr_matrix,
            "top_pairs": top_correlations
        },
        "target_analysis": target_analysis,
        "data_quality_alerts": alerts,
        "preview_records": preview_records
    }
