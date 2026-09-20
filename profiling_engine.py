import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from load_data import load_data

def run_profiling_engine(
    parquet_path: str = "raw_data_loaded.parquet",
    output_json_path: str = "stats_summary.json"
) -> Dict[str, Any]:
    """
    Computes data quality statistics for the dataset:
    - Dataset summary (row counts, column counts, duplicate row count & percentage)
    - Per-column missing-value matrix & cardinality metrics
    - Pairwise symmetric correlation matrix for numerical target columns (minimum_nights, availability_365, number_of_reviews) with note on price pending Sprint 6
    - Distribution statistics for all numeric columns
    """
    if not os.path.exists(parquet_path):
        print(f"Parquet file '{parquet_path}' not found. Loading via load_data pipeline...")
        df = load_data(output_parquet_path=parquet_path)
    else:
        print(f"Loading dataset from '{parquet_path}'...")
        df = pd.read_parquet(parquet_path)
        
    total_rows = len(df)
    total_columns = len(df.columns)
    
    # 1. Duplicate row count
    duplicate_rows = int(df.duplicated().sum())
    duplicate_pct = float(round((duplicate_rows / total_rows) * 100, 4)) if total_rows > 0 else 0.0
    
    dataset_summary = {
        "total_rows": total_rows,
        "total_columns": total_columns,
        "duplicate_row_count": duplicate_rows,
        "duplicate_row_pct": duplicate_pct
    }
    
    # 2. Per-column missing value matrix & cardinality
    per_column_stats = {}
    for col in df.columns:
        series = df[col]
        non_null_count = int(series.notna().sum())
        missing_count = int(series.isna().sum())
        missing_pct = float(round((missing_count / total_rows) * 100, 4)) if total_rows > 0 else 0.0
        unique_count = int(series.nunique(dropna=True))
        
        per_column_stats[col] = {
            "dtype": str(series.dtype),
            "total_rows": total_rows,
            "non_null_count": non_null_count,
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "unique_count": unique_count
        }
        
    # 3. Correlation Matrix (price, minimum_nights, availability_365, number_of_reviews)
    target_corr_cols = ["price", "minimum_nights", "availability_365", "number_of_reviews"]
    numeric_corr_cols = [c for c in target_corr_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    
    corr_df = df[numeric_corr_cols].corr()
    
    # Convert dataframe correlation to symmetric dict of dicts
    matrix_dict = {}
    for r in numeric_corr_cols:
        matrix_dict[r] = {}
        for c in numeric_corr_cols:
            val = corr_df.loc[r, c]
            matrix_dict[r][c] = float(round(val, 6)) if not pd.isna(val) else None
            
    correlation_section = {
        "target_columns": target_corr_cols,
        "computed_numeric_columns": numeric_corr_cols,
        "excluded_columns": ["price"],
        "price_note": "price column is string-formatted in raw dataset (e.g. '$120.00') and excluded from numeric correlation calculation; pending Sprint 6 normalization.",
        "matrix": matrix_dict
    }
    
    # 4. Distribution stats per numeric column
    distribution_stats = {}
    numeric_df = df.select_dtypes(include=[np.number])
    
    for col in numeric_df.columns:
        series = numeric_df[col].dropna()
        if len(series) == 0:
            distribution_stats[col] = {
                "count": 0,
                "mean": None,
                "std": None,
                "min": None,
                "25%": None,
                "50%": None,
                "75%": None,
                "max": None
            }
        else:
            desc = series.describe(percentiles=[0.25, 0.5, 0.75])
            distribution_stats[col] = {
                "count": int(desc["count"]),
                "mean": float(round(desc["mean"], 6)),
                "std": float(round(desc["std"], 6)) if not pd.isna(desc["std"]) else 0.0,
                "min": float(round(desc["min"], 6)),
                "25%": float(round(desc["25%"], 6)),
                "50%": float(round(desc["50%"], 6)),
                "75%": float(round(desc["75%"], 6)),
                "max": float(round(desc["max"], 6))
            }
        
    # Combine deliverable JSON
    stats_summary = {
        "dataset_summary": dataset_summary,
        "per_column_stats": per_column_stats,
        "correlation_matrix": correlation_section,
        "distribution_stats": distribution_stats
    }
    
    print(f"Saving statistics summary to '{output_json_path}'...")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(stats_summary, f, indent=2, ensure_ascii=False)
        
    print(f"Successfully generated deliverable '{output_json_path}'.")
    return stats_summary

if __name__ == "__main__":
    run_profiling_engine()
