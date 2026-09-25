import os
import json
import datetime
import pandas as pd
from typing import Dict, Any, Optional

from load_data import load_data
from profiling_engine import run_profiling_engine
from suspicious_column_rules import run_suspicious_rules
from visualizations import generate_all_visualizations

def run_profiling(
    df: Optional[pd.DataFrame] = None,
    parquet_path: str = "raw_data_loaded.parquet",
    output_json_path: str = "profiling_report.json",
    generate_visuals: bool = True,
    stats_summary_path: Optional[str] = None,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Exposes full data profiling API:
    1. Loads dataset (if df not supplied directly)
    2. Computes dataset summary, per-column missingness & cardinality, numeric correlation matrix, distribution stats
    3. Evaluates suspicious rules (PII columns, format inconsistencies, suspicious row anomalies)
    4. Renders visualization artifacts (heatmap_missing.png, outlier_min_nights.png, correlation_heatmap.png)
    5. Saves and returns profiling_report.json
    """
    if output_dir is None:
        output_dir = os.path.dirname(output_json_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    if stats_summary_path is None:
        stats_summary_path = os.path.join(output_dir, "stats_summary.json")

    if df is None:
        if os.path.exists(parquet_path):
            print(f"[profiling_api] Loading dataset from Parquet '{parquet_path}'...")
            df = pd.read_parquet(parquet_path)
        else:
            print(f"[profiling_api] Parquet '{parquet_path}' not found. Loading raw dataset...")
            df = load_data(output_parquet_path=parquet_path)
    else:
        print(f"[profiling_api] Using provided DataFrame ({len(df)} rows, {len(df.columns)} columns).")

    # 1. Base statistical profiling engine output
    stats_summary = run_profiling_engine(parquet_path=parquet_path, output_json_path=stats_summary_path)

    # 2. Suspicious column rules evaluation
    suspicious_rules = run_suspicious_rules(df)

    # 3. Visualizations rendering
    vis_paths = {}
    if generate_visuals:
        print(f"[profiling_api] Generating visualization artifacts in '{output_dir}'...")
        vis_paths = generate_all_visualizations(df, output_dir=output_dir)
    else:
        vis_paths = {
            "missing_value_heatmap": os.path.join(output_dir, "heatmap_missing.png"),
            "outliers_boxplot": os.path.join(output_dir, "outlier_min_nights.png"),
            "correlation_heatmap": os.path.join(output_dir, "correlation_heatmap.png")
        }

    # 4. Assemble consolidated profiling report schema
    profiling_report = {
        "report_metadata": {
            "generated_at": datetime.datetime.now().isoformat(),
            "version": "1.0.0",
            "engine": "Automated Data Cleaning & Validation System — Sprint 3 Profiling API"
        },
        "dataset_summary": stats_summary["dataset_summary"],
        "per_column_stats": stats_summary["per_column_stats"],
        "correlation_matrix": stats_summary["correlation_matrix"],
        "distribution_stats": stats_summary["distribution_stats"],
        "suspicious_rules_analysis": suspicious_rules,
        "visualizations": vis_paths
    }

    # 5. Write to output_json_path deliverable
    print(f"[profiling_api] Saving deliverable profiling report to '{output_json_path}'...")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(profiling_report, f, indent=2, ensure_ascii=False)

    print(f"[profiling_api] Successfully generated deliverable '{output_json_path}'.")
    return profiling_report

if __name__ == "__main__":
    report = run_profiling()
    print("[profiling_api] Execution completed successfully.")
