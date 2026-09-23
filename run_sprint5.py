import os
import json
import pandas as pd
from imputers import impute_dataframe, MissingValueImputer
from dedup_engine import deduplicate_dataframe, DeduplicationEngine

def run_sprint5_pipeline(
    raw_parquet_path="raw_data_loaded.parquet",
    output_parquet_path="imputed_data.parquet",
    output_report_path="dedup_report.json",
    schema_path="schema.json"
):
    print("==================================================")
    print("RUNNING SPRINT 5 PIPELINE")
    print("==================================================")

    # 1. Load ground truth parquet dataset
    if not os.path.exists(raw_parquet_path):
        raise FileNotFoundError(f"Input dataset '{raw_parquet_path}' not found.")

    df_raw = pd.read_parquet(raw_parquet_path)
    print(f"Loaded raw dataset: {len(df_raw)} rows, {len(df_raw.columns)} columns.")

    # 2. Perform missing value imputation (Statistical + ML)
    print("\nExecuting missing value imputation (Statistical + ML KNN/Iterative Imputer)...")
    imputer = MissingValueImputer(schema_path=schema_path, ml_method="knn")
    df_imputed, imputation_summary = imputer.impute_missing_values(df_raw)
    print(f"Imputation complete. Total missing values before: {imputation_summary['total_missing_before']}, after: {imputation_summary['total_missing_after']}.")

    # 3. Perform near-duplicate listing deduplication
    print("\nExecuting near-duplicate listing deduplication (Exact + Fuzzy matching)...")
    dedup_engine = DeduplicationEngine(similarity_threshold=0.50, coord_precision=4)
    df_clean, dedup_report = dedup_engine.run_deduplication(df_imputed)

    # Add imputation summary to report metadata
    dedup_report["imputation_summary"] = imputation_summary

    print(f"Deduplication complete.")
    print(f"  Rows before: {dedup_report['before_row_count']}")
    print(f"  Rows after:  {dedup_report['after_row_count']}")
    print(f"  Exact duplicates removed: {dedup_report['exact_duplicates_count']}")
    print(f"  Fuzzy duplicates removed: {dedup_report['fuzzy_duplicates_count']}")

    # 4. Save deliverables
    print(f"\nSaving deliverable '{output_parquet_path}'...")
    df_clean.to_parquet(output_parquet_path, index=False)

    print(f"Saving deliverable '{output_report_path}'...")
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(dedup_report, f, indent=2)

    print("\nSPRINT 5 PIPELINE COMPLETED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    run_sprint5_pipeline()
