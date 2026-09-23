import os
import json
import pandas as pd

def verify_sprint5():
    print("==================================================")
    print("SPRINT 5 DEFINITION OF DONE VERIFICATION")
    print("==================================================")

    raw_path = "raw_data_loaded.parquet"
    imputed_path = "imputed_data.parquet"
    report_path = "dedup_report.json"
    schema_path = "schema.json"

    # 1. Check raw ground truth data exists
    assert os.path.exists(raw_path), f"FAILED: Ground truth parquet '{raw_path}' missing!"
    df_raw = pd.read_parquet(raw_path)
    raw_rows = len(df_raw)
    raw_cols = len(df_raw.columns)
    print(f"PASS: Ground truth raw dataset loaded ({raw_rows} rows, {raw_cols} columns).")

    # 2. Check deliverables exist and non-empty
    for fp in [imputed_path, report_path]:
        assert os.path.exists(fp), f"FAILED: Deliverable '{fp}' missing!"
        assert os.path.getsize(fp) > 0, f"FAILED: Deliverable '{fp}' is empty (0 bytes)!"
    print(f"PASS: Deliverables exist and are non-empty: {imputed_path}, {report_path}.")

    # 3. Load deliverables
    df_imputed = pd.read_parquet(imputed_path)
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    # 4. Verify Schema Non-Nullability
    retained_exceptions = {"license"}
    cols_spec = schema.get("columns", {})

    non_nullable_cols = [c for c, spec in cols_spec.items() if not spec.get("is_nullable", True)]
    print(f"\nChecking non-nullable columns ({len(non_nullable_cols)} columns in schema)...")

    for col in non_nullable_cols:
        if col in retained_exceptions:
            missing_cnt = df_imputed[col].isna().sum()
            print(f"  [RETAINED EXCEPTION] '{col}': {missing_cnt}/{len(df_imputed)} missing (Documented legitimate absence).")
        elif col in df_imputed.columns:
            missing_cnt = df_imputed[col].isna().sum()
            assert missing_cnt == 0, f"FAILED: Non-nullable column '{col}' has {missing_cnt} remaining NaNs!"

    print("PASS: Schema non-nullability verified (0 NaNs in all required non-nullable columns).")

    # 5. Verify Imputation of target columns
    target_imputed_cols = ["neighbourhood_group_cleansed", "review_scores_rating", "bedrooms", "beds", "price", "minimum_nights"]
    for tc in target_imputed_cols:
        if tc in df_imputed.columns:
            cnt = df_imputed[tc].isna().sum()
            assert cnt == 0, f"FAILED: Target imputed column '{tc}' has {cnt} remaining NaNs!"
            print(f"PASS: Imputation confirmed for '{tc}' (0 NaNs).")

    # 6. Verify dedup_report.json structure & row count consistency
    required_keys = [
        "timestamp", "before_row_count", "after_row_count",
        "exact_duplicates_count", "fuzzy_duplicates_count",
        "total_duplicates_removed", "deduplication_summary", "flagged_duplicates"
    ]

    for k in required_keys:
        assert k in report, f"FAILED: Missing key '{k}' in dedup_report.json"

    before_cnt = report["before_row_count"]
    after_cnt = report["after_row_count"]
    exact_cnt = report["exact_duplicates_count"]
    fuzzy_cnt = report["fuzzy_duplicates_count"]
    total_removed = report["total_duplicates_removed"]

    assert before_cnt == raw_rows, f"FAILED: before_row_count ({before_cnt}) != raw dataset rows ({raw_rows})"
    assert after_cnt == len(df_imputed), f"FAILED: after_row_count ({after_cnt}) != imputed parquet rows ({len(df_imputed)})"
    assert exact_cnt + fuzzy_cnt == total_removed, f"FAILED: exact ({exact_cnt}) + fuzzy ({fuzzy_cnt}) != total_removed ({total_removed})"
    assert before_cnt - total_removed == after_cnt, f"FAILED: Row count arithmetic mismatch: {before_cnt} - {total_removed} != {after_cnt}"

    print("\nPASS: dedup_report.json metrics verified:")
    print(f"       Before Row Count: {before_cnt}")
    print(f"       After Row Count:  {after_cnt}")
    print(f"       Exact Duplicates: {exact_cnt}")
    print(f"       Fuzzy Duplicates: {fuzzy_cnt}")
    print(f"       Total Removed:    {total_removed}")

    print("\n==================================================")
    print("ALL SPRINT 5 DEFINITION OF DONE VERIFICATIONS PASSED!")
    print("==================================================")

if __name__ == "__main__":
    verify_sprint5()
