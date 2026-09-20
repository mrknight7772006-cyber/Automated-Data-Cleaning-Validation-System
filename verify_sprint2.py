import os
import json
import pandas as pd

def verify_sprint2():
    print("==================================================")
    print("SPRINT 2 DEFINITION OF DONE VERIFICATION")
    print("==================================================")
    
    # 1. Load parquet dataset for ground truth comparison
    parquet_path = "raw_data_loaded.parquet"
    assert os.path.exists(parquet_path), f"FAILED: '{parquet_path}' missing!"
    df = pd.read_parquet(parquet_path)
    total_rows = len(df)
    total_cols = len(df.columns)
    
    # 2. Verify deliverable stats_summary.json exists
    json_path = "stats_summary.json"
    assert os.path.exists(json_path), f"FAILED: '{json_path}' deliverable missing!"
    with open(json_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
        
    print(f"PASS: '{json_path}' exists and loaded successfully.")
    
    # Check top-level keys
    assert "dataset_summary" in summary, "FAILED: Missing 'dataset_summary' in summary JSON"
    assert "per_column_stats" in summary, "FAILED: Missing 'per_column_stats' in summary JSON"
    assert "correlation_matrix" in summary, "FAILED: Missing 'correlation_matrix' in summary JSON"
    assert "distribution_stats" in summary, "FAILED: Missing 'distribution_stats' in summary JSON"
    
    # Check dataset summary values
    ds = summary["dataset_summary"]
    assert ds["total_rows"] == total_rows, f"FAILED: Total rows mismatch ({ds['total_rows']} vs {total_rows})"
    assert ds["total_columns"] == total_cols, f"FAILED: Total columns mismatch ({ds['total_columns']} vs {total_cols})"
    assert ds["duplicate_row_count"] == int(df.duplicated().sum()), "FAILED: Duplicate row count mismatch"
    print(f"PASS: Dataset summary verified ({total_rows} rows, {total_cols} cols, {ds['duplicate_row_count']} duplicates).")
    
    # 3. Check per-column metrics completeness
    col_stats = summary["per_column_stats"]
    assert len(col_stats) == total_cols, f"FAILED: per_column_stats count ({len(col_stats)}) != dataset columns ({total_cols})"
    
    for col in df.columns:
        assert col in col_stats, f"FAILED: Column '{col}' missing from per_column_stats"
        c_meta = col_stats[col]
        assert "dtype" in c_meta, f"FAILED: Missing dtype for {col}"
        assert "non_null_count" in c_meta, f"FAILED: Missing non_null_count for {col}"
        assert "missing_count" in c_meta, f"FAILED: Missing missing_count for {col}"
        assert "missing_pct" in c_meta, f"FAILED: Missing missing_pct for {col}"
        assert "unique_count" in c_meta, f"FAILED: Missing unique_count for {col}"
        
    print("PASS: All columns have complete metrics (dtype, non_null_count, missing_count, missing_pct, unique_count).")
    
    # 4. Manual isna().sum() check for license & review_scores_rating
    manual_license_missing = int(df["license"].isna().sum())
    manual_license_pct = round((manual_license_missing / total_rows) * 100, 4)
    summary_license_missing = col_stats["license"]["missing_count"]
    summary_license_pct = col_stats["license"]["missing_pct"]
    assert manual_license_missing == summary_license_missing, f"FAILED: license missing count ({summary_license_missing}) != ground truth ({manual_license_missing})"
    assert manual_license_pct == summary_license_pct, f"FAILED: license missing % ({summary_license_pct}) != ground truth ({manual_license_pct})"
    print(f"PASS: 'license' missing check matches ground truth ({manual_license_missing}/{total_rows} = {manual_license_pct}%).")

    manual_rating_missing = int(df["review_scores_rating"].isna().sum())
    manual_rating_pct = round((manual_rating_missing / total_rows) * 100, 4)
    summary_rating_missing = col_stats["review_scores_rating"]["missing_count"]
    summary_rating_pct = col_stats["review_scores_rating"]["missing_pct"]
    assert manual_rating_missing == summary_rating_missing, f"FAILED: review_scores_rating missing count ({summary_rating_missing}) != ground truth ({manual_rating_missing})"
    assert manual_rating_pct == summary_rating_pct, f"FAILED: review_scores_rating missing % ({summary_rating_pct}) != ground truth ({manual_rating_pct})"
    print(f"PASS: 'review_scores_rating' missing check matches ground truth ({manual_rating_missing}/{total_rows} = {manual_rating_pct}%).")

    # Verify expected gaps in target columns
    gap_cols = ["host_neighbourhood", "neighbourhood_group_cleansed", "license", "review_scores_rating", "bathrooms_text", "reviews_per_month"]
    for g_col in gap_cols:
        g_missing = col_stats[g_col]["missing_count"]
        assert g_missing > 0, f"FAILED: Expected gap in '{g_col}', found 0 missing values!"
    print(f"PASS: Real gaps confirmed for all target columns: {gap_cols}.")
    
    # 5. Correlation matrix checks
    corr_sec = summary["correlation_matrix"]
    matrix = corr_sec["matrix"]
    numeric_cols = corr_sec["computed_numeric_columns"]
    
    expected_numeric = ["minimum_nights", "availability_365", "number_of_reviews"]
    for col in expected_numeric:
        assert col in numeric_cols, f"FAILED: Expected numeric correlation column '{col}' missing from computed columns"
        assert col in matrix, f"FAILED: Column '{col}' missing from matrix keys"
        
    # Check matrix symmetry: M[i][j] == M[j][i]
    for r in numeric_cols:
        for c in numeric_cols:
            val_rc = matrix[r][c]
            val_cr = matrix[c][r]
            assert val_rc is not None, f"FAILED: Correlation between {r} and {c} is None"
            assert round(val_rc, 5) == round(val_cr, 5), f"FAILED: Correlation matrix non-symmetric at ({r}, {c}): {val_rc} != {val_cr}"
            
    print("PASS: Correlation matrix is symmetric and valid for numeric target columns.")

    # Check price note in correlation metadata
    assert "price" in corr_sec.get("excluded_columns", []), "FAILED: 'price' should be in excluded_columns"
    price_note = corr_sec.get("price_note", "")
    assert "Sprint 6" in price_note, "FAILED: Correlation metadata must reference price pending Sprint 6 normalization"
    print(f"PASS: 'price' exclusion and Sprint 6 normalization note verified: \"{price_note}\".")
    
    # 6. Distribution stats check
    dist_stats = summary["distribution_stats"]
    numeric_cols_df = list(df.select_dtypes(include=["number"]).columns)
    assert len(dist_stats) == len(numeric_cols_df), f"FAILED: Distribution stats count ({len(dist_stats)}) != numeric cols ({len(numeric_cols_df)})"
    
    for num_col in numeric_cols_df:
        assert num_col in dist_stats, f"FAILED: Numeric column '{num_col}' missing from distribution_stats"
        ds_col = dist_stats[num_col]
        for key in ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]:
            assert key in ds_col, f"FAILED: Missing '{key}' stat in distribution_stats for '{num_col}'"
            
    print("PASS: Distribution stats present and verified for all numeric columns.")
    
    print("\n==================================================")
    print("ALL SPRINT 2 DEFINITION OF DONE VERIFICATIONS PASSED!")
    print("==================================================")

if __name__ == "__main__":
    verify_sprint2()
