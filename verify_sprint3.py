import os
import json
import pandas as pd
from PIL import Image

def verify_sprint3():
    print("==================================================")
    print("SPRINT 3 DEFINITION OF DONE VERIFICATION")
    print("==================================================")

    # 1. Load parquet ground truth dataset
    parquet_path = "raw_data_loaded.parquet"
    assert os.path.exists(parquet_path), f"FAILED: Missing ground truth parquet '{parquet_path}'"
    df = pd.read_parquet(parquet_path)
    total_rows = len(df)
    total_cols = len(df.columns)
    print(f"PASS: Ground truth dataset loaded ({total_rows} rows, {total_cols} columns).")

    # 2. Check deliverable deliverables exist
    required_files = [
        "profiling_report.json",
        "heatmap_missing.png",
        "outlier_min_nights.png",
        "correlation_heatmap.png",
        "PROFILING_README.md"
    ]
    for file_path in required_files:
        assert os.path.exists(file_path), f"FAILED: Deliverable missing: '{file_path}'"
        assert os.path.getsize(file_path) > 0, f"FAILED: Deliverable '{file_path}' is empty (0 bytes)"
    print(f"PASS: All 5 required deliverables exist and are non-empty: {required_files}.")

    # 3. Verify images are valid rendering PNG files
    png_files = ["heatmap_missing.png", "outlier_min_nights.png", "correlation_heatmap.png"]
    for img_path in png_files:
        try:
            with Image.open(img_path) as img:
                img.verify()
            print(f"PASS: Image '{img_path}' rendered and verified successfully.")
        except Exception as e:
            assert False, f"FAILED: Image '{img_path}' failed validation: {e}"

    # 4. Load profiling_report.json and verify schema
    json_path = "profiling_report.json"
    with open(json_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    required_schema_keys = [
        "report_metadata",
        "dataset_summary",
        "per_column_stats",
        "correlation_matrix",
        "distribution_stats",
        "suspicious_rules_analysis",
        "visualizations"
    ]
    for key in required_schema_keys:
        assert key in report, f"FAILED: Key '{key}' missing from profiling_report.json schema"
    print("PASS: profiling_report.json schema structure fully verified.")

    # 5. Verify PII Flags
    rules = report["suspicious_rules_analysis"]
    assert "pii_columns" in rules, "FAILED: Missing 'pii_columns' in suspicious_rules_analysis"
    pii = rules["pii_columns"]
    assert pii["flagged"] is True, "FAILED: PII columns should be flagged as True"

    pii_found = pii["pii_columns_found"]
    for required_pii in ["host_name", "host_id", "latitude", "longitude"]:
        assert required_pii in pii_found, f"FAILED: Required PII column '{required_pii}' missing from PII flags!"
    print(f"PASS: Required PII columns correctly flagged: ['host_name', 'host_id', 'latitude', 'longitude']. Found total: {pii_found}")

    # 6. Verify Format Inconsistency Flags
    assert "format_inconsistencies" in rules, "FAILED: Missing 'format_inconsistencies' in rules analysis"
    fmt = rules["format_inconsistencies"]
    assert fmt["flagged"] is True, "FAILED: format_inconsistencies should be flagged True"
    assert "bathrooms_text" in fmt["columns"], "FAILED: 'bathrooms_text' format inconsistency flag missing"
    print(f"PASS: 'bathrooms_text' format inconsistency correctly flagged.")

    # 7. Verify Suspicious Rows Flags (price == "$0.00" and minimum_nights > 365)
    assert "suspicious_rows" in rules, "FAILED: Missing 'suspicious_rows' in rules analysis"
    susp = rules["suspicious_rows"]
    
    assert "zero_price_rows" in susp, "FAILED: Missing 'zero_price_rows' in suspicious_rows"
    zero_price_data = susp["zero_price_rows"]
    # Check ground truth zero price count
    price_series = df["price"].astype(str).str.strip()
    gt_zero_price = int((price_series.isin(["$0.00", "$0", "0", "0.0", "$0.0"]) | (price_series.str.replace(r'[\$,]', '', regex=True).str.strip() == "0.00")).sum())
    assert zero_price_data["count"] == gt_zero_price, f"FAILED: zero_price count ({zero_price_data['count']}) != ground truth ({gt_zero_price})"
    print(f"PASS: Zero price rows verified ({zero_price_data['count']} rows flagged).")

    assert "extreme_minimum_nights" in susp, "FAILED: Missing 'extreme_minimum_nights' in suspicious_rows"
    extreme_min = susp["extreme_minimum_nights"]
    gt_extreme_min = int((pd.to_numeric(df["minimum_nights"], errors="coerce") > 365).sum())
    assert extreme_min["criteria"] == 'minimum_nights > 365', f"FAILED: Criteria should be 'minimum_nights > 365', got '{extreme_min['criteria']}'"
    assert extreme_min["count"] == gt_extreme_min, f"FAILED: extreme_minimum_nights count ({extreme_min['count']}) != ground truth ({gt_extreme_min})"
    assert extreme_min["flagged"] == (gt_extreme_min > 0), f"FAILED: extreme_minimum_nights flagged state mismatch"
    print(f"PASS: Extreme minimum_nights (>365) row rule verified (Criteria: {extreme_min['criteria']}, Count: {extreme_min['count']} rows).")

    # 8. Verify Visualizations metadata links
    vis = report["visualizations"]
    assert vis["missing_value_heatmap"] == "heatmap_missing.png"
    assert vis["outliers_boxplot"] == "outlier_min_nights.png"
    assert vis["correlation_heatmap"] == "correlation_heatmap.png"
    print("PASS: Visualization file paths in report match deliverable filenames.")

    # 9. Verify PROFILING_README.md contents
    with open("PROFILING_README.md", "r", encoding="utf-8") as f:
        readme_text = f.read()
    assert "# Module 1: Data Profiling" in readme_text, "FAILED: PROFILING_README.md header title missing"
    assert "profiling_report.json" in readme_text
    assert "run_profiling" in readme_text
    print("PASS: PROFILING_README.md contains presentation, schema, and API documentation.")

    print("\n==================================================")
    print("ALL SPRINT 3 DEFINITION OF DONE VERIFICATIONS PASSED!")
    print("==================================================")

if __name__ == "__main__":
    verify_sprint3()
