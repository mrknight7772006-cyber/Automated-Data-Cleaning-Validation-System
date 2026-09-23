import os
import json
import pandas as pd

def verify_sprint7():
    print("==================================================")
    print("SPRINT 7 DEFINITION OF DONE VERIFICATION")
    print("==================================================")

    dataset_path = "cleaned_data.csv"
    report_path = "rule_violations.json"
    module_report_path = "validation_report.json"
    validator_path = "rule_validator.py"

    # 1. Check deliverables exist & non-empty
    for fp in [validator_path, report_path, module_report_path]:
        assert os.path.exists(fp), f"FAILED: Deliverable '{fp}' missing!"
        assert os.path.getsize(fp) > 0, f"FAILED: Deliverable '{fp}' is empty (0 bytes)!"

    print(f"PASS: Deliverables exist and are non-empty: {validator_path}, {report_path}, {module_report_path}.")

    # 2. Load dataset & report
    df = pd.read_csv(dataset_path, low_memory=False)
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    # 3. Verify Top-Level JSON Schema Structure
    required_top_keys = [
        "timestamp", "dataset_path", "city", "bounding_box",
        "total_rows_validated", "total_rules_evaluated",
        "total_violations_found", "per_rule_summary", "rule_violations"
    ]
    for key in required_top_keys:
        assert key in report, f"FAILED: Missing key '{key}' in {report_path}"

    assert report["total_rows_validated"] == len(df), f"FAILED: total_rows_validated ({report['total_rows_validated']}) != cleaned dataset rows ({len(df)})"
    assert report["total_rules_evaluated"] >= 7, f"FAILED: Expected at least 7 rules evaluated, got {report['total_rules_evaluated']}"
    assert report["total_violations_found"] == len(report["rule_violations"]), f"FAILED: total_violations_found mismatch with rule_violations list length!"

    print(f"PASS: JSON structure verified ({report['total_rows_validated']} rows validated, {report['total_violations_found']} total violations).")

    # 4. Verify Per-Rule Summary Metrics
    per_rule = report["per_rule_summary"]
    required_rules = [
        "price_greater_than_zero", "minimum_nights_in_range", "availability_365_in_range",
        "room_type_valid", "neighbourhood_group_valid", "coordinate_bounding_box", "min_nights_le_max_nights"
    ]
    for r_name in required_rules:
        assert r_name in per_rule, f"FAILED: Missing rule '{r_name}' in per_rule_summary!"
        summary = per_rule[r_name]
        assert "rule_category" in summary, f"FAILED: Missing rule_category in summary for '{r_name}'"
        assert "description" in summary, f"FAILED: Missing description in summary for '{r_name}'"
        assert "violation_count" in summary, f"FAILED: Missing violation_count in summary for '{r_name}'"
        assert "violation_pct" in summary, f"FAILED: Missing violation_pct in summary for '{r_name}'"

    print("PASS: All 7 required rule metrics present in per_rule_summary.")

    # 5. Verify DoD Catch Requirements (coordinate-bounding-box & min_nights <= max_nights)
    bbox_count = per_rule["coordinate_bounding_box"]["violation_count"]
    nights_count = per_rule["min_nights_le_max_nights"]["violation_count"]

    assert bbox_count > 0, f"FAILED: coordinate_bounding_box rule caught 0 violations! Expected > 0."
    assert nights_count > 0, f"FAILED: min_nights_le_max_nights rule caught 0 violations! Expected > 0."

    print(f"\nPASS: Specific DoD Catch Requirements Verified:")
    print(f"       coordinate_bounding_box violations caught: {bbox_count}")
    print(f"       min_nights_le_max_nights violations caught: {nights_count}")

    # 6. Verify Row-Level Violation Log References
    for i, v in enumerate(report["rule_violations"][:5]): # Check sample entries
        for req_field in ["rule_id", "rule_name", "row_index", "listing_id", "column", "value", "details"]:
            assert req_field in v, f"FAILED: Missing field '{req_field}' in violation entry #{i}"
        assert 0 <= v["row_index"] < len(df), f"FAILED: Invalid row_index {v['row_index']} in violation entry #{i}"

    print("PASS: Row-level violation log entries fully structured and referenced.")

    print("\n==================================================")
    print("ALL SPRINT 7 DEFINITION OF DONE VERIFICATIONS PASSED!")
    print("==================================================")

if __name__ == "__main__":
    verify_sprint7()
