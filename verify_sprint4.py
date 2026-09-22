import os
import json
import pandas as pd

def verify_sprint4():
    print("==================================================")
    print("SPRINT 4 DEFINITION OF DONE VERIFICATION")
    print("==================================================")

    # 1. Check parquet ground truth dataset loaded
    parquet_path = "raw_data_loaded.parquet"
    assert os.path.exists(parquet_path), f"FAILED: Ground truth parquet '{parquet_path}' missing!"
    df = pd.read_parquet(parquet_path)
    total_rows = len(df)
    total_cols = len(df.columns)
    print(f"PASS: Ground truth dataset verified ({total_rows} rows, {total_cols} columns).")

    # 2. Verify required deliverables exist and are non-empty
    required_files = ["schema.json", "quality_score_before.json"]
    for file_path in required_files:
        assert os.path.exists(file_path), f"FAILED: Deliverable '{file_path}' missing!"
        assert os.path.getsize(file_path) > 0, f"FAILED: Deliverable '{file_path}' is empty (0 bytes)!"
    print(f"PASS: All required deliverables exist and are non-empty: {required_files}.")

    # 3. Verify schema.json structure and rules
    with open("schema.json", "r", encoding="utf-8") as f:
        schema = json.load(f)

    required_schema_top_keys = ["schema_version", "dataset_name", "total_columns", "key_columns", "columns"]
    for k in required_schema_top_keys:
        assert k in schema, f"FAILED: Missing key '{k}' in schema.json"

    cols_dict = schema["columns"]
    assert len(cols_dict) == total_cols, f"FAILED: schema.json column count ({len(cols_dict)}) != dataset ({total_cols})"

    # Key columns checking
    required_key_cols = ["id", "price", "minimum_nights", "availability_365", "room_type", "bathrooms_text", "latitude", "longitude"]
    for kc in required_key_cols:
        assert kc in cols_dict, f"FAILED: Key column '{kc}' missing from schema.json columns!"
        assert cols_dict[kc].get("is_key_column") is True, f"FAILED: '{kc}' should have is_key_column: True"

    # Price spec check
    price_spec = cols_dict["price"]
    assert price_spec["min_value"] >= 0.0, "FAILED: Price min_value must be >= 0.0"
    assert "format_pattern" in price_spec, "FAILED: Price missing format_pattern regex"
    print("PASS: 'price' schema rules verified (min_value >= 0.0, format pattern).")

    # Minimum nights spec check
    min_nights_spec = cols_dict["minimum_nights"]
    assert min_nights_spec["min_value"] == 1, "FAILED: minimum_nights min_value should be 1"
    assert min_nights_spec["max_value"] == 365, "FAILED: minimum_nights max_value should be 365"
    print("PASS: 'minimum_nights' schema rules verified (range 1 to 365).")

    # Availability 365 spec check
    avail_spec = cols_dict["availability_365"]
    assert avail_spec["min_value"] == 0, "FAILED: availability_365 min_value should be 0"
    assert avail_spec["max_value"] == 365, "FAILED: availability_365 max_value should be 365"
    print("PASS: 'availability_365' schema rules verified (range 0 to 365).")

    # Room type spec check
    room_spec = cols_dict["room_type"]
    allowed_set = set(room_spec.get("allowed_values", []))
    expected_set = {"Entire home/apt", "Private room", "Hotel room", "Shared room"}
    assert expected_set.issubset(allowed_set), f"FAILED: room_type allowed_values set missing expected values. Got {allowed_set}"
    print(f"PASS: 'room_type' schema rules verified (allowed set: {allowed_set}).")

    # 4. Verify quality_score_before.json structure and sub-scores
    with open("quality_score_before.json", "r", encoding="utf-8") as f:
        score_data = json.load(f)

    required_score_top_keys = ["timestamp", "dataset_summary", "overall_quality_score", "sub_scores", "sub_score_details"]
    for k in required_score_top_keys:
        assert k in score_data, f"FAILED: Missing key '{k}' in quality_score_before.json"

    sub_scores = score_data["sub_scores"]
    required_sub_scores = ["completeness_score", "consistency_score", "validity_score"]
    for ss in required_sub_scores:
        assert ss in sub_scores, f"FAILED: Missing sub-score '{ss}' in sub_scores"
        val = sub_scores[ss]
        assert isinstance(val, (int, float)), f"FAILED: Sub-score '{ss}' must be numeric"
        assert 0.0 <= val <= 100.0, f"FAILED: Sub-score '{ss}' ({val}) out of 0-100 range"

    overall_score = score_data["overall_quality_score"]
    assert isinstance(overall_score, (int, float)), "FAILED: overall_quality_score must be numeric"
    assert 0.0 <= overall_score <= 100.0, f"FAILED: overall_quality_score ({overall_score}) out of 0-100 range"

    # Verify sub_score_details
    details = score_data["sub_score_details"]
    assert "completeness" in details, "FAILED: Missing completeness details"
    assert "consistency" in details, "FAILED: Missing consistency details"
    assert "validity" in details, "FAILED: Missing validity details"

    print(f"PASS: quality_score_before.json verified.")
    print(f"       Overall Score: {overall_score}/100")
    print(f"       Completeness Sub-score: {sub_scores['completeness_score']}/100")
    print(f"       Consistency Sub-score:  {sub_scores['consistency_score']}/100")
    print(f"       Validity Sub-score:     {sub_scores['validity_score']}/100")

    print("\n==================================================")
    print("ALL SPRINT 4 DEFINITION OF DONE VERIFICATIONS PASSED!")
    print("==================================================")

if __name__ == "__main__":
    verify_sprint4()
