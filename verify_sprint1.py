import os
import json
import pandas as pd

def verify_sprint1():
    print("==================================================")
    print("SPRINT 1 DEFINITION OF DONE VERIFICATION")
    print("==================================================")
    
    # 1. Check parquet deliverable
    parquet_path = "raw_data_loaded.parquet"
    assert os.path.exists(parquet_path), f"FAILED: '{parquet_path}' does not exist!"
    df = pd.read_parquet(parquet_path)
    assert len(df) > 0, "FAILED: Parquet dataset is empty!"
    assert len(df.columns) >= 70, f"FAILED: Expected >=70 columns, found {len(df.columns)}"
    print(f"PASS: '{parquet_path}' verified ({len(df)} rows, {len(df.columns)} columns).")
    
    # 2. Check metadata_report.json deliverable
    json_path = "metadata_report.json"
    assert os.path.exists(json_path), f"FAILED: '{json_path}' does not exist!"
    with open(json_path, 'r', encoding='utf-8') as f:
        report = json.load(f)
        
    assert isinstance(report, list), "FAILED: metadata_report.json must be a JSON array!"
    assert len(report) == len(df.columns), f"FAILED: Report column count ({len(report)}) does not match dataset ({len(df.columns)})"
    print(f"PASS: '{json_path}' verified ({len(report)} columns listed).")
    
    # 3. Check every entry schema
    for item in report:
        assert "column_name" in item, "FAILED: Entry missing 'column_name'"
        assert "dtype" in item, "FAILED: Entry missing 'dtype'"
        assert "semantic_type" in item, "FAILED: Entry missing 'semantic_type'"
        assert "is_mixed_type" in item, "FAILED: Entry missing 'is_mixed_type'"
        assert item["semantic_type"] in ["ID", "URL", "date", "numeric", "categorical", "free text"], \
            f"FAILED: Invalid semantic_type '{item['semantic_type']}' for {item['column_name']}"
            
    print("PASS: All entries have required schema keys and valid semantic types.")
    
    # 4. Check specific column flags
    report_dict = {r["column_name"]: r for r in report}
    
    # bathrooms_text check
    assert "bathrooms_text" in report_dict, "FAILED: bathrooms_text not found in metadata report!"
    bathrooms_meta = report_dict["bathrooms_text"]
    assert bathrooms_meta["is_mixed_type"] is True, "FAILED: bathrooms_text must be flagged with is_mixed_type: True"
    print(f"PASS: 'bathrooms_text' correctly flagged as mixed-type (sample: {bathrooms_meta['sample_values']}).")
    
    # price check
    assert "price" in report_dict, "FAILED: price not found in metadata report!"
    price_meta = report_dict["price"]
    assert price_meta["is_mixed_type"] is True, "FAILED: price must be flagged with is_mixed_type: True"
    print(f"PASS: 'price' correctly flagged as mixed-type (sample: {price_meta['sample_values']}).")
    
    # ID semantic types check
    for id_col in ["id", "scrape_id", "host_id", "host_profile_id"]:
        assert report_dict[id_col]["semantic_type"] == "ID", f"FAILED: {id_col} semantic type should be 'ID'"
    print("PASS: ID columns correctly classified as 'ID'.")
    
    # URL semantic types check
    for url_col in ["listing_url", "picture_url", "host_url", "host_picture_url"]:
        assert report_dict[url_col]["semantic_type"] == "URL", f"FAILED: {url_col} semantic type should be 'URL'"
    print("PASS: URL columns correctly classified as 'URL'.")
    
    # Date semantic types check
    for date_col in ["last_scraped", "calendar_last_scraped", "first_review", "last_review"]:
        assert report_dict[date_col]["semantic_type"] == "date", f"FAILED: {date_col} semantic type should be 'date'"
    print("PASS: Date columns correctly classified as 'date'.")
    
    print("\n==================================================")
    print("ALL SPRINT 1 DEFINITION OF DONE VERIFICATIONS PASSED!")
    print("==================================================")

if __name__ == "__main__":
    verify_sprint1()
