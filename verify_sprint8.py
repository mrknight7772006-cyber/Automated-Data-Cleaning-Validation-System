import os
import json
import pandas as pd

def verify_sprint8():
    print("==================================================")
    print("SPRINT 8 DEFINITION OF DONE VERIFICATION")
    print("==================================================")

    detector_script = "anomaly_detector.py"
    nlp_script = "nlp_column_classifier.py"
    anomaly_report_path = "anomaly_report.json"
    nlp_report_path = "nlp_classification_report.json"
    dataset_path = "cleaned_data.csv"

    # 1. Deliverables existence & non-empty check
    for fp in [detector_script, nlp_script, anomaly_report_path, nlp_report_path]:
        assert os.path.exists(fp), f"FAILED: Deliverable '{fp}' missing!"
        assert os.path.getsize(fp) > 0, f"FAILED: Deliverable '{fp}' is empty (0 bytes)!"

    print(f"PASS: Deliverables exist and are non-empty: {detector_script}, {nlp_script}, {anomaly_report_path}, {nlp_report_path}.")

    # 2. Verify Anomaly Report Schema & DoD Criteria
    with open(anomaly_report_path, "r", encoding="utf-8") as f:
        anom_report = json.load(f)

    req_anom_keys = ["timestamp", "dataset_path", "features_evaluated", "summary", "overlap_statistics", "flagged_rows"]
    for k in req_anom_keys:
        assert k in anom_report, f"FAILED: Missing key '{k}' in {anomaly_report_path}"

    expected_features = ['price', 'minimum_nights', 'availability_365', 'number_of_reviews']
    for f_col in expected_features:
        assert f_col in anom_report["features_evaluated"], f"FAILED: Feature '{f_col}' missing from evaluated features!"

    sum_anom = anom_report["summary"]
    assert sum_anom["isolation_forest_anomalies_count"] > 0, "FAILED: Isolation Forest caught 0 anomalies!"
    assert sum_anom["local_outlier_factor_anomalies_count"] > 0, "FAILED: LOF caught 0 anomalies!"
    assert sum_anom["autoencoder_anomalies_count"] > 0, "FAILED: Autoencoder caught 0 anomalies!"

    overlap = anom_report["overlap_statistics"]
    assert "both_if_and_lof_count" in overlap, "FAILED: Missing 'both_if_and_lof_count' in overlap_statistics!"
    assert "jaccard_similarity_if_lof" in overlap, "FAILED: Missing 'jaccard_similarity_if_lof' in overlap_statistics!"
    assert overlap["both_if_and_lof_count"] > 0, "FAILED: Overlap count between IF and LOF is 0!"

    print(f"PASS: Anomaly Detector verified:")
    print(f"       Isolation Forest Anomalies: {sum_anom['isolation_forest_anomalies_count']}")
    print(f"       Local Outlier Factor Anomalies: {sum_anom['local_outlier_factor_anomalies_count']}")
    print(f"       Autoencoder Anomalies: {sum_anom['autoencoder_anomalies_count']}")
    print(f"       IF & LOF Overlap Count: {overlap['both_if_and_lof_count']} (Jaccard: {overlap['jaccard_similarity_if_lof']:.4f})")

    # Check flagged rows detail structure
    for i, r in enumerate(anom_report["flagged_rows"][:5]):
        assert "row_index" in r and "listing_id" in r and "feature_values" in r, f"FAILED: Incomplete flagged row #{i}"
        assert "scores" in r and "flagged_by" in r, f"FAILED: Incomplete scores/flagged_by in row #{i}"
        assert "isolation_forest" in r["flagged_by"] and "local_outlier_factor" in r["flagged_by"], f"FAILED: Missing model flags in row #{i}"

    print("PASS: Anomaly report flagged rows detailed structure fully verified.")

    # 3. Verify NLP Classification Report Schema & DoD Criteria
    with open(nlp_report_path, "r", encoding="utf-8") as f:
        nlp_report = json.load(f)

    req_nlp_keys = ["timestamp", "dataset_path", "total_rows_evaluated", "text_classification_summary", "amenities_parsing_report", "flagged_text_issues"]
    for k in req_nlp_keys:
        assert k in nlp_report, f"FAILED: Missing key '{k}' in {nlp_report_path}"

    txt_sum = nlp_report["text_classification_summary"]
    assert "description" in txt_sum, "FAILED: Missing 'description' in text_classification_summary!"
    assert "neighborhood_overview" in txt_sum, "FAILED: Missing 'neighborhood_overview' in text_classification_summary!"

    desc_cats = txt_sum["description"]["counts_by_category"]
    assert desc_cats.get("GENUINE_LISTING_TEXT", 0) > 0, "FAILED: 0 genuine descriptions found!"
    assert desc_cats.get("MISSING_OR_NULL", 0) > 0, "FAILED: 0 missing descriptions found!"
    assert (desc_cats.get("NEAR_EMPTY_OR_SHORT", 0) + desc_cats.get("PLACEHOLDER", 0) + desc_cats.get("MISSING_OR_NULL", 0)) > 0, "FAILED: 0 placeholder/near-empty descriptions flagged!"

    print(f"PASS: NLP Text Classification verified:")
    print(f"       Genuine Descriptions: {desc_cats.get('GENUINE_LISTING_TEXT', 0)}")
    print(f"       Missing Descriptions: {desc_cats.get('MISSING_OR_NULL', 0)}")
    print(f"       Near-Empty / Short Descriptions: {desc_cats.get('NEAR_EMPTY_OR_SHORT', 0)}")
    print(f"       Placeholder Descriptions: {desc_cats.get('PLACEHOLDER', 0)}")

    amen_report = nlp_report["amenities_parsing_report"]
    assert amen_report["summary"]["total_processed"] == len(pd.read_csv(dataset_path, low_memory=False)), "FAILED: Amenities processed count mismatch!"
    assert amen_report["metrics"]["avg_amenities_per_listing"] > 0, "FAILED: Average amenities per listing is 0!"
    assert len(amen_report["top_20_amenities"]) > 0, "FAILED: top_20_amenities is empty!"

    print(f"PASS: Amenities Parsing & Standardization verified:")
    print(f"       Average Amenities per Listing: {amen_report['metrics']['avg_amenities_per_listing']}")
    print(f"       Unique Standardized Amenities: {amen_report['metrics']['total_unique_standardized_amenities']}")
    print(f"       Top 3 Amenities: {list(amen_report['top_20_amenities'].items())[:3]}")

    print("\n==================================================")
    print("ALL SPRINT 8 DEFINITION OF DONE VERIFICATIONS PASSED!")
    print("==================================================")

if __name__ == "__main__":
    verify_sprint8()
