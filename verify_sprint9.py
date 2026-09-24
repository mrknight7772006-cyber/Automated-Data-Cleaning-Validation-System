import os
import json
import pandas as pd

def verify_sprint9():
    print("==================================================")
    print("SPRINT 9 DEFINITION OF DONE VERIFICATION")
    print("==================================================")

    req_files = [
        "error_detector.py", "validation_scorer.py", "validation_api.py", "evaluate_validation.py",
        "validation_report.json", "evaluation_metrics.json", "roc_curve.png",
        "VALIDATION_README.md", "Module_3_Presentation.md"
    ]

    for fp in req_files:
        assert os.path.exists(fp), f"FAILED: Required deliverable '{fp}' missing!"
        assert os.path.getsize(fp) > 0, f"FAILED: Deliverable '{fp}' is empty (0 bytes)!"

    print(f"PASS: All 9 required deliverables exist and are non-empty.")

    # 1. Verify validation_report.json Unified Schema
    with open("validation_report.json", "r", encoding="utf-8") as f:
        val_rep = json.load(f)

    req_val_keys = [
        "timestamp", "dataset_path", "total_rows_validated", "overall_health_score",
        "health_grade", "sub_scores", "severity_summary", "rule_validation_summary",
        "anomaly_detection_summary", "nlp_classification_summary", "data_drift_report", "flagged_rows_master_log"
    ]
    for k in req_val_keys:
        assert k in val_rep, f"FAILED: Missing key '{k}' in validation_report.json"

    assert 0.0 <= val_rep["overall_health_score"] <= 100.0, "FAILED: Invalid overall_health_score!"
    assert val_rep["health_grade"] in ["A+", "A", "B", "C", "D", "F"], "FAILED: Invalid health_grade!"
    assert len(val_rep["flagged_rows_master_log"]) > 0, "FAILED: Master log is empty!"

    print(f"PASS: Unified validation_report.json schema verified:")
    print(f"       Overall Health Score: {val_rep['overall_health_score']}/100 (Grade: {val_rep['health_grade']})")
    print(f"       Sub-scores: {val_rep['sub_scores']}")
    print(f"       Total Flagged Rows in Log: {len(val_rep['flagged_rows_master_log'])}")

    # 2. Verify evaluation_metrics.json Real Metrics
    with open("evaluation_metrics.json", "r", encoding="utf-8") as f:
        eval_m = json.load(f)

    req_eval_keys = ["sample_size", "precision", "recall", "f1_score", "accuracy", "roc_auc", "confusion_matrix"]
    for k in req_eval_keys:
        assert k in eval_m, f"FAILED: Missing key '{k}' in evaluation_metrics.json"

    assert eval_m["sample_size"] >= 100, f"FAILED: Sample size {eval_m['sample_size']} < 100!"
    assert 0.0 <= eval_m["precision"] <= 1.0, "FAILED: Invalid precision metric!"
    assert 0.0 <= eval_m["recall"] <= 1.0, "FAILED: Invalid recall metric!"
    assert 0.5 <= eval_m["roc_auc"] <= 1.0, "FAILED: Invalid ROC AUC metric!"

    print(f"PASS: Real evaluation metrics verified:")
    print(f"       Sample Size: {eval_m['sample_size']}")
    print(f"       Precision: {eval_m['precision']}")
    print(f"       Recall: {eval_m['recall']}")
    print(f"       F1-Score: {eval_m['f1_score']}")
    print(f"       ROC AUC Score: {eval_m['roc_auc']}")

    # 3. Verify Module 3 Presentation & README Content
    with open("Module_3_Presentation.md", "r", encoding="utf-8") as f:
        pres_text = f.read()
        assert "# Module 3 Presentation" in pres_text, "FAILED: Presentation title missing!"
        assert "Slide 1:" in pres_text, "FAILED: Presentation slide structure missing!"

    with open("VALIDATION_README.md", "r", encoding="utf-8") as f:
        readme_text = f.read()
        assert "# Automated Data Cleaning & Validation System" in readme_text, "FAILED: README title missing!"

    print("PASS: Module_3_Presentation.md and VALIDATION_README.md structure verified.")

    print("\n==================================================")
    print("ALL SPRINT 9 DEFINITION OF DONE VERIFICATIONS PASSED!")
    print("==================================================")

if __name__ == "__main__":
    verify_sprint9()
