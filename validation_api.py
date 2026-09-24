import os
import json
import pandas as pd
from datetime import datetime, timezone
from rule_validator import RuleValidator
from anomaly_detector import detect_anomalies
from nlp_column_classifier import run_nlp_classification
from error_detector import ErrorDetector
from validation_scorer import ValidationScorer

class ValidationAPI:
    """
    Unified Data Validation API.
    Runs rule validation, AI anomaly detection, NLP column classification,
    and statistical error detection, producing a unified validation_report.json.
    """
    def run_validation_pipeline(self, input_path="cleaned_data.csv", output_path="validation_report.json", contamination=0.05):
        print("==================================================")
        print(f"RUNNING UNIFIED VALIDATION PIPELINE (Contamination={contamination})")
        print("==================================================")
        
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Dataset not found at {input_path}")
            
        df = pd.read_csv(input_path, low_memory=False)
        total_rows = len(df)
        
        validator = RuleValidator(input_path)
        rv_res = validator.validate_dataframe(df)
        if isinstance(rv_res, tuple):
            rv_report, rule_violations = rv_res
        else:
            rv_report = rv_res
            rule_violations = rv_report.get("rule_violations", [])
        
        anom_report = detect_anomalies(input_path, "anomaly_report.json", contamination=contamination)
        anomaly_flagged_rows = anom_report.get("flagged_rows", [])
        
        nlp_report = run_nlp_classification(input_path, "nlp_classification_report.json")
        nlp_issues = nlp_report.get("flagged_text_issues", [])
        
        err_detector = ErrorDetector(df)
        stat_errors = err_detector.detect_statistical_errors()
        drift_report = err_detector.detect_data_drift()
        
        scorer = ValidationScorer()
        scoring_res = scorer.score_dataset(
            total_rows=total_rows,
            rule_violations=rule_violations,
            anomaly_flagged_rows=anomaly_flagged_rows,
            nlp_issues=nlp_issues,
            statistical_errors=stat_errors
        )
        
        row_log = []
        for idx in range(total_rows):
            listing_id = int(df.loc[idx, 'id']) if 'id' in df.columns and pd.notna(df.loc[idx, 'id']) else idx
            row_rvs = [v for v in rule_violations if v.get("row_index") == idx]
            row_anoms = [a for a in anomaly_flagged_rows if a.get("row_index") == idx]
            row_nlps = [n for n in nlp_issues if n.get("row_index") == idx]
            row_errs = [e for e in stat_errors if e.get("row_index") == idx]
            
            row_health = scoring_res["row_health_scores"][idx]
            
            if row_rvs or row_anoms or row_nlps or row_errs:
                row_log.append({
                    "row_index": idx,
                    "listing_id": listing_id,
                    "row_health_score": row_health,
                    "rule_violations": row_rvs,
                    "anomalies": row_anoms,
                    "nlp_issues": row_nlps,
                    "statistical_errors": row_errs
                })
                
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset_path": input_path,
            "total_rows_validated": total_rows,
            "overall_health_score": scoring_res["overall_health_score"],
            "health_grade": scoring_res["health_grade"],
            "sub_scores": scoring_res["sub_scores"],
            "severity_summary": scoring_res["severity_summary"],
            "rule_validation_summary": {
                "total_rules_evaluated": rv_report.get("total_rules_evaluated", 7),
                "total_violations_found": rv_report.get("total_violations_found", len(rule_violations)),
                "per_rule_summary": rv_report.get("per_rule_summary", {})
            },
            "anomaly_detection_summary": anom_report.get("summary", {}),
            "anomaly_overlap_statistics": anom_report.get("overlap_statistics", {}),
            "nlp_classification_summary": nlp_report.get("text_classification_summary", {}),
            "amenities_parsing_summary": nlp_report.get("amenities_parsing_report", {}),
            "data_drift_report": drift_report,
            "flagged_rows_master_log": row_log
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
            
        print(f"Unified validation report generated at {output_path}")
        print(f"Overall Dataset Health Score: {scoring_res['overall_health_score']}/100 (Grade: {scoring_res['health_grade']})")
        return report

if __name__ == "__main__":
    api = ValidationAPI()
    api.run_validation_pipeline()
