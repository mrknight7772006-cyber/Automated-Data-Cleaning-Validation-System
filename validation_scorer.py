import numpy as np

class ValidationScorer:
    """
    Validation Scorer Engine.
    Aggregates rule violations, AI anomalies, NLP quality issues, and statistical errors.
    Assigns severity scores and computes an Overall Health Score (0-100) along with sub-scores.
    """
    SEVERITY_WEIGHTS = {
        "CRITICAL": 15.0,
        "HIGH": 10.0,
        "MEDIUM": 5.0,
        "LOW": 2.0
    }
    
    def score_dataset(self, total_rows, rule_violations, anomaly_flagged_rows, nlp_issues, statistical_errors):
        critical_issues = []
        high_issues = []
        medium_issues = []
        low_issues = []
        
        row_penalties = {i: 0.0 for i in range(total_rows)}
        
        for rv in rule_violations:
            r_name = rv.get("rule_name", "")
            r_idx = rv.get("row_index", 0)
            if r_name in ["price_greater_than_zero", "coordinate_bounding_box", "min_nights_le_max_nights"]:
                sev = "CRITICAL"
                critical_issues.append(rv)
            else:
                sev = "HIGH"
                high_issues.append(rv)
            row_penalties[r_idx] += self.SEVERITY_WEIGHTS[sev]
            
        for se in statistical_errors:
            r_idx = se.get("row_index", 0)
            e_type = se.get("error_type", "")
            if e_type in ["IMPOSSIBLE_PRICE", "IMPOSSIBLE_MIN_NIGHTS", "MIN_NIGHTS_GREATER_THAN_MAX"]:
                sev = "CRITICAL"
                critical_issues.append(se)
            else:
                sev = "HIGH"
                high_issues.append(se)
            row_penalties[r_idx] += self.SEVERITY_WEIGHTS[sev]
            
        for an in anomaly_flagged_rows:
            r_idx = an.get("row_index", 0)
            flagged = an.get("flagged_by", {})
            consensus = an.get("anomaly_consensus", False)
            if consensus or (flagged.get("isolation_forest") and flagged.get("local_outlier_factor")):
                sev = "HIGH"
                high_issues.append(an)
            else:
                sev = "MEDIUM"
                medium_issues.append(an)
            row_penalties[r_idx] += self.SEVERITY_WEIGHTS[sev]
            
        for nlp in nlp_issues:
            r_idx = nlp.get("row_index", 0)
            cat = nlp.get("classification_category", "")
            if cat == "MISSING_OR_NULL":
                sev = "MEDIUM"
                medium_issues.append(nlp)
            elif cat in ["PLACEHOLDER", "NEAR_EMPTY_OR_SHORT"]:
                sev = "LOW"
                low_issues.append(nlp)
            else:
                sev = "LOW"
                low_issues.append(nlp)
            row_penalties[r_idx] += self.SEVERITY_WEIGHTS[sev]
            
        rule_penalty = sum(self.SEVERITY_WEIGHTS["CRITICAL" if v.get("rule_name") in ["price_greater_than_zero", "coordinate_bounding_box", "min_nights_le_max_nights"] else "HIGH"] for v in rule_violations)
        rule_subscore = max(0.0, round(100.0 - (rule_penalty / total_rows) * 20.0, 2))
        
        anomaly_penalty = sum(self.SEVERITY_WEIGHTS["HIGH" if a.get("anomaly_consensus") else "MEDIUM"] for a in anomaly_flagged_rows)
        anomaly_subscore = max(0.0, round(100.0 - (anomaly_penalty / total_rows) * 20.0, 2))
        
        nlp_penalty = sum(self.SEVERITY_WEIGHTS["MEDIUM" if n.get("classification_category") == "MISSING_OR_NULL" else "LOW"] for n in nlp_issues)
        nlp_subscore = max(0.0, round(100.0 - (nlp_penalty / total_rows) * 20.0, 2))
        
        stat_penalty = sum(self.SEVERITY_WEIGHTS["CRITICAL" if s.get("error_type") in ["IMPOSSIBLE_PRICE", "IMPOSSIBLE_MIN_NIGHTS", "MIN_NIGHTS_GREATER_THAN_MAX"] else "HIGH"] for s in statistical_errors)
        stat_subscore = max(0.0, round(100.0 - (stat_penalty / total_rows) * 20.0, 2))
        
        overall_health_score = round(
            0.35 * rule_subscore + 0.25 * anomaly_subscore + 0.20 * nlp_subscore + 0.20 * stat_subscore,
            2
        )
        
        if overall_health_score >= 90:
            health_grade = "A+"
        elif overall_health_score >= 80:
            health_grade = "A"
        elif overall_health_score >= 70:
            health_grade = "B"
        elif overall_health_score >= 60:
            health_grade = "C"
        elif overall_health_score >= 50:
            health_grade = "D"
        else:
            health_grade = "F"
            
        row_health_scores = []
        for idx in range(total_rows):
            p = row_penalties[idx]
            r_score = max(0.0, round(100.0 - p * 3.0, 2))
            row_health_scores.append(r_score)
            
        return {
            "overall_health_score": overall_health_score,
            "health_grade": health_grade,
            "sub_scores": {
                "rule_validation_score": rule_subscore,
                "anomaly_health_score": anomaly_subscore,
                "nlp_quality_score": nlp_subscore,
                "data_integrity_score": stat_subscore
            },
            "severity_summary": {
                "critical_count": len(critical_issues),
                "high_count": len(high_issues),
                "medium_count": len(medium_issues),
                "low_count": len(low_issues),
                "total_issues_count": len(critical_issues) + len(high_issues) + len(medium_issues) + len(low_issues)
            },
            "row_health_scores": row_health_scores
        }
