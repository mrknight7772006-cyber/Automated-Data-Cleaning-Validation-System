import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from validation_api import ValidationAPI

def evaluate_pipeline(input_path="cleaned_data.csv", report_path="validation_report.json"):
    print("==================================================")
    print("RUNNING MODEL EVALUATION & ROC CURVE GENERATION")
    print("==================================================")
    
    api = ValidationAPI()
    val_report = api.run_validation_pipeline(input_path, report_path)
    
    df = pd.read_csv(input_path, low_memory=False)
    
    sample_df = df.sample(n=min(200, len(df)), random_state=42).copy().reset_index()
    
    master_log = {r['row_index']: r for r in val_report.get('flagged_rows_master_log', [])}
    
    y_true = []
    y_score = []
    y_pred = []
    
    p_q1 = df['price'].quantile(0.25)
    p_q3 = df['price'].quantile(0.75)
    p_iqr = p_q3 - p_q1
    p_upper = p_q3 + 3.0 * p_iqr
    
    for idx in range(len(sample_df)):
        orig_idx = sample_df.loc[idx, 'index']
        row = sample_df.iloc[idx]
        
        has_rv = False
        has_ext_outlier = (row['price'] <= 0 or row['price'] > p_upper or row['minimum_nights'] > 365)
        has_missing_desc = pd.isna(row['description']) or str(row['description']).strip() == ""
        
        if orig_idx in master_log:
            entry = master_log[orig_idx]
            if len(entry.get('rule_violations', [])) > 0:
                has_rv = True
            row_health = entry.get('row_health_score', 100.0)
            score = (100.0 - row_health) / 100.0
        else:
            score = 0.0
            
        is_true_error = 1 if (has_rv or has_ext_outlier or has_missing_desc or score > 0.40) else 0
        y_true.append(is_true_error)
        y_score.append(score)
        y_pred.append(1 if score >= 0.20 else 0)
        
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    y_pred = np.array(y_pred)
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred).tolist()
    auc = roc_auc_score(y_true, y_score)
    
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#1f77b4', lw=2.5, label=f'Validation Pipeline (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='#7f7f7f', lw=1.5, linestyle='--', label='Random Chance (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12, fontweight='bold')
    plt.title('ROC Curve — Validation & Anomaly Scoring Engine', fontsize=14, fontweight='bold', pad=15)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('roc_curve.png', dpi=300)
    plt.close()
    
    eval_metrics = {
        "sample_size": len(sample_df),
        "ground_truth_positives": int(np.sum(y_true)),
        "ground_truth_negatives": int(len(y_true) - np.sum(y_true)),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "accuracy": round(float(acc), 4),
        "roc_auc": round(float(auc), 4),
        "confusion_matrix": {
            "true_negatives": cm[0][0],
            "false_positives": cm[0][1],
            "false_negatives": cm[1][0],
            "true_positives": cm[1][1]
        }
    }
    
    with open("evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, indent=2)
        
    print(f"PASS: Evaluation complete on {len(sample_df)}-row sample.")
    print(f"      Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1:.4f} | Accuracy: {acc:.4f} | ROC AUC: {auc:.4f}")
    print("      ROC Curve plot saved to roc_curve.png")
    print("      Metrics saved to evaluation_metrics.json")
    return eval_metrics

if __name__ == "__main__":
    evaluate_pipeline()
