import os
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor

def detect_anomalies(input_path='cleaned_data.csv', output_path='anomaly_report.json', contamination=0.05):
    print("==================================================")
    print("RUNNING ANOMALY DETECTOR (IF + LOF + AUTOENCODER)")
    print("==================================================")
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input dataset not found at: {input_path}")
        
    df = pd.read_csv(input_path, low_memory=False)
    print(f"Loaded dataset from {input_path} with {len(df)} rows.")
    
    feature_cols = ['price', 'minimum_nights', 'availability_365', 'number_of_reviews']
    for col in feature_cols:
        if col not in df.columns:
            raise KeyError(f"Missing required feature column: {col}")
            
    X = df[feature_cols].copy()
    X = X.fillna(X.median())
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 1. Isolation Forest
    if_model = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
    if_preds = if_model.fit_predict(X_scaled) # -1 for anomaly, 1 for normal
    if_scores = if_model.decision_function(X_scaled) # lower = more anomalous
    if_score_norm = (if_scores.max() - if_scores) / (if_scores.max() - if_scores.min() + 1e-9)
    is_anomaly_if = (if_preds == -1)
    
    # 2. Local Outlier Factor (LOF)
    lof_model = LocalOutlierFactor(n_neighbors=20, contamination=contamination, novelty=False)
    lof_preds = lof_model.fit_predict(X_scaled) # -1 for anomaly, 1 for normal
    lof_scores = lof_model.negative_outlier_factor_ # lower = more anomalous
    lof_score_norm = (lof_scores.max() - lof_scores) / (lof_scores.max() - lof_scores.min() + 1e-9)
    is_anomaly_lof = (lof_preds == -1)
    
    # 3. Autoencoder (MLP Reconstruction Error Stretch Goal)
    ae_model = MLPRegressor(hidden_layer_sizes=(4, 2, 4), max_iter=500, random_state=42, activation='relu')
    ae_model.fit(X_scaled, X_scaled)
    X_reconstructed = ae_model.predict(X_scaled)
    ae_mse = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)
    ae_threshold = np.percentile(ae_mse, (1 - contamination) * 100)
    is_anomaly_ae = (ae_mse >= ae_threshold)
    ae_score_norm = (ae_mse - ae_mse.min()) / (ae_mse.max() - ae_mse.min() + 1e-9)
    
    total_rows = len(df)
    if_count = int(np.sum(is_anomaly_if))
    lof_count = int(np.sum(is_anomaly_lof))
    ae_count = int(np.sum(is_anomaly_ae))
    
    both_if_lof = int(np.sum(is_anomaly_if & is_anomaly_lof))
    all_three = int(np.sum(is_anomaly_if & is_anomaly_lof & is_anomaly_ae))
    union_if_lof = int(np.sum(is_anomaly_if | is_anomaly_lof))
    
    jaccard_if_lof = float(both_if_lof / union_if_lof) if union_if_lof > 0 else 0.0
    pct_if_in_lof = float(both_if_lof / if_count * 100) if if_count > 0 else 0.0
    pct_lof_in_if = float(both_if_lof / lof_count * 100) if lof_count > 0 else 0.0
    
    price_95 = df['price'].quantile(0.95)
    min_nights_95 = df['minimum_nights'].quantile(0.95)
    avail_95 = df['availability_365'].quantile(0.95)
    reviews_95 = df['number_of_reviews'].quantile(0.95)
    
    flagged_rows = []
    for idx in range(total_rows):
        is_if = bool(is_anomaly_if[idx])
        is_lof = bool(is_anomaly_lof[idx])
        is_ae = bool(is_anomaly_ae[idx])
        
        if is_if or is_lof or is_ae:
            listing_id = int(df.loc[idx, 'id']) if 'id' in df.columns and pd.notna(df.loc[idx, 'id']) else idx
            row_features = {
                'price': float(df.loc[idx, 'price']),
                'minimum_nights': float(df.loc[idx, 'minimum_nights']),
                'availability_365': int(df.loc[idx, 'availability_365']),
                'number_of_reviews': int(df.loc[idx, 'number_of_reviews'])
            }
            
            reasons = []
            if row_features['price'] > price_95:
                reasons.append(f"Price (${row_features['price']:.2f}) exceeds 95th percentile (${price_95:.2f})")
            if row_features['minimum_nights'] > min_nights_95:
                reasons.append(f"Minimum nights ({row_features['minimum_nights']:.0f}) exceeds 95th percentile ({min_nights_95:.0f})")
            if row_features['number_of_reviews'] > reviews_95:
                reasons.append(f"Reviews count ({row_features['number_of_reviews']}) exceeds 95th percentile ({reviews_95:.0f})")
            if not reasons:
                reasons.append("Multi-dimensional feature outlier")
                
            flagged_rows.append({
                'row_index': idx,
                'listing_id': listing_id,
                'feature_values': row_features,
                'scores': {
                    'isolation_forest_decision_score': float(if_scores[idx]),
                    'isolation_forest_anomaly_score': float(if_score_norm[idx]),
                    'lof_negative_factor': float(lof_scores[idx]),
                    'lof_anomaly_score': float(lof_score_norm[idx]),
                    'autoencoder_mse_loss': float(ae_mse[idx]),
                    'autoencoder_anomaly_score': float(ae_score_norm[idx])
                },
                'flagged_by': {
                    'isolation_forest': is_if,
                    'local_outlier_factor': is_lof,
                    'autoencoder': is_ae
                },
                'anomaly_consensus': bool((is_if + is_lof + is_ae) >= 2),
                'anomaly_reasons': reasons
            })
            
    report = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'dataset_path': input_path,
        'features_evaluated': feature_cols,
        'parameters': {
            'contamination': contamination,
            'if_n_estimators': 100,
            'lof_n_neighbors': 20,
            'autoencoder_architecture': 'MLP (4-2-4)'
        },
        'summary': {
            'total_rows_evaluated': total_rows,
            'isolation_forest_anomalies_count': if_count,
            'isolation_forest_anomalies_pct': round(float(if_count / total_rows * 100), 2),
            'local_outlier_factor_anomalies_count': lof_count,
            'local_outlier_factor_anomalies_pct': round(float(lof_count / total_rows * 100), 2),
            'autoencoder_anomalies_count': ae_count,
            'autoencoder_anomalies_pct': round(float(ae_count / total_rows * 100), 2),
            'total_union_flagged_count': union_if_lof,
            'total_consensus_anomalies_count': sum(1 for r in flagged_rows if r['anomaly_consensus'])
        },
        'overlap_statistics': {
            'both_if_and_lof_count': both_if_lof,
            'all_three_models_count': all_three,
            'jaccard_similarity_if_lof': round(jaccard_if_lof, 4),
            'pct_if_anomalies_in_lof': round(pct_if_in_lof, 2),
            'pct_lof_anomalies_in_if': round(pct_lof_in_if, 2)
        },
        'flagged_rows': flagged_rows
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
        
    print(f"PASS: Isolation Forest flagged {if_count} anomalies.")
    print(f"PASS: Local Outlier Factor flagged {lof_count} anomalies.")
    print(f"PASS: Autoencoder flagged {ae_count} anomalies.")
    print(f"PASS: Overlap between IF and LOF: {both_if_lof} rows (Jaccard: {jaccard_if_lof:.4f}).")
    print(f"Report written to {output_path}.")
    return report

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AI Anomaly Detector using IF, LOF, and Autoencoder')
    parser.add_argument('--input', default='cleaned_data.csv', help='Input dataset CSV')
    parser.add_argument('--output', default='anomaly_report.json', help='Output JSON report path')
    parser.add_argument('--contamination', type=float, default=0.05, help='Contamination factor (0.0 to 0.5)')
    args = parser.parse_args()
    detect_anomalies(args.input, args.output, args.contamination)
