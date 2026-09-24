import os
import json
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from datetime import datetime, timezone

class ErrorDetector:
    """
    Statistical Error Detector & Data Drift Engine for Airbnb Dataset.
    Detects impossible numerical values, domain anomalies, and feature data drift.
    """
    def __init__(self, df):
        self.df = df.copy()
        
    def detect_statistical_errors(self):
        errors = []
        df = self.df
        
        p_q1 = df['price'].quantile(0.25)
        p_q3 = df['price'].quantile(0.75)
        p_iqr = p_q3 - p_q1
        p_upper = p_q3 + 3.0 * p_iqr
        
        mn_q1 = df['minimum_nights'].quantile(0.25)
        mn_q3 = df['minimum_nights'].quantile(0.75)
        mn_iqr = mn_q3 - mn_q1
        mn_upper = max(30.0, mn_q3 + 3.0 * mn_iqr)
        
        for idx in range(len(df)):
            row = df.iloc[idx]
            row_id = int(row['id']) if 'id' in df.columns and pd.notna(row['id']) else idx
            
            price = float(row['price'])
            if price <= 0:
                errors.append({
                    "row_index": idx,
                    "listing_id": row_id,
                    "error_type": "IMPOSSIBLE_PRICE",
                    "column": "price",
                    "value": price,
                    "details": "Price must be strictly greater than 0"
                })
            elif price > p_upper:
                errors.append({
                    "row_index": idx,
                    "listing_id": row_id,
                    "error_type": "EXTREME_PRICE_OUTLIER",
                    "column": "price",
                    "value": price,
                    "details": f"Price (${price:.2f}) exceeds 3x IQR upper bound (${p_upper:.2f})"
                })
                
            min_nights = float(row['minimum_nights'])
            if min_nights > 365:
                errors.append({
                    "row_index": idx,
                    "listing_id": row_id,
                    "error_type": "IMPOSSIBLE_MIN_NIGHTS",
                    "column": "minimum_nights",
                    "value": min_nights,
                    "details": f"Minimum nights ({min_nights}) exceeds maximum possible calendar days (365)"
                })
            elif min_nights > mn_upper:
                errors.append({
                    "row_index": idx,
                    "listing_id": row_id,
                    "error_type": "EXTREME_MIN_NIGHTS_OUTLIER",
                    "column": "minimum_nights",
                    "value": min_nights,
                    "details": f"Minimum nights ({min_nights}) exceeds 3x IQR threshold ({mn_upper:.0f})"
                })
                
            avail = int(row['availability_365'])
            if avail == 0 and price > 500:
                errors.append({
                    "row_index": idx,
                    "listing_id": row_id,
                    "error_type": "SUSPICIOUS_UNAVAILABLE_HIGH_PRICE",
                    "column": "availability_365",
                    "value": avail,
                    "details": f"Listing has 0 availability despite high luxury pricing (${price:.2f})"
                })
                
            if 'maximum_nights' in row and pd.notna(row['maximum_nights']):
                max_nights = float(row['maximum_nights'])
                if min_nights > max_nights:
                    errors.append({
                        "row_index": idx,
                        "listing_id": row_id,
                        "error_type": "MIN_NIGHTS_GREATER_THAN_MAX",
                        "column": "minimum_nights",
                        "value": min_nights,
                        "details": f"Minimum nights ({min_nights}) > maximum nights ({max_nights})"
                    })
                    
        return errors
        
    def detect_data_drift(self):
        df = self.df.copy()
        if 'host_since' not in df.columns or df['host_since'].isna().all():
            return {"status": "SKIPPED", "reason": "host_since column missing or all null"}
            
        df['host_since_dt'] = pd.to_datetime(df['host_since'], errors='coerce')
        valid_df = df.dropna(subset=['host_since_dt'])
        
        if len(valid_df) < 20:
            return {"status": "SKIPPED", "reason": "Insufficient host_since date entries"}
            
        median_date = valid_df['host_since_dt'].median()
        c1 = valid_df[valid_df['host_since_dt'] <= median_date]
        c2 = valid_df[valid_df['host_since_dt'] > median_date]
        
        drift_results = {}
        num_cols = ['price', 'minimum_nights', 'availability_365', 'number_of_reviews']
        
        for col in num_cols:
            if col in df.columns:
                s1 = c1[col].dropna()
                s2 = c2[col].dropna()
                if len(s1) > 5 and len(s2) > 5:
                    ks_stat, p_val = ks_2samp(s1, s2)
                    w_dist = wasserstein_distance(s1, s2)
                    is_drift = bool(p_val < 0.05)
                    drift_results[col] = {
                        "ks_statistic": round(float(ks_stat), 4),
                        "p_value": round(float(p_val), 5),
                        "wasserstein_distance": round(float(w_dist), 4),
                        "drift_status": "DRIFT_DETECTED" if is_drift else "STABLE",
                        "cohort1_mean": round(float(s1.mean()), 2),
                        "cohort2_mean": round(float(s2.mean()), 2)
                    }
                    
        return {
            "status": "COMPLETED",
            "cohort_split_method": "host_since_median_split",
            "cohort1_size": len(c1),
            "cohort2_size": len(c2),
            "median_split_date": str(median_date.date()),
            "column_drift": drift_results
        }
