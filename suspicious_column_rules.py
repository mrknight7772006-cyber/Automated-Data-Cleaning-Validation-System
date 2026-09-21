import pandas as pd
import numpy as np
import re
from typing import Dict, Any, List

def flag_pii_columns(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Identifies PII (Personally Identifiable Information) and location-sensitive columns:
    Required flags: host_name, host_id, latitude, longitude
    Additional PII: host_url, listing_url, host_location, host_about, host_thumbnail_url, host_picture_url
    """
    pii_definitions = {
        "host_name": {"category": "Direct Identifier (Name)", "reason": "Contains full or first name of listing host"},
        "host_id": {"category": "Direct Identifier (ID)", "reason": "Unique system identifier for host identity"},
        "latitude": {"category": "Location PII", "reason": "Exact GPS latitude coordinate revealing physical address"},
        "longitude": {"category": "Location PII", "reason": "Exact GPS longitude coordinate revealing physical address"},
        "host_url": {"category": "Direct Identifier (URL)", "reason": "Link to host profile entity"},
        "listing_url": {"category": "Location PII (URL)", "reason": "Direct web link to specific property"},
        "host_location": {"category": "Location PII", "reason": "Free-form text of host home/residence location"},
        "host_about": {"category": "Quasi-Identifier / Free Text", "reason": "Bio text containing potential PII details"},
        "host_thumbnail_url": {"category": "Biometric / Media PII", "reason": "Image URL of host face profile picture"},
        "host_picture_url": {"category": "Biometric / Media PII", "reason": "Image URL of host face profile picture"}
    }
    
    flagged_columns = [col for col in df.columns if col.lower() in pii_definitions]
    details = {col: pii_definitions[col.lower()] for col in flagged_columns}
    
    return {
        "flagged": len(flagged_columns) > 0,
        "pii_columns_found": flagged_columns,
        "required_pii_checked": ["host_name", "host_id", "latitude", "longitude"],
        "all_required_pii_present": all(col in flagged_columns for col in ["host_name", "host_id", "latitude", "longitude"]),
        "details": details
    }

def flag_inconsistent_formats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Flags inconsistent string formatting in columns such as bathrooms_text:
    Checks for mixed numeric and textual formatting (e.g. "1.5 baths", "Half-bath", "1 private bath", nulls).
    """
    inconsistencies = {}
    
    if "bathrooms_text" in df.columns:
        series = df["bathrooms_text"]
        null_count = int(series.isna().sum())
        non_null = series.dropna().astype(str).str.strip()
        
        # Categorize value patterns
        patterns = {
            "standard_numeric_bath": 0,    # e.g., "1 bath", "1.5 baths", "2 baths"
            "half_bath_text": 0,          # e.g., "Half-bath", "Shared half-bath", "Private half-bath"
            "shared_or_private_text": 0,  # e.g., "1 shared bath", "1 private bath"
            "other_unclassified": 0
        }
        
        sample_variations = non_null.unique()[:10].tolist()
        
        for val in non_null:
            v_lower = val.lower()
            if "half" in v_lower:
                patterns["half_bath_text"] += 1
            elif "shared" in v_lower or "private" in v_lower:
                patterns["shared_or_private_text"] += 1
            elif re.match(r'^\d+(\.\d+)?\s+(bath|baths)$', v_lower):
                patterns["standard_numeric_bath"] += 1
            else:
                patterns["other_unclassified"] += 1
                
        inconsistencies["bathrooms_text"] = {
            "flagged": True,
            "reason": "Mixed numeric and textual values requiring parsing and normalization (e.g. '1.5 baths' vs 'Half-bath' vs 'Shared half-bath')",
            "total_rows": len(df),
            "null_count": null_count,
            "unique_patterns_count": int(series.nunique()),
            "pattern_breakdown": patterns,
            "sample_variations": sample_variations
        }
        
    return {
        "flagged": len(inconsistencies) > 0,
        "columns": inconsistencies
    }

def flag_suspicious_rows(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Flags rows with suspicious data anomalies:
    1. price = "$0.00" (or 0)
    2. minimum_nights > 365
    """
    total_rows = len(df)
    
    # 1. price = "$0.00"
    zero_price_count = 0
    zero_price_indices = []
    if "price" in df.columns:
        price_series = df["price"].astype(str).str.strip()
        # Match "$0.00", "$0", "0", "0.0", "$0.0"
        zero_mask = price_series.isin(["$0.00", "$0", "0", "0.0", "$0.0"]) | (price_series.str.replace(r'[\$,]', '', regex=True).str.strip() == "0.00")
        zero_price_count = int(zero_mask.sum())
        zero_price_indices = df[zero_mask].index.tolist()[:10]  # first 10 sample indices
        
    zero_price_pct = float(round((zero_price_count / total_rows) * 100, 4)) if total_rows > 0 else 0.0

    # 2. minimum_nights > 365
    extreme_min_nights_count = 0
    extreme_min_nights_indices = []
    if "minimum_nights" in df.columns:
        min_nights_series = pd.to_numeric(df["minimum_nights"], errors="coerce")
        extreme_mask = min_nights_series > 365
        extreme_min_nights_count = int(extreme_mask.sum())
        extreme_min_nights_indices = df[extreme_mask].index.tolist()[:10]
        
    extreme_min_nights_pct = float(round((extreme_min_nights_count / total_rows) * 100, 4)) if total_rows > 0 else 0.0

    return {
        "zero_price_rows": {
            "flagged": zero_price_count > 0,
            "count": zero_price_count,
            "pct": zero_price_pct,
            "criteria": 'price == "$0.00"',
            "sample_row_indices": zero_price_indices
        },
        "extreme_minimum_nights": {
            "flagged": extreme_min_nights_count > 0,
            "count": extreme_min_nights_count,
            "pct": extreme_min_nights_pct,
            "criteria": 'minimum_nights > 365',
            "sample_row_indices": extreme_min_nights_indices
        }
    }

def run_suspicious_rules(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Executes all suspicious rule evaluations on the DataFrame.
    """
    pii_results = flag_pii_columns(df)
    format_results = flag_inconsistent_formats(df)
    suspicious_row_results = flag_suspicious_rows(df)
    
    return {
        "pii_columns": pii_results,
        "format_inconsistencies": format_results,
        "suspicious_rows": suspicious_row_results
    }

if __name__ == "__main__":
    from load_data import load_data
    df = load_data()
    rules_report = run_suspicious_rules(df)
    import json
    print(json.dumps(rules_report, indent=2))
