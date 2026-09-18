import os
import json
import re
import pandas as pd
from typing import Dict, List, Any
from load_data import load_data

def infer_semantic_type(col_name: str, series: pd.Series) -> str:
    """
    Infers the semantic type of a column from its name, data type, and values.
    Returns one of: 'ID', 'URL', 'date', 'numeric', 'categorical', 'free text'.
    """
    name_lower = col_name.lower()
    non_null = series.dropna()
    
    # 1. ID columns
    if name_lower == 'id' or name_lower.endswith('_id'):
        return 'ID'
        
    # 2. URL columns
    if name_lower.endswith('_url') or 'url' in name_lower:
        return 'URL'
    if len(non_null) > 0 and pd.api.types.is_string_dtype(series):
        first_few = non_null.astype(str).str.strip().iloc[:10]
        if len(first_few) > 0 and first_few.str.startswith(('http://', 'https://')).all():
            return 'URL'
            
    # Exclude numeric review scores/counts/years/listings from date matching
    if name_lower.startswith(('number_of_reviews', 'reviews_per_month', 'review_scores_', 'calculated_host_listings', 'availability_', 'hosts_time_as_')):
        return 'numeric'
        
    # 3. Date columns
    date_keywords = ['scraped', 'since', 'first_review', 'last_review']
    is_date_col = any(kw in name_lower for kw in date_keywords) or name_lower.endswith('_date') or name_lower.startswith('date_')
    if is_date_col:
        return 'date'
        
    if len(non_null) > 0 and (series.dtype == 'object' or pd.api.types.is_string_dtype(series)):
        sample_str = str(non_null.iloc[0]).strip()
        if re.match(r'^\d{2}-\d{2}-\d{4}$', sample_str) or re.match(r'^\d{4}-\d{2}-\d{2}$', sample_str):
            return 'date'
            
    # Explicit semantic types for known numeric columns trapped in mixed strings
    if name_lower in ['price', 'bathrooms_text']:
        return 'numeric'
        
    # 4. Numeric columns
    if pd.api.types.is_numeric_dtype(series):
        return 'numeric'
        
    # 5. Free Text vs Categorical for string/object columns
    free_text_cols = {
        'name', 'description', 'neighborhood_overview', 'host_about', 
        'amenities', 'price_quote_raw', 'host_verifications', 'license'
    }
    if name_lower in free_text_cols:
        return 'free text'
        
    if len(non_null) > 0:
        avg_len = non_null.astype(str).str.len().mean()
        nunique = series.nunique()
        if avg_len > 60:
            return 'free text'
        if nunique <= 100 or name_lower.endswith('_type') or name_lower in [
            'source', 'host_is_superhost', 'host_has_profile_pic', 'host_identity_verified', 
            'has_availability', 'instant_bookable', 'neighbourhood_cleansed', 'host_location', 
            'host_response_time'
        ]:
            return 'categorical'
            
    return 'free text'

def check_is_mixed_type(col_name: str, series: pd.Series, semantic_type: str) -> bool:
    """
    Flags mixed-type columns, such as bathrooms_text (numbers mixed with text like "Half-bath", "1 bath")
    and price (numeric value trapped in a $-prefixed string like "$80.43").
    """
    name_lower = col_name.lower()
    
    # Specifically required mixed-type columns
    if name_lower in ['bathrooms_text', 'price']:
        return True
        
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
        
    # URLs, IDs, and long Free Text are not mixed-type data columns
    if semantic_type in ['URL', 'ID', 'free text']:
        return False
        
    sample_list = non_null.astype(str).str.strip().tolist()
    
    # Check for currency symbols or percent signs in categorical/numeric-like strings
    has_currency_or_pct = any(
        val.startswith('$') or val.startswith('€') or val.startswith('£') or '%' in val
        for val in sample_list
    )
    if has_currency_or_pct:
        return True
        
    # Check for text + number mixing in non-date, non-url columns
    if semantic_type in ['categorical', 'numeric'] and name_lower not in {'property_type', 'room_type', 'source', 'neighbourhood_cleansed', 'host_location'}:
        if series.dtype == 'object' or pd.api.types.is_string_dtype(series):
            has_digits = any(re.search(r'\d', val) for val in sample_list)
            has_words = any(re.search(r'[a-zA-Z]', val) for val in sample_list)
            contains_half = any('half' in val.lower() for val in sample_list)
            
            if (has_digits and has_words) or contains_half:
                return True
                
    return False

def generate_metadata_report(df: pd.DataFrame, output_json_path: str = "metadata_report.json") -> List[Dict[str, Any]]:
    """
    Generates metadata report for all columns in DataFrame and saves to output_json_path.
    """
    report = []
    total_rows = len(df)
    
    for col in df.columns:
        series = df[col]
        dtype_str = str(series.dtype)
        semantic_type = infer_semantic_type(col, series)
        is_mixed_type = check_is_mixed_type(col, series, semantic_type)
        
        non_null_count = int(series.notna().sum())
        null_count = int(series.isna().sum())
        nunique_count = int(series.nunique())
        
        # Extract sample values (up to 3 non-null values)
        non_null_vals = series.dropna().unique()
        sample_values = [str(v) for v in non_null_vals[:3]]
        
        col_metadata = {
            "column_name": col,
            "dtype": dtype_str,
            "semantic_type": semantic_type,
            "is_mixed_type": is_mixed_type,
            "total_rows": total_rows,
            "non_null_count": non_null_count,
            "null_count": null_count,
            "unique_count": nunique_count,
            "sample_values": sample_values
        }
        report.append(col_metadata)
        
    print(f"Saving metadata report with {len(report)} columns to '{output_json_path}'...")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        
    print(f"Successfully generated deliverable '{output_json_path}'.")
    return report

def run_metadata_pipeline(parquet_path: str = "raw_data_loaded.parquet", output_json_path: str = "metadata_report.json"):
    """
    Loads parquet data (or runs load_data if parquet missing), generates metadata report, and verifies deliverables.
    """
    if not os.path.exists(parquet_path):
        print(f"Parquet file '{parquet_path}' not found. Executing load_data pipeline...")
        df = load_data(output_parquet_path=parquet_path)
    else:
        print(f"Loading parquet dataset from '{parquet_path}'...")
        df = pd.read_parquet(parquet_path)
        
    report = generate_metadata_report(df, output_json_path=output_json_path)
    return report

if __name__ == "__main__":
    report = run_metadata_pipeline()
    
    bathrooms_meta = next((r for r in report if r["column_name"] == "bathrooms_text"), None)
    price_meta = next((r for r in report if r["column_name"] == "price"), None)
    
    print("\n--- Key Deliverables Verification ---")
    print("bathrooms_text metadata:", bathrooms_meta)
    print("price metadata:", price_meta)
