import json
import os
import re
import pandas as pd
from datetime import datetime
from typing import Dict, Any

def compute_quality_scores(df: pd.DataFrame, schema: Dict[str, Any], is_post_clean: bool = False) -> Dict[str, Any]:
    """
    Computes reproducible Quality Score (0-100) with 3 sub-scores:
    1. Completeness Sub-Score
    2. Consistency Sub-Score
    3. Validity Sub-Score
    Supports both raw (pre-clean) and post-cleaned data formats.
    """
    total_rows = len(df)
    total_cols = len(df.columns)
    total_cells = total_rows * total_cols

    # -------------------------------------------------------------
    # 1. COMPLETENESS SUB-SCORE
    # -------------------------------------------------------------
    overall_not_null = int(df.notna().sum().sum())
    overall_cell_completeness_pct = float(round((overall_not_null / total_cells) * 100, 2)) if total_cells > 0 else 0.0

    key_columns = schema.get("key_columns", [
        "id", "price", "minimum_nights", "availability_365", "room_type",
        "latitude", "longitude", "bathrooms_text", "license"
    ])
    
    key_not_null = sum(int(df[col].notna().sum()) for col in key_columns if col in df.columns)
    key_columns_completeness_pct = float(round((key_not_null / (len(key_columns) * total_rows)) * 100, 2)) if (len(key_columns) * total_rows) > 0 else 0.0

    completeness_score = float(round(0.5 * overall_cell_completeness_pct + 0.5 * key_columns_completeness_pct, 2))

    # Detailed column gaps
    missing_value_gaps = {}
    for col in df.columns:
        null_cnt = int(df[col].isna().sum())
        if null_cnt > 0:
            missing_value_gaps[col] = {
                "missing_count": null_cnt,
                "missing_pct": float(round((null_cnt / total_rows) * 100, 2))
            }

    # -------------------------------------------------------------
    # 2. CONSISTENCY SUB-SCORE
    # -------------------------------------------------------------
    # a. ID Uniqueness
    id_unique_count = int(df["id"].nunique()) if "id" in df.columns else total_rows
    id_uniqueness_pct = float(round((id_unique_count / total_rows) * 100, 2))

    # b. Price Currency Format / Numeric Matching
    if "price" in df.columns:
        if pd.api.types.is_numeric_dtype(df["price"]):
            price_non_null = df["price"].dropna()
            price_format_consistency_pct = 100.0 if len(price_non_null) > 0 else 0.0
        else:
            price_non_null = df["price"].dropna().astype(str).str.strip()
            price_format_pattern = schema.get("columns", {}).get("price", {}).get("format_pattern", r"^\$\d{1,3}(,\d{3})*(\.\d{2})?$")
            price_format_valid = int(price_non_null.str.match(price_format_pattern).sum())
            price_format_consistency_pct = float(round((price_format_valid / len(price_non_null)) * 100, 2)) if len(price_non_null) > 0 else 100.0
    else:
        price_format_consistency_pct = 100.0

    # c. Bathrooms Format Consistency
    if "bathrooms_text" in df.columns or "bathrooms" in df.columns:
        if "bathrooms" in df.columns and pd.api.types.is_numeric_dtype(df["bathrooms"]):
            bathrooms_format_consistency_pct = 100.0 if df["bathrooms"].notna().sum() > 0 else 0.0
        elif "bathrooms_text" in df.columns:
            bath_non_null = df["bathrooms_text"].dropna().astype(str).str.strip()
            bath_pattern = schema.get("columns", {}).get("bathrooms_text", {}).get("format_pattern", r"^\d+(\.\d+)?\s+")
            bath_format_valid = int(bath_non_null.str.match(bath_pattern).sum())
            bathrooms_format_consistency_pct = float(round((bath_format_valid / len(bath_non_null)) * 100, 2)) if len(bath_non_null) > 0 else 100.0
        else:
            bathrooms_format_consistency_pct = 100.0
    else:
        bathrooms_format_consistency_pct = 100.0

    # d. Logical Night Boundaries (minimum_nights <= maximum_nights)
    if "minimum_nights" in df.columns and "maximum_nights" in df.columns:
        min_n = pd.to_numeric(df["minimum_nights"], errors="coerce")
        max_n = pd.to_numeric(df["maximum_nights"], errors="coerce")
        valid_eval = (min_n.notna()) & (max_n.notna())
        if valid_eval.sum() > 0:
            night_logic_valid = int((min_n[valid_eval] <= max_n[valid_eval]).sum())
            min_max_nights_logic_pct = float(round((night_logic_valid / valid_eval.sum()) * 100, 2))
        else:
            min_max_nights_logic_pct = 100.0
    else:
        min_max_nights_logic_pct = 100.0

    consistency_score = float(round((id_uniqueness_pct + price_format_consistency_pct + bathrooms_format_consistency_pct + min_max_nights_logic_pct) / 4, 2))

    # -------------------------------------------------------------
    # 3. VALIDITY SUB-SCORE
    # -------------------------------------------------------------
    # a. Price Numeric >= 0
    if "price" in df.columns:
        if pd.api.types.is_numeric_dtype(df["price"]):
            valid_prices = df["price"].dropna()
        else:
            clean_prices = df["price"].dropna().astype(str).str.replace(r'[\$,]', '', regex=True).str.strip()
            valid_prices = pd.to_numeric(clean_prices, errors="coerce").dropna()
        price_valid_count = int((valid_prices >= 0.0).sum())
        price_validity_pct = float(round((price_valid_count / len(valid_prices)) * 100, 2)) if len(valid_prices) > 0 else 100.0
    else:
        price_validity_pct = 100.0

    # b. Minimum Nights in range [1, 365]
    if "minimum_nights" in df.columns:
        min_nights_num = pd.to_numeric(df["minimum_nights"], errors="coerce").dropna()
        min_nights_valid_count = int(((min_nights_num >= 1) & (min_nights_num <= 365)).sum())
        minimum_nights_validity_pct = float(round((min_nights_valid_count / len(min_nights_num)) * 100, 2)) if len(min_nights_num) > 0 else 100.0
    else:
        minimum_nights_validity_pct = 100.0

    # c. Availability 365 in range [0, 365]
    if "availability_365" in df.columns:
        avail_num = pd.to_numeric(df["availability_365"], errors="coerce").dropna()
        avail_valid_count = int(((avail_num >= 0) & (avail_num <= 365)).sum())
        availability_365_validity_pct = float(round((avail_valid_count / len(avail_num)) * 100, 2)) if len(avail_num) > 0 else 100.0
    else:
        availability_365_validity_pct = 100.0

    # d. Room Type in Allowed Set
    if "room_type" in df.columns:
        allowed_rooms = set(schema.get("columns", {}).get("room_type", {}).get("allowed_values", [
            "Entire home/apt", "Private room", "Hotel room", "Shared room"
        ]))
        room_type_valid_count = int(df["room_type"].isin(allowed_rooms).sum())
        room_type_validity_pct = float(round((room_type_valid_count / total_rows) * 100, 2))
    else:
        room_type_validity_pct = 100.0

    # e. Geographical Coordinates Validity
    if "latitude" in df.columns and "longitude" in df.columns:
        lat_valid = int(((df["latitude"] >= -90.0) & (df["latitude"] <= 90.0)).sum())
        lon_valid = int(((df["longitude"] >= -180.0) & (df["longitude"] <= 180.0)).sum())
        geo_coordinates_validity_pct = float(round(((lat_valid + lon_valid) / (2 * total_rows)) * 100, 2))
    else:
        geo_coordinates_validity_pct = 100.0

    validity_score = float(round((price_validity_pct + minimum_nights_validity_pct + availability_365_validity_pct + room_type_validity_pct + geo_coordinates_validity_pct) / 5, 2))

    # -------------------------------------------------------------
    # OVERALL QUALITY SCORE
    # -------------------------------------------------------------
    overall_quality_score = float(round((completeness_score + consistency_score + validity_score) / 3, 2))

    report = {
        "timestamp": datetime.now().isoformat(),
        "dataset_summary": {
            "total_rows": total_rows,
            "total_columns": total_cols,
            "total_cells": total_cells,
            "duplicate_rows": int(df.duplicated().sum())
        },
        "overall_quality_score": overall_quality_score,
        "sub_scores": {
            "completeness_score": completeness_score,
            "consistency_score": consistency_score,
            "validity_score": validity_score
        },
        "sub_score_details": {
            "completeness": {
                "overall_cell_completeness_pct": overall_cell_completeness_pct,
                "key_columns_completeness_pct": key_columns_completeness_pct,
                "missing_value_gaps": missing_value_gaps
            },
            "consistency": {
                "id_uniqueness_pct": id_uniqueness_pct,
                "price_format_consistency_pct": price_format_consistency_pct,
                "bathrooms_format_consistency_pct": bathrooms_format_consistency_pct,
                "min_max_nights_logic_pct": min_max_nights_logic_pct
            },
            "validity": {
                "price_validity_pct": price_validity_pct,
                "minimum_nights_validity_pct": minimum_nights_validity_pct,
                "availability_365_validity_pct": availability_365_validity_pct,
                "room_type_validity_pct": room_type_validity_pct,
                "geo_coordinates_validity_pct": geo_coordinates_validity_pct
            }
        }
    }

    return report

def generate_quality_score(parquet_path: str = "raw_data_loaded.parquet", schema_path: str = "schema.json", output_path: str = "quality_score_before.json") -> Dict[str, Any]:
    if not os.path.exists(parquet_path):
        from load_data import load_data
        load_data()
        
    if not os.path.exists(schema_path):
        from schema_inference import generate_schema
        generate_schema(parquet_path=parquet_path, output_schema_path=schema_path)

    df = pd.read_parquet(parquet_path)
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    report = compute_quality_scores(df, schema, is_post_clean=False)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Successfully generated quality score report '{output_path}'.")
    print(f"Overall Quality Score: {report['overall_quality_score']}/100")
    print(f"Sub-Scores -> Completeness: {report['sub_scores']['completeness_score']}, Consistency: {report['sub_scores']['consistency_score']}, Validity: {report['sub_scores']['validity_score']}")

    return report

def generate_post_clean_quality_score(cleaned_csv_path: str = "cleaned_data.csv", schema_path: str = "schema.json", output_path: str = "quality_score_after.json") -> Dict[str, Any]:
    if not os.path.exists(cleaned_csv_path):
        raise FileNotFoundError(f"Cleaned dataset '{cleaned_csv_path}' not found.")

    df = pd.read_csv(cleaned_csv_path, low_memory=False)
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    report = compute_quality_scores(df, schema, is_post_clean=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Successfully generated post-clean quality score report '{output_path}'.")
    print(f"Overall Quality Score (After): {report['overall_quality_score']}/100")
    print(f"Sub-Scores -> Completeness: {report['sub_scores']['completeness_score']}, Consistency: {report['sub_scores']['consistency_score']}, Validity: {report['sub_scores']['validity_score']}")

    return report

def compute_quality_delta(before_json_path: str = "quality_score_before.json", after_json_path: str = "quality_score_after.json", output_path: str = "quality_delta.json") -> Dict[str, Any]:
    with open(before_json_path, "r", encoding="utf-8") as f:
        before = json.load(f)

    with open(after_json_path, "r", encoding="utf-8") as f:
        after = json.load(f)

    b_overall = before["overall_quality_score"]
    a_overall = after["overall_quality_score"]
    overall_delta = round(a_overall - b_overall, 2)

    b_sub = before["sub_scores"]
    a_sub = after["sub_scores"]

    delta_report = {
        "timestamp": datetime.now().isoformat(),
        "before_score": b_overall,
        "after_score": a_overall,
        "overall_delta": overall_delta,
        "sub_score_deltas": {
            "completeness_delta": round(a_sub["completeness_score"] - b_sub["completeness_score"], 2),
            "consistency_delta": round(a_sub["consistency_score"] - b_sub["consistency_score"], 2),
            "validity_delta": round(a_sub["validity_score"] - b_sub["validity_score"], 2)
        },
        "metrics_summary": {
            "before_sub_scores": b_sub,
            "after_sub_scores": a_sub
        },
        "improvement_verdict": "MEASURABLE_IMPROVEMENT_PASSED" if overall_delta > 0 else "NO_IMPROVEMENT"
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(delta_report, f, indent=2)

    print(f"Successfully generated quality delta report '{output_path}'.")
    print(f"Overall Score Delta: {overall_delta:+.2f} (Before: {b_overall} -> After: {a_overall})")

    return delta_report

if __name__ == "__main__":
    generate_quality_score()
