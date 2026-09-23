import os
import json
import pytest
import pandas as pd
import numpy as np

CSV_PATH = "cleaned_data.csv"
LOG_PATH = "cleaning_log.json"
AFTER_SCORE_PATH = "quality_score_after.json"
DELTA_PATH = "quality_delta.json"
SCHEMA_PATH = "schema.json"

@pytest.fixture(scope="module")
def cleaned_df():
    assert os.path.exists(CSV_PATH), f"Deliverable '{CSV_PATH}' missing!"
    assert os.path.getsize(CSV_PATH) > 0, f"Deliverable '{CSV_PATH}' is empty!"
    df = pd.read_csv(CSV_PATH, low_memory=False)
    return df

@pytest.fixture(scope="module")
def schema():
    assert os.path.exists(SCHEMA_PATH), f"Schema '{SCHEMA_PATH}' missing!"
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def test_cleaned_data_exists(cleaned_df):
    assert len(cleaned_df) > 0, "cleaned_data.csv has 0 rows!"
    assert len(cleaned_df.columns) > 0, "cleaned_data.csv has 0 columns!"

def test_no_nans_in_non_nullable_columns(cleaned_df, schema):
    retained_exceptions = {"license"}
    cols_spec = schema.get("columns", {})
    non_nullable_cols = [c for c, spec in cols_spec.items() if not spec.get("is_nullable", True)]

    for col in non_nullable_cols:
        if col in retained_exceptions:
            continue
        if col in cleaned_df.columns:
            nan_count = cleaned_df[col].isna().sum()
            assert nan_count == 0, f"Non-nullable column '{col}' has {nan_count} NaNs in cleaned_data.csv!"

def test_price_fully_numeric(cleaned_df):
    assert "price" in cleaned_df.columns, "price column missing!"
    assert pd.api.types.is_numeric_dtype(cleaned_df["price"]), "price column is not numeric float!"
    assert cleaned_df["price"].isna().sum() == 0, "price column contains NaNs!"
    assert (cleaned_df["price"] >= 0.0).all(), "price column contains negative values!"

def test_no_duplicate_listings(cleaned_df):
    assert "id" in cleaned_df.columns, "id column missing!"
    duplicate_count = cleaned_df["id"].duplicated().sum()
    assert duplicate_count == 0, f"Found {duplicate_count} duplicate listing IDs in cleaned_data.csv!"

def test_all_dates_parse(cleaned_df):
    for date_col in ["last_scraped", "host_since"]:
        assert date_col in cleaned_df.columns, f"{date_col} missing!"
        parsed_dates = pd.to_datetime(cleaned_df[date_col], format="%Y-%m-%d", errors="coerce")
        assert parsed_dates.isna().sum() == 0, f"Column '{date_col}' contains invalid/unparseable date strings!"

def test_bathrooms_numeric(cleaned_df):
    assert "bathrooms" in cleaned_df.columns, "bathrooms column missing!"
    assert pd.api.types.is_numeric_dtype(cleaned_df["bathrooms"]), "bathrooms column is not numeric!"
    assert cleaned_df["bathrooms"].isna().sum() == 0, "bathrooms column contains NaNs!"
    assert (cleaned_df["bathrooms"] >= 0.0).all(), "bathrooms column contains negative values!"
    assert "bathroom_type" in cleaned_df.columns, "bathroom_type flag column missing!"

def test_quality_delta_improvement():
    assert os.path.exists(DELTA_PATH), f"Deliverable '{DELTA_PATH}' missing!"
    with open(DELTA_PATH, "r", encoding="utf-8") as f:
        delta_data = json.load(f)

    assert "overall_delta" in delta_data, "overall_delta missing from quality_delta.json!"
    overall_delta = delta_data["overall_delta"]
    assert overall_delta > 0.0, f"Quality score did not show measurable improvement! delta={overall_delta}"
    assert delta_data.get("improvement_verdict") == "MEASURABLE_IMPROVEMENT_PASSED"

def test_engineered_features(cleaned_df):
    required_features = [
        "host_tenure_days", "price_per_bedroom",
        "room_type_encoded", "neighbourhood_group_cleansed_encoded",
        "bathroom_type"
    ]
    for feat in required_features:
        assert feat in cleaned_df.columns, f"Engineered feature '{feat}' missing!"
        if feat != "bathroom_type":
            assert pd.api.types.is_numeric_dtype(cleaned_df[feat]), f"Feature '{feat}' is not numeric!"
            assert cleaned_df[feat].isna().sum() == 0, f"Feature '{feat}' contains NaNs!"
