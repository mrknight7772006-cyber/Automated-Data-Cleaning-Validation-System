import json
import os
import pandas as pd
from typing import Dict, Any

def infer_dataset_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generates structured schema definition ('correctness' specification)
    for all dataset columns with explicit rules for key columns.
    """
    # Key columns specifications
    key_column_specs = {
        "id": {
            "expected_dtype": "integer",
            "semantic_type": "ID",
            "is_nullable": False,
            "is_unique": True,
            "description": "Unique identifier for listing"
        },
        "price": {
            "expected_dtype": "float",
            "semantic_type": "numeric",
            "is_nullable": False,
            "min_value": 0.0,
            "format_pattern": r"^\$\d{1,3}(,\d{3})*(\.\d{2})?$",
            "description": "Listing price per night, numeric >= 0 after stripping currency symbols"
        },
        "minimum_nights": {
            "expected_dtype": "integer",
            "semantic_type": "numeric",
            "is_nullable": False,
            "min_value": 1,
            "max_value": 365,
            "description": "Reasonable integer range 1 to 365"
        },
        "availability_365": {
            "expected_dtype": "integer",
            "semantic_type": "numeric",
            "is_nullable": False,
            "min_value": 0,
            "max_value": 365,
            "description": "Availability in days per year [0, 365]"
        },
        "room_type": {
            "expected_dtype": "categorical",
            "semantic_type": "categorical",
            "is_nullable": False,
            "allowed_values": [
                "Entire home/apt",
                "Private room",
                "Hotel room",
                "Shared room"
            ],
            "description": "Categorical listing room type from fixed set"
        },
        "bathrooms_text": {
            "expected_dtype": "string",
            "semantic_type": "text",
            "is_nullable": True,
            "format_pattern": r"^\d+(\.\d+)?\s+",
            "description": "Bathrooms textual descriptor, expects numeric quantity prefix"
        },
        "latitude": {
            "expected_dtype": "float",
            "semantic_type": "numeric",
            "is_nullable": False,
            "min_value": -90.0,
            "max_value": 90.0,
            "description": "GPS Latitude coordinate"
        },
        "longitude": {
            "expected_dtype": "float",
            "semantic_type": "numeric",
            "is_nullable": False,
            "min_value": -180.0,
            "max_value": 180.0,
            "description": "GPS Longitude coordinate"
        },
        "review_scores_rating": {
            "expected_dtype": "float",
            "semantic_type": "numeric",
            "is_nullable": True,
            "min_value": 1.0,
            "max_value": 5.0,
            "description": "Review score rating between 1.0 and 5.0"
        },
        "license": {
            "expected_dtype": "string",
            "semantic_type": "text",
            "is_nullable": False,
            "description": "Short-term rental license identifier"
        }
    }

    columns_schema = {}

    for col in df.columns:
        if col in key_column_specs:
            spec = key_column_specs[col].copy()
            spec["is_key_column"] = True
            columns_schema[col] = spec
        else:
            # Infer properties for non-key columns
            dtype_str = str(df[col].dtype)
            is_nullable = bool(df[col].isna().any())
            
            if "int" in dtype_str:
                expected_dtype = "integer"
                semantic_type = "numeric"
            elif "float" in dtype_str:
                expected_dtype = "float"
                semantic_type = "numeric"
            elif df[col].nunique() < 10 and not is_nullable:
                expected_dtype = "categorical"
                semantic_type = "categorical"
            else:
                expected_dtype = "string"
                semantic_type = "free text"

            columns_schema[col] = {
                "expected_dtype": expected_dtype,
                "semantic_type": semantic_type,
                "is_nullable": is_nullable,
                "is_key_column": False,
                "description": f"Inferred schema for column '{col}'"
            }

    schema_data = {
        "schema_version": "1.0",
        "dataset_name": "listings.csv.gz",
        "total_columns": len(df.columns),
        "key_columns": list(key_column_specs.keys()),
        "columns": columns_schema
    }

    return schema_data

def generate_schema(parquet_path: str = "raw_data_loaded.parquet", output_schema_path: str = "schema.json") -> Dict[str, Any]:
    if not os.path.exists(parquet_path):
        from load_data import load_data
        load_data()
        
    df = pd.read_parquet(parquet_path)
    schema = infer_dataset_schema(df)
    
    with open(output_schema_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)
        
    print(f"Successfully generated schema deliverable '{output_schema_path}' ({len(schema['columns'])} columns defined).")
    return schema

if __name__ == "__main__":
    generate_schema()
