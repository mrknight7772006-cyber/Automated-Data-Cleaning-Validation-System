import os
import json
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Tuple

from normalizer import normalize_dataframe
from transformer import transform_dataframe
from quality_scorer import generate_post_clean_quality_score, compute_quality_delta

class CleaningAPI:
    """
    Unified Cleaning & Transformation API orchestrating data normalization,
    feature engineering, quality scoring, and artifact export.
    """

    def __init__(self,
                 imputed_parquet_path: str = "imputed_data.parquet",
                 raw_parquet_path: str = "raw_data_loaded.parquet",
                 schema_path: str = "schema.json"):
        self.imputed_parquet_path = imputed_parquet_path
        self.raw_parquet_path = raw_parquet_path
        self.schema_path = schema_path

    def run_pipeline(self,
                     output_csv_path: str = "cleaned_data.csv",
                     log_json_path: str = "cleaning_log.json",
                     after_score_path: str = "quality_score_after.json",
                     delta_json_path: str = "quality_delta.json",
                     before_score_path: str = "quality_score_before.json") -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Runs full normalization, transformation, artifact export, and post-clean scoring.
        """
        print("Starting Sprint 6 Data Normalization & Transformation Pipeline...")

        # 1. Load data
        if not os.path.exists(self.imputed_parquet_path):
            raise FileNotFoundError(f"Input imputed dataset '{self.imputed_parquet_path}' missing. Run Sprint 5 first.")

        df_imputed = pd.read_parquet(self.imputed_parquet_path)
        raw_df = pd.read_parquet(self.raw_parquet_path) if os.path.exists(self.raw_parquet_path) else None

        before_rows = len(df_imputed)
        before_cols = len(df_imputed.columns)

        # 2. Normalization
        print("Executing data format & date normalization (normalizer.py)...")
        df_normalized = normalize_dataframe(df_imputed, raw_df=raw_df)

        # 3. Transformation & Feature Engineering
        print("Executing categorical encoding & feature extraction (transformer.py)...")
        df_transformed = transform_dataframe(df_normalized)

        after_rows = len(df_transformed)
        after_cols = len(df_transformed.columns)

        # 4. Export Cleaned Dataset
        print(f"Exporting cleaned dataset to '{output_csv_path}'...")
        df_transformed.to_csv(output_csv_path, index=False)

        # 5. Create Cleaning Log
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "input_file": self.imputed_parquet_path,
            "output_file": output_csv_path,
            "before_summary": {
                "rows": before_rows,
                "columns": before_cols
            },
            "after_summary": {
                "rows": after_rows,
                "columns": after_cols
            },
            "normalization_steps": [
                "Price stripped of $ and commas, cast to float64",
                "bathrooms_text parsed into numeric bathrooms and bathroom_type flag",
                "neighbourhood and neighbourhood_cleansed standardized casing and spelling variants",
                "last_scraped and host_since parsed to ISO YYYY-MM-DD datetimes"
            ],
            "transformation_steps": [
                "room_type label encoded into room_type_encoded",
                "neighbourhood_group_cleansed label encoded into neighbourhood_group_cleansed_encoded",
                "host_tenure_days calculated as (last_scraped - host_since) in days",
                "price_per_bedroom calculated as price / max(bedrooms, 1)"
            ],
            "engineered_features": [
                "bathroom_type", "room_type_encoded", "neighbourhood_group_cleansed_encoded",
                "host_tenure_days", "price_per_bedroom"
            ]
        }

        with open(log_json_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2)
        print(f"Successfully generated cleaning log '{log_json_path}'.")

        # 6. Post-Clean Quality Score & Quality Delta
        print("Computing post-clean quality score...")
        after_report = generate_post_clean_quality_score(
            cleaned_csv_path=output_csv_path,
            schema_path=self.schema_path,
            output_path=after_score_path
        )

        print("Computing quality score delta (before vs after)...")
        if os.path.exists(before_score_path):
            delta_report = compute_quality_delta(
                before_json_path=before_score_path,
                after_json_path=after_score_path,
                output_path=delta_json_path
            )
        else:
            delta_report = {"warning": f"Before score file '{before_score_path}' missing."}

        print("\n==================================================")
        print("SPRINT 6 CLEANING & TRANSFORMATION PIPELINE COMPLETE!")
        print("==================================================")

        return df_transformed, log_data

def clean_dataset(imputed_parquet_path="imputed_data.parquet", output_csv_path="cleaned_data.csv") -> pd.DataFrame:
    api = CleaningAPI(imputed_parquet_path=imputed_parquet_path)
    cleaned_df, _ = api.run_pipeline(output_csv_path=output_csv_path)
    return cleaned_df

if __name__ == "__main__":
    api = CleaningAPI()
    api.run_pipeline()
