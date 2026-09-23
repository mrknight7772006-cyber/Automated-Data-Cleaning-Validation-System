import pandas as pd
import numpy as np
from typing import Dict, Any

class DataTransformer:
    """
    Performs feature engineering and categorical encoding on Airbnb dataset.
    """

    def __init__(self):
        self.room_type_map = {
            'Entire home/apt': 0,
            'Private room': 1,
            'Hotel room': 2,
            'Shared room': 3
        }

    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encodes room_type and neighbourhood_group_cleansed into integer codes.
        """
        out = df.copy()

        # Encode room_type
        if 'room_type' in out.columns:
            out['room_type_encoded'] = out['room_type'].map(self.room_type_map).fillna(0).astype(int)

        # Encode neighbourhood_group_cleansed
        if 'neighbourhood_group_cleansed' in out.columns:
            unique_ngc = sorted(list(out['neighbourhood_group_cleansed'].astype(str).unique()))
            ngc_map = {val: i for i, val in enumerate(unique_ngc)}
            out['neighbourhood_group_cleansed_encoded'] = out['neighbourhood_group_cleansed'].map(ngc_map).fillna(0).astype(int)

        return out

    def extract_host_tenure(self, df: pd.DataFrame) -> pd.Series:
        """
        Extracts host_tenure_days = (last_scraped - host_since).dt.days.
        """
        ls_dt = pd.to_datetime(df['last_scraped'], errors='coerce').fillna(pd.to_datetime('2026-06-16'))
        hs_dt = pd.to_datetime(df['host_since'], errors='coerce')
        
        # Fallback if host_since was NaT
        fallback_hs = ls_dt - pd.Timedelta(days=365)
        hs_dt = hs_dt.fillna(fallback_hs)

        tenure_days = (ls_dt - hs_dt).dt.days.clip(lower=0)
        return pd.Series(tenure_days, index=df.index, name='host_tenure_days')

    def extract_price_per_bedroom(self, df: pd.DataFrame) -> pd.Series:
        """
        Extracts price_per_bedroom = price / max(bedrooms, 1).
        """
        price = pd.to_numeric(df['price'], errors='coerce').fillna(0.0)
        bedrooms = pd.to_numeric(df.get('bedrooms', pd.Series(1.0, index=df.index)), errors='coerce').fillna(1.0)
        bedrooms_effective = bedrooms.apply(lambda b: max(float(b), 1.0))

        ppb = (price / bedrooms_effective).round(2)
        return pd.Series(ppb, index=df.index, name='price_per_bedroom')

    def transform_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies categorical encoding and feature extraction pipeline.
        """
        out = self.encode_categorical_features(df)
        out['host_tenure_days'] = self.extract_host_tenure(out)
        out['price_per_bedroom'] = self.extract_price_per_bedroom(out)

        return out

def transform_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    transformer = DataTransformer()
    return transformer.transform_dataset(df)
