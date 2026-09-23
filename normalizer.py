import pandas as pd
import numpy as np
import re
from datetime import datetime
from typing import Tuple, Dict, Any

class DataNormalizer:
    """
    Normalizes data types, string formats, spelling variants, and datetimes
    for Airbnb listings dataset. Ensures zero duplicate listing IDs.
    """

    def __init__(self, raw_df: pd.DataFrame = None):
        self.raw_df = raw_df

    def normalize_price(self, df: pd.DataFrame) -> pd.Series:
        """
        Strips '$' and ',' from price, casts to float.
        Restores raw price strings if imputed dataset lost them, imputes median for invalid/zero prices.
        """
        raw_prices = None
        if self.raw_df is not None and 'price' in self.raw_df.columns:
            s_raw = self.raw_df['price'].astype(str).str.replace(r'[\$,]', '', regex=True).str.strip()
            raw_prices = pd.to_numeric(s_raw, errors='coerce')

        s_curr = df['price'].astype(str).str.replace(r'[\$,]', '', regex=True).str.strip() if 'price' in df.columns else pd.Series(dtype=float)
        s_curr_num = pd.to_numeric(s_curr, errors='coerce')

        valid_prices = s_curr_num[s_curr_num > 0]
        if len(valid_prices) == 0 and raw_prices is not None:
            valid_prices = raw_prices[raw_prices > 0]
        median_price = float(valid_prices.median()) if len(valid_prices) > 0 else 100.0

        cleaned_prices = []
        for idx, row in df.iterrows():
            val = s_curr_num.loc[idx] if idx in s_curr_num.index else np.nan
            if pd.isna(val) or val <= 0:
                row_id = row.get('id')
                if self.raw_df is not None and 'id' in self.raw_df.columns and row_id in self.raw_df['id'].values:
                    match = self.raw_df[self.raw_df['id'] == row_id]
                    if not match.empty:
                        rp_str = str(match['price'].values[0]).replace('$', '').replace(',', '').strip()
                        rp_num = pd.to_numeric(rp_str, errors='coerce')
                        val = rp_num if pd.notna(rp_num) and rp_num > 0 else median_price
                    else:
                        val = median_price
                else:
                    val = median_price
            cleaned_prices.append(float(round(val, 2)))

        return pd.Series(cleaned_prices, index=df.index, name='price')

    def parse_bathrooms(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Parses bathrooms_text into a numeric bathrooms column (float)
        plus a bathroom_type flag ('shared', 'half', 'private', 'standard').
        """
        bathrooms_numeric = []
        bathroom_types = []

        for idx, row in df.iterrows():
            text = str(row.get('bathrooms_text', '')).lower().strip()
            b_num_col = row.get('bathrooms')

            b_type = 'standard'
            if 'shared' in text:
                b_type = 'shared'
            elif 'private' in text:
                b_type = 'private'

            if 'half' in text:
                b_type = 'half' if b_type == 'standard' else f'{b_type}_half'

            match = re.search(r'(\d+(?:\.\d+)?)', text)
            if match:
                num = float(match.group(1))
            elif 'half' in text:
                num = 0.5
            elif pd.notna(b_num_col) and str(b_num_col).replace('.', '', 1).isdigit():
                num = float(b_num_col)
            else:
                num = 1.0

            bathrooms_numeric.append(float(num))
            bathroom_types.append(b_type)

        return pd.Series(bathrooms_numeric, index=df.index, name='bathrooms'), pd.Series(bathroom_types, index=df.index, name='bathroom_type')

    def standardize_neighbourhoods(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Standardizes neighbourhood & neighbourhood_cleansed spelling variants,
        trims whitespace, unifies casing, and populates missing neighbourhood from neighbourhood_cleansed.
        """
        n_cleansed = df['neighbourhood_cleansed'].apply(
            lambda x: str(x).strip().upper() if pd.notna(x) and str(x).strip() != '' else 'UNKNOWN'
        ) if 'neighbourhood_cleansed' in df.columns else pd.Series('UNKNOWN', index=df.index)

        if 'neighbourhood' in df.columns:
            n_raw = df['neighbourhood'].fillna(df['neighbourhood_cleansed'] if 'neighbourhood_cleansed' in df.columns else 'Unknown')
        else:
            n_raw = df['neighbourhood_cleansed'] if 'neighbourhood_cleansed' in df.columns else pd.Series('Unknown', index=df.index)

        n_std = n_raw.apply(
            lambda x: str(x).strip().title() if pd.notna(x) and str(x).strip() != '' else 'Unknown'
        )

        return pd.Series(n_std, index=df.index, name='neighbourhood'), pd.Series(n_cleansed, index=df.index, name='neighbourhood_cleansed')

    def parse_dates(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Parses last_scraped and host_since into consistent ISO datetime strings (YYYY-MM-DD).
        Derives missing host_since from last_scraped and host tenure years/months or first_review.
        """
        if 'last_scraped' in df.columns:
            ls_dt = pd.to_datetime(df['last_scraped'], format='%d-%m-%Y', errors='coerce')
            ls_dt = ls_dt.fillna(pd.to_datetime(df['last_scraped'], errors='coerce')).fillna(pd.to_datetime('2026-06-16'))
        else:
            ls_dt = pd.Series(pd.to_datetime('2026-06-16'), index=df.index)

        hs_raw = pd.to_datetime(df['host_since'], errors='coerce') if 'host_since' in df.columns else pd.Series(pd.NaT, index=df.index)

        derived_hs = []
        for idx, row in df.iterrows():
            hs = hs_raw.loc[idx]
            if pd.isna(hs):
                ls = ls_dt.loc[idx]
                yrs = row.get('hosts_time_as_host_years', 0)
                mths = row.get('hosts_time_as_host_months', 0)
                if pd.notna(yrs) and pd.notna(mths) and (float(yrs) > 0 or float(mths) > 0):
                    days_ago = int(float(yrs) * 365.25 + float(mths) * 30.4375)
                    hs = ls - pd.Timedelta(days=days_ago)
                elif pd.notna(row.get('first_review')):
                    fr_dt = pd.to_datetime(row.get('first_review'), format='%d-%m-%Y', errors='coerce')
                    hs = fr_dt if pd.notna(fr_dt) else (ls - pd.Timedelta(days=365))
                else:
                    hs = ls - pd.Timedelta(days=365)
            derived_hs.append(hs)

        hs_dt = pd.Series(derived_hs, index=df.index)

        ls_str = ls_dt.dt.strftime('%Y-%m-%d')
        hs_str = hs_dt.dt.strftime('%Y-%m-%d')

        return pd.Series(ls_str, index=df.index, name='last_scraped'), pd.Series(hs_str, index=df.index, name='host_since')

    def normalize_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes complete normalization pipeline on DataFrame, including deduplication of listing IDs.
        """
        out = df.copy()

        # Enforce zero duplicate listing IDs
        if 'id' in out.columns:
            out = out.drop_duplicates(subset=['id'], keep='first').reset_index(drop=True)

        out['price'] = self.normalize_price(out)
        out['bathrooms'], out['bathroom_type'] = self.parse_bathrooms(out)
        out['neighbourhood'], out['neighbourhood_cleansed'] = self.standardize_neighbourhoods(out)
        out['last_scraped'], out['host_since'] = self.parse_dates(out)

        return out

def normalize_dataframe(df: pd.DataFrame, raw_df: pd.DataFrame = None) -> pd.DataFrame:
    normalizer = DataNormalizer(raw_df=raw_df)
    return normalizer.normalize_dataset(df)
