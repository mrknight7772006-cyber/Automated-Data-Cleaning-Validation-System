import os
import gzip
import shutil
import pandas as pd
import re

def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes column names by converting to lowercase, removing non-alphanumeric
    characters, and replacing spaces/hyphens with single underscores.
    """
    new_columns = []
    for col in df.columns:
        # Convert to string and strip surrounding whitespace
        c = str(col).strip().lower()
        # Replace non-alphanumeric characters with underscores
        c = re.sub(r'[^a-z0-9_]', '_', c)
        # Collapse multiple underscores into a single underscore
        c = re.sub(r'_+', '_', c)
        # Remove leading and trailing underscores
        c = c.strip('_')
        new_columns.append(c)
    
    df_copy = df.copy()
    df_copy.columns = new_columns
    return df_copy

def ensure_gzipped_input(gz_path: str = "listings.csv.gz", csv_fallback: str = "dataset.csv") -> str:
    """
    Ensures that the gzipped input file exists. If gz_path does not exist,
    compresses csv_fallback into gz_path if csv_fallback is present.
    """
    if os.path.exists(gz_path):
        return gz_path
    
    if os.path.exists(csv_fallback):
        print(f"'{gz_path}' not found. Compressing '{csv_fallback}' into '{gz_path}'...")
        with open(csv_fallback, 'rb') as f_in:
            with gzip.open(gz_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"Successfully created '{gz_path}'.")
        return gz_path
    
    raise FileNotFoundError(f"Neither '{gz_path}' nor '{csv_fallback}' could be found.")

def load_data(input_path: str = "listings.csv.gz", output_parquet_path: str = "raw_data_loaded.parquet") -> pd.DataFrame:
    """
    Reads the gzipped listings raw dataset, standardizes column names,
    saves the result to raw_data_loaded.parquet, and returns the DataFrame.
    """
    actual_input_path = ensure_gzipped_input(gz_path=input_path)
    print(f"Loading raw data from '{actual_input_path}'...")
    
    # Read gzipped CSV with low_memory=False to prevent dtype inference warnings
    df = pd.read_csv(actual_input_path, compression='gzip', low_memory=False)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns.")
    
    # Standardize column names
    df_standardized = standardize_column_names(df)
    print("Column names standardized.")
    
    # Save to Parquet
    print(f"Saving standardized data to '{output_parquet_path}'...")
    df_standardized.to_parquet(output_parquet_path, index=False)
    print(f"Successfully saved deliverable '{output_parquet_path}'.")
    
    return df_standardized

if __name__ == "__main__":
    df = load_data()
    print("Sample standardized columns:", list(df.columns[:10]))
