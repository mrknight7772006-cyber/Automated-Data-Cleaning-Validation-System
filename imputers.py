import os
import json
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer

class MissingValueImputer:
    """
    Imputation engine performing statistical imputation (mode, grouped median)
    and ML-based imputation (KNN/Iterative imputer) on Airbnb listing data.
    Enforces schema non-nullability constraints post-imputation.
    """

    def __init__(self, schema_path="schema.json", ml_method="knn", n_neighbors=5, random_state=42):
        self.schema_path = schema_path
        self.ml_method = ml_method.lower()
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.schema = self._load_schema()

    def _load_schema(self):
        if os.path.exists(self.schema_path):
            with open(self.schema_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"columns": {}}

    def statistical_imputation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform statistical imputation:
        1. Mode for neighbourhood_group_cleansed (falling back to neighbourhood_cleansed mode or 'Unknown').
        2. Grouped median for review_scores_rating by room_type.
        """
        df_out = df.copy()

        # 1. Mode for neighbourhood_group_cleansed
        if "neighbourhood_group_cleansed" in df_out.columns:
            mode_ngc = df_out["neighbourhood_group_cleansed"].mode()
            if len(mode_ngc) > 0 and pd.notna(mode_ngc[0]):
                fill_val = mode_ngc[0]
            elif "neighbourhood_cleansed" in df_out.columns:
                mode_nc = df_out["neighbourhood_cleansed"].mode()
                fill_val = mode_nc[0] if len(mode_nc) > 0 else "Unknown"
            else:
                fill_val = "Unknown"
            
            df_out["neighbourhood_group_cleansed"] = df_out["neighbourhood_group_cleansed"].fillna(fill_val)

        # 2. Grouped median for review_scores_rating by room_type
        if "review_scores_rating" in df_out.columns and "room_type" in df_out.columns:
            # Grouped median transform
            grouped_median = df_out.groupby("room_type")["review_scores_rating"].transform("median")
            global_median = df_out["review_scores_rating"].median()
            fill_series = grouped_median.fillna(global_median)
            df_out["review_scores_rating_stat_imputed"] = df_out["review_scores_rating"].fillna(fill_series)

        return df_out

    def ml_imputation(self, df: pd.DataFrame, target_cols=None) -> pd.DataFrame:
        """
        Perform ML-based imputation (KNNImputer or IterativeImputer) on numeric features:
        bedrooms, beds, review_scores_rating using correlated numeric features.
        """
        df_out = df.copy()

        if target_cols is None:
            target_cols = ["bedrooms", "beds", "review_scores_rating"]

        numeric_predictors = [
            "accommodates", "bathrooms", "price", "minimum_nights",
            "number_of_reviews", "availability_365", "latitude", "longitude",
            "reviews_per_month", "estimated_revenue_l365d"
        ]

        # Standardize price and numeric columns before converting to float
        for col in list(dict.fromkeys(target_cols + numeric_predictors)):
            if col in df_out.columns:
                if df_out[col].dtype == object or isinstance(df_out[col].dtype, pd.ArrowDtype) or str(df_out[col].dtype) == "string":
                    s_str = df_out[col].astype(str).str.replace(r"[\$,]", "", regex=True).str.strip()
                    df_out[col] = pd.to_numeric(s_str, errors="coerce")

        all_ml_cols = list(dict.fromkeys([c for c in target_cols + numeric_predictors if c in df_out.columns]))

        numeric_df = pd.DataFrame(index=df_out.index)
        for col in all_ml_cols:
            numeric_df[col] = pd.to_numeric(df_out[col], errors="coerce")

        valid_ml_cols = [c for c in all_ml_cols if not numeric_df[c].isna().all()]
        numeric_df = numeric_df[valid_ml_cols]

        if self.ml_method == "iterative":
            imputer = IterativeImputer(random_state=self.random_state, max_iter=20)
        else:
            imputer = KNNImputer(n_neighbors=self.n_neighbors)

        imputed_array = imputer.fit_transform(numeric_df)
        imputed_df = pd.DataFrame(imputed_array, columns=valid_ml_cols, index=df_out.index)

        for col in target_cols:
            if col in imputed_df.columns:
                if col in ["bedrooms", "beds"]:
                    df_out[col] = imputed_df[col].round(1)
                else:
                    df_out[col] = imputed_df[col].round(4)

        for col in ["price", "minimum_nights", "bathrooms", "reviews_per_month", "estimated_revenue_l365d"]:
            if col in imputed_df.columns and col in df_out.columns:
                df_out[col] = imputed_df[col]

        return df_out

    def enforce_schema_non_nullability(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure all columns declared as non-nullable in schema.json have 0 NaNs,
        except explicitly retained legitimate absence columns (e.g. 'license').
        """
        df_out = df.copy()
        cols_schema = self.schema.get("columns", {})

        retained_exceptions = {"license"}

        for col, spec in cols_schema.items():
            if col not in df_out.columns:
                continue
            is_nullable = spec.get("is_nullable", True)

            if not is_nullable and col not in retained_exceptions:
                missing_cnt = df_out[col].isna().sum()
                if missing_cnt > 0:
                    dtype_spec = spec.get("expected_dtype", "string")
                    is_num = (dtype_spec in ["integer", "float", "numeric"]) or pd.api.types.is_numeric_dtype(df_out[col])
                    if is_num:
                        s_num = pd.to_numeric(df_out[col], errors="coerce")
                        fill_val = float(s_num.median()) if pd.notna(s_num.median()) else 0.0
                        df_out[col] = s_num.fillna(fill_val)
                    else:
                        mode_val = df_out[col].mode()
                        fill_val = str(mode_val[0]) if len(mode_val) > 0 and pd.notna(mode_val[0]) else "Unknown"
                        df_out[col] = df_out[col].fillna(fill_val)

        return df_out

    def impute_missing_values(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """
        Main pipeline executing statistical, ML-based imputation and schema non-nullability enforcement.
        """
        before_missing = df.isna().sum().to_dict()

        df_stat = self.statistical_imputation(df)
        df_ml = self.ml_imputation(df_stat, target_cols=["bedrooms", "beds", "review_scores_rating"])
        df_final = self.enforce_schema_non_nullability(df_ml)

        # Standardize price to float explicitly
        if "price" in df_final.columns:
            s_price = df_final["price"].astype(str).str.replace(r"[\$,]", "", regex=True).str.strip()
            df_final["price"] = pd.to_numeric(s_price, errors="coerce")
            if df_final["price"].isna().sum() > 0:
                df_final["price"] = df_final["price"].fillna(float(df_final["price"].median()))

        after_missing = df_final.isna().sum().to_dict()

        summary = {
            "ml_method": self.ml_method,
            "statistical_imputed_cols": ["neighbourhood_group_cleansed", "review_scores_rating"],
            "ml_imputed_cols": ["bedrooms", "beds", "review_scores_rating"],
            "retained_nan_columns": ["license"],
            "total_missing_before": int(sum(before_missing.values())),
            "total_missing_after": int(sum(after_missing.values()))
        }

        return df_final, summary

def impute_dataframe(df: pd.DataFrame, schema_path="schema.json", ml_method="knn") -> pd.DataFrame:
    imputer = MissingValueImputer(schema_path=schema_path, ml_method=ml_method)
    clean_df, _ = imputer.impute_missing_values(df)
    return clean_df
