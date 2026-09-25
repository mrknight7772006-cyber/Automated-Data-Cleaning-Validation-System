import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List

def plot_missing_value_heatmap(df: pd.DataFrame, output_path: str = "heatmap_missing.png") -> str:
    """
    Generates and saves a heatmap showing missing values across dataset columns.
    """
    plt.figure(figsize=(14, 7))
    
    # Calculate missing percentage per column to focus on relevant columns if many exist
    missing_pct = df.isna().mean() * 100
    
    # Create missingness heatmap
    sns.heatmap(
        df.isna(),
        cbar=True,
        yticklabels=False,
        cmap="Blues",
        cbar_kws={'label': 'Missing (True / False)'}
    )
    
    plt.title("Missing Value Matrix Heatmap", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Columns", fontsize=11, fontweight="bold")
    plt.ylabel("Rows / Observations", fontsize=11, fontweight="bold")
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Generated missing value heatmap: '{output_path}'")
    return output_path

def plot_outliers(df: pd.DataFrame, output_path: str = "outlier_min_nights.png") -> str:
    """
    Generates and saves boxplots for minimum_nights and availability_365 outliers.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.set_theme(style="whitegrid")
    
    # Boxplot for minimum_nights
    if "minimum_nights" in df.columns:
        sns.boxplot(y=df["minimum_nights"].dropna(), ax=axes[0], color="#5B84B1", flierprops={"markerfacecolor": "#E63946", "marker": "o", "markersize": 5})
        axes[0].set_title("Outliers: minimum_nights", fontsize=12, fontweight="bold")
        axes[0].set_ylabel("Minimum Nights", fontsize=10, fontweight="bold")
        
        # Annotate max outlier if exists
        max_val = df["minimum_nights"].max()
        if not pd.isna(max_val) and max_val > 365:
            axes[0].annotate(
                f"Extreme Outlier: {int(max_val)} days",
                xy=(0, max_val),
                xytext=(0.1, max_val * 0.9),
                arrowprops=dict(facecolor='red', shrink=0.05, width=1.5, headwidth=8),
                fontsize=9, fontweight="bold", color="#D90429"
            )
    else:
        axes[0].text(0.5, 0.5, "minimum_nights not found", ha="center", va="center")
        
    # Boxplot for availability_365
    if "availability_365" in df.columns:
        sns.boxplot(y=df["availability_365"].dropna(), ax=axes[1], color="#457B9D", flierprops={"markerfacecolor": "#E63946", "marker": "o", "markersize": 5})
        axes[1].set_title("Outliers: availability_365", fontsize=12, fontweight="bold")
        axes[1].set_ylabel("Availability (Days in Year)", fontsize=10, fontweight="bold")
    else:
        axes[1].text(0.5, 0.5, "availability_365 not found", ha="center", va="center")
        
    plt.suptitle("Outlier Analysis: minimum_nights & availability_365", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Generated outlier boxplots: '{output_path}'")
    return output_path

def plot_correlation_heatmap(df: pd.DataFrame, output_path: str = "correlation_heatmap.png") -> str:
    """
    Generates and saves correlation heatmap for numeric features.
    """
    plt.figure(figsize=(10, 8))
    
    target_cols = ["minimum_nights", "availability_365", "number_of_reviews", "reviews_per_month", "calculated_host_listings_count"]
    numeric_cols = [c for c in target_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            linewidths=0.5,
            square=True,
            cbar_kws={"shrink": 0.8}
        )
        plt.title("Correlation Heatmap (Numeric Features)", fontsize=14, fontweight="bold", pad=15)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
    else:
        plt.text(0.5, 0.5, "Insufficient numerical columns for correlation plot", ha="center", va="center")
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Generated correlation heatmap: '{output_path}'")
    return output_path

def generate_all_visualizations(df: pd.DataFrame, output_dir: str = ".") -> Dict[str, str]:
    """
    Executes all visualization plots and returns dict of generated file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    missing_path = plot_missing_value_heatmap(df, os.path.join(output_dir, "heatmap_missing.png"))
    outlier_path = plot_outliers(df, os.path.join(output_dir, "outlier_min_nights.png"))
    corr_path = plot_correlation_heatmap(df, os.path.join(output_dir, "correlation_heatmap.png"))
    
    return {
        "missing_value_heatmap": missing_path,
        "outliers_boxplot": outlier_path,
        "correlation_heatmap": corr_path
    }

if __name__ == "__main__":
    from load_data import load_data
    df = load_data()
    generate_all_visualizations(df)
