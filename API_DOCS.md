# Automated Data Cleaning & Validation System — API Reference & Schema Documentation

## Overview

This document provides consolidated technical specifications, Python API function signatures, parameter descriptions, return types, and JSON output schemas for all modules within the Automated Data Cleaning & Validation System.

---

## Table of Contents
1. [Module 1: Profiling & Metadata Engine](#1-module-1-profiling--metadata-engine)
2. [Module 2: Cleaning, Imputation & Scoring Engine](#2-module-2-cleaning-imputation--scoring-engine)
3. [Module 3: Validation, Anomaly Detection & Scoring Engine](#3-module-3-validation-anomaly-detection--scoring-engine)
4. [Module 4: Orchestration & CLI Engine](#4-module-4-orchestration--cli-engine)
5. [Deliverable JSON Schemas](#5-deliverable-json-schemas)

---

## 1. Module 1: Profiling & Metadata Engine

### `load_data.py`
```python
def load_data(
    input_path: str = "listings.csv.gz",
    output_parquet_path: str = "raw_data_loaded.parquet"
) -> pd.DataFrame
```
- **Description**: Reads raw gzipped CSV file, standardizes column headers into strict `snake_case`, writes Parquet cache file, and returns the loaded DataFrame.
- **Parameters**:
  - `input_path` *(str)*: Path to input raw dataset `.csv` or `.csv.gz`.
  - `output_parquet_path` *(str)*: Path where Parquet file will be cached.
- **Returns**: `pd.DataFrame` — Standardized raw DataFrame.

```python
def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame
```
- **Description**: Converts column names to lowercase, strips surrounding whitespace, replaces non-alphanumeric characters with underscores, and removes duplicate/leading/trailing underscores.

---

### `schema_inference.py`
```python
def infer_schema(df: pd.DataFrame) -> Dict[str, Any]
```
- **Description**: Inspects column types, sample values, null counts, and value ranges to infer JSON schema data types (`integer`, `float`, `boolean`, `datetime`, `string`, `categorical`) and boundary constraints.
- **Returns**: `Dict[str, Any]` — Dict mapping column names to inferred schema properties.

---

### `profiling_engine.py`
```python
def run_profiling_engine(
    parquet_path: str = "raw_data_loaded.parquet",
    output_json_path: str = "stats_summary.json"
) -> Dict[str, Any]
```
- **Description**: Computes overall dataset statistics, per-column missingness percentages, cardinality counts, numeric correlation matrix, and percentiles. Writes output JSON.
- **Returns**: `Dict[str, Any]` — Consolidated statistical summary dictionary.

---

### `suspicious_column_rules.py`
```python
def run_suspicious_rules(df: pd.DataFrame) -> Dict[str, Any]
```
- **Description**: Evaluates rule heuristics to detect PII exposure risk (email, phone, SSN patterns), formatting inconsistencies, and suspicious row anomalies.
- **Returns**: `Dict[str, Any]` — Analysis dictionary of flagged suspicious columns and rows.

---

### `visualizations.py`
```python
def generate_all_visualizations(
    df: pd.DataFrame,
    output_dir: str = "."
) -> Dict[str, str]
```
- **Description**: Renders missing value heatmap (`heatmap_missing.png`), outlier boxplots (`outlier_min_nights.png`), and numeric correlation heatmap (`correlation_heatmap.png`).
- **Returns**: `Dict[str, str]` — Dictionary mapping visualization names to generated file paths.

---

### `profiling_api.py`
```python
def run_profiling(
    df: Optional[pd.DataFrame] = None,
    parquet_path: str = "raw_data_loaded.parquet",
    output_json_path: str = "profiling_report.json",
    generate_visuals: bool = True,
    stats_summary_path: Optional[str] = None,
    output_dir: Optional[str] = None
) -> Dict[str, Any]
```
- **Description**: Master API entrypoint for Module 1. Executes profiling, rule evaluation, visualization rendering, and exports `profiling_report.json`.
- **Returns**: `Dict[str, Any]` — Full `profiling_report.json` data structure.

---

## 2. Module 2: Cleaning, Imputation & Scoring Engine

### `dedup_engine.py`
```python
def deduplicate_dataframe(
    df: pd.DataFrame,
    similarity_threshold: float = 0.85
) -> Tuple[pd.DataFrame, Dict[str, Any]]
```
- **Description**: Identifies exact and fuzzy duplicate rows using TF-IDF vectorization and cosine similarity / Levenshtein distance matching.
- **Returns**: `(pd.DataFrame, Dict)` — Deduplicated DataFrame and deduplication summary report.

---

### `imputers.py`
```python
def impute_dataframe(
    df: pd.DataFrame,
    ml_method: str = "median"
) -> pd.DataFrame
```
- **Description**: Imputes missing values across numerical and categorical columns using specified algorithm (`median`, `mean`, `knn`, `iterative`).
- **Returns**: `pd.DataFrame` — Imputed DataFrame.

---

### `normalizer.py`
```python
def normalize_dataframe(
    df: pd.DataFrame,
    raw_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame
```
- **Description**: Strips currency symbols (`$` and commas), parses unstructured bathrooms text into float counts, normalizes text casing/variants, and formats datetimes to ISO `YYYY-MM-DD`.
- **Returns**: `pd.DataFrame` — Format-normalized DataFrame.

---

### `transformer.py`
```python
def transform_dataframe(df: pd.DataFrame) -> pd.DataFrame
```
- **Description**: Performs categorical label encoding and extracts engineered features (`price_per_bedroom`, `host_tenure_days`, `bathroom_type`).
- **Returns**: `pd.DataFrame` — Transformed DataFrame.

---

### `quality_scorer.py`
```python
def generate_post_clean_quality_score(
    cleaned_csv_path: str,
    schema_path: str = "schema.json",
    output_path: str = "quality_score_after.json"
) -> Dict[str, Any]
```
- **Description**: Evaluates completeness, consistency, and validity dimensions for the cleaned dataset and exports scoring JSON.

```python
def compute_quality_delta(
    before_json_path: str,
    after_json_path: str,
    output_path: str = "quality_delta.json"
) -> Dict[str, Any]
```
- **Description**: Computes score changes (delta) between before-clean and after-clean quality reports.

---

### `cleaning_api.py`
```python
class CleaningAPI:
    def run_pipeline(
        self,
        output_csv_path: str = "cleaned_data.csv",
        log_json_path: str = "cleaning_log.json",
        after_score_path: str = "quality_score_after.json",
        delta_json_path: str = "quality_delta.json",
        before_score_path: str = "quality_score_before.json"
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]
```
- **Description**: Master API entrypoint for Module 2. Orchestrates normalization, transformation, scoring, and exports `cleaned_data.csv` and `cleaning_log.json`.

---

## 3. Module 3: Validation, Anomaly Detection & Scoring Engine

### `rule_validator.py`
```python
class RuleValidator:
    def validate_dataframe(self, df: pd.DataFrame) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]
```
- **Description**: Evaluates rule-based business domain constraints (price range, response rate, coordinate boundaries) and returns summary and list of violations.

---

### `anomaly_detector.py`
```python
def detect_anomalies(
    input_path: str = "cleaned_data.csv",
    output_path: str = "anomaly_report.json",
    contamination: float = 0.05
) -> Dict[str, Any]
```
- **Description**: Fits Isolation Forest, Local Outlier Factor, and Autoencoder Neural Network (MLPRegressor) reconstruction models to detect multivariate anomalies. Writes `anomaly_report.json`.

---

### `nlp_column_classifier.py`
```python
def run_nlp_classification(
    input_path: str = "cleaned_data.csv",
    output_path: str = "nlp_classification_report.json"
) -> Dict[str, Any]
```
- **Description**: Performs NLP classification on text fields and parses corrupted JSON/string arrays in amenities.

---

### `error_detector.py`
```python
class ErrorDetector:
    def detect_statistical_errors(self) -> List[Dict[str, Any]]
    def detect_data_drift(self) -> Dict[str, Any]
```
- **Description**: Identifies statistical outliers using Z-Score and IQR methods, and evaluates dataset distribution drift via Kolmogorov-Smirnov tests (`data_drift_report`).

---

### `validation_scorer.py`
```python
class ValidationScorer:
    def score_dataset(
        self,
        total_rows: int,
        rule_violations: List[Dict[str, Any]],
        anomaly_flagged_rows: List[Dict[str, Any]],
        nlp_issues: List[Dict[str, Any]],
        statistical_errors: List[Dict[str, Any]]
    ) -> Dict[str, Any]
```
- **Description**: Calculates row-level health scores and computes overall dataset health score ($0-100$) and letter grade (`A`, `B`, `C`, `D`, `F`).

---

### `evaluate_validation.py`
```python
def evaluate_pipeline(
    input_path: str = "cleaned_data.csv",
    report_path: str = "validation_report.json"
) -> Dict[str, Any]
```
- **Description**: Generates classification metric evaluation metrics (`evaluation_metrics.json`) and renders ROC AUC curve plot (`roc_curve.png`).

---

### `validation_api.py`
```python
class ValidationAPI:
    def run_validation_pipeline(
        self,
        input_path: str = "cleaned_data.csv",
        output_path: str = "validation_report.json",
        contamination: float = 0.05,
        anomaly_report_path: Optional[str] = None,
        nlp_report_path: Optional[str] = None
    ) -> Dict[str, Any]
```
- **Description**: Master API entrypoint for Module 3. Executes rule validation, anomaly detection, NLP classification, and outputs consolidated `validation_report.json`.

---

## 4. Module 4: Orchestration & CLI Engine

### `pipeline_orchestrator.py`
```python
class PipelineOrchestrator:
    def __init__(
        self,
        config_path: str = "config.yaml",
        input_path: Optional[str] = None,
        output_dir: Optional[str] = None
    )
    def run_full_pipeline(self) -> Dict[str, Any]
```
- **Description**: Master orchestrator chaining Stages 1-3 sequentially with structured timing logs.

---

### `pipeline.py`
```python
def run_pipeline(
    input_path: str = "listings.csv.gz",
    output_dir: str = "results",
    config_path: str = "config.yaml"
) -> Dict[str, Any]
```
- **Description**: CLI entrypoint function executing the end-to-end cleaning and validation system.

---

## 5. Deliverable JSON Schemas

### `profiling_report.json`
```json
{
  "report_metadata": {
    "generated_at": "ISO-8601 Timestamp",
    "version": "1.0.0",
    "engine": "String"
  },
  "dataset_summary": {
    "total_rows": "Integer",
    "total_columns": "Integer",
    "memory_usage_mb": "Float"
  },
  "per_column_stats": {
    "column_name": {
      "dtype": "String",
      "null_count": "Integer",
      "null_percentage": "Float",
      "cardinality": "Integer"
    }
  },
  "correlation_matrix": "Object",
  "distribution_stats": "Object",
  "suspicious_rules_analysis": "Object",
  "visualizations": "Object"
}
```

### `validation_report.json`
```json
{
  "timestamp": "ISO-8601 Timestamp",
  "dataset_path": "String",
  "total_rows_validated": "Integer",
  "overall_health_score": "Float (0.0 - 100.0)",
  "health_grade": "String ('A', 'B', 'C', 'D', 'F')",
  "sub_scores": {
    "rule_validation_score": "Float",
    "anomaly_detection_score": "Float",
    "nlp_classification_score": "Float",
    "statistical_error_score": "Float"
  },
  "severity_summary": "Object",
  "rule_validation_summary": "Object",
  "anomaly_detection_summary": "Object",
  "flagged_rows_master_log": [
    {
      "row_index": "Integer",
      "listing_id": "Integer",
      "row_health_score": "Float",
      "rule_violations": "Array",
      "anomalies": "Array",
      "nlp_issues": "Array",
      "statistical_errors": "Array"
    }
  ]
}
```
