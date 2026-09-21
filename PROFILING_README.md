# Module 1: Data Profiling, Quality Architecture & Visualizations

> [!NOTE]
> **Sprint 3 Deliverable Documentation**
> This repository module provides automated profiling, statistical measurement, rule-based anomaly detection, and visual analytics for raw listing datasets.

---

## 1. Executive Summary & Architecture Overview

The **Automated Data Cleaning & Validation System** (Module 1) transforms raw Airbnb listings data into structured statistical profiles, actionable visual diagrams, and rule-driven anomaly flags.

```
                  +--------------------------+
                  |    raw_data_loaded.parquet|
                  +------------+-------------+
                               |
                               v
                     +-------------------+
                     |  profiling_api.py |
                     +---------+---------+
                               |
       +-----------------------+-----------------------+
       |                       |                       |
       v                       v                       v
+--------------+   +-----------------------+   +---------------+
|profiling_    |   |suspicious_column_     |   |visualizations.|
|engine.py     |   |rules.py               |   |py             |
+------+-------+   +-----------+-----------+   +-------+-------+
       |                       |                       |
       +-----------------------+-----------------------+
                               |
                               v
               +-------------------------------+
               |     DELIVERABLE ARTIFACTS     |
               | - profiling_report.json       |
               | - heatmap_missing.png         |
               | - outlier_min_nights.png      |
               | - correlation_heatmap.png     |
               +-------------------------------+
```

---

## 2. API Reference & Usage Guide

Module 1 exposes a single programmatically callable entry point in `profiling_api.py`.

### Python API Example
```python
from profiling_api import run_profiling
import pandas as pd

# Option A: Run profiling on local parquet file (or raw dataset)
report = run_profiling(parquet_path="raw_data_loaded.parquet", output_json_path="profiling_report.json")

# Option B: Run profiling directly on an in-memory DataFrame
df = pd.read_csv("dataset.csv")
report = run_profiling(df=df, generate_visuals=True)
```

### CLI Command Execution
```powershell
python profiling_api.py
```

---

## 3. Profiling Report Schema Specification (`profiling_report.json`)

The deliverable report `profiling_report.json` follows a strict schema structured into 6 primary sections:

| Section Key | Description |
| :--- | :--- |
| `report_metadata` | Contains timestamp, schema version, and engine identification. |
| `dataset_summary` | Total row count, total column count, duplicate row count and percentage. |
| `per_column_stats` | Column data types, non-null counts, missing value counts, missing percentages, and cardinality metrics. |
| `correlation_matrix` | Symmetric pairwise correlation values between target numeric attributes (`minimum_nights`, `availability_365`, `number_of_reviews`, etc.). |
| `distribution_stats` | Standard parametric and non-parametric summary statistics (`count`, `mean`, `std`, `min`, `25%`, `50%`, `75%`, `max`). |
| `suspicious_rules_analysis` | Flagged PII columns, inconsistent formatting in mixed columns, and suspicious row anomalies. |
| `visualizations` | Paths to generated PNG plot deliverables (`heatmap_missing.png`, `outlier_min_nights.png`, `correlation_heatmap.png`). |

---

## 4. Suspicious Rules Engine & Findings

The rule engine (`suspicious_column_rules.py`) screens the dataset across three key vulnerability vectors:

### A. Personally Identifiable Information (PII)
> [!WARNING]
> PII features present privacy and regulatory compliance risks if stored unhashed or exposed.

- **Flagged PII Columns**:
  - `host_name`: Direct Identifier (Host Name)
  - `host_id`: Unique Host Identifier
  - `latitude`: Exact GPS Latitude Coordinate (Location PII)
  - `longitude`: Exact GPS Longitude Coordinate (Location PII)
  - Additional attributes flagged: `host_url`, `listing_url`, `host_thumbnail_url`, `host_picture_url`, `host_location`.

### B. Format Inconsistencies
- **Flagged Column**: `bathrooms_text`
- **Issue**: Mixed string formatting combining numerical counts (`"1.5 baths"`, `"1 bath"`), descriptive text (`"Half-bath"`, `"Shared half-bath"`), and null entries.
- **Action Required**: Upstream normalization rule to parse floats and extract shared/private flags (Sprint 6).

### C. Suspicious Row Anomalies
1. **Zero Price Rows (`price == "$0.00"`)**:
   - Flagged rows where price equals `$0.00`. Represents missing pricing data or promotional test listings.
2. **Extreme Minimum Stay (`minimum_nights > 365`)**:
   - Flagged rows where minimum stay exceeds 365 days (e.g. 365, 366, or 1000 nights). Represents data entry typos or non-standard long-term leases.

---

## 5. Visual Artifacts Summary

1. **`heatmap_missing.png`**:
   - Renders a missingness matrix illustrating missing value distribution across dataset columns. Highlighted gap features include `license`, `review_scores_rating`, `reviews_per_month`, and `bathrooms_text`.
2. **`outlier_min_nights.png`**:
   - Side-by-side boxplots of `minimum_nights` and `availability_365`. Highlights extreme outliers in `minimum_nights` extending beyond 365 days.
3. **`correlation_heatmap.png`**:
   - Heatmap depicting pairwise correlations among numerical attributes, identifying relationships between reviews per month, total number of reviews, and listing availability.

---

## 6. Verification & Automated Test Suite

To verify all deliverables against the Definition of Done:

```powershell
python verify_sprint3.py
```

Output:
```
==================================================
SPRINT 3 DEFINITION OF DONE VERIFICATION
==================================================
PASS: Ground truth dataset loaded (1000 rows, 75 columns).
PASS: All 5 required deliverables exist and are non-empty.
PASS: Image 'heatmap_missing.png' rendered and verified successfully.
PASS: Image 'outlier_min_nights.png' rendered and verified successfully.
PASS: Image 'correlation_heatmap.png' rendered and verified successfully.
PASS: profiling_report.json schema structure fully verified.
PASS: Required PII columns correctly flagged: ['host_name', 'host_id', 'latitude', 'longitude'].
PASS: 'bathrooms_text' format inconsistency correctly flagged.
PASS: Zero price rows verified.
PASS: Extreme minimum_nights (>365) rows verified.
PASS: Visualization file paths in report match deliverable filenames.
PASS: PROFILING_README.md contains presentation, schema, and API documentation.

==================================================
ALL SPRINT 3 DEFINITION OF DONE VERIFICATIONS PASSED!
==================================================
```
