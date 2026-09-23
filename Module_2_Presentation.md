# Module 2 Executive Presentation: Automated Data Cleaning & Validation System

**Project:** CadetX Internship — Airbnb Data Engineering Pipeline  
**Module 2 Focus:** End-to-End Data Ingestion, Profiling, Schema Enforcement, Imputation, Deduplication, Normalization & Feature Engineering  
**Date:** September 2026  

---

##  EXECUTIVE SUMMARY

The objective of Module 2 is to design, implement, and validate an automated data engineering pipeline for unstructured/semi-structured Airbnb listing data. The system takes raw, noisy, incomplete dataset files and transforms them into an analytics-ready, feature-engineered, high-quality analytical dataset (`cleaned_data.csv`).

Key Achievements across Sprints 1–6:
- **Measurable Quality Improvement:** Raised the overall Data Quality Score from **94.45/100** to **96.08/100** (+1.63 points overall delta).
- **100% Schema Non-Nullability Compliance:** Resolved all missing values across all non-nullable schema columns using statistical and ML-based imputers (`KNNImputer`/`IterativeImputer`), retaining documented legitimate exceptions (`license`).
- **Deduplication:** Identified and eliminated exact full-row duplicates, exact location duplicates, and fuzzy near-duplicate listings from the same host at the same address.
- **Normalization:** Stripped currency formatting (`$`, `,`) casting `price` to `float64`, parsed `bathrooms_text` into numeric `bathrooms` and categorical `bathroom_type`, standardized neighbourhood spelling variants, and standardized date formats (`YYYY-MM-DD`).
- **Feature Engineering:** Extracted `host_tenure_days`, `price_per_bedroom`, `room_type_encoded`, `neighbourhood_group_cleansed_encoded`, and `bathroom_type`.
- **Automated Quality Assurance:** Developed a full `pytest` verification suite (`test_cleaning.py`) ensuring zero regression and 100% Definition of Done (DoD) compliance.

---

## PIPELINE ARCHITECTURE & SPRINT ROADMAP

```
[Raw CSV / GZ Data]
       │
       ▼ (Sprint 1: Data Ingestion & Metadata)
[raw_data_loaded.parquet] + [metadata_report.json]
       │
       ▼ (Sprint 2: Statistical Profiling)
[stats_summary.json]
       │
       ▼ (Sprint 3: Advanced Profiling & Visualizations)
[profiling_report.json] + [heatmap_missing.png, outlier_min_nights.png, correlation_heatmap.png]
       │
       ▼ (Sprint 4: Schema Inference & Pre-Clean Scoring)
[schema.json] + [quality_score_before.json] (Score: 94.45)
       │
       ▼ (Sprint 5: Imputation & Deduplication)
[imputed_data.parquet] + [dedup_report.json]
       │
       ▼ (Sprint 6: Normalization, Transformation & Post-Clean Scoring)
[cleaned_data.csv] + [cleaning_log.json] + [quality_score_after.json] (Score: 96.08) + [quality_delta.json] (+1.63)
```

---

## SPRINT-BY-SPRINT IMPLEMENTATION SUMMARY

### Sprint 1 — Data Ingestion & Metadata Engine
- Built `load_data.py` and `metadata_engine.py`.
- Ingested compressed raw CSV (`listings.csv.gz`), converting raw strings to optimized Parquet format (`raw_data_loaded.parquet`).
- Generated `metadata_report.json` cataloging semantic data types (ID, URL, date, numeric, text, mixed-type flags).

### Sprint 2 — Profiling & Summary Statistics
- Built `profiling_engine.py`.
- Generated `stats_summary.json` computing missing percentages, unique value counts, numeric distributions, and missingness gap analysis across all 90 raw columns.
- Identified primary missingness vectors: `license` (100%), `neighbourhood_group_cleansed` (100%), `review_scores_rating` (12.24%), `bedrooms` (22.86%), `price` (6.53%).

### Sprint 3 — Visualization Engine & Anomaly Detection
- Built `visualizations.py`, `suspicious_column_rules.py`, and `profiling_api.py`.
- Detected PII/security risks (host profile details, geographic coordinates).
- Generated publication-grade PNG visualizations:
  - `heatmap_missing.png`: Missingness pattern visualization.
  - `outlier_min_nights.png`: Minimum nights distribution and extreme value flagging.
  - `correlation_heatmap.png`: Symmetric correlation matrix across key numerical features.

### Sprint 4 — Schema Inference & Pre-Clean Quality Scoring
- Built `schema_inference.py` and `quality_scorer.py`.
- Inferred production schema rules (`schema.json`) specifying range constraints, allowed sets, format patterns, and nullability flags.
- Computed pre-cleaning baseline Quality Score (`quality_score_before.json`): **94.45/100**.

### Sprint 5 — Missing Value Imputation & Deduplication Engine
- Built `imputers.py` and `dedup_engine.py`.
- Implemented mode imputation for categorical attributes and grouped median for `review_scores_rating` by `room_type`.
- Implemented ML-based `KNNImputer` for numeric features (`bedrooms`, `beds`).
- Executed exact key matching and fuzzy string matching (sequence matcher threshold 0.50) on host address/coordinates, removing duplicate listings.

### Sprint 6 — Normalization, Feature Engineering & Post-Clean Scoring
- Built `normalizer.py`:
  - `normalize_price`: Stripped `$` and commas, cast to `float64`, imputed non-zero median prices.
  - `parse_bathrooms`: Extracted numeric `bathrooms` (`float64`) and `bathroom_type` flag (`shared`, `half`, `private`, `standard`).
  - `standardize_neighbourhoods`: Standardized casing, whitespace, and unified spelling variants.
  - `parse_dates`: Standardized `last_scraped` and `host_since` to ISO `YYYY-MM-DD` datetimes, deriving missing host tenure.
- Built `transformer.py`:
  - Categorical label encoding for `room_type` -> `room_type_encoded` and `neighbourhood_group_cleansed` -> `neighbourhood_group_cleansed_encoded`.
  - Feature extraction for `host_tenure_days` = `(last_scraped - host_since)` in days.
  - Feature extraction for `price_per_bedroom` = `(price / max(bedrooms, 1))`.
- Built `cleaning_api.py`, `test_cleaning.py` (pytest suite), and `run_sprint6.py`.

---

## EMPIRICAL QUALITY IMPROVEMENT & DELTA ANALYSIS

| Quality Dimension | Pre-Clean Score (`quality_score_before.json`) | Post-Clean Score (`quality_score_after.json`) | Delta Score (`quality_delta.json`) | Impact Description |
| :--- | :---: | :---: | :---: | :--- |
| **Completeness Sub-Score** | 84.66 / 100 | **88.25 / 100** | **+3.59** | Imputed all missing values across key non-nullable columns. |
| **Consistency Sub-Score** | 98.73 / 100 | **100.00 / 100** | **+1.27** | Standardized currency strings, bathroom text formats, and unique listing IDs. |
| **Validity Sub-Score** | 99.96 / 100 | **100.00 / 100** | **+0.04** | Validated price ranges, minimum night boundaries, and room type sets. |
| **OVERALL QUALITY SCORE** | **94.45 / 100** | **96.08 / 100** | **+1.63** | **MEASURABLE IMPROVEMENT VERIFIED** |

---

## DELIVERABLES SUMMARY

1. **`cleaned_data.csv`**: Final analytics-ready cleaned and feature-engineered dataset.
2. **`cleaning_log.json`**: Audited log of all normalization, transformation, and feature extraction operations.
3. **`quality_score_after.json`**: Post-clean Data Quality Score report (**96.08/100**).
4. **`quality_delta.json`**: Quantitative pre-vs-post clean score improvement report (**+1.63**).
5. **`test_cleaning.py`**: Automated Pytest verification suite (8/8 tests passed).
6. **`Module_2_Presentation.md`**: Executive presentation document.

---

## CONCLUSION & NEXT STEPS

Module 2 successfully establishes an automated, reproducible, production-grade data cleaning, feature engineering, and quality validation pipeline. The dataset `cleaned_data.csv` is fully verified against all non-nullability, format, and logical constraints, ready for downstream machine learning modeling, demand forecasting, and analytics dashboards.
