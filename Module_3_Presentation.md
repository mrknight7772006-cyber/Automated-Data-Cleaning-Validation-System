# Module 3 Presentation: Advanced Data Quality, AI Anomaly Detection & Unified Validation API

---

## Slide 1: Executive Overview & Project Goals

- **Objective**: Build an end-to-end, automated data validation, anomaly detection, NLP text quality classification, and scoring engine for Airbnb listings.
- **Business Impact**:
  - Eliminates silent data corruption before downstream analytics/ML modeling.
  - Replaces rigid manual checks with AI-driven statistical & machine learning anomaly detection.
  - Delivers a single, executive-level **Dataset Health Score (0–100)** with clear severity metrics.

---

## Slide 2: Data Validation & Statistical Error Architecture

- **Rule Validation Engine (`rule_validator.py`)**:
  - Evaluates 7 core domain rules: price $> 0$, minimum nights range, availability range, valid room types, valid neighborhood groups, GPS coordinate bounding box, $\text{min\_nights} \le \text{max\_nights}$.
- **Statistical Error Detector (`error_detector.py`)**:
  - Identifies $3 \times \text{IQR}$ extreme upper bound price outliers ($> \$549.30$).
  - Detects impossible minimum night constraints ($> 365$ days).
  - Flags domain contradictions (luxury pricing $> \$500$ with 0 calendar availability).

---

## Slide 3: AI Anomaly Detection Engine (IF + LOF + Autoencoder)

- **Multivariate Feature Matrix**: `price`, `minimum_nights`, `availability_365`, `number_of_reviews`.
- **Tri-Model Ensemble**:
  1. **Isolation Forest (IF)**: Tree partitioning model flagging 23 anomalies ($5.0\%$ contamination).
  2. **Local Outlier Factor (LOF)**: Density-based nearest neighbor outlier detection flagging 23 anomalies.
  3. **Autoencoder (MLP Neural Network)**: Deep neural reconstruction error flagging 23 anomalies.
- **Ensemble Consensus**:
  - Overlap between IF & LOF: **14 listings** (**Jaccard Similarity: 0.4375**).
  - All 3 models consensus: **11 listings**.

---

## Slide 4: NLP Text Quality Classifier & Amenities Standardization

- **NLP Description Classification (`nlp_column_classifier.py`)**:
  - **Genuine Listing Text**: **448 listings** ($97.39\%$).
  - **Missing / Null Descriptions**: **9 listings** ($1.96\%$).
  - **Near-Empty / Short Text**: **3 listings** ($0.65\%$, e.g. *"Great please safe neighborhood"*).
- **Amenities Field Standardization**:
  - Cleaned corrupted unicode artifacts (`\ufffd`), decoded en-dashes (`\u2013`), and normalized whitespace across **205 listings**.
  - Average of **37.95 amenities per listing** across **468 unique standardized amenities**.
  - Top Amenities: Wifi (**439**), Smoke alarm (**439**), Kitchen (**416**), Air conditioning (**382**).

---

## Slide 5: Data Drift Analysis & Statistical Shift Metrics

- **Cohort Splitting**: Divided dataset into `Early Host Cohort` ($n=232$) vs `Recent Host Cohort` ($n=228$) using median `host_since` date (`2019-09-02`).
- **Statistical Test Results**:
  - **Price**: KS Statistic = `0.1620`, $p$-value = `0.0041` $\rightarrow$ **`DRIFT_DETECTED`** (Mean shift: $\$196.22 \rightarrow \$155.57$).
  - **Number of Reviews**: KS Statistic = `0.3128`, $p$-value = `0.0000` $\rightarrow$ **`DRIFT_DETECTED`** (Mean shift: $87.89 \rightarrow 31.28$ reviews).
  - **Minimum Nights**: KS Statistic = `0.0871`, $p$-value = `0.3209` $\rightarrow$ **`STABLE`**.
  - **Availability 365**: KS Statistic = `0.0938`, $p$-value = `0.2424` $\rightarrow$ **`STABLE`**.

---

## Slide 6: Validation Scoring Engine & Dataset Health Score

- **Overall Dataset Health Score**: **71.21 / 100** (**Grade: B**)
- **Sub-Score Breakdown**:
  - **Rule Validation Sub-score**: **91.74 / 100**
  - **Anomaly Health Score**: **79.57 / 100**
  - **NLP Text Quality Sub-score**: **98.04 / 100**
  - **Data Integrity & Statistical Error Sub-score**: **0.00 / 100** (heavy penalty from extreme price/min_nights statistical outliers).
- **Severity Summary**:
  - **Critical Issues**: 12
  - **High Issues**: 47
  - **Medium Issues**: 32
  - **Low Issues**: 3

---

## Slide 7: Unified Data Validation API Architecture

- **`ValidationAPI` (`validation_api.py`)**:
  - Single method execution: `api.run_validation_pipeline()`
  - Integrates all 4 engines (`RuleValidator`, `AnomalyDetector`, `NLPColumnClassifier`, `ErrorDetector`).
  - Emits unified [`validation_report.json`](file:///C:/Users/HAI/OneDrive/Documents/cadetx%20intenship/validation_report.json) containing top-level executive metrics and row-level master violation logs.

---

## Slide 8: Model Evaluation & ROC Curve Performance

- **Evaluation Benchmark**: Evaluated system predictions on a deterministic ground-truth sample of $n=200$ listings.
- **Performance Metrics**:
  - **Precision**: `0.6667`
  - **Recall**: `1.0000` ($100\%$ detection rate of ground-truth errors)
  - **F1-Score**: `0.8000`
  - **Accuracy**: `0.9500` ($95.0\%$ overall classification accuracy)
  - **ROC AUC Score**: `0.9944`
- **Confusion Matrix**:
  - True Negatives (TN): `176`
  - False Positives (FP): `9`
  - False Negatives (FN): `0`
  - True Positives (TP): `15`

---

## Slide 9: Key Insights & Future Recommendations

1. **Automation & API Integration**: Embed `ValidationAPI` into CI/CD pipelines to block corrupted data ingest before database commits.
2. **Dynamic Outlier Thresholds**: Use seasonal percentile thresholds for luxury price outlier detection.
3. **Continuous Drift Monitoring**: Set up automated alerts for feature drift ($p < 0.05$) to trigger model retraining.

---
