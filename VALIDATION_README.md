# Automated Data Cleaning & Validation System
## Module 3 — Error Detection, Validation Scoring & Unified API

---

### 1. Executive Summary & Architecture

Module 3 delivers an enterprise-grade **Unified Validation & Anomaly Scoring Pipeline** for real estate and short-term rental listings datasets. It consolidates four distinct validation layers into a single, cohesive, severity-weighted output with an overall **Dataset Health Score (0–100)** and quantitative evaluation metrics.

```
+-----------------------------------------------------------------------------------+
|                            INPUT DATASET (cleaned_data.csv)                        |
+-----------------------------------------------------------------------------------+
                                          |
      +-------------------+---------------+-------------------+-------------------+
      |                   |                                   |                   |
      v                   v                                   v                   v
+--------------+  +---------------+                   +---------------+  +-------------------+
| Rule         |  | AI Anomaly    |                   | NLP Text      |  | Statistical Error |
| Validation   |  | Detection     |                   | Classifier    |  | & Data Drift      |
| Engine       |  | (IF, LOF, AE) |                   | & Amenities   |  | Engine            |
+--------------+  +---------------+                   +---------------+  +-------------------+
      |                   |                                   |                   |
      +-------------------+---------------+-------------------+-------------------+
                                          |
                                          v
                        +-----------------------------------+
                        | Validation Scoring Engine         |
                        | (Severity Matrix & Health Score)  |
                        +-----------------------------------+
                                          |
                                          v
                        +-----------------------------------+
                        | Unified Validation API            |
                        | (validation_report.json)          |
                        +-----------------------------------+
```

---

### 2. Core Components & Technical Specifications

#### A. Error Detector & Data Drift Engine (`error_detector.py`)
- **Statistical Error Detection**: Flags values exceeding domain constraints or statistical thresholds:
  - Extreme Price Outliers: $\text{Price} > \text{Q3} + 3 \times \text{IQR}$ or $\text{Price} \le 0$.
  - Extreme Minimum Nights Outliers: $\text{Minimum Nights} > 365$ or $> \text{Q3} + 3 \times \text{IQR}$.
  - Suspicious Inconsistencies: Zero calendar availability on high-priced luxury listings ($\text{Price} > \$500$).
  - Logical Contradictions: $\text{Minimum Nights} > \text{Maximum Nights}$.
- **Feature Data Drift Detection**: Splits dataset into temporal host cohorts (`Early Cohort` vs `Recent Cohort` based on `host_since` median split) and evaluates statistical shift using:
  - **2-Sample Kolmogorov-Smirnov (KS) Test**: Evaluates distribution equality ($p < 0.05$ flags drift).
  - **Wasserstein Distance**: Measures Earth Mover's Distance between feature distributions.

#### B. AI Anomaly Detector (`anomaly_detector.py`)
- **Isolation Forest (IF)**: Tree-partitioning anomaly detection ($5\%$ contamination).
- **Local Outlier Factor (LOF)**: Density-based local outlier detection ($k=20$ neighbors).
- **Autoencoder (MLP Neural Network - Stretch Goal)**: $4 \rightarrow 2 \rightarrow 4$ architecture calculating Mean Squared Error (MSE) reconstruction loss.
- **Overlap Analysis**: Calculates Jaccard similarity index and model consensus flags across all three algorithms.

#### C. NLP Column Classifier & Amenities Parser (`nlp_column_classifier.py`)
- **Text Classification**: Categorizes `description` and `neighborhood_overview` into:
  - `GENUINE_LISTING_TEXT`: Substantive, descriptive text.
  - `PLACEHOLDER`: Text matching placeholder patterns (*"N/A"*, *"none"*, *"null"*, single-word listings).
  - `NEAR_EMPTY_OR_SHORT`: Descriptions $\le 5$ words or $< 25$ characters.
  - `MISSING_OR_NULL`: NaN / empty strings.
- **Amenities Parsing & Standardization**: Cleans corrupted unicode artifacts (`\ufffd`), standardizes en-dashes (`\u2013`), normalizes whitespace, deduplicates amenities, and extracts top 20 amenity categories.

#### D. Validation Scorer (`validation_scorer.py`)
Weight-based severity matrix assigning penalties to compute sub-scores and the **Overall Dataset Health Score (0–100)**:

$$\text{Overall Health Score} = 0.35 \times \text{Rule Subscore} + 0.25 \times \text{Anomaly Subscore} + 0.20 \times \text{NLP Subscore} + 0.20 \times \text{Integrity Subscore}$$

| Severity | Weight | Criteria / Description |
| :--- | :--- | :--- |
| **CRITICAL** | `15.0` | Bounding box errors, $\text{Price} \le 0$, $\text{Min Nights} > \text{Max Nights}$. |
| **HIGH** | `10.0` | Statistical 3x IQR outliers, IF & LOF ML consensus anomalies. |
| **MEDIUM** | `5.0` | Single model ML anomaly (IF or LOF only), missing description field. |
| **LOW** | `2.0` | Near-empty text, duplicate amenity items. |

#### E. Model Evaluation Engine (`evaluate_validation.py`)
Evaluates system predictions against a deterministic ground-truth **200-row sample**:
- **Precision**: $\frac{TP}{TP + FP}$
- **Recall**: $\frac{TP}{TP + FN}$
- **F1-Score**: $2 \times \frac{P \times R}{P + R}$
- **Accuracy**: $\frac{TP + TN}{\text{Total Sample}}$
- **ROC AUC Score & Curve**: Plots Sensitivity vs $1 - \text{Specificity}$ (`roc_curve.png`).

---

### 3. Execution & API Quickstart

#### Run Unified Validation Pipeline:
```bash
python validation_api.py
```

#### Run Model Evaluation & ROC Curve Generation:
```bash
python evaluate_validation.py
```

---

### 4. Output Deliverables & Schema

- [`validation_report.json`](file:///C:/Users/HAI/OneDrive/Documents/cadetx%20intenship/validation_report.json): Consolidated master report containing overall health score, sub-scores, severity summary, data drift metrics, and row-by-row violation logs.
- [`evaluation_metrics.json`](file:///C:/Users/HAI/OneDrive/Documents/cadetx%20intenship/evaluation_metrics.json): Model evaluation metrics (Precision, Recall, F1, Accuracy, Confusion Matrix, ROC AUC).
- [`roc_curve.png`](file:///C:/Users/HAI/OneDrive/Documents/cadetx%20intenship/roc_curve.png): Publication-quality ROC curve visualization.

---
