# Automated Data Cleaning & Validation System

[![CI Pipeline](https://github.com/mrknight7772006-cyber/Automated-Data-Cleaning-Validation-System/actions/workflows/ci.yml/badge.svg)](https://github.com/mrknight7772006-cyber/Automated-Data-Cleaning-Validation-System/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Dockerized](https://img.shields.io/badge/docker-ready-brightgreen.svg)](https://www.docker.com/)

An enterprise-grade, end-to-end data engineering system built to profile, clean, impute, transform, validate, and score tabular datasets. Feature highlights include dynamic schema inference, ML imputation, fuzzy deduplication, multi-model AI anomaly detection, NLP text analysis, automated before/after quality delta scoring, and a containerized production CLI.

---

## 📑 Quick Navigation
- 🏗️ [**Architecture Specification (`ARCHITECTURE.md`)**](ARCHITECTURE.md)
- 📖 [**API & Schema Reference (`API_DOCS.md`)**](API_DOCS.md)
- 📊 [**Final Presentation & Demo Guide (`PRESENTATION.md`)**](PRESENTATION.md)
- 🔍 [**Module 1 Profiling Details (`PROFILING_README.md`)**](PROFILING_README.md)
- 🛡️ [**Module 3 Validation Details (`VALIDATION_README.md`)**](VALIDATION_README.md)

---

## ⚡ Quickstart Guide

### 1. Local Python Setup
```bash
# Clone repository
git clone https://github.com/mrknight7772006-cyber/Automated-Data-Cleaning-Validation-System.git
cd Automated-Data-Cleaning-Validation-System

# Install Python dependencies
pip install -r requirements.txt

# Run full automated integration test suite
pytest test_integration.py -v

# Run production CLI pipeline
python pipeline.py --input listings.csv.gz --output results/
```

### 2. Docker Container Execution
```bash
# Build production Docker container
docker build -t data-cleaning-pipeline .

# Execute containerized pipeline
docker run --rm -v $(pwd)/results:/app/results data-cleaning-pipeline --input listings.csv.gz --output results/
```

---

## 📦 Primary Output Deliverables

Running the pipeline populates the specified output directory (e.g., `results/`) with:
1. `profiling_report.json`: Comprehensive data profiling summary, cardinality stats, missingness heatmaps, and suspicious rule diagnostics.
2. `cleaned_data.csv`: Normalized, deduplicated, ML-imputed, and feature-engineered output dataset.
3. `validation_report.json`: Consolidated validation diagnostics, rule violation reports, AI ensemble anomaly logs, NLP classification metrics, and overall dataset health score ($0-100$) / grade (`A`-`F`).

---

## 🧪 Verification & Sprint Test Suites

```bash
# Verify Sprint 10 Orchestration & Config Shifts
python verify_sprint10.py

# Verify Sprint 11 CLI & End-to-End Integration
pytest test_integration.py -v
```

---

## 🎓 CadetX Internship Project Submission
- **Repository URL**: `https://github.com/mrknight7772006-cyber/Automated-Data-Cleaning-Validation-System`
- **Sprints Delivered**: Sprints 1 through 12 complete.
