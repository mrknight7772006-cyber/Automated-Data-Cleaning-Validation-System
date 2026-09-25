"""
Integration Test Suite for Automated Data Cleaning & Validation System (Sprint 11)
Runs the end-to-end pipeline on a sampled subset of raw listings data and asserts
that all 3 primary outputs exist and satisfy schema validation constraints.

Execute with:
    pytest test_integration.py
"""

import os
import json
import gzip
import shutil
import pytest
import pandas as pd
import sys
from pathlib import Path

from pipeline import run_pipeline, main

@pytest.fixture
def sample_raw_dataset_gz(tmp_path: Path) -> str:
    """
    Creates a sampled subset (50 rows) from the raw dataset and saves it
    as a gzipped CSV file inside a temporary directory.
    """
    raw_source = "listings.csv.gz"
    if not os.path.exists(raw_source):
        raw_source = "dataset.csv"
        
    assert os.path.exists(raw_source), f"Raw source dataset '{raw_source}' missing for integration test!"
    
    if raw_source.endswith(".gz"):
        df = pd.read_csv(raw_source, compression="gzip", low_memory=False)
    else:
        df = pd.read_csv(raw_source, low_memory=False)
        
    sample_df = df.head(50)
    
    sample_csv = tmp_path / "sample_listings.csv"
    sample_gz = tmp_path / "sample_listings.csv.gz"
    
    sample_df.to_csv(sample_csv, index=False)
    with open(sample_csv, "rb") as f_in:
        with gzip.open(sample_gz, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
            
    return str(sample_gz)

def test_full_pipeline_integration(sample_raw_dataset_gz: str, tmp_path: Path):
    """
    Runs the full end-to-end pipeline on the sampled dataset and verifies:
    1. All 3 primary output deliverables (profiling_report.json, cleaned_data.csv, validation_report.json) exist.
    2. Output files are schema-valid according to system specifications.
    """
    output_dir = tmp_path / "integration_results"
    
    # Execute full pipeline programmatically
    summary = run_pipeline(
        input_path=sample_raw_dataset_gz,
        output_dir=str(output_dir),
        config_path="config.yaml"
    )
    
    assert summary is not None, "Pipeline run_pipeline() returned None"
    assert summary["raw_rows"] == 50, f"Expected 50 raw rows, got {summary['raw_rows']}"
    assert summary["cleaned_rows"] > 0, "Cleaned rows count must be > 0"

    prof_report_file = output_dir / "profiling_report.json"
    cleaned_csv_file = output_dir / "cleaned_data.csv"
    val_report_file = output_dir / "validation_report.json"
    
    # ----------------------------------------------------
    # 1. ASSERT ALL 3 PRIMARY OUTPUT DELIVERABLES EXIST
    # ----------------------------------------------------
    assert prof_report_file.exists(), f"Stage 1 deliverable missing: {prof_report_file}"
    assert cleaned_csv_file.exists(), f"Stage 2 deliverable missing: {cleaned_csv_file}"
    assert val_report_file.exists(), f"Stage 3 deliverable missing: {val_report_file}"

    # ----------------------------------------------------
    # 2. SCHEMA VALIDATION: Stage 1 — profiling_report.json
    # ----------------------------------------------------
    with open(prof_report_file, "r", encoding="utf-8") as f:
        prof_data = json.load(f)
        
    expected_prof_keys = {
        "report_metadata",
        "dataset_summary",
        "per_column_stats",
        "correlation_matrix",
        "distribution_stats",
        "suspicious_rules_analysis",
        "visualizations"
    }
    assert expected_prof_keys.issubset(set(prof_data.keys())), f"Profiling report missing keys: {expected_prof_keys - set(prof_data.keys())}"
    assert prof_data["dataset_summary"]["total_rows"] == 50, f"Profiling dataset summary total_rows expected 50, got {prof_data['dataset_summary']['total_rows']}"
    assert prof_data["dataset_summary"]["total_columns"] > 0, "Profiling dataset summary total_columns must be > 0"
    assert len(prof_data["per_column_stats"]) > 0, "Profiling per_column_stats is empty"

    # ----------------------------------------------------
    # 3. SCHEMA VALIDATION: Stage 2 — cleaned_data.csv
    # ----------------------------------------------------
    cleaned_df = pd.read_csv(cleaned_csv_file, low_memory=False)
    assert len(cleaned_df) > 0, "Cleaned dataset CSV contains no rows"
    assert len(cleaned_df.columns) > 0, "Cleaned dataset CSV contains no columns"
    assert "price" in cleaned_df.columns or "price_per_bedroom" in cleaned_df.columns, "Cleaned dataset missing expected normalized columns"

    # ----------------------------------------------------
    # 4. SCHEMA VALIDATION: Stage 3 — validation_report.json
    # ----------------------------------------------------
    with open(val_report_file, "r", encoding="utf-8") as f:
        val_data = json.load(f)
        
    expected_val_keys = {
        "timestamp",
        "dataset_path",
        "total_rows_validated",
        "overall_health_score",
        "health_grade",
        "sub_scores",
        "severity_summary",
        "rule_validation_summary",
        "anomaly_detection_summary",
        "flagged_rows_master_log"
    }
    assert expected_val_keys.issubset(set(val_data.keys())), f"Validation report missing keys: {expected_val_keys - set(val_data.keys())}"
    
    score = val_data["overall_health_score"]
    grade = val_data["health_grade"]
    assert isinstance(score, (int, float)), f"overall_health_score should be numeric, got {type(score)}"
    assert 0.0 <= score <= 100.0, f"overall_health_score out of bounds [0, 100]: {score}"
    assert isinstance(grade, str) and len(grade) > 0, f"health_grade must be non-empty string, got {grade}"
    assert val_data["total_rows_validated"] > 0, "total_rows_validated must be > 0"

def test_cli_argument_overrides(sample_raw_dataset_gz: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Verifies that running pipeline.py via CLI with custom --input and --output flags
    correctly generates outputs in the specified target directory.
    """
    cli_out_dir = tmp_path / "cli_override_results"
    
    test_argv = ["pipeline.py", "--input", sample_raw_dataset_gz, "--output", str(cli_out_dir)]
    monkeypatch.setattr(sys, "argv", test_argv)
    
    exit_code = main()
    assert exit_code == 0, f"CLI main() returned error exit code: {exit_code}"
    
    assert (cli_out_dir / "profiling_report.json").exists()
    assert (cli_out_dir / "cleaned_data.csv").exists()
    assert (cli_out_dir / "validation_report.json").exists()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
