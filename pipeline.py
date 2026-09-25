#!/usr/bin/env python3
"""
Automated Data Cleaning & Validation System — Pipeline CLI Entrypoint (Sprint 11)
Executes end-to-end data processing pipeline: Profiling -> Cleaning -> Validation.

Usage:
    python pipeline.py --input listings.csv.gz --output results/
    python pipeline.py --input dataset.csv --output results/ --config config.yaml
"""

import os
import sys
import argparse
from typing import Dict, Any, Optional

from pipeline_orchestrator import PipelineOrchestrator

def run_pipeline(
    input_path: str = "listings.csv.gz",
    output_dir: str = "results",
    config_path: str = "config.yaml"
) -> Dict[str, Any]:
    """
    Programmatic entrypoint to run the end-to-end cleaning and validation pipeline.

    Args:
        input_path: Path to the raw dataset CSV or CSV.GZ file.
        output_dir: Directory where all generated output artifacts will be saved.
        config_path: Path to pipeline configuration YAML file.

    Returns:
        Dict containing pipeline summary metrics (duration, raw_rows, cleaned_rows, health_score, health_grade).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize orchestrator with overridden input and output paths
    orchestrator = PipelineOrchestrator(
        config_path=config_path,
        input_path=input_path,
        output_dir=output_dir
    )
    
    # Run full pipeline stages: Profiling -> Cleaning -> Validation
    summary = orchestrator.run_full_pipeline()
    return summary

def main() -> int:
    """
    CLI Entrypoint parsing command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Automated Data Cleaning & Validation System CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-i", "--input",
        default="listings.csv.gz",
        help="Path to input raw dataset file (e.g. listings.csv.gz)"
    )
    parser.add_argument(
        "-o", "--output",
        default="results",
        help="Path to output directory for results and reports"
    )
    parser.add_argument(
        "-c", "--config",
        default="config.yaml",
        help="Path to configuration YAML file"
    )
    
    args = parser.parse_args()
    
    print("==================================================")
    print("AUTOMATED DATA CLEANING & VALIDATION SYSTEM — CLI")
    print("==================================================")
    print(f"Input Dataset:    {args.input}")
    print(f"Output Directory: {args.output}")
    print(f"Config File:      {args.config}")
    print("==================================================\n")
    
    if not os.path.exists(args.input):
        print(f"ERROR: Input dataset path '{args.input}' does not exist!", file=sys.stderr)
        return 1
        
    try:
        summary = run_pipeline(
            input_path=args.input,
            output_dir=args.output,
            config_path=args.config
        )
        print("\n==================================================")
        print("PIPELINE CLI EXECUTION SUCCESSFUL!")
        print(f"Outputs written to: {os.path.abspath(args.output)}")
        print(f"Raw Rows: {summary['raw_rows']} | Cleaned Rows: {summary['cleaned_rows']}")
        print(f"Overall Health Score: {summary['health_score']}/100 (Grade: {summary['health_grade']})")
        print("==================================================")
        return 0
    except Exception as e:
        print(f"\nERROR: Pipeline execution failed with exception: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
