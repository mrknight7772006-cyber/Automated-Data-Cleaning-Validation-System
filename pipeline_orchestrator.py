import os
import sys
import yaml
import time
import argparse
import pandas as pd
from datetime import datetime, timezone

from logger import get_pipeline_logger, StageTimer
from load_data import load_data
import profiling_api
from dedup_engine import deduplicate_dataframe
from imputers import impute_dataframe
from cleaning_api import CleaningAPI
from validation_api import ValidationAPI
from evaluate_validation import evaluate_pipeline

class PipelineOrchestrator:
    """
    Master Orchestrator chaining Modules 1-3:
    Stage 1: Profiling (run_profiling)
    Stage 2: Cleaning (run_cleaning)
    Stage 3: Validation (run_validation)
    """
    def __init__(self, config_path="config.yaml", input_path=None, output_dir=None):
        self.config_path = config_path
        if os.path.exists(config_path):
            self.config = self.load_config(config_path)
        else:
            self.config = {}
        
        p_cfg = self.config.get("pipeline", {})
        self.city_name = p_cfg.get("city_name", "Albany")
        self.input_path = input_path if input_path is not None else p_cfg.get("input_path", "listings.csv.gz")
        self.output_dir = output_dir if output_dir is not None else p_cfg.get("output_dir", "output")
        self.log_file = "pipeline.log"
        
        os.makedirs(self.output_dir, exist_ok=True)
        self.logger = get_pipeline_logger("PipelineOrchestrator", log_file=self.log_file)
        
    def load_config(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file missing: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
            
    def run_profiling_stage(self):
        """Stage 1: Profiling & Schema Inference"""
        prof_cfg = self.config.get("profiling", {})
        out_json = os.path.join(self.output_dir, "profiling_report.json")
        cache_parquet = os.path.join(self.output_dir, "raw_data_loaded.parquet")
        gen_visuals = prof_cfg.get("generate_visualizations", True)
        
        with StageTimer(self.logger, "STAGE 1: Profiling & Metadata Extraction"):
            self.logger.info(f"Reading raw data from '{self.input_path}'...")
            raw_df = load_data(self.input_path, cache_parquet)
            raw_count = len(raw_df)
            self.logger.info(f"Loaded raw dataset with {raw_count} rows and {len(raw_df.columns)} columns.")
            
            prof_report = profiling_api.run_profiling(
                df=raw_df,
                parquet_path=cache_parquet,
                output_json_path=out_json,
                generate_visuals=gen_visuals,
                output_dir=self.output_dir
            )
            self.logger.info(f"Profiling completed. Report written to '{out_json}'. Raw row count: {raw_count}.")
            return raw_df, prof_report
            
    def run_cleaning_stage(self, raw_df=None):
        """Stage 2: Data Cleaning, Deduplication, Imputation & Quality Scoring"""
        clean_cfg = self.config.get("cleaning", {})
        imputation_method = clean_cfg.get("imputation_method", "median")
        dedup_thresh = clean_cfg.get("deduplication_threshold", 0.85)
        cleaned_csv = os.path.join(self.output_dir, "cleaned_data.csv")
        imputed_parquet = os.path.join(self.output_dir, "imputed_data.parquet")
        cache_parquet = os.path.join(self.output_dir, "raw_data_loaded.parquet")
        
        with StageTimer(self.logger, f"STAGE 2: Data Cleaning & Imputation (Method: {imputation_method.upper()})"):
            if raw_df is None:
                if os.path.exists(cache_parquet):
                    raw_df = pd.read_parquet(cache_parquet)
                else:
                    raw_df = load_data(self.input_path, cache_parquet)
                    
            if not os.path.exists("dataset.csv"):
                raw_df.to_csv("dataset.csv", index=False)
                
            self.logger.info(f"Applying deduplication (threshold={dedup_thresh}), schema normalization, and ML imputation ('{imputation_method}')...")
            
            df_dedup, _ = deduplicate_dataframe(raw_df, similarity_threshold=dedup_thresh)
            df_imputed = impute_dataframe(df_dedup, ml_method=imputation_method)
            df_imputed.to_parquet(imputed_parquet)
            if imputed_parquet != "imputed_data.parquet":
                df_imputed.to_parquet("imputed_data.parquet")
                
            cleaner = CleaningAPI(imputed_parquet_path=imputed_parquet, raw_parquet_path=cache_parquet)
            cleaned_df, clean_meta = cleaner.run_pipeline(
                output_csv_path=cleaned_csv,
                log_json_path=os.path.join(self.output_dir, "cleaning_log.json"),
                after_score_path=os.path.join(self.output_dir, "quality_score_after.json"),
                delta_json_path=os.path.join(self.output_dir, "quality_delta.json"),
                before_score_path=os.path.join(self.output_dir, "quality_score_before.json")
            )
            
            if cleaned_csv != "cleaned_data.csv":
                cleaned_df.to_csv("cleaned_data.csv", index=False)
                
            cleaned_count = len(cleaned_df)
            self.logger.info(f"Cleaning completed. Output dataset shape: {cleaned_df.shape}. Written to '{cleaned_csv}'. Cleaned row count: {cleaned_count}.")
            return cleaned_df
            
    def run_validation_stage(self, cleaned_df=None):
        """Stage 3: Validation, Anomaly Detection & Scoring Engine"""
        val_cfg = self.config.get("validation", {})
        contamination = float(val_cfg.get("contamination", 0.05))
        val_report_path = os.path.join(self.output_dir, "validation_report.json")
        cleaned_csv = os.path.join(self.output_dir, "cleaned_data.csv")
        
        with StageTimer(self.logger, f"STAGE 3: Validation, Anomaly Detection & Scoring (Contamination: {contamination})"):
            val_api = ValidationAPI()
            if not os.path.exists(cleaned_csv) and cleaned_df is not None:
                cleaned_df.to_csv(cleaned_csv, index=False)
            if not os.path.exists("cleaned_data.csv") and cleaned_df is not None:
                cleaned_df.to_csv("cleaned_data.csv", index=False)
                
            self.logger.info(f"Running rule validator, AI anomaly detector (contamination={contamination}), NLP classifier, and error scorer...")
            val_report = val_api.run_validation_pipeline(
                input_path=cleaned_csv if os.path.exists(cleaned_csv) else "cleaned_data.csv",
                output_path=val_report_path,
                contamination=contamination,
                anomaly_report_path=os.path.join(self.output_dir, "anomaly_report.json"),
                nlp_report_path=os.path.join(self.output_dir, "nlp_classification_report.json")
            )
            
            eval_metrics = evaluate_pipeline(
                input_path=cleaned_csv if os.path.exists(cleaned_csv) else "cleaned_data.csv",
                report_path=val_report_path
            )
            
            health_score = val_report.get("overall_health_score", 0.0)
            health_grade = val_report.get("health_grade", "N/A")
            validated_count = val_report.get("total_rows_validated", 0)
            
            self.logger.info(f"Validation completed. Validated row count: {validated_count}. Overall Health Score: {health_score}/100 (Grade: {health_grade}).")
            return val_report
            
    def run_full_pipeline(self):
        """Executes full end-to-end pipeline: Profiling -> Cleaning -> Validation"""
        start_pipeline_time = time.time()
        self.logger.info("==================================================")
        self.logger.info(f"STARTING FULL PIPELINE ORCHESTRATION FOR '{self.city_name.upper()}'")
        self.logger.info("==================================================")
        
        raw_df, prof_report = self.run_profiling_stage()
        cleaned_df = self.run_cleaning_stage(raw_df)
        val_report = self.run_validation_stage(cleaned_df)
        
        total_duration = time.time() - start_pipeline_time
        self.logger.info("==================================================")
        self.logger.info(f"ALL PIPELINE STAGES COMPLETED SUCCESSFULLY IN {total_duration:.2f}s!")
        self.logger.info(f"Summary: Raw Rows = {len(raw_df)}, Cleaned Rows = {len(cleaned_df)}, Health Score = {val_report.get('overall_health_score')}/100 (Grade: {val_report.get('health_grade')})")
        self.logger.info("==================================================\n")
        return {
            "total_duration_seconds": round(total_duration, 2),
            "raw_rows": len(raw_df),
            "cleaned_rows": len(cleaned_df),
            "health_score": val_report.get("overall_health_score"),
            "health_grade": val_report.get("health_grade")
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Orchestrator for Data Cleaning & Validation System")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml file")
    args = parser.parse_args()
    
    orchestrator = PipelineOrchestrator(args.config)
    orchestrator.run_full_pipeline()
