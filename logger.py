import os
import sys
import logging
import time

def get_pipeline_logger(name="PipelineOrchestrator", log_file="pipeline.log", log_level=logging.INFO):
    """
    Configures and returns a structured logger for the Pipeline Orchestrator.
    Logs messages to both file (pipeline.log) and console with timestamps.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    if logger.handlers:
        return logger
        
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    logger.addHandler(console_handler)
    
    return logger

class StageTimer:
    """Helper context manager to log timing of pipeline stages."""
    def __init__(self, logger, stage_name):
        self.logger = logger
        self.stage_name = stage_name
        self.start_time = None
        self.duration = None
        
    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(f">>> STARTING {self.stage_name}...")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.duration = time.time() - self.start_time
        if exc_type is None:
            self.logger.info(f"<<< COMPLETED {self.stage_name} in {self.duration:.2f}s.\n")
        else:
            self.logger.error(f"!!! FAILED {self.stage_name} after {self.duration:.2f}s: {exc_val}\n")
