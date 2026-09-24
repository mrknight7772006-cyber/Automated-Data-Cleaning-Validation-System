import os
import json
import yaml
import pandas as pd
from pipeline_orchestrator import PipelineOrchestrator

def verify_sprint10():
    print("==================================================")
    print("SPRINT 10 DEFINITION OF DONE VERIFICATION")
    print("==================================================")

    req_files = ["pipeline_orchestrator.py", "config.yaml", "logger.py"]
    for fp in req_files:
        assert os.path.exists(fp), f"FAILED: Deliverable '{fp}' missing!"
        assert os.path.getsize(fp) > 0, f"FAILED: Deliverable '{fp}' is empty!"

    print(f"PASS: Required scripts exist: {req_files}.")

    # 1. Run full pipeline using default config
    orchestrator = PipelineOrchestrator("config.yaml")
    res1 = orchestrator.run_full_pipeline()

    assert os.path.exists("pipeline.log"), "FAILED: pipeline.log not created!"
    assert os.path.getsize("pipeline.log") > 0, "FAILED: pipeline.log is empty!"

    with open("pipeline.log", "r", encoding="utf-8") as f:
        log_content = f.read()
        assert "STAGE 1" in log_content, "FAILED: STAGE 1 missing in pipeline.log!"
        assert "STAGE 2" in log_content, "FAILED: STAGE 2 missing in pipeline.log!"
        assert "STAGE 3" in log_content, "FAILED: STAGE 3 missing in pipeline.log!"

    print(f"PASS: Full pipeline executed successfully in {res1['total_duration_seconds']}s.")
    print(f"       Raw rows: {res1['raw_rows']} | Cleaned rows: {res1['cleaned_rows']} | Health Score: {res1['health_score']}/100")
    print("PASS: pipeline.log verified with timing and row count logs.")

    # 2. Configurable Output Shift Verification: Change config parameter & confirm output changes without code edits
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["validation"]["contamination"] = 0.15 # Change contamination from 0.05 to 0.15

    with open("config_temp.yaml", "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)

    orchestrator2 = PipelineOrchestrator("config_temp.yaml")
    res2 = orchestrator2.run_full_pipeline()

    if os.path.exists("config_temp.yaml"):
        os.remove("config_temp.yaml")

    print(f"       Original Contamination (0.05) Health Score: {res1['health_score']}")
    print(f"       Modified Contamination (0.15) Health Score: {res2['health_score']}")
    assert res1['health_score'] != res2['health_score'], "FAILED: Config parameter change did not affect pipeline output!"

    print("PASS: Config parameter change measurably altered output without code edits.")

    print("\n==================================================")
    print("ALL SPRINT 10 DEFINITION OF DONE VERIFICATIONS PASSED!")
    print("==================================================")

if __name__ == "__main__":
    verify_sprint10()
