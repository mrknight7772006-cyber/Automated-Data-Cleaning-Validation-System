import os
import sys
import pytest
from cleaning_api import CleaningAPI

def main():
    print("==================================================")
    print("RUNNING SPRINT 6 PIPELINE & VERIFICATIONS")
    print("==================================================")

    # 1. Execute Cleaning API
    api = CleaningAPI()
    cleaned_df, log_summary = api.run_pipeline()

    # 2. Run Pytest Suite on test_cleaning.py
    print("\nRunning Pytest suite 'test_cleaning.py'...")
    pytest_args = ["test_cleaning.py", "-v"]
    exit_code = pytest.main(pytest_args)

    if exit_code != 0:
        print(f"FAILED: Pytest suite 'test_cleaning.py' returned non-zero exit code {exit_code}")
        sys.exit(exit_code)

    print("PASS: Pytest suite passed successfully!")

    # 3. Regression Testing (Sprints 1-5)
    print("\nRunning Sprint 1-5 Regression Verification Scripts...")
    for sprint_num in range(1, 6):
        script_name = f"verify_sprint{sprint_num}.py"
        if os.path.exists(script_name):
            print(f"Executing {script_name}...")
            # Execute dynamically
            with open(script_name, "r", encoding="utf-8") as f:
                code_str = f.read()
            exec_globals = {"__name__": "__main__"}
            exec(code_str, exec_globals)

    print("\n==================================================")
    print("ALL SPRINT 6 DEFINITION OF DONE CRITERIA PASSED!")
    print("==================================================")

if __name__ == "__main__":
    main()
