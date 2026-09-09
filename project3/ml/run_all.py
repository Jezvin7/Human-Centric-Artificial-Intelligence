from pathlib import Path
import subprocess
import sys


BASE_DIR = Path(__file__).resolve().parent


EXPERIMENT_SCRIPTS = [
    "train_baseline.py",
    "evaluate_expert.py",
    "train_defer.py",
    "active_learning.py",
]


REPORT_SCRIPT = "generate_report.py"


def run_script(script_name):

    script_path = BASE_DIR / script_name

    print("\n" + "=" * 60)
    print(f"Running: {script_name}")
    print("=" * 60)

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
        ]
    )

    if result.returncode != 0:

        print(
            f"\nERROR: {script_name} failed."
        )

        sys.exit(
            result.returncode
        )


def main():

    print(
        "\nStarting Project 3 experiment pipeline..."
    )

    # ---------------------------------------------------------
    # RUN TASKS 1-4
    # ---------------------------------------------------------

    for script in EXPERIMENT_SCRIPTS:
        run_script(script)


    # ---------------------------------------------------------
    # GENERATE REPORT FROM UPDATED RESULTS
    # ---------------------------------------------------------

    print(
        "\nAll experiments completed successfully."
    )

    print(
        "Generating updated Project 3 report..."
    )

    run_script(
        REPORT_SCRIPT
    )


    # ---------------------------------------------------------
    # COMPLETE
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("PROJECT 3 PIPELINE COMPLETE")
    print("=" * 60)

    print(
        "\nTasks 1-4 were rerun successfully."
    )

    print(
        "All JSON result files were updated."
    )

    print(
        "The PDF report was regenerated from "
        "the latest experiment results."
    )


if __name__ == "__main__":
    main()