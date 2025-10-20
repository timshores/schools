"""
Master script to generate the complete school district analysis report.

This script orchestrates the execution of all analysis components:
1. District expenditure plots
2. NSS/Ch70 funding analysis plots
3. Western MA choropleth maps
4. Enrollment distribution plots
5. PDF composition

Usage:
    python generate_report.py
    python generate_report.py --force-recompute  # Bypass cache and recompute from source
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

# Define the pipeline of scripts to execute
PIPELINE = [
    ("threshold_analysis.py", "Threshold analysis for shading thresholds"),
    ("executive_summary_plots.py", "Executive Summary plots"),
    ("district_expend_pp_stack.py", "District expenditure plots"),
    ("nss_ch70_main.py", "NSS/Ch70 funding plots"),
    ("western_map.py", "Western MA choropleth maps"),
    ("western_enrollment_plots_individual.py", "Enrollment distribution plots"),
    ("compose_pdf.py", "PDF composition"),
]


def run_script(script_path: str, description: str, force_recompute: bool = False) -> bool:
    """
    Run a Python script and return success status.

    Args:
        script_path: Path to the script to execute
        description: Human-readable description for logging
        force_recompute: If True, pass --force-recompute flag to script

    Returns:
        True if script succeeded, False otherwise
    """
    print("\n" + "=" * 70)
    print(f"Running: {description}")
    print(f"Script: {script_path}")
    print("=" * 70)

    try:
        cmd = [sys.executable, script_path]
        if force_recompute:
            cmd.append("--force-recompute")

        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,  # Show output in real-time
            text=True
        )
        print(f"[OK] {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[FAIL] {description} failed with exit code {e.returncode}")
        print(f"Error: {e}")
        return False
    except FileNotFoundError:
        print(f"[FAIL] Script not found: {script_path}")
        return False
    except Exception as e:
        print(f"[FAIL] Unexpected error running {script_path}: {e}")
        return False


def main():
    """Execute the complete report generation pipeline."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate school district analysis report")
    parser.add_argument("--force-recompute", action="store_true",
                        help="Bypass cache and recompute all data from source")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("SCHOOL DISTRICT ANALYSIS REPORT GENERATOR")
    print("=" * 70)
    print(f"Working directory: {Path.cwd()}")
    print(f"Python executable: {sys.executable}")
    print(f"Pipeline steps: {len(PIPELINE)}")
    if args.force_recompute:
        print(f"Mode: Force recompute (bypassing cache)")
    else:
        print(f"Mode: Using cache if available")

    # Verify all scripts exist before starting
    missing_scripts = []
    for script_path, _ in PIPELINE:
        if not Path(script_path).exists():
            missing_scripts.append(script_path)

    if missing_scripts:
        print("\n[ERROR] Missing required scripts:")
        for script in missing_scripts:
            print(f"  - {script}")
        print("\nPlease ensure all scripts are in the current directory.")
        sys.exit(1)

    # Execute pipeline
    start_time = __import__('time').time()
    failed_steps: List[str] = []

    for i, (script_path, description) in enumerate(PIPELINE, 1):
        print(f"\n[Step {i}/7]")  # Updated to 7 steps
        success = run_script(script_path, description, force_recompute=args.force_recompute)

        if not success:
            failed_steps.append(description)
            print(f"\n[FAIL] Pipeline failed at step {i}: {description}")
            print("Stopping execution.")
            break

    # Summary
    elapsed = __import__('time').time() - start_time
    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    print(f"Total time: {elapsed:.1f} seconds")

    if failed_steps:
        print(f"Status: FAILED")
        print(f"Failed steps: {', '.join(failed_steps)}")
        sys.exit(1)
    else:
        print("Status: SUCCESS")
        print("\nGenerated files:")
        print("  - PNG plots in output/ directory")
        print("  - Final PDF: output/expenditures_series.pdf")
        print("\n[OK] Report generation complete!")


if __name__ == "__main__":
    main()
