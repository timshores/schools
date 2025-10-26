"""
Master script to generate the complete school district analysis report.

This script orchestrates the execution of all analysis components:
1. District expenditure plots
2. NSS/Ch70 funding analysis plots
3. Western MA choropleth maps
4. Enrollment distribution plots
5. PDF composition

Usage:
    python generate_report.py                    # Generate main PDF only (default)
    python generate_report.py --force-recompute  # Bypass cache and recompute from source
    python generate_report.py --appendices-only  # Generate appendices PDF only

The report is split into two separate PDFs:
- Main PDF: "Western MA Per Pupil Expenditure Report.pdf"
  Contains: Table of Contents, Executive Summary, Sections 1-3
- Appendices PDF: "WMPPE Appendices.pdf"
  Contains: Appendices A-D (methodology, calculations, data tables, additional visualizations)

By default, only the main PDF is generated. Use --appendices-only to regenerate the appendices
PDF independently (useful for updating methodology documentation without regenerating all plots).
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


def run_script(script_path: str, description: str, force_recompute: bool = False, appendices_only: bool = False) -> bool:
    """
    Run a Python script and return success status.

    Args:
        script_path: Path to the script to execute
        description: Human-readable description for logging
        force_recompute: If True, pass --force-recompute flag to script
        appendices_only: If True, pass --appendices-only flag to compose_pdf.py (CR A07)

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
        if appendices_only and script_path == "compose_pdf.py":
            cmd.append("--appendices-only")

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
    parser.add_argument("--appendices-only", action="store_true",
                        help="Generate only the appendices PDF (CR A07)")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("SCHOOL DISTRICT ANALYSIS REPORT GENERATOR")
    print("=" * 70)
    print(f"Working directory: {Path.cwd()}")
    print(f"Python executable: {sys.executable}")

    # CR A07: If appendices_only, only run compose_pdf.py
    if args.appendices_only:
        print(f"Mode: Appendices only")
        pipeline_to_run = [("compose_pdf.py", "PDF composition (appendices only)")]
    else:
        print(f"Pipeline steps: {len(PIPELINE)}")
        pipeline_to_run = PIPELINE

    if args.force_recompute:
        print(f"Mode: Force recompute (bypassing cache)")
    else:
        print(f"Mode: Using cache if available")

    # Verify all scripts exist before starting
    missing_scripts = []
    for script_path, _ in pipeline_to_run:
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

    for i, (script_path, description) in enumerate(pipeline_to_run, 1):
        print(f"\n[Step {i}/{len(pipeline_to_run)}]")
        success = run_script(script_path, description, force_recompute=args.force_recompute, appendices_only=args.appendices_only)

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
        if args.appendices_only:
            print("  - Appendices PDF: output/WMPPE Appendices.pdf")
        else:
            print("  - PNG plots in output/ directory")
            print("  - Main PDF: output/Western MA Per Pupil Expenditure Report.pdf")
        print("\n[OK] Report generation complete!")


if __name__ == "__main__":
    main()
