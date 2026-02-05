"""
Master script to generate the complete school district analysis report.

This script orchestrates the execution of all analysis components:
1. Threshold analysis for shading thresholds
2. Executive Summary plots
3. District expenditure plots
4. NSS/Ch70 funding analysis plots
5. Western MA choropleth maps
6. Enrollment distribution plots
7. PDF composition

Usage:
    python generate_report.py                              # Full pipeline, main PDF
    python generate_report.py --force-recompute            # Bypass cache and recompute
    python generate_report.py --appendices-only            # Appendices PDF only
    python generate_report.py --pdf-only                   # Skip plots, just compose PDF
    python generate_report.py --pdf-only --sections exec   # Skip plots, render exec summary only
    python generate_report.py --sections section2,section3 # Full pipeline, render subset

The report is split into two separate PDFs:
- Main PDF: "Western MA Per Pupil Expenditure Report.pdf"
  Contains: Table of Contents, Executive Summary, Sections 1-3
- Appendices PDF: "WMPPE Appendices.pdf"
  Contains: Appendices A-D (methodology, calculations, data tables, additional visualizations)

By default, only the main PDF is generated. Use --appendices-only to regenerate the appendices
PDF independently (useful for updating methodology documentation without regenerating all plots).

Use --pdf-only to skip plot generation steps (1-6) and jump straight to PDF composition.
This is useful when you've only changed text or layout and the plot images are already up to date.

Use --sections to render only specific section groups of the PDF. This produces a DRAFT file
with no TOC and a single-pass build (faster). Available groups:
  exec, section1, section2, section3, appendix_a, appendix_b, appendix_c, appendix_d
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


def run_script(script_path: str, description: str, force_recompute: bool = False,
               appendices_only: bool = False, sections: str = None) -> bool:
    """
    Run a Python script and return success status.

    Args:
        script_path: Path to the script to execute
        description: Human-readable description for logging
        force_recompute: If True, pass --force-recompute flag to script
        appendices_only: If True, pass --appendices-only flag to compose_pdf.py
        sections: If set, pass --sections flag to compose_pdf.py

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
        if sections and script_path == "compose_pdf.py":
            cmd.extend(["--sections", sections])

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
                        help="Generate only the appendices PDF")
    parser.add_argument("--pdf-only", action="store_true",
                        help="Skip plot generation (steps 1-6), run only compose_pdf.py")
    parser.add_argument("--sections", type=str, default=None,
                        help="Comma-separated section groups to render (forwarded to compose_pdf.py). "
                             "Available: exec, section1, section2, section3, "
                             "appendix_a, appendix_b, appendix_c, appendix_d")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("SCHOOL DISTRICT ANALYSIS REPORT GENERATOR")
    print("=" * 70)
    print(f"Working directory: {Path.cwd()}")
    print(f"Python executable: {sys.executable}")

    # Determine pipeline to run
    if args.pdf_only or args.appendices_only:
        desc = "PDF composition"
        if args.pdf_only:
            desc += " (pdf-only)"
        if args.appendices_only:
            desc += " (appendices-only)"
        if args.sections:
            desc += f" (sections: {args.sections})"
        print(f"Mode: {desc}")
        pipeline_to_run = [("compose_pdf.py", desc)]
    else:
        print(f"Pipeline steps: {len(PIPELINE)}")
        pipeline_to_run = PIPELINE

    if args.force_recompute:
        print(f"Mode: Force recompute (bypassing cache)")
    elif not args.pdf_only and not args.appendices_only:
        print(f"Mode: Using cache if available")

    if args.sections:
        print(f"Sections filter: {args.sections}")

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
        success = run_script(script_path, description,
                             force_recompute=args.force_recompute,
                             appendices_only=args.appendices_only,
                             sections=args.sections)

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
        if args.sections and "exec" in args.sections:
            print("  - Main PDF: output/Western MA Per Pupil Expenditure Report.pdf")
            print("  - Executive Summary excerpt: output/Western MA Per Pupil Expenditure Report - Executive Summary.pdf")
        elif args.sections:
            print("  - DRAFT PDF (partial render) in output/ directory")
        elif args.appendices_only:
            print("  - Appendices PDF: output/WMPPE Appendices.pdf")
        else:
            if not args.pdf_only:
                print("  - PNG plots in output/ directory")
            print("  - Main PDF: output/Western MA Per Pupil Expenditure Report.pdf")
            print("  - Executive Summary excerpt: output/Western MA Per Pupil Expenditure Report - Executive Summary.pdf")
        print("\n[OK] Report generation complete!")


if __name__ == "__main__":
    main()
