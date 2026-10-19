"""
Threshold Analysis for Shading Logic

Calculates global standard deviations across all districts to inform
appropriate thresholds for red/green shading in comparison tables.

Current thresholds:
- $/pupil and Enrollment: 2.0% relative difference
- CAGR: 2.0 percentage points absolute difference

This analysis determines whether these thresholds are appropriate by
examining the natural variation in the data.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from school_shared import (
    load_data,
    get_western_cohort_districts,
    prepare_district_epp_lines,
    get_indistrict_fte_for_year,
    LATEST_YEAR, START_YEAR, YEAR_10, YEAR_5, WINDOW_YEARS,
    OUTPUT_DIR
)


def calculate_cagr(start_val: float, end_val: float, years: int) -> float:
    """Calculate CAGR as percentage."""
    if start_val > 0 and end_val > 0:
        return ((end_val / start_val) ** (1 / years) - 1) * 100
    return np.nan


def collect_all_district_metrics(df: pd.DataFrame, reg: pd.DataFrame) -> pd.DataFrame:
    """
    Collect metrics for all Western MA districts (excluding Springfield outlier).

    Returns DataFrame with columns:
    - district
    - ppe_start, ppe_latest
    - cagr_15y, cagr_10y, cagr_5y
    - enroll_start, enroll_latest
    - enroll_cagr_15y, enroll_cagr_10y, enroll_cagr_5y
    """
    print("\n[Threshold Analysis] Collecting metrics for all districts...")

    cohorts = get_western_cohort_districts(df, reg)
    all_districts = []
    for cohort_name, district_list in cohorts.items():
        # Exclude Springfield - it's a statistical outlier (>10,000 FTE)
        if cohort_name.upper() != "SPRINGFIELD":
            all_districts.extend(district_list)

    print(f"  Found {len(all_districts)} districts across all cohorts (excluding Springfield)")

    metrics = []

    for dist in all_districts:
        try:
            # Get PPE data
            epp_pivot, _ = prepare_district_epp_lines(df, dist)
            if epp_pivot.empty:
                continue

            total_ppe = epp_pivot.sum(axis=1)  # Sum across all categories

            # PPE values
            ppe_start = total_ppe.get(START_YEAR, np.nan)
            ppe_latest = total_ppe.get(LATEST_YEAR, np.nan)
            ppe_y10 = total_ppe.get(YEAR_10, np.nan)
            ppe_y5 = total_ppe.get(YEAR_5, np.nan)

            # PPE CAGRs
            cagr_15y = calculate_cagr(ppe_start, ppe_latest, WINDOW_YEARS)
            cagr_10y = calculate_cagr(ppe_y10, ppe_latest, 10)
            cagr_5y = calculate_cagr(ppe_y5, ppe_latest, 5)

            # Enrollment values
            enroll_start = get_indistrict_fte_for_year(df, dist, START_YEAR)
            enroll_latest = get_indistrict_fte_for_year(df, dist, LATEST_YEAR)
            enroll_y10 = get_indistrict_fte_for_year(df, dist, YEAR_10)
            enroll_y5 = get_indistrict_fte_for_year(df, dist, YEAR_5)

            # Enrollment CAGRs
            enroll_cagr_15y = calculate_cagr(enroll_start, enroll_latest, WINDOW_YEARS)
            enroll_cagr_10y = calculate_cagr(enroll_y10, enroll_latest, 10)
            enroll_cagr_5y = calculate_cagr(enroll_y5, enroll_latest, 5)

            metrics.append({
                'district': dist,
                'ppe_start': ppe_start,
                'ppe_latest': ppe_latest,
                'cagr_15y': cagr_15y,
                'cagr_10y': cagr_10y,
                'cagr_5y': cagr_5y,
                'enroll_start': enroll_start,
                'enroll_latest': enroll_latest,
                'enroll_cagr_15y': enroll_cagr_15y,
                'enroll_cagr_10y': enroll_cagr_10y,
                'enroll_cagr_5y': enroll_cagr_5y
            })

        except Exception as e:
            print(f"  Warning: Error processing {dist}: {e}")
            continue

    print(f"  Successfully collected metrics for {len(metrics)} districts")

    return pd.DataFrame(metrics)


def calculate_global_sds(metrics_df: pd.DataFrame) -> Dict[str, float]:
    """Calculate global standard deviations for all metrics."""
    print("\n[Threshold Analysis] Calculating global standard deviations...")

    sds = {}

    # PPE standard deviations
    sds['ppe_start_sd'] = metrics_df['ppe_start'].std()
    sds['ppe_latest_sd'] = metrics_df['ppe_latest'].std()

    # PPE CAGR standard deviations
    sds['cagr_15y_sd'] = metrics_df['cagr_15y'].std()
    sds['cagr_10y_sd'] = metrics_df['cagr_10y'].std()
    sds['cagr_5y_sd'] = metrics_df['cagr_5y'].std()

    # Enrollment standard deviations
    sds['enroll_start_sd'] = metrics_df['enroll_start'].std()
    sds['enroll_latest_sd'] = metrics_df['enroll_latest'].std()

    # Enrollment CAGR standard deviations
    sds['enroll_cagr_15y_sd'] = metrics_df['enroll_cagr_15y'].std()
    sds['enroll_cagr_10y_sd'] = metrics_df['enroll_cagr_10y'].std()
    sds['enroll_cagr_5y_sd'] = metrics_df['enroll_cagr_5y'].std()

    # Also calculate means for context
    sds['ppe_start_mean'] = metrics_df['ppe_start'].mean()
    sds['ppe_latest_mean'] = metrics_df['ppe_latest'].mean()
    sds['cagr_15y_mean'] = metrics_df['cagr_15y'].mean()
    sds['cagr_10y_mean'] = metrics_df['cagr_10y'].mean()
    sds['cagr_5y_mean'] = metrics_df['cagr_5y'].mean()
    sds['enroll_start_mean'] = metrics_df['enroll_start'].mean()
    sds['enroll_latest_mean'] = metrics_df['enroll_latest'].mean()
    sds['enroll_cagr_15y_mean'] = metrics_df['enroll_cagr_15y'].mean()
    sds['enroll_cagr_10y_mean'] = metrics_df['enroll_cagr_10y'].mean()
    sds['enroll_cagr_5y_mean'] = metrics_df['enroll_cagr_5y'].mean()

    print("  Standard deviations calculated")
    return sds


def calculate_threshold_options(sds: Dict[str, float]) -> pd.DataFrame:
    """
    Calculate threshold options at different SD multiples.

    Returns DataFrame with threshold values at 0.25 SD, 0.5 SD, 0.75 SD, 1.0 SD.
    """
    print("\n[Threshold Analysis] Calculating threshold options...")

    sd_factors = [0.25, 0.5, 0.75, 1.0]

    # For $/pupil, we want to express as % of mean
    # For CAGR, we express as absolute pp

    threshold_data = []

    # PPE thresholds (as % of mean)
    for factor in sd_factors:
        ppe_start_threshold = (sds['ppe_start_sd'] * factor / sds['ppe_start_mean']) * 100
        ppe_latest_threshold = (sds['ppe_latest_sd'] * factor / sds['ppe_latest_mean']) * 100

        threshold_data.append({
            'SD_Factor': f"{factor} SD",
            'Metric_Type': f'PPE {START_YEAR}',
            'SD_Value': sds['ppe_start_sd'],
            'Mean_Value': sds['ppe_start_mean'],
            'Threshold_Pct': ppe_start_threshold,
            'Threshold_Abs': sds['ppe_start_sd'] * factor
        })

        threshold_data.append({
            'SD_Factor': f"{factor} SD",
            'Metric_Type': f'PPE {LATEST_YEAR}',
            'SD_Value': sds['ppe_latest_sd'],
            'Mean_Value': sds['ppe_latest_mean'],
            'Threshold_Pct': ppe_latest_threshold,
            'Threshold_Abs': sds['ppe_latest_sd'] * factor
        })

    # CAGR thresholds (as absolute pp)
    cagr_metrics = [
        ('cagr_15y', 'PPE CAGR 15y'),
        ('cagr_10y', 'PPE CAGR 10y'),
        ('cagr_5y', 'PPE CAGR 5y')
    ]

    for factor in sd_factors:
        for metric_key, metric_name in cagr_metrics:
            threshold_data.append({
                'SD_Factor': f"{factor} SD",
                'Metric_Type': metric_name,
                'SD_Value': sds[f'{metric_key}_sd'],
                'Mean_Value': sds[f'{metric_key}_mean'],
                'Threshold_Pct': np.nan,  # Not applicable for pp
                'Threshold_PP': sds[f'{metric_key}_sd'] * factor
            })

    # Enrollment thresholds (as % of mean)
    for factor in sd_factors:
        enroll_start_threshold = (sds['enroll_start_sd'] * factor / sds['enroll_start_mean']) * 100
        enroll_latest_threshold = (sds['enroll_latest_sd'] * factor / sds['enroll_latest_mean']) * 100

        threshold_data.append({
            'SD_Factor': f"{factor} SD",
            'Metric_Type': f'Enrollment {START_YEAR}',
            'SD_Value': sds['enroll_start_sd'],
            'Mean_Value': sds['enroll_start_mean'],
            'Threshold_Pct': enroll_start_threshold,
            'Threshold_Abs': sds['enroll_start_sd'] * factor
        })

        threshold_data.append({
            'SD_Factor': f"{factor} SD",
            'Metric_Type': f'Enrollment {LATEST_YEAR}',
            'SD_Value': sds['enroll_latest_sd'],
            'Mean_Value': sds['enroll_latest_mean'],
            'Threshold_Pct': enroll_latest_threshold,
            'Threshold_Abs': sds['enroll_latest_sd'] * factor
        })

    # Enrollment CAGR thresholds
    enroll_cagr_metrics = [
        ('enroll_cagr_15y', 'Enrollment CAGR 15y'),
        ('enroll_cagr_10y', 'Enrollment CAGR 10y'),
        ('enroll_cagr_5y', 'Enrollment CAGR 5y')
    ]

    for factor in sd_factors:
        for metric_key, metric_name in enroll_cagr_metrics:
            threshold_data.append({
                'SD_Factor': f"{factor} SD",
                'Metric_Type': metric_name,
                'SD_Value': sds[f'{metric_key}_sd'],
                'Mean_Value': sds[f'{metric_key}_mean'],
                'Threshold_Pct': np.nan,
                'Threshold_PP': sds[f'{metric_key}_sd'] * factor
            })

    print(f"  Generated {len(threshold_data)} threshold options")
    return pd.DataFrame(threshold_data)


def save_analysis_results(sds: Dict[str, float], thresholds_df: pd.DataFrame, output_path: Path):
    """Save analysis results to CSV for use in PDF generation."""
    print(f"\n[Threshold Analysis] Saving results to {output_path}...")

    # Save SDs as CSV
    sds_df = pd.DataFrame([sds])
    sds_path = output_path.parent / 'threshold_sds.csv'
    sds_df.to_csv(sds_path, index=False)

    # Save thresholds as CSV
    thresholds_df.to_csv(output_path, index=False)

    print(f"  Saved standard deviations to: {sds_path}")
    print(f"  Saved threshold options to: {output_path}")


def main():
    """Run threshold analysis."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run threshold analysis for shading logic")
    parser.add_argument("--force-recompute", action="store_true",
                        help="Bypass cache and recompute from source")
    args = parser.parse_args()

    print("=" * 70)
    print("THRESHOLD ANALYSIS FOR SHADING LOGIC")
    print("=" * 70)

    # Load data
    print("\n[1/4] Loading data...")
    df, reg, c70 = load_data(force_recompute=args.force_recompute)

    # Collect metrics for all districts
    print("\n[2/4] Collecting district metrics...")
    metrics_df = collect_all_district_metrics(df, reg)

    # Calculate global standard deviations
    print("\n[3/4] Calculating global SDs...")
    sds = calculate_global_sds(metrics_df)

    # Calculate threshold options
    print("\n[4/4] Calculating threshold options...")
    thresholds_df = calculate_threshold_options(sds)

    # Save results
    output_path = OUTPUT_DIR / 'threshold_analysis.csv'
    save_analysis_results(sds, thresholds_df, output_path)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY OF GLOBAL STANDARD DEVIATIONS")
    print("=" * 70)
    print("\nPPE Standard Deviations:")
    print(f"  {START_YEAR}: ${sds['ppe_start_sd']:,.2f} (mean: ${sds['ppe_start_mean']:,.2f})")
    print(f"  {LATEST_YEAR}: ${sds['ppe_latest_sd']:,.2f} (mean: ${sds['ppe_latest_mean']:,.2f})")

    print("\nPPE CAGR Standard Deviations:")
    print(f"  15-year: {sds['cagr_15y_sd']:.2f}pp (mean: {sds['cagr_15y_mean']:.2f}%)")
    print(f"  10-year: {sds['cagr_10y_sd']:.2f}pp (mean: {sds['cagr_10y_mean']:.2f}%)")
    print(f"  5-year:  {sds['cagr_5y_sd']:.2f}pp (mean: {sds['cagr_5y_mean']:.2f}%)")

    print("\nEnrollment Standard Deviations:")
    print(f"  {START_YEAR}: {sds['enroll_start_sd']:.1f} FTE (mean: {sds['enroll_start_mean']:.1f})")
    print(f"  {LATEST_YEAR}: {sds['enroll_latest_sd']:.1f} FTE (mean: {sds['enroll_latest_mean']:.1f})")

    print("\nEnrollment CAGR Standard Deviations:")
    print(f"  15-year: {sds['enroll_cagr_15y_sd']:.2f}pp (mean: {sds['enroll_cagr_15y_mean']:.2f}%)")
    print(f"  10-year: {sds['enroll_cagr_10y_sd']:.2f}pp (mean: {sds['enroll_cagr_10y_mean']:.2f}%)")
    print(f"  5-year:  {sds['enroll_cagr_5y_sd']:.2f}pp (mean: {sds['enroll_cagr_5y_mean']:.2f}%)")

    print("\n" + "=" * 70)
    print("[SUCCESS] Threshold analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
