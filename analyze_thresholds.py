"""
Analyze distributions and compare threshold methodologies.

Compares:
1. Current fixed thresholds (2% for dollars, 2pp for CAGR)
2. Coefficient of Variation (CV) based thresholds
3. Median Absolute Deviation (MAD) based thresholds (for enrollment)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from school_shared import (
    load_data,
    get_indistrict_fte_for_year,
    initialize_cohort_definitions,
    get_western_cohort_districts,
    COHORT_DEFINITIONS,
    prepare_district_epp_lines,
)
from nss_ch70_plots import compute_cagr_last


def calculate_mad(values: np.ndarray) -> float:
    """Calculate Median Absolute Deviation."""
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    return mad


def analyze_metric_distribution(
    values: List[float],
    metric_name: str,
    cohort_name: str = "All"
) -> Dict:
    """Analyze distribution of a metric and compute various statistics."""
    arr = np.array([v for v in values if not np.isnan(v) and v > 0])

    if len(arr) == 0:
        return {}

    mean_val = np.mean(arr)
    std_val = np.std(arr, ddof=1)
    median_val = np.median(arr)
    mad_val = calculate_mad(arr)

    # Coefficient of Variation
    cv = std_val / mean_val if mean_val != 0 else float('nan')

    # CV-based threshold (half the typical CV)
    cv_threshold_rel = 0.5 * cv
    cv_threshold_abs = mean_val * cv_threshold_rel

    # MAD-based threshold (using 1.4826 for normal equivalence, then half)
    # This gives a robust estimate similar to 0.5 * SD for normal data
    mad_threshold_rel = 0.5 * 1.4826 * (mad_val / median_val) if median_val != 0 else float('nan')
    mad_threshold_abs = median_val * mad_threshold_rel

    return {
        'cohort': cohort_name,
        'metric': metric_name,
        'n': len(arr),
        'mean': mean_val,
        'std': std_val,
        'median': median_val,
        'mad': mad_val,
        'cv': cv,
        'cv_threshold_rel': cv_threshold_rel,
        'cv_threshold_abs': cv_threshold_abs,
        'mad_threshold_rel': mad_threshold_rel,
        'mad_threshold_abs': mad_threshold_abs,
    }


def main():
    print("=" * 80)
    print("THRESHOLD METHODOLOGY ANALYSIS")
    print("=" * 80)
    print()

    # Load data
    df, reg, c70 = load_data()

    # Initialize cohorts (already called by load_data, but keeping for clarity)
    initialize_cohort_definitions(df, reg)
    cohorts = get_western_cohort_districts(df, reg)

    latest_year = int(df["YEAR"].max())
    print(f"Analysis Year: {latest_year}")
    print()

    # Collect data by cohort
    results = []

    for cohort_key in ["TINY", "SMALL", "MEDIUM", "LARGE", "X-LARGE"]:
        districts = cohorts.get(cohort_key, [])
        if not districts:
            continue

        cohort_def = COHORT_DEFINITIONS.get(cohort_key, {})
        cohort_label = cohort_def.get("short_label", cohort_key)

        print(f"\n{cohort_label} Cohort ({len(districts)} districts)")
        print("-" * 60)

        # Collect enrollment data
        enrollments = []
        for dist in districts:
            fte = get_indistrict_fte_for_year(df, dist, latest_year)
            if fte > 0:
                enrollments.append(fte)

        # Analyze enrollment distribution
        enroll_stats = analyze_metric_distribution(enrollments, "Enrollment", cohort_label)
        if enroll_stats:
            results.append(enroll_stats)
            print(f"  Enrollment: n={enroll_stats['n']}, mean={enroll_stats['mean']:.0f}, "
                  f"median={enroll_stats['median']:.0f}, CV={enroll_stats['cv']:.3f}")

        # Collect PPE data (total PPE for latest year)
        ppe_values = []
        for dist in districts:
            epp_pivot, _ = prepare_district_epp_lines(df, dist)
            if latest_year in epp_pivot.index:
                total_ppe = epp_pivot.loc[latest_year].sum()
                if not np.isnan(total_ppe) and total_ppe > 0:
                    ppe_values.append(total_ppe)

        # Analyze PPE distribution
        ppe_stats = analyze_metric_distribution(ppe_values, "Total PPE", cohort_label)
        if ppe_stats:
            results.append(ppe_stats)
            print(f"  Total PPE: n={ppe_stats['n']}, mean=${ppe_stats['mean']:,.0f}, "
                  f"median=${ppe_stats['median']:,.0f}, CV={ppe_stats['cv']:.3f}")

        # Collect 5-year CAGR data
        cagr_5y_values = []
        for dist in districts:
            epp_pivot, _ = prepare_district_epp_lines(df, dist)
            if latest_year in epp_pivot.index and (latest_year - 5) in epp_pivot.index:
                start_val = epp_pivot.loc[latest_year - 5].sum()
                end_val = epp_pivot.loc[latest_year].sum()
                if not np.isnan(start_val) and not np.isnan(end_val) and start_val > 0:
                    cagr = compute_cagr_last(epp_pivot.sum(axis=1), 5)
                    if not np.isnan(cagr):
                        cagr_5y_values.append(cagr * 100)  # Convert to percentage points

        # Analyze CAGR distribution (in pp)
        cagr_stats = analyze_metric_distribution(cagr_5y_values, "5yr CAGR (pp)", cohort_label)
        if cagr_stats:
            results.append(cagr_stats)
            print(f"  5yr CAGR: n={cagr_stats['n']}, mean={cagr_stats['mean']:.2f}pp, "
                  f"median={cagr_stats['median']:.2f}pp, CV={cagr_stats['cv']:.3f}")

    # Global analysis (pooled across all cohorts)
    print("\n" + "=" * 80)
    print("GLOBAL ANALYSIS (All Western MA Traditional Districts)")
    print("=" * 80)

    all_districts = []
    for cohort_key in ["TINY", "SMALL", "MEDIUM", "LARGE", "X-LARGE"]:
        all_districts.extend(cohorts.get(cohort_key, []))

    # Global enrollment
    all_enrollments = []
    for dist in all_districts:
        fte = get_indistrict_fte_for_year(df, dist, latest_year)
        if fte > 0:
            all_enrollments.append(fte)

    global_enroll = analyze_metric_distribution(all_enrollments, "Enrollment", "Global")
    if global_enroll:
        results.append(global_enroll)

    # Global PPE
    all_ppe = []
    for dist in all_districts:
        epp_pivot, _ = prepare_district_epp_lines(df, dist)
        if latest_year in epp_pivot.index:
            total_ppe = epp_pivot.loc[latest_year].sum()
            if not np.isnan(total_ppe) and total_ppe > 0:
                all_ppe.append(total_ppe)

    global_ppe = analyze_metric_distribution(all_ppe, "Total PPE", "Global")
    if global_ppe:
        results.append(global_ppe)

    # Global CAGR
    all_cagr = []
    for dist in all_districts:
        epp_pivot, _ = prepare_district_epp_lines(df, dist)
        if latest_year in epp_pivot.index and (latest_year - 5) in epp_pivot.index:
            start_val = epp_pivot.loc[latest_year - 5].sum()
            end_val = epp_pivot.loc[latest_year].sum()
            if not np.isnan(start_val) and not np.isnan(end_val) and start_val > 0:
                cagr = compute_cagr_last(epp_pivot.sum(axis=1), 5)
                if not np.isnan(cagr):
                    all_cagr.append(cagr * 100)

    global_cagr = analyze_metric_distribution(all_cagr, "5yr CAGR (pp)", "Global")
    if global_cagr:
        results.append(global_cagr)

    # Print summary table
    print("\n" + "=" * 80)
    print("THRESHOLD COMPARISON")
    print("=" * 80)
    print()

    print("ENROLLMENT THRESHOLDS:")
    print("-" * 80)
    print(f"{'Cohort':<15} {'N':<5} {'Median':<12} {'MAD':<10} {'MAD Thresh':<12} {'MAD %':<10}")
    print("-" * 80)
    for r in results:
        if r['metric'] == 'Enrollment':
            print(f"{r['cohort']:<15} {r['n']:<5} {r['median']:>11.0f} {r['mad']:>9.0f} "
                  f"{r['mad_threshold_abs']:>11.0f} {r['mad_threshold_rel']*100:>8.1f}%")
    print("-" * 80)
    print(f"Current fixed threshold: 2.0%")
    print()

    print("\nTOTAL PPE THRESHOLDS:")
    print("-" * 80)
    print(f"{'Cohort':<15} {'N':<5} {'Mean':<12} {'CV':<8} {'CV Thresh':<12} {'CV %':<10}")
    print("-" * 80)
    for r in results:
        if r['metric'] == 'Total PPE':
            print(f"{r['cohort']:<15} {r['n']:<5} ${r['mean']:>10,.0f} {r['cv']:>7.3f} "
                  f"${r['cv_threshold_abs']:>10,.0f} {r['cv_threshold_rel']*100:>8.1f}%")
    print("-" * 80)
    print(f"Current fixed threshold: 2.0%")
    print()

    print("\n5-YEAR CAGR THRESHOLDS:")
    print("-" * 80)
    print(f"{'Cohort':<15} {'N':<5} {'Mean':<10} {'CV':<8} {'CV Thresh':<12} {'CV (abs)':<10}")
    print("-" * 80)
    for r in results:
        if r['metric'] == '5yr CAGR (pp)':
            print(f"{r['cohort']:<15} {r['n']:<5} {r['mean']:>9.2f}pp {r['cv']:>7.3f} "
                  f"{r['cv_threshold_abs']:>11.2f}pp {r['cv_threshold_rel']*100:>8.1f}%")
    print("-" * 80)
    print(f"Current fixed threshold: 2.0pp")
    print()

    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print()

    if global_enroll:
        print("1. ENROLLMENT:")
        print(f"   - Current: 2.0% fixed threshold")
        print(f"   - MAD-based global: {global_enroll['mad_threshold_rel']*100:.1f}% "
              f"(median +/- {global_enroll['mad_threshold_abs']:.0f} students)")
        print(f"   - Distribution: Right-skewed (median={global_enroll['median']:.0f}, "
              f"mean={global_enroll['mean']:.0f})")
        if global_enroll['mad_threshold_rel'] < 0.02:
            print(f"   WARNING: MAD threshold ({global_enroll['mad_threshold_rel']*100:.1f}%) is TIGHTER than current 2%")
        else:
            print(f"   OK: MAD threshold ({global_enroll['mad_threshold_rel']*100:.1f}%) is more lenient than current 2%")

    if global_ppe:
        print("\n2. TOTAL PPE:")
        print(f"   - Current: 2.0% fixed threshold")
        print(f"   - CV-based global: {global_ppe['cv_threshold_rel']*100:.1f}% "
              f"(mean +/- ${global_ppe['cv_threshold_abs']:,.0f})")
        print(f"   - CV = {global_ppe['cv']:.3f} (typical variation)")
        if global_ppe['cv_threshold_rel'] < 0.02:
            print(f"   WARNING: CV threshold ({global_ppe['cv_threshold_rel']*100:.1f}%) is TIGHTER than current 2%")
        else:
            print(f"   OK: CV threshold ({global_ppe['cv_threshold_rel']*100:.1f}%) is more lenient than current 2%")

    if global_cagr:
        print("\n3. CAGR:")
        print(f"   - Current: 2.0pp fixed threshold")
        print(f"   - CV-based global: {global_cagr['cv_threshold_abs']:.2f}pp "
              f"(mean +/- {global_cagr['cv_threshold_abs']:.2f}pp)")
        print(f"   - CV = {global_cagr['cv']:.3f} (typical variation)")
        if global_cagr['cv_threshold_abs'] < 2.0:
            print(f"   WARNING: CV threshold ({global_cagr['cv_threshold_abs']:.2f}pp) is TIGHTER than current 2pp")
        else:
            print(f"   OK: CV threshold ({global_cagr['cv_threshold_abs']:.2f}pp) is more lenient than current 2pp")

    print("\n" + "=" * 80)
    print("INTERPRETATION NOTES")
    print("=" * 80)
    print("""
The Coefficient of Variation (CV) approach is scale-independent and works well
for comparing metrics across different cohorts. CV = SD / Mean tells us the
"typical relative variation" in a metric.

For right-skewed distributions (like enrollment), Median Absolute Deviation (MAD)
is more robust to outliers than standard deviation. MAD = median(|x - median|)
gives the median distance from the median.

Current fixed thresholds (2% and 2pp) represent policy judgments about what
constitutes "material difference." Data-driven thresholds would adjust based on
observed variation in your specific dataset.

Key question: Do you want to flag districts that differ by:
  (A) A fixed policy threshold (current: 2% / 2pp), OR
  (B) A data-driven threshold based on typical variation (CV/MAD approach)?

Option A is simpler to explain: "We flag differences >= 2%"
Option B is more adaptive: "We flag districts in the top/bottom X% of variation"
    """)


if __name__ == "__main__":
    main()
