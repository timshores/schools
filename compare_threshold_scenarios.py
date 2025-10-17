"""
Compare different threshold scenarios for equity/proportionality.

Shows how 2%/2pp vs 5%/Xpp would affect flagging rates.
"""

import numpy as np
import pandas as pd
from school_shared import (
    load_data,
    get_indistrict_fte_for_year,
    initialize_cohort_definitions,
    get_western_cohort_districts,
    prepare_district_epp_lines,
)
from nss_ch70_plots import compute_cagr_last


def main():
    print("=" * 80)
    print("THRESHOLD SCENARIO COMPARISON")
    print("=" * 80)
    print()

    # Load data
    df, reg, c70 = load_data()
    initialize_cohort_definitions(df, reg)
    cohorts = get_western_cohort_districts(df, reg)
    latest_year = int(df["YEAR"].max())

    # Collect all districts
    all_districts = []
    for cohort_key in ["TINY", "SMALL", "MEDIUM", "LARGE", "X-LARGE"]:
        all_districts.extend(cohorts.get(cohort_key, []))

    # Collect PPE and CAGR data for all districts
    ppe_values = []
    cagr_values = []

    for dist in all_districts:
        epp_pivot, _ = prepare_district_epp_lines(df, dist)
        if latest_year in epp_pivot.index:
            total_ppe = epp_pivot.loc[latest_year].sum()
            if not np.isnan(total_ppe) and total_ppe > 0:
                ppe_values.append(total_ppe)

        if latest_year in epp_pivot.index and (latest_year - 5) in epp_pivot.index:
            start_val = epp_pivot.loc[latest_year - 5].sum()
            end_val = epp_pivot.loc[latest_year].sum()
            if not np.isnan(start_val) and not np.isnan(end_val) and start_val > 0:
                cagr = compute_cagr_last(epp_pivot.sum(axis=1), 5)
                if not np.isnan(cagr):
                    cagr_values.append(cagr * 100)  # Convert to pp

    ppe_array = np.array(ppe_values)
    cagr_array = np.array(cagr_values)

    ppe_mean = np.mean(ppe_array)
    ppe_sd = np.std(ppe_array, ddof=1)
    ppe_cv = ppe_sd / ppe_mean

    cagr_mean = np.mean(cagr_array)
    cagr_sd = np.std(cagr_array, ddof=1)
    cagr_cv = cagr_sd / cagr_mean

    print(f"Dataset Statistics (n={len(all_districts)} districts):")
    print(f"  PPE:  mean=${ppe_mean:,.0f}, SD=${ppe_sd:,.0f}, CV={ppe_cv:.3f}")
    print(f"  CAGR: mean={cagr_mean:.2f}pp, SD={cagr_sd:.2f}pp, CV={cagr_cv:.3f}")
    print()

    # Define scenarios
    scenarios = [
        ("Current (Conservative)", 0.02, 2.0),
        ("Simple Multiplier (2.5x)", 0.05, 5.0),
        ("Equal SD Sensitivity", 0.05, 0.05 * (cagr_sd / ppe_sd) * ppe_mean),
        ("Half Typical Variation (CV-based)", 0.5 * ppe_cv, 0.5 * cagr_sd),
        ("Relaxed (Practical)", 0.05, 3.5),
    ]

    print("=" * 80)
    print("SCENARIO COMPARISON")
    print("=" * 80)
    print()

    for scenario_name, ppe_thresh_pct, cagr_thresh_pp in scenarios:
        print(f"\n{scenario_name}:")
        print(f"  Thresholds: {ppe_thresh_pct*100:.1f}% for PPE/enrollment, {cagr_thresh_pp:.2f}pp for CAGR")
        print()

        # Calculate absolute thresholds
        ppe_thresh_abs = ppe_mean * ppe_thresh_pct

        # Calculate how many standard deviations each threshold represents
        ppe_n_sd = ppe_thresh_abs / ppe_sd
        cagr_n_sd = cagr_thresh_pp / cagr_sd

        print(f"  PPE threshold: ${ppe_thresh_abs:,.0f} = {ppe_n_sd:.2f} standard deviations")
        print(f"  CAGR threshold: {cagr_thresh_pp:.2f}pp = {cagr_n_sd:.2f} standard deviations")

        # Calculate what percentage of typical variation each represents
        ppe_pct_of_cv = (ppe_thresh_abs / ppe_sd) / ppe_cv if ppe_cv != 0 else 0
        cagr_pct_of_cv = (cagr_thresh_pp / cagr_sd) / cagr_cv if cagr_cv != 0 else 0

        # Estimate flagging rates (using normal approximation for rough estimate)
        # P(|Z| > n_sd) where Z is standard normal
        from scipy import stats
        ppe_flag_rate = 2 * (1 - stats.norm.cdf(ppe_n_sd))  # two-tailed
        cagr_flag_rate = 2 * (1 - stats.norm.cdf(cagr_n_sd))  # two-tailed

        print(f"  Estimated flagging rates (if normally distributed):")
        print(f"    PPE: ~{ppe_flag_rate*100:.1f}% of comparisons would be flagged")
        print(f"    CAGR: ~{cagr_flag_rate*100:.1f}% of comparisons would be flagged")

        # Balance check
        sensitivity_ratio = cagr_n_sd / ppe_n_sd if ppe_n_sd != 0 else 0
        print(f"  Sensitivity ratio (CAGR/PPE): {sensitivity_ratio:.2f}x")
        if abs(sensitivity_ratio - 1.0) < 0.2:
            print(f"    OK: Well-balanced - CAGR and PPE have similar sensitivity")
        elif sensitivity_ratio > 1.5:
            print(f"    WARNING: CAGR is MORE sensitive than PPE ({sensitivity_ratio:.1f}x)")
        else:
            print(f"    WARNING: PPE is MORE sensitive than CAGR ({1/sensitivity_ratio:.1f}x)")

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print("""
1. SIMPLE MULTIPLIER (5% / 5pp):
   - Easiest to explain: "We increased all thresholds by 2.5x"
   - Maintains same relative looseness across metrics
   - Recommended if you want consistency with minimal complexity

2. PRACTICAL COMPROMISE (5% / 3.5pp):
   - Recognizes that CAGR naturally varies more than PPE levels
   - More balanced sensitivity between the two metrics
   - Still simple to communicate: "5% for amounts, 3.5pp for growth rates"
   - Would flag similar percentages of outliers for both metrics

3. KEEP CURRENT (2% / 2pp):
   - Most conservative option
   - Current system slightly favors flagging CAGR outliers over PPE outliers
   - Good if you want to emphasize growth rate differences

The "equitable" choice depends on your goal:
- If you want EQUAL STRINGENCY: Use 5% / 3.5pp (balances sensitivity)
- If you want PROPORTIONAL LOOSENING: Use 5% / 5pp (2.5x across board)
- If you want SIMPLICITY: Use 5% / 5pp (easiest to explain)
    """)


if __name__ == "__main__":
    main()
