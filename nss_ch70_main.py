"""
Main script to generate all NSS/Ch70 plots and prepare data for PDF integration.

This script:
1. Loads C70 data (Chapter 70 Aid and Net School Spending)
2. Generates stacked bar plots for each district showing:
   - Chapter 70 Aid (bottom stack, green)
   - Required NSS minus Ch70 (middle stack, amber)
   - Actual NSS minus Required NSS (top stack, purple)
3. Prepares table data for comparison (similar to expenditure category tables)
4. Outputs PNG files to the output/ directory
"""

from pathlib import Path
from school_shared import (
    load_data,
    prepare_district_nss_ch70,
    prepare_aggregate_nss_ch70_weighted,
    DISTRICTS_OF_INTEREST,
    OUTPUT_DIR,
    context_for_district,
    get_cohort_ylim,
    latest_total_fte,
    get_enrollment_group,
    compute_districts_fte_ylim,
    get_cohort_label,
    get_cohort_ylim,
    get_western_cohort_districts,
    make_safe_filename,
)
from nss_ch70_plots import (
    plot_nss_ch70,
    compute_nss_ylim,
    build_nss_category_data,
)


def main():
    """Generate all NSS/Ch70 plots and data."""

    print("=" * 60)
    print("Chapter 70 and Net School Spending Analysis")
    print("=" * 60)

    # Load data
    print("\n[1/4] Loading data...")
    df, reg, c70 = load_data()

    if c70.empty:
        print("[ERROR] No C70 data found. Cannot generate NSS/Ch70 plots.")
        return

    print(f"  Loaded C70 data: {len(c70)} rows, {len(c70['DIST_NAME'].unique())} districts")
    print(f"  Years available: {c70['YEAR'].min()} to {c70['YEAR'].max()}")

    # Prepare all district data
    print("\n[2/5] Preparing district data...")
    all_pivots = []
    district_data = {}

    # Western MA aggregates - 6 enrollment groups
    print("  Processing Western MA aggregates (6 enrollment groups)...")

    # Get Western MA traditional districts organized by cohort
    # Use centralized function to ensure consistency with PPE analysis
    cohorts = get_western_cohort_districts(df, reg)
    western_tiny = cohorts["TINY"]
    western_small = cohorts["SMALL"]
    western_medium = cohorts["MEDIUM"]
    western_large = cohorts["LARGE"]
    western_xlarge = cohorts["X-LARGE"]
    western_springfield = cohorts["SPRINGFIELD"]

    # Generate NSS/Ch70 for each enrollment cohort using centralized definitions
    enrollment_groups = [
        ("tiny", western_tiny, f"Western MA {get_cohort_label('TINY')}"),
        ("small", western_small, f"Western MA {get_cohort_label('SMALL')}"),
        ("medium", western_medium, f"Western MA {get_cohort_label('MEDIUM')}"),
        ("large", western_large, f"Western MA {get_cohort_label('LARGE')}"),
        ("x-large", western_xlarge, f"Western MA {get_cohort_label('X-LARGE')}"),
        ("springfield", western_springfield, f"Western MA {get_cohort_label('SPRINGFIELD')}"),
    ]

    for bucket_id, districts, label in enrollment_groups:
        if districts:
            nss_group, enroll_group, foundation_group = prepare_aggregate_nss_ch70_weighted(df, c70, districts)
            if not nss_group.empty:
                district_data[label] = (nss_group, enroll_group, foundation_group)
                all_pivots.append(nss_group)
                print(f"    {label}: {len(districts)} districts")

    # Individual districts
    for dist in DISTRICTS_OF_INTEREST:
        print(f"  Processing {dist}...")
        nss_dist, enroll_dist, foundation_dist = prepare_district_nss_ch70(df, c70, dist)
        if not nss_dist.empty:
            district_data[dist] = (nss_dist, enroll_dist, foundation_dist)
            all_pivots.append(nss_dist)

    print(f"  Successfully prepared {len(district_data)} districts/aggregates")

    # Note: For absolute dollar plots, we use auto-scaling (no global y-limit)
    # because district sizes vary dramatically (e.g., Shutesbury ~$2.6M vs aggregates ~$300M)
    print("\n[3/5] Plot scaling...")
    print(f"  Using auto-scaling for absolute dollar values (no global y-limit)")
    print(f"  Computing foundation enrollment y-axis limits to match PPE plots")

    # Compute global left_ylim_districts from all individual district foundation enrollment
    # (excluding aggregates, matching logic from district_expend_pp_stack.py)
    individual_district_foundations = []
    for dist_name, (_, _, foundation) in district_data.items():
        if "Western MA" not in dist_name and foundation is not None and not foundation.empty:
            individual_district_foundations.append({"Foundation Enrollment": foundation})

    if individual_district_foundations:
        left_ylim_districts = compute_districts_fte_ylim(individual_district_foundations, pad=1.06, step=50)
        print(f"  Global district foundation enrollment y-limit: {left_ylim_districts:,.0f}")
    else:
        left_ylim_districts = 2000  # Fallback

    # Generate plots
    print("\n[4/5] Generating plots...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for dist_name, (nss_piv, enroll, foundation) in district_data.items():
        # Generate safe filename using centralized sanitization function
        # This prevents creation of Windows reserved device names like "nul"
        if "Western MA" in dist_name:
            # Extract enrollment group name for Western MA aggregates
            if "Tiny" in dist_name:
                safe_name = "Western_MA_tiny"
            elif "Small" in dist_name:
                safe_name = "Western_MA_small"
            elif "Medium" in dist_name:
                safe_name = "Western_MA_medium"
            elif "X-Large" in dist_name:  # Check X-Large before Large to avoid false match
                safe_name = "Western_MA_x-large"
            elif "Large" in dist_name:
                safe_name = "Western_MA_large"
            elif "Springfield" in dist_name:
                safe_name = "Western_MA_springfield"
            else:
                safe_name = make_safe_filename(dist_name)
        else:
            safe_name = make_safe_filename(dist_name)

        out_path = OUTPUT_DIR / f"nss_ch70_{safe_name}.png"

        # Determine title, enrollment label, and y-axis limits based on district type
        is_aggregate = "Western MA" in dist_name

        if is_aggregate:
            # Aggregates: Update titles and labels based on enrollment cohort
            # Use centralized cohort definitions for boundaries and labels
            if "Tiny" in dist_name:
                cohort_label = get_cohort_label("TINY")
                title = f"All Western MA Traditional Districts: {cohort_label}"
                enrollment_label = "Weighted avg enrollment per district"
                left_ylim = get_cohort_ylim("TINY")
            elif "Small" in dist_name:
                cohort_label = get_cohort_label("SMALL")
                title = f"All Western MA Traditional Districts: {cohort_label}"
                enrollment_label = "Weighted avg enrollment per district"
                left_ylim = get_cohort_ylim("SMALL")
            elif "Medium" in dist_name:
                cohort_label = get_cohort_label("MEDIUM")
                title = f"All Western MA Traditional Districts: {cohort_label}"
                enrollment_label = "Weighted avg enrollment per district"
                left_ylim = get_cohort_ylim("MEDIUM")
            elif "X-Large" in dist_name:  # Check X-Large before Large to avoid false match
                cohort_label = get_cohort_label("X-LARGE")
                title = f"All Western MA Traditional Districts: {cohort_label}"
                enrollment_label = "Weighted avg enrollment per district"
                left_ylim = get_cohort_ylim("X-LARGE")
            elif "Large" in dist_name:
                cohort_label = get_cohort_label("LARGE")
                title = f"All Western MA Traditional Districts: {cohort_label}"
                enrollment_label = "Weighted avg enrollment per district"
                left_ylim = get_cohort_ylim("LARGE")
            elif "Springfield" in dist_name:
                cohort_label = get_cohort_label("SPRINGFIELD")
                title = f"All Western MA Traditional Districts: {cohort_label}"
                enrollment_label = "Weighted avg enrollment per district"
                left_ylim = get_cohort_ylim("SPRINGFIELD")
            else:
                title = f"{dist_name}: Chapter 70 Aid and Net School Spending"
                enrollment_label = "Foundation Enrollment (FTE)"
                left_ylim = compute_districts_fte_ylim([{"Foundation Enrollment": foundation}], pad=1.06, step=50) if foundation is not None and not foundation.empty else None
        else:
            # Individual districts
            title = f"{dist_name}: Chapter 70 Aid and Net School Spending"
            enrollment_label = "Enrollment"

            # Use cohort-based y-axis limit automatically
            # This ensures each district's enrollment axis matches its cohort range
            context = context_for_district(df, dist_name)
            left_ylim = get_cohort_ylim(context)
            if left_ylim is None:  # Springfield has no fixed ylim
                left_ylim = left_ylim_districts

        plot_nss_ch70(
            out_path=out_path,
            nss_pivot=nss_piv,
            enrollment=enroll,
            title=title,
            right_ylim=None,  # Auto-scale each plot individually
            per_pupil=is_aggregate,
            foundation_enrollment=foundation,
            left_ylim=left_ylim,
            enrollment_label=enrollment_label,
        )

    print("\n" + "=" * 60)
    print("[SUCCESS] All NSS/Ch70 plots generated successfully!")
    print("=" * 60)

    # Print table data preview for one district
    print("\n[5/5] Preview table data...")
    print("\n[TABLE DATA] Amherst:")
    if "Amherst" in district_data:
        nss_piv, _, _ = district_data["Amherst"]
        latest_year = int(nss_piv.index.max())
        cat_rows, cat_total, _ = build_nss_category_data(nss_piv, latest_year)

        print("\n  Category Data:")
        print("  " + "Category".ljust(22) + "2009".rjust(10) + "CAGR 15y".rjust(10) +
              "CAGR 10y".rjust(10) + "CAGR 5y".rjust(10) + "2024".rjust(10))
        print("  " + "-" * 72)
        for row in cat_rows:
            print(f"  {row[0]:20s} {row[1]:>10s} {row[2]:>10s} {row[3]:>10s} {row[4]:>10s} {row[5]:>10s}")
        print("  " + "-" * 72)
        print(f"  {cat_total[0]:20s} {cat_total[1]:>10s} {cat_total[2]:>10s} {cat_total[3]:>10s} {cat_total[4]:>10s} {cat_total[5]:>10s}")

    print("\n[NEXT STEPS]")
    print("  1. Integrate these plots into compose_pdf.py")
    print("  2. Add NSS/Ch70 tables with red/green shading (comparing to aggregates)")
    print("  3. Create new PDF section for NSS/Ch70 analysis")
    print("  4. Update work_log.md with implementation details")


if __name__ == "__main__":
    main()
