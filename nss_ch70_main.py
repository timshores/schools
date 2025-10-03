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
    ALPS_COMPONENTS,
    OUTPUT_DIR,
    latest_total_fte,
    N_THRESHOLD,
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

    # Western MA aggregates
    print("  Processing Western MA aggregates...")

    # Get Western MA traditional districts from regions
    western_mask = (reg["EOHHS_REGION"].str.lower() == "western") & (reg["SCHOOL_TYPE"].str.lower() == "traditional")
    western_districts = sorted(set(reg[western_mask]["DIST_NAME"].str.lower()))
    present = set(df["DIST_NAME"].str.lower())
    western_districts = [d for d in western_districts if d in present]

    # Split by enrollment size
    western_le500 = []
    western_gt500 = []
    for dist in western_districts:
        fte = latest_total_fte(df, dist)
        if fte <= N_THRESHOLD:
            western_le500.append(dist)
        else:
            western_gt500.append(dist)

    # Western ≤500 (weighted per-pupil for aggregates)
    if western_le500:
        nss_w_le, enroll_w_le = prepare_aggregate_nss_ch70_weighted(df, c70, western_le500)
        if not nss_w_le.empty:
            district_data["Western MA (≤500)"] = (nss_w_le, enroll_w_le)
            all_pivots.append(nss_w_le)

    # Western >500 (weighted per-pupil for aggregates)
    if western_gt500:
        nss_w_gt, enroll_w_gt = prepare_aggregate_nss_ch70_weighted(df, c70, western_gt500)
        if not nss_w_gt.empty:
            district_data["Western MA (>500)"] = (nss_w_gt, enroll_w_gt)
            all_pivots.append(nss_w_gt)

    # ALPS Peers aggregate (weighted per-pupil)
    print("  Processing ALPS Peers...")
    alps_peers = ["ALPS PK-12", "Greenfield", "Easthampton", "South Hadley", "Northampton",
                  "East Longmeadow", "Longmeadow", "Agawam", "Hadley", "Hampden-Wilbraham"]
    nss_peers, enroll_peers = prepare_aggregate_nss_ch70_weighted(df, c70, alps_peers)
    if not nss_peers.empty:
        district_data["ALPS Peers"] = (nss_peers, enroll_peers)
        all_pivots.append(nss_peers)

    # ALPS PK-12 aggregate (weighted per-pupil)
    print("  Processing ALPS PK-12...")
    alps_districts = list(ALPS_COMPONENTS)
    nss_alps, enroll_alps = prepare_aggregate_nss_ch70_weighted(df, c70, alps_districts)
    if not nss_alps.empty:
        district_data["ALPS PK-12"] = (nss_alps, enroll_alps)
        all_pivots.append(nss_alps)

    # Individual districts
    for dist in DISTRICTS_OF_INTEREST:
        print(f"  Processing {dist}...")
        nss_dist, enroll_dist = prepare_district_nss_ch70(df, c70, dist)
        if not nss_dist.empty:
            district_data[dist] = (nss_dist, enroll_dist)
            all_pivots.append(nss_dist)

    print(f"  Successfully prepared {len(district_data)} districts/aggregates")

    # Note: For absolute dollar plots, we use auto-scaling (no global y-limit)
    # because district sizes vary dramatically (e.g., Shutesbury ~$2.6M vs aggregates ~$300M)
    print("\n[3/5] Plot scaling...")
    print(f"  Using auto-scaling for absolute dollar values (no global y-limit)")

    # Generate plots
    print("\n[4/5] Generating plots...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for dist_name, (nss_piv, enroll) in district_data.items():
        safe_name = (dist_name.replace("-", "_").replace(" ", "_")
                     .replace("≤", "le").replace(">", "gt")
                     .replace("(", "").replace(")", ""))
        out_path = OUTPUT_DIR / f"nss_ch70_{safe_name}.png"

        title = f"{dist_name}: Chapter 70 Aid and Net School Spending"

        # Aggregates use per-pupil, individual districts use absolute dollars
        is_aggregate = any(x in dist_name for x in ["Western MA", "ALPS"])

        plot_nss_ch70(
            out_path=out_path,
            nss_pivot=nss_piv,
            enrollment=enroll,
            title=title,
            right_ylim=None,  # Auto-scale each plot individually
            per_pupil=is_aggregate,
        )

    print("\n" + "=" * 60)
    print("[SUCCESS] All NSS/Ch70 plots generated successfully!")
    print("=" * 60)

    # Print table data preview for one district
    print("\n[5/5] Preview table data...")
    print("\n[TABLE DATA] Amherst:")
    if "Amherst" in district_data:
        nss_piv, _ = district_data["Amherst"]
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
