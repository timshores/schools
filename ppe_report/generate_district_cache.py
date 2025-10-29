"""
Generate district-level data cache for CR CacheMonster01.

This script generates CSV cache files for ALL Massachusetts districts to:
1. Enable QA by desk-checking CSVs against printed PDF reports
2. Improve performance by computing each district's data only once
3. Make reports extensible to any MA district

Usage:
    python generate_district_cache.py                    # Cache only missing districts
    python generate_district_cache.py --force-recompute  # Regenerate all caches
    python generate_district_cache.py --test             # Test with one district only

The script creates cache/district_data/ with CSV files for each district:
- {district}_epp_pivot.csv    - Expenditure per pupil by category and year
- {district}_lines.csv         - FTE time series (enrollment metrics)
- {district}_nss_pivot.csv     - NSS/Ch70 funding by category and year

After generation, all report scripts will load from cache instead of recomputing.
"""

import argparse
import sys
from pathlib import Path

from school_shared import load_data
from district_cache import ensure_all_districts_cached, ensure_cache_dirs, get_all_districts


def test_single_district(df, reg, c70):
    """
    Test cache generation with a single district.

    Uses Amherst-Pelham as test case since it's in Western MA and well-known.
    """
    from district_cache import save_district_data, load_district_data
    from school_shared import prepare_district_epp_lines, prepare_district_nss_ch70

    test_district = "Amherst-Pelham"
    print(f"\n{'='*70}")
    print(f"TEST MODE: Caching single district ({test_district})")
    print(f"{'='*70}\n")

    # Generate data
    print(f"Generating data for {test_district}...")
    epp_pivot, lines = prepare_district_epp_lines(df, test_district)
    nss_pivot, _, foundation_series = prepare_district_nss_ch70(df, c70, test_district)

    print(f"\nGenerated data:")
    print(f"  epp_pivot shape: {epp_pivot.shape}")
    print(f"  lines keys: {list(lines.keys())}")
    print(f"  nss_pivot shape: {nss_pivot.shape if nss_pivot is not None else 'None'}")
    print(f"  foundation_series length: {len(foundation_series) if foundation_series is not None else 0}")

    # Save to cache
    print(f"\nSaving to cache...")
    save_district_data(test_district, epp_pivot, lines, nss_pivot, foundation_series)

    # Load from cache
    print(f"\nLoading from cache...")
    loaded_epp, loaded_lines, loaded_nss = load_district_data(test_district)

    # Verify
    print(f"\nVerifying cache integrity:")
    epp_match = loaded_epp.equals(epp_pivot)
    lines_match = all(loaded_lines[k].equals(lines[k]) for k in lines.keys())

    # For nss_pivot, check if shapes match and values are close (floating point tolerance)
    import numpy as np
    if nss_pivot is not None and loaded_nss is not None:
        nss_match = (
            loaded_nss.shape == nss_pivot.shape and
            np.allclose(loaded_nss.values, nss_pivot.values, rtol=1e-10, equal_nan=True) and
            list(loaded_nss.columns) == list(nss_pivot.columns)
        )
        if not nss_match:
            print(f"\n  NSS mismatch details:")
            print(f"    Original shape: {nss_pivot.shape}, columns: {list(nss_pivot.columns)}")
            print(f"    Loaded shape: {loaded_nss.shape}, columns: {list(loaded_nss.columns)}")
            print(f"    Max diff: {np.max(np.abs(loaded_nss.values - nss_pivot.values))}")
    else:
        nss_match = True

    print(f"  epp_pivot matches: {epp_match}")
    print(f"  lines match: {lines_match}")
    print(f"  nss_pivot matches: {nss_match}")

    if epp_match and lines_match and nss_match:
        print(f"\n[SUCCESS] Test passed! Cache working correctly.")
        print(f"\nCached files:")
        from district_cache import get_district_cache_paths
        paths = get_district_cache_paths(test_district)
        for name, path in paths.items():
            if path.exists():
                size_kb = path.stat().st_size / 1024
                print(f"  {path.name} ({size_kb:.1f} KB)")
    else:
        print(f"\n[FAIL] Test failed! Cache data doesn't match.")
        sys.exit(1)


def generate_cohort_cache(df, reg, c70):
    """
    Generate cohort cache files from consolidated district data.

    Creates two files:
    1. cohort_members.csv - Individual district values for each cohort/year
    2. cohort_aggregates.csv - Mean values and CAGRs for each cohort/year
    """
    from district_cache import (
        CONSOLIDATED_EPP_PIVOT, CONSOLIDATED_LINES, CONSOLIDATED_NSS_PIVOT,
        CONSOLIDATED_CAGR, save_cohort_members, save_cohort_aggregates
    )
    from school_shared import get_western_cohort_districts_for_year
    import pandas as pd
    import numpy as np

    print(f"\n{'='*70}")
    print(f"COHORT CACHE GENERATION")
    print(f"{'='*70}\n")

    # Load consolidated district data
    print("[1/4] Loading consolidated district data...")
    epp_data = pd.read_csv(CONSOLIDATED_EPP_PIVOT)
    lines_data = pd.read_csv(CONSOLIDATED_LINES)
    nss_data = pd.read_csv(CONSOLIDATED_NSS_PIVOT)
    cagr_data = pd.read_csv(CONSOLIDATED_CAGR)

    print(f"  Loaded {len(epp_data)} expenditure rows")
    print(f"  Loaded {len(lines_data)} enrollment rows")
    print(f"  Loaded {len(nss_data)} NSS rows")

    # Years to process (2009-2024)
    years = sorted(epp_data['YEAR'].unique())
    print(f"\n[2/4] Processing {len(years)} years: {min(years)}-{max(years)}")

    # Build cohort members DataFrame
    all_member_records = []
    all_aggregate_records = []

    for year in years:
        print(f"  Processing year {year}...")

        # Get cohort assignments for this year
        cohort_districts_full = get_western_cohort_districts_for_year(df, reg, year)

        # Debug: show cohort keys
        if year == 2024:
            print(f"    DEBUG: Cohort keys for {year}: {list(cohort_districts_full.keys())}")

        # Process each cohort
        for full_cohort_name, district_list in cohort_districts_full.items():
            if not district_list:
                continue

            # Extract simple cohort name (keys are uppercase like 'TINY', 'SMALL', etc.)
            if full_cohort_name in ['TINY', 'SMALL', 'MEDIUM', 'LARGE', 'SPRINGFIELD']:
                cohort_name = full_cohort_name.capitalize()  # Convert to 'Tiny', 'Small', etc.
            else:
                # Skip non-cohort keys
                continue

            # Debug: show matched cohorts
            if year == 2024:
                print(f"    DEBUG: Matched cohort '{cohort_name}' from '{full_cohort_name}' with {len(district_list)} districts")
                if cohort_name == 'Tiny':
                    print(f"      DEBUG: First 3 Tiny districts: {district_list[:3]}")

            # Get data for all districts in this cohort
            for district in district_list:
                # Capitalize district name to match data (cohort list is lowercase)
                district_proper = district.title()

                # Get expenditure data
                epp_row = epp_data[(epp_data['YEAR'] == year) & (epp_data['District'] == district_proper)]
                if epp_row.empty:
                    if year == 2024 and cohort_name == 'Tiny':
                        print(f"      DEBUG: No EPP data for district '{district}' in {year}")
                    continue

                # Get enrollment data
                lines_row = lines_data[(lines_data['YEAR'] == year) & (lines_data['District'] == district_proper)]
                if lines_row.empty:
                    continue

                # Get NSS data
                nss_row = nss_data[(nss_data['YEAR'] == year) & (nss_data['District'] == district_proper)]

                # Build member record
                member_record = {
                    'Year': year,
                    'Cohort': cohort_name,
                    'District': district_proper,  # Use capitalized name
                }

                # Add enrollment metrics
                fte_in = lines_row[lines_row['Metric'] == 'In-District FTE']['Value'].values
                fte_out = lines_row[lines_row['Metric'] == 'Out-of-District FTE']['Value'].values
                fte_foundation = lines_row[lines_row['Metric'] == 'Foundation Enrollment']['Value'].values

                member_record['FTE_In_District'] = fte_in[0] if len(fte_in) > 0 else np.nan
                member_record['FTE_Out_District'] = fte_out[0] if len(fte_out) > 0 else np.nan
                member_record['FTE_Foundation'] = fte_foundation[0] if len(fte_foundation) > 0 else np.nan

                # Add PPE by category
                for _, cat_row in epp_row.iterrows():
                    category = cat_row['Category']
                    ppe_value = cat_row['PPE']
                    member_record[f'{category}_PPE'] = ppe_value

                # Add NSS metrics
                if not nss_row.empty:
                    for _, nss_cat_row in nss_row.iterrows():
                        component = nss_cat_row['Component']  # NSS uses 'Component' not 'Category'
                        value = nss_cat_row['Value']
                        member_record[component] = value

                all_member_records.append(member_record)

            # Calculate aggregate for this cohort/year
            # Capitalize district names to match data
            district_list_proper = [d.title() for d in district_list]

            cohort_epp = epp_data[
                (epp_data['YEAR'] == year) & (epp_data['District'].isin(district_list_proper))
            ]
            cohort_lines = lines_data[
                (lines_data['YEAR'] == year) & (lines_data['District'].isin(district_list_proper))
            ]
            cohort_nss = nss_data[
                (nss_data['YEAR'] == year) & (nss_data['District'].isin(district_list_proper))
            ]

            if cohort_epp.empty:
                continue

            aggregate_record = {
                'Year': year,
                'Cohort': cohort_name,
                'District_Count': len(district_list),
            }

            # Mean FTE metrics
            fte_in_mean = cohort_lines[cohort_lines['Metric'] == 'In-District FTE']['Value'].mean()
            fte_out_mean = cohort_lines[cohort_lines['Metric'] == 'Out-of-District FTE']['Value'].mean()
            fte_foundation_mean = cohort_lines[cohort_lines['Metric'] == 'Foundation Enrollment']['Value'].mean()

            aggregate_record['Mean_FTE_In_District'] = fte_in_mean
            aggregate_record['Mean_FTE_Out_District'] = fte_out_mean
            aggregate_record['Mean_FTE_Foundation'] = fte_foundation_mean

            # Mean PPE by category
            for category in cohort_epp['Category'].unique():
                cat_ppe = cohort_epp[cohort_epp['Category'] == category]['PPE']
                aggregate_record[f'Mean_{category}_PPE'] = cat_ppe.mean()

            # Mean NSS metrics
            if not cohort_nss.empty:
                for component in cohort_nss['Component'].unique():  # NSS uses 'Component'
                    comp_values = cohort_nss[cohort_nss['Component'] == component]['Value']
                    aggregate_record[f'Mean_{component}'] = comp_values.mean()

            all_aggregate_records.append(aggregate_record)

    # Build DataFrames
    print(f"\n[3/4] Building cohort DataFrames...")
    members_df = pd.DataFrame(all_member_records)
    aggregates_df = pd.DataFrame(all_aggregate_records)

    print(f"  Cohort members: {len(members_df)} rows")
    print(f"  Cohort aggregates: {len(aggregates_df)} rows")

    # Calculate CAGRs for aggregates
    print(f"\n[4/4] Calculating cohort CAGRs...")

    # Add CAGR columns to aggregates
    for cohort in aggregates_df['Cohort'].unique():
        cohort_data = aggregates_df[aggregates_df['Cohort'] == cohort].sort_values('Year')

        if len(cohort_data) >= 6:  # Need at least 6 years for 5-year CAGR
            # Get time series for Total PPE and Foundation FTE
            years_series = cohort_data['Year'].values
            ppe_series = cohort_data['Mean_Total PPE_PPE'].values if 'Mean_Total PPE_PPE' in cohort_data.columns else None
            fte_series = cohort_data['Mean_FTE_Foundation'].values

            # Calculate CAGRs for each year
            for idx, year in enumerate(cohort_data['Year'].values):
                row_idx = aggregates_df[(aggregates_df['Cohort'] == cohort) & (aggregates_df['Year'] == year)].index[0]

                # 5-year CAGR
                if idx >= 5:
                    if ppe_series is not None:
                        start_ppe = ppe_series[idx - 5]
                        end_ppe = ppe_series[idx]
                        if start_ppe > 0:
                            aggregates_df.loc[row_idx, 'CAGR_5yr_PPE'] = (end_ppe / start_ppe) ** (1/5) - 1

                    start_fte = fte_series[idx - 5]
                    end_fte = fte_series[idx]
                    if start_fte > 0:
                        aggregates_df.loc[row_idx, 'CAGR_5yr_FTE'] = (end_fte / start_fte) ** (1/5) - 1

                # 10-year CAGR
                if idx >= 10:
                    if ppe_series is not None:
                        start_ppe = ppe_series[idx - 10]
                        end_ppe = ppe_series[idx]
                        if start_ppe > 0:
                            aggregates_df.loc[row_idx, 'CAGR_10yr_PPE'] = (end_ppe / start_ppe) ** (1/10) - 1

                    start_fte = fte_series[idx - 10]
                    end_fte = fte_series[idx]
                    if start_fte > 0:
                        aggregates_df.loc[row_idx, 'CAGR_10yr_FTE'] = (end_fte / start_fte) ** (1/10) - 1

                # 15-year CAGR
                if idx >= 15:
                    if ppe_series is not None:
                        start_ppe = ppe_series[idx - 15]
                        end_ppe = ppe_series[idx]
                        if start_ppe > 0:
                            aggregates_df.loc[row_idx, 'CAGR_15yr_PPE'] = (end_ppe / start_ppe) ** (1/15) - 1

                    start_fte = fte_series[idx - 15]
                    end_fte = fte_series[idx]
                    if start_fte > 0:
                        aggregates_df.loc[row_idx, 'CAGR_15yr_FTE'] = (end_fte / start_fte) ** (1/15) - 1

    # Save to cache
    print(f"\nSaving cohort cache files...")
    save_cohort_members(members_df)
    save_cohort_aggregates(aggregates_df)

    print(f"\n{'='*70}")
    print(f"COHORT CACHE GENERATION COMPLETE")
    print(f"{'='*70}")
    print(f"  Members: {len(members_df)} rows across {len(years)} years")
    print(f"  Aggregates: {len(aggregates_df)} rows")
    print(f"  Cohorts: {', '.join(aggregates_df['Cohort'].unique())}")
    print(f"{'='*70}\n")


def main():
    """Generate district-level data cache for all MA districts."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Generate district-level data cache for CR CacheMonster01"
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Regenerate all caches even if they exist"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: cache only one district (Amherst-Pelham)"
    )
    args = parser.parse_args()

    # Load source data
    print("\n[1/3] Loading source data...")
    df, reg, c70 = load_data(force_recompute=args.force_recompute)
    print(f"  Loaded {len(df)} expenditure records")
    print(f"  Loaded {len(reg)} regional records")
    print(f"  Loaded {len(c70)} Chapter 70 records")

    # Count unique districts
    all_districts = get_all_districts(df)
    print(f"  Found {len(all_districts)} unique districts")

    # Test mode: cache single district
    if args.test:
        test_single_district(df, reg, c70)
        return

    # Generate cache for all districts
    print("\n[2/3] Generating district cache...")
    ensure_all_districts_cached(df, reg, c70, force=args.force_recompute)

    # Generate cohort cache from consolidated district data
    print("\n[3/3] Generating cohort cache...")
    generate_cohort_cache(df, reg, c70)

    print("\n[SUCCESS] All cache generation complete!")
    print(f"\nNext steps:")
    print(f"1. Review cached CSVs:")
    print(f"   - District data: cache/all_districts_*.csv")
    print(f"   - Cohort data: cache/cohort_*.csv")
    print(f"2. Desk-check CSVs against the PDF report for QA")
    print(f"3. Run report generation - it will now use cached data")


if __name__ == "__main__":
    main()
