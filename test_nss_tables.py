"""Test NSS/Ch70 table data generation"""
from school_shared import (
    load_data,
    prepare_district_nss_ch70,
    prepare_aggregate_nss_ch70,
    ALPS_COMPONENTS,
)
from nss_ch70_plots import build_nss_category_data

# Load data
df, reg, c70 = load_data()

print("=== Test NSS/Ch70 table data for Amherst ===")
nss_piv, enroll = prepare_district_nss_ch70(df, c70, "Amherst")

if not nss_piv.empty:
    latest_year = int(nss_piv.index.max())
    print(f"Latest year: {latest_year}")

    cat_rows, cat_total, cat_start_map = build_nss_category_data(nss_piv, latest_year)

    print("\n=== Category Rows ===")
    print("Format: (category, start_str, c15s, c10s, c5s, latest_str, color, latest_val)")
    for row in cat_rows:
        print(f"  {row[0]:20s} {row[1]:>10s} {row[2]:>8s} {row[3]:>8s} {row[4]:>8s} {row[5]:>10s}")

    print(f"\n=== Total Row ===")
    print(f"  {cat_total[0]:20s} {cat_total[1]:>10s} {cat_total[2]:>8s} {cat_total[3]:>8s} {cat_total[4]:>8s} {cat_total[5]:>10s}")

    print(f"\n=== Start values map ===")
    for k, v in cat_start_map.items():
        print(f"  {k:20s} ${v:,.2f}")

print("\n=== Test NSS/Ch70 table data for ALPS PK-12 ===")
alps_districts = list(ALPS_COMPONENTS)
nss_agg, enroll_agg = prepare_aggregate_nss_ch70(df, c70, alps_districts)

if not nss_agg.empty:
    latest_year_agg = int(nss_agg.index.max())
    print(f"Latest year: {latest_year_agg}")

    cat_rows_agg, cat_total_agg, cat_start_map_agg = build_nss_category_data(nss_agg, latest_year_agg)

    print("\n=== Category Rows (ALPS) ===")
    for row in cat_rows_agg:
        print(f"  {row[0]:20s} {row[1]:>10s} {row[2]:>8s} {row[3]:>8s} {row[4]:>8s} {row[5]:>10s}")

    print(f"\n=== Total Row (ALPS) ===")
    print(f"  {cat_total_agg[0]:20s} {cat_total_agg[1]:>10s} {cat_total_agg[2]:>8s} {cat_total_agg[3]:>8s} {cat_total_agg[4]:>8s} {cat_total_agg[5]:>10s}")

print("\n=== Verification: Total should match sum of stacks ===")
if not nss_piv.empty:
    # For Amherst 2024
    total_from_data = nss_piv.loc[2024].sum()
    print(f"Amherst 2024 total from data: ${total_from_data:,.2f}")
    print(f"Amherst 2024 total from table: {cat_total[5]}")
    print(f"Match: {abs(total_from_data - float(cat_total[5].replace('$', '').replace(',', ''))) < 1.0}")
