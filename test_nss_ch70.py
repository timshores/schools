"""Test NSS/Ch70 data processing"""
from school_shared import (
    load_data,
    prepare_district_nss_ch70,
    prepare_aggregate_nss_ch70,
    DISTRICTS_OF_INTEREST,
    ALPS_COMPONENTS
)

# Load data
df, reg, c70 = load_data()

print("=== Test single district: Amherst ===")
nss_piv, enroll = prepare_district_nss_ch70(df, c70, "Amherst")
print(f"NSS pivot shape: {nss_piv.shape}")
print(f"Years: {nss_piv.index.min()} to {nss_piv.index.max()}")
print("\nLast 5 years:")
print(nss_piv.tail())
print(f"\nEnrollment (last 5 years):")
print(enroll.tail())

# Check for edge cases
print("\n=== Check for Ch70 > Req NSS edge case ===")
for idx in nss_piv.index:
    if c70[(c70['DIST_NAME'].str.lower() == 'amherst') & (c70['YEAR'] == idx)].empty:
        continue
    row = c70[(c70['DIST_NAME'].str.lower() == 'amherst') & (c70['YEAR'] == idx)].iloc[0]
    if row['c70aid'] > row['rqdnss2']:
        print(f"Year {idx}: Ch70={row['c70aid']:,.0f} > ReqNSS={row['rqdnss2']:,.0f}")

print("\n=== Test aggregate: ALPS PK-12 ===")
alps_districts = list(ALPS_COMPONENTS)
nss_agg, enroll_agg = prepare_aggregate_nss_ch70(df, c70, alps_districts)
print(f"NSS aggregate pivot shape: {nss_agg.shape}")
print(f"Years: {nss_agg.index.min()} to {nss_agg.index.max()}")
print("\nLast 5 years:")
print(nss_agg.tail())

print("\n=== Verify stacking makes sense ===")
print("For Amherst 2024:")
if 2024 in nss_piv.index:
    row = nss_piv.loc[2024]
    total = row.sum()
    print(f"  Ch70 Aid:        ${row['Ch70 Aid']:>10,.2f} per pupil")
    print(f"  Req NSS (adj):   ${row['Req NSS (adj)']:>10,.2f} per pupil")
    print(f"  Actual NSS (adj):${row['Actual NSS (adj)']:>10,.2f} per pupil")
    print(f"  Total stack:     ${total:>10,.2f} per pupil")

    # Verify against raw data
    c70_2024 = c70[(c70['DIST_NAME'].str.lower() == 'amherst') & (c70['YEAR'] == 2024)]
    if not c70_2024.empty:
        enr_2024 = enroll.loc[2024]
        raw = c70_2024.iloc[0]
        print(f"\n  Raw data check:")
        print(f"    actualNSS / enrollment = {raw['actualNSS'] / enr_2024:,.2f}")
        print(f"    Should equal total stack = {total:,.2f}")
        print(f"    Match: {abs(raw['actualNSS'] / enr_2024 - total) < 0.01}")
