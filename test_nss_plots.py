"""Test NSS/Ch70 plotting"""
from pathlib import Path
from school_shared import (
    load_data,
    prepare_district_nss_ch70,
    prepare_aggregate_nss_ch70,
    DISTRICTS_OF_INTEREST,
    ALPS_COMPONENTS,
    OUTPUT_DIR,
)
from nss_ch70_plots import plot_nss_ch70, compute_nss_ylim

# Load data
df, reg, c70 = load_data()

# Test single district
print("=== Generating NSS/Ch70 plot for Amherst ===")
nss_piv, enroll = prepare_district_nss_ch70(df, c70, "Amherst")
if not nss_piv.empty:
    plot_nss_ch70(
        OUTPUT_DIR / "nss_ch70_Amherst.png",
        nss_piv,
        enroll,
        title="Amherst: Chapter 70 Aid and Net School Spending (Per Pupil)",
        right_ylim=35000,
        left_ylim=1500,
    )

# Test aggregate
print("\n=== Generating NSS/Ch70 plot for ALPS PK-12 ===")
alps_districts = list(ALPS_COMPONENTS)
nss_agg, enroll_agg = prepare_aggregate_nss_ch70(df, c70, alps_districts)
if not nss_agg.empty:
    plot_nss_ch70(
        OUTPUT_DIR / "nss_ch70_ALPS_PK-12.png",
        nss_agg,
        enroll_agg,
        title="ALPS PK-12: Chapter 70 Aid and Net School Spending (Per Pupil)",
        right_ylim=35000,
        left_ylim=3000,
    )

# Test with computed ylim
print("\n=== Testing computed ylim ===")
all_pivots = [nss_piv, nss_agg]
ylim = compute_nss_ylim(all_pivots)
print(f"Computed ylim: ${ylim:,.0f}")

print("\n=== Generating all districts ===")
for dist in DISTRICTS_OF_INTEREST:
    print(f"Generating {dist}...")
    nss_d, enr_d = prepare_district_nss_ch70(df, c70, dist)
    if not nss_d.empty:
        safe_name = dist.replace("-", "_").replace(" ", "_")
        plot_nss_ch70(
            OUTPUT_DIR / f"nss_ch70_{safe_name}.png",
            nss_d,
            enr_d,
            title=f"{dist}: Chapter 70 Aid and Net School Spending (Per Pupil)",
            right_ylim=ylim,
        )

print("\n=== Done! ===")
