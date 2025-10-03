"""Test script to explore profile_DataC70 data structure"""
from school_shared import load_data, DISTRICTS_OF_INTEREST
import pandas as pd

# Load data
df, reg, c70 = load_data()

print("=== C70 DIST_NAME after standardization ===")
print("Sample DIST_NAME (first 10):")
print(c70['DIST_NAME'].head(10).tolist())

print("\n=== Check our districts of interest in C70 ===")
for d in DISTRICTS_OF_INTEREST:
    # Check in C70 using standardized DIST_NAME
    matches = c70[c70['DIST_NAME'].str.lower() == d.lower()]
    print(f"{d:20s} - Found: {len(matches) > 0}, Count: {len(matches)}")
    if len(matches) > 0:
        print(f"  Years: {sorted(matches['YEAR'].unique())[:5]}...{sorted(matches['YEAR'].unique())[-5:]}")

print("\n=== Check 'ALPS PK-12' virtual district ===")
# ALPS PK-12 is synthetic, won't be in C70
print("ALPS PK-12 needs to be computed from components")

print("\n=== Sample C70 data for Amherst ===")
amherst = c70[c70['DIST_NAME'].str.lower() == 'amherst']
if len(amherst) > 0:
    print(amherst[['DIST_NAME', 'YEAR', 'actualNSS', 'rqdnss2', 'c70aid']].head(10))
else:
    # Try with original District column
    print("Trying with original District column (all caps):")
    amherst_raw = c70[c70['District'].str.strip().str.upper() == 'AMHERST']
    print(f"Found: {len(amherst_raw)} rows")
    if len(amherst_raw) > 0:
        print(amherst_raw[['District', 'fy', 'actualNSS', 'rqdnss2', 'c70aid']].head(10))
