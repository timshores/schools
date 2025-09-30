# Calculation Reference Guide

This document maps every calculation in the analysis pipeline from Excel input to PDF output, allowing you to QA the numbers by tracing through the logic step-by-step.

---

## Data Flow Overview

```
Excel File (E2C_Hub_MA_DESE_Data.xlsx)
  ↓
Load & Parse (school_shared.py: load_data)
  ↓
Filter & Transform (prepare_*_epp_lines functions)
  ↓
Aggregate & Calculate (CAGR, weighted averages, etc.)
  ↓
Visualize (district_expend_pp_stack.py)
  ↓
Generate PDF (compose_pdf.py)
```

---

## Input Data Structure

### Excel File Location
- **Path**: `data/E2C_Hub_MA_DESE_Data.xlsx`
- **Sheets Used**:
  1. `"District Expend by Category"` - expenditure and enrollment data
  2. `"District Regions"` - district region/type classifications

### Sheet 1: "District Expend by Category"

**Key Columns** (after normalization):
- `DIST_NAME` (string) - District name
- `YEAR` (int) - Fiscal year (end year of school year, e.g., 2024 for 2023-24)
- `IND_CAT` (string) - Indicator category
- `IND_SUBCAT` (string) - Indicator subcategory
- `IND_VALUE` (float) - Numeric value

**Key IND_CAT Values**:
- `"expenditures per pupil"` - Per-pupil spending by category
- `"student enrollment"` - FTE enrollment counts

**Key IND_SUBCAT Values for Expenditures**:
- Various spending categories (see "Category Mapping" below)
- `"total expenditures"` - EXCLUDED from analysis
- `"total in-district expenditures"` - EXCLUDED from analysis

**Key IND_SUBCAT Values for Enrollment**:
- `"in-district fte pupils"` - Students enrolled in district
- `"out-of-district fte pupils"` - Students placed out of district
- `"total fte pupils"` - Sum of in-district + out-of-district

### Sheet 2: "District Regions"

**Key Columns** (after normalization):
- `DIST_NAME` (string) - District name
- `EOHHS_REGION` (string) - Regional designation (e.g., "Western")
- `SCHOOL_TYPE` (string) - District type (e.g., "Traditional", "Charter", "Vocational")

---

## Category Mapping

### Canonical Categories (Bottom to Top in Stacked Charts)

These are the standardized spending categories used in all visualizations:

1. **Teachers** - Direct teaching salaries
2. **Insurance, Retirement Programs and Other** - Benefits
3. **Pupil Services** - Student support services
4. **Other Teaching Services** - Non-teacher instructional staff
5. **Operations and Maintenance** - Facilities
6. **Instructional Leadership** - Principals, department heads
7. **Administration** - Central office
8. **Other** - Everything else

### Subcategory → Canonical Mapping

**File**: `school_shared.py` lines 52-70

| Input Subcategory | Canonical Category |
|-------------------|-------------------|
| "teachers" | Teachers |
| "insurance, retirement programs and other" | Insurance, Retirement Programs and Other |
| "pupil services" | Pupil Services |
| "guidance, counseling and testing" | Pupil Services |
| "other teaching services" | Other Teaching Services |
| "operations and maintenance" | Operations and Maintenance |
| "instructional leadership" | Instructional Leadership |
| "administration" | Administration |
| "professional development" | Other |
| "instructional materials, equipment and technology" | Other |
| "student transportation" | Other |
| "transportation" | Other |
| "extraordinary maintenance" | Other |
| "tuition" | Other |
| "special education tuition" | Other |
| "tuition to mass schools" | Other |
| "food services" | Other |

**Normalization**: All input strings are lowercased and whitespace-normalized before matching.

---

## Core Calculations

### 1. Per-Pupil Expenditures by Category (Individual District)

**Function**: `prepare_district_epp_lines()` in `school_shared.py:243-256`

**Input**:
- DataFrame filtered to: `DIST_NAME == district` AND `IND_CAT == "expenditures per pupil"`
- Excludes: `IND_SUBCAT` in `{"total expenditures", "total in-district expenditures"}`

**Processing**:
```python
# Step 1: Pivot raw data
pivot_raw = df.pivot_table(
    index="YEAR",
    columns="IND_SUBCAT",
    values="IND_VALUE",
    aggfunc="sum"
).fillna(0.0)

# Step 2: Map subcategories to canonical categories (aggregate_to_canonical)
for each column in pivot_raw:
    canonical = SUBCAT_TO_CANON.get(normalize(column), "Other")
    # Sum all columns that map to same canonical category

# Step 3: Order columns bottom-to-top
pivot = pivot[CANON_CATS_BOTTOM_TO_TOP order]
```

**Output**: DataFrame with YEAR as index, canonical categories as columns, $/pupil as values

**Example QA Check**:
```
For Amherst-Pelham in 2024:
1. Open Excel, filter to DIST_NAME="Amherst-Pelham", IND_CAT="Expenditures Per Pupil", YEAR=2024
2. Sum all "teachers" subcategory values → should match "Teachers" in output
3. Sum all values except totals → should match total in PDF table
```

---

### 2. Enrollment Trends (Individual District)

**Function**: `prepare_district_epp_lines()` in `school_shared.py:250-256` (lines dict)

**Input**:
- DataFrame filtered to: `DIST_NAME == district` AND `IND_CAT == "student enrollment"`

**Processing**:
```python
for (key, label) in ENROLL_KEYS:
    # key examples: "in-district fte pupils", "out-of-district fte pupils"
    series = df[df["IND_SUBCAT"].str.lower() == key][["YEAR", "IND_VALUE"]]
    # Result: time series of enrollment by year
```

**Output**: Dictionary mapping enrollment type to Series (YEAR → count)

**Example QA Check**:
```
For Amherst-Pelham in 2024:
1. Excel: Filter DIST_NAME="Amherst-Pelham", IND_CAT="Student Enrollment",
   IND_SUBCAT="In-District FTE Pupils", YEAR=2024
2. Value should match black line endpoint in district plot
```

---

### 3. Weighted Average Expenditures (Multiple Districts)

**Function**: `weighted_epp_aggregation()` in `school_shared.py:391-440`

**Purpose**: Aggregate multiple districts using enrollment-weighted averages

**Input**: List of district names (e.g., all Western MA districts ≤500 students)

**Processing**:
```python
# Step 1: Get expenditures for all districts
epp = df[(IND_CAT == "expenditures per pupil") &
         (DIST_NAME in district_list)]

# Step 2: Get enrollment weights
weights = df[(IND_CAT == "student enrollment") &
             (IND_SUBCAT == "in-district fte pupils") &
             (DIST_NAME in district_list)]

# Step 3: Merge on DIST_NAME and YEAR
merged = epp.merge(weights, on=["DIST_NAME", "YEAR"])

# Step 4: For each YEAR and IND_SUBCAT:
weighted_value = sum(epp_value * enrollment) / sum(enrollment)

# If sum(enrollment) == 0, use simple mean as fallback

# Step 5: Aggregate to canonical categories (same as individual districts)
```

**Output**:
- DataFrame with YEAR as index, canonical categories as columns, weighted $/pupil as values
- Series with YEAR → total enrollment sum

**Example QA Check**:
```
For Western MA ≤500 in 2024, "Teachers" category:
1. Excel: Get list of all Western MA Traditional districts with ≤500 FTE
2. For each district in 2024:
   - Get "teachers" $/pupil value
   - Get "in-district fte pupils" value
3. Calculate: sum(teachers_value * enrollment) / sum(enrollment)
4. Should match "Teachers" value in Western ≤500 chart for 2024
```

---

### 4. CAGR (Compound Annual Growth Rate)

**Function**: `compute_cagr_last()` in `compose_pdf.py:105-138`

**Formula**:
```
CAGR = (End_Value / Start_Value)^(1 / years) - 1
```

**Processing**:
```python
def compute_cagr_last(series: pd.Series, years: int) -> float:
    # Step 1: Get latest year in series
    end_year = series.index.max()
    start_year = end_year - years

    # Step 2: Check both endpoints exist
    if start_year not in series or end_year not in series:
        return NaN

    # Step 3: Get values
    v0 = series[start_year]
    v1 = series[end_year]

    # Step 4: Validate (must be positive)
    if v0 <= 0 or v1 <= 0:
        return NaN

    # Step 5: Calculate
    return (v1 / v0) ** (1.0 / years) - 1.0
```

**Output**: Float representing annual growth rate (e.g., 0.05 = 5% per year)

**Example QA Check**:
```
For ALPS PK-12 total PPE, 5-year CAGR:
1. Get latest year (e.g., 2024) and value (e.g., $28,750)
2. Get start year (2024 - 5 = 2019) and value (e.g., $22,500)
3. Calculate: (28750 / 22500)^(1/5) - 1 = 0.0507 = 5.07%
4. Should match "CAGR 5y" in PDF table for Total row
```

**CRITICAL**: For total CAGR, use CAGR of the sum, NOT mean of category CAGRs:
```python
# CORRECT (after refactoring):
total_series = epp_pivot.sum(axis=1)  # Sum across categories for each year
total_cagr = compute_cagr_last(total_series, 5)

# INCORRECT (old bug):
category_cagrs = [compute_cagr_last(epp_pivot[cat], 5) for cat in categories]
total_cagr = mean(category_cagrs)  # WRONG!
```

---

### 5. Virtual District: ALPS PK-12

**Function**: `add_alps_pk12()` in `school_shared.py:315-353`

**Purpose**: Create synthetic district combining Amherst, Amherst-Pelham, Leverett, Pelham, Shutesbury

**Input**: Full DataFrame

**Processing**:

**A. Expenditures per Pupil** (weighted by enrollment):
```python
# Step 1: Filter to component districts
components = ["amherst", "amherst-pelham", "leverett", "pelham", "shutesbury"]
df_parts = df[df["DIST_NAME"].str.lower().isin(components)]

# Step 2: Use weighted aggregation (same as multi-district aggregation)
epp_pivot = weighted_epp_aggregation(df_parts, components)
# This gives enrollment-weighted per-pupil expenditures

# Step 3: Create new rows with DIST_NAME = "ALPS PK-12"
```

**B. Enrollment** (simple sum):
```python
# For each enrollment type (in-district, out-of-district):
for key in ["in-district fte pupils", "out-of-district fte pupils"]:
    # Sum across all component districts for each year
    alps_enrollment[year] = sum(component_enrollment[year]
                                 for each component)
```

**Output**: New rows added to DataFrame with DIST_NAME="ALPS PK-12"

**Example QA Check**:
```
For ALPS PK-12 in 2024, "Teachers" category:
1. Get 2024 data for all 5 components:
   - Amherst: teachers=$X, enrollment=E1
   - Amherst-Pelham: teachers=$Y, enrollment=E2
   - ... (3 more)
2. Weighted average: (X*E1 + Y*E2 + ...) / (E1+E2+...)
3. Should match ALPS PK-12 "Teachers" in output

For ALPS PK-12 in 2024, In-District FTE:
1. Sum enrollment across all 5 components
2. Should match black line endpoint in ALPS plot
```

---

### 6. Comparative Bars Chart (5-Year Change)

**Function**: `plot_ppe_change_bars()` in `district_expend_pp_stack.py:140-290`

**Purpose**: Show PPE change from year T0 to latest year

**Processing**:
```python
# Given: latest year (e.g., 2024), lag=5
t0 = latest - lag  # e.g., 2019

# For each district:
# Step 1: Get total PPE for both years
total_series = epp_pivot.sum(axis=1)  # Sum all categories
p0 = total_series[t0]      # Base year
p1 = total_series[latest]  # Latest year

# Step 2: Calculate delta
delta = p1 - p0

# Step 3: Visualize
# - Bar base: p0 (light blue/gray)
# - Bar extension: delta (dark blue if positive, purple if negative)
# - Total height: p1
```

**Enrollment Subplot**:
```python
# For each district, plot enrollment series from t0 to latest
# Line connects all years in between
# Endpoints labeled with actual values
```

**Example QA Check**:
```
For ALPS PK-12 in comparative chart:
1. Get total PPE for 2019: sum all categories in 2019 → base of bar
2. Get total PPE for 2024: sum all categories in 2024 → top of bar
3. Delta: 2024 - 2019 → height of colored segment
4. Enrollment: Get all FTE values 2019-2024 → should match line plot below
```

---

### 7. PDF Table Calculations

**File**: `compose_pdf.py`, function `build_page_dicts()` lines 354-488

#### Category Table

**For each canonical category**:
- **Latest $/pupil**: `epp_pivot.loc[latest_year, category]`
- **CAGR 5y**: `compute_cagr_last(epp_pivot[category], 5)`
- **CAGR 10y**: `compute_cagr_last(epp_pivot[category], 10)`
- **CAGR 15y**: `compute_cagr_last(epp_pivot[category], 15)`

**Total Row**:
```python
# IMPORTANT: Use CAGR of total, not mean of category CAGRs
total_series = epp_pivot.sum(axis=1)  # Sum across categories for each year

latest_total = total_series[latest_year]
cagr_5 = compute_cagr_last(total_series, 5)
cagr_10 = compute_cagr_last(total_series, 10)
cagr_15 = compute_cagr_last(total_series, 15)
```

#### FTE Table

**For each enrollment type** (In-District, Out-of-District):
- **Latest value**: `enrollment_series[latest_year]`
- **CAGR 5y**: `compute_cagr_last(enrollment_series, 5)`
- **CAGR 10y**: `compute_cagr_last(enrollment_series, 10)`
- **CAGR 15y**: `compute_cagr_last(enrollment_series, 15)`

**Total FTE Row**:
```python
# Sum in-district + out-of-district for each year
total_fte = in_district_series.add(out_of_district_series, fill_value=0)

latest_total = total_fte[latest_year]
cagr_5 = compute_cagr_last(total_fte, 5)
# ... (same for 10y, 15y)
```

---

## Table Shading Logic

### Dollar Amount Shading

**Location**: `compose_pdf.py:218-223`

**Logic**:
```python
# Compare district's latest $/pupil to Western MA baseline for same category

baseline_dollar = western_baseline[category]["DOLLAR"]
district_dollar = epp_pivot.loc[latest_year, category]

relative_difference = (district_dollar - baseline_dollar) / baseline_dollar

# Threshold: ±2% (DOLLAR_THRESHOLD_REL = 0.02)
if abs(relative_difference) >= 0.02:
    if relative_difference > 0:
        shade = RED (higher than Western)
    else:
        shade = GREEN (lower than Western)

    # Intensity based on magnitude:
    # [0.02-0.05): lightest
    # [0.05-0.08): light
    # [0.08-0.12): medium
    # [0.12+):     darkest
```

### CAGR Shading

**Location**: `compose_pdf.py:226-232`

**Logic**:
```python
# Compare district's CAGR to Western MA baseline (percentage point difference)

district_cagr = 0.053  # 5.3%
baseline_cagr = 0.028  # 2.8%

delta_pp = district_cagr - baseline_cagr  # 0.025 = 2.5 percentage points

# Threshold: ±2.0 pp (MATERIAL_DELTA_PCTPTS = 0.02)
if abs(delta_pp) >= 0.02:
    if delta_pp > 0:
        shade = RED (growing faster than Western)
    else:
        shade = GREEN (growing slower than Western)

    # Intensity based on magnitude (same bins as dollar shading)
```

**Example QA Check**:
```
For Amherst-Pelham "Teachers" 5y CAGR in 2024:
1. PDF shows: 4.5% (shaded light red)
2. Get baseline: Western >500 "Teachers" 5y CAGR = 2.8%
3. Delta: 4.5% - 2.8% = 1.7 pp
4. Check: 1.7pp < 2.0pp threshold → should NOT be shaded
5. OR: 1.7pp >= 2.0pp threshold → should be light red (in [0.02, 0.05) bin)

(Verify actual values in your data)
```

---

## Common QA Scenarios

### Scenario 1: Verify a District's Total PPE for One Year

**Steps**:
1. Open `E2C_Hub_MA_DESE_Data.xlsx`, sheet "District Expend by Category"
2. Filter:
   - `DIST_NAME` = your district
   - `IND_CAT` = "Expenditures Per Pupil" (case-insensitive)
   - Year column = your target year
   - `IND_SUBCAT` NOT IN ("total expenditures", "total in-district expenditures")
3. Sum all `IND_VALUE` cells
4. Compare to PDF table "Total" row, "Latest $/pupil" column

### Scenario 2: Verify ALPS PK-12 Enrollment for One Year

**Steps**:
1. Open Excel, filter to:
   - `DIST_NAME` IN ("Amherst", "Amherst-Pelham", "Leverett", "Pelham", "Shutesbury")
   - `IND_CAT` = "Student Enrollment"
   - `IND_SUBCAT` = "In-District FTE Pupils"
   - Year = your target year
2. Sum all `IND_VALUE` for these 5 districts
3. Compare to ALPS PK-12 PDF table, "In-District FTE Pupils" row, latest year column

### Scenario 3: Verify 5-Year CAGR for a Category

**Steps**:
1. Identify the series (e.g., Amherst-Pelham "Teachers")
2. Get values for start year and end year from Excel (following Scenario 1 process)
3. Calculate: `(end_value / start_value) ^ (1/5) - 1`
4. Compare to PDF table

### Scenario 4: Verify Western MA Aggregate ≤500 for One Category/Year

**Steps**:
1. Open Excel sheet "District Regions"
2. Filter: `EOHHS_REGION`="Western", `SCHOOL_TYPE`="Traditional"
3. Get list of districts
4. For each district, check latest total FTE (from enrollment data)
5. Filter to districts with FTE ≤ 500
6. In "District Expend by Category" sheet:
   - For target year and category
   - Get $/pupil value AND in-district FTE for each qualified district
7. Calculate weighted average:
   ```
   weighted_avg = sum(value_i * fte_i) / sum(fte_i)
   ```
8. Compare to Western ≤500 chart or appendix table

---

## File Reference

### Key Functions by Location

| Function | File | Lines | Purpose |
|----------|------|-------|---------|
| `load_data()` | school_shared.py | 149-180 | Load Excel, normalize columns |
| `coalesce_year_column()` | school_shared.py | 113-147 | Parse year from various formats |
| `aggregate_to_canonical()` | school_shared.py | 187-205 | Map subcategories to canonical |
| `prepare_district_epp_lines()` | school_shared.py | 243-256 | Get district EPP + enrollment |
| `prepare_western_epp_lines()` | school_shared.py | 258-293 | Get Western aggregate EPP |
| `weighted_epp_aggregation()` | school_shared.py | 391-440 | Enrollment-weighted average |
| `add_alps_pk12()` | school_shared.py | 315-353 | Create ALPS virtual district |
| `compute_cagr_last()` | compose_pdf.py | 105-138 | Calculate CAGR |
| `plot_ppe_change_bars()` | district_expend_pp_stack.py | 140-290 | Comparative bars chart |
| `build_page_dicts()` | compose_pdf.py | 354-488 | Assemble PDF table data |

---

## Constants & Thresholds

| Constant | Value | Purpose | File |
|----------|-------|---------|------|
| `N_THRESHOLD` | 500 | Enrollment cutoff for small/large | school_shared.py:18 |
| `COLOR_MAP_VERSION` | 4 | Color palette version | school_shared.py:15 |
| `PPE_PEERS_YMAX` | 30000 | Y-axis max for bars chart | school_shared.py:95 |
| `MATERIAL_DELTA_PCTPTS` | 0.02 | CAGR shading threshold (2.0pp) | compose_pdf.py:29 |
| `DOLLAR_THRESHOLD_REL` | 0.02 | Dollar shading threshold (2%) | compose_pdf.py:31 |
| `SHADE_BINS` | [0.02, 0.05, 0.08, 0.12] | Shading intensity levels | compose_pdf.py:34 |

---

## Version Tracking

- Plots: Version stamp in bottom-right corner (e.g., "Code: v2025.09.29-REFACTORED")
- PDF: No version in footer (removed to prevent overflow)
- Find version in: `CODE_VERSION` variable in `district_expend_pp_stack.py:26` and `compose_pdf.py:26`

---

## Notes on Data Quality

### Missing Data Handling
- Missing values in Excel → treated as 0.0 after pivot
- Missing years in CAGR calculation → returns NaN (shown as "—" in PDF)
- Districts not in regions table → excluded from Western aggregates

### Edge Cases
- **Zero enrollment**: Weighted average falls back to simple mean
- **Negative expenditures**: Theoretically possible, not explicitly handled
- **Year gaps**: CAGR requires exact start and end years; gaps cause NaN

### Known Exclusions
- Charter schools: Excluded from Western aggregates (via SCHOOL_TYPE filter)
- Vocational schools: Excluded from Western aggregates
- Total rows: "total expenditures" and "total in-district expenditures" excluded to prevent double-counting