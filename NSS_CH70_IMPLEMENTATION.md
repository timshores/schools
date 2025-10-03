# Chapter 70 and Net School Spending (NSS) Implementation

## Overview
This implementation analyzes Chapter 70 state aid and Net School Spending on a per-pupil basis for Massachusetts school districts. The analysis creates stacked bar charts showing how funding is distributed across three components.

## Stacking Logic (Bottom to Top)

### Visual Representation
```
┌─────────────────────────────────┐
│ Actual NSS (adj)                │  Purple (#a78bfa)
│ Spending above requirement      │
├─────────────────────────────────┤
│ Req NSS (adj)                   │  Amber (#fbbf24)
│ Local contribution required     │
├─────────────────────────────────┤
│ Ch70 Aid                        │  Green (#4ade80)
│ State aid                       │
└─────────────────────────────────┘
```

### Mathematical Formulas

Given raw C70 data for a district in year Y:
- `c70aid` = Chapter 70 aid (dollars)
- `rqdnss2` = Required Net School Spending with adjustments (dollars)
- `actualNSS` = Actual Net School Spending (dollars)
- `enrollment` = In-District FTE enrollment

Per-pupil calculations:
1. **Ch70 Aid** (bottom stack, green)
   ```
   Ch70_pp = c70aid / enrollment
   ```

2. **Req NSS (adj)** (middle stack, amber)
   ```
   ReqNSS_adj_pp = max(0, rqdnss2 - c70aid) / enrollment
   ```
   Note: Uses `max(0, ...)` to handle edge case where Ch70 aid exceeds required NSS

3. **Actual NSS (adj)** (top stack, purple)
   ```
   ActualNSS_adj_pp = (actualNSS - rqdnss2) / enrollment
   ```
   Note: Can be negative if district underfunds (spends less than required)

**Total NSS per pupil:**
```
Total = Ch70_pp + ReqNSS_adj_pp + ActualNSS_adj_pp
      = c70aid/enrollment + (rqdnss2-c70aid)/enrollment + (actualNSS-rqdnss2)/enrollment
      = actualNSS / enrollment
```

## Data Sources

### profile_DataC70 Sheet
- **Source**: MA DESE data on Chapter 70 aid and net school spending
- **Key Columns**:
  - `District`: District name (ALL CAPS with spaces, e.g., "AMHERST PELHAM")
  - `fy`: Fiscal year (maps to school year)
  - `c70aid`: Chapter 70 state aid (dollars)
  - `rqdnss2`: Required Net School Spending with adjustments (dollars)
  - `actualNSS`: Actual Net School Spending (dollars)
- **Years**: 1993-2025 (33 years of data)
- **Districts**: 444 districts

### Data Normalization
District names required normalization:
- C70: "AMHERST PELHAM" (ALL CAPS, space)
- Expenditure data: "Amherst-Pelham" (Title Case, hyphen)
- **Solution**: `.str.title().str.replace(" ", "-")`

## Key Functions

### Data Processing (school_shared.py)

#### `prepare_district_nss_ch70(df, c70, dist)`
Prepares NSS/Ch70 data for a single district on per-pupil basis.

**Returns:**
- `nss_pivot`: DataFrame with columns `['Ch70 Aid', 'Req NSS (adj)', 'Actual NSS (adj)']`
- `enrollment_series`: In-District FTE enrollment by year

**Example:**
```python
nss_piv, enroll = prepare_district_nss_ch70(df, c70, "Amherst")
# nss_piv has years as index, 3 columns for stacking
# enroll has years as index, FTE values
```

#### `prepare_aggregate_nss_ch70(df, c70, districts)`
Prepares aggregate NSS/Ch70 data for multiple districts.

**Aggregation Method:**
- Sum dollar amounts across all districts by year
- Divide by total enrollment for per-pupil values

**Example:**
```python
alps = ["Amherst", "Amherst-Pelham", "Leverett", "Pelham", "Shutesbury"]
nss_agg, enroll_agg = prepare_aggregate_nss_ch70(df, c70, alps)
```

### Plotting (nss_ch70_plots.py)

#### `plot_nss_ch70(out_path, nss_pivot, enrollment, title, ...)`
Creates stacked bar chart with enrollment line overlay.

**Features:**
- Twin axes: enrollment (left), $/pupil (right)
- Stacked bars (bottom to top): Ch70 Aid, Req NSS (adj), Actual NSS (adj)
- Enrollment line with markers
- Legend includes both stacks and enrollment
- Figure size: 11.8" x 7.4" (matches expenditure plots)

#### `compute_nss_ylim(nss_pivots, pad=1.05)`
Computes global y-axis limit for consistent comparison across plots.

**Algorithm:**
1. Find max total NSS (sum of all stacks) across all districts
2. Add 5% padding
3. Round up to nearest $500

### Table Generation (nss_ch70_plots.py)

#### `build_nss_category_data(nss_pivot, latest_year)`
Builds table rows with CAGR calculations for comparison tables.

**Returns:**
- `cat_rows`: List of tuples `(category, start_str, c15s, c10s, c5s, latest_str, color, latest_val)`
- `cat_total`: Tuple `(label, start_str, c15s, c10s, c5s, latest_str)`
- `cat_start_map`: Dict `{category: start_value}` for red/green shading

**Table Format:**
```
Category          2009      CAGR 15y  CAGR 10y  CAGR 5y   2024
----------------------------------------------------------------
Actual NSS (adj)  $6,217      4.29%      5.77%    3.94%  $11,670
Req NSS (adj)     $3,977      7.11%      4.27%    7.11%  $11,151
Ch70 Aid          $4,518      2.21%      2.44%    3.10%   $6,271
----------------------------------------------------------------
Total            $14,712      4.65%      4.39%    4.89%  $29,093
```

## Usage Example

```python
from school_shared import load_data, prepare_district_nss_ch70
from nss_ch70_plots import plot_nss_ch70, build_nss_category_data

# Load data
df, reg, c70 = load_data()

# Process district
nss_piv, enroll = prepare_district_nss_ch70(df, c70, "Amherst")

# Generate plot
plot_nss_ch70(
    out_path=OUTPUT_DIR / "nss_ch70_Amherst.png",
    nss_pivot=nss_piv,
    enrollment=enroll,
    title="Amherst: Chapter 70 Aid and Net School Spending",
    right_ylim=31000,
    left_ylim=1500,
)

# Generate table data
latest_year = int(nss_piv.index.max())
cat_rows, cat_total, cat_start_map = build_nss_category_data(nss_piv, latest_year)
```

## Run Main Script

Generate all NSS/Ch70 plots:
```bash
python nss_ch70_main.py
```

Generates 6 PNG files:
- nss_ch70_ALPS_PK_12.png
- nss_ch70_Amherst_Pelham.png
- nss_ch70_Amherst.png
- nss_ch70_Leverett.png
- nss_ch70_Pelham.png
- nss_ch70_Shutesbury.png

## Future Work

### PDF Integration
1. Add NSS/Ch70 plots to compose_pdf.py workflow
2. Create NSS/Ch70 comparison tables with red/green shading
3. Add new PDF section after expenditure analysis
4. Update Table of Contents with NSS/Ch70 section link
5. Add NSS/Ch70 data tables to Appendix B

### Red/Green Shading Logic
Compare district values to aggregate baselines (similar to expenditure tables):
- **Red**: District spending ≥2% higher than baseline
- **Green**: District spending ≥2% lower than baseline
- Apply to both 2009 and 2024 columns
- Apply to all three CAGR columns

### Additional Analysis
- Identify districts where Ch70 Aid > Required NSS (edge case)
- Analyze correlation between Ch70 Aid and district enrollment
- Track changes in state aid percentage over time
- Compare required vs actual NSS funding gaps

## Data Insights (Amherst 2024)

| Component          | Value/Pupil | % of Total |
|-------------------|-------------|------------|
| Ch70 Aid          | $6,271      | 22%        |
| Req NSS (adj)     | $11,151     | 38%        |
| Actual NSS (adj)  | $11,670     | 40%        |
| **Total NSS**     | **$29,093** | **100%**   |

**Growth Rates (CAGR):**
- 15-year: 4.65%
- 10-year: 4.39%
- 5-year: 4.89%

**Interpretation:**
- Amherst receives 22% of its per-pupil NSS from state Chapter 70 aid
- Required local contribution is 38% ($11,151/pupil)
- District exceeds requirement by 40%, spending $11,670/pupil above required NSS
- Total NSS has grown at ~4.6% annually over the past 15 years
