# Data Products Inventory for Western MA PPE Report
## CR CacheMonster01 - Comprehensive listing of all tables and plots with their input dataframes

---

## Executive Summary

### ES-1: YoY Growth Separate Panes Plot
- **File:** `executive_summary_yoy_panes.png`
- **Type:** Plot
- **Script:** `executive_summary_plots.py` - `plot_yoy_separate_panes()`
- **Input Dataframes:**
  - `df` (main expenditure data) → filtered to districts of interest
  - `reg` (regional district data) → used to assign cohorts
  - **Intermediate dataframes:**
    - `epp_pivot` (per district): from `prepare_district_epp_lines(df, dist)`
    - `epp_pivot` (per cohort): from `weighted_epp_aggregation(df, district_list)`
    - `total_ppe` series: sum of epp_pivot across columns
    - `district_yoy` dict: YoY growth calculated from total_ppe
    - `cohort_yoy` dict: YoY growth calculated from cohort aggregates

### ES-2: CAGR 5-Year Grouped Bars
- **File:** `executive_summary_cagr_grouped.png`
- **Type:** Plot
- **Script:** `executive_summary_plots.py` - `plot_cagr_grouped_bars()`
- **Input Dataframes:** Same as ES-1
- **Intermediate dataframes:**
  - `district_cagr` dict: CAGR chunks (2009-2014, 2014-2019, 2019-2024) per district
  - `cohort_cagr` dict: CAGR chunks per cohort

### ES-3: CAGR 15-Year Bars
- **File:** `executive_summary_cagr_15year.png`
- **Type:** Plot
- **Script:** `executive_summary_plots.py` - `plot_cagr_15year_bars()`
- **Input Dataframes:** Same as ES-1
- **Intermediate dataframes:**
  - `district_cagr_15y` dict: 15-year CAGR (2009-2024) per district
  - `cohort_cagr_15y` dict: 15-year CAGR per cohort

### ES-4: CAGR Legend
- **File:** `executive_summary_cagr_legend.png`
- **Type:** Legend graphic
- **Script:** `executive_summary_plots.py` - `plot_cagr_legend()`
- **Input Dataframes:** Same as ES-1 (for color assignment)

### ES-5: Executive Summary Cohort Tables (in compose_pdf.py)
- **Type:** Tables (embedded in PDF)
- **Script:** `compose_pdf.py` - `build_page_dicts()` lines 2721-2900
- **Input Dataframes:**
  - `df` → `epp_pivot` via `prepare_western_epp_lines()` or `prepare_district_epp_lines()`
  - `reg` → cohort assignment via `get_western_cohort_districts()`
  - `c70` → Ch70 Aid data via `prepare_aggregate_nss_ch70_weighted()` or `prepare_district_nss_ch70()`
- **Intermediate dataframes:**
  - `cohorts_exec`: cohort to district list mapping
  - `epp` pivot table for each cohort/district
  - `nss_piv` for Ch70 Aid data
  - `rows`, `total`, `start_map` from `_build_category_data()`

---

## Section 1: Western MA Traditional District Trends

### S1-1: PPE Overview All Western
- **File:** `ppe_overview_all_western.png`
- **Type:** Plot (stacked bar chart)
- **Script:** `district_expend_pp_stack.py` - `plot_all_western_overview()`
- **Input Dataframes:**
  - `df` (main expenditure data)
  - `reg` (regional district data) → filter Western MA traditional districts
  - `c70` (Chapter 70 data)
- **Intermediate dataframes:**
  - `western_traditional_districts` list from filtering
  - Per-district `epp_pivot` and `lines` from `prepare_district_epp_lines()`
  - Combined data structure for all districts

### S1-2: Category Tables (for each district/cohort)
- **Type:** Tables
- **Script:** `compose_pdf.py` - `_build_category_table()`
- **Input Dataframes:**
  - `epp_pivot` from `prepare_western_epp_lines()` or `prepare_district_epp_lines()`
  - Processes via `_build_category_data(epp_pivot, latest_year, context, cmap_all)`
- **Intermediate dataframes:**
  - `rows`: list of category data rows
  - `total`: total PPE row
  - `start_map`: starting values for CAGR calculations

### S1-3: FTE Tables
- **Type:** Tables
- **Script:** `compose_pdf.py` - `_build_fte_table()`
- **Input Dataframes:**
  - `lines` dict from `prepare_western_epp_lines()` or `prepare_district_epp_lines()`
  - Contains series for: Total FTE, In-District FTE, School Choice Out
- **Intermediate dataframes:**
  - Output from `_build_fte_data(lines, latest_year)`

### S1-4: NSS/Ch70 Tables
- **Type:** Tables
- **Script:** `compose_pdf.py` - `_build_nss_ch70_table()`
- **Input Dataframes:**
  - `nss_data` from `prepare_aggregate_nss_ch70_weighted()` or `prepare_district_nss_ch70()`
  - `c70` dataframe
- **Intermediate dataframes:**
  - `baseline_map` from `_build_nss_ch70_baseline_map()`
  - NSS category breakdown

---

## Section 2: Western MA Cohort Details

### S2-1: Cohort Distribution Tables
- **Type:** Tables
- **Script:** `compose_pdf.py` - `build_cohort_distribution_table()`
- **Input Dataframes:**
  - `dist_df`: dataframe of district-level metrics for a cohort
  - Built from looping through districts and calling `prepare_district_epp_lines()`
- **Intermediate dataframes:**
  - Quartile calculations on `dist_df`
  - Per-district metric values

### S2-2: Cohort Boxplot Images
- **File:** `output/boxplots/cohort_{cohort_name}_{metric_name}.png` (multiple files)
- **Type:** Inline plots
- **Script:** `compose_pdf.py` - embedded matplotlib code in `build_cohort_distribution_table()`
- **Input Dataframes:** Same `dist_df` as S2-1

### S2-3: Scatterplot Table (in report body)
- **Type:** Table
- **Script:** `compose_pdf.py` - `_build_scatterplot_table()`
- **Input Dataframes:**
  - `district_data` list from `_build_scatterplot_district_table(df, reg, latest_year)`
  - Uses `df` and `reg` to build list of tuples with enrollment and PPE data

---

## Section 3: Selected Districts

### S3-1: Individual District Plots (Simple)
- **File:** `expenditures_per_pupil_vs_enrollment_{district}_simple.png`
- **Type:** Plot (stacked area chart with enrollment line)
- **Script:** `district_expend_pp_stack.py` - `plot_one_simple()`
- **Input Dataframes:**
  - `epp_pivot`: from `prepare_district_epp_lines(df, district_name)`
  - `lines`: dict with FTE series from same function
- **Intermediate dataframes:** None (direct plotting)

### S3-2: Individual District Plots (Detailed)
- **File:** `expenditures_per_pupil_vs_enrollment_{district}_detail.png`
- **Type:** Plot (stacked area chart with multiple FTE lines)
- **Script:** `district_expend_pp_stack.py` - `plot_one()`
- **Input Dataframes:** Same as S3-1
- **Intermediate dataframes:** None (direct plotting)

### S3-3: Regional District Plots
- **File:** `regional_expenditures_per_pupil_Western_Traditional_{cohort}.png`
- **Type:** Plot (stacked area chart)
- **Script:** `district_expend_pp_stack.py` - `plot_one()`
- **Input Dataframes:**
  - `epp_pivot` from `prepare_western_epp_lines(df, reg, bucket, c70, districts=district_list)`
  - `lines_sum` and `lines_mean` from same function
- **Intermediate dataframes:** Weighted aggregation across districts in cohort

---

## Appendix A: Data Sources & Methodology

### A-1: Threshold Analysis Summary Table
- **File:** Generated data embedded in PDF
- **Type:** Table
- **Script:** `threshold_analysis.py` generates CSV, `compose_pdf.py` builds table
- **Input Dataframes:**
  - Results from `threshold_analysis.py` - loads `df`, calculates distributions
- **Intermediate dataframes:**
  - Distribution statistics for each cohort and metric
  - Quantile calculations

---

## Appendix B: Calculations and Examples

### B-1: Calculation Example Tables
- **Type:** Tables showing step-by-step calculations
- **Script:** `compose_pdf.py` - embedded in `build_page_dicts()`
- **Input Dataframes:**
  - Same `epp_pivot`, `lines`, `nss_data` as used in main report
  - Shows actual data for example districts

---

## Appendix C: Data Tables

### C-1: Category Data Tables
- **Type:** Large data tables
- **Script:** `compose_pdf.py` - `_build_epp_data_table()`
- **Input Dataframes:**
  - `epp_pivot` from prepare functions
- **Intermediate dataframes:** Transposed view of epp_pivot

### C-2: FTE Data Tables
- **Type:** Large data tables
- **Script:** `compose_pdf.py` - `_build_fte_data_table()`
- **Input Dataframes:**
  - `lines` dict from prepare functions

### C-3: NSS/Ch70 Data Tables
- **Type:** Large data tables
- **Script:** `compose_pdf.py` - `_build_nss_ch70_data_table()`
- **Input Dataframes:**
  - `nss_pivot` from prepare functions

---

## Appendix D: Additional Visualizations

### D-1: Enrollment Scatterplots (by year)
- **File:** `enrollment_1_scatterplot_{year}.png`
- **Type:** Plot (scatter plot)
- **Script:** `western_enrollment_plots_individual.py`
- **Input Dataframes:**
  - `df` (main expenditure data) filtered to Western MA for specific year
  - `reg` (regional district data)
- **Intermediate dataframes:**
  - Per-district enrollment and PPE for that year
  - Cohort assignments

### D-2: Enrollment Histogram
- **File:** `enrollment_3_histogram.png`
- **Type:** Plot
- **Script:** `western_enrollment_plots_individual.py`
- **Input Dataframes:**
  - `df` filtered to latest year, Western MA districts
- **Intermediate dataframes:**
  - Enrollment distribution data

### D-3: Enrollment Grouping Visualization
- **File:** `enrollment_4_grouping.png`
- **Type:** Plot
- **Script:** `western_enrollment_plots_individual.py`
- **Input Dataframes:**
  - Same as D-2
- **Intermediate dataframes:**
  - Cohort boundary calculations

### D-4: Western MA Choropleth Maps
- **File:** `western_ma_choropleth_{year}.png`
- **Type:** Geographic map
- **Script:** `western_map.py`
- **Input Dataframes:**
  - `df` filtered to specific year
  - `reg` for district boundaries
  - GeoJSON data for map shapes
- **Intermediate dataframes:**
  - Merged geodata with PPE metrics

### D-5: Western MA PPE Comparison Maps
- **File:** `western_ma_ppe_comparison_{year}.png`
- **Type:** Geographic comparison maps
- **Script:** `western_map.py`
- **Input Dataframes:** Same as D-4

### D-6: Western MA CAGR Comparison Maps
- **File:** `western_ma_cagr_comparison_2009_{year}.png`
- **Type:** Geographic comparison maps showing growth
- **Script:** `western_map.py`
- **Input Dataframes:**
  - `df` filtered to year range (2009 to {year})
  - CAGR calculations across years
- **Intermediate dataframes:**
  - CAGR per district

### D-7: NSS/Ch70 Plots (per district/cohort)
- **File:** `nss_ch70_{district_or_cohort}.png`
- **Type:** Plot (stacked area chart for funding sources)
- **Script:** `nss_ch70_main.py`
- **Input Dataframes:**
  - `df` (main expenditure data)
  - `c70` (Chapter 70 funding data)
  - `reg` (regional data)
- **Intermediate dataframes:**
  - `nss_pivot` from NSS preparation functions
  - NSS category breakdown per year

---

## Key Shared Dataframes Across Products

### Primary Source Data (loaded once)
1. **`df`** - Main expenditure data (DISTRICT_NAME, YEAR, CATEGORY, EXPENDITURE, various FTE columns)
2. **`reg`** - Regional district information
3. **c70`** - Chapter 70 funding data

### Common Intermediate Dataframes (created repeatedly)
1. **`epp_pivot`** - Expenditure per pupil by category and year (from `prepare_district_epp_lines()` or `prepare_western_epp_lines()`)
2. **`lines`** - Dict of time series (Total FTE, In-District FTE, School Choice, etc.)
3. **`nss_pivot`** - NSS/Ch70 funding breakdown by category and year
4. **`cohorts`** - Dict mapping cohort names to district lists (from `get_western_cohort_districts()`)
5. **`cmap_all`** - Color mapping for categories (from `create_or_load_color_map()`)

### Transformation Pipeline
```
df (source)
  → filter by district/cohort/year
  → prepare_district_epp_lines() OR prepare_western_epp_lines()
  → epp_pivot (categories × years) + lines (FTE time series)
  → _build_category_data() → rows, total, start_map
  → tables and plots
```

---

## Next Steps for CSV Caching

For each intermediate dataframe listed above, we should:
1. Generate CSV output at the point of creation
2. Store in organized cache directory structure
3. Include metadata (district name, cohort, year range, timestamp)
4. Enable QA by desk-checking CSVs against printed PDFs

**Priority dataframes to cache:**
- `epp_pivot` for each district and cohort
- `lines` dict for each district and cohort
- `nss_pivot` for each district and cohort
- `district_yoy` and `cohort_yoy` from executive summary
- `dist_df` for cohort distribution analysis
- Scatterplot district data list
