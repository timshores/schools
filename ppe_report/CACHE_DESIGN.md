# CR CacheMonster01: District-Level Caching Design

## Architecture Overview

**Core Principle:** Compute each district's data exactly once, cache to CSV, reuse everywhere.

```
Source Data (df, reg, c70)
    ↓
Generate District Cache (run once, or when data changes)
    ↓
cache/
  ├── district_data/
  │   ├── adams-cheshire_epp_pivot.csv          # Categories × Years
  │   ├── adams-cheshire_lines.csv              # FTE time series
  │   ├── adams-cheshire_nss_pivot.csv          # NSS/Ch70 × Years
  │   ├── amherst-pelham_epp_pivot.csv
  │   ├── amherst-pelham_lines.csv
  │   ├── amherst-pelham_nss_pivot.csv
  │   ├── boston_epp_pivot.csv
  │   ├── boston_lines.csv
  │   ├── boston_nss_pivot.csv
  │   └── ... (ALL ~150-200 MA districts, 3 files each = ~450-600 CSVs)
  │
  ├── cohort_data/                              # Generated from district cache
  │   ├── tiny_epp_pivot.csv                    # Weighted aggregation
  │   ├── tiny_lines_sum.csv
  │   ├── tiny_lines_mean.csv
  │   └── ... (all cohorts)
  │
  └── metadata.json                             # Cache timestamp, data source info
    ↓
All scripts read from cache
    ↓
QA: Desk check CSVs against printed PDF
```

## Cache Directory Structure

```
ppe_report/
├── cache/
│   ├── district_data/          # ALL MA districts (~150-200 districts)
│   ├── cohort_data/            # Cohort aggregations (optional)
│   ├── executive_summary/      # ES-specific intermediate data
│   └── metadata.json           # Cache metadata
```

**Key Decision:** Cache ALL districts in Massachusetts, not just Western MA.

**Rationale:**
- Marginal cost of caching all districts is minimal
- Makes report extensible to any region (Central MA, Boston, etc.)
- Cache generation is one-time cost
- Provides complete data export for all of MA
- Future reports can select any subset from cache

## File Naming Convention

### District Files
- `{district_slug}_epp_pivot.csv` - Expenditure per pupil by category
- `{district_slug}_lines.csv` - FTE time series (Total FTE, In-District FTE, etc.)
- `{district_slug}_nss_pivot.csv` - NSS/Ch70 funding breakdown
- `{district_slug}_metadata.json` - District-specific metadata (years, enrollment range, etc.)

### Cohort Files (aggregated from districts)
- `{cohort}_epp_pivot.csv` - Weighted mean PPE
- `{cohort}_lines_sum.csv` - Summed FTE across cohort
- `{cohort}_lines_mean.csv` - Mean FTE across cohort
- `{cohort}_nss_pivot.csv` - Weighted mean NSS/Ch70

### Executive Summary Files
- `district_yoy.csv` - YoY growth rates for all districts
- `cohort_yoy.csv` - YoY growth rates for cohorts
- `district_cagr_chunks.csv` - CAGR for 5-year periods
- `district_cagr_15year.csv` - 15-year CAGR

## CSV Format Specifications

### epp_pivot.csv
```csv
YEAR,Administration,Classroom & Specialists,Fixed Charges,... ,Total
2009,1250.50,5200.30,890.20,...,12500.00
2010,1280.75,5350.10,920.50,...,12800.00
...
```
- Index: YEAR
- Columns: Expenditure categories (matching CATEGORY_ORDER)
- Values: Dollars per pupil

### lines.csv
```csv
YEAR,Total FTE,In-District FTE,School Choice Out,Ch70 Foundation Enrollment
2009,1200.5,1150.0,50.5,1180.0
2010,1210.2,1160.5,49.7,1185.0
...
```
- Index: YEAR
- Columns: Different FTE metrics
- Values: FTE counts

### nss_pivot.csv
```csv
YEAR,Ch70 Aid,Required Local Contribution,Required Net School Spending,...
2009,5000000,3000000,8000000,...
2010,5100000,3100000,8200000,...
...
```
- Index: YEAR
- Columns: NSS/Ch70 categories
- Values: Total dollars (will be divided by enrollment when used)

## Implementation Plan

### Phase 1: Cache Generation Script
**New file:** `generate_district_cache.py`

```python
def generate_all_district_cache(df, reg, c70, force=False):
    """
    Generate CSV cache for ALL Massachusetts districts.

    Args:
        df: Main expenditure data
        reg: Regional district data
        c70: Chapter 70 funding data
        force: If True, regenerate even if cache exists
    """
    # Get list of ALL districts in df (unique DISTRICT_NAME values)
    # For each district:
    #   - Check if cache exists and is valid
    #   - If not (or force=True):
    #     - Call prepare_district_epp_lines()
    #     - Call prepare_district_nss_ch70()
    #     - Save to CSV with metadata
    #   - Log progress (e.g., "Processing district 45/178...")
    # Final log: "Cached 178 districts, 534 CSV files"
```

### Phase 2: Cache Read/Write Functions
**Add to:** `cache_manager.py` (if exists) or new `district_cache.py`

```python
def save_district_epp_pivot(district_slug, epp_pivot, cache_dir)
def load_district_epp_pivot(district_slug, cache_dir) -> pd.DataFrame
def save_district_lines(district_slug, lines_dict, cache_dir)
def load_district_lines(district_slug, cache_dir) -> Dict[str, pd.Series]
def save_district_nss_pivot(district_slug, nss_pivot, cache_dir)
def load_district_nss_pivot(district_slug, cache_dir) -> pd.DataFrame
```

### Phase 3: Modify Prepare Functions
**Modify:** `school_shared.py`

```python
def prepare_district_epp_lines(df, district_name, use_cache=True):
    """
    Modified to check cache first.

    If use_cache and cache exists and is valid:
        return load_district_epp_pivot(), load_district_lines()
    else:
        compute as before
        save to cache
        return results
    """
```

### Phase 4: Cohort Aggregation from Cache
**New function in:** `school_shared.py`

```python
def prepare_western_epp_lines_from_cache(bucket, district_list):
    """
    Load individual district caches, compute weighted aggregation.

    Instead of processing raw df, load cached district data:
    - Load each district's epp_pivot and lines from CSV
    - Compute weighted mean based on enrollment
    - Return aggregated epp_pivot and lines
    """
```

### Phase 5: Integration with Pipeline
**Modify:** `generate_report.py`

Add cache generation as first step:
```python
PIPELINE = [
    ("generate_district_cache.py", "Generate district-level data cache"),  # NEW
    ("threshold_analysis.py", "Threshold analysis for shading thresholds"),
    ("executive_summary_plots.py", "Executive Summary plots"),
    # ... rest of pipeline
]
```

## Benefits

### QA Benefits (Primary Goal)
1. **Desk checking:** Print CSV, compare to PDF side-by-side
2. **Transparency:** See exact numbers that go into each visualization
3. **Debugging:** When a number looks wrong, trace it to source CSV
4. **Audit trail:** CSVs serve as documentation of data transformations

### Performance Benefits (Secondary Goal)
1. **Eliminate redundancy:** Compute each district once instead of 10+ times
2. **Incremental updates:** Only regenerate cache when source data changes
3. **Faster iterations:** When tweaking visualizations, reuse cached data
4. **Parallel processing:** Could parallelize district cache generation

### Code Quality Benefits
1. **Separation of concerns:** Data preparation vs. visualization
2. **Testability:** Can unit test against known CSV fixtures
3. **Modularity:** Scripts can be run independently with cached inputs

## Cache Invalidation Strategy

**When to regenerate cache:**
1. Source data files change (df, reg, c70)
2. User passes `--force-recompute` flag
3. Cache metadata version mismatch (if we change CSV schema)

**How to detect:**
- Store source file checksums in metadata.json
- Compare checksums on load
- If mismatch, regenerate that district's cache

## Next Steps

1. Create `generate_district_cache.py` script
2. Implement CSV read/write functions
3. Test with single district (e.g., Amherst-Pelham)
4. Verify CSV matches current output
5. Extend to all districts
6. Update prepare functions to use cache
7. Run full pipeline with caching enabled
8. QA: Desk check CSVs vs. PDF

## Questions to Resolve

1. Should cohort aggregations be cached, or always computed from district cache?
   - **Recommendation:** Always compute from district cache (more flexible)

2. Should executive summary intermediate data be cached?
   - **Recommendation:** Yes, cache YoY and CAGR data for QA

3. What format for lines.csv - wide or long?
   - **Recommendation:** Wide format (as shown above) for easy reading

4. Include metadata in each CSV or separate files?
   - **Recommendation:** Separate metadata.json per district for cleaner CSVs
