# CAGR Cache Design

## Purpose
Cache CAGR (Compound Annual Growth Rate) calculations for all categories and metrics to:
1. Enable QA desk-checking against printed report
2. Avoid recomputing CAGR multiple times
3. Provide transparency into growth calculations

## Cache File Format

### Filename: `{district}_cagr.csv`

Example: `amherst_pelham_cagr.csv`

### Structure:

```csv
Metric_Type,Category,Latest_Year,Latest_Value,Year_5yr,Value_5yr,Year_10yr,Value_10yr,Start_Year,Start_Value,CAGR_5yr,CAGR_10yr,CAGR_15yr
Expenditure,Teachers,2024,8270.00,2019,7295.00,2014,6420.00,2009,5491.00,0.0256,0.0203,0.0277
Expenditure,Insurance Retirement and Other,2024,6248.00,2019,5145.00,2014,4235.00,2009,3305.00,0.0436,0.0468,0.0434
Expenditure,Pupil Services,2024,2766.00,2019,2267.00,2014,1902.00,2009,1645.00,0.0435,0.0460,0.0353
Expenditure,Guidance Counseling and Testing,2024,1465.00,2019,1035.00,2014,798.00,2009,525.00,0.0963,0.0717,0.0708
...
Expenditure,Total PPE,2024,28233.00,2019,24085.00,2014,20456.00,2009,16211.00,0.0483,0.0381,0.0377
Enrollment,In-District FTE,2024,1209.3,2019,1325.5,2014,1429.8,2009,1712.0,-0.0195,-0.0194,-0.0229
Enrollment,Out-of-District FTE,2024,146.7,2019,158.2,2014,143.5,2009,111.5,-0.0151,-0.0014,0.0185
NSS,Ch70 Aid,2024,7658.08,2019,6643.05,2014,5865.27,2009,5487.86,0.0288,0.0270,0.0225
NSS,Req NSS (minus Ch70),2024,10834.81,2019,8271.12,2014,7379.94,2009,4382.68,0.0555,0.0391,0.0622
NSS,Actual NSS (minus Req NSS),2024,6058.93,2019,5364.71,2014,3837.71,2009,3406.99,0.0246,0.0467,0.0391
NSS,Total Actual NSS per pupil,2024,24551.82,2019,20278.88,2014,17082.92,2009,13277.52,0.0390,0.0369,0.0418
Enrollment,Foundation Enrollment,2024,1274.0,2019,1401.0,2014,1509.0,2009,1801.0,-0.0232,-0.0214,-0.0228
```

## Column Definitions

- **Metric_Type**: Type of metric
  - `Expenditure` - Spending categories
  - `Enrollment` - FTE metrics
  - `NSS` - Net School Spending / Ch70 components

- **Category**: Name of category or metric
  - For Expenditure: Category names (Teachers, Insurance, etc.)
  - For Enrollment: FTE metric names (In-District FTE, etc.)
  - For NSS: Component names (Ch70 Aid, Req NSS, etc.)
  - Special: "Total PPE", "Total Actual NSS per pupil"

- **Latest_Year**: Most recent year in data (typically 2024)

- **Latest_Value**: Value in most recent year ($/pupil or FTE count)

- **Year_5yr**: Year for 5-year CAGR calculation (typically 2019)

- **Value_5yr**: Value in Year_5yr (for desk-checking 5-year CAGR)

- **Year_10yr**: Year for 10-year CAGR calculation (typically 2014)

- **Value_10yr**: Value in Year_10yr (for desk-checking 10-year CAGR)

- **Start_Year**: Starting year for 15-year CAGR (typically 2009)

- **Start_Value**: Value in start year (for desk-checking 15-year CAGR)

- **CAGR_5yr**: 5-year CAGR (as decimal, e.g., 0.0435 = 4.35%)

- **CAGR_10yr**: 10-year CAGR

- **CAGR_15yr**: 15-year CAGR

## Calculation Method

CAGR is calculated using `compute_cagr_last()` function from compose_pdf.py:

```python
CAGR = (End_Value / Start_Value)^(1 / Years) - 1
```

Special cases:
- Zero values: CAGR = NaN
- Sign crossing (negative to positive): Uses average annual rate
- Both negative: Uses absolute values

## Implementation

### Save Function
```python
def save_district_cagr(district_name: str, cagr_data: pd.DataFrame) -> None:
    """
    Save district CAGR calculations to CSV.

    Args:
        district_name: District name
        cagr_data: DataFrame with columns:
                   [Metric_Type, Category, Latest_Year, Latest_Value,
                    Year_5yr, Value_5yr, Year_10yr, Value_10yr,
                    Start_Year, Start_Value, CAGR_5yr, CAGR_10yr, CAGR_15yr]
    """
```

### Load Function
```python
def load_district_cagr(district_name: str) -> Optional[pd.DataFrame]:
    """
    Load district CAGR calculations from CSV.

    Returns:
        DataFrame with CAGR data, or None if not cached
    """
```

### Generation
Add to `generate_district_cache.py` after generating epp_pivot, lines, nss_pivot:

```python
# Calculate CAGR for all metrics
cagr_data = calculate_district_cagr(epp_pivot, lines, nss_pivot, latest_year)
save_district_cagr(district_name, cagr_data)
```

## Usage in Reports

When building tables in compose_pdf.py, instead of recalculating CAGR:

```python
# Current: Recalculates CAGR each time
c5 = compute_cagr_last(epp_pivot[sc], 5)
c10 = compute_cagr_last(epp_pivot[sc], 10)
c15 = compute_cagr_last(epp_pivot[sc], 15)

# With cache: Load pre-calculated CAGR
cagr_data = load_district_cagr(district_name)
row = cagr_data[(cagr_data['Metric_Type']=='Expenditure') &
                (cagr_data['Category']==sc)]
c5 = row['CAGR_5yr'].values[0]
c10 = row['CAGR_10yr'].values[0]
c15 = row['CAGR_15yr'].values[0]
```

## Benefits

1. **QA**: Can desk-check CAGR values against printed tables
2. **Performance**: No need to recalculate CAGR multiple times
3. **Consistency**: Same CAGR values used throughout report
4. **Transparency**: All growth calculations documented in CSV
5. **Debugging**: Easy to verify CAGR calculations manually
