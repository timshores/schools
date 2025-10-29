# NSS/Ch70 Calculation Methodology

## Source Data
**Input file:** `E2C_Hub_MA_DESE_Data.xlsx` → Tab: `profile_DataC70`

**Columns used from profile_DataC70:**
- `DIST_NAME` - District name
- `YEAR` - Fiscal year
- `actualNSS` - Actual Net School Spending (total dollars)
- `rqdnss2` - Required Net School Spending (total dollars)
- `c70aid` - Chapter 70 Aid (total dollars)
- `distfoundenro` - District Foundation Enrollment

**Enrollment source:** From main data tab
- `In-District FTE Pupils` from df (expenditure data)

## Calculation Steps

### Step 1: Get Enrollment
From the main expenditure data (`df`):
```
Filter: DIST_NAME = "Amherst-Pelham"
        IND_CAT = "Student Enrollment"
        IND_SUBCAT = "In-District FTE Pupils"
Extract: YEAR, IND_VALUE (enrollment)
```

### Step 2: Get Ch70 Data
From profile_DataC70 (`c70`):
```
Filter: DIST_NAME = "Amherst-Pelham"
        YEAR <= 2024
Extract: YEAR, actualNSS, rqdnss2, c70aid, distfoundenro
```

### Step 3: Merge and Calculate Per-Pupil Values
Join enrollment data with Ch70 data by YEAR, then divide by enrollment:

**Column 1: "Ch70 Aid"**
```
Ch70 Aid = c70aid / enrollment
```
This is Chapter 70 state aid per pupil.

**Column 2: "Req NSS (minus Ch70)"**
```
Req NSS (minus Ch70) = max(0, rqdnss2 - c70aid) / enrollment
```
This is the Required NSS contribution beyond Ch70 aid, per pupil.
- If Ch70 aid exceeds required NSS (rare), this is 0
- Normally this is positive (local contribution required)

**Column 3: "Actual NSS (minus Req NSS)"**
```
Actual NSS (minus Req NSS) = (actualNSS - rqdnss2) / enrollment
```
This is the difference between actual and required NSS, per pupil.
- Positive: District spending above requirement
- Negative: District underfunding (rare)

## Stacking Interpretation

When these three values are stacked in a chart (bottom to top):
1. **Ch70 Aid** (bottom) - State contribution
2. **Req NSS (minus Ch70)** (middle) - Required local contribution
3. **Actual NSS (minus Req NSS)** (top) - Additional spending beyond requirement

**Total height** = Ch70 Aid + Req NSS (minus Ch70) + Actual NSS (minus Req NSS)
                 = c70aid/enroll + (rqdnss2-c70aid)/enroll + (actualNSS-rqdnss2)/enroll
                 = actualNSS / enrollment
                 = **Actual NSS per pupil**

## Example Calculation for Amherst-Pelham 2024

**From profile_DataC70:**
- c70aid = $9,754,000
- rqdnss2 = $23,571,000
- actualNSS = $31,285,000

**From expenditure data (In-District FTE):**
- enrollment = 1,209.3

**Calculated per-pupil values:**
```
Ch70 Aid = $9,754,000 / 1,209.3 = $8,067.81 per pupil

Req NSS (minus Ch70) = max(0, $23,571,000 - $9,754,000) / 1,209.3
                     = $13,817,000 / 1,209.3
                     = $11,414.49 per pupil

Actual NSS (minus Req NSS) = ($31,285,000 - $23,571,000) / 1,209.3
                           = $7,714,000 / 1,209.3
                           = $6,383.10 per pupil
```

**Verification:**
Total = $8,067.81 + $11,414.49 + $6,383.10 = $25,865.40 per pupil
Check: $31,285,000 / 1,209.3 = $25,865.40 ✓

## Common Discrepancies

### Issue 1: Enrollment Mismatch
The calculation uses **In-District FTE Pupils** from the expenditure data, NOT the foundation enrollment (distfoundenro) from Ch70 data.

If you're calculating manually using foundation enrollment, values will differ.

### Issue 2: Missing Enrollment Years
For years where In-District FTE is missing, the code uses the most recent available enrollment as a proxy (see lines 1154-1159 in school_shared.py).

### Issue 3: Rounding
CSV values are stored with full precision. Manual calculations may round at different steps.

### Issue 4: Years Before 1993
The data only includes years from 1993 onwards where Ch70 data exists.

## How to Verify

To verify a specific year's calculation for Amherst-Pelham:

1. **Get enrollment for that year:**
   - Open `E2C_Hub_MA_DESE_Data.xlsx` → Main expenditure tab
   - Filter: DIST_NAME = "Amherst-Pelham", IND_CAT = "Student Enrollment", IND_SUBCAT = "In-District FTE Pupils"
   - Find IND_VALUE for the year

2. **Get Ch70 data for that year:**
   - Open `E2C_Hub_MA_DESE_Data.xlsx` → profile_DataC70 tab
   - Filter: DIST_NAME = "Amherst-Pelham", YEAR = [year]
   - Extract: actualNSS, rqdnss2, c70aid

3. **Calculate per-pupil values:**
   ```
   Ch70 Aid = c70aid / enrollment
   Req NSS (minus Ch70) = max(0, rqdnss2 - c70aid) / enrollment
   Actual NSS (minus Req NSS) = (actualNSS - rqdnss2) / enrollment
   ```

4. **Compare with CSV:**
   - Open `amherst_pelham_nss_pivot.csv`
   - Find row for that year
   - Compare calculated values

## Source Code Reference

Function: `prepare_district_nss_ch70()`
File: `school_shared.py` lines 1089-1180
