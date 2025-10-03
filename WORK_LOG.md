# Work Log - School Data Analysis Project

## 2025-10-03

### Moved ALPS PK-12 NSS/Ch70 Page Back to District Section with Baseline Shading

**Change:** Moved ALPS PK-12 Chapter 70 Aid and Net School Spending page from Appendix A back to the district section (after ALPS district pages), and added baseline comparison shading.

**Implementation** (`compose_pdf.py` lines 1184-1212):
- Removed ALPS PK-12 NSS/Ch70 from Appendix A section
- Added after ALPS PK-12 district pages (following ALPS Peers comparison)
- Computes ALPS Peers baseline for comparison
- Adds red/green shading to show how ALPS PK-12 compares to ALPS Peers aggregate
- Title: "ALPS PK-12: Chapter 70 Aid and Net School Spending (vs ALPS Peers)"

**Files Modified:**
- `compose_pdf.py` - Moved ALPS PK-12 NSS/Ch70 page, added baseline shading
- Regenerated PDF (7.8 MB, 12:57)

---

### Added 2025 Data to Aggregates Using 2024 Enrollment Proxy

**Issue:** Aggregate NSS/Ch70 plots showed 2009-2024 while individual districts showed 2009-2025, causing the 2025 column in district comparison tables to have no red/green shading.

**Root Cause:** Enrollment data only available through 2024, but C70 data includes 2025. Previous code dropped 2025 for aggregates.

**Solution:** For years with missing enrollment (like 2025), use the most recent available enrollment as a proxy.

**Implementation** (`school_shared.py` lines 698-712):
```python
# For each district with missing enrollment for a year:
# 1. Find most recent available enrollment
# 2. Use that value as proxy for missing year
# This allows 2025 C70 data to be included using 2024 enrollment
```

**Verification:**
- Western MA >500: Now includes 2025 (~$224.8M per district)
- ALPS Peers: Now includes 2025
- ALPS PK-12: Now includes 2025
- Individual district comparison tables: 2025 column now has red/green shading vs aggregates

**Files Modified:**
- `school_shared.py` - Added enrollment forward-fill logic
- Regenerated all plots and PDF (7.8 MB, 12:40)

---

### FINAL FIX: Changed to Weighted Avg Per District (Not Per Pupil)

**Clarification:** The aggregates should use **weighted average per DISTRICT**, not per pupil.

**New Calculation:**
1. For each district/year: Compute per-pupil values (dollars / enrollment)
2. Compute enrollment-weighted average per-pupil across all districts
3. Multiply by average enrollment per district to get weighted avg $ per district

**Formula:**
```
weighted_avg_pp = sum(district_pp * district_enrollment) / sum(district_enrollment)
weighted_avg_per_district = weighted_avg_pp * mean(district_enrollment)
```

**Changes Made:**

1. **Rewrote `prepare_aggregate_nss_ch70_weighted()`** (`school_shared.py` lines 690-732):
   - Merges C70 data with enrollment by district/year
   - Computes per-pupil for each district
   - Calculates enrollment-weighted average per-pupil
   - Multiplies by average enrollment to get weighted avg $ per district
   - Added `include_groups=False` to fix pandas FutureWarning

2. **Updated All Labels** to "Weighted Avg $ per District":
   - **nss_ch70_plots.py** line 113: Y-axis label
   - **compose_pdf.py** lines 1227, 1288, 1308: Subtitles

**Verification:**
- Western MA >500: Shows ~$33M-$58M per district (2024)
- Values represent what an average district in the aggregate spends
- Weighted by enrollment so larger districts have more influence

**Files Modified:**
- `school_shared.py` - Complete rewrite of weighted aggregate calculation
- `nss_ch70_plots.py` - Updated y-axis label
- `compose_pdf.py` - Updated subtitles
- Regenerated all plots and PDF (7.8 MB, 12:26)

---

### CRITICAL FIX: NSS/Ch70 Data Alignment and Total Row Shading

**Issues Found:**
1. **2025 data showing 745 million instead of ~$17K per pupil** - enrollment data ended at 2024 but C70 data included 2025
2. **Total row not shaded in comparison tables** - baseline map didn't include "Total"
3. **Labels said "$ per pupil" instead of "Weighted Avg"** - misleading for aggregates
4. **Only one bar showing for 2025 on aggregate plots** - division by zero/NaN issue

**Fixes Applied:**

1. **Fixed `prepare_aggregate_nss_ch70_weighted()`** (`school_shared.py` lines 693-699):
   - Now drops years where enrollment data is missing
   - Prevents division by zero or NaN
   - Ensures consistent per-pupil calculations across all years
   - Before: 2025 showed $745M (undivided absolute dollars)
   - After: 2025 is dropped (no enrollment data available)

2. **Added Total to Baseline Maps** (`compose_pdf.py` lines 691-702):
   - `_build_nss_ch70_baseline_map()` now computes Total row baseline
   - Sums all components to get total series
   - Computes CAGR and dollar values for Total
   - Enables red/green shading on Total row

3. **Updated Labels to "Weighted Avg $ per Pupil"**:
   - **nss_ch70_plots.py** line 113: Y-axis label
   - **compose_pdf.py** lines 1214, 1275, 1295: Subtitles for all aggregates

**Verification:**
- Western MA >500: Now shows bars for 2009-2024 (not 2025), values $8K-$16K per pupil
- Total row shading: Now works in comparison tables
- Labels: Clearly indicate "Weighted Avg $ per Pupil" for aggregates

**Files Modified:**
- `school_shared.py` - Fixed enrollment alignment in weighted aggregate function
- `compose_pdf.py` - Added Total to baseline maps, updated subtitles
- `nss_ch70_plots.py` - Updated y-axis label
- Regenerated all plots and PDF (7.7 MB)

---

### Major NSS/Ch70 Enhancements - Weighted Per-Pupil Aggregates (Earlier Session)

**Changes Made:**

1. **Created Weighted Per-Pupil Aggregate Function** (`school_shared.py` lines 632-707):
   - Added `prepare_aggregate_nss_ch70_weighted()` function
   - Computes enrollment-weighted per-pupil NSS/Ch70 values for aggregates
   - Formula: weighted_avg = sum(dollars) / sum(enrollment) for each year/component
   - Enables proper comparison between different-sized aggregates

2. **Updated All Aggregate Calls to Use Weighted Version**:
   - **compose_pdf.py**: Changed 5 locations to use `prepare_aggregate_nss_ch70_weighted()`:
     - Western MA baselines for district comparisons (lines 1011, 1019)
     - Western MA aggregate pages in Appendix A (line 1168)
     - ALPS Peers aggregate page in Appendix A (line 1230)
     - ALPS PK-12 aggregate page (line 1250)
   - **nss_ch70_main.py**: Updated 4 aggregate calls (lines 76, 83, 92, 100)

3. **Moved ALPS PK-12 to Appendix A** (`compose_pdf.py`):
   - Removed ALPS PK-12 NSS/Ch70 page from district section (line 1133)
   - Added to Appendix A with other aggregates (lines 1248-1266)
   - Ensures all aggregates are grouped together

4. **Added raw_nss Data for Aggregates** (`compose_pdf.py`):
   - Western MA aggregates now store raw_nss for Appendix B tables (line 1184)
   - ALPS Peers stores raw_nss (line 1245)
   - ALPS PK-12 stores raw_nss (line 1265)

5. **Fixed Total Row Shading** (`compose_pdf.py` lines 550-586):
   - Added shading logic for Total row in NSS/Ch70 comparison tables
   - Shades start dollar (col 2), latest dollar (col 6), and all CAGR columns (cols 3-5)
   - Uses same red/green shading rules as component rows

6. **Limited NSS/Ch70 Plots to 2009-2025** (`nss_ch70_plots.py` lines 79-85):
   - Plots now filter to 2009-2025 timeframe
   - Tables remain at 2010-2025 per existing logic in Appendix B

7. **Updated Plot Y-Axis Labels** (`nss_ch70_plots.py`):
   - Added `per_pupil` parameter to `plot_nss_ch70()` function (line 59)
   - Aggregates display "$ per pupil" label
   - Individual districts display "Dollars ($)" label
   - **nss_ch70_main.py** detects aggregates and sets flag (lines 132-133)

8. **Updated Subtitles for Per-Pupil Aggregates** (`compose_pdf.py`):
   - Western MA aggregates: "Funding components (per pupil)" (line 1176)
   - ALPS Peers: "Funding components (per pupil)" (line 1237)
   - ALPS PK-12: "Funding components (per pupil)" (line 1257)
   - Individual districts keep original subtitle without "(per pupil)"

**Impact:**
- ✅ Western MA aggregates now show weighted per-pupil values (comparable across size groups)
- ✅ ALPS Peers and ALPS PK-12 show weighted per-pupil values (proper comparison)
- ✅ Individual districts show absolute dollars (appropriate for single entities)
- ✅ Total row now has proper red/green shading in comparison tables
- ✅ All aggregates grouped together in Appendix A
- ✅ NSS/Ch70 data tables available for all aggregates in Appendix B
- ✅ Plot labels correctly indicate per-pupil vs absolute dollars

**Files Modified:**
- `school_shared.py` - Added weighted per-pupil aggregate function
- `compose_pdf.py` - Updated all aggregate calls, moved ALPS PK-12, fixed Total shading, added raw_nss
- `nss_ch70_main.py` - Updated to use weighted aggregates, detect aggregate vs district
- `nss_ch70_plots.py` - Added per_pupil parameter, limited date range to 2009-2025
- Regenerated all NSS/Ch70 plots and PDF

### Fixed NSS/Ch70 Plots to Display Absolute Dollars (Earlier Session)

**Issue:** NSS/Ch70 plots were displaying values in the $13K-$24K range instead of millions, making it appear they were per-pupil values despite code claiming absolute dollars.

**Root Cause:**
- Data preparation functions (`prepare_district_nss_ch70`, `prepare_aggregate_nss_ch70`) were correctly returning absolute dollar values from profile_DataC70 sheet
- However, `nss_ch70_main.py` was computing a single global y-axis limit across ALL districts using `compute_nss_ylim()`
- This caused small districts (e.g., Shutesbury with ~$2.6M total NSS) to have invisible bars when plotted on the same scale as large aggregates (e.g., Western MA >500 with ~$300M+)
- The result was a blank plot with an inappropriate y-axis scale (0 to 1.6 billion)

**Fix Applied:**

1. **nss_ch70_main.py** (lines 115-118, 137):
   - Removed global y-limit computation with `compute_nss_ylim()`
   - Changed `right_ylim=None` to enable matplotlib auto-scaling for each plot individually
   - Updated console output to reflect auto-scaling approach

2. **nss_ch70_plots.py**:
   - Updated module docstring (line 9): Changed "per-pupil basis" to "absolute dollars (not per-pupil)"
   - Updated `plot_nss_ch70()` docstring (line 72): Changed "per-pupil" to "absolute dollars (not per-pupil)"
   - Deprecated `compute_nss_ylim()` function (line 147): Added note that it's deprecated for absolute dollar plots due to wide variance in district sizes

**Verification:**
- Shutesbury 2024 values now correctly display:
  - Ch70 Aid (green): $645,986
  - Req NSS (adj) (amber): $820,401
  - Actual NSS (adj) (purple): $1,159,959
  - **Total: $2,626,346** (matches source data exactly)
- Plot y-axis appropriately ranges from $0 to ~$3.2M (auto-scaled to Shutesbury's size)
- All three stacked components are now visible and correctly proportioned

**Files Modified:**
- `nss_ch70_main.py` - Removed global y-limit, enabled auto-scaling
- `nss_ch70_plots.py` - Updated documentation, deprecated `compute_nss_ylim()`
- Regenerated all NSS/Ch70 plots in `output/` directory
- Regenerated `output/expenditures_series.pdf` with corrected plots

## 2025-09-29 - Session Start

### Project Context
- Python scripts for analyzing MA school district expenditure data
- Migrated from ChatGPT web interface to Claude Code in VS Code
- Core files working together:
  - `school_shared.py` - Data loading and shared utilities
  - `district_expend_pp_stack.py` - Plot generation
  - `compose_pdf.py` - PDF report compilation

### Previous Work Completed
- ✅ Refactored and verified accuracy of three core files
- ✅ Created CALCULATION_REFERENCE.md for math verification

### Current Task
- 🔄 Add appendix to PDF output showing all relevant data tables used in plots
- Goal: Enable readers to verify/understand plots by looking up values in tables
- Implementation needed in: `compose_pdf.py`

### Implementation Plan
1. ✅ Read compose_pdf.py and district_expend_pp_stack.py
2. ✅ Add data table formatting functions for:
   - Category expenditure per pupil data (epp_pivot DataFrames)
   - FTE enrollment data (lines dictionaries)
3. ✅ Create new appendix section ("Appendix B: Data Tables")
4. ✅ Format tables to show:
   - District/region name
   - All years of data (columns)
   - Categories/enrollment types (rows)
   - Values in readable format

### Changes Made (Session 2)
**Issue 1: Fixed table width overflow in Appendix B**
- Modified `_build_epp_data_table()` to accept doc_width parameter
- Modified `_build_fte_data_table()` to accept doc_width parameter
- Tables now dynamically calculate column widths to fit page width
- Changed table padding and font sizes to accommodate more columns

**Issue 2 & 3: Added calculation methodology and district membership lists**
- Added new "Calculation Methodology" page (Page 2) showing:
  - ALPS PK-12 aggregate calculation: weighted EPP formula with example
  - Member districts for ALPS PK-12: Amherst, Amherst-Pelham, Leverett, Pelham, Shutesbury
  - PK-12 District Aggregate calculation and 10 member districts
  - All Western MA Traditional Districts (≤500) with full member list
  - All Western MA Traditional Districts (>500) with full member list
- Formula shown: Weighted_EPP = Σ(District_EPP × District_In-District_FTE) / Σ(District_In-District_FTE)
- Enrollment calculation: Simple sum across all member districts

### Changes Made (Session 1)
- Added two new functions to compose_pdf.py:
  - `_build_epp_data_table()`: Creates tables showing expenditure per pupil by category across all years
  - `_build_fte_data_table()`: Creates tables showing FTE enrollment data across all years
- Modified `build_page_dicts()` to:
  - Store raw data (raw_epp, raw_lines) in page dictionaries
  - Add "Appendix B: Data Tables" section after Appendix A
  - Create data table pages for each district/region
- Modified `build_pdf()` to handle "data_table" page type
- Result: PDF now includes complete data tables for all districts and regions at the end

### Changes Made (Session 3)
**Issue 1: Move methodology to Appendix C and add CAGR**
- Moved "Calculation Methodology" from Page 2 to Appendix C (after data tables)
- Added detailed CAGR explanation with formula: CAGR = (End/Start)^(1/Years) − 1
- Added example: $10,000 → $12,000 over 5 years = 3.71% CAGR

**Issue 2: Fix data table font wrapping**
- Created new style `style_data_cell` with fontSize=7, leading=9
- Applied to all data cells in EPP and FTE tables in Appendix B
- Maintains readability while preventing line wrapping

**Issue 3-5: Improve first page comparative plot**
- Removed enrollment subplot (microplots) entirely
- Increased BAR_WIDTH from 0.58 to 0.75 (30% wider)
- Reduced SPACING_FACTOR from 1.35 to 0.95 (tighter spacing)
- Added enrollment change annotations directly on bars:
  - Shows absolute change (e.g., "+150" or "-50")
  - Shows percentage change (e.g., "+5.2%" or "-3.1%")
  - Color coded: blue for increases, red for decreases
  - Displayed in bordered boxes above each bar
- Reduced figure height from 14.0 to 10.0 (removed subplot space)
- Updated legend to remove enrollment microplot reference
- Added subtitle text explaining enrollment annotations

### Changes Made (Session 4 - Major Restructuring)
**Issue 1: Fixed data table word-wrapping**
- Reduced data cell font to fontSize=6, leading=8
- Reduced header font to fontSize=7, leading=9
- All values now fit on single lines without wrapping

**Issue 2-3: Consolidated appendix intro pages**
- Appendix A intro now on same page as first Western MA chart
- Appendix B intro now on same page as first data table
- Removed standalone intro pages, saving 2 pages

**Issue 4-7: Improved first page comparative plot**
- Moved legend from bottom to top (loc="upper center")
- Increased enrollment annotation font from 10 to 12
- Increased note font from 13 to 14
- Changed crimson color from #DC2626 to #B91C1C (deeper crimson)
- Note text now matches crimson color
- Adjusted layout: top=0.92, bottom=0.08

**Issue 8-9: BIG CHANGE - Two pages per district**
- Created new `plot_one_simple()` function for solid-color PPE plots
- Each district now has TWO pages:
  1. **Simple version**: Solid indigo color (#4F46E5), no category breakdown, no tables
  2. **Detailed version**: Category stacks with tables (original format)
- Title distinguishes them: "...(Detailed)" added to second page
- Applies to: ALPS PK-12, Amherst-Pelham, Amherst, Leverett, Pelham, Shutesbury
- Doubled plot generation: 6 districts × 2 versions = 12 plots
- Gives readers quick overview first, then detailed breakdown

### Test Results
✅ PDF generates successfully (4.7MB - larger due to double pages)
✅ Tables fit within page width without wrapping
✅ Methodology page shows all formulas and district lists
✅ Appendix A consolidated with first chart
✅ Appendix B consolidated with first data table
✅ Appendix C shows CAGR calculation with examples
✅ First page plot: legend above, larger fonts, crimson color
✅ All districts have both simple and detailed pages
✅ Simple plots use solid indigo color
✅ 27 PNG files generated (15 district plots, 2 Western plots, 1 comparative)

### Data Structures Identified
- **epp_pivot**: pd.DataFrame with years as index, spending categories as columns, $/pupil as values
- **lines**: Dict[str, pd.Series] with enrollment series (In-District FTE, Out-of-District FTE)
- Each page dict already contains cat_rows, cat_total, fte_rows with aggregated data

### Changes Made (Session 5 - Layout and Spacing Improvements)
**Issue 1: Fixed Appendix C pagination**
- Split methodology content into two separate pages to avoid footer overlap
- Page 1: CAGR definition and Aggregate District Calculations (37 lines)
- Page 2: Red/Green Shading Logic (23 lines)
- Each page stays well above the 5-line buffer before footer boundary

**Issue 2: Improved Page 1 (Western overview) layout**
- Increased figure height from 10.0 to 13.0 inches for more breathing room
- Increased x-axis tick label font from 18 to 22
- Increased y-axis label font from 24 to 26
- Increased legend font from 20 to 22
- Set legend ncol=3 explicitly for single-row layout
- Adjusted subplot margins: top from 0.94 to 0.90, giving plot more vertical space
- Legend repositioned from bbox_to_anchor=(0.5, 1.00) to (0.5, 0.97)
- Plot is now much taller and less cramped

**Issue 3: Fixed Page 2 (ALPS & Peers) legend/annotation overlap**
- Increased annotation spacing from 2% to 3% of y-axis max (PPE_PEERS_YMAX = 35000)
- Moved enrollment explainer text from y=0.995 to y=0.985
- Moved legend from bbox_to_anchor=(0.5, 0.98) to (0.5, 0.96)
- Adjusted subplot top margin from 0.92 to 0.88
- Creates more vertical space between bars/annotations and legend/text
- Prevents overlap of enrollment rectangles with legend

### Changes Made (Session 6 - Horizontal Bar Chart Redesign)
**Issue 1: Fixed Appendix C footer overlap with KeepInFrame**
- Implemented `KeepInFrame` wrapper for methodology text blocks in compose_pdf.py
- Calculates available height: page height - 1.5 inch buffer (title/subtitle/footer)
- Uses mode='shrink' to automatically resize content if needed
- Prevents text from overlapping footer regardless of content length
- Future-proof solution: adding more methodology content won't cause overlap

**Issue 2: Redesigned Page 1 Western overview to horizontal bars**
- **Complete 90° rotation**: Changed from vertical bars to horizontal bars (barh)
- **District names on y-axis**: Full names displayed (letter codes removed)
- **Improved readability**: Y-axis font increased from 10pt to 13pt
- **Better gridlines**: Increased alpha from 0.12 to 0.35, linewidth to 0.8
- **Dynamic height**: Chart height scales with number of districts (~40)
- **Fixed width**: 11 inches (fits portrait PDF perfectly)
- **Removed district code mapping table**: No longer needed, names on chart
- **Layout margins**: 28% left margin for district names
- **Sorting maintained**: Districts still sorted lowest to highest PPE (bottom to top)

**Issue 3: Redesigned Page 2 ALPS & Peers to horizontal bars**
- **90° rotation to match Page 1**: Changed from vertical to horizontal bars
- **Enrollment annotations preserved**: Now positioned to right of bars
- **Aggregate separation**: Western MA and PK-12 aggregates at bottom of chart
- **Color coding maintained**: Aggregates use gray palette, peers use blue
- **Font consistency**: Y-axis 13pt (district names), X-axis 16pt (PPE values)
- **Enrollment boxes**: Show FTE count and % change with color coding (blue/red)
- **Position adjustment**: Annotations at bar_end + 2% with left alignment
- **Legend repositioned**: At top with enrollment explainer text
- **Dynamic height**: Scales based on number of districts + aggregates
- **Layout margins**: 25% left for names, 28% right margin for annotations (72% right edge)

**Issue 4: PDF improvements**
- District Code Mapping table converted to 2-fold layout (was 3-fold)
- Added column headers: "Code", "District", "2024 PPE"
- Reduced horizontal spacing: 70% for district name, 25% for PPE
- Each fold uses 50% page width for better readability

**Code Comments Added (for future sessions)**
- Added NOTE comments explaining dynamic height calculations
- Documented bar stacking logic (base + delta segments)
- Explained aggregate vs peer district separation in horizontal layout
- Commented gridline improvements and font size choices
- Noted purpose of KeepInFrame wrapper for methodology content
- Documented removal of letter code mapping functionality

**Issue 5: Page 2 ALPS & Peers chart refinements**
- Fixed x-axis label smooshing: Added 25° rotation to $/pupil labels
- Right-aligned enrollment boxes: Changed from left-aligned (at bar end) to right-aligned at fixed position
- Extended x-axis limit from 35000 to 38000 to create space for annotations
- Positioned enrollment boxes at x=37500 (well beyond longest bar) to prevent overlap
- Extended right margin from 72% to 78% to accommodate right-aligned annotations
- All enrollment boxes now form clean vertical column at right edge
- Fixed version stamp overlap with x-axis: Moved stamp from y=0.01 to y=0.03

**Issue 6: Page 1 Western overview - Highlight districts of interest**
- Highlighted 5 districts of interest in colorblind-friendly dark orange (#FF8C00)
- Districts: Amherst-Pelham, Amherst, Leverett, Pelham, Shutesbury
- Applied bold font weight in addition to color for extra emphasis
- Orange color chosen for strong contrast and visibility in both normal and colorblind vision

**Issue 7: Reordered all comparison tables to match chronological plot order**
- **Category table columns** (old→new):
  - OLD: Swatch, Category, 2024 $/pupil, CAGR 5y, 10y, 15y
  - NEW: Swatch (no header), Category, 2009 $/pupil, CAGR 15y, 10y, 5y, 2024 $/pupil
- **FTE table columns** (old→new):
  - OLD: Swatch, FTE Series, 2024 FTE, CAGR 5y, 10y, 15y
  - NEW: Swatch (no header), FTE Series, 2019 FTE, CAGR 15y, 10y, 5y, 2024 FTE
- **Rationale**: Left-to-right chronology matches plot timeline (oldest→newest)
- **Column width optimization**: Narrowed CAGR and $/pupil columns to prevent overflow
  - Category table: 0.85" for CAGR columns, 0.95" for $/pupil columns
  - FTE table: 0.85" for all numeric columns
- **Updated _build_category_data** to calculate and return start year values (15 years before latest)
  - Changed from t0_year (5 years back) to start_year (15 years back) to match CAGR 15y timeframe
  - Renamed return value from cat_t0_map to cat_start_map for clarity
  - Returns start year values for each spending category (2009 when latest is 2024)
- **Updated shading logic** to work with new column positions:
  - CAGR columns now 3-5 (15y, 10y, 5y) instead of 5-3
  - Latest $/pupil now column 6 instead of column 2
  - **NEW: Added shading for start year $/pupil (column 2, 2009)**
    - Uses same relative comparison as latest $/pupil: (District − Baseline) / Baseline
    - Compares 2009 district spending to 2009 Western baseline
    - Red if ≥2% higher, green if ≥2% lower
- **Added START_DOLLAR to all baseline_map constructions**
  - Baseline maps now include START_DOLLAR key with start year (15y back) value
  - Applied to all 4 baseline construction locations: ALPS vs Western, ALPS vs Peers, Districts vs Western, Districts vs Peers
  - Enables comparison shading for both 2009 and 2024 $/pupil columns
- **Removed swatch column header text** for cleaner appearance
- **Added start year total row values** for both category and FTE tables

**Issue 8: Added Table of Contents with clickable links**
- Created new `build_toc_page()` function to generate TOC page
- TOC includes 11 sections with internal PDF links:
  1. All Western MA Traditional Districts: PPE Overview 2019 -> 2024
  2. ALPS PK-12 & Peers: PPE and Enrollment 2019 -> 2024
  3. ALPS PK-12
  4. Amherst-Pelham Regional
  5. Amherst
  6. Leverett
  7. Pelham
  8. Shutesbury
  9. Appendix A. Aggregate Districts for Comparison
  10. Appendix B. Data Tables
  11. Appendix C. Calculation Methodology
- Added section_id field to relevant pages for anchor targets
- Modified build_pdf() to:
  - Handle TOC page type rendering
  - Add HTML anchors (`<a name="section_id"/>`) to section titles
  - Create clickable links (`<a href="#section_id">`) in TOC
- TOC inserted as first page of PDF

**Issue 9: Fixed duplicate data table pages in Appendix B**
- Problem: Districts with multiple comparison pages (vs Western and vs Peers) were creating duplicate data tables
- Solution: Added deduplication logic using `seen_districts` set
- Now each district appears exactly once in Appendix B data tables
- Reduced redundancy: ALPS PK-12 and each district (Amherst-Pelham, Amherst, Leverett, Pelham, Shutesbury) now have single data table page instead of duplicates

---

## 2025-10-02 - Session 7: Chapter 70 and Net School Spending (NSS) Analysis

### Objective
Integrate profile_DataC70 sheet data to analyze Chapter 70 state aid and Net School Spending on a per-pupil basis. Create stacked bar plots showing:
1. Chapter 70 Aid (bottom stack)
2. Required NSS minus Ch70 Aid (middle stack)
3. Actual NSS minus Required NSS (top stack)

### Implementation Details

**1. Data Integration and Normalization** (school_shared.py:169-197)
- Fixed district name mapping between sheets:
  - C70 data uses ALL CAPS with spaces (e.g., "AMHERST PELHAM")
  - Expenditure data uses Title Case with hyphens (e.g., "Amherst-Pelham")
  - Solution: Applied `.str.title()` and `.str.replace(" ", "-")` to normalize C70 district names
- Column mapping: C70 `fy` (fiscal year) → `YEAR` to match other sheets
- All districts of interest now match correctly (verified: Amherst-Pelham, Amherst, Leverett, Pelham, Shutesbury)

**2. Data Processing Functions** (school_shared.py:497-634)
- Created `prepare_district_nss_ch70()`: Single district NSS/Ch70 data on per-pupil basis
  - Joins C70 data with enrollment (In-District FTE) by year
  - Calculates three stacking components:
    - Ch70 Aid = c70aid / enrollment
    - Req NSS (adj) = max(0, rqdnss2 - c70aid) / enrollment
    - Actual NSS (adj) = (actualNSS - rqdnss2) / enrollment
  - Handles edge case: Ch70 > Required NSS (sets middle stack to 0)
- Created `prepare_aggregate_nss_ch70()`: Aggregate multiple districts
  - Sums dollar amounts across districts by year
  - Divides by total enrollment for per-pupil values
  - Used for ALPS PK-12 virtual district

**3. Plotting Functions** (nss_ch70_plots.py)
- Created `plot_nss_ch70()`: Stacked bar plot with enrollment line overlay
  - Twin axes: enrollment (left), $/pupil (right)
  - Color palette (colorblind-friendly):
    - Green (#4ade80): Chapter 70 Aid
    - Amber (#fbbf24): Required NSS (adj)
    - Purple (#a78bfa): Actual NSS (adj)
  - Legend shows both stacks and enrollment line
  - Figure size: 11.8 x 7.4 inches (matches expenditure plots)
- Created `compute_nss_ylim()`: Global y-axis limit computation
  - Computes max total (sum of all stacks) across all pivots
  - Adds 5% padding and rounds to nearest $500

**4. Table Generation Functions** (nss_ch70_plots.py:191-291)
- Created `build_nss_category_data()`: Table data with CAGR calculations
  - Returns category rows: (category, start_str, c15s, c10s, c5s, latest_str, color, latest_val)
  - Column order (chronological left-to-right): 2009, CAGR 15y, CAGR 10y, CAGR 5y, 2024
  - Computes total row: sum of all three stacks
  - Returns start_map for red/green shading comparisons (future integration)
- Helper functions:
  - `compute_cagr_last()`: CAGR calculation over N years
  - `fmt_pct()`: Percentage formatting with em-dash for NaN values

**5. Main Generation Script** (nss_ch70_main.py)
- Generates NSS/Ch70 plots for all districts of interest:
  - ALPS PK-12 (aggregate)
  - Amherst-Pelham
  - Amherst
  - Leverett
  - Pelham
  - Shutesbury
- Computes shared y-axis limits for consistent comparison
- Outputs PNG files to output/ directory
- Preview table data for verification

### Files Created/Modified
**Created:**
- `nss_ch70_plots.py` - Plotting and table generation functions (292 lines)
- `nss_ch70_main.py` - Main generation script (132 lines)
- `test_c70_data.py` - Data exploration and verification
- `test_nss_ch70.py` - Data processing tests
- `test_nss_plots.py` - Plot generation tests
- `test_nss_tables.py` - Table data generation tests

**Modified:**
- `school_shared.py` - Added C70 normalization and processing functions (138 new lines)

### Test Results
✅ All 6 NSS/Ch70 plots generated successfully
✅ District name mapping verified for all districts
✅ Table data calculations verified (totals match sum of stacks)
✅ Edge case handling confirmed (no Ch70 > Req NSS cases in our districts)
✅ CAGR calculations verified across all timeframes (5y, 10y, 15y)

### Output Files
Generated 6 PNG files in output/ directory:
- nss_ch70_ALPS_PK_12.png
- nss_ch70_Amherst_Pelham.png
- nss_ch70_Amherst.png
- nss_ch70_Leverett.png
- nss_ch70_Pelham.png
- nss_ch70_Shutesbury.png

### Data Insights (Amherst example, 2024)
- Ch70 Aid: $6,271/pupil (22% of total NSS)
- Req NSS (adj): $11,151/pupil (38% of total NSS)
- Actual NSS (adj): $11,670/pupil (40% of total NSS)
- Total NSS: $29,093/pupil
- 15-year CAGR: 4.65% (total NSS growth rate)

### PDF Integration (Completed)
✅ **NSS/Ch70 section successfully integrated into compose_pdf.py:**
1. Added NSS/Ch70 plots to PDF (6 districts: ALPS PK-12 + 5 individual districts)
2. Created `_build_nss_ch70_table()` function for funding component tables
3. Added new PDF section "Chapter 70 Aid and Net School Spending Analysis"
4. Updated Table of Contents with clickable link to NSS/Ch70 section
5. Each NSS/Ch70 page includes:
   - Stacked bar chart (Ch70 Aid, Req NSS adj, Actual NSS adj)
   - Table with funding components, CAGR values, and color swatches
   - Chronological column order (2009, CAGR 15y/10y/5y, 2024)

**PDF File Generated:**
- Output: `output/expenditures_series.pdf` (6.9MB)
- New section appears between individual district pages and Appendix A
- Section includes 6 NSS/Ch70 pages (ALPS PK-12, Amherst-Pelham, Amherst, Leverett, Pelham, Shutesbury)

### Session 7 Updates (2025-10-02 continued)

**1. Removed enrollment from NSS/Ch70 plots** (per user request)
- Removed enrollment line overlay from all NSS/Ch70 plots
- Hidden left y-axis (was used for enrollment)
- Simplified plots to focus on funding components only
- Updated nss_ch70_plots.py:92-112

**2. Reordered PDF pages - NSS/Ch70 grouped with districts**
- Moved NSS/Ch70 pages from separate section to be grouped with each district
- Each district now has 5 pages total:
  1. Simple version (solid color)
  2. Detailed version vs Western MA
  3. Detailed version vs ALPS Peers
  4. NSS/Ch70 vs Western MA (NEW position)
  5. NSS/Ch70 vs ALPS Peers (NEW position)
- NSS/Ch70 pages now appear immediately after expenditure pages for their district
- Removed separate "Chapter 70 Aid and Net School Spending Analysis" section

**3. Created NSS/Ch70 plots for aggregate districts**
- Generated Western MA (≤500) aggregate NSS/Ch70 plot
- Generated Western MA (>500) aggregate NSS/Ch70 plot
- Generated ALPS Peers aggregate NSS/Ch70 plot
- Added safe filename handling for special characters (≤, >, parentheses)
- Updated nss_ch70_main.py to include aggregate plot generation

**4. Added NSS/Ch70 aggregate pages to Appendix A**
- Added NSS/Ch70 page after each Western aggregate's expenditure pages
- Added NSS/Ch70 page after ALPS Peers aggregate's expenditure pages
- Pre-computed Western district lists (≤500 and >500) at start of build_page_dicts()
- Appendix A now shows both expenditure and NSS/Ch70 analysis for aggregates

**5. Implemented red/green shading for NSS/Ch70 tables**
- Created `_build_nss_ch70_baseline_map()` function (compose_pdf.py:551-578)
  - Builds baseline map with DOLLAR, START_DOLLAR, and CAGR (5/10/15y) values
  - Same structure as expenditure baseline maps
- Updated `_build_nss_ch70_table()` to support baseline comparison (compose_pdf.py:427-559)
  - Added baseline_map and baseline_title parameters
  - Shades start year $/pupil (col 2): relative comparison (≥2% threshold)
  - Shades latest $/pupil (col 6): relative comparison (≥2% threshold)
  - Shades CAGR columns (cols 3-5): absolute pp delta (≥2pp threshold)
  - Added legend rows with shading rules (same format as expenditure tables)
  - Red shading: higher than baseline
  - Green shading: lower than baseline
- Each district now has TWO NSS/Ch70 pages with baseline comparison:
  - One vs Western Traditional (≤500 or >500 bucket based on district size)
  - One vs ALPS Peer Districts Aggregate
- Computes baselines from aggregate NSS/Ch70 data for both Western and ALPS Peers
- Moved Western district list computation to start of build_page_dicts() for early access

**Code Changes:**
- compose_pdf.py:551-578 - Added _build_nss_ch70_baseline_map() function
- compose_pdf.py:427-559 - Updated _build_nss_ch70_table() with shading logic
- compose_pdf.py:827-838 - Pre-compute Western district lists at start
- compose_pdf.py:1005-1061 - Generate two NSS/Ch70 pages per district with baselines
- compose_pdf.py:42 - Added latest_total_fte and N_THRESHOLD to imports

**PDF Output:**
- Each district now has 2 NSS/Ch70 comparison pages (vs Western and vs ALPS Peers)
- Tables include red/green shading comparing to appropriate baseline
- Shading uses same logic as expenditure tables (2% dollar threshold, 2pp CAGR threshold)
- Aggregate NSS/Ch70 pages in Appendix A show baseline data for comparisons

**6. Added NSS/Ch70 data tables to Appendix B**
- Created `_build_nss_ch70_data_table()` function (compose_pdf.py:791-847)
  - Builds data table showing NSS/Ch70 funding components by year
  - Rows: Ch70 Aid, Req NSS (adj), Actual NSS (adj), Total NSS
  - Columns: Years (2009-2024)
  - Format: $/pupil with dollar formatting
- Updated Appendix B data collection to gather NSS/Ch70 data (compose_pdf.py:1247-1292)
  - Collects both EPP and NSS/Ch70 data for each district
  - Deduplicates by district name
  - Stores raw_nss in district_data map
- Updated data_table page rendering (compose_pdf.py:1558-1566)
  - Adds NSS/Ch70 data table section if raw_nss is available
  - Includes section header "NSS/Ch70 Funding Components ($/pupil)"
  - Table appears after EPP and FTE tables on each district's data page
- Updated page dict to store raw_nss (compose_pdf.py:1060, 1075)
  - District NSS/Ch70 pages now include raw_nss field
  - Enables data tables in Appendix B
- Updated Appendix B subtitle and note to mention NSS/Ch70 funding

**7. Added NSS/Ch70 calculation methodology to Appendix C**
- Created methodology_page3 with comprehensive NSS/Ch70 explanation (compose_pdf.py:1384-1426)
  - Background: Chapter 70 and Net School Spending definitions
  - Data Sources: c70aid, rqdnss2, actualNSS, In-District FTE
  - Per-Pupil Calculation: Division by enrollment
  - Stacked Components with formulas:
    - Ch70 Aid ($/pupil) = c70aid / In-District FTE
    - Req NSS (adj) = max(0, rqdnss2 − c70aid) / In-District FTE
    - Actual NSS (adj) = (actualNSS − rqdnss2) / In-District FTE
  - Total NSS = Sum of all three components
  - Example Calculation using Amherst FY2024 data:
    - Enrollment: 1,083 students
    - Ch70 Aid: $6,791,000 / 1,083 = $6,271/pupil
    - Req NSS (adj): ($18,859,000 − $6,791,000) / 1,083 = $11,151/pupil
    - Actual NSS (adj): ($31,511,000 − $18,859,000) / 1,083 = $11,670/pupil
    - Total NSS: $6,271 + $11,151 + $11,670 = $29,093/pupil
  - Aggregate Calculation: Sum dollar amounts across districts, divide by total enrollment
  - Shading: Same 2% dollar / 2pp CAGR thresholds as PPE tables
- Added third methodology page to pages list (compose_pdf.py:1447-1454)
- Appendix C now has 3 pages:
  1. CAGR definition and aggregate calculations
  2. Red/green shading logic
  3. NSS/Ch70 calculations with example

**PDF Output:**
- Appendix B now includes NSS/Ch70 funding data tables for all districts
- Appendix C includes complete NSS/Ch70 methodology with worked example
- Each district's Appendix B page shows: PPE table, FTE table, NSS/Ch70 table

**8. Changed NSS/Ch70 to absolute dollars instead of per-pupil**
- Per user request based on DESE Researcher's Guide guidance about out-of-district costs
- Updated school_shared.py functions:
  - `prepare_district_nss_ch70()` (lines 498-564): Removed division by enrollment
    - Now returns absolute dollar values: c70aid, max(0, rqdnss2 - c70aid), actualNSS - rqdnss2
  - `prepare_aggregate_nss_ch70()` (lines 567-629): Removed per-pupil calculation
    - Now sums dollar amounts across districts without dividing by enrollment
- Updated nss_ch70_plots.py (line 105): Changed y-axis label from "$ per pupil" to "Dollars ($)"
- Updated compose_pdf.py throughout:
  - Narrowed Component column in NSS/Ch70 data table from 1.2" to 0.9" (line 828)
  - Removed "per-pupil" references from all NSS/Ch70 page subtitles
  - Updated Appendix B note to clarify "NSS/Ch70 funding components (in absolute dollars)"
  - Updated data table label: "NSS/Ch70 Funding Components ($)" (line 1620)
- Updated Appendix C methodology (lines 1396-1431):
  - Added important note explaining absolute dollars vs per-pupil
  - Updated all formulas to show absolute dollar calculations
  - Updated example calculation for Amherst FY2024:
    - Ch70 Aid: $6,791,000 (not per-pupil)
    - Req NSS (adj): $12,068,000 (not per-pupil)
    - Actual NSS (adj): $12,652,000 (not per-pupil)
    - Total NSS: $31,511,000 (not per-pupil)
  - Updated aggregate calculation: sum dollars across districts (no division)

**9. Added PPE definition to Appendix C**
- Added new section 1 to methodology_page1 (lines 1322-1327)
- Includes DESE Researcher's Guide quote about out-of-district expenditures:
  - "The out-of-district total cannot be properly reported as a per-pupil expenditure because the cost of tuitions varies greatly depending on the reason for going out of district."
- Explains PPE is calculated using in-district FTE only
- Renumbered subsequent sections (CAGR now #2, Aggregate Calculations now #3, etc.)

**PDF Output:**
- NSS/Ch70 plots now show absolute dollar amounts on y-axis
- NSS/Ch70 tables show dollar values without per-pupil division
- Plot scales are much larger (millions of dollars instead of thousands per pupil)
- CAGR calculations remain valid (percentage growth rates independent of per-pupil vs absolute)
- Red/green shading still works correctly for comparison to baselines

**10. Additional NSS/Ch70 refinements and ALPS relocation**
- Added ALPS PK-12 explanation to Appendix C (line 1342):
  - "This simulates the four towns of the Amherst-Pelham Regional District as a PK-12 unified district to support comparison with other PK-12 unified districts."
- Fixed NSS/Ch70 table headers (lines 438, 442):
  - Changed from "$/pupil" to just "$" for absolute dollar clarity
- Narrowed Component column in NSS/Ch70 summary tables (line 496):
  - Set Component column to fixed 1.4" width (was flexible)
  - Increased dollar columns to 1.0" for larger values (millions)
- Relocated ALPS PK-12 section after Shutesbury (lines 1041-1135):
  - Moved from page 2-3 position to after all individual districts
  - New page order: Amherst-Pelham, Amherst, Leverett, Pelham, Shutesbury, ALPS PK-12
  - Maintains all 3 pages per district structure (simple, vs Western, vs Peers, NSS/Ch70)
- Added number abbreviations to Appendix B NSS/Ch70 tables (lines 805-814):
  - Format function: ≥$1M shows as "$X.XM", ≥$1K shows as "$XK"
  - Examples: $6,791,000 → $6.8M, $12,068,000 → $12.1M
  - Prevents line wrapping for large aggregate values
- Limited Appendix B NSS/Ch70 tables to 2009-2024 (lines 797-800):
  - Filters years to 2009-2024 range only
  - Reduces column count from 16 to 16 years (matches data availability)
  - More horizontal space for each year column

**PDF Output:**
- NSS/Ch70 summary tables now accommodate $10M+ values without overflow
- Appendix B NSS/Ch70 tables show abbreviated values (6.8M, 12.1M)
- Table year range constrained to 2009-2024 for optimal layout
- ALPS PK-12 now appears as final district section before Appendix A
- Table of Contents reflects new ALPS PK-12 position

### Future Enhancements (Optional)
1. ✅ ~~Add red/green shading to NSS/Ch70 tables~~ (COMPLETED)
2. ✅ ~~Add NSS/Ch70 data tables to Appendix B~~ (COMPLETED)
3. ✅ ~~Create comparison baseline for NSS/Ch70~~ (COMPLETED - Western MA and ALPS Peers)
4. ✅ ~~Convert NSS/Ch70 to absolute dollars~~ (COMPLETED - per DESE guidance)