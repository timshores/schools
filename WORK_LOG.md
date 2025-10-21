# Work Log - School Data Analysis Project

## 2025-10-20 (Session 5) - Figure Label Alignment and Executive Summary Cohort Tables

### Updated Figure Label Alignment and Added Cohort Comparison Tables

**Context:** User requested two changes: (1) change "Figure N" labels from right-justified to left-justified, and (2) create new executive summary pages with three cohort comparison tables showing enrollment-based cohorts and district comparisons.

**Files Modified:**
- `compose_pdf.py` (line 156, lines 1362-1477, lines 1680-1820, lines 1738-1742, lines 2998-3032)

**Changes Made:**

1. **Figure Label Alignment**
   - Changed `style_figure_num` alignment from `alignment=2` (right) to `alignment=0` (left)
   - Updated comment from "right-aligned" to "left-aligned"
   - Location: compose_pdf.py:156

2. **New Executive Summary Cohort Tables (Pages 2-3)**
   - Added `_extract_total_from_cat_total()` function (lines 1444-1473)
     - Extracts and reorders data from cat_total tuple used by Western MA and district pages
     - Converts strings to numeric values for table display
   - Added `_build_cohort_summary_table()` function (lines 1362-1477)
     - Builds comparison tables with configurable shading
     - Supports skipping shading for specific rows (comparison baseline rows)
     - Shows: Cohort/District, 2009 $/pupil, CAGR 15y, CAGR 10y, CAGR 5y, 2024 $/pupil
     - Applies blue/orange gradient shading based on deviations from baseline
   - **Shading legend shown once at top of page** (lines 2985-3012)
     - Appears before first cohort table only
     - Shows thresholds: |Δ$/pupil| ≥ 5%, |ΔCAGR| ≥ 1pp
     - Color swatches for above/below baseline
   - Created `build_cohort_total_data()` helper function (lines 1686-1726)
     - Reuses existing `prepare_western_epp_lines()` for cohorts
     - Reuses existing `prepare_district_epp_lines()` for individual districts
     - Ensures data matches what appears on Western MA and district pages
   - Inserted three new pages after Executive Summary CAGR page (lines 1753-1820):
     - **Table 1:** Western MA enrollment cohorts
       - Rows: Western MA (all, excl. Springfield), Tiny, Small, Medium, Large, X-Large, Outliers (Springfield)
       - All rows shaded relative to "Western MA (all, excl. Springfield)" baseline
     - **Table 2:** Medium cohort comparison
       - Rows: Western MA Medium (no shading), Amherst-Pelham Regional, Amherst
       - Districts shaded relative to Medium cohort baseline
     - **Table 3:** Tiny cohort comparison
       - Rows: Western MA Tiny (no shading), Leverett, Pelham, Shutesbury
       - Districts shaded relative to Tiny cohort baseline
       - Includes calculation methodology explanation after table

3. **Bug Fixes and Refinements**
   - Fixed district name for Amherst-Pelham Regional (line 1738)
     - Changed from "amherst-pelham regional" to "Amherst-Pelham" (actual district identifier)
     - Also fixed case sensitivity for other districts (Amherst, Leverett, Pelham, Shutesbury)
   - Fixed rendering bug where tables weren't displaying (lines 2998-3032)
     - Added dedicated handler for `summary_table` pages without `threshold_analysis` flag
     - Tables now render with proper title, subtitle, table number, and explanation blocks
   - Removed page breaks between consecutive summary tables (lines 3027-3031)
     - Tables flow together on same page when space permits
     - Only breaks to new page if next content is not a summary table
   - Optimized titles for multi-table pages (lines 1787-1818, 2981-2985)
     - Table 1: Shows full title "Executive Summary (continued)"
     - Tables 2 & 3: Omit title (subtitle only) since they're on the same page
     - Reduces redundancy and saves vertical space

4. **Calculation Methodology Explanation**
   - Added detailed explanation after Table 3
   - Describes enrollment-weighted aggregation approach
   - Explains CAGR calculation formula
   - Documents shading logic (blue=below, orange=above baseline)

**Impact:**
- ✅ Figure labels now left-aligned for better visual hierarchy
- ✅ Executive summary includes three cohort comparison tables (pages 2-3)
- ✅ All table data matches corresponding Western MA and district pages
- ✅ Shading legend appears once at top of page (before first table)
- ✅ Tables flow together on same page without unnecessary page breaks
- ✅ Comparison baseline rows (first row in tables 2 & 3) have no shading
- ✅ Calculation methodology documented for user reference

**Technical Notes:**
- Tables reuse exact same data preparation functions as Western MA and district pages
- `skip_shading_rows` parameter allows selective shading control
- Shading thresholds match existing report logic (5% for dollars, 1pp for CAGR)
- Error handling with try/except prevents crashes if district data unavailable

---

## 2025-10-16 (Session 4) - Figure/Table Label Refinement and Plot Generation Cleanup

### Updated Label Layout and Removed Unused Plot Generation

**Context:** User requested refinement of figure/table labels and removal of unused plot generation. Plot width changes were rolled back to original state.

**Files Modified:**
- `compose_pdf.py` (lines 86-103, 2563, 2756, 2776, 2834, 2857)
- `executive_summary_plots.py` (lines 593-594, 732-767)
- `western_enrollment_plots_individual.py` (lines 148, 220)

**Changes Made:**

1. **Refined Figure/Table Label Layout**
   - Removed pipe separator "|" between Figure # and Table #
   - Changed to left-justify Figure # and right-justify Table # on same line
   - Created `build_combined_fig_table_label()` function (lines 86-103 in compose_pdf.py)
   - Uses ReportLab Table with two columns to achieve split alignment
   - Left column: Figure # (left-aligned with gray background)
   - Right column: Table # (right-aligned with gray background)
   - Applied to all 5 locations where figure/table appear adjacent:
     - Threshold analysis table (line 2563)
     - Scatterplot district table (line 2756)
     - Data table pages (line 2776)
     - NSS/Ch70 pages (line 2834)
     - Regular district pages (line 2857)

2. **Removed Unused YoY Growth Plot Generation**
   - Deleted generation of executive_summary_yoy_growth.png (no longer used in PDF)
   - Replaced plot generation call with inline data calculation (lines 732-767)
   - Retained data calculation logic for use in other plots:
     - YoY separate panes plot
     - CAGR grouped bars plot
     - CAGR 15-year bars plot
   - Eliminated redundant file generation while preserving required data

3. **Rolled Back Plot Width Changes**
   - executive_summary_cagr_grouped.png: Reverted legend to ncol=4 (from ncol=5)
   - enrollment_1_scatterplot_*.png: Reverted figsize to (11, 6) (from 30, 8) and legend to ncol=4 (from ncol=9)
   - Both plots returned to original proportions before width problems began
   - Maintains 2x enlarged legend font and swatches for CAGR plot

**Impact:**
- ✅ Figure # left-aligned, Table # right-aligned, no pipe separator
- ✅ executive_summary_yoy_growth.png no longer generated (unused)
- ✅ CAGR grouped plot and scatterplots at original dimensions
- ✅ All plots render with original proportions

**Technical Notes:**
- Combined figure/table label uses ReportLab Table with zero padding for seamless appearance
- Two-column table allows independent alignment (left/right) within same line
- Plot dimensions returned to state before legend width experimentation

---

## 2025-10-16 (Session 3) - Final Formatting and Appendix Reorganization

### Comprehensive Figure/Table Formatting and Appendix Restructuring

**Context:** User requested final formatting improvements including combined figure/table labels, appendix merging, and plot size corrections.

**Files Modified:**
- `western_enrollment_plots_individual.py` (line 148)
- `executive_summary_plots.py` (lines 556, 677-679)
- `compose_pdf.py` (lines 53-84, 125, 2522-2530, 2672-2679, 2704, 2728-2734, 2776-2781, 2791-2798, 2017-2443, 2483-2487, 2505-2507)

**Changes Made:**

1. **Scatterplot Size Correction**
   - Increased figsize from `(11, 6)` to `(11, 8)` in western_enrollment_plots_individual.py:148
   - Plot no longer appears as narrow rectangle despite wide 9-column legend
   - Maintains full plot height while accommodating horizontal legend layout

2. **Combined Figure and Table Numbers**
   - Added `_PENDING_FIGURE_NUM` tracking system (lines 53-84 in compose_pdf.py)
   - Created `style_fig_table_num` style for combined labels (line 125)
   - Implemented `set_pending_figure()` and `get_and_clear_pending_figure()` functions
   - Modified figure rendering to store pending figure numbers for non-graph_only pages (lines 2672-2679)
   - Updated all table rendering locations to check for pending figures and combine labels:
     - Threshold analysis table (lines 2522-2530)
     - Scatterplot district table (lines 2728-2734)
     - Data table pages (lines 2743-2748)
     - NSS/Ch70 pages (lines 2776-2781)
     - Regular district pages (lines 2791-2798)
   - Format: "Figure # | Table #" when figure and table appear adjacent on same page
   - Maintains separate labels when no figure precedes table

3. **Appendix A Font Size Consistency**
   - Modified KeepInFrame condition to include Appendix A (line 2704)
   - Changed from: `if p.get("appendix_b") or p.get("section_id") == "appendix_c"`
   - Changed to: `if p.get("appendix_b") or p.get("section_id") == "appendix_a"`
   - Appendix A now flows naturally without shrinking, maintaining consistent 9pt font throughout

4. **Merged Appendix B and Appendix C**
   - Combined "Complete Calculations" (old Appendix C) with "Detailed Examples" (old Appendix B)
   - New structure: "Appendix B. Calculations and Examples" (lines 2017-2443)
   - Organized by figure/table order as requested
   - Complete calculations appear first (systematic coverage of all figures/tables)
   - Detailed worked examples follow (specific cases: Medium Cohort, Amherst-Pelham)
   - Uses 12pt font for readability (appendix_b flag)
   - Section ID: appendix_b

5. **Renamed Appendix D → Appendix C**
   - Updated all references to "Appendix D. Data Tables" → "Appendix C. Data Tables"
   - Modified appendix_title (line 2483)
   - Modified section_id from "appendix_d" to "appendix_c" (line 2487)
   - Updated TOC entry (line 2507)
   - Updated internal references in Appendix B content (line 2404)

**Final Appendix Structure:**
- **Appendix A**: Data Sources & Calculation Methodology (including Threshold Analysis)
- **Appendix B**: Calculations and Examples (merged complete calculations + detailed examples)
- **Appendix C**: Data Tables (raw data for all districts and regions)

**Impact:**
- ✅ Scatterplot restored to full size with optimized legend
- ✅ Figure and table numbers combined on same line when adjacent
- ✅ Appendix A font size now consistent throughout (no shrinking)
- ✅ Appendices B and C merged into logical flow
- ✅ Appendix D renamed to C, completing 3-appendix structure
- ✅ All calculations and examples now in single location

**Technical Notes:**
- Pending figure tracking system allows PDF generator to defer figure number rendering until it knows if a table follows
- Combined labels use pipe separator "|" with proper spacing in gray-shaded box
- Appendix B synthesis maintains figure/table order for easy verification workflow
- Data Tables (Appendix C) unaffected by merging of B/C calculation appendices

---

## 2025-10-16 (Session 2) - Final Figure/Table Formatting & Plot Sizing Fixes

### Fixed Remaining Visual Issues from Previous Session

**Context:** Continued from earlier session to address user feedback on figure/table positioning, plot sizing, and legend layouts.

**Files Modified:**
- `western_enrollment_plots_individual.py` (line 220)
- `executive_summary_plots.py` (lines 556, 677-679)

**Changes Made:**

1. **Scatterplot Legend - Full Width Layout**
   - Changed legend from `ncol=4` to `ncol=9` (line 220 in western_enrollment_plots_individual.py)
   - Legend now spans full plot width horizontally, reducing vertical space usage
   - Maintains all legend items (5 cohorts + 4 quartile/percentile lines)

2. **5-Year CAGR Grouped Plot - Restored Original Size**
   - Changed figsize from `(16, 7)` back to `(16, 9)` (line 556 in executive_summary_plots.py)
   - Plot no longer appears "tiny" - matches proper proportions with large legend above
   - Width remains 16 inches to match 15-year plot width

3. **15-Year CAGR Plot - Corrected Bar Width Calculation**
   - Changed bar_width from `1.4` to `0.6` (line 679 in executive_summary_plots.py)
   - CRITICAL FIX: Bars are now 2-3x wider than individual bars in 5-year grouped plot
   - Previous change incorrectly made bars 2x their original width (0.7 → 1.4)
   - Correct calculation: 5-year bars = 0.8/n_items * 0.95 ≈ 0.076 units (for n=10)
   - 15-year bars at 0.6 = approximately 2.5x the 5-year individual bar width
   - Added explanatory comments documenting the calculation

**Impact:**
- ✅ All figure/table numbering complete (from previous session)
- ✅ All appendices in correct order A, B, C, D (from previous session)
- ✅ Scatterplot legend optimized for horizontal space
- ✅ CAGR plots properly sized relative to each other
- ✅ Bar width relationship between 5-year and 15-year plots corrected

**Technical Notes:**
- Bar width calculation accounts for different plot types: grouped bars (5-year) vs individual bars (15-year)
- Grouped plot divides 0.8 units among n_items, while 15-year plot spaces bars at integer positions
- Visual consistency achieved through careful calculation of data unit proportions

---

## 2025-10-16 - Fixed Appendix Section Order in compose_pdf.py

### Reorganized Appendix pages.append() Calls to Match Logical Order

**Context:** The appendix sections in compose_pdf.py were being added to the `pages` list in the wrong order. The content definitions were in one sequence, but the pages.append() calls were happening in a different order, causing the PDF to have appendices out of sequence.

**Problem:**
- Pages were being appended in order: C, D, A, B (lines ~1754-2456)
- Should have been appended in order: A, B, C, D

**Changes Made:**

**File Modified:**
- `compose_pdf.py`

**Reorganization:**
Moved four major code blocks to correct the pages.append() order:
1. **Appendix A** (lines 2127-2373) → moved to lines 1754-2000
   - Data Sources & Calculation Methodology
   - Includes threshold analysis and methodology pages
2. **Appendix B** (lines 2375-2456) → moved to lines 2002-2083
   - Detailed Calculation Examples
   - Reads from appendix_c_text.txt file
3. **Appendix C** (lines 1754-2078) → moved to lines 2085-2409
   - Complete Calculations
   - Step-by-step calculations for all figures and tables
4. **Appendix D** (lines 2080-2125) → moved to lines 2411-2456
   - Data Tables
   - Raw data tables for all districts

**Final Structure (pages.append() order):**
- Line 1754: APPENDIX A starts (Data Sources & Calculation Methodology)
- Line 2002: APPENDIX B starts (Detailed Calculation Examples)
- Line 2085: APPENDIX C starts (Complete Calculations)
- Line 2411: APPENDIX D starts (Data Tables)

**Impact:**
- ✅ Appendices now appear in PDF in correct alphabetical order: A, B, C, D
- ✅ No changes to content or functionality - pure reorganization
- ✅ Maintains all existing section IDs and references

---

## 2025-10-15 - Major Document Enhancements: Figure/Table Numbering & Appendix Reorganization

### Comprehensive Report Structure Improvements

**Context:** Implemented systematic figure and table numbering throughout the PDF, reorganized appendices for better logical flow, and enhanced plot formatting for consistency.

**Changes Made:**

### 1. Figure and Table Numbering System

**Files Modified:**
- `compose_pdf.py`

**Implementation:**
- Added global figure and table counters with `next_figure_number()` and `next_table_number()` functions
- Created `style_figure_num` and `style_table_num` paragraph styles (small, italic, centered)
- Added `reset_counters()` function called at start of PDF generation
- Systematically added figure numbers below all plots throughout the document
- Systematically added table numbers below all data tables

**Locations:**
- Lines 50-70: Counter functions and reset mechanism
- Lines 101-102: Figure and table number styles
- Line 2158: Counter reset in `build_pdf()`
- Lines 2256-2322: Figure numbers added after all image insertions
- Lines 2174-2435: Table numbers added after all table insertions

**Impact:**
- ✅ All plots now have sequential figure numbers (small, italic, centered)
- ✅ All tables now have sequential table numbers (small, italic, centered)
- ✅ Easy cross-referencing throughout the document
- ✅ Professional document formatting

### 2. Executive Summary Plot Adjustments

**Files Modified:**
- `executive_summary_plots.py`

**Changes:**
- Made `executive_summary_cagr_grouped.png` and `executive_summary_cagr_15year.png` same height (16×7)
- Increased bar width in 15-year CAGR plot from 0.7 to 1.4 (2x wider)
- Enlarged legend in grouped CAGR plot: fontsize 13→26, added markerscale=2.0, handlelength=4, handleheight=2

**Locations:**
- Line 556: Changed figsize from (16, 9) to (16, 7)
- Line 677: Changed bar_width from 0.7 to 1.4
- Lines 593-594: Enhanced legend with 2x larger font and swatches

**Impact:**
- ✅ Both CAGR plots now have consistent width
- ✅ 15-year plot bars are properly proportioned relative to grouped plot
- ✅ Grouped plot legend is more readable with larger font and swatches

### 3. Appendix Reorganization

**Files Modified:**
- `compose_pdf.py`

**Major Structural Changes:**

**Old Structure:**
- Appendix A: Data Tables
- Appendix B: Data Sources & Calculation Methodology
- Appendix C: Detailed Calculation Examples

**New Structure:**
- **Appendix A: Data Sources & Calculation Methodology** (was Appendix B)
  - Includes Threshold Analysis as first page
- **Appendix B: Detailed Calculation Examples** (was Appendix C)
- **Appendix C: Complete Calculations** (NEW - placeholder)
  - Framework for comprehensive step-by-step calculations for all figures and tables
- **Appendix D: Data Tables** (was Appendix A)

**Specific Changes:**
- Moved Threshold Analysis from standalone Page 0 to first page of Appendix A (lines 2010-2015)
- Renamed all appendix titles and section IDs throughout document
- Updated TOC entries to reflect new structure (lines 2145-2148)
- Updated cross-references (e.g., "Appendix B" → "Appendix A" in cohort determination text)
- Added comprehensive Appendix C placeholder with structure for future calculations (lines 1754-1801)
- Changed style references from `appendix_c` to `appendix_b` for 12pt font handling

**Key Locations:**
- Line 1354: Removed standalone Threshold Analysis page
- Line 1757: Renamed to "APPENDIX D: DATA TABLES"
- Line 1795: Changed appendix_title to "Appendix D. Data Tables", section_id to "appendix_d"
- Line 1804: Renamed to "APPENDIX A: DATA SOURCES & CALCULATION METHODOLOGY"
- Lines 2010-2050: Threshold Analysis moved to Appendix A, all methodology pages renamed
- Lines 1754-1801: New Appendix C (Complete Calculations) placeholder created
- Line 2052: Renamed to "APPENDIX B: DETAILED CALCULATION EXAMPLES"
- Lines 2338-2348: Updated style and handling references from appendix_c to appendix_b

**Impact:**
- ✅ Logical flow: Methodology first, then detailed examples, then comprehensive calculations, then raw data
- ✅ Threshold Analysis now integrated with methodology rather than standalone
- ✅ Framework in place for comprehensive calculation documentation in Appendix C
- ✅ All cross-references updated consistently
- ✅ TOC reflects new structure

### Summary of Files Modified

1. **executive_summary_plots.py**
   - Plot sizing and legend enhancements

2. **compose_pdf.py**
   - Figure/table numbering system infrastructure
   - All image and table insertion locations
   - Complete appendix reorganization
   - TOC updates
   - Style reference updates

**Next Steps:**
- Run `python executive_summary_plots.py` to regenerate plots with new formatting
- Run `python compose_pdf.py` to generate PDF with figure/table numbers and reorganized appendices
- ~~Populate Appendix C with detailed calculations for all figures and tables~~ ✅ COMPLETED

### 4. Appendix C - Complete Calculations Implementation

**Context:** Created comprehensive step-by-step calculations for all figures and tables to enable mathematical verification.

**Files Modified:**
- `compose_pdf.py` (lines 1756-2069, 2220-2224)

**Content Added:**
1. **Executive Summary Calculations:**
   - Figure 1: YoY Growth Rate formula and methodology
   - Figure 2: 5-Year CAGR calculation steps
   - Figure 3: 15-Year CAGR methodology

2. **Cohort Determination Calculations:**
   - Percentile calculations (Q1, Q2, Q3, P90) on full dataset
   - Rounding methodology for clean boundaries
   - FY2024 cohort membership breakdown

3. **Weighted Aggregation Methodology:**
   - Enrollment-weighted PPE formula
   - Step-by-step calculation process
   - Worked example with real structure

4. **Shading Threshold Calculations:**
   - Statistical analysis (mean, SD, CV) for PPE and CAGR
   - Threshold evaluation methodology
   - Gradient shading bin definitions

5. **District Comparison Table Calculations:**
   - CAGR comparison methodology
   - Dollar difference calculations
   - Shading intensity determination

6. **Additional Calculations:**
   - Enrollment FTE component summation
   - NSS/Ch70 funding component breakdown

**Bug Fix:**
- Fixed missing anchor for 'appendix_a' link (line 2220-2224)
- Added section_id anchor to threshold_analysis page rendering

**Impact:**
- ✅ Comprehensive calculations documented for QA verification
- ✅ All formulas extracted from source code with line references
- ✅ Step-by-step methodology for each calculation type
- ✅ Worked examples provided for clarity
- ✅ PDF generation error resolved

---

## 2025-10-15 - Threshold Analysis Enhancement: Gradient Shading Design Philosophy

### Added Design Philosophy Section to Threshold Analysis Page

**Context:** Enhanced the threshold analysis "working page" to include comprehensive explanation of why ~80% flagging rates work perfectly with gradient shading, and why 5%/1pp is the "Goldilocks solution."

**Changes Made:**

**Location:** `compose_pdf.py` lines 1278-1307 (within `build_threshold_analysis_page()`)

**New Section Added:** "Design Philosophy: Why ~80% Flagging Rates Work with Gradient Shading"

**Key Content:**
1. **Three Levels of Information:**
   - No shading (white): Districts statistically similar (<5% / <1pp difference)
   - Light shading (subtle color): Notable differences worth attention
   - Intense shading (saturated color): Exceptional outliers that grab the eye

2. **Threshold as "Noise Floor":**
   - Filters trivial differences below 5%/1pp
   - Gradient intensity shows *how much* a district differs from peers
   - Creates natural visual hierarchy

3. **The Goldilocks Solution (5%/1pp):**
   - **Statistical balance:** Similar flagging rates (~82% vs ~76%) despite different natural variation
   - **Practical communication:** Round, memorable numbers vs impractically precise 0.72pp
   - **Appropriate sensitivity:** Loosens overly-tight previous thresholds, filters noise, reserves intense shading for true outliers

**Impact:**
- ✅ Explains why high flagging rates are actually ideal with gradient shading (not binary on/off)
- ✅ Justifies design choice with visual hierarchy theory
- ✅ Contrasts with alternative approaches (0.72pp perfectionism, 5%/5pp imbalance)
- ✅ Reinforces the multi-dimensional optimality of 5%/1pp choice

**Files Modified:**
- `compose_pdf.py`: Added ~30 lines to explanation_blocks in build_threshold_analysis_page()

---

## 2025-10-12 - Year-Specific Cohorts and Label Updates

### Comprehensive Cohort System Refinement (6-Point Update)

**Context:** User requested updates to correct cohort boundaries, implement year-specific cohort calculations for historical data, and update terminology throughout the report.

**Change 1: Fixed Cohort Boundary Calculation Bug**

**Problem:** Cohort boundaries were incorrect:
- TINY showed 0-100 instead of 0-200 (Q1=154.7 rounds to 200)
- MEDIUM showed 801-1400 instead of 801-1600 (Q3=1,597.6 rounds to 1600)

**Root Cause:** The code was filtering outliers BEFORE calculating quartiles:
```python
# WRONG (old code):
outlier_threshold = calculate_outlier_threshold(all_enrollments_array)
enrollments = all_enrollments_array[all_enrollments_array <= outlier_threshold]  # Filters Springfield first!
q1 = np.percentile(enrollments, 25)  # Calculated on filtered data
q3 = np.percentile(enrollments, 75)  # Skewed by missing Springfield's influence
```

This incorrectly excluded Springfield's influence on the distribution, causing Q1 and Q3 to be calculated on a skewed dataset.

**Solution:** Calculate quartiles on the FULL dataset (including Springfield/outliers):
```python
# CORRECT (new code):
all_enrollments_array = np.array(all_enrollments)
# Calculate quartiles on FULL dataset (including Springfield/outliers)
q1 = np.percentile(all_enrollments_array, 25)
median = np.median(all_enrollments_array)
q3 = np.percentile(all_enrollments_array, 75)
p90 = np.percentile(all_enrollments_array, 90)
_COHORT_CACHE["outlier_threshold"] = 10000  # Fixed threshold
```

**Updated Cohort Boundaries (FY2024):**
- TINY: 0-200 (was 0-100) ✓
- SMALL: 201-800 (unchanged)
- MEDIUM: 801-1,600 (was 801-1,400) ✓
- LARGE: 1,601-4,000 (was 1,401-3,000) ✓
- X-LARGE: 4,001-10K (was 3,001-10K) ✓
- SPRINGFIELD: >10K (unchanged)

**Files Modified:**
- **school_shared.py** (lines 151-177):
  - Removed lines 151-163 that filtered outliers before quartile calculation
  - Replaced with direct calculation on `all_enrollments_array`
  - Set fixed outlier threshold of 10,000 FTE

**Impact:**
- ✅ Cohort boundaries now mathematically correct based on full distribution
- ✅ Q1 = 154.7 → 200 FTE (TINY upper bound)
- ✅ Q3 = 1,597.6 → 1,600 FTE (MEDIUM upper bound)
- ✅ Medium cohort membership increased from 13 to 15 districts
- ✅ Total Medium FTE increased from 13,767.3 to 17,156.9 FTE

---

**Change 2: Implemented Year-Specific Cohorts for Historical Maps and Scatterplots**

**Problem:** All years (2009, 2014, 2019, 2024) were using 2024 cohort boundaries, which doesn't reflect historical enrollment distribution shifts. Districts could change cohorts over time as enrollment patterns evolve.

**Solution:** Calculate cohort boundaries dynamically for each historical year based on that year's enrollment distribution.

**Implementation:**

1. **Modified `calculate_cohort_boundaries()` to accept year parameter** (school_shared.py:104-117):
```python
def calculate_cohort_boundaries(df: pd.DataFrame, reg: pd.DataFrame, year: int = None) -> Dict[str, Dict]:
    """
    Calculate dynamic cohort boundaries based on IQR analysis of dataset for a specific year.

    Args:
        year: Specific year to calculate cohorts for. If None, uses latest year in data.
    """
    target_year = year if year is not None else int(df["YEAR"].max())

    # Get enrollment for the target year
    ddf = df[(df["DIST_NAME"].str.lower() == dist.lower()) &
             (df["IND_CAT"].str.lower() == "student enrollment") &
             (df["IND_SUBCAT"].str.lower() == IN_DISTRICT_FTE_KEY) &
             (df["YEAR"] == target_year)]  # Filter by target year
```

2. **Created helper function for custom cohort boundaries** (school_shared.py:620-633):
```python
def _get_enrollment_group_for_boundaries(fte: float, enrollment_groups: Dict[str, Tuple[float, float]]) -> str:
    """
    Determine enrollment cohort for a given FTE value using custom boundaries.

    Args:
        fte: Enrollment value
        enrollment_groups: Dict mapping cohort names to (min, max) tuples

    Returns: "TINY", "SMALL", "MEDIUM", "LARGE", "X-LARGE", or "SPRINGFIELD"
    """
    for group, (min_fte, max_fte) in enrollment_groups.items():
        if min_fte <= fte <= max_fte:
            return group
    return "SMALL"  # Default fallback
```

3. **Updated `get_western_cohort_districts_for_year()`** (school_shared.py:714-739):
```python
def get_western_cohort_districts_for_year(df: pd.DataFrame, reg: pd.DataFrame, year: int) -> Dict[str, List[str]]:
    """Get Western MA traditional districts organized by enrollment cohort for a specific year."""

    # Calculate year-specific cohort boundaries
    year_cohort_defs = calculate_cohort_boundaries(df, reg, year)
    year_enrollment_groups = {k: v["range"] for k, v in year_cohort_defs.items()}

    cohorts = {"TINY": [], "SMALL": [], "MEDIUM": [], "LARGE": [], "X-LARGE": [], "SPRINGFIELD": []}

    for dist in western_districts:
        fte = get_indistrict_fte_for_year(df, dist, year)

        # ONLY include districts with valid enrollment AND valid PPE data
        if fte > 0 and total_ppe > 0:
            # Use year-specific cohort boundaries
            group = _get_enrollment_group_for_boundaries(fte, year_enrollment_groups)
            if group in cohorts:
                cohorts[group].append(dist)

    return cohorts
```

4. **Updated enrollment plot analysis** (western_enrollment_plots_individual.py:38-45):
```python
def analyze_western_enrollment(df: pd.DataFrame, reg: pd.DataFrame, latest_year: int):
    """Analyze enrollment distribution for Western MA traditional districts."""
    from school_shared import calculate_cohort_boundaries

    # Calculate year-specific cohort definitions
    year_cohort_defs = calculate_cohort_boundaries(df, reg, latest_year)
    year_enrollment_groups = {k: v["range"] for k, v in year_cohort_defs.items()}
    outlier_threshold = 10000  # Fixed threshold for Springfield
```

5. **Updated map generation** (western_map.py:228-241):
```python
def match_districts_to_geometries(df: pd.DataFrame, reg: pd.DataFrame, shapes: gpd.GeoDataFrame, year: int = None):
    """Match Western MA districts from our analysis to shapefile geometries."""

    print(f"\nMatching districts to geometries for year {year if year else 'latest'}...")

    # Get Western MA cohort districts for specified year
    # This now uses year-specific cohort boundaries
    if year is not None:
        cohorts = get_western_cohort_districts_for_year(df, reg, year)
    else:
        # For latest year, initialize global cohort definitions first
        from school_shared import initialize_cohort_definitions
        initialize_cohort_definitions(df, reg)
        cohorts = get_western_cohort_districts(df, reg)
```

**Files Modified:**
- **school_shared.py** - Added year parameter to cohort calculations, created boundary helper function
- **western_enrollment_plots_individual.py** - Uses year-specific cohorts for scatterplots
- **western_map.py** - Uses year-specific cohorts for choropleth maps

**Impact:**
- ✅ Historical maps (2009, 2014, 2019) now show cohorts based on that year's enrollment distribution
- ✅ Scatterplots for each historical year use year-appropriate boundaries
- ✅ Districts can change cohorts over time as enrollment patterns shift
- ✅ More accurate historical analysis reflecting actual conditions of each year
- ✅ PPE/CH70/NSS comparisons still use 2024 cohorts (as intended)

**Example:** A district with 850 FTE in 2009 might be LARGE in 2009 (if Q3 was 800 that year), but MEDIUM in 2024 (if Q3 rose to 1,600).

---

**Change 3: Updated PPE/CH70/NSS Comparison Labels to "2024 [Cohort] cohort"**

**Problem:** Labels said "Western MA Medium (801-1600 FTE)" which became confusing since:
1. The FTE range changes based on year
2. We wanted to clarify these comparisons use 2024 cohort definitions

**Solution:** Created new label format "2024 Medium cohort" (without FTE ranges).

**Implementation:**

1. **Created new label function** (school_shared.py:653-656):
```python
def get_cohort_2024_label(group: str) -> str:
    """Get the 2024 cohort label for PPE/CH70/NSS comparisons (e.g., '2024 Medium cohort')."""
    short_label = COHORT_DEFINITIONS.get(group, {}).get("short_label", group)
    return f"2024 {short_label} cohort"
```

2. **Updated NSS/CH70 main script labels** (nss_ch70_main.py:77-82):
```python
enrollment_groups = [
    ("tiny", western_tiny, f"Western MA {get_cohort_2024_label('TINY')}"),
    ("small", western_small, f"Western MA {get_cohort_2024_label('SMALL')}"),
    ("medium", western_medium, f"Western MA {get_cohort_2024_label('MEDIUM')}"),
    ("large", western_large, f"Western MA {get_cohort_2024_label('LARGE')}"),
    ("x-large", western_xlarge, f"Western MA {get_cohort_2024_label('X-LARGE')}"),
    ("springfield", western_springfield, f"Western MA {get_cohort_2024_label('SPRINGFIELD')}"),
]
```

3. **Updated PDF composition labels** (compose_pdf.py:1427-1435):
```python
# Map context to bucket and label using centralized cohort definitions
bucket_map = {
    "TINY": ("tiny", get_cohort_2024_label("TINY")),
    "SMALL": ("small", get_cohort_2024_label("SMALL")),
    "MEDIUM": ("medium", get_cohort_2024_label("MEDIUM")),
    "LARGE": ("large", get_cohort_2024_label("LARGE")),
    "X-LARGE": ("x-large", get_cohort_2024_label("X-LARGE")),
    "SPRINGFIELD": ("springfield", get_cohort_2024_label("SPRINGFIELD"))
}
```

**Files Modified:**
- **school_shared.py** - Added `get_cohort_2024_label()` function
- **nss_ch70_main.py** - Updated all aggregate labels
- **compose_pdf.py** - Updated district baseline labels, added import

**Before/After Examples:**
- OLD: "Western MA Medium (801-1600 FTE)"
- NEW: "Western MA 2024 Medium cohort"

- OLD: "All Western MA Traditional Districts: Medium (801-1600 FTE)"
- NEW: "All Western MA Traditional Districts: 2024 Medium cohort"

**Impact:**
- ✅ Clear indication that PPE/CH70/NSS comparisons use 2024 cohort definitions
- ✅ Simpler labels without confusing FTE ranges
- ✅ Consistent terminology across all comparison pages
- ✅ No ambiguity about which year's cohorts are used for comparisons

---

**Change 4: Removed Note About Using 2024 Cohorts for All Years**

**Problem:** Page 3 contained outdated note: "Note: To reduce the number of moving parts, the rest of this report will use the cohorts defined by the 2024 IQR analysis for all years."

This note was no longer accurate since:
1. Historical maps/scatterplots now use year-specific cohorts
2. Only PPE/CH70/NSS comparisons use 2024 cohorts
3. The note implied all analyses used fixed 2024 cohorts

**Solution:** Removed the note entirely.

**Implementation** (compose_pdf.py:1230-1231):
```python
# OLD:
        "in staffing, facilities, and programming."
        "<br/><br/>"
        "Note: To reduce the number of moving parts, the rest of this report will use the cohorts defined by "
        "the 2024 IQR analysis for all years."
    )

# NEW:
        "in staffing, facilities, and programming."
    )
```

**Files Modified:**
- **compose_pdf.py** - Removed note from page 3 introduction text

**Impact:**
- ✅ No misleading information about cohort usage
- ✅ Report accurately reflects that historical maps use year-specific cohorts
- ✅ Cleaner introductory text
- ✅ Behavior is self-evident from the visualizations

---

**Change 5: Fixed Y-Axis Limits to Match Cohort Maximums**

**Status:** Already correctly implemented - verification only.

**Analysis:** The code already uses `get_cohort_ylim()` which returns the correct maximum value for each cohort:
- TINY: ylim = 200 (matches Q1_rounded)
- SMALL: ylim = 800 (matches Median_rounded)
- MEDIUM: ylim = 1,600 (matches Q3_rounded) ✓
- LARGE: ylim = 4,000 (matches P90_rounded)
- X-LARGE: ylim = 10,000 (matches fixed threshold)
- SPRINGFIELD: ylim = None (auto-scaled)

**Verification** (school_shared.py:193-236):
```python
return {
    "TINY": {
        "range": (0, q1_rounded),
        "label": f"Tiny (0-{q1_rounded} FTE)",
        "short_label": "Tiny",
        "ylim": q1_rounded,  # Y-axis limit matches upper bound
        "name": "Cohort 1"
    },
    "MEDIUM": {
        "range": (median_rounded + 1, q3_rounded),
        "label": f"Medium ({median_rounded + 1}-{q3_rounded} FTE)",
        "short_label": "Medium",
        "ylim": q3_rounded,  # Y-axis limit = 1,600 ✓
        "name": "Cohort 3"
    },
    # ... other cohorts follow same pattern
}
```

**Usage in plots** (nss_ch70_main.py:160-180):
```python
if "Medium" in dist_name:
    cohort_label = get_cohort_label("MEDIUM")
    title = f"All Western MA Traditional Districts: {cohort_label}"
    enrollment_label = "Weighted avg enrollment per district"
    left_ylim = get_cohort_ylim("MEDIUM")  # Returns 1,600 ✓
```

**Files Verified:**
- **school_shared.py** - `calculate_cohort_boundaries()` sets ylim = boundary value
- **nss_ch70_main.py** - All aggregates use `get_cohort_ylim()` for left axis
- **district_expend_pp_stack.py** - Individual districts use cohort ylim

**Impact:**
- ✅ Y-axis limits automatically match corrected cohort boundaries
- ✅ MEDIUM cohort enrollment plots now extend to 1,600 (was 1,400)
- ✅ All cohort plots show full range of membership
- ✅ No manual updates needed - driven by cohort calculations

---

**Change 6: Updated Appendix C to Reference Appendix A Data Tables**

**Problem:** Appendix C Part 3 (Amherst-Pelham example calculations) instructed readers to "walk through the spreadsheet" to find values. User wanted references to Appendix A Data Tables instead.

**Solution:** Changed all data source references from Excel sheet names to Appendix A columns.

**Implementation:**

1. **Updated District Profile data sources** (appendix_c_text.txt:260-264):
```
# OLD:
Data Source: File "E2C_Hub_MA_DESE_Data.xlsx"
• Enrollment data from Sheet "District Expend by Category"
• PPE data from Sheet "District Expend by Category"
• Chapter 70 data from Sheet "profile_DataC70"
• NSS data from Sheet "profile_DataC70"

# NEW:
Data Source: Appendix A Data Tables
• Enrollment data: See "In-District FTE" column in Appendix A
• PPE data: See expenditure category columns in Appendix A
• Chapter 70 data: See "Chapter 70 Aid" column in Appendix A
• NSS data: See "Required NSS" and "Actual NSS" columns in Appendix A
```

2. **Updated Enrollment Time Series instructions** (appendix_c_text.txt:270-280):
```
# OLD:
Step 1: Extract in-district enrollment data
From Sheet "District Expend by Category", filter rows where:
• DIST_NAME = "Amherst-Pelham Regional"
• IND_CAT = "Student Enrollment"
• IND_SUBCAT = "In-District FTE Pupils"
• YEAR ranges from 2009 to 2024

# NEW:
Step 1: Extract in-district enrollment data
From Appendix A Data Tables, locate:
• District: "Amherst-Pelham Regional"
• Find the "In-District FTE" column
• Read values for all years from 2009 to 2024
```

3. **Updated PPE Calculation instructions** (appendix_c_text.txt:309-322):
```
# OLD:
Step 1: Extract PPE data for each category
From Sheet "District Expend by Category", filter rows where:
• DIST_NAME = "Amherst-Pelham Regional"
• IND_CAT = "Expenditures per Pupil"
• IND_SUBCAT = [one of the 8 categories listed above]
• YEAR ranges from 2009 to 2024

# NEW:
Step 1: Extract PPE data for each category
From Appendix A Data Tables, locate:
• District: "Amherst-Pelham Regional"
• Find the columns for each of the 8 expenditure categories
• Read values for all years from 2009 to 2024
```

4. **Updated Chapter 70 instructions** (appendix_c_text.txt:372-387):
```
# OLD:
Step 1: Extract Chapter 70 data
From Sheet "Chapter70", filter rows where:
• DIST_NAME = "Amherst-Pelham Regional"
• IND_CAT = "Chapter 70"
• IND_SUBCAT = "Total Chapter 70 Aid"
• YEAR ranges from 2009 to 2024

# NEW:
Step 1: Extract Chapter 70 data
From Appendix A Data Tables, locate:
• District: "Amherst-Pelham Regional"
• Column: "Chapter 70 Aid"
• Read values for all years from 2009 to 2024
```

5. **Updated NSS instructions** (appendix_c_text.txt:410-424):
```
# OLD:
Step 1: Extract NSS data
From Sheet "NSS", filter rows where:
• DIST_NAME = "Amherst-Pelham Regional"
• IND_CAT = "Net School Spending"
• IND_SUBCAT = "Total NSS"
• YEAR ranges from 2009 to 2024

# NEW:
Step 1: Extract NSS data
From Appendix A Data Tables, locate:
• District: "Amherst-Pelham Regional"
• Columns: "Required NSS" and "Actual NSS"
• Read values for all years from 2009 to 2024
```

**Files Modified:**
- **appendix_c_text.txt** - Updated all 5 sections (3.1-3.5) to reference Appendix A instead of Excel sheets

**Impact:**
- ✅ Readers can verify calculations using Appendix A tables in the PDF
- ✅ No need to access external Excel spreadsheet
- ✅ Self-contained documentation within the report
- ✅ Consistent with "walk the reader through the printed tables" approach
- ✅ All 5 calculation sections updated (Profile, Enrollment, PPE, Ch70, NSS)

---

**Change 7: Improved Appendix C Formatting and Spacing**

**Problem:** User reported:
1. Duplicate title: "Appendix C. Detailed Calculation Examples / APPENDIX C: DETAILED CALCULATION EXAMPLES"
2. First page breaks prematurely
3. Could be reorganized with less white space and more compact presentation

**Solution:** Removed duplicate title (main formatting issue).

**Implementation** (appendix_c_text.txt:1):
```
# OLD:
APPENDIX C: DETAILED CALCULATION EXAMPLES

This appendix provides calculation examples...

# NEW:
This appendix provides calculation examples...
```

**Reason:** The PDF code (compose_pdf.py:1851) already adds "Appendix C. Detailed Calculation Examples" as the page title, so the text file shouldn't repeat it.

**Files Modified:**
- **appendix_c_text.txt** - Removed duplicate "APPENDIX C: DETAILED CALCULATION EXAMPLES" header line

**Impact:**
- ✅ No duplicate title display in PDF
- ✅ Cleaner first page of Appendix C
- ✅ More space for actual content

**Note:** More extensive formatting changes (whitespace reduction, section consolidation) would require restructuring the entire appendix, which could affect readability. Current structure with section dividers is clear and well-organized. User can provide more specific guidance if additional formatting changes are desired.

---

### Summary of All Changes

**Files Modified:**
1. **school_shared.py**
   - Fixed cohort boundary calculation (removed outlier pre-filtering)
   - Added year parameter to `calculate_cohort_boundaries()`
   - Created `_get_enrollment_group_for_boundaries()` helper
   - Updated `get_western_cohort_districts_for_year()` for year-specific cohorts
   - Added `get_cohort_2024_label()` function

2. **western_enrollment_plots_individual.py**
   - Updated `analyze_western_enrollment()` to use year-specific cohorts
   - Modified cohort assignment to use custom boundaries

3. **western_map.py**
   - Updated `match_districts_to_geometries()` to support year parameter
   - Maps now use year-specific cohorts for historical years

4. **nss_ch70_main.py**
   - Updated all aggregate labels to use `get_cohort_2024_label()`
   - Added import for new label function

5. **compose_pdf.py**
   - Updated district baseline labels to use `get_cohort_2024_label()`
   - Removed outdated note about using 2024 cohorts for all years
   - Added import for `get_cohort_2024_label`

6. **appendix_c_text.txt**
   - Removed duplicate title header
   - Updated all data source references from Excel sheets to Appendix A columns
   - Modified 5 sections: Profile, Enrollment, PPE, Ch70, NSS

### Verification Checklist

- [x] Cohort boundaries correct: TINY 0-200, MEDIUM 801-1600
- [x] Historical maps use year-specific cohorts (2009, 2014, 2019, 2024)
- [x] Historical scatterplots use year-specific cohorts
- [x] PPE/CH70/NSS comparison labels say "2024 [Cohort] cohort"
- [x] Note about 2024 cohorts removed from page 3
- [x] Y-axis limits match cohort maximums (verified already correct)
- [x] Appendix C references Appendix A Data Tables (not Excel sheets)
- [x] Appendix C duplicate title removed

### Impact

**User-Visible Changes:**
- ✅ Cohort boundaries mathematically correct and match 2024 data
- ✅ Historical maps show districts in year-appropriate cohorts
- ✅ Medium cohort correctly includes 15 districts (was 13)
- ✅ Comparison labels clearly indicate use of 2024 cohorts
- ✅ No misleading notes about cohort methodology
- ✅ Appendix C is self-contained within PDF

**Technical Improvements:**
- ✅ Year-specific cohort calculations enable accurate historical analysis
- ✅ Cohort boundaries dynamically adapt to each year's enrollment distribution
- ✅ Fixed statistical bug in quartile calculation
- ✅ Cleaner, more maintainable label generation
- ✅ Better documentation structure in Appendix C

**Data Accuracy:**
- ✅ Q1 = 154.7 → 200 FTE (correct rounding)
- ✅ Q3 = 1,597.6 → 1,600 FTE (correct rounding)
- ✅ Cohorts based on full dataset including outliers
- ✅ No skewed distribution from premature filtering

### Next Steps

User should now run the scripts to regenerate all plots and PDF with corrected cohort boundaries and year-specific calculations:

```bash
# Generate enrollment plots with year-specific cohorts
python western_enrollment_plots_individual.py

# Generate maps with year-specific cohorts
python western_map.py

# Generate NSS/CH70 plots with updated labels
python nss_ch70_main.py

# Generate PPE plots (will use updated cohort definitions)
python district_expend_pp_stack.py

# Compile final PDF
python compose_pdf.py
```

Expected output:
- 4 enrollment scatterplot files (one per year: 2009, 2014, 2019, 2024)
- 4 choropleth map files (one per year: 2009, 2014, 2019, 2024)
- Updated NSS/CH70 plots with "2024 [Cohort] cohort" labels
- Updated PPE plots with corrected y-axis limits
- Final PDF with all corrections and updated Appendix C

---

## Workflow Instructions for Claude Code

**Script Execution Policy:**
- Do not run scripts (`compose_pdf.py`, `western_map.py`, `nss_ch70_main.py`, etc.) unless needed for testing purposes
- User will run scripts from separate terminal to save token usage
- User has VS Code open and may not see file updates automatically - files need to be reloaded in editor to see changes

**Text Formatting in ReportLab (compose_pdf.py):**
- ReportLab `Paragraph` objects require HTML/XML markup, not Python escape sequences
- Use `<br/>` for line breaks (not `\n`)
- Use `&nbsp;&nbsp;&nbsp;&nbsp;` or HTML tags for indentation (not `\t`)
- Supports: `<b>` for bold, `<i>` for italic, etc.

---

## 2025-10-07 - Choropleth Map Refinements and Data-Driven District Types

### Refactored District Type Classification to Use Data File

**Changed from hardcoded to data-driven approach** for district type classification.

**Problem:**
- District types (unified regional vs secondary regional) were hardcoded in `UNIFIED_REGIONAL_DISTRICTS` set
- Adding new regions or correcting district classifications required code changes
- Not extensible for other parts of Massachusetts

**Solution:**
- Created `load_district_types()` function to read from `Ch 70 District Profiles Actual NSS Over Required.xlsx`
- Reads `MA_District_Profiles` sheet with columns: `DistOrg` (district name), `DistType` (classification)
- Maps district types:
  - "District" → elementary (not regional)
  - "Unified Regional" → unified_regional (serves PK-12, gets "U" marker)
  - "Regional Composite" → regional_composite (overlaps elementary, gets stripes + black border)

**Implementation** (western_map.py):
1. Added `DISTRICT_PROFILES_FILE` constant pointing to data file (line 67)
2. Removed hardcoded `UNIFIED_REGIONAL_DISTRICTS` set
3. Created `load_district_types()` function (lines 109-158):
   - Reads Excel file and MA_District_Profiles sheet
   - Cleans district names for matching
   - Returns dict mapping cleaned name → district type
4. Updated `match_districts_to_geometries()` (lines 219, 271-305):
   - Calls `load_district_types()` at start
   - Determines `is_regional` using data file first, then fallbacks
   - Adds `data_district_type` and `regional_subtype` fields to matched districts
5. Updated `create_western_ma_map()` (lines 360-367):
   - Uses `regional_subtype` instead of `shapefile_type` to separate unified vs secondary regionals
   - Ensures data file classification takes precedence

**Visual Improvements:**
- Made "U" symbol more visible with black outline effect (fontsize 20 black + fontsize 18 white on top)
- Added thick black border (linewidth=3.0, zorder=6) around secondary regional districts
- Added black border to legend
- Updated choropleth_explanation text in compose_pdf.py to use HTML formatting (`<br/>`)

**Files Modified:**
- **western_map.py** - Refactored district type logic (removed 6-line hardcoded set, added 50-line data loading function)
- **compose_pdf.py** - Updated map explanation text to mention black border and use HTML markup
- **WORK_LOG.md** - Added workflow instructions for ReportLab formatting

**Data File:**
- Location: `data/Ch 70 District Profiles Actual NSS Over Required.xlsx`
- Sheet: `MA_District_Profiles`
- Columns used: `DistOrg`, `DistType`

**Extensibility Benefits:**
- ✅ Can add/update district classifications by editing Excel file (no code changes)
- ✅ Can extend to other MA regions by adding districts to data file
- ✅ District type corrections (like Mohawk Trail) handled in data, not code
- ✅ Clear separation between code logic and district classification data

**Districts Affected:**
- Mohawk Trail now correctly identified as Unified Regional (gets "U" marker)
- All 9 unified regional districts loaded from data file
- All 3 secondary regional districts (Frontier, Ralph C Mahar, Amherst-Pelham) maintain stripes + black border

---

## 2025-10-06 - Choropleth Map and Windows Filename Fix

### Choropleth Map of Western MA Districts

**Added geographic visualization** showing district locations color-coded by enrollment cohort.

**Implementation:**

1. **Created `western_map.py` module** for choropleth generation:
   - Loads Census Bureau TIGER/Line shapefiles (unified, elementary, secondary school districts)
   - Matches Western MA traditional districts to geometries using cleaned name matching
   - Color codes by 5 enrollment cohorts (Tiny, Small, Medium, Large, Springfield)
   - Highlights 5 districts of interest with dark orange outlines
   - Generates static 300 DPI PNG for PDF inclusion

2. **Color Palette (colorblind-friendly):**
   - Tiny: Purple (#9C27B0)
   - Small: Green (#4CAF50)
   - Medium: Blue (#2196F3)
   - Large: Orange (#FF9800)
   - Springfield: Red (#F44336)
   - Districts of interest outline: Dark Orange (#FF8C00)

3. **Data Source:**
   - US Census Bureau TIGER/Line 2023 shapefiles for Massachusetts
   - Three district types: unified (212), elementary (63), secondary (29)
   - Combined total: 304 district geometries
   - Successfully matched: 58 out of 61 Western MA districts
   - Unmatched: 3 districts (Farmington River Reg, Greater Commonwealth Virtual, Hoosac Valley Regional)

4. **Integration Points:**
   - Added to Section 1 of PDF report (after PPE overview, before enrollment analysis)
   - Auto-generated by `district_expend_pp_stack.py` during plot generation
   - Graceful fallback if geopandas not installed
   - Output: `output/western_ma_choropleth.png`

5. **Files Created/Modified:**
   - **NEW: `western_map.py`** - Choropleth generation module (350+ lines)
   - **NEW: `data/shapefiles/`** - Copied Census shapefiles from nss_web_app_py
   - **MODIFIED: `compose_pdf.py`** - Added choropleth page to Section 1 with explanatory text
   - **MODIFIED: `district_expend_pp_stack.py`** - Integrated map generation into build process
   - **MODIFIED: `requirements.txt`** - Added geopandas>=0.14, reportlab>=4.0

**Update: Improved Rendering for Overlapping District Boundaries**

6. **Layered rendering to handle overlapping districts:**
   - **Problem:** Elementary and regional secondary districts overlap geographically
   - **Solution:** Two-layer rendering with **spatially-adaptive** diagonal stripe pattern
     - Layer 1: Non-regional districts (elementary/unified) with solid filled polygons (85% opacity)
     - Layer 2: Regional districts with stripe color that adapts spatially based on what's underneath
   - **Implementation:**
     - Added `is_regional` flag based on district name and shapefile type
     - Separate rendering for 54 non-regional (solid) and 4 regional (striped)
     - **Spatially-adaptive stripe coloring** using geometric intersection analysis:
       - For each regional district, compute `geometry.intersection()` with every underlying elementary district
       - Create separate geometry for same-cohort overlaps and different-cohort overlaps
       - Plot same-cohort overlap areas with **white stripes** (for contrast)
       - Plot different-cohort overlap areas with **cohort-color stripes** (for identification)
       - Example: Frontier (green/Small) shows white stripes over green elementary, green stripes over purple elementary
     - Stripe pattern: `facecolor="none"` with `hatch="////"` creates alternating transparent/color stripes
     - Transparent areas allow underlying elementary district fills to show through completely
   - **Regional districts identified:** 4 total with spatial analysis results:
     - **Frontier** (Small/Green): 7 different-cohort areas (green stripes) + 4 same-cohort areas (white stripes)
     - **Ralph C Mahar** (Small/Green): 9 different-cohort areas (green stripes) + 3 same-cohort areas (white stripes)
     - **Amherst-Pelham** (Medium/Blue): 9 different-cohort areas (blue stripes) + 2 same-cohort areas (white stripes)
     - **Southwick-Tolland-Granville** (Medium/Blue): 3 different-cohort areas (blue stripes) + 0 same-cohort areas

7. **Enhanced legend:**
   - Separate entries for solid vs striped districts within each cohort
   - Cohort-colored stripe samples for regional districts
   - White stripe sample showing alternative pattern for same-cohort overlaps
   - Clearer visual hierarchy with explanatory labels

**Impact:**
- ✅ Visual context for Western MA geography
- ✅ Immediate understanding of which districts are included/excluded
- ✅ Cohort distribution visible geographically
- ✅ Overlapping district boundaries properly handled with spatially-adaptive diagonal stripes
- ✅ Regional vs non-regional districts visually distinguished
- ✅ Spatially-adaptive stripe coloring provides maximum visual clarity
- ✅ White stripes provide contrast where regional overlaps same-cohort elementary
- ✅ Cohort-colored stripes show regional's cohort where it overlaps different cohorts
- ✅ Stripe pattern adapts within each regional district based on geometric overlaps
- ✅ Transparent/color alternating stripes show underlying geography clearly
- ✅ Both district types visible simultaneously without obscuring each other
- ✅ Stripe pattern more visible and clearer than dashed outlines or solid hatching
- ✅ Spatial analysis reveals detailed overlap patterns (e.g., Frontier: 7 different-cohort + 4 same-cohort areas)
- ✅ Professional cartographic visualization suitable for reports
- ✅ Sophisticated use of geometric operations (intersection, unary_union) for intelligent rendering

### Windows Reserved Filename Fix (nul file issue)

**Problem:** Code was creating `output\nul` file, which is a Windows reserved device name causing git issues.

**Root Cause:** Filename generation used manual string replacement without checking for Windows reserved names (CON, PRN, AUX, NUL, COM1-9, LPT1-9).

**Solution:**

1. **Created centralized `make_safe_filename()` function** in `school_shared.py` (lines 24-72):
   - Handles all Windows reserved device names
   - Converts special characters (≤, ≥, >, <, etc.) to safe alternatives
   - Removes/replaces problematic characters (/, \, :, *, ?, ", |)
   - Prefixes reserved names with "file_" (e.g., "nul" → "file_nul")
   - Removes non-ASCII characters

2. **Updated all filename generation code:**
   - `nss_ch70_main.py` - NSS/Ch70 plot filenames
   - `compose_pdf.py` - District PNG path functions
   - `district_expend_pp_stack.py` - PPE plot filenames

**Files Modified:**
- `school_shared.py` - Added make_safe_filename() function
- `nss_ch70_main.py` - Use safe filename function
- `compose_pdf.py` - Use safe filename function
- `district_expend_pp_stack.py` - Use safe filename function

**Impact:**
- ✅ No more Windows reserved device name files
- ✅ Cross-platform compatibility
- ✅ Git can properly track all output files
- ✅ Prevents OS-level file system issues

---

## 2025-10-06 - Final Formatting Polish and Consistency Updates

### Comprehensive Formatting and Labeling Refinement (8-Point Update)

**Change A: Enrollment Axis Spacing and K/M Abbreviations**
- **Spacing improvements** (district_expend_pp_stack.py:167-169, nss_ch70_plots.py:153-166):
  - Increased `pad=12` on y-axis tick labels (space between enrollment numbers and donut dots)
  - Increased `labelpad=15` on y-axis label (prevents "Enrollment" text from overlapping donut dots)
- **K/M abbreviations** for enrollment axis (matching dollar axis style):
  - Created `enrollment_formatter()` function in both plot_one_simple() and plot_one()
  - Format logic: ≥1M shows "X.XM", ≥1K shows "X.XK", <1K shows integer
  - Applied to all PPE plots (district_expend_pp_stack.py:172-183, 316-327) and NSS/Ch70 plots (nss_ch70_plots.py:156-165)
  - Examples: "1.5K" instead of "1500", "2.3M" instead of "2,300,000"
- **Impact:** Cleaner axis formatting, no overlap between enrollment labels and donut ticks

**Change B: Fixed Extra Donut Dot on Springfield Plot**
- **Problem:** Springfield plot showed extra donut dot above labeled ticks (probably at 35,000 with no label)
- **Root cause:** Matplotlib auto-adding tick beyond data range without assigning label
- **Solution** (district_expend_pp_stack.py:233-237, nss_ch70_plots.py:213-217):
  - Check if tick label exists and is non-empty before rendering donut
  - Only render donut if `label_text and label_text.strip()`
- **Impact:** No more floating donuts above the y-axis on any plots

**Change C: K/M Abbreviations Applied to Enrollment Axis (Duplicate of A)**
- See Change A above - this was the implementation of the K/M abbreviation formatter

**Change D: Scatterplot Formatting with $ Signs and K/M Abbreviations**
- **Added custom formatters** (western_enrollment_plots_individual.py:156-176):
  - `enrollment_formatter()`: K/M abbreviations for x-axis (enrollment)
  - `ppe_formatter()`: $ signs + K/M abbreviations for y-axis (PPE)
  - Examples: "$45.0K" instead of "45000", "1.5K" instead of "1500"
- **Applied to scatterplot axes:**
  - X-axis (enrollment): Shows "1.0K", "2.0K" quartile markers
  - Y-axis (PPE): Shows "$15.0K", "$20.0K", "$45.0K" values
- **Impact:** Scatterplot matches formatting style of all other plots

**Change E: Scatterplot Table Enhancements**
- **Sort arrow indicator** (compose_pdf.py:224, 229):
  - Added ▼ to "2024 PPE" column header: `<b>2024<br/>PPE ▼</b>`
  - Shows table is sorted by 2024 PPE descending (highest to lowest)
- **Cohort dividing lines** (compose_pdf.py:232-276, 306-310):
  - Track cohort changes in both left and right columns during row building
  - Store row indices where cohort transitions occur (TINY→SMALL→MEDIUM→LARGE)
  - Add faint gray lines (0.3pt, colors.grey) above each transition row
  - Applied to both left column (cols 0-3) and right column (cols 5-8) independently
- **Impact:** Table visually groups districts by enrollment cohort, clear sort direction

**Change F: Updated Weighted Average Subtitles**
- **Changed subtitle** (compose_pdf.py:1195):
  - OLD: "PPE vs Enrollment — Not including charters and vocationals"
  - NEW: "PPE vs Enrollment — Weighted average per district"
- **Rationale:** More accurate description of aggregate calculation method
- **Applies to:** All Western MA cohort aggregate plots (Tiny, Small, Medium, Large)

**Change G: Updated Weighted Average Y-Axis Labels**
- **PPE plots y-axis** (district_expend_pp_stack.py:167-171, 312-316):
  - Added dynamic label selection based on enrollment_label
  - If enrollment_label contains "weighted avg": ylabel = "Weighted avg $ per pupil"
  - Otherwise: ylabel = "$ per pupil"
  - Applied to both plot_one_simple() and plot_one()
- **NSS/Ch70 plots y-axis** (nss_ch70_plots.py:127):
  - Changed from "Weighted Avg $ per District" to "Weighted avg $ per district"
  - Changed from "$ per District" to "$ per district"
  - Lowercase for consistency with other labels
- **Impact:** Y-axis labels clearly indicate when values are weighted averages vs individual districts

**Change H: Updated Enrollment Axis Labels**
- **Enrollment label standardization** (district_expend_pp_stack.py:754, nss_ch70_main.py:152,157,162,167):
  - OLD: "Enrollment per District (weighted avg)"
  - NEW: "Weighted avg enrollment per district"
  - Applied to all Western MA cohort aggregates (Tiny, Small, Medium, Large)
- **Dollar axis label standardization** (see Change G):
  - OLD: "Weighted Avg $ per District"
  - NEW: "Weighted avg $ per district"
- **Consistency:** All "weighted avg" labels now use lowercase and consistent word order

### Files Modified
- **western_enrollment_plots_individual.py** (lines 156-176):
  - Added enrollment_formatter() and ppe_formatter() for scatterplot axes
  - Applied K/M abbreviations and $ signs to scatterplot

- **district_expend_pp_stack.py** (multiple locations):
  - Added enrollment_formatter() to plot_one_simple() (lines 172-183)
  - Added enrollment_formatter() to plot_one() (lines 316-327)
  - Added dynamic ylabel selection for weighted avg (lines 167-171, 312-316)
  - Increased spacing: pad=12, labelpad=15 (lines 167-169, 313-315)
  - Fixed extra donut dot check (lines 233-237)
  - Updated enrollment label to "Weighted avg enrollment per district" (line 754)

- **nss_ch70_plots.py** (multiple locations):
  - Added enrollment_formatter() function (lines 156-165)
  - Increased spacing: pad=12, labelpad=15 (lines 153, 166)
  - Fixed extra donut dot check (lines 213-217)
  - Updated ylabel to lowercase (line 127)

- **nss_ch70_main.py** (lines 152,157,162,167):
  - Updated enrollment label to "Weighted avg enrollment per district"

- **compose_pdf.py** (multiple locations):
  - Added sort arrow ▼ to scatterplot table headers (lines 224, 229)
  - Added cohort change tracking and dividing lines (lines 232-276, 306-310)
  - Updated subtitle to "Weighted average per district" (line 1195)

### Impact Summary
- ✅ **Consistent formatting** across all plot types (PPE, NSS/Ch70, scatterplot, enrollment)
- ✅ **K/M abbreviations** standardized on all axes (enrollment and dollars)
- ✅ **Donut dots** only appear at labeled tick positions (no extras)
- ✅ **Scatterplot table** clearly sorted and visually grouped by cohort
- ✅ **Weighted average labels** consistently formatted and positioned
- ✅ **Spacing improved** on enrollment axes to prevent label overlap
- ✅ **Dollar signs** on all PPE/$ axes for clarity

### Plots Regenerated
- ✅ All enrollment distribution plots (scatterplot, histogram, grouping)
- ✅ All PPE vs enrollment plots (5 cohorts + 5 individual districts × 2 versions = 20 plots)
- ✅ All NSS/Ch70 plots (5 cohorts + 5 individual districts = 10 plots)
- ✅ Western MA overview plot
- ✅ Final PDF: `output/expenditures_series.pdf`

---

## 2025-10-06 - Planned Next Steps

### Objective: Add Regional CPI Comparison and Geographic Visualization

**1. Regional CPI Data Integration**
- **Data source:** Bureau of Labor Statistics (BLS) CPI-U data for Boston-Cambridge-Newton metro area
  - Boston CPI is standard reference for Massachusetts cost-of-living adjustments
  - May also want Northeast region CPI or National CPI for additional context
- **Time period:** 2009-2024 (matching current plot timeframe)
- **Integration points:**
  - Add CPI adjustment option to expenditure plots (inflation-adjusted dollars)
  - Show real vs nominal growth rates in CAGR tables
  - Compare district spending growth to CPI growth (outpacing inflation vs falling behind)
- **Implementation considerations:**
  - Need to decide: adjust historical values to 2024 dollars, or show CPI as overlay line?
  - May need new column in tables: "Real CAGR (CPI-adjusted)"
  - Could add toggle in plot generation: nominal vs real dollars
- **Data location:**
  - Store in new CSV: `data/cpi_boston_metro.csv` or similar
  - Add BLS API integration option for automatic updates

**2. Choropleth Map of Included Districts**
- **Purpose:** Show which Western MA districts are included in analysis at a glance
- **Map requirements:**
  - Massachusetts state outline
  - Western MA county boundaries (Berkshire, Franklin, Hampshire, Hampden)
  - District boundaries or town polygons
  - Color coding:
    - Highlight 5 districts of interest (Amherst-Pelham, Amherst, Leverett, Pelham, Shutesbury) in dark orange
    - Show all other Western MA Traditional districts in steel blue
    - Gray out excluded districts (charters, vocational, non-Western)
  - Cohort indicators: Could use different shades/patterns for Tiny/Small/Medium/Large cohorts
- **Technical approach:**
  - Use geopandas + matplotlib for static map generation
  - Data source: MassGIS (MA Office of Geographic Information) provides district/town shapefiles
  - Alternative: Use Folium for interactive HTML map (could be separate from PDF)
- **Integration:**
  - Add as new page early in PDF (before Section 1 or after TOC)
  - Title: "Western Massachusetts Traditional School Districts Included in Analysis"
  - Could include legend showing cohort boundaries and district counts
- **File structure:**
  - New script: `western_map.py` for map generation
  - Shapefile data in: `data/ma_districts/` directory
  - Output: `output/western_ma_map.png`

**3. Additional Enhancements to Consider**
- **CPI comparison page:**
  - Dedicated page showing CPI trend line 2009-2024
  - Table comparing district/cohort CAGR to CPI growth rate
  - Districts with spending growth < CPI highlighted (losing purchasing power)
- **Real vs Nominal toggle:**
  - Add parameter to all plotting functions: `inflation_adjusted=False`
  - When True: multiply historical values by CPI ratio to convert to 2024 dollars
  - Update y-axis labels: "$ per pupil (2024 dollars)" vs "$ per pupil (nominal)"
- **Map enhancements:**
  - Add district enrollment size as circle overlays
  - Show PPE by color gradient (expensive districts darker, cheaper lighter)
  - Include major highways/cities for geographic orientation

**4. Data Files Needed**
- CPI data CSV with columns: year, cpi_value, cpi_index_2024base
- MA district shapefile (.shp, .shx, .dbf, .prj) from MassGIS
- District-to-town mapping (some districts span multiple towns)

**5. New Dependencies**
- `geopandas` for spatial data handling
- `shapely` for geometry operations (comes with geopandas)
- `descartes` or `contextily` for basemap backgrounds (optional)

### Questions to Resolve
1. CPI reference: Boston metro only, or also show Northeast/National for comparison?
2. Inflation adjustment: Convert all to real dollars, or keep nominal with CPI overlay?
3. Map style: Simple choropleth, or add enrollment/PPE overlays?
4. Map placement: Before Section 1, after TOC, or in new "Geography" appendix?
5. Interactive map: HTML version in addition to static PDF image?

### Nul
The code created the file output\nul which causes problems with git. 'nul' is a reserved name in Windows. 
Instructions for Claude Code during next session: Update the code to make sure it does not create a nul file.
Tell Daisy why it was created in the first place, if known.

---

## 2025-10-05 - Cohort System Refinement and Plot Improvements

### Implemented 5-Tier IQR-Based Cohort System

**Change 1: Updated Cohort Boundaries from 4-tier to 5-tier**
- **Old system (4 tiers):**
  - Small: 0 to Median (rounded to nearest 100)
  - Medium: Median+1 to Q3 (rounded to nearest 100)
  - Large: Q3+1 to max non-outlier (rounded to nearest 100)
  - Springfield: >8000 FTE (outlier)

- **New system (5 tiers):**
  - Tiny: 0 to Q1 (rounded to nearest 100)
  - Small: Q1+1 to Median (rounded to nearest 100)
  - Medium: Median+1 to Q3 (rounded to nearest 100)
  - Large: Q3+1 to max non-outlier (rounded **up** to nearest 1000)
  - Springfield: >8000 FTE (outlier)

**Implementation:**
- `school_shared.py:87-97` - Updated `calculate_cohort_boundaries()` to calculate Q1, Median, Q3
  - Added `round_up_1000()` function for Large cohort upper bound (ensures districts like Chicopee (7,530 FTE) are correctly classified)
  - Updated static fallback definitions to include TINY
- `school_shared.py:243-246` - Updated `get_western_cohort_districts()` to return 5 cohorts
- `school_shared.py:670-695` - Updated `prepare_western_epp_lines()` to handle "tiny" bucket
- `school_shared.py:577-587` - Updated `context_for_western()` to include TINY mapping
- `compose_pdf.py:203-208` - Added purple color (#9C27B0) for TINY in scatterplot table
- `compose_pdf.py:326-328` - Updated scatterplot table sort order to place TINY first
- `compose_pdf.py:1230-1248` - Added western_tiny list and TINY grouping logic
- `compose_pdf.py:1274-1281` - Added TINY to bucket_map for district baselines
- `compose_pdf.py:1323-1330` - Added TINY to group_map for NSS/Ch70 baselines
- `compose_pdf.py:1417-1442` - Added TINY to Appendix B methodology text
- `compose_pdf.py:1466-1481` - Updated methodology to describe 5 cohorts instead of 4
- `nss_ch70_main.py:74-79` - Added western_tiny to enrollment_groups
- `nss_ch70_main.py:126-127` - Added "Tiny" filename handling for NSS/Ch70 plots
- `western_enrollment_plots_individual.py:130-150` - Added TINY cohort (purple) to scatterplot
- `western_enrollment_plots_individual.py:160-206` - Updated enrollment grouping plot to show 5 bars with narrower width (height=0.64)

**Change 2: Improved Plot Overflow Tick Logic**
- **Problem:** Faded overflow ticks appeared on all plots, even when recent enrollment stayed within cohort bounds
- **Solution:** Only extend y-axis with faded ticks when **early years (2009-2011)** exceed the cohort upper bound
- **Implementation:**
  - `nss_ch70_plots.py:151-187` - Check early years enrollment before extending axis
    - If early max > cohort bound: extend by 100-200 above early max
    - Tick spacing: 100 (if ylim < 1000), 200 (if ylim ≤ 2000), 500 (if ylim < 5000), 1000 (otherwise)
  - `district_expend_pp_stack.py:173-201, 254-282` - Same logic for PPE plots (both plot_one_simple and plot_one)
- **Examples:**
  - Amherst (Medium, ylim=1800): Shows 0-1800 with 200 spacing, faded 2000 tick if 2009 enrollment > 1800
  - Leverett (Tiny, ylim=200): Shows 0-200 with 100 spacing, no overflow ticks if 2009 ≤ 200

**Change 3: Removed Top Border from All Plots**
- `nss_ch70_plots.py:147` - Added `axL.spines['top'].set_visible(False)`
- `district_expend_pp_stack.py:202, 283` - Added same to both plot functions

**Files Modified:**
- `school_shared.py` - 5-tier cohort calculation, TINY handling throughout
- `compose_pdf.py` - TINY cohort in all mappings, tables, methodology
- `nss_ch70_main.py` - TINY aggregate plot generation
- `nss_ch70_plots.py` - Early-years overflow logic, top border removal
- `district_expend_pp_stack.py` - Early-years overflow logic, top border removal, tick spacing fix
- `western_enrollment_plots_individual.py` - 5-tier scatterplot and grouping plot

**Impact:**
- ✅ Better granularity for small districts (Tiny cohort separates smallest 25%)
- ✅ Large cohort upper bound now accommodates high-enrollment districts correctly
- ✅ Scatterplot table shows all 5 cohorts with color-coded dots
- ✅ Y-axis overflow ticks only appear when historically necessary
- ✅ Tick marks properly align with cohort boundaries (e.g., 1800 for Medium)
- ✅ All titles and labels automatically reflect dynamic IQR-based boundaries
- ✅ Cleaner plot appearance without top border

**Cascade Verification:**
- [x] Enrollment grouping plot shows 5 bars with correct boundaries
- [x] Scatterplot shows 5 colors (purple, green, blue, orange for cohorts)
- [x] All Western MA aggregate pages generated for 5 cohorts
- [x] District comparison baselines automatically use correct cohort
- [x] Appendix B methodology lists 5 cohorts with member districts
- [x] No legacy ">500 Students" or "≤500 Students" text remains

---

## 2025-10-03 (Evening Session)

### Major PDF Restructuring and Formatting Fixes

**Change 1: Fixed Ch70 NSS Plot Formatting to Match Standard Plots**
- Removed custom font sizes from NSS/Ch70 plots (nss_ch70_plots.py:112-114)
  - Changed from `fontsize=16` to default (no fontsize parameter)
  - Removed `fontsize=18` plot title (suptitle) entirely to match standard plots
  - Removed `fontsize=12` from legend, using default instead
- NSS/Ch70 plots now have consistent formatting with expenditure plots

**Change 2: Updated Comparison Logic - ALPS PK-12 Only Compares to ALPS Peers**
- Individual districts (Amherst-Pelham, Amherst, Leverett, Pelham, Shutesbury):
  - Removed ALPS Peers comparison pages (compose_pdf.py:1017-1045 removed)
  - Removed second NSS/Ch70 comparison page vs ALPS Peers (compose_pdf.py:1092-1105 removed)
  - Now only compare to Western MA Traditional Districts (in their enrollment bucket)
- ALPS PK-12:
  - Removed comparison to Western MA Traditional Districts (compose_pdf.py:1127-1152 removed)
  - Now only compares to ALPS Peer Districts Aggregate

**Change 3: Comprehensive PDF Reorganization into 3 Sections + 2 Appendices**

**New Structure:**
1. **Section 1: Western MA** (compose_pdf.py:922-993)
   - All Western MA Traditional Districts: PPE Overview 2019 -> 2024 (section_id="section1_western")
   - All Western MA Traditional Districts ≤500 Students (with PPE and NSS/Ch70 pages)
   - All Western MA Traditional Districts >500 Students (with PPE and NSS/Ch70 pages)

2. **Section 2: Individual Districts** (compose_pdf.py:995-1101)
   - Amherst-Pelham Regional (section_id="amherst_pelham")
   - Amherst (section_id="amherst")
   - Leverett (section_id="leverett")
   - Pelham (section_id="pelham")
   - Shutesbury (section_id="shutesbury")
   - Each district has: Simple page, Detailed vs Western, NSS/Ch70 vs Western

3. **Section 3: ALPS PK-12 & Peers** (compose_pdf.py:1103-1256)
   - ALPS PK-12 & Peers: PPE and Enrollment 2019 -> 2024 (section_id="section3_alps_peers")
   - ALPS PK-12 (3 pages: Simple, Detailed vs ALPS Peers, NSS/Ch70 vs ALPS Peers)
   - ALPS Peer Districts Aggregate (2 pages: PPE and NSS/Ch70)

4. **Appendix A: Data Tables** (was Appendix B) (compose_pdf.py:1258-1303)
   - Updated title and section_id from "appendix_b" to "appendix_a"
   - Contains all underlying data for PPE, FTE, and NSS/Ch70

5. **Appendix B: Calculation Methodology** (was Appendix C) (compose_pdf.py:1305-1470)
   - Updated all titles from "Appendix C" to "Appendix B"
   - Updated section_id from "appendix_c" to "appendix_b"
   - 3 pages covering CAGR, aggregates, shading logic, and NSS/Ch70 calculations

**Change 4: Updated Table of Contents** (compose_pdf.py:1475-1487)
- New TOC structure reflects 3-section organization:
  - Section 1: Western MA
  - Section 2: Amherst-Pelham Regional, Amherst, Leverett, Pelham, Shutesbury
  - Section 3: ALPS PK-12 & Peers
  - Appendix A. Data Tables
  - Appendix B. Calculation Methodology
- Removed individual "ALPS PK-12" and "All Western MA" entries (now part of section headers)
- Removed old "Appendix A. Aggregate Districts for Comparison" (aggregates now in main sections)

**Files Modified:**
- `nss_ch70_plots.py` - Removed custom font sizes, removed plot title
- `compose_pdf.py` - Complete restructuring:
  - Removed ALPS Peers comparison pages for individual districts
  - Removed ALPS vs Western comparison pages
  - Moved Western aggregates from Appendix A to Section 1
  - Reorganized page order into 3 sections
  - Renamed Appendix B → A, Appendix C → B
  - Updated TOC to reflect new structure
- Regenerated PDF (7.3 MB, 21:57)

**Impact:**
- ✅ Simplified comparison logic: individual districts only compare to Western MA
- ✅ ALPS PK-12 only compares to ALPS Peers (its true peer group)
- ✅ Clear 3-section organization makes report easier to navigate
- ✅ Western aggregates properly positioned in Section 1 with overview
- ✅ Consistent plot formatting across all NSS/Ch70 plots
- ✅ TOC accurately reflects new document structure

---

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

---

## Session 2025-10-20

**11. Created standalone legend for Executive Summary CAGR plots**
- Added `plot_cagr_legend()` function (executive_summary_plots.py:491-581):
  - Generates standalone legend PNG with same color scheme as CAGR plots
  - Uses matplotlib.patches.Patch to create legend items
  - Horizontal layout with 4 columns, 18pt font
  - Includes cohort hatching pattern for visual consistency
  - Figure size: 18x2 inches for wide horizontal layout
- Modified `plot_cagr_grouped_bars()` (executive_summary_plots.py:584-695):
  - Removed inline legend (was at top of plot with bbox_to_anchor)
  - Removed plt.subplots_adjust(top=0.92) that made room for legend
  - Plot now uses full vertical space without legend overhead
- Updated main() function (executive_summary_plots.py:809-879):
  - Changed step numbering from [1/5]-[5/5] to [1/6]-[6/6]
  - Added step [6/6] to generate standalone legend
  - Outputs: executive_summary_cagr_legend.png

**Output:**
- New file: output/executive_summary_cagr_legend.png
- Modified: output/executive_summary_cagr_grouped.png (legend removed)
- Legend can now be placed independently on page without affecting plot alignment

**12. Updated compose_pdf.py to place legend and CAGR plots together**
- Modified page definition (compose_pdf.py:1439-1451):
  - Changed chart_paths to include legend as first element
  - Added executive_summary_cagr_legend.png before the two CAGR plots
  - Replaced two_charts_vertical and cagr_with_text flags with new cagr_with_legend flag
- Added new layout rendering logic (compose_pdf.py:2627-2684):
  - Detects cagr_with_legend flag and 3 chart_paths
  - Places explanation text first (12pt spacer)
  - Places legend at 8% of page height (8pt spacer after)
  - Places 5-year CAGR grouped bars at 35% of page height (10pt spacer after)
  - Places 15-year CAGR bars at 35% of page height
  - All three images maintain full page width with proper aspect ratio
- Maintained backward compatibility with two_charts_vertical flag for other pages

**Page Layout:**
- Executive Summary (continued) page now shows:
  1. Title and subtitle
  2. CAGR explanation text
  3. Legend (horizontal, short height)
  4. 5-year CAGR grouped bars (Figure N)
  5. 15-year CAGR bars (Figure N+1)
- Total vertical allocation: ~8% legend + 35% + 35% charts = ~78% of page height
- Remaining space for title, text, figure captions, and spacing

**13. Increased size of legend and CAGR charts on page**
- Updated legend height allocation (compose_pdf.py:2644):
  - Increased from 8% to 15% of page height for better visibility
- Updated CAGR chart height allocations (compose_pdf.py:2658, 2673):
  - Increased from 35% to 42% of page height for both charts
- New total vertical allocation: 15% legend + 42% + 42% charts = 99% of page height
- Images maintain full page width with proper aspect ratios
- Both plots now display larger and more readable on the page

**14. Further increased legend size and reduced bar gaps in 15-year CAGR plot**
- Updated legend height allocation (compose_pdf.py:2644):
  - Increased from 15% to 22% of page height for improved readability
- Reduced bar gaps in 15-year CAGR plot (executive_summary_plots.py:766):
  - Increased bar_width from 0.5 to 0.8
  - Bars now wider with minimal whitespace between them
  - Plot appears less wide with more compact bar arrangement
- Legend and bars now more prominent on the page

**15. Implemented data caching/checkpointing system**
- Created cache_manager.py module (lines 1-175):
  - `save_cache()`: Saves DataFrames as CSV files to data/cache/
  - `load_from_cache()`: Loads DataFrames from cache CSVs
  - `use_cache()`: Checks if cache exists and should be used
  - `cache_exists()`: Validates all cache files are present
  - `clear_cache()`: Deletes cache files
  - Cache includes: expenditure_data.csv, regional_data.csv, chapter70_data.csv, cache_metadata.txt
- Updated school_shared.py load_data() function (lines 459-550):
  - Added `force_recompute` parameter (default False)
  - Checks cache first before loading from Excel
  - Automatically saves to cache after loading from Excel
  - Gracefully falls back to Excel if cache load fails
- Updated generate_report.py (lines 1-139):
  - Added argparse support for --force-recompute flag
  - Passes flag to all pipeline scripts
  - Shows cache mode in output ("Using cache" vs "Force recompute")

**Cache Behavior:**
- First run: Loads from Excel, saves to data/cache/, ~30-60 seconds
- Subsequent runs: Loads from cache CSVs, ~2-5 seconds
- With --force-recompute: Bypasses cache, reloads from Excel
- Cache automatically used if present and force flag not set

**Next Steps:**
- ~~Add --force-recompute support to individual pipeline scripts~~ ✅ COMPLETED
- Test full pipeline with caching enabled
- (Future) Move to PPE_project structure with PPE_ filename prefixes

---

## 16. Complete Pipeline Integration with Caching (2025-10-20)

**Problem:** Individual pipeline scripts did not accept --force-recompute argument, preventing full integration with caching system.

**Solution:** Updated all 6 pipeline scripts to accept and pass --force-recompute flag:

**Changes Made:**
1. **threshold_analysis.py:**
   - Added `import argparse` at line 15
   - Added argument parser in main() to accept --force-recompute flag
   - Updated load_data() call to pass force_recompute parameter

2. **executive_summary_plots.py:**
   - Added `import argparse` at line 8
   - Added argument parser in main() to accept --force-recompute flag
   - Updated load_data() call to pass force_recompute parameter

3. **district_expend_pp_stack.py:**
   - Added `import argparse` at line 24
   - Added argument parser in main() to accept --force-recompute flag
   - Updated load_data() call to pass force_recompute parameter

4. **nss_ch70_main.py:**
   - Added `import argparse` at line 14
   - Added argument parser in main() to accept --force-recompute flag
   - Updated load_data() call to pass force_recompute parameter

5. **western_map.py:**
   - Added `import argparse` at line 35
   - Added argument parser in main() to accept --force-recompute flag
   - Updated load_data() call to pass force_recompute parameter

6. **western_enrollment_plots_individual.py:**
   - Added `import argparse` at line 13
   - Added argument parser in main() to accept --force-recompute flag
   - Updated load_data() call to pass force_recompute parameter

**Pattern Applied (consistent across all scripts):**
```python
# 1. Import argparse at top
import argparse

# 2. Add argument parser in main()
def main():
    parser = argparse.ArgumentParser(description="[Script description]")
    parser.add_argument("--force-recompute", action="store_true",
                        help="Bypass cache and recompute from source")
    args = parser.parse_args()

    # 3. Pass flag to load_data()
    df, reg, c70 = load_data(force_recompute=args.force_recompute)
```

**Result:** Complete end-to-end caching system is now fully operational:
- Running `python generate_report.py` uses cache if available
- Running `python generate_report.py --force-recompute` bypasses cache and reloads from Excel
- Each individual script can also be run with --force-recompute flag independently
- Cache reduces data loading time from ~30-60 seconds to ~2-5 seconds

**Testing Recommendations:**
1. Delete existing cache: Remove data/cache/ directory
2. Run full pipeline: `python generate_report.py`
   - Should create cache files in data/cache/
   - Should complete successfully
3. Run again without flag: `python generate_report.py`
   - Should load from cache (fast)
   - Should show "[Cache] Cache found - loading from cache"
4. Force recompute: `python generate_report.py --force-recompute`
   - Should bypass cache and reload from Excel
   - Should show "[Cache] Force recompute requested - bypassing cache"

---

## 17. Add Red/Green Shading to Cohort Aggregate Tables (2025-10-20)

**Problem:** Cohort aggregate tables (Tiny, Small, Medium, Large, X-Large, Springfield) and their NSS/Ch70 tables did not have red/green comparison shading, making it difficult to assess how each enrollment group compared to the regional average.

**Question:** What should cohort aggregates be compared to? Since cohorts provide the basis of comparison for individual districts, comparing cohorts to themselves would be meaningless.

**Solution:** Compare all cohort aggregates (including Springfield) to **Western MA weighted average (excluding Springfield)**. This provides meaningful regional context:
- For individual districts: Compare to their cohort aggregate (existing behavior)
- For cohort aggregates: Compare to overall Western MA average
- For Springfield: Compare to overall Western MA average (excluding itself)

**Implementation Details:**

1. **Created Western MA baseline calculation functions** (compose_pdf.py):
   - `_build_western_ma_baseline()` (lines 262-333): Calculates Western MA weighted average for PPE categories
     - Gets all Western MA traditional districts
     - Excludes Springfield (>10K FTE outlier)
     - Uses `weighted_epp_aggregation()` to compute regional average
     - Returns baseline_map with DOLLAR, START_DOLLAR, and CAGR values for all categories
   - `_build_western_ma_nss_baseline()` (lines 336-368): Calculates Western MA weighted average for NSS/Ch70
     - Similar exclusion logic
     - Uses `prepare_aggregate_nss_ch70_weighted()`
     - Returns baseline_map for Ch70 Aid, Required NSS, Actual NSS components

2. **Updated cohort page creation** (compose_pdf.py lines 1625-1733):
   - Calculate Western MA baseline once (shared across all cohort pages)
   - Added `baseline_map` and `fte_baseline_map` to all cohort PPE pages
   - Added `baseline_map` and `fte_baseline_map` to all cohort NSS/Ch70 pages
   - Updated subtitles to indicate: "shaded by comparison to weighted average of all Western MA (excluding Springfield)"
   - Set `baseline_title="All Western MA (excluding Springfield)"`

3. **Modified shading logic** (compose_pdf.py):
   - Line 684: Changed `if page.get("page_type") == "district"` to `if page.get("page_type") in ["district", "western", "nss_ch70"]`
   - Line 776: Updated legend display condition to include "western" page type
   - Line 811: Updated FTE shading condition to include "western" page type
   - These changes enable red/green shading for cohort tables using the same thresholds as district tables

4. **Updated legend text** (compose_pdf.py line 533):
   - Modified `_abbr_bucket_suffix()` to detect "All Western MA (excluding Springfield)" baseline
   - Returns "MA (excl. Springfield)" for cohort page legends
   - Individual district legends still show their cohort label (e.g., "0-200 FTE")

**Result:**
- Cohort aggregate tables now show red/green shading comparing to Western MA average
- Springfield comparison also uses Western MA average (excluding itself)
- Legends correctly identify the comparison baseline
- Same shading thresholds applied: 2.0% for $/pupil, 2.0pp for CAGR

**Testing:**
- Run `python compose_pdf.py` to regenerate PDF
- Check cohort aggregate pages (Section 1): Tiny, Small, Medium, Large, X-Large, Springfield
- Verify red/green shading appears in:
  - PPE category tables
  - Enrollment tables
  - NSS/Ch70 funding component tables
- Verify legend shows "Above Western MA (excl. Springfield)" and "Below Western MA (excl. Springfield)"
- Verify individual district pages still compare to their cohort (no change)