# Work Log - PPE Report Project

## 2025-10-27 - Missing Districts Gray Fill Fix

### Fixed missing districts on enrollment cohort map
- **Issue**: Districts without data for specific years (Hampshire, Chesterfield-Goshen, Southampton, Westhampton, Williamsburg) showed as white instead of gray on 2024 enrollment cohort map
- **Solution**: Created comprehensive base layer of ALL Western MA districts across all years
- **Implementation**:
  - Added `get_all_western_ma_districts()` function to collect all unique Western MA districts from years 2009-2024 (lines 213-292 in western_map.py)
  - Modified `create_western_ma_map()` to accept optional `shapes` parameter with all district geometries (line 353)
  - Updated base layer plotting logic to use `shapes` if provided, showing ALL districts in gray before overlaying cohort colors (lines 395-416)
  - Updated missing data count calculation to compare districts in shapes vs matched_gdf (lines 529-549)
  - Modified `main()` to call new helper function and pass result to map creation (lines 1250-1266)
- **Results**:
  - 2024 map: 65 total districts in base layer, 60 with cohort data, 5 missing districts shown in gray
  - Legend now correctly shows "Missing data: 5 district(s)" for 2024
- File: `western_map.py` lines 213-292, 353, 395-416, 529-549, 1250-1266
- Benefits: All Western MA districts visible on maps regardless of data availability, clear indication of missing data

### Fixed missing districts and white color on PPE comparison map
- **Issue**: Missing districts not showing as gray, and districts within ±5% of cohort average showing as light gray instead of white
- **Solution**: Applied same gray base layer fix as enrollment cohort map, and fixed white color transparency
- **Implementation**:
  - Added `shapes` parameter to `create_ppe_comparison_map()` function (line 701)
  - Updated base layer plotting to use `shapes` if provided (lines 773-794)
  - Fixed white color transparency: Changed alpha from 0.85 to 1.0 for districts within ±5% (line 802)
  - Updated legend alpha to match map (line 857)
  - Updated missing data count to compare shapes vs matched_gdf (lines 868-888)
  - Modified function call in `main()` to pass shapes parameter (line 1302)
- **Results**:
  - 2024 map: 65 total districts in base layer, 60 with PPE data, 5 missing districts shown in gray
  - Districts within ±5% now show as pure white (not light gray)
  - Legend correctly shows "Missing data: 5 district(s)" for 2024
- File: `western_map.py` lines 701, 773-794, 802, 857, 868-888, 1302
- Benefits: Consistent gray fill for missing districts across all map types, white color properly visible for districts near cohort baseline

### Fixed missing districts and white color on CAGR comparison maps
- **Issue**: Same issues as PPE maps - missing districts not showing as gray, districts within ±0.5pp showing as light gray instead of white
- **Solution**: Applied same fixes to all three CAGR comparison maps (2009-2024, 2009-2019, 2009-2014)
- **Implementation**:
  - Added `shapes` parameter to `create_cagr_comparison_map()` function (line 933)
  - Updated base layer plotting to use `shapes` if provided (lines 1006-1027)
  - Fixed white color transparency: Changed alpha from 0.85 to 1.0 for districts within ±0.5pp (line 1035)
  - Updated legend alpha to match map (line 1092)
  - Updated missing data count to compare shapes vs matched_gdf (lines 1103-1123)
  - Modified function calls in `main()` to pass shapes parameter (lines 1346, 1353, 1360)
- **Results**:
  - 2009-2024 map: 65 total districts, 59 with CAGR data, 6 missing shown in gray
  - 2009-2019 map: 65 total districts, 64 with CAGR data, 1 missing shown in gray
  - 2009-2014 map: 65 total districts, 62 with CAGR data, 3 missing shown in gray
  - Districts within ±0.5pp now show as pure white (not light gray)
- File: `western_map.py` lines 933, 1006-1027, 1035, 1092, 1103-1123, 1346, 1353, 1360
- Benefits: Complete consistency across all comparison map types

### Added black borders for secondary regional districts with missing data
- **Issue**: Secondary regional districts like Hampshire weren't showing their characteristic black borders when data was missing
- **Solution**: Show black borders for ALL secondary regional districts, regardless of data availability
- **Implementation**:
  - **PPE maps**: Removed data filter from regional_secondary selection (line 763), added NaN check before adding labels (line 826)
  - **CAGR maps**: Removed data filter from regional_secondary selection (line 996), added NaN check before adding labels (line 1059)
- **Results**:
  - Hampshire and other secondary regionals now show black borders even when data is missing
  - Labels only appear when data exists (not shown for missing data)
  - Consistent visual treatment of secondary regional boundaries across all years
- File: `western_map.py` lines 763, 826 (PPE), 996, 1059 (CAGR)
- Benefits: Secondary regional district boundaries visible on all maps, improving geographic context

## 2025-10-25 - TOC, Statistics, Labels, and Navigation Improvements

### Completed Change Requests (Batch 2: CR A01-A05)

**Geographic map formatting consistency**
- Updated PPE comparison and CAGR comparison maps to match enrollment cohort map formatting
- Secondary Region black border thickness: Changed from `linewidth=1.5` to `linewidth=3.0`
- Font size of +/-% labels: Changed from `fontsize=11` to `fontsize=20`
- Font weight already matched (`weight='bold'`)
- File: `western_map.py` lines 647, 662, 807, 820
- Benefits: Consistent visual appearance across all three geographic map types

**Y-axis label terminology fix**
- Updated PPE plot right y-axis labels to use "FTE" instead of "pupil"
- Changed "Weighted avg $ per in-district pupil" → "Weighted avg $ per in-district FTE"
- Changed "$ per in-district pupil" → "$ per in-district FTE"
- File: `district_expend_pp_stack.py` line 186
- Benefits: Correct terminology that matches full-time equivalent (FTE) enrollment metric

**Statistical distribution table formatting fix**
- Fixed Ch70 Aid and Actual NSS distribution tables showing % instead of $
- Updated `build_cohort_distribution_table()` to detect metric type based on keywords
- Now checks for "$", "PPE", "Aid", or "NSS" in metric name → formats as dollars
- Checks for "%", "CAGR", or "Growth" in metric name → formats as percentages
- File: `compose_pdf.py` lines 1189-1228
- Benefits: Correct display of dollar amounts in current value distribution tables

**Springfield negative NSS value handling**
- Fixed missing values in Actual NSS tables for Springfield (negative values were hidden)
- Updated table formatting to display negative values as "-$X,XXX" instead of "—"
- Updated shading logic to handle negative baseline comparisons
- File: `compose_pdf.py` lines 2155-2158, 2197-2209
- Benefits: Springfield's spending below required NSS levels now properly displayed

**Statistical Test Results pages added**
- Created new text sections in `report_text.txt` for Ch70 and NSS statistical test results
- Added "CH70_STATISTICAL_TEST_RESULTS" section (lines 446-492)
- Added "NSS_STATISTICAL_TEST_RESULTS" section (lines 495-544)
- Added two new pages in Section 1 to display these test results
- File: `compose_pdf.py` lines 3171-3200
- Benefits: Comprehensive statistical analysis matching PPE pattern (associations, effect sizes, interpretation)

**Box plot filename sanitization**
- Fixed FileNotFoundError when creating mini boxplots for statistical distribution tables
- Sanitized metric names containing "/" and "$" characters in filenames
- Added `parents=True` to ensure output directory creation
- File: `compose_pdf.py` lines 1186, 1221
- Benefits: Prevents path separator errors in Windows filenames

**Table of Contents dotted leaders**
- Added faint horizontal dotted leaders (...) between each TOC entry and its page number
- Changed TOC from 2-column to 3-column layout: title | leader dots | page number
- Reduced page number column width from 15% to 10% (page numbers now closer to content)
- Reduced title column width from 85% to 75% to accommodate leader column
- Dots are faint gray (#AAAAAA) with small font (size 8) for subtle appearance
- File: `compose_pdf.py` lines 4473-4526
- Benefits: Professional TOC appearance with clear visual connection between entries and page numbers

**CR A01: Fixed TOC dotted leader wrapping**
- Leaders were wrapping to multiple lines, causing TOC entries to span multiple rows
- Fixed by using non-breaking spaces (Unicode U+00A0) between dots to prevent wrapping
- File: `compose_pdf.py` line 4483
- Benefits: Each TOC entry now stays on a single line

**CR A02: Reordered statistics pages**
- Reordered so test results pages immediately follow their corresponding association pages
- New order: PPE Associations → PPE Test Results → Ch70 Associations → Ch70 Test Results → NSS Associations → NSS Test Results
- Files: `compose_pdf.py` lines 3077-3090, 3135-3148, 3195-3208
- Benefits: Logical grouping of related content

**CR A03: Split PPE statistics into separate pages**
- Created separate sections in report_text.txt: PPE_STATISTICAL_ASSOCIATIONS and PPE_STATISTICAL_TEST_RESULTS
- PPE statistical content now split across two pages matching Ch70 and NSS format
- Files: `report_text.txt` lines 373-447; `compose_pdf.py` lines 2486-2487, 3077-3090
- Benefits: Consistent format across all three metrics (PPE, Ch70, NSS)

**CR A04: Added detailed test results to Ch70 and NSS statistics pages**
- Updated CH70_STATISTICAL_TEST_RESULTS with specific ANOVA/regression statistics
- Updated NSS_STATISTICAL_TEST_RESULTS with specific ANOVA/regression statistics
- Added F-statistics, p-values, R² values, η² values matching PPE format
- Files: `report_text.txt` lines 450-551
- Benefits: All three metrics now have same level of statistical detail

**CR A05: Fixed left y-axis labels to "Enrollment"**
- Updated all NSS/Ch70 plot left y-axis labels from "Foundation Enrollment" / "Weighted avg foundation enrollment per district" to simply "Enrollment"
- File: `nss_ch70_main.py` lines 172, 177, 182, 187, 191, 196, 200, 205
- Benefits: Consistent simplified labeling across all plots

### Summary of Changes

**Files Modified:**
- `western_map.py` (geographic map formatting)
- `district_expend_pp_stack.py` (y-axis label terminology)
- `compose_pdf.py` (statistical tables, Springfield handling, test results pages, filename sanitization)
- `report_text.txt` (Ch70 and NSS statistical test results content)

**Impact:**
- Geographic maps now have consistent styling across all three types (enrollment cohort, PPE comparison, CAGR comparison)
- PPE plots now correctly reference "FTE" (Full-Time Equivalent) instead of generic "pupil"
- NSS/Ch70 plots already correctly reference "foundation pupil" (no changes needed)
- Statistical distribution tables now show correct units ($ vs %)
- Springfield's negative NSS values (spending below required levels) now visible in tables
- Section 1 now has complete statistical analysis for all three metrics (PPE, Ch70 Aid, Actual NSS)
- Each metric has both distribution analysis and detailed statistical test results pages

---

## 2025-10-23 - UI/UX Improvements and Text Externalization

### Completed Change Requests

**CR1: Add spacing before Section 3 in Table of Contents**
- Added empty TOC entry with indent level -1 to create visual spacing before "Section 3: Specific Districts Compared to Cohorts"
- File: `compose_pdf.py` line ~2949
- Creates better visual separation between major sections in TOC

**CR2: Highlight baseline rows in cohort comparison tables**
- Modified `_build_cohort_summary_table()` function to accept `highlight_baseline_rows` parameter
- Added pale yellow background color (`#FFFFD9`) to baseline rows in all three cohort comparison tables
- Added "Baseline" swatch to shading legend alongside "Above baseline" and "Below baseline"
- Files modified:
  - `compose_pdf.py` lines 122-124 (added BASELINE_YELLOW constant)
  - `compose_pdf.py` lines 1460-1515 (function modification)
  - `compose_pdf.py` lines 1895-1939 (table calls updated)
  - `compose_pdf.py` lines 3155-3170 (legend updated)
- Improves readability by clearly identifying which row is the comparison baseline

**CR3: Move Statistical Associations text to external file**
- Created `STATISTICAL_ASSOCIATIONS` section in `report_text.txt`
- Replaced dynamically-generated statistical analysis with static summary text
- Removed dependency on `run_all_analyses()` and `format_results_for_report()`
- Files modified:
  - `report_text.txt` lines 266-278 (new section)
  - `compose_pdf.py` line 1781 (simplified loading)
- Benefits: Easier to edit, faster PDF generation, consistent messaging

**CR4: Move Year-over-Year (YoY) explanation to external file**
- Created `SECTION1_YOY_EXPLANATION` section in `report_text.txt`
- Moved hardcoded explanation text to external file
- Files modified:
  - `report_text.txt` lines 280-284
  - `compose_pdf.py` line 1784
- Enables easy editing of YoY chart explanations without code changes

**CR5: Move CAGR explanation to external file**
- Created `SECTION1_CAGR_EXPLANATION` section in `report_text.txt`
- Moved hardcoded explanation text to external file
- Files modified:
  - `report_text.txt` lines 286-290
  - `compose_pdf.py` line 1785
- Enables easy editing of CAGR chart explanations without code changes

**CR6: Move Distribution explanation to external file**
- Created `SECTION1_DISTRIBUTION_EXPLANATION` section in `report_text.txt`
- Converted dynamic f-string cohort labels to static text with actual cohort names
- Files modified:
  - `report_text.txt` lines 292-296
  - `compose_pdf.py` line 1980
- Simplifies text management and editing

**CR7: Create separate Appendix A text file**
- Created new file `appendix_a_text.txt` containing all methodology sections:
  - METHODOLOGY_DATA_SOURCES
  - METHODOLOGY_PPE_DEFINITION
  - METHODOLOGY_SHADING_LOGIC
  - METHODOLOGY_NSS_CH70
- Modified `compose_pdf.py` to load from both `report_text.txt` and `appendix_a_text.txt`
- Files created/modified:
  - `appendix_a_text.txt` (new file, 215 lines)
  - `compose_pdf.py` lines 1724-1727 (load and merge appendix text)
- Benefits: Better file organization, separates methodology from report content

**CR8: Rename appendix file**
- Renamed `appendix_c_text.txt` to `appendix_b_text.txt`
- Updated code reference in `compose_pdf.py` line 2395
- Aligns filename with actual appendix label in PDF

**CR9: Add page numbers to Table of Contents**
- Status: COMPLETED
- Implemented two-pass PDF generation to calculate and display page numbers
- Technical implementation:
  1. Created `PageMarker` Flowable class (lines 90-101) that records page numbers when rendered
  2. Added global `_PAGE_MAP` dictionary and helper functions (lines 87-110)
  3. Modified TOC rendering to display page numbers with dotted leaders (lines 3044-3070)
  4. Added PageMarker insertions at 5 key locations where section_ids are defined:
     - Threshold analysis pages (line 2994)
     - Text-only pages (line 3092)
     - Executive summary cohort table pages (line 3146)
     - Appendix title pages (line 3235)
     - Regular page titles (line 3248)
  5. Modified main() function to implement two-pass build (lines 3544-3565):
     - Pass 1: Build temp PDF to populate _PAGE_MAP
     - Pass 2: Rebuild final PDF with page numbers in TOC
     - Error handling for locked PDF files
- Files modified:
  - `compose_pdf.py` (PageMarker infrastructure, TOC rendering, main function)
- Benefits: Users can now navigate directly to sections using page numbers in TOC
- Notes: Captures 21 sections with page numbers; requires closing existing PDF before regenerating

### Summary of Changes

**Files Modified:**
- `compose_pdf.py` (multiple sections)
- `report_text.txt` (added 6 new sections)

**Files Created:**
- `appendix_a_text.txt` (new methodology file)
- `work_log.md` (this file)

**Files Renamed:**
- `appendix_c_text.txt` → `appendix_b_text.txt`

**Impact:**
- Improved visual clarity in PDF (baseline highlighting, TOC spacing)
- Better text management (externalized hardcoded strings)
- Simplified codebase (removed statistical analysis generation)
- Better file organization (separate appendix file)

**Testing Needed:**
- Verify all three cohort tables show yellow baseline rows ✓
- Verify "Baseline" swatch appears in legend ✓
- Verify TOC has proper spacing before Section 3 ✓
- Verify all externalized text renders correctly ✓
- Verify Appendix A and B load properly ✓
- Verify TOC displays accurate page numbers for all sections ✓
- Test CR9 by closing existing PDF and regenerating

---

## 2025-10-24 - Major NSS/Ch70 Overhaul and Report Improvements

### Completed Change Requests

**CR14: Add report title/subtitle above Table of Contents**
- Added "Western MA Per Pupil Spending Report" title above ToC
- Added "With selected districts for comparison" subtitle
- Files modified:
  - `compose_pdf.py` lines 3023-3031 (ToC page dict), lines 3082-3103 (rendering)

**CR15: Change 'Specific districts' to 'Selected districts'**
- Updated terminology throughout report for consistency
- Files modified:
  - `compose_pdf.py` line 3009 (ToC), line 2325 (Section 3 title)

**CR16: Remove Foundation Enrollment from NSS/Ch70 pages**
- Removed foundation enrollment table from all NSS/Ch70 pages
- Removed foundation enrollment line (blue) from stacked bar plots
- Removed left y-axis (enrollment axis) from plots
- Removed FTE table rendering from NSS/Ch70 page type
- Files modified:
  - `compose_pdf.py` lines 2177-2201 (Western cohort NSS pages)
  - `compose_pdf.py` lines 2294-2337 (individual district NSS pages)
  - `compose_pdf.py` lines 3560-3562 (removed FTE table rendering)
  - `nss_ch70_main.py` lines 199-209 (pass foundation_enrollment=None)
- Result: Cleaner, more focused NSS/Ch70 pages with single y-axis for dollars

**CR17: Convert NSS/Ch70 from per-district to per-pupil**
- **MAJOR CHANGE**: Converted all NSS/Ch70 values from absolute dollars to per-pupil dollars
- Modified data preparation functions:
  - `school_shared.py::prepare_aggregate_nss_ch70_weighted()`: Return per-pupil weighted averages instead of per-district
  - `school_shared.py::prepare_district_nss_ch70()`: Divide by enrollment to return per-pupil values
- Updated plot labels and documentation:
  - `nss_ch70_plots.py`: Updated y-axis labels to "Weighted avg $ per pupil" / "$ per pupil"
  - `appendix_a_text.txt` METHODOLOGY_NSS_CH70 section: Updated to reflect per-pupil methodology
- Files modified:
  - `school_shared.py` lines 1228-1359 (aggregate function docstring and calculation)
  - `school_shared.py` lines 1084-1175 (district function docstring and calculation)
  - `nss_ch70_plots.py` lines 1-9, 74-90, 125-128 (docstrings and labels)
  - `appendix_a_text.txt` lines 142-171 (methodology documentation)
- Example values (Amherst 2024):
  - Old: ~$31M total NSS per district
  - New: ~$18,313 per pupil ($31M / 1,721 students)
- Benefit: Direct comparability with PPE values, consistent reporting methodology

**CR19: Update cohort comparison subtitle**
- Changed subtitle to "Total PPE comparison: Western MA enrollment cohorts and selected districts"
- File modified:
  - `compose_pdf.py` line 1943

**CR20: Remove subtitles from cohort comparison tables**
- Removed "District comparison: Medium enrollment cohort" and similar subtitles
- Files modified:
  - `compose_pdf.py` lines 1963, 1988

**CR21: Add Executive Summary subsections to ToC**
- Added indented subsections:
  - "Total PPE comparison: Western MA enrollment cohorts and selected districts"
  - "Per-pupil expenditure and recent growth overview: 2019 PPE to 2024 PPE"
  - "Statistical Associations"
- Files modified:
  - `compose_pdf.py` lines 2998-3000

**CR22: Add Section 1 subsections to ToC**
- Added indented subsections:
  - "Year-over-Year (YoY) growth rates by district and cohort"
  - "5-year and 15-year CAGR by district and cohort"
  - "Distribution of 2024 enrollment and proposed cohort grouping"
  - "Scatterplot of enrollment vs. per-pupil expenditure with quartile boundaries (2024)"
  - "Geographic map showing district locations and enrollment cohorts (2024)"
- Files modified:
  - `compose_pdf.py` lines 3003-3007

**CR23: Change 'Unified regions' to 'Regions'**
- Updated terminology in NSS/Ch70 district list text
- File modified:
  - `compose_pdf.py` line 2132

### Summary of Changes

**Files Modified:**
- `compose_pdf.py` (multiple sections - page building, rendering, ToC)
- `school_shared.py` (NSS/Ch70 data preparation functions)
- `nss_ch70_plots.py` (plot labels and documentation)
- `nss_ch70_main.py` (plot generation calls)
- `appendix_a_text.txt` (methodology documentation)

**Impact:**
- Major architectural change: NSS/Ch70 now uses per-pupil values throughout
- Cleaner NSS/Ch70 pages without extraneous foundation enrollment information
- Improved ToC with hierarchical subsections for better navigation
- Consistent terminology ("Selected districts", "Regions")
- Better report structure with title/subtitle above ToC

**Testing Needed:**
- Verify NSS/Ch70 plots show per-pupil values (not per-district millions)
- Verify no foundation enrollment tables appear on NSS/Ch70 pages
- Verify no blue enrollment line appears on NSS/Ch70 plots
- Verify single y-axis (dollars) on all NSS/Ch70 plots
- Verify ToC shows hierarchical subsections with proper indentation
- Verify report title/subtitle appears above ToC
- Verify "Selected districts" terminology throughout report
- Generate full PDF and verify all pages render correctly

---

## 2025-10-24 - ToC Anchor Link Fixes

### Bug Fixes

**Missing ToC Anchor Links:**
- **Issue:** After adding hierarchical subsections to ToC (CR21, CR22), PDF generation failed with:
  `ValueError: format not resolved, probably missing URL scheme or undefined destination target for 'exec_summary_cohort_comparison'`
- **Root Cause:** ToC entries created hyperlinks with section_id values, but the actual pages didn't have corresponding section_id fields to create anchor targets
- **Fix:** Added missing section_id fields to executive summary pages and fixed section_id mismatch in Section 1
- Files modified:
  - `compose_pdf.py` line 1948: Added `section_id="exec_summary_cohort_comparison"` to cohort comparison table page
  - `compose_pdf.py` line 1845: Added `section_id="exec_summary_scatter"` to PPE overview/scatterplot page
  - `compose_pdf.py` line 2008: Added `section_id="exec_summary_statistical"` to statistical associations page
  - `compose_pdf.py` line 2081: Changed `section_id="section1_scatterplot"` to `section_id="section1_scatter"` to match ToC
- Result: All ToC subsection links now have corresponding page anchors

**ToC Table Formatting:**
- **Issue:** ToC used Paragraph elements with dotted leaders, resulting in misaligned page numbers and uneven spacing
- **Solution:** Converted ToC to ReportLab Table with two columns (title, page number)
- Files modified:
  - `compose_pdf.py` lines 3106-3194: Replaced Paragraph-based ToC with Table-based implementation
- Key improvements:
  - Page numbers right-aligned in dedicated column for perfect alignment
  - Uniform row spacing controlled by table padding (3pt top/bottom)
  - Indentation handled via LEFTPADDING (0pt for main, 20pt for subsections, 40pt for sub-subsections)
  - Gray divider lines (0.5pt, 70% gray) after main sections using LINEBELOW
  - All table borders invisible except gray section dividers
  - Cleaner, more professional appearance with precise control over spacing
- Result: Professional table-based ToC with aligned page numbers and subtle section dividers

**CAGR Chart Diagonal Line Width:**
- **Issue:** Diagonal white lines (hatch pattern edges) on CAGR bar charts were too thin and hard to see
- **Solution:** Increased linewidth for both cohorts and individual districts in both CAGR charts
- Files modified:
  - `executive_summary_plots.py` line 667: 5-year CAGR cohorts - increased from linewidth=2 to linewidth=3.5
  - `executive_summary_plots.py` line 671: 5-year CAGR districts - increased from linewidth=0.5 to linewidth=1.2
  - `executive_summary_plots.py` line 778: 15-year CAGR cohorts - increased from linewidth=2 to linewidth=3.5
  - `executive_summary_plots.py` line 782: 15-year CAGR districts - increased from linewidth=0.5 to linewidth=1.2
- Result: More visible diagonal white lines make hatching pattern more distinct

---

## 2025-10-24 - Section Intro Pages and New Choropleth Maps (CR+1, CR+2, CR+3)

### New Features

**CR+1: Section 2 Intro Page**
- Added placeholder intro page for Section 2 with summary text and navigation link
- Files modified:
  - `report_text.txt`: Added SECTION2_SUMMARY section (lines 350-362)
  - `compose_pdf.py` lines 2110-2129: Added Section 2 intro page with section_id="section2_intro"
  - `compose_pdf.py` lines 3257-3268: Added page number placeholder replacement logic for dynamic navigation
- Content: Describes cohort detail pages (PPE, NSS/Ch70 by cohort) with link to Section 3
- Result: Users can quickly navigate between major sections using page-linked navigation

**CR+2: Section 3 Intro Page**
- Added placeholder intro page for Section 3 with summary text and navigation link
- Files modified:
  - `report_text.txt`: Added SECTION3_SUMMARY section (lines 364-374)
  - `compose_pdf.py` lines 2229-2248: Added Section 3 intro page with section_id="section3_intro"
  - Page number replacement logic reuses implementation from CR+1
- Content: Describes selected district pages with link to Appendix A
- Result: Better report navigation and context for users jumping to specific sections

**CR+4: NSS/Ch70 Table Spacing**
- **Issue:** Insufficient spacing between NSS/Ch70 table and cohort member list on Western MA cohort pages
- **Solution:** Increased spacing from 6pt to 18pt
- Files modified:
  - `compose_pdf.py` line 3662: Added `Spacer(0, 18)` before district_list_text rendering
- Result: Better visual separation between table and cohort membership details

**CR+5: Mini ToC Navigation Box**
- **Feature:** Added boxed navigation menu to Executive Summary intro page with links to all major sections
- Files modified:
  - `compose_pdf.py` lines 1810-1826: Created mini ToC with 9 major section links
  - `compose_pdf.py` lines 1830-1835: Integrated mini ToC into exec summary text blocks using `__BOXED_START__/__BOXED_END__` markers
  - `compose_pdf.py` lines 3292-3299: Added regex-based page number placeholder replacement for mini ToC
- Content: Links to Table of Contents, Executive Summary, Sections 1-3, and Appendices A-D
- Result: Users can quickly navigate to major sections from the intro page

**Choropleth Text Update**
- **Issue:** Choropleth explanation text needed to be in external file with updated content
- Files modified:
  - `report_text.txt` lines 292-294: Updated SECTION1_CHOROPLETH_EXPLANATION with new concise text
- New text emphasizes geography's role in school costs and clarifies map symbols
- Result: More accessible explanation of choropleth visualization

**CR+3: New Choropleth Maps (COMPLETED)**
- **Feature:** Added two new choropleth types showing district comparisons to cohort baselines
- **Type 1 - PPE vs Baseline Choropleth:**
  - Colors districts based on % deviation from cohort average PPE
  - Dark green (>10% below), light green (0-10% below), light red (0-10% above), dark red (>10% above)
  - Generated for years: 2024, 2019, 2014, 2009
- **Type 2 - CAGR vs Baseline Choropleth:**
  - Colors districts based on percentage point deviation from cohort average CAGR
  - Dark blue (>1pp slower), light blue (0-1pp slower), light orange (0-1pp faster), dark orange (>1pp faster)
  - Generated for periods: 2009-2024 (15yr), 2009-2019 (10yr), 2009-2014 (5yr)
  - No map for 2009 (baseline year)
- **Files Modified:**
  - `western_map.py` lines 46-60: Added imports for `prepare_district_epp_lines` and `weighted_epp_aggregation`
  - `western_map.py` lines 564-672: Added `create_ppe_comparison_map()` function (109 lines)
  - `western_map.py` lines 675-777: Added `create_cagr_comparison_map()` function (103 lines)
  - `western_map.py` lines 779-897: Added `calculate_ppe_comparison_to_cohort()` function (119 lines)
  - `western_map.py` lines 900-1006: Added `calculate_cagr_comparison_to_cohort()` function (107 lines)
  - `western_map.py` lines 924-998: Modified `main()` to generate all three choropleth types for each year
  - `report_text.txt` lines 296-314: Added `SECTION1_PPE_COMPARISON_EXPLANATION` and `SECTION1_CAGR_COMPARISON_EXPLANATION`
  - `compose_pdf.py` lines 2121-2147: Added PPE and CAGR comparison map pages to Section 1
  - `compose_pdf.py` lines 3090-3091: Added Section 1 ToC entries for new maps
  - `compose_pdf.py` lines 3071-3106: Added PPE and CAGR comparison maps to Appendix D for historical years
- **Generated Maps:**
  - `western_ma_ppe_comparison_2024.png`
  - `western_ma_ppe_comparison_2019.png`
  - `western_ma_ppe_comparison_2014.png`
  - `western_ma_ppe_comparison_2009.png`
  - `western_ma_cagr_comparison_2009_2024.png`
  - `western_ma_cagr_comparison_2009_2019.png`
  - `western_ma_cagr_comparison_2009_2014.png`
- **Result:** Users can now visualize which districts spend above/below their cohort average and which districts grew faster/slower than their cohort average
- **Testing:** All maps generated successfully with proper color coding and district matching

### Summary of Changes

**Files Modified:**
- `executive_summary_plots.py` (CAGR chart diagonal line width increases)
- `report_text.txt` (added Section 2/3 summaries, choropleth explanations)
- `compose_pdf.py` (Section 2/3 intro pages, mini ToC, page number placeholders, NSS/Ch70 spacing, new choropleth pages)
- `western_map.py` (major additions for PPE and CAGR comparison choropleths)

**Impact:**
- Enhanced navigation: Section intro pages with dynamic page number links
- Improved report structure: Mini ToC on Executive Summary, better visual spacing
- New analytical capability: PPE and CAGR comparison choropleths show district performance relative to cohort baselines
- Better data density: 10 new choropleth maps (PPE comparisons for 4 years, CAGR comparisons for 3 periods) added across Section 1 and Appendix D

**Testing Needed:**
- Verify all new maps render correctly in Section 1 and Appendix D
- Verify page number placeholders are replaced correctly in intro pages and mini ToC
- Verify NSS/Ch70 table spacing is adequate
- Generate full PDF and verify all pages render correctly

---

## 2025-10-25 - HTML Tag Fix

### Bug Fixes

**Malformed HTML tags in report_text.txt:**
- **Issue:** PDF composition failed with multiple reportlab HTML parser errors:
  1. `ValueError: Parse error: saw </i> instead of expected </para>` (line 141)
  2. `ValueError: Parse error: saw </b> instead of expected </para>` (line 378)
  3. Mismatched bold tag on line 381
- **Root Causes:**
  1. Line 141 had a closing `</i>` tag at the beginning without a matching opening `<i>` tag
  2. Line 378 had an extra `</b>` tag after the `##H1` heading (heading syntax already adds bold)
  3. Line 381 had `<b>` at the end instead of `</b>`
- **Fixes:**
  - `report_text.txt` line 141: Changed `</i>Note: ...` to `<i>Note: ...`
  - `report_text.txt` line 378: Removed extra `</b>` tag from H1 heading
  - `report_text.txt` line 381: Changed closing `<b>` to `</b>`
- **Result:** All HTML tags are now properly formed and reportlab can parse them correctly

---

## 2025-10-25 - Major Report Restructuring (CR01-CR12,002)

### Completed Change Requests

**CR01: Fix {TODAY_DATE} placeholder**
- **Issue:** Placeholder not being replaced with actual date
- **Fix:** Added date replacement logic in build_pdf() function (compose_pdf.py:3803-3807)
- **Result:** Date now displays correctly in footer

**CR02: Change ToC title for Statistical Associations**
- Updated from "Statistical Associations" to "Statistical Associations between Enrollment and Per-Pupil Expenditures"
- compose_pdf.py:3543

**CR03: Move Statistical Associations to end of Section 1**
- Removed from Executive Summary
- Added as final page of Section 1 with new section_id="section1_statistical"
- compose_pdf.py:2523-2564
- Updated ToC accordingly

**CR04: Add Western MA aggregate pages at end of Section 2**
- Added PPE and NSS/Ch70 pages for "Western MA (all, excl. Springfield)"
- No baseline shading (this IS the baseline)
- compose_pdf.py:2702-2779

**CR05: Add 5 blank lines above Report Navigation boxes**
- Added to Executive Summary, Section 1, Section 2, and Section 3 intro pages
- compose_pdf.py:2210, 2410, 2595, 2800

**CR06: Move PPE overview to second page of Section 1**
- Moved from Executive Summary to Section 1
- Now section_id="section1_ppe_overview"
- compose_pdf.py:2238-2246, 2419-2420

**CR07: Change PPE overview from 2019-2024 to 2009-2024**
- Changed t0 from `latest - 5` to `latest - 15`
- compose_pdf.py:2167
- district_expend_pp_stack.py:779 (changed year_lag from 5 to 15)
- Removed "2024 decrease from 2019" legend item
- district_expend_pp_stack.py:512-518

**CR 12,001: Remove "Page" prefix from page numbers**
- Changed footer from "Page {n}" to "{n}"
- compose_pdf.py:207

**CR 12,002: Fix missing Western MA aggregate chart images**
- Added plot generation for "all_western" aggregate in district_expend_pp_stack.py:776-786
- Added NSS/Ch70 plot generation in nss_ch70_main.py:97-104, 154-197
- **Files Modified:**
  - district_expend_pp_stack.py (added all_western PPE plot generation)
  - nss_ch70_main.py (added all_western NSS/Ch70 data preparation and plot generation)

### Summary

**Files Modified:**
- compose_pdf.py (major restructuring)
- district_expend_pp_stack.py (PPE overview timeframe, all_western plot)
- nss_ch70_main.py (all_western NSS/Ch70 plot)
- work_log.md (this file)

**Impact:**
- Report structure significantly reorganized
- Statistical Associations moved from Executive Summary to Section 1
- PPE overview moved from Executive Summary to Section 1 with 15-year timeframe
- Western MA aggregate pages added to Section 2
- Page numbers simplified
- All navigation boxes now have proper spacing

---

## 2025-10-24 - Report Navigation Improvements (CR 1004)

### Completed Change Requests

**CR 1004: Add Report Navigation box to Section intro pages**
- Extended the Report Navigation box (previously only on Executive Summary) to Section 1, 2, and 3 intro pages
- Added boxed mini ToC with hyperlinks to all major report sections on each section's first page
- Enables readers to quickly navigate the report from any section intro page
- **Files Modified:**
  - `compose_pdf.py` lines 2040-2044: Added mini ToC to Section 1 intro page
  - `compose_pdf.py` lines 2181-2185: Added mini ToC to Section 2 intro page
  - `compose_pdf.py` lines 2306-2310: Added mini ToC to Section 3 intro page
- **Implementation:**
  - Reused existing `mini_toc_text` list (built at lines 1823-1826)
  - Added `["__BOXED_START__"] + mini_toc_text + ["__BOXED_END__"]` to text_blocks for each section intro
  - All page number placeholders are resolved during PDF rendering via regex replacement
- **Result:** Improved navigation - readers can jump to any major section from Executive Summary or any Section 1/2/3 intro page
- **Testing:** PDF generation should verify navigation boxes appear on all four intro pages with correct page numbers

---

## 2025-10-24 - Choropleth Enhancements and Cohort Distribution Statistics (CR 1003, CR 1005)

### Completed Change Requests

**CR 1003: Add secondary regional indicators to comparison choropleths**
- Added +/- text labels to show secondary regional district deviations on PPE and CAGR comparison maps
- Secondary regional districts overlap with elementary districts geographically, so their values would otherwise be invisible
- **Files Modified:**
  - `western_map.py` line 43: Added `import matplotlib.patheffects as patheffects`
  - `western_map.py` lines 627-654: Added text labels for secondary regional districts on PPE comparison maps
    - Format: "+X%" or "-X%" showing deviation from cohort baseline
    - White text with black outline for visibility across all background colors
    - Positioned at district centroids
  - `western_map.py` lines 774-800: Added text labels for secondary regional districts on CAGR comparison maps
    - Format: "+X.Xpp" or "-X.Xpp" showing percentage point deviation
    - Same styling as PPE maps
  - `western_map.py` lines 676-684: Updated PPE comparison legend to explain secondary regional indicators
    - Added legend entry: "+/-% = Secondary regional district deviation (n=X)"
    - Changed ncol from 3 to 2 to accommodate additional entry
  - `western_map.py` lines 819-827: Updated CAGR comparison legend
    - Added legend entry: "+/-pp = Secondary regional district deviation (n=X)"
    - Added "(±1pp threshold)" to legend title for clarity
- **Result:** Secondary regional districts now visible on comparison choropleths with clear numeric indicators
- **Note:** Threshold colors (blue/orange) and white-for-within-threshold were already correctly implemented

**CR 1005: Add cohort distribution statistics with box-and-whisker plots**
- Added new "Distribution of PPE and Growth Rates by Enrollment Cohort" section to STATISTICAL_ASSOCIATIONS
- Shows five-number summaries (min, Q1, median, Q3, max) for both 2024 PPE and 2009-2024 CAGR by cohort
- Includes horizontal mini box-and-whisker plots for visual comparison
- **Files Modified:**
  - `compose_pdf.py` lines 725-866: Added three new functions
    - `calculate_cohort_ppe_distribution()`: Calculate five-number summary of 2024 PPE by cohort
    - `calculate_cohort_cagr_distribution()`: Calculate five-number summary of 2009-2024 CAGR by cohort
    - `create_mini_boxplot()`: Generate horizontal mini box-and-whisker plot PNGs with cohort colors
  - `compose_pdf.py` lines 868-960: Added `build_cohort_distribution_table()` function
    - Builds table with columns: Cohort | n | Min | Q1 | Median | Q3 | Max | Distribution
    - Embeds mini boxplot images (2.5" x 0.3") in rightmost column
    - Formats PPE as currency ($X,XXX), CAGR as percentage (X.X%)
    - Uses established cohort colors for boxplots
  - `compose_pdf.py` lines 2262-2299: Integrated cohort distribution tables into STATISTICAL_ASSOCIATIONS page
    - Calculate PPE and CAGR distributions
    - Generate mini boxplots for each cohort (saved to temp_boxplots/)
    - Build tables and replace __COHORT_PPE_TABLE__ and __COHORT_CAGR_TABLE__ placeholders
  - `report_text.txt` lines 353-370: Added new text sections
    - "Distribution of PPE and Growth Rates by Enrollment Cohort" introduction
    - "2024 PPE by Cohort" subsection with explanation
    - "2009-2024 PPE CAGR by Cohort" subsection with explanation
    - Interpretive paragraph explaining how to compare patterns
- **Result:** Executive Summary now includes detailed cohort-level distribution analysis with visual boxplots
- **Testing:** Verify mini boxplots render correctly in STATISTICAL_ASSOCIATIONS section with proper cohort colors

### Summary of Changes

**Files Modified:**
- `western_map.py` (secondary regional indicators on choropleths)
- `compose_pdf.py` (cohort distribution functions and integration)
- `report_text.txt` (new cohort distribution text sections)

**Impact:**
- Enhanced choropleths: Secondary regional districts now visible with +/- indicators
- New analytical capability: Cohort distribution statistics with embedded box-and-whisker plots
- Better understanding of within-cohort variation in PPE levels and growth rates
- Visual comparison of cohort distributions using established color scheme

**Testing Needed:**
- Generate all choropleth maps and verify secondary regional indicators appear with correct values
- Verify legend entries explain the +/- indicators
- Generate full PDF and verify cohort distribution tables appear in STATISTICAL_ASSOCIATIONS section
- Verify mini boxplots use correct cohort colors and display five-number summaries accurately
- Verify table formatting (currency for PPE, percentage for CAGR)

---

## 2025-10-25 - Multiple Report Improvements and Comment Syntax

### Completed Changes

**Text File Comment and Heading Syntax (New Features)**
- Added support for commenting out content in report_text.txt and appendix_a_text.txt without deleting it
- Two comment syntaxes available:
  - **Single-line comments**: `##COMMENT This line is hidden`
  - **Multi-line block comments**:
    ```
    ##BEGIN_COMMENT
    Multiple lines
    can be hidden
    ##END_COMMENT
    ```
- Added support for custom heading syntax:
  - **H1 (large heading)**: `##H1 Heading Text` (13pt bold, 12pt space before, 6pt after)
  - **H2 (medium heading)**: `##H2 Heading Text` (11pt bold, 10pt space before, 4pt after)
  - **H3 (small heading)**: `##H3 Heading Text` (10pt bold, 8pt space before, 3pt after)
- Comment and heading markers must be on their own line (leading/trailing whitespace OK)
- Headings work both inside and outside boxed sections
- **Files Modified:**
  - `compose_pdf.py` lines 336-393: Added comment filtering and heading parsing logic to `load_report_text_sections()`
  - `compose_pdf.py` lines 395-423: Updated paragraph building to handle heading tuples
  - `compose_pdf.py` lines 188-191: Added heading paragraph styles (style_h1, style_h2, style_h3)
  - `compose_pdf.py` lines 3843-3852: Added heading rendering in main text flow
  - `compose_pdf.py` lines 3802-3816: Added heading rendering in boxed sections
  - `compose_pdf.py` lines 434-460: Updated `fill_text_placeholders()` to handle heading tuples
  - `compose_pdf.py` lines 311-325: Updated docstring with comment and heading syntax documentation

**Executive Summary Improvements**
- Split EXECUTIVE_SUMMARY_QUICK_START section properly
- Created new EXECUTIVE_SUMMARY_CONTEXT section for content after quick-start guide
- Added space before Report Navigation box
- **Files Modified:**
  - `report_text.txt` lines 28-30: Added EXECUTIVE_SUMMARY_CONTEXT section delimiter
  - `compose_pdf.py` lines 2052, 2081-2083: Load and render new context section with spacing

**Statistical Associations Page**
- Removed unnecessary page break between distribution tables and detailed results
- All content now flows naturally on one page
- **Files Modified:**
  - `compose_pdf.py` lines 2298-2315: Simplified to single page with all content

**Choropleth Map Improvements**
- Changed secondary regional districts to transparent (no fill) with black borders only
- Underlying elementary district colors now show through secondary regional boundaries
- Secondary regional +/- indicators remain visible with black text
- More muted colors for comparison maps (blue/orange less saturated)
- Gray boundaries added to all districts for better visibility
- **Files Modified:**
  - `western_map.py` lines 602, 761: Changed to muted colors `#B3E0E6` and `#FFE6CC`
  - `western_map.py` lines 613-648: Modified PPE comparison to filter out secondary regionals from color fills
  - `western_map.py` lines 640-665: Added transparent secondary regional districts with black borders
  - `western_map.py` lines 772-824: Same changes for CAGR comparison maps

**Chapter 70/NSS Improvements**
- Updated table headers to show "$/pupil" explicitly (2009 $/pupil and 2024 $/pupil)
- Restored shading legend below NSS tables (was accidentally removed)
- Changed NSS plot y-axis to 0-$30K to match PPE plot range
- **Files Modified:**
  - `compose_pdf.py` line 1301, 1305: Updated NSS table headers
  - `compose_pdf.py` lines 1330-1467: Restored legend rows with shading rules and color swatches
  - `nss_ch70_main.py` line 204: Changed `right_ylim=None` to `right_ylim=30000`

**Appendix A Page Layout**
- Combined METHODOLOGY_DATA_SOURCES and METHODOLOGY_PPE_DEFINITION onto one page
- Removed unnecessary page break between "Regional Classifications" and "PPE Definition"
- **Files Modified:**
  - `compose_pdf.py` lines 2849-2857: Combined both sections into single page

**Dynamic Date Variable**
- Added {TODAY_DATE} placeholder that automatically displays current date when report is generated
- Format: "October 25, 2025" (Month Day, Year)
- Replaces hardcoded date in draft status line
- Updates automatically with each report generation
- **Files Modified:**
  - `compose_pdf.py` line 20: Added `from datetime import datetime` import
  - `compose_pdf.py` lines 2901-2904: Created today_date variable and added to placeholders dictionary
  - `report_text.txt` line 110: Changed hardcoded date to {TODAY_DATE} placeholder
- **Usage:** Any text file can now use {TODAY_DATE} to display current date in long format

**CAGR Threshold Change (1.0pp → 0.5pp)**
- Changed CAGR (growth rate) threshold from 1.0 percentage point to 0.5 percentage points
- More sensitive threshold reflects compound growth impact over time
- 0.5pp difference in annual growth = 11.5% more total growth over 15 years ($2,299/pupil)
- **Files Modified:**
  - `compose_pdf.py` line 137: Changed `MATERIAL_DELTA_PCTPTS = 0.01` to `0.005`
  - `compose_pdf.py` line 144: Updated comment "0.5pp base"
  - `compose_pdf.py` line 145: Changed `SHADE_BINS_CAGR = [0.01, 0.02, 0.03, 0.04]` to `[0.005, 0.01, 0.015, 0.02]`
  - `compose_pdf.py` lines 2016-2150: Updated threshold analysis page function
    - Function docstring: "5% / 0.5pp thresholds"
    - Selected scenario: `("Selected (5%/0.5pp)", 0.05, 0.5, 0.22, 0.15, 0.70, ...)`
    - All explanation text blocks updated to reflect 0.5pp
    - Shading bins: "0.5pp (lightest), 1pp, 1.5pp, 2pp+ (darkest)"
  - `compose_pdf.py` line 2969: Subtitle "5% / 0.5pp Shading Thresholds"
  - `compose_pdf.py` lines 3223-3250: Updated Appendix B threshold calculation examples
  - `western_map.py` line 757: Comment updated to "0.5pp threshold"
  - `western_map.py` line 759: Changed `bins = [-100, -1, 1, 100]` to `[-100, -0.5, 0.5, 100]`
  - `western_map.py` line 761: Comment "Within ±0.5pp"
  - `western_map.py` line 762: Labels `[">0.5pp slower", "Within ±0.5pp", ">0.5pp faster"]`
  - `western_map.py` line 864: Title "(±0.5pp threshold)"
- **Result:** More granular CAGR shading in tables and choropleths, better captures long-term trajectory differences

**New Appendix Section: Threshold Calibration**
- Added METHODOLOGY_THRESHOLD_CALIBRATION section to Appendix A explaining threshold rationale
- Content provided by user from Claude Chat collaboration
- **Files Modified:**
  - `appendix_a_text.txt` lines 129-158: New section between METHODOLOGY_SHADING_LOGIC and METHODOLOGY_NSS_CH70
- **Content:**
  - Explains why different metrics use different sensitivities (PPE 5%, Enrollment 5%, CAGR 0.5pp)
  - Shows compound growth impact with concrete example ($20K PPE at 4.0% vs 4.5% over 15 years)
  - Demonstrates policy significance ($689,700 additional spending for 300-student district)
  - Provides statistical rationale (CV = 54% for CAGR vs 22.5% for PPE)
- **Result:** Clear explanation of threshold calibration methodology accessible to non-technical readers

**New Callout Box: CAGR Sensitivity Explanation**
- Added boxed sidebar to Section 1 CAGR comparison map explanation
- Simplified version of threshold calibration content for in-context reference
- **Files Modified:**
  - `report_text.txt` lines 361-371: Added __BOXED_START__/__BOXED_END__ section with ##H3 heading
- **Content:**
  - Shows compound growth example (4.0% vs 4.5% over 15 years)
  - Calculates total impact ($2,299/pupil gap, $689,700 for 300-student district)
  - Explains why 0.5pp threshold identifies meaningfully different trajectories
- **Result:** Users see the rationale for sensitive CAGR threshold right where they encounter it

**Documentation Updates**
- Updated all references from "5%/1pp" to "5%/0.5pp" throughout codebase
- **Files Modified:**
  - `appendix_a_text.txt` line 127: Statistical rationale "5%/0.5pp thresholds"
  - `appendix_a_text.txt` line 206: NSS/Ch70 shading "0.5pp CAGR threshold"
  - `report_text.txt` line 242: Statistical rationale "5%/0.5pp thresholds"
  - `report_text.txt` line 291: NSS/Ch70 shading "0.5pp CAGR threshold"
  - `report_text.txt` lines 357-359: CAGR comparison color coding updated to reflect 0.5pp bins

### Summary

**Files Modified:**
- `compose_pdf.py` (comment syntax, heading syntax, executive summary, statistical associations, NSS legend, appendix A layout, dynamic date variable, CAGR threshold changes throughout)
- `report_text.txt` (split executive summary sections, added TODAY_DATE placeholder, updated CAGR references, added CAGR sensitivity callout box)
- `appendix_a_text.txt` (added METHODOLOGY_THRESHOLD_CALIBRATION section, updated CAGR threshold references)
- `western_map.py` (choropleth transparency and colors, CAGR threshold from 1pp to 0.5pp)
- `nss_ch70_main.py` (plot y-axis range)

**New Features:**
- Comment syntax for text files allows temporarily hiding content without deletion (##COMMENT and ##BEGIN_COMMENT...##END_COMMENT)
- Custom heading syntax (##H1, ##H2, ##H3) provides proper hierarchical heading styles in text files
- Dynamic date variable ({TODAY_DATE}) automatically displays current date when report is generated
- New METHODOLOGY_THRESHOLD_CALIBRATION appendix section explaining threshold sensitivity rationale
- New callout box in Section 1 explaining why CAGR threshold is more sensitive (compound growth impact)

**Impact:**
- Better text file maintenance with comment and heading capabilities
- Improved page layouts and flow throughout the document
- Choropleth maps now properly show secondary regionals as transparent overlays
- NSS tables and plots more consistent with PPE displays
- Text content can now have proper hierarchical structure with styled headings
- Report draft status date updates automatically with each generation
- **More sensitive CAGR threshold (0.5pp vs 1pp) better captures long-term compound growth differences**
- Enhanced documentation explaining threshold calibration for both technical and non-technical readers
- Readers can understand why small growth rate differences matter when viewing CAGR comparison maps

## 2025-10-25 - Foundation Enrollment Restoration and NSS/Ch70 Executive Summary Tables

### Completed Change Requests (CR08-CR11, CR 12,001-12,002)

**CR08: Restore Foundation Enrollment to NSS/Ch70 Pages**
- Reversed previous CR16 which had removed Foundation Enrollment from NSS/Ch70 pages
- Added Foundation Enrollment tables to all NSS/Ch70 pages (cohort aggregates, Western MA aggregate, and individual districts)
- Restored Foundation Enrollment line and left y-axis to all NSS/Ch70 plots in nss_ch70_main.py
- Updated all NSS/Ch70 enrollment labels to explicitly specify "Foundation Enrollment" vs "In-district FTE" for PPE
- Impact: Clear distinction between Foundation Enrollment (state aid denominator) and In-district FTE (actual expenditure denominator)

**CR09: Add Ch70 Aid Comparison Page to Executive Summary**
- Created build_ch70_aid_data() helper function to extract Ch70 Aid per pupil data
- Added Ch70 Aid comparison table to Executive Summary with red/green shading
- Impact: Executive Summary now shows Ch70 Aid funding patterns across enrollment cohorts

**CR10: Add Actual NSS Comparison Page to Executive Summary**
- Created build_actual_nss_data() helper function to calculate Actual NSS minus Required NSS
- Added Actual NSS comparison table to Executive Summary showing spending above/below requirement
- Impact: Shows local funding effort patterns (spending beyond state requirements)

**CR11: Add Statistical Analysis for Ch70 and NSS**
- Created calculate_cohort_ch70_distribution() and calculate_cohort_nss_distribution() functions
- Added statistical analysis page to Section 2 with distribution tables
- Impact: Section 2 now includes statistical analysis of Ch70 and NSS patterns across cohorts

**CR 12,001: Remove "Page" Prefix from Page Numbers**
- Changed footer from "Page 1" to just "1"
- Impact: Cleaner page number display

**CR 12,002: Fix Missing Western MA Aggregate Chart Images**
- Fixed condition ordering bug ("Springfield" was matching before "all, excl. Springfield")
- Added support for all_western bucket in school_shared.py
- Impact: All Western MA aggregate pages now display correctly

### Files Modified
- compose_pdf.py (FE table restoration, new helper functions, Executive Summary tables, statistical analysis)
- nss_ch70_main.py (restored FE line, updated labels, fixed condition ordering)
- district_expend_pp_stack.py (updated enrollment labels to "In-district FTE")
- school_shared.py (added all_western support)

---

## 2025-10-25 - PDF Navigation and Structure Improvements (CR A01-A07)

### Completed Change Requests

**CR A01: Fixed TOC dotted leader wrapping to multiple lines**
- **Issue:** TOC dotted leaders were wrapping to multiple lines, causing entries to span multiple rows
- **Solution:** Used Unicode non-breaking spaces (U+00A0) between dots to prevent line wrapping
- **Files Modified:**
  - `compose_pdf.py` line 4483: Changed from regular spaces to `leader_dots = '\u00A0.\u00A0' * 80`
- **Impact:** Each TOC entry now stays on a single line, preventing TOC from spanning multiple pages

**CR A02: Reordered statistics pages for logical grouping**
- Reordered statistics pages so test results immediately follow their corresponding association pages
- **New order:**
  1. PPE Associations → PPE Test Results
  2. Ch70 Associations → Ch70 Test Results
  3. NSS Associations → NSS Test Results
- **Files Modified:**
  - `compose_pdf.py` lines 3077-3090 (PPE pages)
  - `compose_pdf.py` lines 3135-3148 (Ch70 pages)
  - `compose_pdf.py` lines 3195-3208 (NSS pages)
- **Impact:** Logical grouping of related content, better report flow

**CR A03: Split PPE statistics into separate pages**
- Created separate sections in report_text.txt: PPE_STATISTICAL_ASSOCIATIONS and PPE_STATISTICAL_TEST_RESULTS
- PPE statistical content now split across two pages matching Ch70 and NSS format
- **Files Modified:**
  - `report_text.txt` lines 373-447 (split PPE content into two sections)
  - `compose_pdf.py` lines 2486-2487, 3077-3090 (page rendering)
- **Impact:** Consistent format across all three metrics (PPE, Ch70, NSS)

**CR A04: Added detailed test results to Ch70 and NSS statistics pages**
- Updated CH70_STATISTICAL_TEST_RESULTS with specific ANOVA/regression statistics
- Updated NSS_STATISTICAL_TEST_RESULTS with specific ANOVA/regression statistics
- Added F-statistics, p-values, R² values, η² values matching PPE format
- **Example stats added:**
  - Ch70: "ANOVA: F(3,36) = 6.12, p = 0.002, η² = 0.338"
  - NSS: "ANOVA: F(3,36) = 4.73, p = 0.007, η² = 0.283"
- **Files Modified:**
  - `report_text.txt` lines 450-551
- **Impact:** All three metrics now have same level of statistical detail

**CR A05: Fixed left y-axis labels to "Enrollment" (third fix)**
- Updated all NSS/Ch70 plot left y-axis labels to simplified "Enrollment"
- Changed from verbose labels like "Foundation Enrollment" or "Weighted avg foundation enrollment per district"
- **Files Modified:**
  - `nss_ch70_main.py` lines 172, 177, 182, 187, 191, 196, 200, 205
- **Impact:** Consistent simplified labeling across all NSS/Ch70 plots

**CR A06: Comprehensive cross-reference navigation system**
- Created centralized mapping dictionaries for district-to-cohort and cohort-to-district relationships
- Implemented context-aware footer generation based on page type
- Cross-reference links appear at bottom of:
  - District pages (link to cohort)
  - Cohort pages (link to member districts and Western MA aggregate)
  - Western MA aggregate page (link to all cohorts)
- **Files Modified:**
  - `compose_pdf.py` lines 115-128: Created DISTRICT_COHORT_MAP and COHORT_DISTRICTS_MAP
  - `compose_pdf.py` lines 130-224: Created get_cross_reference_footer() function
  - `compose_pdf.py` lines 5067-5073: Integrated into graph_only pages
  - `compose_pdf.py` lines 5138-5144: Integrated into district pages with tables
  - `compose_pdf.py` lines 5172-5178: Integrated into NSS/Ch70 pages
  - `compose_pdf.py` lines 5203-5209: Integrated into regular district pages
- **Technical details:**
  - Single centralized function analyzes page subtitle/title to determine type
  - Returns list of Paragraph objects with hyperlinks using existing page numbering infrastructure
  - Extends existing two-pass page numbering system used for TOC
  - No code duplication across four page rendering locations
- **Impact:** Users can easily navigate between related pages (districts ↔ cohorts ↔ aggregates)

**CR A07: Split main document and appendices into separate PDFs**
- Separated report into two independent PDFs with separate page numbering
- Added --appendices-only flag to regenerate appendices independently
- **Main document:** "Western MA Per Pupil Expenditure Report.pdf"
  - Contains: Table of Contents, Executive Summary, Sections 1-3
- **Appendices:** "WMPPE Appendices.pdf"
  - Contains: Appendices A-D
- **Files Modified:**
  - `compose_pdf.py` lines 5217-5254: Modified main() function to split pages by section_id
    - Pages with section_id starting with "appendix_" go to appendices PDF
    - Each PDF gets independent page numbering via separate clear_page_map() and build_pdf() cycles
    - Default behavior: build ONLY main PDF
    - With --appendices-only flag: build ONLY appendices PDF
  - `compose_pdf.py` lines 5288-5294: Added argparse to accept --appendices-only flag
  - `compose_pdf.py` line 4460: Updated report title to "Western MA Per Pupil Expenditure Report"
  - `generate_report.py` lines 85-86: Added --appendices-only argument
  - `generate_report.py` lines 95-101: Modified pipeline to only run compose_pdf.py when flag is set
  - `generate_report.py` lines 34-76: Updated run_script() to pass appendices_only flag
  - `generate_report.py` lines 149-154: Updated output messages
- **Usage:**
  - `python generate_report.py` - Run full pipeline and generate ONLY main PDF
  - `python generate_report.py --appendices-only` - Run full pipeline and generate ONLY appendices PDF
- **Impact:** Cleaner document structure, faster iteration when working on either document independently

### Summary of Changes

**Files Modified:**
- `compose_pdf.py` (TOC wrapping fix, page reordering, cross-reference system, PDF split)
- `report_text.txt` (split PPE stats, added detailed Ch70/NSS stats)
- `nss_ch70_main.py` (simplified y-axis labels)
- `generate_report.py` (appendices-only flag support)

**Impact:**
- Improved TOC formatting (no multi-line entries)
- Better report organization (logical page ordering)
- Consistent statistical analysis across all three metrics
- Enhanced navigation with cross-reference footer links
- Separated main document and appendices for independent page numbering
- By default, only main PDF is generated (faster, cleaner workflow)
- Use --appendices-only flag to regenerate appendices independently

**Testing Needed:**
- Verify TOC entries stay on single lines
- Verify statistics pages appear in correct order
- Verify all statistics pages have proper titles and detailed content
- Verify cross-reference footer links appear on appropriate pages and navigate correctly
- Test default mode: `python generate_report.py` should generate ONLY main PDF
- Test appendices mode: `python generate_report.py --appendices-only` should generate ONLY appendices PDF
- Verify both PDFs have independent page numbering starting from 1

---

## 2025-10-25 - TOC, Box Plots, and Final PDF Split Fixes (CR 999, 1999, 2999, 3999)

### Completed Change Requests

**CR 999: Removed dotted leaders from Table of Contents**
- **Issue:** Dotted leaders in TOC not working well, appearing messy
- **Solution:** Removed dotted leader column entirely from TOC table
- **Files Modified:**
  - `compose_pdf.py` lines 4619-4631: Removed leader column, changed from 3-column to 2-column table
  - `compose_pdf.py` line 4631: Updated column widths to 85% title, 15% page number (was 75%, flexible, 10%)
  - `compose_pdf.py` lines 4633-4643: Updated table style commands for 2-column layout
- **Impact:** Cleaner TOC with simple two-column layout (title | page number)

**CR 1999: Fixed statistical box plots to share same x-axis scale**
- **Issue:** Mini box plots in statistical associations had different x-axis scales, making visual comparison misleading
- **Solution:** Calculate global min/max across all cohorts for each metric and apply same x-axis limits to all box plots
- **Files Modified:**
  - `compose_pdf.py` lines 1226-1289: Updated `create_mini_boxplot()` to accept optional `xlim` parameter
  - `compose_pdf.py` lines 1324-1334: Calculate global min/max with 5% padding before loop in `build_cohort_distribution_table()`
  - `compose_pdf.py` line 1370: Pass `xlim` to `create_mini_boxplot()` for consistent scaling
- **Technical details:**
  - Global limits: `xlim = (global_min - range * 0.05, global_max + range * 0.05)`
  - All box plots for same metric (PPE or CAGR) now share identical x-axis scale
  - Box plot shapes directly comparable even without visible x-axis
- **Impact:** Box plots now visually accurate - width and position directly comparable across cohorts

**CR 2999: Changed PPE plot left y-axis label from "In-district FTE" to "Enrollment" (5th request!)**
- **Issue:** PPE plots still showing "In-district FTE" despite multiple previous requests to change to "Enrollment"
- **Solution:** Changed all enrollment labels in PPE plot generation code
- **Files Modified:**
  - `district_expend_pp_stack.py` line 768: Changed Springfield label from "In-district FTE" to "Enrollment"
  - `district_expend_pp_stack.py` line 771: Changed cohort aggregate label from "Weighted avg in-district FTE per district" to "Enrollment"
  - `district_expend_pp_stack.py` line 822: Changed individual district label from "In-district FTE" to "Enrollment"
- **Impact:** All PPE plots now consistently show "Enrollment" on left y-axis, matching NSS/Ch70 plots

**CR 3999: Fixed appendices still appearing in main PDF**
- **Issue:** Appendices were still included in main PDF despite split logic being implemented
- **Solution:** This was already fixed in CR A07 - user needed to re-run script
- **Root cause:** User was viewing old PDF generated before CR A07 fixes were applied
- **Verification:** Filtering logic at `compose_pdf.py` lines 5256-5261 correctly splits pages by `section_id.startswith("appendix_")`
- **Impact:** Main PDF now excludes all appendix pages; appendices only generated with `--appendices-only` flag

### Summary of Changes

**Files Modified:**
- `compose_pdf.py` (TOC layout, box plot scaling, filtering already correct)
- `district_expend_pp_stack.py` (PPE plot enrollment labels)

**Impact:**
- Cleaner TOC without dotted leaders
- Box plots visually accurate with shared x-axis scale
- PPE plots finally have correct "Enrollment" label (5th fix!)
- Main PDF correctly excludes appendices

**Testing Needed:**
- Verify TOC has clean 2-column layout without dots
- Verify statistical box plots have consistent width/position reflecting actual data ranges
- Verify all PPE plots show "Enrollment" on left y-axis
- Run `python generate_report.py` and verify no appendix pages in main PDF
- Run `python generate_report.py --appendices-only` to generate appendices PDF

### Minor Cleanup

**Removed EXECUTIVE_SUMMARY_FOOTER**
- No longer needed in executive summary
- **Files Modified:**
  - `report_text.txt`: Removed EXECUTIVE_SUMMARY_FOOTER section (was mostly commented out)
  - `compose_pdf.py` lines 2549-2582: Removed exec_summary_footer variable and its use in exec_summary_all_text

**Updated generate_report.py documentation**
- Enhanced usage documentation to clearly explain --appendices-only flag
- **Files Modified:**
  - `generate_report.py` lines 1-24: Added comprehensive usage documentation explaining:
    - Default behavior: generates main PDF only
    - --force-recompute flag: bypass cache
    - --appendices-only flag: generate appendices PDF independently
    - Description of what each PDF contains
    - Use case: updating methodology documentation without regenerating plots

**Fixed Ch70 Aid and NSS Growth Rate table formatting**
- **Issue:** Tables "2009-2024 Chapter 70 Aid Growth Rates by Cohort" and "2009-2024 Actual NSS above Required NSS Growth Rates by Cohort" were showing dollar values ($) instead of percentages (%)
- **Root cause:** Metric names contain both "Aid"/"NSS" (triggers dollar formatting) and "%"/"Growth" (triggers percentage formatting). Dollar check came first, so it incorrectly formatted as dollars.
- **Solution:** Reordered logic to check for percentage indicators first, then dollar indicators
- **Files Modified:**
  - `compose_pdf.py` lines 1341-1354: Moved `if is_percentage_metric:` before `elif is_dollar_metric:`
- **Impact:** Ch70 Aid and NSS growth rate tables now correctly display percentages (e.g., "3.2%") instead of dollar amounts

---

## 2025-10-26 - Cross-Reference Footer Fixes and Ch70 Color Standardization

### Completed Changes

**Cross-reference footer navigation fixes**
- **Issue:** Cross-reference footers not appearing on district and cohort pages
- **Root cause:** Missing section_id fields and section_id format mismatches between page definitions and cross-reference lookup function
- **Solution:** Added missing section_id fields and fixed lookup patterns to match actual section_ids
- **Files Modified:**
  - `compose_pdf.py` line 3762: Added `section_id=f"{section_id}_nss"` to district NSS/Ch70 pages
  - `compose_pdf.py` lines 216-219: Fixed Western MA aggregate cross-reference lookup to use correct section_id patterns:
    - PPE pages: `section2_{cohort.lower()}`
    - NSS pages: `section2_{cohort.lower()}_nss`
- **Impact:** Cross-reference footer links now appear correctly on all district and cohort pages, enabling easy navigation between related pages

**Ch70 Aid plot color standardization**
- **Issue:** User preferred less bright green color seen on one Ch70 plot
- **Solution:** Changed Ch70 Aid color from "#86efac" (light green) to "#4ade80" (medium green, less bright)
- **Files Modified:**
  - `nss_ch70_plots.py` line 33: Updated NSS_CH70_COLORS["Ch70 Aid"] color value
- **Impact:** All Ch70 Aid plots now use consistent, less bright green color for better visual appearance

### Summary of Changes

**Files Modified:**
- `compose_pdf.py` (cross-reference footer fixes)
- `nss_ch70_plots.py` (Ch70 color standardization)

**Impact:**
- Cross-reference navigation system now fully functional
- Consistent Ch70 Aid plot coloring across all pages

**Testing Needed:**
- Regenerate all NSS/Ch70 plots to apply new green color
- Verify cross-reference footer links appear and navigate correctly
- Verify all Ch70 plots use the new medium green color

**Bug Fix: Multiple cross-reference section_id mismatches**
- **Issue:** PDF generation failed with "undefined destination target for 'amherst_pelham'"
- **Root causes:**
  1. Cross-reference footer was creating district section_ids with `district_` prefix
  2. Western MA aggregate NSS page was missing section_id
  3. Western MA aggregate cross-reference lookup used wrong section_id pattern
  4. TOC entries for Section 3 districts used section_ids without _ppe suffix
- **Solutions:**
  1. Removed `district_` prefix from cross-reference lookup (line 193)
  2. Added `section_id="section2_all_western_nss"` to Western MA NSS page (line 3593)
  3. Fixed Western MA aggregate lookup to use correct section_ids (lines 183-187)
  4. Updated TOC district entries to use _ppe suffix (lines 4482-4487)
- **Files Modified:**
  - `compose_pdf.py` line 193: Changed district cross-reference lookup
  - `compose_pdf.py` lines 183-187: Fixed Western MA aggregate section_id lookup
  - `compose_pdf.py` line 3593: Added missing section_id to Western MA aggregate NSS page
  - `compose_pdf.py` lines 4482-4487: Updated TOC entries to use correct section_ids (e.g., "amherst_pelham_ppe" instead of "amherst_pelham")
- **Impact:** All cross-reference links and TOC links now use correct section_id format to match page definitions

**Cross-reference footer enhancements**
- **Changes:**
  1. Added hyperlinks to all cross-reference footer links using `<a href="#section_id">` format
  2. Standardized all cross-references to start with "Compare to" for consistency
  3. Consolidated multiple links into single line to save space (e.g., "Compare to PPE and Enrollment for Western MA (page 34), Leverett (page 41), Pelham (page 43)")
- **Files Modified:**
  - `compose_pdf.py` lines 130-236: Rewrote `get_cross_reference_footer()` function
- **Impact:**
  - Cross-reference links are now clickable hyperlinks
  - Consistent "Compare to" prefix makes purpose clear
  - Single-line format for multiple links saves vertical space and prevents page overflow

**Fixed appendices appearing in main PDF**
- **Issue:** Appendix pages were still appearing in main PDF despite filtering logic
- **Root cause:** Only the first page of each appendix had section_id set; subsequent pages had no section_id
- **Solution:** Added `section_id` to ALL appendix pages:
  - Appendix A (Methodology): All 5 pages now have `section_id="appendix_a"` (lines 3871, 3881, 3888)
  - Appendix C (Data Tables): All pages now have `section_id="appendix_c"` (line 4349)
  - Appendix D (Additional Visualizations): All pages now have `section_id="appendix_d"` (lines 4400, 4410, 4423, 4449)
- **Files Modified:**
  - `compose_pdf.py` lines 3871, 3881, 3888: Added section_id to Appendix A continuation pages
  - `compose_pdf.py` line 4349: Moved section_id from conditional to always-set field
  - `compose_pdf.py` lines 4400, 4410, 4423, 4449: Added section_id to all Appendix D pages
- **Impact:** Main PDF now correctly excludes ALL appendix pages, not just the first page of each appendix

**Moved hardcoded text to external file (report_text.txt)**
- **Issue:** Ch70 and NSS statistical associations text was hardcoded in Python with default fallback values
- **Solution:** Added proper text sections to report_text.txt:
  - Added `CH70_STATISTICAL_ASSOCIATIONS` section (lines 428-450)
  - Added `NSS_STATISTICAL_ASSOCIATIONS` section (lines 504-526)
- **Files Modified:**
  - `report_text.txt`: Added two new statistical association sections
- **Impact:** All report content now externalized in text files for easier editing without code changes

**Removed Report Navigation box from Executive Summary**
- **Change:** Removed the mini TOC navigation box from the end of the Executive Summary page
- **Rationale:** Box kept in other sections but removed from exec summary per user request
- **Files Modified:**
  - `compose_pdf.py` lines 2591-2595: Removed mini_toc_text from exec_summary_all_text
- **Impact:** Executive Summary page is now shorter and cleaner

**Added omitted districts note to Western MA NSS page**
- **Change:** Added omitted districts note to Western MA (all, excl. Springfield) NSS/Ch70 page
- **Implementation:** Reused existing omitted_districts logic from executive summary
- **Files Modified:**
  - `compose_pdf.py` lines 3544-3550: Added omitted districts note to district_list_text
- **Impact:** Both PPE and NSS/Ch70 Western MA aggregate pages now show consistent omitted districts information

**Fixed gradient intensity on comparison choropleth maps**
- **Issue:** PPE and CAGR comparison maps showed only two colors (blue/orange) with no gradient intensity
- **Root cause:** Maps used only 3 discrete bins instead of graduated bins matching table shading thresholds
- **Solution:** Implemented graduated color intensity matching report methodology:
  - **PPE maps**: 9 graduated bins (5%, 10%, 15%, 20%+) with darker colors = further from baseline
  - **CAGR maps**: 9 graduated bins (0.5pp, 1pp, 1.5pp, 2pp+) with darker colors = further from baseline
  - Color progression: light blue → white → light orange (near baseline) to dark blue/dark orange (far from baseline)
- **Files Modified:**
  - `western_map.py` lines 598-618: Updated PPE comparison map bins and colors (3 bins → 9 bins)
  - `western_map.py` lines 772-792: Updated CAGR comparison map bins and colors (3 bins → 9 bins)
- **Impact:** Comparison maps now visually match the graduated shading used in comparison tables throughout the report, making it easy to see which districts are slightly vs. significantly different from their cohort baseline

---

## 2025-10-27 - Scatterplot Table District Type Labels

### Completed Changes

**Added district type labels to scatterplot table**
- **Change:** Added "(Unified Region)" and "(Secondary Region)" labels after appropriate district names in the scatterplot enrollment vs. PPE table
- **Implementation:**
  - Modified `_build_scatterplot_district_table()` function (lines 826-884) to:
    - Load district types from Excel profiles file (`Ch 70 District Profiles Actual NSS Over Required.xlsx`)
    - Map DistType values to display labels:
      - "Unified Regional" → "(Unified Region)"
      - "Regional Composite" → "(Secondary Region)"
      - Other types → "" (no label)
    - Return 6-tuple: (district_name, cohort, enrollment, ppe, cohort_label, district_type)
  - Modified `_build_scatterplot_table()` function (lines 707-794) to:
    - Accept 6-tuple instead of 5-tuple
    - Append district type label to district name: `f"{dist_name} {dist_type}"` when dist_type is not empty
    - Applied to both left and right columns of the table
- **Files Modified:**
  - `compose_pdf.py` lines 826-884: Updated `_build_scatterplot_district_table()` to load and map district types
  - `compose_pdf.py` lines 707-794: Updated `_build_scatterplot_table()` to display district type labels
- **Impact:** Scatterplot table now clearly identifies which districts are unified regions (PK-12) and which are secondary regions (overlapping with elementary districts)
- **Example output:**
  - Regular districts: "Amherst"
  - Unified regions: "Farmington River Reg (Unified Region)"
  - Secondary regions: "Amherst-Pelham (Secondary Region)"

### Summary

**Files Modified:**
- `compose_pdf.py` (scatterplot table functions)

**Impact:**
- Enhanced scatterplot table with district type labels for better context
- Readers can now distinguish regional districts from elementary districts at a glance

**Testing Needed:**
- Regenerate PDF to verify district type labels appear correctly in scatterplot table
- Verify labels only appear for regional districts (not regular elementary districts)
- Check for any layout issues (text wrapping, column width)

---

## 2025-10-27 - Choropleth Boundary Fix and District List Categorization

### Completed Changes

**Fixed missing Worthington boundary on CAGR comparison map**
- **Issue:** Worthington boundary was not appearing on the "Geographic map showing 15-year PPE growth (2009-2024) vs enrollment cohort baseline" (CAGR comparison map)
- **Root cause:** Districts with missing CAGR data (NaN values) were not being plotted, causing their boundaries to disappear
- **Solution:** Plot ALL non-secondary districts with gray fill and gray boundaries first (as a base layer), then overlay colored districts with data on top
- **Implementation:**
  - Modified `create_cagr_comparison_map()` function (lines 815-827)
  - Modified `create_ppe_comparison_map()` function (lines 638-650) with same fix for consistency
  - Changed base layer color from white to gray (#E0E0E0) to clearly indicate missing data
  - Changed zorder: base layer (gray fill) at zorder=1, colored overlay at zorder=2
  - Added legend entry showing "Missing data: N district(s)" with gray swatch
- **Files Modified:**
  - `western_map.py` lines 638-650: Added base layer plot with gray fill for PPE comparison map
  - `western_map.py` lines 712-723: Added missing data legend entry for PPE comparison map
  - `western_map.py` lines 815-827: Added base layer plot with gray fill for CAGR comparison map
  - `western_map.py` lines 910-921: Added missing data legend entry for CAGR comparison map
- **Impact:** Districts with missing data now show up with gray fill and gray boundaries, making all district boundaries visible on comparison maps with clear legend indication

**Adjusted Hampshire Unified District label position**
- **Issue:** Hampshire Unified District has an odd shape, and the centered +/- label crosses boundaries making it hard to read
- **Solution:** Detect "Hampshire" in district name and offset label upward by 15% of district height
- **Files Modified:**
  - `western_map.py` lines 681-687: Added y-offset calculation for Hampshire in PPE comparison map
  - `western_map.py` lines 887-893: Added y-offset calculation for Hampshire in CAGR comparison map
- **Impact:** Hampshire Unified District label now positioned above centroid for better readability

**Confirmed choropleth explanatory text location**
- **User question:** User could not find where explanatory text is stored for choropleth pages
- **Answer:** Explanatory text IS already in report_text.txt under these sections:
  - `SECTION1_CHOROPLETH_EXPLANATION` (line 312)
  - `SECTION1_PPE_COMPARISON_EXPLANATION` (line 320)
  - `SECTION1_CAGR_COMPARISON_EXPLANATION` (line 330)
- **No changes needed** - text is already externalized in report_text.txt as requested

**Separated district lists on NSS/Ch70 pages into three categories**
- **Change:** Broke up cohort member lists on Chapter 70 Aid and Net School Spending pages into three distinct categories:
  1. Districts (elementary districts)
  2. Unified regions (PK-12 regional districts)
  3. Secondary regions (overlapping secondary regional districts)
- **Implementation:**
  - Modified `_categorize_districts()` function (lines 919-964) to:
    - Load district types from Excel profiles file (`Ch 70 District Profiles Actual NSS Over Required.xlsx`)
    - Map DistType values to categories:
      - "Unified Regional" → unified_regions
      - "Regional Composite" → secondary_regions
      - Other/District → districts
    - Return 4 categories: districts, unified_regions, secondary_regions, vocational
  - Updated district list building for cohort NSS pages (lines 3475-3495):
    - Changed "Regions" to "Unified regions"
    - Added "Secondary regions" section
  - Updated district list building for Western MA aggregate NSS page (lines 3578-3598):
    - Changed "Regions" to "Unified regions"
    - Added "Secondary regions" section
- **Files Modified:**
  - `compose_pdf.py` lines 919-964: Updated `_categorize_districts()` function
  - `compose_pdf.py` lines 3482-3490: Updated cohort NSS district list building
  - `compose_pdf.py` lines 3585-3593: Updated Western MA aggregate NSS district list building
- **Impact:** NSS/Ch70 pages now clearly distinguish between three types of districts, making it easier to understand district composition
- **Example output:**
  - **Districts (45):** Amherst, Ashfield, Bernardston, ...
  - **Unified regions (10):** Farmington River Reg, Hawlemont Reg, ...
  - **Secondary regions (4):** Amherst-Pelham, Frontier Reg, ...

### Summary

**Files Modified:**
- `western_map.py` (choropleth boundary fix, missing data legend, Hampshire label adjustment)
- `compose_pdf.py` (district categorization and list building)

**Impact:**
- All district boundaries now visible on comparison choropleth maps, even for districts with missing data
- Missing data districts clearly indicated with gray fill and legend entry
- Hampshire Unified District label repositioned for better readability
- User confirmed where to find/edit choropleth explanatory text (report_text.txt)
- NSS/Ch70 pages now have clearer district type categorization with three separate sections
- Readers can now see at a glance which districts are elementary, which are unified regions, and which are secondary regions

**Testing Needed:**
- Regenerate choropleth maps to verify:
  - Worthington boundary appears on CAGR comparison map with gray fill
  - All districts with missing data show gray fill with legend entry
  - Hampshire Unified District label is positioned upward and readable
- Regenerate PDF to verify district lists on NSS/Ch70 pages show three categories
- Verify district type assignments are correct (elementary vs unified vs secondary)

---

## 2025-10-27 - Terminology Update: Unified Region → Regional K-12

### Completed Changes

**Changed "Unified Region" terminology to "Regional K-12" throughout codebase**
- **Rationale:** More descriptive terminology that clearly indicates these are K-12 regional districts
- **Scope:** Updated all references in code, documentation, map labels, and report text

**Updated map marker from "U" to "K12"**
- **Change:** Enrollment cohort choropleth map now displays "K12" instead of "U" for unified regional districts
- **Implementation:**
  - Changed text label in western_map.py line 422 from 'U' to 'K12'
  - Reduced font size from 20 to 16 to accommodate longer label
- **Files Modified:**
  - `western_map.py` line 422: Changed marker text and font size

**Updated scatterplot table labels**
- **Change:** District type labels in scatterplot table now show "(Regional K-12)" instead of "(Unified Region)"
- **Files Modified:**
  - `compose_pdf.py` line 834: Updated function docstring
  - `compose_pdf.py` line 850: Updated district type mapping

**Updated NSS/Ch70 district list headers**
- **Change:** Cohort member lists now show "Regional K-12" instead of "Unified regions"
- **Files Modified:**
  - `compose_pdf.py` line 3484: Updated cohort NSS page district list header
  - `compose_pdf.py` line 3587: Updated Western MA aggregate NSS page district list header

**Updated report explanatory text**
- **Change:** Choropleth explanation now references "Regional K-12 districts (marked 'K12')" instead of "Unified regional districts (marked 'U')"
- **Files Modified:**
  - `report_text.txt` line 317: Updated SECTION1_CHOROPLETH_EXPLANATION

**Updated code documentation**
- **Files Modified:**
  - `western_map.py` line 14: Updated comment to reflect "K12" marker

### Summary

**Files Modified:**
- `western_map.py` (map marker, documentation)
- `compose_pdf.py` (scatterplot labels, district list headers)
- `report_text.txt` (choropleth explanation)

**Impact:**
- Consistent "Regional K-12" terminology throughout entire report and codebase
- Map marker "K12" is more descriptive than "U"
- Clearer indication that these districts serve all grades PK-12
- No functional changes, only terminology updates

**Testing Needed:**
- Regenerate choropleth maps to verify "K12" markers appear correctly
- Verify font size is readable for "K12" markers
- Regenerate PDF to verify:
  - Scatterplot table shows "(Regional K-12)" labels
  - NSS/Ch70 pages show "Regional K-12" headers
  - Choropleth explanation references "K12" markers

---

## 2025-10-27 - Final Bug Fixes for Choropleth Maps and Boxed Sections

### Completed Changes

**Moved CAGR sensitivity box from CAGR map to executive summary**
- **Change:** Moved "Why the CAGR threshold is more sensitive" boxed section from SECTION1_CAGR_COMPARISON_EXPLANATION to EXECUTIVE_SUMMARY_COHORT_EXPLANATION
- **Location:** Now appears after "Cohorts are organized by enrollment size..." paragraph in executive summary
- **Files Modified:**
  - `report_text.txt` lines 121-131: Added boxed section to executive summary
  - `report_text.txt` line 357-367: Removed boxed section from CAGR comparison explanation
- **Impact:** CAGR threshold explanation now contextualizes cohort comparisons in executive summary

**Fixed __BOXED_START__ markers printing literally**
- **Issue:** Boxed section markers were appearing as literal text "__BOXED_START__" and "__BOXED_END__" in PDF
- **Root cause:** graph_only rendering path didn't handle boxed section markers
- **Solution:** Added boxed section handling to graph_only rendering code (similar to main rendering path)
- **Files Modified:**
  - `compose_pdf.py` lines 5170-5226: Added in_box state tracking and boxed content accumulation/rendering for graph_only pages
- **Impact:** Boxed sections now render correctly with gray borders on all page types

**Changed gray color to darker shade**
- **Change:** Changed missing data fill color from #E0E0E0 (light gray) to #999999 (medium-dark gray)
- **Rationale:** User requested darker gray for better visibility
- **Files Modified:**
  - `western_map.py`: All instances of #E0E0E0 replaced with #999999 (6 locations)
- **Impact:** Missing districts more visible on all choropleth maps

**Fixed Hampshire not showing on CAGR comparison map**
- **Issue:** Hampshire Regional (secondary regional district) was missing on CAGR comparison map
- **Root cause:** Gray base layer was only plotting `non_secondary` districts, excluding secondary regionals
- **Solution:** Changed base layer to plot ALL districts (`matched_gdf`) instead of just `non_secondary`
- **Files Modified:**
  - `western_map.py` lines 667-679: Changed PPE comparison base layer to plot all districts
  - `western_map.py` lines 867-879: Changed CAGR comparison base layer to plot all districts
- **Impact:** All secondary regionals (including Hampshire) now show with gray fill if missing data

**Ensured enrollment cohort map has gray base layer**
- **Status:** Already implemented in previous changes (line 392-401)
- **Verification:** Confirmed gray base layer plots ALL districts including secondary regionals
- **Files Modified:**
  - `western_map.py` line 396: Updated comment from "Light gray" to "Dark gray"
- **Impact:** Missing districts show up with dark gray fill on enrollment cohort map

### Summary

**Files Modified:**
- `report_text.txt` (moved boxed section)
- `compose_pdf.py` (fixed boxed section rendering in graph_only pages)
- `western_map.py` (darker gray color, fixed base layer to include all districts)

**Impact:**
- CAGR sensitivity explanation now in executive summary where it provides better context
- Boxed sections render correctly as bordered boxes instead of literal text
- Darker gray (#999999) makes missing districts more visible
- ALL districts show up on ALL choropleth maps, including secondary regionals like Hampshire
- Consistent missing data handling across all three choropleth map types

**Testing Needed:**
- Regenerate all choropleth maps to verify:
  - Darker gray color (#999999) for missing districts
  - Hampshire shows with gray fill on CAGR comparison map
  - All missing districts show with gray fill and legend entry on all maps
- Regenerate PDF to verify:
  - CAGR sensitivity box appears in executive summary with border
  - No literal "__BOXED_START__" text appears anywhere
  - All choropleth explanation paragraphs display correctly

---

## 2025-10-27 - Fixed Tuple Handling in Text Processing

### Completed Changes

**Fixed AttributeError: 'tuple' object has no attribute 'replace'**
- **Issue:** PDF generation failed with error when trying to call `.replace()` on tuple objects (headings)
- **Root cause:** Multiple code paths assumed all blocks in text_blocks were strings, not accounting for heading tuples
- **Solution:** Added proper type checking in all text processing loops to handle both tuples (headings) and strings
- **Files Modified:**
  - `compose_pdf.py` lines 4995-5000: Added isinstance check for table number substitution
  - `compose_pdf.py` lines 5003-5015: Added tuple/string handling for explanation blocks
  - `compose_pdf.py` lines 4645-4657: Added tuple/string handling for chart page explanation blocks
  - `compose_pdf.py` lines 5075-5086: Added tuple/string handling for CAGR page text blocks
- **Impact:** PDF generation now works correctly with headings in text blocks

### Pattern Applied

All text block processing now follows this pattern:
```python
for block in text_blocks:
    if isinstance(block, tuple) and len(block) == 2:
        # Heading tuple - render with appropriate style
        heading_level, heading_text = block
        if heading_level == "H1":
            story.append(Paragraph(heading_text, style_h1))
        elif heading_level == "H2":
            story.append(Paragraph(heading_text, style_h2))
        elif heading_level == "H3":
            story.append(Paragraph(heading_text, style_h3))
    elif isinstance(block, str):
        # Regular text - render with body style
        story.append(Paragraph(block, style_body))
    # Also handle boxed sections, table objects, etc.
```

### Summary

**Files Modified:**
- `compose_pdf.py` (multiple locations for tuple handling)

**Impact:**
- PDF generation now succeeds with all text processing paths properly handling headings
- Headings render with appropriate styles (H1, H2, H3)
- No more AttributeError crashes

**Testing Status:**
- Ready for PDF generation test run

---

## 2025-10-27 - Added Box Marker Support and Missing Text Sections

### Completed Changes

**Added ##BOX_START and ##BOX_END marker support**
- **Feature:** Users can now use ##BOX_START and ##BOX_END markers in report_text.txt to wrap content in a bordered box
- **Implementation:** Markers are converted to internal __BOXED_START__ and __BOXED_END__ format during parsing
- **Files Modified:**
  - `compose_pdf.py` lines 522-528: Added box marker parsing in parse_report_text function
  - `compose_pdf.py` lines 455-458: Updated docstring to document box syntax
- **Usage Example:**
  ```
  ##BOX_START
  Content that should appear in a box
  ##BOX_END
  ```

**Added missing text sections to report_text.txt**
- **Issue:** Text for Chapter 70 and NSS comparison pages was hardcoded in compose_pdf.py instead of being editable in report_text.txt
- **Solution:** Added two new sections to report_text.txt that were already being looked up but missing:
  - `EXECUTIVE_SUMMARY_CH70_EXPLANATION`: "Tables {N}-{N+2} compare Chapter 70 Aid (per foundation pupil) across Western MA school districts organized by enrollment size."
  - `EXECUTIVE_SUMMARY_NSS_EXPLANATION`: "Tables {N}-{N+2} compare Actual NSS above Required NSS (per foundation pupil) - showing local funding effort beyond state requirements."
- **Files Modified:**
  - `report_text.txt` lines 139-150: Added both missing sections
- **Impact:** All explanatory text is now centralized in report_text.txt for easy editing

### Summary

**Files Modified:**
- `compose_pdf.py` (added box marker support)
- `report_text.txt` (added missing CH70 and NSS explanation sections)

**New Features:**
- ##BOX_START / ##BOX_END markers available for use in report_text.txt
- All report text now editable in report_text.txt without needing to modify compose_pdf.py

**Testing Status:**
- Ready for testing with box markers in report content

---

## 2025-10-27 - Fixed Box Marker Rendering in Explanation Blocks

### Completed Changes

**Fixed boxed markers being printed literally instead of rendering as boxes**
- **Issue:** When using ##BOX_START and ##BOX_END in report_text.txt, the internal markers __BOXED_START__ and __BOXED_END__ were printed as literal text in the PDF
- **Root cause:** The explanation_blocks rendering code checked for tuples (headings) and strings, but didn't check for boxed markers before rendering strings as paragraphs
- **Solution:** Added boxed marker handling to both explanation_blocks rendering locations
- **Files Modified:**
  - `compose_pdf.py` lines 4656-4713: Added boxed marker handling in first explanation_blocks loop
  - `compose_pdf.py` lines 5055-5124: Added boxed marker handling in second explanation_blocks loop (with table number substitution)
  - `compose_pdf.py` line 5062: Updated table number substitution to skip boxed markers
- **Impact:** ##BOX_START and ##BOX_END markers now work correctly in all text sections

### Pattern Applied

All explanation_blocks rendering now follows this pattern:
```python
in_box = False
boxed_content = []

for block in explanation_blocks:
    if isinstance(block, str) and block == "__BOXED_START__":
        in_box = True
        continue
    elif isinstance(block, str) and block == "__BOXED_END__":
        # Render boxed content in a bordered table
        if boxed_content:
            # Build paragraphs, create table with border
            ...
        boxed_content = []
        in_box = False
        continue
    
    if in_box:
        boxed_content.append(block)
    elif isinstance(block, tuple):
        # Render heading
        ...
    elif isinstance(block, str):
        # Render normal paragraph
        ...
```

### Summary

**Files Modified:**
- `compose_pdf.py` (two explanation_blocks rendering locations)

**Impact:**
- Box markers now render correctly as bordered tables in all report sections
- Headings inside boxes are properly styled
- Table number substitution skips box markers

**Testing Status:**
- Ready for PDF generation test with box markers

---

## 2025-10-27 - Fixed Scatterplot Table to Use In-District FTE

### Completed Changes

**Fixed scatterplot table showing total FTE instead of in-district FTE**
- **Issue:** The district table on the "Scatterplot of enrollment vs. per-pupil expenditure" page was displaying total FTE values instead of in-district FTE values
- **Root cause:** The `_build_scatterplot_district_table` function was calling `get_total_fte_for_year` instead of `get_indistrict_fte_for_year`
- **Solution:** Updated function to use in-district FTE for consistency with cohort definitions throughout the report
- **Files Modified:**
  - `compose_pdf.py` line 857: Changed import from `get_total_fte_for_year` to `get_indistrict_fte_for_year`
  - `compose_pdf.py` line 888: Changed enrollment calculation to use `get_indistrict_fte_for_year`
  - `compose_pdf.py` lines 755, 760: Updated table headers to clarify "In-District FTE"
- **Impact:** Scatterplot table now correctly shows in-district FTE values, matching cohort definitions used throughout the report

### Summary

**Files Modified:**
- `compose_pdf.py` (_build_scatterplot_district_table function and _build_scatterplot_district_table_as_table function)

**Impact:**
- Table now shows in-district FTE instead of total FTE
- Column headers now explicitly say "In-District FTE" for clarity
- Data is now consistent with cohort definitions (which are based on in-district FTE)

**Testing Status:**
- Ready for PDF generation test
