# Work Log - PPE Report Project

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
