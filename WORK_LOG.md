# Work Log - School Data Analysis Project

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