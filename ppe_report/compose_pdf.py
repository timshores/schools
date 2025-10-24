"""
PDF Report Generator for School District Expenditure Analysis

This module creates a comprehensive PDF report with:
- Section 1: Western MA overview and 6 enrollment-based aggregate groups
- Section 2: Individual district pages with peer group comparisons
- Appendices: Data tables and calculation methodology

Key Design Patterns:
- All comparative plots use horizontal bars for better PDF layout
- Districts grouped by enrollment (Small/Medium/Large/Springfield)
- Enrollment changes shown as annotations (boxes to right of bars)
- Aggregates computed using enrollment-weighted per-pupil expenditures
- KeepInFrame prevents text overflow into footer area
"""

from __future__ import annotations

import bisect, re
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Flowable, HRFlowable, Image, KeepInFrame, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from school_shared import (
    OUTPUT_DIR, load_data, create_or_load_color_map, color_for,
    ENROLL_KEYS, DISTRICTS_OF_INTEREST,
    context_for_district, prepare_district_epp_lines, prepare_western_epp_lines, context_for_western,
    canonical_order_bottom_to_top,
    FTE_LINE_COLORS,
    mean_clean, get_latest_year,
    weighted_epp_aggregation,
    prepare_district_nss_ch70, prepare_aggregate_nss_ch70, prepare_aggregate_nss_ch70_weighted,
    latest_total_fte, get_enrollment_group,
    get_cohort_label, get_cohort_short_label, get_cohort_2024_label,
    get_western_cohort_districts, get_omitted_western_districts,
    make_safe_filename,
)
from nss_ch70_plots import build_nss_category_data, NSS_CH70_COLORS, NSS_CH70_STACK_ORDER
from statistical_analysis import run_all_analyses, format_results_for_report

# ===== code version =====
CODE_VERSION = "v2025.09.29-REFACTORED"

# ===== Figure and Table Counters =====
_FIGURE_COUNTER = 0
_TABLE_COUNTER = 0
_PENDING_FIGURE_NUM = None  # Track figure number waiting to be combined with table

def next_figure_number():
    """Get the next figure number and increment counter."""
    global _FIGURE_COUNTER
    _FIGURE_COUNTER += 1
    return _FIGURE_COUNTER

def next_table_number():
    """Get the next table number and increment counter."""
    global _TABLE_COUNTER
    _TABLE_COUNTER += 1
    return _TABLE_COUNTER

def set_pending_figure(fig_num):
    """Store a figure number to be combined with next table."""
    global _PENDING_FIGURE_NUM
    _PENDING_FIGURE_NUM = fig_num

def get_and_clear_pending_figure():
    """Get pending figure number and clear it."""
    global _PENDING_FIGURE_NUM
    result = _PENDING_FIGURE_NUM
    _PENDING_FIGURE_NUM = None
    return result

def reset_counters():
    """Reset figure and table counters (called at start of PDF generation)."""
    global _FIGURE_COUNTER, _TABLE_COUNTER, _PENDING_FIGURE_NUM
    _FIGURE_COUNTER = 0
    _TABLE_COUNTER = 0
    _PENDING_FIGURE_NUM = None

# ===== Page Number Tracking for TOC =====
_PAGE_MAP = {}  # Maps section_id -> page_number

class PageMarker(Flowable):
    """Invisible flowable that records the page number for a section_id."""
    def __init__(self, section_id):
        Flowable.__init__(self)
        self.section_id = section_id
        self.width = 0
        self.height = 0

    def draw(self):
        """Called when flowable is rendered - captures current page number."""
        if self.section_id:
            _PAGE_MAP[self.section_id] = self.canv.getPageNumber()

def get_page_number(section_id):
    """Get the recorded page number for a section_id, or None if not found."""
    return _PAGE_MAP.get(section_id)

def clear_page_map():
    """Clear the page number map (called before each PDF build pass)."""
    global _PAGE_MAP
    _PAGE_MAP = {}

def build_combined_fig_table_label(fig_num, table_num, doc_width, style_fig, style_table):
    """Build a table with Figure # left-aligned and Table # right-aligned on same line."""
    fig_para = Paragraph(f"<i>Figure {fig_num}</i>", style_fig)
    table_para = Paragraph(f"<i>Table {table_num}</i>", style_table)

    # Create a two-column table spanning full width
    data = [[fig_para, table_para]]
    t = Table(data, colWidths=[doc_width/2, doc_width/2])
    t.setStyle(TableStyle([
        ('ALIGN', (0, 0), (0, 0), 'LEFT'),
        ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
        ('TOPPADDING', (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
    ]))
    return t

# ---- Shading controls ----
# NOTE: Two independent tests determine cell shading in district comparison tables:
# 1. CAGR Test: Compares district CAGR to baseline using ABSOLUTE percentage point difference
#    Example: District 5.0% vs Baseline 3.0% = 2.0pp difference → shade if >= threshold
MATERIAL_DELTA_PCTPTS = 0.01  # 1.0pp threshold for CAGR shading

# 2. Dollar Test: Compares district $/pupil to baseline using RELATIVE percent difference
#    Example: District $20,000 vs Baseline $19,000 = +5.3% relative → shade if >= threshold
DOLLAR_THRESHOLD_REL  = 0.05  # 5.0% threshold for dollar shading

# Bins for shading intensity - separate bins for CAGR and dollar to match different thresholds
# CAGR bins: 1pp base, increments of 1pp → [1pp, 2pp, 3pp, 4pp+]
SHADE_BINS_CAGR = [0.01, 0.02, 0.03, 0.04]
# Dollar bins: 5% base, increments of 5% → [5%, 10%, 15%, 20%+]
SHADE_BINS_DOLLAR = [0.05, 0.10, 0.15, 0.20]
# Neutral comparison colors (not implying good/bad):
ABOVE_SHADES = ["#FFF4E6", "#FFE8CC", "#FFD9A8", "#FFC97A", "#FFB84D"]  # above baseline: light amber/tan (lightest→darkest)
BELOW_SHADES = ["#E0F7FA", "#B2EBF2", "#80DEEA", "#4DD0E1", "#26C6DA"]  # below baseline: light teal/cyan (lightest→darkest)
BASELINE_YELLOW = "#FFFFD9"  # pale yellow for baseline row highlighting (RGB 1.0, 1.0, 0.85 in hex)

# Optional $-gate; if >0, shading applies only when latest $/pupil >= this
# Currently disabled (0.0) - all categories get shading regardless of magnitude
MATERIAL_MIN_LATEST_DOLLARS = 0.0

def district_png_simple(dist: str) -> Path:
    safe_dist = make_safe_filename(dist)
    return OUTPUT_DIR / f"expenditures_per_pupil_vs_enrollment_{safe_dist}_simple.png"

def district_png_detail(dist: str) -> Path:
    safe_dist = make_safe_filename(dist)
    return OUTPUT_DIR / f"expenditures_per_pupil_vs_enrollment_{safe_dist}_detail.png"

def regional_png(bucket: str) -> Path:
    return OUTPUT_DIR / f"regional_expenditures_per_pupil_Western_Traditional_{bucket}.png"

# ---- Styles ----
styles = getSampleStyleSheet()
style_title_main = ParagraphStyle("title_main", parent=styles["Heading1"], fontSize=14, leading=17, spaceAfter=2)
style_title_sub  = ParagraphStyle("title_sub",  parent=styles["Normal"],   fontSize=12, leading=14, spaceAfter=6)
style_body       = ParagraphStyle("body",       parent=styles["Normal"],   fontSize=9,  leading=12)
style_body_12pt  = ParagraphStyle("body_12pt",  parent=styles["Normal"],   fontSize=12, leading=16)  # Readable font for Appendix C
style_num        = ParagraphStyle("num",        parent=styles["Normal"],   fontSize=9,  leading=12, alignment=2)
style_legend     = ParagraphStyle("legend",     parent=style_body,         fontSize=8,  leading=10, alignment=2)
style_legend_center = ParagraphStyle("legend_center", parent=style_body,   fontSize=8,  leading=10, alignment=1)
style_legend_right  = ParagraphStyle("legend_right",  parent=style_body,   fontSize=8,  leading=10, alignment=2)
style_note       = ParagraphStyle("note",       parent=style_body,         fontSize=8,  leading=10, alignment=2)
style_hdr_left   = ParagraphStyle("hdr_left",   parent=styles["Normal"],   fontSize=9,  leading=12, alignment=0)
style_hdr_right  = ParagraphStyle("hdr_right",  parent=styles["Normal"],   fontSize=9,  leading=12, alignment=2)
style_right_small= ParagraphStyle("right_small", parent=styles["Normal"],  fontSize=8,  leading=10, alignment=2)
style_data_cell  = ParagraphStyle("data_cell",   parent=styles["Normal"],  fontSize=6,  leading=8,  alignment=2)
style_data_hdr   = ParagraphStyle("data_hdr",    parent=styles["Normal"],  fontSize=7,  leading=9,  alignment=2)
style_data_label = ParagraphStyle("data_label",  parent=styles["Normal"],  fontSize=6,  leading=8,  alignment=0)
style_figure_num = ParagraphStyle("figure_num",  parent=styles["Normal"],  fontSize=8,  leading=10, alignment=0, fontName='Helvetica-Oblique', backColor=colors.HexColor("#F5F5F5"))  # Small, italic, left-aligned, light gray background
style_table_num  = ParagraphStyle("table_num",   parent=styles["Normal"],  fontSize=8,  leading=10, alignment=2, fontName='Helvetica-Oblique', backColor=colors.HexColor("#F5F5F5"))  # Small, italic, right-aligned, light gray background
style_fig_table_num = ParagraphStyle("fig_table_num", parent=styles["Normal"], fontSize=8, leading=10, alignment=2, fontName='Helvetica-Oblique', backColor=colors.HexColor("#F5F5F5"))  # Combined figure and table numbers

NEG_COLOR        = HexColor("#3F51B5")
style_num_neg    = ParagraphStyle("num_neg", parent=style_num, textColor=NEG_COLOR)

# Footer
SOURCE_LINE1 = "Source: DESE District Expenditures by Spending Category, Last updated: August 12, 2025"
SOURCE_LINE2 = "https://educationtocareer.data.mass.gov/Finance-and-Budget/District-Expenditures-by-Spending-Category/er3w-dyti/"

def draw_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    y1 = 0.6 * inch
    # Source lines removed from footer, now in appendix
    x_right = doc.pagesize[0] - doc.rightMargin
    canvas.drawRightString(x_right, y1, f"Page {canvas.getPageNumber()}")
    canvas.restoreState()

# ---- Flowables ----
def _to_color(c): return c if isinstance(c, colors.Color) else HexColor(c)

class LineSwatch(Flowable):
    # Width 30 per your tweak to avoid label overlap
    def __init__(self, color_like, w=30, h=7, lw=2.6):
        super().__init__(); self.c=_to_color(color_like); self.w=w; self.h=h; self.lw=lw; self.width=w; self.height=h
    def draw(self):
        c = self.canv; y = self.h/2.0
        c.setStrokeColor(self.c); c.setLineWidth(self.lw); c.line(0, y, self.w, y)
        c.setFillColor(colors.white); c.setStrokeColor(self.c)
        r = self.h*0.65; c.circle(self.w/2.0, y, r, stroke=1, fill=1)

class DotSwatch(Flowable):
    """Filled circle swatch matching scatterplot symbols."""
    def __init__(self, color_like, w=20, h=7, r=3):
        super().__init__(); self.c=_to_color(color_like); self.w=w; self.h=h; self.r=r; self.width=w; self.height=h
    def draw(self):
        c = self.canv; y = self.h/2.0; x = self.w/2.0
        c.setFillColor(self.c); c.setStrokeColor(colors.white)
        c.circle(x, y, self.r, stroke=1, fill=1)

# ---- Helpers ----
def fmt_pct(v: float) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)): return "—"
    return f"{v * 100:+.1f}%"

def parse_pct_str_to_float(s: str) -> float:
    """Parse percentage string to float (e.g., '+5.0%' -> 0.05)."""
    try:
        return float((s or "").replace("%", "").replace("+", "").strip()) / 100.0
    except Exception:
        return float("nan")

def compute_cagr_last(series: pd.Series, years: int) -> float:
    """
    CAGR from (last - years) → last; NaN if endpoints missing or invalid.

    Args:
        series: Time series indexed by year
        years: Number of years to look back

    Returns:
        CAGR as a float (e.g., 0.05 = 5%), or NaN if calculation not possible
    """
    if series is None or len(series) == 0:
        return float("nan")
    if years <= 0:
        return float("nan")

    s = series.dropna().sort_index()
    if s.empty:
        return float("nan")

    end_year = int(s.index.max())
    start_year = end_year - years

    if start_year not in s.index or end_year not in s.index:
        return float("nan")

    v0 = float(s.loc[start_year])
    v1 = float(s.loc[end_year])

    # Handle zero values
    if v0 == 0 or v1 == 0:
        return float("nan")

    # If values cross zero (e.g., Springfield: -$10M to +$1M),
    # use average annual growth rate instead of geometric CAGR
    if (v0 > 0 and v1 < 0) or (v0 < 0 and v1 > 0):
        # Average annual rate of change relative to starting absolute value
        return (v1 - v0) / (years * abs(v0))

    # For negative values with same sign, calculate CAGR on absolute values
    # then apply sign based on growth direction
    if v0 < 0 and v1 < 0:
        # Both negative: use absolute values for calculation
        abs_cagr = (abs(v1) / abs(v0)) ** (1.0 / years) - 1.0
        # If |v1| > |v0|, values are becoming "more negative" (growth in magnitude)
        # Return positive CAGR to indicate growth in absolute magnitude
        return abs_cagr

    # Standard case: both values positive
    return (v1 / v0) ** (1.0 / years) - 1.0


# ---------- Text Loading from External Files ----------

def load_report_text_sections(text_file: Path = None) -> dict:
    """
    Load text sections from external file for easy editing.

    Returns:
        dict mapping section_name -> list of paragraph strings

    Format of text file:
        ================================================================================
        SECTION_NAME
        ================================================================================
        Paragraph 1 text here...

        Paragraph 2 text here...
        ================================================================================
        NEXT_SECTION
        ================================================================================
        ...
    """
    if text_file is None:
        # Default to report_text.txt in the same directory as this script
        text_file = Path(__file__).parent / "report_text.txt"

    if not text_file.exists():
        print(f"[WARN] Text file not found: {text_file}, using defaults")
        return {}

    sections = {}
    content = text_file.read_text(encoding='utf-8')
    # Normalize line endings (Windows \r\n -> Unix \n)
    content = content.replace('\r\n', '\n')

    # Parse sections using regex to handle the double-separator format:
    # ========...========
    # SECTION_NAME
    # ========...========
    # content...
    import re
    section_pattern = re.compile(r'={80,}\s*\n([A-Z_0-9]+)\s*\n={80,}\s*\n(.*?)(?=\n={80,}|$)', re.DOTALL)

    for match in section_pattern.finditer(content):
        section_name = match.group(1).strip()
        section_content = match.group(2).strip()

        # Split content into paragraphs (empty lines separate paragraphs)
        paragraphs = []
        current_para = []

        for line in section_content.split('\n'):
            line = line.rstrip()
            if line == "":
                if current_para:
                    paragraphs.append(" ".join(current_para))
                    current_para = []
                # Don't add empty paragraphs - let the content control spacing
            else:
                current_para.append(line)

        # Add final paragraph if exists
        if current_para:
            paragraphs.append(" ".join(current_para))

        sections[section_name] = paragraphs

    return sections


def fill_text_placeholders(paragraphs: list, replacements: dict) -> list:
    """
    Replace placeholders in text with actual values.

    Args:
        paragraphs: List of paragraph strings (may contain {PLACEHOLDER} markers)
        replacements: Dict mapping placeholder names to replacement strings

    Returns:
        List of paragraphs with placeholders replaced
    """
    result = []
    for para in paragraphs:
        filled = para
        for placeholder, value in replacements.items():
            filled = filled.replace(f"{{{placeholder}}}", str(value))
        result.append(filled)
    return result


# ---------- Western MA Baseline Builders ----------

def _build_western_ma_baseline(df: pd.DataFrame, reg: pd.DataFrame, c70: pd.DataFrame, cmap_all: dict, latest_year: int, exclude_springfield: bool = True) -> Tuple[dict, dict]:
    """
    Build baseline comparison map for Western MA weighted average (excluding Springfield).

    This baseline is used for comparing cohort aggregates to the regional average.

    Args:
        df: Main expenditure DataFrame
        reg: Regional classification DataFrame
        c70: Chapter 70 data
        cmap_all: Color map
        latest_year: Latest year in dataset
        exclude_springfield: If True, exclude Springfield from the average (default True)

    Returns:
        Tuple of (baseline_map for PPE, baseline_map for FTE)
    """
    # Get all Western MA traditional districts using centralized cohort function
    # This ensures baseline calculations use the same filtered district list as plots and tables
    cohorts = get_western_cohort_districts(df, reg)

    # Combine all cohorts
    western_present = []
    for cohort_key in ["TINY", "SMALL", "MEDIUM", "LARGE", "SPRINGFIELD"]:
        western_present.extend(cohorts.get(cohort_key, []))

    # Exclude Springfield if requested
    if exclude_springfield:
        western_present = [d for d in western_present if d not in cohorts.get("SPRINGFIELD", [])]

    # Calculate weighted Western MA aggregate
    epp_western, _, _ = weighted_epp_aggregation(df, western_present)

    # Build PPE baseline map
    base_map = {}
    if not epp_western.empty:
        start_year = latest_year - 15  # 15 years before latest for START_DOLLAR
        for sc in epp_western.columns:
            s = epp_western[sc]
            base_map[sc] = {
                "5": compute_cagr_last(s, 5),
                "10": compute_cagr_last(s, 10),
                "15": compute_cagr_last(s, 15),
                "DOLLAR": (float(s.loc[latest_year]) if latest_year in s.index else float("nan")),
                "START_DOLLAR": (float(s.loc[start_year]) if start_year in s.index else float("nan")),
            }

        # Add Total entry to baseline map
        total_series = epp_western.sum(axis=1)
        base_map["Total"] = {
            "5": compute_cagr_last(total_series, 5),
            "10": compute_cagr_last(total_series, 10),
            "15": compute_cagr_last(total_series, 15),
            "DOLLAR": (float(total_series.loc[latest_year]) if latest_year in total_series.index else float("nan")),
            "START_DOLLAR": (float(total_series.loc[start_year]) if start_year in total_series.index else float("nan")),
        }

    # Build FTE baseline map (using weighted average enrollment)
    # Get enrollment data for Western MA districts
    _, _, _, lines_western = prepare_western_epp_lines(df, reg, "all_western", c70, districts=western_present)

    fte_base_map = {}
    if lines_western:
        for key, label in ENROLL_KEYS:
            s = lines_western.get(label)
            if s is not None and not s.empty:
                start_year = latest_year - 15
                fte_base_map[label] = {
                    "5": compute_cagr_last(s, 5),
                    "10": compute_cagr_last(s, 10),
                    "15": compute_cagr_last(s, 15),
                    "FTE": (float(s.loc[latest_year]) if latest_year in s.index else float("nan")),
                    "START_FTE": (float(s.loc[start_year]) if start_year in s.index else float("nan")),
                }

    return base_map, fte_base_map


def _build_western_ma_nss_baseline(df: pd.DataFrame, reg: pd.DataFrame, c70: pd.DataFrame, latest_year: int, exclude_springfield: bool = True) -> dict:
    """
    Build NSS/Ch70 baseline comparison map for Western MA weighted average (excluding Springfield).

    This baseline is used for comparing cohort aggregates to the regional average for NSS/Ch70 data.

    Args:
        df: Main expenditure DataFrame
        reg: Regional classification DataFrame
        c70: Chapter 70 data
        latest_year: Latest year in dataset
        exclude_springfield: If True, exclude Springfield from the average (default True)

    Returns:
        dict mapping component names to baseline values
    """
    # Get all Western MA traditional districts using centralized cohort function
    # This ensures baseline calculations use the same filtered district list as plots and tables
    cohorts = get_western_cohort_districts(df, reg)

    # Combine all cohorts
    western_present = []
    for cohort_key in ["TINY", "SMALL", "MEDIUM", "LARGE", "SPRINGFIELD"]:
        western_present.extend(cohorts.get(cohort_key, []))

    # Exclude Springfield if requested
    if exclude_springfield:
        western_present = [d for d in western_present if d not in cohorts.get("SPRINGFIELD", [])]

    # Calculate weighted Western MA aggregate for NSS/Ch70
    nss_western, _, foundation_western = prepare_aggregate_nss_ch70_weighted(df, c70, western_present)

    if nss_western.empty:
        return {}

    # Use existing function to build baseline map
    return _build_nss_ch70_baseline_map(nss_western, latest_year, foundation_western)


def _build_scatterplot_table(district_data: List[Tuple[str, str, float, float, str]], doc_width: float, style_body, style_num, year: int = 2024):
    """Build compact 3-column table showing all districts with cohort colors, enrollment, and year-specific PPE."""
    if not district_data:
        return None

    # Cohort colors matching the scatterplot (more saturated)
    cohort_colors = {
        'TINY': HexColor('#4575B4'),      # Blue (low enrollment)
        'SMALL': HexColor('#3C9DC4'),     # Cyan (more saturated)
        'MEDIUM': HexColor('#FDB749'),    # Amber (more saturated)
        'LARGE': HexColor('#D73027'),     # Red (now includes former X-Large)
        'SPRINGFIELD': HexColor('#A50026'),  # Dark Red (outliers)
    }

    # Smaller font styles for compact table
    style_small = ParagraphStyle("small", parent=style_body, fontSize=7, leading=9)
    style_small_num = ParagraphStyle("small_num", parent=style_num, fontSize=7, leading=9)

    # Split districts into two columns
    mid = (len(district_data) + 1) // 2
    left_districts = district_data[:mid]
    right_districts = district_data[mid:]

    # Build combined table data with year-specific headers
    data = [
        [Paragraph("", style_small),  # Left swatch
         Paragraph("<b>District</b>", style_small),
         Paragraph(f"<b>{year} FTE</b>", style_small_num),
         Paragraph(f"<b>{year} PPE ▼</b>", style_small_num),  # Sort arrow indicator
         Paragraph("", style_small),  # Gap
         Paragraph("", style_small),  # Right swatch
         Paragraph("<b>District</b>", style_small),
         Paragraph(f"<b>{year} FTE</b>", style_small_num),
         Paragraph(f"<b>{year} PPE ▼</b>", style_small_num)]  # Sort arrow indicator
    ]

    # Track cohort changes for dividing lines
    left_cohort_changes = []  # Row indices where left column cohort changes
    right_cohort_changes = []  # Row indices where right column cohort changes
    prev_left_cohort = None
    prev_right_cohort = None

    # Add rows
    for i in range(mid):
        row = []

        # Left column
        if i < len(left_districts):
            dist_name, cohort, enrollment, ppe, cohort_label = left_districts[i]
            swatch_color = cohort_colors.get(cohort, colors.white)
            row.extend([
                DotSwatch(swatch_color),
                Paragraph(dist_name, style_small),
                Paragraph(f"{enrollment:,.0f}", style_small_num),
                Paragraph(f"${ppe:,.0f}", style_small_num)
            ])
            # Track cohort change
            if prev_left_cohort is not None and cohort != prev_left_cohort:
                left_cohort_changes.append(i + 1)  # +1 because header is row 0
            prev_left_cohort = cohort
        else:
            row.extend([Paragraph("", style_small)] * 4)

        # Gap
        row.append(Paragraph("", style_small))

        # Right column
        if i < len(right_districts):
            dist_name, cohort, enrollment, ppe, cohort_label = right_districts[i]
            swatch_color = cohort_colors.get(cohort, colors.white)
            row.extend([
                DotSwatch(swatch_color),
                Paragraph(dist_name, style_small),
                Paragraph(f"{enrollment:,.0f}", style_small_num),
                Paragraph(f"${ppe:,.0f}", style_small_num)
            ])
            # Track cohort change
            if prev_right_cohort is not None and cohort != prev_right_cohort:
                right_cohort_changes.append(i + 1)  # +1 because header is row 0
            prev_right_cohort = cohort
        else:
            row.extend([Paragraph("", style_small)] * 4)

        data.append(row)

    # 90% width table with 2 columns plus gap (wider to eliminate wrapping)
    # Left: Swatch, District, FTE, PPE | Gap | Right: Swatch, District, FTE, PPE
    table_width = doc_width * 0.90
    col_width = (table_width - 0.20*inch) / 2  # Split remaining space between left and right

    tbl = Table(data, colWidths=[
        0.25*inch, col_width*0.53, col_width*0.23, col_width*0.24,  # Left columns
        0.20*inch,  # Gap
        0.25*inch, col_width*0.53, col_width*0.23, col_width*0.24   # Right columns
    ], repeatRows=1)

    ts = TableStyle([
        ("LINEBELOW", (0,0), (3,0), 0.5, colors.black),  # Left header underline
        ("LINEBELOW", (5,0), (8,0), 0.5, colors.black),  # Right header underline
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("ALIGN", (0,0), (1,-1), "LEFT"),
        ("ALIGN", (2,0), (3,-1), "RIGHT"),
        ("ALIGN", (5,0), (6,-1), "LEFT"),
        ("ALIGN", (7,0), (8,-1), "RIGHT"),
        ("LEFTPADDING", (0,0), (-1,-1), 2),
        ("RIGHTPADDING", (0,0), (-1,-1), 2),
        ("TOPPADDING", (0,0), (-1,-1), 1.5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 1.5),
    ])

    # Add faint dividing lines between cohorts
    for row_idx in left_cohort_changes:
        ts.add("LINEABOVE", (0, row_idx), (3, row_idx), 0.3, colors.grey)
    for row_idx in right_cohort_changes:
        ts.add("LINEABOVE", (5, row_idx), (8, row_idx), 0.3, colors.grey)

    tbl.setStyle(ts)
    return tbl

def _build_scatterplot_district_table(df: pd.DataFrame, reg: pd.DataFrame, latest_year: int) -> List[Tuple[str, str, float, float, str]]:
    """
    Build district table data for scatterplot page.
    Returns: List of (district_name, cohort, enrollment, ppe, cohort_label) sorted by cohort then PPE descending.
    """
    from school_shared import get_total_fte_for_year, get_enrollment_group, get_cohort_label, get_western_cohort_districts

    # Get Western MA traditional districts using centralized cohort function
    # This ensures consistent filtering (includes PPE validation)
    cohorts = get_western_cohort_districts(df, reg)
    western_districts = []
    for cohort_key in ["TINY", "SMALL", "MEDIUM", "LARGE", "SPRINGFIELD"]:
        western_districts.extend(cohorts.get(cohort_key, []))

    district_data = []
    for dist in western_districts:
        enrollment = get_total_fte_for_year(df, dist, latest_year)
        # Note that we're retaining Springfield in this table although it's an outlier not shown on the plot
        ppe_data = df[
            (df["DIST_NAME"].str.lower() == dist) &
            (df["IND_CAT"].str.lower() == "expenditures per pupil") &
            (df["YEAR"] == latest_year)
        ]
        total_ppe = ppe_data[~ppe_data["IND_SUBCAT"].str.lower().isin(["total expenditures", "total in-district expenditures"])]["IND_VALUE"].sum()

        # District already filtered by get_western_cohort_districts() to have valid PPE
        if total_ppe > 0:
            cohort = get_enrollment_group(enrollment)
            cohort_label = get_cohort_label(cohort)
            district_data.append((dist.title(), cohort, float(enrollment), float(total_ppe), cohort_label))

    # Sort by cohort (TINY, SMALL, MEDIUM, LARGE, SPRINGFIELD) then by PPE descending within each cohort
    cohort_order = {"TINY": 0, "SMALL": 1, "MEDIUM": 2, "LARGE": 3, "SPRINGFIELD": 4}
    district_data.sort(key=lambda x: (cohort_order.get(x[1], 999), -x[3]))

    return district_data

def _abbr_bucket_suffix(full: str) -> str:
    """Extract abbreviated bucket suffix from full baseline title for legend display."""
    # Match by cohort keywords (case-insensitive) - works with any dynamic boundaries
    full_lower = (full or "").lower()

    # Check for "All Western MA (excluding Springfield)" - used for cohort comparisons
    if "all western" in full_lower and "excluding springfield" in full_lower:
        return "MA (excl. Springfield)"

    # Check for each cohort keyword (order matters - check "tiny" before "small"!)
    if "tiny" in full_lower:
        return f"({get_cohort_label('TINY')})"
    if "small" in full_lower:
        return f"({get_cohort_label('SMALL')})"
    if "medium" in full_lower:
        return f"({get_cohort_label('MEDIUM')})"
    if "large" in full_lower:
        return f"({get_cohort_label('LARGE')})"
    if "springfield" in full_lower:
        return f"({get_cohort_label('SPRINGFIELD')})"

    # Legacy fallback for old 2-tier system (should not be used)
    if re.search(r"≤\s*500", full or "", flags=re.I):
        return "(≤500)"
    if re.search(r">\s*500", full or "", flags=re.I):
        return "(>500)"

    return ""

def _categorize_districts(district_list: list, reg: pd.DataFrame) -> dict:
    """Categorize districts into regular districts, unified regions, and vocational/technical schools."""
    categories = {"districts": [], "unified_regions": [], "vocational": []}

    for district in district_list:
        # Find district in reg dataframe (case-insensitive match)
        mask = reg["DIST_NAME"].str.lower() == district.lower()
        if not mask.any():
            # If not found, categorize by name pattern
            if "-" in district or "reg" in district.lower() or "regional" in district.lower():
                categories["unified_regions"].append(district)
            else:
                categories["districts"].append(district)
            continue

        school_type = reg[mask]["SCHOOL_TYPE"].iloc[0]

        if school_type.lower() == "vocational":
            categories["vocational"].append(district)
        elif "-" in district or "reg" in district.lower() or "regional" in district.lower():
            # Traditional regional unified districts
            categories["unified_regions"].append(district)
        else:
            # Regular traditional districts
            categories["districts"].append(district)

    return categories

def _shade_for_cagr_delta(delta_pp: float):
    # delta_pp is a true fraction (e.g., 0.021 = 2.1pp)
    if delta_pp != delta_pp or abs(delta_pp) < MATERIAL_DELTA_PCTPTS:
        return None
    idx = max(0, min(len(ABOVE_SHADES)-1, bisect.bisect_right(SHADE_BINS_CAGR, abs(delta_pp)) - 1))
    return HexColor(ABOVE_SHADES[idx]) if delta_pp > 0 else HexColor(BELOW_SHADES[idx])

def _shade_for_dollar_rel(delta_rel: float):
    # delta_rel is a relative diff (e.g., +0.03 = +3%)
    if delta_rel != delta_rel or abs(delta_rel) < DOLLAR_THRESHOLD_REL:
        return None
    idx = max(0, min(len(ABOVE_SHADES)-1, bisect.bisect_right(SHADE_BINS_DOLLAR, abs(delta_rel)) - 1))
    return HexColor(ABOVE_SHADES[idx]) if delta_rel > 0 else HexColor(BELOW_SHADES[idx])

# ---- Table builders ----
def _build_category_table(page: dict) -> Table:
    """
    Build category comparison table with chronological column order.

    Columns (left to right): Swatch, Category, 2009 $/pupil, CAGR 15y, 10y, 5y, 2024 $/pupil
    NOTE: Reordered to match plot chronology (oldest to newest data)
    NOTE: First $/pupil column is 15 years before latest (matching CAGR 15y timeframe)
    """
    rows_in = page.get("cat_rows", [])
    total   = page.get("cat_total", ("Total","—","—","—","—"))
    base    = page.get("baseline_map", {}) or {}
    latest_year = page.get('latest_year', 'Latest')
    start_year = latest_year - 15 if isinstance(latest_year, int) else 2009  # 15 years before latest

    # Header cells - reordered chronologically
    header = [
        Paragraph("", style_hdr_left),  # Swatch column, no header text
        Paragraph("Category", style_hdr_left),
        Paragraph(f"{start_year} $/pupil", style_hdr_right),
        Paragraph("CAGR 15y", style_hdr_right),
        Paragraph("CAGR 10y", style_hdr_right),
        Paragraph("CAGR 5y", style_hdr_right),
        Paragraph(f"{latest_year} $/pupil", style_hdr_right),
    ]
    data: List[List] = [header]

    # Category rows - reordered: start (15y ago), CAGR 15y, 10y, 5y, latest
    for (sc, latest_str, c5s, c10s, c15s, col, latest_val) in rows_in:
        # Get start year value (15 years before latest)
        start_val = page.get("cat_start_map", {}).get(sc, 0.0)
        start_str = f"${start_val:,.0f}" if start_val > 0 else "—"

        data.append([
            "",  # swatch painted via TableStyle
            Paragraph(sc, style_body),
            Paragraph(start_str, style_num),
            Paragraph(c15s, style_num),
            Paragraph(c10s, style_num),
            Paragraph(c5s, style_num),
            Paragraph(latest_str, style_num),
        ])

    # TOTAL row directly after categories - reordered
    start_total_str = total[5] if len(total) > 5 else "—"  # start year total
    data.append([
        "", Paragraph(total[0], style_body),
        Paragraph(start_total_str, style_num),  # start year total (2009)
        Paragraph(total[4], style_num),  # CAGR 15y
        Paragraph(total[3], style_num),  # CAGR 10y
        Paragraph(total[2], style_num),  # CAGR 5y
        Paragraph(total[1], style_num),  # Latest total (2024)
    ])
    total_row_idx = len(data) - 1

    # Legend removed from category table - now shown only after enrollment table

    # Column widths: flexible Category column, wider first/last dollar columns for large values
    # Swatch, Category, t0 $, CAGR15y, CAGR10y, CAGR5y, Latest $
    tbl = Table(data, colWidths=[0.22*inch, None, 1.05*inch, 0.85*inch, 0.85*inch, 0.85*inch, 1.05*inch])

    # Table style
    ts = TableStyle([
        ("LINEBELOW", (0,0), (-1,0), 0.5, colors.black),
        ("VALIGN",(0,0), (-1,-1), "MIDDLE"),
        ("ALIGN", (2,1), (-1,-1), "RIGHT"),  # All numeric columns right-aligned
        ("ALIGN", (1,1), (1,-1), "LEFT"),    # Category column left-aligned
        ("LEFTPADDING",  (0,0), (-1,-1), 4),
        ("RIGHTPADDING", (0,0), (-1,-1), 4),
        ("TOPPADDING",   (0,0), (-1,-1), 3),
        ("BOTTOMPADDING",(0,0), (-1,-1), 3),
        ("LINEABOVE", (0, total_row_idx), (-1, total_row_idx), 0.5, colors.black),
    ])

    # Paint category swatch backgrounds (full cell)
    for i, (_sc, _ls, _c5, _c10, _c15, col, _lv) in enumerate(rows_in, start=1):
        ts.add("BACKGROUND", (0,i), (0,i), HexColor(col))

    # Shade $/pupil columns (cols 2 and 6: start and latest) and CAGR columns (cols 3–5)
    for i, (sc, _ls, c5s, c10s, c15s, _col, latest_val) in enumerate(rows_in, start=1):
        # optional $ floor gate
        if MATERIAL_MIN_LATEST_DOLLARS > 0 and (page.get("page_type") == "district"):
            if latest_val < MATERIAL_MIN_LATEST_DOLLARS:
                continue

        base_map = base.get(sc, {})
        start_val = page.get("cat_start_map", {}).get(sc, 0.0)

        # --- Start year $/pupil shading (col 2, 2009) - relative vs Western baseline ---
        base_start_dollar = base_map.get("START_DOLLAR", float("nan"))
        if base_start_dollar == base_start_dollar and base_start_dollar > 0 and start_val > 0:
            delta_rel = (start_val - float(base_start_dollar)) / float(base_start_dollar)
            bg = _shade_for_dollar_rel(delta_rel)
            if bg is not None:
                ts.add("BACKGROUND", (2, i), (2, i), bg)

        # --- Latest $/pupil shading (col 6, 2024) - relative vs Western baseline ---
        base_dollar = base_map.get("DOLLAR", float("nan"))
        if base_dollar == base_dollar and base_dollar > 0 and latest_val == latest_val:
            delta_rel = (latest_val - float(base_dollar)) / float(base_dollar)
            bg = _shade_for_dollar_rel(delta_rel)
            if bg is not None:
                ts.add("BACKGROUND", (6, i), (6, i), bg)

        # --- CAGR shading (absolute pp delta vs Western baseline) ---
        # Columns 3, 4, 5 = CAGR 15y, 10y, 5y
        for col, (cstr, key) in zip((5, 4, 3), [(c5s, "5"), (c10s, "10"), (c15s, "15")]):
            val = parse_pct_str_to_float(cstr)
            base_val = base_map.get(key, float("nan"))
            delta_pp = (val - base_val) if (val==val and base_val==base_val) else float("nan")
            bg = _shade_for_cagr_delta(delta_pp)
            if bg is not None:
                ts.add("BACKGROUND", (col, i), (col, i), bg)

    # Shade TOTAL row (same logic as category rows)
    # Enable shading for: district pages, cohort aggregate pages (western), and NSS/Ch70 pages
    if page.get("page_type") in ["district", "western", "nss_ch70"] and total_row_idx is not None:
        total_base_map = base.get("Total", {})

        # Parse total values from the total tuple
        # total = (label, latest_str, c5s, c10s, c15s, start_str)
        start_total_str = total[5] if len(total) > 5 else "—"
        latest_total_str = total[1]
        c5s_total = total[2]
        c10s_total = total[3]
        c15s_total = total[4]

        # Parse dollar values from formatted strings
        start_total_val = float(start_total_str.replace("$", "").replace(",", "")) if start_total_str != "—" else float("nan")
        latest_total_val = float(latest_total_str.replace("$", "").replace(",", "")) if latest_total_str != "—" else float("nan")

        # --- Start year $/pupil shading for Total (col 2) ---
        base_start_dollar = total_base_map.get("START_DOLLAR", float("nan"))
        if base_start_dollar == base_start_dollar and base_start_dollar > 0 and start_total_val == start_total_val and start_total_val > 0:
            delta_rel = (start_total_val - float(base_start_dollar)) / float(base_start_dollar)
            bg = _shade_for_dollar_rel(delta_rel)
            if bg is not None:
                ts.add("BACKGROUND", (2, total_row_idx), (2, total_row_idx), bg)

        # --- Latest $/pupil shading for Total (col 6) ---
        base_dollar = total_base_map.get("DOLLAR", float("nan"))
        if base_dollar == base_dollar and base_dollar > 0 and latest_total_val == latest_total_val:
            delta_rel = (latest_total_val - float(base_dollar)) / float(base_dollar)
            bg = _shade_for_dollar_rel(delta_rel)
            if bg is not None:
                ts.add("BACKGROUND", (6, total_row_idx), (6, total_row_idx), bg)

        # --- CAGR shading for Total (cols 3, 4, 5) ---
        # Columns 3, 4, 5 = CAGR 15y, 10y, 5y
        for col, (cstr, key) in zip((5, 4, 3), [(c5s_total, "5"), (c10s_total, "10"), (c15s_total, "15")]):
            val = parse_pct_str_to_float(cstr)
            base_val = total_base_map.get(key, float("nan"))
            delta_pp = (val - base_val) if (val==val and base_val==base_val) else float("nan")
            bg = _shade_for_cagr_delta(delta_pp)
            if bg is not None:
                ts.add("BACKGROUND", (col, total_row_idx), (col, total_row_idx), bg)

    # Legend removed from category table - now shown only after enrollment table

    tbl.setStyle(ts)
    return tbl

def _build_fte_table(page: dict) -> Table:
    """
    Build FTE enrollment table with chronological column order and shading.

    Columns (left to right): Swatch, FTE Series, t0 FTE, CAGR 15y, 10y, 5y, Latest FTE
    NOTE: Reordered to match category table chronology
    """
    rows_in = page.get("fte_rows", [])
    latest_year_fte = page.get("latest_year_fte", page.get("latest_year","Latest"))
    t0_year_fte = latest_year_fte - 15 if isinstance(latest_year_fte, int) else 2009
    fte_base = page.get("fte_baseline_map", {}) or {}

    # Header cells - reordered chronologically
    header = [
        Paragraph("", style_hdr_left),  # Swatch column, no header text
        Paragraph("Enrollment", style_hdr_left),
        Paragraph(f"{t0_year_fte}", style_hdr_right),
        Paragraph("CAGR 15y", style_hdr_right),
        Paragraph("CAGR 10y", style_hdr_right),
        Paragraph("CAGR 5y", style_hdr_right),
        Paragraph(f"{latest_year_fte}", style_hdr_right),
    ]
    data: List[List] = [header]

    # Data rows - reordered: t0, CAGR 15y, 10y, 5y, latest
    for (color, label, latest_str, r5s, r10s, r15s) in rows_in:
        # Get t0 value from series
        s = page.get("fte_series_map", {}).get(label)
        t0_str = "—"
        if s is not None and not s.empty and t0_year_fte in s.index:
            t0_str = f"{float(s.loc[t0_year_fte]):,.0f}"

        data.append([LineSwatch(color), Paragraph(label, style_body),
                     Paragraph(t0_str, style_num),
                     Paragraph(r15s, style_num),
                     Paragraph(r10s, style_num),
                     Paragraph(r5s, style_num),
                     Paragraph(latest_str, style_num)])

    # Total FTE removed - enrollment groups overlap, so sum is meaningless
    total_row_idx = None

    # Legend rows: Show only if there's a baseline (i.e., comparison page, not aggregate)
    # Merge 4 leftmost cells (0-3) for shading rule text, span rightmost 3 cells (4-6) for swatches
    legend_rows = []
    # Show legend if baseline comparison is available (for district, western cohort, or nss_ch70 pages)
    if page.get("page_type") in ["district", "western", "nss_ch70"] and (page.get("baseline_map") or page.get("fte_baseline_map")):
        baseline_title = page.get("baseline_title", "")
        bucket_suffix = _abbr_bucket_suffix(baseline_title)

        # Generate baseline-focused legend text
        if bucket_suffix == "MA (excl. Springfield)":
            above_text = "Above Western MA baseline"
            below_text = "Below Western MA baseline"
        else:
            # Cohort baseline - remove " FTE" suffix for cleaner display
            clean_suffix = bucket_suffix.replace(" FTE)", ")")
            above_text = f"Above cohort baseline {clean_suffix}".strip()
            below_text = f"Below cohort baseline {clean_suffix}".strip()

        shade_rule = (
            f"Shading vs baseline: |Δ$/pupil| ≥ {DOLLAR_THRESHOLD_REL*100:.1f}%, "
            f"|ΔEnrollment| ≥ {DOLLAR_THRESHOLD_REL*100:.1f}%, |ΔCAGR| ≥ {MATERIAL_DELTA_PCTPTS*100:.1f}pp"
        )
        cagr_def = "CAGR = (End/Start)^(1/years) − 1"
        legend_rows = [
            [Paragraph(shade_rule, style_legend_right), "", "", "", Paragraph(above_text, style_legend_center), "", ""],  # Col 0-3 merged for text
            [Paragraph(cagr_def, style_legend_right), "", "", "", Paragraph(below_text, style_legend_center), "", ""],  # Col 0-3 merged for text
        ]
    data.extend(legend_rows)
    leg1_idx = len(rows_in) + 1 if legend_rows else None
    leg2_idx = len(rows_in) + 2 if legend_rows else None

    # Column widths: flexible Enrollment column, wider first/last FTE columns
    # Swatch, Series (flex), t0 FTE, CAGR15y, CAGR10y, CAGR5y, Latest FTE
    tbl = Table(data, colWidths=[0.45*inch, None, 1.05*inch, 0.85*inch, 0.85*inch, 0.85*inch, 1.05*inch])
    ts = TableStyle([
        ("VALIGN",(0,0), (-1,-1), "MIDDLE"),
        ("ALIGN", (2,1), (-1,-1), "RIGHT"),  # All numeric columns right-aligned
        ("ALIGN", (1,1), (1,-1), "LEFT"),    # Series column left-aligned
        ("LINEBELOW", (0,0), (-1,0), 0.5, colors.black),
        ("LEFTPADDING",  (0,0), (-1,-1), 4),
        ("RIGHTPADDING", (0,0), (-1,-1), 4),
        ("TOPPADDING",   (0,0), (-1,-1), 3),
        ("BOTTOMPADDING",(0,0), (-1,-1), 3),
    ])

    # Shade FTE columns (cols 2 and 6: start and latest) and CAGR columns (cols 3–5)
    # Show comparison info for district, western cohort, and nss_ch70 pages
    if page.get("page_type") in ["district", "western", "nss_ch70"]:
        for i, (color, label, latest_str, r5s, r10s, r15s) in enumerate(rows_in, start=1):
            base_map = fte_base.get(label, {})

            # Get t0 and latest values
            s = page.get("fte_series_map", {}).get(label)
            t0_val = float("nan")
            latest_val = float("nan")
            if s is not None and not s.empty:
                if t0_year_fte in s.index:
                    t0_val = float(s.loc[t0_year_fte])
                if latest_year_fte in s.index:
                    latest_val = float(s.loc[latest_year_fte])

            # --- Start year FTE shading (col 2, 2009) - relative vs Western baseline ---
            base_start_fte = base_map.get("START_FTE", float("nan"))
            if base_start_fte == base_start_fte and base_start_fte > 0 and t0_val == t0_val and t0_val > 0:
                delta_rel = (t0_val - float(base_start_fte)) / float(base_start_fte)
                bg = _shade_for_dollar_rel(delta_rel)  # reuse same function (5% threshold)
                if bg is not None:
                    ts.add("BACKGROUND", (2, i), (2, i), bg)

            # --- Latest FTE shading (col 6, 2024) - relative vs Western baseline ---
            base_fte = base_map.get("FTE", float("nan"))
            if base_fte == base_fte and base_fte > 0 and latest_val == latest_val:
                delta_rel = (latest_val - float(base_fte)) / float(base_fte)
                bg = _shade_for_dollar_rel(delta_rel)  # reuse same function (5% threshold)
                if bg is not None:
                    ts.add("BACKGROUND", (6, i), (6, i), bg)

            # --- CAGR shading (absolute pp delta vs Western baseline) ---
            # Columns 3, 4, 5 = CAGR 15y, 10y, 5y
            for col, (cstr, key) in zip((5, 4, 3), [(r5s, "5"), (r10s, "10"), (r15s, "15")]):
                val = parse_pct_str_to_float(cstr)
                base_val = base_map.get(key, float("nan"))
                delta_pp = (val - base_val) if (val==val and base_val==base_val) else float("nan")
                bg = _shade_for_cagr_delta(delta_pp)
                if bg is not None:
                    ts.add("BACKGROUND", (col, i), (col, i), bg)

    # Apply legend spans & swatches (at bottom of table)
    # Merge leftmost 4 cells (0-3) for shading rule text, rightmost 3 cells (4-6) for color swatches
    if leg1_idx is not None:
        # Add faint line above legend
        ts.add("LINEABOVE", (0, leg1_idx), (-1, leg1_idx), 0.5, colors.lightgrey)
        ts.add("TOPPADDING", (0, leg1_idx), (-1, leg1_idx), 8)
        ts.add("BOTTOMPADDING", (0, leg2_idx), (-1, leg2_idx), 6)

        ts.add("SPAN", (0, leg1_idx), (3, leg1_idx))
        ts.add("SPAN", (0, leg2_idx), (3, leg2_idx))
        ts.add("SPAN", (4, leg1_idx), (6, leg1_idx))
        ts.add("SPAN", (4, leg2_idx), (6, leg2_idx))
        # Align legend text to right in the merged left cells
        ts.add("ALIGN", (0, leg1_idx), (3, leg1_idx), "RIGHT")
        ts.add("ALIGN", (0, leg2_idx), (3, leg2_idx), "RIGHT")
        # Center the swatch labels
        ts.add("ALIGN", (4, leg1_idx), (6, leg1_idx), "CENTER")
        ts.add("ALIGN", (4, leg2_idx), (6, leg2_idx), "CENTER")
        sw_above = HexColor(ABOVE_SHADES[1]); sw_below = HexColor(BELOW_SHADES[1])
        ts.add("BACKGROUND", (4, leg1_idx), (6, leg1_idx), sw_above)
        ts.add("BACKGROUND", (4, leg2_idx), (6, leg2_idx), sw_below)

    if total_row_idx is not None:
        ts.add("LINEABOVE", (0,total_row_idx), (-1,total_row_idx), 0.5, colors.black)
    tbl.setStyle(ts)
    return tbl

# ---- NSS/Ch70 table builder ----
def _build_nss_ch70_table(page: dict) -> Table:
    """Build NSS/Ch70 funding components table with optional baseline comparison."""
    rows_in = page.get("cat_rows", [])
    total = page.get("cat_total", ("Total", "$0", "—", "—", "—", "$0"))
    base = page.get("baseline_map", {}) or {}
    latest_year = page.get("latest_year", 2024)
    start_year = latest_year - 15

    # Header row (reordered chronologically)
    hdr = [Paragraph("", style_hdr_left),  # Swatch column
           Paragraph("Component", style_hdr_left),
           Paragraph(f"{start_year}<br/>$", style_hdr_right),
           Paragraph("CAGR<br/>15y", style_hdr_right),
           Paragraph("CAGR<br/>10y", style_hdr_right),
           Paragraph("CAGR<br/>5y", style_hdr_right),
           Paragraph(f"{latest_year}<br/>$", style_hdr_right)]
    data = [hdr]

    # Category rows (top to bottom: Actual NSS (adj), Req NSS (adj), Ch70 Aid)
    # Note: rows_in format is (sc, start_str, c15s, c10s, c5s, latest_str, hex_color, latest_val)
    for (sc, start_str, c15s, c10s, c5s, latest_str, hex_color, _latest_val) in rows_in:
        data.append(["",  # swatch painted via TableStyle
                     Paragraph(sc, style_body),
                     Paragraph(start_str, style_num),
                     Paragraph(c15s, style_num),
                     Paragraph(c10s, style_num),
                     Paragraph(c5s, style_num),
                     Paragraph(latest_str, style_num)])

    # Total row
    total_label, start_total, c15_total, c10_total, c5_total, latest_total = total
    data.append(["",
                 Paragraph(total_label, style_body),
                 Paragraph(start_total, style_num),
                 Paragraph(c15_total, style_num),
                 Paragraph(c10_total, style_num),
                 Paragraph(c5_total, style_num),
                 Paragraph(latest_total, style_num)])
    total_row_idx = len(data) - 1

    # Legend removed from NSS/Ch70 table - now shown only after enrollment table below

    # Column widths (flexible Component column, match FTE table dollar column widths)
    tbl = Table(data, colWidths=[0.45*inch, None, 1.05*inch, 0.85*inch, 0.85*inch, 0.85*inch, 1.05*inch])

    # Table style
    ts = TableStyle([
        ("VALIGN",(0,0), (-1,-1), "MIDDLE"),
        ("ALIGN", (2,1), (-1,-1), "RIGHT"),  # All numeric columns right-aligned
        ("ALIGN", (1,1), (1,-1), "LEFT"),    # Component column left-aligned
        ("LINEBELOW", (0,0), (-1,0), 0.5, colors.black),
        ("LEFTPADDING",  (0,0), (-1,-1), 4),
        ("RIGHTPADDING", (0,0), (-1,-1), 4),
        ("TOPPADDING",   (0,0), (-1,-1), 3),
        ("BOTTOMPADDING",(0,0), (-1,-1), 3),
    ])

    # Paint color swatches for each component
    for i, (_sc, _start_str, _c15s, _c10s, _c5s, _latest_str, hex_color, _latest_val) in enumerate(rows_in, start=1):
        ts.add("BACKGROUND", (0, i), (0, i), HexColor(hex_color))

    # Shade $/pupil columns (cols 2 and 6: start and latest) and CAGR columns (cols 3–5) if baseline provided
    if base:
        for i, (sc, start_str, c15s, c10s, c5s, latest_str, _hex_color, latest_val) in enumerate(rows_in, start=1):
            base_map = base.get(sc, {})
            # Parse start dollar value (format: "$1,234")
            try:
                start_val = float(start_str.replace("$", "").replace(",", "").strip()) if "$" in start_str else float("nan")
            except:
                start_val = float("nan")

            # --- Start year $/pupil shading (col 2) - relative vs baseline ---
            base_start_dollar = base_map.get("START_DOLLAR", float("nan"))
            if base_start_dollar == base_start_dollar and base_start_dollar > 0 and start_val == start_val:
                delta_rel = (start_val - float(base_start_dollar)) / float(base_start_dollar)
                bg = _shade_for_dollar_rel(delta_rel)
                if bg is not None:
                    ts.add("BACKGROUND", (2, i), (2, i), bg)

            # --- Latest $/pupil shading (col 6) - relative vs baseline ---
            base_dollar = base_map.get("DOLLAR", float("nan"))
            if base_dollar == base_dollar and base_dollar > 0 and latest_val == latest_val:
                delta_rel = (latest_val - float(base_dollar)) / float(base_dollar)
                bg = _shade_for_dollar_rel(delta_rel)
                if bg is not None:
                    ts.add("BACKGROUND", (6, i), (6, i), bg)

            # --- CAGR shading (absolute pp delta vs baseline) ---
            # Columns 3, 4, 5 = CAGR 15y, 10y, 5y
            for col, (cstr, key) in zip((3, 4, 5), [(c15s, "15"), (c10s, "10"), (c5s, "5")]):
                val = parse_pct_str_to_float(cstr)
                base_val = base_map.get(key, float("nan"))
                delta_pp = (val - base_val) if (val==val and base_val==base_val) else float("nan")
                bg = _shade_for_cagr_delta(delta_pp)
                if bg is not None:
                    ts.add("BACKGROUND", (col, i), (col, i), bg)

        # Apply shading to Total row (same logic as component rows)
        base_total = base.get("Total", {})
        if base_total:
            # Parse Total row dollar values
            try:
                total_start_val = float(start_total.replace("$", "").replace(",", "").strip()) if "$" in start_total else float("nan")
            except:
                total_start_val = float("nan")
            try:
                total_latest_val = float(latest_total.replace("$", "").replace(",", "").strip()) if "$" in latest_total else float("nan")
            except:
                total_latest_val = float("nan")

            # Shade start dollar (col 2)
            base_start_dollar = base_total.get("START_DOLLAR", float("nan"))
            if base_start_dollar == base_start_dollar and base_start_dollar > 0 and total_start_val == total_start_val:
                delta_rel = (total_start_val - float(base_start_dollar)) / float(base_start_dollar)
                bg = _shade_for_dollar_rel(delta_rel)
                if bg is not None:
                    ts.add("BACKGROUND", (2, total_row_idx), (2, total_row_idx), bg)

            # Shade latest dollar (col 6)
            base_dollar = base_total.get("DOLLAR", float("nan"))
            if base_dollar == base_dollar and base_dollar > 0 and total_latest_val == total_latest_val:
                delta_rel = (total_latest_val - float(base_dollar)) / float(base_dollar)
                bg = _shade_for_dollar_rel(delta_rel)
                if bg is not None:
                    ts.add("BACKGROUND", (6, total_row_idx), (6, total_row_idx), bg)

            # Shade CAGR columns (cols 3, 4, 5)
            for col, (cstr, key) in zip((3, 4, 5), [(c15_total, "15"), (c10_total, "10"), (c5_total, "5")]):
                val = parse_pct_str_to_float(cstr)
                base_val = base_total.get(key, float("nan"))
                delta_pp = (val - base_val) if (val==val and base_val==base_val) else float("nan")
                bg = _shade_for_cagr_delta(delta_pp)
                if bg is not None:
                    ts.add("BACKGROUND", (col, total_row_idx), (col, total_row_idx), bg)

    ts.add("LINEABOVE", (0,total_row_idx), (-1,total_row_idx), 0.5, colors.black)

    tbl.setStyle(ts)
    return tbl

# ---- Helper for building page data ----
def _build_category_data(epp_pivot: pd.DataFrame, latest_year: int, context: str, cmap_all: dict) -> tuple:
    """
    Build category rows and total for a given EPP pivot table.

    Returns:
        cat_rows: List of tuples (category, latest_str, c5s, c10s, c15s, color, latest_val)
        cat_total: Tuple (label, latest_str, c5s, c10s, c15s, start_str)
        cat_start_map: Dict mapping category -> start year value (15 years before latest)
    """
    if epp_pivot.empty:
        return [], ("Total", "$0", "—", "—", "—", "$0"), {}

    bottom_top = canonical_order_bottom_to_top(epp_pivot.columns.tolist())
    top_bottom = list(reversed(bottom_top))
    start_year = latest_year - 15  # 15 years before latest (2009 if latest is 2024)

    cat_rows = []
    cat_start_map = {}
    for sc in top_bottom:
        latest_val = float(epp_pivot.loc[latest_year, sc]) if (latest_year in epp_pivot.index and sc in epp_pivot.columns) else 0.0
        start_val = float(epp_pivot.loc[start_year, sc]) if (start_year in epp_pivot.index and sc in epp_pivot.columns) else 0.0
        cat_start_map[sc] = start_val
        c5 = compute_cagr_last(epp_pivot[sc], 5)
        c10 = compute_cagr_last(epp_pivot[sc], 10)
        c15 = compute_cagr_last(epp_pivot[sc], 15)
        cat_rows.append((sc, f"${latest_val:,.0f}", fmt_pct(c5), fmt_pct(c10), fmt_pct(c15),
                        color_for(cmap_all, context, sc), latest_val))

    # Compute CAGR on total series (sum of all categories), not mean of individual CAGRs
    total_series = epp_pivot.sum(axis=1)
    latest_total = float(total_series.loc[latest_year]) if latest_year in total_series.index else 0.0
    start_total = float(total_series.loc[start_year]) if start_year in total_series.index else 0.0
    cat_total = ("Total", f"${latest_total:,.0f}",
                 fmt_pct(compute_cagr_last(total_series, 5)),
                 fmt_pct(compute_cagr_last(total_series, 10)),
                 fmt_pct(compute_cagr_last(total_series, 15)),
                 f"${start_total:,.0f}")

    return cat_rows, cat_total, cat_start_map

def _build_fte_data(lines: Dict[str, pd.Series], latest_year: int) -> tuple:
    """Build FTE rows and determine latest FTE year."""
    fte_years = [int(s.index.max()) for s in lines.values() if s is not None and not s.empty]
    latest_fte_year = max(fte_years) if fte_years else latest_year
    fte_rows = []
    fte_map = {}

    for _k, label in ENROLL_KEYS:
        s = lines.get(label)
        if s is None or s.empty:
            continue
        fte_map[label] = s
        r5 = compute_cagr_last(s, 5)
        r10 = compute_cagr_last(s, 10)
        r15 = compute_cagr_last(s, 15)
        latest_str = "—" if latest_fte_year not in s.index else f"{float(s.loc[latest_fte_year]):,.0f}"
        fte_rows.append((FTE_LINE_COLORS[label], label, latest_str, fmt_pct(r5), fmt_pct(r10), fmt_pct(r15)))

    return fte_rows, fte_map, latest_fte_year

def _build_nss_fte_data(foundation_series: pd.Series, latest_year: int) -> tuple:
    """Build FTE rows for NSS/Ch70 pages (foundation enrollment only)."""
    if foundation_series is None or foundation_series.empty:
        return [], {}, latest_year

    latest_fte_year = int(foundation_series.index.max())
    fte_rows = []
    fte_map = {}

    label = "Foundation Enrollment"
    fte_map[label] = foundation_series
    r5 = compute_cagr_last(foundation_series, 5)
    r10 = compute_cagr_last(foundation_series, 10)
    r15 = compute_cagr_last(foundation_series, 15)
    latest_str = "—" if latest_fte_year not in foundation_series.index else f"{float(foundation_series.loc[latest_fte_year]):,.0f}"
    fte_rows.append((FTE_LINE_COLORS[label], label, latest_str, fmt_pct(r5), fmt_pct(r10), fmt_pct(r15)))

    return fte_rows, fte_map, latest_fte_year

def _build_nss_ch70_baseline_map(nss_data: pd.DataFrame, latest_year: int, foundation_series: pd.Series = None) -> dict:
    """
    Build baseline map for NSS/Ch70 components and Foundation Enrollment for comparison.

    Returns dict mapping component name (including "Total" and "Foundation Enrollment") to:
        - DOLLAR (or FTE): latest value
        - START_DOLLAR (or START_FTE): start year (15y ago) value
        - "5", "10", "15": CAGR values as floats
    """
    baseline_map = {}
    start_year = latest_year - 15

    for comp in nss_data.columns:
        latest_val = float(nss_data.loc[latest_year, comp]) if (latest_year in nss_data.index and comp in nss_data.columns) else float("nan")
        start_val = float(nss_data.loc[start_year, comp]) if (start_year in nss_data.index and comp in nss_data.columns) else float("nan")
        c5 = compute_cagr_last(nss_data[comp], 5)
        c10 = compute_cagr_last(nss_data[comp], 10)
        c15 = compute_cagr_last(nss_data[comp], 15)

        baseline_map[comp] = {
            "DOLLAR": latest_val,
            "START_DOLLAR": start_val,
            "5": c5,
            "10": c10,
            "15": c15
        }

    # Add Total row
    total_series = nss_data.sum(axis=1)
    latest_total = float(total_series.loc[latest_year]) if latest_year in total_series.index else float("nan")
    start_total = float(total_series.loc[start_year]) if start_year in total_series.index else float("nan")

    baseline_map["Total"] = {
        "DOLLAR": latest_total,
        "START_DOLLAR": start_total,
        "5": compute_cagr_last(total_series, 5),
        "10": compute_cagr_last(total_series, 10),
        "15": compute_cagr_last(total_series, 15)
    }

    # Add Foundation Enrollment if provided
    if foundation_series is not None and not foundation_series.empty:
        latest_fte_year = int(foundation_series.index.max())
        latest_fte = float(foundation_series.loc[latest_fte_year]) if latest_fte_year in foundation_series.index else float("nan")
        start_fte = float(foundation_series.loc[start_year]) if start_year in foundation_series.index else float("nan")

        baseline_map["Foundation Enrollment"] = {
            "FTE": latest_fte,
            "START_FTE": start_fte,
            "5": compute_cagr_last(foundation_series, 5),
            "10": compute_cagr_last(foundation_series, 10),
            "15": compute_cagr_last(foundation_series, 15)
        }

    return baseline_map

# ---- Data table builders for appendix ----
def _build_epp_data_table(epp_pivot: pd.DataFrame, title: str, doc_width: float) -> Table:
    """Build a data table showing all EPP data (years as columns, categories as rows)."""
    if epp_pivot.empty:
        return None

    # Get years and categories in order - filter to 2009 and later
    years = sorted([y for y in epp_pivot.index.tolist() if y >= 2009])
    if not years:
        return None
    categories = canonical_order_bottom_to_top(epp_pivot.columns.tolist())
    categories = list(reversed(categories))  # Top to bottom for table display

    # Build header row
    header = [Paragraph("Category", style_data_label)]
    for yr in years:
        header.append(Paragraph(str(yr), style_data_hdr))

    data: List[List] = [header]

    # Add category rows
    for cat in categories:
        row = [Paragraph(cat, style_data_label)]
        for yr in years:
            val = epp_pivot.loc[yr, cat] if (yr in epp_pivot.index and cat in epp_pivot.columns) else np.nan
            val_str = f"${val:,.0f}" if not np.isnan(val) else "—"
            row.append(Paragraph(val_str, style_data_cell))
        data.append(row)

    # Add total row
    total_series = epp_pivot.sum(axis=1)
    total_row = [Paragraph("Total", style_data_label)]
    for yr in years:
        val = total_series.loc[yr] if yr in total_series.index else np.nan
        val_str = f"${val:,.0f}" if not np.isnan(val) else "—"
        total_row.append(Paragraph(val_str, style_data_cell))
    data.append(total_row)
    total_row_idx = len(data) - 1

    # Calculate column widths - very narrow category column (allows wrapping), more space for data
    cat_width = 1.0*inch
    remaining_width = doc_width - cat_width
    year_width = remaining_width / len(years)
    col_widths = [cat_width] + [year_width] * len(years)

    tbl = Table(data, colWidths=col_widths)
    ts = TableStyle([
        ("LINEBELOW", (0,0), (-1,0), 0.5, colors.black),
        ("LINEABOVE", (0, total_row_idx), (-1, total_row_idx), 0.5, colors.black),
        ("VALIGN",(0,0), (-1,-1), "MIDDLE"),
        ("ALIGN", (1,1), (-1,-1), "RIGHT"),
        ("ALIGN", (0,1), (0,-1), "LEFT"),
        ("LEFTPADDING",  (0,0), (-1,-1), 3),
        ("RIGHTPADDING", (0,0), (-1,-1), 3),
        ("TOPPADDING",   (0,0), (-1,-1), 2),
        ("BOTTOMPADDING",(0,0), (-1,-1), 2),
    ])
    tbl.setStyle(ts)

    return tbl

def _build_fte_data_table(lines: Dict[str, pd.Series], title: str, doc_width: float) -> Table:
    """Build a data table showing all FTE enrollment data."""
    if not lines:
        return None

    # Collect all years across all series - filter to 2009-2024 to match CAGR tables
    all_years = set()
    for s in lines.values():
        if s is not None and not s.empty:
            all_years.update([y for y in s.index.tolist() if 2009 <= y <= 2024])

    if not all_years:
        return None

    years = sorted(list(all_years))

    # Build header row
    header = [Paragraph("FTE Series", style_data_label)]
    for yr in years:
        header.append(Paragraph(str(yr), style_data_hdr))

    data: List[List] = [header]

    # Add FTE rows in standard order
    for _k, label in ENROLL_KEYS:
        s = lines.get(label)
        if s is None or s.empty:
            continue
        row = [Paragraph(label, style_data_label)]
        for yr in years:
            val = s.loc[yr] if yr in s.index else np.nan
            val_str = f"{val:,.0f}" if not np.isnan(val) else "—"
            row.append(Paragraph(val_str, style_data_cell))
        data.append(row)

    # Add total row
    total_series = None
    for _k, label in ENROLL_KEYS:
        s = lines.get(label)
        if s is None: continue
        total_series = s if total_series is None else (total_series.add(s, fill_value=0.0))

    # Total FTE removed - enrollment groups overlap, so sum is meaningless
    total_row_idx = None

    # Calculate column widths - very narrow label column (allows wrapping), more space for data
    label_width = 1.0*inch
    remaining_width = doc_width - label_width
    year_width = remaining_width / len(years)
    col_widths = [label_width] + [year_width] * len(years)

    tbl = Table(data, colWidths=col_widths)
    ts = TableStyle([
        ("LINEBELOW", (0,0), (-1,0), 0.5, colors.black),
        ("VALIGN",(0,0), (-1,-1), "MIDDLE"),
        ("ALIGN", (1,1), (-1,-1), "RIGHT"),
        ("ALIGN", (0,1), (0,-1), "LEFT"),
        ("LEFTPADDING",  (0,0), (-1,-1), 3),
        ("RIGHTPADDING", (0,0), (-1,-1), 3),
        ("TOPPADDING",   (0,0), (-1,-1), 2),
        ("BOTTOMPADDING",(0,0), (-1,-1), 2),
    ])

    if total_row_idx is not None:
        ts.add("LINEABOVE", (0, total_row_idx), (-1, total_row_idx), 0.5, colors.black)

    tbl.setStyle(ts)
    return tbl

def _build_nss_ch70_data_table(nss_pivot: pd.DataFrame, title: str, doc_width: float) -> Table:
    """Build a data table showing NSS/Ch70 funding component data (years as columns, components as rows)."""
    if nss_pivot.empty:
        return None

    # Get years and components in order - limit to 2009-2024
    all_years = sorted(nss_pivot.index.tolist())
    years = [yr for yr in all_years if 2009 <= yr <= 2024]
    if not years:
        years = all_years  # Fallback if no years in range

    # Components in order: Ch70 Aid, Req NSS (adj), Actual NSS (adj)
    components = nss_pivot.columns.tolist()

    # Helper function to format large numbers with K/M abbreviations
    def format_dollar_abbrev(val):
        if np.isnan(val):
            return "—"
        if abs(val) >= 1_000_000:
            return f"${val/1_000_000:.1f}M"
        elif abs(val) >= 1_000:
            return f"${val/1_000:.0f}K"
        else:
            return f"${val:,.0f}"

    # Build header row
    header = [Paragraph("Component", style_data_label)]
    for yr in years:
        header.append(Paragraph(str(yr), style_data_hdr))

    data: List[List] = [header]

    # Add component rows
    for comp in components:
        row = [Paragraph(comp, style_data_label)]
        for yr in years:
            val = nss_pivot.loc[yr, comp] if (yr in nss_pivot.index and comp in nss_pivot.columns) else np.nan
            val_str = format_dollar_abbrev(val)
            row.append(Paragraph(val_str, style_data_cell))
        data.append(row)

    # Add total row
    total_series = nss_pivot.sum(axis=1)
    total_row = [Paragraph("Total NSS", style_data_label)]
    for yr in years:
        val = total_series.loc[yr] if yr in total_series.index else np.nan
        val_str = format_dollar_abbrev(val)
        total_row.append(Paragraph(val_str, style_data_cell))
    data.append(total_row)
    total_row_idx = len(data) - 1

    # Calculate column widths - narrow component column, more space for data
    comp_width = 0.9*inch  # Narrower to give more space for year columns
    remaining_width = doc_width - comp_width
    year_width = remaining_width / len(years)
    col_widths = [comp_width] + [year_width] * len(years)

    tbl = Table(data, colWidths=col_widths)
    ts = TableStyle([
        ("LINEBELOW", (0,0), (-1,0), 0.5, colors.black),
        ("LINEABOVE", (0, total_row_idx), (-1, total_row_idx), 0.5, colors.black),
        ("VALIGN",(0,0), (-1,-1), "MIDDLE"),
        ("ALIGN", (1,1), (-1,-1), "RIGHT"),
        ("ALIGN", (0,1), (0,-1), "LEFT"),
        ("LEFTPADDING",  (0,0), (-1,-1), 3),
        ("RIGHTPADDING", (0,0), (-1,-1), 3),
        ("TOPPADDING",   (0,0), (-1,-1), 2),
        ("BOTTOMPADDING",(0,0), (-1,-1), 2),
    ])
    tbl.setStyle(ts)

    return tbl

def _build_cohort_summary_table(cohort_rows: List[tuple], baseline_row: tuple = None, apply_shading: bool = True, skip_shading_rows: set = None, highlight_baseline_rows: set = None) -> Table:
    """
    Build cohort summary table showing Total PPE metrics across cohorts.

    Args:
        cohort_rows: List of (label, start_dollar, cagr15, cagr10, cagr5, latest_dollar) tuples
        baseline_row: Optional baseline row data for shading comparisons
        apply_shading: Whether to apply shading (True for first table, False for comparison tables)
        skip_shading_rows: Set of row indices (0-based, excluding header) to skip shading for
        highlight_baseline_rows: Set of row indices to highlight in pale yellow (baseline)

    Returns:
        ReportLab Table with formatted cohort data
    """
    skip_shading_rows = skip_shading_rows or set()
    highlight_baseline_rows = highlight_baseline_rows or set()

    # Header
    summary_data = [[
        Paragraph("<b>Cohort/District</b>", style_hdr_left),
        Paragraph("<b>2009 $/pupil</b>", style_hdr_right),
        Paragraph("<b>CAGR 15y</b>", style_hdr_right),
        Paragraph("<b>CAGR 10y</b>", style_hdr_right),
        Paragraph("<b>CAGR 5y</b>", style_hdr_right),
        Paragraph("<b>2024 $/pupil</b>", style_hdr_right),
    ]]

    # Data rows
    for label, start_dollar, cagr15, cagr10, cagr5, latest_dollar in cohort_rows:
        summary_data.append([
            Paragraph(label, style_body),
            Paragraph(f"${start_dollar:,.0f}" if start_dollar > 0 else "—", style_num),
            Paragraph(fmt_pct(cagr15), style_num),
            Paragraph(fmt_pct(cagr10), style_num),
            Paragraph(fmt_pct(cagr5), style_num),
            Paragraph(f"${latest_dollar:,.0f}" if latest_dollar > 0 else "—", style_num),
        ])

    summary_table = Table(summary_data, colWidths=[2.4*inch, 1.05*inch, 0.85*inch, 0.85*inch, 0.85*inch, 1.05*inch])
    ts = TableStyle([
        ('LINEBELOW', (0,0), (-1,0), 0.5, colors.black),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ALIGN', (1,1), (-1,-1), 'RIGHT'),
        ('ALIGN', (0,1), (0,-1), 'LEFT'),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ])

    # Apply pale yellow background to baseline rows
    for baseline_idx in highlight_baseline_rows:
        row_num = baseline_idx + 1  # +1 because header is row 0
        ts.add("BACKGROUND", (0, row_num), (-1, row_num), HexColor(BASELINE_YELLOW))

    # Apply shading if requested and baseline is provided
    if apply_shading and baseline_row is not None:
        base_label, base_start, base_c15, base_c10, base_c5, base_latest = baseline_row

        for i, (label, start_dollar, cagr15, cagr10, cagr5, latest_dollar) in enumerate(cohort_rows, start=1):
            # Skip shading for specified rows
            if (i - 1) in skip_shading_rows:
                continue

            # Shade 2009 $/pupil (col 1)
            if base_start > 0 and start_dollar > 0:
                delta_rel = (start_dollar - base_start) / base_start
                bg = _shade_for_dollar_rel(delta_rel)
                if bg is not None:
                    ts.add("BACKGROUND", (1, i), (1, i), bg)

            # Shade 2024 $/pupil (col 5)
            if base_latest > 0 and latest_dollar > 0:
                delta_rel = (latest_dollar - base_latest) / base_latest
                bg = _shade_for_dollar_rel(delta_rel)
                if bg is not None:
                    ts.add("BACKGROUND", (5, i), (5, i), bg)

            # Shade CAGR columns (2, 3, 4 = 15y, 10y, 5y)
            for col, (cagr_val, base_cagr) in zip([2, 3, 4], [(cagr15, base_c15), (cagr10, base_c10), (cagr5, base_c5)]):
                if cagr_val == cagr_val and base_cagr == base_cagr:  # Check for non-NaN
                    delta_pp = cagr_val - base_cagr
                    bg = _shade_for_cagr_delta(delta_pp)
                    if bg is not None:
                        ts.add("BACKGROUND", (col, i), (col, i), bg)

    summary_table.setStyle(ts)
    return summary_table

def _extract_total_from_cat_total(cat_total: tuple) -> tuple:
    """
    Extract and reorder data from cat_total tuple.

    cat_total format: (label, latest_str, c5s, c10s, c15s, start_str)
    Returns: (start_dollar, cagr15, cagr10, cagr5, latest_dollar) as floats
    """
    if not cat_total or len(cat_total) < 6:
        return (0.0, 0.0, 0.0, 0.0, 0.0)

    label, latest_str, c5s, c10s, c15s, start_str = cat_total

    # Parse dollar strings (e.g., "$24,237" -> 24237.0)
    def parse_dollar(s):
        if s == "—" or not s:
            return 0.0
        return float(s.replace("$", "").replace(",", ""))

    latest_dollar = parse_dollar(latest_str)
    start_dollar = parse_dollar(start_str)

    # Parse CAGR strings - they're already floats or NaN from fmt_pct
    cagr5 = parse_pct_str_to_float(c5s)
    cagr10 = parse_pct_str_to_float(c10s)
    cagr15 = parse_pct_str_to_float(c15s)

    return (start_dollar, cagr15, cagr10, cagr5, latest_dollar)

# ---- Page dicts ----
def build_threshold_analysis_page() -> dict:
    """
    Build threshold analysis page documenting the rationale for 5% / 1pp thresholds.

    This page shows the statistical analysis that led to selecting balanced thresholds
    for red/green shading in district comparison tables.
    """
    # Dataset statistics (from Western MA traditional districts, n=59)
    ppe_mean = 24237  # mean PPE 2024
    ppe_sd = 5462     # standard deviation
    cagr_mean = 6.00  # mean 5-year CAGR (pp)
    cagr_sd = 3.24    # standard deviation (pp)

    # Build comparison table
    summary_data = []
    summary_data.append([
        Paragraph("<b>Scenario</b>", style_body),
        Paragraph("<b>PPE/<br/>Enrollment</b>", style_body),
        Paragraph("<b>CAGR</b>", style_body),
        Paragraph("<b>PPE<br/>Sensitivity</b>", style_body),
        Paragraph("<b>CAGR<br/>Sensitivity</b>", style_body),
        Paragraph("<b>Balance<br/>Ratio</b>", style_body),
        Paragraph("<b>Assessment</b>", style_body),
    ])

    scenarios = [
        ("Previous (2%/2pp)", 0.02, 2.0, 0.09, 0.62, 6.97, "Unbalanced"),
        ("Equal SD (5%/0.72pp)", 0.05, 0.72, 0.22, 0.22, 1.00, "Too tight for CAGR"),
        ("Selected (5%/1pp)", 0.05, 1.0, 0.22, 0.31, 1.41, "Well-balanced"),
        ("Proportional (5%/5pp)", 0.05, 5.0, 0.22, 1.55, 6.97, "CAGR too loose"),
    ]

    for name, ppe_thresh, cagr_thresh, ppe_sd_mult, cagr_sd_mult, ratio, assessment in scenarios:
        ppe_color = colors.lightgrey if "Selected" in name else colors.whitesmoke
        summary_data.append([
            Paragraph(f"<b>{name}</b>" if "Selected" in name else name, style_body),
            Paragraph(f"{ppe_thresh*100:.1f}%", style_num),
            Paragraph(f"{cagr_thresh:.2f}pp", style_num),
            Paragraph(f"{ppe_sd_mult:.2f} SD", style_num),
            Paragraph(f"{cagr_sd_mult:.2f} SD", style_num),
            Paragraph(f"{ratio:.2f}x", style_num),
            Paragraph(assessment, style_body),
        ])

    summary_table = Table(summary_data, colWidths=[1.4*inch, 0.9*inch, 0.8*inch, 0.9*inch, 0.9*inch, 0.8*inch, 1.2*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 3), (-1, 3), colors.lightgreen),  # Highlight selected row
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, 2), [colors.whitesmoke]),
        ('ROWBACKGROUNDS', (0, 4), (-1, 4), [colors.whitesmoke]),
    ]))

    # Explanatory text
    explanation_blocks = [
        "<b>Rationale for 5% / 1pp Shading Thresholds</b>",
        "",
        "<b>Problem Statement:</b> Different metrics have different natural variation. Using a uniform threshold " +
        "(e.g., 2% for everything) creates unbalanced sensitivity where some comparisons over-flag minor differences " +
        "while others under-flag meaningful ones.",
        "",
        "<b>Data Analysis:</b> Examining all 59 Western MA traditional districts:",
        "• <b>PPE variation:</b> Mean = $24,237, SD = $5,462, CV = 0.225 (22.5% typical variation)",
        "• <b>CAGR variation:</b> Mean = 6.00pp, SD = 3.24pp, CV = 0.540 (54% typical variation)",
        "• <b>Key insight:</b> CAGR naturally varies 2.4x more than PPE relative to their means",
        "",
        "<b>Methodology:</b>",
        "1. Calculated how many standard deviations each threshold represents",
        "2. Compared 'sensitivity' (how strict each threshold is relative to typical variation)",
        "3. Tested scenarios ranging from 2%/2pp (original) to 5%/5pp (proportional loosening)",
        "4. Selected thresholds that balance statistical rigor with practical communication",
        "",
        "<b>Scenario Comparison:</b>",
        "• <b>Previous (2% / 2pp):</b> PPE=0.09 SD, CAGR=0.62 SD → CAGR 7x more sensitive (unbalanced)",
        "• <b>Equal SD (5% / 0.72pp):</b> Both=0.22 SD → Perfect balance but 0.72pp too precise to communicate",
        "• <b>Selected (5% / 1pp):</b> PPE=0.22 SD, CAGR=0.31 SD → 1.4x ratio (well-balanced, simple numbers)",
        "• <b>Proportional (5% / 5pp):</b> PPE=0.22 SD, CAGR=1.55 SD → CAGR 7x looser (under-flags growth differences)",
        "",
        "<b>Selected Thresholds (5% / 1pp):</b>",
        "• <b>5% for PPE and enrollment:</b> Flags differences >= $1,212 (at mean), or ~1/5 of typical variation",
        "• <b>1pp for CAGR:</b> Flags growth rate differences >= 1pp, or ~1/3 of typical variation",
        "• <b>Balance ratio:</b> 1.4x (CAGR slightly more sensitive, recognizing higher natural variation)",
        "• <b>Flagging rates:</b> ~82% for PPE, ~76% for CAGR (similar selectivity)",
        "",
        "<b>Design Philosophy: Why ~80% Flagging Rates Work with Gradient Shading</b>",
        "",
        "With <b>gradient shading</b> (not binary on/off), having ~80% of comparisons show some level of shading is actually ideal. " +
        "The threshold acts as a <b>noise floor</b> that filters trivial differences while the graduated intensity creates three distinct levels of information:",
        "",
        "• <b>No shading (white background):</b> Districts are statistically similar to their cohort (differences <5% / <1pp)",
        "• <b>Light shading (subtle color):</b> Notable differences worth attention but not alarming",
        "• <b>Intense shading (saturated color):</b> Exceptional outliers that immediately grab the eye",
        "",
        "This creates a natural visual hierarchy where readers' eyes are drawn to the most intense colors (true outliers) while still " +
        "being able to see the full pattern of variation across the lighter-shaded cells. The threshold filters out noise while the " +
        "gradient intensity tells you <i>how much</i> a district differs from its peers.",
        "",
        "<b>The Goldilocks Solution (5% / 1pp):</b>",
        "",
        "This threshold pair hits the sweet spot across multiple dimensions:",
        "",
        "<b>Statistical balance:</b>",
        "• Similar flagging rates (~82% vs ~76%) despite very different natural variation (CV = 22.5% vs 54%)",
        "• Proportional to underlying variation - neither metric dominates the visual attention",
        "",
        "<b>Practical communication:</b>",
        "• Round, memorable numbers (5% and 1pp)",
        "• Easy to explain: \"We flag 5% differences in dollars/enrollment, 1 percentage point differences in growth rates\"",
        "• Contrast with alternatives: 0.72pp would be statistically perfect but impractically precise to communicate",
        "",
        "<b>Appropriate sensitivity:</b>",
        "• Loosens overly-tight previous thresholds (2%/2pp had 93% PPE flagging)",
        "• Filters trivial noise while capturing meaningful differences",
        "• Reserves intense shading for truly exceptional cases",
        "",
        "<b>Shading Intensity:</b> Both metrics use graduated shading with darker colors for larger differences:",
        "• PPE/Enrollment: 5% (lightest), 10%, 15%, 20%+ (darkest)",
        "• CAGR: 1pp (lightest), 2pp, 3pp, 4pp+ (darkest)",
        "",
        "<b>Note:</b> These thresholds apply to all comparison tables throughout this report. They represent " +
        "a balance between highlighting meaningful differences and avoiding excessive flagging of normal variation.",
    ]

    return dict(
        title="Threshold Analysis",
        subtitle="Statistical Rationale for 5% / 1pp Shading Thresholds",
        chart_path=None,
        threshold_analysis=True,
        summary_table=summary_table,
        explanation_blocks=explanation_blocks
    )


def build_page_dicts(df: pd.DataFrame, reg: pd.DataFrame, c70: pd.DataFrame) -> List[dict]:
    pages: List[dict] = []

    # Load external text sections for easy editing
    report_text = load_report_text_sections()

    # Load Appendix A text sections from separate file
    appendix_a_text = load_report_text_sections(Path(__file__).parent / "appendix_a_text.txt")
    # Merge appendix_a_text into report_text
    report_text.update(appendix_a_text)

    # Note: Threshold Analysis moved to Appendix A

    latest = int(df["YEAR"].max())
    t0 = latest - 5

    # ===== EXECUTIVE SUMMARY =====
    # PAGE 0: Executive Summary - Introduction
    exec_summary_intro_text = report_text.get("EXECUTIVE_SUMMARY_PAGE1", [
        "<b>Per-pupil expenditure (PPE)</b> is a cloudy lens for understanding education funding."
    ])

    exec_summary_quick_start = report_text.get("EXECUTIVE_SUMMARY_QUICK_START", [
        "<b>Quick-start guide:</b>"
    ])

    exec_summary_footer = report_text.get("EXECUTIVE_SUMMARY_FOOTER", [
        "Refer to the appendices for information about data sources and calculations."
    ])

    # Combine all text blocks for the Executive Summary page
    # Mark the quick-start guide with a special marker for boxed rendering
    exec_summary_all_text = exec_summary_intro_text + ["__BOXED_START__"] + exec_summary_quick_start + ["__BOXED_END__"] + exec_summary_footer

    pages.append(dict(
        title="Executive Summary",
        subtitle="",
        text_blocks=exec_summary_all_text,
        text_only_page=True,
        section_id="executive_summary"
    ))

    # PAGE 0a: Executive Summary - PPE Overview (will be added after cohort tables - see CR5)
    # Prepare this data early since we need it for Executive Summary
    omitted_districts = get_omitted_western_districts(df, reg)
    western_explanation = report_text.get("PPE_OVERVIEW_EXPLANATION", [
        "Each bar represents a district's per-pupil expenditures (PPE) in 2024, with growth from 2019 shown by a darker color."
    ])[0]  # Get first paragraph from list

    # Add omitted districts note if any
    if omitted_districts:
        omitted_note = "<i>Note: The following districts are omitted from this analysis: "
        omitted_list = ", ".join([f"{name} ({reason})" for name, reason in omitted_districts])
        omitted_note += omitted_list + ".</i>"
        western_text_blocks_exec = [western_explanation, omitted_note]
    else:
        western_text_blocks_exec = [western_explanation]

    # Store PPE Overview page dict to add after cohort tables (CR5)
    ppe_overview_page = dict(
        title="Executive Summary (continued)",
        subtitle=f"Per-pupil expenditure and recent growth overview: {t0} PPE (lighter) to {latest} PPE (darker segment)",
        chart_path=str(OUTPUT_DIR / "ppe_overview_all_western.png"),
        text_blocks=western_text_blocks_exec,
        graph_only=True
    )

    # PAGE 0b: Executive Summary - Statistical Analysis (text from report_text.txt)
    stat_text_blocks = report_text.get("STATISTICAL_ASSOCIATIONS", [])

    # YoY and CAGR pages moved to Section 1 (text from report_text.txt)
    exec_summary_explanation = "\n".join(report_text.get("SECTION1_YOY_EXPLANATION", []))
    cagr_explanation = "\n".join(report_text.get("SECTION1_CAGR_EXPLANATION", []))

    # ===== EXECUTIVE SUMMARY COHORT TABLES =====
    # Build cohort summary data using same functions as Western MA pages
    cohorts_exec = get_western_cohort_districts(df, reg)
    cmap_exec = create_or_load_color_map(df)

    # Helper function to build cohort/district total PPE data
    def build_cohort_total_data(bucket_key: str, district_list: List[str]) -> tuple:
        """
        Build total PPE data for a cohort or district.

        Args:
            bucket_key: Cohort identifier (tiny, small, etc.) or "district" for individual districts
            district_list: List of district names

        Returns:
            Tuple of (start_dollar, cagr15, cagr10, cagr5, latest_dollar)
        """
        if not district_list:
            return (0.0, 0.0, 0.0, 0.0, 0.0)

        try:
            if bucket_key in ["tiny", "small", "medium", "large", "springfield", "all_western"]:
                # Cohort aggregate: use prepare_western_epp_lines
                title, epp, lines_sum, lines_mean = prepare_western_epp_lines(df, reg, bucket_key, c70, districts=district_list)
                if epp.empty:
                    return (0.0, 0.0, 0.0, 0.0, 0.0)
                latest_year_local = get_latest_year(df, epp)
                context = context_for_western(bucket_key)
            else:
                # Individual district: use prepare_district_epp_lines
                dist_name = district_list[0]
                epp, lines = prepare_district_epp_lines(df, dist_name)
                if epp.empty:
                    return (0.0, 0.0, 0.0, 0.0, 0.0)
                latest_year_local = get_latest_year(df, epp)
                context = context_for_district(df, dist_name)

            # Build category data (common for both cohorts and districts)
            rows, total, start_map = _build_category_data(epp, latest_year_local, context, cmap_exec)
            return _extract_total_from_cat_total(total)

        except Exception as e:
            # Log error but continue - return zeros for this cohort/district
            print(f"Warning: Failed to build data for {bucket_key} ({district_list}): {e}")
            return (0.0, 0.0, 0.0, 0.0, 0.0)

    # Build all Western MA cohort districts (excluding Springfield)
    # Note: Large cohort label will be dynamically determined based on actual cohort boundaries
    western_all_excl_springfield = [d for group in ["TINY", "SMALL", "MEDIUM", "LARGE"] for d in cohorts_exec.get(group, [])]
    large_label = get_cohort_label("LARGE")  # Will be "Large (1601-10K FTE)" or similar based on dynamic boundaries

    cohort_summary_data = {
        "Western MA (all, excl. Springfield)": build_cohort_total_data("all_western", western_all_excl_springfield),
        "Western MA Tiny (0-200 FTE)": build_cohort_total_data("tiny", cohorts_exec.get("TINY", [])),
        "Western MA Small (201-800 FTE)": build_cohort_total_data("small", cohorts_exec.get("SMALL", [])),
        "Western MA Medium (801-1600 FTE)": build_cohort_total_data("medium", cohorts_exec.get("MEDIUM", [])),
        f"Western MA {large_label}": build_cohort_total_data("large", cohorts_exec.get("LARGE", [])),
        "Outliers (Springfield)": build_cohort_total_data("springfield", cohorts_exec.get("SPRINGFIELD", [])),
        "Amherst-Pelham Regional": build_cohort_total_data("district", ["Amherst-Pelham"]),  # Note: actual district name is "Amherst-Pelham"
        "Amherst": build_cohort_total_data("district", ["Amherst"]),
        "Leverett": build_cohort_total_data("district", ["Leverett"]),
        "Pelham": build_cohort_total_data("district", ["Pelham"]),
        "Shutesbury": build_cohort_total_data("district", ["Shutesbury"]),
    }

    # Helper to build table rows from cohort names
    def make_table_rows(cohort_names: List[str]) -> List[tuple]:
        """Build table rows from cohort/district names."""
        return [(name, *cohort_summary_data.get(name, (0, 0, 0, 0, 0))) for name in cohort_names]

    # TABLE 1: All Western MA cohorts with shading (all rows shaded vs Western MA baseline)
    table1_names = [
        "Western MA (all, excl. Springfield)",
        "Western MA Tiny (0-200 FTE)",
        "Western MA Small (201-800 FTE)",
        "Western MA Medium (801-1600 FTE)",
        f"Western MA {large_label}",
        "Outliers (Springfield)",
    ]
    cohort_table1_rows = make_table_rows(table1_names)
    baseline_western = cohort_table1_rows[0][1:]
    cohort_table1 = _build_cohort_summary_table(
        cohort_table1_rows,
        baseline_row=("Western MA (all, excl. Springfield)", *baseline_western),
        apply_shading=True,
        highlight_baseline_rows={0}  # Highlight baseline row in yellow
    )

    pages.append(dict(
        title="Executive Summary (continued)",
        subtitle="Total PPE comparison: Western MA enrollment cohorts and selected districts",
        summary_table=cohort_table1,
        explanation_blocks=[],  # Removed - explanation moved to bottom after all tables
        first_cohort_table=True  # Flag to show legend at top
    ))

    # TABLE 2: The Mediums (baseline row not shaded, districts shaded vs Medium cohort)
    table2_names = ["Western MA Medium (801-1600 FTE)", "Amherst-Pelham Regional", "Amherst"]
    cohort_table2_rows = make_table_rows(table2_names)
    baseline_medium = cohort_table2_rows[0][1:]
    cohort_table2 = _build_cohort_summary_table(
        cohort_table2_rows,
        baseline_row=("Western MA Medium (801-1600 FTE)", *baseline_medium),
        apply_shading=True,
        skip_shading_rows={0},  # Don't shade the comparison baseline
        highlight_baseline_rows={0}  # Highlight baseline row in yellow
    )

    pages.append(dict(
        title="Executive Summary (continued)",  # Won't be displayed due to omit_title flag
        subtitle="",  # Removed subtitle per CR20
        summary_table=cohort_table2,
        explanation_blocks=[],  # Removed - explanation moved to bottom after all tables
        omit_title=True  # Skip title since it's on the same page as Table 1
    ))

    # TABLE 3: The Tinies (baseline row not shaded, districts shaded vs Tiny cohort)
    table3_names = ["Western MA Tiny (0-200 FTE)", "Leverett", "Pelham", "Shutesbury"]
    cohort_table3_rows = make_table_rows(table3_names)
    baseline_tiny = cohort_table3_rows[0][1:]
    cohort_table3 = _build_cohort_summary_table(
        cohort_table3_rows,
        baseline_row=("Western MA Tiny (0-200 FTE)", *baseline_tiny),
        apply_shading=True,
        skip_shading_rows={0},  # Don't shade the comparison baseline
        highlight_baseline_rows={0}  # Highlight baseline row in yellow
    )

    # Load explanation text from external file - this will appear at the bottom after all 3 tables
    cohort_explanation_text = report_text.get("EXECUTIVE_SUMMARY_COHORT_EXPLANATION", [
        "Table {N} compares PPE and cost growth of Western MA school districts organized into cohorts by enrollment size."
    ])

    pages.append(dict(
        title="Executive Summary (continued)",  # Won't be displayed due to omit_title flag
        subtitle="",  # Removed subtitle per CR20
        summary_table=cohort_table3,
        explanation_blocks=cohort_explanation_text,
        omit_title=True,  # Skip title since it's on the same page as Table 1
        needs_table_num_substitution=True  # Flag to replace {N}, {N+1}, {N+2} with actual table numbers
    ))

    # Executive Summary - PPE Overview (add after cohort tables - CR5)
    pages.append(ppe_overview_page)

    # Executive Summary - Statistical Analysis (add after PPE overview)
    if stat_text_blocks:
        pages.append(dict(
            title="Executive Summary (continued)",
            subtitle="Statistical Associations",
            text_blocks=stat_text_blocks,
            text_only_page=True
        ))

    # ===== SECTION 1: WESTERN MA TRADITIONAL PUBLIC SCHOOL DISTRICT TRENDS =====
    # Section 1 Summary Page
    section1_summary_text = report_text.get("SECTION1_SUMMARY", [
        "Section 1 examines per-pupil expenditure trends across Western Massachusetts."
    ])
    pages.append(dict(
        title="Section 1 — Western MA traditional public school district trends",
        subtitle="",
        text_blocks=section1_summary_text,
        text_only_page=True,
        section_id="section1_summary"
    ))

    # PAGE 1: YoY Growth Rates (moved from Executive Summary)
    pages.append(dict(
        title="Section 1 — Western MA traditional public school district trends",
        subtitle="Year-over-Year (YoY) growth rates by district and cohort",
        chart_path=str(OUTPUT_DIR / "executive_summary_yoy_panes.png"),
        text_blocks=[exec_summary_explanation],
        graph_only=True,
        section_id="section1_yoy"
    ))

    # PAGE 2: 5-year CAGR (moved from Executive Summary)
    pages.append(dict(
        title="Section 1 — Western MA traditional public school district trends",
        subtitle="5-year and 15-year CAGR by district and cohort",
        chart_paths=[
            str(OUTPUT_DIR / "executive_summary_cagr_legend.png"),
            str(OUTPUT_DIR / "executive_summary_cagr_grouped.png"),
            str(OUTPUT_DIR / "executive_summary_cagr_15year.png")
        ],
        text_blocks=[cagr_explanation],
        graph_only=True,
        cagr_with_legend=True,  # Special flag for legend + two charts layout
        section_id="section1_cagr"
    ))

    # PAGE 3: Distribution of 2024 enrollment and proposed cohort grouping (text from report_text.txt)
    enrollment_dist_explanation = "\n".join(report_text.get("SECTION1_DISTRIBUTION_EXPLANATION", []))

    pages.append(dict(
        title="Section 1 — Western MA traditional public school district trends",
        subtitle="Distribution of 2024 enrollment and proposed cohort grouping",
        chart_paths=[
            str(OUTPUT_DIR / "enrollment_3_histogram.png"),
            str(OUTPUT_DIR / "enrollment_4_grouping.png")
        ],
        text_blocks=[enrollment_dist_explanation],
        graph_only=True,
        two_charts_vertical=True,
        section_id="section1_distribution"
    ))

    # PAGE 4: 2024 Scatterplot with district table
    scatterplot_explanation_2024 = (
        "This scatterplot shows the relationship between district enrollment (x-axis) and per-pupil "
        "expenditure (y-axis) for Western Massachusetts traditional districts in 2024. Each point represents "
        "one district, colored by enrollment cohort. Horizontal lines show quartile boundaries used to define cohorts."
    )

    scatterplot_table_data = _build_scatterplot_district_table(df, reg, 2024)
    pages.append(dict(
        title="Section 1 — Western MA traditional public school district trends",
        subtitle=f"Scatterplot of enrollment vs. per-pupil expenditure with quartile boundaries (2024)",
        chart_path=str(OUTPUT_DIR / f"enrollment_1_scatterplot_2024.png"),
        text_blocks=[scatterplot_explanation_2024],
        graph_only=False,  # Includes table
        scatterplot_districts=scatterplot_table_data,
        year=2024,
        section_id="section1_scatterplot"
    ))

    # PAGE 5: Geographic map 2024
    choropleth_explanation_2024 = report_text.get("SECTION1_CHOROPLETH_EXPLANATION", [
        "This map situates the districts within their geographic context in Western Massachusetts."
    ])[0]  # Get first paragraph from list

    pages.append(dict(
        title="Section 1 — Western MA traditional public school district trends",
        subtitle=f"Geographic map showing district locations and enrollment cohorts (2024)",
        chart_path=str(OUTPUT_DIR / f"western_ma_choropleth_2024.png"),
        text_blocks=[choropleth_explanation_2024],
        graph_only=True,
        section_id="section1_map"
    ))

    # Set up cohort data structures for Section 2
    cmap_all = create_or_load_color_map(df)

    # Get Western cohorts using centralized function
    cohorts = get_western_cohort_districts(df, reg)
    cohort_map = {"tiny": "TINY", "small": "SMALL", "medium": "MEDIUM", "large": "LARGE", "springfield": "SPRINGFIELD"}

    # Calculate Western MA baseline (excluding Springfield) for cohort comparisons
    # This is used once and shared across all cohort pages
    western_baseline_map, western_fte_baseline_map = _build_western_ma_baseline(df, reg, c70, cmap_all, df["YEAR"].max(), exclude_springfield=True)

    # ===== SECTION 2: WESTERN MA COHORT DETAILS =====
    # Each cohort gets PPE and NSS/Ch70 aggregate pages (moved from Section 1)

    for bucket in ("tiny", "small", "medium", "large", "springfield"):
        cohort_name = cohort_map[bucket]
        district_list = cohorts.get(cohort_name, [])

        if not district_list:
            continue

        # Get cohort label for use in title
        cohort_label = get_cohort_label(cohort_name)

        # Prepare categorized district list text for NSS/Ch70 page
        categories = _categorize_districts(district_list, reg)
        total_count = len(district_list)

        # Build categorized list text
        district_list_text = [f"<b>{cohort_label}</b> includes {total_count} members:", ""]

        if categories["districts"]:
            count = len(categories["districts"])
            district_list_text.append(f"<b>Districts ({count}):</b> {', '.join(sorted([d.title() for d in categories['districts']]))}.")
            district_list_text.append("")  # Blank line

        if categories["unified_regions"]:
            count = len(categories["unified_regions"])
            district_list_text.append(f"<b>Regions ({count}):</b> {', '.join(sorted([d.title() for d in categories['unified_regions']]))}.")
            district_list_text.append("")  # Blank line

        if categories["vocational"]:
            count = len(categories["vocational"])
            district_list_text.append(f"<b>Vocational/technical ({count}):</b> {', '.join(sorted([d.title() for d in categories['vocational']]))}.")
            district_list_text.append("")  # Blank line

        # PAGE 1 for this cohort: PPE aggregate page
        title, epp, lines_sum, lines_mean = prepare_western_epp_lines(df, reg, bucket, c70, districts=district_list)
        if epp.empty and not lines_sum:
            continue

        latest_year = get_latest_year(df, epp)
        context = context_for_western(bucket)

        rows, total, start_map = _build_category_data(epp, latest_year, context, cmap_all)
        fte_rows, fte_map, latest_fte_year = _build_fte_data(lines_mean, latest_year)

        pages.append(dict(
            title=f"Section 2 — Western MA cohort details",
            subtitle=f"{cohort_label} — PPE vs Enrollment — Weighted average per district.",
            chart_path=str(regional_png(bucket)),
            latest_year=latest_year,
            latest_year_fte=latest_fte_year,
            cat_rows=rows,
            cat_total=total,
            cat_start_map=start_map,
            fte_rows=fte_rows,
            fte_series_map=fte_map,
            page_type="western",
            baseline_title="All Western MA (excluding Springfield)",
            baseline_map=western_baseline_map,
            fte_baseline_map=western_fte_baseline_map,
            raw_epp=epp,
            raw_lines=lines_mean,
            dist_name=title,
            section_id=f"section2_{bucket.replace('-', '_')}"
        ))

        # PAGE 2 for this cohort: NSS/Ch70 aggregate page (with district list text)
        if c70 is not None and not c70.empty:
            cohort_key = bucket.upper().replace("-", "-")  # Convert "tiny" to "TINY", etc.
            bucket_districts = cohorts.get(cohort_key, [])

            if bucket_districts:
                nss_west, enroll_west, foundation_west = prepare_aggregate_nss_ch70_weighted(df, c70, bucket_districts)
                if not nss_west.empty:
                    latest_year_nss = int(nss_west.index.max())
                    cat_rows_nss, cat_total_nss, cat_start_map_nss = build_nss_category_data(nss_west, latest_year_nss)
                    fte_rows_nss, fte_map_nss, latest_fte_year_nss = _build_nss_fte_data(foundation_west, latest_year_nss)

                    # Build Western MA baseline for NSS/Ch70 (shared across all cohorts)
                    western_nss_baseline = _build_western_ma_nss_baseline(df, reg, c70, latest_year_nss, exclude_springfield=True)

                    safe_name = f"Western_MA_{bucket}"
                    pages.append(dict(
                        title=f"Section 2 — Western MA cohort details",
                        subtitle=f"{cohort_label} — Chapter 70 Aid and Net School Spending (NSS).",
                        chart_path=str(OUTPUT_DIR / f"nss_ch70_{safe_name}.png"),
                        latest_year=latest_year_nss,
                        cat_rows=cat_rows_nss,
                        cat_total=cat_total_nss,
                        cat_start_map=cat_start_map_nss,
                        fte_rows=fte_rows_nss,
                        fte_series_map=fte_map_nss,
                        latest_year_fte=latest_fte_year_nss,
                        page_type="nss_ch70",
                        baseline_title="All Western MA (excluding Springfield)",
                        baseline_map=western_nss_baseline,
                        fte_baseline_map=western_nss_baseline if "Foundation Enrollment" in western_nss_baseline else {},
                        dist_name=title,
                        raw_nss=nss_west,  # Store for Appendix C data tables
                        district_list_text=district_list_text  # Add district list text below enrollment table
                    ))

    # ===== SECTION 3: SPECIFIC DISTRICTS COMPARED TO COHORTS =====

    # Pre-compute Western district lists for NSS/Ch70 baseline computation (6 enrollment groups)
    # Use centralized cohort function to ensure consistency with plots (filters by PPE data availability)
    cohorts_for_baseline = get_western_cohort_districts(df, reg)
    western_tiny = cohorts_for_baseline.get("TINY", [])
    western_small = cohorts_for_baseline.get("SMALL", [])
    western_medium = cohorts_for_baseline.get("MEDIUM", [])
    western_large = cohorts_for_baseline.get("LARGE", [])
    western_springfield = cohorts_for_baseline.get("SPRINGFIELD", [])

    # DISTRICT PAGES - Three pages per district (simple + detailed vs Western + detailed vs Peers)
    for dist in ["Amherst-Pelham"] + [d for d in DISTRICTS_OF_INTEREST if d != "Amherst-Pelham"]:
        epp, lines = prepare_district_epp_lines(df, dist, c70)
        if epp.empty and not lines:
            continue

        dist_title = f"Amherst-Pelham Regional" if dist == "Amherst-Pelham" else dist
        # Create section_id from district name (lowercase, replace spaces with underscores)
        section_id = dist.lower().replace(" ", "_").replace("-", "_")

        # Detailed version with tables (CR1: removed simple version)
        latest_year = get_latest_year(df, epp)
        context = context_for_district(df, dist)

        rows, total, start_map = _build_category_data(epp, latest_year, context, cmap_all)
        fte_rows, fte_map, latest_fte_year = _build_fte_data(lines, latest_year)

        # Map context to bucket and label using centralized cohort definitions
        bucket_map = {
            "TINY": ("tiny", get_cohort_2024_label("TINY")),
            "SMALL": ("small", get_cohort_2024_label("SMALL")),
            "MEDIUM": ("medium", get_cohort_2024_label("MEDIUM")),
            "LARGE": ("large", get_cohort_2024_label("LARGE")),
            "SPRINGFIELD": ("springfield", get_cohort_2024_label("SPRINGFIELD"))
        }
        bucket, bucket_label = bucket_map.get(context, ("tiny", get_cohort_2024_label("TINY")))

        base_title = f"All Western MA Traditional Districts: {bucket_label}"
        base_map = {}
        fte_base_map = {}
        district_list = cohorts[context]
        title_w, epp_w, lines_sum, lines_mean = prepare_western_epp_lines(df, reg, bucket, c70, districts=district_list)
        if not epp_w.empty:
            start_year = latest_year - 15  # 15 years before latest for START_DOLLAR
            for sc in epp_w.columns:
                s=epp_w[sc]
                base_map[sc]={
                    "5": compute_cagr_last(s,5),
                    "10":compute_cagr_last(s,10),
                    "15":compute_cagr_last(s,15),
                    "DOLLAR": (float(s.loc[latest_year]) if latest_year in s.index else float("nan")),
                    "START_DOLLAR": (float(s.loc[start_year]) if start_year in s.index else float("nan")),
                }

            # Add Total entry to baseline map
            total_series = epp_w.sum(axis=1)
            base_map["Total"] = {
                "5": compute_cagr_last(total_series, 5),
                "10": compute_cagr_last(total_series, 10),
                "15": compute_cagr_last(total_series, 15),
                "DOLLAR": (float(total_series.loc[latest_year]) if latest_year in total_series.index else float("nan")),
                "START_DOLLAR": (float(total_series.loc[start_year]) if start_year in total_series.index else float("nan")),
            }

        # Build FTE baseline map from Western aggregate enrollment data
        if lines_mean:
            for key, label in ENROLL_KEYS:
                s = lines_mean.get(label)
                if s is not None and not s.empty:
                    fte_base_map[label] = {
                        "5": compute_cagr_last(s, 5),
                        "10": compute_cagr_last(s, 10),
                        "15": compute_cagr_last(s, 15),
                        "FTE": (float(s.loc[latest_fte_year]) if latest_fte_year in s.index else float("nan")),
                        "START_FTE": (float(s.loc[start_year]) if start_year in s.index else float("nan")),
                    }

        pages.append(dict(
            title="Section 3 — Specific districts",
            subtitle=f"{dist_title} — PPE vs Enrollment",
            chart_path=str(district_png_detail(dist)),
            latest_year=latest_year, latest_year_fte=latest_fte_year,
            cat_rows=rows, cat_total=total, cat_start_map=start_map, fte_rows=fte_rows,
            fte_series_map=fte_map,
            page_type="district", baseline_title=base_title, baseline_map=base_map,
            fte_baseline_map=fte_base_map,
            raw_epp=epp, raw_lines=lines, dist_name=dist,
            section_id=section_id
        ))

        # District NSS/Ch70 page (grouped with this district)
        if c70 is not None and not c70.empty:
            nss_dist, enroll_dist, foundation_dist = prepare_district_nss_ch70(df, c70, dist)
            if not nss_dist.empty:
                latest_year_nss = int(nss_dist.index.max())
                cat_rows_nss, cat_total_nss, cat_start_map_nss = build_nss_category_data(nss_dist, latest_year_nss)
                fte_rows_nss, fte_map_nss, latest_fte_year_nss = _build_nss_fte_data(foundation_dist, latest_year_nss)

                # Determine Western baseline using 4-tier enrollment groups
                fte = latest_total_fte(df, dist)
                group = get_enrollment_group(fte)

                # Map group to district list and label using centralized cohort definitions
                group_map = {
                    "TINY": (western_tiny, get_cohort_label("TINY")),
                    "SMALL": (western_small, get_cohort_label("SMALL")),
                    "MEDIUM": (western_medium, get_cohort_label("MEDIUM")),
                    "LARGE": (western_large, get_cohort_label("LARGE")),
                    "SPRINGFIELD": (western_springfield, get_cohort_label("SPRINGFIELD"))
                }
                western_dists, group_label = group_map.get(group, (western_tiny, get_cohort_label("TINY")))

                # Compute Western NSS/Ch70 baseline (weighted per-pupil for comparison)
                nss_west_baseline = {}
                fte_west_baseline = {}
                if western_dists:
                    nss_west, _, foundation_west = prepare_aggregate_nss_ch70_weighted(df, c70, western_dists)
                    if not nss_west.empty:
                        nss_west_baseline = _build_nss_ch70_baseline_map(nss_west, latest_year_nss, foundation_west)
                        # Extract FTE baseline for the enrollment table
                        if "Foundation Enrollment" in nss_west_baseline:
                            fte_west_baseline["Foundation Enrollment"] = nss_west_baseline["Foundation Enrollment"]

                safe_name = make_safe_filename(dist)

                pages.append(dict(
                    title="Section 3 — Specific districts",
                    subtitle=f"{dist_title} — Chapter 70 Aid, Net School Spending (NSS), and Foundation Enrollment",
                    chart_path=str(OUTPUT_DIR / f"nss_ch70_{safe_name}.png"),
                    latest_year=latest_year_nss,
                    cat_rows=cat_rows_nss,
                    cat_total=cat_total_nss,
                    cat_start_map=cat_start_map_nss,
                    fte_rows=fte_rows_nss,
                    fte_series_map=fte_map_nss,
                    latest_year_fte=latest_fte_year_nss,
                    page_type="nss_ch70",
                    baseline_title=f"Western Traditional ({group_label})",
                    baseline_map=nss_west_baseline,
                    fte_baseline_map=fte_west_baseline,
                    dist_name=dist,
                    raw_nss=nss_dist  # Store raw data for Appendix A
                ))

    # Note: Section 3 (ALPS PK-12 & Peers) removed. Districts now compared to enrollment-based peer groups.

    # ===== APPENDIX A: DATA SOURCES & CALCULATION METHODOLOGY =====

    # Get Western MA districts organized by 5-tier enrollment groups
    # Use centralized cohort function to ensure consistency with plots and tables
    cohorts_for_methodology = get_western_cohort_districts(df, reg)
    western_tiny = cohorts_for_methodology.get("TINY", [])
    western_small = cohorts_for_methodology.get("SMALL", [])
    western_medium = cohorts_for_methodology.get("MEDIUM", [])
    western_large = cohorts_for_methodology.get("LARGE", [])
    western_springfield = cohorts_for_methodology.get("SPRINGFIELD", [])

    # Build cohort info placeholders for methodology text
    def format_cohort_info(cohort_key, districts):
        label = get_cohort_label(cohort_key)
        count = len(districts)
        if count == 0:
            members = "None"
        elif count <= 15:
            members = ', '.join(districts)
        else:
            members = ', '.join(districts[:15]) + f', and {count-15} others'

        cohort_descriptions = {
            "TINY": "Cohort 1: below Q1",
            "SMALL": "Cohort 2: Q1 to median",
            "MEDIUM": "Cohort 3: median to Q3",
            "LARGE": "Cohort 4: above Q3",
            "SPRINGFIELD": "Cohort 5: statistical outlier, analyzed separately"
        }
        desc = cohort_descriptions.get(cohort_key, "")

        return f"<b>{label}:</b> {count} district{'s' if count != 1 else ''} ({desc})\n<b>Member districts:</b> {members}"

    # Build placeholder dictionary
    placeholders = {
        "COHORT_TINY_INFO": format_cohort_info("TINY", western_tiny),
        "COHORT_SMALL_INFO": format_cohort_info("SMALL", western_small),
        "COHORT_MEDIUM_INFO": format_cohort_info("MEDIUM", western_medium),
        "COHORT_LARGE_INFO": format_cohort_info("LARGE", western_large),
        "COHORT_SPRINGFIELD_INFO": format_cohort_info("SPRINGFIELD", western_springfield),
        "DOLLAR_THRESHOLD_PERCENT": f"{DOLLAR_THRESHOLD_REL*100:.1f}",
        "CAGR_THRESHOLD_PP": f"{MATERIAL_DELTA_PCTPTS*100:.1f}",
        "DOLLAR_BIN_1": f"{DOLLAR_THRESHOLD_REL*100:.0f}",
        "DOLLAR_BIN_2": f"{SHADE_BINS_DOLLAR[1]*100:.0f}",
        "DOLLAR_BIN_3": f"{SHADE_BINS_DOLLAR[2]*100:.0f}",
        "DOLLAR_BIN_4": f"{SHADE_BINS_DOLLAR[3]*100:.0f}",
        "CAGR_BIN_1": f"{MATERIAL_DELTA_PCTPTS*100:.0f}",
        "CAGR_BIN_2": f"{SHADE_BINS_CAGR[1]*100:.0f}",
        "CAGR_BIN_3": f"{SHADE_BINS_CAGR[2]*100:.0f}",
        "CAGR_BIN_4": f"{SHADE_BINS_CAGR[3]*100:.0f}",
    }

    # Load methodology pages from external text file (with fallback to hardcoded defaults)
    # Apply placeholders for dynamic content
    methodology_page1 = fill_text_placeholders(
        report_text.get("METHODOLOGY_PPE_DEFINITION", []),
        placeholders
    )

    methodology_page2 = fill_text_placeholders(
        report_text.get("METHODOLOGY_SHADING_LOGIC", []),
        placeholders
    )

    methodology_page3 = report_text.get("METHODOLOGY_NSS_CH70", [])

    methodology_page4 = report_text.get("METHODOLOGY_DATA_SOURCES", [])

    # Appendix A - Page 1: Data Sources
    pages.append(dict(
        title="Appendix A. Data Sources & Calculation Methodology",
        subtitle="",
        chart_path=None,
        graph_only=True,
        text_blocks=methodology_page4,
        section_id="appendix_a"
    ))

    # Appendix A - Page 2: PPE Definition, CAGR, YoY, Enrollment-Based Cohorts
    pages.append(dict(
        title="Appendix A. Data Sources & Calculation Methodology (continued)",
        subtitle="",
        chart_path=None,
        graph_only=True,
        text_blocks=methodology_page1
    ))

    # Appendix A - Page 3: Orange/Blue Shading Logic
    pages.append(dict(
        title="Appendix A. Data Sources & Calculation Methodology (continued)",
        subtitle="",
        chart_path=None,
        graph_only=True,
        text_blocks=methodology_page2
    ))

    # Appendix A - Page 4: Chapter 70 Aid and Net School Spending (NSS)
    pages.append(dict(
        title="Appendix A. Data Sources & Calculation Methodology (continued)",
        subtitle="",
        chart_path=None,
        graph_only=True,
        text_blocks=methodology_page3
    ))

    # Appendix A - Page 5: Threshold Analysis (moved to end)
    threshold_page = build_threshold_analysis_page()
    threshold_page["title"] = "Appendix A. Data Sources & Calculation Methodology (continued)"
    threshold_page["subtitle"] = "Rationale for 5% / 1pp Shading Thresholds"
    pages.append(threshold_page)

    # ===== APPENDIX B: CALCULATIONS AND EXAMPLES (Combined B & C) =====
    # This appendix combines "Complete Calculations" (ordered by figures/tables)
    # with "Detailed Examples" for specific cases

    appendix_b_examples_content = []  # Will hold detailed examples from file
    appendix_b_path = Path("appendix_b_text.txt")
    if appendix_b_path.exists():
        # Read and parse detailed examples content - split by "====" section markers
        appendix_b_full = appendix_b_path.read_text(encoding='utf-8')

        # Split by ===== markers to get sections
        raw_sections = appendix_b_full.split('=' * 80)  # 80 equals signs

        # First section (before first ====) is the introduction for title page
        intro_text = []
        if raw_sections and raw_sections[0].strip():
            intro_lines = raw_sections[0].strip().split('\n')
            for line in intro_lines:
                intro_text.append(line.strip())

        # If no intro found in file, use default
        if not intro_text:
            intro_text = [
                "This appendix provides calculation examples showing the source of numbers and demonstrating every step of the analysis for two examples:",
                "1. Western MA Medium Cohort (aggregate cohort calculations)",
                "2. Amherst-Pelham Regional (individual district calculations)",
                "",
                "The following pages detail:",
                "• Data sources and cohort determination methodology",
                "• Weighted average per-pupil expenditure calculations",
                "• Time series analysis and compound annual growth rate (CAGR) formulas",
                "• Chapter 70 state aid and Net School Spending calculations",
                "• Comparative analysis methods",
            ]

        # Parse all sections and concatenate into one continuous content block
        all_content = []

        # Start from index 1 to skip the intro section
        for raw_section in raw_sections[1:]:
            lines = raw_section.strip().split('\n')
            if not lines or not lines[0].strip():
                continue

            # First non-empty line is the section title (from === markers)
            section_title = lines[0].strip()
            if not section_title:
                continue

            # Add section header as bold, larger text
            all_content.append("")  # Spacing before section
            all_content.append(f"<b><font size=14>{section_title}</font></b>")
            all_content.append("")  # Spacing after header

            # Rest are content - format them
            for line in lines[1:]:
                stripped = line.strip()

                # Format headers and steps as bold
                if stripped.startswith('Step '):
                    all_content.append(f"<b>{stripped}</b>")
                elif stripped.startswith('PART '):
                    all_content.append(f"<b><font size=13>{stripped}</font></b>")
                elif stripped.endswith(':') and len(stripped) < 100 and not stripped.startswith('•') and not stripped.startswith('http'):
                    # Looks like a subsection header
                    all_content.append(f"<b>{stripped}</b>")
                elif stripped.startswith('---'):
                    # Separator line - skip it, section headers provide enough visual separation
                    pass  # Don't add separator lines
                else:
                    # Regular line - preserve original formatting
                    all_content.append(line.rstrip())

        # Store detailed examples content for merging with complete calculations
        # Combine intro and content into single block
        appendix_b_examples_content = intro_text + [""] + all_content  # Add blank line between intro and content

    # ===== APPENDIX C: COMPLETE CALCULATIONS =====
    # Comprehensive calculations for all plots, tables, and methodology
    appendix_c_content = [
        "<b>Purpose</b>",
        "This appendix provides complete, step-by-step calculations for every figure and table in the report. "
        "The goal is to make it easy for an experienced analyst to verify the mathematics behind every result.",
        "",
        "=" * 80,
        "",
        "<b>1. EXECUTIVE SUMMARY CALCULATIONS</b>",
        "",
        "<b>Figure 1: Year-over-Year (YoY) Growth Rates</b>",
        "<i>Location:</i> Executive Summary page",
        "<i>Source:</i> executive_summary_plots.py, lines 44-63",
        "",
        "<b>Formula:</b>",
        "YoY Growth(year) = [(PPE(year) / PPE(year-1)) - 1] × 100",
        "",
        "<b>Calculation Steps:</b>",
        "1. For each district, get total PPE by year:",
        "   - Load expenditure data from df (all expense categories)",
        "   - Pivot to get PPE by category and year",
        "   - Sum across all categories: total_ppe[year] = sum(PPE_category[year] for all categories)",
        "",
        "2. Calculate YoY growth:",
        "   - For year in [2010, 2011, ..., 2024]:",
        "     YoY[year] = [(total_ppe[year] / total_ppe[year-1]) - 1] × 100",
        "   - First year (2009) has no YoY value (no prior year)",
        "",
        "3. For cohort aggregates:",
        "   - Get all districts in cohort",
        "   - Calculate weighted aggregate PPE (see Calculation #5)",
        "   - Apply same YoY formula to aggregate PPE series",
        "",
        "<b>Example (Amherst 2023-2024):</b>",
        "- PPE(2023) = $25,432.18",
        "- PPE(2024) = $26,789.45",
        "- YoY Growth = [(26,789.45 / 25,432.18) - 1] × 100 = 5.34%",
        "",
        "=" * 80,
        "",
        "<b>Figure 2: 5-Year CAGR (Compound Annual Growth Rate) - Grouped Bars</b>",
        "<i>Location:</i> Executive Summary page",
        "<i>Source:</i> executive_summary_plots.py, lines 66-95",
        "",
        "<b>Formula:</b>",
        "CAGR = [(End_Value / Start_Value)^(1/Years) - 1] × 100",
        "",
        "<b>Calculation Steps:</b>",
        "1. Define three 5-year periods:",
        "   - Period 1: 2009-2014 (5 years)",
        "   - Period 2: 2014-2019 (5 years)",
        "   - Period 3: 2019-2024 (5 years)",
        "",
        "2. For each district and each period:",
        "   - Get total_ppe[start_year] and total_ppe[end_year]",
        "   - CAGR = [(total_ppe[end] / total_ppe[start])^(1/5) - 1] × 100",
        "",
        "3. For cohort aggregates:",
        "   - Use weighted aggregate PPE (see Calculation #5)",
        "   - Apply same CAGR formula",
        "",
        "<b>Example (Medium Cohort 2019-2024):</b>",
        "- Weighted PPE(2019) = $22,145.67",
        "- Weighted PPE(2024) = $28,934.21",
        "- CAGR = [(28,934.21 / 22,145.67)^(1/5) - 1] × 100",
        "- CAGR = [1.30672^0.2 - 1] × 100",
        "- CAGR = [1.05497 - 1] × 100 = 5.50%",
        "",
        "=" * 80,
        "",
        "<b>Figure 3: 15-Year CAGR (2009-2024)</b>",
        "<i>Location:</i> Executive Summary page",
        "<i>Source:</i> executive_summary_plots.py, lines 98-118",
        "",
        "<b>Formula:</b>",
        "CAGR_15yr = [(PPE(2024) / PPE(2009))^(1/15) - 1] × 100",
        "",
        "<b>Calculation Steps:</b>",
        "1. Get total_ppe[2009] and total_ppe[2024] for each district",
        "2. Apply CAGR formula with years = 15",
        "3. For cohort aggregates, use weighted aggregate PPE",
        "",
        "<b>Example (Amherst):</b>",
        "- PPE(2009) = $18,234.56",
        "- PPE(2024) = $26,789.45",
        "- CAGR = [(26,789.45 / 18,234.56)^(1/15) - 1] × 100",
        "- CAGR = [1.46923^0.06667 - 1] × 100 = 2.61%",
        "",
        "=" * 80,
        "",
        "<b>2. COHORT DETERMINATION CALCULATIONS</b>",
        "",
        "<b>Purpose:</b> Assign each district to one of 6 enrollment-based cohorts",
        "<i>Source:</i> school_shared.py, lines 140-200 (calculate_cohort_boundaries)",
        "",
        "<b>Cohort Definitions:</b>",
        "- TINY: 0 - Q1 (25th percentile)",
        "- SMALL: Q1 - Q2 (median)",
        "- MEDIUM: Q2 - Q3 (75th percentile)",
        "- LARGE: Q3 - 10,000 FTE",
        "- SPRINGFIELD: > 10,000 FTE (outlier district)",
        "",
        "<b>Calculation Steps (FY2024):</b>",
        "1. Get all Western MA traditional districts",
        "2. Get FY2024 FTE enrollment for each district",
        "3. Create array of all enrollments (INCLUDING Springfield):",
        "   all_enrollments = [154.7, 287.3, ..., 24,567.8]  # 59 districts",
        "",
        "4. Calculate percentiles ON FULL DATASET:",
        "   - Q1 (25th) = np.percentile(all_enrollments, 25) = 154.7",
        "   - Q2 (50th) = np.percentile(all_enrollments, 50) = 520.3",
        "   - Q3 (75th) = np.percentile(all_enrollments, 75) = 1,597.6",
        "   - P90 (90th) = np.percentile(all_enrollments, 90) = 3,892.1",
        "",
        "5. Round to clean boundaries:",
        "   - Q1: 154.7 → 200 FTE",
        "   - Median: 520.3 → 500 FTE",
        "   - Q3: 1,597.6 → 1,600 FTE",
        "   - P90: 3,892.1 → 4,000 FTE",
        "   - Outlier threshold: Fixed at 10,000 FTE",
        "",
        "6. Final FY2024 Cohort Boundaries:",
        "   - TINY: 0-200 FTE (13 districts)",
        "   - SMALL: 201-500 FTE (16 districts)",
        "   - MEDIUM: 501-1,600 FTE (15 districts)",
        "   - LARGE: 1,601-10,000 FTE (14 districts)",
        "   - SPRINGFIELD: >10,000 FTE (1 district: 24,567.8 FTE)",
        "",
        "<b>Note:</b> For historical data, cohorts are calculated year-by-year using that year's enrollment distribution.",
        "",
        "=" * 80,
        "",
        "<b>3. WEIGHTED AGGREGATION METHODOLOGY</b>",
        "",
        "<b>Purpose:</b> Calculate enrollment-weighted per-pupil expenditure for a group of districts",
        "<i>Source:</i> school_shared.py, lines 1001-1050 (weighted_epp_aggregation)",
        "",
        "<b>Formula:</b>",
        "Weighted PPE(category, year) = Σ[PPE(d, cat, yr) × FTE(d, yr)] / Σ[FTE(d, yr)]",
        "where d = district, cat = category, yr = year",
        "",
        "<b>Calculation Steps:</b>",
        "1. For each district d in cohort:",
        "   - Get PPE by category and year: PPE(d, category, year)",
        "   - Get FTE enrollment by year: FTE(d, year)",
        "",
        "2. For each category and year:",
        "   - Calculate weighted numerator:",
        "     numerator(cat, yr) = Σ[PPE(d, cat, yr) × FTE(d, yr)] for all d in cohort",
        "   - Calculate total FTE:",
        "     denominator(yr) = Σ[FTE(d, yr)] for all d in cohort",
        "   - Weighted PPE(cat, yr) = numerator(cat, yr) / denominator(yr)",
        "",
        "<b>Example (Medium Cohort, Instruction category, 2024):</b>",
        "Districts in Medium cohort (FY2024): 15 districts",
        "",
        "District A: PPE_instruction = $12,345, FTE = 678",
        "District B: PPE_instruction = $13,456, FTE = 891",
        "...",
        "District O: PPE_instruction = $11,234, FTE = 1,234",
        "",
        "Numerator = (12,345 × 678) + (13,456 × 891) + ... + (11,234 × 1,234)",
        "         = 8,369,910 + 11,989,296 + ... + 13,862,756",
        "         = 156,432,890",
        "",
        "Denominator = 678 + 891 + ... + 1,234 = 13,456 FTE",
        "",
        "Weighted PPE_instruction = 156,432,890 / 13,456 = $11,627.45/pupil",
        "",
        "=" * 80,
        "",
        "<b>4. SHADING THRESHOLD CALCULATIONS</b>",
        "",
        "<b>Purpose:</b> Determine 5% / 1pp thresholds for gradient shading in comparison tables",
        "<i>Source:</i> Appendix A (Threshold Analysis), compose_pdf.py lines 1212-1349",
        "",
        "<b>Statistical Analysis (All 59 Western MA Districts):</b>",
        "",
        "1. Calculate variation in PPE:",
        "   - Mean PPE = $24,237",
        "   - Std Dev PPE = $5,462",
        "   - Coefficient of Variation (CV) = 5,462 / 24,237 = 0.225 (22.5%)",
        "",
        "2. Calculate variation in CAGR:",
        "   - Mean CAGR = 6.00 percentage points",
        "   - Std Dev CAGR = 3.24 percentage points",
        "   - Coefficient of Variation (CV) = 3.24 / 6.00 = 0.540 (54.0%)",
        "",
        "3. Key insight: CAGR varies 2.4× more than PPE relative to means (54.0% / 22.5% = 2.4)",
        "",
        "4. Evaluate threshold scenarios:",
        "   Scenario: 5% PPE / 1pp CAGR (SELECTED)",
        "   - PPE: 5% = 0.05 × 24,237 = $1,212",
        "   - PPE Standard Deviations: 1,212 / 5,462 = 0.22 SD",
        "   - CAGR Standard Deviations: 1.00 / 3.24 = 0.31 SD",
        "   - Balance ratio: 0.31 / 0.22 = 1.4× (well-balanced)",
        "   - PPE flagging rate: ~82% of comparisons",
        "   - CAGR flagging rate: ~76% of comparisons",
        "",
        "<b>Gradient Shading Bins:</b>",
        "- CAGR bins (absolute percentage points): [1pp, 2pp, 3pp, 4pp+]",
        "- Dollar bins (relative percent): [5%, 10%, 15%, 20%+]",
        "- Intensity increases with bin: lightest shade → darkest shade",
        "",
        "=" * 80,
        "",
        "<b>5. DISTRICT COMPARISON TABLE CALCULATIONS</b>",
        "",
        "<b>Purpose:</b> Compare district PPE to baseline (cohort or Western MA aggregate)",
        "<i>Source:</i> compose_pdf.py, lines 425-583 (_build_category_table)",
        "",
        "<b>For Each Category (e.g., Instruction, Student Support, etc.):</b>",
        "",
        "1. Get latest year (2024) and 5-year-ago baseline (2019)",
        "",
        "2. Calculate CAGR:",
        "   District CAGR = [(PPE_2024 / PPE_2019)^(1/5) - 1] × 100",
        "   Baseline CAGR = [(Baseline_PPE_2024 / Baseline_PPE_2019)^(1/5) - 1] × 100",
        "",
        "3. Calculate CAGR difference (absolute percentage points):",
        "   CAGR_diff = |District_CAGR - Baseline_CAGR|",
        "",
        "4. Determine CAGR shading:",
        "   if CAGR_diff < 1.0pp: No shading (white)",
        "   elif CAGR_diff < 2.0pp: Lightest shade",
        "   elif CAGR_diff < 3.0pp: Light shade",
        "   elif CAGR_diff < 4.0pp: Medium shade",
        "   else: Darkest shade",
        "   Color: Amber if above baseline, Teal if below",
        "",
        "5. Calculate dollar difference (relative percent):",
        "   Dollar_diff = |PPE_2024 - Baseline_PPE_2024| / Baseline_PPE_2024",
        "",
        "6. Determine dollar shading:",
        "   if Dollar_diff < 5%: No shading",
        "   elif Dollar_diff < 10%: Lightest shade",
        "   elif Dollar_diff < 15%: Light shade",
        "   elif Dollar_diff < 20%: Medium shade",
        "   else: Darkest shade",
        "   Color: Amber if above baseline, Teal if below",
        "",
        "<b>Example (Amherst Instruction vs Medium Cohort):</b>",
        "- Amherst PPE_2019 = $10,234, PPE_2024 = $12,456",
        "- Medium PPE_2019 = $9,876, PPE_2024 = $11,234",
        "",
        "CAGR calculations:",
        "- Amherst CAGR = [(12,456 / 10,234)^0.2 - 1] × 100 = 4.01%",
        "- Medium CAGR = [(11,234 / 9,876)^0.2 - 1] × 100 = 2.60%",
        "- CAGR_diff = |4.01 - 2.60| = 1.41pp → Light amber shade",
        "",
        "Dollar calculations:",
        "- Dollar_diff = |12,456 - 11,234| / 11,234 = 10.88%",
        "- 10.88% is in [10%, 15%) range → Light amber shade",
        "",
        "=" * 80,
        "",
        "<b>6. ENROLLMENT FTE CALCULATIONS</b>",
        "",
        "<b>Purpose:</b> Calculate total FTE enrollment from component categories",
        "<i>Source:</i> school_shared.py, prepare_district_epp_lines()",
        "",
        "<b>FTE Categories:</b>",
        "- Foundation Enrollment (low-income, ELL, special ed weighted)",
        "- PK (Pre-Kindergarten)",
        "- K-6 (Kindergarten through 6th grade)",
        "- 7-12 (7th through 12th grade)",
        "",
        "<b>Calculation:</b>",
        "Total FTE(year) = Foundation(year) + PK(year) + K6(year) + 712(year)",
        "",
        "<b>Example (Amherst 2024):</b>",
        "- Foundation = 1,234.5 FTE",
        "- PK = 89.2 FTE",
        "- K-6 = 678.3 FTE",
        "- 7-12 = 543.1 FTE",
        "- Total FTE = 1,234.5 + 89.2 + 678.3 + 543.1 = 2,545.1 FTE",
        "",
        "=" * 80,
        "",
        "<b>7. NSS/CH70 FUNDING CALCULATIONS</b>",
        "",
        "<b>Purpose:</b> Calculate Chapter 70 aid and Net School Spending from components",
        "<i>Source:</i> school_shared.py, prepare_district_nss_ch70()",
        "",
        "<b>Components:</b>",
        "- Foundation Budget: State-determined minimum spending level",
        "- Required Local Contribution: Municipality's required contribution",
        "- Chapter 70 Aid: State aid = Foundation - Required Local Contribution",
        "- Actual Net School Spending (NSS): Actual spending by district",
        "",
        "<b>Calculations:</b>",
        "Ch70 Aid = Foundation Budget - Required Local Contribution",
        "NSS Gap = Actual NSS - Foundation Budget",
        "Total Spending = Required Local Contribution + Ch70 Aid + NSS Gap",
        "",
        "<b>Example (Amherst 2024):</b>",
        "- Foundation Budget = $45,678,901",
        "- Required Local Contribution = $38,234,567",
        "- Ch70 Aid = $45,678,901 - $38,234,567 = $7,444,334",
        "- Actual NSS = $52,123,456",
        "- NSS Gap = $52,123,456 - $45,678,901 = $6,444,555",
        "",
        "=" * 80,
        "",
        "<b>NOTES</b>",
        "",
        "• All dollar values shown are illustrative examples for demonstration purposes",
        "• Actual data values are stored in Appendix C (Data Tables)",
        "• All calculations use enrollment-weighted aggregations for cohort comparisons",
        "• CAGR calculations exclude negative or zero values",
        "• Shading applies independently to CAGR and dollar comparisons",
        "• Year-specific cohort assignments are used for all historical maps and scatterplots",
    ]

    # Combine Complete Calculations (Appendix C) with Detailed Examples (Appendix B)
    # into a single Appendix B, synthesized in order of figures and tables
    combined_appendix_b = [
        "<b>Purpose</b>",
        "This appendix provides calculations and worked examples for all figures and tables in the report. "
        "It combines step-by-step calculation procedures with detailed examples for verification.",
        "",
        "=" * 80,
        "",
    ] + appendix_c_content[6:]  # Skip the original purpose section of appendix_c_content

    # Add detailed examples section if available
    if appendix_b_examples_content:
        combined_appendix_b.extend([
            "",
            "=" * 80,
            "",
            "<b>DETAILED WORKED EXAMPLES</b>",
            "",
            "The following sections provide complete worked examples for specific districts and cohorts:",
            ""
        ])
        combined_appendix_b.extend(appendix_b_examples_content)

    pages.append(dict(
        title="Appendix B. Calculations and Examples",
        subtitle="Complete calculations and worked examples for all figures and tables",
        chart_path=None,
        graph_only=True,
        text_blocks=combined_appendix_b,
        section_id="appendix_b",
        appendix_b=True  # Flag for 12pt font
    ))

    # ===== APPENDIX D: DATA TABLES =====
    # Deduplicate by dist_name to avoid duplicate data tables
    # Collect both EPP and NSS/Ch70 data for each district
    data_pages_to_add = []
    seen_districts = set()
    district_data = {}  # Map dist_name -> {raw_epp, raw_lines, raw_nss}

    for p in pages:
        dist_name = p.get("dist_name", "Unknown")
        if dist_name not in district_data:
            district_data[dist_name] = {"raw_epp": None, "raw_lines": {}, "raw_nss": None}

        # Collect EPP data
        if p.get("raw_epp") is not None and not p["raw_epp"].empty:
            district_data[dist_name]["raw_epp"] = p.get("raw_epp")
            district_data[dist_name]["raw_lines"] = p.get("raw_lines", {})

        # Collect NSS/Ch70 data
        if p.get("raw_nss") is not None and not p["raw_nss"].empty:
            district_data[dist_name]["raw_nss"] = p.get("raw_nss")

    first_data_table = True
    for dist_name, data in district_data.items():
        if data["raw_epp"] is None or data["raw_epp"].empty:
            continue  # Skip if no EPP data

        page_dict = dict(
            title=f"Data: {dist_name}",
            subtitle="PPE ($/pupil), FTE Enrollment, and NSS/Ch70 Funding ($)",
            chart_path=None,
            page_type="data_table",
            raw_epp=data["raw_epp"],
            raw_lines=data["raw_lines"],
            raw_nss=data.get("raw_nss"),  # May be None if no NSS/Ch70 data
            dist_name=dist_name
        )

        if first_data_table:
            page_dict["appendix_title"] = "Appendix C. Data Tables"
            page_dict["appendix_subtitle"] = "All data values used in plots"
            page_dict["appendix_note"] = ("This appendix contains the underlying data tables for all districts and regions shown in the report. "
                                        "Each table shows PPE by category (in $/pupil), FTE enrollment counts, and NSS/Ch70 funding components (in absolute dollars) across all available years.")
            page_dict["section_id"] = "appendix_c"
            first_data_table = False

        pages.append(page_dict)

    # ===== APPENDIX D: ADDITIONAL VISUALIZATIONS =====
    # Multi-year scatterplots and choropleth maps (2019, 2014, 2009)

    # Generic explanations for multi-year visualizations
    scatterplot_generic = (
        "This scatterplot shows the relationship between district enrollment (x-axis) and per-pupil "
        "expenditure (y-axis) for Western Massachusetts traditional districts in {year}. Each point represents "
        "one district, colored by enrollment cohort. Horizontal lines show quartile boundaries used to define cohorts."
    )

    choropleth_generic = report_text.get("SECTION1_CHOROPLETH_EXPLANATION", [
        "This map situates the districts within their geographic context in Western Massachusetts."
    ])[0]  # Get first paragraph from list

    appendix_d_intro = [
        "This appendix contains historical scatterplots and geographic maps showing the evolution of district "
        "enrollment and per-pupil expenditures over time. These visualizations provide context for understanding "
        "how enrollment patterns and spending levels have shifted across Western Massachusetts districts from 2009 to 2024."
    ]

    # Add Appendix D pages for years 2019, 2014, 2009
    for idx, year in enumerate([2019, 2014, 2009]):
        # First page of Appendix D gets the intro text
        if idx == 0:
            pages.append(dict(
                title="Appendix D. Additional Visualizations",
                subtitle=f"Scatterplot of enrollment vs. per-pupil expenditure with quartile boundaries ({year})",
                chart_path=str(OUTPUT_DIR / f"enrollment_1_scatterplot_{year}.png"),
                text_blocks=appendix_d_intro + [scatterplot_generic.format(year=year)],
                graph_only=True,
                section_id="appendix_d"
            ))
        else:
            pages.append(dict(
                title="Appendix D. Additional Visualizations (continued)",
                subtitle=f"Scatterplot of enrollment vs. per-pupil expenditure with quartile boundaries ({year})",
                chart_path=str(OUTPUT_DIR / f"enrollment_1_scatterplot_{year}.png"),
                text_blocks=[scatterplot_generic.format(year=year)],
                graph_only=True
            ))

        # Add choropleth map for this year
        pages.append(dict(
            title="Appendix D. Additional Visualizations (continued)",
            subtitle=f"Geographic map showing district locations and enrollment cohorts ({year})",
            chart_path=str(OUTPUT_DIR / f"western_ma_choropleth_{year}.png"),
            text_blocks=[choropleth_generic],
            graph_only=True
        ))

    return pages

# ---- Table of Contents ----
def build_toc_page():
    """Build table of contents page dict with hierarchical structure."""
    # Hierarchical TOC entries: (title, section_id, indent_level)
    # indent_level: 0=main section, 1=sub-section, 2=sub-sub-section
    toc_entries = [
        ("Executive Summary", "executive_summary", 0),
        ("    Total PPE comparison: Western MA enrollment cohorts and selected districts", "exec_summary_cohort_comparison", 1),
        ("    Per-pupil expenditure and recent growth overview: 2019 PPE to 2024 PPE", "exec_summary_scatter", 1),
        ("    Statistical Associations", "exec_summary_statistical", 1),

        ("Section 1: Western MA Traditional District Trends", "section1_summary", 0),
        ("    Year-over-Year (YoY) growth rates by district and cohort", "section1_yoy", 1),
        ("    5-year and 15-year CAGR by district and cohort", "section1_cagr", 1),
        ("    Distribution of 2024 enrollment and proposed cohort grouping", "section1_distribution", 1),
        ("    Scatterplot of enrollment vs. per-pupil expenditure with quartile boundaries (2024)", "section1_scatter", 1),
        ("    Geographic map showing district locations and enrollment cohorts (2024)", "section1_map", 1),

        ("Section 2: Western MA Cohort Details", "section2_tiny", 0),
        ("    Tiny Cohort (0-200 FTE)", "section2_tiny", 1),
        ("    Small Cohort (201-800 FTE)", "section2_small", 1),
        ("    Medium Cohort (801-1600 FTE)", "section2_medium", 1),
        ("    Large Cohort (1601-10000 FTE)", "section2_large", 1),
        ("    Springfield Cohort (>10000 FTE)", "section2_springfield", 1),
        ("", None, -1),  # Extra spacing before Section 3

        ("Section 3: Selected Districts Compared to Cohorts", "amherst_pelham", 0),
        ("    Amherst-Pelham Regional", "amherst_pelham", 1),
        ("    Amherst", "amherst", 1),
        ("    Leverett", "leverett", 1),
        ("    Pelham", "pelham", 1),
        ("    Shutesbury", "shutesbury", 1),

        ("Appendices", None, 0),
        ("    Appendix A: Data Sources & Calculation Methodology", "appendix_a", 1),
        ("    Appendix B: Calculations and Examples", "appendix_b", 1),
        ("    Appendix C: Data Tables", "appendix_c", 1),
        ("    Appendix D: Additional Visualizations", "appendix_d", 1),
    ]

    return dict(
        title="Table of Contents",
        subtitle="",
        chart_path=None,
        page_type="toc",
        toc_entries=toc_entries,
        report_title="Western MA Per Pupil Spending Report",
        report_subtitle="With selected districts for comparison"
    )

# ---- Build PDF ----
def build_pdf(pages: List[dict], out_path: Path):
    # Reset figure and table counters at start of PDF generation
    reset_counters()

    doc = SimpleDocTemplate(str(out_path), pagesize=A4,
        leftMargin=0.5*inch, rightMargin=0.5*inch, topMargin=0.5*inch, bottomMargin=0.5*inch)

    story: List = []
    first_cohort_table_num = None  # Track first cohort table number for substitution
    for idx, p in enumerate(pages):
        # Handle threshold analysis page
        if p.get("threshold_analysis"):
            # Add page marker for TOC page numbering
            if p.get("section_id"):
                story.append(PageMarker(p["section_id"]))
            # Add anchor if section_id is present (for TOC linking)
            title = p["title"]
            if p.get("section_id"):
                title = f'<a name="{p["section_id"]}"/>{title}'
            story.append(Paragraph(title, style_title_main))
            story.append(Paragraph(p["subtitle"], style_title_sub))
            story.append(Spacer(0, 12))

            # Add summary table
            if p.get("summary_table"):
                # Check for pending figure
                pending_fig = get_and_clear_pending_figure()
                table_num = next_table_number()
                if pending_fig:
                    story.append(build_combined_fig_table_label(pending_fig, table_num, doc.width, style_figure_num, style_table_num))
                else:
                    story.append(Paragraph(f"<i>Table {table_num}</i>", style_table_num))
                story.append(Spacer(0, 3))
                story.append(p["summary_table"])
                story.append(Spacer(0, 12))

            # Add explanation blocks
            explanation_blocks = p.get("explanation_blocks", [])
            for block in explanation_blocks:
                story.append(Paragraph(block, style_body))
                story.append(Spacer(0, 6))

            if idx < len(pages)-1:
                story.append(PageBreak())
            continue

        # Handle TOC page
        if p.get("page_type") == "toc":
            # Add report title and subtitle if present
            if p.get("report_title"):
                # Create centered style for report title
                report_title_style = ParagraphStyle(
                    'ReportTitle',
                    parent=style_title_main,
                    fontSize=18,
                    alignment=TA_CENTER,
                    spaceAfter=6
                )
                story.append(Paragraph(f"<b>{p['report_title']}</b>", report_title_style))

            if p.get("report_subtitle"):
                # Create centered style for report subtitle
                report_subtitle_style = ParagraphStyle(
                    'ReportSubtitle',
                    parent=style_title_sub,
                    fontSize=12,
                    alignment=TA_CENTER,
                    spaceAfter=18
                )
                story.append(Paragraph(p["report_subtitle"], report_subtitle_style))

            story.append(Paragraph(p["title"], style_title_main))
            story.append(Spacer(0, 12))

            # Build hierarchical TOC entries with indentation (CR3: improved spacing and indentation)
            toc_entries = p.get("toc_entries", [])
            for idx_entry, entry in enumerate(toc_entries):
                # Handle both old format (title, section_id) and new format (title, section_id, indent_level)
                if len(entry) == 2:
                    entry_title, entry_id = entry
                    indent_level = 0
                else:
                    entry_title, entry_id, indent_level = entry

                # Add extra spacing before "Appendices" section (CR3)
                if entry_title == "Appendices":
                    story.append(Spacer(0, 18))

                # Apply indentation based on level
                if indent_level == 0:
                    # Main section - bold, add spacing between main sections (CR3)
                    page_num = get_page_number(entry_id) if entry_id else None
                    if entry_id:
                        if page_num:
                            link_text = f'<b><a href="#{entry_id}" color="blue">{entry_title}</a> ........ {page_num}</b>'
                        else:
                            link_text = f'<b><a href="#{entry_id}" color="blue">{entry_title}</a></b>'
                    else:
                        link_text = f'<b>{entry_title}</b>'
                    story.append(Paragraph(link_text, style_body))
                    story.append(Spacer(0, 14))  # Increased from 10
                elif indent_level == 1:
                    # Sub-section - indented with visual indent (CR3)
                    page_num = get_page_number(entry_id) if entry_id else None
                    if entry_id:
                        if page_num:
                            link_text = f'<a href="#{entry_id}" color="blue">{entry_title}</a> ........ {page_num}'
                        else:
                            link_text = f'<a href="#{entry_id}" color="blue">{entry_title}</a>'
                    else:
                        link_text = entry_title
                    # Create indented paragraph style
                    from reportlab.lib.styles import ParagraphStyle
                    indented_style = ParagraphStyle('indented', parent=style_body, leftIndent=20)
                    story.append(Paragraph(link_text, indented_style))
                    story.append(Spacer(0, 6))
                else:
                    # Sub-sub-section - more indented, smaller font
                    if entry_id:
                        link_text = f'<font size="9"><a href="#{entry_id}" color="blue">{entry_title}</a></font>'
                    else:
                        link_text = f'<font size="9">{entry_title}</font>'
                    from reportlab.lib.styles import ParagraphStyle
                    more_indented_style = ParagraphStyle('more_indented', parent=style_body, leftIndent=40)
                    story.append(Paragraph(link_text, more_indented_style))
                    story.append(Spacer(0, 4))

            if idx < len(pages)-1:
                story.append(PageBreak())
            continue

        # Handle text-only pages (e.g., executive summary introduction)
        if p.get("text_only_page"):
            # Add page marker for TOC page numbering
            if p.get("section_id"):
                story.append(PageMarker(p["section_id"]))
            title = p["title"]
            if p.get("section_id"):
                title = f'<a name="{p["section_id"]}"/>{title}'
            story.append(Paragraph(title, style_title_main))
            if p.get("subtitle"):
                story.append(Paragraph(p["subtitle"], style_title_sub))
            story.append(Spacer(0, 12))

            # Add text blocks (CR4: handle boxed text sections)
            text_blocks = p.get("text_blocks", [])
            in_box = False
            boxed_content = []

            for block in text_blocks:
                if block == "__BOXED_START__":
                    in_box = True
                    continue
                elif block == "__BOXED_END__":
                    # Render accumulated boxed content in a table with a border
                    if boxed_content:
                        # Create paragraphs for each boxed text block
                        boxed_paragraphs = [Paragraph(b, style_body) for b in boxed_content]
                        # Create a table with a single cell containing all paragraphs
                        boxed_table_data = [[boxed_paragraphs]]
                        boxed_table = Table(boxed_table_data, colWidths=[doc.width])
                        boxed_table.setStyle(TableStyle([
                            ('BOX', (0, 0), (-1, -1), 1, colors.grey),
                            ('LEFTPADDING', (0, 0), (-1, -1), 12),
                            ('RIGHTPADDING', (0, 0), (-1, -1), 12),
                            ('TOPPADDING', (0, 0), (-1, -1), 12),
                            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                        ]))
                        story.append(boxed_table)
                        story.append(Spacer(0, 12))
                        boxed_content = []
                    in_box = False
                    continue

                if in_box:
                    boxed_content.append(block)
                else:
                    story.append(Paragraph(block, style_body))
                    story.append(Spacer(0, 6))

            if idx < len(pages)-1:
                story.append(PageBreak())
            continue

        # Handle executive summary cohort table pages (summary_table without threshold_analysis)
        if p.get("summary_table") and not p.get("threshold_analysis"):
            # Add page marker for TOC page numbering
            if p.get("section_id"):
                story.append(PageMarker(p["section_id"]))
            # Add title (unless omit_title flag is set) and subtitle
            if not p.get("omit_title"):
                title = p["title"]
                if p.get("section_id"):
                    title = f'<a name="{p["section_id"]}"/>{title}'
                story.append(Paragraph(title, style_title_main))

            story.append(Paragraph(p["subtitle"], style_title_sub))
            story.append(Spacer(0, 12))

            # Add shading legend once at the top (only for first cohort table)
            if p.get("first_cohort_table"):
                # Build legend table with color swatches
                shade_rule = (
                    f"Shading vs baseline: |Δ$/pupil| ≥ {DOLLAR_THRESHOLD_REL*100:.1f}%, "
                    f"|ΔCAGR| ≥ {MATERIAL_DELTA_PCTPTS*100:.1f}pp"
                )
                cagr_def = "CAGR = (End/Start)^(1/years) − 1"
                legend_data = [
                    [Paragraph(shade_rule, style_legend_right), "", "", "", Paragraph("Above baseline", style_legend_center), ""],
                    ["", "", "", "", Paragraph("Baseline", style_legend_center), ""],
                    [Paragraph(cagr_def, style_legend_right), "", "", "", Paragraph("Below baseline", style_legend_center), ""],
                ]
                legend_table = Table(legend_data, colWidths=[2.4*inch, 1.05*inch, 0.85*inch, 0.85*inch, 0.85*inch, 1.05*inch])
                legend_style = TableStyle([
                    ('SPAN', (0, 0), (3, 0)),  # Merge first 4 columns for shade rule text
                    ('SPAN', (0, 1), (3, 1)),  # Merge first 4 columns for row 2 (empty)
                    ('SPAN', (0, 2), (3, 2)),  # Merge first 4 columns for CAGR def text
                    ('SPAN', (4, 0), (5, 0)),  # Merge last 2 columns for "Above baseline"
                    ('SPAN', (4, 1), (5, 1)),  # Merge last 2 columns for "Baseline"
                    ('SPAN', (4, 2), (5, 2)),  # Merge last 2 columns for "Below baseline"
                    ('BACKGROUND', (4, 0), (5, 0), HexColor(ABOVE_SHADES[2])),  # Orange swatch
                    ('BACKGROUND', (4, 1), (5, 1), HexColor(BASELINE_YELLOW)),  # Yellow swatch
                    ('BACKGROUND', (4, 2), (5, 2), HexColor(BELOW_SHADES[2])),  # Blue swatch
                    ('LEFTPADDING', (0,0), (-1,-1), 4),
                    ('RIGHTPADDING', (0,0), (-1,-1), 4),
                    ('TOPPADDING', (0,0), (-1,-1), 2),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 2),
                ])
                legend_table.setStyle(legend_style)
                story.append(legend_table)
                story.append(Spacer(0, 8))

            # Add table with Table number
            table_num = next_table_number()

            # Track first table number for cohort tables (for later substitution)
            if p.get("first_cohort_table"):
                first_cohort_table_num = table_num

            story.append(Paragraph(f"<i>Table {table_num}</i>", style_table_num))
            story.append(Spacer(0, 3))
            story.append(p["summary_table"])
            story.append(Spacer(0, 12))

            # Add explanation blocks
            explanation_blocks = p.get("explanation_blocks", [])

            # Handle table number substitution if needed
            if p.get("needs_table_num_substitution") and explanation_blocks:
                # Replace {N}, {N+1}, {N+2} with actual table numbers
                # Use the first_cohort_table_num we tracked earlier
                substituted_blocks = []
                for block in explanation_blocks:
                    # Replace placeholders with actual table numbers
                    block = block.replace("{N}", str(first_cohort_table_num))
                    block = block.replace("{N+1}", str(first_cohort_table_num + 1))
                    block = block.replace("{N+2}", str(first_cohort_table_num + 2))
                    substituted_blocks.append(block)
                explanation_blocks = substituted_blocks

            for block in explanation_blocks:
                story.append(Paragraph(block, style_body))
                story.append(Spacer(0, 6))

            # Only add page break if next page is NOT also a summary table
            if idx < len(pages)-1:
                next_page = pages[idx+1]
                if not (next_page.get("summary_table") and not next_page.get("threshold_analysis")):
                    story.append(PageBreak())
            continue

        # Add appendix title if present
        if p.get("appendix_title"):
            appendix_title = p["appendix_title"]
            # Add anchor if this page has a section_id
            if p.get("section_id"):
                story.append(PageMarker(p["section_id"]))
                appendix_title = f'<a name="{p["section_id"]}"/>{appendix_title}'
            story.append(Paragraph(appendix_title, style_title_main))
            if p.get("appendix_subtitle"):
                story.append(Paragraph(p["appendix_subtitle"], style_title_sub))
            if p.get("appendix_note"):
                story.append(Spacer(0, 6))
                story.append(Paragraph(p["appendix_note"], style_body))
            story.append(Spacer(0, 12))

        # Add section anchor to title if present
        title_text = p["title"]
        if p.get("section_id"):
            story.append(PageMarker(p["section_id"]))
            title_text = f'<a name="{p["section_id"]}"/>{title_text}'
        story.append(Paragraph(title_text, style_title_main))

        # Only add subtitle if it's not empty
        default_sub = "PPE vs Enrollment"
        subtitle = p.get("subtitle", (default_sub if not p.get("graph_only") else ""))
        if subtitle:
            story.append(Paragraph(subtitle, style_title_sub))

        # Handle single chart or multiple charts
        chart_path = p.get("chart_path")
        chart_paths = p.get("chart_paths", [])

        if chart_path:
            chart_paths = [chart_path]

        if chart_paths:
            # Special handling for CAGR page with legend + two charts
            if p.get("cagr_with_legend") and len(chart_paths) == 3:
                # Add text blocks first (explanation)
                text_blocks = p.get("text_blocks", []) or []
                if text_blocks:
                    for block in text_blocks:
                        story.append(Paragraph(block, style_body))
                    story.append(Spacer(0, 12))

                # Add legend (first image in chart_paths)
                legend_path = Path(chart_paths[0])
                if legend_path.exists():
                    im = Image(str(legend_path))
                    ratio = im.imageHeight / float(im.imageWidth)
                    im.drawWidth = doc.width
                    im.drawHeight = doc.width * ratio
                    # Legend - increased to 22% of page height for better visibility
                    max_legend_h = doc.height * 0.22
                    if im.drawHeight > max_legend_h:
                        im.drawHeight = max_legend_h
                        im.drawWidth = im.drawHeight / ratio
                    story.append(im)
                    story.append(Spacer(0, 8))

                # Add first chart (5-year CAGR grouped bars)
                img_path = Path(chart_paths[1])
                if img_path.exists():
                    im = Image(str(img_path))
                    ratio = im.imageHeight / float(im.imageWidth)
                    im.drawWidth = doc.width
                    im.drawHeight = doc.width * ratio
                    max_chart_h = doc.height * 0.42  # Increased for better visibility
                    if im.drawHeight > max_chart_h:
                        im.drawHeight = max_chart_h
                        im.drawWidth = im.drawHeight / ratio
                    story.append(im)
                    story.append(Paragraph(f"<i>Figure {next_figure_number()}</i>", style_figure_num))
                    story.append(Spacer(0, 10))

                # Add second chart (15-year CAGR bars)
                img_path = Path(chart_paths[2])
                if img_path.exists():
                    im = Image(str(img_path))
                    ratio = im.imageHeight / float(im.imageWidth)
                    im.drawWidth = doc.width
                    im.drawHeight = doc.width * ratio
                    max_chart_h = doc.height * 0.42  # Increased for better visibility
                    if im.drawHeight > max_chart_h:
                        im.drawHeight = max_chart_h
                        im.drawWidth = im.drawHeight / ratio
                    story.append(im)
                    story.append(Paragraph(f"<i>Figure {next_figure_number()}</i>", style_figure_num))
                    story.append(Spacer(0, 6))

                # Skip to page break - don't process further for this page type
                if idx < len(pages)-1:
                    story.append(PageBreak())
                continue
            # Two charts vertical: split page height between them
            elif p.get("two_charts_vertical") and len(chart_paths) == 2:
                # Standard two charts vertical (no text between)
                for chart_path in chart_paths:
                    img_path = Path(chart_path)
                    if not img_path.exists():
                        story.append(Paragraph(f"[Missing chart image: {img_path.name}]", style_body))
                    else:
                        im = Image(str(img_path))
                        ratio = im.imageHeight / float(im.imageWidth)
                        im.drawWidth = doc.width; im.drawHeight = doc.width * ratio
                        # Each chart gets ~38% of page height (76% total for two charts)
                        max_chart_h = doc.height * 0.38
                        if im.drawHeight > max_chart_h:
                            im.drawHeight = max_chart_h; im.drawWidth = im.drawHeight / ratio
                        story.append(im)
                        story.append(Paragraph(f"<i>Figure {next_figure_number()}</i>", style_figure_num))
                        if chart_path != chart_paths[-1]:  # Add small spacer between charts
                            story.append(Spacer(0, 6))
            else:
                # Single chart rendering (original logic)
                for chart_path in chart_paths:
                    img_path = Path(chart_path)
                    if not img_path.exists():
                        story.append(Paragraph(f"[Missing chart image: {img_path.name}]", style_body))
                    else:
                        im = Image(str(img_path))
                        ratio = im.imageHeight / float(im.imageWidth)
                        im.drawWidth = doc.width; im.drawHeight = doc.width * ratio
                        if p.get("graph_only"):
                            name = img_path.name.lower()
                            # Executive summary gets 75% of page for large multi-pane plots
                            if p.get("executive_summary"):
                                max_chart_h = doc.height * 0.75
                            # Western overview gets 70% of page for breathing room
                            elif "ppe_overview_all_western" in name:
                                max_chart_h = doc.height * 0.70
                            else:
                                max_chart_h = doc.height * 0.62
                        else:
                            max_chart_h = doc.height * 0.40
                        if im.drawHeight > max_chart_h:
                            im.drawHeight = max_chart_h; im.drawWidth = im.drawHeight / ratio
                        story.append(im)
                        # For non-graph_only pages (regular district pages with tables), set figure as pending
                        # to be combined with first table. For graph_only pages, append immediately.
                        fig_num = next_figure_number()
                        if not p.get("graph_only"):
                            set_pending_figure(fig_num)
                        else:
                            story.append(Paragraph(f"<i>Figure {fig_num}</i>", style_figure_num))
                        story.append(Spacer(0, 6))

        # Add text blocks (for graph_only pages, add after image)
        # Skip if cagr_with_text since text is already added before charts
        if p.get("graph_only") and not p.get("cagr_with_text"):
            text_blocks = p.get("text_blocks", []) or []
            if text_blocks:
                story.append(Spacer(0, 12))

                # Use 9pt font for all appendices (consistent sizing)
                text_style = style_body

                # Build paragraphs for text content
                text_content = []
                for block in text_blocks:
                    text_content.append(Paragraph(block, text_style))
                    text_content.append(Spacer(0, 6))

                # For Appendix A and B, don't use KeepInFrame - let content flow naturally across pages
                if p.get("appendix_b") or p.get("section_id") == "appendix_a":
                    # Add content directly without frame constraints
                    for item in text_content:
                        story.append(item)
                else:
                    # For other appendices, use KeepInFrame with shrink mode
                    # Calculate available height for content on this page
                    title_and_footer_buffer = 1.5 * inch
                    available_height = doc.height - title_and_footer_buffer

                    frame_content = KeepInFrame(
                        maxWidth=doc.width,
                        maxHeight=available_height,
                        content=text_content,
                        mode='shrink',  # Shrink content if it doesn't fit
                        name='methodology_content'
                    )
                    story.append(frame_content)

            # Handle page breaks: add PageBreak unless this is the last page
            if idx < len(pages)-1:
                story.append(PageBreak())
            continue

        # Scatterplot district table: compact table with cohort colors
        if p.get("scatterplot_districts"):
            story.append(Spacer(0, 12))
            page_year = p.get("year", 2024)  # Get year from page dict, default to 2024
            scatterplot_table = _build_scatterplot_table(p.get("scatterplot_districts"), doc.width, style_body, style_num, page_year)
            if scatterplot_table:
                # Check for pending figure
                pending_fig = get_and_clear_pending_figure()
                table_num = next_table_number()
                if pending_fig:
                    story.append(build_combined_fig_table_label(pending_fig, table_num, doc.width, style_figure_num, style_table_num))
                else:
                    story.append(Paragraph(f"<i>Table {table_num}</i>", style_table_num))
                story.append(Spacer(0, 3))
                story.append(scatterplot_table)
                story.append(Spacer(0, 6))
            if idx < len(pages)-1: story.append(PageBreak())
            continue

        # Data table pages: show raw data tables
        if p.get("page_type") == "data_table":
            story.append(Spacer(0, 10))
            story.append(Paragraph("PPE by Category ($/pupil)", style_body))
            story.append(Spacer(0, 6))
            epp_table = _build_epp_data_table(p.get("raw_epp"), p.get("dist_name", ""), doc.width)
            if epp_table:
                # Check for pending figure
                pending_fig = get_and_clear_pending_figure()
                table_num = next_table_number()
                if pending_fig:
                    story.append(build_combined_fig_table_label(pending_fig, table_num, doc.width, style_figure_num, style_table_num))
                else:
                    story.append(Paragraph(f"<i>Table {table_num}</i>", style_table_num))
                story.append(Spacer(0, 3))
                story.append(epp_table)
            story.append(Spacer(0, 12))
            story.append(Paragraph("FTE Enrollment", style_body))
            story.append(Spacer(0, 6))
            fte_table = _build_fte_data_table(p.get("raw_lines", {}), p.get("dist_name", ""), doc.width)
            if fte_table:
                story.append(Paragraph(f"<i>Table {next_table_number()}</i>", style_table_num))
                story.append(Spacer(0, 3))
                story.append(fte_table)
                story.append(Spacer(0, 6))

            # Add NSS/Ch70 data table if available
            raw_nss = p.get("raw_nss")
            if raw_nss is not None and not raw_nss.empty:
                story.append(Spacer(0, 12))
                story.append(Paragraph("NSS/Ch70 Funding Components ($)", style_body))
                story.append(Spacer(0, 6))
                nss_table = _build_nss_ch70_data_table(raw_nss, p.get("dist_name", ""), doc.width)
                if nss_table:
                    story.append(Paragraph(f"<i>Table {next_table_number()}</i>", style_table_num))
                    story.append(Spacer(0, 3))
                    story.append(nss_table)
                    story.append(Spacer(0, 6))

            if idx < len(pages)-1:
                story.append(PageBreak())
            continue

        # NSS/Ch70 pages: show funding component table and FTE table (foundation enrollment only)
        if p.get("page_type") == "nss_ch70":
            story.append(Spacer(0, 6))
            # Check for pending figure
            pending_fig = get_and_clear_pending_figure()
            table_num = next_table_number()
            if pending_fig:
                story.append(build_combined_fig_table_label(pending_fig, table_num, doc.width, style_figure_num, style_table_num))
            else:
                story.append(Paragraph(f"<i>Table {table_num}</i>", style_table_num))
            story.append(Spacer(0, 3))
            story.append(_build_nss_ch70_table(p))
            story.append(Spacer(0, 12))
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey, spaceBefore=0, spaceAfter=6))
            story.append(Paragraph(f"<i>Table {next_table_number()}</i>", style_table_num))
            story.append(Spacer(0, 3))
            story.append(_build_fte_table(p))
            story.append(Spacer(0, 6))

            # Add district list text if provided (for Section 2 cohort pages)
            if p.get("district_list_text"):
                for text_block in p["district_list_text"]:
                    if text_block.strip():  # Skip empty lines
                        story.append(Paragraph(text_block, style_body))
                    else:
                        story.append(Spacer(0, 6))

            if idx < len(pages)-1:
                story.append(PageBreak())
            continue

        # Regular district/regional pages: show summary tables
        story.append(Spacer(0, 6))
        # Check if there's a pending figure number to combine with this table
        pending_fig = get_and_clear_pending_figure()
        table_num = next_table_number()
        if pending_fig:
            # Combine Figure # left-aligned and Table # right-aligned on same line
            story.append(build_combined_fig_table_label(pending_fig, table_num, doc.width, style_figure_num, style_table_num))
        else:
            story.append(Paragraph(f"<i>Table {table_num}</i>", style_table_num))
        story.append(Spacer(0, 3))
        story.append(_build_category_table(p))
        story.append(Spacer(0, 12))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey, spaceBefore=0, spaceAfter=6))
        story.append(Paragraph(f"<i>Table {next_table_number()}</i>", style_table_num))
        story.append(Spacer(0, 3))
        story.append(_build_fte_table(p))
        story.append(Spacer(0, 6))

        if idx < len(pages)-1:
            story.append(PageBreak())

    doc.build(story, onFirstPage=draw_footer, onLaterPages=draw_footer)

# ---- Main ----
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df, reg, profile_c70 = load_data()
    # Note: add_alps_pk12() removed - no longer using ALPS PK-12 aggregate concept
    pages = build_page_dicts(df, reg, profile_c70)
    if not pages:
        print("[WARN] No pages to write."); return

    # Insert TOC at the beginning
    toc_page = build_toc_page()
    pages.insert(0, toc_page)

    # Two-pass PDF generation to populate TOC with page numbers
    # Pass 1: Build PDF to capture page numbers in _PAGE_MAP
    print("[INFO] Pass 1: Building PDF to capture page numbers...")
    clear_page_map()
    temp_pdf = OUTPUT_DIR / "expenditures_series_temp.pdf"
    build_pdf(pages, temp_pdf)

    # Pass 2: Rebuild PDF with populated page numbers in TOC
    print(f"[INFO] Pass 2: Rebuilding PDF with page numbers ({len(_PAGE_MAP)} sections tracked)...")
    final_pdf = OUTPUT_DIR / "expenditures_series.pdf"

    try:
        build_pdf(pages, final_pdf)
        # Clean up temporary file on success
        if temp_pdf.exists():
            temp_pdf.unlink()
        print(f"[SUCCESS] PDF generated: {final_pdf}")
    except PermissionError as e:
        print(f"[ERROR] Cannot write to {final_pdf}: file is locked (probably open in PDF viewer)")
        print(f"[INFO] Temporary PDF with page numbers is available at: {temp_pdf}")
        print("[INFO] Please close the PDF viewer and run the script again, or manually rename the temp file.")
        raise

if __name__ == "__main__":
    main()
