"""
PDF Report Generator for School District Expenditure Analysis

This module creates a comprehensive PDF report with:
- Section 1: Western MA overview and 4 enrollment-based aggregate groups
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
from reportlab.platypus import Flowable, Image, KeepInFrame, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

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
    get_cohort_label, get_cohort_short_label,
    get_western_cohort_districts, get_omitted_western_districts,
)
from nss_ch70_plots import build_nss_category_data, NSS_CH70_COLORS, NSS_CH70_STACK_ORDER

# ===== code version =====
CODE_VERSION = "v2025.09.29-REFACTORED"

# ---- Shading controls ----
# NOTE: Two independent tests determine cell shading in district comparison tables:
# 1. CAGR Test: Compares district CAGR to baseline using ABSOLUTE percentage point difference
#    Example: District 5.0% vs Baseline 3.0% = 2.0pp difference → shade if >= threshold
MATERIAL_DELTA_PCTPTS = 0.02  # 2.0pp threshold for CAGR shading

# 2. Dollar Test: Compares district $/pupil to baseline using RELATIVE percent difference
#    Example: District $20,000 vs Baseline $19,000 = +5.3% relative → shade if >= threshold
DOLLAR_THRESHOLD_REL  = 0.02  # 2.0% threshold for dollar shading

# Bins for shading intensity (used for both CAGR pp-delta and $ relative delta)
# Higher deltas get darker shading: [2%, 5%, 8%, 12%] bins → 5 shade levels
SHADE_BINS = [0.02, 0.05, 0.08, 0.12]
# Neutral comparison colors (not implying good/bad):
ABOVE_SHADES = ["#FFF4E6", "#FFE8CC", "#FFD9A8", "#FFC97A", "#FFB84D"]  # above baseline: light amber/tan (lightest→darkest)
BELOW_SHADES = ["#E0F7FA", "#B2EBF2", "#80DEEA", "#4DD0E1", "#26C6DA"]  # below baseline: light teal/cyan (lightest→darkest)

# Optional $-gate; if >0, shading applies only when latest $/pupil >= this
# Currently disabled (0.0) - all categories get shading regardless of magnitude
MATERIAL_MIN_LATEST_DOLLARS = 0.0

def district_png_simple(dist: str) -> Path:
    return OUTPUT_DIR / f"expenditures_per_pupil_vs_enrollment_{dist.replace(' ', '_')}_simple.png"

def district_png_detail(dist: str) -> Path:
    return OUTPUT_DIR / f"expenditures_per_pupil_vs_enrollment_{dist.replace(' ', '_')}_detail.png"

def regional_png(bucket: str) -> Path:
    return OUTPUT_DIR / f"regional_expenditures_per_pupil_Western_Traditional_{bucket}.png"

# ---- Styles ----
styles = getSampleStyleSheet()
style_title_main = ParagraphStyle("title_main", parent=styles["Heading1"], fontSize=14, leading=17, spaceAfter=2)
style_title_sub  = ParagraphStyle("title_sub",  parent=styles["Normal"],   fontSize=12, leading=14, spaceAfter=6)
style_body       = ParagraphStyle("body",       parent=styles["Normal"],   fontSize=9,  leading=12)
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

def _build_scatterplot_table(district_data: List[Tuple[str, str, float, float, str]], doc_width: float, style_body, style_num):
    """Build compact 2-column table showing all districts with cohort colors, enrollment, and 2024 PPE."""
    if not district_data:
        return None

    # Cohort colors matching the scatterplot
    cohort_colors = {
        'TINY': HexColor('#9C27B0'),    # Purple
        'SMALL': HexColor('#4CAF50'),   # Green
        'MEDIUM': HexColor('#2196F3'),  # Blue
        'LARGE': HexColor('#FF9800'),   # Orange
    }

    # Smaller font styles for compact table
    style_small = ParagraphStyle("small", parent=style_body, fontSize=7, leading=9)
    style_small_num = ParagraphStyle("small_num", parent=style_num, fontSize=7, leading=9)

    # Split districts into two columns
    mid = (len(district_data) + 1) // 2
    left_districts = district_data[:mid]
    right_districts = district_data[mid:]

    # Build combined table data
    data = [
        [Paragraph("", style_small),  # Left swatch
         Paragraph("<b>District</b>", style_small),
         Paragraph("<b>2024<br/>In-district<br/>FTE</b>", style_small_num),
         Paragraph("<b>2024<br/>PPE ▼</b>", style_small_num),  # Sort arrow indicator
         Paragraph("", style_small),  # Gap
         Paragraph("", style_small),  # Right swatch
         Paragraph("<b>District</b>", style_small),
         Paragraph("<b>2024<br/>In-district<br/>FTE</b>", style_small_num),
         Paragraph("<b>2024<br/>PPE ▼</b>", style_small_num)]  # Sort arrow indicator
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
    from school_shared import latest_total_fte, get_enrollment_group, get_cohort_label

    # Get Western MA traditional districts
    mask = (reg["EOHHS_REGION"].str.lower() == "western") & (reg["SCHOOL_TYPE"].str.lower() == "traditional")
    western_districts = sorted(set(reg[mask]["DIST_NAME"].str.lower()))
    present = set(df["DIST_NAME"].str.lower())
    western_districts = [d for d in western_districts if d in present]

    district_data = []
    for dist in western_districts:
        enrollment = latest_total_fte(df, dist)
        if enrollment > 0 and enrollment <= 8000:  # Exclude Springfield
            # Get total PPE for latest year
            ppe_data = df[
                (df["DIST_NAME"].str.lower() == dist) &
                (df["IND_CAT"].str.lower() == "expenditures per pupil") &
                (df["YEAR"] == latest_year)
            ]
            total_ppe = ppe_data[~ppe_data["IND_SUBCAT"].str.lower().isin(["total expenditures", "total in-district expenditures"])]["IND_VALUE"].sum()

            # ONLY include districts with valid PPE data
            if total_ppe > 0:
                cohort = get_enrollment_group(enrollment)
                cohort_label = get_cohort_label(cohort)
                district_data.append((dist.title(), cohort, float(enrollment), float(total_ppe), cohort_label))

    # Sort by cohort (TINY, SMALL, MEDIUM, LARGE) then by PPE descending within each cohort
    cohort_order = {"TINY": 0, "SMALL": 1, "MEDIUM": 2, "LARGE": 3}
    district_data.sort(key=lambda x: (cohort_order.get(x[1], 999), -x[3]))

    return district_data

def _abbr_bucket_suffix(full: str) -> str:
    """Extract abbreviated bucket suffix from full baseline title for legend display."""
    # Match by cohort keywords (case-insensitive) - works with any dynamic boundaries
    full_lower = (full or "").lower()

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
        return "(Springfield: >8000 FTE)"

    # Legacy fallback for old 2-tier system (should not be used)
    if re.search(r"≤\s*500", full or "", flags=re.I):
        return "(≤500)"
    if re.search(r">\s*500", full or "", flags=re.I):
        return "(>500)"

    return ""

def _shade_for_cagr_delta(delta_pp: float):
    # delta_pp is a true fraction (e.g., 0.021 = 2.1pp)
    if delta_pp != delta_pp or abs(delta_pp) < MATERIAL_DELTA_PCTPTS:
        return None
    idx = max(0, min(len(ABOVE_SHADES)-1, bisect.bisect_right(SHADE_BINS, abs(delta_pp)) - 1))
    return HexColor(ABOVE_SHADES[idx]) if delta_pp > 0 else HexColor(BELOW_SHADES[idx])

def _shade_for_dollar_rel(delta_rel: float):
    # delta_rel is a relative diff (e.g., +0.03 = +3%)
    if delta_rel != delta_rel or abs(delta_rel) < DOLLAR_THRESHOLD_REL:
        return None
    idx = max(0, min(len(ABOVE_SHADES)-1, bisect.bisect_right(SHADE_BINS, abs(delta_rel)) - 1))
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

    # Legend rows: left explanations + right swatches spanning the rightmost three columns
    # NOTE: Legend now spans columns 4-6 (CAGR 10y, 5y, Latest $) - rightmost 3 columns
    legend_rows = []
    if page.get("page_type") == "district":
        baseline_title = page.get("baseline_title", "")
        bucket_suffix = _abbr_bucket_suffix(baseline_title)

        above_text = f"Above Western {bucket_suffix}".strip()
        below_text = f"Below Western {bucket_suffix}".strip()
        shade_rule = (
            f"Shading vs Western: |Δ$/pupil| ≥ {DOLLAR_THRESHOLD_REL*100:.1f}% "
            f"or |ΔCAGR| ≥ {MATERIAL_DELTA_PCTPTS*100:.1f}pp"
        )
        cagr_def   = "CAGR = (End/Start)^(1/years) − 1"
        legend_rows = [
            ["", Paragraph(shade_rule, style_legend_right), "", "", Paragraph(above_text, style_legend_center), "", ""],
            ["", Paragraph(cagr_def,   style_legend_right), "", "", Paragraph(below_text, style_legend_center), "", ""],
        ]
    data.extend(legend_rows)
    leg1_idx = total_row_idx + 1 if legend_rows else None
    leg2_idx = total_row_idx + 2 if legend_rows else None

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

    # Apply legend spans & swatches (immediately below TOTAL row)
    # Legend text spans cols 1-3, swatch spans cols 4-6 (rightmost 3 columns: CAGR 10y, 5y, Latest $)
    if leg1_idx is not None:
        ts.add("SPAN", (1, leg1_idx), (3, leg1_idx))
        ts.add("SPAN", (1, leg2_idx), (3, leg2_idx))
        ts.add("SPAN", (4, leg1_idx), (6, leg1_idx))
        ts.add("SPAN", (4, leg2_idx), (6, leg2_idx))
        sw_above = HexColor(ABOVE_SHADES[1]); sw_below = HexColor(BELOW_SHADES[1])
        ts.add("BACKGROUND", (4, leg1_idx), (6, leg1_idx), sw_above)
        ts.add("BACKGROUND", (4, leg2_idx), (6, leg2_idx), sw_below)

    tbl.setStyle(ts)
    return tbl

def _build_fte_table(page: dict) -> Table:
    """
    Build FTE enrollment table with chronological column order.

    Columns (left to right): Swatch, FTE Series, t0 FTE, CAGR 15y, 10y, 5y, Latest FTE
    NOTE: Reordered to match category table chronology
    """
    rows_in = page.get("fte_rows", [])
    latest_year_fte = page.get("latest_year_fte", page.get("latest_year","Latest"))
    t0_year_fte = latest_year_fte - 15 if isinstance(latest_year_fte, int) else 2009

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

    # Legend rows: left explanations + right swatches spanning the rightmost three columns
    legend_rows = []
    if page.get("page_type") == "nss_ch70" and base:
        baseline_title = page.get("baseline_title", "")
        bucket_suffix = _abbr_bucket_suffix(baseline_title)

        above_text = f"Above Western {bucket_suffix}".strip()
        below_text = f"Below Western {bucket_suffix}".strip()
        shade_rule = (
            f"Shading vs Western: |Δ$/pupil| ≥ {DOLLAR_THRESHOLD_REL*100:.1f}% "
            f"or |ΔCAGR| ≥ {MATERIAL_DELTA_PCTPTS*100:.1f}pp"
        )
        cagr_def = "CAGR = (End/Start)^(1/years) − 1"
        legend_rows = [
            ["", Paragraph(shade_rule, style_legend_right), "", "", Paragraph(above_text, style_legend_center), "", ""],
            ["", Paragraph(cagr_def, style_legend_right), "", "", Paragraph(below_text, style_legend_center), "", ""],
        ]
    data.extend(legend_rows)
    leg1_idx = total_row_idx + 1 if legend_rows else None
    leg2_idx = total_row_idx + 2 if legend_rows else None

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

    # Apply legend spans & swatches (immediately below TOTAL row)
    # Legend text spans cols 1-3, swatch spans cols 4-6 (rightmost 3 columns)
    if leg1_idx is not None:
        ts.add("SPAN", (1, leg1_idx), (3, leg1_idx))
        ts.add("SPAN", (1, leg2_idx), (3, leg2_idx))
        ts.add("SPAN", (4, leg1_idx), (6, leg1_idx))
        ts.add("SPAN", (4, leg2_idx), (6, leg2_idx))
        sw_above = HexColor(ABOVE_SHADES[1]); sw_below = HexColor(BELOW_SHADES[1])
        ts.add("BACKGROUND", (4, leg1_idx), (6, leg1_idx), sw_above)
        ts.add("BACKGROUND", (4, leg2_idx), (6, leg2_idx), sw_below)

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

def _build_nss_ch70_baseline_map(nss_data: pd.DataFrame, latest_year: int) -> dict:
    """
    Build baseline map for NSS/Ch70 components for comparison.

    Returns dict mapping component name (including "Total") to:
        - DOLLAR: latest $/pupil value
        - START_DOLLAR: start year (15y ago) $/pupil value
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

# ---- Page dicts ----
def build_page_dicts(df: pd.DataFrame, reg: pd.DataFrame, c70: pd.DataFrame) -> List[dict]:
    pages: List[dict] = []

    latest = int(df["YEAR"].max())
    t0 = latest - 5

    # ===== SECTION 1: WESTERN MA =====
    # Get omitted districts for note on first page
    omitted_districts = get_omitted_western_districts(df, reg)

    # PAGE 1: All Western MA Districts Overview (graph-only)
    # Plot 1: PPE Overview (horizontal bar chart with district names, sorted by 2024 PPE)
    western_explanation = "Each bar represents one district's per-pupil expenditures (PPE). Districts are sorted by 2024 PPE (lowest to highest)."

    # Add omitted districts note if any
    if omitted_districts:
        omitted_note = "<i>Note: The following districts are omitted from this analysis: "
        omitted_list = ", ".join([f"{name} ({reason})" for name, reason in omitted_districts])
        omitted_note += omitted_list + ".</i>"
        western_text_blocks = [western_explanation, omitted_note]
    else:
        western_text_blocks = [western_explanation]

    pages.append(dict(
        title="All Western MA Traditional Districts",
        subtitle=f"Per-pupil expenditure overview: {t0} PPE (lighter) to {latest} PPE (darker segment)",
        chart_path=str(OUTPUT_DIR / "ppe_overview_all_western.png"),
        text_blocks=western_text_blocks,
        graph_only=True,
        section_id="section1_western"
    ))

    # Add enrollment distribution analysis pages (3 plots total)
    # NEW ORDER: PPE Overview, Histogram, Grouping, Scatterplot
    # Explanations updated per user requirements

    # Combined explanation for histogram and grouping plots
    enrollment_dist_explanation = (
        "The histogram (top) shows right-skewed enrollment distribution using 250 FTE bins, "
        "with a long right tail and sparse districts at higher enrollments. "
        "Based on IQR analysis, districts are grouped into four enrollment-based cohorts (bottom): "
        f"{get_cohort_label('SMALL')}, {get_cohort_label('MEDIUM')}, "
        f"{get_cohort_label('LARGE')}, "
        "and Springfield (>8000 FTE, statistical outlier). "
        "These cohorts enable meaningful comparisons between districts facing similar scale-related challenges "
        "in staffing, facilities, and programming."
    )

    scatterplot_explanation = ("This scatterplot shows the relationship between district enrollment and per-pupil expenditures. "
                              "Each point represents one district's 2024 in-district FTE enrollment (x-axis) and total PPE (y-axis). "
                              "Points are colored by enrollment cohort: Small (0-800 FTE), Medium (801-1800 FTE), and Large (1801-8000 FTE). "
                              "Springfield is omitted as a high-enrollment outlier. "
                              "Horizontal lines show quartile boundaries: Q1=208 FTE, Q2 (Median)=798 FTE, Q3=1768 FTE.")

    # Plot 2: Combined histogram + grouping on one page
    pages.append(dict(
        title="All Western MA Traditional Districts",
        subtitle="Enrollment distribution and proposed four-tier grouping",
        chart_paths=[
            str(OUTPUT_DIR / "enrollment_3_histogram.png"),
            str(OUTPUT_DIR / "enrollment_4_grouping.png")
        ],
        text_blocks=[enrollment_dist_explanation],
        graph_only=True,
        two_charts_vertical=True  # Stack charts vertically
    ))

    # Plot 4: Scatterplot (now enrollment vs PPE with cohort colors) - with district table
    # Build district table data for scatterplot page
    scatterplot_table_data = _build_scatterplot_district_table(df, reg, latest)

    pages.append(dict(
        title="All Western MA Traditional Districts",
        subtitle="Scatterplot of enrollment vs. per-pupil expenditure with quartile boundaries",
        chart_path=str(OUTPUT_DIR / "enrollment_1_scatterplot.png"),
        text_blocks=[scatterplot_explanation],
        graph_only=False,  # Now includes table
        scatterplot_districts=scatterplot_table_data  # Custom table data
    ))

    cmap_all = create_or_load_color_map(df)

    # Get Western cohorts using centralized function
    cohorts = get_western_cohort_districts(df, reg)
    cohort_map = {"tiny": "TINY", "small": "SMALL", "medium": "MEDIUM", "large": "LARGE", "springfield": "SPRINGFIELD"}

    # Add Western MA aggregate pages to Section 1 (5 enrollment groups)
    for bucket in ("tiny", "small", "medium", "large", "springfield"):
        district_list = cohorts[cohort_map[bucket]]
        title, epp, lines_sum, lines_mean = prepare_western_epp_lines(df, reg, bucket, c70, districts=district_list)
        if epp.empty and not lines_sum:
            continue

        latest_year = get_latest_year(df, epp)
        context = context_for_western(bucket)

        rows, total, start_map = _build_category_data(epp, latest_year, context, cmap_all)
        fte_rows, fte_map, latest_fte_year = _build_fte_data(lines_mean, latest_year)

        pages.append(dict(title=title,
                        subtitle="PPE vs Enrollment — Weighted average per district",
                        chart_path=str(regional_png(bucket)),
                        latest_year=latest_year, latest_year_fte=latest_fte_year,
                        cat_rows=rows, cat_total=total, cat_start_map=start_map, fte_rows=fte_rows,
                        fte_series_map=fte_map, page_type="western",
                        raw_epp=epp, raw_lines=lines_mean, dist_name=title))

        # Add NSS/Ch70 page for this Western aggregate (weighted per-pupil)
        if c70 is not None and not c70.empty:
            # Get district list for this bucket using enrollment groups
            western_mask = (reg["EOHHS_REGION"].str.lower() == "western") & (reg["SCHOOL_TYPE"].str.lower() == "traditional")
            western_all = sorted(set(reg[western_mask]["DIST_NAME"].str.lower()))
            western_present = [d for d in western_all if d in set(df["DIST_NAME"].str.lower())]

            bucket_districts = []
            for dist in western_present:
                fte = latest_total_fte(df, dist)
                group = get_enrollment_group(fte).lower()
                if group == bucket:
                    bucket_districts.append(dist)

            if not bucket_districts:
                continue  # Skip if no districts in this enrollment group

            nss_west, enroll_west, foundation_west = prepare_aggregate_nss_ch70_weighted(df, c70, bucket_districts)
            if not nss_west.empty:
                latest_year_nss = int(nss_west.index.max())
                cat_rows_nss, cat_total_nss, cat_start_map_nss = build_nss_category_data(nss_west, latest_year_nss)
                fte_rows_nss, fte_map_nss, latest_fte_year_nss = _build_nss_fte_data(foundation_west, latest_year_nss)

                safe_name = f"Western_MA_{bucket}"
                pages.append(dict(
                    title=title,
                    subtitle="Chapter 70 Aid and Net School Spending (NSS). Weighted avg funding per district: State aid (Ch70), Required local contribution, and Actual spending above requirement",
                    chart_path=str(OUTPUT_DIR / f"nss_ch70_{safe_name}.png"),
                    latest_year=latest_year_nss,
                    cat_rows=cat_rows_nss,
                    cat_total=cat_total_nss,
                    cat_start_map=cat_start_map_nss,
                    fte_rows=fte_rows_nss,
                    fte_series_map=fte_map_nss,
                    latest_year_fte=latest_fte_year_nss,
                    page_type="nss_ch70",
                    dist_name=title,
                    raw_nss=nss_west  # Store for Appendix A data tables
                ))

    # ===== SECTION 2: INDIVIDUAL DISTRICTS =====

    # Pre-compute Western district lists for NSS/Ch70 baseline computation (4 enrollment groups)
    western_mask = (reg["EOHHS_REGION"].str.lower() == "western") & (reg["SCHOOL_TYPE"].str.lower() == "traditional")
    western_all = sorted(set(reg[western_mask]["DIST_NAME"].str.lower()))
    western_present = [d for d in western_all if d in set(df["DIST_NAME"].str.lower())]

    # Organize districts into 5 enrollment groups
    western_tiny = []
    western_small = []
    western_medium = []
    western_large = []
    western_springfield = []

    for dist in western_present:
        fte = latest_total_fte(df, dist)
        group = get_enrollment_group(fte)
        if group == "TINY":
            western_tiny.append(dist)
        elif group == "SMALL":
            western_small.append(dist)
        elif group == "MEDIUM":
            western_medium.append(dist)
        elif group == "LARGE":
            western_large.append(dist)
        elif group == "SPRINGFIELD":
            western_springfield.append(dist)

    # DISTRICT PAGES - Three pages per district (simple + detailed vs Western + detailed vs Peers)
    for dist in ["Amherst-Pelham"] + [d for d in DISTRICTS_OF_INTEREST if d != "Amherst-Pelham"]:
        epp, lines = prepare_district_epp_lines(df, dist, c70)
        if epp.empty and not lines:
            continue

        dist_title = f"Amherst-Pelham Regional" if dist == "Amherst-Pelham" else dist
        # Create section_id from district name (lowercase, replace spaces with underscores)
        section_id = dist.lower().replace(" ", "_").replace("-", "_")

        # Simple version (solid color, no tables)
        pages.append(dict(
            title=dist_title,
            subtitle="PPE vs Enrollment",
            chart_path=str(district_png_simple(dist)),
            graph_only=True,
            section_id=section_id
        ))

        # Detailed version with tables
        latest_year = get_latest_year(df, epp)
        context = context_for_district(df, dist)

        rows, total, start_map = _build_category_data(epp, latest_year, context, cmap_all)
        fte_rows, fte_map, latest_fte_year = _build_fte_data(lines, latest_year)

        # Map context to bucket and label using centralized cohort definitions
        bucket_map = {
            "TINY": ("tiny", get_cohort_label("TINY")),
            "SMALL": ("small", get_cohort_label("SMALL")),
            "MEDIUM": ("medium", get_cohort_label("MEDIUM")),
            "LARGE": ("large", get_cohort_label("LARGE")),
            "SPRINGFIELD": ("springfield", "Springfield (>8000 FTE)")
        }
        bucket, bucket_label = bucket_map.get(context, ("tiny", get_cohort_label("TINY")))

        base_title = f"All Western MA Traditional Districts: {bucket_label}"
        base_map = {}
        district_list = cohorts[context]
        title_w, epp_w, _ls, _lm = prepare_western_epp_lines(df, reg, bucket, c70, districts=district_list)
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

        pages.append(dict(
            title=dist_title,
            subtitle=f"PPE vs Enrollment. Per-pupil expenditures stacked by expense category, expense category table shaded by comparison to weighted average of Western MA {bucket_label}",
            chart_path=str(district_png_detail(dist)),
            latest_year=latest_year, latest_year_fte=latest_fte_year,
            cat_rows=rows, cat_total=total, cat_start_map=start_map, fte_rows=fte_rows,
            fte_series_map=fte_map,
            page_type="district", baseline_title=base_title, baseline_map=base_map,
            raw_epp=epp, raw_lines=lines, dist_name=dist
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
                    "SPRINGFIELD": (western_springfield, "Springfield (>8000 FTE)")
                }
                western_dists, group_label = group_map.get(group, (western_tiny, get_cohort_label("TINY")))

                # Compute Western NSS/Ch70 baseline (weighted per-pupil for comparison)
                nss_west_baseline = {}
                if western_dists:
                    nss_west, _, _ = prepare_aggregate_nss_ch70_weighted(df, c70, western_dists)
                    if not nss_west.empty:
                        nss_west_baseline = _build_nss_ch70_baseline_map(nss_west, latest_year_nss)

                safe_name = dist.replace("-", "_").replace(" ", "_")

                pages.append(dict(
                    title=dist_title,
                    subtitle=f"Chapter 70 Aid, Net School Spending (NSS), and Foundation Enrollment. Funding component table shaded by comparison to weighted average of Western MA {group_label}",
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
                    dist_name=dist,
                    raw_nss=nss_dist  # Store raw data for Appendix A
                ))

    # Note: Section 3 (ALPS PK-12 & Peers) removed. Districts now compared to enrollment-based peer groups.

    # ===== APPENDIX A: DATA TABLES (was Appendix B) =====
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
            page_dict["appendix_title"] = "Appendix A. Data Tables"
            page_dict["appendix_subtitle"] = "All data values used in plots"
            page_dict["appendix_note"] = ("This appendix contains the underlying data tables for all districts and regions shown in the report. "
                                        "Each table shows PPE by category (in $/pupil), FTE enrollment counts, and NSS/Ch70 funding components (in absolute dollars) across all available years.")
            page_dict["section_id"] = "appendix_a"
            first_data_table = False

        pages.append(page_dict)

    # ===== APPENDIX B: CALCULATION METHODOLOGY =====

    # Get Western MA districts organized by 5-tier enrollment groups
    mask = (reg["EOHHS_REGION"].str.lower() == "western") & (reg["SCHOOL_TYPE"].str.lower() == "traditional")
    western_districts = sorted(reg[mask]["DIST_NAME"].unique())

    western_tiny = []
    western_small = []
    western_medium = []
    western_large = []
    western_springfield = []

    for dist in western_districts:
        dist_enr = df[
            (df["DIST_NAME"].str.lower() == dist.lower()) &
            (df["IND_CAT"].str.lower() == "student enrollment") &
            (df["IND_SUBCAT"].str.lower() == "in-district fte pupils") &
            (df["YEAR"] == latest)
        ]["IND_VALUE"]
        if not dist_enr.empty:
            enr = float(dist_enr.iloc[0])
            group = get_enrollment_group(enr)
            if group == "TINY":
                western_tiny.append(dist)
            elif group == "SMALL":
                western_small.append(dist)
            elif group == "MEDIUM":
                western_medium.append(dist)
            elif group == "LARGE":
                western_large.append(dist)
            elif group == "SPRINGFIELD":
                western_springfield.append(dist)

    # Split methodology into three pages to avoid footer overlap
    methodology_page1 = [
        "<b>2. Per-Pupil Expenditure (PPE) Definition</b>",
        "",
        "Per-pupil expenditure (PPE) is reported in the End of Year Report (EOYR) for municipal and regional districts, and is calculated by in-district FTE.",
        "",
        "Per the DESE's Researcher's Guide, section XV. Using financial data:",
        "<i>\"The out-of-district total cannot be properly reported as a per-pupil expenditure because the cost of tuitions varies greatly depending on the reason for going out of district.\"</i>",
        "",
        "<b>3. Compound Annual Growth Rate (CAGR)</b>",
        "",
        "CAGR measures the mean annual growth rate over a specified time period, assuming constant growth.",
        "",
        "<b>Formula:</b> CAGR = (End_Value / Start_Value)^(1 / Years) − 1",
        "",
        "<b>Example:</b> If expenditures grow from $10,000 to $12,000 over 5 years:",
        "CAGR = ($12,000 / $10,000)^(1/5) − 1 = 1.2^0.2 − 1 = 0.0371 = 3.71% per year",
        "",
        "Note: CAGR requires positive values at both endpoints and is undefined if data is missing.",
        "",
        "<b>4. Enrollment-Based Peer Groups</b>",
        "",
        "Districts are grouped by total in-district FTE enrollment into five cohorts (based on IQR analysis) for meaningful peer comparison:",
        "",
        f"<b>{get_cohort_label('TINY')}:</b> {len(western_tiny)} districts (Cohort 1: below Q1)",
        f"<b>Member districts:</b> {', '.join(western_tiny) if len(western_tiny) <= 15 else ', '.join(western_tiny[:15]) + f', and {len(western_tiny)-15} others'}",
        "",
        f"<b>{get_cohort_label('SMALL')}:</b> {len(western_small)} districts (Cohort 2: Q1 to median)",
        f"<b>Member districts:</b> {', '.join(western_small) if len(western_small) <= 15 else ', '.join(western_small[:15]) + f', and {len(western_small)-15} others'}",
        "",
        f"<b>{get_cohort_label('MEDIUM')}:</b> {len(western_medium)} districts (Cohort 3: median to Q3)",
        f"<b>Member districts:</b> {', '.join(western_medium) if len(western_medium) <= 15 else ', '.join(western_medium[:15]) + f', and {len(western_medium)-15} others'}",
        "",
        f"<b>{get_cohort_label('LARGE')}:</b> {len(western_large)} districts (Cohort 4: above Q3)",
        f"<b>Member districts:</b> {', '.join(western_large)}",
        "",
        f"<b>Springfield (&gt;8000 FTE):</b> {len(western_springfield)} district (Cohort 5: statistical outlier, analyzed separately)",
        f"<b>Member districts:</b> {', '.join(western_springfield) if western_springfield else 'None'}",
        "",
        "<b>Weighted EPP Formula (for aggregate calculations):</b>",
        "For each category and year: Weighted_EPP = Σ(District_EPP × District_In-District_FTE) / Σ(District_In-District_FTE)",
        "",
        "<b>Example:</b> District A spends $5,000/pupil (500 students), District B spends $6,000/pupil (300 students)",
        "Weighted average = ($5,000×500 + $6,000×300) / (500+300) = $5,375/pupil",
        "",
        "<b>Enrollment Calculation:</b> For each series (In-District FTE, Out-of-District FTE) and year: Sum across all member districts in the enrollment group.",
    ]

    methodology_page2 = [
        "<b>5. Red/Green Shading Logic (District Comparison Tables)</b>",
        "",
        "District pages include tables comparing each district's per-pupil expenditures (PPE) and growth rates (CAGR) to their enrollment-based peer group aggregate (Small, Medium, Large, or Springfield).",
        "",
        "<b>Two independent tests determine shading:</b>",
        "",
        "<b>Test 1 - Dollar Amount (2024 $/pupil column):</b>",
        f"Compares the district's 2024 PPE to the baseline's 2024 PPE using relative difference: (District − Baseline) / Baseline",
        f"• <b>Red shading:</b> District spending is ≥{DOLLAR_THRESHOLD_REL*100:.1f}% higher than baseline",
        f"• <b>Green shading:</b> District spending is ≥{DOLLAR_THRESHOLD_REL*100:.1f}% lower than baseline",
        f"• <b>No shading:</b> Difference is less than {DOLLAR_THRESHOLD_REL*100:.1f}%",
        "",
        "<b>Test 2 - CAGR (5y, 10y, 15y columns):</b>",
        f"Compares the district's CAGR to the baseline's CAGR using absolute percentage point difference: District_CAGR − Baseline_CAGR",
        f"• <b>Red shading:</b> District CAGR is ≥{MATERIAL_DELTA_PCTPTS*100:.1f} percentage points higher than baseline",
        f"• <b>Green shading:</b> District CAGR is ≥{MATERIAL_DELTA_PCTPTS*100:.1f} percentage points lower than baseline",
        f"• <b>No shading:</b> Difference is less than {MATERIAL_DELTA_PCTPTS*100:.1f} percentage points",
        "",
        "<b>Key insight:</b> The tests are independent. A red 2024 $/pupil with unshaded CAGRs typically means:",
        "• The district started at a higher baseline 15 years ago, AND",
        "• The district has been growing at roughly the same rate as peers",
        "• Therefore it remains higher in absolute dollars but isn't growing faster",
    ]

    methodology_page3 = [
        "<b>6. Chapter 70 Aid and Net School Spending (NSS) Calculations</b>",
        "",
        "Chapter 70 is Massachusetts' primary state aid program for K-12 education. Net School Spending (NSS) is the total amount a district spends on education from local and state sources.",
        "",
        "<b>Data Sources:</b>",
        "• <b>Chapter 70 Aid (c70aid):</b> State aid received by the district (DESE profile_DataC70 sheet)",
        "• <b>Required NSS (rqdnss2):</b> Minimum spending required by state law, adjusted (DESE profile_DataC70 sheet)",
        "• <b>Actual NSS (actualNSS):</b> Total district spending on education (DESE profile_DataC70 sheet)",
        "",
        "<b>Important Note:</b>",
        "NSS/Ch70 values are reported in <b>absolute dollars</b>, NOT per-pupil. Unlike PPE (which divides by in-district FTE), NSS/Ch70 values represent total district funding amounts.",
        "",
        "<b>Stacked Components (bottom to top in plots):</b>",
        "1. <b>Ch70 Aid ($):</b> c70aid",
        "   • Green bar in plots • State funding received by district",
        "",
        "2. <b>Req NSS (adj) ($):</b> max(0, rqdnss2 − c70aid)",
        "   • Amber bar in plots • Required local contribution (after subtracting Ch70)",
        "   • Uses max(0, ...) to handle rare cases where Ch70 > Required NSS",
        "",
        "3. <b>Actual NSS (adj) ($):</b> actualNSS − rqdnss2",
        "   • Purple bar in plots • Spending above minimum requirement",
        "   • Represents discretionary local spending beyond state mandates",
        "",
        "<b>Total NSS ($):</b> Sum of all three components = actualNSS",
        "",
        "<b>Example Calculation (Amherst, FY2024):</b>",
        "• Ch70 Aid: $6,791,000",
        "• Required NSS: $18,859,000; Req NSS (adj): $18,859,000 − $6,791,000 = $12,068,000",
        "• Actual NSS: $31,511,000; Actual NSS (adj): $31,511,000 − $18,859,000 = $12,652,000",
        "• Total NSS: $6,791,000 + $12,068,000 + $12,652,000 = $31,511,000",
        "",
        "<b>Aggregate Calculation:</b>",
        "For aggregate enrollment groups (Small, Medium, Large, Springfield), sum dollar amounts across all member districts by year:",
        "• Ch70 Aid ($) = Σ(District_Ch70)",
        "• Same calculation method for Req NSS (adj) and Actual NSS (adj)",
        "",
        "<b>Shading:</b>",
        "NSS/Ch70 comparison tables use the same red/green shading logic as PPE tables (2% dollar threshold, 2pp CAGR threshold).",
    ]

    # Add Data Sources page first (now #1)
    methodology_page4 = [
        "<b>1. Data Sources</b>",
        "",
        "<b>District Expenditures by Spending Category:</b>",
        "Source: Massachusetts Department of Elementary and Secondary Education (DESE)",
        "Last updated: August 12, 2025",
        "URL: https://educationtocareer.data.mass.gov/Finance-and-Budget/District-Expenditures-by-Spending-Category/er3w-dyti/",
        "",
        "This dataset provides detailed expenditure data by category for all Massachusetts school districts, including:",
        "• Expenditures per pupil (EPP) by spending category",
        "• Student enrollment (In-District FTE, Out-of-District FTE, Total FTE)",
        "• Annual data from FY1993 to present",
        "",
        "<b>Chapter 70 District Profiles:</b>",
        "Source: Massachusetts Department of Elementary and Secondary Education (DESE)",
        "URL: Various sources compiled in profile_DataC70 spreadsheet tab",
        "",
        "From the Ch70 website:",
        "\"Chapter 70 District Profiles: The on-line Chapter 70 database shows, for each school district, yearly spending and state aid totals in comparison to the foundation budget. Trend data is available for each year going back to FY1993.\"",
        "",
        "This dataset provides:",
        "• Chapter 70 Aid (c70aid): State aid received by each district",
        "• Required Net School Spending (rqdnss2): Minimum spending required by state law",
        "• Actual Net School Spending (actualNSS): Total district spending on education",
        "• Foundation Enrollment (distfoundenro): District foundation enrollment used for Ch70 funding calculations",
        "",
        "<b>Regional Classifications:</b>",
        "Source: DESE district profiles and regional service mappings",
        "• Districts classified as Western MA based on EOHHS regional designations",
        "• School type classifications (Traditional, Regional, Charter, etc.)",
    ]

    pages.append(dict(
        title="Appendix B. Calculation Methodology",
        subtitle="Data Sources",
        chart_path=None,
        graph_only=True,
        text_blocks=methodology_page4,
        section_id="appendix_b"
    ))

    # Add second methodology page (formulas and district memberships, now #2-4)
    pages.append(dict(
        title="Appendix B. Calculation Methodology (continued)",
        subtitle="",
        chart_path=None,
        graph_only=True,
        text_blocks=methodology_page1
    ))

    # Add third methodology page (continuation)
    pages.append(dict(
        title="Appendix B. Calculation Methodology (continued)",
        subtitle="",
        chart_path=None,
        graph_only=True,
        text_blocks=methodology_page2
    ))

    # Add fourth methodology page (NSS/Ch70)
    pages.append(dict(
        title="Appendix B. Calculation Methodology (continued)",
        subtitle="",
        chart_path=None,
        graph_only=True,
        text_blocks=methodology_page3
    ))

    return pages

# ---- Table of Contents ----
def build_toc_page():
    """Build table of contents page dict."""
    toc_entries = [
        ("Section 1: Western MA", "section1_western"),
        ("Section 2: Amherst-Pelham Regional", "amherst_pelham"),
        ("Section 2: Amherst", "amherst"),
        ("Section 2: Leverett", "leverett"),
        ("Section 2: Pelham", "pelham"),
        ("Section 2: Shutesbury", "shutesbury"),
        ("Appendix A. Data Tables", "appendix_a"),
        ("Appendix B. Calculation Methodology", "appendix_b"),
    ]

    return dict(
        title="Table of Contents",
        subtitle="",
        chart_path=None,
        page_type="toc",
        toc_entries=toc_entries
    )

# ---- Build PDF ----
def build_pdf(pages: List[dict], out_path: Path):
    doc = SimpleDocTemplate(str(out_path), pagesize=A4,
        leftMargin=0.5*inch, rightMargin=0.5*inch, topMargin=0.5*inch, bottomMargin=0.5*inch)

    story: List = []
    for idx, p in enumerate(pages):
        # Handle TOC page
        if p.get("page_type") == "toc":
            story.append(Paragraph(p["title"], style_title_main))
            story.append(Spacer(0, 12))

            # Build TOC entries as clickable links
            toc_entries = p.get("toc_entries", [])
            for entry_title, entry_id in toc_entries:
                # Create clickable link to section
                link_text = f'<a href="#{entry_id}" color="blue">{entry_title}</a>'
                story.append(Paragraph(link_text, style_body))
                story.append(Spacer(0, 8))

            if idx < len(pages)-1:
                story.append(PageBreak())
            continue

        # Add appendix title if present
        if p.get("appendix_title"):
            appendix_title = p["appendix_title"]
            # Add anchor if this page has a section_id
            if p.get("section_id"):
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
            title_text = f'<a name="{p["section_id"]}"/>{title_text}'
        story.append(Paragraph(title_text, style_title_main))
        default_sub = "PPE vs Enrollment"
        story.append(Paragraph(p.get("subtitle", (default_sub if not p.get("graph_only") else "")), style_title_sub))

        # Handle single chart or multiple charts
        chart_path = p.get("chart_path")
        chart_paths = p.get("chart_paths", [])

        if chart_path:
            chart_paths = [chart_path]

        if chart_paths:
            # Two charts vertical: split page height between them
            if p.get("two_charts_vertical") and len(chart_paths) == 2:
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
                            # Western overview gets 70% of page for breathing room, others 62%
                            if "ppe_overview_all_western" in name:
                                max_chart_h = doc.height * 0.70
                            else:
                                max_chart_h = doc.height * 0.62
                        else:
                            max_chart_h = doc.height * 0.40
                        if im.drawHeight > max_chart_h:
                            im.drawHeight = max_chart_h; im.drawWidth = im.drawHeight / ratio
                        story.append(im)

        # Add text blocks (for graph_only pages, add after image)
        if p.get("graph_only"):
            text_blocks = p.get("text_blocks", []) or []
            if text_blocks:
                story.append(Spacer(0, 12))

                # Build paragraphs for text content
                text_content = []
                for block in text_blocks:
                    text_content.append(Paragraph(block, style_body))
                    text_content.append(Spacer(0, 6))

                # Calculate available height for content on this page
                # Must account for:
                # - Title (~22 pt + 2 pt space = ~24 pt = 0.33 inch)
                # - Subtitle (~14 pt + 6 pt space = ~20 pt = 0.28 inch)
                # - Spacer before content (12 pt = 0.17 inch)
                # - Footer buffer (footer at 0.6 inch, need ~0.3 inch clearance above)
                # Total overhead: ~1.1 inches
                # Use conservative 1.5 inch buffer for title + subtitle + footer
                title_and_footer_buffer = 1.5 * inch

                # Available height = doc.height minus title/subtitle/footer
                available_height = doc.height - title_and_footer_buffer

                # Use KeepInFrame to constrain content within available height
                # If content exceeds height, KeepInFrame will shrink or allow overflow
                # with mode='shrink' it shrinks to fit; with mode='truncate' it clips
                # Use mode='error' to raise error if too tall, forcing us to fix it
                frame_content = KeepInFrame(
                    maxWidth=doc.width,
                    maxHeight=available_height,
                    content=text_content,
                    mode='shrink',  # Shrink content if it doesn't fit
                    name='methodology_content'
                )
                story.append(frame_content)

            if idx < len(pages)-1: story.append(PageBreak())
            continue

        # Scatterplot district table: compact table with cohort colors
        if p.get("scatterplot_districts"):
            story.append(Spacer(0, 12))
            scatterplot_table = _build_scatterplot_table(p.get("scatterplot_districts"), doc.width, style_body, style_num)
            if scatterplot_table:
                story.append(scatterplot_table)
            if idx < len(pages)-1: story.append(PageBreak())
            continue

        # Data table pages: show raw data tables
        if p.get("page_type") == "data_table":
            story.append(Spacer(0, 10))
            story.append(Paragraph("PPE by Category ($/pupil)", style_body))
            story.append(Spacer(0, 6))
            epp_table = _build_epp_data_table(p.get("raw_epp"), p.get("dist_name", ""), doc.width)
            if epp_table:
                story.append(epp_table)
            story.append(Spacer(0, 12))
            story.append(Paragraph("FTE Enrollment", style_body))
            story.append(Spacer(0, 6))
            fte_table = _build_fte_data_table(p.get("raw_lines", {}), p.get("dist_name", ""), doc.width)
            if fte_table:
                story.append(fte_table)

            # Add NSS/Ch70 data table if available
            raw_nss = p.get("raw_nss")
            if raw_nss is not None and not raw_nss.empty:
                story.append(Spacer(0, 12))
                story.append(Paragraph("NSS/Ch70 Funding Components ($)", style_body))
                story.append(Spacer(0, 6))
                nss_table = _build_nss_ch70_data_table(raw_nss, p.get("dist_name", ""), doc.width)
                if nss_table:
                    story.append(nss_table)

            if idx < len(pages)-1:
                story.append(PageBreak())
            continue

        # NSS/Ch70 pages: show funding component table and FTE table (foundation enrollment only)
        if p.get("page_type") == "nss_ch70":
            story.append(Spacer(0, 6))
            story.append(_build_nss_ch70_table(p))
            story.append(Spacer(0, 6))
            story.append(_build_fte_table(p))

            if idx < len(pages)-1:
                story.append(PageBreak())
            continue

        # Regular district/regional pages: show summary tables
        story.append(Spacer(0, 6))
        story.append(_build_category_table(p))
        story.append(Spacer(0, 6))
        story.append(_build_fte_table(p))

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

    build_pdf(pages, OUTPUT_DIR / "expenditures_series.pdf")

if __name__ == "__main__":
    main()
