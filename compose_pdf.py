"""
PDF Report Generator for School District Expenditure Analysis

This module creates a comprehensive PDF report with:
- Page 1: All Western MA districts overview (horizontal bars)
- Page 2: ALPS & Peers comparison (horizontal bars with enrollment annotations)
- District pages: Simple and detailed views with category breakdowns
- Appendices: Aggregate comparisons, data tables, methodology

Key Design Patterns:
- All comparative plots use horizontal bars for better PDF layout
- Dynamic heights scale with number of districts
- Enrollment changes shown as annotations (boxes to right of bars)
- Aggregates visually separated at bottom of charts
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
    add_alps_pk12,
    FTE_LINE_COLORS,
    mean_clean, get_latest_year,
    weighted_epp_aggregation,
)

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
RED_SHADES = ["#FDE2E2", "#FAD1D1", "#F5B8B8", "#EF9E9E", "#E88080"]  # higher than Western (lightest→darkest)
GRN_SHADES = ["#E6F4EA", "#D5EDE0", "#C4E6D7", "#B3DFCD", "#A1D8C4"]  # lower than Western (lightest→darkest)

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
style_title_main = ParagraphStyle("title_main", parent=styles["Heading1"], fontSize=18, leading=22, spaceAfter=2)
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
    y1 = 0.6 * inch; y2 = 0.44 * inch
    canvas.drawString(doc.leftMargin, y1, SOURCE_LINE1)
    canvas.drawString(doc.leftMargin, y2, SOURCE_LINE2)
    x_right = doc.pagesize[0] - doc.rightMargin
    canvas.drawRightString(x_right, y1, f"Page {canvas.getPageNumber()}")
    # Removed code version from footer to avoid overflow with source text
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

    # Handle zero or negative values
    if v0 <= 0 or v1 <= 0:
        return float("nan")

    return (v1 / v0) ** (1.0 / years) - 1.0

def _abbr_bucket_suffix(full: str) -> str:
    if re.search(r"≤\s*500", full or "", flags=re.I): return "(≤500)"
    if re.search(r">\s*500", full or "", flags=re.I): return "(>500)"
    return ""

def _shade_for_cagr_delta(delta_pp: float):
    # delta_pp is a true fraction (e.g., 0.021 = 2.1pp)
    if delta_pp != delta_pp or abs(delta_pp) < MATERIAL_DELTA_PCTPTS:
        return None
    idx = max(0, min(len(RED_SHADES)-1, bisect.bisect_right(SHADE_BINS, abs(delta_pp)) - 1))
    return HexColor(RED_SHADES[idx]) if delta_pp > 0 else HexColor(GRN_SHADES[idx])

def _shade_for_dollar_rel(delta_rel: float):
    # delta_rel is a relative diff (e.g., +0.03 = +3%)
    if delta_rel != delta_rel or abs(delta_rel) < DOLLAR_THRESHOLD_REL:
        return None
    idx = max(0, min(len(RED_SHADES)-1, bisect.bisect_right(SHADE_BINS, abs(delta_rel)) - 1))
    return HexColor(RED_SHADES[idx]) if delta_rel > 0 else HexColor(GRN_SHADES[idx])

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

    # Legend rows: left explanations + right swatches spanning the three CAGR columns
    # NOTE: Legend now spans columns 3-5 (CAGR 15y, 10y, 5y) in new layout
    legend_rows = []
    if page.get("page_type") == "district":
        baseline_title = page.get("baseline_title", "")

        # Determine if comparing to ALPS Peers or Western
        if "ALPS Peer" in baseline_title:
            comparison_label = "ALPS Peers"
            bucket_suffix = ""
        else:
            comparison_label = "Western"
            bucket_suffix = _abbr_bucket_suffix(baseline_title)

        red_text = f"&gt; {comparison_label} CAGR {bucket_suffix}".strip()
        grn_text = f"&lt; {comparison_label} CAGR {bucket_suffix}".strip()
        shade_rule = (
            f"Shading vs {comparison_label}: |Δ$/pupil| ≥ {DOLLAR_THRESHOLD_REL*100:.1f}% "
            f"or |ΔCAGR| ≥ {MATERIAL_DELTA_PCTPTS*100:.1f}pp"
        )
        cagr_def   = "CAGR = (End/Start)^(1/years) − 1"
        legend_rows = [
            ["", Paragraph(shade_rule, style_legend_right), "", Paragraph(red_text, style_legend_center), "", "", ""],
            ["", Paragraph(cagr_def,   style_legend_right), "", Paragraph(grn_text, style_legend_center), "", "", ""],
        ]
    data.extend(legend_rows)
    leg1_idx = total_row_idx + 1 if legend_rows else None
    leg2_idx = total_row_idx + 2 if legend_rows else None

    # Column widths: narrower for $/pupil and CAGR columns to prevent overflow
    # Swatch, Category (flex), t0 $, CAGR15y, CAGR10y, CAGR5y, Latest $
    tbl = Table(data, colWidths=[0.22*inch, None, 0.95*inch, 0.85*inch, 0.85*inch, 0.85*inch, 0.95*inch])

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
    # Legend text spans cols 1-2, swatch spans cols 3-5 (CAGR columns)
    if leg1_idx is not None:
        ts.add("SPAN", (1, leg1_idx), (2, leg1_idx))
        ts.add("SPAN", (1, leg2_idx), (2, leg2_idx))
        ts.add("SPAN", (3, leg1_idx), (5, leg1_idx))
        ts.add("SPAN", (3, leg2_idx), (5, leg2_idx))
        sw_red = HexColor(RED_SHADES[1]); sw_grn = HexColor(GRN_SHADES[1])
        ts.add("BACKGROUND", (3, leg1_idx), (5, leg1_idx), sw_red)
        ts.add("BACKGROUND", (3, leg2_idx), (5, leg2_idx), sw_grn)

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
    t0_year_fte = latest_year_fte - 5 if isinstance(latest_year_fte, int) else 2019

    # Header cells - reordered chronologically
    header = [
        Paragraph("", style_hdr_left),  # Swatch column, no header text
        Paragraph("FTE Series", style_hdr_left),
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

    # Total FTE (sum series) - reordered
    total_series = None
    for (_c, label, _ls, _r5, _r10, _r15) in rows_in:
        s = page.get("fte_series_map", {}).get(label)
        if s is None: continue
        total_series = s if total_series is None else (total_series.add(s, fill_value=0.0))
    if total_series is not None and not total_series.empty:
        ly = int(total_series.index.max())
        latest_str = f"{float(total_series.loc[ly]):,.0f}" if ly in total_series.index else "—"
        t0_str = f"{float(total_series.loc[t0_year_fte]):,.0f}" if t0_year_fte in total_series.index else "—"
        r5 = fmt_pct(compute_cagr_last(total_series,5))
        r10= fmt_pct(compute_cagr_last(total_series,10))
        r15= fmt_pct(compute_cagr_last(total_series,15))
        data.append(["", Paragraph("Total FTE", style_body),
                     Paragraph(t0_str, style_num),
                     Paragraph(r15, style_num),
                     Paragraph(r10, style_num),
                     Paragraph(r5, style_num),
                     Paragraph(latest_str, style_num)])
        total_row_idx = len(data) - 1
    else:
        total_row_idx = None

    # Column widths: narrower for FTE and CAGR columns to prevent overflow
    # Swatch, Series (flex), t0 FTE, CAGR15y, CAGR10y, CAGR5y, Latest FTE
    tbl = Table(data, colWidths=[0.45*inch, None, 0.85*inch, 0.85*inch, 0.85*inch, 0.85*inch, 0.85*inch])
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

# ---- Data table builders for appendix ----
def _build_epp_data_table(epp_pivot: pd.DataFrame, title: str, doc_width: float) -> Table:
    """Build a data table showing all EPP data (years as columns, categories as rows)."""
    if epp_pivot.empty:
        return None

    # Get years and categories in order
    years = sorted(epp_pivot.index.tolist())
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

    # Collect all years across all series
    all_years = set()
    for s in lines.values():
        if s is not None and not s.empty:
            all_years.update(s.index.tolist())

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

    if total_series is not None and not total_series.empty:
        total_row = [Paragraph("Total FTE", style_data_label)]
        for yr in years:
            val = total_series.loc[yr] if yr in total_series.index else np.nan
            val_str = f"{val:,.0f}" if not np.isnan(val) else "—"
            total_row.append(Paragraph(val_str, style_data_cell))
        data.append(total_row)
        total_row_idx = len(data) - 1
    else:
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

# ---- Page dicts ----
def build_page_dicts(df: pd.DataFrame, reg: pd.DataFrame) -> List[dict]:
    pages: List[dict] = []

    latest = int(df["YEAR"].max())
    t0 = latest - 5

    # PAGE 1: All Western MA Districts Overview (graph-only)
    # NOTE: Horizontal bar chart with district names on y-axis, sorted by 2024 PPE
    western_explanation = ("All Western MA Traditional Districts: Each bar represents one district's per-pupil expenditures (PPE). "
                          "Districts are sorted by 2024 PPE (lowest to highest).")

    pages.append(dict(
        title="All Western MA Traditional Districts: PPE Overview 2019 -> 2024",
        subtitle=f"Bars show {t0} PPE with darker segment to {latest} (purple = decrease).",
        chart_path=str(OUTPUT_DIR / "ppe_overview_all_western.png"),
        text_blocks=[western_explanation],
        graph_only=True,
        section_id="western_overview"
    ))

    # PAGE 2: ALPS + Peers (graph-only)
    # NOTE: Horizontal bar chart with peer districts, aggregate groups separated at bottom
    peer_explanation = ("ALPS Peers: Weighted aggregate of ALPS PK-12 and 9 comparison districts "
                       "to benchmark against similar PK-12 systems in Western MA. Aggregate groups shown at bottom.")
    pages.append(dict(
        title="ALPS PK-12 & Peers: PPE and Enrollment 2019 -> 2024",
        subtitle=f"Bars show {t0} PPE with darker segment to {latest} (purple = decrease). Enrollment change shown to right of bars.",
        chart_path=str(OUTPUT_DIR / "ppe_change_bars_ALPS_and_peers.png"),
        text_blocks=[peer_explanation],
        graph_only=True,
        section_id="alps_peers"
    ))

    cmap_all = create_or_load_color_map(df)

    # PAGE 2: ALPS - Simple version (solid color, no tables)
    epp_alps, lines_alps = prepare_district_epp_lines(df, "ALPS PK-12")
    if not epp_alps.empty or lines_alps:
        pages.append(dict(
            title="ALPS PK-12: PPE vs Enrollment",
            subtitle="Total per-pupil expenditures; black/red lines show in-/out-of-district FTE trend (ALPS uses its own FTE scale).",
            chart_path=str(district_png_simple("ALPS PK-12")),
            graph_only=True,
            section_id="alps_pk12"
        ))

    # PAGE 3: ALPS - Detailed version with tables
    if not epp_alps.empty or lines_alps:
        latest_year = get_latest_year(df, epp_alps)
        context = context_for_district(df, "ALPS PK-12")

        cat_rows, cat_total, cat_start_map = _build_category_data(epp_alps, latest_year, context, cmap_all)
        fte_rows, fte_map, latest_fte_year = _build_fte_data(lines_alps, latest_year)

        # Western baseline (same bucket); add baseline $ for latest_year
        bucket = "le_500" if context == "SMALL" else "gt_500"
        title_w, epp_w, _ls, _lm = prepare_western_epp_lines(df, reg, bucket)
        base_map = {}
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
            title="ALPS PK-12: PPE vs Enrollment (vs Western MA Districts)",
            subtitle="Stacked per-pupil expenditures by category; black/red lines show in-/out-of-district FTE trend (ALPS uses its own FTE scale).",
            chart_path=str(district_png_detail("ALPS PK-12")),
            latest_year=latest_year, latest_year_fte=latest_fte_year,
            cat_rows=cat_rows, cat_total=cat_total, cat_start_map=cat_start_map, fte_rows=fte_rows,
            fte_series_map=fte_map,
            page_type="district", baseline_title=title_w, baseline_map=base_map,
            raw_epp=epp_alps, raw_lines=lines_alps, dist_name="ALPS PK-12"
        ))

        # ALPS third page: comparison to ALPS Peer Districts
        alps_peers = ["ALPS PK-12", "Greenfield", "Easthampton", "South Hadley", "Northampton",
                      "East Longmeadow", "Longmeadow", "Agawam", "Hadley", "Hampden-Wilbraham"]
        peers_epp_temp, peers_enr_in_temp, peers_enr_out_temp = weighted_epp_aggregation(df, alps_peers)
        peer_base_map = {}
        if not peers_epp_temp.empty:
            start_year = latest_year - 15  # 15 years before latest for START_DOLLAR
            for sc in peers_epp_temp.columns:
                s = peers_epp_temp[sc]
                peer_base_map[sc] = {
                    "5": compute_cagr_last(s, 5),
                    "10": compute_cagr_last(s, 10),
                    "15": compute_cagr_last(s, 15),
                    "DOLLAR": (float(s.loc[latest_year]) if latest_year in s.index else float("nan")),
                    "START_DOLLAR": (float(s.loc[start_year]) if start_year in s.index else float("nan")),
                }

        pages.append(dict(
            title="ALPS PK-12: PPE vs Enrollment (vs ALPS Peers)",
            subtitle="Comparison to ALPS Peer Districts aggregate",
            chart_path=str(district_png_detail("ALPS PK-12")),
            latest_year=latest_year, latest_year_fte=latest_fte_year,
            cat_rows=cat_rows, cat_total=cat_total, cat_start_map=cat_start_map, fte_rows=fte_rows,
            fte_series_map=fte_map,
            page_type="district",
            baseline_title="ALPS Peer Districts Aggregate",
            baseline_map=peer_base_map,
            raw_epp=epp_alps, raw_lines=lines_alps, dist_name="ALPS PK-12"
        ))

    # DISTRICT PAGES - Three pages per district (simple + detailed vs Western + detailed vs Peers)
    for dist in ["Amherst-Pelham"] + [d for d in DISTRICTS_OF_INTEREST if d != "Amherst-Pelham"]:
        epp, lines = prepare_district_epp_lines(df, dist)
        if epp.empty and not lines:
            continue

        dist_title = f"Amherst-Pelham Regional" if dist == "Amherst-Pelham" else dist
        # Create section_id from district name (lowercase, replace spaces with underscores)
        section_id = dist.lower().replace(" ", "_").replace("-", "_")

        # Simple version (solid color, no tables)
        pages.append(dict(
            title=f"{dist_title}: PPE vs Enrollment",
            subtitle="Total per-pupil expenditures; black/red lines show in-/out-of-district FTE trend.",
            chart_path=str(district_png_simple(dist)),
            graph_only=True,
            section_id=section_id
        ))

        # Detailed version with tables
        latest_year = get_latest_year(df, epp)
        context = context_for_district(df, dist)

        rows, total, start_map = _build_category_data(epp, latest_year, context, cmap_all)
        fte_rows, fte_map, latest_fte_year = _build_fte_data(lines, latest_year)

        bucket = "le_500" if context == "SMALL" else "gt_500"
        base_title = f"All Western MA Traditional Districts {'≤500' if bucket=='le_500' else '>500'} Students"
        base_map = {}
        title_w, epp_w, _ls, _lm = prepare_western_epp_lines(df, reg, bucket)
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
            title=f"{dist_title}: PPE vs Enrollment (vs Western MA Districts)",
            subtitle="Stacked per-pupil expenditures by category; black/red lines show in-/out-of-district FTE trend.",
            chart_path=str(district_png_detail(dist)),
            latest_year=latest_year, latest_year_fte=latest_fte_year,
            cat_rows=rows, cat_total=total, cat_start_map=start_map, fte_rows=fte_rows,
            fte_series_map=fte_map,
            page_type="district", baseline_title=base_title, baseline_map=base_map,
            raw_epp=epp, raw_lines=lines, dist_name=dist
        ))

        # Third page: comparison to ALPS Peer Districts
        alps_peers = ["ALPS PK-12", "Greenfield", "Easthampton", "South Hadley", "Northampton",
                      "East Longmeadow", "Longmeadow", "Agawam", "Hadley", "Hampden-Wilbraham"]
        peers_epp_temp, peers_enr_in_temp, peers_enr_out_temp = weighted_epp_aggregation(df, alps_peers)
        peer_base_map = {}
        if not peers_epp_temp.empty:
            start_year = latest_year - 15  # 15 years before latest for START_DOLLAR
            for sc in peers_epp_temp.columns:
                s = peers_epp_temp[sc]
                peer_base_map[sc] = {
                    "5": compute_cagr_last(s, 5),
                    "10": compute_cagr_last(s, 10),
                    "15": compute_cagr_last(s, 15),
                    "DOLLAR": (float(s.loc[latest_year]) if latest_year in s.index else float("nan")),
                    "START_DOLLAR": (float(s.loc[start_year]) if start_year in s.index else float("nan")),
                }

        pages.append(dict(
            title=f"{dist_title}: PPE vs Enrollment (vs ALPS Peers)",
            subtitle="Comparison to ALPS Peer Districts aggregate",
            chart_path=str(district_png_detail(dist)),
            latest_year=latest_year, latest_year_fte=latest_fte_year,
            cat_rows=rows, cat_total=total, cat_start_map=start_map, fte_rows=fte_rows,
            fte_series_map=fte_map,
            page_type="district",
            baseline_title="ALPS Peer Districts Aggregate",
            baseline_map=peer_base_map,
            raw_epp=epp, raw_lines=lines, dist_name=dist
        ))

    # APPENDIX A - first page will include intro text
    first_appendix_a = True
    for bucket in ("le_500", "gt_500"):
        title, epp, lines_sum, lines_mean = prepare_western_epp_lines(df, reg, bucket)
        if epp.empty and not lines_sum:
            continue

        latest_year = get_latest_year(df, epp)
        context = context_for_western(bucket)

        rows, total, start_map = _build_category_data(epp, latest_year, context, cmap_all)
        fte_rows, fte_map, latest_fte_year = _build_fte_data(lines_mean, latest_year)

        page_dict = dict(title=title,
                        subtitle="PPE vs Enrollment — Not including charters and vocationals",
                        chart_path=str(regional_png(bucket)),
                        latest_year=latest_year, latest_year_fte=latest_fte_year,
                        cat_rows=rows, cat_total=total, cat_start_map=start_map, fte_rows=fte_rows,
                        fte_series_map=fte_map, page_type="western",
                        raw_epp=epp, raw_lines=lines_mean, dist_name=title)

        if first_appendix_a:
            page_dict["appendix_title"] = "Appendix A. Aggregate Districts for Comparison"
            page_dict["appendix_subtitle"] = "(Western MA Traditional by enrollment size, plus ALPS Peer group)"
            page_dict["section_id"] = "appendix_a"
            first_appendix_a = False

        pages.append(page_dict)

    # Add ALPS Peer Districts aggregate to Appendix A
    alps_peers = ["ALPS PK-12", "Greenfield", "Easthampton", "South Hadley", "Northampton",
                  "East Longmeadow", "Longmeadow", "Agawam", "Hadley", "Hampden-Wilbraham"]
    peers_epp, peers_enr_in, peers_enr_out = weighted_epp_aggregation(df, alps_peers)
    if not peers_epp.empty:
        latest_year = get_latest_year(df, peers_epp)
        context = "LARGE"  # Peer group is large enrollment

        rows, total, start_map = _build_category_data(peers_epp, latest_year, context, cmap_all)

        # Create FTE data from enrollment sums (both in-district and out-of-district)
        fte_map = {
            "In-District FTE Pupils": peers_enr_in,
            "Out-of-District FTE Pupils": peers_enr_out
        }
        fte_rows = []
        if not peers_enr_in.empty:
            r5 = compute_cagr_last(peers_enr_in, 5)
            r10 = compute_cagr_last(peers_enr_in, 10)
            r15 = compute_cagr_last(peers_enr_in, 15)
            latest_str = "—" if latest_year not in peers_enr_in.index else f"{float(peers_enr_in.loc[latest_year]):,.0f}"
            fte_rows.append((FTE_LINE_COLORS["In-District FTE Pupils"], "In-District FTE Pupils", latest_str, fmt_pct(r5), fmt_pct(r10), fmt_pct(r15)))
        if not peers_enr_out.empty:
            r5 = compute_cagr_last(peers_enr_out, 5)
            r10 = compute_cagr_last(peers_enr_out, 10)
            r15 = compute_cagr_last(peers_enr_out, 15)
            latest_str = "—" if latest_year not in peers_enr_out.index else f"{float(peers_enr_out.loc[latest_year]):,.0f}"
            fte_rows.append((FTE_LINE_COLORS["Out-of-District FTE Pupils"], "Out-of-District FTE Pupils", latest_str, fmt_pct(r5), fmt_pct(r10), fmt_pct(r15)))
        latest_fte_year = latest_year

        pages.append(dict(
            title="ALPS Peer Districts Aggregate",
            subtitle="Weighted aggregate of ALPS PK-12 and 9 peer districts",
            chart_path=str(OUTPUT_DIR / "regional_expenditures_per_pupil_ALPS_Peers_Aggregate.png"),
            latest_year=latest_year, latest_year_fte=latest_fte_year,
            cat_rows=rows, cat_total=total, cat_start_map=start_map, fte_rows=fte_rows,
            fte_series_map=fte_map,
            page_type="western",
            raw_epp=peers_epp, raw_lines=fte_map, dist_name="ALPS Peer Districts Aggregate"
        ))

    # APPENDIX B: Data Tables - first page includes intro
    # Deduplicate by dist_name to avoid duplicate data tables
    data_pages_to_add = []
    seen_districts = set()
    for p in pages:
        if p.get("raw_epp") is not None and not p["raw_epp"].empty:
            dist_name = p.get("dist_name", "Unknown")
            if dist_name not in seen_districts:
                data_pages_to_add.append(p)
                seen_districts.add(dist_name)

    first_data_table = True
    for p in data_pages_to_add:
        dist_name = p.get("dist_name", "Unknown")
        raw_epp = p.get("raw_epp")
        raw_lines = p.get("raw_lines", {})

        page_dict = dict(
            title=f"Data: {dist_name}",
            subtitle="PPE ($/pupil) and FTE Enrollment",
            chart_path=None,
            page_type="data_table",
            raw_epp=raw_epp,
            raw_lines=raw_lines,
            dist_name=dist_name
        )

        if first_data_table:
            page_dict["appendix_title"] = "Appendix B. Data Tables"
            page_dict["appendix_subtitle"] = "All data values used in plots"
            page_dict["appendix_note"] = ("This appendix contains the underlying data tables for all districts and regions shown in the report. "
                                        "Each table shows ppe by category (in dollars) and FTE enrollment counts across all available years.")
            page_dict["section_id"] = "appendix_b"
            first_data_table = False

        pages.append(page_dict)

    # APPENDIX C: Calculation Methodology
    from school_shared import ALPS_COMPONENTS
    alps_list = sorted([d.title() for d in ALPS_COMPONENTS])
    pk12_districts = ["ALPS PK-12", "Easthampton", "Longmeadow", "Hampden-Wilbraham",
                      "East Longmeadow", "South Hadley", "Agawam", "Northampton",
                      "Greenfield", "Hadley"]

    # Get Western MA districts
    mask = (reg["EOHHS_REGION"].str.lower() == "western") & (reg["SCHOOL_TYPE"].str.lower() == "traditional")
    western_districts = sorted(reg[mask]["DIST_NAME"].unique())
    western_le500 = []
    western_gt500 = []

    for dist in western_districts:
        dist_enr = df[
            (df["DIST_NAME"].str.lower() == dist.lower()) &
            (df["IND_CAT"].str.lower() == "student enrollment") &
            (df["IND_SUBCAT"].str.lower() == "in-district fte pupils") &
            (df["YEAR"] == latest)
        ]["IND_VALUE"]
        if not dist_enr.empty:
            enr = float(dist_enr.iloc[0])
            if enr <= 500:
                western_le500.append(dist)
            else:
                western_gt500.append(dist)

    # Split methodology into two pages to avoid footer overlap
    methodology_page1 = [
        "<b>1. Compound Annual Growth Rate (CAGR)</b>",
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
        "<b>2. Aggregate District Calculations</b>",
        "",
        "<b>ALPS PK-12 Aggregate:</b> Weighted aggregation of member districts using enrollment-weighted per-pupil expenditures.",
        f"<b>Member districts:</b> {', '.join(alps_list)}",
        "",
        "<b>Weighted EPP Formula:</b>",
        "For each category and year: Weighted_EPP = Σ(District_EPP × District_In-District_FTE) / Σ(District_In-District_FTE)",
        "",
        "<b>Example:</b> District A spends $5,000/pupil (500 students), District B spends $6,000/pupil (300 students)",
        "Weighted average = ($5,000×500 + $6,000×300) / (500+300) = $5,375/pupil",
        "",
        "<b>Enrollment Calculation:</b> For each series (In-District FTE, Out-of-District FTE) and year: Sum across all member districts.",
        "",
        "<b>PK-12 District Aggregate:</b> Weighted aggregation of selected peer districts (includes ALPS PK-12).",
        f"<b>Member districts:</b> {', '.join(pk12_districts)}",
        "<i>Uses same weighted calculation method as ALPS PK-12.</i>",
        "",
        f"<b>All Western MA Traditional Districts (≤500 students):</b> Weighted aggregation of {len(western_le500)} districts.",
        f"<b>Member districts:</b> {', '.join(western_le500) if len(western_le500) <= 15 else ', '.join(western_le500[:15]) + f', and {len(western_le500)-15} others'}",
        "<i>Uses same weighted calculation method as ALPS PK-12.</i>",
        "",
        f"<b>All Western MA Traditional Districts (&gt;500 students):</b> Weighted aggregation of {len(western_gt500)} districts.",
        f"<b>Member districts:</b> {', '.join(western_gt500)}",
        "<i>Uses same weighted calculation method as ALPS PK-12.</i>",
    ]

    methodology_page2 = [
        "<b>3. Red/Green Shading Logic (District Comparison Tables)</b>",
        "",
        "District pages include tables comparing each district's per-pupil expenditures (PPE) and growth rates (CAGR) to baseline aggregates (Western MA or ALPS Peers).",
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

    # Add first methodology page
    pages.append(dict(
        title="Appendix C. Calculation Methodology",
        subtitle="Formulas and district memberships",
        chart_path=None,
        graph_only=True,
        text_blocks=methodology_page1,
        section_id="appendix_c"
    ))

    # Add second methodology page (continuation)
    pages.append(dict(
        title="Appendix C. Calculation Methodology (continued)",
        subtitle="",
        chart_path=None,
        graph_only=True,
        text_blocks=methodology_page2
    ))

    return pages

# ---- Table of Contents ----
def build_toc_page():
    """Build table of contents page dict."""
    toc_entries = [
        ("All Western MA Traditional Districts: PPE Overview 2019 -> 2024", "western_overview"),
        ("ALPS PK-12 & Peers: PPE and Enrollment 2019 -> 2024", "alps_peers"),
        ("ALPS PK-12", "alps_pk12"),
        ("Amherst-Pelham Regional", "amherst_pelham"),
        ("Amherst", "amherst"),
        ("Leverett", "leverett"),
        ("Pelham", "pelham"),
        ("Shutesbury", "shutesbury"),
        ("Appendix A. Aggregate Districts for Comparison", "appendix_a"),
        ("Appendix B. Data Tables", "appendix_b"),
        ("Appendix C. Calculation Methodology", "appendix_c"),
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

        chart_path = p.get("chart_path")
        if chart_path:
            img_path = Path(chart_path)
            if not img_path.exists():
                story.append(Paragraph(f"[Missing chart image: {img_path.name}]", style_body))
            else:
                if p.get("graph_only") and "ppe_change_bars_ALPS_and_peers" in img_path.name:
                    story.append(Spacer(0, 12))
                im = Image(str(img_path))
                ratio = im.imageHeight / float(im.imageWidth)
                im.drawWidth = doc.width; im.drawHeight = doc.width * ratio
                if p.get("graph_only"):
                    name = img_path.name.lower()
                    # Western overview gets 70% of page for breathing room, ALPS peers 80%, others 62%
                    if "ppe_overview_all_western" in name:
                        max_chart_h = doc.height * 0.70
                    elif "ppe_change_bars_alps_and_peers" in name:
                        max_chart_h = doc.height * 0.80
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
    df = add_alps_pk12(df)
    pages = build_page_dicts(df, reg)
    if not pages:
        print("[WARN] No pages to write."); return

    # Insert TOC at the beginning
    toc_page = build_toc_page()
    pages.insert(0, toc_page)

    build_pdf(pages, OUTPUT_DIR / "expenditures_series.pdf")

if __name__ == "__main__":
    main()
