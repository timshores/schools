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
from reportlab.platypus import Flowable, Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from school_shared import (
    OUTPUT_DIR, load_data, create_or_load_color_map, color_for,
    ENROLL_KEYS, DISTRICTS_OF_INTEREST,
    context_for_district, prepare_district_epp_lines, prepare_western_epp_lines, context_for_western,
    canonical_order_bottom_to_top,
    add_alps_pk12,
    FTE_LINE_COLORS,
    mean_clean, get_latest_year,
)

# ===== code version =====
CODE_VERSION = "v2025.09.29-REFACTORED"

# ---- Shading controls ----
# CAGR shading threshold is in absolute percentage points (e.g., 0.02 = 2.0pp)
MATERIAL_DELTA_PCTPTS = 0.02
# Dollar shading threshold is in relative percent difference (e.g., 0.02 = 2.0%)
DOLLAR_THRESHOLD_REL  = 0.02

# Bins for intensity (used for both CAGR pp-delta and $ relative delta)
SHADE_BINS = [0.02, 0.05, 0.08, 0.12]
RED_SHADES = ["#FDE2E2", "#FAD1D1", "#F5B8B8", "#EF9E9E", "#E88080"]  # higher than Western
GRN_SHADES = ["#E6F4EA", "#D5EDE0", "#C4E6D7", "#B3DFCD", "#A1D8C4"]  # lower than Western

# Optional $-gate; if >0, shading applies only when latest $/pupil >= this
MATERIAL_MIN_LATEST_DOLLARS = 0.0

def district_png(dist: str) -> Path:
    return OUTPUT_DIR / f"expenditures_per_pupil_vs_enrollment_{dist.replace(' ', '_')}.png"

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
    rows_in = page.get("cat_rows", [])
    total   = page.get("cat_total", ("Total","—","—","—","—"))
    base    = page.get("baseline_map", {}) or {}

    # Header cells
    header = [
        Paragraph("", style_hdr_left),
        Paragraph("Category", style_hdr_left),
        Paragraph(f"{page.get('latest_year','Latest')} $/pupil", style_hdr_right),
        Paragraph("CAGR 5y", style_hdr_right),
        Paragraph("CAGR 10y", style_hdr_right),
        Paragraph("CAGR 15y", style_hdr_right),
    ]
    data: List[List] = [header]

    # Category rows
    for (sc, latest_str, c5s, c10s, c15s, col, latest_val) in rows_in:
        data.append([
            "",  # swatch painted via TableStyle
            Paragraph(sc, style_body),
            Paragraph(latest_str, style_num),
            Paragraph(c5s, style_num),
            Paragraph(c10s, style_num),
            Paragraph(c15s, style_num),
        ])

    # TOTAL row directly after categories
    data.append([
        "", Paragraph(total[0], style_body),
        Paragraph(total[1], style_num),
        Paragraph(total[2], style_num),
        Paragraph(total[3], style_num),
        Paragraph(total[4], style_num),
    ])
    total_row_idx = len(data) - 1

    # Legend rows: left explanations + right swatches spanning the three CAGR columns
    legend_rows = []
    if page.get("page_type") == "district":
        bucket_suffix = _abbr_bucket_suffix(page.get("baseline_title", ""))
        red_text = f"&gt; Western CAGR {bucket_suffix}".strip()
        grn_text = f"&lt; Western CAGR {bucket_suffix}".strip()
        shade_rule = (
            f"Shading vs Western: |Δ$/pupil| ≥ {DOLLAR_THRESHOLD_REL*100:.1f}% "
            f"or |ΔCAGR| ≥ {MATERIAL_DELTA_PCTPTS*100:.1f}pp"
        )
        cagr_def   = "CAGR = (End/Start)^(1/years) − 1"
        legend_rows = [
            ["", Paragraph(shade_rule, style_legend_right), "", Paragraph(red_text, style_legend_center), "", ""],
            ["", Paragraph(cagr_def,   style_legend_right), "", Paragraph(grn_text, style_legend_center), "", ""],
        ]
    data.extend(legend_rows)
    leg1_idx = total_row_idx + 1 if legend_rows else None
    leg2_idx = total_row_idx + 2 if legend_rows else None

    tbl = Table(data, colWidths=[0.22*inch, None, 1.2*inch, 0.95*inch, 0.95*inch, 0.95*inch])

    # Table style
    ts = TableStyle([
        ("LINEBELOW", (0,0), (-1,0), 0.5, colors.black),
        ("VALIGN",(0,0), (-1,-1), "MIDDLE"),
        ("ALIGN", (2,1), (-1,-1), "RIGHT"),
        ("ALIGN", (1,1), (1,-1), "LEFT"),
        ("LEFTPADDING",  (0,0), (-1,-1), 4),
        ("RIGHTPADDING", (0,0), (-1,-1), 4),
        ("TOPPADDING",   (0,0), (-1,-1), 3),
        ("BOTTOMPADDING",(0,0), (-1,-1), 3),
        ("LINEABOVE", (0, total_row_idx), (-1, total_row_idx), 0.5, colors.black),
    ])

    # Paint category swatch backgrounds (full cell)
    for i, (_sc, _ls, _c5, _c10, _c15, col, _lv) in enumerate(rows_in, start=1):
        ts.add("BACKGROUND", (0,i), (0,i), HexColor(col))

    # Shade $/pupil (col 2) and all 3 CAGR columns (cols 3–5)
    for i, (sc, _ls, c5s, c10s, c15s, _col, latest_val) in enumerate(rows_in, start=1):
        # optional $ floor gate
        if MATERIAL_MIN_LATEST_DOLLARS > 0 and (page.get("page_type") == "district"):
            if latest_val < MATERIAL_MIN_LATEST_DOLLARS:
                continue

        base_map = base.get(sc, {})

        # --- $/pupil shading (relative vs Western baseline) ---
        base_dollar = base_map.get("DOLLAR", float("nan"))
        if base_dollar == base_dollar and base_dollar > 0 and latest_val == latest_val:
            delta_rel = (latest_val - float(base_dollar)) / float(base_dollar)
            bg = _shade_for_dollar_rel(delta_rel)
            if bg is not None:
                ts.add("BACKGROUND", (2, i), (2, i), bg)

        # --- CAGR shading (absolute pp delta vs Western baseline) ---
        for col, (cstr, key) in zip((3, 4, 5), [(c5s, "5"), (c10s, "10"), (c15s, "15")]):
            val = parse_pct_str_to_float(cstr)
            base_val = base_map.get(key, float("nan"))
            delta_pp = (val - base_val) if (val==val and base_val==base_val) else float("nan")
            bg = _shade_for_cagr_delta(delta_pp)
            if bg is not None:
                ts.add("BACKGROUND", (col, i), (col, i), bg)

    # Apply legend spans & swatches (immediately below TOTAL row)
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
    rows_in = page.get("fte_rows", [])
    latest_year_fte = page.get("latest_year_fte", page.get("latest_year","Latest"))

    header = [
        Paragraph("", style_hdr_left),
        Paragraph("FTE Series", style_hdr_left),
        Paragraph(f"{latest_year_fte}", style_hdr_right),
        Paragraph("CAGR 5y", style_hdr_right),
        Paragraph("CAGR 10y", style_hdr_right),
        Paragraph("CAGR 15y", style_hdr_right),
    ]
    data: List[List] = [header]

    for (color, label, latest_str, r5s, r10s, r15s) in rows_in:
        data.append([LineSwatch(color), Paragraph(label, style_body),
                     Paragraph(latest_str, style_num),
                     Paragraph(r5s, style_num),
                     Paragraph(r10s, style_num),
                     Paragraph(r15s, style_num)])

    # Total FTE (sum series)
    total_series = None
    for (_c, label, _ls, _r5, _r10, _r15) in rows_in:
        s = page.get("fte_series_map", {}).get(label)
        if s is None: continue
        total_series = s if total_series is None else (total_series.add(s, fill_value=0.0))
    if total_series is not None and not total_series.empty:
        ly = int(total_series.index.max())
        latest_str = f"{float(total_series.loc[ly]):,.0f}" if ly in total_series.index else "—"
        r5 = fmt_pct(compute_cagr_last(total_series,5))
        r10= fmt_pct(compute_cagr_last(total_series,10))
        r15= fmt_pct(compute_cagr_last(total_series,15))
        data.append(["", Paragraph("Total FTE", style_body),
                     Paragraph(latest_str, style_num),
                     Paragraph(r5, style_num),
                     Paragraph(r10, style_num),
                     Paragraph(r15, style_num)])
        total_row_idx = len(data) - 1
    else:
        total_row_idx = None

    tbl = Table(data, colWidths=[0.45*inch, None, 0.95*inch, 0.95*inch, 0.95*inch, 0.95*inch])
    ts = TableStyle([
        ("VALIGN",(0,0), (-1,-1), "MIDDLE"),
        ("ALIGN", (2,1), (-1,-1), "RIGHT"),
        ("ALIGN", (1,1), (1,-1), "LEFT"),
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
    """Build category rows and total for a given EPP pivot table."""
    if epp_pivot.empty:
        return [], ("Total", "$0", "—", "—", "—")

    bottom_top = canonical_order_bottom_to_top(epp_pivot.columns.tolist())
    top_bottom = list(reversed(bottom_top))

    cat_rows = []
    for sc in top_bottom:
        latest_val = float(epp_pivot.loc[latest_year, sc]) if (latest_year in epp_pivot.index and sc in epp_pivot.columns) else 0.0
        c5 = compute_cagr_last(epp_pivot[sc], 5)
        c10 = compute_cagr_last(epp_pivot[sc], 10)
        c15 = compute_cagr_last(epp_pivot[sc], 15)
        cat_rows.append((sc, f"${latest_val:,.0f}", fmt_pct(c5), fmt_pct(c10), fmt_pct(c15),
                        color_for(cmap_all, context, sc), latest_val))

    # Compute CAGR on total series (sum of all categories), not mean of individual CAGRs
    total_series = epp_pivot.sum(axis=1)
    cat_total = ("Total", f"${float(total_series.loc[latest_year]) if latest_year in total_series.index else 0.0:,.0f}",
                 fmt_pct(compute_cagr_last(total_series, 5)),
                 fmt_pct(compute_cagr_last(total_series, 10)),
                 fmt_pct(compute_cagr_last(total_series, 15)))

    return cat_rows, cat_total

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

# ---- Page dicts ----
def build_page_dicts(df: pd.DataFrame, reg: pd.DataFrame) -> List[dict]:
    pages: List[dict] = []

    latest = int(df["YEAR"].max())
    t0 = latest - 5

    # PAGE 1: ALPS + Peers (graph-only)
    pages.append(dict(
        title="ALPS PK-12 & Peers: PPE and Enrollment 2019 -> 2024",
        subtitle=f"Top: Bars show {t0} PPE with darker segment to {latest} (purple = decrease). Bottom: FTE enrollment trends {t0}->{latest}.",
        chart_path=str(OUTPUT_DIR / "ppe_change_bars_ALPS_and_peers.png"),
        graph_only=True
    ))

    cmap_all = create_or_load_color_map(df)

    # PAGE 2: ALPS-only (district-style WITH tables)
    epp_alps, lines_alps = prepare_district_epp_lines(df, "ALPS PK-12")
    if not epp_alps.empty or lines_alps:
        latest_year = get_latest_year(df, epp_alps)
        context = context_for_district(df, "ALPS PK-12")

        cat_rows, cat_total = _build_category_data(epp_alps, latest_year, context, cmap_all)
        fte_rows, fte_map, latest_fte_year = _build_fte_data(lines_alps, latest_year)

        # Western baseline (same bucket); add baseline $ for latest_year
        bucket = "le_500" if context == "SMALL" else "gt_500"
        title_w, epp_w, _ls, _lm = prepare_western_epp_lines(df, reg, bucket)
        base_map = {}
        if not epp_w.empty:
            for sc in epp_w.columns:
                s=epp_w[sc]
                base_map[sc]={
                    "5": compute_cagr_last(s,5),
                    "10":compute_cagr_last(s,10),
                    "15":compute_cagr_last(s,15),
                    "DOLLAR": (float(s.loc[latest_year]) if latest_year in s.index else float("nan")),
                }

        pages.append(dict(
            title="ALPS PK-12: Expenditures Per Pupil vs Enrollment",
            subtitle="Stacked per-pupil expenditures by category; black/red lines show in-/out-of-district FTE trend (ALPS uses its own FTE scale).",
            chart_path=str(district_png("ALPS PK-12")),
            latest_year=latest_year, latest_year_fte=latest_fte_year,
            cat_rows=cat_rows, cat_total=cat_total, fte_rows=fte_rows,
            fte_series_map=fte_map,
            page_type="district", baseline_title=title_w, baseline_map=base_map
        ))

    # DISTRICT PAGES
    for dist in ["Amherst-Pelham"] + [d for d in DISTRICTS_OF_INTEREST if d != "Amherst-Pelham"]:
        epp, lines = prepare_district_epp_lines(df, dist)
        if epp.empty and not lines:
            continue

        latest_year = get_latest_year(df, epp)
        context = context_for_district(df, dist)

        rows, total = _build_category_data(epp, latest_year, context, cmap_all)
        fte_rows, fte_map, latest_fte_year = _build_fte_data(lines, latest_year)

        bucket = "le_500" if context == "SMALL" else "gt_500"
        base_title = f"All Western MA Traditional Districts {'≤500' if bucket=='le_500' else '>500'} Students"
        base_map = {}
        title_w, epp_w, _ls, _lm = prepare_western_epp_lines(df, reg, bucket)
        if not epp_w.empty:
            for sc in epp_w.columns:
                s=epp_w[sc]
                base_map[sc]={
                    "5": compute_cagr_last(s,5),
                    "10":compute_cagr_last(s,10),
                    "15":compute_cagr_last(s,15),
                    "DOLLAR": (float(s.loc[latest_year]) if latest_year in s.index else float("nan")),
                }

        pages.append(dict(title=(f"Amherst-Pelham Regional: Expenditures Per Pupil vs Enrollment" if dist=="Amherst-Pelham"
                                 else f"{dist}: Expenditures Per Pupil vs Enrollment"),
                          subtitle="Stacked per-pupil expenditures by category; black/red lines show in-/out-of-district FTE trend.",
                          chart_path=str(district_png(dist)),
                          latest_year=latest_year, latest_year_fte=latest_fte_year,
                          cat_rows=rows, cat_total=total, fte_rows=fte_rows,
                          fte_series_map=fte_map,
                          page_type="district", baseline_title=base_title, baseline_map=base_map))

    # APPENDIX A
    pages.append(dict(
        title="Appendix A. All Western MA Traditional Districts",
        subtitle="(By enrollment size)",
        chart_path=None,
        graph_only=True,
        text_blocks=[]
    ))
    for bucket in ("le_500", "gt_500"):
        title, epp, lines_sum, lines_mean = prepare_western_epp_lines(df, reg, bucket)
        if epp.empty and not lines_sum:
            continue

        latest_year = get_latest_year(df, epp)
        context = context_for_western(bucket)

        rows, total = _build_category_data(epp, latest_year, context, cmap_all)
        fte_rows, fte_map, latest_fte_year = _build_fte_data(lines_mean, latest_year)

        pages.append(dict(title=title,
                          subtitle="Expenditures Per Pupil vs Enrollment — Not including charters and vocationals",
                          chart_path=str(regional_png(bucket)),
                          latest_year=latest_year, latest_year_fte=latest_fte_year,
                          cat_rows=rows, cat_total=total, fte_rows=fte_rows,
                          fte_series_map=fte_map, page_type="western"))

    return pages

# ---- Build PDF ----
def build_pdf(pages: List[dict], out_path: Path):
    doc = SimpleDocTemplate(str(out_path), pagesize=A4,
        leftMargin=0.5*inch, rightMargin=0.5*inch, topMargin=0.5*inch, bottomMargin=0.5*inch)

    story: List = []
    for idx, p in enumerate(pages):
        story.append(Paragraph(p["title"], style_title_main))
        default_sub = "Expenditures Per Pupil vs Enrollment"
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
                    max_chart_h = doc.height * (0.80 if "ppe_change_bars_alps_and_peers" in name else 0.62)
                else:
                    max_chart_h = doc.height * 0.40
                if im.drawHeight > max_chart_h:
                    im.drawHeight = max_chart_h; im.drawWidth = im.drawHeight / ratio
                story.append(im)

        for block in p.get("text_blocks", []) or []:
            story.append(Spacer(0, 6))
            story.append(Paragraph(block, style_body))

        if p.get("graph_only"):
            if idx < len(pages)-1: story.append(PageBreak())
            continue

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
    df, reg = load_data()
    df = add_alps_pk12(df)
    pages = build_page_dicts(df, reg)
    if not pages:
        print("[WARN] No pages to write."); return
    build_pdf(pages, OUTPUT_DIR / "expenditures_series.pdf")

if __name__ == "__main__":
    main()
