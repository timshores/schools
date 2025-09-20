from __future__ import annotations

import bisect, re
from pathlib import Path
from typing import Dict, List
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
)

# ===== code version =====
CODE_VERSION = "v2025.09.20-ALPS-9"

# ---- Tunables ----
# Shade only when |district CAGR − Western CAGR| >= 2 percentage points
DELTA_THRESHOLD_ABS = 0.02
# Bins for darker shades as |delta| increases (in percentage points)
SHADE_BINS = [0.02, 0.05, 0.08, 0.12]  # edges for 5 shades (overflow uses the last)
RED_SHADES = ["#FDE2E2", "#FAD1D1", "#F5B8B8", "#EF9E9E", "#E88080"]
GRN_SHADES = ["#E6F4EA", "#D5EDE0", "#C4E6D7", "#B3DFCD", "#A1D8C4"]

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
style_note       = ParagraphStyle("note",       parent=style_body,         fontSize=8,  leading=10, alignment=2)
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
    canvas.drawRightString(x_right, y2, f"Code: {CODE_VERSION}")
    canvas.restoreState()

# ---- Flowables ----
def _to_color(c): return c if isinstance(c, colors.Color) else HexColor(c)
class LineSwatch(Flowable):
    def __init__(self, color_like, w=42, h=8, lw=3.6):
        super().__init__(); self.c=_to_color(color_like); self.w=w; self.h=h; self.lw=lw; self.width=w; self.height=h
    def draw(self):
        c = self.canv; y = self.h/2.0
        c.setStrokeColor(self.c); c.setLineWidth(self.lw); c.line(0, y, self.w, y)
        c.setFillColor(colors.white); c.setStrokeColor(self.c)
        r = self.h*0.7; c.circle(self.w/2.0, y, r, stroke=1, fill=1)

# ---- Helpers ----
def fmt_pct(v: float) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)): return "—"
    return f"{v * 100:+.1f}%"

def parse_pct_str_to_float(s: str) -> float:
    try: return float((s or "").replace("%","").replace("+","").strip())/100.0
    except Exception: return float("nan")

def _abbr_bucket_suffix(full: str) -> str:
    if re.search(r"≤\s*500", full or "", flags=re.I): return "(≤500)"
    if re.search(r">\s*500", full or "", flags=re.I): return "(>500)"
    return ""

def _shade_for_delta(delta: float):
    if delta != delta or abs(delta) < DELTA_THRESHOLD_ABS:
        return None
    idx = max(0, min(len(RED_SHADES)-1, bisect.bisect_right(SHADE_BINS, abs(delta)) - 1))
    return HexColor(RED_SHADES[idx]) if delta > 0 else HexColor(GRN_SHADES[idx])

# ---- Page dicts ----
def build_page_dicts(df: pd.DataFrame, reg: pd.DataFrame) -> List[dict]:
    pages: List[dict] = []

    latest = int(df["YEAR"].max())
    t0 = latest - 5

    # PAGE 1: ALPS + Peers (graph-only, with subtitle)
    pages.append(dict(
        title="ALPS PK-12 & Peers: PPE and Enrollment 2019 -> 2024",
        subtitle=f"Bars show {t0} PPE with darker segment to {latest} (purple = decrease). Pale-orange mini-areas above each bar trace FTE enrollment over {t0}->{latest}.",
        chart_path=str(OUTPUT_DIR / "ppe_change_bars_ALPS_and_peers.png"),
        graph_only=True
    ))

    # PAGE 2: ALPS-only (district-style page WITH tables)
    cmap_all = create_or_load_color_map(df)
    epp_alps, lines_alps = prepare_district_epp_lines(df, "ALPS PK-12")
    if not epp_alps.empty or lines_alps:
        bottom_top = canonical_order_bottom_to_top(epp_alps.columns.tolist()) if not epp_alps.empty else []
        top_bottom  = list(reversed(bottom_top))
        latest_year = int(epp_alps.index.max()) if not epp_alps.empty else latest
        context     = context_for_district(df, "ALPS PK-12")

        cat_rows = []
        for sc in top_bottom:
            latest_val = float(epp_alps.loc[latest_year, sc]) if (latest_year and sc in epp_alps.columns) else 0.0
            c5 = compute_cagr_last(epp_alps[sc],5); c10=compute_cagr_last(epp_alps[sc],10); c15=compute_cagr_last(epp_alps[sc],15)
            cat_rows.append((sc, f"${latest_val:,.0f}", fmt_pct(c5), fmt_pct(c10), fmt_pct(c15),
                             color_for(cmap_all, context, sc), latest_val))

        latest_vals = [epp_alps.loc[latest_year, sc] if sc in epp_alps.columns else 0.0 for sc in epp_alps.columns]
        def mean_clean(arr): arr=[a for a in arr if a==a]; return float(np.mean(arr)) if arr else float("nan")
        cat_total = ("Total", f"${float(np.nansum(latest_vals)):,.0f}",
                     fmt_pct(mean_clean([compute_cagr_last(epp_alps[sc],5)  for sc in epp_alps.columns])),
                     fmt_pct(mean_clean([compute_cagr_last(epp_alps[sc],10) for sc in epp_alps.columns])),
                     fmt_pct(mean_clean([compute_cagr_last(epp_alps[sc],15) for sc in epp_alps.columns])))

        fte_years = [int(s.index.max()) for s in lines_alps.values() if s is not None and not s.empty]
        latest_fte_year = max(fte_years) if fte_years else latest_year
        fte_rows = []
        for _k, label in ENROLL_KEYS:
            s = lines_alps.get(label)
            if s is None or s.empty: continue
            r5=compute_cagr_last(s,5); r10=compute_cagr_last(s,10); r15=compute_cagr_last(s,15)
            fte_rows.append((FTE_LINE_COLORS[label], label,
                             ("—" if latest_fte_year not in s.index else f"{float(s.loc[latest_fte_year]):,.0f}"),
                             fmt_pct(r5), fmt_pct(r10), fmt_pct(r15)))

        bucket = "le_500" if context == "SMALL" else "gt_500"
        title_w, epp_w, _ls, _lm = prepare_western_epp_lines(df, reg, bucket)
        base_map = {}
        if not epp_w.empty:
            for sc in epp_w.columns:
                s=epp_w[sc]; base_map[sc]={"5":compute_cagr_last(s,5),"10":compute_cagr_last(s,10),"15":compute_cagr_last(s,15)}

        pages.append(dict(
            title="ALPS PK-12: Expenditures Per Pupil vs Enrollment",
            subtitle="Stacked per-pupil expenditures by category; black/red lines show in-/out-of-district FTE trend (ALPS uses its own FTE scale).",
            chart_path=str(district_png("ALPS PK-12")),
            latest_year=latest_year, latest_year_fte=latest_fte_year,
            cat_rows=cat_rows, cat_total=cat_total, fte_rows=fte_rows,
            page_type="district", baseline_title=title_w, baseline_map=base_map
        ))

    # Western pages (with tables)
    for bucket in ("le_500", "gt_500"):
        title, epp, lines_sum, lines_mean = prepare_western_epp_lines(df, reg, bucket)
        if epp.empty and not lines_sum: continue
        bottom_top = canonical_order_bottom_to_top(epp.columns.tolist())
        top_bottom  = list(reversed(bottom_top))
        latest_year = int(epp.index.max()) if not epp.empty else latest
        context     = context_for_western(bucket)

        rows = []
        for sc in top_bottom:
            latest_val = float(epp.loc[latest_year, sc]) if (latest_year and sc in epp.columns) else 0.0
            c5 = compute_cagr_last(epp[sc],5); c10 = compute_cagr_last(epp[sc],10); c15 = compute_cagr_last(epp[sc],15)
            rows.append((sc, f"${latest_val:,.0f}", fmt_pct(c5), fmt_pct(c10), fmt_pct(c15),
                         color_for(cmap_all, context, sc), latest_val))

        latest_vals = [epp.loc[latest_year, sc] if sc in epp.columns else 0.0 for sc in epp.columns]
        def mean_clean(arr): arr=[a for a in arr if a==a]; return float(np.mean(arr)) if arr else float("nan")
        total = ("Total", f"${float(np.nansum(latest_vals)):,.0f}",
                 fmt_pct(mean_clean([compute_cagr_last(epp[sc],5)  for sc in epp.columns])),
                 fmt_pct(mean_clean([compute_cagr_last(epp[sc],10) for sc in epp.columns])),
                 fmt_pct(mean_clean([compute_cagr_last(epp[sc],15) for sc in epp.columns])))

        fte_rows = []
        fte_years = [int(s.index.max()) for s in lines_mean.values() if s is not None and not s.empty]
        latest_fte_year = max(fte_years) if fte_years else latest_year
        for _k, label in ENROLL_KEYS:
            s = lines_mean.get(label)
            if s is None or s.empty: continue
            r5=compute_cagr_last(s,5); r10=compute_cagr_last(s,10); r15=compute_cagr_last(s,15)
            val = "—" if latest_fte_year not in s.index else f"{float(s.loc[latest_fte_year]):,.0f}"
            fte_rows.append((FTE_LINE_COLORS[label], label, val, fmt_pct(r5), fmt_pct(r10), fmt_pct(r15)))

        pages.append(dict(title=title,
                          subtitle="Expenditures Per Pupil vs Enrollment — Not including charters and vocationals",
                          chart_path=str(regional_png(bucket)),
                          latest_year=latest_year, latest_year_fte=latest_fte_year,
                          cat_rows=rows, cat_total=total, fte_rows=fte_rows, page_type="western"))

    # District pages with tables (the five)
    for dist in ["Amherst-Pelham"] + [d for d in DISTRICTS_OF_INTEREST if d != "Amherst-Pelham"]:
        epp, lines = prepare_district_epp_lines(df, dist)
        if epp.empty and not lines: continue
        bottom_top = canonical_order_bottom_to_top(epp.columns.tolist())
        top_bottom = list(reversed(bottom_top))
        latest_year = int(epp.index.max()) if not epp.empty else latest
        context = context_for_district(df, dist)

        rows = []
        for sc in top_bottom:
            latest_val = float(epp.loc[latest_year, sc]) if (latest_year and sc in epp.columns) else 0.0
            c5 = compute_cagr_last(epp[sc],5); c10=compute_cagr_last(epp[sc],10); c15=compute_cagr_last(epp[sc],15)
            rows.append((sc, f"${latest_val:,.0f}", fmt_pct(c5), fmt_pct(c10), fmt_pct(c15),
                         color_for(cmap_all, context, sc), latest_val))

        latest_vals = [epp.loc[latest_year, sc] if sc in epp.columns else 0.0 for sc in epp.columns]
        def mean_clean(arr): arr=[a for a in arr if a==a]; return float(np.mean(arr)) if arr else float("nan")
        total = ("Total", f"${float(np.nansum(latest_vals)):,.0f}",
                 fmt_pct(mean_clean([compute_cagr_last(epp[sc],5)  for sc in epp.columns])),
                 fmt_pct(mean_clean([compute_cagr_last(epp[sc],10) for sc in epp.columns])),
                 fmt_pct(mean_clean([compute_cagr_last(epp[sc],15) for sc in epp.columns])))

        fte_years = [int(s.index.max()) for s in lines.values() if s is not None and not s.empty]
        latest_fte_year = max(fte_years) if fte_years else latest_year
        fte_rows = []
        for _k, label in ENROLL_KEYS:
            s = lines.get(label)
            if s is None or s.empty: continue
            r5=compute_cagr_last(s,5); r10=compute_cagr_last(s,10); r15=compute_cagr_last(s,15)
            fte_rows.append((FTE_LINE_COLORS[label], label,
                             ("—" if latest_fte_year not in s.index else f"{float(s.loc[latest_fte_year]):,.0f}"),
                             fmt_pct(r5), fmt_pct(r10), fmt_pct(r15)))

        bucket = "le_500" if context == "SMALL" else "gt_500"
        base_title = f"All Western MA Traditional Districts {'≤500' if bucket=='le_500' else '>500'} Students"
        base_map = {}
        title_w, epp_w, _ls, _lm = prepare_western_epp_lines(df, reg, bucket)
        if not epp_w.empty:
            for sc in epp_w.columns:
                s=epp_w[sc]; base_map[sc]={"5":compute_cagr_last(s,5),"10":compute_cagr_last(s,10),"15":compute_cagr_last(s,15)}

        pages.append(dict(title=(f"Amherst-Pelham Regional: Expenditures Per Pupil vs Enrollment" if dist=="Amherst-Pelham"
                                 else f"{dist}: Expenditures Per Pupil vs Enrollment"),
                          subtitle="Stacked per-pupil expenditures by category; black/red lines show in-/out-of-district FTE trend.",
                          chart_path=str(district_png(dist)),
                          latest_year=latest_year, latest_year_fte=latest_fte_year,
                          cat_rows=rows, cat_total=total, fte_rows=fte_rows,
                          page_type="district", baseline_title=base_title, baseline_map=base_map))

    return pages

# ---- Build PDF ----
def build_pdf(pages: List[dict], out_path: Path):
    doc = SimpleDocTemplate(str(out_path), pagesize=A4,
        leftMargin=0.5*inch, rightMargin=0.5*inch, topMargin=0.5*inch, bottomMargin=0.5*inch)

    story: List = []
    for idx, p in enumerate(pages):
        # Title & Subtitle
        story.append(Paragraph(p["title"], style_title_main))
        default_sub = "Expenditures Per Pupil vs Enrollment"
        story.append(Paragraph(p.get("subtitle", (default_sub if not p.get("graph_only") else "")), style_title_sub))

        # Chart
        img_path = Path(p["chart_path"])
        if not img_path.exists():
            story.append(Paragraph(f"[Missing chart image: {img_path.name}]", style_body))
        else:
            im = Image(str(img_path))
            ratio = im.imageHeight / float(im.imageWidth)
            im.drawWidth = doc.width; im.drawHeight = doc.width * ratio
            if p.get("graph_only"):
                name = img_path.name.lower()
                max_chart_h = doc.height * (0.74 if "ppe_change_bars_alps_and_peers" in name else 0.62)
            else:
                max_chart_h = doc.height * 0.40
            if im.drawHeight > max_chart_h:
                im.drawHeight = max_chart_h; im.drawWidth = im.drawHeight / ratio
            story.append(im)

        # Graph-only pages: no tables
        if p.get("graph_only"):
            if idx < len(pages)-1: story.append(PageBreak())
            continue

        story.append(Spacer(0, 24))

        # -------- Category table --------
        cat_header = ["", "Category", f"{p['latest_year']} $/pupil", "5y CAGR", "10y CAGR", "15y CAGR"]
        cat_rows_rl, swatch_rows = [], []
        for i, (label, latest, c5, c10, c15, hexcol, latest_num) in enumerate(p["cat_rows"], start=1):
            cat_rows_rl.append(["", Paragraph(label, style_body),
                                Paragraph(latest, style_num),
                                Paragraph(c5, style_num), Paragraph(c10, style_num), Paragraph(c15, style_num)])
            swatch_rows.append((i, HexColor(hexcol)))

        total = ["", Paragraph(p["cat_total"][0], style_body),
                 Paragraph(p["cat_total"][1], style_num),
                 Paragraph(p["cat_total"][2], style_num),
                 Paragraph(p["cat_total"][3], style_num),
                 Paragraph(p["cat_total"][4], style_num)]

        # Legend rows: two rows IMMEDIATELY under Total
        legend_rows = []
        legend_positions = None
        if p.get("page_type") == "district":
            bucket_suffix = _abbr_bucket_suffix(p.get("baseline_title",""))
            rule_text = f"Shading compares each district CAGR to Western MA {bucket_suffix} (same category)."
            red_text  = f"Higher than Western MA {bucket_suffix}"
            grn_text  = f"Lower than Western MA {bucket_suffix}"
            legend_rows = [
                ["", Paragraph(rule_text, style_legend), "", Paragraph(red_text, style_legend), "", ""],  # row 1 (red)
                ["", Paragraph("CAGR = (End/Start)^(1/years) − 1", style_legend), "", Paragraph(grn_text, style_legend), "", ""],  # row 2 (green)
            ]
            legend_positions = ("red", "green", bucket_suffix)

        cat_data = [cat_header] + cat_rows_rl + [total] + legend_rows
        cat_tbl = Table(cat_data,
            colWidths=[0.08*doc.width, 0.42*doc.width, 0.18*doc.width, 0.10*doc.width, 0.10*doc.width, 0.12*doc.width],
            hAlign="LEFT", repeatRows=1, splitByRow=1)

        ts = [("FONT",(0,0),(-1,0),"Helvetica-Bold",9),
              ("ALIGN",(2,0),(-1,0),"RIGHT"),
              ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
              ("ALIGN",(2,1),(-1,-1),"RIGHT"),
              ("LINEBELOW",(0,0),(-1,0),0.5,colors.black),
              ("ROWSPACING",(0,1),(-1,-1),2),
              ("TOPPADDING",(0,1),(-1,-1),4),
              ("BOTTOMPADDING",(0,1),(-1,-1),4)]

        # swatches for each category row
        for r, colr in swatch_rows:
            ts.append(("BACKGROUND", (0, r), (0, r), colr))

        # Total row styling
        total_row_idx = 1 + len(cat_rows_rl)
        ts += [("BACKGROUND",(0,total_row_idx),(-1,total_row_idx),colors.whitesmoke),
               ("FONT",(0,total_row_idx),(-1,total_row_idx),"Helvetica-Bold",9)]

        # District-page legend formatting: rows immediately under Total
        if legend_positions:
            red_row   = total_row_idx + 1
            green_row = total_row_idx + 2

            # Left explainer spans Category + $/pupil; right-justified
            ts += [("SPAN", (1, red_row), (2, red_row)),
                   ("SPAN", (1, green_row), (2, green_row)),
                   ("ALIGN",(1, red_row), (2, red_row), "RIGHT"),
                   ("ALIGN",(1, green_row), (2, green_row), "RIGHT")]

            # Right: colored blocks across CAGR columns with text inside
            ts += [("SPAN", (3, red_row), (5, red_row)),
                   ("SPAN", (3, green_row), (5, green_row)),
                   ("BACKGROUND", (3, red_row), (5, red_row), HexColor("#F5B8B8")),
                   ("BACKGROUND", (3, green_row), (5, green_row), HexColor("#C4E6D7")),
                   ("ALIGN", (3, red_row), (5, red_row), "CENTER"),
                   ("ALIGN", (3, green_row), (5, green_row), "CENTER")]

        # -------- Cell-level shading vs. Western MA (bucket-aligned) --------
        if p.get("page_type") == "district":
            base_map = p.get("baseline_map", {})
            for row_idx, row_raw in enumerate(p["cat_rows"], start=1):
                subcat = row_raw[0]
                base = base_map.get(subcat)
                if not base: continue
                vals = [parse_pct_str_to_float(row_raw[2]), parse_pct_str_to_float(row_raw[3]), parse_pct_str_to_float(row_raw[4])]
                bases= [base.get("5"), base.get("10"), base.get("15")]
                for c_off, (dv, bv) in enumerate(zip(vals, bases), start=3):
                    if dv == dv and bv == bv:
                        delta = dv - bv
                        col = _shade_for_delta(delta)
                        if col is not None:
                            ts.append(("BACKGROUND", (c_off, row_idx), (c_off, row_idx), col))

        cat_tbl.setStyle(TableStyle(ts))
        story.append(cat_tbl)
        story.append(Spacer(0, 24))

        # -------- FTE table --------
        fte_header_label = f"{p['latest_year_fte']} FTE" if p.get("page_type")=="district" else f"{p['latest_year_fte']} avg FTE"
        fte_header = ["", "Pupil Group", fte_header_label, "5y CAGR", "10y CAGR", "15y CAGR"]
        fte_data = [fte_header]
        for (hexcol, label, fte, c5, c10, c15) in p["fte_rows"]:
            fte_data.append([LineSwatch(hexcol), Paragraph(label, style_body),
                             Paragraph(fte, style_num),
                             Paragraph(c5, style_num), Paragraph(c10, style_num), Paragraph(c15, style_num)])
        fte_tbl = Table(fte_data,
            colWidths=[0.16*doc.width, 0.34*doc.width, 0.18*doc.width, 0.10*doc.width, 0.10*doc.width, 0.12*doc.width],
            hAlign="LEFT", repeatRows=1, splitByRow=1)
        fte_tbl.setStyle(TableStyle([
            ("FONT",(0,0),(-1,0),"Helvetica-Bold",9),
            ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
            ("ALIGN",(2,1),(-1,-1),"RIGHT"),
            ("ALIGN",(2,0),(-1,0),"RIGHT"),
            ("LINEBELOW",(0,0),(-1,0),0.5,colors.black),
            ("ROWSPACING",(0,1),(-1,-1),2),
            ("TOPPADDING",(0,1),(-1,-1),4),
            ("BOTTOMPADDING",(0,1),(-1,-1),4),
        ]))
        story.append(fte_tbl)
        story.append(Spacer(0, 24))

        if idx < len(pages)-1:
            story.append(PageBreak())

    doc.build(story, onFirstPage=draw_footer, onLaterPages=draw_footer)
    print(f"[OK] Wrote PDF: {out_path}")

# ---- CAGR helper (kept here to avoid circular refs) ----
def compute_cagr_last(series: pd.Series, years_back: int) -> float:
    if series is None or series.empty: return float("nan")
    s = series.sort_index().astype(float).replace([np.inf,-np.inf], np.nan).dropna()
    if s.size < 2: return float("nan")
    end_year = int(s.index.max())
    cands = s[s.index <= end_year - years_back]
    start_year = int(cands.index.max()) if not cands.empty else int(s.index.min())
    n = end_year - start_year
    if n <= 0: return float("nan")
    a = float(s.loc[start_year]); b = float(s.loc[end_year])
    if a <= 0 or b <= 0: return float("nan")
    return (b/a)**(1.0/n) - 1.0

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
