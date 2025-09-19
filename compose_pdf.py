# compose_pdf.py — A4 portrait PDF compositor using ReportLab
# pip install reportlab pandas numpy

from __future__ import annotations

import bisect
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Flowable,
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ---------------- Paths ----------------
DATA_DIR = Path("./data")
OUTPUT_DIR = Path("./output")
EXCEL_FILE = DATA_DIR / "E2C_Hub_MA_DESE_Data.xlsx"
OUTPUT_PDF = OUTPUT_DIR / "expenditures_series.pdf"


def district_png(dist: str) -> Path:
    return OUTPUT_DIR / f"expenditures_per_pupil_vs_enrollment_{dist.replace(' ', '_')}.png"


def regional_png(key: str) -> Path:
    return OUTPUT_DIR / f"regional_expenditures_per_pupil_{key}.png"


# ---------------- Domain constants ----------------
SHEET_EXPEND = "District Expend by Category"
SHEET_REGIONS = "District Regions"

# Page order (Western first; then these districts)
DISTRICTS = ["Amherst-Pelham", "Amherst", "Leverett", "Pelham", "Shutesbury"]

EXCLUDE_SUBCATS = {"total expenditures", "total in-district expenditures"}
ENROLL_KEYS = [
    ("in-district fte pupils", "In-District FTE Pupils"),
    ("out-of-district fte pupils", "Out-of-District FTE Pupils"),
]
TOTAL_FTE_KEY = "total fte pupils"

# Category color palettes
COLOR_OVERRIDES_DIST = {"professional development": "#6A3D9A"}
PALETTE_DIST = [
    "#0072B2", "#009E73", "#D55E00", "#CC79A7", "#F0E442", "#56B4E9",
    "#E69F00", "#999999", "#332288", "#88CCEE", "#44AA99", "#882255",
]
COLOR_OVERRIDES_APEL = {"professional development": "#2A9D8F"}
PALETTE_APEL = [
    "#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51", "#8AB17D",
    "#7A5195", "#4ECDC4", "#C7F464", "#FF6B6B", "#FFE66D", "#355C7D",
]
COLOR_OVERRIDES_REGION = {"professional development": "#1B9E77"}
PALETTE_REGION = [
    "#8DD3C7", "#FFFFB3", "#BEBADA", "#FB8072", "#80B1D3", "#FDB462",
    "#B3DE69", "#FCCDE5", "#D9D9D9", "#BC80BD", "#CCEBC5", "#FFED6F",
]

# FTE legend line colors (match plotting)
LINE_COLORS_DIST = {"In-District FTE Pupils": "#000000", "Out-of-District FTE Pupils": "#AA0000"}
LINE_COLORS_APEL = {"In-District FTE Pupils": "#1F4E79", "Out-of-District FTE Pupils": "#D35400"}
LINE_COLORS_REGION = {"In-District FTE Pupils": "#1B7837", "Out-of-District FTE Pupils": "#5E3C99"}

# ---------------- Text styles ----------------
styles = getSampleStyleSheet()
style_title_main = ParagraphStyle("title_main", parent=styles["Heading1"], fontSize=18, leading=22, spaceAfter=2)
style_title_sub = ParagraphStyle("title_sub", parent=styles["Normal"], fontSize=12, leading=14, spaceAfter=6)
style_body = ParagraphStyle("body", parent=styles["Normal"], fontSize=9, leading=12)
style_num = ParagraphStyle("num", parent=styles["Normal"], fontSize=9, leading=12, alignment=2)  # right-align
style_src = ParagraphStyle("src", parent=styles["Normal"], fontSize=8, leading=10)
style_legend = ParagraphStyle("legend", parent=style_body, fontSize=8, leading=10, alignment=2)  # RIGHT

# Neutral emphasis for negatives (no red/green baggage)
NEG_COLOR = HexColor("#3F51B5")
style_num_neg = ParagraphStyle("num_neg", parent=style_num, textColor=NEG_COLOR)

# Footer text (every page) — 2 lines only
SOURCE_LINE1 = "Source: DESE District Expenditures by Spending Category , Last updated: August 12, 2025"
SOURCE_LINE2 = "https://educationtocareer.data.mass.gov/Finance-and-Budget/District-Expenditures-by-Spending-Category/er3w-dyti/"

def draw_footer(canvas, doc):
    """Left: 2-line source; Right: page number."""
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    y1 = 0.6 * inch
    y2 = 0.44 * inch
    canvas.drawString(doc.leftMargin, y1, SOURCE_LINE1)
    canvas.drawString(doc.leftMargin, y2, SOURCE_LINE2)
    # page number on the right, aligned with the first line
    x_right = doc.pagesize[0] - doc.rightMargin
    canvas.drawRightString(x_right, y1, f"Page {canvas.getPageNumber()}")
    canvas.restoreState()


# ---------------- Flowables ----------------
def _to_color(c):
    return c if isinstance(c, colors.Color) else HexColor(c)

class LineSwatch(Flowable):
    """Line + white-filled marker to match chart look."""
    def __init__(self, color_like, w=42, h=8, lw=3.6):
        super().__init__()
        self.c = _to_color(color_like)
        self.w, self.h, self.lw = w, h, lw
        self.width, self.height = w, h
    def draw(self):
        c = self.canv
        y = self.h / 2.0
        c.setStrokeColor(self.c)
        c.setLineWidth(self.lw)
        c.line(0, y, self.w, y)
        c.setFillColor(colors.white)
        c.setStrokeColor(self.c)
        r = self.h * 0.7
        c.circle(self.w / 2.0, y, r, stroke=1, fill=1)

class ColorSwatch(Flowable):
    """Solid color block for Category legend."""
    def __init__(self, color_like, w=14, h=10):
        super().__init__()
        self.c = _to_color(color_like)
        self.w, self.h = w, h
        self.width, self.height = w, h
    def draw(self):
        c = self.canv
        c.setFillColor(self.c)
        c.setStrokeColor(self.c)
        c.rect(0, 0, self.w, self.h, fill=1, stroke=0)


# ---------------- Data utils ----------------
def find_sheet_name(xls: pd.ExcelFile, target: str) -> str:
    t = target.strip().lower()
    for n in xls.sheet_names:
        if n.strip().lower() == t:
            return n
    for n in xls.sheet_names:
        nl = n.strip().lower()
        if t in nl or nl in t:
            return n
    return xls.sheet_names[0]

def coalesce_year_column(df: pd.DataFrame) -> pd.DataFrame:
    name_map = {str(c).strip().lower(): c for c in df.columns}
    chosen = None
    for key in ("fiscal_year", "year", "sy", "school_year"):
        if key in name_map:
            chosen = name_map[key]
            break
    if chosen is None:
        first = df.columns[0]
        if str(first).strip().lower() in ("sy", "school_year"):
            chosen = first
    if chosen is None:
        for c in df.columns:
            sample = str(df[c].dropna().astype(str).head(20).tolist())
            if re.search(r"\b(19|20)\d{2}\b", sample):
                chosen = c
                break
    if chosen is None:
        raise ValueError("Could not find a year column.")

    def parse_to_year(val):
        if pd.isna(val):
            return np.nan
        if isinstance(val, (int, float)) and not pd.isna(val):
            v = int(val)
            if 1900 <= v <= 2100:
                return v
        s = str(val).strip()
        m = re.search(r"\b((?:19|20)\d{2})\s*[–\-/]\s*(\d{2,4})\b", s)
        if m:
            y1 = int(m.group(1))
            y2r = m.group(2)
            y2 = (y1 // 100) * 100 + int(y2r) if len(y2r) == 2 else int(y2r)
            if len(y2r) == 2 and (y2 % 100) < (y1 % 100):
                y2 += 100
            return y2
        m2 = re.search(r"\b((?:19|20)\d{2})\b", s)
        if m2:
            return int(m2.group(1))
        return np.nan

    df = df.copy()
    df["YEAR"] = df[chosen].apply(parse_to_year)
    df = df.dropna(subset=["YEAR"])
    df["YEAR"] = df["YEAR"].astype(int)
    return df

def compute_cagr_last(series: pd.Series, years_back: int) -> float:
    if series is None or series.empty:
        return np.nan
    s = series.sort_index().astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if s.size < 2:
        return np.nan
    end_year = int(s.index.max())
    candidates = s[s.index <= end_year - years_back]
    start_year = int(candidates.index.max()) if not candidates.empty else int(s.index.min())
    n_years = end_year - start_year
    if n_years <= 0:
        return np.nan
    start_val = float(s.loc[start_year])
    end_val = float(s.loc[end_year])
    if start_val <= 0 or end_val <= 0:
        return np.nan
    return (end_val / start_val) ** (1.0 / n_years) - 1.0

def order_subcats_by_mean(piv: pd.DataFrame) -> List[str]:
    if piv.shape[1] <= 1:
        return list(piv.columns)
    return list(piv.mean(axis=0).sort_values(ascending=False).index)

def fmt_pct(v: float) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v * 100:+.1f}%"

def parse_pct_str_to_float(s: str) -> float:
    try:
        return float((s or "").replace("%", "").replace("+", "").strip()) / 100.0
    except Exception:
        return float("nan")

def make_color_map(subcats: List[str], palette: List[str], overrides: Dict[str, str]) -> Dict[str, str]:
    base = (palette * ((len(subcats) // len(palette)) + 1))[: len(subcats)]
    cmap = {sc: base[i] for i, sc in enumerate(subcats)}
    for sc in subcats:
        key = sc.strip().lower()
        if key in overrides:
            cmap[sc] = overrides[key]
    return cmap


# ---- intensity shades for relative-to-Western highlighting ----
# bins in absolute percentage points (e.g., 0.00, 0.02 = 2pp, 0.05 = 5pp, 0.08 = 8pp, 0.12 = 12pp+)
DIFF_BINS = [0.00, 0.02, 0.05, 0.08, 0.12]
RED_SHADES = ["#FDE2E2", "#FAD1D1", "#F5B8B8", "#EF9E9E", "#E88080"]
GRN_SHADES = ["#E6F4EA", "#D5EDE0", "#C4E6D7", "#B3DFCD", "#A1D8C4"]

def shade_for_delta(delta: float) -> colors.Color | None:
    """Return a more intense shade as |delta| grows; None when delta ~ 0."""
    if delta is None or not (delta == delta):  # NaN
        return None
    if abs(delta) < 1e-6:
        return None
    idx = bisect.bisect_right(DIFF_BINS, abs(delta)) - 1
    idx = max(0, min(idx, len(DIFF_BINS) - 1))
    return HexColor(RED_SHADES[idx]) if delta > 0 else HexColor(GRN_SHADES[idx])


# ---------------- Load & Prepare ----------------
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    xls = pd.ExcelFile(EXCEL_FILE)

    df = pd.read_excel(xls, sheet_name=find_sheet_name(xls, SHEET_EXPEND))
    df.columns = [str(c).strip() for c in df.columns]
    ren = {}
    for c in df.columns:
        low = str(c).strip().lower()
        if low == "dist_name": ren[c] = "DIST_NAME"
        elif low == "ind_cat": ren[c] = "IND_CAT"
        elif low == "ind_subcat": ren[c] = "IND_SUBCAT"
        elif low == "ind_value": ren[c] = "IND_VALUE"
        elif low in ("sy", "school_year", "fiscal_year", "year"): ren[c] = c
    df = df.rename(columns=ren)
    if "IND_VALUE" not in df.columns:
        raise ValueError("IND_VALUE missing")
    df = coalesce_year_column(df)
    for c in ("DIST_NAME", "IND_CAT", "IND_SUBCAT"):
        df[c] = df[c].astype(str).str.strip()
    df["IND_VALUE"] = pd.to_numeric(df["IND_VALUE"], errors="coerce")

    reg = pd.read_excel(xls, sheet_name=find_sheet_name(xls, SHEET_REGIONS))
    reg.columns = [str(c).strip() for c in reg.columns]
    cmap = {}
    for c in reg.columns:
        low = c.strip().lower()
        if low in ("district_name", "dist_name", "name"): cmap[c] = "DIST_NAME"
        elif low in ("eohhs_region", "region"): cmap[c] = "EOHHS_REGION"
        elif low in ("school_type", "type", "district_type", "org_type"): cmap[c] = "SCHOOL_TYPE"
    reg = reg.rename(columns=cmap)
    for c in ("DIST_NAME", "EOHHS_REGION", "SCHOOL_TYPE"):
        reg[c] = reg[c].astype(str).str.strip()
    return df, reg

def prepare_district(df: pd.DataFrame, dist: str) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
    ddf = df[df["DIST_NAME"].str.lower() == dist.lower()].copy()

    epp = ddf[ddf["IND_CAT"].str.lower() == "expenditures per pupil"].copy()
    epp["IND_SUBCAT"] = epp["IND_SUBCAT"].replace({np.nan: "Unspecified"}).fillna("Unspecified")
    epp = epp[~epp["IND_SUBCAT"].str.lower().isin(EXCLUDE_SUBCATS)].copy()
    epp_pivot = (
        epp.pivot_table(index="YEAR", columns="IND_SUBCAT", values="IND_VALUE", aggfunc="sum")
        .sort_index()
        .fillna(0.0)
    )

    lines: Dict[str, pd.Series] = {}
    enroll = ddf[ddf["IND_CAT"].str.lower() == "student enrollment"].copy()
    for key, label in ENROLL_KEYS:
        ser = enroll[enroll["IND_SUBCAT"].str.lower() == key][["YEAR", "IND_VALUE"]]
        if ser.empty:
            continue
        s = ser.dropna().drop_duplicates(subset=["YEAR"]).set_index("YEAR")["IND_VALUE"].sort_index()
        lines[label] = s
    return epp_pivot, lines

def prepare_western(
    df: pd.DataFrame, reg: pd.DataFrame, size_bucket: str, school_type: str
) -> Tuple[str, pd.DataFrame, Dict[str, pd.Series], Dict[str, pd.Series]]:
    mask = (reg["EOHHS_REGION"].str.lower() == "western") & (
        reg["SCHOOL_TYPE"].str.lower() == school_type.lower()
    )
    names = set(reg[mask]["DIST_NAME"].str.lower())
    present = set(df["DIST_NAME"].str.lower())
    members = sorted(n for n in names if n in present)

    enroll = df[df["IND_CAT"].str.lower() == "student enrollment"].copy()
    totals = {}
    for nm in members:
        dsub = enroll[enroll["DIST_NAME"].str.lower() == nm]
        tot = dsub[dsub["IND_SUBCAT"].str.lower() == TOTAL_FTE_KEY]
        if not tot.empty:
            y = int(tot["YEAR"].max())
            totals[nm] = float(tot.loc[tot["YEAR"] == y, "IND_VALUE"].iloc[0])

    if size_bucket == "le_500":
        members = [n for n, v in totals.items() if v <= 500]
        suffix = "≤500 Students"
    elif size_bucket == "gt_500":
        members = [n for n, v in totals.items() if v > 500]
        suffix = ">500 Students"
    else:
        suffix = None

    title = f"All Western MA {school_type.capitalize()} Districts"
    if suffix:
        title += f" {suffix}"
    if not members:
        return title, pd.DataFrame(), {}, {}

    epp = df[
        (df["IND_CAT"].str.lower() == "expenditures per pupil") & (df["DIST_NAME"].str.lower().isin(members))
    ][["DIST_NAME", "YEAR", "IND_SUBCAT", "IND_VALUE"]].copy()
    epp = epp[~epp["IND_SUBCAT"].str.lower().isin(EXCLUDE_SUBCATS)].copy()

    # Weighted mean by In-District FTE
    wts = df[
        (df["IND_CAT"].str.lower() == "student enrollment")
        & (df["IND_SUBCAT"].str.lower() == "in-district fte pupils")
        & (df["DIST_NAME"].str.lower().isin(members))
    ][["DIST_NAME", "YEAR", "IND_VALUE"]].rename(columns={"IND_VALUE": "WEIGHT"})
    m = epp.merge(wts, on=["DIST_NAME", "YEAR"], how="left")
    m["WEIGHT"] = pd.to_numeric(m["WEIGHT"], errors="coerce").fillna(0.0)
    m["P"] = m["IND_VALUE"] * m["WEIGHT"]
    out = m.groupby(["YEAR", "IND_SUBCAT"], as_index=False).agg(NUM=("P", "sum"), DEN=("WEIGHT", "sum"), MEAN=("IND_VALUE", "mean"))
    out["VALUE"] = np.where(out["DEN"] > 0, out["NUM"] / out["DEN"], out["MEAN"])
    epp_pivot = out.pivot(index="YEAR", columns="IND_SUBCAT", values="VALUE").sort_index().fillna(0.0)

    # Enrollment: sum for chart; mean for table
    lines_sum, lines_mean = {}, {}
    for key, label in ENROLL_KEYS:
        sub = df[
            (df["IND_CAT"].str.lower() == "student enrollment")
            & (df["IND_SUBCAT"].str.lower() == key)
            & (df["DIST_NAME"].str.lower().isin(members))
        ][["DIST_NAME", "YEAR", "IND_VALUE"]]
        if sub.empty:
            continue
        lines_sum[label] = sub.groupby("YEAR")["IND_VALUE"].sum().sort_index()
        lines_mean[label] = sub.groupby("YEAR")["IND_VALUE"].mean().sort_index()

    return title, epp_pivot, lines_sum, lines_mean


# ---------------- Baseline helpers ----------------
def baseline_map_from_epp(epp_pivot: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    m: Dict[str, Dict[str, float]] = {}
    if epp_pivot is None or epp_pivot.empty:
        return m
    for sc in epp_pivot.columns:
        s = epp_pivot[sc]
        m[sc] = {"5": compute_cagr_last(s, 5), "10": compute_cagr_last(s, 10), "15": compute_cagr_last(s, 15)}
    return m

def latest_total_fte_bucket(df: pd.DataFrame, dist: str) -> str:
    ddf = df[
        (df["DIST_NAME"].str.lower() == dist.lower())
        & (df["IND_CAT"].str.lower() == "student enrollment")
        & (df["IND_SUBCAT"].str.lower() == "total fte pupils")
    ]
    if ddf.empty:
        idf = df[
            (df["DIST_NAME"].str.lower() == dist.lower())
            & (df["IND_CAT"].str.lower() == "student enrollment")
            & (df["IND_SUBCAT"].str.lower().isin(["in-district fte pupils", "out-of-district fte pupils"]))
        ][["YEAR", "IND_VALUE"]]
        if idf.empty:
            return "gt_500"
        s = idf.groupby("YEAR")["IND_VALUE"].sum()
        latest = float(s.loc[s.index.max()])
    else:
        y = int(ddf["YEAR"].max())
        latest = float(ddf.loc[ddf["YEAR"] == y, "IND_VALUE"].iloc[0])
    return "le_500" if latest <= 500 else "gt_500"


# ---------------- Page assembly ----------------
def build_page_dicts(df: pd.DataFrame, reg: pd.DataFrame) -> List[dict]:
    pages: List[dict] = []

    # Precompute baselines for Western Traditional buckets (<=500, >500)
    baseline: Dict[str, dict] = {}
    for bucket in ("le_500", "gt_500"):
        title_w, epp_w, _ls, _lm = prepare_western(df, reg, bucket, "Traditional")
        baseline[bucket] = {"title": title_w, "map": baseline_map_from_epp(epp_w)}

    # 1) Western Traditional pages FIRST (compose only these)
    for size_bucket, school_type, key in [
        ("le_500", "Traditional", "Western_Traditional_le_500"),
        ("gt_500", "Traditional", "Western_Traditional_gt_500"),
    ]:
        title, epp, lines_sum, lines_mean = prepare_western(df, reg, size_bucket, school_type)
        if epp.empty and not lines_sum:
            continue

        subcats = order_subcats_by_mean(epp)
        color_map = make_color_map(subcats, PALETTE_REGION, COLOR_OVERRIDES_REGION)

        years = sorted(set(epp.index) | set().union(*(set(s.index) for s in lines_sum.values() if s is not None)))
        epp = epp.reindex(years).fillna(0.0)
        latest_year = int(epp.index.max()) if not epp.empty else 0

        latest_vals = [epp.loc[latest_year, sc] if sc in epp.columns else 0.0 for sc in subcats]
        c5 = [compute_cagr_last(epp[sc], 5) for sc in subcats]
        c10 = [compute_cagr_last(epp[sc], 10) for sc in subcats]
        c15 = [compute_cagr_last(epp[sc], 15) for sc in subcats]

        rev = list(reversed(subcats))
        cat_rows = [
            (sc, f"${epp.loc[latest_year, sc]:,.0f}", fmt_pct(c5[subcats.index(sc)]),
             fmt_pct(c10[subcats.index(sc)]), fmt_pct(c15[subcats.index(sc)]), color_map[sc])
            for sc in rev
        ]
        total_latest = float(np.nansum(latest_vals)) if latest_vals else 0.0

        def mean_clean(arr):
            vs = [v for v in arr if v is not None and not (isinstance(v, float) and np.isnan(v))]
            return float(np.mean(vs)) if vs else np.nan

        cat_total = ("Total", f"${total_latest:,.0f}", fmt_pct(mean_clean(c5)), fmt_pct(mean_clean(c10)), fmt_pct(mean_clean(c15)))

        # FTE table uses MEAN for Western pages
        fte_years = [int(s.index.max()) for s in lines_mean.values() if s is not None and not s.empty]
        latest_fte_year = max(fte_years) if fte_years else latest_year
        fte_rows = []
        for _k, label in ENROLL_KEYS:
            s = lines_mean.get(label)
            if s is None or s.empty:
                continue
            r5 = compute_cagr_last(s, 5)
            r10 = compute_cagr_last(s, 10)
            r15 = compute_cagr_last(s, 15)
            val = "—" if latest_fte_year not in s.index else f"{float(s.loc[latest_fte_year]):,.0f}"
            fte_rows.append((LINE_COLORS_REGION[label], label, val, fmt_pct(r5), fmt_pct(r10), fmt_pct(r15)))

        pages.append(
            dict(
                title=title,
                chart_path=str(regional_png(key)),
                latest_year=latest_year,
                latest_year_fte=latest_fte_year,
                cat_rows=cat_rows,
                cat_total=cat_total,
                fte_rows=fte_rows,
                page_type="western",
            )
        )

    # 2) Then individual districts
    for dist in DISTRICTS:
        epp, lines = prepare_district(df, dist)
        if epp.empty and not lines:
            continue

        subcats = order_subcats_by_mean(epp)
        if dist == "Amherst-Pelham":
            palette, overrides, line_cols = PALETTE_APEL, COLOR_OVERRIDES_APEL, LINE_COLORS_APEL
            title = "Amherst-Pelham Regional: Expenditures Per Pupil vs Enrollment"
        else:
            palette, overrides, line_cols = PALETTE_DIST, COLOR_OVERRIDES_DIST, LINE_COLORS_DIST
            title = f"{dist}: Expenditures Per Pupil vs Enrollment"

        color_map = make_color_map(subcats, palette, overrides)

        years = sorted(set(epp.index) | set().union(*(set(s.index) for s in lines.values() if s is not None)))
        epp = epp.reindex(years).fillna(0.0)
        latest_year = int(epp.index.max()) if not epp.empty else 0

        latest_vals = [epp.loc[latest_year, sc] if sc in epp.columns else 0.0 for sc in subcats]
        c5 = [compute_cagr_last(epp[sc], 5) for sc in subcats]
        c10 = [compute_cagr_last(epp[sc], 10) for sc in subcats]
        c15 = [compute_cagr_last(epp[sc], 15) for sc in subcats]

        rev = list(reversed(subcats))
        cat_rows = [
            (sc, f"${epp.loc[latest_year, sc]:,.0f}", fmt_pct(c5[subcats.index(sc)]),
             fmt_pct(c10[subcats.index(sc)]), fmt_pct(c15[subcats.index(sc)]), color_map[sc])
            for sc in rev
        ]
        total_latest = float(np.nansum(latest_vals)) if latest_vals else 0.0

        def mean_clean(arr):
            vs = [v for v in arr if v is not None and not (isinstance(v, float) and np.isnan(v))]
            return float(np.mean(vs)) if vs else np.nan

        cat_total = ("Total", f"${total_latest:,.0f}", fmt_pct(mean_clean(c5)), fmt_pct(mean_clean(c10)), fmt_pct(mean_clean(c15)))

        fte_years = [int(s.index.max()) for s in lines.values() if s is not None and not s.empty]
        latest_fte_year = max(fte_years) if fte_years else latest_year
        fte_rows = []
        for _k, label in ENROLL_KEYS:
            s = lines.get(label)
            if s is None or s.empty:
                continue
            r5 = compute_cagr_last(s, 5)
            r10 = compute_cagr_last(s, 10)
            r15 = compute_cagr_last(s, 15)
            fte_rows.append(
                (line_cols[label], label,
                 ("—" if latest_fte_year not in s.index else f"{float(s.loc[latest_fte_year]):,.0f}"),
                 fmt_pct(r5), fmt_pct(r10), fmt_pct(r15))
            )

        bucket = latest_total_fte_bucket(df, dist)
        base_title = baseline.get(bucket, {}).get(
            "title",
            "All Western MA Traditional Districts ≤500 Students" if bucket == "le_500" else "All Western MA Traditional Districts >500 Students",
        )
        base_map = baseline.get(bucket, {}).get("map", {})

        pages.append(
            dict(
                title=title,
                chart_path=str(district_png(dist)),
                latest_year=latest_year,
                latest_year_fte=latest_fte_year,
                cat_rows=cat_rows,
                cat_total=cat_total,
                fte_rows=fte_rows,
                page_type="district",
                baseline_title=base_title,
                baseline_map=base_map,
            )
        )

    return pages


# ---------------- Formatting helpers ----------------
def para_num_auto(text: str) -> Paragraph:
    t = (text or "").strip()
    return Paragraph(text, style_num_neg if t.startswith("-") else style_num)


# ---------------- Build PDF ----------------
def build_pdf(pages: List[dict], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=0.5 * inch,
        rightMargin=0.5 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
        allowSplitting=1,
    )

    story: List = []
    for idx, p in enumerate(pages):
        # Title + Subtitle
        main_title, sub_title = p["title"], "Expenditures Per Pupil vs Enrollment"
        if ":" in p["title"]:
            left, right = p["title"].split(":", 1)
            main_title, sub_title = left.strip(), right.strip()
        if ("Western" in p["title"]) and ("Traditional" in p["title"]):
            sub_title = f"{sub_title} — Not including charters and vocationals"

        story.append(Paragraph(main_title, style_title_main))
        story.append(Paragraph(sub_title, style_title_sub))

        # Chart (cap height)
        img_path = Path(p["chart_path"])
        if not img_path.exists():
            story.append(Paragraph(f"[Missing chart image: {img_path.name}]", style_body))
        else:
            im = Image(str(img_path))
            ratio = im.imageHeight / float(im.imageWidth)
            im.drawWidth = doc.width
            im.drawHeight = doc.width * ratio
            max_chart_h = doc.height * 0.40
            if im.drawHeight > max_chart_h:
                im.drawHeight = max_chart_h
                im.drawWidth = im.drawHeight / ratio
            story.append(im)

        story.append(Spacer(0, 24))

        # -------- Category table --------
        cat_header = ["", "Category", f"{p['latest_year']} $/pupil", "5y CAGR", "10y CAGR", "15y CAGR"]

        cat_rows_rl = []
        for (label, latest, c5, c10, c15, hexcol) in p["cat_rows"]:
            cat_rows_rl.append(
                [
                    ColorSwatch(hexcol),
                    Paragraph(label, style_body),
                    Paragraph(latest, style_num),
                    para_num_auto(c5),
                    para_num_auto(c10),
                    para_num_auto(c15),
                ]
            )

        total = ["", Paragraph(p["cat_total"][0], style_body)] + [
            Paragraph(p["cat_total"][1], style_num),
            para_num_auto(p["cat_total"][2]),
            para_num_auto(p["cat_total"][3]),
            para_num_auto(p["cat_total"][4]),
        ]

        # Legend rows (district pages only) — label RIGHT under Category col; swatch spans all CAGR cols (3..5)
        legend_rows = []
        if p.get("page_type") == "district":
            base_label = p.get("baseline_title", "Western baseline")
            legend_rows = [
                ["", Paragraph(f"Greater than CAGR for {base_label}", style_legend), "", "", "", ""],
                ["", Paragraph(f"Less than CAGR for {base_label}",  style_legend), "", "", "", ""],
            ]

        cat_data = [cat_header] + cat_rows_rl + [total] + legend_rows

        cat_tbl = Table(
            cat_data,
            colWidths=[
                0.08 * doc.width,  # swatch
                0.42 * doc.width,  # category (legend label lives here)
                0.18 * doc.width,  # latest $/pupil
                0.10 * doc.width,  # 5y
                0.10 * doc.width,  # 10y
                0.12 * doc.width,  # 15y (legend swatch spans 3..5)
            ],
            hAlign="LEFT",
            repeatRows=1,
            splitByRow=1,
        )

        ts = [
            ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 9),
            ("ALIGN", (2, 0), (-1, 0), "RIGHT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN", (2, 1), (-1, -1), "RIGHT"),
            ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
            ("ROWSPACING", (0, 1), (-1, -1), 2),
        ]

        total_row_idx = 1 + len(cat_rows_rl)
        ts += [
            ("BACKGROUND", (0, total_row_idx), (-1, total_row_idx), colors.whitesmoke),
            ("FONT", (0, total_row_idx), (-1, total_row_idx), "Helvetica-Bold", 9),
        ]

        # Legend formatting (only if legend rows exist)
        if legend_rows:
            red_row_idx = total_row_idx + 1
            grn_row_idx = total_row_idx + 2
            ts += [
                # Label stays only under Category col; right-aligned (style_legend already right)
                # Fill the three CAGR columns as a large swatch
                ("BACKGROUND", (3, red_row_idx), (5, red_row_idx), HexColor(RED_SHADES[2])),
                ("BACKGROUND", (3, grn_row_idx), (5, grn_row_idx), HexColor(GRN_SHADES[2])),
                ("TOPPADDING", (0, red_row_idx), (-1, grn_row_idx), 2),
                ("BOTTOMPADDING", (0, red_row_idx), (-1, grn_row_idx), 2),
            ]

        # Relative-to-Western shading on district CAGR cells with intensity
        if p.get("page_type") == "district":
            base_map = p.get("baseline_map", {})
            for row_idx, row_raw in enumerate(p["cat_rows"], start=1):
                subcat = row_raw[0]
                base = base_map.get(subcat)
                if not base:
                    continue
                d5 = parse_pct_str_to_float(row_raw[2])
                d10 = parse_pct_str_to_float(row_raw[3])
                d15 = parse_pct_str_to_float(row_raw[4])
                for col, dv, bv in ((3, d5, base.get("5")), (4, d10, base.get("10")), (5, d15, base.get("15"))):
                    if dv == dv and bv == bv:  # both finite
                        shade = shade_for_delta(dv - bv)
                        if shade is not None:
                            ts.append(("BACKGROUND", (col, row_idx), (col, row_idx), shade))

        cat_tbl.setStyle(TableStyle(ts))
        story.append(cat_tbl)
        story.append(Spacer(0, 24))

        # -------- FTE table --------
        fte_header_label = f"{p['latest_year_fte']} FTE" if p.get("page_type") == "district" else f"{p['latest_year_fte']} avg FTE"
        fte_header = ["", "Pupil Group", fte_header_label, "5y CAGR", "10y CAGR", "15y CAGR"]
        fte_data = [fte_header]
        for (hexcol, label, fte, c5, c10, c15) in p["fte_rows"]:
            fte_data.append(
                [
                    LineSwatch(hexcol),
                    Paragraph(label, style_body),
                    Paragraph(fte, style_num),
                    para_num_auto(c5),
                    para_num_auto(c10),
                    para_num_auto(c15),
                ]
            )

        fte_tbl = Table(
            fte_data,
            colWidths=[0.16 * doc.width, 0.34 * doc.width, 0.18 * doc.width, 0.10 * doc.width, 0.10 * doc.width, 0.12 * doc.width],
            hAlign="LEFT",
            repeatRows=1,
            splitByRow=1,
        )
        fte_tbl.setStyle(
            TableStyle(
                [
                    ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 9),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("ALIGN", (2, 1), (-1, -1), "RIGHT"),
                    ("ALIGN", (2, 0), (-1, 0), "RIGHT"),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
                    ("ROWSPACING", (0, 1), (-1, -1), 2),
                ]
            )
        )

        story.append(fte_tbl)
        story.append(Spacer(0, 24))

        if idx < len(pages) - 1:
            story.append(PageBreak())

    doc.build(story, onFirstPage=draw_footer, onLaterPages=draw_footer)
    print(f"[OK] Wrote PDF: {out_path}")


# ---------------- Main ----------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df, reg = load_data()
    pages = build_page_dicts(df, reg)
    if not pages:
        print("[WARN] No pages to write.")
        return
    build_pdf(pages, OUTPUT_PDF)


if __name__ == "__main__":
    main()
