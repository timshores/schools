"""
district_expend_pp_stack.py

Single-district series:
  Stacked bars = Expenditures Per Pupil by IND_SUBCAT (excluding totals)
  Lines = In-District and Out-of-District FTE
  Right panel = 2 tables:
      - Category $/pupil (latest year) + 5y/10y/15y CAGRs (plus Total row)
      - FTE (latest year) + 5y/10y/15y CAGRs with colored line/marker swatches

Special case:
  Amherst-Pelham uses a DISTINCT palette and line colors.
  Title becomes "Amherst-Pelham Regional: Expenditures Per Pupil vs Enrollment"
  (This plot appears FIRST in the series.)

Regional series (Western MA), split by SCHOOL_TYPE in this order:
  - Traditional ≤500 FTE
  - Traditional >500 FTE
  - Charter (all FTE)
  - Vocational (all FTE)

  Source sheet: "District Regions" (filter eohhs_region == "Western")
  Keep only districts present in "District Expend by Category"
  Charts:
      Bars = mean EPP per category across districts
              (default: enrollment-weighted by In-District FTE)
      Lines = total FTE In-District and Out-of-District (sum across districts)
  Tables:
      Category $/pupil = Mean (weighted by In-District FTE by default)
      FTE = Mean across districts (average district size)

Outputs:
- One PNG per plot in ./output/
- A single landscape PDF with one plot per page: ./output/expenditures_series.pdf
"""

from __future__ import annotations

import os
import sys
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ---------------------------
# Config
# ---------------------------

DATA_DIR = Path("./data")
OUTPUT_DIR = Path("./output")
EXCEL_FILE = DATA_DIR / "E2C_Hub_MA_DESE_Data.xlsx"

SHEET_EXPEND = "District Expend by Category"
SHEET_REGIONS = "District Regions"

# PDF output (multi-page)
OUTPUT_PDF = OUTPUT_DIR / "expenditures_series.pdf"

# Single-districts of interest (Amherst-Pelham first)
DISTRICTS = ["Amherst-Pelham", "Amherst", "Leverett", "Pelham", "Shutesbury"]

# Exclude these from the stacked bars
EXCLUDE_SUBCATS = {"total expenditures", "total in-district expenditures"}

# Enrollment subcategories to plot (omit Total)
ENROLL_KEYS = [
    ("in-district fte pupils", "In-District FTE Pupils"),
    ("out-of-district fte pupils", "Out-of-District FTE Pupils"),
]
TOTAL_FTE_KEY = "total fte pupils"  # used only for grouping

# --- Single-district color theme ---
LINE_COLORS_DIST = {
    "In-District FTE Pupils": "#000000",   # black
    "Out-of-District FTE Pupils": "#AA0000",  # deep red
}
COLOR_OVERRIDES_DIST = {"professional development": "#6A3D9A"}  # deep purple
PALETTE_DIST = [
    "#0072B2", "#009E73", "#D55E00", "#CC79A7", "#F0E442", "#56B4E9",
    "#E69F00", "#999999", "#332288", "#88CCEE", "#44AA99", "#882255"
]

# --- Amherst-Pelham special theme (distinct from both above palettes) ---
LINE_COLORS_APEL = {
    "In-District FTE Pupils": "#1F4E79",   # dark blue
    "Out-of-District FTE Pupils": "#D35400",  # burnt orange
}
COLOR_OVERRIDES_APEL = {"professional development": "#2A9D8F"}  # teal-ish
PALETTE_APEL = [
    "#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51", "#8AB17D",
    "#7A5195", "#4ECDC4", "#C7F464", "#FF6B6B", "#FFE66D", "#355C7D"
]

# --- Regional color theme (distinct from district & APEL) ---
LINE_COLORS_REGION = {
    "In-District FTE Pupils": "#1B7837",  # dark green
    "Out-of-District FTE Pupils": "#5E3C99",  # purple
}
COLOR_OVERRIDES_REGION = {"professional development": "#1B9E77"}  # teal-ish
PALETTE_REGION = [
    "#8DD3C7", "#FFFFB3", "#BEBADA", "#FB8072", "#80B1D3", "#FDB462",
    "#B3DE69", "#FCCDE5", "#D9D9D9", "#BC80BD", "#CCEBC5", "#FFED6F"
]

# Styling
plt.rcParams.update({
    "figure.dpi": 150,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlelocation": "left",
    "axes.titlepad": 10,
})

# Layout tuning
RIGHT_PANEL_LEFT_PAD = 0.06  # whitespace between plot and tables

# Chart line styling (also used to draw table swatches)
FTE_LINEWIDTH = 2.8
FTE_MARKER_SIZE = 6
FTE_MARKER_EDGEWIDTH = 1.2

# --- Regional EPP mean method: "weighted" (by In-District FTE) or "simple"
REGION_EPP_MEAN_METHOD = "weighted"

# Expected SCHOOL_TYPE values and display order
SCHOOL_TYPES_ORDER = ["Traditional", "Charter", "Vocational"]


# ---------------------------
# Helpers
# ---------------------------

def find_sheet_name(xls: pd.ExcelFile, target: str) -> str:
    names = xls.sheet_names
    for n in names:
        if n.strip().lower() == target.strip().lower():
            return n
    t = target.strip().lower()
    for n in names:
        nl = n.strip().lower()
        if t in nl or nl in t:
            return n
    return names[0]


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
            if re.search(r"\b(19|20)\d{2}\b", sample) or re.search(r"(19|20)\d{2}\s*[–-]\s*\d{2,4}", sample):
                chosen = c
                break
    if chosen is None:
        raise ValueError("Could not find a year column ('SY' or 'FISCAL_YEAR' or 'YEAR').")

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
            y2raw = m.group(2)
            if len(y2raw) == 2:
                y2 = (y1 // 100) * 100 + int(y2raw)
                if (y2 % 100) < (y1 % 100):
                    y2 += 100
            else:
                y2 = int(y2raw)
            return y2
        m2 = re.search(r"(?i)\bsy\b\s*(\d{2})\s*[–\-/]?\s*(\d{2})", s)
        if m2:
            a = int(m2.group(1)); b = int(m2.group(2))
            century = 2000
            y1 = century + a; y2 = century + b
            if b < a:
                y2 += 100
            return y2
        m3 = re.search(r"\b((?:19|20)\d{2})\b", s)
        if m3:
            return int(m3.group(1))
        return np.nan

    df["YEAR"] = df[chosen].apply(parse_to_year)
    df = df.dropna(subset=["YEAR"]).copy()
    df["YEAR"] = df["YEAR"].astype(int)
    return df


def clean_strings(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df


def format_dollars(ax):
    import matplotlib.ticker as mtick
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))


def format_people(ax):
    import matplotlib.ticker as mtick
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))


def color_luminance(hex_color: str) -> float:
    hc = hex_color.lstrip('#')
    r = int(hc[0:2], 16) / 255.0
    g = int(hc[2:4], 16) / 255.0
    b = int(hc[4:6], 16) / 255.0
    def s2l(x): return x/12.92 if x <= 0.04045 else ((x+0.055)/1.055)**2.4
    R, G, B = s2l(r), s2l(g), s2l(b)
    return 0.2126*R + 0.7152*G + 0.0722*B


def compute_cagr_last(series: pd.Series, years_back: int) -> float | np.nan:
    """
    CAGR over the last `years_back` years, using the latest year Y and
    the latest available start in [Y-years_back, Y).
    Falls back to the earliest available year if there is no point in the window.
    """
    if series is None or series.empty:
        return np.nan
    s = series.sort_index().astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if s.size < 2:
        return np.nan
    end_year = int(s.index.max())
    candidates = s[s.index <= end_year - years_back]
    if not candidates.empty:
        start_year = int(candidates.index.max())
    else:
        start_year = int(s.index.min())
    n_years = end_year - start_year
    if n_years <= 0:
        return np.nan
    start_val = float(s.loc[start_year])
    end_val = float(s.loc[end_year])
    if start_val <= 0 or end_val <= 0:
        return np.nan
    return (end_val / start_val) ** (1.0 / n_years) - 1.0


# ---------------------------
# Data loading & shaping
# ---------------------------

def load_expend_data() -> pd.DataFrame:
    if not EXCEL_FILE.exists():
        raise FileNotFoundError(f"Missing Excel file: {EXCEL_FILE}")
    xls = pd.ExcelFile(EXCEL_FILE)
    sheet = find_sheet_name(xls, SHEET_EXPEND)
    df = pd.read_excel(xls, sheet_name=sheet)
    df.columns = [str(c).strip() for c in df.columns]

    canon = {}
    for c in df.columns:
        low = str(c).strip().lower()
        if low == "dist_name":
            canon[c] = "DIST_NAME"
        elif low == "ind_cat":
            canon[c] = "IND_CAT"
        elif low == "ind_subcat":
            canon[c] = "IND_SUBCAT"
        elif low == "ind_value":
            canon[c] = "IND_VALUE"
        elif low in ("sy", "school_year", "fiscal_year", "year"):
            canon[c] = c
    df = df.rename(columns=canon)
    if "IND_VALUE" not in df.columns:
        raise ValueError("IND_VALUE column not found in the sheet.")

    df = coalesce_year_column(df)
    df = clean_strings(df, ["DIST_NAME", "IND_CAT", "IND_SUBCAT"])
    df["IND_VALUE"] = pd.to_numeric(df["IND_VALUE"], errors="coerce")
    return df


def load_regions() -> pd.DataFrame:
    xls = pd.ExcelFile(EXCEL_FILE)
    sheet = find_sheet_name(xls, SHEET_REGIONS)
    reg = pd.read_excel(xls, sheet_name=sheet)
    reg.columns = [str(c).strip() for c in reg.columns]
    cmap = {}
    for c in reg.columns:
        low = c.strip().lower()
        if low in ("district_name", "dist_name", "name"):
            cmap[c] = "DIST_NAME"
        elif low in ("eohhs_region", "region"):
            cmap[c] = "EOHHS_REGION"
        elif low in ("school_type", "type", "district_type", "org_type"):
            cmap[c] = "SCHOOL_TYPE"
    reg = reg.rename(columns=cmap)
    required = {"DIST_NAME", "EOHHS_REGION", "SCHOOL_TYPE"}
    missing = required - set(reg.columns)
    if missing:
        raise ValueError(f"Expected columns {sorted(required)} in 'District Regions'. Missing: {sorted(missing)}")
    for c in ["DIST_NAME", "EOHHS_REGION", "SCHOOL_TYPE"]:
        reg[c] = reg[c].astype(str).str.strip()
    return reg


def prepare_epp_and_enroll(df: pd.DataFrame, district: str) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
    ddf = df[df["DIST_NAME"].str.lower() == district.lower()].copy()

    # EPP stacked bars
    epp = ddf[ddf["IND_CAT"].str.lower() == "expenditures per pupil"].copy()
    epp["IND_SUBCAT"] = epp["IND_SUBCAT"].replace({np.nan: "Unspecified"}).fillna("Unspecified")
    epp = epp[~epp["IND_SUBCAT"].str.lower().isin(EXCLUDE_SUBCATS)].copy()
    epp_pivot = (
        epp.pivot_table(index="YEAR", columns="IND_SUBCAT", values="IND_VALUE", aggfunc="sum")
           .sort_index()
           .fillna(0.0)
    )

    # Enrollment lines
    enroll_lines: Dict[str, pd.Series] = {}
    enroll_ddf = ddf[ddf["IND_CAT"].str.lower() == "student enrollment"].copy()
    if not enroll_ddf.empty:
        for key, label in ENROLL_KEYS:
            ser = enroll_ddf[enroll_ddf["IND_SUBCAT"].str.lower() == key][["YEAR", "IND_VALUE"]]
            if not ser.empty:
                ser = (ser.dropna()
                         .drop_duplicates(subset=["YEAR"])
                         .set_index("YEAR")["IND_VALUE"]
                         .sort_index()
                         .round()
                         .astype(int))
                enroll_lines[label] = ser

    return epp_pivot, enroll_lines


def epp_group_pivot_mean(
    epp_df: pd.DataFrame,
    weights_df: Optional[pd.DataFrame],
    method: str
) -> pd.DataFrame:
    """
    epp_df: rows with DIST_NAME, YEAR, IND_SUBCAT, IND_VALUE (per-pupil)
    weights_df: rows with DIST_NAME, YEAR, WEIGHT (In-District FTE)
    method: "weighted" (by WEIGHT) or "simple"
    """
    if method == "weighted" and weights_df is not None and not weights_df.empty:
        m = epp_df.merge(weights_df, on=["DIST_NAME", "YEAR"], how="left")
        m["WEIGHT"] = pd.to_numeric(m["WEIGHT"], errors="coerce").fillna(0.0)

        def wmean(g):
            wsum = float(g["WEIGHT"].sum())
            if wsum <= 0:
                return float(g["IND_VALUE"].mean())
            return float((g["IND_VALUE"] * g["WEIGHT"]).sum() / wsum)

        out = (
            m.groupby(["YEAR", "IND_SUBCAT"], as_index=False)
             .apply(lambda g: pd.Series({"VALUE": wmean(g)}))
        )
        pivot = out.pivot(index="YEAR", columns="IND_SUBCAT", values="VALUE").sort_index().fillna(0.0)
        return pivot

    # Simple mean
    return (
        epp_df.pivot_table(index="YEAR", columns="IND_SUBCAT", values="IND_VALUE", aggfunc="mean")
              .sort_index()
              .fillna(0.0)
    )


def prepare_group_western(
    df: pd.DataFrame,
    reg: pd.DataFrame,
    size_bucket: str,         # "le_500", "gt_500", or "all"
    school_type: str          # "Traditional", "Charter", "Vocational"
) -> Tuple[str, pd.DataFrame, Dict[str, pd.Series], Dict[str, pd.Series], List[str]]:
    """
    Returns:
      title,
      epp_pivot_mean (by category, YEAR),
      enroll_lines_sum (for chart),
      enroll_lines_mean (for FTE table),
      member_district_pretty_names
    """
    # Western + school_type districts present in expend sheet
    mask_west = reg["EOHHS_REGION"].str.lower() == "western"
    mask_type = reg["SCHOOL_TYPE"].str.strip().str.lower() == school_type.strip().lower()
    western = reg[mask_west & mask_type].copy()

    western_names = set(western["DIST_NAME"].str.strip().str.lower())
    present = set(df["DIST_NAME"].str.strip().str.lower())
    western_present = sorted({n for n in western_names if n in present})

    # Latest Total FTE per district (or sum In+Out)
    enroll = df[df["IND_CAT"].str.lower() == "student enrollment"].copy()
    totals = {}
    for name in western_present:
        dsub = enroll[enroll["DIST_NAME"].str.lower() == name]
        tot = dsub[dsub["IND_SUBCAT"].str.lower() == TOTAL_FTE_KEY]
        if not tot.empty:
            y = int(tot["YEAR"].max())
            totals[name] = float(tot.loc[tot["YEAR"] == y, "IND_VALUE"].iloc[0])
        else:
            tmp = {}
            for key, _label in ENROLL_KEYS:
                ser = dsub[dsub["IND_SUBCAT"].str.lower() == key][["YEAR", "IND_VALUE"]]
                if not ser.empty:
                    tmp[key] = ser.set_index("YEAR")["IND_VALUE"].sort_index()
            if tmp:
                years = set()
                for s in tmp.values():
                    years |= set(s.index)
                if years:
                    y = max(years)
                    tot_val = 0.0
                    for s in tmp.values():
                        if y in s.index:
                            tot_val += float(s.loc[y])
                    totals[name] = tot_val

    # Size filter
    if size_bucket == "le_500":
        members = [n for n, v in totals.items() if v <= 500]
        title_suffix = "≤500 FTE"
    elif size_bucket == "gt_500":
        members = [n for n, v in totals.items() if v > 500]
        title_suffix = ">500 FTE"
    else:
        members = list(totals.keys())
        title_suffix = None

    school_type_title = school_type.capitalize()
    if title_suffix:
        title = f"All Western MA {school_type_title} Districts {title_suffix}"
    else:
        title = f"All Western MA {school_type_title} Districts"

    if not members:
        return title, pd.DataFrame(), {}, {}, []

    # EPP rows for members
    epp = df[
        (df["IND_CAT"].str.lower() == "expenditures per pupil") &
        (df["DIST_NAME"].str.lower().isin(members))
    ][["DIST_NAME", "YEAR", "IND_SUBCAT", "IND_VALUE"]].copy()
    epp = epp[~epp["IND_SUBCAT"].str.lower().isin(EXCLUDE_SUBCATS)].copy()

    # Weights for weighted mean: In-District FTE
    weights = df[
        (df["IND_CAT"].str.lower() == "student enrollment") &
        (df["IND_SUBCAT"].str.lower() == "in-district fte pupils") &
        (df["DIST_NAME"].str.lower().isin(members))
    ][["DIST_NAME", "YEAR", "IND_VALUE"]].rename(columns={"IND_VALUE": "WEIGHT"}).copy()

    epp_pivot = epp_group_pivot_mean(epp, weights, REGION_EPP_MEAN_METHOD)

    # Enrollment series for chart: SUM across members
    enroll_lines_sum: Dict[str, pd.Series] = {}
    # Enrollment series for table: MEAN across members (average district size)
    enroll_lines_mean: Dict[str, pd.Series] = {}

    for key, label in ENROLL_KEYS:
        sub = df[
            (df["IND_CAT"].str.lower() == "student enrollment") &
            (df["IND_SUBCAT"].str.lower() == key) &
            (df["DIST_NAME"].str.lower().isin(members))
        ][["DIST_NAME", "YEAR", "IND_VALUE"]]

        if not sub.empty:
            sum_series = sub.groupby("YEAR")["IND_VALUE"].sum().sort_index().round().astype(int)
            mean_series = sub.groupby("YEAR")["IND_VALUE"].mean().sort_index()
            enroll_lines_sum[label] = sum_series
            enroll_lines_mean[label] = mean_series

    pretty_members = [df[df['DIST_NAME'].str.lower()==m]['DIST_NAME'].iloc[0] for m in members]
    return title, epp_pivot, enroll_lines_sum, enroll_lines_mean, pretty_members


def order_subcats_by_mean(epp_pivot: pd.DataFrame) -> List[str]:
    if epp_pivot.shape[1] <= 1:
        return list(epp_pivot.columns)
    return list(epp_pivot.mean(axis=0).sort_values(ascending=False).index)


# ---------------------------
# Plotting helpers
# ---------------------------

def draw_two_tables_and_source(
    fig, right_axes, years, epp_pivot, color_map,
    line_colors, enroll_lines_chart,
    enroll_lines_table=None,
    label_dollar_col="",
    label_fte_col="",
    right_panel_left_pad=RIGHT_PANEL_LEFT_PAD
):
    """Right-side tables (category + FTE) and source block."""
    if enroll_lines_table is None:
        enroll_lines_table = enroll_lines_chart

    right_axes.axis("off")
    bbox = right_axes.get_position(fig)
    x0 = bbox.x0 + right_panel_left_pad * bbox.width
    w = bbox.width * (1.0 - right_panel_left_pad)
    inner_top    = fig.add_axes([x0, bbox.y0 + 0.32 * bbox.height, w, 0.68 * bbox.height])
    inner_bottom = fig.add_axes([x0, bbox.y0 + 0.10 * bbox.height, w, 0.22 * bbox.height])
    for ax in (inner_top, inner_bottom):
        ax.axis("off")

    latest_year = max([y for y in years if y in epp_pivot.index])

    # Category table rows
    subcats = list(epp_pivot.columns)
    latest_vals = [epp_pivot.loc[latest_year, sc] if sc in epp_pivot.columns else 0.0 for sc in subcats]
    cagr5  = [compute_cagr_last(epp_pivot[sc], 5)  for sc in subcats]
    cagr10 = [compute_cagr_last(epp_pivot[sc], 10) for sc in subcats]
    cagr15 = [compute_cagr_last(epp_pivot[sc], 15) for sc in subcats]

    disp_subcats = list(reversed(subcats))
    disp_colors  = [color_map[sc] for sc in reversed(subcats)]
    disp_latest  = list(reversed(latest_vals))
    disp_c5      = list(reversed(cagr5))
    disp_c10     = list(reversed(cagr10))
    disp_c15     = list(reversed(cagr15))

    def fmt_pct(v): return "—" if (v is None or (isinstance(v, float) and np.isnan(v))) else f"{v*100:+.1f}%"

    cat_rows = list(zip(
        disp_subcats,
        [f"${v:,.0f}" for v in disp_latest],
        [fmt_pct(v) for v in disp_c5],
        [fmt_pct(v) for v in disp_c10],
        [fmt_pct(v) for v in disp_c15],
    ))

    total_latest = float(np.nansum(latest_vals)) if len(latest_vals) else 0.0
    def mean_clean(arr):
        vals = [v for v in arr if (v is not None and not (isinstance(v, float) and np.isnan(v)))]
        return float(np.mean(vals)) if vals else np.nan
    avg5, avg10, avg15 = mean_clean(cagr5), mean_clean(cagr10), mean_clean(cagr15)
    cat_rows.append(("Total", f"${total_latest:,.0f}", fmt_pct(avg5), fmt_pct(avg10), fmt_pct(avg15)))

    cat_col_widths = [0.58, 0.14, 0.09, 0.09, 0.10]
    tbl1 = inner_top.table(
        cellText=cat_rows,
        colLabels=["Category", label_dollar_col, "5y CAGR", "10y CAGR", "15y CAGR"],
        cellLoc="left", loc="upper left",
        colWidths=cat_col_widths
    )
    tbl1.auto_set_font_size(False); tbl1.set_fontsize(9); tbl1.scale(1.0, 1.18)

    n_cat_rows = len(disp_subcats)
    for r, hexc in enumerate(disp_colors, start=1):
        L = color_luminance(hexc)
        cell = tbl1[(r, 0)]
        cell.set_facecolor(hexc); cell.get_text().set_color("black" if L > 0.5 else "white")
    total_row_idx = n_cat_rows + 1
    for c in range(len(cat_col_widths)):
        tcell = tbl1[(total_row_idx, c)]
        tcell.set_facecolor("#E6E6E6"); tcell.get_text().set_color("black")
        try: tcell.get_text().set_fontweight("bold")
        except Exception: pass
    for r in range(1, len(cat_rows) + 1):
        for c in range(1, len(cat_col_widths)):
            tbl1[(r, c)]._loc = 'right'

    # FTE table (use enroll_lines_table)
    fte_years = []
    for s in enroll_lines_table.values():
        if s is not None and not s.empty:
            fte_years.append(int(s.index.max()))
    latest_year_fte = max(fte_years) if fte_years else latest_year

    fte_rows = []
    fte_labels = []
    for _key, label in ENROLL_KEYS:
        s = enroll_lines_table.get(label)
        if s is None or s.empty:
            continue
        latest_val = float(s.get(latest_year_fte, np.nan))
        r5  = compute_cagr_last(s, 5)
        r10 = compute_cagr_last(s, 10)
        r15 = compute_cagr_last(s, 15)
        fte_rows.append(["", label,
                         ("—" if np.isnan(latest_val) else f"{latest_val:,.0f}"),
                         fmt_pct(r5), fmt_pct(r10), fmt_pct(r15)])
        fte_labels.append(label)

    if fte_rows:
        fte_col_widths = [0.10, 0.48, 0.14, 0.09, 0.09, 0.10]
        tbl2 = inner_bottom.table(
            cellText=fte_rows,
            colLabels=["", "Pupil Group", label_fte_col, "5y CAGR", "10y CAGR", "15y CAGR"],
            cellLoc="left", loc="upper left",
            colWidths=fte_col_widths
        )
        tbl2.auto_set_font_size(False); tbl2.set_fontsize(9); tbl2.scale(1.0, 1.18)
        for r in range(1, len(fte_rows) + 1):
            for c in (2, 3, 4, 5):
                tbl2[(r, c)]._loc = 'right'

        # Draw swatches inside swatch cells
        fig.canvas.draw()
        for r, label in enumerate(fte_labels, start=1):
            cell = tbl2[(r, 0)]
            renderer = fig.canvas.get_renderer()
            bbox_disp = cell.get_window_extent(renderer=renderer)
            (x0, y0) = inner_bottom.transAxes.inverted().transform((bbox_disp.x0, bbox_disp.y0))
            (x1, y1) = inner_bottom.transAxes.inverted().transform((bbox_disp.x1, bbox_disp.y1))
            pad = (x1 - x0) * 0.12
            xs, xe = x0 + pad, x1 - pad
            xm, ym = (xs + xe) / 2.0, (y0 + y1) / 2.0
            col = line_colors.get(label, "#000000")
            inner_bottom.plot([xs, xe], [ym, ym], transform=inner_bottom.transAxes,
                              color=col, linewidth=FTE_LINEWIDTH, solid_capstyle="round", zorder=8, clip_on=False)
            inner_bottom.plot([xm], [ym], transform=inner_bottom.transAxes,
                              marker="o", markersize=FTE_MARKER_SIZE,
                              markerfacecolor="white", markeredgecolor=col,
                              markeredgewidth=FTE_MARKER_EDGEWIDTH, color=col, zorder=9, clip_on=False)

    # Source block
    inner_bottom.text(
        0.0, -0.01,
        "Source: DESE District Expenditures by Spending Category\n"
        "Last updated: August 12, 2025\n"
        "https://educationtocareer.data.mass.gov/Finance-and-Budget/District-Expenditures-by-Spending-Category/er3w-dyti/",
        transform=inner_bottom.transAxes, fontsize=8, va="top", ha="left"
    )


def plot_one(
    title: str,
    epp_pivot: pd.DataFrame,
    enroll_lines_chart: Dict[str, pd.Series],
    enroll_lines_table: Optional[Dict[str, pd.Series]],
    palette: List[str],
    color_overrides: Dict[str, str],
    line_colors: Dict[str, str],
    y_max_dollars: Optional[float],
    y_max_people: Optional[float],
    label_dollar_col: str,
    label_fte_col: str,
) -> plt.Figure:
    """Generic plot routine used by both district and regional series. Returns the Figure."""
    if epp_pivot.empty and not enroll_lines_chart:
        raise ValueError(f"No data for '{title}'")

    # Collect union of years across bars and lines
    year_sets = [set(epp_pivot.index)]
    for s in enroll_lines_chart.values():
        year_sets.append(set(s.index))
    years = sorted(set().union(*year_sets))
    if not years:
        raise ValueError(f"No years for '{title}'")
    epp_pivot = epp_pivot.reindex(years).fillna(0.0)

    # Order categories & colors
    subcats = order_subcats_by_mean(epp_pivot)
    epp_pivot = epp_pivot.reindex(columns=subcats)
    base_colors = (palette * ((len(subcats) // len(palette)) + 1))[:len(subcats)]
    color_map = {sc: col for sc, col in zip(subcats, base_colors)}
    for sc in subcats:
        key = sc.strip().lower()
        if key in color_overrides:
            color_map[sc] = color_overrides[key]

    # Layout — landscape by figure size
    fig = plt.figure(figsize=(15.0, 6.8))
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[2.2, 3.6], wspace=0.14)
    axL = fig.add_subplot(gs[0, 0])
    axRight = fig.add_subplot(gs[0, 1])

    # Right $ axis for bars (drawn behind)
    axR = axL.twinx()
    axR.spines["right"].set_visible(True)
    axR.set_zorder(2)
    axL.set_zorder(3)
    axL.patch.set_alpha(0.0)

    # Clean look
    for a in (axL, axR):
        a.margins(x=0.0, y=0.0)
        a.grid(False)

    # Bars
    bottom = np.zeros(len(years))
    for sc in subcats:
        vals = epp_pivot[sc].values
        axR.bar(years, vals, bottom=bottom, width=0.7, align="center",
                zorder=1, color=color_map[sc], edgecolor="none")
        bottom += vals

    if y_max_dollars is not None and y_max_dollars > 0:
        axR.set_ylim(0, y_max_dollars)

    # Lines (chart uses totals or district values as passed)
    for _key, label in ENROLL_KEYS:
        series = enroll_lines_chart.get(label)
        if series is None:
            continue
        series = series.reindex(years)
        lc = line_colors.get(label, "#000000")
        axL.plot(
            years, series.values,
            marker="o", markersize=FTE_MARKER_SIZE, markerfacecolor="white",
            markeredgewidth=FTE_MARKER_EDGEWIDTH,
            linewidth=FTE_LINEWIDTH, color=lc, zorder=10, clip_on=False
        )

    if y_max_people is not None and y_max_people > 0:
        axL.set_ylim(0, y_max_people)

    # Labels
    format_people(axL); axL.set_ylabel("Student Enrollment (FTE)")
    format_dollars(axR); axR.set_ylabel("Expenditures per Pupil ($)")
    axL.set_xlabel("School Year")
    axL.set_title(title, loc="left")

    # Layout & tables
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.22)
    draw_two_tables_and_source(
        fig, axRight, years, epp_pivot, color_map, line_colors,
        enroll_lines_chart, enroll_lines_table,
        label_dollar_col=label_dollar_col, label_fte_col=label_fte_col
    )

    return fig


# ---------------------------
# Main
# ---------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_expend_data()
    cats = {"expenditures per pupil", "student enrollment"}
    df = df[df["IND_CAT"].str.lower().isin(cats)].copy()

    # ---- Single-district series (uniform left axis across all five) ----
    per_district: Dict[str, Tuple[pd.DataFrame, Dict[str, pd.Series]]] = {}
    shared_max_dollars = 0.0
    shared_max_people = 0.0

    for dist in DISTRICTS:
        epp_pivot, enroll_lines = prepare_epp_and_enroll(df, dist)
        per_district[dist] = (epp_pivot, enroll_lines)

        if not epp_pivot.empty:
            max_total = float(epp_pivot.sum(axis=1).max())
            if not (np.isnan(max_total) or np.isinf(max_total)):
                shared_max_dollars = max(shared_max_dollars, max_total)

        for s in enroll_lines.values():
            if s is not None and not s.empty:
                mv = float(s.max())
                if not (np.isnan(mv) or np.isinf(mv)):
                    shared_max_people = max(shared_max_people, mv)

    y_dollars_shared = shared_max_dollars if shared_max_dollars > 0 else None
    y_people_shared  = shared_max_people if shared_max_people > 0 else None

    # ---- Regional series: Western, by SCHOOL_TYPE (distinct colors) ----
    reg = load_regions()

    # Build the requested four plots in this order:
    regional_specs = []
    # 1) Traditional ≤500
    regional_specs.append(("le_500", "Traditional"))
    # 2) Traditional >500
    regional_specs.append(("gt_500", "Traditional"))
    # 3) Charter (all)
    regional_specs.append(("all", "Charter"))
    # 4) Vocational (all)
    regional_specs.append(("all", "Vocational"))

    # Compute shared axes across ALL regional plots
    compiled_specs = []
    regional_max_dollars = 0.0
    regional_max_people  = 0.0

    for size_bucket, school_type in regional_specs:
        title, epp_pivot, enroll_lines_sum, enroll_lines_mean, members = prepare_group_western(
            df, reg, size_bucket=size_bucket, school_type=school_type
        )
        if epp_pivot.empty and not enroll_lines_sum:
            print(f"[WARN] No members for {title}; skipping.")
            continue
        if not epp_pivot.empty:
            mt = float(epp_pivot.sum(axis=1).max())
            if not (np.isnan(mt) or np.isinf(mt)):
                regional_max_dollars = max(regional_max_dollars, mt)
        for s in enroll_lines_sum.values():
            if s is not None and not s.empty:
                mv = float(s.max())
                if not (np.isnan(mv) or np.isinf(mv)):
                    regional_max_people = max(regional_max_people, mv)
        # filename key
        key = f"Western_{school_type}_{size_bucket}".replace(" ", "_")
        compiled_specs.append((title, epp_pivot, enroll_lines_sum, enroll_lines_mean, key))

    y_dollars_reg = regional_max_dollars if regional_max_dollars > 0 else None
    y_people_reg  = regional_max_people if regional_max_people > 0 else None

    # ---- Render and save: PNGs + multi-page PDF ----
    figures_and_pngs: List[Tuple[plt.Figure, Path]] = []

    # District plots (Amherst-Pelham will appear first because of DISTRICTS order)
    for dist, (epp_pivot, enroll_lines) in per_district.items():
        # Label columns
        latest_epp_year = int(epp_pivot.index.max()) if not epp_pivot.empty else ""
        latest_fte_year = 0
        for s in enroll_lines.values():
            if not s.empty:
                latest_fte_year = max(latest_fte_year, int(s.index.max()))
        if latest_fte_year == 0 and latest_epp_year != "":
            latest_fte_year = latest_epp_year

        # Amherst-Pelham gets its own palette + title
        if dist == "Amherst-Pelham":
            title = "Amherst-Pelham Regional: Expenditures Per Pupil vs Enrollment"
            fig = plot_one(
                title, epp_pivot,
                enroll_lines_chart=enroll_lines,
                enroll_lines_table=enroll_lines,
                palette=PALETTE_APEL, color_overrides=COLOR_OVERRIDES_APEL, line_colors=LINE_COLORS_APEL,
                y_max_dollars=y_dollars_shared, y_max_people=y_people_shared,
                label_dollar_col=f"${latest_epp_year} $/pupil",
                label_fte_col=f"{latest_fte_year} FTE"
            )
        else:
            title = f"{dist}: Expenditures Per Pupil vs Enrollment"
            fig = plot_one(
                title, epp_pivot,
                enroll_lines_chart=enroll_lines,
                enroll_lines_table=enroll_lines,
                palette=PALETTE_DIST, color_overrides=COLOR_OVERRIDES_DIST, line_colors=LINE_COLORS_DIST,
                y_max_dollars=y_dollars_shared, y_max_people=y_people_shared,
                label_dollar_col=f"${latest_epp_year} $/pupil",
                label_fte_col=f"{latest_fte_year} FTE"
            )

        png_path = OUTPUT_DIR / f"expenditures_per_pupil_vs_enrollment_{dist.replace(' ', '_')}.png"
        figures_and_pngs.append((fig, png_path))

    # Regional plots (tables show MEANS; chart lines use SUMS)
    for title, epp_pivot, enroll_lines_sum, enroll_lines_mean, fname_key in compiled_specs:
        latest_epp_year = int(epp_pivot.index.max()) if not epp_pivot.empty else ""
        latest_fte_year = 0
        for s in enroll_lines_mean.values():
            if not s.empty:
                latest_fte_year = max(latest_fte_year, int(s.index.max()))
        if latest_fte_year == 0 and latest_epp_year != "":
            latest_fte_year = latest_epp_year

        fig = plot_one(
            title, epp_pivot,
            enroll_lines_chart=enroll_lines_sum,
            enroll_lines_table=enroll_lines_mean,
            palette=PALETTE_REGION, color_overrides=COLOR_OVERRIDES_REGION, line_colors=LINE_COLORS_REGION,
            y_max_dollars=y_dollars_reg, y_max_people=y_people_reg,
            label_dollar_col=f"Mean {latest_epp_year} $/pupil",
            label_fte_col=f"Mean {latest_fte_year} FTE"
        )
        png_path = OUTPUT_DIR / f"regional_expenditures_per_pupil_{fname_key}.png"
        figures_and_pngs.append((fig, png_path))

    # Save all PNGs and build a single PDF with one page per figure
    with PdfPages(OUTPUT_PDF) as pdf:
        for fig, png_path in figures_and_pngs:
            fig.savefig(png_path, bbox_inches="tight")
            fig.savefig(pdf, format="pdf", bbox_inches="tight")
            plt.close(fig)

    print(f"[OK] Wrote multi-page PDF: {OUTPUT_PDF}")
    for _, png_path in figures_and_pngs:
        print(f"[OK] Saved {png_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
