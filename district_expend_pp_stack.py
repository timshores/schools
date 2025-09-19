"""
district_expend_pp_stack.py — CHART-ONLY
Generates per-page PNGs with stacked $/pupil bars and two FTE lines.
No tables, no internal PDF. compose_pdf.py handles the tables & final A4 PDF.

Key bits retained:
- Excludes "Total Expenditures" and "Total In-District Expenditures" from stacks
- Lines: In-District & Out-of-District FTE (foreground, contrasting)
- Amherst-Pelham first with distinct palette & title
- Western MA regional charts by school_type (Traditional ≤500, Traditional >500, Charter all, Vocational all)
- Shared y-axis scaling across districts, and across regional charts
"""

from __future__ import annotations
import os, sys, re
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ---------------------------
# Paths & constants
# ---------------------------
DATA_DIR   = Path("./data")
OUTPUT_DIR = Path("./output")
EXCEL_FILE = DATA_DIR / "E2C_Hub_MA_DESE_Data.xlsx"

SHEET_EXPEND  = "District Expend by Category"
SHEET_REGIONS = "District Regions"

# Districts, Amherst-Pelham first
DISTRICTS = ["Amherst-Pelham", "Amherst", "Leverett", "Pelham", "Shutesbury"]

# Figure size (landscape chart image)
PAGE_FIGSIZE = (12.5, 7.5)

# Exclusions and keys
EXCLUDE_SUBCATS = {"total expenditures", "total in-district expenditures"}
ENROLL_KEYS = [
    ("in-district fte pupils", "In-District FTE Pupils"),
    ("out-of-district fte pupils", "Out-of-District FTE Pupils"),
]
TOTAL_FTE_KEY = "total fte pupils"

# Palettes
LINE_COLORS_DIST = {"In-District FTE Pupils": "#000000", "Out-of-District FTE Pupils": "#AA0000"}
COLOR_OVERRIDES_DIST = {"professional development": "#6A3D9A"}
PALETTE_DIST = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#F0E442", "#56B4E9",
                "#E69F00", "#999999", "#332288", "#88CCEE", "#44AA99", "#882255"]

LINE_COLORS_APEL = {"In-District FTE Pupils": "#1F4E79", "Out-of-District FTE Pupils": "#D35400"}
COLOR_OVERRIDES_APEL = {"professional development": "#2A9D8F"}
PALETTE_APEL = ["#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51", "#8AB17D",
                "#7A5195", "#4ECDC4", "#C7F464", "#FF6B6B", "#FFE66D", "#355C7D"]

LINE_COLORS_REGION = {"In-District FTE Pupils": "#1B7837", "Out-of-District FTE Pupils": "#5E3C99"}
COLOR_OVERRIDES_REGION = {"professional development": "#1B9E77"}
PALETTE_REGION = ["#8DD3C7", "#FFFFB3", "#BEBADA", "#FB8072", "#80B1D3", "#FDB462",
                  "#B3DE69", "#FCCDE5", "#D9D9D9", "#BC80BD", "#CCEBC5", "#FFED6F"]

# Plot styling
plt.rcParams.update({
    "figure.dpi": 150,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlelocation": "left",
    "axes.titlepad": 10,
    # bigger fonts
    "font.size": 16,        # base
    "axes.labelsize": 13,   # axis labels
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

### plot line and bar constants
FTE_LINEWIDTH = 5.6
FTE_MARKER_SIZE = 10
FTE_MARKER_EDGEWIDTH = 2.0
BAR_WIDTH = 0.70

# ---------------------------
# Helpers
# ---------------------------
def find_sheet_name(xls: pd.ExcelFile, target: str) -> str:
    for n in xls.sheet_names:
        if n.strip().lower() == target.strip().lower():
            return n
    t = target.strip().lower()
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
            chosen = name_map[key]; break
    if chosen is None:
        first = df.columns[0]
        if str(first).strip().lower() in ("sy", "school_year"):
            chosen = first
    if chosen is None:
        for c in df.columns:
            sample = str(df[c].dropna().astype(str).head(20).tolist())
            if re.search(r"\b(19|20)\d{2}\b", sample) or re.search(r"(19|20)\d{2}\s*[–-]\s*\d{2,4}", sample):
                chosen = c; break
    if chosen is None:
        raise ValueError("Could not find a year column ('SY' or 'FISCAL_YEAR' or 'YEAR').")

    def parse_to_year(val):
        if pd.isna(val): return np.nan
        if isinstance(val, (int, float)) and not pd.isna(val):
            v = int(val)
            if 1900 <= v <= 2100: return v
        s = str(val).strip()
        m = re.search(r"\b((?:19|20)\d{2})\s*[–\-/]\s*(\d{2,4})\b", s)
        if m:
            y1 = int(m.group(1)); y2raw = m.group(2)
            y2 = (y1 // 100) * 100 + int(y2raw) if len(y2raw) == 2 else int(y2raw)
            if len(y2raw) == 2 and (y2 % 100) < (y1 % 100): y2 += 100
            return y2
        m3 = re.search(r"\b((?:19|20)\d{2})\b", s)
        if m3: return int(m3.group(1))
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

def sparse_year_ticks(ax, years: List[int]):
    if not years: return
    n = len(years)
    if n <= 10: ticks = years
    elif n <= 16: ticks = years[::2]
    else:
        step = max(3, n // 10)
        ticks = years[::step]
    ax.set_xticks(ticks)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

def compute_cagr_last(series: pd.Series, years_back: int) -> float | np.nan:
    if series is None or series.empty: return np.nan
    s = series.sort_index().astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if s.size < 2: return np.nan
    end_year = int(s.index.max())
    candidates = s[s.index <= end_year - years_back]
    start_year = int(candidates.index.max()) if not candidates.empty else int(s.index.min())
    n_years = end_year - start_year
    if n_years <= 0: return np.nan
    start_val = float(s.loc[start_year]); end_val = float(s.loc[end_year])
    if start_val <= 0 or end_val <= 0: return np.nan
    return (end_val / start_val) ** (1.0 / n_years) - 1.0

# ---------------------------
# Data loading
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
        if low == "dist_name": canon[c] = "DIST_NAME"
        elif low == "ind_cat": canon[c] = "IND_CAT"
        elif low == "ind_subcat": canon[c] = "IND_SUBCAT"
        elif low == "ind_value": canon[c] = "IND_VALUE"
        elif low in ("sy","school_year","fiscal_year","year"): canon[c] = c
    df = df.rename(columns=canon)
    if "IND_VALUE" not in df.columns:
        raise ValueError("IND_VALUE column not found in the sheet.")
    df = coalesce_year_column(df)
    df = clean_strings(df, ["DIST_NAME","IND_CAT","IND_SUBCAT"])
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
        if low in ("district_name","dist_name","name"): cmap[c] = "DIST_NAME"
        elif low in ("eohhs_region","region"): cmap[c] = "EOHHS_REGION"
        elif low in ("school_type","type","district_type","org_type"): cmap[c] = "SCHOOL_TYPE"
    reg = reg.rename(columns=cmap)
    for c in ["DIST_NAME","EOHHS_REGION","SCHOOL_TYPE"]:
        reg[c] = reg[c].astype(str).str.strip()
    return reg

# ---------------------------
# Transform helpers
# ---------------------------
def prepare_epp_and_enroll(df: pd.DataFrame, district: str) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
    ddf = df[df["DIST_NAME"].str.lower() == district.lower()].copy()
    epp = ddf[ddf["IND_CAT"].str.lower() == "expenditures per pupil"].copy()
    epp["IND_SUBCAT"] = epp["IND_SUBCAT"].replace({np.nan: "Unspecified"}).fillna("Unspecified")
    epp = epp[~epp["IND_SUBCAT"].str.lower().isin(EXCLUDE_SUBCATS)].copy()
    epp_pivot = epp.pivot_table(index="YEAR", columns="IND_SUBCAT", values="IND_VALUE", aggfunc="sum").sort_index().fillna(0.0)

    enroll_lines: Dict[str, pd.Series] = {}
    enroll_ddf = ddf[ddf["IND_CAT"].str.lower() == "student enrollment"].copy()
    if not enroll_ddf.empty:
        for key, label in ENROLL_KEYS:
            ser = enroll_ddf[enroll_ddf["IND_SUBCAT"].str.lower() == key][["YEAR","IND_VALUE"]]
            if not ser.empty:
                ser = (ser.dropna().drop_duplicates(subset=["YEAR"])
                         .set_index("YEAR")["IND_VALUE"].sort_index())
                enroll_lines[label] = ser
    return epp_pivot, enroll_lines

def epp_group_pivot_mean(epp_df: pd.DataFrame, weights_df: Optional[pd.DataFrame], method: str="weighted") -> pd.DataFrame:
    if method == "weighted" and weights_df is not None and not weights_df.empty:
        m = epp_df.merge(weights_df, on=["DIST_NAME","YEAR"], how="left")
        m["WEIGHT"] = pd.to_numeric(m["WEIGHT"], errors="coerce").fillna(0.0)
        m["P"] = m["IND_VALUE"] * m["WEIGHT"]
        grp = m.groupby(["YEAR","IND_SUBCAT"], as_index=False, sort=False)
        out = grp.agg(NUM=("P","sum"), DEN=("WEIGHT","sum"), MEAN=("IND_VALUE","mean"))
        out["VALUE"] = np.where(out["DEN"]>0, out["NUM"]/out["DEN"], out["MEAN"])
        return out.pivot(index="YEAR", columns="IND_SUBCAT", values="VALUE").sort_index().fillna(0.0)
    return epp_df.pivot_table(index="YEAR", columns="IND_SUBCAT", values="IND_VALUE", aggfunc="mean").sort_index().fillna(0.0)

def prepare_group_western(df: pd.DataFrame, reg: pd.DataFrame, size_bucket: str, school_type: str):
    mask = (reg["EOHHS_REGION"].str.lower()=="western") & (reg["SCHOOL_TYPE"].str.lower()==school_type.lower())
    names = set(reg[mask]["DIST_NAME"].str.lower())
    present = set(df["DIST_NAME"].str.lower())
    members = sorted(n for n in names if n in present)

    enroll = df[df["IND_CAT"].str.lower()=="student enrollment"].copy()
    totals = {}
    for nm in members:
        dsub = enroll[enroll["DIST_NAME"].str.lower()==nm]
        tot = dsub[dsub["IND_SUBCAT"].str.lower()==TOTAL_FTE_KEY]
        if not tot.empty:
            y = int(tot["YEAR"].max())
            totals[nm] = float(tot.loc[tot["YEAR"]==y,"IND_VALUE"].iloc[0])

    if size_bucket=="le_500":
        members = [n for n,v in totals.items() if v<=500]
        title_suffix = "≤500 FTE"
    elif size_bucket=="gt_500":
        members = [n for n,v in totals.items() if v>500]
        title_suffix = ">500 FTE"
    else:
        title_suffix = None

    title = f"All Western MA {school_type.capitalize()} Districts"
    if title_suffix: title += f" {title_suffix}"
    if not members:
        return title, pd.DataFrame(), {}, {}, []

    epp = df[(df["IND_CAT"].str.lower()=="expenditures per pupil") & (df["DIST_NAME"].str.lower().isin(members))][["DIST_NAME","YEAR","IND_SUBCAT","IND_VALUE"]].copy()
    epp = epp[~epp["IND_SUBCAT"].str.lower().isin(EXCLUDE_SUBCATS)].copy()
    weights = df[(df["IND_CAT"].str.lower()=="student enrollment") &
                 (df["IND_SUBCAT"].str.lower()=="in-district fte pupils") &
                 (df["DIST_NAME"].str.lower().isin(members))][["DIST_NAME","YEAR","IND_VALUE"]].rename(columns={"IND_VALUE":"WEIGHT"})
    epp_pivot = epp_group_pivot_mean(epp, weights, "weighted")

    enroll_lines_sum, enroll_lines_mean = {}, {}
    for key, label in ENROLL_KEYS:
        sub = df[(df["IND_CAT"].str.lower()=="student enrollment") &
                 (df["IND_SUBCAT"].str.lower()==key) &
                 (df["DIST_NAME"].str.lower().isin(members))][["DIST_NAME","YEAR","IND_VALUE"]]
        if not sub.empty:
            enroll_lines_sum[label]  = sub.groupby("YEAR")["IND_VALUE"].sum().sort_index()
            enroll_lines_mean[label] = sub.groupby("YEAR")["IND_VALUE"].mean().sort_index()
    pretty_members = [df[df['DIST_NAME'].str.lower()==m]['DIST_NAME'].iloc[0] for m in members]
    return title, epp_pivot, enroll_lines_sum, enroll_lines_mean, pretty_members

def order_subcats_by_mean(epp_pivot: pd.DataFrame) -> List[str]:
    if epp_pivot.shape[1] <= 1: return list(epp_pivot.columns)
    return list(epp_pivot.mean(axis=0).sort_values(ascending=False).index)

# ---------------------------
# Plotting (charts only)
# ---------------------------
def plot_one_chart(title, epp_pivot, enroll_lines_chart,
                   palette, color_overrides, line_colors,
                   y_max_dollars=None, y_max_people=None):
    # Align year index
    year_sets = [set(epp_pivot.index)] + [set(s.index) for s in enroll_lines_chart.values() if s is not None]
    years = sorted(set().union(*year_sets))
    epp_pivot = epp_pivot.reindex(years).fillna(0.0)

    # Order subcats by average contribution
    subcats = order_subcats_by_mean(epp_pivot)
    epp_pivot = epp_pivot.reindex(columns=subcats)

    # Colors per subcat
    base_colors = (palette * ((len(subcats)//len(palette))+1))[:len(subcats)]
    color_map = {sc: col for sc, col in zip(subcats, base_colors)}
    for sc in subcats:
        key = sc.strip().lower()
        if key in color_overrides: color_map[sc] = color_overrides[key]

    # Figure & axes
    fig = plt.figure(figsize=PAGE_FIGSIZE)
    axL = fig.add_subplot(111)
    axR = axL.twinx()
    axR.spines["right"].set_visible(True)
    axR.set_zorder(2); axL.set_zorder(3); axL.patch.set_alpha(0.0)

    for a in (axL, axR):
        a.margins(x=0.0, y=0.0)
        a.grid(False)

    # Stacked bars on right-axis ($)
    bottom = np.zeros(len(years))
    for sc in subcats:
        vals = epp_pivot[sc].values
        axR.bar(years, vals, bottom=bottom, width=BAR_WIDTH, align="center",
                zorder=1, color=color_map[sc], edgecolor="none")
        bottom += vals

    if y_max_dollars: axR.set_ylim(0, y_max_dollars)

    # Enrollment lines on left-axis (foreground)
    for _key, label in ENROLL_KEYS:
        s = enroll_lines_chart.get(label)
        if s is None: continue
        s = s.reindex(years)
        lc = line_colors.get(label, "#000000")
        axL.plot(years, s.values, marker="o", markersize=FTE_MARKER_SIZE, markerfacecolor="white",
                 markeredgewidth=FTE_MARKER_EDGEWIDTH, linewidth=FTE_LINEWIDTH, color=lc, zorder=10, clip_on=False)

    if y_max_people: axL.set_ylim(0, y_max_people)
    sparse_year_ticks(axL, years)

    # Labels & title
    format_people(axL); axL.set_ylabel("Student Enrollment (FTE)")
    format_dollars(axR); axR.set_ylabel("Expenditures per Pupil ($)")
    axL.set_xlabel("School Year")

    # No legend here — ReportLab draws the line swatches in the FTE table.
    fig.subplots_adjust(left=0.07, right=0.94, bottom=0.12, top=0.90)
    return fig

# ---------------------------
# Main
# ---------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_expend_data()
    df = df[df["IND_CAT"].str.lower().isin({"expenditures per pupil","student enrollment"})].copy()

    # Shared scales across the five districts
    per_district = {}
    shared_max_dollars = 0.0
    shared_max_people  = 0.0
    for dist in DISTRICTS:
        epp_pivot, enroll_lines = prepare_epp_and_enroll(df, dist)
        per_district[dist] = (epp_pivot, enroll_lines)
        if not epp_pivot.empty:
            shared_max_dollars = max(shared_max_dollars, float(epp_pivot.sum(axis=1).max()))
        for s in enroll_lines.values():
            if s is not None and not s.empty:
                shared_max_people = max(shared_max_people, float(s.max()))
    y_dollars_shared = shared_max_dollars or None
    y_people_shared  = shared_max_people  or None

    # Regional groups (Western)
    reg = load_regions()
    regional_specs = [("le_500","Traditional"), ("gt_500","Traditional"),
                      ("all","Charter"), ("all","Vocational")]
    compiled = []
    reg_max_dollars = 0.0
    reg_max_people  = 0.0
    for size_bucket, school_type in regional_specs:
        title, epp_pivot, enroll_sum, enroll_mean, members = prepare_group_western(df, reg, size_bucket, school_type)
        if epp_pivot.empty and not enroll_sum:
            print(f"[WARN] No members for {title}; skipping.")
            continue
        if not epp_pivot.empty:
            reg_max_dollars = max(reg_max_dollars, float(epp_pivot.sum(axis=1).max()))
        for s in enroll_sum.values():
            if s is not None and not s.empty:
                reg_max_people = max(reg_max_people, float(s.max()))
        key = f"Western_{school_type}_{size_bucket}".replace(" ", "_")
        compiled.append((title, epp_pivot, enroll_sum, key))
    y_dollars_reg = reg_max_dollars or None
    y_people_reg  = reg_max_people  or None

    # Render & save district charts
    for dist, (epp_pivot, enroll_lines) in per_district.items():
        if dist == "Amherst-Pelham":
            title = "Amherst-Pelham Regional: Expenditures Per Pupil vs Enrollment"
            fig = plot_one_chart(title, epp_pivot, enroll_lines,
                                 PALETTE_APEL, COLOR_OVERRIDES_APEL, LINE_COLORS_APEL,
                                 y_dollars_shared, y_people_shared)
        else:
            title = f"{dist}: Expenditures Per Pupil vs Enrollment"
            fig = plot_one_chart(title, epp_pivot, enroll_lines,
                                 PALETTE_DIST, COLOR_OVERRIDES_DIST, LINE_COLORS_DIST,
                                 y_dollars_shared, y_people_shared)
        png = OUTPUT_DIR / f"expenditures_per_pupil_vs_enrollment_{dist.replace(' ','_')}.png"
        fig.savefig(png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Saved {png}")

    # Render & save Western region charts
    for title, epp_pivot, enroll_sum, key in compiled:
        fig = plot_one_chart(title, epp_pivot, enroll_sum,
                             PALETTE_REGION, COLOR_OVERRIDES_REGION, LINE_COLORS_REGION,
                             y_dollars_reg, y_people_reg)
        png = OUTPUT_DIR / f"regional_expenditures_per_pupil_{key}.png"
        fig.savefig(png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Saved {png}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
