# school_shared.py
# Shared constants, loaders, color mapping, and helpers for plotting & composing.

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ---------------- Paths ----------------
DATA_DIR = Path("./data")
OUTPUT_DIR = Path("./output")
EXCEL_FILE = DATA_DIR / "E2C_Hub_MA_DESE_Data.xlsx"
COLOR_MAP_PATH = OUTPUT_DIR / "category_color_map.json"

# ---------------- Enrollment threshold (palette switch) ----------------
N_THRESHOLD = 500  # switchable later if requested

# ---------------- District set ----------------
DISTRICTS_OF_INTEREST = ["Amherst-Pelham", "Amherst", "Leverett", "Pelham", "Shutesbury"]

# ---------------- Domain constants ----------------
SHEET_EXPEND = "District Expend by Category"
SHEET_REGIONS = "District Regions"

EXCLUDE_SUBCATS = {"total expenditures", "total in-district expenditures"}

ENROLL_KEYS = [
    ("in-district fte pupils", "In-District FTE Pupils"),
    ("out-of-district fte pupils", "Out-of-District FTE Pupils"),
]
TOTAL_FTE_KEY = "total fte pupils"

# ---------------- Two color palettes (SMALL, LARGE) ----------------
# Keep PD distinct across both palettes for contrast.
PALETTE_SMALL = [
    "#0072B2", "#009E73", "#D55E00", "#CC79A7", "#F0E442", "#56B4E9",
    "#E69F00", "#999999", "#332288", "#88CCEE", "#44AA99", "#882255",
]
PALETTE_LARGE = [
    "#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51", "#8AB17D",
    "#7A5195", "#4ECDC4", "#C7F464", "#FF6B6B", "#FFE66D", "#355C7D",
]
SMALL_OVERRIDES = {"professional development": "#6A3D9A"}
LARGE_OVERRIDES = {"professional development": "#2A9D8F"}

# ---------------- FTE line colors ----------------
LINE_COLORS_DIST = {  # for individual district plots
    "In-District FTE Pupils": "#000000",
    "Out-of-District FTE Pupils": "#AA0000",
}
LINE_COLORS_WESTERN = {  # for Western aggregate plots
    "In-District FTE Pupils": "#1B7837",
    "Out-of-District FTE Pupils": "#5E3C99",
}

# ---------------- Basic utils ----------------
def norm_label(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip()).lower()

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

# ---------------- Load data ----------------
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    xls = pd.ExcelFile(EXCEL_FILE)

    df = pd.read_excel(xls, sheet_name=find_sheet_name(xls, SHEET_EXPEND))
    df.columns = [str(c).strip() for c in df.columns]
    ren = {}
    for c in df.columns:
        low = str(c).strip().lower()
        if low == "dist_name":
            ren[c] = "DIST_NAME"
        elif low == "ind_cat":
            ren[c] = "IND_CAT"
        elif low == "ind_subcat":
            ren[c] = "IND_SUBCAT"
        elif low == "ind_value":
            ren[c] = "IND_VALUE"
        elif low in ("sy", "school_year", "fiscal_year", "year"):
            ren[c] = c
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
        if low in ("district_name", "dist_name", "name"):
            cmap[c] = "DIST_NAME"
        elif low in ("eohhs_region", "region"):
            cmap[c] = "EOHHS_REGION"
        elif low in ("school_type", "type", "district_type", "org_type"):
            cmap[c] = "SCHOOL_TYPE"
    reg = reg.rename(columns=cmap)
    for c in ("DIST_NAME", "EOHHS_REGION", "SCHOOL_TYPE"):
        reg[c] = reg[c].astype(str).str.strip()
    return df, reg

# ---------------- Global subcats + color map ----------------
def build_global_subcat_list(df: pd.DataFrame) -> List[str]:
    mask = df["IND_CAT"].str.lower().eq("expenditures per pupil")
    subs = df.loc[mask, "IND_SUBCAT"].astype(str).map(norm_label).tolist()
    return sorted({s for s in subs if s and s not in EXCLUDE_SUBCATS})

def _assign_colors(subcats_norm: List[str], palette: List[str], overrides_norm: Dict[str, str]) -> Dict[str, str]:
    m = {}
    for i, sc in enumerate(subcats_norm):  # alphabetical list
        m[sc] = palette[i % len(palette)]
    for k, v in overrides_norm.items():
        if k in m:
            m[k] = v
    return m

def _overrides_norm(d: Dict[str, str]) -> Dict[str, str]:
    return {norm_label(k): v for k, v in d.items()}

def _rebuild_color_map(df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    subcats = build_global_subcat_list(df)
    return {
        "SMALL": _assign_colors(subcats, PALETTE_SMALL, _overrides_norm(SMALL_OVERRIDES)),
        "LARGE": _assign_colors(subcats, PALETTE_LARGE, _overrides_norm(LARGE_OVERRIDES)),
    }

def create_or_load_color_map(df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    """
    Loads /output/category_color_map.json if present.
    If it's from the old schema ('DIST'/'APEL'/'REGION') or malformed,
    rebuild to the new schema ('SMALL'/'LARGE') and overwrite the file.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # No file? Build fresh.
    if not COLOR_MAP_PATH.exists():
        cmap = _rebuild_color_map(df)
        COLOR_MAP_PATH.write_text(json.dumps(cmap, indent=2), encoding="utf-8")
        return cmap

    # Try reading existing file
    try:
        cmap = json.loads(COLOR_MAP_PATH.read_text(encoding="utf-8"))
    except Exception:
        # Corrupt? Rebuild.
        cmap = _rebuild_color_map(df)
        COLOR_MAP_PATH.write_text(json.dumps(cmap, indent=2), encoding="utf-8")
        return cmap

    # Already in the new format?
    if isinstance(cmap, dict) and "SMALL" in cmap and "LARGE" in cmap:
        return cmap

    # Old format detected (DIST/APEL/REGION) or something else — rebuild.
    cmap = _rebuild_color_map(df)
    COLOR_MAP_PATH.write_text(json.dumps(cmap, indent=2), encoding="utf-8")
    return cmap

def color_for(cmap_all: Dict[str, Dict[str, str]], context: str, subcat_label: str) -> str:
    """
    Safe color fetch: tolerate missing context keys.
    """
    cmap_ctx = cmap_all.get(context) or {}
    return cmap_ctx.get(norm_label(subcat_label), "#777777")

# ---------------- Context helpers ----------------
def latest_total_fte(df: pd.DataFrame, dist: str) -> float:
    ddf = df[
        (df["DIST_NAME"].str.lower() == dist.lower())
        & (df["IND_CAT"].str.lower() == "student enrollment")
        & (df["IND_SUBCAT"].str.lower() == TOTAL_FTE_KEY)
    ]
    if ddf.empty:
        parts = df[
            (df["DIST_NAME"].str.lower() == dist.lower())
            & (df["IND_CAT"].str.lower() == "student enrollment")
            & (df["IND_SUBCAT"].str.lower().isin([k for k, _ in ENROLL_KEYS]))
        ][["YEAR", "IND_VALUE"]]
        if parts.empty:
            return 0.0
        s = parts.groupby("YEAR")["IND_VALUE"].sum().sort_index()
        return float(s.iloc[-1])
    y = int(ddf["YEAR"].max())
    return float(ddf.loc[ddf["YEAR"] == y, "IND_VALUE"].iloc[0])

def context_for_district(df: pd.DataFrame, dist: str) -> str:
    return "LARGE" if latest_total_fte(df, dist) > N_THRESHOLD else "SMALL"

def context_for_western(bucket: str) -> str:
    # 'le_500' -> SMALL, 'gt_500' -> LARGE
    return "SMALL" if bucket == "le_500" else "LARGE"

# ---------------- EPP + Enrollment prep ----------------
def prepare_district_epp_lines(df: pd.DataFrame, dist: str) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
    ddf = df[df["DIST_NAME"].str.lower() == dist.lower()].copy()

    epp = ddf[ddf["IND_CAT"].str.lower() == "expenditures per pupil"].copy()
    epp = epp[~epp["IND_SUBCAT"].str.lower().isin(EXCLUDE_SUBCATS)].copy()
    piv = (
        epp.pivot_table(index="YEAR", columns="IND_SUBCAT", values="IND_VALUE", aggfunc="sum")
        .sort_index()
        .fillna(0.0)
    )

    lines: Dict[str, pd.Series] = {}
    enroll = ddf[ddf["IND_CAT"].str.lower() == "student enrollment"].copy()
    for key, label in ENROLL_KEYS:
        ser = enroll[enroll["IND_SUBCAT"].str.lower() == key][["YEAR", "IND_VALUE"]]
        if not ser.empty:
            s = ser.dropna().drop_duplicates(subset=["YEAR"]).set_index("YEAR")["IND_VALUE"].sort_index()
            lines[label] = s
    return piv, lines

def prepare_western_epp_lines(df: pd.DataFrame, reg: pd.DataFrame, bucket: str) -> Tuple[str, pd.DataFrame, Dict[str, pd.Series], Dict[str, pd.Series]]:
    mask = (reg["EOHHS_REGION"].str.lower() == "western") & (reg["SCHOOL_TYPE"].str.lower() == "traditional")
    members = sorted(set(reg[mask]["DIST_NAME"].str.lower()))
    present = set(df["DIST_NAME"].str.lower())
    members = [m for m in members if m in present]

    # bucket by latest Total FTE
    totals = {}
    enroll = df[df["IND_CAT"].str.lower() == "student enrollment"].copy()
    for nm in members:
        dsub = enroll[enroll["DIST_NAME"].str.lower() == nm]
        tot = dsub[dsub["IND_SUBCAT"].str.lower() == TOTAL_FTE_KEY]
        if not tot.empty:
            y = int(tot["YEAR"].max())
            totals[nm] = float(tot.loc[tot["YEAR"] == y, "IND_VALUE"].iloc[0])

    if bucket == "le_500":
        members = [n for n, v in totals.items() if v <= N_THRESHOLD]
        suffix = "≤500 Students"
    else:
        members = [n for n, v in totals.items() if v > N_THRESHOLD]
        suffix = ">500 Students"

    title = f"All Western MA Traditional Districts {suffix}"
    if not members:
        return title, pd.DataFrame(), {}, {}

    epp = df[
        (df["IND_CAT"].str.lower() == "expenditures per pupil")
        & (df["DIST_NAME"].str.lower().isin(members))
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
    piv = out.pivot(index="YEAR", columns="IND_SUBCAT", values="VALUE").sort_index().fillna(0.0)

    # Enrollment lines: SUM for chart; MEAN also returned (for compose tables if needed)
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

    return title, piv, lines_sum, lines_mean

# ---------------- Axis limits ----------------
def _nice_ceiling(x: float, step: int) -> float:
    if x <= 0:
        return step
    return math.ceil(x / step) * step

def compute_global_dollar_ylim(pivots: List[pd.DataFrame], pad: float = 1.05, step: int = 500) -> float:
    tops = []
    for piv in pivots:
        if piv is None or piv.empty:
            continue
        totals = piv.sum(axis=1)
        if not totals.empty:
            tops.append(float(totals.max()))
    if not tops:
        return step
    m = max(tops) * pad
    return _nice_ceiling(m, step)

def compute_districts_fte_ylim(lines_list: List[Dict[str, pd.Series]], pad: float = 1.05, step: int = 50) -> float:
    tops = []
    for lines in lines_list:
        for s in lines.values():
            if s is not None and not s.empty:
                tops.append(float(s.max()))
    if not tops:
        return step
    m = max(tops) * pad
    return _nice_ceiling(m, step)
