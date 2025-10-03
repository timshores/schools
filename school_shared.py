from __future__ import annotations

import json, math, re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ---------------- Paths ----------------
DATA_DIR = Path("./data")
OUTPUT_DIR = Path("./output")
EXCEL_FILE = DATA_DIR / "E2C_Hub_MA_DESE_Data.xlsx"
COLOR_MAP_PATH = OUTPUT_DIR / "category_color_map.json"
COLOR_MAP_VERSION = 4  # unified, colorblind-safe palette

# ---------------- Enrollment threshold ----------------
N_THRESHOLD = 500

# ---------------- District set ----------------
DISTRICTS_OF_INTEREST = ["Amherst-Pelham", "Amherst", "Leverett", "Pelham", "Shutesbury"]

# ---------------- Sheets ----------------
SHEET_EXPEND  = "District Expend by Category"
SHEET_REGIONS = "District Regions"

# Exclude totals
EXCLUDE_SUBCATS = {"total expenditures", "total in-district expenditures"}

# Enrollment keys
ENROLL_KEYS = [
    ("in-district fte pupils", "In-District FTE Pupils"),
    ("out-of-district fte pupils", "Out-of-District FTE Pupils"),
]
TOTAL_FTE_KEY = "total fte pupils"

# ---------------- Canonical categories (bottom -> top; "Other" always last) ----------------
CANON_CATS_BOTTOM_TO_TOP = [
    "Teachers",
    "Insurance, Retirement Programs and Other",
    "Pupil Services",
    "Other Teaching Services",
    "Operations and Maintenance",
    "Instructional Leadership",
    "Administration",
    "Other",
]

def norm_label(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip()).lower()

SUBCAT_TO_CANON = {
    "teachers": "Teachers",
    "insurance, retirement programs and other": "Insurance, Retirement Programs and Other",
    "pupil services": "Pupil Services",
    "other teaching services": "Other Teaching Services",
    "operations and maintenance": "Operations and Maintenance",
    "instructional leadership": "Instructional Leadership",
    "administration": "Administration",
    "guidance, counseling and testing": "Pupil Services",
    "professional development": "Other",
    "instructional materials, equipment and technology": "Other",
    "student transportation": "Other",
    "transportation": "Other",
    "extraordinary maintenance": "Other",
    "tuition": "Other",
    "special education tuition": "Other",
    "tuition to mass schools": "Other",
    "food services": "Other",
}

# ---------------- Unified palette (Okabe–Ito) ----------------
UNIFIED_PALETTE = {
    "Teachers":                                   "#A8D5F2",  # lighter blue
    "Insurance, Retirement Programs and Other":    "#F5D6A8",  # lighter orange
    "Pupil Services":                              "#A8E6D0",  # lighter teal
    "Other Teaching Services":                     "#E8C8DC",  # lighter pink
    "Operations and Maintenance":                  "#C8E6F5",  # lighter sky blue
    "Instructional Leadership":                    "#F8F4B8",  # lighter yellow
    "Administration":                              "#F0C5A8",  # lighter burnt orange
    "Other":                                       "#D8D8D8",  # lighter gray
}

# ---------------- Enrollment line/swatch colors (unified) ----------------
ENROLL_IN  = "In-District FTE Pupils"
ENROLL_OUT = "Out-of-District Placed FTE Pupils"

FTE_LINE_COLORS = {
    ENROLL_IN:  "#000000",  # black
    ENROLL_OUT: "#D32F2F",  # red
    "Out-of-District FTE Pupils": "#D32F2F",  # legacy alias
}

# ---------------- Peers PNG (ALPS & peers) styling ----------------
PPE_PEERS_YMAX          = 35000.0  # Increased from 30000 to lift legend above bars
PPE_PEERS_REMOVE_SPINES = True   # remove all boundary lines
PPE_PEERS_BAR_EDGES     = False  # no bar edge lines

# Micro-area (enrollment sparkline) colors
MICRO_AREA_FILL = "#F5CBA7"  # pale orange
MICRO_AREA_EDGE = "#C97E2C"  # darker orange


# ---------------- Loaders & year coercion ----------------
def find_sheet_name(xls: pd.ExcelFile, target: str) -> str:
    t = target.strip().lower()
    for n in xls.sheet_names:
        if n.strip().lower() == t: return n
    for n in xls.sheet_names:
        if t in n.strip().lower(): return n
    return xls.sheet_names[0]

def coalesce_year_column(df: pd.DataFrame) -> pd.DataFrame:
    name_map = {str(c).strip().lower(): c for c in df.columns}
    chosen = None
    for key in ("fiscal_year", "year", "sy", "school_year"):
        if key in name_map: chosen = name_map[key]; break
    if chosen is None:
        first = df.columns[0]
        if str(first).strip().lower() in ("sy", "school_year"): chosen = first
    if chosen is None:
        for c in df.columns:
            sample = str(df[c].dropna().astype(str).head(20).tolist())
            if re.search(r"\b(19|20)\d{2}\b", sample): chosen = c; break
    if chosen is None: raise ValueError("Could not find a year column.")

    def parse_to_year(val):
        if pd.isna(val): return np.nan
        if isinstance(val, (int, float)) and not pd.isna(val):
            v = int(val)
            if 1900 <= v <= 2100: return v
        s = str(val).strip()
        m = re.search(r"\b((?:19|20)\d{2})\s*[–\-/]\s*(\d{2,4})\b", s)
        if m:
            y1 = int(m.group(1)); y2r = m.group(2)
            y2 = (y1 // 100) * 100 + int(y2r) if len(y2r) == 2 else int(y2r)
            if len(y2r) == 2 and (y2 % 100) < (y1 % 100): y2 += 100
            return y2
        m2 = re.search(r"\b((?:19|20)\d{2})\b", s)
        if m2: return int(m2.group(1))
        return np.nan

    df = df.copy()
    df["YEAR"] = df[chosen].apply(parse_to_year)
    df = df.dropna(subset=["YEAR"])
    df["YEAR"] = df["YEAR"].astype(int)
    return df

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    if "IND_VALUE" not in df.columns: raise ValueError("IND_VALUE missing")
    df = coalesce_year_column(df)
    for c in ("DIST_NAME", "IND_CAT", "IND_SUBCAT"):
        df[c] = df[c].astype(str).str.strip()
    df["IND_VALUE"] = pd.to_numeric(df["IND_VALUE"], errors="coerce")

    # Load profile_DataC70 sheet if available
    profile_c70 = pd.DataFrame()
    try:
        if "profile_DataC70" in xls.sheet_names:
            profile_c70 = pd.read_excel(xls, sheet_name="profile_DataC70")
            # Filter out header rows (Org4Code == 0)
            profile_c70 = profile_c70[profile_c70["Org4Code"] > 0].copy()
            # Normalize column names
            profile_c70.columns = [str(c).strip() for c in profile_c70.columns]
            # Standardize district names and year column
            if "District" in profile_c70.columns:
                # Normalize C70 district names to match expenditure data format
                # C70 uses ALL CAPS with spaces (e.g., "AMHERST PELHAM")
                # Expenditure data uses Title Case with hyphens (e.g., "Amherst-Pelham")
                profile_c70["DIST_NAME"] = (
                    profile_c70["District"]
                    .astype(str)
                    .str.strip()
                    .str.title()  # ALL CAPS -> Title Case
                    .str.replace(" ", "-")  # spaces -> hyphens for multi-word districts
                )
            if "fy" in profile_c70.columns:
                profile_c70["YEAR"] = pd.to_numeric(profile_c70["fy"], errors="coerce").astype("Int64")
            # Convert numeric columns
            for col in profile_c70.columns:
                if col not in ["DIST_NAME", "YEAR", "District", "Source"]:
                    profile_c70[col] = pd.to_numeric(profile_c70[col], errors="coerce")
    except Exception as e:
        print(f"[WARN] Could not load profile_DataC70: {e}")

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
    return df, reg, profile_c70

# ---------------- Canonical aggregation ----------------
def canonical_order_bottom_to_top(cols_present: List[str]) -> List[str]:
    present = set(cols_present)
    return [c for c in CANON_CATS_BOTTOM_TO_TOP if c in present]

def aggregate_to_canonical(piv: pd.DataFrame) -> pd.DataFrame:
    if piv is None or piv.empty: return piv
    colmap = {}
    for col in piv.columns:
        k = norm_label(col)
        target = SUBCAT_TO_CANON.get(k)
        if target is None:
            canon_by_label = next((c for c in CANON_CATS_BOTTOM_TO_TOP if norm_label(c) == k), None)
            target = canon_by_label if canon_by_label else "Other"
        colmap[col] = target
    parts = []
    for canon in CANON_CATS_BOTTOM_TO_TOP:
        cols_here = [c for c, grp in colmap.items() if grp == canon]
        if not cols_here: continue
        parts.append(piv[cols_here].sum(axis=1).rename(canon))
    out = pd.concat(parts, axis=1)
    out = out.loc[:, (out.abs().sum(axis=0) > 0)]
    out = out[canonical_order_bottom_to_top(out.columns.tolist())]
    return out

# ---------------- Color map (unified for both contexts) ----------------
def _rebuild_color_map() -> Dict[str, Dict[str, str]]:
    return {"_version": COLOR_MAP_VERSION, "SMALL": UNIFIED_PALETTE, "LARGE": UNIFIED_PALETTE, "_mode": "unified"}

def create_or_load_color_map(_df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    def write(c): COLOR_MAP_PATH.write_text(json.dumps(c, indent=2), encoding="utf-8"); return c
    if not COLOR_MAP_PATH.exists(): return write(_rebuild_color_map())
    try:
        cmap = json.loads(COLOR_MAP_PATH.read_text(encoding="utf-8"))
    except Exception:
        return write(_rebuild_color_map())
    if int(cmap.get("_version", 0)) < COLOR_MAP_VERSION or "SMALL" not in cmap or "LARGE" not in cmap:
        return write(_rebuild_color_map())
    return cmap

def color_for(cmap_all: Dict[str, Dict[str, str]], context: str, canon_label: str) -> str:
    return (cmap_all.get("SMALL") or {}).get(canon_label, "#777777")  # unified palette

# ---------------- Context & prep ----------------
def latest_total_fte(df: pd.DataFrame, dist: str) -> float:
    ddf = df[(df["DIST_NAME"].str.lower() == dist.lower()) & (df["IND_CAT"].str.lower() == "student enrollment") & (df["IND_SUBCAT"].str.lower() == TOTAL_FTE_KEY)]
    if ddf.empty:
        parts = df[(df["DIST_NAME"].str.lower() == dist.lower()) & (df["IND_CAT"].str.lower() == "student enrollment") & (df["IND_SUBCAT"].str.lower().isin([k for k, _ in ENROLL_KEYS]))][["YEAR", "IND_VALUE"]]
        if parts.empty: return 0.0
        s = parts.groupby("YEAR")["IND_VALUE"].sum().sort_index()
        return float(s.iloc[-1])
    y = int(ddf["YEAR"].max())
    return float(ddf.loc[ddf["YEAR"] == y, "IND_VALUE"].iloc[0])

def context_for_district(df: pd.DataFrame, dist: str) -> str:
    return "LARGE" if latest_total_fte(df, dist) > N_THRESHOLD else "SMALL"

def context_for_western(bucket: str) -> str:
    return "SMALL" if bucket == "le_500" else "LARGE"

def prepare_district_epp_lines(df: pd.DataFrame, dist: str) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
    ddf = df[df["DIST_NAME"].str.lower() == dist.lower()].copy()
    epp = ddf[ddf["IND_CAT"].str.lower() == "expenditures per pupil"].copy()
    epp = epp[~epp["IND_SUBCAT"].str.lower().isin(EXCLUDE_SUBCATS)].copy()
    piv_raw = epp.pivot_table(index="YEAR", columns="IND_SUBCAT", values="IND_VALUE", aggfunc="sum").sort_index().fillna(0.0)
    piv = aggregate_to_canonical(piv_raw)
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
    totals = {}
    enroll = df[df["IND_CAT"].str.lower() == "student enrollment"].copy()
    for nm in members:
        dsub = enroll[enroll["DIST_NAME"].str.lower() == nm]
        tot = dsub[dsub["IND_SUBCAT"].str.lower() == TOTAL_FTE_KEY]
        if not tot.empty:
            y = int(tot["YEAR"].max())
            totals[nm] = float(tot.loc[tot["YEAR"] == y, "IND_VALUE"].iloc[0])
    if bucket == "le_500":
        members = [n for n, v in totals.items() if v <= N_THRESHOLD]; suffix = "≤500 Students"
    else:
        members = [n for n, v in totals.items() if v > N_THRESHOLD];  suffix = ">500 Students"
    title = f"All Western MA Traditional Districts {suffix}"
    if not members: return title, pd.DataFrame(), {}, {}
    epp = df[(df["IND_CAT"].str.lower() == "expenditures per pupil") & (df["DIST_NAME"].str.lower().isin(members))][["DIST_NAME", "YEAR", "IND_SUBCAT", "IND_VALUE"]].copy()
    epp = epp[~epp["IND_SUBCAT"].str.lower().isin(EXCLUDE_SUBCATS)].copy()
    wts = df[(df["IND_CAT"].str.lower() == "student enrollment") & (df["IND_SUBCAT"].str.lower() == "in-district fte pupils") & (df["DIST_NAME"].str.lower().isin(members))][["DIST_NAME", "YEAR", "IND_VALUE"]].rename(columns={"IND_VALUE": "WEIGHT"})
    m = epp.merge(wts, on=["DIST_NAME", "YEAR"], how="left")
    m["WEIGHT"] = pd.to_numeric(m["WEIGHT"], errors="coerce").fillna(0.0)
    m["P"] = m["IND_VALUE"] * m["WEIGHT"]
    out = m.groupby(["YEAR", "IND_SUBCAT"], as_index=False).agg(NUM=("P", "sum"), DEN=("WEIGHT", "sum"), MEAN=("IND_VALUE", "mean"))
    out["VALUE"] = np.where(out["DEN"] > 0, out["NUM"]/out["DEN"], out["MEAN"])
    piv_raw = out.pivot(index="YEAR", columns="IND_SUBCAT", values="VALUE").sort_index().fillna(0.0)
    piv = aggregate_to_canonical(piv_raw)
    lines_sum, lines_mean = {}, {}
    for key, label in ENROLL_KEYS:
        sub = df[(df["IND_CAT"].str.lower() == "student enrollment") & (df["IND_SUBCAT"].str.lower() == key) & (df["DIST_NAME"].str.lower().isin(members))][["DIST_NAME", "YEAR", "IND_VALUE"]]
        if sub.empty: continue
        lines_sum[label]  = sub.groupby("YEAR")["IND_VALUE"].sum().sort_index()
        lines_mean[label] = sub.groupby("YEAR")["IND_VALUE"].mean().sort_index()
    return title, piv, lines_sum, lines_mean

# ===== Virtual district (ALPS PK-12) =====
ALPS_COMPONENTS = {"amherst", "leverett", "pelham", "shutesbury", "amherst-pelham"}

def _weighted_epp_from_parts(df_parts: pd.DataFrame) -> pd.DataFrame:
    epp = df_parts[df_parts["IND_CAT"].str.lower() == "expenditures per pupil"].copy()
    epp = epp[~epp["IND_SUBCAT"].str.lower().isin(EXCLUDE_SUBCATS)].copy()
    wts = df_parts[
        (df_parts["IND_CAT"].str.lower() == "student enrollment")
        & (df_parts["IND_SUBCAT"].str.lower() == "in-district fte pupils")
    ][["DIST_NAME", "YEAR", "IND_VALUE"]].rename(columns={"IND_VALUE": "WEIGHT"}).copy()
    m = epp.merge(wts, on=["DIST_NAME", "YEAR"], how="left")
    m["WEIGHT"] = pd.to_numeric(m["WEIGHT"], errors="coerce").fillna(0.0)
    m["P"] = m["IND_VALUE"] * m["WEIGHT"]
    out = m.groupby(["YEAR", "IND_SUBCAT"], as_index=False).agg(
        NUM=("P", "sum"), DEN=("WEIGHT", "sum"), MEAN=("IND_VALUE", "mean")
    )
    out["VALUE"] = np.where(out["DEN"] > 0, out["NUM"] / out["DEN"], out["MEAN"])
    piv_raw = out.pivot(index="YEAR", columns="IND_SUBCAT", values="VALUE").sort_index().fillna(0.0)
    return aggregate_to_canonical(piv_raw)

def add_alps_pk12(df: pd.DataFrame) -> pd.DataFrame:
    base = df[df["DIST_NAME"].str.lower().isin(ALPS_COMPONENTS)].copy()
    if base.empty:
        return df

    # 1) Category EPP (weighted, by subcategory → canonical columns)
    piv_epp = _weighted_epp_from_parts(base)
    epp_rows = piv_epp.stack().reset_index()
    epp_rows = epp_rows.rename(columns={"level_1": "IND_SUBCAT", 0: "IND_VALUE"})
    epp_rows["DIST_NAME"] = "ALPS PK-12"
    epp_rows["IND_CAT"] = "Expenditures Per Pupil"

    # 2) Enrollment lines: sum across components for each key
    enr_parts = []
    for key, _label in ENROLL_KEYS:
        sub = base[
            (base["IND_CAT"].str.lower() == "student enrollment")
            & (base["IND_SUBCAT"].str.lower() == key)
        ][["YEAR", "IND_VALUE"]].copy()
        if sub.empty:
            continue
        s = sub.groupby("YEAR")["IND_VALUE"].sum().sort_index()
        out = s.reset_index().rename(columns={"IND_VALUE": "IND_VALUE"})
        out["DIST_NAME"] = "ALPS PK-12"
        out["IND_CAT"] = "Student Enrollment"
        out["IND_SUBCAT"] = key
        enr_parts.append(out)

    rows_enroll = pd.concat(enr_parts, ignore_index=True) if enr_parts else pd.DataFrame(
        columns=["YEAR","IND_VALUE","DIST_NAME","IND_CAT","IND_SUBCAT"]
    )

    to_add = pd.concat([
        epp_rows[["DIST_NAME","YEAR","IND_SUBCAT","IND_VALUE","IND_CAT"]],
        rows_enroll[["DIST_NAME","YEAR","IND_SUBCAT","IND_VALUE","IND_CAT"]]
    ], ignore_index=True)
    to_add = to_add.astype({"DIST_NAME": str, "IND_CAT": str, "IND_SUBCAT": str})
    out = pd.concat([df, to_add], ignore_index=True)
    return out

# ---------------- Axis helpers ----------------
def _nice_ceiling(x: float, step: int) -> float:
    if x <= 0: return step
    return math.ceil(x / step) * step

def compute_global_dollar_ylim(pivots: List[pd.DataFrame], pad: float = 1.05, step: int = 500) -> float:
    tops = []
    for piv in pivots:
        if piv is None or piv.empty: continue
        totals = piv.sum(axis=1)
        if not totals.empty: tops.append(float(totals.max()))
    if not tops: return step
    m = max(tops) * pad
    return _nice_ceiling(m, step)

def compute_districts_fte_ylim(lines_list: List[Dict[str, pd.Series]], pad: float = 1.05, step: int = 50) -> float:
    tops = []
    for lines in lines_list:
        for s in lines.values():
            if s is not None and not s.empty: tops.append(float(s.max()))
    if not tops: return step
    m = max(tops) * pad
    return _nice_ceiling(m, step)

# ---------------- Utility functions for calculations ----------------
def mean_clean(arr: List[float]) -> float:
    """Compute mean of array, filtering out NaN values."""
    arr = [a for a in arr if a == a and not np.isnan(a)]
    return float(np.mean(arr)) if arr else float("nan")

def get_latest_year(df: pd.DataFrame, pivot: pd.DataFrame = None) -> int:
    """Get latest year consistently - prefer pivot index if available, else df YEAR column."""
    if pivot is not None and not pivot.empty and len(pivot.index) > 0:
        return int(pivot.index.max())
    return int(df["YEAR"].max())

def weighted_epp_aggregation(df: pd.DataFrame, districts: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Compute weighted per-pupil expenditures for a list of districts.

    Args:
        df: DataFrame with district expenditure and enrollment data
        districts: List of district names to aggregate

    Returns:
        Tuple of (epp_pivot, enrollment_series):
        - epp_pivot: DataFrame with years as index, canonical categories as columns
        - enrollment_series: Series with in-district FTE enrollment sum by year

    Notes:
        - Uses in-district FTE pupils as weights
        - Returns empty DataFrame/Series if no matching districts found
        - Weighted average: sum(value * weight) / sum(weight) for each year/category
    """
    if df is None or df.empty:
        return pd.DataFrame(), pd.Series(dtype=float)
    if not districts:
        return pd.DataFrame(), pd.Series(dtype=float)

    names_low = {d.lower() for d in districts}
    present = set(df["DIST_NAME"].str.lower())
    members = [n for n in names_low if n in present]

    if not members:
        return pd.DataFrame(), pd.Series(dtype=float)

    # Get expenditures per pupil data
    epp = df[
        (df["IND_CAT"].str.lower() == "expenditures per pupil") &
        (df["DIST_NAME"].str.lower().isin(members))
    ][["DIST_NAME", "YEAR", "IND_SUBCAT", "IND_VALUE"]].copy()
    epp = epp[~epp["IND_SUBCAT"].str.lower().isin(EXCLUDE_SUBCATS)].copy()

    # Get enrollment weights
    wts = df[
        (df["IND_CAT"].str.lower() == "student enrollment") &
        (df["IND_SUBCAT"].str.lower() == "in-district fte pupils") &
        (df["DIST_NAME"].str.lower().isin(members))
    ][["DIST_NAME", "YEAR", "IND_VALUE"]].rename(columns={"IND_VALUE": "WEIGHT"}).copy()

    # Merge and compute weighted average
    m = epp.merge(wts, on=["DIST_NAME", "YEAR"], how="left")
    m["WEIGHT"] = pd.to_numeric(m["WEIGHT"], errors="coerce").fillna(0.0)
    m["P"] = m["IND_VALUE"] * m["WEIGHT"]

    out = m.groupby(["YEAR", "IND_SUBCAT"], as_index=False).agg(
        NUM=("P", "sum"),
        DEN=("WEIGHT", "sum"),
        MEAN=("IND_VALUE", "mean")
    )
    out["VALUE"] = np.where(out["DEN"] > 0, out["NUM"] / out["DEN"], out["MEAN"])

    piv_raw = out.pivot(index="YEAR", columns="IND_SUBCAT", values="VALUE").sort_index().fillna(0.0)
    piv = aggregate_to_canonical(piv_raw)

    # Get enrollment sums (both in-district and out-of-district)
    enr_in = df[
        (df["IND_CAT"].str.lower() == "student enrollment") &
        (df["IND_SUBCAT"].str.lower() == "in-district fte pupils") &
        (df["DIST_NAME"].str.lower().isin(members))
    ][["YEAR", "IND_VALUE"]]
    enroll_in_sum = enr_in.groupby("YEAR")["IND_VALUE"].sum().sort_index() if not enr_in.empty else pd.Series(dtype=float)

    enr_out = df[
        (df["IND_CAT"].str.lower() == "student enrollment") &
        (df["IND_SUBCAT"].str.lower() == "out-of-district fte pupils") &
        (df["DIST_NAME"].str.lower().isin(members))
    ][["YEAR", "IND_VALUE"]]
    enroll_out_sum = enr_out.groupby("YEAR")["IND_VALUE"].sum().sort_index() if not enr_out.empty else pd.Series(dtype=float)

    return piv, enroll_in_sum, enroll_out_sum

# ---------------- Chapter 70 and Net School Spending (NSS) Data ----------------
def prepare_district_nss_ch70(
    df: pd.DataFrame,
    c70: pd.DataFrame,
    dist: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare Chapter 70 and Net School Spending data for a single district in absolute dollars.

    Args:
        df: Expenditure/enrollment DataFrame
        c70: profile_DataC70 DataFrame
        dist: District name

    Returns:
        Tuple of (nss_pivot, enrollment_series):
        - nss_pivot: DataFrame with years as index, columns: ['Ch70 Aid', 'Req NSS (adj)', 'Actual NSS (adj)']
          All values are in absolute dollars
        - enrollment_series: In-District FTE enrollment by year

    Notes:
        Stacking logic (bottom to top):
        1. Chapter 70 Aid (c70aid)
        2. Required NSS minus Ch70 (max(0, rqdnss2 - c70aid))
        3. Actual NSS minus Required NSS (actualNSS - rqdnss2)

        If c70aid > rqdnss2 (rare edge case), the "Req NSS (adj)" component will be 0.
    """
    if c70 is None or c70.empty:
        return pd.DataFrame(), pd.Series(dtype=float)

    # Get enrollment for this district
    enr_data = df[
        (df["DIST_NAME"].str.lower() == dist.lower()) &
        (df["IND_CAT"].str.lower() == "student enrollment") &
        (df["IND_SUBCAT"].str.lower() == "in-district fte pupils")
    ][["YEAR", "IND_VALUE"]].copy()

    if enr_data.empty:
        return pd.DataFrame(), pd.Series(dtype=float)

    enroll_series = enr_data.set_index("YEAR")["IND_VALUE"].sort_index()

    # Get C70 data for this district
    c70_dist = c70[c70["DIST_NAME"].str.lower() == dist.lower()][
        ["YEAR", "actualNSS", "rqdnss2", "c70aid"]
    ].copy()

    if c70_dist.empty:
        return pd.DataFrame(), enroll_series

    # Use C70 data directly (absolute dollars, not per-pupil)
    merged = c70_dist.set_index("YEAR").copy()

    # Calculate absolute dollar values
    # Stack 1 (bottom): Chapter 70 Aid
    merged["Ch70 Aid"] = merged["c70aid"]

    # Stack 2 (middle): Required NSS minus Ch70 (can be 0 if Ch70 > Required)
    merged["Req NSS (adj)"] = np.maximum(0, merged["rqdnss2"] - merged["c70aid"])

    # Stack 3 (top): Actual NSS minus Required NSS (can be negative if underfunding)
    merged["Actual NSS (adj)"] = merged["actualNSS"] - merged["rqdnss2"]

    # Create pivot with stacking columns
    nss_pivot = merged[["Ch70 Aid", "Req NSS (adj)", "Actual NSS (adj)"]].sort_index()

    return nss_pivot, enroll_series


def prepare_aggregate_nss_ch70(
    df: pd.DataFrame,
    c70: pd.DataFrame,
    districts: List[str]
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare aggregate Chapter 70 and Net School Spending data for multiple districts in absolute dollars.

    Args:
        df: Expenditure/enrollment DataFrame
        c70: profile_DataC70 DataFrame
        districts: List of district names to aggregate

    Returns:
        Tuple of (nss_pivot, enrollment_series):
        - nss_pivot: DataFrame with years as index, columns: ['Ch70 Aid', 'Req NSS (adj)', 'Actual NSS (adj)']
          All values are summed absolute dollars
        - enrollment_series: Total in-district FTE enrollment sum by year

    Notes:
        Aggregation method: sum(value) across all districts for each year/category
        Same stacking logic as single district version
    """
    if c70 is None or c70.empty:
        return pd.DataFrame(), pd.Series(dtype=float)

    if not districts:
        return pd.DataFrame(), pd.Series(dtype=float)

    names_low = {d.lower() for d in districts}

    # Get enrollment for all districts
    enr_data = df[
        (df["DIST_NAME"].str.lower().isin(names_low)) &
        (df["IND_CAT"].str.lower() == "student enrollment") &
        (df["IND_SUBCAT"].str.lower() == "in-district fte pupils")
    ][["DIST_NAME", "YEAR", "IND_VALUE"]].copy()

    if enr_data.empty:
        return pd.DataFrame(), pd.Series(dtype=float)

    # Sum enrollment across districts by year
    enroll_series = enr_data.groupby("YEAR")["IND_VALUE"].sum().sort_index()

    # Get C70 data for all districts
    c70_agg = c70[c70["DIST_NAME"].str.lower().isin(names_low)][
        ["DIST_NAME", "YEAR", "actualNSS", "rqdnss2", "c70aid"]
    ].copy()

    if c70_agg.empty:
        return pd.DataFrame(), enroll_series

    # Sum dollar amounts across districts by year
    c70_sums = c70_agg.groupby("YEAR")[["actualNSS", "rqdnss2", "c70aid"]].sum().sort_index()

    # Calculate absolute dollar values (same logic as single district)
    c70_sums["Ch70 Aid"] = c70_sums["c70aid"]
    c70_sums["Req NSS (adj)"] = np.maximum(0, c70_sums["rqdnss2"] - c70_sums["c70aid"])
    c70_sums["Actual NSS (adj)"] = c70_sums["actualNSS"] - c70_sums["rqdnss2"]

    nss_pivot = c70_sums[["Ch70 Aid", "Req NSS (adj)", "Actual NSS (adj)"]].sort_index()

    return nss_pivot, enroll_series


def prepare_aggregate_nss_ch70_weighted(
    df: pd.DataFrame,
    c70: pd.DataFrame,
    districts: List[str]
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare weighted per-district Chapter 70 and Net School Spending data for multiple districts.

    This computes weighted average dollars per district (weighted by enrollment) to enable
    proper comparison across different-sized district aggregates.

    Args:
        df: Expenditure/enrollment DataFrame
        c70: profile_DataC70 DataFrame
        districts: List of district names to aggregate

    Returns:
        Tuple of (nss_pivot, enrollment_series):
        - nss_pivot: DataFrame with years as index, columns: ['Ch70 Aid', 'Req NSS (adj)', 'Actual NSS (adj)']
          All values are weighted average dollars per district
        - enrollment_series: Total in-district FTE enrollment sum by year

    Notes:
        Aggregation method:
        1. Convert each district's absolute dollars to per-pupil
        2. Compute enrollment-weighted average per-pupil across districts
        3. Multiply by average enrollment to get weighted avg $ per district
        This produces per-district values weighted by enrollment size
    """
    if c70 is None or c70.empty:
        return pd.DataFrame(), pd.Series(dtype=float)

    if not districts:
        return pd.DataFrame(), pd.Series(dtype=float)

    names_low = {d.lower() for d in districts}

    # Get enrollment for all districts
    enr_data = df[
        (df["DIST_NAME"].str.lower().isin(names_low)) &
        (df["IND_CAT"].str.lower() == "student enrollment") &
        (df["IND_SUBCAT"].str.lower() == "in-district fte pupils")
    ][["DIST_NAME", "YEAR", "IND_VALUE"]].copy()

    if enr_data.empty:
        return pd.DataFrame(), pd.Series(dtype=float)

    # Sum enrollment across districts by year
    enroll_series = enr_data.groupby("YEAR")["IND_VALUE"].sum().sort_index()

    # Get C70 data for all districts
    c70_agg = c70[c70["DIST_NAME"].str.lower().isin(names_low)][
        ["DIST_NAME", "YEAR", "actualNSS", "rqdnss2", "c70aid"]
    ].copy()

    if c70_agg.empty:
        return pd.DataFrame(), enroll_series

    # Merge enrollment with C70 data to compute per-district weighted averages
    # First, get enrollment for each district/year
    enr_pivot = enr_data.pivot_table(index="YEAR", columns="DIST_NAME", values="IND_VALUE", fill_value=0)

    # Merge C70 data with enrollment
    merged = c70_agg.merge(enr_data.rename(columns={"IND_VALUE": "enrollment"}),
                           on=["DIST_NAME", "YEAR"], how="left")

    # For years with missing enrollment, use most recent available enrollment as proxy
    # This handles cases like 2025 where C70 data exists but enrollment data hasn't been released
    for dist_name in merged["DIST_NAME"].unique():
        dist_mask = merged["DIST_NAME"] == dist_name
        dist_data = merged[dist_mask].copy()

        # Find years with missing enrollment for this district
        missing_enr = dist_data[dist_data["enrollment"].isna() | (dist_data["enrollment"] == 0)]
        if not missing_enr.empty:
            # Get most recent available enrollment for this district
            available_enr = dist_data[dist_data["enrollment"] > 0].sort_values("YEAR", ascending=False)
            if not available_enr.empty:
                most_recent_enr = available_enr.iloc[0]["enrollment"]
                # Fill missing enrollment with most recent value
                merged.loc[dist_mask & (merged["enrollment"].isna() | (merged["enrollment"] == 0)), "enrollment"] = most_recent_enr

    # Drop rows that still have missing enrollment after forward-filling
    merged = merged[merged["enrollment"] > 0].copy()

    if merged.empty:
        return pd.DataFrame(), enroll_series

    # Compute per-pupil for each district
    merged["actualNSS_pp"] = merged["actualNSS"] / merged["enrollment"]
    merged["rqdnss2_pp"] = merged["rqdnss2"] / merged["enrollment"]
    merged["c70aid_pp"] = merged["c70aid"] / merged["enrollment"]

    # Compute enrollment-weighted average per-pupil across districts for each year
    weighted_avg = merged.groupby("YEAR").apply(
        lambda g: pd.Series({
            "actualNSS_pp": (g["actualNSS_pp"] * g["enrollment"]).sum() / g["enrollment"].sum(),
            "rqdnss2_pp": (g["rqdnss2_pp"] * g["enrollment"]).sum() / g["enrollment"].sum(),
            "c70aid_pp": (g["c70aid_pp"] * g["enrollment"]).sum() / g["enrollment"].sum(),
            "avg_enrollment": g["enrollment"].mean()  # Average enrollment per district
        }),
        include_groups=False
    ).sort_index()

    # Multiply weighted avg per-pupil by average enrollment to get weighted avg $ per district
    c70_per_district = weighted_avg.copy()
    c70_per_district["actualNSS"] = weighted_avg["actualNSS_pp"] * weighted_avg["avg_enrollment"]
    c70_per_district["rqdnss2"] = weighted_avg["rqdnss2_pp"] * weighted_avg["avg_enrollment"]
    c70_per_district["c70aid"] = weighted_avg["c70aid_pp"] * weighted_avg["avg_enrollment"]

    # Calculate stacked components
    c70_per_district["Ch70 Aid"] = c70_per_district["c70aid"]
    c70_per_district["Req NSS (adj)"] = np.maximum(0, c70_per_district["rqdnss2"] - c70_per_district["c70aid"])
    c70_per_district["Actual NSS (adj)"] = c70_per_district["actualNSS"] - c70_per_district["rqdnss2"]

    nss_pivot = c70_per_district[["Ch70 Aid", "Req NSS (adj)", "Actual NSS (adj)"]].sort_index()

    return nss_pivot, enroll_series
