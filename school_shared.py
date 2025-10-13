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

# Windows reserved device names (case-insensitive)
_WINDOWS_RESERVED_NAMES = {
    'con', 'prn', 'aux', 'nul',
    'com1', 'com2', 'com3', 'com4', 'com5', 'com6', 'com7', 'com8', 'com9',
    'lpt1', 'lpt2', 'lpt3', 'lpt4', 'lpt5', 'lpt6', 'lpt7', 'lpt8', 'lpt9'
}

def make_safe_filename(name: str) -> str:
    """
    Convert a string to a safe filename by removing/replacing problematic characters.

    Handles:
    - Windows reserved device names (CON, PRN, AUX, NUL, COM1-9, LPT1-9)
    - Special characters that can't be used in filenames
    - Unicode characters that might cause issues

    Args:
        name: The original filename/path component

    Returns:
        A safe filename that won't cause OS-level issues
    """
    if not name:
        return "unnamed"

    # Replace problematic characters with underscores
    # Handle special chars that can't be in filenames on Windows
    safe = name.replace(" ", "_")
    safe = safe.replace("-", "_")
    safe = safe.replace("(", "")
    safe = safe.replace(")", "")
    safe = safe.replace("≤", "le")  # less than or equal to
    safe = safe.replace("≥", "ge")  # greater than or equal to
    safe = safe.replace(">", "gt")   # greater than
    safe = safe.replace("<", "lt")   # less than
    safe = safe.replace("/", "_")
    safe = safe.replace("\\", "_")
    safe = safe.replace(":", "_")
    safe = safe.replace("*", "_")
    safe = safe.replace("?", "_")
    safe = safe.replace('"', "_")
    safe = safe.replace("|", "_")

    # Remove any remaining non-ASCII characters
    safe = safe.encode('ascii', 'ignore').decode('ascii')

    # Check if the result (or its stem before extension) is a Windows reserved name
    stem = safe.split('.')[0].lower()
    if stem in _WINDOWS_RESERVED_NAMES:
        safe = f"file_{safe}"  # Prefix with "file_" to avoid reserved name

    # Ensure it's not empty after sanitization
    if not safe or safe == "":
        safe = "unnamed"

    return safe

# ---------------- Enrollment Cohorts (4-tier system) ----------------
# Cohort boundaries are calculated dynamically based on IQR analysis
# Small: 0 to median (rounded to nearest 100)
# Medium: median+1 to Q3 (rounded to nearest 100)
# Large: Q3+1 to max non-outlier (rounded to nearest 100)
# Outliers: Above outlier threshold (currently >8000 FTE)

# Global cache for cohort definitions (recalculated when data changes)
_COHORT_CACHE = {"definitions": None, "data_hash": None, "outlier_threshold": None}

def calculate_outlier_threshold(enrollments: np.ndarray) -> float:
    """
    Calculate outlier threshold using IQR (Interquartile Range) method for extreme outliers.

    Uses Q3 + 3 * IQR instead of the standard 1.5 * IQR.
    This identifies only extreme high-enrollment outliers (like Springfield)
    while keeping typical large districts (like Chicopee, Holyoke, Pittsfield, Westfield).

    Args:
        enrollments: Array of enrollment values

    Returns:
        Outlier threshold value
    """
    q1 = np.percentile(enrollments, 25)
    q3 = np.percentile(enrollments, 75)
    iqr = q3 - q1
    outlier_threshold = q3 + 3.0 * iqr  # Use 3*IQR for extreme outliers only
    return outlier_threshold

def calculate_cohort_boundaries(df: pd.DataFrame, reg: pd.DataFrame, year: int = None) -> Dict[str, Dict]:
    """
    Calculate dynamic cohort boundaries based on IQR analysis of dataset for a specific year.

    This ensures cohort boundaries automatically adjust when districts are added/removed,
    enabling the system to work with any group of districts (Western MA, all MA, etc.).

    Args:
        df: DataFrame with enrollment and expenditure data
        reg: DataFrame with regional classifications
        year: Specific year to calculate cohorts for. If None, uses latest year in data.

    Returns:
        Dict with cohort definitions including ranges, labels, ylim, etc.
    """
    # Get Western MA traditional districts with valid data
    mask = (reg["EOHHS_REGION"].str.lower() == "western") & (reg["SCHOOL_TYPE"].str.lower() == "traditional")
    western_districts = sorted(set(reg[mask]["DIST_NAME"].str.lower()))
    present = set(df["DIST_NAME"].str.lower())
    western_districts = [d for d in western_districts if d in present and d not in EXCLUDE_DISTRICTS]

    # Use specified year or latest year
    target_year = year if year is not None else int(df["YEAR"].max())
    all_enrollments = []  # Track all enrollments including potential outliers

    # Collect enrollments for districts with valid PPE data (must import locally to avoid circular dependency)
    # Use IN-DISTRICT FTE for cohort calculations (not total FTE)
    for dist in western_districts:
        # Get enrollment for the target year
        ddf = df[(df["DIST_NAME"].str.lower() == dist.lower()) &
                 (df["IND_CAT"].str.lower() == "student enrollment") &
                 (df["IND_SUBCAT"].str.lower() == IN_DISTRICT_FTE_KEY) &
                 (df["YEAR"] == target_year)]
        if ddf.empty:
            # Fallback: if no in-district FTE data for this year, skip this district
            fte = 0.0
        else:
            fte = float(ddf["IND_VALUE"].iloc[0])

        # Check for valid PPE data for the target year
        ppe_data = df[
            (df["DIST_NAME"].str.lower() == dist) &
            (df["IND_CAT"].str.lower() == "expenditures per pupil") &
            (df["YEAR"] == target_year)
        ]
        total_ppe = ppe_data[~ppe_data["IND_SUBCAT"].str.lower().isin(
            ["total expenditures", "total in-district expenditures"])]["IND_VALUE"].sum()

        if fte > 0 and total_ppe > 0:
            all_enrollments.append(fte)

    if not all_enrollments:
        # Fallback to static boundaries if no data
        _COHORT_CACHE["outlier_threshold"] = 8000  # Fallback value
        return _get_static_cohort_definitions()

    # Calculate IQR statistics on ALL districts (don't filter outliers first)
    # This ensures cohort boundaries reflect the true distribution
    all_enrollments_array = np.array(all_enrollments)

    if len(all_enrollments_array) == 0:
        # Fallback if no districts
        _COHORT_CACHE["outlier_threshold"] = 10000
        return _get_static_cohort_definitions()

    # Calculate quartiles on FULL dataset (including Springfield/outliers)
    q1 = np.percentile(all_enrollments_array, 25)
    median = np.median(all_enrollments_array)
    q3 = np.percentile(all_enrollments_array, 75)
    p90 = np.percentile(all_enrollments_array, 90)  # 90th percentile for X-Large cutoff
    max_enrollment = np.max(all_enrollments_array)

    # Store fixed outlier threshold (10,000 FTE) for Springfield cutoff
    _COHORT_CACHE["outlier_threshold"] = 10000

    # Round to nearest 100 (for quartile boundaries)
    def round_100(x):
        return int(round(x / 100.0) * 100)

    # Round to nearest 1000 (for Large upper boundary)
    def round_1000(x):
        return int(round(x / 1000.0) * 1000)

    q1_rounded = round_100(q1)
    median_rounded = round_100(median)
    q3_rounded = round_100(q3)
    p90_rounded = round_1000(p90)  # Round to nearest 1000 for Large upper boundary

    # Build dynamic cohort definitions (6 tiers)
    return {
        "TINY": {
            "range": (0, q1_rounded),
            "label": f"Tiny (0-{q1_rounded} FTE)",
            "short_label": "Tiny",
            "ylim": q1_rounded,
            "name": "Cohort 1"
        },
        "SMALL": {
            "range": (q1_rounded + 1, median_rounded),
            "label": f"Small ({q1_rounded + 1}-{median_rounded} FTE)",
            "short_label": "Small",
            "ylim": median_rounded,
            "name": "Cohort 2"
        },
        "MEDIUM": {
            "range": (median_rounded + 1, q3_rounded),
            "label": f"Medium ({median_rounded + 1}-{q3_rounded} FTE)",
            "short_label": "Medium",
            "ylim": q3_rounded,
            "name": "Cohort 3"
        },
        "LARGE": {
            "range": (q3_rounded + 1, p90_rounded),
            "label": f"Large ({q3_rounded + 1}-{p90_rounded} FTE)",
            "short_label": "Large",
            "ylim": p90_rounded,
            "name": "Cohort 4"
        },
        "X-LARGE": {
            "range": (p90_rounded + 1, 10000),
            "label": f"X-Large ({p90_rounded + 1}-10K FTE)",
            "short_label": "X-Large",
            "ylim": 10000,
            "name": "Cohort 5"
        },
        "SPRINGFIELD": {
            "range": (10001, float('inf')),
            "label": "Outliers (Springfield >10K FTE)",
            "short_label": "Springfield",
            "ylim": None,
            "name": "Cohort 6"
        }
    }

def _get_static_cohort_definitions():
    """Fallback static definitions when data is unavailable."""
    return {
        "TINY": {
            "range": (0, 400),
            "label": "Tiny (0-400 FTE)",
            "short_label": "Tiny",
            "ylim": 400,
            "name": "Cohort 1"
        },
        "SMALL": {
            "range": (401, 800),
            "label": "Small (401-800 FTE)",
            "short_label": "Small",
            "ylim": 800,
            "name": "Cohort 2"
        },
        "MEDIUM": {
            "range": (801, 1800),
            "label": "Medium (801-1800 FTE)",
            "short_label": "Medium",
            "ylim": 1800,
            "name": "Cohort 3"
        },
        "LARGE": {
            "range": (1801, 5000),
            "label": "Large (1801-5000 FTE)",
            "short_label": "Large",
            "ylim": 5000,
            "name": "Cohort 4"
        },
        "X-LARGE": {
            "range": (5001, 10000),
            "label": "X-Large (5001-10K FTE)",
            "short_label": "X-Large",
            "ylim": 10000,
            "name": "Cohort 5"
        },
        "SPRINGFIELD": {
            "range": (10001, float('inf')),
            "label": "Outliers (Springfield >10K FTE)",
            "short_label": "Springfield",
            "ylim": None,
            "name": "Cohort 6"
        }
    }

def get_cohort_definitions(df: pd.DataFrame, reg: pd.DataFrame) -> Dict[str, Dict]:
    """
    Get cohort definitions, using cached values if data hasn't changed.

    This is the primary function for accessing cohort boundaries throughout the codebase.
    """
    # Create a simple hash of the data to detect changes
    data_hash = f"{len(df)}_{int(df['YEAR'].max() if not df.empty else 0)}"

    if _COHORT_CACHE["data_hash"] != data_hash or _COHORT_CACHE["definitions"] is None:
        _COHORT_CACHE["definitions"] = calculate_cohort_boundaries(df, reg)
        _COHORT_CACHE["data_hash"] = data_hash

    return _COHORT_CACHE["definitions"]

def initialize_cohort_definitions(df: pd.DataFrame, reg: pd.DataFrame):
    """
    Initialize global COHORT_DEFINITIONS and ENROLLMENT_GROUPS with dynamic boundaries.

    This MUST be called after loading data to ensure all helper functions use correct boundaries.
    """
    global COHORT_DEFINITIONS, ENROLLMENT_GROUPS

    COHORT_DEFINITIONS = get_cohort_definitions(df, reg)
    ENROLLMENT_GROUPS = {k: v["range"] for k, v in COHORT_DEFINITIONS.items()}

    return COHORT_DEFINITIONS

# Backward compatibility: static definitions for initial import
# These will be overridden by dynamic calculations when data is loaded
COHORT_DEFINITIONS = _get_static_cohort_definitions()
ENROLLMENT_GROUPS = {k: v["range"] for k, v in COHORT_DEFINITIONS.items()}

# Legacy threshold for backward compatibility (deprecated)
N_THRESHOLD = 500

# ---------------- District set ----------------
DISTRICTS_OF_INTEREST = ["Amherst-Pelham", "Amherst", "Leverett", "Pelham", "Shutesbury"]

# ---------------- Sheets ----------------
SHEET_EXPEND  = "District Expend by Category"
SHEET_REGIONS = "District Regions"

# Exclude totals
EXCLUDE_SUBCATS = {"total expenditures", "total in-district expenditures"}

# Exclude non-traditional districts (virtual schools, etc.) from analysis
EXCLUDE_DISTRICTS = {
    "greater commonwealth virtual district",  # Online school, not a traditional district
}

# Enrollment keys (order determines table row order)
ENROLL_KEYS = [
    ("in-district fte pupils", "In-District FTE Pupils"),
    ("foundation enrollment", "Foundation Enrollment"),
    ("out-of-district fte pupils", "Out-of-District FTE Pupils"),
]
TOTAL_FTE_KEY = "total fte pupils"
IN_DISTRICT_FTE_KEY = "in-district fte pupils"  # Used for cohort calculations

# ---------------- Canonical categories (bottom -> top; "Other" always last) ----------------
CANON_CATS_BOTTOM_TO_TOP = [
    "Teachers",
    "Insurance, Retirement and Other",
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
    "insurance, retirement programs and other": "Insurance, Retirement and Other",
    "insurance, retirement and other": "Insurance, Retirement and Other",
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
    "Insurance, Retirement and Other":            "#FFE082",  # golden yellow (changed from orange)
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
    "Foundation Enrollment": "#1976D2",  # blue
}

# ---------------- Horizontal Bar Plot Styling (Western MA Overview) ----------------
PPE_PEERS_YMAX          = 35000.0  # Max x-axis value for horizontal bar plots
PPE_PEERS_REMOVE_SPINES = True     # Remove all boundary lines for cleaner appearance
PPE_PEERS_BAR_EDGES     = False    # No bar edge lines

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

    # Initialize dynamic cohort definitions based on loaded data
    initialize_cohort_definitions(df, reg)

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
def get_total_fte_for_year(df: pd.DataFrame, dist: str, year: int) -> float:
    """Get total FTE enrollment for a district in a specific year."""
    ddf = df[(df["DIST_NAME"].str.lower() == dist.lower()) &
             (df["IND_CAT"].str.lower() == "student enrollment") &
             (df["IND_SUBCAT"].str.lower() == TOTAL_FTE_KEY) &
             (df["YEAR"] == year)]
    if not ddf.empty:
        return float(ddf["IND_VALUE"].iloc[0])

    # If no total FTE, sum component parts for this year
    parts = df[(df["DIST_NAME"].str.lower() == dist.lower()) &
               (df["IND_CAT"].str.lower() == "student enrollment") &
               (df["IND_SUBCAT"].str.lower().isin([k for k, _ in ENROLL_KEYS])) &
               (df["YEAR"] == year)]
    if parts.empty:
        return 0.0
    return float(parts["IND_VALUE"].sum())

def get_indistrict_fte_for_year(df: pd.DataFrame, dist: str, year: int) -> float:
    """Get in-district FTE enrollment for a district in a specific year (used for cohort calculations)."""
    ddf = df[(df["DIST_NAME"].str.lower() == dist.lower()) &
             (df["IND_CAT"].str.lower() == "student enrollment") &
             (df["IND_SUBCAT"].str.lower() == IN_DISTRICT_FTE_KEY) &
             (df["YEAR"] == year)]
    if not ddf.empty:
        return float(ddf["IND_VALUE"].iloc[0])
    return 0.0

def latest_total_fte(df: pd.DataFrame, dist: str) -> float:
    """Get latest total FTE enrollment for a district."""
    ddf = df[(df["DIST_NAME"].str.lower() == dist.lower()) & (df["IND_CAT"].str.lower() == "student enrollment") & (df["IND_SUBCAT"].str.lower() == TOTAL_FTE_KEY)]
    if ddf.empty:
        parts = df[(df["DIST_NAME"].str.lower() == dist.lower()) & (df["IND_CAT"].str.lower() == "student enrollment") & (df["IND_SUBCAT"].str.lower().isin([k for k, _ in ENROLL_KEYS]))][["YEAR", "IND_VALUE"]]
        if parts.empty: return 0.0
        s = parts.groupby("YEAR")["IND_VALUE"].sum().sort_index()
        return float(s.iloc[-1])
    y = int(ddf["YEAR"].max())
    return float(ddf.loc[ddf["YEAR"] == y, "IND_VALUE"].iloc[0])

def latest_indistrict_fte(df: pd.DataFrame, dist: str) -> float:
    """Get latest in-district FTE enrollment for a district (used for cohort calculations)."""
    ddf = df[(df["DIST_NAME"].str.lower() == dist.lower()) &
             (df["IND_CAT"].str.lower() == "student enrollment") &
             (df["IND_SUBCAT"].str.lower() == IN_DISTRICT_FTE_KEY)]
    if ddf.empty:
        return 0.0
    y = int(ddf["YEAR"].max())
    return float(ddf.loc[ddf["YEAR"] == y, "IND_VALUE"].iloc[0])

def _get_enrollment_group_for_boundaries(fte: float, enrollment_groups: Dict[str, Tuple[float, float]]) -> str:
    """
    Determine enrollment cohort for a given FTE value using custom boundaries.

    Args:
        fte: Enrollment value
        enrollment_groups: Dict mapping cohort names to (min, max) tuples

    Returns: "TINY", "SMALL", "MEDIUM", "LARGE", "X-LARGE", or "SPRINGFIELD"
    """
    for group, (min_fte, max_fte) in enrollment_groups.items():
        if min_fte <= fte <= max_fte:
            return group
    return "SMALL"  # Default fallback

def get_enrollment_group(fte: float) -> str:
    """
    Determine enrollment cohort for a given FTE value using global boundaries.
    Returns: "TINY", "SMALL", "MEDIUM", "LARGE", "X-LARGE", or "SPRINGFIELD"
    """
    for group, (min_fte, max_fte) in ENROLLMENT_GROUPS.items():
        if min_fte <= fte <= max_fte:
            return group
    return "SMALL"  # Default fallback

def get_cohort_label(group: str) -> str:
    """Get the full label for a cohort (e.g., 'Small (0-800 FTE)')."""
    return COHORT_DEFINITIONS.get(group, {}).get("label", group)

def get_cohort_short_label(group: str) -> str:
    """Get the short label for a cohort (e.g., 'Small')."""
    return COHORT_DEFINITIONS.get(group, {}).get("short_label", group)

def get_cohort_2024_label(group: str) -> str:
    """Get the 2024 cohort label for PPE/CH70/NSS comparisons (e.g., '2024 Medium cohort')."""
    short_label = COHORT_DEFINITIONS.get(group, {}).get("short_label", group)
    return f"2024 {short_label} cohort"

def get_cohort_ylim(group: str) -> int | None:
    """Get the y-axis limit for a cohort's enrollment plots."""
    return COHORT_DEFINITIONS.get(group, {}).get("ylim")

def get_cohort_range(group: str) -> tuple[int, int | float]:
    """Get the (min, max) FTE range for a cohort."""
    return COHORT_DEFINITIONS.get(group, {}).get("range", (0, 0))

def get_outlier_threshold() -> float:
    """Get the statistical outlier threshold (Q3 + 1.5*IQR) from cache."""
    return _COHORT_CACHE.get("outlier_threshold", 8000)  # Fallback to 8000 if not calculated

def get_western_cohort_districts(df: pd.DataFrame, reg: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Get Western MA traditional districts organized by enrollment cohort.

    This ensures PPE and NSS analyses use identical district lists by centralizing
    the filtering logic in one place.

    ONLY includes districts with valid PPE data (total_ppe > 0 for latest year).
    This filters out districts with missing/incomplete expenditure data.

    Cohorts are determined by IN-DISTRICT FTE enrollment (not total FTE).

    Returns:
        Dict with keys "TINY", "SMALL", "MEDIUM", "LARGE", "X-LARGE", "SPRINGFIELD", each containing
        a list of lowercase district names that have enrollment data and valid PPE.
    """
    mask = (reg["EOHHS_REGION"].str.lower() == "western") & (reg["SCHOOL_TYPE"].str.lower() == "traditional")
    western_districts = sorted(set(reg[mask]["DIST_NAME"].str.lower()))
    present = set(df["DIST_NAME"].str.lower())
    western_districts = [d for d in western_districts if d in present and d not in EXCLUDE_DISTRICTS]

    cohorts = {"TINY": [], "SMALL": [], "MEDIUM": [], "LARGE": [], "X-LARGE": [], "SPRINGFIELD": []}
    latest_year = int(df["YEAR"].max())

    for dist in western_districts:
        fte = latest_indistrict_fte(df, dist)  # Use IN-DISTRICT FTE for cohort assignment

        # Check if district has valid PPE data for latest year
        ppe_data = df[
            (df["DIST_NAME"].str.lower() == dist) &
            (df["IND_CAT"].str.lower() == "expenditures per pupil") &
            (df["YEAR"] == latest_year)
        ]
        total_ppe = ppe_data[~ppe_data["IND_SUBCAT"].str.lower().isin(
            ["total expenditures", "total in-district expenditures"])]["IND_VALUE"].sum()

        # ONLY include districts with valid enrollment AND valid PPE data
        if fte > 0 and total_ppe > 0:
            group = get_enrollment_group(fte)
            if group in cohorts:
                cohorts[group].append(dist)

    return cohorts

def get_western_cohort_districts_for_year(df: pd.DataFrame, reg: pd.DataFrame, year: int) -> Dict[str, List[str]]:
    """
    Get Western MA traditional districts organized by enrollment cohort for a specific year.

    Like get_western_cohort_districts but for a specific year instead of latest.

    ONLY includes districts with valid PPE data (total_ppe > 0 for specified year).
    This filters out districts with missing/incomplete expenditure data.

    Cohorts are determined by IN-DISTRICT FTE enrollment (not total FTE).

    Returns:
        Dict with keys "TINY", "SMALL", "MEDIUM", "LARGE", "X-LARGE", "SPRINGFIELD", each containing
        a list of lowercase district names that have enrollment data and valid PPE for the year.
    """
    mask = (reg["EOHHS_REGION"].str.lower() == "western") & (reg["SCHOOL_TYPE"].str.lower() == "traditional")
    western_districts = sorted(set(reg[mask]["DIST_NAME"].str.lower()))
    present = set(df["DIST_NAME"].str.lower())
    western_districts = [d for d in western_districts if d in present and d not in EXCLUDE_DISTRICTS]

    # Calculate year-specific cohort boundaries
    year_cohort_defs = calculate_cohort_boundaries(df, reg, year)
    year_enrollment_groups = {k: v["range"] for k, v in year_cohort_defs.items()}

    cohorts = {"TINY": [], "SMALL": [], "MEDIUM": [], "LARGE": [], "X-LARGE": [], "SPRINGFIELD": []}

    for dist in western_districts:
        fte = get_indistrict_fte_for_year(df, dist, year)  # Use IN-DISTRICT FTE for cohort assignment

        # Check if district has valid PPE data for specified year
        ppe_data = df[
            (df["DIST_NAME"].str.lower() == dist) &
            (df["IND_CAT"].str.lower() == "expenditures per pupil") &
            (df["YEAR"] == year)
        ]
        total_ppe = ppe_data[~ppe_data["IND_SUBCAT"].str.lower().isin(
            ["total expenditures", "total in-district expenditures"])]["IND_VALUE"].sum()

        # ONLY include districts with valid enrollment AND valid PPE data
        if fte > 0 and total_ppe > 0:
            # Use year-specific cohort boundaries
            group = _get_enrollment_group_for_boundaries(fte, year_enrollment_groups)
            if group in cohorts:
                cohorts[group].append(dist)

    return cohorts

def get_omitted_western_districts(df: pd.DataFrame, reg: pd.DataFrame) -> List[str]:
    """
    Get list of Western MA traditional districts omitted from analysis due to missing PPE data.

    Returns:
        List of (district_name, reason) tuples for districts excluded from analysis.
    """
    mask = (reg["EOHHS_REGION"].str.lower() == "western") & (reg["SCHOOL_TYPE"].str.lower() == "traditional")
    western_districts = sorted(set(reg[mask]["DIST_NAME"].str.lower()))
    present = set(df["DIST_NAME"].str.lower())
    western_districts = [d for d in western_districts if d in present]

    latest_year = int(df["YEAR"].max())
    omitted = []

    for dist in western_districts:
        fte = latest_total_fte(df, dist)

        # Check if district has valid PPE data for latest year
        ppe_data = df[
            (df["DIST_NAME"].str.lower() == dist) &
            (df["IND_CAT"].str.lower() == "expenditures per pupil") &
            (df["YEAR"] == latest_year)
        ]
        total_ppe = ppe_data[~ppe_data["IND_SUBCAT"].str.lower().isin(
            ["total expenditures", "total in-district expenditures"])]["IND_VALUE"].sum()

        # Identify why district was omitted
        if fte <= 0:
            omitted.append((dist.title(), "no enrollment data"))
        elif total_ppe <= 0:
            omitted.append((dist.title(), "missing expenditure data"))

    return omitted

def context_for_district(df: pd.DataFrame, dist: str) -> str:
    """Get enrollment group context for a district (TINY, SMALL, MEDIUM, LARGE, X-LARGE, or SPRINGFIELD).
    Uses IN-DISTRICT FTE for cohort assignment."""
    fte = latest_indistrict_fte(df, dist)
    return get_enrollment_group(fte)

def context_for_western(bucket: str) -> str:
    """Get context for Western MA aggregate bucket."""
    # Map bucket names to enrollment groups
    bucket_map = {
        "tiny": "TINY",
        "small": "SMALL",
        "medium": "MEDIUM",
        "large": "LARGE",
        "x-large": "X-LARGE",
        "springfield": "SPRINGFIELD"
    }
    return bucket_map.get(bucket.lower(), "TINY")

def prepare_district_epp_lines(df: pd.DataFrame, dist: str, c70: pd.DataFrame = None) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
    ddf = df[df["DIST_NAME"].str.lower() == dist.lower()].copy()
    epp = ddf[ddf["IND_CAT"].str.lower() == "expenditures per pupil"].copy()
    epp = epp[~epp["IND_SUBCAT"].str.lower().isin(EXCLUDE_SUBCATS)].copy()
    piv_raw = epp.pivot_table(index="YEAR", columns="IND_SUBCAT", values="IND_VALUE", aggfunc="sum").sort_index().fillna(0.0)
    piv = aggregate_to_canonical(piv_raw)
    lines: Dict[str, pd.Series] = {}
    enroll = ddf[ddf["IND_CAT"].str.lower() == "student enrollment"].copy()
    for key, label in ENROLL_KEYS:
        if key == "foundation enrollment":
            # Get foundation enrollment from c70 DataFrame (cap at 2024 since 2025 is projection)
            if c70 is not None:
                c70_dist = c70[(c70["DIST_NAME"].str.lower() == dist.lower()) & (c70["YEAR"] <= 2024)].copy()
                if not c70_dist.empty:
                    ser = c70_dist[["YEAR", "distfoundenro"]].dropna()
                    if not ser.empty:
                        s = ser.drop_duplicates(subset=["YEAR"]).set_index("YEAR")["distfoundenro"].sort_index()
                        lines[label] = s
        else:
            ser = enroll[enroll["IND_SUBCAT"].str.lower() == key][["YEAR", "IND_VALUE"]]
            if not ser.empty:
                s = ser.dropna().drop_duplicates(subset=["YEAR"]).set_index("YEAR")["IND_VALUE"].sort_index()
                lines[label] = s
    return piv, lines

def prepare_western_epp_lines(df: pd.DataFrame, reg: pd.DataFrame, bucket: str, c70: pd.DataFrame = None, districts: List[str] = None) -> Tuple[str, pd.DataFrame, Dict[str, pd.Series], Dict[str, pd.Series]]:
    """
    Prepare Western MA aggregate data for a given enrollment bucket.

    Args:
        df: Main data DataFrame
        reg: Regions DataFrame
        bucket: One of "small", "medium", "large", or "springfield"
        c70: Chapter 70 data DataFrame (optional, for foundation enrollment)
        districts: Optional pre-filtered list of district names. If provided, uses this list
                  instead of filtering internally. This ensures consistency with NSS analysis.

    Returns:
        (title, epp_pivot, lines_sum, lines_mean)
    """
    # Use pre-filtered district list if provided, otherwise filter internally (legacy mode)
    if districts is not None:
        members = districts
    else:
        mask = (reg["EOHHS_REGION"].str.lower() == "western") & (reg["SCHOOL_TYPE"].str.lower() == "traditional")
        members = sorted(set(reg[mask]["DIST_NAME"].str.lower()))
        present = set(df["DIST_NAME"].str.lower())
        members = [m for m in members if m in present]

        # Calculate IN-DISTRICT enrollment for each district (used for cohort assignment)
        totals = {}
        enroll = df[df["IND_CAT"].str.lower() == "student enrollment"].copy()
        for nm in members:
            dsub = enroll[enroll["DIST_NAME"].str.lower() == nm]
            tot = dsub[dsub["IND_SUBCAT"].str.lower() == IN_DISTRICT_FTE_KEY]
            if not tot.empty:
                y = int(tot["YEAR"].max())
                totals[nm] = float(tot.loc[tot["YEAR"] == y, "IND_VALUE"].iloc[0])

        # Filter members based on bucket - use centralized cohort definitions
        bucket_lower = bucket.lower()
        if bucket_lower == "tiny":
            min_fte, max_fte = ENROLLMENT_GROUPS["TINY"]
            members = [n for n, v in totals.items() if min_fte <= v <= max_fte]
        elif bucket_lower == "small":
            min_fte, max_fte = ENROLLMENT_GROUPS["SMALL"]
            members = [n for n, v in totals.items() if min_fte <= v <= max_fte]
        elif bucket_lower == "medium":
            min_fte, max_fte = ENROLLMENT_GROUPS["MEDIUM"]
            members = [n for n, v in totals.items() if min_fte <= v <= max_fte]
        elif bucket_lower == "large":
            min_fte, max_fte = ENROLLMENT_GROUPS["LARGE"]
            members = [n for n, v in totals.items() if min_fte <= v <= max_fte]
        elif bucket_lower == "x-large":
            min_fte, max_fte = ENROLLMENT_GROUPS["X-LARGE"]
            members = [n for n, v in totals.items() if min_fte <= v <= max_fte]
        elif bucket_lower == "springfield":
            min_fte, max_fte = ENROLLMENT_GROUPS["SPRINGFIELD"]
            members = [n for n, v in totals.items() if min_fte <= v <= max_fte]
        else:
            # Legacy support: le_500 -> small, gt_500 -> large
            if bucket == "le_500":
                members = [n for n, v in totals.items() if v <= 500]
            else:
                members = [n for n, v in totals.items() if v > 500]

    # Generate suffix/title using centralized labels
    bucket_lower = bucket.lower()
    if bucket_lower == "tiny":
        suffix = get_cohort_label("TINY")
    elif bucket_lower == "small":
        suffix = get_cohort_label("SMALL")
    elif bucket_lower == "medium":
        suffix = get_cohort_label("MEDIUM")
    elif bucket_lower == "large":
        suffix = get_cohort_label("LARGE")
    elif bucket_lower == "x-large":
        suffix = get_cohort_label("X-LARGE")
    elif bucket_lower == "springfield":
        # Get latest IN-DISTRICT FTE for title - format as "Outliers (Springfield at X FTE in YYYY)"
        if members:
            enroll = df[df["IND_CAT"].str.lower() == "student enrollment"].copy()
            nm = members[0]  # Should be Springfield
            dsub = enroll[enroll["DIST_NAME"].str.lower() == nm]
            tot = dsub[dsub["IND_SUBCAT"].str.lower() == IN_DISTRICT_FTE_KEY]
            if not tot.empty:
                y = int(tot["YEAR"].max())
                fte_val = float(tot.loc[tot["YEAR"] == y, "IND_VALUE"].iloc[0])
                suffix = f"Outliers ({nm.title()} at {fte_val:,.0f} FTE in {y})"
            else:
                suffix = "Outliers (Springfield)"
        else:
            suffix = "Outliers (Springfield)"
    else:
        # Legacy fallback (should not be used with new 6-tier system)
        suffix = "≤500 Students" if bucket == "le_500" else ">500 Students"

    title = f"All Western MA Traditional Districts: {suffix}"
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
        if key == "foundation enrollment":
            # Get foundation enrollment from c70 DataFrame (cap at 2024 since 2025 is projection)
            if c70 is not None:
                c70_members = c70[(c70["DIST_NAME"].str.lower().isin(members)) & (c70["YEAR"] <= 2024)].copy()
                if not c70_members.empty:
                    ser = c70_members[["YEAR", "distfoundenro"]].dropna()
                    if not ser.empty:
                        lines_sum[label] = ser.groupby("YEAR")["distfoundenro"].sum().sort_index()
                        lines_mean[label] = ser.groupby("YEAR")["distfoundenro"].mean().sort_index()
        else:
            sub = df[(df["IND_CAT"].str.lower() == "student enrollment") & (df["IND_SUBCAT"].str.lower() == key) & (df["DIST_NAME"].str.lower().isin(members))][["DIST_NAME", "YEAR", "IND_VALUE"]]
            if sub.empty: continue
            lines_sum[label]  = sub.groupby("YEAR")["IND_VALUE"].sum().sort_index()
            lines_mean[label] = sub.groupby("YEAR")["IND_VALUE"].mean().sort_index()
    return title, piv, lines_sum, lines_mean

# Note: ALPS PK-12 aggregate concept removed.
# Individual districts now compared to enrollment-based peer groups.

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
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare Chapter 70 and Net School Spending data for a single district in absolute dollars.

    Args:
        df: Expenditure/enrollment DataFrame
        c70: profile_DataC70 DataFrame
        dist: District name

    Returns:
        Tuple of (nss_pivot, enrollment_series, foundation_enrollment):
        - nss_pivot: DataFrame with years as index, columns: ['Ch70 Aid', 'Req NSS (adj)', 'Actual NSS (adj)']
          All values are in absolute dollars
        - enrollment_series: In-District FTE enrollment by year
        - foundation_enrollment: Foundation enrollment by year (from c70 distfoundenro, capped at 2024)

    Notes:
        Stacking logic (bottom to top):
        1. Chapter 70 Aid (c70aid)
        2. Required NSS minus Ch70 (max(0, rqdnss2 - c70aid))
        3. Actual NSS minus Required NSS (actualNSS - rqdnss2)

        If c70aid > rqdnss2 (rare edge case), the "Req NSS (adj)" component will be 0.
    """
    if c70 is None or c70.empty:
        return pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=float)

    # Get enrollment for this district
    enr_data = df[
        (df["DIST_NAME"].str.lower() == dist.lower()) &
        (df["IND_CAT"].str.lower() == "student enrollment") &
        (df["IND_SUBCAT"].str.lower() == "in-district fte pupils")
    ][["YEAR", "IND_VALUE"]].copy()

    if enr_data.empty:
        return pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=float)

    enroll_series = enr_data.set_index("YEAR")["IND_VALUE"].sort_index()

    # Get C70 data for this district (including foundation enrollment)
    c70_dist = c70[(c70["DIST_NAME"].str.lower() == dist.lower()) & (c70["YEAR"] <= 2024)][
        ["YEAR", "actualNSS", "rqdnss2", "c70aid", "distfoundenro"]
    ].copy()

    if c70_dist.empty:
        return pd.DataFrame(), enroll_series, pd.Series(dtype=float)

    # Extract foundation enrollment
    foundation_enr = c70_dist[["YEAR", "distfoundenro"]].dropna()
    if not foundation_enr.empty:
        foundation_series = foundation_enr.set_index("YEAR")["distfoundenro"].sort_index()
    else:
        foundation_series = pd.Series(dtype=float)

    # Use C70 data directly (absolute dollars, not per-pupil)
    merged = c70_dist.set_index("YEAR").copy()

    # Calculate absolute dollar values
    # Stack 1 (bottom): Chapter 70 Aid
    merged["Ch70 Aid"] = merged["c70aid"]

    # Stack 2 (middle): Required NSS minus Ch70 (can be 0 if Ch70 > Required)
    merged["Req NSS (minus Ch70)"] = np.maximum(0, merged["rqdnss2"] - merged["c70aid"])

    # Stack 3 (top): Actual NSS minus Required NSS (can be negative if underfunding)
    merged["Actual NSS (minus Req NSS)"] = merged["actualNSS"] - merged["rqdnss2"]

    # Create pivot with stacking columns
    nss_pivot = merged[["Ch70 Aid", "Req NSS (minus Ch70)", "Actual NSS (minus Req NSS)"]].sort_index()

    return nss_pivot, enroll_series, foundation_series


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

    # Compute weighted average enrollment per district (sum / count of districts)
    enroll_sum = enr_data.groupby("YEAR")["IND_VALUE"].sum()
    district_count = enr_data.groupby("YEAR")["IND_VALUE"].count()
    enroll_series = (enroll_sum / district_count).sort_index()

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
    c70_sums["Req NSS (minus Ch70)"] = np.maximum(0, c70_sums["rqdnss2"] - c70_sums["c70aid"])
    c70_sums["Actual NSS (minus Req NSS)"] = c70_sums["actualNSS"] - c70_sums["rqdnss2"]

    nss_pivot = c70_sums[["Ch70 Aid", "Req NSS (minus Ch70)", "Actual NSS (minus Req NSS)"]].sort_index()

    return nss_pivot, enroll_series


def prepare_aggregate_nss_ch70_weighted(
    df: pd.DataFrame,
    c70: pd.DataFrame,
    districts: List[str]
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare weighted per-district Chapter 70 and Net School Spending data for multiple districts.

    This computes weighted average dollars per district (weighted by enrollment) to enable
    proper comparison across different-sized district aggregates.

    Args:
        df: Expenditure/enrollment DataFrame
        c70: profile_DataC70 DataFrame
        districts: List of district names to aggregate

    Returns:
        Tuple of (nss_pivot, enrollment_series, foundation_enrollment):
        - nss_pivot: DataFrame with years as index, columns: ['Ch70 Aid', 'Req NSS (adj)', 'Actual NSS (adj)']
          All values are weighted average dollars per district
        - enrollment_series: Total in-district FTE enrollment sum by year
        - foundation_enrollment: Total foundation enrollment sum by year (from c70 distfoundenro, capped at 2024)

    Notes:
        Aggregation method:
        1. Convert each district's absolute dollars to per-pupil
        2. Compute enrollment-weighted average per-pupil across districts
        3. Multiply by average enrollment to get weighted avg $ per district
        This produces per-district values weighted by enrollment size
    """
    if c70 is None or c70.empty:
        return pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=float)

    if not districts:
        return pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=float)

    names_low = {d.lower() for d in districts}

    # Get enrollment for all districts
    enr_data = df[
        (df["DIST_NAME"].str.lower().isin(names_low)) &
        (df["IND_CAT"].str.lower() == "student enrollment") &
        (df["IND_SUBCAT"].str.lower() == "in-district fte pupils")
    ][["DIST_NAME", "YEAR", "IND_VALUE"]].copy()

    if enr_data.empty:
        return pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=float)

    # Compute weighted average enrollment per district (sum / count of districts)
    enroll_sum = enr_data.groupby("YEAR")["IND_VALUE"].sum()
    district_count = enr_data.groupby("YEAR")["IND_VALUE"].count()
    enroll_series = (enroll_sum / district_count).sort_index()

    # Get C70 data for all districts
    # Get C70 data for all districts (including foundation enrollment, capped at 2024)
    c70_agg = c70[(c70["DIST_NAME"].str.lower().isin(names_low)) & (c70["YEAR"] <= 2024)][
        ["DIST_NAME", "YEAR", "actualNSS", "rqdnss2", "c70aid", "distfoundenro"]
    ].copy()

    if c70_agg.empty:
        return pd.DataFrame(), enroll_series, pd.Series(dtype=float)

    # Extract foundation enrollment and compute weighted average per district
    # (sum of enrollment divided by number of districts)
    foundation_enr = c70_agg[["YEAR", "distfoundenro"]].dropna()
    if not foundation_enr.empty:
        foundation_sum = foundation_enr.groupby("YEAR")["distfoundenro"].sum()
        district_count = foundation_enr.groupby("YEAR")["distfoundenro"].count()
        foundation_series = (foundation_sum / district_count).sort_index()
    else:
        foundation_series = pd.Series(dtype=float)

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
    c70_per_district["Req NSS (minus Ch70)"] = np.maximum(0, c70_per_district["rqdnss2"] - c70_per_district["c70aid"])
    c70_per_district["Actual NSS (minus Req NSS)"] = c70_per_district["actualNSS"] - c70_per_district["rqdnss2"]

    nss_pivot = c70_per_district[["Ch70 Aid", "Req NSS (minus Ch70)", "Actual NSS (minus Req NSS)"]].sort_index()

    return nss_pivot, enroll_series, foundation_series
