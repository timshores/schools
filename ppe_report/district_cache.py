"""
District-level data caching system for CR CacheMonster01.

This module provides CSV caching for individual district data to:
1. Enable QA by desk-checking CSVs against printed PDF reports
2. Improve performance by computing each district's data only once
3. Make reports extensible to any MA district (not just Western MA)

Architecture:
- Cache ALL Massachusetts districts (not just Western MA)
- Store 3 CSV files per district: epp_pivot, lines, nss_pivot
- Reports load from cache instead of recomputing
- Cohort aggregations load district caches and aggregate

Usage:
    from district_cache import ensure_all_districts_cached, load_district_data

    # Generate cache for all districts (run once or when source data changes)
    ensure_all_districts_cached(df, reg, c70, force=False)

    # Load a specific district's data
    epp_pivot, lines, nss_pivot = load_district_data("Amherst-Pelham")
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import pandas as pd
import json
from datetime import datetime

# Cache directory structure
CACHE_DIR = Path("./cache")
DISTRICT_CACHE_DIR = Path("./cache/district_data")
COHORT_CACHE_DIR = Path("./cache/cohort_data")
EXECUTIVE_SUMMARY_CACHE_DIR = Path("./cache/executive_summary")
METADATA_FILE = Path("./cache/metadata.json")

# Consolidated cache files (all districts in single files)
CONSOLIDATED_EPP_PIVOT = CACHE_DIR / "all_districts_epp_pivot.csv"
CONSOLIDATED_LINES = CACHE_DIR / "all_districts_lines.csv"
CONSOLIDATED_NSS_PIVOT = CACHE_DIR / "all_districts_nss_pivot.csv"
CONSOLIDATED_CAGR = CACHE_DIR / "all_districts_cagr.csv"
CONSOLIDATED_METADATA = CACHE_DIR / "all_districts_metadata.json"

# Cohort cache files (all cohorts in single files)
COHORT_MEMBERS = CACHE_DIR / "cohort_members.csv"
COHORT_AGGREGATES = CACHE_DIR / "cohort_aggregates.csv"


def ensure_cache_dirs():
    """Create cache directory structure if it doesn't exist."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    DISTRICT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    COHORT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    EXECUTIVE_SUMMARY_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def district_slug(district_name: str) -> str:
    """
    Convert district name to filename-safe slug.

    Args:
        district_name: District name (e.g., "Amherst-Pelham")

    Returns:
        Filename-safe slug (e.g., "amherst_pelham")
    """
    from school_shared import make_safe_filename
    # Lowercase and use existing safe filename function
    return make_safe_filename(district_name.lower())


def get_district_cache_paths(district_name: str) -> Dict[str, Path]:
    """
    Get cache file paths for a district.

    Args:
        district_name: District name

    Returns:
        Dict with keys: epp_pivot, lines, nss_pivot, cagr, metadata
    """
    slug = district_slug(district_name)
    return {
        "epp_pivot": DISTRICT_CACHE_DIR / f"{slug}_epp_pivot.csv",
        "lines": DISTRICT_CACHE_DIR / f"{slug}_lines.csv",
        "nss_pivot": DISTRICT_CACHE_DIR / f"{slug}_nss_pivot.csv",
        "cagr": DISTRICT_CACHE_DIR / f"{slug}_cagr.csv",
        "metadata": DISTRICT_CACHE_DIR / f"{slug}_metadata.json"
    }


def district_cache_exists(district_name: str) -> bool:
    """
    Check if cache exists for a district in consolidated files.

    Args:
        district_name: District name

    Returns:
        True if district data exists in consolidated files, False otherwise
    """
    # Check if consolidated metadata exists and contains this district
    if CONSOLIDATED_METADATA.exists():
        with open(CONSOLIDATED_METADATA, 'r') as f:
            all_metadata = json.load(f)
        return district_name in all_metadata
    return False


def save_district_epp_pivot(district_name: str, epp_pivot: pd.DataFrame) -> None:
    """
    Save district's expenditure per pupil pivot table to CSV.

    Args:
        district_name: District name
        epp_pivot: DataFrame with years as index, categories as columns
    """
    ensure_cache_dirs()
    path = get_district_cache_paths(district_name)["epp_pivot"]
    epp_pivot.to_csv(path, index=True)
    print(f"  Saved: {path.name}")


def load_district_epp_pivot(district_name: str) -> Optional[pd.DataFrame]:
    """
    Load district's expenditure per pupil pivot table from CSV.

    Args:
        district_name: District name

    Returns:
        DataFrame with years as index, categories as columns, or None if not cached
    """
    path = get_district_cache_paths(district_name)["epp_pivot"]
    if not path.exists():
        return None

    df = pd.read_csv(path, index_col=0)
    # Convert index to int (years)
    df.index = df.index.astype(int)
    return df


def save_district_lines(district_name: str, lines: Dict[str, pd.Series]) -> None:
    """
    Save district's FTE time series to CSV.

    Args:
        district_name: District name
        lines: Dict mapping metric names to time series
               e.g., {"Total FTE": Series, "In-District FTE": Series, ...}
    """
    ensure_cache_dirs()
    path = get_district_cache_paths(district_name)["lines"]

    # Convert dict of Series to DataFrame
    df = pd.DataFrame(lines)
    df.to_csv(path, index=True, index_label="YEAR")
    print(f"  Saved: {path.name}")


def load_district_lines(district_name: str) -> Optional[Dict[str, pd.Series]]:
    """
    Load district's FTE time series from CSV.

    Args:
        district_name: District name

    Returns:
        Dict mapping metric names to time series, or None if not cached
    """
    path = get_district_cache_paths(district_name)["lines"]
    if not path.exists():
        return None

    df = pd.read_csv(path, index_col=0)
    # Convert index to int (years)
    df.index = df.index.astype(int)

    # Convert DataFrame back to dict of Series
    lines = {col: df[col] for col in df.columns}
    return lines


def save_district_nss_pivot(district_name: str, nss_pivot: pd.DataFrame) -> None:
    """
    Save district's NSS/Ch70 funding pivot table to CSV.

    Args:
        district_name: District name
        nss_pivot: DataFrame with years as index, NSS categories as columns
    """
    ensure_cache_dirs()
    path = get_district_cache_paths(district_name)["nss_pivot"]
    nss_pivot.to_csv(path, index=True)
    print(f"  Saved: {path.name}")


def load_district_nss_pivot(district_name: str) -> Optional[pd.DataFrame]:
    """
    Load district's NSS/Ch70 funding pivot table from CSV.

    Args:
        district_name: District name

    Returns:
        DataFrame with years as index, NSS categories as columns, or None if not cached
    """
    path = get_district_cache_paths(district_name)["nss_pivot"]
    if not path.exists():
        return None

    df = pd.read_csv(path, index_col=0)
    # Convert index to int (years)
    df.index = df.index.astype(int)
    return df


def save_district_metadata(district_name: str, metadata: dict) -> None:
    """
    Save district metadata to JSON.

    Args:
        district_name: District name
        metadata: Dict with cache metadata (timestamp, year range, etc.)
    """
    ensure_cache_dirs()
    path = get_district_cache_paths(district_name)["metadata"]
    with open(path, 'w') as f:
        json.dump(metadata, f, indent=2)


def load_district_metadata(district_name: str) -> Optional[dict]:
    """
    Load district metadata from JSON.

    Args:
        district_name: District name

    Returns:
        Dict with metadata, or None if not cached
    """
    path = get_district_cache_paths(district_name)["metadata"]
    if not path.exists():
        return None

    with open(path, 'r') as f:
        return json.load(f)


def save_district_cagr(district_name: str, cagr_data: pd.DataFrame) -> None:
    """
    Save district CAGR calculations to CSV.

    Args:
        district_name: District name
        cagr_data: DataFrame with columns:
                   [Metric_Type, Category, Latest_Year, Latest_Value,
                    Start_Year, Start_Value, CAGR_5yr, CAGR_10yr, CAGR_15yr]
    """
    ensure_cache_dirs()
    path = get_district_cache_paths(district_name)["cagr"]
    cagr_data.to_csv(path, index=False)
    print(f"  Saved: {path.name}")


def load_district_cagr(district_name: str) -> Optional[pd.DataFrame]:
    """
    Load district CAGR calculations from CSV.

    Args:
        district_name: District name

    Returns:
        DataFrame with CAGR data, or None if not cached
    """
    path = get_district_cache_paths(district_name)["cagr"]
    if not path.exists():
        return None

    return pd.read_csv(path)


def calculate_district_cagr(
    epp_pivot: pd.DataFrame,
    lines: Dict[str, pd.Series],
    nss_pivot: Optional[pd.DataFrame],
    foundation_series: Optional[pd.Series]
) -> pd.DataFrame:
    """
    Calculate CAGR for all metrics.

    Args:
        epp_pivot: Expenditure per pupil pivot table
        lines: FTE time series dict
        nss_pivot: NSS/Ch70 pivot table (optional)
        foundation_series: Foundation enrollment series (optional)

    Returns:
        DataFrame with columns: [Metric_Type, Category, Latest_Year, Latest_Value,
                                 Year_5yr, Value_5yr, Year_10yr, Value_10yr,
                                 Start_Year, Start_Value, CAGR_5yr, CAGR_10yr, CAGR_15yr]
    """
    from compose_pdf import compute_cagr_last
    import numpy as np

    rows = []

    if not epp_pivot.empty:
        latest_year = int(epp_pivot.index.max())
        year_5yr = latest_year - 5
        year_10yr = latest_year - 10
        start_year = latest_year - 15

        # Calculate CAGR for each expenditure category
        for category in epp_pivot.columns:
            series = epp_pivot[category]
            latest_val = float(series.loc[latest_year]) if latest_year in series.index else np.nan
            val_5yr = float(series.loc[year_5yr]) if year_5yr in series.index else np.nan
            val_10yr = float(series.loc[year_10yr]) if year_10yr in series.index else np.nan
            start_val = float(series.loc[start_year]) if start_year in series.index else np.nan

            rows.append({
                "Metric_Type": "Expenditure",
                "Category": category,
                "Latest_Year": latest_year,
                "Latest_Value": latest_val,
                "Year_5yr": year_5yr,
                "Value_5yr": val_5yr,
                "Year_10yr": year_10yr,
                "Value_10yr": val_10yr,
                "Start_Year": start_year,
                "Start_Value": start_val,
                "CAGR_5yr": compute_cagr_last(series, 5),
                "CAGR_10yr": compute_cagr_last(series, 10),
                "CAGR_15yr": compute_cagr_last(series, 15)
            })

        # Calculate CAGR for Total PPE
        total_series = epp_pivot.sum(axis=1)
        latest_total = float(total_series.loc[latest_year]) if latest_year in total_series.index else np.nan
        total_5yr = float(total_series.loc[year_5yr]) if year_5yr in total_series.index else np.nan
        total_10yr = float(total_series.loc[year_10yr]) if year_10yr in total_series.index else np.nan
        start_total = float(total_series.loc[start_year]) if start_year in total_series.index else np.nan

        rows.append({
            "Metric_Type": "Expenditure",
            "Category": "Total PPE",
            "Latest_Year": latest_year,
            "Latest_Value": latest_total,
            "Year_5yr": year_5yr,
            "Value_5yr": total_5yr,
            "Year_10yr": year_10yr,
            "Value_10yr": total_10yr,
            "Start_Year": start_year,
            "Start_Value": start_total,
            "CAGR_5yr": compute_cagr_last(total_series, 5),
            "CAGR_10yr": compute_cagr_last(total_series, 10),
            "CAGR_15yr": compute_cagr_last(total_series, 15)
        })

        # Calculate CAGR for FTE metrics
        for metric_name, series in lines.items():
            if series is not None and not series.empty:
                fte_latest_year = int(series.index.max())
                fte_year_5yr = fte_latest_year - 5
                fte_year_10yr = fte_latest_year - 10
                fte_start_year = fte_latest_year - 15
                latest_val = float(series.loc[fte_latest_year]) if fte_latest_year in series.index else np.nan
                val_5yr = float(series.loc[fte_year_5yr]) if fte_year_5yr in series.index else np.nan
                val_10yr = float(series.loc[fte_year_10yr]) if fte_year_10yr in series.index else np.nan
                start_val = float(series.loc[fte_start_year]) if fte_start_year in series.index else np.nan

                rows.append({
                    "Metric_Type": "Enrollment",
                    "Category": metric_name,
                    "Latest_Year": fte_latest_year,
                    "Latest_Value": latest_val,
                    "Year_5yr": fte_year_5yr,
                    "Value_5yr": val_5yr,
                    "Year_10yr": fte_year_10yr,
                    "Value_10yr": val_10yr,
                    "Start_Year": fte_start_year,
                    "Start_Value": start_val,
                    "CAGR_5yr": compute_cagr_last(series, 5),
                    "CAGR_10yr": compute_cagr_last(series, 10),
                    "CAGR_15yr": compute_cagr_last(series, 15)
                })

    # Calculate CAGR for NSS/Ch70 components
    if nss_pivot is not None and not nss_pivot.empty:
        nss_latest_year = int(nss_pivot.index.max())
        nss_year_5yr = nss_latest_year - 5
        nss_year_10yr = nss_latest_year - 10
        nss_start_year = nss_latest_year - 15

        for component in nss_pivot.columns:
            series = nss_pivot[component]
            latest_val = float(series.loc[nss_latest_year]) if nss_latest_year in series.index else np.nan
            val_5yr = float(series.loc[nss_year_5yr]) if nss_year_5yr in series.index else np.nan
            val_10yr = float(series.loc[nss_year_10yr]) if nss_year_10yr in series.index else np.nan
            start_val = float(series.loc[nss_start_year]) if nss_start_year in series.index else np.nan

            rows.append({
                "Metric_Type": "NSS",
                "Category": component,
                "Latest_Year": nss_latest_year,
                "Latest_Value": latest_val,
                "Year_5yr": nss_year_5yr,
                "Value_5yr": val_5yr,
                "Year_10yr": nss_year_10yr,
                "Value_10yr": val_10yr,
                "Start_Year": nss_start_year,
                "Start_Value": start_val,
                "CAGR_5yr": compute_cagr_last(series, 5),
                "CAGR_10yr": compute_cagr_last(series, 10),
                "CAGR_15yr": compute_cagr_last(series, 15)
            })

        # Calculate Total Actual NSS per pupil
        total_nss_series = nss_pivot.sum(axis=1)
        latest_nss_total = float(total_nss_series.loc[nss_latest_year]) if nss_latest_year in total_nss_series.index else np.nan
        total_nss_5yr = float(total_nss_series.loc[nss_year_5yr]) if nss_year_5yr in total_nss_series.index else np.nan
        total_nss_10yr = float(total_nss_series.loc[nss_year_10yr]) if nss_year_10yr in total_nss_series.index else np.nan
        start_nss_total = float(total_nss_series.loc[nss_start_year]) if nss_start_year in total_nss_series.index else np.nan

        rows.append({
            "Metric_Type": "NSS",
            "Category": "Total Actual NSS per pupil",
            "Latest_Year": nss_latest_year,
            "Latest_Value": latest_nss_total,
            "Year_5yr": nss_year_5yr,
            "Value_5yr": total_nss_5yr,
            "Year_10yr": nss_year_10yr,
            "Value_10yr": total_nss_10yr,
            "Start_Year": nss_start_year,
            "Start_Value": start_nss_total,
            "CAGR_5yr": compute_cagr_last(total_nss_series, 5),
            "CAGR_10yr": compute_cagr_last(total_nss_series, 10),
            "CAGR_15yr": compute_cagr_last(total_nss_series, 15)
        })

    # Calculate CAGR for Foundation Enrollment
    if foundation_series is not None and not foundation_series.empty:
        found_latest_year = int(foundation_series.index.max())
        found_year_5yr = found_latest_year - 5
        found_year_10yr = found_latest_year - 10
        found_start_year = found_latest_year - 15
        latest_val = float(foundation_series.loc[found_latest_year]) if found_latest_year in foundation_series.index else np.nan
        val_5yr = float(foundation_series.loc[found_year_5yr]) if found_year_5yr in foundation_series.index else np.nan
        val_10yr = float(foundation_series.loc[found_year_10yr]) if found_year_10yr in foundation_series.index else np.nan
        start_val = float(foundation_series.loc[found_start_year]) if found_start_year in foundation_series.index else np.nan

        rows.append({
            "Metric_Type": "Enrollment",
            "Category": "Foundation Enrollment",
            "Latest_Year": found_latest_year,
            "Latest_Value": latest_val,
            "Year_5yr": found_year_5yr,
            "Value_5yr": val_5yr,
            "Year_10yr": found_year_10yr,
            "Value_10yr": val_10yr,
            "Start_Year": found_start_year,
            "Start_Value": start_val,
            "CAGR_5yr": compute_cagr_last(foundation_series, 5),
            "CAGR_10yr": compute_cagr_last(foundation_series, 10),
            "CAGR_15yr": compute_cagr_last(foundation_series, 15)
        })

    return pd.DataFrame(rows)


def save_district_data(
    district_name: str,
    epp_pivot: pd.DataFrame,
    lines: Dict[str, pd.Series],
    nss_pivot: Optional[pd.DataFrame] = None,
    foundation_series: Optional[pd.Series] = None
) -> None:
    """
    Save all district data to cache.

    Args:
        district_name: District name
        epp_pivot: Expenditure per pupil pivot table
        lines: FTE time series dict
        nss_pivot: NSS/Ch70 funding pivot table (optional)
        foundation_series: Foundation enrollment series (optional)
    """
    ensure_cache_dirs()

    # Save data files
    save_district_epp_pivot(district_name, epp_pivot)
    save_district_lines(district_name, lines)
    if nss_pivot is not None and not nss_pivot.empty:
        save_district_nss_pivot(district_name, nss_pivot)

    # Calculate and save CAGR
    cagr_data = calculate_district_cagr(epp_pivot, lines, nss_pivot, foundation_series)
    save_district_cagr(district_name, cagr_data)

    # Save metadata
    metadata = {
        "district_name": district_name,
        "cached_at": datetime.now().isoformat(),
        "year_range": [int(epp_pivot.index.min()), int(epp_pivot.index.max())],
        "has_nss_data": nss_pivot is not None and not nss_pivot.empty,
        "has_cagr_data": True,
        "categories": list(epp_pivot.columns),
        "fte_metrics": list(lines.keys())
    }
    save_district_metadata(district_name, metadata)


def save_district_data_consolidated(
    district_name: str,
    epp_pivot: pd.DataFrame,
    lines: Dict[str, pd.Series],
    nss_pivot: Optional[pd.DataFrame] = None,
    foundation_series: Optional[pd.Series] = None
) -> None:
    """
    Save district data to consolidated cache files (all districts in same files).

    Args:
        district_name: District name
        epp_pivot: Expenditure per pupil pivot table
        lines: FTE time series dict
        nss_pivot: NSS/Ch70 funding pivot table (optional)
        foundation_series: Foundation enrollment series (optional)
    """
    import numpy as np

    ensure_cache_dirs()

    # 1. Save EPP Pivot (melt to long format with District column)
    if not epp_pivot.empty:
        epp_long = epp_pivot.reset_index().melt(
            id_vars=['YEAR'],
            var_name='Category',
            value_name='PPE'
        )
        epp_long['District'] = district_name
        epp_long = epp_long[['District', 'YEAR', 'Category', 'PPE']]

        # Append or create
        if CONSOLIDATED_EPP_PIVOT.exists():
            existing = pd.read_csv(CONSOLIDATED_EPP_PIVOT)
            # Remove existing data for this district
            existing = existing[existing['District'] != district_name]
            combined = pd.concat([existing, epp_long], ignore_index=True)
            combined.to_csv(CONSOLIDATED_EPP_PIVOT, index=False)
        else:
            epp_long.to_csv(CONSOLIDATED_EPP_PIVOT, index=False)

    # 2. Save Lines (FTE metrics)
    lines_records = []
    for metric_name, series in lines.items():
        if series is not None and not series.empty:
            for year, value in series.items():
                lines_records.append({
                    'District': district_name,
                    'YEAR': int(year),
                    'Metric': metric_name,
                    'Value': float(value)
                })

    if lines_records:
        lines_df = pd.DataFrame(lines_records)
        if CONSOLIDATED_LINES.exists():
            existing = pd.read_csv(CONSOLIDATED_LINES)
            existing = existing[existing['District'] != district_name]
            combined = pd.concat([existing, lines_df], ignore_index=True)
            combined.to_csv(CONSOLIDATED_LINES, index=False)
        else:
            lines_df.to_csv(CONSOLIDATED_LINES, index=False)

    # 3. Save NSS Pivot
    if nss_pivot is not None and not nss_pivot.empty:
        nss_long = nss_pivot.reset_index().melt(
            id_vars=['YEAR'],
            var_name='Component',
            value_name='Value'
        )
        nss_long['District'] = district_name
        nss_long = nss_long[['District', 'YEAR', 'Component', 'Value']]

        if CONSOLIDATED_NSS_PIVOT.exists():
            existing = pd.read_csv(CONSOLIDATED_NSS_PIVOT)
            existing = existing[existing['District'] != district_name]
            combined = pd.concat([existing, nss_long], ignore_index=True)
            combined.to_csv(CONSOLIDATED_NSS_PIVOT, index=False)
        else:
            nss_long.to_csv(CONSOLIDATED_NSS_PIVOT, index=False)

    # 4. Calculate and save CAGR
    cagr_data = calculate_district_cagr(epp_pivot, lines, nss_pivot, foundation_series)
    if not cagr_data.empty:
        cagr_data['District'] = district_name
        # Reorder columns to put District first
        cols = ['District'] + [col for col in cagr_data.columns if col != 'District']
        cagr_data = cagr_data[cols]

        if CONSOLIDATED_CAGR.exists():
            existing = pd.read_csv(CONSOLIDATED_CAGR)
            existing = existing[existing['District'] != district_name]
            combined = pd.concat([existing, cagr_data], ignore_index=True)
            combined.to_csv(CONSOLIDATED_CAGR, index=False)
        else:
            cagr_data.to_csv(CONSOLIDATED_CAGR, index=False)

    # 5. Save metadata
    metadata_entry = {
        "district_name": district_name,
        "cached_at": datetime.now().isoformat(),
        "year_range": [int(epp_pivot.index.min()), int(epp_pivot.index.max())],
        "has_nss_data": nss_pivot is not None and not nss_pivot.empty,
        "has_cagr_data": True,
        "categories": list(epp_pivot.columns),
        "fte_metrics": list(lines.keys())
    }

    # Load or create metadata dict
    if CONSOLIDATED_METADATA.exists():
        with open(CONSOLIDATED_METADATA, 'r') as f:
            all_metadata = json.load(f)
    else:
        all_metadata = {}

    all_metadata[district_name] = metadata_entry

    with open(CONSOLIDATED_METADATA, 'w') as f:
        json.dump(all_metadata, f, indent=2)


def load_district_data(district_name: str) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, pd.Series]], Optional[pd.DataFrame]]:
    """
    Load all district data from cache.

    Args:
        district_name: District name

    Returns:
        Tuple of (epp_pivot, lines, nss_pivot)
        Any component may be None if not cached
    """
    epp_pivot = load_district_epp_pivot(district_name)
    lines = load_district_lines(district_name)
    nss_pivot = load_district_nss_pivot(district_name)

    return epp_pivot, lines, nss_pivot


def get_all_districts(df: pd.DataFrame) -> list:
    """
    Get list of all unique districts in the data.

    Args:
        df: Main expenditure dataframe

    Returns:
        List of district names
    """
    return sorted(df['DIST_NAME'].unique())


def ensure_all_districts_cached(
    df: pd.DataFrame,
    reg: pd.DataFrame,
    c70: pd.DataFrame,
    force: bool = False
) -> None:
    """
    Generate cache for all Massachusetts districts (BATCHED VERSION).

    Collects all district data in memory, then writes consolidated files once at the end.
    Much faster than writing 421 times.

    Args:
        df: Main expenditure data
        reg: Regional district data
        c70: Chapter 70 funding data
        force: If True, regenerate cache even if it exists
    """
    from school_shared import prepare_district_epp_lines, prepare_district_nss_ch70
    import numpy as np

    ensure_cache_dirs()

    all_districts = get_all_districts(df)
    total = len(all_districts)

    print(f"\n{'='*70}")
    print(f"DISTRICT CACHE GENERATION - CR CacheMonster01 (Consolidated Batch)")
    print(f"{'='*70}")
    print(f"Total districts: {total}")
    print(f"Cache directory: {CACHE_DIR}")
    print(f"Output files: all_districts_*.csv (consolidated)")
    print(f"Force regenerate: {force}")
    print(f"{'='*70}\n")

    # Batch collectors
    all_epp_records = []
    all_lines_records = []
    all_nss_records = []
    all_cagr_records = []
    all_metadata = {}

    cached_count = 0
    skipped_count = 0
    error_count = 0

    for idx, district in enumerate(all_districts, 1):
        print(f"[{idx}/{total}] Processing: {district}")

        # Skip if cache exists and force=False
        if not force and district_cache_exists(district):
            print(f"  [SKIP] Cache exists")
            skipped_count += 1
            continue

        try:
            # Generate epp_pivot and lines
            epp_pivot, lines = prepare_district_epp_lines(df, district)

            if epp_pivot.empty:
                print(f"  [WARN] No expenditure data for {district}")
                error_count += 1
                continue

            # Generate nss_pivot (may be None for some districts)
            nss_pivot = None
            foundation_series = None
            try:
                nss_pivot, _, foundation_series = prepare_district_nss_ch70(df, c70, district)
            except Exception as e:
                pass  # Silently skip NSS for districts without data

            # Collect EPP data
            epp_long = epp_pivot.reset_index().melt(
                id_vars=['YEAR'],
                var_name='Category',
                value_name='PPE'
            )
            epp_long['District'] = district
            all_epp_records.append(epp_long[['District', 'YEAR', 'Category', 'PPE']])

            # Collect Lines data
            for metric_name, series in lines.items():
                if series is not None and not series.empty:
                    for year, value in series.items():
                        all_lines_records.append({
                            'District': district,
                            'YEAR': int(year),
                            'Metric': metric_name,
                            'Value': float(value)
                        })

            # Collect NSS data
            if nss_pivot is not None and not nss_pivot.empty:
                nss_long = nss_pivot.reset_index().melt(
                    id_vars=['YEAR'],
                    var_name='Component',
                    value_name='Value'
                )
                nss_long['District'] = district
                all_nss_records.append(nss_long[['District', 'YEAR', 'Component', 'Value']])

            # Collect CAGR data
            cagr_data = calculate_district_cagr(epp_pivot, lines, nss_pivot, foundation_series)
            if not cagr_data.empty:
                cagr_data['District'] = district
                cols = ['District'] + [col for col in cagr_data.columns if col != 'District']
                all_cagr_records.append(cagr_data[cols])

            # Collect metadata
            all_metadata[district] = {
                "district_name": district,
                "cached_at": datetime.now().isoformat(),
                "year_range": [int(epp_pivot.index.min()), int(epp_pivot.index.max())],
                "has_nss_data": nss_pivot is not None and not nss_pivot.empty,
                "has_cagr_data": True,
                "categories": list(epp_pivot.columns),
                "fte_metrics": list(lines.keys())
            }

            cached_count += 1

        except Exception as e:
            print(f"  [ERROR] Failed to cache {district}: {e}")
            error_count += 1
            continue

    # Write all consolidated files at once
    print(f"\n{'='*70}")
    print("Writing consolidated cache files...")
    print(f"{'='*70}\n")

    if all_epp_records:
        epp_df = pd.concat(all_epp_records, ignore_index=True)
        epp_df.to_csv(CONSOLIDATED_EPP_PIVOT, index=False)
        print(f"  Wrote: {CONSOLIDATED_EPP_PIVOT.name} ({len(epp_df)} rows)")

    if all_lines_records:
        lines_df = pd.DataFrame(all_lines_records)
        lines_df.to_csv(CONSOLIDATED_LINES, index=False)
        print(f"  Wrote: {CONSOLIDATED_LINES.name} ({len(lines_df)} rows)")

    if all_nss_records:
        nss_df = pd.concat(all_nss_records, ignore_index=True)
        nss_df.to_csv(CONSOLIDATED_NSS_PIVOT, index=False)
        print(f"  Wrote: {CONSOLIDATED_NSS_PIVOT.name} ({len(nss_df)} rows)")

    if all_cagr_records:
        cagr_df = pd.concat(all_cagr_records, ignore_index=True)
        cagr_df.to_csv(CONSOLIDATED_CAGR, index=False)
        print(f"  Wrote: {CONSOLIDATED_CAGR.name} ({len(cagr_df)} rows)")

    if all_metadata:
        with open(CONSOLIDATED_METADATA, 'w') as f:
            json.dump(all_metadata, f, indent=2)
        print(f"  Wrote: {CONSOLIDATED_METADATA.name} ({len(all_metadata)} districts)")

    # Summary
    print(f"\n{'='*70}")
    print(f"CACHE GENERATION COMPLETE")
    print(f"{'='*70}")
    print(f"Total districts: {total}")
    print(f"Newly cached: {cached_count}")
    print(f"Skipped (already cached): {skipped_count}")
    print(f"Errors: {error_count}")
    print(f"{'='*70}\n")

    # Save global metadata
    global_metadata = {
        "generated_at": datetime.now().isoformat(),
        "total_districts": total,
        "cached_districts": cached_count + skipped_count,
        "source_data_hash": "TODO",  # Could add MD5 of source files
        "cache_version": "1.0"
    }

    with open(METADATA_FILE, 'w') as f:
        json.dump(global_metadata, f, indent=2)


# =============================================================================
# COHORT CACHE FUNCTIONS
# =============================================================================

def save_cohort_members(cohort_members_df: pd.DataFrame) -> None:
    """
    Save cohort member data to consolidated CSV.

    Args:
        cohort_members_df: DataFrame with columns:
            - Year: Fiscal year (int)
            - Cohort: Cohort name (Tiny, Small, Medium, Large, Springfield)
            - District: District name
            - FTE_In_District: In-district FTE
            - FTE_Out_District: Out-of-district FTE
            - FTE_Foundation: Foundation enrollment
            - Total_PPE: Total per-pupil expenditure
            - Teachers_PPE: Teachers PPE
            - ... (all other PPE categories)
            - Ch70_Aid: Chapter 70 aid per pupil
            - Req_NSS: Required NSS (minus Ch70) per pupil
            - Actual_NSS: Actual NSS (minus Req NSS) per pupil
            - Total_Actual_NSS: Total Actual NSS per pupil
    """
    ensure_cache_dirs()
    cohort_members_df.to_csv(COHORT_MEMBERS, index=False)
    print(f"  Saved cohort members: {COHORT_MEMBERS} ({len(cohort_members_df)} rows)")


def save_cohort_aggregates(cohort_aggregates_df: pd.DataFrame) -> None:
    """
    Save cohort aggregate data to consolidated CSV.

    Args:
        cohort_aggregates_df: DataFrame with columns:
            - Year: Fiscal year (int)
            - Cohort: Cohort name (Tiny, Small, Medium, Large, Springfield)
            - District_Count: Number of districts in cohort
            - Mean_FTE_In_District: Mean in-district FTE
            - Mean_FTE_Out_District: Mean out-of-district FTE
            - Mean_FTE_Foundation: Mean foundation enrollment
            - Mean_Total_PPE: Mean total PPE
            - Mean_Teachers_PPE: Mean teachers PPE
            - ... (all other mean PPE categories)
            - Mean_Ch70_Aid: Mean Ch70 aid per pupil
            - Mean_Req_NSS: Mean required NSS per pupil
            - Mean_Actual_NSS: Mean actual NSS per pupil
            - Mean_Total_Actual_NSS: Mean total actual NSS per pupil
            - CAGR_5yr_PPE: 5-year CAGR for total PPE
            - CAGR_10yr_PPE: 10-year CAGR for total PPE
            - CAGR_15yr_PPE: 15-year CAGR for total PPE
            - CAGR_5yr_FTE: 5-year CAGR for foundation enrollment
            - CAGR_10yr_FTE: 10-year CAGR for foundation enrollment
            - CAGR_15yr_FTE: 15-year CAGR for foundation enrollment
    """
    ensure_cache_dirs()
    cohort_aggregates_df.to_csv(COHORT_AGGREGATES, index=False)
    print(f"  Saved cohort aggregates: {COHORT_AGGREGATES} ({len(cohort_aggregates_df)} rows)")


def load_cohort_members() -> Optional[pd.DataFrame]:
    """
    Load cohort member data from consolidated CSV.

    Returns:
        DataFrame with cohort member data, or None if not cached
    """
    if not COHORT_MEMBERS.exists():
        return None

    return pd.read_csv(COHORT_MEMBERS)


def load_cohort_aggregates() -> Optional[pd.DataFrame]:
    """
    Load cohort aggregate data from consolidated CSV.

    Returns:
        DataFrame with cohort aggregate data, or None if not cached
    """
    if not COHORT_AGGREGATES.exists():
        return None

    return pd.read_csv(COHORT_AGGREGATES)
