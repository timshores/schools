"""
Data caching system for school district analysis.

This module provides checkpoint/caching functionality to avoid recomputing
expensive operations on each run. Cached data is saved as CSV files and
automatically reloaded unless --force-recompute is specified.

Usage:
    from cache_manager import use_cache, save_cache

    # Check if cache exists and is valid
    if use_cache():
        df, reg, c70 = load_from_cache()
    else:
        df, reg, c70 = load_data_from_excel()
        save_cache(df, reg, c70)
"""

from pathlib import Path
from typing import Tuple
import pandas as pd

# Cache directory (relative to project root)
CACHE_DIR = Path("./data/cache")

# Cache file names
CACHE_FILES = {
    "expenditure": CACHE_DIR / "expenditure_data.csv",
    "regions": CACHE_DIR / "regional_data.csv",
    "chapter70": CACHE_DIR / "chapter70_data.csv",
    "metadata": CACHE_DIR / "cache_metadata.txt"
}


def ensure_cache_dir():
    """Create cache directory if it doesn't exist."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def cache_exists() -> bool:
    """
    Check if all required cache files exist.

    Returns:
        True if cache is present, False otherwise
    """
    return all(path.exists() for path in CACHE_FILES.values())


def save_cache(df: pd.DataFrame, reg: pd.DataFrame, c70: pd.DataFrame) -> None:
    """
    Save DataFrames to cache as CSV files.

    Args:
        df: Main expenditure/enrollment DataFrame
        reg: Regional classifications DataFrame
        c70: Chapter 70 funding DataFrame
    """
    ensure_cache_dir()

    print("\n[Cache] Saving data to cache...")

    # Save DataFrames as CSVs
    df.to_csv(CACHE_FILES["expenditure"], index=False)
    print(f"  Saved: {CACHE_FILES['expenditure']} ({len(df)} rows)")

    reg.to_csv(CACHE_FILES["regions"], index=False)
    print(f"  Saved: {CACHE_FILES['regions']} ({len(reg)} rows)")

    if not c70.empty:
        c70.to_csv(CACHE_FILES["chapter70"], index=False)
        print(f"  Saved: {CACHE_FILES['chapter70']} ({len(c70)} rows)")
    else:
        # Create empty file if c70 is empty
        CACHE_FILES["chapter70"].write_text("", encoding="utf-8")
        print(f"  Saved: {CACHE_FILES['chapter70']} (empty)")

    # Save metadata (source file info, timestamp)
    import time
    metadata = f"Cache created: {time.ctime()}\n"
    metadata += f"Expenditure rows: {len(df)}\n"
    metadata += f"Regional rows: {len(reg)}\n"
    metadata += f"Chapter 70 rows: {len(c70)}\n"

    CACHE_FILES["metadata"].write_text(metadata, encoding="utf-8")
    print(f"  Cache saved successfully")


def load_from_cache() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load DataFrames from cache CSV files.

    Returns:
        Tuple of (df, reg, c70) DataFrames

    Raises:
        FileNotFoundError: If cache files don't exist
    """
    if not cache_exists():
        raise FileNotFoundError("Cache files not found. Use save_cache() first.")

    print("\n[Cache] Loading data from cache...")

    # Load expenditure data
    df = pd.read_csv(CACHE_FILES["expenditure"])

    # Restore YEAR column as int
    if "YEAR" in df.columns:
        df["YEAR"] = df["YEAR"].astype(int)

    print(f"  Loaded: {CACHE_FILES['expenditure']} ({len(df)} rows)")

    # Load regional data
    reg = pd.read_csv(CACHE_FILES["regions"])
    print(f"  Loaded: {CACHE_FILES['regions']} ({len(reg)} rows)")

    # Load chapter 70 data
    c70 = pd.read_csv(CACHE_FILES["chapter70"])

    # Restore YEAR column as Int64 (nullable integer)
    if "YEAR" in c70.columns and not c70.empty:
        c70["YEAR"] = c70["YEAR"].astype("Int64")

    # Handle Org4Code column restoration
    if "Org4Code" in c70.columns and not c70.empty:
        c70["Org4Code"] = pd.to_numeric(c70["Org4Code"], errors="coerce")

    print(f"  Loaded: {CACHE_FILES['chapter70']} ({len(c70)} rows)")
    print(f"  Cache loaded successfully")

    return df, reg, c70


def use_cache(force_recompute: bool = False) -> bool:
    """
    Determine whether to use cached data or recompute from source.

    Args:
        force_recompute: If True, bypass cache and recompute

    Returns:
        True if cache should be used, False if data should be recomputed
    """
    if force_recompute:
        print("\n[Cache] Force recompute requested - bypassing cache")
        return False

    if not cache_exists():
        print("\n[Cache] No cache found - will compute from source")
        return False

    print("\n[Cache] Cache found - loading from cache")
    print("  (Use --force-recompute to bypass cache)")
    return True


def clear_cache() -> None:
    """Delete all cache files."""
    print("\n[Cache] Clearing cache...")

    for name, path in CACHE_FILES.items():
        if path.exists():
            path.unlink()
            print(f"  Deleted: {path}")

    print("  Cache cleared")


def get_cache_info() -> dict:
    """
    Get information about the current cache.

    Returns:
        Dict with cache metadata
    """
    if not cache_exists():
        return {"status": "not_found"}

    metadata_path = CACHE_FILES["metadata"]
    if metadata_path.exists():
        metadata = metadata_path.read_text(encoding="utf-8")
        return {"status": "found", "metadata": metadata}

    return {"status": "found", "metadata": "No metadata available"}
