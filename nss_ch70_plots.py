"""
Plotting Module for Chapter 70 and Net School Spending Analysis

This module generates stacked bar charts showing:
1. Chapter 70 Aid (bottom stack)
2. Required NSS minus Ch70 Aid (middle stack)
3. Actual NSS minus Required NSS (top stack)

All values displayed in absolute dollars (not per-pupil).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from school_shared import (
    OUTPUT_DIR,
    ENROLL_KEYS,
    FTE_LINE_COLORS,
)

# Version stamp
CODE_VERSION = "v2025.10.02-NSS-CH70"

# Color palette for NSS/Ch70 stacks (distinct from expenditure categories)
# Using a green-to-purple gradient to represent funding sources
NSS_CH70_COLORS = {
    "Ch70 Aid": "#4ade80",          # Green - state aid (bottom)
    "Req NSS (adj)": "#fbbf24",     # Amber - required local contribution (middle)
    "Actual NSS (adj)": "#a78bfa",  # Purple - spending above requirement (top)
}

# Stack order (bottom to top)
NSS_CH70_STACK_ORDER = ["Ch70 Aid", "Req NSS (adj)", "Actual NSS (adj)"]


def comma_formatter():
    """Returns formatter for thousands separator."""
    return FuncFormatter(lambda x, _: f"{int(x):,}")


def _stamp(fig, y_pos=0.01):
    """Add version stamp to figure."""
    fig.text(0.99, y_pos, f"Code: {CODE_VERSION}", ha="right", va="bottom",
             fontsize=10.5, color="#666666")


def plot_nss_ch70(
    out_path: Path,
    nss_pivot: pd.DataFrame,
    enrollment: pd.Series,
    title: str,
    right_ylim: float | None = None,
    per_pupil: bool = False,
):
    """
    Plot Chapter 70 and Net School Spending stacked bars.

    Args:
        out_path: Output PNG file path
        nss_pivot: DataFrame with columns ['Ch70 Aid', 'Req NSS (adj)', 'Actual NSS (adj)']
        enrollment: Series with in-district FTE enrollment by year (not plotted, kept for compatibility)
        title: Plot title
        right_ylim: Optional y-axis limit for dollars
        per_pupil: If True, label y-axis as "$ per pupil"; if False, label as "Dollars ($)"

    Notes:
        - Stacks are plotted bottom-to-top: Ch70 Aid, Req NSS (adj), Actual NSS (adj)
        - For individual districts: absolute dollars; for aggregates: weighted per-pupil
        - Enrollment parameter retained for compatibility but not displayed
    """
    if nss_pivot is None or nss_pivot.empty:
        print(f"[SKIP] No NSS/Ch70 data for {out_path}")
        return

    # Filter to 2009-2025 for plots (tables will use 2010-2025)
    all_years = nss_pivot.index.tolist()
    years = [yr for yr in all_years if 2009 <= yr <= 2025]
    if not years:
        print(f"[SKIP] No data in 2009-2025 range for {out_path}")
        return
    nss_pivot = nss_pivot.loc[years]

    # Create figure with twin axes
    fig, axL = plt.subplots(figsize=(11.8, 7.4))
    axR = axL.twinx()

    # Set z-order so enrollment line appears above bars
    axL.set_zorder(3)
    axR.set_zorder(2)
    axL.patch.set_alpha(0.0)

    # Plot stacked bars (bottom to top)
    bottom = np.zeros(len(years))
    for stack_name in NSS_CH70_STACK_ORDER:
        if stack_name not in nss_pivot.columns:
            continue
        vals = nss_pivot[stack_name].reindex(years).fillna(0.0).values
        col = NSS_CH70_COLORS[stack_name]
        axR.bar(
            years, vals, bottom=bottom, color=col, width=0.8,
            edgecolor="white", linewidth=0.5, zorder=1, label=stack_name
        )
        bottom = bottom + vals

    # Labels and formatting
    axL.set_xlabel("School Year", fontsize=16)
    ylabel = "Weighted Avg $ per District" if per_pupil else "Dollars ($)"
    axR.set_ylabel(ylabel, fontsize=16)
    axR.yaxis.set_major_formatter(comma_formatter())

    # Hide left axis (not used)
    axL.set_yticks([])
    axL.spines['left'].set_visible(False)

    # Set y-limits
    if right_ylim is not None:
        axR.set_ylim(0, right_ylim)

    # Grid and margins
    axL.grid(False)
    axR.grid(False)
    axL.margins(x=0.02)
    axR.margins(x=0.02)

    # Title
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.98)

    # Legend (stacks only, reversed order for top-to-bottom reading)
    handles_r, labels_r = axR.get_legend_handles_labels()
    handles_r = handles_r[::-1]
    labels_r = labels_r[::-1]

    axR.legend(
        handles_r, labels_r,
        loc="upper left", fontsize=12, framealpha=0.95
    )

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _stamp(fig)
    fig.savefig(out_path, dpi=320, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {str(out_path)}")


def compute_nss_ylim(nss_pivots: list[pd.DataFrame], pad: float = 1.05) -> float:
    """
    Compute global y-axis limit for NSS/Ch70 plots.

    NOTE: This function is deprecated for absolute dollar plots due to wide variance
    in district sizes. Use right_ylim=None for auto-scaling instead.

    Args:
        nss_pivots: List of NSS pivot DataFrames
        pad: Padding factor (default 1.05 = 5% above max)

    Returns:
        Y-axis limit rounded to nearest $500
    """
    tops = []
    for piv in nss_pivots:
        if piv is None or piv.empty:
            continue
        # Sum all stacks to get total
        totals = piv.sum(axis=1)
        if not totals.empty:
            tops.append(float(totals.max()))

    if not tops:
        return 30000.0  # Default fallback

    max_val = max(tops) * pad
    # Round up to nearest $500
    return np.ceil(max_val / 500) * 500


# ================ Table Building Functions ================

def compute_cagr_last(series: pd.Series, n_years: int) -> float:
    """
    Compute CAGR over last N years of a pandas Series.

    Args:
        series: Time series with years as index
        n_years: Number of years to look back

    Returns:
        CAGR as a decimal (e.g., 0.05 for 5%)
    """
    if series is None or series.empty or len(series) < 2:
        return float("nan")

    idx_sorted = sorted(series.index)
    latest_yr = idx_sorted[-1]
    start_yr = latest_yr - n_years

    if start_yr not in series.index:
        return float("nan")

    v0 = float(series.loc[start_yr])
    v1 = float(series.loc[latest_yr])

    if v0 <= 0 or v1 <= 0:
        return float("nan")

    return (v1 / v0) ** (1.0 / n_years) - 1.0


def fmt_pct(val: float) -> str:
    """Format percentage for display."""
    if val != val or np.isnan(val):  # Check for NaN
        return "—"
    return f"{val * 100:.2f}%"


def build_nss_category_data(nss_pivot: pd.DataFrame, latest_year: int) -> tuple:
    """
    Build category rows and total for NSS/Ch70 pivot table.

    Args:
        nss_pivot: DataFrame with columns ['Ch70 Aid', 'Req NSS (adj)', 'Actual NSS (adj)']
        latest_year: Latest year in the data

    Returns:
        Tuple of (cat_rows, cat_total, cat_start_map):
        - cat_rows: List of tuples (category, start_str, c15s, c10s, c5s, latest_str, color, latest_val)
        - cat_total: Tuple (label, start_str, c15s, c10s, c5s, latest_str)
        - cat_start_map: Dict mapping category -> start year value (15 years before latest)
    """
    if nss_pivot.empty:
        return [], ("Total", "$0", "—", "—", "—", "$0"), {}

    # Top to bottom for display (reverse of stack order)
    top_bottom = list(reversed(NSS_CH70_STACK_ORDER))
    start_year = latest_year - 15  # 15 years before latest

    cat_rows = []
    cat_start_map = {}

    for sc in top_bottom:
        if sc not in nss_pivot.columns:
            continue

        latest_val = float(nss_pivot.loc[latest_year, sc]) if (latest_year in nss_pivot.index and sc in nss_pivot.columns) else 0.0
        start_val = float(nss_pivot.loc[start_year, sc]) if (start_year in nss_pivot.index and sc in nss_pivot.columns) else 0.0
        cat_start_map[sc] = start_val

        c5 = compute_cagr_last(nss_pivot[sc], 5)
        c10 = compute_cagr_last(nss_pivot[sc], 10)
        c15 = compute_cagr_last(nss_pivot[sc], 15)

        cat_rows.append((
            sc,
            f"${start_val:,.0f}",
            fmt_pct(c15),
            fmt_pct(c10),
            fmt_pct(c5),
            f"${latest_val:,.0f}",
            NSS_CH70_COLORS[sc],
            latest_val
        ))

    # Compute total series (sum of all categories)
    total_series = nss_pivot.sum(axis=1)
    latest_total = float(total_series.loc[latest_year]) if latest_year in total_series.index else 0.0
    start_total = float(total_series.loc[start_year]) if start_year in total_series.index else 0.0

    cat_total = (
        "Total",
        f"${start_total:,.0f}",
        fmt_pct(compute_cagr_last(total_series, 15)),
        fmt_pct(compute_cagr_last(total_series, 10)),
        fmt_pct(compute_cagr_last(total_series, 5)),
        f"${latest_total:,.0f}"
    )

    return cat_rows, cat_total, cat_start_map
