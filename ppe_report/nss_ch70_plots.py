"""
Plotting Module for Chapter 70 and Net School Spending Analysis

This module generates stacked bar charts showing:
1. Chapter 70 Aid (bottom stack)
2. Required NSS minus Ch70 Aid (middle stack)
3. Actual NSS minus Required NSS (top stack)

All values displayed in dollars per pupil.
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
    "Ch70 Aid": "#86efac",                   # Lighter green - state aid (bottom)
    "Req NSS (minus Ch70)": "#fef08a",       # Lighter golden yellow - required local contribution (middle)
    "Actual NSS (minus Req NSS)": "#c4b5fd", # Lighter purple - spending above requirement (top)
}

# Stack order (bottom to top)
NSS_CH70_STACK_ORDER = ["Ch70 Aid", "Req NSS (minus Ch70)", "Actual NSS (minus Req NSS)"]


def comma_formatter():
    """Returns formatter for thousands separator."""
    return FuncFormatter(lambda x, _: f"{int(x):,}")

def smart_dollar_formatter():
    """Returns formatter that uses M for millions, K for thousands."""
    def format_func(x, _):
        if x >= 1_000_000:
            return f"${x/1_000_000:.1f}M".replace('.0M', 'M')
        elif x >= 1000:
            return f"${int(x/1000):,}K"
        else:
            return f"${int(x):,}"
    return FuncFormatter(format_func)


def _stamp(fig, y_pos=0.01):
    """Placeholder for version stamp (removed per user request)."""
    pass  # Version stamp removed


def plot_nss_ch70(
    out_path: Path,
    nss_pivot: pd.DataFrame,
    enrollment: pd.Series,
    title: str,
    right_ylim: float | None = None,
    per_pupil: bool = False,
    foundation_enrollment: pd.Series | None = None,
    left_ylim: float | None = None,
    enrollment_label: str = "Foundation Enrollment (FTE)",
):
    """
    Plot Chapter 70 and Net School Spending stacked bars with optional foundation enrollment line.

    Args:
        out_path: Output PNG file path
        nss_pivot: DataFrame with columns ['Ch70 Aid', 'Req NSS (adj)', 'Actual NSS (adj)']
        enrollment: Series with in-district FTE enrollment by year (not plotted, kept for compatibility)
        title: Plot title
        right_ylim: Optional y-axis limit for dollars
        per_pupil: If True, label y-axis as "Weighted avg $ per pupil"; if False, label as "$ per pupil"
        foundation_enrollment: Optional Series with foundation enrollment by year (plotted as blue line)

    Notes:
        - Stacks are plotted bottom-to-top: Ch70 Aid, Req NSS (adj), Actual NSS (adj)
        - All values are in dollars per pupil (weighted average for aggregates, direct per-pupil for individual districts)
        - Foundation enrollment shown as blue line on left axis when provided
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

    # Labels and formatting (match font sizes from district_expend_pp_stack.py)
    axL.set_xlabel("School Year", fontsize=20)
    ylabel = "Weighted avg $ per pupil" if per_pupil else "$ per pupil"
    axR.set_ylabel(ylabel, fontsize=20)
    axR.yaxis.set_major_formatter(smart_dollar_formatter())
    axL.tick_params(axis='both', labelsize=14)
    axR.tick_params(axis='both', labelsize=14)

    # Set y-limits
    if right_ylim is not None:
        axR.set_ylim(0, right_ylim)

    # Grid and margins - add faint horizontal gridlines for $ axis
    axL.grid(False)
    axR.grid(True, axis='y', alpha=0.22, linewidth=0.5, linestyle='-', color='gray')
    axL.margins(x=0.02)
    axR.margins(x=0.02)

    # Remove top, left, right borders - keep only bottom
    axR.spines['top'].set_visible(False)
    axR.spines['left'].set_visible(False)
    axR.spines['right'].set_visible(False)

    # Plot foundation enrollment line if provided
    if foundation_enrollment is not None and not foundation_enrollment.empty:
        # Show left axis for enrollment (but hide the spine itself)
        axL.spines['left'].set_visible(False)  # Keep hidden per user request
        axL.spines['top'].set_visible(False)  # Remove top border
        axL.set_ylabel(enrollment_label, fontsize=20, labelpad=15)  # Add padding
        # Custom formatter that hides "0" and abbreviates with K/M on enrollment axis
        from matplotlib.ticker import FuncFormatter
        def enrollment_formatter(x, _):
            if x == 0:
                return ""
            elif x >= 1_000_000:
                return f"{x/1_000_000:.1f}M".rstrip('0').rstrip('.')
            elif x >= 1_000:
                return f"{x/1_000:.1f}K".rstrip('0').rstrip('.')
            else:
                return f"{int(x)}"
        axL.yaxis.set_major_formatter(FuncFormatter(enrollment_formatter))
        axL.tick_params(axis='y', labelsize=14, pad=12)  # Add space between labels and donut dots

        # Check if data exceeds upper limit and extend with faded ticks if needed
        # Only add faded overflow range if EARLY years (2009-2011) exceed the cohort bound
        early_years = [yr for yr in years if 2009 <= yr <= 2011]
        early_enrollment_vals = foundation_enrollment.reindex(early_years).dropna()
        max_early_enrollment = float(early_enrollment_vals.max()) if not early_enrollment_vals.empty else 0

        if left_ylim is not None and max_early_enrollment > left_ylim:
            # Early years exceed bound - extend axis with faded overflow ticks
            # Round extended limit to just above early max (100 or 200 higher)
            overflow_amount = max_early_enrollment - left_ylim
            if overflow_amount <= 100:
                extended_ylim = left_ylim + 100
            elif overflow_amount <= 200:
                extended_ylim = left_ylim + 200
            else:
                # Round to nearest 100
                extended_ylim = np.ceil(max_early_enrollment / 100) * 100

            axL.set_ylim(0, extended_ylim)

            # Create custom tick locator with faded overflow ticks
            tick_spacing = 100 if left_ylim < 1000 else 200 if left_ylim <= 2000 else 500 if left_ylim < 5000 else 1000
            ticks = list(range(0, int(left_ylim) + 1, tick_spacing))

            # Add overflow ticks (faded)
            overflow_ticks = list(range(int(left_ylim) + tick_spacing, int(extended_ylim) + 1, tick_spacing))
            all_ticks = ticks + overflow_ticks

            axL.set_yticks(all_ticks)

            # Color tick labels: normal for in-bounds, faded for overflow
            tick_labels = axL.get_yticklabels()
            for i, label in enumerate(tick_labels):
                if all_ticks[i] > left_ylim:
                    label.set_alpha(0.4)
        elif left_ylim is not None:
            axL.set_ylim(0, left_ylim)

        # Add donut dots at enrollment tick positions (only for labeled ticks)
        # Turn off default tick lines and add circular markers instead at y-axis
        axL.tick_params(axis='y', length=0)  # Hide default tick lines
        tick_positions = axL.get_yticks()
        tick_labels = [label.get_text() for label in axL.get_yticklabels()]
        # Use axis transform to place markers at y-axis regardless of data
        # Only add donut for ticks with visible labels (not empty or hidden)
        for tick_y, label_text in zip(tick_positions, tick_labels):
            if label_text and label_text.strip():  # Only if label exists and is not empty
                axL.plot(0, tick_y, 'o', color='black', markersize=5,
                        markerfacecolor='white', markeredgecolor='black', markeredgewidth=1.5,
                        transform=axL.get_yaxis_transform(), clip_on=False, zorder=10)

        # Plot foundation enrollment line
        y_vals = foundation_enrollment.reindex(years).values
        axL.plot(years, y_vals, color="#1976D2", lw=3.4, marker="o", ms=8.0,
                 markerfacecolor="white", markeredgecolor="#1976D2", markeredgewidth=2.0,
                 zorder=6, clip_on=False, label="Foundation Enrollment")
    else:
        # Hide left axis if not used
        axL.set_yticks([])
        axL.spines['left'].set_visible(False)
        axL.spines['top'].set_visible(False)  # Remove top border

    # No legend - Component table with color swatches serves as legend

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

    Handles negative values and crossing zero (e.g., Springfield NSS going from -$10M to +$1M).

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

    # Handle zero values
    if v0 == 0 or v1 == 0:
        return float("nan")

    # If values cross zero (e.g., Springfield: -$10M to +$1M),
    # use average annual growth rate instead of geometric CAGR
    if (v0 > 0 and v1 < 0) or (v0 < 0 and v1 > 0):
        # Average annual rate of change relative to starting absolute value
        return (v1 - v0) / (n_years * abs(v0))

    # For negative values with same sign, calculate CAGR on absolute values
    if v0 < 0 and v1 < 0:
        # Both negative: use absolute values for calculation
        abs_cagr = (abs(v1) / abs(v0)) ** (1.0 / n_years) - 1.0
        return abs_cagr

    # Standard case: both values positive
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
        return [], ("Total (Actual NSS)", "$0", "—", "—", "—", "$0"), {}

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
        "Total (Actual NSS)",
        f"${start_total:,.0f}",
        fmt_pct(compute_cagr_last(total_series, 15)),
        fmt_pct(compute_cagr_last(total_series, 10)),
        fmt_pct(compute_cagr_last(total_series, 5)),
        f"${latest_total:,.0f}"
    )

    return cat_rows, cat_total, cat_start_map
