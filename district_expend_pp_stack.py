"""
Plotting Module for School District Expenditure Analysis

This module generates all matplotlib charts for the PDF report:
1. Western MA Overview: Horizontal bar chart of all districts (plot_all_western_overview)
2. Western MA Enrollment Groups: 4 aggregate plots (Small/Medium/Large/Springfield)
3. District Detail Plots: Both simple (solid) and detailed (stacked) versions (plot_one, plot_one_simple)

Key Design Patterns:
- Horizontal bars (barh) used for multi-district comparisons (better for PDF portrait layout)
- Vertical bars (bar) used for time-series district plots (better for showing trends)
- Dynamic figure heights scale with number of items to display
- Enrollment-based peer groups (4-tier system) for meaningful comparisons
- All charts use consistent color palettes and styling

Plot Types:
- Simple: Solid color total PPE with enrollment lines (for quick overview)
- Detailed: Stacked categories with enrollment lines (for deep analysis)
- Western Overview: Horizontal bars showing 5-year change for all districts
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from school_shared import (
    OUTPUT_DIR, load_data, create_or_load_color_map, color_for,
    context_for_district, context_for_western,
    prepare_district_epp_lines, prepare_western_epp_lines,
    DISTRICTS_OF_INTEREST, ENROLL_KEYS,
    compute_global_dollar_ylim, compute_districts_fte_ylim,
    canonical_order_bottom_to_top,
    EXCLUDE_SUBCATS, aggregate_to_canonical,
    FTE_LINE_COLORS, PPE_PEERS_YMAX, PPE_PEERS_REMOVE_SPINES, PPE_PEERS_BAR_EDGES,
    MICRO_AREA_FILL, MICRO_AREA_EDGE,
    weighted_epp_aggregation,
    get_cohort_ylim,
    get_western_cohort_districts,
)

# ===== version footer for images =====
CODE_VERSION = "v2025.09.29-REFACTORED"

# ===== Plot styling constants =====
# Comparative bars plot settings
SPACING_FACTOR = 0.95  # Horizontal spacing multiplier (not used in barh plots)
BAR_WIDTH = 0.75       # Bar width for vertical bar plots (not used in barh)
DOT_OFFSET_FACTOR = 0.45  # multiplied by BAR_WIDTH for dot position
YEAR_LAG_DEFAULT = 5   # Default years for PPE change comparison (2019→2024)

# Color Palettes:
# NOTE: Blue palette for individual districts/peers, Gray palette for aggregates
# This visual distinction helps readers quickly identify aggregate vs individual data

# Peer/District colors (blue palette)
BLUE_BASE = "#8fbcd4"   # Light blue for t0 (2019) base PPE
BLUE_DELTA = "#1b6ca8"  # Dark blue for positive change (2019→2024 increase)
PURP_DECL = "#955196"   # Purple for negative change (2019→2024 decrease)

# Aggregate colors (gray palette) - visually distinct from peer blue
AGG_BASE = "#b3c3c9"    # Light gray for t0 aggregate base PPE
AGG_DELTA = "#4b5563"   # Dark gray for positive aggregate change
AGG_DECL = "#8e6aa6"    # Purple-gray for negative aggregate change
AGG_AREA_FILL = "#DBEAFE"   # Fill for aggregate enrollment areas (unused in current layout)
AGG_AREA_EDGE = "#1D4ED8"   # Edge for aggregate enrollment areas (unused in current layout)

# Micro-area (enrollment sparkline) settings
DEFAULT_GAP = 2400.0
DEFAULT_AMP = 6000.0
MIN_GAP = 400.0
MIN_AMP = 800.0
AREA_GAP_FACTOR = 0.22
AREA_SAFETY_MARGIN = 300.0

def _stamp(fig, y_pos=0.01):
    """
    Placeholder for version stamp (removed per user request).

    Args:
        fig: matplotlib figure
        y_pos: Y position for stamp (default 0.01). Increase for plots with crowded bottom areas.
    """
    pass  # Version stamp removed

def _boost_plot_fonts():
    plt.rcParams.update({
        "font.size": 15,
        "axes.labelsize": 20,
        "axes.titlesize": 20,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    })

def comma_formatter():
    return FuncFormatter(lambda x, pos: f"{x:,.0f}")

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

# ---------- Helpers ----------
def _total_ppe_series_from_pivot(piv: pd.DataFrame) -> pd.Series:
    return piv.sum(axis=1).sort_index() if (piv is not None and not piv.empty) else pd.Series(dtype=float)

def _western_all_total_series(df: pd.DataFrame, reg: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Get total PPE and enrollment for all Western MA traditional districts."""
    mask = (reg["EOHHS_REGION"].str.lower() == "western") & (reg["SCHOOL_TYPE"].str.lower() == "traditional")
    members = sorted(set(reg[mask]["DIST_NAME"].str.lower()))
    piv, enroll_in, enroll_out = weighted_epp_aggregation(df, list(members))
    return _total_ppe_series_from_pivot(piv), enroll_in

def _weighted_total_series_for_list(df: pd.DataFrame, districts: List[str]) -> tuple[pd.Series, pd.Series]:
    """Get total PPE and enrollment for a specific list of districts."""
    piv, enroll_in, enroll_out = weighted_epp_aggregation(df, districts)
    return _total_ppe_series_from_pivot(piv), enroll_in

# ---------- Simplified district plot (solid color, no categories) ----------
def plot_one_simple(out_path: Path, epp_pivot: pd.DataFrame, lines: Dict[str, pd.Series],
                   context: str, right_ylim: float, left_ylim: float | None,
                   line_colors: Dict[str, str], enrollment_label: str = "Pupils (FTE)"):
    """Plot district with solid color for total PPE, no category breakdown."""

    # Calculate total PPE series
    total_ppe = _total_ppe_series_from_pivot(epp_pivot) if (epp_pivot is not None and not epp_pivot.empty) else pd.Series(dtype=float)
    years = (total_ppe.index.tolist() if not total_ppe.empty
             else sorted(set().union(*(set(s.index) for s in lines.values() if s is not None))))
    # Filter years to 2009 onwards
    years = [yr for yr in years if yr >= 2009]

    fig, axL = plt.subplots(figsize=(11.8, 7.4))
    axR = axL.twinx()

    axL.set_zorder(3); axR.set_zorder(2)
    axL.patch.set_alpha(0.0)

    # Plot solid bar for total PPE (fainter blue to not compete with lines)
    if not total_ppe.empty:
        vals = total_ppe.reindex(years).fillna(0.0).values
        axR.bar(years, vals, color="#BFDBFE", width=0.8,
                edgecolor="white", linewidth=0.5, zorder=1)

    # Plot enrollment lines
    for _key, label in ENROLL_KEYS:
        s = lines.get(label)
        if s is None or s.empty: continue
        y = s.reindex(years).values
        lc = line_colors.get(label, "#666666")  # Default to gray if color not found
        axL.plot(years, y, color=lc, lw=3.4, marker="o", ms=8.0,
                 markerfacecolor="white", markeredgecolor=lc, markeredgewidth=2.0,
                 zorder=6, clip_on=False)

    axL.set_xlabel("School Year")
    axL.set_ylabel(enrollment_label, labelpad=15)  # Add padding to prevent overlap
    # Use weighted avg label if enrollment_label indicates weighted average
    ylabel = "Weighted avg $ per pupil" if "weighted avg" in enrollment_label.lower() else "$ per pupil"
    axR.set_ylabel(ylabel)
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
    axL.tick_params(axis='y', pad=12)  # Add space between labels and ticks/donut dots
    axR.yaxis.set_major_formatter(smart_dollar_formatter())

    if right_ylim is not None: axR.set_ylim(0, right_ylim)

    # Check if enrollment data exceeds upper limit and extend with faded ticks if needed
    # Only add faded overflow range if EARLY years (2009-2011) exceed the cohort bound
    early_years_range = [yr for yr in years if 2009 <= yr <= 2011]
    max_early_enrollment = 0
    for s in lines.values():
        if s is not None and not s.empty:
            early_vals = s.reindex(early_years_range).dropna()
            if not early_vals.empty:
                max_early_enrollment = max(max_early_enrollment, float(early_vals.max()))

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

    # Add donut dots at enrollment tick positions
    # Turn off default tick lines and add circular markers instead at y-axis
    axL.tick_params(axis='y', length=0)  # Hide default tick lines
    tick_positions = axL.get_yticks()
    # Use axis transform to place markers at y-axis regardless of data
    for tick_y in tick_positions:
        axL.plot(0, tick_y, 'o', color='black', markersize=5,
                markerfacecolor='white', markeredgecolor='black', markeredgewidth=1.5,
                transform=axL.get_yaxis_transform(), clip_on=False, zorder=10)

    # Remove top, left, right borders - keep only bottom
    axL.spines['top'].set_visible(False)
    axL.spines['left'].set_visible(False)
    axL.spines['right'].set_visible(False)
    axR.spines['top'].set_visible(False)
    axR.spines['left'].set_visible(False)
    axR.spines['right'].set_visible(False)

    # Add faint horizontal gridlines for $ axis only (not enrollment axis)
    axL.grid(False)
    axR.grid(True, axis='y', alpha=0.22, linewidth=0.5, linestyle='-', color='gray')
    axL.margins(x=0.02); axR.margins(x=0.02)

    # Add legend below plot
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor="#BFDBFE", edgecolor="white", label='Per-Pupil Expenditures')
    ]
    # Add enrollment lines to legend
    for _key, label in ENROLL_KEYS:
        if label in lines and lines[label] is not None and not lines[label].empty:
            lc = line_colors.get(label, "#666666")
            legend_elements.append(Line2D([0], [0], color=lc, lw=3, marker='o',
                                         markersize=7, markerfacecolor='white',
                                         markeredgecolor=lc, markeredgewidth=2, label=label))

    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.08),
               ncol=4, frameon=False, fontsize=12)
    plt.subplots_adjust(bottom=0.15)  # Make room for legend

    out_path.parent.mkdir(parents=True, exist_ok=True)
    _stamp(fig)
    fig.savefig(out_path, dpi=320, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {out_path}")

# ---------- Main district page plot ----------
def plot_one(out_path: Path, epp_pivot: pd.DataFrame, lines: Dict[str, pd.Series],
             context: str, right_ylim: float, left_ylim: float | None,
             line_colors: Dict[str, str], cmap_all: Dict[str, Dict[str, str]], enrollment_label: str = "Pupils (FTE)"):

    cols = list(epp_pivot.columns) if (epp_pivot is not None and not epp_pivot.empty) else []
    sub_order_bottom_top = canonical_order_bottom_to_top(cols)
    years = (epp_pivot.index.tolist() if cols
             else sorted(set().union(*(set(s.index) for s in lines.values() if s is not None))))
    # Filter years to 2009 onwards
    years = [yr for yr in years if yr >= 2009]

    fig, axL = plt.subplots(figsize=(11.8, 7.4))
    axR = axL.twinx()

    axL.set_zorder(3); axR.set_zorder(2)
    axL.patch.set_alpha(0.0)

    if sub_order_bottom_top:
        bottom = np.zeros(len(years))
        for sc in sub_order_bottom_top:
            vals = epp_pivot[sc].reindex(years).fillna(0.0).values
            col = color_for(cmap_all, context, sc)
            axR.bar(years, vals, bottom=bottom, color=col, width=0.8,
                    edgecolor="white", linewidth=0.5, zorder=1)
            bottom = bottom + vals

    for _key, label in ENROLL_KEYS:
        s = lines.get(label)
        if s is None or s.empty: continue
        y = s.reindex(years).values
        lc = line_colors.get(label, "#666666")  # Default to gray if color not found
        axL.plot(years, y, color=lc, lw=3.4, marker="o", ms=8.0,
                 markerfacecolor="white", markeredgecolor=lc, markeredgewidth=2.0,
                 zorder=6, clip_on=False)

    axL.set_xlabel("School Year")
    axL.set_ylabel(enrollment_label, labelpad=15)  # Add padding to prevent overlap
    # Use weighted avg label if enrollment_label indicates weighted average
    ylabel = "Weighted avg $ per pupil" if "weighted avg" in enrollment_label.lower() else "$ per pupil"
    axR.set_ylabel(ylabel)
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
    axL.tick_params(axis='y', pad=12)  # Add space between labels and ticks/donut dots
    axR.yaxis.set_major_formatter(smart_dollar_formatter())

    if right_ylim is not None: axR.set_ylim(0, right_ylim)

    # Check if enrollment data exceeds upper limit and extend with faded ticks if needed
    # Only add faded overflow range if EARLY years (2009-2011) exceed the cohort bound
    early_years_range = [yr for yr in years if 2009 <= yr <= 2011]
    max_early_enrollment = 0
    for s in lines.values():
        if s is not None and not s.empty:
            early_vals = s.reindex(early_years_range).dropna()
            if not early_vals.empty:
                max_early_enrollment = max(max_early_enrollment, float(early_vals.max()))

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

    # Add donut dots at enrollment tick positions
    # Turn off default tick lines and add circular markers instead at y-axis
    axL.tick_params(axis='y', length=0)  # Hide default tick lines
    tick_positions = axL.get_yticks()
    # Use axis transform to place markers at y-axis regardless of data
    for tick_y in tick_positions:
        axL.plot(0, tick_y, 'o', color='black', markersize=5,
                markerfacecolor='white', markeredgecolor='black', markeredgewidth=1.5,
                transform=axL.get_yaxis_transform(), clip_on=False, zorder=10)

    # Remove top, left, right borders - keep only bottom
    axL.spines['top'].set_visible(False)
    axL.spines['left'].set_visible(False)
    axL.spines['right'].set_visible(False)
    axR.spines['top'].set_visible(False)
    axR.spines['left'].set_visible(False)
    axR.spines['right'].set_visible(False)

    # Add faint horizontal gridlines for $ axis only (not enrollment axis)
    axL.grid(False)
    axR.grid(True, axis='y', alpha=0.22, linewidth=0.5, linestyle='-', color='gray')
    axL.margins(x=0.02); axR.margins(x=0.02)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    _stamp(fig)
    fig.savefig(out_path, dpi=320, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {out_path}")

# ===== PPE comparative bars (5-year change) with enrollment mini-areas =====
def plot_all_western_overview(out_path: Path, df: pd.DataFrame, reg: pd.DataFrame, c70: pd.DataFrame, year_lag: int = YEAR_LAG_DEFAULT):
    """Plot all Western MA districts as horizontal bars (vertical layout)."""
    latest = int(df["YEAR"].max())
    t0 = latest - year_lag

    # Get all Western MA traditional districts
    mask = (reg["EOHHS_REGION"].str.lower() == "western") & (reg["SCHOOL_TYPE"].str.lower() == "traditional")
    western_districts = sorted(reg[mask]["DIST_NAME"].unique())

    # Collect district data
    # NOTE: Building list of all Western MA districts with PPE data for t0 and latest year
    dist_labels, dist_p0, dist_p1 = [], [], []

    for dist in western_districts:
        piv, lines = prepare_district_epp_lines(df, dist, c70)
        if piv.empty:
            continue
        total = _total_ppe_series_from_pivot(piv)
        p0 = total.get(t0, np.nan)
        p1 = total.get(latest, np.nan)
        if np.isnan(p0) or np.isnan(p1):
            continue

        dist_labels.append(dist)
        dist_p0.append(float(p0))
        dist_p1.append(float(p1))

    if not dist_labels:
        print("[WARN] Western overview plot: no data"); return

    # Sort by p1 (latest value) - lowest to highest
    # NOTE: This ensures bars are arranged from lowest to highest PPE for easy comparison
    dist_p0 = np.array(dist_p0)
    dist_p1 = np.array(dist_p1)
    order = np.argsort(dist_p1)
    dist_labels = [dist_labels[i] for i in order]
    dist_p0 = dist_p0[order]
    dist_p1 = dist_p1[order]
    delta_all = dist_p1 - dist_p0

    n = len(dist_labels)
    y_pos = np.arange(n)  # Vertical positions for horizontal bars

    # Create taller figure - height based on number of districts
    # NOTE: Dynamic height calculation ensures all districts fit comfortably
    bar_height = 0.95  # Reduced to add spacing between bars (was 1.13)
    base_fig_height = max(12.0, n * bar_height * 0.35 + 3)
    fig_height = base_fig_height * 1.2  # Increased by 20% for vertical height
    fig_width = 11.0 * 1.4  # Increased by 40% (11.0 → 15.4) for breathing room
    fig, ax_main = plt.subplots(1, 1, figsize=(fig_width, fig_height))

    # Font sizes: Increased 20% more for readability
    ax_main.tick_params(axis='y', labelsize=28)  # Y-axis (district names): 23 * 1.2 ≈ 28
    ax_main.tick_params(axis='x', labelsize=28)  # X-axis (PPE values): 23 * 1.2 ≈ 28
    ax_main.xaxis.label.set_size(35)  # X-axis label: 29 * 1.2 ≈ 35

    # Remove top, left, right borders - keep only bottom
    ax_main.spines['top'].set_visible(False)
    ax_main.spines['left'].set_visible(False)
    ax_main.spines['right'].set_visible(False)

    edge = "white" if PPE_PEERS_BAR_EDGES else None

    # Plot horizontal bars (base + delta segments)
    # NOTE: Uses stacked bars to show 2019 base + change to 2024
    pos = np.clip(delta_all, 0, None)
    neg = np.clip(delta_all, None, 0)
    ax_main.barh(y_pos, dist_p0, height=bar_height, color=BLUE_BASE, edgecolor=edge, linewidth=0.0, zorder=1)
    if np.any(pos > 0):
        ax_main.barh(y_pos, pos, left=dist_p0, height=bar_height, color=BLUE_DELTA, edgecolor=edge, linewidth=0.0, zorder=1)
    if np.any(neg < 0):
        ax_main.barh(y_pos, neg, left=dist_p0, height=bar_height, color=PURP_DECL, edgecolor=edge, linewidth=0.0, zorder=1)

    ax_main.set_xlim(0, PPE_PEERS_YMAX)
    # Improved gridlines: increased alpha and added linewidth for better visibility
    ax_main.grid(axis="x", alpha=0.35, linewidth=0.8)
    ax_main.set_axisbelow(True)
    ax_main.set_xlabel("$ per pupil")
    ax_main.xaxis.set_major_formatter(comma_formatter())

    # Y-axis: Show only district names (letter codes removed per user request)
    # NOTE: Highlighting districts of interest with colorblind-friendly orange
    ax_main.set_yticks(y_pos, dist_labels)

    # Highlight districts of interest with colorblind-friendly color
    # Using dark orange (#FF8C00) which is distinct for both normal and colorblind vision
    districts_of_interest = {"Amherst-Pelham", "Amherst", "Leverett", "Pelham", "Shutesbury"}
    for i, label in enumerate(dist_labels):
        if label in districts_of_interest:
            ax_main.get_yticklabels()[i].set_color('#FF8C00')  # Dark orange
            ax_main.get_yticklabels()[i].set_weight('bold')    # Bold for extra emphasis

    # Legend above plot - 3 items in single row
    handles = [
        Patch(facecolor=BLUE_BASE,  edgecolor=edge, label=f"{t0} PPE"),
        Patch(facecolor=BLUE_DELTA, edgecolor=edge, label=f"{latest} increase from {t0}"),
        Patch(facecolor=PURP_DECL,  edgecolor=edge, label=f"{latest} decrease from {t0}"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.99),
               ncol=3, frameon=False, fontsize=28)  # Increased 20% more: 23 * 1.2 ≈ 28
    plt.subplots_adjust(top=0.96, bottom=0.06, left=0.28, right=0.95)  # More left margin for labels

    _stamp(fig)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=320, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {out_path}")

def plot_ppe_change_bars(out_path: Path, df: pd.DataFrame, reg: pd.DataFrame, c70: pd.DataFrame,
                         districts: list[str], year_lag: int = YEAR_LAG_DEFAULT,
                         title: str | None = None):
    """
    Plot ALPS & Peers as horizontal bars with enrollment annotations.
    NOTE: Rotated 90° from original vertical layout to match Western overview format.
    Districts shown on y-axis, aggregates separated at bottom for visual distinction.
    """

    latest = int(df["YEAR"].max())
    t0 = latest - year_lag

    # Separate peer districts from aggregate groups
    peers_input = [d for d in districts if d.lower() not in {"western ma (aggregate)", "pk-12 district aggregate"}]

    # Collect peer district data
    peer_labels, peer_p0, peer_p1, peer_series = [], [], [], []
    def add_peer(name: str):
        piv, lines = prepare_district_epp_lines(df, name, c70)
        if piv.empty: return
        total = _total_ppe_series_from_pivot(piv)
        p0 = total.get(t0, np.nan); p1 = total.get(latest, np.nan)
        s = lines.get("In-District FTE Pupils")
        if np.isnan(p0) or np.isnan(p1): return
        peer_labels.append(name); peer_p0.append(float(p0)); peer_p1.append(float(p1))
        peer_series.append(s.sort_index() if isinstance(s, pd.Series) and not s.empty else None)
    for d in peers_input: add_peer(d)

    # Western MA aggregate
    ppe_w, enr_w = _western_all_total_series(df, reg)
    has_west = (not ppe_w.empty) and (t0 in ppe_w.index) and (latest in ppe_w.index)
    west_p0 = float(ppe_w.loc[t0]) if has_west else np.nan
    west_p1 = float(ppe_w.loc[latest]) if has_west else np.nan

    # PK-12 aggregate
    pk12_list = ["ALPS PK-12","Easthampton","Longmeadow","Hampden-Wilbraham","East Longmeadow","South Hadley","Agawam","Northampton","Greenfield","Hadley"]
    ppe_pk12, enr_pk12 = _weighted_total_series_for_list(df, pk12_list)
    has_pk12 = (not ppe_pk12.empty) and (t0 in ppe_pk12.index) and (latest in ppe_pk12.index)
    pk12_p0 = float(ppe_pk12.loc[t0]) if has_pk12 else np.nan
    pk12_p1 = float(ppe_pk12.loc[latest]) if has_pk12 else np.nan

    if not peer_labels and not (has_west or has_pk12):
        print("[WARN] comparative plot: nothing to draw."); return

    # Sort peers by PPE (lowest to highest)
    if peer_labels:
        peer_p0 = np.array(peer_p0); peer_p1 = np.array(peer_p1)
        order = np.argsort(peer_p1)
        peer_labels = [peer_labels[i] for i in order]
        peer_p0 = peer_p0[order]; peer_p1 = peer_p1[order]
        peer_series = [peer_series[i] for i in order]
    else:
        peer_p0 = np.array([]); peer_p1 = np.array([]); peer_series = []

    # Build combined lists: aggregates at bottom (idx 0-1), then peers
    # NOTE: This keeps aggregates visually separated at bottom of horizontal chart
    labels = []; series_list: List[pd.Series | None] = []; p0_all = []; p1_all = []

    # Add aggregates first (will be at bottom of chart)
    if has_west:
        labels.append("Western MA (aggregate)")
        series_list.append(enr_w.sort_index())
        p0_all.append(west_p0); p1_all.append(west_p1)
    if has_pk12:
        labels.append("PK-12 District Aggregate")
        series_list.append(enr_pk12.sort_index())
        p0_all.append(pk12_p0); p1_all.append(pk12_p1)

    # Then add peers (will be stacked above aggregates)
    labels.extend(peer_labels)
    series_list.extend(peer_series)
    p0_all.extend(peer_p0.tolist())
    p1_all.extend(peer_p1.tolist())

    n_aggs = (1 if has_west else 0) + (1 if has_pk12 else 0)
    n_total = len(labels)
    y_pos = np.arange(n_total)

    p0_all = np.array(p0_all); p1_all = np.array(p1_all)
    delta_all = p1_all - p0_all

    # Dynamic figure height based on number of items
    bar_height = 0.65
    fig_height = max(11.0, n_total * bar_height * 0.45 + 2)
    fig, ax_main = plt.subplots(1, 1, figsize=(11.0, fig_height))

    ax_main.tick_params(axis='y', labelsize=13)  # Y-axis (district names)
    ax_main.tick_params(axis='x', labelsize=16)  # X-axis (PPE values)
    ax_main.xaxis.label.set_size(20)

    if PPE_PEERS_REMOVE_SPINES:
        for s in ("top","right","left","bottom"):
            ax_main.spines[s].set_visible(False)

    edge = "white" if PPE_PEERS_BAR_EDGES else None

    # Plot horizontal bars - aggregates use gray palette, peers use blue palette
    for i in range(n_total):
        p0 = p0_all[i]; p1 = p1_all[i]; d = delta_all[i]

        if i < n_aggs:  # Aggregate districts
            ax_main.barh([y_pos[i]], [p0], height=bar_height, color=AGG_BASE, edgecolor=edge, linewidth=0.0, zorder=1)
            ax_main.barh([y_pos[i]], [d], left=[p0], height=bar_height,
                        color=(AGG_DELTA if d >= 0 else AGG_DECL), edgecolor=edge, linewidth=0.0, zorder=1)
        else:  # Peer districts
            pos = max(0, d); neg = min(0, d)
            ax_main.barh([y_pos[i]], [p0], height=bar_height, color=BLUE_BASE, edgecolor=edge, linewidth=0.0, zorder=1)
            if pos > 0:
                ax_main.barh([y_pos[i]], [pos], left=[p0], height=bar_height, color=BLUE_DELTA, edgecolor=edge, linewidth=0.0, zorder=1)
            if neg < 0:
                ax_main.barh([y_pos[i]], [neg], left=[p0], height=bar_height, color=PURP_DECL, edgecolor=edge, linewidth=0.0, zorder=1)

    # X-axis configuration with improved label spacing
    # NOTE: Rotate labels slightly to prevent overlap, increase font for readability
    # Extended x-limit to 38000 (from 35000) to create space for enrollment annotations
    ax_main.set_xlim(0, 38000)
    ax_main.grid(axis="x", alpha=0.35, linewidth=0.8)
    ax_main.set_axisbelow(True)
    ax_main.set_xlabel("$ per pupil")
    ax_main.xaxis.set_major_formatter(comma_formatter())
    ax_main.tick_params(axis='x', labelsize=15, rotation=25)  # Slight rotation to prevent smooshing
    ax_main.set_yticks(y_pos, labels)

    # Add enrollment change annotations to right of bars
    # NOTE: Positioned far right at 37500 (well beyond longest bar) to avoid any overlap
    for i, s in enumerate(series_list):
        if s is None or len(s) < 2:
            continue
        ys = s.dropna().values.astype(float)

        # Calculate enrollment change
        enr_start = ys[0]
        enr_end = ys[-1]
        enr_change = enr_end - enr_start
        enr_pct = (enr_change / enr_start * 100) if enr_start != 0 else 0

        # Position at fixed right location (well beyond data range to avoid bar overlap)
        # Using 37500 (out of 38000 max) for consistent right-aligned positioning
        x_pos = 37500

        # Format annotation text
        sign = "+" if enr_change >= 0 else ""
        enr_text = f"{sign}{int(enr_change):,} ({sign}{enr_pct:.1f}%)"

        # Color based on change direction
        text_color = "#2563EB" if enr_change >= 0 else "#B91C1C"

        # Add annotation - RIGHT-ALIGNED at fixed position far from bars
        ax_main.text(x_pos, y_pos[i], enr_text, ha="right", va="center",
                    fontsize=11, color=text_color, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                             edgecolor=text_color, linewidth=1.2, alpha=0.9))

    # Add enrollment explainer text at top
    fig.text(0.5, 0.98, f"Enrollment change {t0}→{latest} shown to right of bars (FTE count and %)",
             ha="center", va="top", fontsize=14, style='italic', color='#B91C1C')

    # Legend at top
    handles = [
        Patch(facecolor=BLUE_BASE,  edgecolor=edge, label=f"{t0} PPE"),
        Patch(facecolor=BLUE_DELTA, edgecolor=edge, label=f"{latest} increase from {t0}"),
        Patch(facecolor=PURP_DECL,  edgecolor=edge, label=f"{latest} decrease from {t0}"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.94),
               ncol=3, frameon=False, fontsize=14)
    # NOTE: Right margin extended to 0.78 to accommodate right-aligned enrollment annotations
    plt.subplots_adjust(top=0.88, bottom=0.06, left=0.25, right=0.78)

    # Version stamp positioned higher (y=0.03) to avoid overlap with rotated x-axis labels
    _stamp(fig, y_pos=0.03)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=320, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {out_path}")

# ---- main ----
if __name__ == "__main__":
    """
    Main execution flow for generating all plots:
    1. Load data and create color mappings
    2. Generate Western MA regional aggregates (4 enrollment groups)
    3. Generate Western MA overview (all districts, horizontal bars)
    4. Generate all district plots (simple + detailed versions)

    NOTE: All plots saved to OUTPUT_DIR with standardized naming:
    - regional_expenditures_per_pupil_*.png for aggregates
    - expenditures_per_pupil_vs_enrollment_*_simple.png for simple district views
    - expenditures_per_pupil_vs_enrollment_*_detail.png for detailed district views
    - ppe_overview_all_western.png for Western overview
    """
    _boost_plot_fonts()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df, reg, c70 = load_data()
    # Note: add_alps_pk12() removed - no longer using ALPS PK-12 aggregate concept
    cmap_all = create_or_load_color_map(df)

    pivots_all, district_lines_all = [], []

    # Get Western cohorts using centralized function (ensures consistency with NSS)
    cohorts = get_western_cohort_districts(df, reg)

    # Western MA - 4 enrollment groups
    western_prepared = {}
    cohort_map = {"tiny": "TINY", "small": "SMALL", "medium": "MEDIUM", "large": "LARGE", "springfield": "SPRINGFIELD"}
    for bucket in ("tiny", "small", "medium", "large", "springfield"):
        district_list = cohorts[cohort_map[bucket]]
        title, piv, lines_sum, lines_mean = prepare_western_epp_lines(df, reg, bucket, c70, districts=district_list)
        # Use weighted average (mean) for aggregate enrollment lines, not sum
        western_prepared[bucket] = (title, piv, lines_mean)
        if not piv.empty: pivots_all.append(piv)

    # Districts
    district_prepared = {}
    for dist in DISTRICTS_OF_INTEREST:
        piv, lines = prepare_district_epp_lines(df, dist, c70)
        district_prepared[dist] = (piv, lines)
        if not piv.empty: pivots_all.append(piv)
        district_lines_all.append(lines)

    right_ylim = compute_global_dollar_ylim(pivots_all, pad=1.06, step=500)
    left_ylim_districts = compute_districts_fte_ylim(district_lines_all, pad=1.06, step=50)

    # Western plots - 4 enrollment groups
    for bucket in ("tiny", "small", "medium", "large", "springfield"):
        _title, piv, lines_mean = western_prepared[bucket]
        if piv.empty:
            print(f"[SKIP] No data for Western MA {bucket} group")
            continue
        context = context_for_western(bucket)
        out = OUTPUT_DIR / f"regional_expenditures_per_pupil_Western_Traditional_{bucket}.png"

        # Determine enrollment label and y-axis scale based on group
        # Use centralized cohort definitions
        cohort_key = bucket.upper()
        if bucket == "springfield":
            enrollment_label = "Enrollment"
            left_ylim_agg = compute_districts_fte_ylim([lines_mean], pad=1.06, step=1000)
        else:
            enrollment_label = "Weighted avg enrollment per district"
            left_ylim_agg = get_cohort_ylim(cohort_key)

        plot_one(out, piv, lines_mean, context, right_ylim, left_ylim_agg, FTE_LINE_COLORS, cmap_all, enrollment_label)

    # Western MA overview plot (all districts as horizontal bars)
    plot_all_western_overview(OUTPUT_DIR / "ppe_overview_all_western.png", df, reg, c70, year_lag=5)

    # District plots - generate BOTH simple and detailed versions
    ordered = ["Amherst-Pelham"] + [d for d in DISTRICTS_OF_INTEREST if d != "Amherst-Pelham"]
    for dist in ordered:
        piv, lines = district_prepared[dist]
        context = context_for_district(df, dist)

        # Use cohort-based y-axis limit automatically
        # This ensures each district's enrollment axis matches its cohort range
        left_ylim_dist = get_cohort_ylim(context)
        if left_ylim_dist is None:  # Springfield has no fixed ylim
            left_ylim_dist = left_ylim_districts

        # Individual districts use "Enrollment" label
        enrollment_label = "Enrollment"

        # Simple version (solid color, no categories)
        out_simple = OUTPUT_DIR / f"expenditures_per_pupil_vs_enrollment_{dist.replace(' ', '_')}_simple.png"
        plot_one_simple(out_simple, piv, lines, context, right_ylim, left_ylim_dist, FTE_LINE_COLORS, enrollment_label)

        # Detailed version (with category breakdown)
        out_detail = OUTPUT_DIR / f"expenditures_per_pupil_vs_enrollment_{dist.replace(' ', '_')}_detail.png"
        plot_one(out_detail, piv, lines, context, right_ylim, left_ylim_dist, FTE_LINE_COLORS, cmap_all, enrollment_label)
