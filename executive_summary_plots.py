"""
Executive Summary Plots - YoY Growth Analysis

Generates year-over-year (YoY) growth rate plots for districts of interest
and their cohort aggregates.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

from school_shared import (
    OUTPUT_DIR,
    load_data,
    DISTRICTS_OF_INTEREST,
    prepare_district_epp_lines,
    weighted_epp_aggregation,
    get_western_cohort_districts,
    context_for_district,
)

# Cohort colors (from western_map.py)
COHORT_COLORS = {
    "TINY": "#4575B4",      # Blue
    "SMALL": "#3C9DC4",     # Cyan
    "MEDIUM": "#FDB749",    # Amber
    "LARGE": "#F46D43",     # Orange
    "X-LARGE": "#D73027",   # Red
    "SPRINGFIELD": "#A50026"  # Dark Red
}

# Line styles for districts within cohorts
LINE_STYLES = {
    0: '-',      # Solid (cohort aggregate)
    1: '--',     # Dashed
    2: ':',      # Dotted
    3: '-.',     # Dash-dot
    4: (0, (3, 1, 1, 1)),  # Dash-dot-dot
}


def calculate_yoy_growth(series: pd.Series) -> pd.Series:
    """
    Calculate year-over-year growth rate as percentage.

    YoY Growth = (Value_year / Value_previous_year - 1) * 100

    Args:
        series: Time series with years as index

    Returns:
        Series of YoY growth percentages (starts one year later than input)
    """
    # Shift by 1 to get previous year's values
    prev_values = series.shift(1)

    # Calculate YoY growth: (current / previous - 1) * 100
    yoy = ((series / prev_values) - 1) * 100

    # Drop the first year (NaN since no previous year)
    return yoy.dropna()


def calculate_cagr_chunks(series: pd.Series) -> pd.Series:
    """
    Calculate CAGR for 5-year chunks: 2009-2014, 2014-2019, 2019-2024.

    CAGR = (End_Value / Start_Value)^(1 / Years) - 1

    Args:
        series: Time series with years as index

    Returns:
        Series with CAGR values indexed by end year of each period
    """
    periods = [
        (2009, 2014),
        (2014, 2019),
        (2019, 2024)
    ]

    cagr_values = {}
    for start_year, end_year in periods:
        if start_year in series.index and end_year in series.index:
            start_val = series[start_year]
            end_val = series[end_year]
            years = end_year - start_year

            if start_val > 0 and end_val > 0:
                cagr = ((end_val / start_val) ** (1 / years) - 1) * 100
                cagr_values[end_year] = cagr

    return pd.Series(cagr_values)


def calculate_cagr_15year(series: pd.Series) -> float:
    """
    Calculate 15-year CAGR from 2009 to 2024.

    Args:
        series: Time series with years as index

    Returns:
        CAGR value as percentage, or 0 if data not available
    """
    start_year, end_year = 2009, 2024
    if start_year in series.index and end_year in series.index:
        start_val = series[start_year]
        end_val = series[end_year]
        years = end_year - start_year

        if start_val > 0 and end_val > 0:
            cagr = ((end_val / start_val) ** (1 / years) - 1) * 100
            return cagr

    return 0


def plot_yoy_growth_districts_of_interest(
    df: pd.DataFrame,
    reg: pd.DataFrame,
    output_path: Path
):
    """
    Plot YoY growth for districts of interest and their cohorts.

    Shows:
    - Each district of interest as a line (with district-specific style)
    - Cohort aggregates as thicker solid lines
    - Color indicates cohort membership
    - Line style distinguishes individual districts
    """
    print("\n[Executive Summary] Generating YoY growth plot...")

    # Get cohort assignments for each district
    cohorts = get_western_cohort_districts(df, reg)

    # Map districts to cohorts
    district_cohort_map = {}
    for cohort_name, district_list in cohorts.items():
        for dist in district_list:
            district_cohort_map[dist.lower()] = cohort_name

    # Organize districts by cohort
    cohort_districts = {}
    for dist in DISTRICTS_OF_INTEREST:
        cohort = district_cohort_map.get(dist.lower())
        if cohort:
            if cohort not in cohort_districts:
                cohort_districts[cohort] = []
            cohort_districts[cohort].append(dist)

    print(f"  Districts by cohort:")
    for cohort, dists in cohort_districts.items():
        print(f"    {cohort}: {', '.join(dists)}")

    # Calculate YoY growth for each district
    district_yoy = {}
    for dist in DISTRICTS_OF_INTEREST:
        epp_pivot, _ = prepare_district_epp_lines(df, dist)
        if not epp_pivot.empty:
            total_ppe = epp_pivot.sum(axis=1)  # Sum across all categories
            yoy = calculate_yoy_growth(total_ppe)
            district_yoy[dist] = yoy
            print(f"  {dist}: {len(yoy)} years of YoY data")

    # Calculate YoY growth for each cohort aggregate
    cohort_yoy = {}
    for cohort, dists in cohort_districts.items():
        # Get all districts in this cohort (not just districts of interest)
        all_cohort_districts = cohorts[cohort]

        # Calculate weighted aggregate PPE
        epp_pivot, _, _ = weighted_epp_aggregation(df, all_cohort_districts)
        if not epp_pivot.empty:
            total_ppe = epp_pivot.sum(axis=1)
            yoy = calculate_yoy_growth(total_ppe)
            cohort_yoy[cohort] = yoy
            print(f"  {cohort} cohort aggregate: {len(yoy)} years of YoY data")

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot cohort aggregates first (thicker, solid lines) - these go in background
    for cohort, yoy_data in cohort_yoy.items():
        color = COHORT_COLORS.get(cohort, "#777777")
        ax.plot(
            yoy_data.index,
            yoy_data.values,
            color=color,
            linestyle='-',
            linewidth=3,
            alpha=0.6,
            label=f"{cohort.title()} Cohort",
            zorder=1
        )

    # Plot individual districts (thinner, varied line styles) - these go in foreground
    for cohort, dists in cohort_districts.items():
        color = COHORT_COLORS.get(cohort, "#777777")

        # Assign line styles to districts in this cohort
        for idx, dist in enumerate(dists, start=1):
            if dist in district_yoy:
                yoy_data = district_yoy[dist]
                linestyle = LINE_STYLES.get(idx, '-')

                ax.plot(
                    yoy_data.index,
                    yoy_data.values,
                    color=color,
                    linestyle=linestyle,
                    linewidth=2,
                    alpha=0.9,
                    label=dist,
                    zorder=2
                )

    # Add horizontal line at 0% for reference
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=0)

    # Formatting
    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel('YoY Growth Rate (%)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}%'))

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"  Saved: {output_path}")

    # Return the data for use in other plots
    return district_yoy, cohort_yoy, cohort_districts, cohorts


def plot_cagr_chunks_districts_of_interest(
    df: pd.DataFrame,
    reg: pd.DataFrame,
    output_path: Path
):
    """
    Plot CAGR for 5-year chunks (2009-2014, 2014-2019, 2019-2024).

    Shows:
    - Each district of interest as a line
    - Cohort aggregates as thicker solid lines
    - Color indicates cohort membership
    - Line style distinguishes individual districts
    """
    print("\n[Executive Summary] Generating CAGR chunks plot...")

    # Get cohort assignments
    cohorts = get_western_cohort_districts(df, reg)
    district_cohort_map = {}
    for cohort_name, district_list in cohorts.items():
        for dist in district_list:
            district_cohort_map[dist.lower()] = cohort_name

    # Organize districts by cohort
    cohort_districts = {}
    for dist in DISTRICTS_OF_INTEREST:
        cohort = district_cohort_map.get(dist.lower())
        if cohort:
            if cohort not in cohort_districts:
                cohort_districts[cohort] = []
            cohort_districts[cohort].append(dist)

    # Calculate CAGR chunks for each district
    district_cagr = {}
    for dist in DISTRICTS_OF_INTEREST:
        epp_pivot, _ = prepare_district_epp_lines(df, dist)
        if not epp_pivot.empty:
            total_ppe = epp_pivot.sum(axis=1)
            cagr = calculate_cagr_chunks(total_ppe)
            district_cagr[dist] = cagr
            print(f"  {dist}: {len(cagr)} CAGR chunks")

    # Calculate CAGR chunks for each cohort aggregate
    cohort_cagr = {}
    for cohort, dists in cohort_districts.items():
        all_cohort_districts = cohorts[cohort]
        epp_pivot, _, _ = weighted_epp_aggregation(df, all_cohort_districts)
        if not epp_pivot.empty:
            total_ppe = epp_pivot.sum(axis=1)
            cagr = calculate_cagr_chunks(total_ppe)
            cohort_cagr[cohort] = cagr
            print(f"  {cohort} cohort aggregate: {len(cagr)} CAGR chunks")

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot cohort aggregates (thicker lines)
    for cohort, cagr_data in cohort_cagr.items():
        color = COHORT_COLORS.get(cohort, "#777777")
        ax.plot(
            cagr_data.index,
            cagr_data.values,
            color=color,
            linestyle='-',
            linewidth=3,
            alpha=0.6,
            label=f"{cohort.title()} Cohort",
            marker='o',
            markersize=8,
            zorder=1
        )

    # Plot individual districts
    for cohort, dists in cohort_districts.items():
        color = COHORT_COLORS.get(cohort, "#777777")
        for idx, dist in enumerate(dists, start=1):
            if dist in district_cagr:
                cagr_data = district_cagr[dist]
                linestyle = LINE_STYLES.get(idx, '-')
                ax.plot(
                    cagr_data.index,
                    cagr_data.values,
                    color=color,
                    linestyle=linestyle,
                    linewidth=2,
                    alpha=0.9,
                    label=dist,
                    marker='o',
                    markersize=6,
                    zorder=2
                )

    # Formatting
    ax.set_xlabel('Period End Year', fontsize=14)
    ax.set_ylabel('5-Year CAGR (%)', fontsize=14)
    ax.set_xticks([2014, 2019, 2024])
    ax.set_xticklabels(['2009-2014', '2014-2019', '2019-2024'])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}%'))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"  Saved: {output_path}")


def plot_yoy_separate_panes(
    district_yoy: Dict,
    cohort_yoy: Dict,
    cohort_districts: Dict,
    output_path: Path
):
    """
    Plot YoY growth with each district/cohort in its own horizontal pane.
    Uses color shading by cohort and diagonal hatching for cohort aggregates.
    """
    print("\n[Executive Summary] Generating YoY separate panes plot...")

    from school_shared import COHORT_DEFINITIONS

    # Organize all items to plot by cohort with color assignments
    plot_items = []  # (name, yoy_data, color, is_cohort)

    for cohort in sorted(cohort_districts.keys()):
        base_color = COHORT_COLORS.get(cohort, "#777777")
        cohort_items = []

        # Get FTE range for cohort label
        cohort_def = COHORT_DEFINITIONS.get(cohort, {})
        fte_range = cohort_def.get('range', (0, 0))
        fte_label = f" ({fte_range[0]:.0f}-{fte_range[1]:.0f} FTE)"

        # Add cohort aggregate with FTE range
        if cohort in cohort_yoy:
            cohort_items.append((f"{cohort.title()} Cohort{fte_label}", cohort_yoy[cohort], True))

        # Add districts in this cohort
        for dist in cohort_districts[cohort]:
            if dist in district_yoy:
                cohort_items.append((dist, district_yoy[dist], False))

        # Generate color shades for this cohort's items
        n_items = len(cohort_items)
        if n_items > 1:
            shades = generate_cohort_shades(base_color, n_items)
        else:
            shades = [base_color]

        # Assign colors to items
        for (name, yoy_data, is_cohort), color in zip(cohort_items, shades):
            plot_items.append((name, yoy_data, color, is_cohort))

    n_items = len(plot_items)
    fig, axes = plt.subplots(n_items, 1, figsize=(12, 2 * n_items), sharex=True, sharey=True)
    if n_items == 1:
        axes = [axes]

    # Calculate global y-axis range across all plots
    all_values = []
    for name, yoy_data, color, is_cohort in plot_items:
        all_values.extend(yoy_data.values)
    y_min, y_max = min(all_values), max(all_values)
    y_padding = (y_max - y_min) * 0.1
    y_range = (y_min - y_padding, y_max + y_padding)

    # Ensure we show -10% if needed
    if y_range[0] > -10:
        y_range = (min(y_range[0], -10), y_range[1])

    for idx, (name, yoy_data, color, is_cohort) in enumerate(plot_items):
        ax = axes[idx]
        linewidth = 5.5 if is_cohort else 2  # Bigger cohort line thickness
        alpha = 0.8 if is_cohort else 0.9

        # Plot line (fill area with hatching for cohorts)
        if is_cohort:
            # Plot with hatching for cohorts
            ax.plot(yoy_data.index, yoy_data.values, color=color, linewidth=linewidth, alpha=alpha)
            # Add hatched fill
            ax.fill_between(yoy_data.index, 0, yoy_data.values, color=color, alpha=0.3, hatch='////', edgecolor=color, linewidth=0)
        else:
            ax.plot(yoy_data.index, yoy_data.values, color=color, linewidth=linewidth, alpha=alpha)

        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_ylabel('YoY (%)', fontsize=14)
        ax.set_title(name, fontsize=14, fontweight='bold' if is_cohort else 'normal', loc='left')
        ax.set_ylim(y_range)  # Set consistent y-axis range

        # Set specific tick marks at -10%, 0%, 10%, 20%
        ax.set_yticks([-10, 0, 10, 20])

        ax.grid(True, alpha=0.3)
        # Format y-axis as integers without decimals
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y)}%'))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Increase tick label sizes
        ax.tick_params(axis='both', labelsize=12)

    axes[-1].set_xlabel('Year', fontsize=14)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"  Saved: {output_path}")


def generate_cohort_shades(base_color: str, n_shades: int) -> List[str]:
    """
    Generate shades of a base color for districts within a cohort.

    Args:
        base_color: Hex color (e.g., "#4575B4")
        n_shades: Number of shades to generate

    Returns:
        List of hex color strings
    """
    import matplotlib.colors as mcolors

    # Convert hex to RGB
    rgb = mcolors.hex2color(base_color)

    # Generate shades by adjusting brightness
    shades = []
    for i in range(n_shades):
        # Vary brightness from 0.5 to 1.2
        factor = 0.5 + (i / max(n_shades - 1, 1)) * 0.7
        new_rgb = tuple(min(1.0, c * factor) for c in rgb)
        shades.append(mcolors.rgb2hex(new_rgb))

    return shades


def plot_cagr_grouped_bars(
    df: pd.DataFrame,
    cohort_districts: Dict,
    cohorts: Dict,
    output_path: Path
):
    """
    Plot 5-year CAGR as vertical grouped bars with time periods on x-axis.
    Colors are cohort-based with shades for different districts.
    Cohorts shown with thicker diagonal white hatching.
    """
    print("\n[Executive Summary] Generating CAGR grouped bars plot...")

    # Calculate CAGR for all districts
    district_cagr = {}
    for dist in DISTRICTS_OF_INTEREST:
        epp_pivot, _ = prepare_district_epp_lines(df, dist)
        if not epp_pivot.empty:
            total_ppe = epp_pivot.sum(axis=1)
            cagr = calculate_cagr_chunks(total_ppe)
            district_cagr[dist] = cagr

    # Calculate CAGR for cohort aggregates
    cohort_cagr = {}
    for cohort, dists in cohort_districts.items():
        all_cohort_districts = cohorts[cohort]
        epp_pivot, _, _ = weighted_epp_aggregation(df, all_cohort_districts)
        if not epp_pivot.empty:
            total_ppe = epp_pivot.sum(axis=1)
            cagr = calculate_cagr_chunks(total_ppe)
            cohort_cagr[cohort] = cagr

    # Organize plot items by cohort and assign colors
    plot_items = []  # (name, cagr_data, color, is_cohort)

    for cohort in sorted(cohort_districts.keys()):
        base_color = COHORT_COLORS.get(cohort, "#777777")
        cohort_items = []

        # Add cohort aggregate first
        if cohort in cohort_cagr:
            cohort_items.append((f"{cohort.title()} Cohort", cohort_cagr[cohort], True))

        # Add districts in this cohort
        for dist in cohort_districts[cohort]:
            if dist in district_cagr:
                cohort_items.append((dist, district_cagr[dist], False))

        # Generate shades for this cohort's items
        n_items = len(cohort_items)
        if n_items > 1:
            shades = generate_cohort_shades(base_color, n_items)
        else:
            shades = [base_color]

        # Assign colors to items
        for (name, cagr_data, is_cohort), color in zip(cohort_items, shades):
            plot_items.append((name, cagr_data, color, is_cohort))

    # Set up the figure
    periods = [2014, 2019, 2024]
    period_labels = ['2009-2014', '2014-2019', '2019-2024']
    n_periods = len(periods)
    n_items = len(plot_items)

    fig, ax = plt.subplots(figsize=(16, 9))  # Same width as legend and 15-year plot

    # Bar dimensions
    bar_width = 0.8 / n_items  # Total width of 0.8 per period group
    x_positions = np.arange(n_periods)

    # Plot bars
    for item_idx, (name, cagr_data, color, is_cohort) in enumerate(plot_items):
        x_offset = (item_idx - n_items / 2) * bar_width + bar_width / 2

        cagr_values = []
        for period in periods:
            cagr_val = cagr_data.get(period, 0) if period in cagr_data.index else 0
            cagr_values.append(cagr_val)

        # Plot bars with thicker white diagonal lines for cohorts
        if is_cohort:
            ax.bar(x_positions + x_offset, cagr_values, bar_width * 0.95,
                   color=color, alpha=0.8, edgecolor='white', linewidth=2,
                   hatch='////', label=name)
        else:
            ax.bar(x_positions + x_offset, cagr_values, bar_width * 0.95,
                   color=color, alpha=0.9, edgecolor='white', linewidth=0.5,
                   label=name)

    # Customize axes
    ax.set_xlabel('Time Period', fontsize=20, fontweight='bold')
    ax.set_ylabel('5-Year CAGR (%) of PPE', fontsize=20, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(period_labels, fontsize=18)
    ax.tick_params(axis='y', labelsize=17)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}%'))

    # Add horizontal line at 0
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=0)

    # Add legend above plot (horizontal layout, 2x bigger font and swatches)
    legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.08), fontsize=26,
                       framealpha=0.95, ncol=4, markerscale=2.0, handlelength=4, handleheight=2)

    # Grid and spines
    ax.grid(True, axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # Make room for legend above
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"  Saved: {output_path}")


def plot_cagr_15year_bars(
    df: pd.DataFrame,
    cohort_districts: Dict,
    cohorts: Dict,
    output_path: Path
):
    """
    Plot 15-year CAGR (2009-2024) as a single group of vertical bars.
    Colors are cohort-based with shades for different districts.
    Cohorts shown with thicker diagonal white hatching.
    """
    print("\n[Executive Summary] Generating 15-year CAGR bars plot...")

    # Calculate 15-year CAGR for all districts
    district_cagr_15y = {}
    for dist in DISTRICTS_OF_INTEREST:
        epp_pivot, _ = prepare_district_epp_lines(df, dist)
        if not epp_pivot.empty:
            total_ppe = epp_pivot.sum(axis=1)
            cagr = calculate_cagr_15year(total_ppe)
            if cagr > 0:
                district_cagr_15y[dist] = cagr

    # Calculate 15-year CAGR for cohort aggregates
    cohort_cagr_15y = {}
    for cohort, dists in cohort_districts.items():
        all_cohort_districts = cohorts[cohort]
        epp_pivot, _, _ = weighted_epp_aggregation(df, all_cohort_districts)
        if not epp_pivot.empty:
            total_ppe = epp_pivot.sum(axis=1)
            cagr = calculate_cagr_15year(total_ppe)
            if cagr > 0:
                cohort_cagr_15y[cohort] = cagr

    # Organize plot items by cohort and assign colors
    plot_items = []  # (name, cagr_value, color, is_cohort)

    for cohort in sorted(cohort_districts.keys()):
        base_color = COHORT_COLORS.get(cohort, "#777777")
        cohort_items = []

        # Add cohort aggregate first
        if cohort in cohort_cagr_15y:
            cohort_items.append((f"{cohort.title()} Cohort", cohort_cagr_15y[cohort], True))

        # Add districts in this cohort
        for dist in cohort_districts[cohort]:
            if dist in district_cagr_15y:
                cohort_items.append((dist, district_cagr_15y[dist], False))

        # Generate shades for this cohort's items
        n_items = len(cohort_items)
        if n_items > 1:
            shades = generate_cohort_shades(base_color, n_items)
        else:
            shades = [base_color]

        # Assign colors to items
        for (name, cagr_value, is_cohort), color in zip(cohort_items, shades):
            plot_items.append((name, cagr_value, color, is_cohort))

    # Set up the figure
    n_items = len(plot_items)
    fig, ax = plt.subplots(figsize=(16, 7))

    # Bar positions - bars 2-3x wider than individual grouped plot bars
    x_positions = np.arange(n_items)
    # Grouped plot bars are: 0.8/n_items * 0.95 ≈ 0.076 (for n=10)
    # For 2.5x that width: bar_width ≈ 0.6
    bar_width = 0.6

    # Extract values and colors
    names = [name for name, _, _, _ in plot_items]
    cagr_values = [cagr for _, cagr, _, _ in plot_items]
    colors = [color for _, _, color, _ in plot_items]
    is_cohorts = [is_cohort for _, _, _, is_cohort in plot_items]

    # Plot bars
    for i, (x, cagr_val, color, is_cohort) in enumerate(zip(x_positions, cagr_values, colors, is_cohorts)):
        if is_cohort:
            ax.bar(x, cagr_val, bar_width,
                   color=color, alpha=0.8, edgecolor='white', linewidth=2,
                   hatch='////')
        else:
            ax.bar(x, cagr_val, bar_width,
                   color=color, alpha=0.9, edgecolor='white', linewidth=0.5)

    # Customize axes
    ax.set_xlabel('District / Cohort', fontsize=20, fontweight='bold')
    ax.set_ylabel('15-Year CAGR (%) of PPE\n2009-2024', fontsize=20, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=14)
    ax.tick_params(axis='y', labelsize=17)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}%'))

    # Add horizontal line at 0
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=0)

    # Grid and spines
    ax.grid(True, axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"  Saved: {output_path}")


def main():
    """Generate Executive Summary plots."""
    print("=" * 70)
    print("Executive Summary - Growth Analysis")
    print("=" * 70)

    # Load data
    print("\n[1/5] Loading data...")
    df, reg, c70 = load_data()
    print(f"  Loaded {len(df)} expenditure records")

    # Calculate YoY growth data for other plots (no longer generating standalone YoY plot)
    print("\n[2/5] Calculating YoY growth data...")
    # Get cohort assignments
    cohorts = get_western_cohort_districts(df, reg)
    district_cohort_map = {}
    for cohort_name, district_list in cohorts.items():
        for dist in district_list:
            district_cohort_map[dist.lower()] = cohort_name

    # Organize districts by cohort
    cohort_districts = {}
    for dist in DISTRICTS_OF_INTEREST:
        cohort = district_cohort_map.get(dist.lower())
        if cohort:
            if cohort not in cohort_districts:
                cohort_districts[cohort] = []
            cohort_districts[cohort].append(dist)

    # Calculate YoY growth for each district
    district_yoy = {}
    for dist in DISTRICTS_OF_INTEREST:
        epp_pivot, _ = prepare_district_epp_lines(df, dist)
        if not epp_pivot.empty:
            total_ppe = epp_pivot.sum(axis=1)
            yoy = calculate_yoy_growth(total_ppe)
            district_yoy[dist] = yoy

    # Calculate YoY growth for each cohort aggregate
    cohort_yoy = {}
    for cohort, dists in cohort_districts.items():
        all_cohort_districts = cohorts[cohort]
        epp_pivot, _, _ = weighted_epp_aggregation(df, all_cohort_districts)
        if not epp_pivot.empty:
            total_ppe = epp_pivot.sum(axis=1)
            yoy = calculate_yoy_growth(total_ppe)
            cohort_yoy[cohort] = yoy

    # Generate YoY separate panes plot
    print("\n[3/5] Generating YoY separate panes plot...")
    output_path = OUTPUT_DIR / "executive_summary_yoy_panes.png"
    plot_yoy_separate_panes(district_yoy, cohort_yoy, cohort_districts, output_path)

    # Generate CAGR 5-year grouped bars plot
    print("\n[4/5] Generating 5-year CAGR grouped bars plot...")
    output_path = OUTPUT_DIR / "executive_summary_cagr_grouped.png"
    plot_cagr_grouped_bars(df, cohort_districts, cohorts, output_path)

    # Generate CAGR 15-year bars plot
    print("\n[5/5] Generating 15-year CAGR bars plot...")
    output_path = OUTPUT_DIR / "executive_summary_cagr_15year.png"
    plot_cagr_15year_bars(df, cohort_districts, cohorts, output_path)

    print("\n" + "=" * 70)
    print("[SUCCESS] Executive Summary plots generated!")
    print("=" * 70)


if __name__ == "__main__":
    main()
