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
    """
    print("\n[Executive Summary] Generating YoY separate panes plot...")

    # Organize all items to plot (cohorts first, then districts within each cohort)
    plot_items = []
    for cohort in sorted(cohort_districts.keys()):
        # Add cohort aggregate
        if cohort in cohort_yoy:
            plot_items.append((f"{cohort.title()} Cohort", cohort_yoy[cohort], cohort, True))
        # Add districts in this cohort
        for dist in cohort_districts[cohort]:
            if dist in district_yoy:
                plot_items.append((dist, district_yoy[dist], cohort, False))

    n_items = len(plot_items)
    fig, axes = plt.subplots(n_items, 1, figsize=(12, 2 * n_items), sharex=True)
    if n_items == 1:
        axes = [axes]

    for idx, (name, yoy_data, cohort, is_cohort) in enumerate(plot_items):
        ax = axes[idx]
        color = COHORT_COLORS.get(cohort, "#777777")
        linewidth = 3 if is_cohort else 2
        alpha = 0.7 if is_cohort else 0.9

        ax.plot(yoy_data.index, yoy_data.values, color=color, linewidth=linewidth, alpha=alpha)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_ylabel('YoY (%)', fontsize=10)
        ax.set_title(name, fontsize=11, fontweight='bold' if is_cohort else 'normal', loc='left')
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}%'))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[-1].set_xlabel('Year', fontsize=12)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"  Saved: {output_path}")


def plot_cagr_separate_panes(
    df: pd.DataFrame,
    cohort_districts: Dict,
    cohorts: Dict,
    output_path: Path
):
    """
    Plot CAGR chunks with each district/cohort in its own horizontal pane.
    """
    print("\n[Executive Summary] Generating CAGR separate panes plot...")

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

    # Organize plot items
    plot_items = []
    for cohort in sorted(cohort_districts.keys()):
        if cohort in cohort_cagr:
            plot_items.append((f"{cohort.title()} Cohort", cohort_cagr[cohort], cohort, True))
        for dist in cohort_districts[cohort]:
            if dist in district_cagr:
                plot_items.append((dist, district_cagr[dist], cohort, False))

    n_items = len(plot_items)
    fig, axes = plt.subplots(n_items, 1, figsize=(12, 2 * n_items), sharex=True)
    if n_items == 1:
        axes = [axes]

    for idx, (name, cagr_data, cohort, is_cohort) in enumerate(plot_items):
        ax = axes[idx]
        color = COHORT_COLORS.get(cohort, "#777777")
        linewidth = 3 if is_cohort else 2
        alpha = 0.7 if is_cohort else 0.9

        ax.plot(cagr_data.index, cagr_data.values, color=color, linewidth=linewidth,
                alpha=alpha, marker='o', markersize=8 if is_cohort else 6)
        ax.set_ylabel('CAGR (%)', fontsize=10)
        ax.set_title(name, fontsize=11, fontweight='bold' if is_cohort else 'normal', loc='left')
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}%'))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[-1].set_xlabel('Period End Year', fontsize=12)
    axes[-1].set_xticks([2014, 2019, 2024])
    axes[-1].set_xticklabels(['2009-2014', '2014-2019', '2019-2024'])
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
    print("\n[1/6] Loading data...")
    df, reg, c70 = load_data()
    print(f"  Loaded {len(df)} expenditure records")

    # Generate YoY growth plot (returns data for other plots)
    print("\n[2/6] Generating YoY growth plot...")
    output_path = OUTPUT_DIR / "executive_summary_yoy_growth.png"
    district_yoy, cohort_yoy, cohort_districts, cohorts = plot_yoy_growth_districts_of_interest(df, reg, output_path)

    # Generate CAGR chunks plot
    print("\n[3/6] Generating CAGR chunks plot...")
    output_path = OUTPUT_DIR / "executive_summary_cagr_chunks.png"
    plot_cagr_chunks_districts_of_interest(df, reg, output_path)

    # Generate YoY separate panes plot
    print("\n[4/6] Generating YoY separate panes plot...")
    output_path = OUTPUT_DIR / "executive_summary_yoy_panes.png"
    plot_yoy_separate_panes(district_yoy, cohort_yoy, cohort_districts, output_path)

    # Generate CAGR separate panes plot
    print("\n[5/6] Generating CAGR separate panes plot...")
    output_path = OUTPUT_DIR / "executive_summary_cagr_panes.png"
    plot_cagr_separate_panes(df, cohort_districts, cohorts, output_path)

    print("\n[6/6] All plots generated!")

    print("\n" + "=" * 70)
    print("[SUCCESS] Executive Summary plots generated!")
    print("=" * 70)


if __name__ == "__main__":
    main()
