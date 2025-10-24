"""
Western MA Enrollment Distribution - Individual Plots for PDF

Creates 4 separate plots for PDF integration:
1. Scatterplot with distribution curve
2. Box-and-whisker with quartile boundaries
3. Distribution histogram (250 FTE bins)
4. Proposed 4-group enrollment bands

Each plot saved separately for PDF layout (2 per page).
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.interpolate import make_interp_spline

from school_shared import (
    OUTPUT_DIR, load_data, DISTRICTS_OF_INTEREST,
    latest_indistrict_fte, get_indistrict_fte_for_year, get_enrollment_group, get_cohort_label,
    get_outlier_threshold, EXCLUDE_DISTRICTS
)

# Styling constants
HIGHLIGHT_COLOR = "#FF8C00"  # Dark orange for districts of interest
WESTERN_COLOR = "#4682B4"  # Steel blue for other Western MA districts
MEAN_COLOR = "#DC143C"  # Crimson for mean line
MEDIAN_COLOR = "#228B22"  # Forest green for median line
CURVE_COLOR = "#708090"  # Slate gray for distribution curve

CODE_VERSION = "v2025.10.04-ENROLLMENT-DIST-V2"

# Outlier threshold calculated dynamically using IQR method (Q3 + 1.5*IQR)


def analyze_western_enrollment(df: pd.DataFrame, reg: pd.DataFrame, latest_year: int):
    """Analyze enrollment distribution for Western MA traditional districts."""
    from school_shared import calculate_cohort_boundaries, get_western_cohort_districts_for_year

    # Calculate year-specific cohort definitions for display
    year_cohort_defs = calculate_cohort_boundaries(df, reg, latest_year)
    year_enrollment_groups = {k: v["range"] for k, v in year_cohort_defs.items()}
    outlier_threshold = 10000  # Fixed threshold for Springfield

    # Get Western MA traditional districts using centralized year-specific cohort function
    # This ensures consistent filtering (includes PPE validation)
    cohorts = get_western_cohort_districts_for_year(df, reg, latest_year)
    western_districts = []
    for cohort_key in ["TINY", "SMALL", "MEDIUM", "LARGE", "SPRINGFIELD"]:
        western_districts.extend(cohorts.get(cohort_key, []))

    # Get enrollment and PPE for each district
    district_data = []
    for dist in western_districts:
        # Use year-specific IN-DISTRICT enrollment for cohort calculation
        enrollment = get_indistrict_fte_for_year(df, dist, latest_year)
        if enrollment > 0:
            # Get total PPE for latest year
            ppe_data = df[
                (df["DIST_NAME"].str.lower() == dist) &
                (df["IND_CAT"].str.lower() == "expenditures per pupil") &
                (df["YEAR"] == latest_year)
            ]
            total_ppe = ppe_data[~ppe_data["IND_SUBCAT"].str.lower().isin(["total expenditures", "total in-district expenditures"])]["IND_VALUE"].sum()

            # District already filtered by get_western_cohort_districts_for_year() to have valid PPE
            if total_ppe > 0:
                # Get cohort assignment using year-specific boundaries
                from school_shared import _get_enrollment_group_for_boundaries
                cohort = _get_enrollment_group_for_boundaries(enrollment, year_enrollment_groups)

                district_data.append({
                    'name': dist,
                    'enrollment': enrollment,
                    'ppe': total_ppe,
                    'cohort': cohort,
                    'is_highlight': dist in [d.lower() for d in DISTRICTS_OF_INTEREST],
                })

    # Sort by enrollment
    district_data.sort(key=lambda x: x['enrollment'])

    # Extract lists
    districts = [d['name'] for d in district_data]
    enrollments = np.array([d['enrollment'] for d in district_data])
    ppes = np.array([d['ppe'] for d in district_data])
    cohorts = [d['cohort'] for d in district_data]
    is_highlight = [d['is_highlight'] for d in district_data]

    # Calculate statistics
    mean_enr = np.mean(enrollments)
    median_enr = np.median(enrollments)
    q1, q2, q3 = np.percentile(enrollments, [25, 50, 75])
    # Note: P90 is no longer used as Large cohort now extends to 10K FTE

    # Find Springfield (>10,000 FTE outliers)
    # Use fixed 10,000 threshold instead of statistical outlier_threshold
    SPRINGFIELD_THRESHOLD = 10000
    springfield_idx = [i for i, e in enumerate(enrollments) if e > SPRINGFIELD_THRESHOLD]
    springfield_name = [districts[i].title() for i in springfield_idx]
    springfield_value = [enrollments[i] for i in springfield_idx]

    # Calculate x_max for plots (highest non-outlier district)
    non_springfield = [e for e in enrollments if e <= SPRINGFIELD_THRESHOLD]
    x_max = max(non_springfield) * 1.1 if non_springfield else SPRINGFIELD_THRESHOLD

    return {
        'districts': districts,
        'enrollments': enrollments,
        'ppes': ppes,
        'cohorts': cohorts,
        'is_highlight': is_highlight,
        'mean': mean_enr,
        'median': median_enr,
        'quartiles': [q1, q2, q3],
        'latest_year': latest_year,
        'x_max': x_max,
        'springfield_name': springfield_name[0] if springfield_name else None,
        'springfield_value': springfield_value[0] if springfield_value else None,
        'outlier_threshold': SPRINGFIELD_THRESHOLD,  # Use fixed 10,000 threshold for display
        'year_cohort_defs': year_cohort_defs,  # Include year-specific cohort definitions
    }


def plot_scatterplot(data: dict, out_path: Path):
    """Plot 1: Scatterplot of enrollment vs PPE with cohort colors."""
    enrollments = data['enrollments']
    ppes = data['ppes']
    cohorts = data['cohorts']
    q1, q2, q3 = data['quartiles']
    latest_year = data['latest_year']
    year_cohort_defs = data.get('year_cohort_defs', {})

    # Define cohort colors (6 tiers) - more saturated for visibility
    cohort_colors = {
        'TINY': '#4575B4',       # Blue (low enrollment)
        'SMALL': '#3C9DC4',      # Cyan (more saturated)
        'MEDIUM': '#FDB749',     # Amber (more saturated)
        'LARGE': '#D73027',      # Red (now includes former X-Large)
        'SPRINGFIELD': '#A50026'  # Dark Red (outliers)
    }

    fig, ax = plt.subplots(figsize=(11, 5))

    # Filter out extreme outliers (SPRINGFIELD) for plotting
    outlier_threshold = data['outlier_threshold']
    mask = enrollments <= outlier_threshold
    enrollments_plot = enrollments[mask]
    ppes_plot = ppes[mask]
    cohorts_plot = [c for c, m in zip(cohorts, mask) if m]

    # Plot points by cohort
    for cohort_key in ['TINY', 'SMALL', 'MEDIUM', 'LARGE']:
        cohort_mask = np.array([c == cohort_key for c in cohorts_plot])
        if cohort_mask.any():
            # Use year-specific label if available, otherwise fall back to global
            cohort_label = year_cohort_defs.get(cohort_key, {}).get('label', get_cohort_label(cohort_key))
            ax.scatter(enrollments_plot[cohort_mask], ppes_plot[cohort_mask],
                      c=cohort_colors[cohort_key], s=100, alpha=0.7,
                      edgecolors='white', linewidth=1.5,
                      label=cohort_label, zorder=3)

    # Add quartile and percentile vertical lines
    ax.axvline(q1, color='purple', linestyle=':', linewidth=2,
               label=f'Q1: {q1:.0f} FTE', zorder=2, alpha=0.7)
    ax.axvline(q2, color='purple', linestyle='--', linewidth=2,
               label=f'Median: {q2:.0f} FTE', zorder=2, alpha=0.7)
    ax.axvline(q3, color='purple', linestyle=':', linewidth=2,
               label=f'Q3: {q3:.0f} FTE', zorder=2, alpha=0.7)

    # Note: P90 line removed as Large cohort now extends to 10K FTE

    # Add K/M formatter to x-axis (enrollment)
    from matplotlib.ticker import FuncFormatter
    def enrollment_formatter(x, _):
        if x >= 1_000_000:
            return f"{x/1_000_000:.1f}M".rstrip('0').rstrip('.')
        elif x >= 1_000:
            return f"{x/1_000:.1f}K".rstrip('0').rstrip('.')
        else:
            return f"{int(x)}"

    # Add $ and K/M formatter to y-axis (PPE)
    def ppe_formatter(x, _):
        if x >= 1_000_000:
            return f"${x/1_000_000:.1f}M".rstrip('0').rstrip('.')
        elif x >= 1_000:
            return f"${x/1_000:.1f}K".rstrip('0').rstrip('.')
        else:
            return f"${int(x)}"

    ax.xaxis.set_major_formatter(FuncFormatter(enrollment_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(ppe_formatter))

    ax.set_xlabel('In-District FTE Enrollment', fontsize=14)
    ax.set_ylabel(f'Total Per-Pupil Expenditure', fontsize=14)
    # No title - page subtitle does the job
    ax.grid(True, alpha=0.3)

    # Add year annotation at top right corner
    ax.text(0.98, 0.90, str(latest_year), transform=ax.transAxes,
            fontsize=20, fontweight='bold', ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', linewidth=2))

    # Remove top, left, right borders - keep only bottom
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend below the plot
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=5, fontsize=10,
             title="Springfield (>10K FTE) omitted as high-enrollment outlier",
             title_fontsize=9)

    # Annotation box removed per user request
    # Adjust layout to make room for legend without overlapping x-axis label
    plt.subplots_adjust(bottom=0.25)
    # Version stamp removed


    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {out_path}")


def plot_boxplot(data: dict, out_path: Path):
    """Plot 2: Box-and-whisker with quartile boundaries."""
    enrollments = data['enrollments']
    is_highlight = data['is_highlight']
    q1, q2, q3 = data['quartiles']
    x_max = data['x_max']

    fig, ax = plt.subplots(figsize=(11, 6))

    # Create box plot (excluding Springfield for box calculation)
    outlier_threshold = data['outlier_threshold']
    enrollments_box = enrollments[enrollments <= outlier_threshold]
    bp = ax.boxplot([enrollments_box], vert=False, widths=0.5, patch_artist=True,
                     boxprops=dict(facecolor='lightblue', edgecolor='black', linewidth=2),
                     whiskerprops=dict(linewidth=2),
                     capprops=dict(linewidth=2),
                     medianprops=dict(color='red', linewidth=3))

    # Overlay individual points
    y_pos = np.ones(len(enrollments)) + np.random.normal(0, 0.02, len(enrollments))
    for i in range(len(enrollments)):
        if enrollments[i] > outlier_threshold:
            continue  # Skip Springfield
        color = HIGHLIGHT_COLOR if is_highlight[i] else WESTERN_COLOR
        size = 150 if is_highlight[i] else 80
        alpha = 0.8 if is_highlight[i] else 0.6
        ax.scatter(enrollments[i], y_pos[i], c=color, s=size, alpha=alpha,
                   zorder=3, edgecolors='white', linewidth=1.5)

    # Add quartile labels
    ax.text(q1, 0.5, f'Q1: {q1:.0f}', fontsize=12, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(q2, 0.5, f'Q2 (Median): {q2:.0f}', fontsize=12, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    ax.text(q3, 0.5, f'Q3: {q3:.0f}', fontsize=12, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Set x-axis limit
    ax.set_xlim(0, x_max)

    ax.set_xlabel('In-District FTE Enrollment', fontsize=14)
    ax.set_yticks([])
    # No title - page subtitle does the job
    ax.grid(True, alpha=0.3, axis='x')

    # Remove top, left, right borders - keep only bottom
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add explanatory text
    explanation = ('The box shows the middle 50% of districts (Q1 to Q3). The red line is the median.\n'
                   'Whiskers extend to show the full range (excluding extreme outliers like Springfield).')
    fig.text(0.5, 0.02, explanation, ha='center', fontsize=9, style='italic',
             wrap=True, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    # Version stamp removed


    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {out_path}")


def plot_histogram(data: dict, out_path: Path):
    """Plot 3: Distribution histogram with smooth distribution curve."""
    enrollments = data['enrollments']
    median_val = data['median']
    q1, q2, q3 = data['quartiles']
    x_max = data['x_max']
    springfield_name = data['springfield_name']
    springfield_value = data['springfield_value']

    fig, ax = plt.subplots(figsize=(11, 6))

    # Create histogram with 250 FTE bins (excluding Springfield)
    outlier_threshold = data['outlier_threshold']
    hist_data = enrollments[enrollments <= outlier_threshold]
    bins = np.arange(0, x_max + 250, 250)
    n, bin_edges, patches = ax.hist(hist_data, bins=bins, color='lightblue',
                                     edgecolor='black', alpha=0.7, label='District Count')

    # Add vertical lines for quartiles and percentiles
    ax.axvline(median_val, color=MEDIAN_COLOR, linestyle='--', linewidth=2.5,
               label=f'Median: {median_val:.0f}', zorder=6)
    ax.axvline(q1, color='purple', linestyle=':', linewidth=2,
               label=f'Q1: {q1:.0f}', zorder=4)
    ax.axvline(q3, color='purple', linestyle=':', linewidth=2,
               label=f'Q3: {q3:.0f}', zorder=4)

    # Note: P90 line removed as Large cohort now extends to 10K FTE

    # Add IQR shading
    ax.axvspan(q1, q3, alpha=0.15, color='purple', label=f'IQR: {q3-q1:.0f}')

    # Fix y-axis: show 1-7, then skip to max
    max_count = int(np.max(n))
    if max_count > 7:
        y_ticks = list(range(1, 8)) + [max_count]
    else:
        y_ticks = list(range(1, max_count + 1))
    ax.set_yticks(y_ticks)
    ax.set_ylim(0, max_count * 1.1)

    ax.set_xlim(0, x_max)
    ax.set_xlabel('In-District FTE Enrollment (250 FTE bins)', fontsize=14)
    ax.set_ylabel('Frequency (Number of Districts)', fontsize=14)
    # No title - page subtitle does the job
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Remove top, left, right borders - keep only bottom
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Annotation box removed per user request
    plt.tight_layout()
    # Version stamp removed


    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {out_path}")


def plot_grouping(data: dict, out_path: Path):
    """Plot 4: Proposed 5-group enrollment bands."""
    enrollments = data['enrollments']
    springfield_name = data['springfield_name']
    springfield_value = data['springfield_value']
    year_cohort_defs = data.get('year_cohort_defs', {})

    fig, ax = plt.subplots(figsize=(11, 6))

    # Use year-specific cohort boundaries if available, otherwise fall back to global
    if year_cohort_defs:
        tiny_range = year_cohort_defs['TINY']['range']
        small_range = year_cohort_defs['SMALL']['range']
        medium_range = year_cohort_defs['MEDIUM']['range']
        large_range = year_cohort_defs['LARGE']['range']
    else:
        from school_shared import COHORT_DEFINITIONS
        tiny_range = COHORT_DEFINITIONS['TINY']['range']
        small_range = COHORT_DEFINITIONS['SMALL']['range']
        medium_range = COHORT_DEFINITIONS['MEDIUM']['range']
        large_range = COHORT_DEFINITIONS['LARGE']['range']

    # Create 4 enrollment groups using dynamic boundaries (Springfield excluded as outlier)
    proposed_thresholds = [tiny_range[0], small_range[0], medium_range[0], large_range[0], large_range[1] + 1]
    proposed_groups = []
    proposed_labels = [
        f"{tiny_range[0]}-{tiny_range[1]}\n(Tiny)",
        f"{small_range[0]}-{small_range[1]}\n(Small)",
        f"{medium_range[0]}-{medium_range[1]}\n(Medium)",
        f"{large_range[0]}-{large_range[1]:,}\n(Large)"
    ]

    # Count districts in each group
    for i in range(len(proposed_thresholds) - 1):
        count = sum((enrollments >= proposed_thresholds[i]) & (enrollments < proposed_thresholds[i+1]))
        proposed_groups.append(count)

    # Add Springfield as separate group
    if springfield_name and springfield_value:
        proposed_labels.append(f'{springfield_name}\n({springfield_value:.0f} FTE)')
        proposed_groups.append(1)

    # Create bar chart with narrower bars (50% narrower: 0.5 height)
    colors_list = ['#4575B4', '#3C9DC4', '#FDB749', '#F46D43', '#D73027', '#A50026']  # Blue to Red gradient (saturated)
    bars = ax.barh(proposed_labels, proposed_groups, height=0.5, color=colors_list[:len(proposed_groups)],
                   edgecolor='black', alpha=0.7, linewidth=1.5)

    # Add count labels and percentages
    total_districts = len(enrollments)
    for i, (bar, count) in enumerate(zip(bars, proposed_groups)):
        if count > 0:
            pct = 100 * count / total_districts
            ax.text(count + 0.5, i, f'{count} ({pct:.1f}%)', va='center',
                    fontsize=12, fontweight='bold')

    ax.set_xlabel('Number of Districts', fontsize=14)
    ax.set_ylabel('Enrollment Group', fontsize=14)
    # No title - page subtitle does the job
    ax.grid(True, alpha=0.3, axis='x')

    # Remove top, left, right borders - keep only bottom
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # No explanatory text annotation - removed per user request
    plt.tight_layout()
    # Version stamp removed


    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {out_path}")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate enrollment distribution plots")
    parser.add_argument("--force-recompute", action="store_true",
                        help="Bypass cache and recompute from source")
    args = parser.parse_args()

    print("="*70)
    print("Western MA Enrollment Distribution - Individual Plots")
    print("="*70)

    # Load data
    print("\n[1/5] Loading data...")
    df, reg, c70 = load_data(force_recompute=args.force_recompute)
    latest_year = int(df["YEAR"].max())
    print(f"  Latest year: {latest_year}")

    # Years to generate plots for (in reverse order: 2024, 2019, 2014, 2009)
    years = [2024, 2019, 2014, 2009]

    for year in years:
        print(f"\n{'='*70}")
        print(f"Generating plots for FY {year}")
        print(f"{'='*70}")

        # Analyze enrollment for this year
        print(f"\n[1/3] Analyzing enrollment distribution for {year}...")
        data = analyze_western_enrollment(df, reg, year)
        print(f"  Total districts: {len(data['enrollments'])}")
        print(f"  Median enrollment: {data['median']:.0f} FTE")

        # Create scatterplot for this year
        print(f"\n[2/3] Creating scatterplot for {year}...")
        plot_scatterplot(data, OUTPUT_DIR / f"enrollment_1_scatterplot_{year}.png")

        # Histogram and grouping only for latest year
        if year == latest_year:
            print("\n[3/3] Creating histogram and grouping (latest year only)...")
            plot_histogram(data, OUTPUT_DIR / "enrollment_3_histogram.png")
            plot_grouping(data, OUTPUT_DIR / "enrollment_4_grouping.png")

    print("\n" + "="*70)
    print("Individual plots created successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
