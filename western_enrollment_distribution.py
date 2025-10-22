"""
Western MA Enrollment Distribution Analysis

Creates visualizations to understand enrollment patterns across Western MA traditional districts:
1. Scatterplot with distribution curve, mean/median lines
2. Box-and-whisker plot showing quartile boundaries
3. Standard deviation visualization
4. Highlights districts of interest (Amherst-Pelham, Amherst, Leverett, Pelham, Shutesbury)
5. Includes ALPS PK-12 aggregate for comparison
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import stats
from scipy.interpolate import make_interp_spline

from school_shared import (
    OUTPUT_DIR, load_data, DISTRICTS_OF_INTEREST,
    latest_total_fte
)

# Styling constants
HIGHLIGHT_COLOR = "#FF8C00"  # Dark orange for districts of interest
ALPS_COLOR = "#8B4513"  # Saddle brown for ALPS aggregate
WESTERN_COLOR = "#4682B4"  # Steel blue for other Western MA districts
MEAN_COLOR = "#DC143C"  # Crimson for mean line
MEDIAN_COLOR = "#228B22"  # Forest green for median line
CURVE_COLOR = "#708090"  # Slate gray for distribution curve

CODE_VERSION = "v2025.10.04-ENROLLMENT-DIST"


def analyze_western_enrollment(df: pd.DataFrame, reg: pd.DataFrame, latest_year: int):
    """
    Analyze enrollment distribution for Western MA traditional districts.

    Returns:
        dict with keys:
        - districts: list of district names
        - enrollments: list of enrollment values
        - is_highlight: list of booleans (True if district of interest)
        - is_alps: list of booleans (True if ALPS aggregate)
        - mean: mean enrollment
        - median: median enrollment
        - std: standard deviation
        - quartiles: [Q1, Q2, Q3]
    """
    from school_shared import get_western_cohort_districts

    # Get Western MA traditional districts using centralized cohort function
    # This ensures consistent filtering (includes PPE validation)
    cohorts = get_western_cohort_districts(df, reg)
    western_districts = []
    for cohort_key in ["TINY", "SMALL", "MEDIUM", "LARGE", "X-LARGE", "SPRINGFIELD"]:
        western_districts.extend(cohorts.get(cohort_key, []))

    # Get enrollment for each district
    district_data = []
    for dist in western_districts:
        enrollment = latest_total_fte(df, dist)
        if enrollment > 0:
            district_data.append({
                'name': dist,
                'enrollment': enrollment,
                'is_highlight': dist in [d.lower() for d in DISTRICTS_OF_INTEREST],
                'is_alps': False
            })

    # ALPS PK-12 aggregate removed - individual districts now used for peer comparisons
    # alps_enrollment = 0
    # for component in ALPS_COMPONENTS:
    #     alps_enrollment += latest_total_fte(df, component)
    #
    # if alps_enrollment > 0:
    #     district_data.append({
    #         'name': 'ALPS PK-12',
    #         'enrollment': alps_enrollment,
    #         'is_highlight': False,
    #         'is_alps': True
    #     })

    # Sort by enrollment
    district_data.sort(key=lambda x: x['enrollment'])

    # Extract lists
    districts = [d['name'] for d in district_data]
    enrollments = np.array([d['enrollment'] for d in district_data])
    is_highlight = [d['is_highlight'] for d in district_data]
    is_alps = [d['is_alps'] for d in district_data]

    # Calculate statistics
    mean_enr = np.mean(enrollments)
    median_enr = np.median(enrollments)
    std_enr = np.std(enrollments)
    q1, q2, q3 = np.percentile(enrollments, [25, 50, 75])

    return {
        'districts': districts,
        'enrollments': enrollments,
        'is_highlight': is_highlight,
        'is_alps': is_alps,
        'mean': mean_enr,
        'median': median_enr,
        'std': std_enr,
        'quartiles': [q1, q2, q3],
        'latest_year': latest_year
    }


def plot_enrollment_distribution(data: dict, out_path: Path):
    """Create comprehensive enrollment distribution visualization."""

    enrollments = data['enrollments']
    districts = data['districts']
    is_highlight = data['is_highlight']
    is_alps = data['is_alps']
    mean_val = data['mean']
    median_val = data['median']
    std_val = data['std']
    q1, q2, q3 = data['quartiles']
    latest_year = data['latest_year']

    # Identify outliers (districts > 8000 for labeling)
    outlier_threshold = 8000
    outlier_indices = [i for i, e in enumerate(enrollments) if e > outlier_threshold]
    outlier_names = [districts[i].title() for i in outlier_indices]
    outlier_values = [enrollments[i] for i in outlier_indices]

    # Set x-axis range to exclude extreme outliers from main visualization
    # Use penultimate district if there are outliers
    if outlier_indices:
        # Find the highest enrollment that's NOT an outlier
        non_outlier_enrollments = [e for e in enrollments if e <= outlier_threshold]
        if non_outlier_enrollments:
            x_max = max(non_outlier_enrollments) * 1.1  # Add 10% padding
        else:
            x_max = outlier_threshold
    else:
        x_max = enrollments.max() * 1.05

    # Create figure with 4 subplots
    fig = plt.figure(figsize=(16, 12))

    # --- Subplot 1: Scatterplot with distribution curve ---
    ax1 = plt.subplot(2, 2, 1)

    # Create x-axis positions (rank order)
    x_pos = np.arange(len(enrollments))

    # Plot points
    colors = []
    for i in range(len(enrollments)):
        if is_alps[i]:
            colors.append(ALPS_COLOR)
        elif is_highlight[i]:
            colors.append(HIGHLIGHT_COLOR)
        else:
            colors.append(WESTERN_COLOR)

    ax1.scatter(x_pos, enrollments, c=colors, s=100, alpha=0.7, zorder=3, edgecolors='white', linewidth=1.5)

    # Fit smooth curve through points
    if len(x_pos) > 3:
        x_smooth = np.linspace(x_pos.min(), x_pos.max(), 300)
        spl = make_interp_spline(x_pos, enrollments, k=3)
        y_smooth = spl(x_smooth)
        ax1.plot(x_smooth, y_smooth, color=CURVE_COLOR, linewidth=2, alpha=0.6, zorder=2, label='Distribution Curve')

    # Add mean and median lines
    ax1.axhline(mean_val, color=MEAN_COLOR, linestyle='--', linewidth=2, label=f'Mean: {mean_val:.0f}', zorder=4)
    ax1.axhline(median_val, color=MEDIAN_COLOR, linestyle='--', linewidth=2, label=f'Median: {median_val:.0f}', zorder=4)

    # Set y-axis limit to exclude outliers
    ax1.set_ylim(0, x_max)

    # Add annotations for outliers
    for name, value in zip(outlier_names, outlier_values):
        ax1.annotate(f'{name}\n({value:.0f} FTE)',
                    xy=(len(enrollments) - 1, x_max * 0.95),
                    xytext=(len(enrollments) - 5, x_max * 0.85),
                    fontsize=10, fontweight='bold', color='red',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))

    ax1.set_xlabel('District Rank (by enrollment)', fontsize=12)
    ax1.set_ylabel('Total FTE Enrollment', fontsize=12)
    ax1.set_title(f'Western MA Traditional Districts: Enrollment Distribution ({latest_year})',
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # --- Subplot 2: Box-and-Whisker Plot ---
    ax2 = plt.subplot(2, 2, 2)

    # Create box plot
    bp = ax2.boxplot([enrollments], vert=False, widths=0.5, patch_artist=True,
                      boxprops=dict(facecolor='lightblue', edgecolor='black', linewidth=2),
                      whiskerprops=dict(linewidth=2),
                      capprops=dict(linewidth=2),
                      medianprops=dict(color='red', linewidth=3))

    # Overlay individual points
    y_pos = np.ones(len(enrollments)) + np.random.normal(0, 0.02, len(enrollments))
    for i in range(len(enrollments)):
        if is_alps[i]:
            ax2.scatter(enrollments[i], y_pos[i], c=ALPS_COLOR, s=150, alpha=0.8,
                       zorder=3, edgecolors='white', linewidth=2, marker='D')
        elif is_highlight[i]:
            ax2.scatter(enrollments[i], y_pos[i], c=HIGHLIGHT_COLOR, s=150, alpha=0.8,
                       zorder=3, edgecolors='white', linewidth=2)
        else:
            ax2.scatter(enrollments[i], y_pos[i], c=WESTERN_COLOR, s=80, alpha=0.6,
                       zorder=3, edgecolors='white', linewidth=1)

    # Add quartile labels
    ax2.text(q1, 0.5, f'Q1: {q1:.0f}', fontsize=11, ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax2.text(q2, 0.5, f'Q2 (Median): {q2:.0f}', fontsize=11, ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    ax2.text(q3, 0.5, f'Q3: {q3:.0f}', fontsize=11, ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Set x-axis limit and add outlier annotation
    ax2.set_xlim(0, x_max)

    # Add outlier annotations
    for name, value in zip(outlier_names, outlier_values):
        ax2.annotate(f'{name} ({value:.0f})',
                    xy=(x_max * 0.95, 1.3),
                    fontsize=10, fontweight='bold', color='red',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    ax2.set_xlabel('Total FTE Enrollment', fontsize=12)
    ax2.set_yticks([])
    ax2.set_title('Box-and-Whisker: Quartile Boundaries', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')

    # --- Subplot 3: Distribution Histogram (250 FTE bins) ---
    ax3 = plt.subplot(2, 2, 3)

    # Create histogram with 250 FTE bins, excluding outliers
    hist_data = enrollments[enrollments <= outlier_threshold]
    bins = np.arange(0, x_max + 250, 250)  # 250 FTE bins starting at 0
    n, bin_edges, patches = ax3.hist(hist_data, bins=bins, color='lightblue', edgecolor='black', alpha=0.7)

    # Add vertical lines for quartiles and median
    ax3.axvline(median_val, color=MEDIAN_COLOR, linestyle='--', linewidth=2.5, label=f'Median: {median_val:.0f}', zorder=5)
    ax3.axvline(q1, color='purple', linestyle=':', linewidth=2, label=f'Q1: {q1:.0f}', zorder=4)
    ax3.axvline(q3, color='purple', linestyle=':', linewidth=2, label=f'Q3: {q3:.0f}', zorder=4)

    # Add IQR shading
    ax3.axvspan(q1, q3, alpha=0.15, color='purple', label=f'IQR: {q3-q1:.0f}')

    # Add outlier annotation
    if outlier_names:
        outlier_text = ', '.join([f'{n} ({v:.0f})' for n, v in zip(outlier_names, outlier_values)])
        ax3.text(0.98, 0.98, f'Outliers (>{outlier_threshold}):\n{outlier_text}',
                transform=ax3.transAxes, fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    ax3.set_xlim(0, x_max)
    ax3.set_xlabel('Total FTE Enrollment (250 FTE bins)', fontsize=12)
    ax3.set_ylabel('Frequency (Number of Districts)', fontsize=12)
    ax3.set_title('Distribution Shape & Quartile Boundaries', fontsize=14, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')

    # --- Subplot 4: Proposed Enrollment Groups Analysis ---
    ax4 = plt.subplot(2, 2, 4)

    # Find penultimate (highest non-outlier) for grouping
    non_outliers = [e for e in enrollments if e <= outlier_threshold]
    if non_outliers:
        penultimate = max(non_outliers)
    else:
        penultimate = outlier_threshold

    # Create proposed enrollment cohorts based on IQR analysis: 0-800, 801-1800, 1801-penultimate, outliers
    proposed_thresholds = [0, 800, 1800, penultimate + 1]
    proposed_groups = []
    proposed_labels = ['0-800\n(Small)', '801-1800\n(Medium)', f'1801-{int(penultimate)}\n(Large)']

    # Count districts in each group (excluding outliers from these groups)
    for i in range(len(proposed_thresholds) - 1):
        count = sum((enrollments >= proposed_thresholds[i]) & (enrollments < proposed_thresholds[i+1]))
        proposed_groups.append(count)

    # Add outliers as separate group
    outlier_count = len(outlier_indices)
    if outlier_count > 0:
        proposed_labels.append(f'Outliers\n(>{outlier_threshold})')
        proposed_groups.append(outlier_count)

    # Create bar chart
    colors_list = ['lightgreen', 'lightblue', 'orange', 'red'] if outlier_count > 0 else ['lightgreen', 'lightblue', 'orange']
    bars = ax4.barh(proposed_labels, proposed_groups, color=colors_list[:len(proposed_groups)],
                    edgecolor='black', alpha=0.7, linewidth=1.5)

    # Add count labels and percentages
    total_districts = len(enrollments)
    for i, (bar, count) in enumerate(zip(bars, proposed_groups)):
        if count > 0:
            pct = 100 * count / total_districts
            ax4.text(count + 0.5, i, f'{count} ({pct:.1f}%)', va='center', fontsize=11, fontweight='bold')

    # Add outlier names annotation if they exist
    if outlier_names:
        outlier_label = ', '.join(outlier_names)
        ax4.text(0.98, 0.02, f'Outlier districts: {outlier_label}',
                transform=ax4.transAxes, fontsize=9, verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                style='italic')

    ax4.set_xlabel('Number of Districts', fontsize=12)
    ax4.set_ylabel('Proposed Enrollment Groups', fontsize=12)
    ax4.set_title('Proposed Grouping: 0-500 / 500-1500 / 1500+ / Outliers', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')

    # Overall figure adjustments
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])

    # Add version stamp
    fig.text(0.99, 0.01, f"Code: {CODE_VERSION}", ha="right", va="bottom",
             fontsize=9, color="#666666")

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {out_path}")


def print_summary_statistics(data: dict):
    """Print summary statistics to console."""
    enrollments = data['enrollments']
    districts = data['districts']
    is_highlight = data['is_highlight']
    is_alps = data['is_alps']
    mean_val = data['mean']
    median_val = data['median']
    std_val = data['std']
    q1, q2, q3 = data['quartiles']

    print("\n" + "="*70)
    print("WESTERN MA ENROLLMENT DISTRIBUTION ANALYSIS")
    print("="*70)

    print(f"\nTotal Districts: {len(enrollments)}")
    print(f"Year: {data['latest_year']}")

    print("\n--- Central Tendency ---")
    print(f"Mean:   {mean_val:8.1f} FTE")
    print(f"Median: {median_val:8.1f} FTE")

    print("\n--- Spread (Appropriate for Bounded Data) ---")
    print(f"Min:       {enrollments.min():8.1f} FTE")
    print(f"Max:       {enrollments.max():8.1f} FTE")
    print(f"Range:     {enrollments.max() - enrollments.min():8.1f} FTE")
    print(f"IQR (Q3 - Q1):         {q3 - q1:8.1f} FTE  (better measure for skewed data)")

    print("\n--- Quartiles (Primary Spread Metric) ---")
    print(f"Q1 (25th percentile):  {q1:8.1f} FTE")
    print(f"Q2 (50th percentile):  {q2:8.1f} FTE")
    print(f"Q3 (75th percentile):  {q3:8.1f} FTE")

    print("\n--- Note on Statistical Measures ---")
    print("For bounded, right-skewed data like enrollment:")
    print("  • Median and IQR are more appropriate than mean and SD")
    print("  • Standard deviation assumes normal distribution (not valid here)")
    print("  • Use quartiles and percentiles to describe spread")

    print("\n--- Current Grouping (N_THRESHOLD = 500) ---")
    le_500 = sum(enrollments <= 500)
    gt_500 = sum(enrollments > 500)
    print(f"<=500 FTE:  {le_500:3d} districts ({100*le_500/len(enrollments):.1f}%)")
    print(f">500 FTE:   {gt_500:3d} districts ({100*gt_500/len(enrollments):.1f}%)")

    print("\n--- Proposed Grouping (Based on Round Numbers & MA Practice) ---")
    g1 = sum(enrollments <= 500)
    g2 = sum((enrollments > 500) & (enrollments <= 1500))
    # Find penultimate for upper bound
    outlier_threshold = 8000
    non_outliers = enrollments[enrollments <= outlier_threshold]
    penultimate = max(non_outliers) if len(non_outliers) > 0 else outlier_threshold
    g3 = sum((enrollments > 1500) & (enrollments <= penultimate))
    g4 = sum(enrollments > outlier_threshold)

    print(f"0-500 (Small):      {g1:3d} districts ({100*g1/len(enrollments):.1f}%)")
    print(f"500-1500 (Medium):  {g2:3d} districts ({100*g2/len(enrollments):.1f}%)")
    print(f"1500-{int(penultimate)} (Large):     {g3:3d} districts ({100*g3/len(enrollments):.1f}%)")
    if g4 > 0:
        print(f"Outliers (>{outlier_threshold}):    {g4:3d} districts ({100*g4/len(enrollments):.1f}%)")

    print("\n--- Districts of Interest ---")
    for i, dist in enumerate(districts):
        if is_highlight[i]:
            print(f"{dist.title():25s}: {enrollments[i]:6.0f} FTE")

    print("\n--- ALPS PK-12 Aggregate ---")
    for i, dist in enumerate(districts):
        if is_alps[i]:
            print(f"{dist:25s}: {enrollments[i]:6.0f} FTE")
            percentile = (sum(enrollments < enrollments[i]) / len(enrollments)) * 100
            print(f"  (At {percentile:.1f}th percentile)")

    print("\n" + "="*70)


def main():
    print("="*70)
    print("Western MA Enrollment Distribution Analysis")
    print("="*70)

    # Load data
    print("\n[1/3] Loading data...")
    df, reg, c70 = load_data()
    latest_year = int(df["YEAR"].max())
    print(f"  Latest year: {latest_year}")

    # Analyze enrollment
    print("\n[2/3] Analyzing enrollment distribution...")
    data = analyze_western_enrollment(df, reg, latest_year)

    # Print statistics
    print_summary_statistics(data)

    # Create visualization
    print("\n[3/3] Creating visualization...")
    out_path = OUTPUT_DIR / "western_enrollment_distribution.png"
    plot_enrollment_distribution(data, out_path)

    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)


if __name__ == "__main__":
    main()
