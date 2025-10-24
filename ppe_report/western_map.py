"""
Generate choropleth maps of Western Massachusetts school districts for PDF report.

This module:
1. Loads district types from Ch 70 District Profiles (MA_District_Profiles sheet)
2. Loads MA school district shapefiles (Census Bureau TIGER/Line)
3. Matches Western MA traditional districts to geometries
4. Color codes districts by enrollment cohort
5. Adds cohort indicators to regional districts (U for unified, T/S/M/L for secondary)
6. Generates static PNG map for PDF inclusion

District Type Classification (from data file):
- "District" -> Elementary district (not regional)
- "Unified Regional" -> Serves all grades PK-12 across multiple towns (gets "U" marker)
- "Regional Composite" -> Secondary regional that overlaps elementary districts (gets cohort letter + black border)

Map Design:
- 5 enrollment cohorts (Tiny, Small, Medium, Large, Springfield) each with unique color
- Non-regional districts: solid filled polygons (85% opacity)
- Unified regional districts: solid filled polygons (85% opacity) with white "U" marker
- Secondary regional districts: thick black border + cohort letter indicator (T, S, M, L)
  * Letter indicates the cohort of the secondary regional district
  * T = Tiny, S = Small, M = Medium, L = Large
  * Similar styling to "U" marker (black outline + white text)
- Legend showing cohort definitions, district counts, and regional markers
- Handles geographic overlap between elementary and secondary regional districts

Extensibility:
- District types loaded from data file (not hardcoded)
- Can be extended to other regions by updating data file
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as patheffects
from matplotlib.colors import ListedColormap
import pandas as pd

from school_shared import (
    load_data,
    OUTPUT_DIR,
    DISTRICTS_OF_INTEREST,
    get_western_cohort_districts,
    get_western_cohort_districts_for_year,
    get_total_fte_for_year,
    get_enrollment_group,
    get_cohort_label,
    get_cohort_short_label,
    initialize_cohort_definitions,
    EXCLUDE_DISTRICTS,
    prepare_district_epp_lines,
    weighted_epp_aggregation,
)

# Shapefile paths
SHAPEFILE_DIR = Path("./data/shapefiles")
UNIFIED_SHP = SHAPEFILE_DIR / "tl_2023_25_unsd.shp"
ELEMENTARY_SHP = SHAPEFILE_DIR / "tl_2023_25_elsd.shp"
SECONDARY_SHP = SHAPEFILE_DIR / "tl_2023_25_scsd.shp"

# Color palette for cohorts (colorblind-friendly, more saturated)
COHORT_COLORS = {
    "TINY": "#4575B4",      # Blue (low enrollment)
    "SMALL": "#3C9DC4",     # Cyan (more saturated)
    "MEDIUM": "#FDB749",    # Amber (more saturated)
    "LARGE": "#D73027",     # Red (now includes former X-Large)
    "SPRINGFIELD": "#A50026",  # Dark Red (outliers)
}

# Highlight color for districts of interest
HIGHLIGHT_COLOR = "#FF8C00"  # Dark orange outline
EXCLUDED_COLOR = "#EEEEEE"   # Light gray for excluded districts

# District profiles data file
DISTRICT_PROFILES_FILE = Path("./data/Ch 70 District Profiles Actual NSS Over Required.xlsx")

# Version stamp
CODE_VERSION = "v2025.10.11-MAPS-COHORT-LETTERS"


def clean_district_name(name: str) -> str:
    """
    Clean district name for matching with shapefiles.

    Shapefiles use different naming conventions than our data, so we need to normalize:
    - Remove suffixes like "School District", "Public Schools", "SD"
    - Remove location details like " in Savoy (7-12)"
    - Standardize separators (hyphens, spaces)
    - Convert to lowercase for case-insensitive matching
    """
    cleaned = str(name).lower()

    # Remove grade level indicators like (7-12), (9-12), etc.
    import re
    cleaned = re.sub(r'\s*\(\d+-\d+\)', '', cleaned)

    # Remove location details like " in Savoy", " in Hancock", etc.
    cleaned = re.sub(r'\s+in\s+[a-z\s]+$', '', cleaned)
    cleaned = re.sub(r'\s+in\s+[a-z\s]+\s+town[s]?', '', cleaned)

    # Remove common suffixes
    cleaned = (
        cleaned
        .replace("school district", "")
        .replace("public schools", "")
        .replace(" sd", "")
        .replace(",", "")
        .replace(".", "")
    )

    # Normalize separators - keep hyphens for now as they're meaningful
    cleaned = cleaned.strip()

    return cleaned


def load_district_types() -> Dict[str, str]:
    """
    Load district types from MA_District_Profiles sheet.

    Maps district names to their type classification:
    - "District" -> "elementary" (elementary district, not regional)
    - "Unified Regional" -> "unified_regional" (serves all grades PK-12 across multiple towns)
    - "Regional Composite" -> "regional_composite" (overlaps with multiple elementary districts)

    Returns:
        Dictionary mapping cleaned district name -> district type
    """
    district_types = {}

    try:
        if not DISTRICT_PROFILES_FILE.exists():
            print(f"[WARN] District profiles file not found: {DISTRICT_PROFILES_FILE}")
            return district_types

        profiles = pd.read_excel(DISTRICT_PROFILES_FILE, sheet_name="MA_District_Profiles")

        for idx, row in profiles.iterrows():
            dist_name = str(row.get("DistOrg", "")).strip()
            dist_type = str(row.get("DistType", "")).strip()

            if not dist_name or not dist_type:
                continue

            # Clean the district name for matching
            clean_name = clean_district_name(dist_name)

            # Map DistType to our internal classification
            if dist_type == "District":
                district_types[clean_name] = "elementary"
            elif dist_type == "Unified Regional":
                district_types[clean_name] = "unified_regional"
            elif dist_type == "Regional Composite":
                district_types[clean_name] = "regional_composite"
            # Ignore other types for now

        print(f"  Loaded district types for {len(district_types)} districts")
        unified_count = sum(1 for t in district_types.values() if t == "unified_regional")
        composite_count = sum(1 for t in district_types.values() if t == "regional_composite")
        print(f"    - Unified Regional: {unified_count}")
        print(f"    - Regional Composite: {composite_count}")

    except Exception as e:
        print(f"[WARN] Could not load district types: {e}")

    return district_types


def load_shapefiles() -> gpd.GeoDataFrame:
    """
    Load and combine all three types of MA school district shapefiles.

    Returns:
        Combined GeoDataFrame with unified, elementary, and secondary districts
    """
    print("Loading shapefiles...")

    # Load all three shapefile types
    unified = gpd.read_file(UNIFIED_SHP)
    elementary = gpd.read_file(ELEMENTARY_SHP)
    secondary = gpd.read_file(SECONDARY_SHP)

    # Add type identifier
    unified["district_type"] = "unified"
    elementary["district_type"] = "elementary"
    secondary["district_type"] = "secondary"

    # Standardize column names across all three types
    unified = unified.rename(columns={"UNSDLEA": "LEA", "NAME": "district_name"})
    elementary = elementary.rename(columns={"ELSDLEA": "LEA", "NAME": "district_name"})
    secondary = secondary.rename(columns={"SCSDLEA": "LEA", "NAME": "district_name"})

    # Combine all districts
    all_districts = pd.concat([unified, elementary, secondary], ignore_index=True)

    # Create cleaned name column for matching
    all_districts["clean_name"] = all_districts["district_name"].apply(clean_district_name)

    print(f"  Loaded {len(unified)} unified, {len(elementary)} elementary, {len(secondary)} secondary districts")
    print(f"  Total: {len(all_districts)} district geometries")

    return all_districts


def match_districts_to_geometries(
    df: pd.DataFrame,
    reg: pd.DataFrame,
    shapes: gpd.GeoDataFrame,
    year: int = None
) -> gpd.GeoDataFrame:
    """
    Match Western MA districts from our analysis to shapefile geometries.

    Args:
        df: Main expenditure data
        reg: District regions/metadata
        shapes: GeoDataFrame with district geometries
        year: Specific year to use for enrollment/cohorts (defaults to latest if None)

    Returns:
        GeoDataFrame with matched districts, cohort assignments, and geometries
    """
    print(f"\nMatching districts to geometries for year {year if year else 'latest'}...")

    # Load district types from data file
    district_types = load_district_types()

    # Get Western MA cohort districts for specified year
    # This now uses year-specific cohort boundaries
    if year is not None:
        cohorts = get_western_cohort_districts_for_year(df, reg, year)
    else:
        # For latest year, initialize global cohort definitions first
        from school_shared import initialize_cohort_definitions
        initialize_cohort_definitions(df, reg)
        cohorts = get_western_cohort_districts(df, reg)

    # Create mapping of district name -> cohort
    district_cohort_map = {}
    for cohort_name, districts in cohorts.items():
        for dist in districts:
            district_cohort_map[dist.lower()] = cohort_name

    # Create mapping of district name -> is_of_interest
    districts_of_interest_lower = [d.lower() for d in DISTRICTS_OF_INTEREST]

    # Match each district to geometry
    matched_districts = []
    unmatched = []

    for dist_name, cohort_name in district_cohort_map.items():
        clean_name = clean_district_name(dist_name)

        # Try exact match first
        matches = shapes[shapes["clean_name"] == clean_name]

        if len(matches) == 0:
            # Try fuzzy matching as fallback
            # Look for partial matches by checking if clean_name is in shapefile name or vice versa
            potential_matches = []
            for idx, row in shapes.iterrows():
                shape_clean = row["clean_name"]
                # Check if either name contains the other (for cases like "hoosac valley" matching "hoosac valley...")
                if clean_name in shape_clean or shape_clean in clean_name:
                    potential_matches.append(row)
                # Also check for key words match (for multi-word names)
                elif len(clean_name.split()) > 1:
                    clean_words = set(clean_name.split())
                    shape_words = set(shape_clean.split())
                    # If more than half of the words match, consider it
                    if len(clean_words & shape_words) >= max(2, len(clean_words) * 0.6):
                        potential_matches.append(row)

            if potential_matches:
                # Take the first potential match (could improve with scoring)
                geom_row = potential_matches[0]
                matches = gpd.GeoDataFrame([geom_row])

        if len(matches) > 0:
            # Take first match (usually only one)
            geom_row = matches.iloc[0] if isinstance(matches, gpd.GeoDataFrame) else matches

            # Determine district type from data file
            clean_name_check = clean_district_name(dist_name)
            data_dist_type = district_types.get(clean_name_check, "")

            # Determine if this is a regional district based on:
            # 1. District type from data file (unified_regional or regional_composite)
            # 2. Name contains "Reg" or "Regional" (fallback for legacy data)
            # 3. Shapefile type is "secondary" (fallback)
            is_regional = (
                data_dist_type in ("unified_regional", "regional_composite") or
                "reg" in dist_name.lower() or
                "regional" in dist_name.lower() or
                geom_row["district_type"] == "secondary"
            )

            # Determine regional subtype (unified vs secondary/composite)
            # If data file says "unified_regional", it's unified regardless of shapefile
            # If data file says "regional_composite", it's secondary/composite
            if data_dist_type == "unified_regional":
                regional_subtype = "unified"
            elif data_dist_type == "regional_composite":
                regional_subtype = "secondary"
            else:
                # Fall back to shapefile type for districts not in data file
                regional_subtype = geom_row["district_type"]

            matched_districts.append({
                "district_name": dist_name.title(),
                "cohort": cohort_name,
                "cohort_label": get_cohort_short_label(cohort_name),
                "is_of_interest": dist_name in districts_of_interest_lower,
                "is_regional": is_regional,
                "shapefile_type": geom_row["district_type"],
                "data_district_type": data_dist_type,  # Type from data file
                "regional_subtype": regional_subtype,  # unified vs secondary
                "geometry": geom_row["geometry"]
            })
        else:
            unmatched.append(dist_name)

    print(f"  Matched: {len(matched_districts)} districts")
    if unmatched:
        print(f"  Unmatched: {len(unmatched)} districts")
        for name in unmatched[:5]:  # Show first 5
            print(f"    - {name}")
        if len(unmatched) > 5:
            print(f"    ... and {len(unmatched) - 5} more")

    # Create GeoDataFrame
    matched_gdf = gpd.GeoDataFrame(matched_districts, geometry="geometry", crs=shapes.crs)

    # Convert to WGS84 (EPSG:4326) for standard lat/lon coordinates
    matched_gdf = matched_gdf.to_crs(epsg=4326)

    return matched_gdf


def create_western_ma_map(
    matched_gdf: gpd.GeoDataFrame,
    output_path: Path,
    year: int = 2024,
    title: str = "Western Massachusetts Traditional School Districts"
) -> None:
    """
    Generate choropleth map of Western MA districts color-coded by cohort.

    Uses different rendering for different district types:
    - Elementary/Non-regional districts: solid filled polygons (85% opacity)
    - Unified regional districts (PK-12): solid filled polygons (85% opacity) with "U" marker
    - Secondary regional districts: thick black border + cohort letter indicator
      * Letter indicates cohort: T (Tiny), S (Small), M (Medium), L (Large)
      * Similar styling to "U" marker (black outline + white text on top)
      * Black border shows the boundary of the secondary regional district

    Args:
        matched_gdf: GeoDataFrame with matched districts and cohort assignments
        output_path: Path to save PNG file
        title: Map title
    """
    print(f"\nGenerating map: {output_path.name}")

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))

    # Separate districts by type
    # Unified regional districts (serve all grades PK-12) should render as solid fills with "U" marker
    # Secondary regional districts (overlap with elementary) should render with stripes and black border
    non_regional = matched_gdf[~matched_gdf["is_regional"]]
    regional_secondary = matched_gdf[
        (matched_gdf["is_regional"]) &
        (matched_gdf["regional_subtype"] == "secondary")
    ]
    regional_unified = matched_gdf[
        (matched_gdf["is_regional"]) &
        (matched_gdf["regional_subtype"] == "unified")
    ]

    print(f"  Rendering {len(non_regional)} non-regional districts (solid fill)")
    print(f"  Rendering {len(regional_unified)} unified regional districts (solid fill)")
    print(f"  Rendering {len(regional_secondary)} secondary regional districts (diagonal stripes)")

    # LAYER 1: Plot non-regional districts with filled polygons
    for cohort_name, color in COHORT_COLORS.items():
        cohort_districts = non_regional[non_regional["cohort"] == cohort_name]
        if len(cohort_districts) > 0:
            cohort_districts.plot(
                ax=ax,
                color=color,
                edgecolor="white",
                linewidth=0.5,
                alpha=0.85,
                zorder=1
            )

    # Plot unified regional districts with filled polygons (no special border)
    for cohort_name, color in COHORT_COLORS.items():
        cohort_districts = regional_unified[regional_unified["cohort"] == cohort_name]
        if len(cohort_districts) > 0:
            cohort_districts.plot(
                ax=ax,
                color=color,
                edgecolor="white",
                linewidth=0.5,
                alpha=0.85,
                zorder=1
            )
            # Add "U" label to each unified regional district
            for idx, row in cohort_districts.iterrows():
                centroid = row.geometry.centroid
                # Black text only (no outline)
                ax.text(
                    centroid.x, centroid.y, 'U',
                    color='black',
                    fontsize=20,
                    fontweight='bold',
                    ha='center',
                    va='center',
                    family='sans-serif',
                    zorder=5
                )

    # LAYER 2: Plot secondary regional districts with black border and cohort letter indicator
    print(f"  Rendering {len(regional_secondary)} secondary regional districts (black border + cohort letter):")

    # Cohort letter mapping
    cohort_letters = {
        "TINY": "T",
        "SMALL": "S",
        "MEDIUM": "M",
        "LARGE": "L"
    }

    for idx, regional_row in regional_secondary.iterrows():
        regional_geom = regional_row.geometry
        regional_cohort = regional_row["cohort"]
        regional_name = regional_row["district_name"]

        # Add black border around secondary regional district boundary
        gpd.GeoDataFrame([{"geometry": regional_geom}], geometry="geometry", crs=regional_secondary.crs).plot(
            ax=ax,
            facecolor="none",
            edgecolor="black",
            linewidth=3.0,
            zorder=6  # Top-most layer to cover all other colors
        )

        # Add cohort letter indicator (T, S, M, L, XL) in center of district
        centroid = regional_geom.centroid
        cohort_letter = cohort_letters.get(regional_cohort, "?")

        # Black text only (no outline)
        ax.text(
            centroid.x, centroid.y, cohort_letter,
            color='black',
            fontsize=20,
            fontweight='bold',
            ha='center',
            va='center',
            family='sans-serif',
            zorder=7
        )

        print(f"    {regional_name}: Black border + '{cohort_letter}' marker")

    # Remove axes
    ax.set_axis_off()

    # Create legend
    legend_elements = []

    # Add cohort colors to legend with counts
    for cohort_name in ["TINY", "SMALL", "MEDIUM", "LARGE", "SPRINGFIELD"]:
        cohort_label = get_cohort_label(cohort_name)
        n_total = len(matched_gdf[matched_gdf["cohort"] == cohort_name])
        n_regional_secondary = len(regional_secondary[regional_secondary["cohort"] == cohort_name])
        n_regional_unified = len(regional_unified[regional_unified["cohort"] == cohort_name])
        n_non_regional = len(non_regional[non_regional["cohort"] == cohort_name])

        # Total solid-fill districts (non-regional + unified regional)
        n_solid = n_non_regional + n_regional_unified

        # Show filled box for cohort, including both non-regional and all regional districts
        if n_total > 0:
            legend_elements.append(
                mpatches.Patch(
                    facecolor=COHORT_COLORS[cohort_name],
                    edgecolor="white",
                    alpha=0.85,
                    label=f"{cohort_label}: {n_total} district(s)"
                )
            )

    # Add explanation for secondary regional black border and cohort letters
    if len(regional_secondary) > 0:
        legend_elements.append(
            mpatches.Patch(
                facecolor="none",
                edgecolor="none",
                label=""
            )
        )
        legend_elements.append(
            mpatches.Patch(
                facecolor="none",
                edgecolor="black",
                linewidth=3,
                label="Secondary regional (black border)"
            )
        )
        legend_elements.append(
            mpatches.Patch(
                facecolor="none",
                edgecolor="none",
                label="  Cohort: T, S, M, L, XL"
            )
        )

    # Add explanation for "U" label on unified regional districts
    if len(regional_unified) > 0:
        # Add "U" explanation (text only, no symbol)
        legend_elements.append(
            mpatches.Patch(
                facecolor="none",
                edgecolor="none",
                label="U = Unified regional district (PK-12)")
        )

    # Add legend to plot - positioned below the map to avoid overlap
    ax.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),  # Position below the map
        fontsize=14,
        frameon=True,
        fancybox=True,
        shadow=True,
        title=f"Enrollment Cohorts - {year}",
        title_fontsize=16,  # Bigger title font
        ncol=2  # Use 2 columns to make it more compact
    )

    # Adjust layout
    plt.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"  Saved: {output_path}")
    print(f"  Resolution: 300 DPI")
    print(f"  Format: PNG with white background")


def create_ppe_comparison_map(
    matched_gdf: gpd.GeoDataFrame,
    ppe_comparisons: Dict[str, float],
    output_path: Path,
    year: int = 2024,
    title: str = "PPE vs Cohort Baseline"
) -> None:
    """
    Generate choropleth map showing district PPE compared to cohort baseline.

    Districts are colored based on percentage deviation from their cohort's weighted average:
    - Blue (#80DEEA): >5% below cohort baseline (below baseline)
    - White: Within ±5% of baseline (similar to cohort)
    - Orange (#FFD9A8): >5% above cohort baseline (above baseline)

    Matches the blue/orange color scheme used in comparison tables (CR 1003).

    Args:
        matched_gdf: GeoDataFrame with matched districts and cohort assignments
        ppe_comparisons: Dictionary mapping district name -> percentage deviation
        output_path: Path to save PNG file
        year: Year of analysis
        title: Map title
    """
    print(f"\nGenerating PPE comparison map: {output_path.name}")

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))

    # Add comparison values to matched_gdf (case-insensitive matching)
    ppe_comparisons_lower = {k.lower(): v for k, v in ppe_comparisons.items()}
    matched_gdf["ppe_deviation"] = matched_gdf["district_name"].str.lower().map(ppe_comparisons_lower)

    # Define color bins based on percentage deviation (CR 1003: 5% threshold)
    # Blue = below baseline, White = within threshold, Orange = above baseline
    bins = [-100, -5, 5, 100]  # Extended range to handle outliers
    # More muted colors per user request
    colors = ["#B3E0E6", "#FFFFFF", "#FFE6CC"]  # Muted blue (below), White (within ±5%), Muted orange (above)
    labels = [">5% below", "Within ±5%", ">5% above"]

    # Categorize districts by deviation
    matched_gdf["color_category"] = pd.cut(
        matched_gdf["ppe_deviation"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    # Separate secondary regional districts from others
    # Secondary regionals should only have black borders and text labels (no fill)
    regional_secondary = matched_gdf[
        (matched_gdf["is_regional"]) &
        (matched_gdf["regional_subtype"] == "secondary") &
        (~matched_gdf["ppe_deviation"].isna())
    ]

    # Non-secondary districts (get color fills)
    non_secondary = matched_gdf[
        ~((matched_gdf["is_regional"]) & (matched_gdf["regional_subtype"] == "secondary"))
    ]

    # Plot non-secondary districts by color category with gray boundaries
    for idx, (label, color) in enumerate(zip(labels, colors)):
        category_districts = non_secondary[non_secondary["color_category"] == label]
        if len(category_districts) > 0:
            # Use gray edges for all districts to make boundaries visible
            category_districts.plot(
                ax=ax,
                color=color,
                edgecolor="gray",
                linewidth=0.8,
                alpha=0.85,
                zorder=1
            )

    # Plot secondary regional districts with black borders and text labels (no fill)
    for idx, row in regional_secondary.iterrows():
        # Draw black border (no fill - transparent to show underlying elementary districts)
        gpd.GeoDataFrame([{"geometry": row.geometry}], geometry="geometry", crs=regional_secondary.crs).plot(
            ax=ax,
            facecolor="none",  # Transparent - shows underlying elementary district colors
            edgecolor="black",
            linewidth=1.5,
            zorder=2
        )

        # Add text label showing deviation
        deviation = row["ppe_deviation"]
        centroid = row.geometry.centroid

        # Format as +X% or -X% with sign
        sign = "+" if deviation > 0 else ""
        label_text = f'{sign}{deviation:.0f}%'

        # Use black text (more readable on muted colors)
        ax.text(
            centroid.x, centroid.y, label_text,
            fontsize=11, ha='center', va='center',
            color='black', weight='bold',
            zorder=10
        )

    # Remove axes
    ax.set_axis_off()

    # Create legend
    legend_elements = []
    for label, color in zip(labels, colors):
        n_districts = len(matched_gdf[matched_gdf["color_category"] == label])
        if n_districts > 0:
            # Use gray edges for all legend entries (consistent with map)
            legend_elements.append(
                mpatches.Patch(
                    facecolor=color,
                    edgecolor="gray",
                    linewidth=0.8,
                    alpha=0.85,
                    label=f"{label} cohort avg: {n_districts} district(s)"
                )
            )

    # Add legend entry for secondary regional indicators
    if len(regional_secondary) > 0:
        legend_elements.append(
            mpatches.Patch(
                facecolor="none",
                edgecolor="none",
                label=f"+/-% = Secondary regional district deviation (n={len(regional_secondary)})"
            )
        )

    # Add legend
    ax.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        fontsize=14,
        frameon=True,
        fancybox=True,
        shadow=True,
        title=f"Total PPE vs Cohort Baseline - {year} (±5% threshold)",
        title_fontsize=16,
        ncol=2  # Changed from 3 to 2 to accommodate additional legend entry
    )

    # Adjust layout
    plt.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"  Saved: {output_path}")
    print(f"  Districts mapped: {len(matched_gdf[~matched_gdf['ppe_deviation'].isna()])}")


def create_cagr_comparison_map(
    matched_gdf: gpd.GeoDataFrame,
    cagr_comparisons: Dict[str, float],
    output_path: Path,
    start_year: int = 2009,
    end_year: int = 2024,
    title: str = "CAGR vs Cohort Baseline"
) -> None:
    """
    Generate choropleth map showing district CAGR compared to cohort baseline.

    Districts are colored based on percentage point difference from their cohort's CAGR:
    - Blue (#80DEEA): >1.0pp below cohort baseline (below baseline growth)
    - White: Within ±1.0pp of baseline (similar growth to cohort)
    - Orange (#FFD9A8): >1.0pp above cohort baseline (above baseline growth)

    Matches the blue/orange color scheme used in comparison tables (CR 1003).

    Args:
        matched_gdf: GeoDataFrame with matched districts and cohort assignments
        cagr_comparisons: Dictionary mapping district name -> percentage point difference
        output_path: Path to save PNG file
        start_year: Period start year
        end_year: Period end year
        title: Map title
    """
    print(f"\nGenerating CAGR comparison map: {output_path.name}")

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))

    # Add comparison values to matched_gdf (case-insensitive matching)
    cagr_comparisons_lower = {k.lower(): v for k, v in cagr_comparisons.items()}
    matched_gdf["cagr_deviation"] = matched_gdf["district_name"].str.lower().map(cagr_comparisons_lower)

    # Define color bins based on percentage point deviation (CR 1003: 1.0pp threshold)
    # Blue = below baseline, White = within threshold, Orange = above baseline
    bins = [-100, -1, 1, 100]  # Extended range to handle outliers
    # More muted colors per user request
    colors = ["#B3E0E6", "#FFFFFF", "#FFE6CC"]  # Muted blue (below), White (within ±1pp), Muted orange (above)
    labels = [">1.0pp slower", "Within ±1.0pp", ">1.0pp faster"]

    # Categorize districts by deviation
    matched_gdf["color_category"] = pd.cut(
        matched_gdf["cagr_deviation"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    # Separate secondary regional districts from others
    # Secondary regionals should only have black borders and text labels (no fill)
    regional_secondary = matched_gdf[
        (matched_gdf["is_regional"]) &
        (matched_gdf["regional_subtype"] == "secondary") &
        (~matched_gdf["cagr_deviation"].isna())
    ]

    # Non-secondary districts (get color fills)
    non_secondary = matched_gdf[
        ~((matched_gdf["is_regional"]) & (matched_gdf["regional_subtype"] == "secondary"))
    ]

    # Plot non-secondary districts by color category with gray boundaries
    for idx, (label, color) in enumerate(zip(labels, colors)):
        category_districts = non_secondary[non_secondary["color_category"] == label]
        if len(category_districts) > 0:
            # Use gray edges for all districts to make boundaries visible
            category_districts.plot(
                ax=ax,
                color=color,
                edgecolor="gray",
                linewidth=0.8,
                alpha=0.85,
                zorder=1
            )

    # Plot secondary regional districts with black borders and text labels (no fill)
    for idx, row in regional_secondary.iterrows():
        # Draw black border (no fill - transparent to show underlying elementary districts)
        gpd.GeoDataFrame([{"geometry": row.geometry}], geometry="geometry", crs=regional_secondary.crs).plot(
            ax=ax,
            facecolor="none",  # Transparent - shows underlying elementary district colors
            edgecolor="black",
            linewidth=1.5,
            zorder=2
        )

        # Add text label showing deviation
        deviation = row["cagr_deviation"]
        centroid = row.geometry.centroid

        # Format as +X.Xpp or -X.Xpp with sign
        sign = "+" if deviation > 0 else ""
        label_text = f'{sign}{deviation:.1f}pp'

        # Use black text (more readable on muted colors)
        ax.text(
            centroid.x, centroid.y, label_text,
            fontsize=11, ha='center', va='center',
            color='black', weight='bold',
            zorder=10
        )

    # Remove axes
    ax.set_axis_off()

    # Create legend
    legend_elements = []
    for label, color in zip(labels, colors):
        n_districts = len(matched_gdf[matched_gdf["color_category"] == label])
        if n_districts > 0:
            # Use gray edges for all legend entries (consistent with map)
            legend_elements.append(
                mpatches.Patch(
                    facecolor=color,
                    edgecolor="gray",
                    linewidth=0.8,
                    alpha=0.85,
                    label=f"{label} than cohort: {n_districts} district(s)"
                )
            )

    # Add legend entry for secondary regional indicators
    if len(regional_secondary) > 0:
        legend_elements.append(
            mpatches.Patch(
                facecolor="none",
                edgecolor="none",
                label=f"+/-pp = Secondary regional district deviation (n={len(regional_secondary)})"
            )
        )

    # Add legend
    ax.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        fontsize=14,
        frameon=True,
        fancybox=True,
        shadow=True,
        title=f"Total PPE CAGR vs Cohort Baseline - {start_year} to {end_year} (±1pp threshold)",
        title_fontsize=16,
        ncol=2
    )

    # Adjust layout
    plt.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"  Saved: {output_path}")
    print(f"  Districts mapped: {len(matched_gdf[~matched_gdf['cagr_deviation'].isna()])}")


def calculate_ppe_comparison_to_cohort(
    df: pd.DataFrame,
    reg: pd.DataFrame,
    year: int
) -> Dict[str, float]:
    """
    Calculate PPE percentage deviation from cohort baseline for each district.

    Args:
        df: Main expenditure data
        reg: District regions/metadata
        year: Year to analyze

    Returns:
        Dictionary mapping district name -> percentage deviation from cohort baseline
        Positive values indicate district spends more than cohort average
        Negative values indicate district spends less than cohort average
    """
    print(f"\nCalculating PPE comparison to cohort for year {year}...")

    # Get cohort assignments for this year
    cohorts = get_western_cohort_districts_for_year(df, reg, year)

    # Calculate cohort baseline PPE (weighted average) for each cohort
    cohort_baselines = {}
    for cohort_name, districts in cohorts.items():
        if districts:
            epp_pivot, _, _ = weighted_epp_aggregation(df, districts)
            if not epp_pivot.empty and year in epp_pivot.index:
                total_ppe = epp_pivot.loc[year].sum()  # Sum all categories for the year
                cohort_baselines[cohort_name] = total_ppe
                print(f"  {cohort_name} baseline PPE ({year}): ${total_ppe:,.0f}")

    # Calculate district PPE and percentage deviation from cohort baseline
    district_comparisons = {}
    for cohort_name, districts in cohorts.items():
        if cohort_name not in cohort_baselines:
            continue

        baseline = cohort_baselines[cohort_name]
        for dist in districts:
            epp_pivot, _ = prepare_district_epp_lines(df, dist)
            if not epp_pivot.empty and year in epp_pivot.index:
                total_ppe = epp_pivot.loc[year].sum()  # Sum all categories for the year
                pct_deviation = ((total_ppe - baseline) / baseline) * 100
                district_comparisons[dist] = pct_deviation
                print(f"    {dist}: ${total_ppe:,.0f} ({pct_deviation:+.1f}%)")

    return district_comparisons


def calculate_cagr_comparison_to_cohort(
    df: pd.DataFrame,
    reg: pd.DataFrame,
    start_year: int,
    end_year: int
) -> Dict[str, float]:
    """
    Calculate CAGR percentage point difference from cohort baseline for each district.

    Args:
        df: Main expenditure data
        reg: District regions/metadata
        start_year: Period start year
        end_year: Period end year

    Returns:
        Dictionary mapping district name -> percentage point difference from cohort baseline CAGR
        Positive values indicate district grew faster than cohort average
        Negative values indicate district grew slower than cohort average
    """
    print(f"\nCalculating CAGR comparison to cohort for period {start_year}-{end_year}...")

    # Helper function to calculate CAGR
    def calc_cagr(series: pd.Series, start: int, end: int) -> float:
        """Calculate CAGR between two years."""
        if start in series.index and end in series.index:
            start_val = series.loc[start]
            end_val = series.loc[end]
            if start_val > 0:
                years = end - start
                cagr = ((end_val / start_val) ** (1 / years) - 1) * 100
                return cagr
        return 0.0

    # Get cohort assignments for end year (most recent cohort definitions)
    cohorts = get_western_cohort_districts_for_year(df, reg, end_year)

    # Calculate cohort baseline CAGR for each cohort
    cohort_baselines = {}
    for cohort_name, districts in cohorts.items():
        if districts:
            epp_pivot, _, _ = weighted_epp_aggregation(df, districts)
            if not epp_pivot.empty:
                total_ppe = epp_pivot.sum(axis=1)  # Sum all categories across years
                cagr = calc_cagr(total_ppe, start_year, end_year)
                if cagr > 0:
                    cohort_baselines[cohort_name] = cagr
                    print(f"  {cohort_name} baseline CAGR ({start_year}-{end_year}): {cagr:.2f}%")

    # Calculate district CAGR and percentage point difference from cohort baseline
    district_comparisons = {}
    for cohort_name, districts in cohorts.items():
        if cohort_name not in cohort_baselines:
            continue

        baseline = cohort_baselines[cohort_name]
        for dist in districts:
            epp_pivot, _ = prepare_district_epp_lines(df, dist)
            if not epp_pivot.empty:
                total_ppe = epp_pivot.sum(axis=1)  # Sum all categories across years
                cagr = calc_cagr(total_ppe, start_year, end_year)
                if cagr > 0:
                    pp_difference = cagr - baseline  # Percentage point difference
                    district_comparisons[dist] = pp_difference
                    print(f"    {dist}: {cagr:.2f}% ({pp_difference:+.2f} pp)")

    return district_comparisons


def main():
    """Generate Western MA choropleth maps for multiple years."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate Western MA choropleth maps")
    parser.add_argument("--force-recompute", action="store_true",
                        help="Bypass cache and recompute from source")
    args = parser.parse_args()

    print("=" * 60)
    print("Western Massachusetts District Choropleth Map Generator")
    print("=" * 60)

    # Load data
    print("\n[1/3] Loading data...")
    df, reg, c70 = load_data(force_recompute=args.force_recompute)
    print(f"  Loaded {len(df)} expenditure records")
    print(f"  Loaded {len(reg)} district metadata records")

    # Load shapefiles (only once)
    print("\n[2/3] Loading shapefiles...")
    shapes = load_shapefiles()

    # Years to generate maps for (in reverse order: 2024, 2019, 2014, 2009)
    years = [2024, 2019, 2014, 2009]

    for year in years:
        print(f"\n{'='*60}")
        print(f"Generating maps for FY {year}")
        print(f"{'='*60}")

        # Match districts to geometries for this year
        print(f"\n[3a] Matching districts for {year}...")
        matched_gdf = match_districts_to_geometries(df, reg, shapes, year=year)

        # Generate enrollment cohort map for this year
        print(f"\n[3b] Generating enrollment cohort map...")
        output_path = OUTPUT_DIR / f"western_ma_choropleth_{year}.png"
        create_western_ma_map(matched_gdf, output_path, year=year)

        # Generate PPE comparison map for this year
        print(f"\n[3c] Generating PPE comparison map...")
        ppe_comparisons = calculate_ppe_comparison_to_cohort(df, reg, year)
        if ppe_comparisons:
            output_path_ppe = OUTPUT_DIR / f"western_ma_ppe_comparison_{year}.png"
            # Create a copy of matched_gdf to avoid modifying original
            matched_gdf_copy = matched_gdf.copy()
            create_ppe_comparison_map(matched_gdf_copy, ppe_comparisons, output_path_ppe, year=year)

        # Generate CAGR comparison maps for different periods
        # For 2024: 15-year CAGR (2009-2024)
        # For 2019: 10-year CAGR (2009-2019)
        # For 2014: 5-year CAGR (2009-2014)
        # For 2009: No CAGR map (it's the starting year)
        if year == 2024:
            print(f"\n[3d] Generating CAGR comparison map (2009-2024)...")
            cagr_comparisons = calculate_cagr_comparison_to_cohort(df, reg, 2009, 2024)
            if cagr_comparisons:
                output_path_cagr = OUTPUT_DIR / f"western_ma_cagr_comparison_2009_2024.png"
                matched_gdf_copy = matched_gdf.copy()
                create_cagr_comparison_map(matched_gdf_copy, cagr_comparisons, output_path_cagr, 2009, 2024)
        elif year == 2019:
            print(f"\n[3d] Generating CAGR comparison map (2009-2019)...")
            cagr_comparisons = calculate_cagr_comparison_to_cohort(df, reg, 2009, 2019)
            if cagr_comparisons:
                output_path_cagr = OUTPUT_DIR / f"western_ma_cagr_comparison_2009_2019.png"
                matched_gdf_copy = matched_gdf.copy()
                create_cagr_comparison_map(matched_gdf_copy, cagr_comparisons, output_path_cagr, 2009, 2019)
        elif year == 2014:
            print(f"\n[3d] Generating CAGR comparison map (2009-2014)...")
            cagr_comparisons = calculate_cagr_comparison_to_cohort(df, reg, 2009, 2014)
            if cagr_comparisons:
                output_path_cagr = OUTPUT_DIR / f"western_ma_cagr_comparison_2009_2014.png"
                matched_gdf_copy = matched_gdf.copy()
                create_cagr_comparison_map(matched_gdf_copy, cagr_comparisons, output_path_cagr, 2009, 2014)
        # No CAGR map for 2009 (baseline year)

        print(f"\n[SUCCESS] Maps for {year} generated!")
        print(f"  Enrollment cohort map: {OUTPUT_DIR / f'western_ma_choropleth_{year}.png'}")
        if ppe_comparisons:
            print(f"  PPE comparison map: {OUTPUT_DIR / f'western_ma_ppe_comparison_{year}.png'}")
        if year == 2024:
            print(f"  CAGR comparison map: {OUTPUT_DIR / 'western_ma_cagr_comparison_2009_2024.png'}")
        elif year == 2019:
            print(f"  CAGR comparison map: {OUTPUT_DIR / 'western_ma_cagr_comparison_2009_2019.png'}")
        elif year == 2014:
            print(f"  CAGR comparison map: {OUTPUT_DIR / 'western_ma_cagr_comparison_2009_2014.png'}")

        print(f"\nDistricts mapped: {len(matched_gdf)}")
        print(f"  Non-regional (solid fill): {len(matched_gdf[~matched_gdf['is_regional']])}")
        print(f"  Regional (black border + cohort letter): {len(matched_gdf[matched_gdf['is_regional']])}")
        print("\nCohort breakdown:")
        for cohort_name in ["TINY", "SMALL", "MEDIUM", "LARGE", "SPRINGFIELD"]:
            n_total = len(matched_gdf[matched_gdf["cohort"] == cohort_name])
            n_regional = len(matched_gdf[(matched_gdf["cohort"] == cohort_name) & (matched_gdf["is_regional"])])
            n_non_regional = n_total - n_regional
            label = get_cohort_label(cohort_name)
            if n_regional > 0:
                print(f"  {label}: {n_non_regional} district(s), {n_regional} regional")
            else:
                print(f"  {label}: {n_non_regional} district(s)")

    print("\n" + "=" * 60)
    print("[SUCCESS] All maps generated successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
