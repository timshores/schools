"""
Generate choropleth maps of Western Massachusetts school districts for PDF report.

This module:
1. Loads district types from Ch 70 District Profiles (MA_District_Profiles sheet)
2. Loads MA school district shapefiles (Census Bureau TIGER/Line)
3. Matches Western MA traditional districts to geometries
4. Color codes districts by enrollment cohort
5. Handles overlapping district boundaries using spatially-adaptive stripe patterns
6. Generates static PNG map for PDF inclusion

District Type Classification (from data file):
- "District" -> Elementary district (not regional)
- "Unified Regional" -> Serves all grades PK-12 across multiple towns (gets "U" marker)
- "Regional Composite" -> Secondary regional that overlaps elementary districts (gets stripes + black border)

Map Design:
- 5 enrollment cohorts (Tiny, Small, Medium, Large, Springfield) each with unique color
- Non-regional districts: solid filled polygons (85% opacity)
- Unified regional districts: solid filled polygons (85% opacity) with white "U" marker
- Secondary regional districts: diagonal stripes with spatially-adaptive coloring + thick black border
  * Within each regional district, stripe color adapts based on underlying cohorts
  * White stripes: in areas where regional overlaps same-cohort elementary
  * Cohort-color stripes: in areas where regional overlaps different-cohort elementary
  * Example: Green regional shows white stripes over green elementary,
    green stripes over purple/blue elementary
- Legend showing cohort definitions, district counts, and regional markers
- Handles geographic overlap between elementary and secondary regional districts

Extensibility:
- District types loaded from data file (not hardcoded)
- Can be extended to other regions by updating data file
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import pandas as pd

from school_shared import (
    load_data,
    OUTPUT_DIR,
    DISTRICTS_OF_INTEREST,
    get_western_cohort_districts,
    get_cohort_label,
    get_cohort_short_label,
    initialize_cohort_definitions,
)

# Shapefile paths
SHAPEFILE_DIR = Path("./data/shapefiles")
UNIFIED_SHP = SHAPEFILE_DIR / "tl_2023_25_unsd.shp"
ELEMENTARY_SHP = SHAPEFILE_DIR / "tl_2023_25_elsd.shp"
SECONDARY_SHP = SHAPEFILE_DIR / "tl_2023_25_scsd.shp"

# Color palette for cohorts (colorblind-friendly)
COHORT_COLORS = {
    "TINY": "#9C27B0",      # Purple
    "SMALL": "#4CAF50",     # Green
    "MEDIUM": "#2196F3",    # Blue
    "LARGE": "#FF9800",     # Orange
    "SPRINGFIELD": "#F44336",  # Red
}

# Highlight color for districts of interest
HIGHLIGHT_COLOR = "#FF8C00"  # Dark orange outline
EXCLUDED_COLOR = "#EEEEEE"   # Light gray for excluded districts

# District profiles data file
DISTRICT_PROFILES_FILE = Path("./data/Ch 70 District Profiles Actual NSS Over Required.xlsx")

# Version stamp
CODE_VERSION = "v2025.10.06-MAPS"


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
    shapes: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Match Western MA districts from our analysis to shapefile geometries.

    Args:
        df: Main expenditure data
        reg: District regions/metadata
        shapes: GeoDataFrame with district geometries

    Returns:
        GeoDataFrame with matched districts, cohort assignments, and geometries
    """
    print("\nMatching districts to geometries...")

    # Initialize cohort definitions from data
    initialize_cohort_definitions(df, reg)

    # Load district types from data file
    district_types = load_district_types()

    # Get Western MA cohort districts
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
    title: str = "Western Massachusetts Traditional School Districts"
) -> None:
    """
    Generate choropleth map of Western MA districts color-coded by cohort.

    Uses different rendering for different district types:
    - Elementary/Non-regional districts: solid filled polygons (85% opacity)
    - Unified regional districts (PK-12): solid filled polygons (85% opacity)
    - Secondary regional districts: diagonal striped pattern with spatially-adaptive coloring
      * White stripes: in areas where regional overlaps same-cohort elementary districts
      * Cohort-color stripes: in areas where regional overlaps different-cohort districts

    The stripe color adapts spatially within each secondary regional district based on
    what's underneath. For example, a green regional district will show white stripes
    over green elementary districts (for contrast) and green stripes over purple/blue
    elementary districts (for cohort identification).

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
            # Add white "U" label to each unified regional district
            for idx, row in cohort_districts.iterrows():
                centroid = row.geometry.centroid
                # Add black outline for better visibility
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
                # Add white text on top
                ax.text(
                    centroid.x, centroid.y, 'U',
                    color='white',
                    fontsize=18,
                    fontweight='bold',
                    ha='center',
                    va='center',
                    family='sans-serif',
                    zorder=5
                )

    # LAYER 2: Plot secondary regional districts with spatially-adaptive stripe coloring
    # Stripe color changes based on what's underneath each part of the regional district
    print(f"  Analyzing spatial stripe patterns for {len(regional_secondary)} secondary regional districts:")
    for idx, regional_row in regional_secondary.iterrows():
        regional_geom = regional_row.geometry
        regional_cohort = regional_row["cohort"]
        regional_name = regional_row["district_name"]
        regional_color = COHORT_COLORS[regional_cohort]

        # Find which non-regional districts this regional district overlaps
        overlapping = non_regional[non_regional.intersects(regional_geom)]

        # Separate overlaps into same-cohort and different-cohort areas
        same_cohort_geoms = []
        diff_cohort_geoms = []

        for _, overlap_row in overlapping.iterrows():
            overlap_geom = overlap_row.geometry
            overlap_cohort = overlap_row["cohort"]

            # Get the intersection area
            intersection = regional_geom.intersection(overlap_geom)

            if not intersection.is_empty:
                if overlap_cohort == regional_cohort:
                    # Same cohort underneath - will use white stripes
                    same_cohort_geoms.append(intersection)
                else:
                    # Different cohort underneath - will use cohort-color stripes
                    diff_cohort_geoms.append(intersection)

        # Combine geometries for each stripe type
        from shapely.ops import unary_union

        n_same = len(same_cohort_geoms)
        n_diff = len(diff_cohort_geoms)

        print(f"    {regional_name}: {n_diff} different-cohort areas (color), {n_same} same-cohort areas (white)")

        # Plot different-cohort overlaps with cohort-color stripes
        if diff_cohort_geoms:
            diff_union = unary_union(diff_cohort_geoms)

            # Plot with diagonal stripes in regional cohort color
            gpd.GeoDataFrame([{"geometry": diff_union}], geometry="geometry", crs=regional_secondary.crs).plot(
                ax=ax,
                facecolor="none",
                edgecolor=regional_color,
                linewidth=0.8,
                hatch="////",
                zorder=2
            )
            print(f"      -> Plotted {n_diff} color-striped areas")

        # Plot same-cohort overlaps with white stripes
        if same_cohort_geoms:
            same_union = unary_union(same_cohort_geoms)

            # Plot with white diagonal stripes for contrast
            gpd.GeoDataFrame([{"geometry": same_union}], geometry="geometry", crs=regional_secondary.crs).plot(
                ax=ax,
                facecolor="none",
                edgecolor="white",
                linewidth=0.8,
                hatch="////",
                zorder=3
            )
            print(f"      -> Plotted {n_same} same-cohort areas (white stripes)")

        # Add black border around entire secondary regional district boundary
        # to show how collections of elementary districts are joined together
        gpd.GeoDataFrame([{"geometry": regional_geom}], geometry="geometry", crs=regional_secondary.crs).plot(
            ax=ax,
            facecolor="none",
            edgecolor="black",
            linewidth=3.0,
            zorder=6  # Top-most layer to cover all other colors
        )

        if not diff_cohort_geoms and not same_cohort_geoms:
            print(f"      -> WARNING: No geometries to plot for {regional_name}!")

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

        # Show filled box for cohort
        if n_solid > 0:
            legend_elements.append(
                mpatches.Patch(
                    facecolor=COHORT_COLORS[cohort_name],
                    edgecolor="white",
                    alpha=0.85,
                    label=f"{cohort_label}: {n_solid} district(s)"
                )
            )

        # Show striped pattern for secondary regional districts in this cohort
        if n_regional_secondary > 0:
            legend_elements.append(
                mpatches.Patch(
                    facecolor="none",
                    edgecolor=COHORT_COLORS[cohort_name],
                    hatch="////",
                    label=f"{cohort_label}: {n_regional_secondary} regional"
                )
            )

    # Add explanation for secondary regional black border
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

    # Add explanation for "U" label on unified regional districts
    if len(regional_unified) > 0:
        # Add "U" explanation
        from matplotlib.lines import Line2D
        legend_elements.append(
            Line2D([0], [0], marker='$U$', color='w',
                   markerfacecolor='gray', markersize=12,
                   linestyle='None',
                   label="U = Unified regional district")
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
        title="Enrollment Cohorts",
        title_fontsize=11,
        ncol=2  # Use 2 columns to make it more compact
    )

    # Add version stamp
    fig.text(
        0.99, 0.01,
        CODE_VERSION,
        fontsize=8,
        color="gray",
        ha="right",
        va="bottom",
        alpha=0.6
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


def main():
    """Generate Western MA choropleth map."""
    print("=" * 60)
    print("Western Massachusetts District Choropleth Map Generator")
    print("=" * 60)

    # Load data
    print("\n[1/4] Loading data...")
    df, reg, c70 = load_data()
    print(f"  Loaded {len(df)} expenditure records")
    print(f"  Loaded {len(reg)} district metadata records")

    # Load shapefiles
    print("\n[2/4] Loading shapefiles...")
    shapes = load_shapefiles()

    # Match districts to geometries
    print("\n[3/4] Matching districts...")
    matched_gdf = match_districts_to_geometries(df, reg, shapes)

    # Generate map
    print("\n[4/4] Generating map...")
    output_path = OUTPUT_DIR / "western_ma_choropleth.png"
    create_western_ma_map(matched_gdf, output_path)

    print("\n" + "=" * 60)
    print("[SUCCESS] Map generated successfully!")
    print("=" * 60)
    print(f"\nOutput: {output_path}")
    print(f"Districts mapped: {len(matched_gdf)}")
    print(f"  Non-regional (solid fill): {len(matched_gdf[~matched_gdf['is_regional']])}")
    print(f"  Regional (diagonal stripes): {len(matched_gdf[matched_gdf['is_regional']])}")
    print("\nCohort breakdown:")
    for cohort_name in ["TINY", "SMALL", "MEDIUM", "LARGE", "SPRINGFIELD"]:
        n_total = len(matched_gdf[matched_gdf["cohort"] == cohort_name])
        n_regional = len(matched_gdf[(matched_gdf["cohort"] == cohort_name) & (matched_gdf["is_regional"])])
        n_non_regional = n_total - n_regional
        label = get_cohort_label(cohort_name)
        if n_regional > 0:
            print(f"  {label}: {n_non_regional} district(s), {n_regional} regional (striped)")
        else:
            print(f"  {label}: {n_non_regional} district(s)")


if __name__ == "__main__":
    main()
