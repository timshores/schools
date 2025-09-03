#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Make district→EOHHS region CSV (authoritative GIS + town backfill)
Reads from:  .\data\ma school districts.csv
             .\data\schooldistricts.zip
             .\data\reg_eohhs.zip
Writes to:   .\data\ma_districts_with_eohhs_regions_complete.csv

Dependencies: pandas, fiona, shapely, pyproj
  pip install pandas fiona shapely pyproj
Tested on Windows (paths via pathlib).

Method:
1) Authoritative: polygon overlay (district vs EOHHS regions), pick largest area.
2) Backfill Unknown: town→region lookup (REGEOHHS_POLY_TOWNS) using:
   a) curated charter→host-town mapping,
   b) town-name detection inside district name.
"""

from pathlib import Path
import sys
import re
import pandas as pd
import fiona
from shapely.geometry import shape
from shapely.ops import transform as shp_transform
import pyproj


# ---------------------------
# Config
# ---------------------------
DATA_DIR = Path.cwd() / "data"
DISTRICT_LIST_FILE = DATA_DIR / "ma school districts.csv"
SCHOOLDISTRICTS_ZIP = DATA_DIR / "schooldistricts.zip"
REG_EOHHS_ZIP = DATA_DIR / "reg_eohhs.zip"
OUTPUT_FILE = DATA_DIR / "ma_districts_with_eohhs_regions_complete.csv"

# Layer names to auto-detect exact names
DISTRICTS_LAYER_CANDIDATES = ["SCHOOLDISTRICTS_POLY", "schooldistricts_poly", "SchoolDistricts_Poly"]
EOHHS_LAYER_CANDIDATES = ["REGEOHHS_POLY", "reg_eohhs_poly", "RegEOHHS_Poly"]
EOHHS_TOWNS_LAYER_CANDIDATES = ["REGEOHHS_POLY_TOWNS", "reg_eohhs_poly_towns", "RegEOHHS_Poly_Towns"]

# Curated (not exhaustive) mapping of charter to host town, for backfill
CHARTER_HOST = {
    # Western / Pioneer Valley
    "Four Rivers Charter Public": "Greenfield",
    "Paulo Freire Social Justice Charter": "Chicopee",   # former Holyoke site
    "Pioneer Valley Chinese Immersion Charter": "Hadley",
    "Hampden Charter School of Science": "Chicopee",
    "Veritas Preparatory Charter": "Springfield",
    "Baystate Academy Charter Public School": "Springfield",
    "Sabis International Charter": "Springfield",
    "Berkshire Arts and Technology Charter": "Adams",
    # Central
    "Abby Kelley Foster Charter Public": "Worcester",
    "Seven Hills Charter": "Worcester",
    # Metro West
    "Advanced Math and Science Academy Charter": "Marlborough",
    "Francis W. Parker Charter": "Devens",
    "Christa McAuliffe Charter": "Framingham",
    # Boston / NE 
    "Academy Of the Pacific Rim Charter": "Boston",
    "Boston Collegiate Charter": "Boston",
    "Boston Preparatory Charter": "Boston",
    "MATCH Charter Public": "Boston",
    "Conservatory Lab Charter": "Boston",
    "Edward M. Kennedy Academy for Health Careers": "Boston",
    "Mystic Valley Regional Charter": "Malden",
    "Innovations Academy Charter": "Tyngsborough",
    "Lowell Community Charter Public": "Lowell",
    "Lowell Collegiate Charter": "Lowell",
    "Community Charter School of Cambridge": "Cambridge",
    "Benjamin Banneker Charter Public": "Cambridge",
    "KIPP Academy Lynn": "Lynn",
    "Pioneer Charter School of Science": "Everett",
    "Phoenix Charter Academy": "Chelsea",  # default site; also Lawrence/Springfield
    "Excel Academy Charter": "Boston",
    "Brooke Charter": "Boston",
    "City on a Hill Charter": "Boston",
    # Southeast / Cape
    "Argosy Collegiate Charter School": "Fall River",
    "Atlantis Charter": "Fall River",
    "Alma del Mar Charter School": "New Bedford",
    "Global Learning Charter": "New Bedford",
    "Rising Tide Charter Public School": "Plymouth",
    "Old Sturbridge Academy Charter": "Sturbridge",
    "South Shore Charter Public School": "Norwell",
    "New Heights Charter School of Brockton": "Brockton",
    "Benjamin Franklin Classical Charter": "Franklin",
    "Barnstable Community Horace Mann Charter": "Barnstable",
    "Barnstable Horace Mann Charter": "Barnstable",
    "Bentley Academy Charter School": "Salem",
    # Statewide/virtual specifics
    "Greenfield Commonwealth Virtual School": "Greenfield",
}

# ---------------------------
# Helpers
# ---------------------------

def fail(msg: str) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(1)


def list_layers(zip_path: Path):
    return fiona.listlayers(f"zip://{zip_path}")


def pick_layer(zip_path: Path, candidates):
    layers = list_layers(zip_path)
    for cand in candidates:
        if cand in layers:
            return cand
    # fall back to first polygon layer if possible
    return layers[0] if layers else None


def read_records(zip_path: Path, layer: str):
    with fiona.open(f"zip://{zip_path}", layer=layer) as src:
        crs = src.crs_wkt or src.crs
        schema_props = src.schema.get("properties", {})
        recs = list(src)
    return recs, schema_props, crs


def ensure_data_exists():
    missing = []
    if not DISTRICT_LIST_FILE.exists():
        missing.append(str(DISTRICT_LIST_FILE))
    if not SCHOOLDISTRICTS_ZIP.exists():
        missing.append(str(SCHOOLDISTRICTS_ZIP))
    if not REG_EOHHS_ZIP.exists():
        missing.append(str(REG_EOHHS_ZIP))
    if missing:
        fail("Missing required file(s):\n  " + "\n  ".join(missing))


def zfill8(x: str) -> str:
    return (x or "").strip().zfill(8)


def safe_org8(props: dict) -> str | None:
    c8 = props.get("ORG8CODE")
    if c8 and str(c8).strip():
        return zfill8(str(c8))
    c4 = props.get("ORG4CODE")
    if c4 and str(c4).strip():
        return str(c4).strip().zfill(4) + "0000"
    return None


def reprojector(from_crs, to_crs):
    if from_crs == to_crs:
        return None
    transformer = pyproj.Transformer.from_crs(
        pyproj.CRS.from_user_input(from_crs),
        pyproj.CRS.from_user_input(to_crs),
        always_xy=True
    )
    return lambda geom: shp_transform(transformer.transform, geom)


def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def infer_town_from_name(district_name: str, towns_upper_set: set[str]) -> str | None:
    # Whole-word scan: pad and strip punctuation to produce word boundaries
    up = " " + normalize_spaces(re.sub(r"[-/()\\.,&]", " ", district_name.upper())) + " "
    for town_u in towns_upper_set:
        token = " " + town_u + " "
        if token in up:
            return town_u
    return None


# ---------------------------
# Main
# ---------------------------

def main():
    ensure_data_exists()

    # Load user list (tab-delimited; preserve leading zeros by dtype=str)
    user = pd.read_csv(DISTRICT_LIST_FILE, sep="\t", header=None,
                       names=["district_code", "district_name"], dtype=str)
    user["district_code"] = user["district_code"].apply(zfill8)

    # Detect layers
    dist_layer = pick_layer(SCHOOLDISTRICTS_ZIP, DISTRICTS_LAYER_CANDIDATES)
    eohhs_layer = pick_layer(REG_EOHHS_ZIP, EOHHS_LAYER_CANDIDATES)
    towns_layer = pick_layer(REG_EOHHS_ZIP, EOHHS_TOWNS_LAYER_CANDIDATES)
    if not dist_layer or not eohhs_layer or not towns_layer:
        fail("Could not detect expected layers in the provided ZIPs.")

    # Read features
    dist_recs, dist_props, dist_crs = read_records(SCHOOLDISTRICTS_ZIP, dist_layer)
    eohhs_recs, eohhs_props, eohhs_crs = read_records(REG_EOHHS_ZIP, eohhs_layer)
    towns_recs, towns_props, towns_crs = read_records(REG_EOHHS_ZIP, towns_layer)

    # Determine region-name field
    reg_field = ("REG_NAME" if "REG_NAME" in eohhs_props else
                 ("REG_NAM" if "REG_NAM" in eohhs_props else
                  ("REGION" if "REGION" in eohhs_props else None)))
    if not reg_field:
        fail("Could not find region name field in EOHHS polygon attributes.")

    town_field = "TOWN" if "TOWN" in towns_props else ("TOWN_NAME" if "TOWN_NAME" in towns_props else None)
    town_reg_field = ("REG_NAME" if "REG_NAME" in towns_props else
                      ("REG_NAM" if "REG_NAM" in towns_props else
                       ("REGION" if "REGION" in towns_props else None)))
    if not town_field or not town_reg_field:
        fail("Could not find expected fields in REGEOHHS_POLY_TOWNS.")

    # Prepare reprojection of EOHHS regions to district CRS (for area calculation)
    reproj_regions = reprojector(eohhs_crs, dist_crs)

    # Build regions list
    regions = []
    for rec in eohhs_recs:
        geom = shape(rec["geometry"]) if rec["geometry"] else None
        if reproj_regions and geom is not None:
            geom = reproj_regions(geom)
        regions.append((normalize_spaces(rec["properties"].get(reg_field, "")), geom))

    # Build districts list
    districts = []
    for rec in dist_recs:
        props = rec["properties"]
        code = safe_org8(props)
        name = normalize_spaces(props.get("DISTRICT_NAME", ""))
        geom = shape(rec["geometry"]) if rec["geometry"] else None
        districts.append((code, name, geom))

    # Authoritative assignment: largest-area intersection
    map_rows = []
    for code, dname, dgeom in districts:
        if code is None or dgeom is None:
            map_rows.append((code, dname, "Unknown", 0.0))
            continue
        best_name = "Unknown"
        best_area = 0.0
        for rname, rgeom in regions:
            if rgeom is None:
                continue
            if not dgeom.intersects(rgeom):
                continue
            area = dgeom.intersection(rgeom).area
            if area > best_area:
                best_area = area
                best_name = rname
        map_rows.append((code, dname, best_name, best_area))

    map_df = pd.DataFrame(map_rows, columns=["ORG8CODE", "DISTRICT_NAME", "eohhs_region", "intersect_area"])

    # Merge to user's list (by district_code)
    result = user.merge(map_df[["ORG8CODE", "eohhs_region"]], left_on="district_code", right_on="ORG8CODE", how="left")
    result.drop(columns=["ORG8CODE"], inplace=True)
    result["eohhs_region"] = result["eohhs_region"].fillna("Unknown")
    result["region_confidence"] = result["eohhs_region"].apply(lambda x: "largest-area" if x != "Unknown" else "no_intersection")

    # Build town→region lookup
    town_rows = []
    for rec in towns_recs:
        t = normalize_spaces(rec["properties"].get(town_field, ""))
        r = normalize_spaces(rec["properties"].get(town_reg_field, ""))
        if t:
            town_rows.append((t, r))
    town_df = pd.DataFrame(town_rows, columns=["town", "eohhs_region"])
    town_df["town_upper"] = town_df["town"].str.upper().str.strip()
    town_to_region = dict(zip(town_df["town_upper"], town_df["eohhs_region"]))
    towns_upper_set = set(town_df["town_upper"].unique())

    # Curated charter mapping (upper)
    charter_host_upper = {k.upper(): v for k, v in CHARTER_HOST.items()}

    def backfill_region(row):
        # Keep authoritative
        if row["eohhs_region"] != "Unknown":
            return row["eohhs_region"], row["region_confidence"]

        name = row["district_name"] or ""
        name_u = name.upper()

        # 1) Curated charter→host-town
        for key_u, host_town in charter_host_upper.items():
            if key_u in name_u:
                reg = town_to_region.get(host_town.upper())
                if reg:
                    return reg, "host-town-curated"

        # 2) Host town inferred from district name
        town_u = infer_town_from_name(name, towns_upper_set)
        if town_u:
            reg = town_to_region.get(town_u)
            if reg:
                return reg, "host-town-inferred"

        # 3) Still unknown
        return "Unknown", "no_intersection"

    result[["eohhs_region", "region_confidence"]] = result.apply(
        lambda r: pd.Series(backfill_region(r)),
        axis=1
    )

    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    # Summary
    by_conf = result["region_confidence"].value_counts(dropna=False).to_dict()
    by_region = result["eohhs_region"].value_counts(dropna=False).to_dict()

    print(f"✓ Wrote: {OUTPUT_FILE}")
    print("Confidence breakdown:", by_conf)
    print("Region counts:", by_region)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        fail(str(e))