"""
Outlines of Massachusetts's 351 cities and towns, for the explorer's map, from the Census
Bureau's cartographic boundary file of county subdivisions (1:500,000, clipped to the
shoreline):

    data/town_boundaries.json
        {"source", "vintage", "width", "height", "towns": {COUSUB_FIPS: {"name", "town", "sqmi", "d"}}}

"sqmi" is the town's land area in square miles, from the file's ALAND, for densities. Each "d"
is an SVG path in a width x height box: longitude and latitude projected
equirectangularly at the state's middle latitude, then rounded to whole units, which both
shrinks the file and drops points closer together than a unit. Neighbouring towns share
their border points in the source, so they round the same way and still meet. "town" is the
name census_town_population.csv uses, matched on COUSUB_FIPS, for joining the rest of the data.

The file is a shapefile; this reads the two parts it needs (.shp polygons, .dbf names) with
the standard library, so the pipeline needs no geographic packages.

    python update_town_boundaries.py
"""

import io
import json
import math
import struct
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

DATA_DIR = Path("./data")
OUT = DATA_DIR / "town_boundaries.json"
POP_CSV = DATA_DIR / "census_town_population.csv"
VINTAGE = 2023
URL = f"https://www2.census.gov/geo/tiger/GENZ{VINTAGE}/shp/cb_{VINTAGE}_25_cousub_500k.zip"
WIDTH = 1600  # units across; about 190 m each
SQ_M_PER_SQ_MI = 2_589_988.11


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def read_dbf(raw: bytes) -> list[dict]:
    """The attribute table: one dict per record, text fields stripped."""
    n, header_len, rec_len = struct.unpack("<xxxxIHH", raw[:12])
    fields, pos = [], 32
    while raw[pos] != 0x0D:
        name = raw[pos:pos + 11].split(b"\0")[0].decode("latin-1")
        fields.append((name, raw[pos + 16]))
        pos += 32
    out = []
    for i in range(n):
        rec = raw[header_len + i * rec_len: header_len + (i + 1) * rec_len]
        row, at = {}, 1  # the first byte flags deleted records
        for name, size in fields:
            row[name] = rec[at:at + size].decode("latin-1").strip()
            at += size
        out.append(row)
    return out


def read_shp(raw: bytes) -> list[list[list[tuple[float, float]]]]:
    """Each record's rings of (longitude, latitude); polygons only (shape type 5)."""
    shapes, pos = [], 100
    while pos < len(raw):
        _, length = struct.unpack(">ii", raw[pos:pos + 8])
        body = raw[pos + 8: pos + 8 + length * 2]
        pos += 8 + length * 2
        kind = struct.unpack("<i", body[:4])[0]
        if kind == 0:
            shapes.append([])
            continue
        if kind != 5:
            raise ValueError(f"shape type {kind}, expected polygons (5)")
        n_parts, n_points = struct.unpack("<ii", body[36:44])
        parts = list(struct.unpack(f"<{n_parts}i", body[44:44 + 4 * n_parts])) + [n_points]
        pts = struct.unpack(f"<{2 * n_points}d", body[44 + 4 * n_parts: 44 + 4 * n_parts + 16 * n_points])
        xy = list(zip(pts[0::2], pts[1::2]))
        shapes.append([xy[parts[k]:parts[k + 1]] for k in range(n_parts)])
    return shapes


def main():
    z = zipfile.ZipFile(io.BytesIO(fetch(URL)))
    name = lambda ext: next(n for n in z.namelist() if n.endswith(ext))
    rows = read_dbf(z.read(name(".dbf")))
    shapes = read_shp(z.read(name(".shp")))
    # A town that became a city got a new code (Amherst, Framingham, ...), and the population file
    # has both, so each town takes whichever of its codes the outlines use
    pop = pd.read_csv(POP_CSV, dtype={"COUSUB_FIPS": str})
    town_of = dict(zip(pop["COUSUB_FIPS"], pop["Town"]))
    keep = [(r, s) for r, s in zip(rows, shapes) if r["GEOID"] in town_of and s]
    missing = sorted(set(pop["Town"]) - {town_of[r["GEOID"]] for r, _ in keep})
    if missing or len(keep) != pop["Town"].nunique():
        raise ValueError(f"{len(keep)} outlines for {pop['Town'].nunique()} towns; none for {missing[:5]}")

    lons = [x for _, s in keep for ring in s for x, _ in ring]
    lats = [y for _, s in keep for ring in s for _, y in ring]
    lat0 = (min(lats) + max(lats)) / 2
    kx = math.cos(math.radians(lat0))
    scale = WIDTH / ((max(lons) - min(lons)) * kx)
    height = math.ceil((max(lats) - min(lats)) * scale)
    proj = lambda lon, lat: (round((lon - min(lons)) * kx * scale), round((max(lats) - lat) * scale))

    out, points = {}, 0
    for r, s in keep:
        d = []
        for ring in s:
            xy = [proj(*p) for p in ring]
            pts = [xy[0]] + [p for a, p in zip(xy, xy[1:]) if p != a]
            if len(pts) < 4:
                continue  # a sliver smaller than a unit
            points += len(pts)
            seg = [f"M{pts[0][0]} {pts[0][1]}"]
            seg += [f"l{b[0] - a[0]} {b[1] - a[1]}" for a, b in zip(pts, pts[1:-1])]
            d.append("".join(seg) + "z")
        out[r["GEOID"]] = {"name": r["NAME"], "town": town_of[r["GEOID"]], "sqmi": round(int(r["ALAND"]) / SQ_M_PER_SQ_MI, 2),
                           "d": "".join(d)}
    result = {"source": URL, "vintage": VINTAGE, "width": WIDTH, "height": height, "towns": out}
    OUT.write_text(json.dumps(result, separators=(",", ":")))
    print(f"  {OUT}: {len(out)} towns, {points:,} points, {OUT.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    sys.exit(main())
