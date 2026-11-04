"""
Who lives in every Massachusetts city and town: residents by sex and age (table B01001), by race and
Hispanic or Latino origin (B03002), and with a disability (B18101), from the Census Bureau's American
Community Survey five-year estimates, for the four releases that don't overlap: 2005-2009,
2010-2014, 2015-2019, and 2020-2024. Five-year estimates that share years shouldn't be compared,
so these four are the series. B18101 begins with 2008-2012, so 2005-2009 has no disability figures.

    data/census_town_demographics.csv
        Vintage (last year of the five), COUSUB_FIPS, then counts: population, male, female; ages
        under 18, 18-24, 25-44, 45-64, 65 and older; White, Black, Asian, and other or more than one
        race, each not Hispanic or Latino, and Hispanic or Latino of any race; the civilian
        population outside institutions, and those of them with a disability

As update_acs_households.py does, this reads the summary files, which need no API key: for 2009-2020
the members of the Massachusetts zip that hold each table, by HTTP range requests; for 2021 on, each
table's national file, keeping Massachusetts's county subdivisions.

    python update_acs_demographics.py
"""

import csv
import io
import re
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from update_acs_households import BASE, LOOKUP, STATE_ZIP, HttpFile, fetch

DATA_DIR = Path("./data")
OUT = DATA_DIR / "census_town_demographics.csv"
VINTAGES = [2009, 2014, 2019, 2024]
FIRST_TABLE_BASED = 2021
TABLE_FILE = BASE + "table-based-SF/data/5YRData/acsdt5y{y}-{t}.dat"
TABLES = {"B01001": 49, "B03002": 21, "B18101": 39}


def cells(prefix: str, *nums: int) -> list[str]:
    return [f"{prefix}_{n:03d}" for n in nums]


# Each output column: the cells it adds up
COLUMNS = {
    "population": cells("B01001", 1),
    "male": cells("B01001", 2),
    "female": cells("B01001", 26),
    "age_under_18": cells("B01001", *range(3, 7), *range(27, 31)),
    "age_18_24": cells("B01001", *range(7, 11), *range(31, 35)),
    "age_25_44": cells("B01001", *range(11, 15), *range(35, 39)),
    "age_45_64": cells("B01001", *range(15, 20), *range(39, 44)),
    "age_65_up": cells("B01001", *range(20, 26), *range(44, 50)),
    "white": cells("B03002", 3),
    "black": cells("B03002", 4),
    "asian": cells("B03002", 6),
    "other_race": cells("B03002", 5, 7, 8, 9),
    "hispanic": cells("B03002", 12),
    "disability_universe": cells("B18101", 1),
    "disability": cells("B18101", 4, 7, 10, 13, 16, 19, 23, 26, 29, 32, 35, 38),
}


def lookup(y: int) -> dict[str, tuple[int, int]]:
    """Each table's sequence and the column of its first cell in that sequence's file."""
    text = fetch((LOOKUP["old"] if y <= 2012 else LOOKUP["new"]).format(y=y)).decode("latin-1")
    out = {}
    for row in csv.reader(io.StringIO(text)):
        if len(row) > 5 and row[1].strip() in TABLES and row[4].strip():
            n = int(re.match(r"\d+", row[5].strip()).group(0))
            if n != TABLES[row[1].strip()]:
                raise ValueError(f"{y}: {row[1]} has {n} cells, not {TABLES[row[1].strip()]}")
            out[row[1].strip()] = (int(row[2]), int(row[4]))
    if set(out) != set(TABLES) - ({"B18101"} if y < 2012 else set()):
        raise ValueError(f"{y}: tables missing from the lookup: {set(TABLES) - set(out)}")
    return out


def from_state_zip(y: int) -> dict[str, dict[str, float]]:
    """COUSUB_FIPS -> {cell: estimate}, from the year's Massachusetts zip."""
    z = zipfile.ZipFile(io.BufferedReader(HttpFile(STATE_ZIP.format(y=y)), buffer_size=1 << 18))
    names = {n.lower(): n for n in z.namelist()}
    logrec = {}
    if f"g{y}5ma.csv" in names:
        for row in csv.reader(io.TextIOWrapper(z.open(names[f"g{y}5ma.csv"]), encoding="latin-1")):
            if row[2] == "060" and row[3] == "00":
                gid = next(f for f in row if f.startswith("06000US25"))
                logrec[row[4].strip().zfill(7)] = gid[-10:]
    else:  # before 2012, the fixed-width geography file only
        for line in io.TextIOWrapper(z.open(names[f"g{y}5ma.txt"]), encoding="latin-1"):
            if line[8:11] == "060" and line[11:13] == "00":
                gid = re.search(r"06000US25\d{8}", line)
                if gid:
                    logrec[line[13:20].strip().zfill(7)] = gid.group(0)[-10:]
    out: dict[str, dict[str, float]] = {}
    for table, (seq, start) in lookup(y).items():
        for row in csv.reader(io.TextIOWrapper(z.open(names[f"e{y}5ma{seq:04d}000.txt"]), encoding="latin-1")):
            rec = row[5].strip().zfill(7)
            if rec not in logrec:
                continue
            vals = row[start - 1:start - 1 + TABLES[table]]
            got = out.setdefault(logrec[rec], {})
            for i, v in enumerate(vals, 1):
                got[f"{table}_{i:03d}"] = float(v) if v not in ("", ".") else float("nan")
    return out


def from_table_files(y: int) -> dict[str, dict[str, float]]:
    """COUSUB_FIPS -> {cell: estimate}, streaming each table's national file for Massachusetts's towns."""
    out: dict[str, dict[str, float]] = {}
    for table in TABLES:
        req = urllib.request.Request(TABLE_FILE.format(y=y, t=table.lower()), headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=600) as r:
            lines = io.TextIOWrapper(r, encoding="latin-1")
            head = next(lines).rstrip("\n").split("|")
            want = {i: f"{table}_{int(h.split('_E')[1]):03d}" for i, h in enumerate(head) if re.fullmatch(rf"{table}_E\d+", h)}
            for line in lines:
                if line.startswith("0600000US25"):
                    f = line.rstrip("\n").split("|")
                    got = out.setdefault(f[0][-10:], {})
                    for i, cell in want.items():
                        got[cell] = float(f[i]) if f[i] not in ("", ".") else float("nan")
    return out


def main():
    towns = set(pd.read_csv(DATA_DIR / "census_town_population.csv", dtype={"COUSUB_FIPS": str})["COUSUB_FIPS"])
    rows = []
    for y in VINTAGES:
        raw = from_table_files(y) if y >= FIRST_TABLE_BASED else from_state_zip(y)
        found = towns & set(raw)
        for fips in sorted(found):
            c = raw[fips]
            row = {"Vintage": y, "COUSUB_FIPS": fips}
            for col, parts in COLUMNS.items():
                vals = [c.get(p) for p in parts]
                row[col] = None if any(v is None or v != v for v in vals) else round(sum(vals))
            rows.append(row)
        print(f"  {y - 4}-{y}: {len(found)} of {len(towns)} towns")
    df = pd.DataFrame(rows)
    # The parts add to the whole, within rounding: a check that each cell landed in its column
    for whole, parts in [("population", ["male", "female"]),
                         ("population", ["age_under_18", "age_18_24", "age_25_44", "age_45_64", "age_65_up"]),
                         ("population", ["white", "black", "asian", "other_race", "hispanic"])]:
        off = (df[parts].sum(axis=1) - df[whole]).abs() > 2
        if off.any():
            raise ValueError(f"{int(off.sum())} rows where {'+'.join(parts)} misses {whole}")
    df.to_csv(OUT, index=False)
    print(f"  {OUT}: {len(df):,} rows")


if __name__ == "__main__":
    main()
