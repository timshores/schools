"""
Households in every Massachusetts city and town, from the Census Bureau's American
Community Survey five-year estimates (table B11001, total households), for each
five-year release from 2005-2009 through the latest:

    data/census_town_households.csv
        Vintage (last year of the five), COUSUB_FIPS, Households, Source

The Census API now needs a key, so this reads the summary files instead, which
don't. For 2021 on, the table-based file for B11001 covers the whole country;
this keeps Massachusetts's county subdivisions. For 2009-2020, each year's
Massachusetts file is a zip of about 150 MB, so this reads only the two members
it needs, the geography file and the sequence holding B11001, with HTTP range
requests. Towns join the population file (census_town_population.csv) on
COUSUB_FIPS.

    python update_acs_households.py
"""

import csv
import io
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

DATA_DIR = Path("./data")
OUT = DATA_DIR / "census_town_households.csv"
POP_CSV = DATA_DIR / "census_town_population.csv"
BASE = "https://www2.census.gov/programs-surveys/acs/summary_file/{y}/"
STATE_ZIP = BASE + "data/5_year_by_state/Massachusetts_All_Geographies_Not_Tracts_Block_Groups.zip"
LOOKUP = {  # where each year's sequence/table lookup lives
    "old": BASE + "documentation/5_year/user_tools/Sequence_Number_and_Table_Number_Lookup.txt",
    "new": BASE + "documentation/user_tools/ACS_5yr_Seq_Table_Number_Lookup.txt",
}
TABLE_FILE = BASE + "table-based-SF/data/5YRData/acsdt5y{y}-b11001.dat"
FIRST, FIRST_TABLE_BASED = 2009, 2021
TABLE = "B11001"


def fetch(url: str, headers: dict | None = None) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", **(headers or {})})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


class HttpFile(io.RawIOBase):
    """A read-only file over HTTP range requests: enough for zipfile to read one member."""

    def __init__(self, url: str):
        self.url, self.pos = url, 0
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size or not len(b):
            return 0
        end = min(self.size, self.pos + len(b)) - 1
        data = fetch(self.url, {"Range": f"bytes={self.pos}-{end}"})
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def sequence_of(y: int) -> int:
    """The summary-file sequence that holds B11001 this year (its first estimate is column 7)."""
    text = fetch((LOOKUP["old"] if y <= 2012 else LOOKUP["new"]).format(y=y)).decode("latin-1")
    for row in csv.reader(io.StringIO(text)):
        if len(row) > 4 and row[1].strip() == TABLE and row[4].strip():
            if row[4].strip() != "7":
                raise ValueError(f"{y}: {TABLE} starts at column {row[4]}, not 7; read the lookup again")
            return int(row[2])
    raise ValueError(f"{y}: {TABLE} not in the lookup file")


def from_state_zip(y: int) -> dict[str, int]:
    """COUSUB_FIPS -> households, from the year's Massachusetts zip."""
    url = STATE_ZIP.format(y=y)
    z = zipfile.ZipFile(io.BufferedReader(HttpFile(url), buffer_size=1 << 18))
    names = {n.lower(): n for n in z.namelist()}
    est = names[f"e{y}5ma{sequence_of(y):04d}000.txt"]
    logrec = {}
    if f"g{y}5ma.csv" in names:
        for row in csv.reader(io.TextIOWrapper(z.open(names[f"g{y}5ma.csv"]), encoding="latin-1")):
            # FILEID, STUSAB, SUMLEVEL, COMPONENT, LOGRECNO, ...; GEOID is the field like 06000US25...
            if row[2] == "060" and row[3] == "00":
                gid = next(f for f in row if f.startswith("06000US25"))
                logrec[row[4].strip().zfill(7)] = gid[-10:]
    else:
        # Before 2012 only the fixed-width file: SUMLEVEL at 9-11, COMPONENT 12-13, LOGRECNO 14-20
        for line in io.TextIOWrapper(z.open(names[f"g{y}5ma.txt"]), encoding="latin-1"):
            if line[8:11] == "060" and line[11:13] == "00":
                gid = re.search(r"06000US25\d{8}", line)
                if gid:
                    logrec[line[13:20].strip().zfill(7)] = gid.group(0)[-10:]
    out = {}
    for row in csv.reader(io.TextIOWrapper(z.open(est), encoding="latin-1")):
        rec = row[5].strip().zfill(7)
        if rec in logrec and row[6] not in ("", "."):
            out[logrec[rec]] = int(float(row[6]))
    return out


def from_table_file(y: int) -> dict[str, int]:
    """COUSUB_FIPS -> households, from the year's national B11001 file."""
    lines = fetch(TABLE_FILE.format(y=y)).decode("latin-1").splitlines()
    head = lines[0].split("|")
    col = head.index("B11001_E001")
    return {f[0][-10:]: int(float(f[col])) for f in (ln.split("|") for ln in lines[1:]) if f[0].startswith("0600000US25")}


def main():
    towns = set(pd.read_csv(POP_CSV, dtype={"COUSUB_FIPS": str})["COUSUB_FIPS"])
    rows = []
    y = FIRST
    while True:
        try:
            counts = from_table_file(y) if y >= FIRST_TABLE_BASED else from_state_zip(y)
        except urllib.error.HTTPError as e:
            if e.code == 404 and y > FIRST_TABLE_BASED:
                break  # no release yet for this year
            raise
        source = TABLE_FILE.format(y=y) if y >= FIRST_TABLE_BASED else STATE_ZIP.format(y=y)
        found = {k: v for k, v in counts.items() if k in towns}
        missing = sorted(towns - set(found))
        print(f"  ACS {y - 4}-{y}: {len(found)} towns" + (f"; missing {missing[:5]}" if missing else ""), flush=True)
        rows += [{"Vintage": y, "COUSUB_FIPS": k, "Households": v, "Source": source} for k, v in sorted(found.items())]
        y += 1
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"  {OUT}: {len(rows):,} rows")


if __name__ == "__main__":
    sys.exit(main())
