"""
Download what the explorer's Housing and enrollment tab needs beyond the pipeline's other data (Tim, seahag prompt
log #249): the housing counts and outside studies it sets beside DESE's enrollment and DLS's new growth.

    data/census_town_housing.csv      every city and town's housing units and occupied units, April 1, 2010 and
                                      April 1, 2020, from the decennial census's redistricting files (P.L. 94-171,
                                      table H1); the counts MAPC's 2024 update used
    data/mapc_community_types.csv     MAPC's community type and subtype for every city and town (its 2008
                                      classification, as MAPC's own data keys carry it)
    data/mapc_enrollment_2017.csv     the data MAPC published with its October 2017 brief: each municipal district's
                                      housing units permitted 2010-2016 (single-family, multifamily), as a share of
                                      its 2010 housing, and its enrollment, 2010/11 and 2016/17, as published
    data/aft_cocs_studies.csv         every study in American Farmland Trust's Cost of Community Services fact sheet
                                      (October 2025): community, state, the three ratios, author and year, read from
                                      the PDF's table with pdftotext (poppler-utils)

The raw downloads are cached in data/census_raw/ (not in git): the census files are 37 MB zipped, and the fact sheet is
AFT's publication, which the CSV cites rather than copies. Delete a cached file to fetch it again.

    python update_housing_data.py
"""

import csv
import datetime as dt
import io
import json
import re
import subprocess
import urllib.request
import zipfile
from pathlib import Path

import openpyxl

DATA_DIR = Path("./data")
RAW_DIR = DATA_DIR / "census_raw"
SOURCES_JSON = DATA_DIR / "housing_sources.json"
PL = {
    2010: "https://www2.census.gov/census_2010/01-Redistricting_File--PL_94-171/Massachusetts/ma2010.pl.zip",
    2020: ("https://www2.census.gov/programs-surveys/decennial/2020/data/01-Redistricting_File--PL_94-171/"
           "Massachusetts/ma2020.pl.zip"),
}
MAPC_KEYS = "https://raw.githubusercontent.com/MAPC/mapcdatakeys/HEAD/data-raw/muni_data_keys_2024.csv"
MAPC_2017 = "https://www.mapc.org/wp-content/uploads/2017/10/SchoolEnrollmentandHousing_Data_Metadata.xlsx"
AFT = "https://farmlandinfo.org/wp-content/uploads/sites/2/2019/09/cost-of-community-services-fact-sheet.pdf"


def fetch(url: str, path: Path) -> Path:
    if not path.exists():
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=600) as r:
            path.write_bytes(r.read())
    return path


def housing(year: int, url: str) -> list[dict]:
    """County subdivisions (summary level 060) from one year's geographic header and its table H1: housing units,
    occupied and vacant, the last three fields of the file's second segment"""
    z = zipfile.ZipFile(fetch(url, RAW_DIR / url.rsplit("/", 1)[1]))
    geo_name = next(n for n in z.namelist() if "geo" in n)
    seg2 = next(n for n in z.namelist() if re.search(r"00002\d{4}\.pl$", n))
    towns = {}
    for line in io.TextIOWrapper(z.open(geo_name), encoding="latin-1"):
        if year == 2020:
            f = line.rstrip("\n").split("|")
            sumlev, comp, logrec, county, cousub, name = f[2], f[4], f[7], f[14], f[17], f[87]
        else:  # 2010's header is fixed-width
            sumlev, comp, logrec = line[8:11], line[11:13], line[18:25]
            county, cousub, name = line[29:32], line[36:41], line[226:316].strip()
        if sumlev == "060" and comp == "00" and cousub != "00000":
            towns[logrec] = {"COUSUB_FIPS": f"25{county}{cousub}", "Census_name": name}
    sep = "|" if year == 2020 else ","
    rows = []
    for line in io.TextIOWrapper(z.open(seg2), encoding="latin-1"):
        f = line.rstrip("\n").split(sep)
        if f[4] in towns:
            rows.append({"Year": year, **towns[f[4]], "Housing_units": int(f[-3]), "Occupied_units": int(f[-2]),
                         "Source": url})
    return rows


def aft_table(pdf: Path) -> list[dict]:
    """The fact sheet's Summary of Cost of Community Services Studies: each row's ratios, cost of services per dollar
    of revenue, for residential, commercial and industrial, and working and open land. A source that wraps puts its
    first part on the line above the ratios and its year on the line below."""
    lines = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True,
                           check=True).stdout.splitlines()
    ratio = r"1\s*[:;]\s*([\d.]+)"
    full = re.compile(rf"^\s+(\S.*?)\s{{2,}}{ratio}\s+{ratio}\s+{ratio}\s+(.+?),?\s*(\d{{4}})\s*$")
    bare = re.compile(rf"^\s+(\S.*?)\s{{2,}}{ratio}\s+{ratio}\s+{ratio}\s*$")
    state, rows = None, []
    for i, line in enumerate(lines):
        if "CONTINUED FROM PAGE 1" in line:
            break
        if m := full.match(line):
            name, r, c, w, src, yr = m.groups()
        elif m := bare.match(line):
            name, r, c, w = m.groups()
            src, yr = re.match(r"(.+?),?\s*(\d{4})$", f"{lines[i - 1].strip()} {lines[i + 1].strip()}").groups()
        else:
            if re.fullmatch(r"[A-Z][a-z]+( [A-Z][a-z]+)*", line.strip()) and not line.startswith("   "):
                state = line.strip()
            continue
        rows.append({"State": state, "Community": name.strip(), "Residential": float(r),
                     "Commercial_industrial": float(c), "Working_open_land": float(w), "Source": src.strip(),
                     "Year": int(yr)})
    return rows


def write(path: Path, rows: list[dict]):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"  {path}: {len(rows)} rows")


def main():
    today = dt.date.today().isoformat()
    names = {r["COUSUB_FIPS"]: r["Town"] for r in csv.DictReader((DATA_DIR / "census_town_population.csv").open())}
    rows = [r for y, u in PL.items() for r in housing(y, u)]
    for r in rows:
        r["Town"] = names.get(r["COUSUB_FIPS"], "")
    missing = sorted({r["Census_name"] for r in rows if not r["Town"]})
    assert not missing, f"county subdivisions with no town name: {missing}"
    write(DATA_DIR / "census_town_housing.csv",
          [{k: r[k] for k in ("Year", "COUSUB_FIPS", "Town", "Housing_units", "Occupied_units", "Source")} for r in rows])

    keys = list(csv.DictReader(io.StringIO(fetch(MAPC_KEYS, RAW_DIR / "mapc_muni_data_keys_2024.csv").read_text())))
    write(DATA_DIR / "mapc_community_types.csv",
          [{"muni_id": k["muni_id"], "Town": k["muni_name"], "COUSUB_FIPS": k["cosub_cn20"],
            "Community_type": k["cmtyp08"], "Subtype": k["cmsbt08"], "MAPC_region": k["mapc"]} for k in keys])

    ws = openpyxl.load_workbook(fetch(MAPC_2017, RAW_DIR / "mapc_enrollment_2017.xlsx"), read_only=True)["Data"]
    head, *data = list(ws.iter_rows(values_only=True))
    write(DATA_DIR / "mapc_enrollment_2017.csv", [dict(zip(head, r)) for r in data if r[0] is not None])

    write(DATA_DIR / "aft_cocs_studies.csv", aft_table(fetch(AFT, RAW_DIR / "aft_cocs_fact_sheet.pdf")))

    SOURCES_JSON.write_text(json.dumps({
        "downloaded": today,
        "census_town_housing": {"source": PL, "table": "H1, occupancy status (redistricting data, P.L. 94-171)"},
        "mapc_community_types": {"source": MAPC_KEYS, "repository": "https://github.com/MAPC/mapcdatakeys",
                                 "classification": "MAPC community types, 2008 (cmtyp08, cmsbt08)"},
        "mapc_enrollment_2017": {"source": MAPC_2017,
                                 "report": "https://www.mapc.org/learn/research-analysis/enrollment-2017/"},
        "aft_cocs_studies": {"source": AFT, "page": "https://farmlandinfo.org/?p=19267",
                             "edition": "October 2025"},
    }, indent=1) + "\n")
    print(f"  {SOURCES_JSON}")


if __name__ == "__main__":
    main()
