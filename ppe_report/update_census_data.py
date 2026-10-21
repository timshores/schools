"""
Download the Census Bureau's Annual Survey of School System Finances (F-33)
for Massachusetts, FY2009 onward, and join it to DESE district codes.

1. One file per fiscal year (elsecYY.txt, every school system in the US,
   dollar amounts in thousands), cached in data/census_raw/ (not in git).
   Massachusetts rows (Census STATE 22) for every year go to
   data/census_f33_ma.csv with all of Census's columns, plus FY, NCESID,
   Org4Code and DIST_CODE (DESE's codes, DIST_CODE = Org4Code x 10000 as in
   the expenditure data).
2. The NCES-to-DESE crosswalk comes from the NCES Common Core of Data
   district directory (via the Urban Institute's Education Data API), whose
   state_leaid is "MA-" plus DESE's four-digit code. Several directory years
   are combined so districts that merged or closed still match. Saved to
   data/nces_ma_districts.csv; delete it to fetch again.
3. Population of every Massachusetts city and town, by fiscal year, from the
   Census Bureau's subcounty population estimates, for spending per resident
   -> data/census_town_population.csv. A fiscal year takes the July 1
   estimate at its start (FY2025: July 1, 2024), from three vintages: the
   2000-2010 intercensal file, the 2010-2020 file, and the 2020s file. FY2021
   uses the April 1, 2020 estimates base instead of the July 2020 estimate,
   which drops by more than a quarter in Amherst (27,806 against 39,264 in April and
   40,260 a year later) and in other college towns, then recovers.

Census publishes a fiscal year about 18 months after it ends (FY2024 in
2026), so the latest year here usually trails DESE by one.

    python update_census_data.py
"""

import datetime as dt
import json
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

DATA_DIR = Path("./data")
RAW_DIR = DATA_DIR / "census_raw"
FIRST_YEAR = 2009
CENSUS_URL = ("https://www2.census.gov/programs-surveys/school-finances/tables/"
              "{fy}/secondary-education-finance/elsec{yy:02d}.{ext}")
# A national file has about 14,000 school systems. Census's FY2022 elsec22.txt
# stops after 1,729 of them (checked 2026-10-21), so a short text file falls
# back to the same year's Excel file.
MIN_SYSTEMS = 10000
CENSUS_SOURCE = ("Census Bureau, Annual Survey of School System Finances: "
                 "https://www.census.gov/programs-surveys/school-finances.html")
CENSUS_MA = "22"  # Census's own state code, not the FIPS code (25)
NCES_URL = "https://educationdata.urban.org/api/v1/school-districts/ccd/directory/{year}/?fips=25"
NCES_YEARS = [2009, 2014, 2019, 2024]
OUT = DATA_DIR / "census_f33_ma.csv"
NCES_XWALK = DATA_DIR / "nces_ma_districts.csv"
SOURCES_JSON = DATA_DIR / "census_sources.json"
ID_COLS = ["STATE", "IDCENSUS", "PID6", "UNIT_TYPE", "FIPST", "NAME", "CONUM", "CSA", "CBSA",
           "SCHLEV", "NCESID", "YRDATA"]
POP_URL = "https://www2.census.gov/programs-surveys/popest/datasets/"
# (file, {fiscal year: column}): each fiscal year's estimate and the vintage it comes from
POP_FILES = [
    ("2000-2010/intercensal/cities/sub-est00int.csv", {2009: "POPESTIMATE2008", 2010: "POPESTIMATE2009"}),
    ("2010-2020/cities/SUB-EST2020_25.csv", {fy: f"POPESTIMATE{fy - 1}" for fy in range(2011, 2021)}),
    ("2020-2024/cities/totals/sub-est2024_25.csv",
     {2021: "ESTIMATESBASE2020", **{fy: f"POPESTIMATE{fy - 1}" for fy in range(2022, 2026)}}),
]
POP_OUT = DATA_DIR / "census_town_population.csv"
COUNTIES = {"001": "Barnstable", "003": "Berkshire", "005": "Bristol", "007": "Dukes", "009": "Essex",
            "011": "Franklin", "013": "Hampden", "015": "Hampshire", "017": "Middlesex", "019": "Nantucket",
            "021": "Norfolk", "023": "Plymouth", "025": "Suffolk", "027": "Worcester"}


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def download(fy: int, ext: str) -> Path | None:
    """The cached national file for one year, downloading it if needed; None on a 404."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"elsec{fy % 100:02d}.{ext}"
    if not path.exists():
        try:
            path.write_bytes(fetch(CENSUS_URL.format(fy=fy, yy=fy % 100, ext=ext)))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            raise
    return path


def census_year(fy: int) -> pd.DataFrame | None:
    """Massachusetts rows for one fiscal year, or None if Census hasn't published it."""
    path = download(fy, "txt")
    if path is None:
        return None
    # Data rows end in a comma the header lacks; index_col=False keeps columns aligned
    raw = pd.read_csv(path, dtype=str, index_col=False)
    if len(raw) < MIN_SYSTEMS:
        xlsx = download(fy, "xlsx")
        if xlsx is None:
            raise ValueError(f"{path.name} has only {len(raw):,} school systems and there is no Excel file")
        print(f"  {path.name} has only {len(raw):,} school systems; reading {xlsx.name}")
        raw = pd.read_excel(xlsx, dtype=str)
    ma = raw[raw["STATE"].str.zfill(2) == CENSUS_MA].copy()
    ma = pd.concat([pd.Series(fy, index=ma.index, name="FY"),
                    ma.apply(lambda s: s if s.name in ID_COLS else pd.to_numeric(s, errors="coerce"))], axis=1)
    return ma


def nces_crosswalk() -> pd.DataFrame:
    """NCES LEA ID -> DESE Org4Code, from several CCD directory years."""
    if NCES_XWALK.exists():
        return pd.read_csv(NCES_XWALK, dtype=str)
    rows = []
    for year in NCES_YEARS:
        for r in json.loads(fetch(NCES_URL.format(year=year)))["results"]:
            sid = str(r.get("state_leaid") or "")
            code = sid.split("-")[-1][:4] if sid else ""
            if code.isdigit():
                rows.append({"NCESID": str(r["leaid"]).zfill(7), "Org4Code": code.zfill(4),
                             "nces_name": r["lea_name"], "directory_year": year})
    xw = (pd.DataFrame(rows).sort_values("directory_year")
          .drop_duplicates("NCESID", keep="last").sort_values("NCESID"))
    clash = xw.groupby("Org4Code")["NCESID"].nunique()
    if (clash > 1).any():
        print(f"  [WARN] DESE codes with more than one NCES ID: {', '.join(clash[clash > 1].index)}")
    xw.to_csv(NCES_XWALK, index=False)
    print(f"  {NCES_XWALK}: {len(xw)} districts from CCD directories {NCES_YEARS}")
    return xw


def town_population() -> pd.DataFrame:
    """Population of each Massachusetts city and town (county subdivision) by fiscal year."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for file, columns in POP_FILES:
        path = RAW_DIR / Path(file).name
        if not path.exists():
            path.write_bytes(fetch(POP_URL + file))
        raw = pd.read_csv(path, dtype=str, encoding="latin-1")
        towns = raw[(raw["SUMLEV"] == "061") & (raw["STATE"] == "25")]
        for fy, col in columns.items():
            rows.append(pd.DataFrame({
                "FY": fy,
                "COUSUB_FIPS": "25" + towns["COUNTY"] + towns["COUSUB"],
                # "Amherst Town city", "Agawam Town city", "Leverett town" -> "Amherst", "Agawam", "Leverett"
                "Town": towns["NAME"].str.replace(r"(\s+Town)?\s+(city|town)$", "", regex=True),
                "County": towns["COUNTY"].map(COUNTIES),
                "Population": pd.to_numeric(towns[col]),
                "Estimate": col,
                "File": file,
            }))
    pop = pd.concat(rows, ignore_index=True)
    counts = pop.groupby("FY")["COUSUB_FIPS"].nunique()
    if (counts != 351).any():
        raise ValueError(f"Expected 351 towns every year: {counts[counts != 351].to_dict()}")
    return pop


def main():
    today = dt.date.today()
    print("[Census] Town population estimates...")
    pop = town_population()
    pop.to_csv(POP_OUT, index=False)
    # A few towns' FIPS codes change between vintages; join on the town name
    print(f"  {POP_OUT}: 351 towns a year, FY{pop.FY.min()}-FY{pop.FY.max()}")
    print("[Census] Downloading school system finances...")
    years = []
    fy = FIRST_YEAR
    while (ma := census_year(fy)) is not None:
        years.append(ma)
        fy += 1
    df = pd.concat(years, ignore_index=True)
    df["NCESID"] = df["NCESID"].str.zfill(7)
    xw = nces_crosswalk()
    df = df.merge(xw[["NCESID", "Org4Code"]], on="NCESID", how="left")
    df["DIST_CODE"] = pd.to_numeric(df["Org4Code"], errors="coerce").mul(10000).astype("Int64")
    unmatched = df[df["Org4Code"].isna()]
    if len(unmatched):
        print(f"  {unmatched['NCESID'].nunique()} Census systems have no DESE code "
              f"(e.g. {', '.join(unmatched['NAME'].drop_duplicates().head(5))})")
    lead = ["FY", "DIST_CODE", "Org4Code", "NCESID", "NAME"]
    df = df[lead + [c for c in df.columns if c not in lead]]
    df.to_csv(OUT, index=False)
    print(f"  {OUT}: {len(df):,} rows, fiscal years {df.FY.min()}-{df.FY.max()}")
    SOURCES_JSON.write_text(json.dumps({
        "source": CENSUS_SOURCE,
        "first_year": int(df.FY.min()), "latest_year": int(df.FY.max()),
        "crosswalk": "NCES Common Core of Data district directory, via educationdata.urban.org",
        "town_population": "Census Bureau subcounty population estimates: " + ", ".join(f for f, _ in POP_FILES),
        "town_population_years": [int(pop.FY.min()), int(pop.FY.max())],
        "downloaded": today.isoformat(),
    }, indent=1) + "\n")
    print(f"  {SOURCES_JSON}")


if __name__ == "__main__":
    main()
