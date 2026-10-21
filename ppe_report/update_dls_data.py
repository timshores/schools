"""
Download DLS Schedule A general fund revenues and expenditures for every
Massachusetts municipality, FY2002 onward.

The Division of Local Services (Department of Revenue) publishes each
municipality's year-end Schedule A in its Municipal Databank. The DLS Gateway
report "Schedule A General Fund" exports one fiscal year at a time (several
years in one request returns an error page), so this asks for every
municipality, once per year and amount type, and writes one long table:

    data/dls_schedule_a_general_fund.csv
        DOR_CODE, Municipality, FY, Type (Revenues / Expenditures), Category, Amount

Expenditure categories are Schedule A's functions (General Government,
Public Safety, Education, ... Debt Service, Total Expenditures); revenue
categories are its sources (Taxes, State Revenue, ... Total Revenues).
Schedule A is due November 30 for the year that ended in June, so the latest
year covers only the municipalities that have filed; see the FY counts printed
at the end.

Education includes a town's assessments from regional school districts as
well as its own school department, so it does not line up with any one
district's spending.

    python update_dls_data.py
"""

import datetime as dt
import io
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

DATA_DIR = Path("./data")
BASE = "https://dls-gw.dor.state.ma.us/reports/rdPage.aspx"
FORM = BASE + "?rdReport=ScheduleA.GeneralFund&rdSubReport=True"
EXPORT = (BASE + "?rdReport=ScheduleA.GeneralFund&rdReportFormat=NativeExcel&rdExportTableID=xtGenFund"
          "&rdExportFilename=GenFund&rdExcelOutputFormat=Excel2007")
SOURCE = ("DLS Municipal Databank, Schedule A General Fund: "
          "https://dls-gw.dor.state.ma.us/reports/rdPage.aspx?rdReport=ScheduleA.GenFund_MAIN")
TYPES = ["Expenditures", "Revenues"]
OUT = DATA_DIR / "dls_schedule_a_general_fund.csv"
SOURCES_JSON = DATA_DIR / "dls_sources.json"
PAUSE = 0.5  # seconds between requests to the Gateway


def fetch(url: str, data: list[tuple[str, str]] | None = None) -> bytes:
    body = urllib.parse.urlencode(data).encode() if data else None
    req = urllib.request.Request(url, data=body, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def form_options() -> tuple[list[str], list[int], str]:
    """Municipality codes, fiscal years, and the 'data current as of' date, from the report's form."""
    page = fetch(FORM).decode("utf-8", errors="ignore")
    munis = re.findall(r'name="iclMuni"[^>]*value="(\d+)"', page)
    years = sorted(int(y) for y in re.findall(r'name="islYear"[^>]*value="(\d{4})"', page))
    current = re.search(r"current as of\s*(\d{1,2}/\d{1,2}/\d{4})", re.sub(r"<[^>]+>", " ", page))
    as_of = dt.datetime.strptime(current.group(1), "%m/%d/%Y").date().isoformat() if current else ""
    if not munis or not years:
        raise ValueError("The Schedule A form lists no municipalities or years; has the Gateway page changed?")
    return munis, years, as_of


def export(munis: list[str], year: int, kind: str) -> pd.DataFrame:
    """One year of one amount type for every municipality, long format."""
    raw = fetch(EXPORT, [("iclMuni", m) for m in munis] + [("islYear", str(year)), ("islAmountType", kind)])
    if not raw.startswith(b"PK"):  # an .xlsx is a zip; anything else is the Gateway's error page
        raise ValueError(f"FY{year} {kind}: the Gateway returned an error page, not a spreadsheet")
    wide = pd.read_excel(io.BytesIO(raw), dtype={"DOR Code": str})
    wide = wide[wide["DOR Code"].str.fullmatch(r"\d+", na=False)]  # drops the "Totals:" row
    long = wide.melt(id_vars=["DOR Code", "Municipality", "Fiscal Year"], var_name="Category", value_name="Amount")
    return pd.DataFrame({
        "DOR_CODE": long["DOR Code"].str.zfill(3),
        "Municipality": long["Municipality"].str.strip(),
        "FY": long["Fiscal Year"].astype(int),
        "Type": kind,
        "Category": long["Category"].str.strip(),
        "Amount": pd.to_numeric(long["Amount"], errors="coerce"),
    })


def main():
    today = dt.date.today()
    print("[DLS] Reading the Schedule A form...")
    munis, years, as_of = form_options()
    print(f"  {len(munis)} municipalities, FY{years[0]}-FY{years[-1]}, data current as of {as_of}")
    frames = []
    for year in years:
        for kind in TYPES:
            frames.append(export(munis, year, kind))
            time.sleep(PAUSE)
        print(f"  FY{year}", end="", flush=True)
    print()
    df = pd.concat(frames, ignore_index=True).sort_values(["Type", "FY", "DOR_CODE", "Category"])
    df.to_csv(OUT, index=False)
    totals = df[df["Category"].isin(["Total Expenditures", "Total Revenues"])]
    filed = totals[totals["Amount"] > 0].groupby(["FY", "Type"])["DOR_CODE"].nunique().unstack()
    print(f"  {OUT}: {len(df):,} rows")
    print("  Municipalities with a nonzero total, by year:")
    print("    " + filed.to_string().replace("\n", "\n    "))
    SOURCES_JSON.write_text(json.dumps({
        "source": SOURCE,
        "data_current_as_of": as_of,
        "first_year": years[0], "latest_year": years[-1],
        "downloaded": today.isoformat(),
    }, indent=1) + "\n")
    print(f"  {SOURCES_JSON}")


if __name__ == "__main__":
    main()
