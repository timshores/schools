"""
Download price indexes for the inflation addendum and write fiscal-year
averages to data/inflation_indices.csv.

Massachusetts school fiscal years run July 1 to June 30, so FY2024 averages
July 2023 through June 2024. DESE's school year (SY) 2024 is the same period.

Series:
- cpi_u_northeast: BLS CPI-U, Northeast region, all items, not seasonally
  adjusted (CUUR0100SA0). Monthly. The report's earlier data/cpi_annual.csv is
  this series' calendar-year average.
- cpi_u_us: BLS CPI-U, U.S. city average, all items, NSA (CUUR0000SA0). Monthly.
- sl_gov_deflator: BEA implicit price deflator for state and local government
  consumption expenditures and gross investment (NIPA Table 1.1.9, line 26,
  series A829RD). Quarterly. Chapter 70's foundation inflation index is built
  on this deflator (M.G.L. c.70, s.2, capped at 4.5% a year).

A fiscal year is written only when all of its months (or quarters) are
published. BLS published no October 2025 CPI because of the federal
shutdown, so FY2026 CPI averages are left blank rather than estimated.
"""

import csv
import datetime as dt
import io
import json
import urllib.request
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path("./data")
OUT = DATA_DIR / "inflation_indices.csv"
BLS_URL = "https://api.bls.gov/publicAPI/v1/timeseries/data/"
BLS_SERIES = {"cpi_u_northeast": "CUUR0100SA0", "cpi_u_us": "CUUR0000SA0"}
BEA_URL = "https://apps.bea.gov/national/Release/TXT/NipaDataQ.txt"
BEA_SERIES = {"sl_gov_deflator": "A829RD"}
FIRST_FY, LAST_FY = 2008, 2027


def _get(url, data=None):
    headers = {"User-Agent": "Mozilla/5.0"}
    if data is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(data).encode()
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def bls_monthly():
    """{name: {(year, month): value}}. The keyless v1 API serves 10 years per call."""
    out = defaultdict(dict)
    this_year = dt.date.today().year  # the API rejects future years
    for start in range(FIRST_FY - 1, this_year + 1, 10):
        body = {"seriesid": list(BLS_SERIES.values()),
                "startyear": str(start), "endyear": str(min(start + 9, this_year))}
        resp = json.loads(_get(BLS_URL, body))
        if resp.get("status") != "REQUEST_SUCCEEDED":
            raise RuntimeError(f"BLS: {resp.get('status')} {resp.get('message')}")
        for s in resp["Results"]["series"]:
            name = next(k for k, v in BLS_SERIES.items() if v == s["seriesID"])
            for d in s["data"]:
                if d["period"].startswith("M") and d["period"] != "M13" and d["value"] != "-":
                    out[name][(int(d["year"]), int(d["period"][1:]))] = float(d["value"])
    return out


def bea_quarterly():
    """{name: {(year, quarter): value}} from BEA's bulk NIPA file."""
    text = _get(BEA_URL).decode("utf-8")
    out = defaultdict(dict)
    for row in csv.reader(io.StringIO(text)):
        if len(row) == 3 and row[0] in BEA_SERIES.values():
            name = next(k for k, v in BEA_SERIES.items() if v == row[0])
            year, q = row[1].split("Q")
            out[name][(int(year), int(q))] = float(row[2].replace(",", ""))
    return out


def fy_average_monthly(series, fy):
    months = [(fy - 1, m) for m in range(7, 13)] + [(fy, m) for m in range(1, 7)]
    vals = [series.get(k) for k in months]
    return sum(vals) / 12 if all(v is not None for v in vals) else None


def fy_average_quarterly(series, fy):
    quarters = [(fy - 1, 3), (fy - 1, 4), (fy, 1), (fy, 2)]
    vals = [series.get(k) for k in quarters]
    return sum(vals) / 4 if all(v is not None for v in vals) else None


def main():
    monthly = bls_monthly()
    quarterly = bea_quarterly()
    names = list(BLS_SERIES) + list(BEA_SERIES)
    rows = []
    for fy in range(FIRST_FY, LAST_FY + 1):
        row = {"fiscal_year": fy}
        for name in BLS_SERIES:
            row[name] = fy_average_monthly(monthly[name], fy)
        for name in BEA_SERIES:
            row[name] = fy_average_quarterly(quarterly[name], fy)
        if any(row[n] is not None for n in names):
            rows.append(row)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["fiscal_year"] + names)
        w.writeheader()
        for row in rows:
            w.writerow({k: ("" if v is None else (round(v, 3) if k != "fiscal_year" else v))
                        for k, v in row.items()})
    print(f"[Inflation] {OUT}: fiscal years {rows[0]['fiscal_year']}-{rows[-1]['fiscal_year']}")


if __name__ == "__main__":
    main()
