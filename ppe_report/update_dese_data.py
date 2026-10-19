"""
Refresh the two DESE sources the report depends on, without touching the
hand-built Excel workbook.

1. District Expenditures by Spending Category (E2C Hub dataset er3w-dyti),
   all school years, written to data/dese_expenditures.csv in the column
   layout of the workbook's "District Expend by Category" sheet.
2. Chapter 70 District Profiles (DESE profile.xlsx, DataC70 sheet), written to
   data/dese_c70_profile.csv in the column layout of the workbook's
   "profile_DataC70" sheet.

school_shared.load_data() prefers these CSVs when they exist. After running
this script, rebuild the cache:

    python update_dese_data.py
    python generate_report.py --force-recompute

The downloaded profile.xlsx is kept in data/ so a run can be reproduced from
the same file.
"""

import datetime as dt
import io
import urllib.request
from pathlib import Path

import pandas as pd

DATA_DIR = Path("./data")
EXPEND_URL = (
    "https://educationtocareer.data.mass.gov/resource/er3w-dyti.csv"
    "?$limit=500000&$order=sy,dist_code,ind_cat,ind_subcat"
)
EXPEND_SOURCE = (
    "https://educationtocareer.data.mass.gov/Finance-and-Budget/"
    "District-Expenditures-by-Spending-Category/er3w-dyti/about_data"
)
C70_URL = "https://www.doe.mass.edu/finance/chapter70/profile.xlsx"
C70_SOURCE = (
    "Chapter 70 District Profiles: "
    "https://www.doe.mass.edu/finance/chapter70/default.html"
)
EXPEND_OUT = DATA_DIR / "dese_expenditures.csv"
C70_XLSX = DATA_DIR / "dese_c70_profile.xlsx"
C70_OUT = DATA_DIR / "dese_c70_profile.csv"
DOWNLOADED_BY = "update_dese_data.py"


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def refresh_expenditures(today: dt.date) -> pd.DataFrame:
    raw = pd.read_csv(io.BytesIO(fetch(EXPEND_URL)), dtype={"dist_code": str})
    df = pd.DataFrame({
        "SY": raw["sy"].astype(int),
        "DIST_CODE": raw["dist_code"].str.zfill(8),
        "DIST_NAME": raw["dist_name"].str.strip(),
        "IND_CAT": raw["ind_cat"].str.strip(),
        "IND_SUBCAT": raw["ind_subcat"].str.strip(),
        "IND_VALUE": pd.to_numeric(raw["ind_value"], errors="coerce"),
        "IND_VALUE_TYPE": raw["ind_value_type"],
    })
    df["Source_and_Data_Dictionary"] = EXPEND_SOURCE
    df["Last_Updated"] = ""  # the API does not expose the dataset's update date per row
    df["Download_Date"] = today.isoformat()
    df["Download_By"] = DOWNLOADED_BY
    df.to_csv(EXPEND_OUT, index=False)
    print(f"  {EXPEND_OUT}: {len(df):,} rows, school years {df.SY.min()}-{df.SY.max()}")
    return df


def refresh_c70() -> pd.DataFrame:
    C70_XLSX.write_bytes(fetch(C70_URL))
    sheet = pd.read_excel(C70_XLSX, sheet_name="DataC70", header=None, dtype=object)

    # Rows 0-3 are DESE's notes and column headers; data starts at row 4.
    # Columns 0-12 hold the profile data; columns 15-18 hold a code-to-name table.
    data = sheet.iloc[4:, 0:13].copy()
    data.columns = ["Org4Codefy", "Org4Code", "LEANumCode", "fy", "distfoundenro",
                    "distfoundbudget", "distrlc", "c70aid", "rqdnss", "rqdnss2",
                    "actualNSS", "index", "c70aid.1"]
    data = data[data["fy"].notna()].copy()
    data["Org4Code"] = data["Org4Code"].astype(str).str.zfill(4)

    names = sheet.iloc[6:, [15, 18]].dropna()
    names.columns = ["Org4Code", "District"]
    name_map = dict(zip(names["Org4Code"].astype(str).str.zfill(4), names["District"].astype(str)))
    name_map["0000"] = "STATE TOTAL"
    data["District"] = data["Org4Code"].map(name_map)
    missing = data["District"].isna().sum()
    if missing:
        raise ValueError(f"{missing} Chapter 70 rows have no district name")

    out = pd.DataFrame({
        "Org4Code": data["Org4Code"],
        "District": data["District"],
        "LEANumCode": data["LEANumCode"],
        "fy": data["fy"].astype(int),
        "distfoundenro": data["distfoundenro"],
        "distfoundbudget": data["distfoundbudget"],
        "distrlc": data["distrlc"],
        "c70aid": data["c70aid"],
        "Checkc70aid-per-fe": None,
        "rqdnss": data["rqdnss"],
        "rqdnss2": data["rqdnss2"],
        "Checkrqdnss2-per-fe": None,
        "actualNSS": data["actualNSS"],
        "CheckactualNSS-per-fe": None,
        "index": data["index"],
        "c70aid.1": data["c70aid.1"],
        "Source": C70_SOURCE,
    })
    out.to_csv(C70_OUT, index=False)
    print(f"  {C70_OUT}: {len(out):,} rows, fiscal years {out.fy.min()}-{out.fy.max()}")
    return out


def main():
    today = dt.date.today()
    print("[DESE] Downloading expenditures by spending category...")
    refresh_expenditures(today)
    print("[DESE] Downloading Chapter 70 district profiles...")
    refresh_c70()
    print("[DESE] Done. Rebuild with: python generate_report.py --force-recompute")


if __name__ == "__main__":
    main()
