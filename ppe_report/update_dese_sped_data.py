"""
Download DESE's special education spending and Circuit Breaker reimbursements
for every Massachusetts district, for the explorer's special education panels.

1. Special Education Direct Expenditure Trends (RADAR's spreadsheet, the "data"
   sheet): each district's direct special education spending, FY2008 on, in
   eight categories, with its total spending, written to
   data/dese_sped_expenditures.csv. DESE posts the workbook on the RADAR page
   and dates it in its header; a new year replaces the file at the same URL.
2. Special Education Circuit Breaker Reimbursements (E2C Hub dataset
   ab34-d3ma): what the state paid each district, FY2006 on, and the students
   and costs it claimed, written to data/dese_circuit_breaker.csv. FY is the
   year the state paid: quarterly payments reimburse the prior year's costs,
   extraordinary relief the same year's.

    python update_dese_sped_data.py
"""

import datetime as dt
import io
import json
import urllib.request
from pathlib import Path

import pandas as pd

DATA_DIR = Path("./data")
SPED_URL = "https://www.doe.mass.edu/research/radar/direct-expenditure-trends.xlsx"
SPED_SOURCE = "Special Education Direct Expenditure Trends: https://www.doe.mass.edu/research/radar/"
CB_URL = "https://educationtocareer.data.mass.gov/resource/ab34-d3ma.csv?$limit=100000&$order=fy,dist_code"
CB_META_URL = "https://educationtocareer.data.mass.gov/api/views/ab34-d3ma.json"
CB_SOURCE = ("https://educationtocareer.data.mass.gov/Finance-and-Budget/"
             "Special-Education-Circuit-Breaker-Reimbursements/ab34-d3ma/about_data")
SPED_XLSX = DATA_DIR / "dese_sped_direct_expenditure_trends.xlsx"
SPED_OUT = DATA_DIR / "dese_sped_expenditures.csv"
CB_OUT = DATA_DIR / "dese_circuit_breaker.csv"
SOURCES_JSON = DATA_DIR / "dese_sped_sources.json"

# The data sheet's columns, as headed (E to L are the categories, M their sum, N all spending)
SPED_COLUMNS = {
    "Teaching": "teaching",
    "Other Instructional": "other_instructional",
    "In-District Transportation": "in_transport",
    "Mass. Public Schools and Collaboratives": "ma_public",
    "Mass Private and Out-of-State Schools": "private",
    "Out-of-District Transportation": "ood_transport",
    "Non-public Health Services": "health",
    "Spending from Grants and Revolving Funds": "grants",
    "Combined Special Ed Expenditures (sum of E through L)": "sped_total",
    "Total Expenditures": "total_exp",
}
CB_MONEY = ["tot_elig_expenses", "threshold_amt", "net_elig_instr_tuit_costs", "net_elig_trans_costs", "tot_net_claim",
            "reimb_instr_tuit", "reimb_spec_ind_instr_tuit", "reimb_trans", "reimb_spec_ind_trans", "prior_year_adj",
            "tot_qtly_payment", "extra_relief_payment", "addl_suppl_payment"]


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def refresh_sped() -> str:
    """Download the workbook; return the date in its header (YYYY-MM-DD)."""
    SPED_XLSX.write_bytes(fetch(SPED_URL))
    head = pd.read_excel(SPED_XLSX, sheet_name="data", header=None, nrows=3)
    dated = next(v for v in head.values.ravel() if isinstance(v, (dt.datetime, pd.Timestamp)))
    raw = pd.read_excel(SPED_XLSX, sheet_name="data", header=5, dtype={"LEA": str})
    missing = [c for c in [*SPED_COLUMNS, "LEA", "District Name", "Fiscal Year"] if c not in raw.columns]
    if missing:
        raise ValueError(f"The data sheet has changed; missing {missing}")
    raw = raw[raw["LEA"].str.fullmatch(r"\d+", na=False)]
    df = pd.DataFrame({"DIST_CODE": raw["LEA"].str.zfill(4) + "0000", "DIST_NAME": raw["District Name"].str.strip(),
                       "FY": raw["Fiscal Year"].astype(int)})
    for col, key in SPED_COLUMNS.items():
        df[key] = pd.to_numeric(raw[col], errors="coerce").round(2)
    # The sum column is the categories' sum; a mismatch means the sheet's layout moved
    parts = df[list(SPED_COLUMNS.values())[:8]].sum(axis=1)
    off = (parts - df["sped_total"]).abs() > 1
    if off.any():
        raise ValueError(f"{int(off.sum())} rows' categories don't add to their total, e.g. {df[off].iloc[0].to_dict()}")
    df.sort_values(["FY", "DIST_CODE"]).to_csv(SPED_OUT, index=False)
    print(f"  {SPED_OUT}: {len(df):,} rows, {df['DIST_CODE'].nunique()} districts, FY{df['FY'].min()}-FY{df['FY'].max()}")
    return pd.Timestamp(dated).date().isoformat()


def refresh_circuit_breaker() -> str:
    """Download the dataset; return DESE's rows-updated date (YYYY-MM-DD)."""
    meta = json.loads(fetch(CB_META_URL))
    raw = pd.read_csv(io.BytesIO(fetch(CB_URL)), dtype={"dist_code": str})
    df = raw.rename(columns={"fy": "FY", "dist_code": "DIST_CODE", "dist_name": "DIST_NAME",
                             "elig_stu_claim_cnt": "students"})
    df["DIST_CODE"] = df["DIST_CODE"].str.zfill(8)
    for c in CB_MONEY + ["students"]:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    df = df[["FY", "DIST_CODE", "DIST_NAME", "students", *CB_MONEY, "comments"]]
    df.sort_values(["FY", "DIST_CODE"]).to_csv(CB_OUT, index=False)
    print(f"  {CB_OUT}: {len(df):,} rows, FY{df['FY'].min()}-FY{df['FY'].max()}")
    return dt.date.fromtimestamp(meta["rowsUpdatedAt"]).isoformat()


def main():
    today = dt.date.today()
    sped_date = refresh_sped()
    cb_date = refresh_circuit_breaker()
    SOURCES_JSON.write_text(json.dumps({
        "sped": {"source": SPED_SOURCE, "url": SPED_URL, "file_date": sped_date},
        "circuit_breaker": {"source": CB_SOURCE, "url": CB_URL, "last_updated": cb_date},
        "downloaded": today.isoformat(),
    }, indent=1) + "\n")
    print(f"  {SOURCES_JSON}")


if __name__ == "__main__":
    main()
