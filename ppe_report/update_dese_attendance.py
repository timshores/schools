"""
Download DESE's student attendance for every Massachusetts district and the
state, for the explorer.

E2C Hub dataset ak6h-9k7x (Student Attendance), the figures DESE's profiles
show: end-of-year attendance for students in grades PK-12 enrolled at least 20
days, SY2018 on, for all students and fourteen student groups. A school year
is its fiscal year: SY2026 is 2025-26. DESE counts a student as chronically
absent who misses 10 percent or more of the days enrolled, whatever the
reason. The dataset also has a March snapshot from SY2021, which this leaves
out: a year's figures are its end-of-year ones.

Economically disadvantaged (SY2018-SY2021) and low income (from SY2022) are
two measures, kept apart as DESE labels them.

Writes data/dese_attendance.csv, one row per district (or the state), year and
student group, and data/dese_attendance_sources.json.

    python update_dese_attendance.py
"""

import datetime as dt
import io
import json
import urllib.request
from pathlib import Path

import pandas as pd

DATA_DIR = Path("./data")
URL = ("https://educationtocareer.data.mass.gov/resource/ak6h-9k7x.csv?$limit=200000"
       "&$where=org_type in ('District','State') AND attend_period='End of Year'&$order=sy,dist_code,stu_grp")
META_URL = "https://educationtocareer.data.mass.gov/api/views/ak6h-9k7x.json"
SOURCE = "https://educationtocareer.data.mass.gov/Students-and-Teachers/Student-Attendance/ak6h-9k7x/about_data"
OUT = DATA_DIR / "dese_attendance.csv"
SOURCES_JSON = DATA_DIR / "dese_attendance_sources.json"
# attendance rate; average days absent per student; share absent 10 or more
# days; share chronically absent (10% or more of days, and 20% or more); share
# with 10 or more unexcused absences
MEASURES = ["attend_rate", "cnt_avg_abs", "pct_abs_10_days", "pct_chron_abs_10", "pct_chron_abs_20", "pct_unexc_10_days"]


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url.replace(" ", "%20"), headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def main():
    meta = json.loads(fetch(META_URL))
    raw = pd.read_csv(io.BytesIO(fetch(URL)), dtype={"dist_code": str, "org_code": str})
    need = ["sy", "dist_code", "dist_name", "org_type", "stu_grp", *MEASURES]
    missing = [c for c in need if c not in raw.columns]
    if missing:
        raise ValueError(f"The dataset has changed; missing {missing}")
    df = raw[need].rename(columns={"sy": "FY", "dist_code": "DIST_CODE", "dist_name": "DIST_NAME",
                                   "org_type": "ORG_TYPE", "stu_grp": "GROUP"})
    df["DIST_CODE"] = df["DIST_CODE"].str.zfill(8)
    for c in MEASURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df.duplicated(["FY", "DIST_CODE", "GROUP"]).any():
        raise ValueError("A district's student group appears twice in a year")
    every = df[df["GROUP"] == "All Students"]
    if every["pct_chron_abs_10"].max() > 1.0001 or every["attend_rate"].max() > 1.0001:
        raise ValueError("Rates are no longer fractions of one: the dataset has changed")
    df.sort_values(["FY", "ORG_TYPE", "DIST_CODE", "GROUP"]).to_csv(OUT, index=False)
    d = df[df["ORG_TYPE"] == "District"]
    print(f"  {OUT}: {len(df):,} rows, {d['DIST_CODE'].nunique()} districts, {df['GROUP'].nunique()} student groups, "
          f"SY{df['FY'].min()}-SY{df['FY'].max()}")
    SOURCES_JSON.write_text(json.dumps({
        "source": SOURCE, "url": URL,
        "last_updated": dt.date.fromtimestamp(meta["rowsUpdatedAt"]).isoformat(),
        "downloaded": dt.date.today().isoformat(),
    }, indent=1) + "\n")
    print(f"  {SOURCES_JSON}")


if __name__ == "__main__":
    main()
