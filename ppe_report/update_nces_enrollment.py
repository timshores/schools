"""
Download each Massachusetts school district's fall enrollment, FY1990 to FY2006, from the NCES
Common Core of Data district directory, and join it to DESE district codes.

DESE's per-pupil reports for FY1990 to FY1996 (data/dese_archive/) give each district's spending
per pupil but not its pupils, so the explorer has no weights for its regional and cohort averages
in those years. NCES counts every district's enrolled pupils each fall, from 1989-90. It is a
different count from DESE's average membership, which the workbooks give from FY1997: a head count
in the fall against full-time equivalents averaged over the year. Against DESE's membership in
FY1997 and FY1998 the median ratio is 1.000, with the middle 90% between 0.94 and 1.09; vocational
and agricultural schools differ most.

Directory years are school years by their fall: 1989 is 1989-90, fiscal year 1990. The district
code is DESE's: state_leaid is the three- or four-digit LEA code (two districts carry three
trailing zeros in the early years, "753000" for Quabbin), checked against nces_ma_districts.csv
where the NCES id is there. Enrollment NCES marks missing (negative codes) or zero is left out.

    python update_nces_enrollment.py   ->  data/nces_ma_enrollment_fy1990_2006.csv
"""

import csv
import json
import urllib.request
from pathlib import Path

DATA_DIR = Path("./data")
URL = "https://educationdata.urban.org/api/v1/school-districts/ccd/directory/{year}/?fips=25"
SOURCE = "https://educationdata.urban.org/documentation/school-districts.html#ccd-directory"
FIRST_FY, LAST_FY = 1990, 2006
OUT = DATA_DIR / "nces_ma_enrollment_fy1990_2006.csv"
XWALK = DATA_DIR / "nces_ma_districts.csv"


def fetch(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def org4(rec, xwalk):
    """DESE's four-digit code for a directory row, or None."""
    s = str(rec.get("state_leaid") or "").split("-")[-1].strip()
    code = None
    if s.isdigit():
        n = int(s)
        code = n // 1000 if n > 9999 and n % 1000 == 0 else n
    known = xwalk.get(str(rec.get("leaid")))
    if known is not None and code is not None and known != code:
        raise ValueError(f"{rec['lea_name']}: state_leaid {s} but crosswalk {known}")
    return code if code is not None else known


def main():
    with open(XWALK, newline="") as f:
        xwalk = {r["NCESID"]: int(r["Org4Code"]) for r in csv.DictReader(f)}
    rows = []
    for fy in range(FIRST_FY, LAST_FY + 1):
        year = fy - 1
        data = fetch(URL.format(year=year))
        if data.get("next"):
            raise RuntimeError(f"{year}: more than one page of results")
        n = 0
        for rec in data["results"]:
            e = rec.get("enrollment")
            code = org4(rec, xwalk)
            if e is None or e <= 0 or code is None:
                continue
            rows.append({"FY": fy, "school_year": f"{year}-{str(year + 1)[2:]}", "Org4Code": f"{code:04d}",
                         "DIST_CODE": f"{code:04d}0000", "NCESID": rec["leaid"], "nces_name": rec["lea_name"],
                         "enrollment": e, "source_url": URL.format(year=year)})
            n += 1
        print(f"  FY{fy}: {n} districts with enrollment")
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["FY"], r["DIST_CODE"])))
    print(f"[NCES] {OUT}: {len(rows):,} rows")


if __name__ == "__main__":
    main()
