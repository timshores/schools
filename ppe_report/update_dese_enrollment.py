"""
Download DESE's enrollment by race and ethnicity, gender, and selected
populations for every Massachusetts district, for the explorer's students
panels.

E2C Hub dataset t8td-gens (Enrollment: Grade, Race/Ethnicity, Gender, and
Selected Populations), the figures DESE's profiles show: each district's
students on October 1 of each school year, SY1992 on. A school year is its
fiscal year: SY2026 is 2025-26, counted October 1, 2025. Race and gender are
percents of all students; the selected populations are counts and percents.
Low income through SY2014, economically disadvantaged SY2015-SY2021, and a new
low income from SY2022 are three measures, kept apart.

Writes data/dese_enrollment.csv, one row per district and year, and
data/dese_enrollment_sources.json.

    python update_dese_enrollment.py
"""

import datetime as dt
import io
import json
import urllib.request
from pathlib import Path

import pandas as pd

DATA_DIR = Path("./data")
URL = ("https://educationtocareer.data.mass.gov/resource/t8td-gens.csv?$limit=100000"
       "&$where=org_type='District'&$order=sy,dist_code")
META_URL = "https://educationtocareer.data.mass.gov/api/views/t8td-gens.json"
SOURCE = ("https://educationtocareer.data.mass.gov/Students-and-Teachers/"
          "Enrollment-Grade-Race-Ethnicity-Gender-and-Selected-Populations/t8td-gens/about_data")
OUT = DATA_DIR / "dese_enrollment.csv"
SOURCES_JSON = DATA_DIR / "dese_enrollment_sources.json"
RACE = ["wh_pct", "baa_pct", "as_pct", "hl_pct", "aian_pct", "nhpi_pct", "mnhl_pct"]
GENDER = ["fe_pct", "ma_pct", "nb_pct"]
GROUPS = ["swd", "el", "flne", "li", "ecd", "hn"]


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def main():
    meta = json.loads(fetch(META_URL))
    raw = pd.read_csv(io.BytesIO(fetch(URL)), dtype={"dist_code": str, "org_code": str})
    cols = ["total_cnt", *RACE, *GENDER, *[f"{g}_{k}" for g in GROUPS for k in ("cnt", "pct")]]
    missing = [c for c in cols + ["sy", "dist_code", "dist_name"] if c not in raw.columns]
    if missing:
        raise ValueError(f"The dataset has changed; missing {missing}")
    df = raw[["sy", "dist_code", "dist_name", *cols]].rename(columns={"sy": "FY", "dist_code": "DIST_CODE", "dist_name": "DIST_NAME"})
    df["DIST_CODE"] = df["DIST_CODE"].str.zfill(8)
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # A year with no students has zeros for its percents, which aren't figures
    empty = ~(df["total_cnt"] > 0)
    df.loc[empty, [c for c in cols if c != "total_cnt"]] = None
    # Each year's race groups are every student, so their percents add to one; DESE's file has a stray row or two
    race = df[RACE].sum(axis=1, min_count=1)
    off = race.notna() & ((race - 1).abs() > 0.02)
    for _, r in df[off].iterrows():
        print(f"  [WARN] SY{r['FY']} {r['DIST_NAME']}: race percents add to {race[_]:.1%}")
    if off.sum() > 5:
        raise ValueError(f"{int(off.sum())} rows' race percents don't add to 100%: the dataset has changed")
    if df.duplicated(["FY", "DIST_CODE"]).any():
        raise ValueError("A district appears twice in a year")
    df.sort_values(["FY", "DIST_CODE"]).to_csv(OUT, index=False)
    print(f"  {OUT}: {len(df):,} rows, {df['DIST_CODE'].nunique()} districts, SY{df['FY'].min()}-SY{df['FY'].max()}")
    SOURCES_JSON.write_text(json.dumps({
        "source": SOURCE, "url": URL,
        "last_updated": dt.date.fromtimestamp(meta["rowsUpdatedAt"]).isoformat(),
        "downloaded": dt.date.today().isoformat(),
    }, indent=1) + "\n")
    print(f"  {SOURCES_JSON}")


if __name__ == "__main__":
    main()
