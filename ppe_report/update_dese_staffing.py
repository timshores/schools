"""
Download DESE's educator reports for every Massachusetts district, for the
explorer's teachers and staff panels.

Seven of the statewide reports behind DESE's School and District Profiles
(profiles.doe.mass.edu/statereport/), each read for every year it offers:

  teacherdata.aspx                    teachers (FTE), percent licensed, students per teacher
  teachersalaries.aspx                salary total, average salary, FTE
  programareastaffing.aspx            teachers (FTE) by program area
  staffingRetentionRates.aspx         superintendents, principals and teachers retained
  teacherbyracegender.aspx            all staff (FTE) by race and ethnicity, and by gender
  agestaffing.aspx                    all staff (FTE) by age group
  educatorevaluationperformance.aspx  all educators' evaluation ratings

A school year is its fiscal year: 2026 is 2025-26. The retention report's year
is the year the staff were retained into. Each report is a form: one GET for
its fields, then one POST for each year. A cell DESE leaves blank or prints as
"-" (too few people to report) is empty here.

Writes data/dese_staffing.csv, one row per district and year, and
data/dese_staffing_sources.json.

    python update_dese_staffing.py
"""

import datetime as dt
import html
import http.cookiejar
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

DATA_DIR = Path("./data")
BASE = "https://profiles.doe.mass.edu/statereport/"
OUT = DATA_DIR / "dese_staffing.csv"
SOURCES_JSON = DATA_DIR / "dese_staffing_sources.json"
FIELD = "ctl00$ContentPlaceHolder1$"

# report page: (fields to set besides the year, {column header as DESE prints it: our column})
# A header that changed over the years is listed under each of its names.
REPORTS = {
    "teacherdata.aspx": ({"ddSubject": "42"}, {
        "Total # of Teachers (FTE)": "teachers_fte", "Total # of Teachers": "teachers_fte",
        "% of Teachers Licensed": "licensed_pct", "% of Teachers Licensed in Teaching Assignment": "licensed_pct",
        "Student / Teacher Ratio": "students_per_teacher",
        "Percent of Experienced Teachers": "experienced_pct",
        "Percent of Teachers without Waiver or Provisional License": "no_waiver_pct",
        "Percent Teaching In-Field": "in_field_pct"}),
    "teachersalaries.aspx": ({}, {"Salary Totals": "salary_total", "Average Salary": "salary_avg", "FTE Count": "salary_fte"}),
    "programareastaffing.aspx": ({"ddDisplay": "Number"}, {
        "General Education (#)": "pa_general", "Special Education (#)": "pa_sped",
        "Career Technical Education (#)": "pa_cte", "English Learners (#)": "pa_el", "FTE Count": "pa_fte"}),
    "staffingRetentionRates.aspx": ({}, None),  # two header rows: read by position
    "teacherbyracegender.aspx": ({"ddDisplay": "Number", "ddClassification": "1100"}, {
        "American Indian or Alaska Native (#)": "staff_aian", "Asian (#)": "staff_as",
        "Black or African American (#)": "staff_baa", "Hispanic or Latino (#)": "staff_hl",
        "Multi-Race, Not Hispanic or Latino (#)": "staff_mnhl",
        "Native Hawaiian or Other Pacific Islander (#)": "staff_nhpi", "White (#)": "staff_wh",
        "Female (#)": "staff_fe", "Male (#)": "staff_ma", "FTE Count": "staff_fte"}),
    "agestaffing.aspx": ({"ddDisplay": "Number", "ddJobClassification": "1100"}, {
        "<26 yrs (#)": "age_u26", "26-32 yrs (#)": "age_26_32", "33-40 yrs (#)": "age_33_40",
        "41-48 yrs (#)": "age_41_48", "49-56 yrs (#)": "age_49_56", "57-64 yrs (#)": "age_57_64",
        "Over 64 yrs (#)": "age_65", "FTE Count": "age_fte"}),
    "educatorevaluationperformance.aspx": ({"ddGroupType": "AS"}, {
        "# of Educators to be Evaluated": "eval_due", "# Evaluated": "eval_done",
        "% Exemplary": "eval_exemplary", "% Proficient": "eval_proficient",
        "% Needs Improvement": "eval_needs", "% Unsatisfactory": "eval_unsat"}),
}
RETENTION = ["supt_n", "supt_kept", None, "prin_n", "prin_kept", None, "tch_n", "tch_kept", None]

opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
opener.addheaders = [("User-Agent", "Mozilla/5.0")]


def fetch(page: str, data=None) -> str:
    body = urllib.parse.urlencode(data).encode() if data else None
    for attempt in range(3):
        try:
            return opener.open(BASE + page, body, timeout=120).read().decode("utf-8", "ignore")
        except OSError:
            if attempt == 2:
                raise
            time.sleep(5)
    raise AssertionError


def selects(page_html: str) -> dict:
    """{field name: [(value, selected)]} for each drop-down"""
    return {m.group(1): [(v, "selected" in pre) for pre, v in re.findall(r'<option([^>]*?)value="([^"]*)"', m.group(0))]
            for m in re.finditer(r'<select[^>]*name="([^"]+)".*?</select>', page_html, re.S)}


def table(page_html: str) -> list:
    cell = lambda r: [html.unescape(re.sub(r"<[^>]+>", "", t)).strip()
                      for t in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", r, re.S)]
    return [c for c in (cell(r) for r in re.findall(r"<tr[^>]*>(.*?)</tr>", page_html, re.S)) if c]


def number(text: str):
    """'$907,725' -> 907725.0, '8.6 to 1' -> 8.6; '' and '-' -> None"""
    t = text.replace("$", "").replace(",", "").replace("to 1", "").strip()
    try:
        return float(t)
    except ValueError:
        return None


def report(page: str, fixed: dict, columns) -> pd.DataFrame:
    first = fetch(page)
    form = dict(re.findall(r'<input type="hidden" name="([^"]+)" id="[^"]*" value="([^"]*)"', first))
    drop = selects(first)
    for name, options in drop.items():
        form[name] = next((v for v, sel in options if sel), options[0][0])
    for name, value in fixed.items():
        if FIELD + name not in drop:
            raise ValueError(f"{page}: no field {name}; the report has changed")
        form[FIELD + name] = value
    for name, value in re.findall(r'<input type="submit" name="([^"]+)" value="([^"]*)"', first):
        if "ViewReport" in name:
            form[name] = value
    rows = []
    for year, _ in drop[FIELD + "ddYear"]:
        rs = table(fetch(page, {**form, FIELD + "ddYear": year}))
        head = rs[0]
        if head[:2] != ["District Name", "District Code"]:
            raise ValueError(f"{page} {year}: unexpected table {head[:3]}")
        if columns is None:
            at = dict(zip(range(2, 11), RETENTION))
        else:
            at = {i: columns[h] for i, h in enumerate(head) if h in columns}
            if not at:
                raise ValueError(f"{page} {year}: none of the expected columns in {head}")
        n = 0
        for r in rs[1:]:
            if len(r) < 3 or not re.fullmatch(r"\d{8}", r[1]) or r[1] == "00000000":
                continue
            rows.append({"FY": int(year), "DIST_CODE": r[1], "DIST_NAME": r[0],
                         **{c: number(r[i]) for i, c in at.items() if c and i < len(r)}})
            n += 1
        print(f"  {page} {year}: {n} districts")
        time.sleep(0.3)
    return pd.DataFrame(rows)


def main():
    merged = None
    for page, (fixed, columns) in REPORTS.items():
        df = report(page, fixed, columns)
        # A code DESE prints against two districts in one year can't be told apart: both rows go (the salary
        # report's 2012-13 has two such pairs, vocational and agricultural schools)
        twice = df.duplicated(["FY", "DIST_CODE"], keep=False)
        for _, r in df[twice].iterrows():
            print(f"  [WARN] {page} {r['FY']}: code {r['DIST_CODE']} appears twice ({r['DIST_NAME']}); dropped")
        if twice.sum() > 6:
            raise ValueError(f"{page}: {int(twice.sum())} rows share a district code and year: the report has changed")
        df = df[~twice]
        merged = df if merged is None else merged.merge(df.drop(columns="DIST_NAME"), on=["FY", "DIST_CODE"], how="outer")
        names = df.drop_duplicates("DIST_CODE", keep="last").set_index("DIST_CODE")["DIST_NAME"]
        merged["DIST_NAME"] = merged["DIST_NAME"].fillna(merged["DIST_CODE"].map(names))
    merged = merged.sort_values(["FY", "DIST_CODE"])
    merged.to_csv(OUT, index=False)
    print(f"  {OUT}: {len(merged):,} rows, {merged['DIST_CODE'].nunique()} districts, FY{merged['FY'].min()}-FY{merged['FY'].max()}")
    SOURCES_JSON.write_text(json.dumps({
        "source": BASE + "#Educators",
        "reports": {p: BASE + p for p in REPORTS},
        "downloaded": dt.date.today().isoformat(),
    }, indent=1) + "\n")
    print(f"  {SOURCES_JSON}")


if __name__ == "__main__":
    main()
