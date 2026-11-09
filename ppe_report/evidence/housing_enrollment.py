"""
The evidence record for the explorer's Housing and enrollment tab (Tim, seahag prompt log #249): MAPC's 2024 update
rebuilt from the Census Bureau's and DESE's counts, MAPC's 2017 data, each town's new growth, American Farmland Trust's
table of cost of community services studies, and the Chapter 70 figures the tab cites. Each link sets a figure the
site draws beside the figure its publisher printed, and writes evidence/housing_enrollment.md and its part of the
site's src/data/massppe/evidence.json, which the tab's "Show the work" reads.

    1. the census counts: the towns' housing units add up to the state's own record in each file
    2. DESE's enrollment against MAPC's 2024 counts: districts, how many lost students, grew, and by how much
    3. MAPC's region by community type, against MAPC's 2024 table
    4. the straight line: no slope to tell from none, and where it crosses zero, against MAPC's 6 percent
    5. MAPC's 2017 data against its brief: districts, how many lost students, Hopkinton
    6. AFT's table: its medians against AFT's chart
    7. DLS's new growth: the site's share for every town and year against DLS's own percentage
    8. Chapter 70: towns below and above target, and the property percentage, against DESE's FY2027 summary
    9. Hancock: MAPC's 41 to 47 students, against DESE's counts

AI-written checks of published figures; the outside figures are the publishers' own words, typed in below with the page
they come from. A person accepts the record by setting REVIEWED.

    python evidence/housing_enrollment.py [SITE_ROOT]    # from ppe_report; SITE_ROOT defaults to ../../mywebsite
"""

import csv
import datetime as dt
import io
import json
import re
import statistics
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
OUT = HERE / "housing_enrollment.md"
RECORD_URL = "https://github.com/timshores/schools/blob/main/ppe_report/evidence/housing_enrollment.md"
MAPC_2024 = "https://www.mapc.org/learn/research-analysis/enrollment/"
MAPC_2017 = "https://www.mapc.org/learn/research-analysis/enrollment-2017/"
AFT = "https://farmlandinfo.org/wp-content/uploads/sites/2/2019/09/cost-of-community-services-fact-sheet.pdf"
DESE_FY27 = "https://www.doe.mass.edu/finance/chapter70/fy2027/chapter-2027-whitepaper.docx"
# MAPC's 2024 update: "Of the 231 local school districts statewide, two-thirds (155) saw enrollment decline"; "Out of
# 76 local districts that saw enrollment growth between 2010 and 2020, 28 grew by over 10 percent. Of those that
# experienced a decline in enrollment, 22 out of 155 shrank by over 20 percent"; "school enrollment declined 0.7
# percent overall"; "Statewide, school enrollment would have, according to the model, declined 6 percent in the
# absence of any change in the number of housing units"
MAPC24 = {"districts": 231, "declined": 155, "grew": 76, "over_10": 28, "under_20": 22, "statewide": -0.7, "intercept": -6}
# Its Figure 2, MAPC's region: housing change, enrollment change, districts
MAPC24_TYPES = {"Inner Core": (9.3, 2.2, 16), "Regional Urban Center": (6.3, 4.7, 11), "Maturing Suburb": (6.0, -2.5, 43),
                "Developing Suburb": (9.6, -8.2, 23), "All": (7.9, -0.3, 93)}
# MAPC's 2017 brief: "159 out of 234 local school districts saw enrollment declines"; "Hopkinton, at 18%, saw almost no
# change in enrollment (increase of 0.23%)"
MAPC17 = {"districts": 234, "declined": 159, "hopkinton": (18, 0.23)}
# MAPC's update: Hancock "saw an 86 percent growth in housing units and only a 14 percent growth in its school
# enrollment, from 41 to 47 students"
HANCOCK = (41, 47)
# AFT's chart, "Median COCS Results": residential $1.16, business $.30, agriculture $.37; "Ratios from the Indiana
# project are excluded"
AFT_MEDIANS = {"res": 1.16, "ci": 0.30, "wol": 0.37}
# DESE's FY2027 summary, Appendix A: "this impacts contributions for 87 or 25 percent of the 351 cities and towns";
# "263 cities and towns have preliminary contributions that are below target"; "the property percentage is set at
# 0.331 percent and is applied to each municipality's 2024 aggregate equalized property valuation"
DESE = {"above": 87, "below": 263, "property_pct": 0.331, "eqv_year": 2024}
# Who has read this record and accepted it, and when; empty until then
REVIEWED: str | None = None


def norm(s):
    return re.sub(r"[^a-z]", "", s.lower())


def state_units(year):
    """The state's own record (summary level 040) of housing units in the cached redistricting file"""
    z = next((DATA / "census_raw").glob(f"ma{year}.pl.zip"), None)
    if z is None:
        return None
    z = zipfile.ZipFile(z)
    geo = next(n for n in z.namelist() if "geo" in n)
    seg2 = next(n for n in z.namelist() if re.search(r"00002\d{4}\.pl$", n))
    rec = None
    for line in io.TextIOWrapper(z.open(geo), encoding="latin-1"):
        f = line.split("|") if year == 2020 else None
        sumlev, comp, logrec = (f[2], f[4], f[7]) if f else (line[8:11], line[11:13], line[18:25])
        if sumlev == "040" and comp == "00":
            rec = logrec
            break
    sep = "|" if year == 2020 else ","
    for line in io.TextIOWrapper(z.open(seg2), encoding="latin-1"):
        f = line.rstrip("\n").split(sep)
        if f[4] == rec:
            return int(f[-3])


def main():
    site = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent.parent / "mywebsite"
    sdata = site / "src/data/massppe"
    hd = json.loads((sdata / "housing.json").read_text())
    lines, failed, ran = [], [], []

    def link(n, title, ok, *body):
        lines.extend([f"## {n}. {title}: {'PASS' if ok else 'FAIL'}", "", *body, ""])
        ran.append(n)
        if not ok:
            failed.append(n)

    # 1. The census counts
    rows = list(csv.DictReader((DATA / "census_town_housing.csv").open()))
    sums = {y: sum(int(r["Housing_units"]) for r in rows if int(r["Year"]) == y) for y in (2010, 2020)}
    state = {y: state_units(y) for y in (2010, 2020)}
    link(1, "The towns' housing units add up to the state's", all(state[y] == sums[y] for y in sums),
         *[f"- {y}: the 351 towns sum to {sums[y]:,}; the file's state record says "
           f"{state[y]:,}." if state[y] else f"- {y}: the towns sum to {sums[y]:,}; the cached file isn't here to check."
           for y in sums])

    # 2. DESE's enrollment against MAPC's 2024 counts
    ds = hd["districts"]
    ch = [d["enr"][1] / d["enr"][0] - 1 for d in ds]
    enr = list(csv.DictReader((DATA / "dese_enrollment.csv").open()))
    tot = {fy: sum(float(r["total_cnt"]) for r in enr if r["FY"] == str(fy) and r["total_cnt"]) for fy in hd["meta"]["enroll_fy"]}
    statewide = (tot[hd["meta"]["enroll_fy"][1]] / tot[hd["meta"]["enroll_fy"][0]] - 1) * 100
    mine = {"districts": len(ds), "declined": sum(c < 0 for c in ch), "grew": sum(c >= 0 for c in ch),
            "over_10": sum(c > 0.10 for c in ch), "under_20": sum(c < -0.20 for c in ch), "statewide": round(statewide, 1)}
    link(2, "DESE's enrollment against MAPC's 2024 counts", all(mine[k] == MAPC24[k] for k in mine),
         f"- MAPC's update ({MAPC_2024}) and the site's {len(ds)} districts, DESE's counts for 2010/11 and 2019/20:",
         *[f"  - {k}: MAPC {MAPC24[k]}, the site {mine[k]}" for k in mine],
         "- MAPC counts the one district with no change among those that grew; every public school's students, "
         "charter schools included, for the statewide change.")

    # 3. MAPC's region by community type
    out3, ok3 = [], True
    for t, (hu, en, n) in MAPC24_TYPES.items():
        g = [d for d in ds if d["mapc"] and (t == "All" or d["type"] == t)]
        h = round((sum(d["hu"][1] for d in g) / sum(d["hu"][0] for d in g) - 1) * 100, 1)
        e = round((sum(d["enr"][1] for d in g) / sum(d["enr"][0] for d in g) - 1) * 100, 1)
        ok3 &= (h, e, len(g)) == (hu, en, n)
        out3.append(f"  - {t}: MAPC {hu}%, {en}%, {n} districts; the site {h}%, {e}%, {len(g)}")
    link(3, "MAPC's region by community type", ok3, "- Housing change, enrollment change, districts, summed over each type:",
         *out3)

    # 4. The straight line
    f = hd["fits"]["housing"]
    link(4, "The straight line", f["p"] > 0.05 and abs(f["intercept"] - MAPC24["intercept"]) < 1,
         f"- Least squares over {f['n']} districts: slope {f['slope']:.3f}, p = {f['p']:.3f}, R² = {f['r2']:.3f}; MAPC: "
         "\"the association between the two variables is not different from zero.\"",
         f"- Where it crosses zero housing growth: {f['intercept']:.1f}%; MAPC: \"declined 6 percent.\"",
         f"- Without the districts drawn off the chart ({', '.join(d['name'] for d in ds if d.get('off'))}): slope "
         f"{hd['fits']['housing_drawn']['slope']:.3f}, "
         f"p = {hd['fits']['housing_drawn']['p']:.3f}.")

    # 5. MAPC's 2017 data
    m = hd["mapc17"]
    hop = next(r for r in m if r["name"] == "Hopkinton")
    ok5 = (len(m) == MAPC17["districts"] and sum(r["chg"] < 0 for r in m) == MAPC17["declined"]
           and round(hop["perm"]) == MAPC17["hopkinton"][0] and hop["chg"] == MAPC17["hopkinton"][1])
    link(5, "MAPC's 2017 data against its brief", ok5,
         f"- The data file ({MAPC_2017}): {len(m)} districts, {sum(r['chg'] < 0 for r in m)} lost students; the brief: "
         f"{MAPC17['districts']} and {MAPC17['declined']}.",
         f"- Hopkinton: permits {hop['perm']}% of its 2010 housing, enrollment {hop['chg']:+}%; the brief: "
         f"{MAPC17['hopkinton'][0]}% and +{MAPC17['hopkinton'][1]}%.")

    # 6. AFT's table
    st = [s for s in hd["cocs"] if s["state"] != "Indiana"]
    med = {k: statistics.median(s[k] for s in st) for k in AFT_MEDIANS}
    link(6, "AFT's medians, to within a cent", all(round(abs(med[k] - AFT_MEDIANS[k]), 4) <= 0.01 for k in med),
         f"- {len(st)} rows outside Indiana, read from the fact sheet's table ({AFT}).",
         *[f"  - {k}: the table's median {med[k]:.2f}; AFT's chart {AFT_MEDIANS[k]:.2f}" for k in med],
         "- Working and open land is a cent off AFT's chart; the chart may count a different set of studies.")

    # 7. DLS's new growth
    years = json.loads((sdata / "towns.json").read_text())["meta"]["years"]
    dls = {(norm(r["Municipality"]), int(r["Fiscal Year"])): r for r in csv.DictReader((DATA / "dls_new_growth.csv").open())}
    n7, off7 = 0, []
    for fp in (sdata / "towns").glob("*.json"):
        for t in json.loads(fp.read_text())["towns"]:
            if t.get("kind") != "town":
                continue
            for fy, v in zip(years, t.get("new_growth") or []):
                r = dls.get((norm(t["name"]), fy))
                if v is None or not r or not r["Total New Growth Applied to Limit as a % of PY Levy Limit"]:
                    continue
                n7 += 1
                if abs(v * 100 - float(r["Total New Growth Applied to Limit as a % of PY Levy Limit"])) > 0.006:
                    off7.append((t["name"], fy, v))
    link(7, "DLS's new growth", n7 > 0 and not off7,
         f"- {n7} town-years: {n7 - len(off7)} of the site's shares equal DLS's own percentage, new growth over the "
         "prior year's levy limit, to DLS's two decimals." + (f" First: {off7[:3]}" if off7 else ""))

    # 8. Chapter 70
    c = hd["c70"]
    ok8 = (c["below"] == DESE["below"] and c["above"] == DESE["above"] and round(c["property_pct"] * 100, 3) == DESE["property_pct"]
           and c["eqv_year"] == DESE["eqv_year"])
    link(8, f"Chapter 70, FY{c['fy']}", ok8,
         f"- DESE's spreadsheet ({c['source']}): {c['below']} towns below target, {c['above']} above, the property "
         f"percentage {c['property_pct'] * 100:.4f}% of {c['eqv_year']} valuation.",
         f"- DESE's summary ({DESE_FY27}): {DESE['below']}, {DESE['above']}, {DESE['property_pct']}% of {DESE['eqv_year']}.")

    # 9. Hancock
    hk = {r["FY"]: r["total_cnt"] for r in enr if r["DIST_NAME"] == "Hancock"}
    e0, e1 = hd["meta"]["enroll_fy"]
    ok9 = (float(hk.get(str(e0 - 1)) or 0), float(hk.get(str(e1)) or 0)) == HANCOCK
    link(9, "Hancock's students", ok9,
         f"- MAPC gives {HANCOCK[0]} to {HANCOCK[1]}. DESE counts {hk.get(str(e0 - 1))} in {e0 - 2}/{str(e0 - 1)[2:]}, "
         f"{hk.get(str(e0))} in {e0 - 1}/{str(e0)[2:]} and {hk.get(str(e1))} in {e1 - 1}/{str(e1)[2:]}: MAPC's first "
         "figure is the year before the others it gives. The site uses 2010/11, as MAPC's other counts do.")

    today = dt.date.today().isoformat()
    n = len(ran)
    head = ["# Evidence record: the Housing and enrollment tab", "",
            f"Written by `evidence/housing_enrollment.py` on {today}. {n - len(failed)} of {n} links pass; "
            + (f"reviewed by {REVIEWED}." if REVIEWED else "not yet reviewed by a person.") + " These are AI-written "
            "checks of the site's figures against the ones MAPC, AFT, DESE and DLS published.", "",
            "**To accept it:** read the links below, then set `REVIEWED` in `evidence/housing_enrollment.py` to who and "
            "when, or ask Claude to, and rerun the script.", ""]
    OUT.write_text("\n".join(head + lines))
    ev_path = sdata / "evidence.json"
    ev = json.loads(ev_path.read_text()) if ev_path.exists() else {}
    ev["housing"] = {"checked": today, "record": RECORD_URL, "links": n, "passed": n - len(failed), "reviewed": REVIEWED}
    ev_path.write_text(json.dumps(ev, indent=1) + "\n")
    print(f"[evidence] housing and enrollment: {n - len(failed)} of {n} links pass -> {OUT.relative_to(ROOT)}, "
          f"{site.name}/src/data/massppe/evidence.json")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
