"""
The evidence record for the single-family tax bill as a share of income per capita (Project Regain Custody; seahag
prompt log #246): the towns view's tax bill chart and tile, the map's measure, and About the data's "Amherst's FY2025
bill is 41% of its income per capita, the highest share in the state". It checks each link from the Division of
Local Services' report to the figures the site draws, and writes evidence/bill_income.md and its part of the site's
src/data/massppe/evidence.json, which the tax bill chart's "Show the work" reads.

    1. DLS's report: its own "Average Tax Bill as a % of Income" against its bill over its income per capita
    2. the report -> the site's town files: every town's share, year by year
    3. the ranking: the highest share in the latest year, which About the data states, from the map's data
    4. an outside report: the Amherst Indy's figure for Amherst, FY2025

AI-written checks of one agency report; the outside report is a newspaper's reading of the same DLS figure. A person
accepts the record by setting REVIEWED.

    python evidence/bill_income.py [SITE_ROOT]    # from ppe_report; SITE_ROOT defaults to ../../mywebsite
"""

import csv
import datetime as dt
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BILLS = ROOT / "data" / "dls_single_family_tax_bill.csv"
OUT = HERE / "bill_income.md"
RECORD_URL = "https://github.com/timshores/schools/blob/main/ppe_report/evidence/bill_income.md"
DLS_REPORT = "https://dls-gw.dor.state.ma.us/reports/rdPage.aspx?rdReport=AverageSingleTaxBill.SingleFamTaxBill_wRange"
# A newspaper's reading of DLS's figure: the Amherst Indy, August 1, 2025
INDY = {"url": "https://www.amherstindy.org/2025/08/01/by-this-measure-amherst-has-the-least-affordable-property-taxes-in-massachusetts/",
        "town": "Amherst", "fy": 2025, "pct": 40.76}
# Who has read this record and accepted it, and when; empty until then
REVIEWED: str | None = None


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main():
    site = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent.parent / "mywebsite"
    lines, failed, ran = [], [], []

    def link(n, title, ok, *body):
        lines.extend([f"## {n}. {title}: {'PASS' if ok else 'FAIL'}", "", *body, ""])
        ran.append(n)
        if not ok:
            failed.append(n)

    rows = [r for r in csv.DictReader(BILLS.open())]
    have = [(r["Municipality"], int(r["FY"]), num(r["Single-Family Tax Bill"]), num(r["DOR Income Per Capita"]),
             num(r["Average Tax Bill as a % of Income"])) for r in rows]
    have = [h for h in have if None not in h[2:] and h[3]]

    # 1. DLS's own percentage against its own bill and income
    off1 = [h for h in have if abs(h[2] / h[3] * 100 - h[4]) > 0.005]
    link(1, "DLS's percentage is its bill over its income per capita", not off1,
         f"- `data/dls_single_family_tax_bill.csv`, DLS's report ({DLS_REPORT}): {len(have)} town-years with a bill, an "
         f"income per capita and DLS's percentage; {len(have) - len(off1)} agree to DLS's two decimals.")

    # 2. The site's town files
    share = {}
    for f in (site / "src/data/massppe/towns").glob("*.json"):
        for t in json.loads(f.read_text())["towns"]:
            share[t["name"]] = t.get("sf_bill_income") or []
    years = json.loads((site / "src/data/massppe/towns.json").read_text())["meta"]["years"]
    off2, n2 = [], 0
    for town, fy, bill, inc, _ in have:
        if fy in years and town in share:
            v = share[town][years.index(fy)]
            n2 += 1
            if v is None or abs(v - bill / inc) > 0.00005:
                off2.append((town, fy, v, round(bill / inc, 4)))
    link(2, "DLS's report to the site's town files", not off2 and n2 > 0,
         f"- {n2} town-years, FY{years[0]} to FY{years[-1]}; {n2 - len(off2)} of the site's shares equal DLS's bill over "
         "its income per capita, to four decimals." + (f" First: {off2[:3]}" if off2 else ""))

    # 3. The ranking About the data states, from the map's data (each town's latest year)
    # The site's latest year, which every town has; DLS's file runs ahead for the towns that have set a new rate
    latest = years[-1]
    ranked = sorted((h for h in have if h[1] == latest), key=lambda h: -h[4])
    mp = json.loads((site / "src/data/massppe/map.json").read_text())
    on_map = sorted(((t["name"], t["v"].get("bill_income")) for t in mp["towns"] if (t["v"].get("bill_income") or [None])[0] is not None),
                    key=lambda x: -x[1][0])
    top_map = on_map[0]
    link(3, f"The highest share, FY{latest}", ranked[0][0] == top_map[0] and abs(top_map[1][0] * 100 - ranked[0][4]) < 0.01,
         f"- DLS's figures rank {ranked[0][0]} first in FY{latest}, at {ranked[0][4]:.2f}% (a ${ranked[0][2]:,.0f} bill over "
         f"${ranked[0][3]:,.0f} of income per capita), then {ranked[1][0]}, {ranked[1][4]:.2f}%, and {ranked[2][0]}, "
         f"{ranked[2][4]:.2f}%; {len(ranked)} towns in all.",
         f"- The map's data, which About the data reads for its sentence, ranks {top_map[0]} first at "
         f"{top_map[1][0] * 100:.2f}%.")

    # 4. The Amherst Indy
    dls = next(h for h in have if h[0] == INDY["town"] and h[1] == INDY["fy"])
    link(4, "An outside report: the Amherst Indy", abs(dls[4] - INDY["pct"]) < 0.005,
         f"- The Amherst Indy ({INDY['url']}) reported {INDY['town']}'s FY{INDY['fy']} bill at {INDY['pct']:.2f}% of income "
         f"per capita, the highest in the state; DLS's figure is {dls[4]:.2f}%.",
         "- The newspaper read the same DLS report, so agreement shows the figure is DLS's, not that DLS's inputs are right.")

    today = dt.date.today().isoformat()
    n = len(ran)
    head = ["# Evidence record: the single-family tax bill as a share of income per capita", "",
            f"Written by `evidence/bill_income.py` on {today}. {n - len(failed)} of {n} links pass; "
            + (f"reviewed by {REVIEWED}." if REVIEWED else "not yet reviewed by a person.") + " These are AI-written "
            "checks of one DLS report; the outside report is a newspaper's reading of the same figure.", "",
            "**To accept it:** read the links below, then set `REVIEWED` in `evidence/bill_income.py` to who and when, "
            "or ask Claude to, and rerun the script.", ""]
    OUT.write_text("\n".join(head + lines))
    ev_path = site / "src/data/massppe/evidence.json"
    ev = json.loads(ev_path.read_text()) if ev_path.exists() else {}
    ev["bill_income"] = {"checked": today, "record": RECORD_URL, "links": n, "passed": n - len(failed),
                         "town_years": len(have), "latest": latest, "top": ranked[0][0], "top_pct": ranked[0][4],
                         "reviewed": REVIEWED}
    ev_path.write_text(json.dumps(ev, indent=1) + "\n")
    print(f"[evidence] bill share: {n - len(failed)} of {n} links pass -> {OUT.relative_to(ROOT)}, {site.name}/src/data/massppe/evidence.json")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
