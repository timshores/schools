"""Numbers for the six-slide LEF forum deck (build_forum6.js), from the Municipal spending explorer's data files
(mywebsite/src/data/massppe, written by ppe_report/export_explorer_data.py), so the deck and the explorer agree.

    python presentations/lef_forum_2026/forum6_data.py [SITE_DATA_DIR]

Writes forum6_data.json beside this file. Inflation is the explorer's: CPI-U Northeast by fiscal year.
The Four Towns Fiscal Study Group figures are typed in from its Report to the Community, October 28, 2026.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SITE = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/projects/mywebsite/src/data/massppe")

ex = json.load(open(os.path.join(SITE, "explorer.json")))
years = ex["meta"]["years"]
cpi = {int(k): v for k, v in ex["meta"]["cpi"].items()}
latest = years[-1]
ents = {}
for f in sorted(os.listdir(os.path.join(SITE, "districts"))):
    for x in json.load(open(os.path.join(SITE, "districts", f)))["entities"]:
        ents[x["id"]] = x

def _fit(vals):
    """The least-squares line through a yearly series: its first and last values and the change between them"""
    m = len(vals)
    xb, yb = (m - 1) / 2, sum(vals) / m
    slope = sum((i - xb) * (v - yb) for i, v in enumerate(vals)) / sum((i - xb) ** 2 for i in range(m))
    first, last = yb - slope * xb, yb + slope * xb
    return {"first": round(first, 1), "last": round(last, 1), "change": last / first - 1}


GROUPS = {
    "leverett": "Leverett", "pelham": "Pelham", "shutesbury": "Shutesbury", "amherst": "Amherst",
    "amherst-pelham": "Amherst-Pelham Regional", "cohort-tiny-western": "Western MA, 200 or fewer students",
    "cohort-medium-western": "Western MA, 801 to 1,600 students",
    "cohort-gateway": "Gateway cities",
    "type-rural": "Rural schools",
    "region-western": "Western MA", "region-central": "Central MA", "region-northeast": "Northeast MA",
    "region-metro-west": "Metro West", "region-southeast": "Southeast MA", "region-boston": "Boston area",
    "statewide-all": "Statewide",
}


def total(x):
    cats = x["ppe"]
    return [sum(v[i] for v in cats.values() if v[i] is not None) if any(v[i] is not None for v in cats.values()) else None
            for i in range(len(years))]


def real(vals):
    return [None if v is None else v * cpi[latest] / cpi[y] for v, y in zip(vals, years)]


out = {"years": years, "first": years[0], "latest": latest, "cpi_factor": cpi[latest] / cpi[years[0]],
       # Northeast CPI's yearly rise: the latest fiscal year's, and the average over the deck's years
       "cpi_rate_latest": cpi[latest] / cpi[latest - 1] - 1, "cpi_rate_avg": (cpi[latest] / cpi[years[0]]) ** (1 / (latest - years[0])) - 1,
       "groups": {}}
n = len(years) - 1
for gid, name in GROUPS.items():
    x = ents[gid]
    t = total(x)
    r = real(t)
    nss = x["nss"]
    parts = {k: nss[k] for k in ("ch70_aid", "required_local", "above_required")}
    shares = {}
    for i in (0, n):
        tot = sum(parts[k][i] for k in parts)
        shares[str(years[i])] = {k: parts[k][i] / tot for k in parts}
    enroll = x["enroll"]["in_district"]
    out["groups"][gid] = {
        "name": name, "members": len(x["members"]) if isinstance(x.get("members"), list) else x.get("members"),
        "ppe": [round(v) for v in t], "ppe_real": [round(v) for v in r],
        "nominal_change": t[n] / t[0] - 1, "real_change": r[n] / r[0] - 1,
        "real_rate": (r[n] / r[0]) ** (1 / n) - 1, "nominal_rate": (t[n] / t[0]) ** (1 / n) - 1,
        "real_index": [round(100 * v / r[0], 1) for v in r],
        "enroll_first": enroll[0], "enroll_latest": enroll[n], "enroll_change": enroll[n] / enroll[0] - 1,
        # Enrollment year by year and its straight-line trend (least squares over every year), for the dashed line
        "enroll": [round(v, 1) for v in enroll], "enroll_trend": _fit(enroll),
        "aid_per_pupil": parts["ch70_aid"][n], "nss_shares": shares,
    }

cap = json.load(open(os.path.join(SITE, "cap.json")))["years"]
fy = max(cap)
lev = next(t for t in cap[fy]["towns"] if t["id"] == "leverett")
tot = cap[fy]["totals"]
out["cap"] = {"fy": int(fy), "towns": len(cap[fy]["towns"]), "at_cap": tot["capped"], "lowered": tot["up"], "raised": tot["down"],
              "shifted": tot["up_sum"], "leverett_target": lev["target"], "leverett_nocap": lev["nocap"],
              "leverett_target_change": lev["nocap"] - lev["target"]}

# Four Towns Fiscal Study Group, Report to the Community, October 28, 2026 (pages 2, 4, 5 and the appendix)
# The same report's appendix, "Commercial and Industrial Property Values as Percent of Total Tax Levy, FY25" (from DLS,
# Tax Levy by Class), and its residential shares of the levy and Amherst's tax-exempt share of value (page 4)
out["tax_base"] = {
    "ci_share": [["Amherst", 0.063], ["Leverett", 0.008], ["Pelham", 0.011], ["Shutesbury", 0.009], ["Hadley", 0.274], ["Deerfield", 0.175],
                 ["Greenfield", 0.166], ["Northampton", 0.138], ["East Longmeadow", 0.129], ["Easthampton", 0.085]],
    "ours": ["Amherst", "Leverett", "Pelham", "Shutesbury"],
    "residential": {"Amherst": 0.88, "Leverett": 0.88, "Pelham": 0.93, "Shutesbury": 0.94}, "amherst_exempt": 0.30,
}
out["four_towns"] = {
    "bill_1992": 2136, "bill_2026": 7852, "bill_at_cpi": 5065, "bill_pct_of_cpi": 1.55,
    "state_aid_share_2008": 0.131, "state_aid_share_2026": 0.096, "discretionary_fy2025": 0.42,
    "residential_share_of_levy": 0.88,
}
# Leverett's income and equalized property value per resident against Western MA as a whole (all its towns pooled),
# from the figures the state uses for the Chapter 70 calculation of the cap's year
_t = cap[fy]["towns"]
_lev = next(t for t in _t if t["id"] == "leverett")
_w = [t for t in _t if t["region"] == "Western"]
_s = [t for t in _t if t["pop"] and t["pop_income"]]
_rank = lambda rows, f: sorted(rows, key=f, reverse=True).index(_lev) + 1
_inc, _eqv = (lambda t: t["income"] / t["pop_income"]), (lambda t: t["eqv"] / t["pop"])
_pool = lambda rows, k, d: sum(t[k] for t in rows) / sum(t[d] for t in rows)
out["wealth"] = {"income_pc": _inc(_lev), "eqv_pc": _eqv(_lev), "western_towns": len(_w),
                 "income_vs_western": _inc(_lev) / _pool(_w, "income", "pop_income"), "eqv_vs_western": _eqv(_lev) / _pool(_w, "eqv", "pop"),
                 "western_income_pc": _pool(_w, "income", "pop_income"), "western_eqv_pc": _pool(_w, "eqv", "pop"),
                 "income_rank_western": _rank(_w, _inc), "eqv_rank_western": _rank(_w, _eqv),
                 "income_vs_state": _inc(_lev) / _pool(_s, "income", "pop_income"), "income_rank_state": _rank(_s, _inc), "state_towns": len(_s)}
# The municipal revenue growth factor: DESE's history workbook FY2007 on (ppe_report/data/dese_c70_town_history.csv)
# and the latest year's formula sheet (dese_c70_townwide.csv). The medians are across all the state's towns.
import csv
import statistics
_data = os.path.join(HERE, "..", "..", "ppe_report", "data")
_g = {}
for _r in csv.DictReader(open(os.path.join(_data, "dese_c70_town_history.csv"))):
    _g[(int(_r["fy"]), _r["town"].strip())] = float(_r["mrgf"])
_tw = {}
for _r in csv.DictReader(open(os.path.join(_data, "dese_c70_townwide.csv"))):
    if not _r["mrgf"]:
        continue
    _g[(int(_r["fy"]), _r["town"].strip())] = float(_r["mrgf"])
    _tw[(int(_r["fy"]), _r["town"].strip())] = _r
_gy = sorted({y for y, _ in _g})
_series = lambda town: [_g[(y, town)] for y in _gy]
_median = [statistics.median(v for (y, _), v in _g.items() if y == yy) for yy in _gy]
_wnames = {t["name"] for t in _w}
_wmed = [statistics.median(v for (y, n), v in _g.items() if y == yy and n in _wnames) for yy in _gy]
_avg = lambda xs: sum(xs) / len(xs)
def _step(town, y):
    r = _tw[(y, town)]
    f = lambda k: float(r[k] or 0)
    return {"prior": f("rlc_prior"), "mrgf": f("mrgf"), "preliminary": f("preliminary"), "target": f("target"),
            "increment": f("increment"), "excess_reduction": f("excess_reduction"), "required": f("required"),
            "change": f("required") / f("rlc_prior") - 1}
out["mrgf"] = {"years": _gy, "towns_counted": sum(1 for (y, _) in _g if y == _gy[-1]), "western_towns": sum(1 for (y, n) in _g if y == _gy[-1] and n in _wnames),
               "leverett": _series("Leverett"), "amherst": _series("Amherst"), "pelham": _series("Pelham"), "shutesbury": _series("Shutesbury"),
               "median": _median, "western_median": _wmed,
               "avg": {"leverett": _avg(_series("Leverett")), "amherst": _avg(_series("Amherst")), "median": _avg(_median), "western_median": _avg(_wmed)},
               "steps": {t.lower(): {str(y): _step(t, y) for y in _gy[-2:]} for t in ("Leverett", "Amherst")}}
json.dump(out, open(os.path.join(HERE, "forum6_data.json"), "w"), indent=1)
g = out["groups"]
for k in ("leverett", "cohort-tiny-western", "region-western", "statewide-all"):
    print(f"{g[k]['name']:36s} FY{years[0]} ${g[k]['ppe'][0]:,} -> FY{latest} ${g[k]['ppe'][n]:,}  nominal {g[k]['nominal_change']:+.0%}  real {g[k]['real_change']:+.0%}"
          f"  ({g[k]['real_rate']:+.1%}/yr)  enrollment {g[k]['enroll_change']:+.0%}  above-required share {g[k]['nss_shares'][str(years[0])]['above_required']:.0%} -> {g[k]['nss_shares'][str(latest)]['above_required']:.0%}"
          f"  aid share {g[k]['nss_shares'][str(latest)]['ch70_aid']:.0%}")
print("regions' real rates:", {v["name"]: f"{v['real_rate']:.1%}" for k, v in g.items() if k.startswith("region")})
print("cap:", out["cap"])
