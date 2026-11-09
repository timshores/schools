"""
The evidence record for the map's note on the residential share of property value (Tim, seahag prompt log #247,
#248): towns whose property is mostly homes have passed more Proposition 2½ overrides, and new growth accounts for
about half of the link. It recomputes every figure the note gives from the site's data and DLS's new growth report,
and writes evidence/res_overrides.md. An exploratory finding: correlations across towns, not a cause shown.

    python evidence/res_overrides.py [SITE_ROOT]    # from ppe_report; SITE_ROOT defaults to ../../mywebsite
"""

import csv
import datetime as dt
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
NEW_GROWTH = ROOT / "data" / "dls_new_growth.csv"
OUT = HERE / "res_overrides.md"
YEARS = range(2006, 2026)  # the override count runs from FY2005's votes; new growth over the years they took effect


def norm(s):
    return re.sub(r"[^a-z]", "", s.lower())


def partial(x, y, zs):
    """Spearman's ρ of x and y with the zs held steady: each ranked, the zs' ranks regressed out of both"""
    X = np.column_stack([np.ones(len(x))] + [rankdata(z) for z in zs])
    rx, ry = rankdata(x), rankdata(y)
    ex = rx - X @ np.linalg.lstsq(X, rx, rcond=None)[0]
    ey = ry - X @ np.linalg.lstsq(X, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])


def main():
    site = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent.parent / "mywebsite"
    data = site / "src/data/massppe"
    years = json.loads((data / "towns.json").read_text())["meta"]["years"]
    i = years.index(2025)
    ng = {}
    for r in csv.DictReader(NEW_GROWTH.open()):
        if int(r["Fiscal Year"]) in YEARS and r["Prior Year's Levy Limit"]:
            d = ng.setdefault(norm(r["Municipality"]), [0.0, 0.0, 0.0])
            d[0] += float(r["Total New Growth Applied to Levy Limit"] or 0)
            d[1] += float(r["Residential New Growth Applied to the Levy Limit"] or 0)
            d[2] += float(r["Prior Year's Levy Limit"])
    v = lambda t, k: (t["v"].get(k) or [None])[0]  # noqa: E731
    towns = {}
    for t in json.loads((data / "map.json").read_text())["towns"]:
        g = ng.get(norm(t["name"]))
        towns[t["name"]] = {"res": v(t, "res_share"), "ov": v(t, "overrides"), "cap": v(t, "override"), "inc": v(t, "income"),
                            "dens": v(t, "density"), "ng_nonres": (g[0] - g[1]) / g[2] if g and g[2] else None}
    for f in (data / "towns").glob("*.json"):
        for t in json.loads(f.read_text())["towns"]:
            if t["name"] in towns:
                towns[t["name"]]["eqv"] = (t.get("eqv") or [None] * len(years))[i]
                towns[t["name"]]["pop"] = (t.get("pop") or [None] * len(years))[i]
    rows = [d for d in towns.values() if None not in d.values()]
    col = lambda k: [d[k] for d in rows]  # noqa: E731
    held = [col(k) for k in ("inc", "eqv", "pop", "dens")]
    q = np.quantile(col("res"), [0.25, 0.75])
    low = [d for d in rows if d["res"] < q[0]]
    high = [d for d in rows if d["res"] >= q[1]]
    figures = {
        "towns": len(rows),
        "high_quarter": f"{sum(1 for d in high if d['ov'] > 0)} of {len(high)}",
        "low_quarter": f"{sum(1 for d in low if d['ov'] > 0)} of {len(low)}",
        "rho_overrides": spearmanr(col("res"), col("ov")).correlation,
        "rho_overrides_held": partial(col("res"), col("ov"), held),
        "rho_overrides_held_ng": partial(col("res"), col("ov"), held + [col("ng_nonres")]),
        "rho_capacity_held": partial(col("res"), col("cap"), held),
        "rho_capacity_held_ng": partial(col("res"), col("cap"), held + [col("ng_nonres")]),
        "rho_res_ng": spearmanr(col("res"), col("ng_nonres")).correlation,
        "rho_ng_overrides_held": partial(col("ng_nonres"), col("ov"), held),
    }
    f = figures
    lines = [
        "# Evidence record: the residential share and Proposition 2½ overrides", "",
        f"Written by `evidence/res_overrides.py` on {dt.date.today().isoformat()}. An exploratory finding, which Tim asked "
        "for (#247, #248): correlations across the state's towns, not a cause shown. Spearman's ρ throughout; \"held "
        "steady\" means the towns' ranks on income per capita, property value per resident, population and density "
        "regressed out of both measures.", "",
        "## The figures the map's note gives", "",
        f"- {f['towns']} towns, FY2025 residential share (the map's measure) and overrides passed from FY2005 on (DLS).",
        f"- Most residential quarter: {f['high_quarter']} towns passed at least one override; least residential quarter: "
        f"{f['low_quarter']}.",
        f"- Residential share and overrides passed: ρ = {f['rho_overrides']:.2f}; held steady, {f['rho_overrides_held']:.2f}.",
        f"- Residential share and non-residential new growth (applied to the levy limit, FY2006-FY2025, as a share of "
        f"the prior year's limit, from DLS's New Growth Analysis): ρ = {f['rho_res_ng']:.2f}.",
        f"- Non-residential new growth and overrides, held steady: ρ = {f['rho_ng_overrides_held']:.2f}.",
        f"- Residential share and overrides, holding non-residential new growth steady too: {f['rho_overrides_held_ng']:.2f}, "
        f"down from {f['rho_overrides_held']:.2f}: new growth accounts for about half of the link. Override capacity: "
        f"{f['rho_capacity_held']:.2f}, then {f['rho_capacity_held_ng']:.2f}.",
        "- The residential share and non-residential new growth move closely together, so how the link splits between "
        "them is imprecise.", "",
    ]
    OUT.write_text("\n".join(lines))
    print(f"[evidence] residential share and overrides: {f['high_quarter']} vs {f['low_quarter']}; "
          f"ρ {f['rho_overrides_held']:.2f} held, {f['rho_overrides_held_ng']:.2f} with new growth -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
