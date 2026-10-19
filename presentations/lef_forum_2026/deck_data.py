"""
Numbers for the Leverett Education Foundation Education Forum deck (November 2026).

Run from ppe_report/ so the report's data paths resolve:

    cd ppe_report
    python ../presentations/lef_forum_2026/deck_data.py

Writes deck_data.json next to this script. Every figure on a slide comes
from this file, so the deck can be rebuilt when DESE data changes.
"""

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "ppe_report"))

from school_shared import (DISTRICTS_OF_INTEREST, load_data,  # noqa: E402
                           get_indistrict_fte_for_year,
                           get_western_cohort_districts_for_year,
                           weighted_epp_aggregation)

FIRST_YEAR = 2009
TINY_MAX_FTE = 200
LABELS = {"Amherst-Pelham": "Amherst-Pelham Regional"}
INDEX_CSV = Path("./data/inflation_indices.csv")


def series_dict(s: pd.Series, years) -> dict:
    return {int(y): (None if pd.isna(s.get(y)) else round(float(s.get(y)), 1)) for y in years}


def main():
    df, reg, c70 = load_data()
    latest = int(df["YEAR"].max())
    years = list(range(FIRST_YEAR, latest + 1))
    idx = pd.read_csv(INDEX_CSV).set_index("fiscal_year")["cpi_u_northeast"]

    cohorts = get_western_cohort_districts_for_year(df, reg, latest)
    western = [d for k, m in cohorts.items() if k != "SPRINGFIELD" for d in m]
    # Peer group: the report's Tiny cohort (0-200 in-district FTE, the series
    # standard set from FY2024 enrollment), applied to latest-year enrollment.
    # Stated explicitly so the deck does not depend on PPE_COHORT_RULE.
    cohorts["TINY"] = [d for d in western
                       if 0 < get_indistrict_fte_for_year(df, d, latest) <= TINY_MAX_FTE]

    def ppe(districts):
        piv, enr_in, _ = weighted_epp_aggregation(df, districts)
        return piv.sum(axis=1), enr_in

    groups = {"Western MA": western, "Western MA Tiny": cohorts["TINY"]}
    groups.update({LABELS.get(d, d): [d] for d in DISTRICTS_OF_INTEREST})

    out = {"latest_year": latest, "first_year": FIRST_YEAR, "years": years,
           "tiny_cohort_size": len(cohorts["TINY"]),
           "western_district_count": len(western),
           "ppe": {}, "ppe_real": {}, "enrollment": {}}
    for name, members in groups.items():
        s, enr = ppe(members)
        real = s * idx[latest] / idx.reindex(s.index)
        out["ppe"][name] = series_dict(s, years)
        out["ppe_real"][name] = series_dict(real, years)
        out["enrollment"][name] = series_dict(enr, years)

    # Leverett's place among the Tiny cohort in the latest year
    ranks = []
    for d in cohorts["TINY"]:
        s, _ = ppe([d])
        if latest in s.index:
            ranks.append((d, float(s[latest])))
    ranks.sort(key=lambda x: -x[1])
    out["tiny_ranking"] = [{"district": d.title(), "ppe": round(v)} for d, v in ranks]

    # Foundation enrollment (residents the district pays for) vs in-district FTE
    lev_c70 = c70[c70["DIST_NAME"].str.lower() == "leverett"].set_index("YEAR")
    out["leverett_foundation_enrollment"] = {int(y): float(lev_c70.loc[y, "distfoundenro"])
                                            for y in years if y in lev_c70.index}

    # Chapter 70 aid and spending above the required minimum, per foundation pupil
    def c70_per_pupil(districts, year):
        sub = c70[(c70["DIST_NAME"].str.lower().isin([d.lower() for d in districts]))
                  & (c70["YEAR"] == year)]
        # Districts DESE has not reported actual NSS for drop out of every component
        sub = sub.dropna(subset=["distfoundenro", "c70aid", "rqdnss2", "actualNSS"])
        fe = sub["distfoundenro"].sum()
        return {"aid": float(sub["c70aid"].sum() / fe),
                "required_nss": float(sub["rqdnss2"].sum() / fe),
                "actual_nss": float(sub["actualNSS"].sum() / fe),
                "above_required": float((sub["actualNSS"] - sub["rqdnss2"]).sum() / fe)}
    out["c70"] = {}
    for name, members in groups.items():
        out["c70"][name] = {str(y): {k: round(v) for k, v in c70_per_pupil(members, y).items()}
                            for y in (FIRST_YEAR, latest)}

    # Teacher pay and staffing (DESE Teacher Salaries indicators)
    ts = df[df["IND_CAT"].str.lower() == "teacher salaries"]
    pay = {}
    for name, members in groups.items():
        sub = ts[ts["DIST_NAME"].str.lower().isin([m.lower() for m in members])]
        sal = sub[sub["IND_SUBCAT"] == "Average Teacher Salary"][["DIST_NAME", "YEAR", "IND_VALUE"]]
        fte = sub[sub["IND_SUBCAT"] == "Teacher FTE"][["DIST_NAME", "YEAR", "IND_VALUE"]]
        m = sal.merge(fte, on=["DIST_NAME", "YEAR"], suffixes=("_sal", "_fte")).dropna()
        m = m[m["IND_VALUE_sal"] > 0]
        # Average salary weighted by teacher FTE, so a cohort figure is per teacher
        by_year = m.groupby("YEAR").apply(
            lambda g: (g["IND_VALUE_sal"] * g["IND_VALUE_fte"]).sum() / g["IND_VALUE_fte"].sum(),
            include_groups=False)
        per100 = sub[sub["IND_SUBCAT"] == "Teachers per 100 FTE students"].groupby("YEAR")["IND_VALUE"].mean()
        pay[name] = {"avg_salary": series_dict(by_year, years),
                     "teacher_fte": series_dict(fte.groupby("YEAR")["IND_VALUE"].sum(), years),
                     "teachers_per_100": series_dict(per100, years)}
    out["teacher_pay"] = pay

    # Teachers spending per pupil (the category most tied to pay)
    epp = df[(df["IND_CAT"] == "Expenditures Per Pupil") & (df["IND_SUBCAT"] == "Teachers")]
    out["teachers_ppe"] = {LABELS.get(d, d): series_dict(
        epp[epp["DIST_NAME"] == d].set_index("YEAR")["IND_VALUE"], years) for d in DISTRICTS_OF_INTEREST}

    out["cpi_u_northeast_fy"] = {int(y): float(idx[y]) for y in years if y in idx.index}
    path = HERE / "deck_data.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"[deck] {path} (FY{FIRST_YEAR}-FY{latest})")


if __name__ == "__main__":
    main()
