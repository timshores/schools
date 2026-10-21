"""
Export the data behind the interactive explorer at timshores.com/massppe.

One JSON file with every Western MA traditional district in the report, each
cohort aggregate, and the Western MA aggregate, FY START_YEAR..LATEST_YEAR:
per-pupil expenditure by category, enrollment, and Chapter 70 / net school
spending per foundation pupil. Values come from the same functions the PDF
report uses, so the explorer and the report agree.

Revenue by source comes from the Census Bureau's school finance survey
(data/census_f33_ma.csv, written by update_census_data.py), per in-district
FTE pupil like spending, for regional districts only: for town-run districts
Census's local revenue leaves out much of what the town spends on its schools
(Amherst FY2024: $18.4M of revenue against $37.6M of spending). Town-run
districts get Census's latest totals instead, so the explorer can say so.
Aggregates pool their regional members, weighted by in-district FTE.

    python export_explorer_data.py ../../mywebsite/public/massppe/data/explorer.json
"""

import datetime as dt
import json
import re
import sys
from pathlib import Path

import pandas as pd

from school_shared import (CANON_CATS_BOTTOM_TO_TOP, IN_DISTRICT_FTE_KEY, LATEST_YEAR, START_YEAR, get_cohort_label,
                           get_cohort_range, get_cohort_short_label, get_indistrict_fte_for_year, get_western_cohort_districts, load_data,
                           prepare_aggregate_nss_ch70_weighted, prepare_district_epp_lines,
                           prepare_district_nss_ch70, prepare_western_epp_lines)

YEARS = list(range(START_YEAR, LATEST_YEAR + 1))
ENROLL = {"In-District FTE Pupils": "in_district", "Foundation Enrollment": "foundation",
          "Out-of-District FTE Pupils": "out_of_district"}
NSS = {"Ch70 Aid": "ch70_aid", "Req NSS (minus Ch70)": "required_local",
       "Actual NSS (minus Req NSS)": "above_required"}
COHORT_ORDER = ["TINY", "SMALL", "MEDIUM", "LARGE", "SPRINGFIELD"]

CENSUS_CSV = Path("data/census_f33_ma.csv")
# DESE numbers regional academic districts from 0600; DIST_CODE is that times 10000
REGIONAL_MIN_CODE = 6000000
# An aggregate with fewer regional members than this gets no revenue average
MIN_REVENUE_MEMBERS = 3
# Revenue sources, bottom to top in the explorer's stack: Census F-33 item codes
# (dollars in thousands). "Other" rows are their total less the rows named.
# School building aid is on top because it arrives in large, irregular payments.
REVENUE = {
    "towns": ["D23"],              # revenue from cities and towns: member assessments, tuition
    "ch70": ["C01"],               # general formula assistance (matches DESE Chapter 70 aid)
    "retirement": ["C38"],         # state payments on behalf: teachers' retirement
    "transport": ["C12"],
    "other_state": ["TSTREV", "-C01", "-C38", "-C12", "-C11"],
    "federal": ["TFEDREV"],
    "districts": ["D11"],          # revenue from other school systems
    "other_local": ["TLOCREV", "-D23", "-D11"],
    "building": ["C11"],           # capital outlay and debt service (MSBA)
}


# Town view (towns.json, beside explorer.json): every member town of the
# explorer's districts, with its general fund spending and revenue per resident.
DLS_CSV = Path("data/dls_schedule_a_general_fund.csv")
POP_CSV = Path("data/census_town_population.csv")
MEMBERS_CSV = Path("data/dese_c70_regional_members.csv")
C70_CSV = Path("data/dese_c70_profile.csv")
# Schedule A functions, bottom to top in the explorer's stack
TOWN_SPEND = {
    "education": "Education", "fixed": "Fixed Costs", "safety": "Public Safety", "works": "Public Works",
    "general": "General Government", "debt": "Debt Service", "assessments": "Intergov Assessments",
    "human": "Human Services", "culture": "Culture and Recreation", "other": "Other Expenditures",
}
# Schedule A revenue sources, grouped: six of its eleven are small in most towns
TOWN_REV = {
    "taxes": ["Taxes"],
    "state": ["State Revenue"],
    "federal": ["Federal Revenue"],
    "charges": ["Service Charges", "Licenses and Permits", "Fines and Forfeitures", "Special Assessments"],
    "other": ["Miscellaneous", "Revenue from Other Governments"],
    "transfers": ["Transfers", "Other Financing Sources"],
}
# Kept out of the Western MA towns average, as Springfield is kept out of the district averages
TOWN_AVERAGE_EXCLUDES = {"Springfield"}
# Peer groups by population: the estimate behind the latest fiscal year (July 2024 for
# FY2025), as district cohorts use latest-year enrollment. Round bounds near the quintiles
# of the 96 towns (median 1,868): 20, 35, 21 and 20 towns on FY2025 data.
# (id, lowest population, first population above the group, name, mid-sentence name, label)
TOWN_PEERS = [
    ("towns-under-1000", 0, 1000, "Towns under 1,000 average",
     "the average for towns under 1,000", "under 1,000 residents"),
    ("towns-1000-2499", 1000, 2500, "Towns of 1,000–2,499 average",
     "the average for towns of 1,000–2,499", "1,000–2,499 residents"),
    ("towns-2500-9999", 2500, 10000, "Towns of 2,500–9,999 average",
     "the average for towns of 2,500–9,999", "2,500–9,999 residents"),
    ("towns-10000-plus", 10000, None, "Towns of 10,000+ average",
     "the average for towns of 10,000 or more", "10,000 residents or more"),
]


# Labels for the revenue sources, as the explorer's legend names them (CSV headers)
REVENUE_LABELS = {
    "towns": "Payments from towns", "ch70": "Chapter 70 state aid", "retirement": "Teacher retirement paid by the state",
    "transport": "Transportation aid", "other_state": "Other state aid", "federal": "Federal",
    "districts": "From other school districts", "other_local": "Other local", "building": "School building aid",
}

# Ultimate sources, for the CSVs' source_url columns
URL_DESE_EXPEND = "https://educationtocareer.data.mass.gov/d/er3w-dyti"
URL_C70_PROFILE = "https://www.doe.mass.edu/finance/chapter70/profile.xlsx"
URL_C70_REGIONAL = "https://www.doe.mass.edu/finance/chapter70/fy{fy}/chapter-{fy}-reg.xlsx"
URL_CENSUS_FILE = "https://www2.census.gov/programs-surveys/school-finances/tables/{fy}/secondary-education-finance/elsec{yy:02d}.{ext}"
URL_DLS = "https://dls-gw.dor.state.ma.us/reports/rdPage.aspx?rdReport=ScheduleA.GenFund_MAIN"
URL_POP = "https://www2.census.gov/programs-surveys/popest/datasets/"
URL_BLS = "https://data.bls.gov/timeseries/CUUR0100SA0"


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def series(s, digits):
    """Values for YEARS, None where missing."""
    if s is None or len(s) == 0:
        return [None] * len(YEARS)
    s = s.copy(); s.index = [int(i) for i in s.index]
    return [None if (y not in s.index or pd.isna(s[y])) else round(float(s[y]), digits) for y in YEARS]


def pack(piv, lines, nss):
    return {
        "ppe": {c: series(piv[c] if c in piv.columns else None, 0) for c in CANON_CATS_BOTTOM_TO_TOP},
        "enroll": {key: series(lines.get(label), 1) for label, key in ENROLL.items()},
        "nss": {key: series(nss[col] if col in nss.columns else None, 0) for col, key in NSS.items()},
    }


def revenue_per_pupil(census, codes):
    """Revenue by source per in-district FTE pupil, pooled over the districts' DESE codes."""
    rows = census[census["DIST_CODE"].isin(codes) & (census["FTE"] > 0)]
    by_year = rows.groupby("FY").sum(numeric_only=True)
    out = {}
    for key, items in REVENUE.items():
        dollars = sum(by_year[i.lstrip("-")] * (-1 if i.startswith("-") else 1) for i in items)
        out[key] = series(dollars * 1000 / by_year["FTE"], 0)
    return out


def revenue_fields(census, codes, regional):
    """The entity's revenue fields: per-pupil series when every member is regional,
    Census's latest totals for a town-run district."""
    if census is None:
        return {}
    if regional:
        return {"rev": revenue_per_pupil(census, codes)}
    latest = census[census["DIST_CODE"].isin(codes)].sort_values("FY").tail(1)
    if latest.empty:
        return {}
    r = latest.iloc[0]
    return {"rev_check": {"fy": int(r["FY"]), "revenue": int(r["TOTALREV"]) * 1000,
                          "expenditure": int(r["TOTALEXP"]) * 1000}}


def norm(name: str) -> str:
    """A town name as DLS, DESE and Census all spell it ("Manchester By The Sea", "Manchester-by-the-Sea")."""
    return re.sub(r"[^a-z]", "", name.lower())


def dollars(v):
    """Whole dollars, or None for a missing value (JSON has no NaN)."""
    return None if v is None or pd.isna(v) else int(round(float(v)))


def town_series(rows, population):
    """Per-resident series for one town or a pooled group: rows are DLS rows for its towns,
    population a (town key, FY) -> residents map. A year counts only towns that filed."""
    wide = rows.pivot_table(index=["key", "FY", "Type"], columns="Category", values="Amount", aggfunc="sum")
    wide = wide.reset_index()
    wide["pop"] = [population.get((k, fy)) for k, fy in zip(wide["key"], wide["FY"])]
    out = {}
    for kind, total_col, groups in [("spend", "Total Expenditures", {k: [v] for k, v in TOWN_SPEND.items()}),
                                    ("rev", "Total Revenues", TOWN_REV)]:
        w = wide[(wide["Type"] == ("Expenditures" if kind == "spend" else "Revenues"))
                 & (wide[total_col] > 0) & wide["pop"].notna()]
        by_year = w.groupby("FY").sum(numeric_only=True)
        out[kind] = {k: series(by_year[cols].sum(axis=1) / by_year["pop"], 0) for k, cols in groups.items()}
        out[f"{kind}_total"] = series(by_year[total_col], 0)
    return out


def town_data(districts, latest):
    """towns.json content, and the member towns of each district.

    districts: (entity id, name, Org4Code) for every district in the explorer."""
    dls = pd.read_csv(DLS_CSV, dtype={"DOR_CODE": str})
    dls = dls[dls["FY"].isin(YEARS)].assign(key=lambda d: d["Municipality"].map(norm))
    pop = pd.read_csv(POP_CSV).assign(key=lambda d: d["Town"].map(norm))
    population = {(k, fy): p for k, fy, p in zip(pop["key"], pop["FY"], pop["Population"])}
    county = pop.sort_values("FY").groupby("key")["County"].last().to_dict()
    members = pd.read_csv(MEMBERS_CSV, dtype={"Org4Code": str}).assign(key=lambda d: d["Town"].map(norm))
    c70 = pd.read_csv(C70_CSV, dtype={"Org4Code": str})
    rlc = c70[c70["fy"] == latest].set_index("Org4Code")["distrlc"].to_dict()
    names = dls.drop_duplicates("key").set_index("key")["Municipality"].to_dict()

    by_org = {f"{org:04d}": (eid, name) for eid, name, org in districts}
    district_towns = {}
    for eid, name, org in districts:
        if org * 10000 < REGIONAL_MIN_CODE:
            district_towns[eid] = [norm(name)]
        else:
            district_towns[eid] = members.loc[members["Org4Code"] == f"{org:04d}", "key"].tolist()
            if not district_towns[eid]:
                raise ValueError(f"No FY{latest} member towns for {name}")
    keys = sorted({k for ks in district_towns.values() for k in ks})
    missing = [k for k in keys if k not in names or (k, latest) not in population]
    if missing:
        raise ValueError(f"Towns without Schedule A or population data: {missing}")

    pooled = [k for k in keys if names[k] not in TOWN_AVERAGE_EXCLUDES]
    peer_of = {}
    for pid, lo, hi, *_ in TOWN_PEERS:
        for k in pooled:
            if lo <= population[(k, latest)] and (hi is None or population[(k, latest)] < hi):
                peer_of[k] = pid

    in_view = set(keys)
    # Every member town of every regional district, named as DLS spells it where it can be
    member_towns = {org: [{"name": names.get(mk, town), "id": slug(names[mk]) if mk in in_view else None, "key": mk}
                          for mk, town in zip(g["key"], g["Town"])]
                    for org, g in members.groupby("Org4Code")}

    towns = []
    schools_of = {}
    for k in keys:
        schools = []
        own = next(((eid, name, org) for eid, name, org in districts
                    if org * 10000 < REGIONAL_MIN_CODE and norm(name) == k), None)
        if own:
            schools.append({"name": own[1], "id": own[0], "kind": "town",
                            "contribution": dollars(rlc.get(f"{own[2]:04d}")), "with": []})
        for _, m in members[members["key"] == k].iterrows():
            org = int(m["Org4Code"])
            linked = by_org.get(m["Org4Code"])
            schools.append({"name": linked[1] if linked else m["District"], "id": linked[0] if linked else None,
                            "kind": "regional" if 600 <= org < 800 else "vocational",
                            "contribution": dollars(m["Required_contribution"]),
                            # The district's other member towns
                            "with": [{"name": o["name"], "id": o["id"]} for o in member_towns[m["Org4Code"]]
                                     if o["key"] != k]})
        schools_of[k] = schools
        towns.append({"id": slug(names[k]), "name": names[k], "kind": "town", "county": county.get(k),
                      "peer": peer_of.get(k),
                      "pop": series(pd.Series({fy: population.get((k, fy)) for fy in YEARS}), 0),
                      **town_series(dls[dls["key"] == k], population), "districts": schools})

    def average(aid, name, members, **fields):
        """A pooled per-resident average; pop is the members' combined population."""
        return {"id": aid, "name": name, "kind": "average", "members": len(members), **fields,
                "pop": series(pd.Series({fy: sum(population.get((k, fy), 0) for k in members) for fy in YEARS}), 0),
                **town_series(dls[dls["key"].isin(members)], population)}

    averages = [average("western-towns", "Western MA towns average", pooled)]
    for pid, lo, hi, name, ref, label in TOWN_PEERS:
        members = [k for k in pooled if peer_of.get(k) == pid]
        averages.append(average(pid, name, members, ref=ref, label=label))
    out = {
        "meta": {
            "years": YEARS, "latest": latest, "start": START_YEAR, "built": dt.date.today().isoformat(),
            "members_fy": latest, "excluded": sorted(TOWN_AVERAGE_EXCLUDES),
            # July 1 of this year sets each town's peer group
            "peer_year": latest - 1,
            "sources": {
                "finances": "DLS Municipal Databank, Schedule A general fund revenues and expenditures",
                "population": "Census Bureau subcounty population estimates, July 1 at the start of each fiscal year "
                              "(FY2021: April 1, 2020 estimates base)",
                "members": f"DESE FY{latest} Chapter 70 summary for regional districts",
            },
        },
        "towns": averages + towns,
    }
    def town_link(eid, k):
        """A district's member town: what it must contribute to this district, and its other districts."""
        mine = next(s for s in schools_of[k] if s["id"] == eid)
        return {"id": slug(names[k]), "name": names[k], "contribution": mine["contribution"],
                "also": [{"name": s["name"], "id": s["id"], "kind": s["kind"]} for s in schools_of[k] if s is not mine]}

    town_links = {eid: [town_link(eid, k) for k in ks] for eid, ks in district_towns.items()}
    return out, town_links


def write_csvs(out_dir: Path, districts, towns_out, census, latest):
    """The source data behind the explorer, as CSV files beside its JSON: one row per
    district-year or town-year, values as each agency published them, and a source_url on
    every row. districts: (entity id, name, Org4Code) for the explorer's districts."""
    orgs = {org for _, _, org in districts}
    written = []

    def save(name, frame):
        frame.to_csv(out_dir / name, index=False)
        written.append(f"{name} ({len(frame):,} rows, {(out_dir / name).stat().st_size / 1024:.0f} KB)")

    # DESE expenditures per pupil and enrollment, wide: one column per DESE subcategory
    exp = pd.read_csv("data/dese_expenditures.csv", dtype={"DIST_CODE": str})
    exp = exp[exp["DIST_CODE"].astype(int).floordiv(10000).isin(orgs) & exp["SY"].between(START_YEAR, latest)
              & exp["IND_CAT"].isin(["Expenditures Per Pupil", "Student Enrollment"])]
    exp = exp.assign(column=exp["IND_CAT"].map({"Expenditures Per Pupil": "Per pupil: ", "Student Enrollment": "Enrollment: "})
                     + exp["IND_SUBCAT"])
    wide = exp.pivot_table(index=["SY", "DIST_CODE", "DIST_NAME"], columns="column", values="IND_VALUE", aggfunc="first")
    wide = wide.reset_index().rename(columns={"SY": "fiscal_year", "DIST_CODE": "dese_code", "DIST_NAME": "district"})
    wide["source_url"] = URL_DESE_EXPEND
    wide["source_updated"] = json.loads(Path("data/dese_sources.json").read_text())["expenditures_last_updated"]
    save("districts-spending-per-pupil.csv", wide.sort_values(["district", "fiscal_year"]))

    # DESE Chapter 70 profiles, DESE's own columns
    c70 = pd.read_csv(C70_CSV, dtype={"Org4Code": str})
    c70 = c70[c70["Org4Code"].astype(int).isin(orgs) & c70["fy"].between(START_YEAR, latest)].copy()
    # FY2023 required and actual NSS that update_dese_data.py restored from DESE's August 2025 file
    raw = pd.read_excel("data/dese_c70_profile.xlsx", sheet_name="DataC70", header=None, dtype=object).iloc[4:, [1, 3, 8, 9, 10]]
    raw.columns = ["Org4Code", "fy", "rqdnss", "rqdnss2", "actualNSS"]
    raw = raw[raw["fy"] == 2023].assign(Org4Code=lambda d: d["Org4Code"].astype(str).str.zfill(4)).set_index("Org4Code")
    def note(r):
        notes = []
        if r["fy"] == 2023 and r["Org4Code"] in raw.index and any(
                abs(float(raw.at[r["Org4Code"], f] or 0) - float(r[f] or 0)) > 1 for f in ("rqdnss", "rqdnss2", "actualNSS")):
            notes.append("FY2023 rqdnss, rqdnss2 and actualNSS from DESE's August 2025 profile: the March 2026 "
                         "profile assigns this year's regional district values to the wrong districts")
        if not r["actualNSS"] or float(r["actualNSS"]) <= 0:
            notes.append("actualNSS 0 means DESE has not reported actual spending; the explorer treats it as missing")
        return "; ".join(notes)
    c70["note"] = c70.apply(note, axis=1)
    # The explorer's name for each district; DESE's profile names some differently ("ADAMS CHESHIRE")
    explorer_name = {org: name for _, name, org in districts}
    c70["district"] = c70["Org4Code"].astype(int).map(explorer_name)
    c70["profile_name"] = c70["District"].str.strip()
    c70 = c70.rename(columns={"fy": "fiscal_year", "Org4Code": "dese_org_code"})
    c70["source_url"] = URL_C70_PROFILE
    c70["source_dated"] = json.loads(Path("data/dese_sources.json").read_text())["c70_profile_date"]
    save("districts-chapter70.csv", c70[["fiscal_year", "dese_org_code", "district", "profile_name", "distfoundenro", "distfoundbudget",
                                         "distrlc", "c70aid", "rqdnss", "rqdnss2", "actualNSS", "note", "source_url",
                                         "source_dated"]].sort_values(["district", "fiscal_year"]))

    # Census revenue by the explorer's sources, in dollars, regional districts only
    if census is not None:
        reg = census[census["DIST_CODE"].floordiv(10000).isin(orgs) & (census["DIST_CODE"] >= REGIONAL_MIN_CODE)].copy()
        out = reg[["FY", "DIST_CODE", "NCESID", "NAME"]].rename(columns={"FY": "fiscal_year", "DIST_CODE": "dese_code",
                                                                       "NCESID": "nces_id", "NAME": "census_name"})
        out["dese_code"] = out["dese_code"].astype(int).astype(str).str.zfill(8)  # as DESE writes it
        out.insert(3, "district", out["dese_code"].astype(int).floordiv(10000).map({org: n for _, n, org in districts}))
        for key, items in REVENUE.items():
            formula = "".join(i if k == 0 else (i if i.startswith("-") else f"+{i}") for k, i in enumerate(items))
            out[f"{REVENUE_LABELS[key]} ({formula})"] = sum(
                reg[i.lstrip("-")] * (-1 if i.startswith("-") else 1) for i in items) * 1000
        out["Total revenue (TOTALREV)"] = reg["TOTALREV"] * 1000
        out["in_district_fte"] = reg["FTE"]
        # Census's FY2022 text file is truncated, so that year was read from its Excel file
        out["source_url"] = [URL_CENSUS_FILE.format(fy=fy, yy=fy % 100, ext="xlsx" if fy == 2022 else "txt")
                             for fy in out["fiscal_year"]]
        save("districts-revenue.csv", out.sort_values(["census_name", "fiscal_year"]))

    if towns_out:
        keys = {norm(x["name"]): x for x in towns_out["towns"] if x["kind"] == "town"}
        dls = pd.read_csv(DLS_CSV, dtype={"DOR_CODE": str})
        dls = dls[dls["FY"].between(START_YEAR, latest) & dls["Municipality"].map(norm).isin(keys)]
        pop = pd.read_csv(POP_CSV).assign(key=lambda d: d["Town"].map(norm))
        pop = pop.set_index(["key", "FY"])
        for kind, total, name in [("Expenditures", "Total Expenditures", "towns-spending.csv"),
                                  ("Revenues", "Total Revenues", "towns-revenue.csv")]:
            w = dls[dls["Type"] == kind].pivot_table(index=["FY", "DOR_CODE", "Municipality"], columns="Category",
                                                     values="Amount", aggfunc="sum").reset_index()
            w = w[w[total] > 0]  # years a town filed
            cols = [c for c in w.columns if c not in ("FY", "DOR_CODE", "Municipality", total)] + [total]
            k = w["Municipality"].map(norm)
            frame = pd.DataFrame({"fiscal_year": w["FY"], "dor_code": w["DOR_CODE"], "town": w["Municipality"],
                                  "county": [keys[x]["county"] for x in k]})
            frame = pd.concat([frame, w[cols]], axis=1)
            frame["population"] = [pop.at[(x, fy), "Population"] for x, fy in zip(k, w["FY"])]
            # "POPESTIMATE2024" -> "July 1, 2024"; "ESTIMATESBASE2020" -> "April 1, 2020 estimates base"
            frame["population_estimate"] = [
                "April 1, 2020 estimates base" if e == "ESTIMATESBASE2020" else f"July 1, {e[-4:]}"
                for e in (pop.at[(x, fy), "Estimate"] for x, fy in zip(k, w["FY"]))]
            frame["source_url"] = URL_DLS
            frame["population_source_url"] = [URL_POP + pop.at[(x, fy), "File"] for x, fy in zip(k, w["FY"])]
            save(name, frame.sort_values(["town", "fiscal_year"]))

        rows = []
        for x in towns_out["towns"]:
            for d in x.get("districts") or []:
                rows.append({"town": x["name"], "county": x["county"], "school_district": d["name"],
                             "district_kind": {"town": "town district"}.get(d["kind"], d["kind"]),
                             "in_explorer": "yes" if d["id"] else "no",
                             f"fy{latest}_required_local_contribution": d["contribution"],
                             "source_url": URL_C70_PROFILE if d["kind"] == "town" else URL_C70_REGIONAL.format(fy=latest)})
        save("towns-school-districts.csv", pd.DataFrame(rows))

    cpi = pd.read_csv("data/inflation_indices.csv")
    cpi = cpi[cpi["fiscal_year"].between(START_YEAR, latest)][["fiscal_year", "cpi_u_northeast"]]
    save("cpi.csv", cpi.assign(average_of="July-June monthly CPI-U Northeast, not seasonally adjusted", source_url=URL_BLS))
    print(f"{out_dir}: " + "; ".join(written))


def main(out_path: Path):
    df, reg, c70 = load_data()
    cohorts = get_western_cohort_districts(df, reg)
    proper = {n.lower(): n for n in df["DIST_NAME"].unique()}
    code = df.assign(name=df["DIST_NAME"].str.lower()).groupby("name")["DIST_CODE"].first().astype(int).to_dict()
    idx = pd.read_csv("data/inflation_indices.csv").set_index("fiscal_year")["cpi_u_northeast"]
    census = pd.read_csv(CENSUS_CSV, dtype={"Org4Code": str, "NCESID": str}) if CENSUS_CSV.exists() else None
    if census is None:
        print(f"  [WARN] {CENSUS_CSV} not found (run update_census_data.py); no revenue data")
    else:
        fte = df[(df["IND_CAT"].str.lower() == "student enrollment") &
                 (df["IND_SUBCAT"].str.lower() == IN_DISTRICT_FTE_KEY)]
        fte = fte.rename(columns={"YEAR": "FY", "IND_VALUE": "FTE"})[["DIST_CODE", "FY", "FTE"]]
        census = census.merge(fte.astype({"DIST_CODE": int, "FY": int}), on=["DIST_CODE", "FY"], how="left")

    entities = []
    # Aggregates first: Western MA (excluding Springfield) and each cohort
    western = [d for k in ["TINY", "SMALL", "MEDIUM", "LARGE"] for d in cohorts.get(k, [])]
    groups = [("western-ma", "Western MA average", "western", None, western)]
    groups += [(f"cohort-{k.lower()}", f"{get_cohort_short_label(k)} cohort average", "cohort", k, cohorts[k])
               for k in ["TINY", "SMALL", "MEDIUM", "LARGE"] if cohorts.get(k)]
    for eid, name, kind, cohort, members in groups:
        bucket = "all_western" if kind == "western" else cohort.lower()
        _t, piv, _sum, lines_mean = prepare_western_epp_lines(df, reg, bucket, c70, districts=members)
        nss, _e, foundation = prepare_aggregate_nss_ch70_weighted(df, c70, members)
        lines = dict(lines_mean)
        if foundation is not None and len(foundation):
            lines["Foundation Enrollment"] = foundation
        regional = [code[d] for d in members if code.get(d, 0) >= REGIONAL_MIN_CODE]
        rev = {"rev_members": len(regional)}
        if census is not None and len(regional) >= MIN_REVENUE_MEMBERS:
            rev |= revenue_fields(census, regional, True)
        entities.append({"id": eid, "name": name, "kind": kind, "cohort": cohort, "members": len(members),
                         **pack(piv, lines, nss), **rev})

    # Districts, grouped by cohort
    for k in COHORT_ORDER:
        for d in sorted(cohorts.get(k, [])):
            name = proper.get(d, d.title())
            piv, lines = prepare_district_epp_lines(df, name, c70)
            nss, _e, _f = prepare_district_nss_ch70(df, c70, name)
            regional = code[d] >= REGIONAL_MIN_CODE
            entities.append({"id": slug(name), "name": name, "kind": "district", "cohort": k,
                             "regional": regional,
                             "fte": round(get_indistrict_fte_for_year(df, d, LATEST_YEAR), 1),
                             "compare": f"cohort-{k.lower()}" if k != "SPRINGFIELD" else "western-ma",
                             **pack(piv, lines, nss), **revenue_fields(census, [code[d]], regional)})

    out = {
        "meta": {
            "title": "Western MA per-pupil expenditure",
            "years": YEARS, "latest": LATEST_YEAR, "start": START_YEAR,
            "built": dt.date.today().isoformat(),
            "categories": CANON_CATS_BOTTOM_TO_TOP,
            "cohorts": {k: {"label": get_cohort_label(k), "short": get_cohort_short_label(k),
                            "range": [r if r != float("inf") else None for r in get_cohort_range(k)]}
                        for k in COHORT_ORDER},
            "cpi": {str(y): float(idx[y]) for y in YEARS if y in idx.index},
            "revenue": ({"first": int(census["FY"].min()), "latest": int(census["FY"].max()),
                         "min_members": MIN_REVENUE_MEMBERS} if census is not None else None),
            "sources": {
                "expenditures": "DESE District Expenditures by Spending Category (E2C Hub er3w-dyti)",
                "chapter70": "DESE Chapter 70 District Profiles",
                "cpi": "BLS CPI-U Northeast (CUUR0100SA0), July-June fiscal-year averages",
                "revenue": "Census Bureau Annual Survey of School System Finances (F-33), Massachusetts",
            },
        },
        "entities": entities,
    }
    towns_out = None
    districts = [(e["id"], e["name"], code[e["name"].lower()] // 10000) for e in entities if e["kind"] == "district"]
    if all(p.exists() for p in (DLS_CSV, POP_CSV, MEMBERS_CSV)):
        towns_out, town_links = town_data(districts, LATEST_YEAR)
        for e in entities:
            if e["kind"] == "district":
                e["towns"] = town_links[e["id"]]
    else:
        print("  [WARN] town data missing (run update_dls_data.py, update_census_data.py, "
              "update_dese_data.py); no town view")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, separators=(",", ":")))
    n_d = sum(e["kind"] == "district" for e in entities)
    print(f"{out_path}: {n_d} districts, {len(entities) - n_d} aggregates, "
          f"{out_path.stat().st_size / 1024:.0f} KB")
    write_csvs(out_path.parent, districts, towns_out, census, LATEST_YEAR)
    if towns_out:
        towns_path = out_path.with_name("towns.json")
        towns_path.write_text(json.dumps(towns_out, separators=(",", ":")))
        n_t = sum(x["kind"] == "town" for x in towns_out["towns"])
        print(f"{towns_path}: {n_t} towns, {len(towns_out['towns']) - n_t} averages, "
              f"{towns_path.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output/explorer.json"))
