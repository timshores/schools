"""
Export the data behind the interactive explorer at timshores.com/mass-school-spending.

Every Massachusetts school district (traditional, vocational and charter, less a
few special-purpose state schools) and every city and town, FY START_YEAR..
LATEST_YEAR. Districts: per-pupil expenditure by category, enrollment, and
Chapter 70 / net school spending per foundation pupil, from the same functions
the PDF report uses, so the explorer and the report agree; districts join size
cohorts by the report's rule (get_enrollment_group), applied statewide.
Averages: statewide, each EOHHS region, each cohort statewide and within each
region. As the report kept Springfield out of the Western MA average, region and
statewide averages leave out the very large districts (over 10,000 FTE), which
have their own average; the Western MA average is the report's.

Revenue by source comes from the Census Bureau's school finance survey
(data/census_f33_ma.csv, written by update_census_data.py), per in-district
FTE pupil like spending, for regional districts only: for town-run districts
Census's local revenue leaves out much of what the town spends on its schools
(Amherst FY2024: $18.4M of revenue against $37.6M of spending). Town-run
districts get Census's latest totals instead, so the explorer can say so.
Aggregates pool their regional members, weighted by in-district FTE.

Towns: general fund spending and revenue per resident (DLS Schedule A), with
population peer groups and the same kinds of averages; the cities whose school
districts are very large stand apart from them.

Writes into a copy of the website (github.com/timshores/mywebsite): the JSON to
src/data/massppe/, where the page imports it, so it ships as compressed, hashed
JavaScript chunks (the host compresses JavaScript but not JSON). explorer.json
and towns.json hold the averages and a light list; each region's full district
and town data is its own file under districts/ and towns/, loaded when first
needed. The CSVs of source data go to public/mass-school-spending/data/.

    python export_explorer_data.py ../../mywebsite
"""

import datetime as dt
import json
import re
import sys
from pathlib import Path

import pandas as pd

from school_shared import (CANON_CATS_BOTTOM_TO_TOP, IN_DISTRICT_FTE_KEY, LATEST_YEAR, START_YEAR, get_cohort_label,
                           get_cohort_range, get_cohort_short_label, get_enrollment_group, latest_indistrict_fte,
                           load_data, prepare_aggregate_nss_ch70_weighted, prepare_district_epp_lines,
                           prepare_district_nss_ch70, prepare_western_epp_lines)

YEARS = list(range(START_YEAR, LATEST_YEAR + 1))
ENROLL = {"In-District FTE Pupils": "in_district", "Foundation Enrollment": "foundation",
          "Out-of-District FTE Pupils": "out_of_district"}
NSS = {"Ch70 Aid": "ch70_aid", "Req NSS (minus Ch70)": "required_local",
       "Actual NSS (minus Req NSS)": "above_required"}
# Districts: every traditional, vocational and charter district with latest-year
# enrollment and spending, less these special-purpose state schools
SCHOOL_TYPES = {"Traditional": "traditional", "Vocational": "vocational", "Charter": "charter"}
EXCLUDE_DISTRICTS = {"institutional schools", "ma academy for math and science",
                     "tec connections academy commonwealth virtual school district",
                     "greater commonwealth virtual district"}
# Size cohorts for traditional districts (the report's), then groups by type
SIZE_COHORTS = ["TINY", "SMALL", "MEDIUM", "LARGE", "XLARGE"]
COHORT_ORDER = SIZE_COHORTS + ["VOCATIONAL", "CHARTER"]
COHORT_TEXT = {  # cohorts the report doesn't define: (label, short label)
    "XLARGE": ("Very large (over 10,000 FTE)", "Very large"),
    "GATEWAY": ("Gateway Cities", "Gateway"),
    "VOCATIONAL": ("Vocational and technical districts", "Vocational"),
    "CHARTER": ("Charter schools", "Charter"),
}
# EOHHS regions, as DESE assigns districts to them (data/cache/regional_data.csv)
REGIONS = {"Western": "Western MA", "Central": "Central MA", "Northeast": "Northeast MA",
           "Metro West": "Metro West", "Southeast": "Southeast MA", "Boston": "Boston area"}
# A cohort average within a region needs this many members; with fewer, comparisons use
# the statewide cohort
MIN_AVERAGE_MEMBERS = 3
# Gateway Cities, as the Commonwealth lists them (M.G.L. c. 23A, s. 3A: population 35,000-
# 250,000, median household income and bachelor's attainment below the state's). The list
# dates from 2014. Their districts and the cities themselves get averages of their own,
# alongside their size cohorts and peer groups.
GATEWAY_CITIES = ["Attleboro", "Barnstable", "Brockton", "Chelsea", "Chicopee", "Everett", "Fall River", "Fitchburg",
                  "Haverhill", "Holyoke", "Lawrence", "Leominster", "Lowell", "Lynn", "Malden", "Methuen",
                  "New Bedford", "Peabody", "Pittsfield", "Quincy", "Revere", "Salem", "Springfield", "Taunton",
                  "Westfield", "Worcester"]

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


# Town view: every city and town, general fund spending and revenue per resident
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
# Peer groups by population: the estimate behind the latest fiscal year (July 2024 for
# FY2025), as district cohorts use latest-year enrollment. Round bounds; statewide
# FY2025: 26, 42, 98, 103, 55 and 27 towns (the last 17 without the very large cities).
# (id suffix, lowest population, first population above the group, label)
TOWN_PEERS = [
    ("under-1000", 0, 1000, "under 1,000"),
    ("1000-2499", 1000, 2500, "1,000–2,499"),
    ("2500-9999", 2500, 10000, "2,500–9,999"),
    ("10000-24999", 10000, 25000, "10,000–24,999"),
    ("25000-49999", 25000, 50000, "25,000–49,999"),
    ("50000-plus", 50000, None, "50,000 or more"),
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


# Region for the few towns with no school district of their own (Monroe, Tyringham,
# Gosnold...): their county's, where the county lies in one EOHHS region
COUNTY_REGION = {"Berkshire": "Western", "Franklin": "Western", "Hampden": "Western", "Hampshire": "Western",
                 "Worcester": "Central", "Barnstable": "Southeast", "Bristol": "Southeast", "Dukes": "Southeast",
                 "Nantucket": "Southeast", "Plymouth": "Southeast"}
# DESE's Chapter 70 tables call Manchester-by-the-Sea "Manchester"
TOWN_ALIASES = {"manchester": "manchesterbythesea"}


def norm(name: str) -> str:
    """A town name as DLS, DESE and Census all spell it ("Manchester By The Sea", "Manchester-by-the-Sea")."""
    key = re.sub(r"[^a-z]", "", name.lower())
    return TOWN_ALIASES.get(key, key)


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


def explorer_districts(df, reg):
    """The explorer's districts: the report's rule for its Western MA cohorts
    (school_shared.get_western_cohort_districts: latest-year in-district FTE and
    per-pupil spending both present, cohort by get_enrollment_group) applied to every
    region and school type. Each is a dict: key (lowercase name), name, code (DESE
    DIST_CODE), org (four-digit code), type, cohort, region (None if DESE gives none)."""
    present = {n.lower(): n for n in df["DIST_NAME"].unique()}
    code = df.assign(key=df["DIST_NAME"].str.lower()).groupby("key")["DIST_CODE"].first().astype(int)
    latest = int(df["YEAR"].max())
    epp = df[(df["IND_CAT"].str.lower() == "expenditures per pupil") & (df["YEAR"] == latest)
             & ~df["IND_SUBCAT"].str.lower().isin(["total expenditures", "total in-district expenditures"])]
    total_ppe = epp.groupby(epp["DIST_NAME"].str.lower())["IND_VALUE"].sum()
    out = []
    for _, r in reg[reg["SCHOOL_TYPE"].isin(SCHOOL_TYPES)].drop_duplicates("DIST_NAME").iterrows():
        key = r["DIST_NAME"].lower()
        if key not in present or key in EXCLUDE_DISTRICTS:
            continue
        fte = latest_indistrict_fte(df, key)
        if fte <= 0 or total_ppe.get(key, 0) <= 0:
            continue
        kind = SCHOOL_TYPES[r["SCHOOL_TYPE"]]
        if kind == "traditional":
            group = get_enrollment_group(fte)
            cohort = "XLARGE" if group == "SPRINGFIELD" else group
        else:
            cohort = kind.upper()
        out.append({"key": key, "name": present[key], "code": int(code[key]), "org": int(code[key]) // 10000,
                    "type": kind, "cohort": cohort, "fte": fte,
                    "region": r["EOHHS_REGION"] if r["EOHHS_REGION"] in REGIONS else None})
    return out


def chunk_of(region, kind="traditional"):
    """The data file a district or town loads from: its region, or its school type."""
    if kind != "traditional":
        return kind
    return slug(region) if region else "other"


def cohort_name(k, region=None):
    """'Statewide Tiny cohort average', 'Western MA Tiny cohort average', 'Very large districts average'"""
    if k in ("XLARGE", "VOCATIONAL", "CHARTER"):
        return {"XLARGE": "Very large districts average", "VOCATIONAL": "Vocational districts average",
                "CHARTER": "Charter schools average"}[k]
    return f"{REGIONS[region] if region else 'Statewide'} {get_cohort_short_label(k)} cohort average"


def town_data(dists, reg, latest):
    """The town view: (towns.json content, {chunk: full towns}, every full town, each
    district's towns). dists: explorer_districts() with each district's explorer id."""
    dls = pd.read_csv(DLS_CSV, dtype={"DOR_CODE": str})
    dls = dls[dls["FY"].isin(YEARS)].assign(key=lambda d: d["Municipality"].map(norm))
    pop = pd.read_csv(POP_CSV).assign(key=lambda d: d["Town"].map(norm))
    population = {(k, fy): p for k, fy, p in zip(pop["key"], pop["FY"], pop["Population"])}
    county = pop.sort_values("FY").groupby("key")["County"].last().to_dict()
    members = pd.read_csv(MEMBERS_CSV, dtype={"Org4Code": str}).assign(key=lambda d: d["Town"].map(norm))
    c70 = pd.read_csv(C70_CSV, dtype={"Org4Code": str})
    rlc = c70[c70["fy"] == latest].set_index("Org4Code")["distrlc"].to_dict()
    names = dls.drop_duplicates("key").set_index("key")["Municipality"].to_dict()
    keys = sorted(names)
    missing = [k for k in keys if (k, latest) not in population]
    if missing:
        raise ValueError(f"Towns without population data: {missing}")

    by_org = {f"{d['org']:04d}": d for d in dists}
    own_of = {norm(d["name"]): d for d in dists if d["code"] < REGIONAL_MIN_CODE and d["type"] == "traditional"
              and norm(d["name"]) in names}
    member_towns = {org: [{"name": names.get(mk, town), "id": slug(names[mk]) if mk in names else None, "key": mk}
                          for mk, town in zip(g["key"], g["Town"])]
                    for org, g in members.groupby("Org4Code")}

    # Each district's towns: its member towns, or the town of the same name
    district_towns, towns_kind = {}, {}
    for d in dists:
        org = f"{d['org']:04d}"
        if org in member_towns:
            district_towns[d["id"]], towns_kind[d["id"]] = [m["key"] for m in member_towns[org]], "members"
        elif norm(d["name"]) in own_of:
            district_towns[d["id"]], towns_kind[d["id"]] = [norm(d["name"])], "town"

    # The cities whose own school districts are very large stand apart from the averages
    apart = {k for k, d in own_of.items() if d["cohort"] == "XLARGE"}
    pooled = [k for k in keys if k not in apart]
    gateway = {norm(c) for c in GATEWAY_CITIES}
    if not gateway <= set(keys):
        raise ValueError(f"Gateway Cities missing from the town list: {sorted(gateway - set(keys))}")

    def peer_of(k):
        if k in apart:
            return None
        n = population[(k, latest)]
        return next(f"towns-{sfx}" for sfx, lo, hi, _ in TOWN_PEERS if lo <= n and (hi is None or n < hi))

    full, schools_of, region_of = [], {}, {}
    for k in keys:
        schools = []
        own = own_of.get(k)
        if own:
            schools.append({"name": own["name"], "id": own["id"], "kind": "town",
                            "contribution": dollars(rlc.get(f"{own['org']:04d}")), "with": []})
        for _, m in members[members["key"] == k].iterrows():
            org = int(m["Org4Code"])
            linked = by_org.get(m["Org4Code"])
            schools.append({"name": linked["name"] if linked else m["District"], "id": linked["id"] if linked else None,
                            "kind": "regional" if 600 <= org < 800 else "vocational",
                            "contribution": dollars(m["Required_contribution"]),
                            "with": [{"name": o["name"], "id": o["id"]} for o in member_towns[m["Org4Code"]]
                                     if o["key"] != k]})
        schools_of[k] = schools
        # Region: the town's own district's, else its regional academic district's, else any of
        # its districts', else its county's (regional_data.csv's district_code is not DESE's
        # code for districts whose region was inferred from the host town, so match districts)
        mine = ([own] if own else []) + sorted(
            (by_org[o] for o in members.loc[members["key"] == k, "Org4Code"] if o in by_org),
            key=lambda d: d["type"] != "traditional")
        region_of[k] = next((d["region"] for d in mine if d["region"]), None) or COUNTY_REGION.get(county.get(k))
        full.append({"id": slug(names[k]), "name": names[k], "kind": "town", "county": county.get(k),
                     "region": region_of[k], "peer": peer_of(k), "gateway": k in gateway,
                     "pop": series(pd.Series({fy: population.get((k, fy)) for fy in YEARS}), 0),
                     **town_series(dls[dls["key"] == k], population), "districts": schools})

    def average(aid, name, scope, group, region, ks, **fields):
        """A pooled per-resident average; pop is the members' combined population."""
        return {"id": aid, "name": name, "kind": "average", "scope": scope, "peer": group, "region": region,
                "members": len(ks), "chunk": slug(region) if region else "statewide", **fields,
                "pop": series(pd.Series({fy: sum(population.get((k, fy), 0) for k in ks) for fy in YEARS}), 0),
                **town_series(dls[dls["key"].isin(ks)], population)}

    averages = [average("towns-statewide", "Statewide towns average", "state", None, None, pooled, cmp={})]
    for rk, rl in REGIONS.items():
        ks = [k for k in pooled if region_of[k] == rk]
        averages.append(average(f"towns-region-{slug(rk)}", f"{rl} towns average", "region", None, rk, ks,
                                cmp={"state": "towns-statewide"}))
    have = {a["id"] for a in averages}
    for sfx, lo, hi, label in TOWN_PEERS:
        pid = f"towns-{sfx}"
        phrase = f"towns {label}" if label.startswith("under") else f"towns of {label}"
        ks = [k for k in pooled if peer_of(k) == pid]
        averages.append(average(pid, f"{phrase[0].upper()}{phrase[1:]} average", "peer", pid, None, ks,
                                ref=f"the average for {phrase}", label=f"{label} residents", cmp={"state": "towns-statewide"}))
        for rk, rl in REGIONS.items():
            ks = [k for k in pooled if peer_of(k) == pid and region_of[k] == rk]
            if len(ks) >= MIN_AVERAGE_MEMBERS:
                averages.append(average(f"{pid}-{slug(rk)}", f"{rl} {phrase} average", "region-peer", pid, rk, ks,
                                        ref=f"the average for {rl} {phrase}", label=f"{label} residents",
                                        cmp={"peer_state": pid, "region": f"towns-region-{slug(rk)}",
                                             "state": "towns-statewide"}))
    # Gateway Cities: the Commonwealth's list, very large cities included
    averages.append(average("towns-gateway", "Gateway Cities average", "group", None, None, sorted(gateway),
                            cmp={"state": "towns-statewide"}))
    for a in averages:
        a["cmp"]["gateway"] = "towns-gateway" if a["id"] != "towns-gateway" else None
    have = {a["id"] for a in averages}
    chunks = {}
    for a in averages:
        chunks.setdefault(a["chunk"], []).append(a)
    for x in full:
        pid, rk = x["peer"], x["region"]
        x["cmp"] = {"peer": f"{pid}-{slug(rk)}" if pid and rk and f"{pid}-{slug(rk)}" in have else None,
                    "peer_state": pid, "region": f"towns-region-{slug(rk)}" if rk else None, "state": "towns-statewide",
                    "gateway": "towns-gateway"}
        x["chunk"] = chunk_of(rk)
        chunks.setdefault(x["chunk"], []).append(x)

    index = {
        "meta": {
            "years": YEARS, "latest": latest, "start": START_YEAR, "built": dt.date.today().isoformat(),
            "members_fy": latest, "regions": REGIONS,
            "apart": sorted(names[k] for k in apart),
            # July 1 of this year sets each town's peer group
            "peer_year": latest - 1,
            "peers": [{"id": f"towns-{sfx}", "label": f"{label} residents"} for sfx, _, _, label in TOWN_PEERS],
            "sources": {
                "finances": "DLS Municipal Databank, Schedule A general fund revenues and expenditures",
                "population": "Census Bureau subcounty population estimates, July 1 at the start of each fiscal year "
                              "(FY2021: April 1, 2020 estimates base)",
                "members": f"DESE FY{latest} Chapter 70 summary for regional districts",
            },
        },
        "averages": [{"id": a["id"], "name": a["name"], "scope": a["scope"], "peer": a["peer"], "region": a["region"],
                      "chunk": a["chunk"]} for a in averages],
        "towns": [{"id": x["id"], "name": x["name"], "county": x["county"], "region": x["region"],
                   "peer": x["peer"], "chunk": x["chunk"]} for x in full],
    }

    def town_link(eid, k):
        """A district's town: what it must contribute to this district, and its other districts."""
        mine = next(s for s in schools_of[k] if s["id"] == eid)
        return {"id": slug(names[k]), "name": names[k], "contribution": mine["contribution"],
                "also": [{"name": s["name"], "id": s["id"], "kind": s["kind"]} for s in schools_of[k] if s is not mine]}

    links = {eid: {"towns": [town_link(eid, k) for k in ks], "towns_kind": towns_kind[eid]}
             for eid, ks in district_towns.items()}
    return index, chunks, full, links


def write_csvs(out_dir: Path, districts, all_towns, census, latest):
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

    if all_towns:
        keys = {norm(x["name"]): x for x in all_towns}
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
        for x in all_towns:
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


def write_json(path: Path, content) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, separators=(",", ":")))
    return path.stat().st_size


def main(site: Path):
    df, reg, c70 = load_data()
    idx = pd.read_csv("data/inflation_indices.csv").set_index("fiscal_year")["cpi_u_northeast"]
    census = pd.read_csv(CENSUS_CSV, dtype={"Org4Code": str, "NCESID": str}) if CENSUS_CSV.exists() else None
    if census is None:
        print(f"  [WARN] {CENSUS_CSV} not found (run update_census_data.py); no revenue data")
    else:
        fte = df[(df["IND_CAT"].str.lower() == "student enrollment") &
                 (df["IND_SUBCAT"].str.lower() == IN_DISTRICT_FTE_KEY)]
        fte = fte.rename(columns={"YEAR": "FY", "IND_VALUE": "FTE"})[["DIST_CODE", "FY", "FTE"]]
        census = census.merge(fte.astype({"DIST_CODE": int, "FY": int}), on=["DIST_CODE", "FY"], how="left")

    dists = explorer_districts(df, reg)
    gateway = {norm(c) for c in GATEWAY_CITIES}
    for d in dists:
        d["id"] = slug(d["name"])
        d["gateway"] = d["type"] == "traditional" and d["code"] < REGIONAL_MIN_CODE and norm(d["name"]) in gateway
    if sum(d["gateway"] for d in dists) != len(GATEWAY_CITIES):
        raise ValueError(f"Expected {len(GATEWAY_CITIES)} Gateway City districts: "
                         f"{sorted(c for c in gateway if not any(norm(d['name']) == c for d in dists))} missing")
    if len({d["id"] for d in dists}) != len(dists):
        raise ValueError("Two districts share an id")
    print(f"  {len(dists)} districts: " + ", ".join(f"{k} {sum(d['cohort'] == k for d in dists)}" for k in COHORT_ORDER))

    # Averages. Region and statewide averages are traditional districts, less the very large.
    pooled = [d for d in dists if d["type"] == "traditional" and d["cohort"] != "XLARGE"]
    groups = [("statewide", "Statewide average", "state", None, None, pooled, {})]
    for rk, rl in REGIONS.items():
        groups.append((f"region-{slug(rk)}", f"{rl} average", "region", None, rk,
                       [d for d in pooled if d["region"] == rk], {"state": "statewide"}))
    for k in COHORT_ORDER:
        groups.append((f"cohort-{k.lower()}", cohort_name(k), "cohort", k, None,
                       [d for d in dists if d["cohort"] == k], {"state": "statewide"}))
    groups.append(("cohort-gateway", "Gateway City districts average", "group", "GATEWAY", None,
                   [d for d in dists if d["gateway"]], {"state": "statewide"}))
    for k in SIZE_COHORTS[:-1]:
        for rk in REGIONS:
            ms = [d for d in dists if d["cohort"] == k and d["region"] == rk]
            if len(ms) >= MIN_AVERAGE_MEMBERS:
                groups.append((f"cohort-{k.lower()}-{slug(rk)}", cohort_name(k, rk), "region-cohort", k, rk, ms,
                               {"cohort_state": f"cohort-{k.lower()}", "region": f"region-{slug(rk)}", "state": "statewide"}))
    have = {g[0] for g in groups}

    averages = []
    for eid, name, scope, cohort, region, members, cmp in groups:
        keys = [d["key"] for d in members]
        _t, piv, _sum, lines_mean = prepare_western_epp_lines(df, reg, "all_western", c70, districts=keys)
        nss, _e, foundation = prepare_aggregate_nss_ch70_weighted(df, c70, keys)
        lines = dict(lines_mean)
        if foundation is not None and len(foundation):
            lines["Foundation Enrollment"] = foundation
        regional = [d["code"] for d in members if d["code"] >= REGIONAL_MIN_CODE]
        rev = {"rev_members": len(regional)}
        if census is not None and len(regional) >= MIN_REVENUE_MEMBERS:
            rev |= revenue_fields(census, regional, True)
        cmp = {**cmp, "gateway": "cohort-gateway" if eid != "cohort-gateway" else None}
        averages.append({"id": eid, "name": name, "kind": "average", "scope": scope, "cohort": cohort,
                         "region": region, "members": len(members), "cmp": cmp,
                         "chunk": slug(region) if region else "statewide", **pack(piv, lines, nss), **rev})

    full = []
    for d in sorted(dists, key=lambda d: d["name"]):
        piv, lines = prepare_district_epp_lines(df, d["name"], c70)
        nss, _e, _f = prepare_district_nss_ch70(df, c70, d["name"])
        k, rk = d["cohort"], d["region"]
        regional = d["code"] >= REGIONAL_MIN_CODE
        rc = f"cohort-{k.lower()}-{slug(rk)}" if rk else None
        full.append({"id": d["id"], "name": d["name"], "kind": "district", "type": d["type"], "cohort": k,
                     "region": rk, "regional": regional, "fte": round(d["fte"], 1), "gateway": d["gateway"],
                     "cmp": {"cohort": rc if rc in have else None, "cohort_state": f"cohort-{k.lower()}",
                             "region": f"region-{slug(rk)}" if rk else None, "state": "statewide",
                             "gateway": "cohort-gateway"},
                     "chunk": chunk_of(rk, d["type"]),
                     **pack(piv, lines, nss), **revenue_fields(census, [d["code"]], regional)})

    towns_index, town_chunks, all_towns = None, {}, []
    if all(p.exists() for p in (DLS_CSV, POP_CSV, MEMBERS_CSV)):
        towns_index, town_chunks, all_towns, links = town_data(dists, reg, LATEST_YEAR)
        for e in full:
            e.update(links.get(e["id"], {}))
    else:
        print("  [WARN] town data missing (run update_dls_data.py, update_census_data.py, "
              "update_dese_data.py); no town view")

    cohorts = {k: {"label": get_cohort_label(k), "short": get_cohort_short_label(k),
                   "range": [r if r != float("inf") else None for r in get_cohort_range(k)]} for k in SIZE_COHORTS[:-1]}
    cohorts |= {k: {"label": COHORT_TEXT[k][0], "short": COHORT_TEXT[k][1],
                    "range": [10001, None] if k == "XLARGE" else None} for k in COHORT_TEXT}
    index = {
        "meta": {
            "title": "Massachusetts school spending",
            "years": YEARS, "latest": LATEST_YEAR, "start": START_YEAR,
            "built": dt.date.today().isoformat(),
            "categories": CANON_CATS_BOTTOM_TO_TOP,
            "cohorts": {k: cohorts[k] for k in COHORT_ORDER + ["GATEWAY"]},
            "regions": REGIONS,
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
        # A light list for the menus; each entity's data is in its chunk's file
        "averages": [{"id": a["id"], "name": a["name"], "scope": a["scope"], "cohort": a["cohort"],
                      "region": a["region"], "chunk": a["chunk"]} for a in averages],
        "districts": [{"id": e["id"], "name": e["name"], "type": e["type"], "cohort": e["cohort"], "region": e["region"],
                       "regional": e["regional"], "fte": e["fte"], "chunk": e["chunk"]} for e in full],
    }

    json_dir = site / "src" / "data" / "massppe"
    csv_dir = site / "public" / "mass-school-spending" / "data"
    for sub in ("districts", "towns"):  # stale region files would ship as dead chunks
        for f in (json_dir / sub).glob("*.json"):
            f.unlink()
    sizes = {"explorer.json": write_json(json_dir / "explorer.json", index)}
    for chunk in sorted({e["chunk"] for e in full + averages}):
        sizes[f"districts/{chunk}.json"] = write_json(json_dir / "districts" / f"{chunk}.json",
                                                      {"entities": [e for e in averages + full if e["chunk"] == chunk]})
    if towns_index:
        sizes["towns.json"] = write_json(json_dir / "towns.json", towns_index)
        for chunk, xs in sorted(town_chunks.items()):
            sizes[f"towns/{chunk}.json"] = write_json(json_dir / "towns" / f"{chunk}.json", {"towns": xs})
    print(f"{json_dir}: {len(full)} districts, {len(averages)} district averages, {len(all_towns)} towns, "
          f"{len(towns_index['averages']) if towns_index else 0} town averages")
    print("  " + "; ".join(f"{k} {v / 1024:.0f} KB" for k, v in sizes.items()))
    csv_dir.mkdir(parents=True, exist_ok=True)
    write_csvs(csv_dir, [(e["id"], e["name"], e_code // 10000) for e, e_code in
                         ((e, next(d["code"] for d in dists if d["id"] == e["id"])) for e in full)],
               all_towns, census, LATEST_YEAR)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output/site"))
