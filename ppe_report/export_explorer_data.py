"""
Export the data behind the interactive explorer at timshores.com/muni-spending.

Every Massachusetts school district (traditional, vocational and charter, less a
few special-purpose state schools) and every city and town, FY EXPLORER_START (2005, the
first year of DESE's functional measure)..LATEST_YEAR; the PDF report keeps FY2009. Districts: per-pupil expenditure by category, enrollment, and
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
(Amherst FY2024: $18.4M of revenue against $37.6M of spending), and the
survey has no charter schools, so neither has revenue figures (#237).
Aggregates pool their regional members, weighted by in-district FTE.

Payments to out-of-district schools, per out-of-district FTE pupil, come from DESE's
totals, which the nine categories leave out (explorer_common.ood_per_pupil). FY2005-FY2008 come
from DESE's archive (data/dese_expenditures_fy2005_2008.csv), stacked under the Hub's rows
(explorer_common.with_functional). The years before EXPLORER_START, for the "Since 1990"
switch, are explorer_history.py's.

Towns: general fund spending and revenue per resident (DLS Schedule A), with
population groups and the same kinds of averages; the cities whose school
districts are very large stand apart from the statewide and regional ones. Each town's property: assessed
value per resident and parcels by type (the assessors' LA-4), equalized
valuation (EQV) per resident, and the average single-family tax bill, from
update_dls_property_data.py.

Writes into a copy of the website (github.com/timshores/mywebsite): the JSON to
src/data/massppe/, where the page imports it, so it ships as compressed, hashed
JavaScript chunks (the host compresses JavaScript but not JSON). explorer.json
and towns.json hold the averages and a light list; each region's full district
and town data is its own file under districts/ and towns/, loaded when first
needed. The CSVs of source data go to public/muni-spending/data/. Both directories
are written beside the site's and moved into place at the end (staged), and the
export stops before reading anything if an input is missing (check_inputs).

    python export_explorer_data.py ../../mywebsite [--force-recompute]

--force-recompute rebuilds load_data's cache (data/cache) from its sources; it is
rebuilt anyway once a source is newer than it. What this file shares with
explorer_history.py is in explorer_common.py. The steps (main): load, district_list,
district_groups, the district and town entities, explorer_index, write_site.
"""

import argparse
import datetime as dt
import json
import os
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import pandas as pd

from explorer_common import (EXCLUDE_DISTRICTS, EXPLORER_START, FUNC_CSV, HIST_YEARS, REGIONS, YEARS, Hub,
                             is_regional, norm, pack, series, slug, with_functional)
from explorer_history import DAY_CSV, NCES_CSV, history
from school_shared import (CANON_CATS_BOTTOM_TO_TOP, EXCEL_FILE, IN_DISTRICT_FTE_KEY, LATEST_YEAR,
                           get_cohort_definitions, get_enrollment_group, latest_indistrict_fte, load_data)

# Districts: every traditional, vocational and charter district with latest-year enrollment and spending, less
# the special-purpose state schools explorer_common.EXCLUDE_DISTRICTS names
SCHOOL_TYPES = {"Traditional": "traditional", "Vocational": "vocational", "Charter": "charter"}
# Size cohorts for traditional districts (the report's), then groups by type
SIZE_COHORTS = ["TINY", "SMALL", "MEDIUM", "LARGE", "XLARGE"]
COHORT_ORDER = SIZE_COHORTS + ["VOCATIONAL", "CHARTER"]
COHORT_TEXT = {  # cohorts the report doesn't define: (label, short label)
    "XLARGE": ("Very large (over 10,000 FTE)", "Very large"),
    "GATEWAY": ("Gateway Cities", "Gateway"),
    "VOCATIONAL": ("Vocational and technical districts", "Vocational"),
    "CHARTER": ("Charter schools", "Charter"),
}
# A cohort average within a region needs this many members; with fewer, comparisons use
# the statewide cohort
MIN_AVERAGE_MEMBERS = 3
# Gateway Cities, as the Commonwealth lists them (M.G.L. c. 23A, s. 3A: population 35,000-
# 250,000, median household income and bachelor's attainment below the state's). The list
# dates from 2014. Their districts and the cities themselves get averages of their own,
# alongside their enrollment and population groups.
GATEWAY_CITIES = ["Attleboro", "Barnstable", "Brockton", "Chelsea", "Chicopee", "Everett", "Fall River", "Fitchburg",
                  "Haverhill", "Holyoke", "Lawrence", "Leominster", "Lowell", "Lynn", "Malden", "Methuen",
                  "New Bedford", "Peabody", "Pittsfield", "Quincy", "Revere", "Salem", "Springfield", "Taunton",
                  "Westfield", "Worcester"]

CENSUS_CSV = Path("data/census_f33_ma.csv")
# DESE special education spending and Circuit Breaker payments (update_dese_sped_data.py)
SPED_CSV = Path("data/dese_sped_expenditures.csv")
CB_CSV = Path("data/dese_circuit_breaker.csv")
SPED_SOURCES = Path("data/dese_sped_sources.json")
# Special education's categories, bottom to top: in the district's own schools, then out of district, then
# what grants and revolving funds paid (RADAR's Special Education Direct Expenditure Trends)
SPED = ["teaching", "other_instructional", "in_transport", "ma_public", "private", "ood_transport", "health", "grants"]
# A year's claim is paid the next year, in quarterly payments and any supplement; extraordinary relief, the same year
CB_NEXT_YEAR = ["tot_qtly_payment", "addl_suppl_payment"]
BUDGET_CSV = Path("data/dese_nss_budgets.csv")
BUDGET_SOURCES = Path("data/dese_nss_budget_sources.json")
ENROLL_CSV = Path("data/dese_enrollment.csv")
ENROLL_SOURCES = Path("data/dese_enrollment_sources.json")
# Students by race and ethnicity, as the towns' residents are grouped: Hispanic or Latino of any race, each race
# of those who aren't, and the smaller groups together
STUDENT_RACE = {"white": ["wh_pct"], "black": ["baa_pct"], "asian": ["as_pct"], "other_race": ["aian_pct", "nhpi_pct", "mnhl_pct"],
                "hispanic": ["hl_pct"]}
# Students in selected groups: low income through FY2014 and from FY2022, two measures, and economically
# disadvantaged between, each its own series
STUDENT_GROUPS = {"swd": "swd", "el": "el", "low_income": "li", "econ": "ecd", "high_needs": "hn"}
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
# The Chapter 70 profiles' row for the state as a whole
STATE_TOTAL_ORG = "0000"
EXP_CSV = Path("data/dese_expenditures.csv")
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
# Property (update_dls_property_data.py). The LA-4's groups of use codes, gathered into the
# explorer's property types, bottom to top in its stack. Personal property (business
# equipment, utilities) is assessed value but not parcels; exempt property is neither.
LA4_CSV = Path("data/dls_la4_parcels.csv")
EQV_CSV = Path("data/dls_eqv.csv")
SF_BILL_CSV = Path("data/dls_single_family_tax_bill.csv")
LEVY_CSV = Path("data/dls_levy_capacity.csv")
# Reserves, debt, capital projects, Prop 2 1/2 votes, labor force (update_dls_fiscal_data.py)
FREE_CASH_CSV = Path("data/dls_free_cash.csv")
STAB_CSV = Path("data/dls_stabilization.csv")
DEBT_CSV = Path("data/dls_long_term_debt.csv")
AUTH_DEBT_CSV = Path("data/dls_authorized_debt.csv")
RATINGS_CSV = Path("data/dls_bond_ratings.csv")
VOTES_CSV = Path("data/dls_prop25_votes.csv")
CAPITAL_CSV = Path("data/dls_capital_projects.csv")
LABOR_CSV = Path("data/dls_labor_force.csv")
FISCAL_SOURCES = Path("data/dls_fiscal_sources.json")
PROOF_CSV = Path("data/dls_free_cash_proof.csv")
# The free cash proof's lines, grouped for the chart, bottom to top: what carried over, what went unspent, revenue
# above its estimates, and the adjustments (owed taxes and the like) that can take some back
PROOF_GROUPS = {"carried": ["prior_not_appropriated"], "unspent": ["unexpended"], "receipts": ["local_receipts"],
                "state_aid": ["state_aid"], "other_rev": ["not_estimated", "overlay_surplus"],
                "adjust": ["revenue_deficits", "negative_prior", "receivables", "adjustment_change", "other_adjustments"]}
# Schedule A Part 4's purposes, bottom to top
CAPITAL = {"Schools": "schools", "Municipal Buildings": "buildings", "Highways (Chapter 90)": "highways",
           "Water": "water", "Sewer": "sewer", "Landfill": "landfill", "Other": "other"}
VOTE_KINDS = {"Override or underride": "override", "Debt exclusion": "debt", "Capital exclusion": "capital",
              "Stabilization fund": "stabilization"}
# Households: ACS five-year estimates (update_acs_households.py). Each fiscal year takes the
# release ending the calendar year it begins (FY2025: 2020-2024); FY2009 takes the first
# release, 2005-2009, and earlier years have none. A town missing from a release (ten whose
# Census codes changed in 2010 are missing from 2005-2009) takes the next one.
HH_CSV = Path("data/census_town_households.csv")
# Residents by sex, age, race and Hispanic origin, and disability, ACS five-year releases that don't overlap
DEMO_CSV = Path("data/census_town_demographics.csv")
DEMO_COLS = ["population", "male", "female", "age_under_18", "age_18_24", "age_25_44", "age_45_64", "age_65_up",
             "white", "black", "asian", "other_race", "hispanic", "disability_universe", "disability"]
# Town outlines for the map, with land areas (update_town_boundaries.py)
BOUNDARIES_JSON = Path("data/town_boundaries.json")
# Community types, by the Commonwealth's own lines where it draws them. The Inner Core is Boston and
# its densest neighbors: the Metropolitan Area Planning Council's Metropolitan Core Communities and
# Streetcar Suburbs (its community types, as its DataCommon profiles give them). The Gateway Cities
# are the state's 26 mid-size cities; the four in the Inner Core count there. A district is rural at
# 35 or fewer students per square mile (foundation enrollment over its towns' land area), DESE's test
# for rural school aid, in its three tiers; a town is rural under 500 residents per square mile, the
# Rural Policy Advisory Commission's line. The state has no name for the rest; here, suburban and other.
INNER_CORE = ["Boston", "Cambridge", "Chelsea", "Everett", "Malden", "Revere", "Somerville",
              "Arlington", "Belmont", "Brookline", "Medford", "Melrose", "Newton", "Waltham", "Watertown", "Winthrop"]
RURAL_STUDENT_TIERS = (11, 21, 35)
RURAL_RESIDENTS = 500
PROPERTY_TYPES = {
    "single": ["Single Family 101"],
    "condo": ["Condominiums 102"],
    "multi": ["Two Family 104", "Three Family 105", "Apartment 111-125"],
    "other_res": ["Miscellaneous Residential 103,109", "Vacant/ Accessory Land 130-132,106"],
    "mixed": ["Multi-Use 012-043"],
    "commercial": ["Commercial 300-393"],
    "industrial": ["Industrial 400-442", "Industrial Power Plant 450-452"],
    "farm": ["CH 61A LAND Agriculture"],
    "forest": ["CH 61 LAND Forest", "CH 61B LAND Recreation", "Open Space 200-231"],
    "personal": ["Individuals/ Partnerships/ Associations/ Trusts/ LLC 501", "Corporations 502", "Manufacturing 503",
                 "Public Utilities 504", "Centrally Valued Telephone 505", "Centrally Valued Pipelines 506",
                 "Wireless Telephone 508", "Electric Generating Plant 550-552"],
}
LA4_EXEMPT = "Exempt Parcel Count & Value"
# FY2025 Chapter 70 (DESE white paper, Appendix A): the uniform percentages applied to each
# town's 2022 EQV and 2021 income, the statewide target local share, and the cap
C70_TARGET = {"fy": 2025, "eqv_year": 2022, "income_year": 2021, "property_pct": 0.3906, "income_pct": 1.4312,
              "target_share": 59, "cap": 82.5}

# Population groups (the site's name; "peer groups" in the code): the estimate behind the
# latest fiscal year (July 2024 for FY2025), as district cohorts use latest-year enrollment.
# Round bounds. Every city and town is in one (Tim, #217: Springfield had none); the top
# band splits at 100,000 so the largest cities, which stand apart from the statewide and
# regional averages, have a group of their own rather than outweighing the 50,000s, as the
# very large districts have theirs. FY2025: 26, 42, 98, 103, 55, 18 and 9 towns.
# (id suffix, lowest population, first population above the group, label)
TOWN_PEERS = [
    ("under-1000", 0, 1000, "under 1,000"),
    ("1000-2499", 1000, 2500, "1,000–2,499"),
    ("2500-9999", 2500, 10000, "2,500–9,999"),
    ("10000-24999", 10000, 25000, "10,000–24,999"),
    ("25000-49999", 25000, 50000, "25,000–49,999"),
    ("50000-99999", 50000, 100000, "50,000–99,999"),
    ("100000-plus", 100000, None, "100,000 or more"),
]


# Labels for the revenue sources, as the explorer's legend names them (CSV headers)
REVENUE_LABELS = {
    "towns": "Payments from towns", "ch70": "Chapter 70 state aid", "retirement": "Teacher retirement paid by the state",
    "transport": "Transportation aid", "other_state": "Other state aid", "federal": "Federal",
    "districts": "From other school districts", "other_local": "Other local", "building": "School building aid",
}

# Ultimate sources, for the CSVs' source_url columns and cap.json, in one table (#237). They were 17 constants and two
# strings inline. The fetch scripts record some sources in data/*_sources.json, and the export reads those there:
# special education, the NSS budget files and DLS's fiscal reports by their URLs, and the Chapter 70 formula
# workbooks (dese_c70_townwide_sources.json, whose URLs are these exactly). The Circuit Breaker and enrollment files
# record their own dataset pages, spelled unlike the short links below the CSVs carry, and the others record no URL:
# moving those into the files would mean running the fetch scripts again, so they stay here.
SOURCE_URLS = {
    "dese_expenditures": "https://educationtocareer.data.mass.gov/d/er3w-dyti",
    # FY2005-FY2008: DESE's workbooks, from a public records request, as received and read
    "dese_archive": "https://github.com/timshores/schools/blob/main/ppe_report/data/dese_archive/README.md",
    "c70_profile": "https://www.doe.mass.edu/finance/chapter70/profile.xlsx",
    "c70_regional": "https://www.doe.mass.edu/finance/chapter70/fy{fy}/chapter-{fy}-reg.xlsx",
    "c70_whitepaper": "https://www.doe.mass.edu/finance/chapter70/fy{fy}/chapter-{fy}-whitepaper.docx",
    "c70_key_factors": "https://www.doe.mass.edu/finance/chapter70/keyfactors.xlsx",
    "census_f33": "https://www2.census.gov/programs-surveys/school-finances/tables/{fy}/secondary-education-finance/elsec{yy:02d}.{ext}",
    "circuit_breaker": "https://educationtocareer.data.mass.gov/d/ab34-d3ma",
    "students": "https://educationtocareer.data.mass.gov/d/t8td-gens",
    "dls_schedule_a": "https://dls-gw.dor.state.ma.us/reports/rdPage.aspx?rdReport=ScheduleA.GenFund_MAIN",
    "dls_la4": "https://dls-gw.dor.state.ma.us/reports/rdPage.aspx?rdReport=PropertyTaxInformation.LA4.Parcel_counts_vals",
    "dls_eqv": "https://dls-gw.dor.state.ma.us/reports/rdPage.aspx?rdReport=PropertyTaxInformation.EQV.EQV",
    "dls_levy": "https://dls-gw.dor.state.ma.us/reports/rdPage.aspx?rdReport=Prop2.5.ExcessLevyCapandOverride_10_pres",
    "dls_sf_bill": "https://dls-gw.dor.state.ma.us/reports/rdPage.aspx?rdReport=AverageSingleTaxBill.SingleFamTaxBill_wRange",
    "census_population": "https://www2.census.gov/programs-surveys/popest/datasets/",  # + each estimate's file
    "census_acs": "https://www2.census.gov/programs-surveys/acs/summary_file/",
    "bls_cpi": "https://data.bls.gov/timeseries/CUUR0100SA0",
}
# The last fiscal year DESE's archive gives (districts-spending-per-pupil.csv's source); the Hub's from the next
FUNC_LAST = 2008
C70_TOWNWIDE_CSV = Path("data/dese_c70_townwide.csv")
C70_TOWNWIDE_SOURCES = Path("data/dese_c70_townwide_sources.json")
C70_HISTORY_CSV = Path("data/dese_c70_town_history.csv")
# The Chapter 70 profile workbook, for the FY2023 figures update_dese_data.py restored (districts-chapter70.csv)
C70_XLSX = Path("data/dese_c70_profile.xlsx")
DESE_SOURCES = Path("data/dese_sources.json")
CPI_CSV = Path("data/inflation_indices.csv")
FISCAL_CSVS = [FREE_CASH_CSV, STAB_CSV, DEBT_CSV, AUTH_DEBT_CSV, RATINGS_CSV, VOTES_CSV, CAPITAL_CSV, LABOR_CSV,
               PROOF_CSV, LEVY_CSV]
# Every input: load_data's (the workbook and the refreshed DESE files, EXP_CSV and C70_CSV), explorer_history's, and
# the rest, which read_sources reads. All are committed; check_inputs stops the export without any.
INPUTS = [EXCEL_FILE, EXP_CSV, C70_CSV, C70_XLSX, DESE_SOURCES, FUNC_CSV, DAY_CSV, NCES_CSV, CPI_CSV, CENSUS_CSV,
          SPED_CSV, CB_CSV, SPED_SOURCES, BUDGET_CSV, BUDGET_SOURCES, ENROLL_CSV, ENROLL_SOURCES, DLS_CSV, POP_CSV,
          MEMBERS_CSV, C70_TOWNWIDE_CSV, C70_TOWNWIDE_SOURCES, C70_HISTORY_CSV, LA4_CSV, EQV_CSV, SF_BILL_CSV, *FISCAL_CSVS, FISCAL_SOURCES,
          HH_CSV, DEMO_CSV, BOUNDARIES_JSON]


def revenue_per_pupil(census, codes):
    """Revenue by source per in-district FTE pupil, pooled over the districts' DESE codes, and the
    in-district FTE of their average district, which turns the pooled figures back into an average
    district's dollars (one district's own FTE when there is only one)."""
    rows = census[census["DIST_CODE"].isin(codes) & (census["FTE"] > 0)]
    by_year = rows.groupby("FY").sum(numeric_only=True)
    out = {}
    for key, items in REVENUE.items():
        dollars = sum(by_year[i.lstrip("-")] * (-1 if i.startswith("-") else 1) for i in items)
        out[key] = series(dollars * 1000 / by_year["FTE"], 0)
    return out, series(by_year["FTE"] / rows.groupby("FY").size(), 1)


def revenue_fields(census, codes, regional):
    """The entity's revenue fields: per-pupil series when every member is regional. A town-run district has none
    (its Census totals, once written as rev_check, had no reader; #237)."""
    if not regional:
        return {}
    rev, fte = revenue_per_pupil(census, codes)
    return {"rev": rev, "rev_fte": fte}


def sped_pupils(hub):
    """Each district's FTE pupils in district and out, by year: everyone its special education spending serves,
    as DESE counts them for total spending per pupil."""
    rows = hub.df[(hub.cat == "student enrollment") & hub.subcat.isin([IN_DISTRICT_FTE_KEY, "out-of-district fte pupils"])]
    out = rows.groupby(["DIST_CODE", "YEAR"])["IND_VALUE"].sum().reset_index()
    return out.rename(columns={"YEAR": "FY", "IND_VALUE": "pupils"}).astype({"DIST_CODE": int, "FY": int})


def sped_fields(sped, cb, pupils, codes):
    """Special education spending by category, and what the state paid back through the Circuit Breaker for
    the same year's costs, per FTE pupil (in district and out), pooled over the districts' codes; special
    education's share of their spending; and the pupils and students claimed of their average district (one
    district's own)."""
    s = sped[sped["DIST_CODE"].isin(codes)].merge(pupils, on=["DIST_CODE", "FY"])
    s = s[s["pupils"] > 0]
    if s.empty:
        return {}
    by = s.groupby("FY")
    sums = by.sum(numeric_only=True)
    out = {"sped": {k: series(sums[k] / sums["pupils"], 0) for k in SPED},
           "sped_share": series(sums["sped_total"] / sums["total_exp"] * 100, 1),
           "sped_fte": series(sums["pupils"] / by.size(), 1)}
    # By the year of the costs, as the columns are: the claim the next year's row carries and what it paid,
    # and the year's own extraordinary relief. Every member in the year counts, one with no claim as repaid
    # nothing, over the years the dataset's claims cover.
    mine = cb[cb["DIST_CODE"].isin(codes)]
    claim = mine.assign(FY=mine["FY"] - 1, repaid=mine[CB_NEXT_YEAR].fillna(0).sum(axis=1))
    claim = claim[["DIST_CODE", "FY", "repaid", "students", "tot_elig_expenses", "tot_net_claim"]]
    p = pupils[pupils["DIST_CODE"].isin(codes) & (pupils["pupils"] > 0) &
               pupils["FY"].between(cb["FY"].min() - 1, cb["FY"].max() - 1)]
    c = p.merge(claim, on=["DIST_CODE", "FY"], how="left").merge(
        mine[["DIST_CODE", "FY", "extra_relief_payment"]], on=["DIST_CODE", "FY"], how="left")
    c["paid"] = c["repaid"].fillna(0) + c["extra_relief_payment"].fillna(0)
    cs = c.groupby("FY")
    csum = cs.sum(numeric_only=True)
    out["cb"] = {"paid": series(csum["paid"] / csum["pupils"], 0),
                 "claimed": series(csum["tot_elig_expenses"] / csum["pupils"], 0),
                 "above": series(csum["tot_net_claim"] / csum["pupils"], 0),
                 "students": series(csum["students"] / cs.size(), 1),
                 "fte": series(csum["pupils"] / cs.size(), 1)}
    return out


def budget_fields(budgets, c70, codes):
    """Budgeted net school spending per foundation pupil, pooled over the districts' codes that year with a budget
    and pupils; actual spending against the budget, in percent, pooled over those with both, as DESE paired them;
    and the newest budget, a year past the axis, against the year before's actual and budget."""
    leas = {c // 10000 for c in codes}
    b = budgets[budgets["lea"].isin(leas)]
    fe = c70[c70["Org4Code"].isin(leas)][["Org4Code", "YEAR", "distfoundenro"]].rename(
        columns={"Org4Code": "lea", "YEAR": "fy", "distfoundenro": "fe"})
    m = b.merge(fe, on=["lea", "fy"], how="left")
    have = m.dropna(subset=["budgeted_nss"])
    have = have[have["fe"] > 0]
    if have.empty:
        return {}
    per = have.groupby("fy")[["budgeted_nss", "fe"]].sum()
    both = m.dropna(subset=["budgeted_nss", "actual_nss"]).groupby("fy")[["budgeted_nss", "actual_nss"]].sum()
    out = {"nss_budget": series(per["budgeted_nss"] / per["fe"], 0),
           "nss_gap": series((both["actual_nss"] / both["budgeted_nss"] - 1) * 100, 1)}
    nxt = LATEST_YEAR + 1
    now = m[m["fy"] == nxt].set_index("lea")["budgeted_nss"].dropna()
    if not now.empty:
        act = m[m["fy"] == LATEST_YEAR].set_index("lea")["actual_nss"].dropna()
        bud = m[m["fy"] == LATEST_YEAR].set_index("lea")["budgeted_nss"].dropna()
        req = m[m["fy"] == nxt].set_index("lea")["budget_required_nss"].dropna()
        ka, kb, kr = now.index.intersection(act.index), now.index.intersection(bud.index), now.index.intersection(req.index)
        out["nss_budget_next"] = {
            "fy": nxt,
            "vs_actual": round(float(now[ka].sum() / act[ka].sum() - 1) * 100, 1) if len(ka) else None,
            "vs_budget": round(float(now[kb].sum() / bud[kb].sum() - 1) * 100, 1) if len(kb) else None,
            "vs_required": round(float(now[kr].sum() / req[kr].sum() - 1) * 100, 1) if len(kr) else None,
            # One district's own dollars; an average's would be a sum no one budgeted
            "budget": round(float(now.sum())) if len(leas) == 1 else None,
        }
    return out


def student_fields(enr, codes):
    """Students by race and ethnicity and in selected groups, in percent of all students, pooled over the districts'
    codes (each measure over the districts that report it that year); the students of their average district (one
    district's own); and the newest year's figures, a year past the axis."""
    rows = enr[enr["DIST_CODE"].isin(codes) & (enr["total_cnt"] > 0)]
    if rows.empty:
        return {}
    def pooled(part, has):
        r = rows[has]
        by = r.assign(part=part[has]).groupby("FY")[["part", "total_cnt"]].sum()
        return by["part"] / by["total_cnt"] * 100
    race = {k: pooled(sum(rows[c].fillna(0) for c in cs) * rows["total_cnt"], rows[cs[0]].notna())
            for k, cs in STUDENT_RACE.items()}
    groups = {k: pooled(rows[f"{c}_cnt"], rows[f"{c}_cnt"].notna()) for k, c in STUDENT_GROUPS.items()}
    by = rows.groupby("FY")
    total = by["total_cnt"].sum() / by.size()
    out = {"students": {"total": series(total, 1), "race": {k: series(v, 1) for k, v in race.items()},
                        "groups": {k: series(v, 1) for k, v in groups.items()}}}
    nxt = LATEST_YEAR + 1
    if nxt in total.index:
        out["students_next"] = {"fy": nxt, "total": round(float(total[nxt]), 1),
                                **{k: round(float(v[nxt]), 1) for k, v in groups.items() if nxt in v.index and pd.notna(v[nxt])}}
    return out


# Region for the few towns with no school district of their own (Monroe, Tyringham,
# Gosnold...): their county's, where the county lies in one EOHHS region
COUNTY_REGION = {"Berkshire": "Western", "Franklin": "Western", "Hampden": "Western", "Hampshire": "Western",
                 "Worcester": "Central", "Barnstable": "Southeast", "Bristol": "Southeast", "Dukes": "Southeast",
                 "Nantucket": "Southeast", "Plymouth": "Southeast"}
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
        # The map's spending per resident reads spending's total; revenue's had no reader (#237)
        if kind == "spend":
            out["spend_total"] = series(by_year[total_col], 0)
    return out


def load_property(la4, eqv, bills, levy):
    """LA-4 value and parcels by (town key, FY) and property type, EQV by (town key, year), single-family tax
    bills and levy capacity by (town key, FY), from the files as read (read_sources)."""
    type_of = {c: t for t, cats in PROPERTY_TYPES.items() for c in cats}
    la4 = la4[la4["FY"].isin(YEARS)].assign(key=lambda d: d["Municipality"].map(norm))
    unknown = set(la4["Category"]) - set(type_of) - {LA4_EXEMPT}
    if unknown:
        raise ValueError(f"LA-4 categories without a property type: {sorted(unknown)}")
    la4 = la4.assign(type=la4["Category"].map(type_of))
    taxable = la4[la4["type"].notna()]
    value = taxable.pivot_table(index=["key", "FY"], columns="type", values="Value", aggfunc="sum")
    parcels = taxable[taxable["type"] != "personal"].pivot_table(index=["key", "FY"], columns="type",
                                                                values="Parcels", aggfunc="sum")
    eqv = eqv.assign(key=lambda d: d["Municipality"].map(norm))
    bills = bills.assign(key=lambda d: d["Municipality"].map(norm))
    bills = bills[bills["FY"].isin(YEARS)].set_index(["key", "FY"])
    levy = levy.assign(key=lambda d: d["Municipality"].map(norm))
    levy = levy[levy["FY"].isin(YEARS)].set_index(["key", "FY"])
    return {"value": value, "parcels": parcels, "eqv": eqv.set_index(["key", "Year"])["EQV"], "bills": bills, "levy": levy}


def property_series(prop, ks, population):
    """Property for one town, or pooled over a group of towns (ks): assessed value per resident
    by type, parcels by type (the towns' combined count), EQV per resident and in all, and the
    average single-family tax bill with its share of income per capita.

    EQV is set as of January 1 of even years; each sits at the fiscal year whose assessments
    share that valuation date (EQV of January 1, 2024 at FY2025), per resident of that year."""
    ks = set(ks)
    pop = lambda idx: pd.Series([population.get(i, 0) for i in idx], index=idx)  # noqa: E731
    v = prop["value"][prop["value"].index.get_level_values("key").isin(ks)]
    v_res = v.groupby(level="FY").sum().div(pop(v.index).groupby(level="FY").sum(), axis=0)
    n = prop["parcels"][prop["parcels"].index.get_level_values("key").isin(ks)].groupby(level="FY").sum()
    e = prop["eqv"][prop["eqv"].index.get_level_values("key").isin(ks)]
    e = e.groupby(level="Year").sum()
    e.index = e.index + 1  # valuation year -> fiscal year
    e = e[e.index.isin(YEARS)]
    e_pop = pd.Series({fy: sum(population.get((k, fy), 0) for k in ks) for fy in e.index})
    b = prop["bills"][prop["bills"].index.get_level_values("key").isin(ks)].dropna(subset=["Single-Family Tax Bill"])
    b = b[b["Single-Family Parcels"] > 0]
    bill = (b["Single-Family Tax Bill"] * b["Single-Family Parcels"]).groupby(level="FY").sum() / \
        b["Single-Family Parcels"].groupby(level="FY").sum()
    # DOR's income per resident begins at FY2007; a year without it has none, rather than an income of 0
    bi = b[b["DOR Income Per Capita"] > 0]
    bi_pop = pop(bi.index)
    income = (bi["DOR Income Per Capita"] * bi_pop).groupby(level="FY").sum() / bi_pop.groupby(level="FY").sum()
    # The tax rate on single-family homes, per $1,000: the bills over the homes' value
    rate = (b["Single-Family Tax Bill"] * b["Single-Family Parcels"]).groupby(level="FY").sum() / \
        b["Single-Family Values"].groupby(level="FY").sum() * 1000
    lv = prop["levy"][prop["levy"].index.get_level_values("key").isin(ks)]
    sums = lv[["Total Tax Levy", "Excess Levy Capacity", "Maximum Levy Limit", "Levy Ceiling"]].groupby(level="FY").sum()
    per = pop(lv.index).groupby(level="FY").sum()
    return {
        # Prop 2 1/2: the levy raised and the capacity left under the limit, per resident and in all
        "levy": {"raised": series(sums["Total Tax Levy"] / per, 0), "unused": series(sums["Excess Levy Capacity"] / per, 0)},
        "levy_total": series(sums["Total Tax Levy"], 0), "levy_max": series(sums["Maximum Levy Limit"], 0),
        "levy_ceiling": series(sums["Levy Ceiling"], 0),
        "sf_rate": series(rate, 2),
        "income_pc": series(income, 0),
        "prop": {t: series(v_res[t] if t in v_res.columns else None, 0) for t in PROPERTY_TYPES},
        "parcels": {t: series(n[t] if t in n.columns else None, 0) for t in PROPERTY_TYPES if t != "personal"},
        "value_total": series(v.groupby(level="FY").sum().sum(axis=1), 0),
        "eqv": series(e / e_pop, 0),
        "eqv_total": series(e, 0),
        "sf_bill": series(bill, 0),
        "sf_bill_income": series(bill / income, 4),
    }


def load_fiscal(raw):
    """Reserves, budgets, debt, capital projects, levy capacity, ratings, votes, labor force and the free cash proof
    by town key, from the files as read ({path: frame}, read_sources)."""
    read = lambda path, fy: raw[path].assign(key=lambda d: d["Municipality"].map(norm)).rename(columns={fy: "FY"})  # noqa: E731
    fc = read(FREE_CASH_CSV, "Fiscal Year")
    # Free cash certified as of July 1 is the year before's June 30 balance: FY2027's is FY2026's year-end
    fc = fc.assign(FY=fc["FY"] - 1).set_index(["key", "FY"])
    stab = read(STAB_CSV, "Schedule A Fiscal Year").set_index(["key", "FY"])
    debt = read(DEBT_CSV, "Fiscal Year").set_index(["key", "FY"])
    auth = read(AUTH_DEBT_CSV, "Fiscal Year").set_index(["key", "FY"])
    cap = read(CAPITAL_CSV, "Fiscal Year")
    cap = cap[cap["Type"] == "Expenditures"].set_index(["key", "FY"])
    # The labor force is a calendar year's average; the year before a fiscal year's July 1 population
    # estimate begins it centers on that day, so calendar 2022 sits at FY2023, as the population does
    labor = read(LABOR_CSV, "Year")
    labor = labor.assign(FY=labor["FY"] + 1).set_index(["key", "FY"])
    levy = raw[LEVY_CSV].assign(key=lambda d: d["Municipality"].map(norm)).set_index(["key", "FY"])
    ratings = raw[RATINGS_CSV].assign(key=lambda d: d["Name"].map(norm))
    # The proof's year is the fiscal year whose June 30 balance sheet it reads, as the reserves' are
    proof = raw[PROOF_CSV].assign(key=lambda d: d["Municipality"].map(norm))
    for g, lines in PROOF_GROUPS.items():
        proof[g] = proof[lines].fillna(0).sum(axis=1)
    proof = proof.set_index(["key", "FY"])
    votes = raw[VOTES_CSV].assign(key=lambda d: d["Municipality"].map(norm))
    return {"fc": fc, "stab": stab, "debt": debt, "auth": auth, "cap": cap, "labor": labor, "levy": levy,
            "ratings": ratings, "votes": votes, "proof": proof}


def fiscal_series(fiscal, ks, population):
    """Reserves, the operating budget, debt and capital projects per resident, the labor force, and
    override capacity, for one town or pooled over a group (ks); each year counts the towns with a
    figure for it. Reserves are at each fiscal year's end: June 30's stabilization balances and the
    free cash certified from that balance sheet. latest: the newest year-end's dollars, which the
    explorer's axis stops short of (reserves to FY2026, levy capacity to FY2026)."""
    ks = set(ks)
    mine = lambda df: df[df.index.get_level_values("key").isin(ks)]  # noqa: E731

    def per_resident(df, cols):
        """{col: per-resident series} over YEARS, each year pooled over the towns with that column"""
        out = {}
        for c, key in cols.items():
            x = df[c].dropna()
            res = pd.Series([population.get(i, 0) for i in x.index], index=x.index)
            out[key] = series(x.groupby(level="FY").sum() / res.groupby(level="FY").sum(), 0)
        return out

    def latest(df, cols):
        """The newest year nine in ten of the group's reporting towns have every column for, in dollars,
        summed over those towns (a town's own newest year)"""
        x = df[list(cols)].dropna()
        if x.empty:
            return None
        n = x.groupby(level="FY").size()
        fy = int(n[n >= 0.9 * n.max()].index.max())
        row = x.xs(fy, level="FY").sum()
        return {"fy": fy, **{key: round(float(row[c])) for c, key in cols.items()}}

    fc, stab, debt, auth, cap = (mine(fiscal[k]) for k in ("fc", "stab", "debt", "auth", "cap"))
    res_cols = {"Stabilization Fund Balance": "stab", "Special Purpose Stabilization Fund Balance": "sp_stab"}
    both = stab[list(res_cols) + ["Operating Budget"]].join(fc[["Certified Free Cash as of 7/1"]], how="inner")
    out = {
        "reserves": {**per_resident(both, res_cols), **per_resident(both, {"Certified Free Cash as of 7/1": "free_cash"})},
        "budget": per_resident(stab, {"Operating Budget": "v"})["v"],
        "reserves_latest": latest(both, {**res_cols, "Certified Free Cash as of 7/1": "free_cash", "Operating Budget": "budget"}),
        "debt": {**per_resident(debt, {"Total Outstanding Debt (Schedule A Part 10)": "outstanding"}),
                 **per_resident(auth, {"Unissued Debt": "unissued"})},
        "capital": per_resident(cap, {c: k for c, k in CAPITAL.items()}),
    }
    d = debt[["GF Debt Service (Schedule A Part 2)", "Total Budget", "Total Outstanding Debt (Schedule A Part 10)",
              "Equalized Values (EQV)"]].dropna().groupby(level="FY").sum()
    out["debt_service"] = series(d["GF Debt Service (Schedule A Part 2)"] / d["Total Budget"] * 100, 2)
    out["debt_eqv"] = series(d["Total Outstanding Debt (Schedule A Part 10)"] / d["Equalized Values (EQV)"] * 100, 2)
    out["debt_latest"] = latest(debt, {"Total Outstanding Debt (Schedule A Part 10)": "outstanding",
                                      "GF Debt Service (Schedule A Part 2)": "service", "Total Budget": "budget"})
    lab = mine(fiscal["labor"])[["Labor Force", "Employed", "Unemployed"]].dropna().groupby(level="FY").sum()
    out["labor"] = {"force": series(lab["Labor Force"], 0), "employed": series(lab["Employed"], 0),
                    "rate": series(lab["Unemployed"] / lab["Labor Force"] * 100, 1)}
    lv = mine(fiscal["levy"])[["Override Capacity", "Levy Ceiling", "Levy Limit w/o Debt & Capital Exclusions"]].dropna()
    lsum = lv.groupby(level="FY").sum()
    out["override_pct"] = series(lsum["Override Capacity"] / lsum["Levy Ceiling"] * 100, 1)
    out["levy_ceiling_pr"] = per_resident(lv, {"Levy Ceiling": "v"})["v"]
    out["override_latest"] = latest(lv, {"Override Capacity": "capacity", "Levy Ceiling": "ceiling"})
    # What each year's free cash was made of, per resident, and the newest year's dollars
    pr = mine(fiscal["proof"])
    out["fc_proof"] = per_resident(pr, {g: g for g in PROOF_GROUPS})
    out["fc_identified"] = per_resident(pr, {"identified": "v"})["v"]
    out["fc_proof_latest"] = latest(pr, {**{g: g for g in PROOF_GROUPS}, "identified": "identified"})
    # Prop 2 1/2 votes since the explorer's first year: held and won, by kind
    v = fiscal["votes"][fiscal["votes"]["key"].isin(ks) & (fiscal["votes"]["Fiscal Year"] >= EXPLORER_START)]
    out["votes"] = {VOTE_KINDS[k]: [int((v["Vote"] == k).sum()), int(((v["Vote"] == k) & (v["Win / Loss"] == "WIN")).sum())]
                    for k in VOTE_KINDS}
    return out


def vote_kind(vote, vote_type):
    """A Prop 2 1/2 vote's kind, as the explorer names it: an override or underride by its type, one with no type
    an override; the others by VOTE_KINDS. town_votes and town_passed each had their own cleanup (#237), which
    differed only for a missing type: no vote in DLS's records lacks one."""
    if vote != "Override or underride":
        return VOTE_KINDS[vote]
    return (vote_type if isinstance(vote_type, str) and vote_type else "Override").lower()


def town_votes(fiscal, names):
    """votes.json: every town's Prop 2 1/2 votes, newest first, as DLS lists them: fiscal year, date,
    kind (override, underride, debt or capital exclusion, stabilization fund), department, purpose,
    amount, won, yes and no votes."""
    v = fiscal["votes"].copy()
    v["yes"] = v["Yes Votes"].fillna(v["Number Yes"])
    v["no"] = v["No Votes"].fillna(v["Number No"])
    v["kind"] = [vote_kind(k, t) for k, t in zip(v["Vote"], v["Vote Type"])]
    v["date"] = pd.to_datetime(v["Vote Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out = {}
    for k, g in v[v["key"].isin(names)].sort_values(["date", "Fiscal Year"], ascending=False).groupby("key", sort=False):
        out[slug(names[k])] = [[int(r["Fiscal Year"]), r["date"] if isinstance(r["date"], str) else None, r["kind"],
                                (r["Department"] or "").strip().capitalize() if isinstance(r["Department"], str) else None,
                                r["Description"].strip() if isinstance(r["Description"], str) else None,
                                None if pd.isna(r["Amount"]) else round(float(r["Amount"])),
                                r["Win / Loss"] == "WIN",
                                None if pd.isna(r["yes"]) else int(r["yes"]), None if pd.isna(r["no"]) else int(r["no"])]
                               for _, r in g.iterrows()]
    return out


def town_passed(fiscal, k):
    """A town's Prop 2 1/2 votes that passed, from the explorer's first year: [fiscal year, kind, amount or None,
    purpose], for the marks on its charts. Debt exclusions carry no amount in DLS's records."""
    v = fiscal["votes"]
    v = v[(v["key"] == k) & (v["Win / Loss"] == "WIN") & (v["Fiscal Year"] >= EXPLORER_START)]
    out = []
    for _, r in v.sort_values(["Fiscal Year", "Vote Date"]).iterrows():
        kind = vote_kind(r["Vote"], r["Vote Type"])
        dept = r["Department"].strip().capitalize() if isinstance(r["Department"], str) else ""
        desc = r["Description"].strip() if isinstance(r["Description"], str) else ""
        what = ": ".join(x for x in (dept, desc) if x)  # as the list under the levy chart gives it
        out.append([int(r["Fiscal Year"]), kind, None if pd.isna(r["Amount"]) else round(float(r["Amount"])), what[:90]])
    return {"passed": out} if out else {}


def town_ratings(fiscal, k):
    """A town's bond ratings by agency: [year, rating] each year DLS lists one."""
    r = fiscal["ratings"][fiscal["ratings"]["key"] == k]
    if r.empty:
        return {}
    years = [c for c in r.columns if str(c).isdigit()]
    out = {}
    for _, row in r.iterrows():
        got = [[int(y), row[y].strip()] for y in years if isinstance(row[y], str) and row[y].strip()]
        if got:
            out["moodys" if row["Rating Agency"].startswith("Mood") else "sp"] = got
    return {"ratings": out} if out else {}


# Commonwealth tab: every district in DESE's Chapter 70 profiles, with its member towns'
# property and income pooled, for scatterplots of any two measures
VALLEY_COUNTIES = {"Franklin", "Hampshire", "Hampden"}
RESIDENTIAL_TYPES = ["single", "condo", "multi", "other_res"]


def commonwealth_data(dists, full, src):
    """commonwealth.json: per district and year, its Chapter 70 figures and spending (DESE)
    and its towns' wealth (DLS), pooled over member towns. EQV is the latest as of each
    fiscal year's January 1 valuation date (EQV of January 1, 2022 for FY2023 and FY2024)."""
    # Every district but the state total (code 0000). Matching "COUNTY" in the name had also dropped four vocational
    # districts DESE names that way: Franklin County, Tri County, Bristol County and Norfolk County (#242)
    c70 = src.c70[src.c70["fy"].isin(YEARS) & (src.c70["Org4Code"] != STATE_TOTAL_ORG)]
    prop, population, county = src.prop, src.population, src.county
    names = src.dls[["Municipality"]].drop_duplicates().assign(
        key=lambda d: d["Municipality"].map(norm)).set_index("key")["Municipality"].to_dict()
    member_keys = src.members.groupby("Org4Code")["key"].apply(list).to_dict()
    by_org = {f"{d['org']:04d}": d for d in dists}
    entity = {e["id"]: e for e in full}
    # A district counts in a year only if it ran schools then, and DESE's district profiles
    # (spending, enrollment, staff) cover those years alone. A town whose pupils all go to regional
    # districts keeps a row in the Chapter 70 profiles, with a few pupils it pays tuition for and
    # leftover sums that divide into nonsense (Williamstown, FY2025: $318,114 of actual spending
    # against $14 required).
    exp = src.exp[["SY", "DIST_CODE"]].drop_duplicates()
    ran = {(f"{int(c) // 10000:04d}", int(fy)) for c, fy in zip(exp["DIST_CODE"], exp["SY"])}

    def r(v, digits):
        return None if v is None or pd.isna(v) else round(float(v), digits)

    def ratio(a, b):
        return None if a is None or b is None or pd.isna(a) or pd.isna(b) or b == 0 else a / b

    out = []
    for org, g in c70.groupby("Org4Code"):
        g = g.set_index("fy")
        if not (g["distfoundenro"].fillna(0) > 0).any() or not (g["actualNSS"].fillna(0) > 0).any():
            continue  # towns that send all their pupils elsewhere, districts that no longer operate
        d = by_org.get(org)
        dese_name = g["District"].iloc[-1].strip()
        own = norm(dese_name)
        ks = member_keys.get(org) or ([own] if own in names else [])
        counties = [county.get(k) for k in ks]
        e = entity.get(d["id"]) if d else None
        col = lambda c: [g[c].get(fy) if fy in g.index else None for fy in YEARS]  # noqa: E731
        fe, fb, rlc, aid, rqd = (col(c) for c in ("distfoundenro", "distfoundbudget", "distrlc", "c70aid", "rqdnss2"))
        act = [v if v is not None and not pd.isna(v) and v > 0 else None for v in col("actualNSS")]
        ppe = total_ppe(e, strict=True) if e else [None] * len(YEARS)
        fte = e["enroll"]["in_district"] if e else [None] * len(YEARS)
        w = property_series(prop, ks, population) if ks else {}
        pops = [sum(population.get((k, fy), 0) for k in ks) or None for fy in YEARS]
        eqv = pd.Series(w.get("eqv_total") or [None] * len(YEARS), dtype=float).ffill().tolist()
        values = [sum(w["prop"][t][i] or 0 for t in PROPERTY_TYPES) if w else None for i in range(len(YEARS))]
        homes = [sum(w["prop"][t][i] or 0 for t in RESIDENTIAL_TYPES) if w else None for i in range(len(YEARS))]
        v = {
            "fe": [r(x, 1) for x in fe],
            "fte": fte,
            "nss_pct_fb": [r(ratio(a, b), 4) for a, b in zip(act, fb)],
            "nss_pct_req": [r(ratio(a, b), 4) for a, b in zip(act, rqd)],
            "nss_pfe": [r(ratio(a, b), 0) for a, b in zip(act, fe)],
            "above_pfe": [r(ratio(a - b, f), 0) if a is not None and b is not None else None for a, b, f in zip(act, rqd, fe)],
            "ppe": [r(x, 0) for x in ppe],
            "aid_pfe": [r(ratio(a, b), 0) for a, b in zip(aid, fe)],
            "rlc_pfe": [r(ratio(a, b), 0) for a, b in zip(rlc, fe)],
            "aid_share": [r(ratio(a, b), 4) for a, b in zip(aid, fb)],
            "eqv_res": [r(ratio(a, b), 0) for a, b in zip(eqv, pops)],
            "av_res": [r(x, 0) for x in values],
            "income_pc": w.get("income_pc") or [None] * len(YEARS),
            "sf_bill": w.get("sf_bill") or [None] * len(YEARS),
            "bill_income": w.get("sf_bill_income") or [None] * len(YEARS),
            "res_share": [r(ratio(a, b), 4) for a, b in zip(homes, values)],
            # The district's own: its students' shares, special education's share, spending against its budget, and
            # payments per out-of-district pupil
            **({k: x for k, x in district_shares(e).items() if k != "high_needs"} if e else {}),
            "ood_pp": (e.get("ood") if e else None) or [None] * len(YEARS),
        }
        off = [(org, fy) not in ran for fy in YEARS]
        if all(off):
            continue
        v = {k: [None if o else x for x, o in zip(s, off)] for k, s in v.items()}
        out.append({
            "org": org, "name": d["name"] if d else dese_name.title(), "id": d["id"] if d else None,
            "type": d["type"] if d else ("vocational" if int(org) >= 800 else "traditional"),
            "region": d["region"] if d else None,
            "towns": [names.get(k, k) for k in ks],
            "valley": bool(ks) and sum(c in VALLEY_COUNTIES for c in counties) * 2 > len(ks),
            "ct": (d.get("community") or {}).get("type") if d else None,
            "fifth": (d.get("income") or {}).get("fifth") if d else None,
            "v": v,
        })
    return {"meta": {"years": YEARS, "latest": LATEST_YEAR, "built": dt.date.today().isoformat(),
                     "regions": REGIONS, "c70_profile_date": src.dese["c70_profile_date"]},
            "districts": out}


def nocap_factor(targets, yields, foundations):
    """The factor that takes the 82.5% cap off a year's Chapter 70 targets (cap_data): the capped targets' total
    over the uncapped combined effort yields' total, so the targets still add up to 59% of all foundation
    budgets; a town's no-cap target is its yield times it, rounded. A town with no foundation budget (Gosnold, with
    no pupils of its own) has no target under any cap, so it takes no share of the total (#242). cap_data, the map's
    yearly figures (map_series) and towns-chapter70-targets.csv (targets_csv) figure it this way (#237), as the cap
    tab does in the browser (cap.ts solve)."""
    has = [f is not None and f > 0 for f in foundations]
    return sum(t for t, h in zip(targets, has) if h) / sum(c for c, h in zip(yields, has) if h)


def at_cap(targets, yields, foundations):
    """Whether the cap held each town's target below its combined effort yield: DESE's target is the lesser of the
    yield and the cap's share of the foundation budget, so a capped town's target is below its yield. The cap
    tab's every count of towns at the cap uses this test (#242); it agrees town for town with DESE's maximum local
    contribution column in FY2026 and FY2027."""
    return [f is not None and f > 0 and t is not None and c is not None and t < c - 1
            for t, c, f in zip(targets, yields, foundations)]


def cap_data(all_towns, src):
    """cap.json: each year's town targets from DESE's formula spreadsheet, with and without the
    82.5% cap. Without it, DESE's rule that targets add up to 59% of all foundation budgets
    stays, so both statewide percentages fall by one factor: the capped targets' total over
    the uncapped combined effort yields' total. A town's target is then its yield times that
    factor. evidence/cap_fy2026.py checks it against DESE's spreadsheet and Michael Naughton's FY2026 workbook:
    every town within $1, but Gosnold, which has no foundation budget and no target here (he gives it one)."""
    tw, population, households = src.tw, src.population, src.households
    last_pop = int(src.pop["FY"].max())
    # Taxable real estate parcels: the latest LA-4 year every town has filed
    la4 = src.la4.assign(key=lambda d: d["Municipality"].map(norm))
    la4 = la4[~la4["Category"].isin([LA4_EXEMPT] + PROPERTY_TYPES["personal"])]
    filed = la4[la4["Parcels"] > 0].groupby("FY")["key"].nunique()
    parcel_fy = int(filed[filed >= 351].index.max())
    parcels = la4[la4["FY"] == parcel_fy].groupby("key")["Parcels"].sum().to_dict()
    by_key = {norm(x["name"]): x for x in all_towns}
    years = {}
    for fy, g in tw.groupby("fy"):
        g = g.assign(key=g["town"].map(norm))
        missing = sorted(set(g["key"]) - set(by_key))
        if missing:
            raise ValueError(f"FY{fy} formula towns not in the town list: {missing}")
        factor = nocap_factor(g["target"], g["cey"], g["foundation"])
        nocap = (g["cey"] * factor).round().where(g["foundation"] > 0, 0)
        income_year, eqv_year = int(g["income_year"].iloc[0]), int(g["eqv_year"].iloc[0])
        capped = pd.Series(at_cap(g["target"], g["cey"], g["foundation"]), index=g.index)
        change = nocap - g["target"]
        towns = []
        for (_, r), nc, cap in zip(g.iterrows(), nocap, capped):
            x = by_key[r["key"]]
            towns.append({
                "lea": int(r["lea"]), "name": x["name"], "id": x["id"], "county": x["county"], "region": x["region"],
                # Residents: the latest estimate for per-resident dollars; income per resident uses the
                # estimate for July of the income year
                "pop": dollars(population.get((r["key"], min(fy, last_pop)))),
                "pop_income": dollars(population.get((r["key"], income_year + 1))),
                "hh": dollars(households.get((r["key"], min(fy, last_pop)))),
                "parcels": dollars(parcels.get(r["key"])),
                "eqv": dollars(r["eqv"]), "income": dollars(r["income"]), "cey": dollars(r["cey"]),
                "foundation": dollars(r["foundation"]), "max": dollars(r["max_local"]), "target": dollars(r["target"]),
                "nocap": dollars(nc), "required": dollars(r["required"]), "prior": dollars(r["rlc_prior"]),
                # How the required contribution moved: growth factor, extra increment, and the start it gave
                "mrgf": None if pd.isna(r["mrgf"]) else round(float(r["mrgf"]), 4),
                "increment": None if pd.isna(r["pct_increment"]) else round(float(r["pct_increment"]), 4),
                "preliminary": dollars(r["preliminary"]),
                "capped": bool(cap), "floor": bool(r["floor_increment"] > 0),
            })
        years[str(fy)] = {
            "fy": int(fy), "eqv_year": eqv_year, "income_year": income_year,
            "property_pct": float(g["property_pct"].iloc[0]), "income_pct": float(g["income_pct"].iloc[0]),
            "max_pct": float(g["max_pct"].iloc[0]), "factor": float(factor),
            "totals": {"cey": dollars(g["cey"].sum()), "target": dollars(g["target"].sum()),
                       "foundation": dollars(g["foundation"].sum()), "required": dollars(g["required"].sum()),
                       "capped": int(capped.sum()), "up": int((change > 0).sum()), "up_sum": dollars(change[change > 0].sum()),
                       "down": int((change < 0).sum()), "down_sum": dollars(change[change < 0].sum())},
            "source": src.c70_townwide_sources["urls"][str(fy)], "whitepaper": SOURCE_URLS["c70_whitepaper"].format(fy=fy),
            "pop_year": min(int(fy), last_pop) - 1, "parcel_fy": parcel_fy,
            "towns": towns,
        }
    # How far districts spend above the required minimum (actual net school spending over required),
    # the latest year with actual spending reported
    c70 = src.c70
    c70 = c70[(c70["fy"] == LATEST_YEAR) & (c70["Org4Code"] != STATE_TOTAL_ORG)
              & (c70["actualNSS"] > 0) & (c70["rqdnss2"] > 0)]
    ratio = c70["actualNSS"] / c70["rqdnss2"]
    spending = {"fy": LATEST_YEAR, "districts": int(len(c70)), "above": int((ratio > 1.0005).sum()),
                "median": round(float(ratio.median()), 3), "half_again": int((ratio >= 1.5).sum()),
                "double": int((ratio >= 2).sum()), "required": dollars(c70["rqdnss2"].sum()),
                "over": dollars((c70["actualNSS"] - c70["rqdnss2"]).clip(lower=0).sum())}
    # Each town's yield, foundation budget, target, and required contribution, FY2007 on (DESE's
    # trends workbook), with the latest formula year added, so any cap can be redone year by year
    h = src.c70_history.assign(key=lambda d: d["town"].map(norm))
    latest_fy = int(tw["fy"].max())
    if latest_fy not in set(h["fy"]):
        t = tw[tw["fy"] == latest_fy].assign(key=lambda d: d["town"].map(norm))
        h = pd.concat([h, pd.DataFrame({"fy": latest_fy, "key": t["key"], "cey": t["cey"], "foundation": t["foundation"],
                                        "target": t["target"], "required": t["required"]})], ignore_index=True)
    missing = sorted(set(h["key"]) - set(by_key))
    if missing:
        raise ValueError(f"History towns not in the town list: {missing}")
    hy = sorted(int(y) for y in h["fy"].unique())
    hist_towns = {}
    for k, g in h.groupby("key"):
        g = g.set_index("fy")
        hist_towns[by_key[k]["id"]] = {c: [dollars(g[c].get(y)) if y in g.index else None for y in hy]
                                       for c in ("cey", "foundation", "target", "required")}
    # Each year's count of towns at the cap, by the cap tab's one test (at_cap), for About the data's FY2025 count
    capped_by_year = [sum(at_cap(*([t[c][i] for t in hist_towns.values()] for c in ("target", "cey", "foundation"))))
                      for i in range(len(hy))]
    history = {"years": hy, "source": SOURCE_URLS["c70_key_factors"], "capped": capped_by_year,
               "towns": hist_towns}
    return {"meta": {"built": dt.date.today().isoformat(), "latest": max(int(y) for y in years), "spending": spending},
            "years": years, "history": history}


def load_demographics(pop, demo):
    """(town key, ACS release's last year) -> its counts, from DEMO_CSV as read. A town the Census Bureau gave a
    new code keeps both in the population file, so both find it."""
    key_of = pop.drop_duplicates("COUSUB_FIPS").set_index("COUSUB_FIPS")["key"]
    d = demo.assign(key=lambda x: x["COUSUB_FIPS"].map(key_of))
    return {(k, int(v)): row for k, v, row in zip(d["key"], d["Vintage"], d[DEMO_COLS].to_dict("records")) if isinstance(k, str)}


def demo_series(demo, ks):
    """The towns' combined counts, each release at the fiscal year after its five years end (2020-2024 at FY2025),
    as the households are; a release counts the towns it has."""
    if not demo:
        return {}
    vintages = sorted({v for _, v in demo})
    out = {c: [None] * len(YEARS) for c in DEMO_COLS}
    for v in vintages:
        if v + 1 not in YEARS:
            continue
        rows = [demo[(k, v)] for k in ks if (k, v) in demo]
        for c in DEMO_COLS:
            vals = [r[c] for r in rows if r[c] == r[c] and r[c] is not None]
            if rows and len(vals) == len(rows):
                out[c][YEARS.index(v + 1)] = round(sum(vals))
    return {"demo": out} if any(x is not None for x in out["population"]) else {}


def load_households(pop, hh):
    """(town key, fiscal year) -> households, from HH_CSV as read."""
    key_of = pop.drop_duplicates("COUSUB_FIPS").set_index("COUSUB_FIPS")["key"]
    hh = hh.assign(key=hh["COUSUB_FIPS"].map(key_of))
    by = {(k, v): n for k, v, n in zip(hh["key"], hh["Vintage"], hh["Households"]) if isinstance(k, str)}
    first, last = int(hh["Vintage"].min()), int(hh["Vintage"].max())
    out = {}
    for k in set(hh["key"].dropna()):
        for fy in YEARS:
            if fy < first:
                continue
            v = min(max(fy - 1, first), last)
            n = by.get((k, v)) or by.get((k, v + 1))
            if n:
                out[(k, fy)] = n
    return out


def land_areas(geo):
    """Each town's land area in square miles, by town key, from the outlines file (BOUNDARIES_JSON)"""
    return {norm(g["town"]): g["sqmi"] for g in geo["towns"].values()}


def district_members(dists, towns_of):
    """Each traditional district's towns, by town key: a regional district's members (towns_of, from DESE's
    regional members file); a town's own district is the town."""
    return {d["id"]: towns_of.get(d["code"] // 10000, []) if is_regional(d) else [norm(d["name"])]
            for d in dists if d["type"] == "traditional"}


FIFTH_NAMES = {1: "lowest", 2: "second", 3: "middle", 4: "fourth", 5: "highest"}


def town_incomes(tw, population):
    """Each town's income per resident, as the cap tab counts it: the Chapter 70 formula's income for
    the latest fiscal year (tw: C70_TOWNWIDE_CSV) over the population estimate for July of that income year.
    {town key: (income, population, year)}"""
    tw = tw[tw["fy"] == tw["fy"].max()]
    out = {}
    for town, inc, yr in zip(tw["town"], tw["income"], tw["income_year"]):
        k, p = norm(town), population.get((norm(town), int(yr) + 1))
        if p and inc == inc:
            out[k] = (float(inc), float(p), int(yr))
    return out


def income_fifths(per_resident):
    """{key: {"fifth": 1 (lowest) to 5 (highest), "income_pc", "year"}}, from {key: (income, population, year)}:
    ranked by income per resident, in five groups as near equal in size as the count allows"""
    ranked = sorted(per_resident.items(), key=lambda kv: kv[1][0] / kv[1][1])
    n = len(ranked)
    return {k: {"fifth": min(4, i * 5 // n) + 1, "income_pc": round(v[0] / v[1]), "year": v[2]}
            for i, (k, v) in enumerate(ranked)}


def district_types(dists, c70, area, towns_of):
    """Each traditional district's community type: {id: {"type", "density" (students per square
    mile), "tier" (DESE's rural tier, else None), "fy" (of the enrollment)}}. A regional district's
    towns are its members (towns_of, from DESE's regional members file); a town's own district is the town."""
    fe = c70.dropna(subset=["distfoundenro"]).sort_values("fy").groupby("Org4Code").last()
    core = {norm(t) for t in INNER_CORE}
    out, missing = {}, []
    for d in dists:
        if d["type"] != "traditional":
            continue
        org = d["code"] // 10000
        ks = towns_of.get(org, []) if is_regional(d) else [norm(d["name"])]
        if not ks or any(k not in area for k in ks) or org not in fe.index:
            missing.append(d["name"])
            continue
        density = float(fe.loc[org, "distfoundenro"]) / sum(area[k] for k in ks)
        tier = next((i + 1 for i, lim in enumerate(RURAL_STUDENT_TIERS) if density <= lim), None)
        local = not is_regional(d)
        kind = ("core" if local and norm(d["name"]) in core else "gateway" if d["gateway"] else "rural" if tier
                else "suburban")
        out[d["id"]] = {"type": kind, "density": round(density, 1), "tier": tier, "fy": int(fe.loc[org, "fy"])}
    if missing:
        print(f"  [WARN] no community type for {len(missing)} districts: {missing[:8]}")
    return out


def edu_share(t):
    """Education's share of a town's general fund spending each year: its spending per resident on
    education over its spending per resident on every function, as the explorer's tile figures it"""
    spend = t.get("spend") or {}
    out = []
    for y in range(len(YEARS)):
        parts = [s[y] for s in spend.values()]
        edu = spend.get("education", [None] * len(YEARS))[y]
        out.append(None if not spend or edu is None or any(p is None for p in parts) or not sum(parts) else edu / sum(parts))
    return out


def fiscal_shares(t):
    """A town's fiscal measures for the map, each year as a share: override capacity of the levy ceiling, reserves of
    the operating budget, general fund debt service of the budget, and the unemployment rate (2022 at FY2023)."""
    pct = lambda vals: [None if v is None else v / 100 for v in vals] if vals else None  # noqa: E731
    res = t.get("reserves")
    reserves = ([None if t["budget"][i] in (None, 0) or any(res[k][i] is None for k in res)
                 else sum(res[k][i] for k in res) / t["budget"][i] for i in range(len(YEARS))] if res and t.get("budget") else None)
    return {"override": pct(t.get("override_pct")), "reserves": reserves, "debt_service": pct(t.get("debt_service")),
            "unemp": pct((t.get("labor") or {}).get("rate"))}


def total_ppe(e, strict=False):
    """A district's (or district average's) in-district spending per pupil each year, every category together:
    None in a year no category has a figure for, or with strict, a year any lacks one (the Commonwealth tab's
    scatterplots); else a missing category counts as nothing (the map's). One function for the four places that
    summed them (#237); every category has had a figure in every year that has any, so the two agree today."""
    cats = list(e["ppe"].values())

    def total(vals):
        if strict:
            return None if any(v is None for v in vals) else sum(vals)
        return None if all(v is None for v in vals) else sum(v or 0 for v in vals)
    return [total([c[y] for c in cats]) for y in range(len(YEARS))]


def required_pfe(nss):
    """Required net school spending per foundation pupil each year: Chapter 70 aid and the required local
    contribution above it (pack's "nss")"""
    return [None if a is None or r is None else a + r
            for a, r in zip(nss.get("ch70_aid") or [], nss.get("required_local") or [])]


def levy_room(t):
    """A town's (or town average's) room under its levy limit each year, as a share of the limit"""
    return [None if m is None or x is None or not m else (m - x) / m
            for x, m in zip(t.get("levy_total") or [], t.get("levy_max") or [])]


def frac(vals, start=None):
    """Percents as fractions, for the map's and the Commonwealth tab's shares; years before start left off"""
    return [None if v is None or (start and y < start) else round(v / 100, 4) for v, y in zip(vals or [None] * len(YEARS), YEARS)]


def district_shares(e):
    """A district's students and budget measures, year by year, as fractions: its students with disabilities,
    English learners, low income (the measure DESE has used since FY2022), high needs and students of color; special
    education's share of its spending; and its spending against its budget"""
    st = e.get("students") or {}
    g, race = st.get("groups") or {}, st.get("race") or {}
    return {"swd": frac(g.get("swd")), "el": frac(g.get("el")), "low_income": frac(g.get("low_income"), 2022),
            "high_needs": frac(g.get("high_needs")),
            "soc": [None if w is None else round(1 - w / 100, 4) for w in race.get("white") or [None] * len(YEARS)],
            "sped_share": frac(e.get("sped_share")), "gap": frac(e.get("nss_gap"))}


def town_shares(t):
    """A town's residents 65 and older, of color (all but non-Hispanic white), and with a disability, as fractions
    of its residents (the last of those not living in institutions), each census release at its fiscal year; and its
    long-term debt per resident"""
    dm = t.get("demo") or {}
    of = lambda part, whole: [None if p is None or not w else round(p / w, 4) for p, w in
                              zip(dm.get(part) or [None] * len(YEARS), dm.get(whole) or [None] * len(YEARS))]
    return {"older": of("age_65_up", "population"),
            "poc": [None if x is None else round(1 - x, 4) for x in of("white", "population")],
            "disability": of("disability", "disability_universe"),
            "debt_pc": (t.get("debt") or {}).get("outstanding") or [None] * len(YEARS)}


# The map's groupings: each town's or district's comparison group of one kind, as its cmp names it, in the order
# the groups run (by size, or income, or density); ordered groupings color their groups on one ramp
MAP_GROUPINGS = {
    "towns": [("region", "Region", False), ("peer_state", "Population group, statewide", True), ("peer", "Population group, within its region", False),
              ("type", "Community type", True), ("fifth", "Income fifth", True), ("schools", "School districts", True)],
    "districts": [("region", "Region", False), ("cohort_state", "Enrollment group, statewide", True), ("cohort", "Enrollment group, within its region", False),
                  ("type", "Community type", True), ("fifth", "Income fifth", True)],
}


def real_dollars(vals, cpi):
    """Values in LATEST_YEAR dollars; cpi: fiscal year -> Northeast CPI"""
    return [None if v is None else v * cpi[LATEST_YEAR] / cpi[y] for v, y in zip(vals, YEARS)]


def ratios(a, b):
    return [None if x is None or y is None or not y else x / y for x, y in zip(a, b)]


def ends(vals):
    """The first and last year indexes with a positive value, if two"""
    at = [i for i, v in enumerate(vals) if v is not None and v > 0]
    return (at[0], at[-1]) if len(at) >= 2 and at[-1] > at[0] else None


def trend(vals):
    """The least-squares line's yearly growth, from its first year to its last, as the explorer's charts
    draw it: [rate, first year, last year]"""
    pts = [(i, v) for i, v in enumerate(vals) if v is not None]
    if len(pts) < 3:
        return None
    n = len(pts)
    mx = sum(i for i, _ in pts) / n
    my = sum(v for _, v in pts) / n
    slope = sum((i - mx) * (v - my) for i, v in pts) / sum((i - mx) ** 2 for i, _ in pts)
    a, b = pts[0][0], pts[-1][0]
    fa, fb = my + slope * (a - mx), my + slope * (b - mx)
    if fa <= 0 or fb <= 0:
        return None
    return [round((fb / fa) ** (1 / (b - a)) - 1, 5), YEARS[a], YEARS[b]]


def growth(vals, cpi):
    """Trend growth in LATEST_YEAR dollars, then nominal: [real rate, first year, last year, nominal rate].
    Each is fitted on its own series, so one can't be had from the other."""
    r, nom = trend(real_dollars(vals, cpi)), trend(vals)
    return [*r, nom[0]] if r and nom else None


def change(vals):
    """The change from the first year with a positive value to the last: [change, first year, last year]"""
    if not (e := ends(vals)):
        return None
    a, b = e
    return [round(vals[b] / vals[a] - 1, 4), YEARS[a], YEARS[b]]


def last_value(vals):
    """[the last value, its year]"""
    at = [i for i, v in enumerate(vals) if v is not None]
    return [round(vals[at[-1]], 4), YEARS[at[-1]]] if at else None


def reported_nss(e):
    """The year indexes with the district's above-required, aid and required spending per foundation pupil"""
    nss = e.get("nss") or {}
    return {i: {"above": a, "aid": c, "req": q, "fy": YEARS[i]}
            for i, (a, c, q) in enumerate(zip(nss.get("above_required", []), nss.get("ch70_aid", []), required_pfe(nss)))
            if a is not None and q is not None}


def combined(e1, w1, e2, w2, cpi):
    """Two districts as one, each weighted by the town's pupils in it (w1, w2); spending against the
    requirement at the latest year both have reported"""
    if not w1 or not w2:
        return {}
    p1, p2 = total_ppe(e1), total_ppe(e2)
    r1, r2 = reported_nss(e1), reported_nss(e2)
    both = sorted(set(r1) & set(r2))
    n1, n2 = (r1[both[-1]], r2[both[-1]]) if both else (None, None)
    out = {}
    mix = [None if a is None or b is None else (w1 * a + w2 * b) / (w1 + w2) for a, b in zip(p1, p2)]
    if (g := growth(mix, cpi)):
        out["ppe"] = g
    # Each district's change over the years both have, so the figure covers the years it's labeled with
    v1, v2 = e1["enroll"]["in_district"], e2["enroll"]["in_district"]
    x1, x2 = ends(v1), ends(v2)
    if x1 and x2 and (a := max(x1[0], x2[0])) < (b := min(x1[1], x2[1])) and all(
            v and v > 0 for v in (v1[a], v2[a], v1[b], v2[b])):
        c1, c2 = v1[b] / v1[a] - 1, v2[b] / v2[a] - 1
        out["enroll"] = [round((w1 * c1 + w2 * c2) / (w1 + w2), 4), YEARS[a], YEARS[b]]
    s1, s2 = district_shares(e1), district_shares(e2)
    for k in s1:
        both_k = [j for j, (a, b) in enumerate(zip(s1[k], s2[k])) if a is not None and b is not None]
        if both_k:
            j = both_k[-1]
            out[k] = [round((w1 * s1[k][j] + w2 * s2[k][j]) / (w1 + w2), 4), YEARS[j]]
    if n1 and n2:
        req = w1 * n1["req"] + w2 * n2["req"]
        out["above"] = [round((w1 * n1["above"] + w2 * n2["above"]) / req, 4), n1["fy"]]
        out["aid"] = [round((w1 * n1["aid"] + w2 * n2["aid"]) / req, 4), n1["fy"]]
    return out


def town_measures(t, cpi, cap_towns, cap_year):
    """A town's measures for the map, or a town average's: the ones figured for one town alone (the cap, overrides
    passed, density, income fifths) a town's only. cap_towns: the cap tab's towns in its latest year (cap_year)."""
    c = cap_towns.get(t["id"])
    v = {
        "spend": growth(ratios(t["spend_total"], t["pop"]), cpi),
        "pop": change(t["pop"]),
        "eqv": growth(t["eqv"], cpi) if t.get("eqv") else None,
        "levy_room": last_value(levy_room(t)),
        "bill_income": last_value(t["sf_bill_income"]) if t.get("sf_bill_income") else None,
        "edu": last_value(edu_share(t)),
        "cap_hh": [round((c["nocap"] - c["target"]) / c["hh"]), cap_year] if c and c.get("hh") else None,
        **{k: last_value(vals) for k, vals in fiscal_shares(t).items() if vals},
        **{k: last_value(vals) for k, vals in town_shares(t).items()},
        # Overrides its voters passed, FY2005 on
        "overrides": [sum(1 for fy, kind, *_ in t.get("passed") or [] if kind == "override"),
                      max([fy for fy, *_ in t.get("passed") or []], default=LATEST_YEAR)] if t["kind"] == "town" else None,
    }
    com = t.get("community") or {}
    if com:
        v["density"] = [com["density"], com["fy"]]
    if t.get("income"):
        v["income"] = [t["income"]["income_pc"], t["income"]["year"]]
    return {m: x for m, x in v.items() if x is not None}


def district_measures(e, cpi):
    """A district's measures for the map, or a district average's"""
    nss = e.get("nss") or {}
    req = required_pfe(nss)
    v = {
        "ppe": growth(total_ppe(e), cpi),
        "enroll": change(e["enroll"]["in_district"]),
        "above": last_value(ratios(nss.get("above_required", []), req)) if req else None,
        "aid": last_value(ratios(nss.get("ch70_aid", []), req)) if req else None,
        **{k: last_value(vals) for k, vals in district_shares(e).items()},
    }
    com = e.get("community") or {}
    if com:
        v["density"] = [com["density"], com["fy"]]
    if e.get("income"):
        v["income"] = [e["income"]["income_pc"], e["income"]["year"]]
    return {m: x for m, x in v.items() if x is not None}


def fifth_cuts(xs):
    """Where each income fifth begins, so the map's income colors are exactly the fifths"""
    return [min(x["income_pc"] for x in xs if x["fifth"] == f) for f in (2, 3, 4, 5)] if xs else None


def map_group_of(u, key, by_id):
    """A town's or district's map group of one kind (MAP_GROUPINGS' key), as its cmp names it. Within its region:
    one whose region has too few of its size for a comparison group stays in its region's group of that size,
    however few share it, a regional outlier there (Tim, #219, #221); the explorer compares it with the statewide
    group. With no region, the statewide group. by_id: the groups there are."""
    fallback = {"peer": "peer_state", "cohort": "cohort_state"}.get(key)
    c = u.get("cmp") or {}
    if fallback and not c.get(key) and c.get(fallback) and u.get("region"):
        mine = f"{c[fallback]}-{slug(u['region'])}"
        if mine in by_id:
            return mine
    return c.get(key) or (c.get(fallback) if fallback else None)


def map_towns(towns, entity, cap, cpi, org4, src):
    """Each town's map record: its outline, the district that serves its elementary grades and its secondary
    grades, its measures, and for a town whose grades split between two districts, the two combined ("all");
    with the districts the map shows and a split town's pupils in each ({town id: [elementary, secondary]})"""
    outline = {norm(g["town"]): g["d"] for g in src.geo["towns"].values()}
    count = {i: len(e.get("towns") or []) for i, e in entity.items()}
    mem = src.members
    pupils_in = {(int(o), norm(t)): float(f) for o, t, f in zip(mem["Org4Code"], mem["Town"], mem["Foundation_enrollment"])}

    def pupils(did, town, kind):
        """The town's pupils in a district: all of a district of its own, its share of a regional one"""
        if kind == "town":
            fe = entity[did]["enroll"]["foundation"]
            at = [v for v in fe if v is not None]
            return at[-1] if at else None
        return pupils_in.get((org4.get(did), norm(town)))

    cap_year = cap["meta"]["latest"]
    cap_towns = {t["id"]: t for t in cap["years"][str(cap_year)]["towns"]}
    used, out, weights = set(), [], {}
    for t in towns:
        k = norm(t["name"])
        if k not in outline:
            raise ValueError(f"No outline for {t['name']}")
        # Its own district for the elementary grades, else its regional one (of two, the smaller);
        # for high school, a regional district if it has one (of two, the larger), else its own
        ds = [d for d in t.get("districts") or [] if d["kind"] != "vocational" and d["id"] in entity]
        own = [d["id"] for d in ds if d["kind"] == "town"]
        regs = sorted((d["id"] for d in ds if d["kind"] == "regional"), key=lambda i: count[i])
        elem = own[0] if own else regs[0] if regs else None
        high = regs[-1] if regs else own[0] if own else None
        used |= {elem, high} - {None}
        kind_of = {d["id"]: d["kind"] for d in ds}
        pair = [(i, pupils(i, t["name"], kind_of[i])) for i in (elem, high)] if elem and high and elem != high else None
        both = combined(entity[pair[0][0]], pair[0][1], entity[pair[1][0]], pair[1][1], cpi) if pair else None
        if both and all(w for _, w in pair):
            weights[t["id"]] = [round(w, 1) for _, w in pair]
        com = t.get("community") or {}
        out.append({"id": t["id"], "name": t["name"], "d": outline[k], "elem": elem, "high": high, "ct": com.get("type"),
                    "v": town_measures(t, cpi, cap_towns, cap_year), **({"all": both} if both else {})})
    return out, used, weights


def map_data(districts, towns, cap, cpi, org4, src, town_avgs=(), district_avgs=()):
    """map.json: the Commonwealth tab's map. Each town's outline, the district that serves its
    elementary grades and its secondary grades, and a few measures of how towns and districts
    have changed. Growth is the trend's, as the explorer's charts draw it: the least-squares line's
    yearly growth from its first year to its last, in LATEST_YEAR dollars and then nominal, each
    fitted on its own series; each value carries its years. A town whose grades split between two
    districts also gets the two combined ("all"), each weighted by the town's own pupils in it:
    a district of its own counts all its foundation pupils, a regional one the town's foundation
    enrollment there (DESE's regional members file, its one year). districts and towns: the
    explorer's full entities; cap: cap_data(); cpi: fiscal year -> Northeast CPI; org4: district
    id -> its four-digit DESE code; src: read_sources(); town_avgs, district_avgs: the averages the
    map can group by, the explorer's and the map's own within regions."""
    geo = src.geo
    entity = {e["id"]: e for e in districts}
    out_towns, used, weights = map_towns(towns, entity, cap, cpi, org4, src)
    out_districts = {}
    for i in sorted(used):
        e = entity[i]
        com = e.get("community") or {}
        out_districts[i] = {"name": e["name"], "ct": com.get("type"), "v": district_measures(e, cpi)}
    # The groupings: each town's and district's group of each kind, and each group's values, as its own average's
    by_id = {a["id"]: a for a in [*town_avgs, *district_avgs]}
    groupings = {}
    for layer, units in (("towns", towns), ("districts", [entity[i] for i in sorted(used)])):
        groupings[layer] = []
        for key, label, ordered in MAP_GROUPINGS[layer]:
            of = {u["id"]: map_group_of(u, key, by_id) for u in units}
            of = {u: g for u, g in of.items() if g in by_id}
            # Colored by region, a town or district its region's average leaves out (the very large) shows the group
            # that pools it instead, so each group's count in the key and the figure each place shows are the average's
            # own (#242). `of` keeps its region, which the groupings within regions outline.
            apart = {}
            if key == "region":
                home = "towns-apart" if layer == "towns" else "cohort-xlarge"
                apart = {u: home for u, g in of.items() if u not in set(by_id[g]["member_ids"])}
                assert set(apart) <= set(by_id[home]["member_ids"]), f"{layer}: left out of a region but not in {home}"
            ids = list(dict.fromkeys(a["id"] for a in (town_avgs if layer == "towns" else district_avgs)
                                     if a["id"] in set(of.values()) - set(apart.values())))
            ids += sorted(set(apart.values()))  # last, as the map lists a statewide group within regions
            groupings[layer].append({"key": key, "label": label, "ordered": ordered, "groups": ids, "of": of,
                                     **({"apart": apart} if apart else {})})
    used_groups = sorted({g for gs in groupings.values() for x in gs for g in x["groups"]})
    town_ids = {a["id"] for a in town_avgs}
    cap_year = cap["meta"]["latest"]
    cap_towns = {t["id"]: t for t in cap["years"][str(cap_year)]["towns"]}
    groups = {g: {"name": by_id[g]["name"], "members": by_id[g].get("members"),
                  "v": town_measures(by_id[g], cpi, cap_towns, cap_year) if g in town_ids else district_measures(by_id[g], cpi),
                  **({"outlier": True} if by_id[g].get("outlier") else {})} for g in used_groups}
    years = map_series(entity, sorted(used), towns, cap, cpi, org4, src, weights,
                       [by_id[g] for g in used_groups if g in town_ids],
                       [by_id[g] for g in used_groups if g in {a["id"] for a in district_avgs}])
    return {"meta": {"built": dt.date.today().isoformat(), "start": EXPLORER_START, "latest": LATEST_YEAR,
                     "cap_fy": cap_year, "width": geo["width"], "height": geo["height"],
                     "outlines": geo["source"],
                     "fifths": {"towns": fifth_cuts([t["income"] for t in towns if t.get("income")]),
                                "districts": fifth_cuts([e["income"] for e in districts if e.get("income")]),
                                "year": max((t["income"]["year"] for t in towns if t.get("income")), default=None)}},
            "towns": out_towns, "districts": out_districts, "groupings": groupings, "groups": groups, "series": years}


def map_series(entity, used, towns, cap, cpi, org4, src, weights, town_groups=(), district_groups=()):
    """The map's measures year by year, so it can show any span of years for a change and any year for a
    level: every array runs along "years", FY{EXPLORER_START} to the cap tab's latest year, trailing
    years without a value left off. Growth and change are figured in the browser, by the same rules as
    map_data (trend growth on the years chosen, before and after inflation with "cpi"; change between the
    first and last years with a value). Towns: spending per resident, population, property value per
    resident, room under the levy limit, the tax bill's share of income, education's share of spending,
    residents per square mile, and the cap's change per household with no cap (FY2009 on, the first year
    with a household count; FY2026 and FY2027 as the cap tab figures them). Districts: in-district
    spending per pupil, in-district enrollment, actual spending above required, Chapter 70 aid, and
    required spending, each per foundation pupil, and students per square mile (the last year, the
    latest foundation enrollment, as the community types count it). "w": a split town's pupils in its
    elementary and its secondary district, for combining the two."""
    cap_years = sorted(int(y) for y in cap["years"])
    axis = YEARS + [y for y in cap_years if y > LATEST_YEAR]
    at = {y: i for i, y in enumerate(axis)}

    def trim(vals, digits=None):
        out = [None if v is None or v != v else (round(v, digits) if digits is not None else round(v)) for v in vals]
        while out and out[-1] is None:
            out.pop()
        return out if any(v is not None for v in out) else None

    def on_axis(vals, years=YEARS):
        row = [None] * len(axis)
        for v, y in zip(vals, years):
            if y in at:
                row[at[y]] = v
        return row

    area, towns_of = src.area, src.towns_of

    # The cap's change per household: each history year's no-cap targets keep that year's statewide total;
    # the cap tab's own years replace them, so the latest year is the figure map_data shows
    H = cap.get("history") or {}
    hyears = H.get("years") or []
    factor = {}
    for j, y in enumerate(hyears):
        rows = [(h["cey"][j], h["target"][j], h["foundation"][j]) for h in H["towns"].values()
                if h["cey"][j] is not None and h["target"][j] is not None]
        if rows and sum(c for c, _, f in rows if f):
            factor[y] = nocap_factor([t for _, t, _ in rows], [c for c, _, _ in rows], [f for _, _, f in rows])
    cap_town = {y: {t["id"]: t for t in cap["years"][str(y)]["towns"]} for y in cap_years}

    def town_row(t):
        """A town's measures year by year, or a town average's: all but density and the cap, figured for a town alone"""
        pop = t["pop"]
        return {
            "spend": trim(on_axis([None if s is None or not p else s / p for s, p in zip(t["spend_total"], pop)]), 2),
            "pop": trim(on_axis(pop)),
            "eqv": trim(on_axis(t["eqv"])) if t.get("eqv") else None,
            "levy_room": trim(on_axis(levy_room(t)), 4),
            "bill_income": trim(on_axis(t["sf_bill_income"]), 4) if t.get("sf_bill_income") else None,
            "edu": trim(on_axis(edu_share(t)), 4),
            **{k: trim(on_axis(vals), 4) for k, vals in fiscal_shares(t).items() if vals},
            **{k: trim(on_axis(vals), 0 if k == "debt_pc" else 4) for k, vals in town_shares(t).items()},
        }

    def district_row(e, ppe):
        """A district's measures year by year, or a district average's: all but density"""
        nss = e.get("nss") or {}
        return {
            "ppe": trim(on_axis(ppe)),
            "enroll": trim(on_axis(e["enroll"]["in_district"]), 1),
            "above": trim(on_axis(nss.get("above_required") or [])),
            "aid": trim(on_axis(nss.get("ch70_aid") or [])),
            "req": trim(on_axis(required_pfe(nss))),
            **{k: trim(on_axis(vals), 4) for k, vals in district_shares(e).items()},
        }

    s_towns = {}
    for t in towns:
        k = norm(t["name"])
        pop = t["pop"]
        hh = t.get("hh") or [None] * len(YEARS)
        cap_hh = [None] * len(axis)
        h = H.get("towns", {}).get(t["id"])
        for j, y in enumerate(hyears):
            if y not in at:
                continue
            if y in cap_town:
                c = cap_town[y].get(t["id"])
                if c and c.get("hh"):
                    cap_hh[at[y]] = (c["nocap"] - c["target"]) / c["hh"]
            elif h and y <= LATEST_YEAR and y in factor and h["cey"][j] is not None and h["target"][j] is not None and hh[YEARS.index(y)]:
                nocap = round(h["cey"][j] * factor[y]) if h["foundation"][j] else 0
                cap_hh[at[y]] = (nocap - h["target"][j]) / hh[YEARS.index(y)]
        row = town_row(t) | {
            "density": trim(on_axis([None if p is None or not area.get(k) else p / area[k] for p in pop]), 1),
            "cap_hh": trim(cap_hh),
        }
        s_towns[t["id"]] = {m: v for m, v in row.items() if v}
        if t["id"] in weights:
            s_towns[t["id"]]["w"] = weights[t["id"]]

    s_districts = {}
    for i in used:
        e = entity[i]
        nss = e.get("nss") or {}
        ks = towns_of.get(org4.get(i), []) if e.get("regional") else [norm(e["name"])]
        sqmi = sum(area[x] for x in ks) if ks and all(x in area for x in ks) else None
        dens = on_axis([None if f is None or not sqmi else f / sqmi for f in e["enroll"]["foundation"]])
        com = e.get("community") or {}
        if com.get("fy") in at:
            dens[at[com["fy"]]] = com["density"]
        row = district_row(e, total_ppe(e)) | {"density": trim(dens, 1)}
        s_districts[i] = {m: v for m, v in row.items() if v}
    # The group averages, as the map colors a town by its group's
    s_groups = {t["id"]: {m: v for m, v in town_row(t).items() if v} for t in town_groups}
    for e in district_groups:
        s_groups[e["id"]] = {m: v for m, v in district_row(e, total_ppe(e)).items() if v}
    return {"years": axis, "cpi": [round(cpi[y], 3) if y in cpi else None for y in axis],
            "towns": s_towns, "districts": s_districts, "groups": s_groups}


def explorer_districts(hub, defs):
    """The explorer's districts: the report's rule for its Western MA cohorts
    (school_shared.get_western_cohort_districts: latest-year in-district FTE and
    per-pupil spending both present, cohort by get_enrollment_group) applied to every
    region and school type. Each is a dict: key (lowercase name), name, code (DESE
    DIST_CODE), org (four-digit code), type, cohort, region (None if DESE gives none). defs: the cohort
    definitions (get_cohort_definitions), whose boundaries the cohorts are by.
    Each district's FTE is read from its own rows (Hub.rows): latest_indistrict_fte over the
    whole frame lower-cased all of it for each district, 27 seconds of the export (#237)."""
    df, reg = hub.df, hub.reg
    present = {n.lower(): n for n in df["DIST_NAME"].unique()}
    code = df["DIST_CODE"].groupby(hub.name).first().astype(int)
    latest = int(df["YEAR"].max())
    epp = df[(hub.cat == "expenditures per pupil") & (df["YEAR"] == latest)
             & ~hub.subcat.isin(["total expenditures", "total in-district expenditures"])]
    total_ppe = epp.groupby(hub.name[epp.index])["IND_VALUE"].sum()
    out = []
    for _, r in reg[reg["SCHOOL_TYPE"].isin(SCHOOL_TYPES)].drop_duplicates("DIST_NAME").iterrows():
        key = r["DIST_NAME"].lower()
        if key not in present or key in EXCLUDE_DISTRICTS:
            continue
        fte = latest_indistrict_fte(hub.rows([key]), key)
        if fte <= 0 or total_ppe.get(key, 0) <= 0:
            continue
        kind = SCHOOL_TYPES[r["SCHOOL_TYPE"]]
        if kind == "traditional":
            group = get_enrollment_group(fte, {k: v["range"] for k, v in defs.items()})
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


def cohort_short(k, defs):
    """A cohort's short label: the report's (defs, get_cohort_definitions), or COHORT_TEXT's for the ones it doesn't
    define"""
    return COHORT_TEXT[k][1] if k in COHORT_TEXT else defs.get(k, {}).get("short_label", k)


def cohort_name(k, defs):
    """'Statewide Tiny enrollment group average', 'Very large districts average'; within a region,
    within_regions names the group. The site calls the report's size cohorts enrollment groups, as it calls the
    towns' population groups (Tim, #217; #215 had "size peer groups" for both)"""
    if k in ("XLARGE", "VOCATIONAL", "CHARTER"):
        return {"XLARGE": "Very large districts average", "VOCATIONAL": "Vocational districts average",
                "CHARTER": "Charter schools average"}[k]
    return f"Statewide {cohort_short(k, defs)} enrollment group average"


def town_data(dists, latest, src, fifths):
    """The town view: (towns.json content, {chunk: full towns}, every full town, each
    district's towns, the map's outlier groups). dists: explorer_districts() with each district's
    explorer id; src: read_sources(); fifths: income_fifths() of the towns. The outlier groups are the
    population groups within a region that have too few towns for a comparison (fewer than
    MIN_AVERAGE_MEMBERS): the map keeps such towns in their region, marked as outliers, while the
    explorer compares them with the statewide group."""
    tc = town_context(dists, latest, src)
    full, schools_of = town_entities(tc, src, latest)
    type_of = town_types(tc, src, latest)
    setup = {norm(x["name"]): school_setup(x) for x in full}
    groups = town_groups(tc, fifths, type_of, setup)
    averages = [town_average(g, tc, src) for g in groups if g.outlier is None]
    outliers = [town_average(g, tc, src) for g in groups if g.outlier is not None]
    have = {a["id"] for a in averages}
    chunks = {}
    for a in averages:
        chunks.setdefault(a["chunk"], []).append(a)
    for x in full:
        pid, rk = x["peer"], x["region"]
        k = norm(x["name"])
        x["cmp"] = {"peer": f"{pid}-{slug(rk)}" if pid and rk and f"{pid}-{slug(rk)}" in have else None,
                    "peer_state": pid, "region": f"towns-region-{slug(rk)}" if rk else None, "state": "towns-statewide",
                    "state_all": "towns-statewide-all", "gateway": "towns-gateway",
                    "type": f"towns-type-{type_of[k]['type']}" if k in type_of and f"towns-type-{type_of[k]['type']}" in have else None,
                    "fifth": f"towns-fifth-{fifths[k]['fifth']}" if fifths and k in fifths else None,
                    "schools": f"towns-schools-{setup[k]}" if f"towns-schools-{setup.get(k)}" in have else None}
        if k in type_of:
            x["community"] = type_of[k]
        if fifths and k in fifths:
            x["income"] = fifths[k]
        x["chunk"] = chunk_of(rk)
        chunks.setdefault(x["chunk"], []).append(x)
    return towns_index(tc, averages, full, latest, src), chunks, full, district_links(tc, schools_of), outliers


def town_context(dists, latest, src):
    """What the town view's steps share: the DLS rows of the explorer's years and each town's name by key (keys,
    sorted), the explorer's districts by four-digit code and each town's own district, each regional district's
    member towns, each district's towns, the cities that stand apart, each town's population group and region"""
    tc = SimpleNamespace()
    tc.dls = src.dls[src.dls["FY"].isin(YEARS)].assign(key=lambda d: d["Municipality"].map(norm))
    tc.names = names = tc.dls.drop_duplicates("key").set_index("key")["Municipality"].to_dict()
    tc.keys = keys = sorted(names)
    missing = [k for k in keys if (k, latest) not in src.population]
    if missing:
        raise ValueError(f"Towns without population data: {missing}")
    tc.by_org = by_org = {f"{d['org']:04d}": d for d in dists}
    tc.own_of = own_of = {norm(d["name"]): d for d in dists if not is_regional(d) and d["type"] == "traditional"
                          and norm(d["name"]) in names}
    tc.member_towns = {org: [{"name": names.get(mk, town), "id": slug(names[mk]) if mk in names else None, "key": mk}
                             for mk, town in zip(g["key"], g["Town"])]
                       for org, g in src.members.groupby("Org4Code")}
    # Each district's towns: its member towns, or the town of the same name
    tc.district_towns, tc.towns_kind = {}, {}
    for d in dists:
        org = f"{d['org']:04d}"
        if org in tc.member_towns:
            tc.district_towns[d["id"]], tc.towns_kind[d["id"]] = [m["key"] for m in tc.member_towns[org]], "members"
        elif norm(d["name"]) in own_of:
            tc.district_towns[d["id"]], tc.towns_kind[d["id"]] = [norm(d["name"])], "town"
    # The cities whose own school districts are very large stand apart from the statewide and regional averages;
    # they are in their population groups, as the very large districts are in theirs
    tc.apart = {k for k, d in own_of.items() if d["cohort"] == "XLARGE"}
    tc.pooled = [k for k in keys if k not in tc.apart]
    tc.gateway = {norm(c) for c in GATEWAY_CITIES}
    if not tc.gateway <= set(keys):
        raise ValueError(f"Gateway Cities missing from the town list: {sorted(tc.gateway - set(keys))}")
    tc.peer = {k: next(f"towns-{sfx}" for sfx, lo, hi, _ in TOWN_PEERS
                       if lo <= src.population[(k, latest)] and (hi is None or src.population[(k, latest)] < hi))
               for k in keys}
    # Region: the town's own district's, else its regional academic district's, else any of
    # its districts', else its county's (regional_data.csv's district_code is not DESE's
    # code for districts whose region was inferred from the host town, so match districts)
    tc.region = {}
    for k in keys:
        own = own_of.get(k)
        mine = ([own] if own else []) + sorted(
            (by_org[o] for o in src.members.loc[src.members["key"] == k, "Org4Code"] if o in by_org),
            key=lambda d: d["type"] != "traditional")
        tc.region[k] = next((d["region"] for d in mine if d["region"]), None) or COUNTY_REGION.get(src.county.get(k))
    return tc


def town_entities(tc, src, latest):
    """(every town's full record, each town's school districts by key)"""
    rlc = src.c70[src.c70["fy"] == latest].set_index("Org4Code")["distrlc"].to_dict()
    population, members = src.population, src.members
    full, schools_of = [], {}
    for k in tc.keys:
        schools = []
        own = tc.own_of.get(k)
        if own:
            schools.append({"name": own["name"], "id": own["id"], "kind": "town",
                            "contribution": dollars(rlc.get(f"{own['org']:04d}")), "with": []})
        for _, m in members[members["key"] == k].iterrows():
            org = int(m["Org4Code"])
            linked = tc.by_org.get(m["Org4Code"])
            # The members file lists regional districts alone: academic from 0600, vocational from 0800
            schools.append({"name": linked["name"] if linked else m["District"], "id": linked["id"] if linked else None,
                            "kind": "regional" if 600 <= org < 800 else "vocational",
                            "contribution": dollars(m["Required_contribution"]),
                            "with": [{"name": o["name"], "id": o["id"]} for o in tc.member_towns[m["Org4Code"]]
                                     if o["key"] != k]})
        schools_of[k] = schools
        name = tc.names[k]
        full.append({"id": slug(name), "name": name, "kind": "town", "county": src.county.get(k),
                     "region": tc.region[k], "peer": tc.peer[k], "gateway": k in tc.gateway,
                     "pop": series(pd.Series({fy: population.get((k, fy)) for fy in YEARS}), 0),
                     "hh": series(pd.Series({fy: src.households.get((k, fy)) for fy in YEARS}), 0),
                     **town_series(tc.dls[tc.dls["key"] == k], population), **property_series(src.prop, [k], population),
                     **fiscal_series(src.fiscal, [k], population), **town_ratings(src.fiscal, k),
                     **town_passed(src.fiscal, k), **demo_series(src.demo, [k]),
                     "districts": schools})
    return full, schools_of


def town_types(tc, src, latest):
    """Each town's community type: the Inner Core, the other Gateway Cities, rural under RURAL_RESIDENTS per
    square mile, and the rest; {key: {"type", "density", "fy"}}"""
    core = {norm(t) for t in INNER_CORE}
    type_of = {}
    for k in tc.keys:
        p, a = src.population.get((k, latest)), src.area.get(k)
        if p and a:
            kind = "core" if k in core else "gateway" if k in tc.gateway else "rural" if p / a < RURAL_RESIDENTS else "suburban"
            type_of[k] = {"type": kind, "density": round(p / a), "fy": latest}
    return type_of


def school_setup(x):
    """How much of its schooling a town runs itself. None (it tuitions its pupils to other districts), regional
    districts only, its own district and a regional one (one for its elementary grades, the other for its
    secondary, or for some of each), or its own for every grade; vocational districts aside"""
    kinds = {d["kind"] for d in x.get("districts") or []}
    own, reg = "town" in kinds, "regional" in kinds
    return "own" if own and not reg else "own-regional" if own else "regional" if reg else "none"


def peer_phrase(label):
    """'towns under 1,000', 'towns of 1,000–2,499'"""
    return f"towns {label}" if label.startswith("under") else f"towns of {label}"


def town_groups(tc, fifths, type_of, setup):
    """The towns' comparison groups, as Groups, in the order the explorer lists them, with the map's regional
    outliers among them (outlier True): the town counterpart of district_groups"""
    keys, state = tc.keys, {"state": "towns-statewide"}
    # Two statewide averages: every city and town, the default (Tim, #242), and the same without the cities whose
    # school districts are very large
    groups = [Group("towns-statewide", f"Statewide towns average, without the {len(tc.apart)} cities with very large "
                    "districts", "state", None, None, tc.pooled, {}),
              Group("towns-statewide-all", "Statewide towns average", "state", None, None, list(keys), state)]
    for rk, rl in REGIONS.items():
        groups.append(Group(f"towns-region-{slug(rk)}", f"{rl} towns average", "region", None, rk,
                            [k for k in tc.pooled if tc.region[k] == rk], state))
    for sfx, lo, hi, label in TOWN_PEERS:
        pid, phrase = f"towns-{sfx}", peer_phrase(label)
        groups.append(Group(pid, f"{phrase[0].upper()}{phrase[1:]} average", "peer", pid, None,
                            [k for k in keys if tc.peer[k] == pid], state))
        groups += within_regions(pid, "region-peer", lambda rk: [k for k in keys if tc.peer[k] == pid and tc.region[k] == rk],
                                 lambda rk: f"{pid}-{slug(rk)}", lambda rl: f"{rl} {phrase}",
                                 lambda rk: {"peer_state": pid, "region": f"towns-region-{slug(rk)}", "state": "towns-statewide"})
    # Gateway Cities: the Commonwealth's list, very large cities included
    groups.append(Group("towns-gateway", "Gateway Cities average", "group", None, None, sorted(tc.gateway), state))
    # The cities the region averages leave out, together: the map's region grouping shows them as this group,
    # where it had shown each with the region average that leaves it out (#242)
    groups.append(Group("towns-apart", "Cities with very large school districts average", "apart", None, None,
                        sorted(tc.apart), state))
    # Income fifths, as the cap tab counts them; every town of each, the very large cities included. Community types
    # and school districts likewise; a kind no town is of has no group
    kinds = [(f"towns-fifth-{f}", f"Towns in the {word} income fifth average", "fifth",
              [k for k in keys if (fifths or {}).get(k, {}).get("fifth") == f]) for f, word in FIFTH_NAMES.items()]
    kinds += [(f"towns-type-{t}", name, "type", [k for k in keys if type_of.get(k, {}).get("type") == t])
              for t, name in (("core", "Inner Core average"), ("gateway", "Gateway Cities outside the Inner Core average"),
                              ("suburban", "Suburban and other towns average"), ("rural", "Rural towns average"))]
    kinds += [(f"towns-schools-{sk}", name, "schools", [k for k in keys if setup.get(k) == sk])
              for sk, name in (("none", "Towns with no school district average"),
                               ("regional", "Towns in regional districts only average"),
                               ("own-regional", "Towns with their own district and a regional one average"),
                               ("own", "Towns with their own district for every grade average"))]
    groups += [Group(gid, name, scope, None, None, ks, state) for gid, name, scope, ks in kinds if ks]
    return groups


def town_average(g, tc, src):
    """A town Group's pooled per-resident average; pop is the members' combined population. A population group's
    average says what it is (ref, label); the map's regional outliers compare with nothing."""
    ks, population, households = g.members, src.population, src.households
    text = {}
    if g.size and g.outlier is None:
        label = next(label for sfx, _, _, label in TOWN_PEERS if f"towns-{sfx}" == g.size)
        where = f"{REGIONS[g.region]} " if g.region else ""
        text = {"ref": f"the average for {where}{peer_phrase(label)}", "label": f"{label} residents"}
    return {"id": g.id, "name": g.name, "kind": "average", "scope": g.scope, "peer": g.size, "region": g.region,
            "members": len(ks), "member_ids": [slug(tc.names[k]) for k in sorted(ks, key=lambda k: tc.names[k])],
            "chunk": slug(g.region) if g.region else "statewide", **text,
            **({"outlier": True} if g.outlier else {}),
            "cmp": g.cmp if g.outlier is not None else shared_cmp(g, "towns-gateway", "towns-statewide-all"),
            "pop": series(pd.Series({fy: sum(population.get((k, fy), 0) for k in ks) for fy in YEARS}), 0),
            "hh": series(pd.Series({fy: sum(households.get((k, fy), 0) for k in ks) or None for fy in YEARS}), 0),
            **town_series(tc.dls[tc.dls["key"].isin(ks)], population), **property_series(src.prop, ks, population),
            **fiscal_series(src.fiscal, ks, population), **demo_series(src.demo, ks)}


def towns_index(tc, averages, full, latest, src):
    """towns.json: the town view's meta, and light lists of its averages and towns for the menus"""
    return {
        "meta": {
            "years": YEARS, "latest": latest, "start": EXPLORER_START, "built": dt.date.today().isoformat(),
            "members_fy": latest, "regions": REGIONS,
            "apart": sorted(tc.names[k] for k in tc.apart),
            # July 1 of this year sets each town's population group
            "peer_year": latest - 1,
            "peers": [{"id": f"towns-{sfx}", "label": f"{label} residents"} for sfx, _, _, label in TOWN_PEERS],
            "sources": {
                "finances": "DLS Municipal Databank, Schedule A general fund revenues and expenditures",
                "population": "Census Bureau subcounty population estimates, July 1 at the start of each fiscal year "
                              "(FY2021: April 1, 2020 estimates base)",
                "members": f"DESE FY{latest} Chapter 70 summary for regional districts",
                "property": "DLS Municipal Databank: LA-4 parcel counts and values by use, equalized valuations, "
                            "average single-family tax bill",
            },
            # Both are always there: every input is required (#237); the page reads them as flags
            "property": True,
            "fiscal": {k: v.get("data_current_as_of") for k, v in src.fiscal_sources.items() if isinstance(v, dict)},
            "c70_target": C70_TARGET,
        },
        "averages": [{"id": a["id"], "name": a["name"], "scope": a["scope"], "peer": a["peer"], "region": a["region"],
                      "chunk": a["chunk"]} for a in averages],
        "towns": [{"id": x["id"], "name": x["name"], "county": x["county"], "region": x["region"],
                   "peer": x["peer"], "gateway": x["gateway"], "chunk": x["chunk"]} for x in full],
    }


def district_links(tc, schools_of):
    """Each district's towns, for its page: what each must contribute to it, and the town's other districts"""
    def town_link(eid, k):
        mine = next(s for s in schools_of[k] if s["id"] == eid)
        return {"id": slug(tc.names[k]), "name": tc.names[k], "contribution": mine["contribution"],
                "also": [{"name": s["name"], "id": s["id"], "kind": s["kind"]} for s in schools_of[k] if s is not mine]}
    return {eid: {"towns": [town_link(eid, k) for k in ks], "towns_kind": tc.towns_kind[eid]}
            for eid, ks in tc.district_towns.items()}


def write_csvs(out_dir: Path, dists, all_towns, src, latest):
    """The source data behind the explorer, as CSV files beside its JSON: one row per
    district-year or town-year, values as each agency published them, and a source_url on
    every row. dists: the explorer's districts (explorer_districts, with ids), by name; src: read_sources(),
    whose frames are as each file was read, so each step copies a frame before changing it. Returns what it
    wrote, for the log."""
    written = []
    for name, frame in [*district_csvs(dists, src, latest), *town_csvs(all_towns, src, latest),
                        ("cpi.csv", cpi_csv(src, latest))]:
        frame.to_csv(out_dir / name, index=False)
        written.append(f"{name} ({len(frame):,} rows, {(out_dir / name).stat().st_size / 1024:.0f} KB)")
    return written


def district_csvs(dists, src, latest):
    """(file name, frame) for each district CSV, in the order they're written"""
    orgs = {d["org"] for d in dists}
    names = {d["org"]: d["name"] for d in dists}
    yield "districts-spending-per-pupil.csv", spending_csv(src, orgs, latest)
    yield "districts-chapter70.csv", chapter70_csv(src, orgs, names, latest)
    yield "districts-revenue.csv", revenue_csv(src.census, {d["org"] for d in dists if is_regional(d)}, names)
    yield from sped_csvs(src, orgs, names, latest)
    yield "districts-nss-budgets.csv", budgets_csv(src, orgs, names, latest)
    yield "districts-students.csv", students_csv(src, orgs, names, latest)


def by_org(frame, orgs, code="DIST_CODE"):
    """The rows of the explorer's districts, by the four-digit code in DESE's eight-digit one"""
    return frame[frame[code].astype(int).floordiv(10000).isin(orgs)]


def spending_csv(src, orgs, latest):
    """DESE expenditures per pupil and enrollment, wide: one column per DESE subcategory. FY2005-FY2008
    from DESE's archive, which also printed payments to out-of-district schools per pupil"""
    exp = by_org(pd.concat([src.func, src.exp], ignore_index=True), orgs)
    exp = exp[exp["SY"].between(EXPLORER_START, latest) & exp["IND_CAT"].isin(["Expenditures Per Pupil", "Student Enrollment"])]
    exp = exp.assign(column=exp["IND_CAT"].map({"Expenditures Per Pupil": "Per pupil: ", "Student Enrollment": "Enrollment: "})
                     + exp["IND_SUBCAT"])
    wide = exp.pivot_table(index=["SY", "DIST_CODE", "DIST_NAME"], columns="column", values="IND_VALUE", aggfunc="first")
    wide = wide.reset_index().rename(columns={"SY": "fiscal_year", "DIST_CODE": "dese_code", "DIST_NAME": "district"})
    hub = wide["fiscal_year"] > FUNC_LAST
    wide["source_url"] = [SOURCE_URLS["dese_expenditures"] if h else SOURCE_URLS["dese_archive"] for h in hub]
    wide["source_updated"] = [src.dese["expenditures_last_updated"] if h else None for h in hub]
    return wide.sort_values(["district", "fiscal_year"])


def chapter70_csv(src, orgs, names, latest):
    """DESE Chapter 70 profiles, DESE's own columns"""
    c70 = src.c70[src.c70["Org4Code"].astype(int).isin(orgs) & src.c70["fy"].between(EXPLORER_START, latest)].copy()
    # FY2023 required and actual NSS that update_dese_data.py restored from DESE's August 2025 file
    raw = pd.read_excel(C70_XLSX, sheet_name="DataC70", header=None, dtype=object).iloc[4:, [1, 3, 8, 9, 10]]
    raw.columns = ["Org4Code", "fy", "rqdnss", "rqdnss2", "actualNSS"]
    raw = raw[raw["fy"] == 2023].assign(Org4Code=lambda d: d["Org4Code"].astype(str).str.zfill(4)).set_index("Org4Code")
    def note(r):
        notes = []
        if r["fy"] == 2023 and r["Org4Code"] in raw.index and any(
                abs(float(raw.at[r["Org4Code"], f] or 0) - float(r[f] or 0)) > 1 for f in ("rqdnss", "rqdnss2", "actualNSS")):
            notes.append("FY2023 rqdnss, rqdnss2 and actualNSS from DESE's August 2025 profile: the March 2026 "
                         "profile assigns some districts' FY2023 values, most of them regional districts', to the "
                         "wrong districts")
        if not r["actualNSS"] or float(r["actualNSS"]) <= 0:
            notes.append("actualNSS 0 means DESE has not reported actual spending; the explorer treats it as missing")
        return "; ".join(notes)
    c70["note"] = c70.apply(note, axis=1)
    # The explorer's name for each district; DESE's profile names some differently ("ADAMS CHESHIRE")
    c70["district"] = c70["Org4Code"].astype(int).map(names)
    c70["profile_name"] = c70["District"].str.strip()
    c70 = c70.rename(columns={"fy": "fiscal_year", "Org4Code": "dese_org_code"})
    c70["source_url"] = SOURCE_URLS["c70_profile"]
    c70["source_dated"] = src.dese["c70_profile_date"]
    return c70[["fiscal_year", "dese_org_code", "district", "profile_name", "distfoundenro", "distfoundbudget",
                "distrlc", "c70aid", "rqdnss", "rqdnss2", "actualNSS", "note", "source_url",
                "source_dated"]].sort_values(["district", "fiscal_year"])


def revenue_csv(census, regional, names):
    """Census revenue by the explorer's sources, in dollars, regional districts only (regional: their four-digit
    codes, by is_regional, as the explorer's revenue series are)"""
    reg = by_org(census, regional).copy()
    out = reg[["FY", "DIST_CODE", "NCESID", "NAME"]].rename(columns={"FY": "fiscal_year", "DIST_CODE": "dese_code",
                                                                   "NCESID": "nces_id", "NAME": "census_name"})
    out["dese_code"] = out["dese_code"].astype(int).astype(str).str.zfill(8)  # as DESE writes it
    out.insert(3, "district", out["dese_code"].astype(int).floordiv(10000).map(names))
    for key, items in REVENUE.items():
        formula = "".join(i if k == 0 else (i if i.startswith("-") else f"+{i}") for k, i in enumerate(items))
        out[f"{REVENUE_LABELS[key]} ({formula})"] = sum(
            reg[i.lstrip("-")] * (-1 if i.startswith("-") else 1) for i in items) * 1000
    out["Total revenue (TOTALREV)"] = reg["TOTALREV"] * 1000
    out["in_district_fte"] = reg["FTE"]
    # Census's FY2022 text file is truncated, so that year was read from its Excel file
    out["source_url"] = [SOURCE_URLS["census_f33"].format(fy=fy, yy=fy % 100, ext="xlsx" if fy == 2022 else "txt")
                         for fy in out["fiscal_year"]]
    return out.sort_values(["census_name", "fiscal_year"])


def sped_csvs(src, orgs, names, latest):
    """DESE special education spending and Circuit Breaker payments, in dollars, as DESE published them"""
    meta = src.sped_sources
    sp = by_org(src.sped_raw, orgs)
    sp = sp[sp["FY"].between(EXPLORER_START, latest)]
    sp = sp.rename(columns={"FY": "fiscal_year", "DIST_CODE": "dese_code", "DIST_NAME": "radar_name"})
    sp.insert(2, "district", sp["dese_code"].astype(int).floordiv(10000).map(names))
    sp["source_url"], sp["source_dated"] = meta["sped"]["url"], meta["sped"]["file_date"]
    yield "districts-special-education.csv", sp.sort_values(["district", "fiscal_year"])
    cbs = by_org(src.cb_raw, orgs)
    cbs = cbs[cbs["FY"].between(EXPLORER_START, latest + 1)]
    cbs = cbs.rename(columns={"FY": "fiscal_year_paid", "DIST_CODE": "dese_code", "DIST_NAME": "dese_name",
                              "students": "elig_stu_claim_cnt"})
    cbs.insert(2, "district", cbs["dese_code"].astype(int).floordiv(10000).map(names))
    cbs["source_url"], cbs["source_updated"] = SOURCE_URLS["circuit_breaker"], meta["circuit_breaker"]["last_updated"]
    yield "districts-circuit-breaker.csv", cbs.sort_values(["district", "fiscal_year_paid"])


def budgets_csv(src, orgs, names, latest):
    """DESE's budgeted and actual net school spending, in dollars, as its compliance summaries reported them"""
    bd = src.budgets[src.budgets["lea"].isin(orgs) & src.budgets["fy"].between(EXPLORER_START, latest + 1)]
    bd = bd.rename(columns={"fy": "fiscal_year", "lea": "dese_lea", "district": "dese_name"})
    bd.insert(2, "district", bd["dese_lea"].map(names))
    urls = {v["file"]: v["url"] for v in src.budget_sources["files"].values()}
    bd["budget_source_url"], bd["actual_source_url"] = bd["budget_file"].map(urls), bd["actual_file"].map(urls)
    return bd.drop(columns=["budget_file", "actual_file"]).sort_values(["district", "fiscal_year"])


def students_csv(src, orgs, names, latest):
    """DESE's students by race and ethnicity, gender, and selected groups, as its profiles show them"""
    st = by_org(src.enr_raw, orgs)
    st = st[st["FY"].between(EXPLORER_START, latest + 1)]
    st = st.rename(columns={"FY": "school_year", "DIST_CODE": "dese_code", "DIST_NAME": "dese_name"})
    st.insert(2, "district", st["dese_code"].astype(int).floordiv(10000).map(names))
    st["source_url"], st["source_updated"] = SOURCE_URLS["students"], src.enroll_sources["last_updated"]
    return st.sort_values(["district", "school_year"])


def town_csvs(all_towns, src, latest):
    """(file name, frame) for each town CSV, in the order they're written"""
    keys = {norm(x["name"]): x for x in all_towns}
    yield from finance_csvs(src, keys, latest)
    yield "towns-school-districts.csv", town_districts_csv(all_towns, latest)
    yield from property_csvs(src, keys, latest)
    yield "towns-chapter70-targets.csv", targets_csv(src)
    yield "towns-households.csv", households_csv(src)
    yield "towns-demographics.csv", demographics_csv(src)
    yield from fiscal_csvs(src, set(keys))


def finance_csvs(src, keys, latest):
    """DLS Schedule A spending and revenue, one column per category, with the population each year's per-resident
    figures divide by"""
    dls = src.dls[src.dls["FY"].between(EXPLORER_START, latest) & src.dls["Municipality"].map(norm).isin(keys)]
    pop = src.pop.set_index(["key", "FY"])
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
        frame["source_url"] = SOURCE_URLS["dls_schedule_a"]
        frame["population_source_url"] = [SOURCE_URLS["census_population"] + pop.at[(x, fy), "File"] for x, fy in zip(k, w["FY"])]
        yield name, frame.sort_values(["town", "fiscal_year"])


def town_districts_csv(all_towns, latest):
    """Each town's school districts and what it must contribute to each"""
    rows = []
    for x in all_towns:
        for d in x.get("districts") or []:
            rows.append({"town": x["name"], "county": x["county"], "school_district": d["name"],
                         "district_kind": {"town": "town district"}.get(d["kind"], d["kind"]),
                         "in_explorer": "yes" if d["id"] else "no",
                         f"fy{latest}_required_local_contribution": d["contribution"],
                         "source_url": SOURCE_URLS["c70_profile"] if d["kind"] == "town" else SOURCE_URLS["c70_regional"].format(fy=latest)})
    return pd.DataFrame(rows)


def property_csvs(src, keys, latest):
    """Property: the LA-4 as DLS publishes it, one column per category, with the year's single-family tax bill;
    EQV for every year DLS lists; levy capacity"""
    la4 = src.la4[src.la4["FY"].between(EXPLORER_START, latest)]
    order = [c for cats in PROPERTY_TYPES.values() for c in cats] + [LA4_EXEMPT]
    wide = []
    for measure in ("Parcels", "Value"):
        w = la4.pivot_table(index=["FY", "DOR_CODE", "Municipality"], columns="Category", values=measure, aggfunc="sum")
        wide.append(w[[c for c in order if c in w.columns]].add_prefix(f"{measure.lower()}: "))
    frame = pd.concat(wide, axis=1).reset_index()
    frame.insert(3, "county", [keys[norm(m)]["county"] for m in frame["Municipality"]])
    frame = frame.rename(columns={"FY": "fiscal_year", "DOR_CODE": "dor_code", "Municipality": "town"})
    frame["taxable_value"] = frame[[f"value: {c}" for c in order[:-1] if f"value: {c}" in frame.columns]].sum(axis=1)
    bills = src.bills.rename(columns={"FY": "fiscal_year", "DOR_CODE": "dor_code"})
    bills = bills.drop(columns=["Municipality"]).rename(columns=lambda c: c if c in ("fiscal_year", "dor_code")
                                                         else f"single-family tax bill table: {c}")
    frame = frame.merge(bills, on=["fiscal_year", "dor_code"], how="left")
    frame["source_url"] = SOURCE_URLS["dls_la4"]
    frame["tax_bill_source_url"] = SOURCE_URLS["dls_sf_bill"]
    yield "towns-property.csv", frame.sort_values(["town", "fiscal_year"])
    eqv = src.eqv.rename(columns={"Year": "valuation_year", "DOR_CODE": "dor_code", "Municipality": "town",
                                  "County": "county", "EQV": "equalized_valuation"})
    eqv["valuation_date"] = "January 1, " + eqv["valuation_year"].astype(str)
    eqv["source_url"] = SOURCE_URLS["dls_eqv"]
    yield "towns-eqv.csv", eqv[["valuation_year", "valuation_date", "dor_code", "town", "county", "equalized_valuation",
                                "source_url"]].sort_values(["town", "valuation_year"])
    levy = src.levy[src.levy["FY"].between(EXPLORER_START, latest)].rename(
        columns={"FY": "fiscal_year", "DOR_CODE": "dor_code", "Municipality": "town"})
    levy["source_url"] = SOURCE_URLS["dls_levy"]
    yield "towns-levy.csv", levy.sort_values(["town", "fiscal_year"])


def targets_csv(src):
    """Chapter 70 targets with and without the 82.5% cap (cap_data's method)"""
    tw = src.tw.copy()
    # Each town's yield times the year's factor, as cap.json figures it (nocap_factor), in floating point: multiplying
    # the yield by the targets' total first, in 64-bit integers, overflowed for Boston (-$407,169,082 for FY2026; #237)
    factor = tw["fy"].map({fy: nocap_factor(g["target"], g["cey"], g["foundation"]) for fy, g in tw.groupby("fy")})
    tw["target_without_cap"] = (tw["cey"].astype(float) * factor).round().where(tw["foundation"] > 0, 0)
    tw["change_without_cap"] = tw["target_without_cap"] - tw["target"]
    tw["without_cap_method"] = ("computed here: combined effort yield times (statewide targets / statewide yields), "
                                "so targets still total 59% of foundation budgets")
    tw = tw.rename(columns={"fy": "fiscal_year", "cey": "combined_effort_yield", "foundation": "foundation_budget",
                            "max_local": "maximum_local_contribution", "target": "target_local_contribution",
                            "rlc_prior": "prior_year_required_contribution", "required": "required_contribution"})
    return tw.sort_values(["fiscal_year", "town"])


def town_of_fips(src):
    """Each Census county subdivision code's town name, as the population file spells it"""
    return src.pop.drop_duplicates("COUSUB_FIPS").set_index("COUSUB_FIPS")["Town"]


def households_csv(src):
    hh = src.hh_raw.copy()
    hh.insert(2, "town", hh["COUSUB_FIPS"].map(town_of_fips(src)))
    hh = hh.rename(columns={"Vintage": "acs_five_year_ending", "COUSUB_FIPS": "census_cousub_fips",
                            "Households": "households", "Source": "source_url"})
    hh.insert(1, "acs_five_year_span", (hh["acs_five_year_ending"] - 4).astype(str) + "-" + hh["acs_five_year_ending"].astype(str))
    return hh.sort_values(["town", "acs_five_year_ending"])


def demographics_csv(src):
    dm = src.demo_raw.copy()
    dm.insert(2, "town", dm["COUSUB_FIPS"].map(town_of_fips(src)))
    dm.insert(1, "acs_five_year_span", (dm["Vintage"] - 4).astype(str) + "-" + dm["Vintage"].astype(str))
    dm["tables"] = "B01001 (sex, age), B03002 (race, Hispanic or Latino origin), B18101 (disability, 2012 releases on)"
    dm["source_url"] = SOURCE_URLS["census_acs"]
    return dm.rename(columns={"Vintage": "acs_five_year_ending", "COUSUB_FIPS": "census_cousub_fips"}).sort_values(
        ["town", "acs_five_year_ending"])


def fiscal_csvs(src, keys):
    """DLS reserves, debt, ratings, votes, capital projects and labor force, DLS's own columns. A file whose
    source dls_fiscal_sources.json doesn't name is left out."""
    meta = src.fiscal_sources
    for path, key, name in [(FREE_CASH_CSV, "free_cash", "towns-free-cash.csv"),
                            (STAB_CSV, "stabilization", "towns-stabilization.csv"),
                            (DEBT_CSV, "long_term_debt", "towns-debt.csv"),
                            (AUTH_DEBT_CSV, "authorized_debt", "towns-authorized-debt.csv"),
                            (RATINGS_CSV, "bond_ratings", "towns-bond-ratings.csv"),
                            (VOTES_CSV, "votes: Override or underride", "towns-prop25-votes.csv"),
                            (CAPITAL_CSV, "capital_projects", "towns-capital-projects.csv"),
                            (LABOR_CSV, "labor_force", "towns-labor-force.csv"),
                            (PROOF_CSV, "free_cash_proof", "towns-free-cash-proof.csv")]:
        if key not in meta:
            continue
        f = src.fiscal_raw[path]
        town_col = "Name" if "Name" in f.columns else "Municipality"
        f = f[f[town_col].map(norm).isin(keys)]
        f["source_url"] = (f["Vote"].map(lambda k: meta[f"votes: {k}"]["source"]) if "Vote" in f.columns
                           else meta[key]["source"])
        f["data_current_as_of"] = meta[key]["data_current_as_of"]
        yield name, f


def cpi_csv(src, latest):
    cpi = src.cpi[src.cpi["fiscal_year"].between(EXPLORER_START, latest)][["fiscal_year", "cpi_u_northeast"]]
    return cpi.assign(average_of="July-June monthly CPI-U Northeast, not seasonally adjusted", source_url=SOURCE_URLS["bls_cpi"])


def write_json(path: Path, content) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, separators=(",", ":")))
    return path.stat().st_size


def read_sources():
    """Every file the export reads, but load_data's and explorer_history's day-program files, each read once, with
    the shapes several steps share built once: town keys, population, counties, households and residents by
    town, the regional members (latest year) and each regional district's towns, property, fiscal records, land
    areas (#237). Each step read its own: POP_CSV seven times and the population dict four, MEMBERS_CSV five
    times (two without the latest-year filter, harmless only while the file holds one year), load_fiscal,
    load_property and land_areas more than once. The frames are shared, as read: a step copies one before
    changing it (pandas' copy-on-write keeps a filtered frame's changes its own)."""
    s = SimpleNamespace()
    s.func = pd.read_csv(FUNC_CSV, dtype={"DIST_CODE": str})
    s.exp = pd.read_csv(EXP_CSV, dtype={"DIST_CODE": str})
    s.c70 = pd.read_csv(C70_CSV, dtype={"Org4Code": str})
    s.dese = json.loads(DESE_SOURCES.read_text())
    s.cpi = pd.read_csv(CPI_CSV)
    s.census = pd.read_csv(CENSUS_CSV, dtype={"Org4Code": str, "NCESID": str})
    # As read for the CSVs (DESE's eight-digit codes as text), and with the codes as numbers for the explorer
    s.sped_raw = pd.read_csv(SPED_CSV, dtype={"DIST_CODE": str})
    s.cb_raw = pd.read_csv(CB_CSV, dtype={"DIST_CODE": str})
    s.enr_raw = pd.read_csv(ENROLL_CSV, dtype={"DIST_CODE": str})
    s.sped, s.cb, s.enr = (f.astype({"DIST_CODE": int}) for f in (s.sped_raw, s.cb_raw, s.enr_raw))
    s.budgets = pd.read_csv(BUDGET_CSV, dtype={"lea": int})
    s.sped_sources, s.budget_sources, s.enroll_sources, s.fiscal_sources = (
        json.loads(p.read_text()) for p in (SPED_SOURCES, BUDGET_SOURCES, ENROLL_SOURCES, FISCAL_SOURCES))
    s.dls = pd.read_csv(DLS_CSV, dtype={"DOR_CODE": str})
    s.pop = pd.read_csv(POP_CSV, dtype={"COUSUB_FIPS": str}).assign(key=lambda d: d["Town"].map(norm))
    s.population = {(k, fy): p for k, fy, p in zip(s.pop["key"], s.pop["FY"], s.pop["Population"])}
    s.county = s.pop.sort_values("FY").groupby("key")["County"].last().to_dict()
    members = pd.read_csv(MEMBERS_CSV, dtype={"Org4Code": str})
    s.members = members[members["FY"] == members["FY"].max()].assign(key=lambda d: d["Town"].map(norm))
    s.towns_of = {}  # a regional district's member towns, by its four-digit code
    for o, k in zip(s.members["Org4Code"].astype(int), s.members["key"]):
        s.towns_of.setdefault(o, []).append(k)
    s.tw = pd.read_csv(C70_TOWNWIDE_CSV)
    s.c70_townwide_sources = json.loads(C70_TOWNWIDE_SOURCES.read_text())
    s.c70_history = pd.read_csv(C70_HISTORY_CSV)
    s.la4, s.eqv, s.bills = (pd.read_csv(p, dtype={"DOR_CODE": str}) for p in (LA4_CSV, EQV_CSV, SF_BILL_CSV))
    s.fiscal_raw = {p: pd.read_csv(p, dtype={"DOR_CODE": str}) for p in FISCAL_CSVS}
    s.levy = s.fiscal_raw[LEVY_CSV]
    s.prop = load_property(s.la4, s.eqv, s.bills, s.levy)
    s.fiscal = load_fiscal(s.fiscal_raw)
    s.hh_raw = pd.read_csv(HH_CSV, dtype={"COUSUB_FIPS": str})
    s.households = load_households(s.pop, s.hh_raw)
    s.demo_raw = pd.read_csv(DEMO_CSV, dtype={"COUSUB_FIPS": str})
    s.demo = load_demographics(s.pop, s.demo_raw)
    s.geo = json.loads(BOUNDARIES_JSON.read_text())
    s.area = land_areas(s.geo)
    return s


def census_with_fte(census, hub):
    """The Census's rows with each district's in-district FTE that year, which its revenue per pupil divides by"""
    fte = hub.df[(hub.cat == "student enrollment") & (hub.subcat == IN_DISTRICT_FTE_KEY)]
    fte = fte.rename(columns={"YEAR": "FY", "IND_VALUE": "FTE"})[["DIST_CODE", "FY", "FTE"]]
    return census.merge(fte.astype({"DIST_CODE": int, "FY": int}), on=["DIST_CODE", "FY"], how="left")


def check_inputs():
    """Stop, before anything is read, if an input is missing: every one is committed, and the export once wrote
    what it could without one, a WARN in its log and exit 0, so a site could ship without a panel (#237)."""
    missing = [str(p) for p in INPUTS if not p.exists()]
    if missing:
        sys.exit("export_explorer_data.py: inputs missing (each is committed with the repository; restore it from git):\n  "
                 + "\n  ".join(missing))


def load(force_recompute=False):
    """(the Hub, read_sources(), the cohort definitions). load_data's frames are the Hub's, with FY2005-FY2008
    stacked under its rows; the cohort definitions are the ones load_data set (get_cohort_definitions returns them
    from its cache, keyed by the frame), taken by name rather than through school_shared's globals (#237)."""
    df, reg, c70 = load_data(force_recompute=force_recompute)
    defs = get_cohort_definitions(df, reg)
    src = read_sources()
    hub = Hub(with_functional(df, src.func), reg, c70)
    src.census = census_with_fte(src.census, hub)
    return hub, src, defs


def district_list(hub, src, defs):
    """(the explorer's districts, the towns' income fifths). Each district (explorer_districts) gains its id,
    whether it is a Gateway City's, its community type (district_types) and its income fifth, by its towns'
    combined income per resident; later steps read all four."""
    dists = explorer_districts(hub, defs)
    gateway = {norm(c) for c in GATEWAY_CITIES}
    for d in dists:
        d["id"] = slug(d["name"])
        d["gateway"] = d["type"] == "traditional" and not is_regional(d) and norm(d["name"]) in gateway
    if sum(d["gateway"] for d in dists) != len(GATEWAY_CITIES):
        raise ValueError(f"Expected {len(GATEWAY_CITIES)} Gateway City districts: "
                         f"{sorted(c for c in gateway if not any(norm(d['name']) == c for d in dists))} missing")
    if len({d["id"] for d in dists}) != len(dists):
        raise ValueError("Two districts share an id")
    print(f"  {len(dists)} districts: " + ", ".join(f"{k} {sum(d['cohort'] == k for d in dists)}" for k in COHORT_ORDER))
    # Community types: rural (and DESE's three rural tiers), suburban and other, Boston and the Gateway Cities
    types = district_types(dists, hub.c70, src.area, src.towns_of)
    for d in dists:
        d["community"] = types.get(d["id"])
    # Income fifths: towns by income per resident, and districts by their towns' combined income per resident
    incomes = town_incomes(src.tw, src.population)
    district_income = {}
    for i, ks in district_members(dists, src.towns_of).items():
        if ks and all(k in incomes for k in ks):
            district_income[i] = (sum(incomes[k][0] for k in ks), sum(incomes[k][1] for k in ks), incomes[ks[0]][2])
    district_fifths = income_fifths(district_income)
    for d in dists:
        d["income"] = district_fifths.get(d["id"])
    return dists, income_fifths(incomes)


class Group(NamedTuple):
    """A comparison group, before its average is figured: the average's id and name, its scope (state, region,
    cohort, region-cohort, group, fifth, type; for towns peer, region-peer, schools), the size group it averages
    (a district cohort, or GATEWAY; a town population group's id), its region, its members (districts or town
    keys), and the averages its own compares with. outlier is for the map's groups within a region alone (Tim,
    #219, #221): None for the explorer's own groups, else whether too few share it for a comparison. Groups were
    7-tuples, with the map's flag riding in cmp until the averages popped it (#237)."""
    id: str
    name: str
    scope: str
    size: str | None
    region: str | None
    members: list
    cmp: dict
    outlier: bool | None = None


def within_regions(size, scope, members_of, gid, base, cmp_of, comparable=True):
    """The groups of one size within each region, for districts and towns alike: where a region has at least
    MIN_AVERAGE_MEMBERS of the size, a comparison group (named base(rl) + " average", comparing with cmp_of(rk));
    where it has fewer, or the size has no regional comparison (comparable False: the very large districts), a group
    for the map alone, named base(rl), a regional outlier if too few share it (Tim, #219, #221). members_of(rk): the
    region's members of the size; gid(rk): the group's id."""
    out = []
    for rk, rl in REGIONS.items():
        ms = members_of(rk)
        if comparable and len(ms) >= MIN_AVERAGE_MEMBERS:
            out.append(Group(gid(rk), f"{base(rl)} average", scope, size, rk, ms, cmp_of(rk)))
        elif ms:
            out.append(Group(gid(rk), base(rl), scope, size, rk, ms, {}, outlier=len(ms) < MIN_AVERAGE_MEMBERS))
    return out


def shared_cmp(g, gateway, state_all):
    """A group's comparisons, with the two every average offers but itself: the Gateway Cities' and the
    statewide average of all"""
    return {**g.cmp, "gateway": gateway if g.id != gateway else None, "state_all": state_all if g.id != state_all else None}


def district_groups(dists, defs):
    """(the explorer's comparison groups, the map's groups within a region it has none of), as Groups. Region
    averages are traditional districts, less the very large, as the report kept Springfield out of its Western MA
    average. Two statewide averages: every traditional district, the default (Tim, #242), and the same without the
    very large districts. Vocational and charter schools stand apart from both. The map's enrollment groups within a
    region (Tim, #221): every size in every region, the very large included, where the explorer has no comparison
    group for it; one with fewer than MIN_AVERAGE_MEMBERS districts is a regional outlier. Only the map has them: the
    explorer compares these districts with the statewide group of their size."""
    traditional = [d for d in dists if d["type"] == "traditional"]
    pooled = [d for d in traditional if d["cohort"] != "XLARGE"]
    state = {"state": "statewide"}
    groups = [Group("statewide", f"Statewide average, without the {len(traditional) - len(pooled)} very large districts",
                    "state", None, None, pooled, {}),
              Group("statewide-all", "Statewide average", "state", None, None, traditional, state)]
    for rk, rl in REGIONS.items():
        groups.append(Group(f"region-{slug(rk)}", f"{rl} average", "region", None, rk,
                            [d for d in pooled if d["region"] == rk], state))
    for k in COHORT_ORDER:
        groups.append(Group(f"cohort-{k.lower()}", cohort_name(k, defs), "cohort", k, None,
                            [d for d in dists if d["cohort"] == k], state))
    groups.append(Group("cohort-gateway", "Gateway City districts average", "group", "GATEWAY", None,
                        [d for d in dists if d["gateway"]], state))
    regional = [g for k in SIZE_COHORTS for g in within_regions(
        k, "region-cohort", lambda rk: [d for d in dists if d["cohort"] == k and d["region"] == rk],
        lambda rk: f"cohort-{k.lower()}-{slug(rk)}", lambda rl: f"{rl} {cohort_short(k, defs)} enrollment group",
        lambda rk: {"cohort_state": f"cohort-{k.lower()}", "region": f"region-{slug(rk)}", "state": "statewide"},
        comparable=k != "XLARGE")]
    groups += [g for g in regional if g.outlier is None]
    community = lambda d, field: (d["community"] or {}).get(field)  # noqa: E731
    for k, word in FIFTH_NAMES.items():
        groups.append(Group(f"fifth-{k}", f"Districts in the {word} income fifth average", "fifth", None, None,
                            [d for d in dists if (d["income"] or {}).get("fifth") == k], state))
    for t, name in (("core", "Inner Core districts average"),
                    ("gateway", "Gateway City districts outside the Inner Core average"),
                    ("suburban", "Suburban and other districts average"), ("rural", "Rural districts average")):
        groups.append(Group(f"type-{t}", name, "type", None, None, [d for d in dists if community(d, "type") == t], state))
    for tier, words in zip((1, 2, 3), ("11 or fewer", "over 11 to 21", "over 21 to 35")):
        groups.append(Group(f"type-rural-{tier}", f"Rural districts, {words} students per square mile, average", "type",
                            None, None, [d for d in dists if community(d, "tier") == tier],
                            {"type": "type-rural", "state": "statewide"}))
    return groups, [g for g in regional if g.outlier is not None]


def district_average(g, hub, src):
    """A district Group's average: spending, enrollment and Chapter 70 pooled over its members (Hub.aggregate),
    revenue where enough of them are regional, and the rest pooled by DESE code"""
    members = g.members
    keys = [d["key"] for d in members]
    codes = [d["code"] for d in members]
    regional = [d["code"] for d in members if is_regional(d)]
    rev = {"rev_members": len(regional)}
    if len(regional) >= MIN_REVENUE_MEMBERS:
        rev |= revenue_fields(src.census, regional, True)
    return {"id": g.id, "name": g.name, "kind": "average", "scope": g.scope, "cohort": g.size,
            **({"outlier": True} if g.outlier else {}),
            "region": g.region, "members": len(members),
            "member_ids": [d["id"] for d in sorted(members, key=lambda d: d["name"])],
            "cmp": shared_cmp(g, "cohort-gateway", "statewide-all"),
            "chunk": slug(g.region) if g.region else "statewide", **pack(*hub.aggregate(keys)),
            "ood": hub.ood(keys), **rev,
            **sped_fields(src.sped, src.cb, src.pupils, codes),
            **budget_fields(src.budgets, hub.c70, codes),
            **student_fields(src.enr, codes)}


def district_entity(d, hub, src, have):
    """A district's full record; have: the ids of the explorer's groups, for its comparisons"""
    k, rk = d["cohort"], d["region"]
    regional = is_regional(d)
    rc = f"cohort-{k.lower()}-{slug(rk)}" if rk else None
    com = d["community"]
    return {"id": d["id"], "name": d["name"], "kind": "district", "type": d["type"], "cohort": k,
            "region": rk, "regional": regional, "fte": round(d["fte"], 1), "gateway": d["gateway"],
            "cmp": {"cohort": rc if rc in have else None, "cohort_state": f"cohort-{k.lower()}",
                    "region": f"region-{slug(rk)}" if rk else None, "state": "statewide",
                    "gateway": "cohort-gateway", "state_all": "statewide-all",
                    "type": f"type-{com['type']}" if com else None,
                    "fifth": f"fifth-{d['income']['fifth']}" if d["income"] else None},
            **({"community": com} if com else {}),
            **({"income": d["income"]} if d["income"] else {}),
            "chunk": chunk_of(rk, d["type"]),
            **pack(*hub.district(d["name"])), "ood": hub.ood([d["key"]]),
            **revenue_fields(src.census, [d["code"]], regional),
            **sped_fields(src.sped, src.cb, src.pupils, [d["code"]]),
            **budget_fields(src.budgets, hub.c70, [d["code"]]),
            **student_fields(src.enr, [d["code"]])}


def explorer_index(averages, full, former, hist_meta, defs, src):
    """explorer.json: the district view's meta, and light lists of its averages and districts for the menus"""
    idx = src.cpi.set_index("fiscal_year")["cpi_u_northeast"]
    census, sped, cb, budgets, enr = src.census, src.sped, src.cb, src.budgets, src.enr
    cohorts = {k: {"label": defs[k]["label"], "short": defs[k]["short_label"],
                   "range": [r if r != float("inf") else None for r in defs[k]["range"]]} for k in SIZE_COHORTS[:-1]}
    cohorts |= {k: {"label": COHORT_TEXT[k][0], "short": COHORT_TEXT[k][1],
                    "range": [10001, None] if k == "XLARGE" else None} for k in COHORT_TEXT}
    return {
        "meta": {
            "title": "Massachusetts school spending",
            "years": YEARS, "latest": LATEST_YEAR, "start": EXPLORER_START,
            "built": dt.date.today().isoformat(),
            "categories": CANON_CATS_BOTTOM_TO_TOP,
            "cohorts": {k: cohorts[k] for k in COHORT_ORDER + ["GATEWAY"]},
            "regions": REGIONS,
            "cpi": {str(y): float(idx[y]) for y in HIST_YEARS + YEARS if y in idx.index},
            "history": hist_meta,
            "revenue": {"first": int(census["FY"].min()), "latest": int(census["FY"].max()),
                        "min_members": MIN_REVENUE_MEMBERS},
            "sped": {"first": int(sped["FY"].min()), "latest": int(sped["FY"].max()),
                     "cb_first": int(cb["FY"].min()), "cb_latest": int(cb["FY"].max()),
                     **{k: v for k, v in src.sped_sources.items() if k == "downloaded"}},
            "budgets": {"first": int(budgets.dropna(subset=["budgeted_nss"])["fy"].min()),
                        "latest": int(budgets.dropna(subset=["budgeted_nss"])["fy"].max()),
                        **{k: v for k, v in src.budget_sources.items() if k in ("downloaded", "not_found")}},
            "students": {"first": int(enr["FY"].min()), "latest": int(enr["FY"].max()),
                         **{k: v for k, v in src.enroll_sources.items() if k in ("downloaded", "last_updated")}},
            "sources": {
                "expenditures": "DESE District Expenditures by Spending Category (E2C Hub er3w-dyti)",
                "chapter70": "DESE Chapter 70 District Profiles",
                "cpi": "BLS CPI-U Northeast (CUUR0100SA0), July-June fiscal-year averages",
                "revenue": "Census Bureau Annual Survey of School System Finances (F-33), Massachusetts",
                "sped": "DESE Special Education Direct Expenditure Trends (RADAR)",
                "circuit_breaker": "DESE Special Education Circuit Breaker Reimbursements (E2C Hub ab34-d3ma)",
                "budgets": "DESE Net School Spending compliance summaries (budgeted and actual NSS)",
                "students": "DESE Enrollment: Grade, Race/Ethnicity, Gender, and Selected Populations (E2C Hub t8td-gens)",
            },
        },
        # A light list for the menus; each entity's data is in its chunk's file
        "averages": [{"id": a["id"], "name": a["name"], "scope": a["scope"], "cohort": a["cohort"],
                      "region": a["region"], "chunk": a["chunk"]} for a in averages],
        "districts": [{"id": e["id"], "name": e["name"], "type": e["type"], "cohort": e["cohort"], "region": e["region"],
                       "regional": e["regional"], "fte": e["fte"], "gateway": e["gateway"], "chunk": e["chunk"]} for e in full],
        # Districts the older files have and the Hub doesn't, shown with the earlier years
        "former": former,
    }


@contextmanager
def staged(*targets):
    """New directories beside the targets, which replace them when the block finishes: each file the export
    writes ships with all the others or not at all (#237). It once wrote into the site as it went, deleting only
    the chunk files first, so a run that failed partway, or lacked an input, left new files beside old ones. The
    export owns these directories: anything else in them goes too. On an error the new ones are removed and the
    site stays as it was."""
    umask = os.umask(0)
    os.umask(umask)
    new = []
    try:
        for t in targets:
            t.parent.mkdir(parents=True, exist_ok=True)
            new.append(Path(tempfile.mkdtemp(prefix=f".{t.name}.new-", dir=t.parent)))
            os.chmod(new[-1], 0o777 & ~umask)  # mkdtemp's 0700, as a directory made the usual way
        yield new
    except BaseException:
        for n in new:
            shutil.rmtree(n, ignore_errors=True)
        raise
    for t, n in zip(targets, new):
        old = n.with_name(n.name.replace(".new-", ".old-"))
        if t.exists():
            t.rename(old)
        n.rename(t)
        if old.exists():
            shutil.rmtree(old)


def write_site(site, out):
    """Every file the explorer reads, into the site root: the JSON to src/data/massppe/ and the CSVs to
    public/muni-spending/data/, staged beside them and moved into place at the end. out: what main built."""
    json_dir = site / "src" / "data" / "massppe"
    csv_dir = site / "public" / "muni-spending" / "data"
    with staged(json_dir, csv_dir) as (json_new, csv_new):
        sizes = {}

        def put(name, content):
            sizes[name] = write_json(json_new / name, content)
        put("explorer.json", out.index)
        for chunk in sorted({e["chunk"] for e in out.full + out.averages}):
            put(f"districts/{chunk}.json", {"entities": [shipped(e) for e in out.averages + out.full if e["chunk"] == chunk]})
        for chunk, xs in sorted(out.hist_chunks.items()):
            put(f"districts-history/{chunk}.json", {"entities": xs})
        put("towns.json", out.towns_index)
        put("votes.json", out.votes)  # every town's Prop 2 1/2 votes, loaded when a town's list opens
        for chunk, xs in sorted(out.town_chunks.items()):
            put(f"towns/{chunk}.json", {"towns": [shipped(x) for x in xs]})
        put("commonwealth.json", out.common)
        put("cap.json", out.cap)
        # The years behind the map's measures, a file of their own: the map draws without them
        put("map-years.json", out.map_years)
        put("map.json", out.map)
        print(f"{json_dir}: {len(out.full)} districts, {len(out.averages)} district averages, {len(out.all_towns)} towns, "
              f"{len(out.towns_index['averages'])} town averages")
        print("  " + "; ".join(f"{k} {v / 1024:.0f} KB" for k, v in sizes.items()))
        print(f"{csv_dir}: " + "; ".join(write_csvs(csv_new, out.dists_by_name, out.all_towns, out.src, LATEST_YEAR)))
        write_groups(csv_new, json_new, out.averages, out.full, out.town_list)


def main(site: Path, force_recompute=False):
    """Load, group, build the entities and the files' content, then write them all. The steps run in this order
    because each reads what the one before added: district_list adds ids, community types and income fifths to the
    districts; the towns' links add each district's towns to its entity, which the earlier years read."""
    check_inputs()
    hub, src, defs = load(force_recompute)
    src.pupils = sped_pupils(hub)
    dists, town_fifths = district_list(hub, src, defs)
    groups, map_only = district_groups(dists, defs)
    have = {g.id for g in groups}
    averages = [district_average(g, hub, src) for g in groups]
    map_averages = [district_average(g, hub, src) for g in map_only]
    full = [district_entity(d, hub, src, have) for d in sorted(dists, key=lambda d: d["name"])]

    towns_index, town_chunks, all_towns, links, town_outliers = town_data(dists, LATEST_YEAR, src, town_fifths)
    for e in full:
        e.update(links.get(e["id"], {}))
    # The earlier years, for the "Since 1990" switch: their own files, loaded only when it's on
    hist_chunks, hist_meta, former = history(hub, dists, groups, full, all_towns, src.func)

    out = SimpleNamespace(src=src, full=full, averages=averages, all_towns=all_towns, towns_index=towns_index,
                          town_chunks=town_chunks, hist_chunks=hist_chunks,
                          dists_by_name=sorted(dists, key=lambda d: d["name"]),
                          town_list=[t for xs in town_chunks.values() for t in xs])
    out.index = explorer_index(averages, full, former, hist_meta, defs, src)
    out.votes = town_votes(src.fiscal, {norm(x["name"]): x["name"] for x in all_towns if x["kind"] == "town"})
    out.common = commonwealth_data(dists, full, src)
    print(f"  commonwealth.json: {len(out.common['districts'])} districts")
    out.cap = cap_data(all_towns, src)
    print("  cap.json: FY" + ", FY".join(out.cap["years"]))
    idx = src.cpi.set_index("fiscal_year")["cpi_u_northeast"]
    out.map = map_data(full, [t for t in all_towns if t["kind"] == "town"], out.cap,
                       {int(y): float(idx[y]) for y in YEARS if y in idx.index}, {d["id"]: d["code"] // 10000 for d in dists},
                       src, [t for t in out.town_list if t["kind"] != "town"] + town_outliers, averages + map_averages)
    out.map_years = out.map.pop("series")
    print(f"  map.json: {len(out.map['towns'])} towns, {len(out.map['districts'])} districts")
    write_site(site, out)


def shipped(x):
    """A record as the explorer loads it: an average's member list is for the groups page alone"""
    return {k: v for k, v in x.items() if k != "member_ids"}


def write_groups(csv_dir: Path, json_dir: Path, averages, districts, towns):
    """Every average the explorer offers and its members, as the explorer counts them (the Data tab
    gives each group's rule): groups.json for the groups page, which reads it when the site is built,
    and groups.csv beside the other CSVs, one row per member."""
    name_of = {e["id"]: e["name"] for e in districts} | {t["id"]: t["name"] for t in towns if t["kind"] == "town"}
    views = (("districts", "school districts", averages), ("towns", "towns", [t for t in towns if t["kind"] == "average"]))
    write_json(json_dir / "groups.json", {key: [{"id": a["id"], "name": a["name"], "scope": a["scope"],
                                                 "members": [{"id": m, "name": name_of[m]} for m in a["member_ids"]]}
                                                for a in avgs] for key, _, avgs in views})
    rows = [{"view": view, "group_id": a["id"], "group": a["name"], "scope": a["scope"], "members": a["members"],
             "member_id": m, "member": name_of[m]}
            for _, view, avgs in views for a in avgs for m in a["member_ids"]]
    frame = pd.DataFrame(rows)
    frame.to_csv(csv_dir / "groups.csv", index=False)
    print(f"  groups.json, groups.csv: {frame['group_id'].nunique()} groups, {len(frame):,} members")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Export the Municipal spending explorer's data into a site root.")
    ap.add_argument("site", nargs="?", type=Path, default=Path("output/site"),
                    help="the site root (a copy of github.com/timshores/mywebsite, or a scratch directory)")
    ap.add_argument("--force-recompute", action="store_true",
                    help="rebuild load_data's cache (data/cache) from its sources, even if it is newer than they are")
    args = ap.parse_args()
    main(args.site, args.force_recompute)
