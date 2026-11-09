"""
Download each Massachusetts municipality's reserves, budget, debt, Proposition 2 1/2 votes,
capital projects and labor force from the DLS Municipal Databank, for the explorer's town view.

    data/dls_free_cash.csv            certified free cash as of July 1 and the prior year's
                                      operating budget, FY2002 on
    data/dls_stabilization.csv        stabilization and special-purpose stabilization fund
                                      balances at June 30 and that year's operating budget
    data/dls_long_term_debt.csv       outstanding debt, EQV, population, general fund debt
                                      service and the total budget, FY2003 on
    data/dls_authorized_debt.csv      debt authorized, issued, retired or rescinded, and
                                      authorized but unissued, FY2003 on
    data/dls_bond_ratings.csv         Moody's and S&P ratings by year, 2001 on
    data/dls_prop25_votes.csv         override, underride, debt exclusion, capital exclusion
                                      and stabilization fund votes, with results and amounts
    data/dls_debt_exclusion_levy.csv  each debt exclusion's net excludable debt by fiscal year
    data/dls_capital_projects.csv     Schedule A Part 4: capital projects fund revenues and
                                      expenditures by purpose, FY2002 on
    data/dls_labor_force.csv          labor force, employed, unemployed, unemployment rate,
                                      by calendar year, 1990 on
    data/dls_free_cash_proof.csv      the Bureau of Accounts' free cash proof: what each year's free
                                      cash was made of (local receipts and state aid above or below
                                      their estimates, appropriations left unspent, the year before's
                                      free cash not spent, adjustments), FY2021 on

Every report is a Logi form of checkbox lists (municipalities, years, and for votes the
department, result and type); this asks for every choice at once and exports the report's
table. Certified free cash is set as of July 1, from the June 30 balance sheet of the year
before: the FY2027 figure is what FY2026 left. DLS divides it by the prior year's operating
budget, and the stabilization balances by the same year's. The operating budget is the tax
rate recap's revenue without enterprise and Community Preservation funds (it matches DLS's
Revenue by Source report's total to the cent).

    python update_dls_fiscal_data.py
"""

import datetime as dt
import io
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

from update_dls_data import BASE, PAUSE, fetch

DATA_DIR = Path("./data")
SOURCES_JSON = DATA_DIR / "dls_fiscal_sources.json"
# name: (report, export table, output file)
REPORTS = {
    "free_cash": ("Dashboard.Cat_1_Reports.CertifiedFreeCashBudget351", "TblCFC_PerBudg", "dls_free_cash.csv"),
    "stabilization": ("Dashboard.Cat_1_Reports.StablPerBudget351", "TblStabl_PerBudg", "dls_stabilization.csv"),
    "long_term_debt": ("Dashboard.Cat_6_Reports.LongTermDebt351", "tblLongTermDebt", "dls_long_term_debt.csv"),
    "authorized_debt": ("Dashboard.Cat_6_Reports.TotalAuthUnisDebt351", "tblTotAuthUniDebt", "dls_authorized_debt.csv"),
    "bond_ratings": ("DLS_Bond_Ratings", "xtblBondRatings", "dls_bond_ratings.csv"),
    "debt_exclusion_levy": ("Votes.Prop2_5.DebtExclusionLevyAmt", "tblDebtExcLevyAmt", "dls_debt_exclusion_levy.csv"),
    "labor_force": ("Dashboard.TrendAnalysisReports.LaborForce", "tblLaborForce", "dls_labor_force.csv"),
    # New growth: the levy limit's yearly increase from new construction and other additions to the tax base, with
    # its residential part, FY2003 on (#248; DLS's New Growth Analysis; the trend report of that name stops at FY2017)
    "new_growth": ("NewGrowth.NewGrowth_dash_v2_test", "tblNewGrowth", "dls_new_growth.csv"),
}
VOTES = {
    "Override or underride": "Votes.Prop2_5.OverrideUnderride",
    "Debt exclusion": "Votes.Prop2_5.DebtExclusionVotes",
    "Capital exclusion": "Votes.Prop2_5.Capital",
    "Stabilization fund": "Votes.Prop2_5.Stabilization",
}
CAPITAL = "ScheduleA.CapitalProjects.CapitalProjects"
PROOF = "BalanceSheet.FreecashProofComp"
# The proof's lines, as DLS heads them, and the short names this file gives them. The lines from revenue_deficits
# to other_adjustments add up to identified free cash, each with its own sign; certified_prior and certified are the
# certifications either side of the year.
PROOF_LINES = {
    "Free Cash Certified Prior Year": "certified_prior",
    "Current Year Calculation": "certified",
    "Revenue Deficits": "revenue_deficits",
    "Excess/Shortfall Local Receipts (CL#6)": "local_receipts",
    "Add Actual Revenue Received but not Estimated (CL#7)": "not_estimated",
    "Excess/Shortfall Cherry Sheet Receipts (CL#8)": "state_aid",
    "Add Unencumbered/Unexpended Appropriations (CL#11)": "unexpended",
    "Subtract Negative Free Cash Certified July 1 Prior Year": "negative_prior",
    "Add Prior Year Free Cash Not Appropriated (CL#12)": "prior_not_appropriated",
    "Add Overlay Surplus Closed (CL#13)": "overlay_surplus",
    "Prior & Current Year Outstanding Receivables Total": "receivables",
    "Net Change in Adjustment to Free Cash": "adjustment_change",
    "Other Adjustments": "other_adjustments",
    "Identified Free Cash July 1,": "identified",
}


def form(report: str) -> tuple[list[tuple[str, str]], str, str | None]:
    """Every choice on the report's form (each checkbox list's values, each select's first option),
    its 'data current as of' date, and the export table the page names."""
    page = fetch(f"{BASE}?rdReport={report}").decode("utf-8", errors="ignore")
    choices = [(n, v) for n, v in re.findall(r'name="(icl[A-Za-z]+)"[^>]*value="([^"]*)"', page)]
    if not choices:
        raise ValueError(f"The {report} form lists no choices; has the Gateway page changed?")
    current = re.search(r"current as of\s*(\d{1,2}/\d{1,2}/\d{4})", re.sub(r"<[^>]+>", " ", page))
    as_of = dt.datetime.strptime(current.group(1), "%m/%d/%Y").date().isoformat() if current else ""
    table = re.search(r"rdExportTableID=([A-Za-z0-9_]+)", page)
    return choices, as_of, table.group(1) if table else None


def export(report: str, table: str, data: list[tuple[str, str]], header: int = 0) -> pd.DataFrame:
    url = (f"{BASE}?rdReport={report}&rdReportFormat=NativeExcel&rdExportTableID={table}"
           "&rdExcelOutputFormat=Excel2007")
    raw = fetch(url, data)
    if not raw.startswith(b"PK"):  # an .xlsx is a zip; anything else is the Gateway's error page
        raise ValueError(f"{report} {table}: the Gateway returned an error page, not a spreadsheet")
    time.sleep(PAUSE)
    return pd.read_excel(io.BytesIO(raw), header=header, dtype=str)


def tidy(df: pd.DataFrame, codes: dict[str, str] | None = None) -> pd.DataFrame:
    """One header row, single-spaced names; rows with a numeric DOR code only (no totals). A report
    that leaves its codes blank (the labor force) takes them from the municipality's name."""
    df = df.rename(columns=lambda c: re.sub(r"\s+", " ", str(c)).strip())
    code = next(c for c in df.columns if c.upper() == "DOR CODE")
    df = df.rename(columns={code: "DOR_CODE"})
    if codes and df["DOR_CODE"].isna().all():
        df["DOR_CODE"] = df["Municipality"].str.strip().map(codes)
        if df["DOR_CODE"].isna().any():
            raise ValueError(f"No DOR code for {sorted(df.loc[df['DOR_CODE'].isna(), 'Municipality'].unique())[:5]}")
    df = df[df["DOR_CODE"].astype(str).str.fullmatch(r"\d+", na=False)].copy()
    df["DOR_CODE"] = df["DOR_CODE"].str.zfill(3)
    for c in df.columns:
        if c in ("DOR_CODE", "Municipality", "Name") or df[c].isna().all():
            continue
        num = pd.to_numeric(df[c].str.replace(r"[$,%]", "", regex=True), errors="coerce")
        if num.notna().sum() >= df[c].notna().sum() * 0.9:  # a numeric column, not text or dates
            df[c] = num
    return df


def free_cash_proof() -> tuple[pd.DataFrame, str]:
    """Every town's free cash proof, one request a town (the form takes one at a time): a row per town and year,
    the year being the fiscal year whose June 30 balance sheet the proof reads."""
    page = fetch(f"{BASE}?rdReport={PROOF}").decode("utf-8", errors="ignore")
    munis = re.findall(r'VALUE="([^"]*)"', re.search(r'NAME="islMuniCCP".*?</SELECT>', page, re.S | re.I).group(0), re.I)
    years = re.findall(r'name="iclYearCCP"[^>]*value="([^"]*)"', page)
    current = re.search(r"current as of\s*(\d{1,2}/\d{1,2}/\d{4})", re.sub(r"<[^>]+>", " ", page))
    as_of = dt.datetime.strptime(current.group(1), "%m/%d/%Y").date().isoformat() if current else ""
    rows = []
    for m in munis:
        raw = export(PROOF, "tblProofCompare", [("islMuniCCP", m)] + [("iclYearCCP", y) for y in years], header=None)
        head = [str(v).strip() for v in raw.iloc[0]]
        for _, r in raw.iloc[1:].iterrows():
            line = re.sub(r"\s+", " ", str(r.iloc[0])).strip()
            if line not in PROOF_LINES:
                continue
            for y, v in zip(head[1:], r.iloc[1:]):
                if y.isdigit():
                    s = str(v).replace(",", "").strip()
                    s = "-" + s[1:-1] if s.startswith("(") and s.endswith(")") else s
                    rows.append({"Municipality": m, "FY": int(y), "line": PROOF_LINES[line], "amount": pd.to_numeric(s, errors="coerce")})
    long = pd.DataFrame(rows)
    wide = long.pivot_table(index=["Municipality", "FY"], columns="line", values="amount", aggfunc="first").reset_index()
    parts = ["revenue_deficits", "local_receipts", "not_estimated", "state_aid", "unexpended", "negative_prior",
             "prior_not_appropriated", "overlay_surplus", "receivables", "adjustment_change", "other_adjustments"]
    off = (wide[parts].sum(axis=1) - wide["identified"]).abs() > 2
    if off.any():
        raise ValueError(f"{int(off.sum())} proofs whose lines don't add to identified free cash, e.g. "
                         f"{wide[off].iloc[0][['Municipality', 'FY']].to_dict()}")
    return wide[["Municipality", "FY", "certified_prior", *parts, "identified", "certified"]], as_of


def main():
    today = dt.date.today()
    # Named reports alone (python update_dls_fiscal_data.py new_growth): fetched, and their sources merged into the
    # file's, leaving every other file as it is
    only = set(sys.argv[1:])
    if only - set(REPORTS):
        raise SystemExit(f"Not a report here: {sorted(only - set(REPORTS))}; choose from {sorted(REPORTS)}")
    sources = json.loads(SOURCES_JSON.read_text()) if only else {"downloaded": today.isoformat()}
    codes = None
    for name, (report, table, out) in REPORTS.items():
        if only and name not in only:
            continue
        choices, as_of, found = form(report)
        if name == "bond_ratings":  # one agency at a time
            df = pd.concat([tidy(export(report, table or found, choices + [("islCompany", a)])) for a in ("Moodys", "S&P")],
                           ignore_index=True)
        else:
            df = tidy(export(report, table or found, choices), codes)
        if name == "free_cash":
            codes = dict(zip(df["Municipality"].str.strip(), df["DOR_CODE"]))
        df.to_csv(DATA_DIR / out, index=False)
        years = [c for c in df.columns if c.lower() in ("fiscal year", "year", "schedule a fiscal year", "fy excluded")]
        span = f", {int(df[years[0]].min())}-{int(df[years[0]].max())}" if years and pd.api.types.is_numeric_dtype(df[years[0]]) else ""
        print(f"  {DATA_DIR / out}: {len(df):,} rows{span}, current as of {as_of}")
        sources[name] = {"source": f"{BASE}?rdReport={report}", "data_current_as_of": as_of}
    if only:
        SOURCES_JSON.write_text(json.dumps(sources, indent=1) + "\n")
        return

    votes = []
    for kind, report in VOTES.items():
        choices, as_of, table = form(report)
        df = tidy(export(report, table, choices))
        df.insert(1, "Vote", kind)
        votes.append(df)
        sources[f"votes: {kind}"] = {"source": f"{BASE}?rdReport={report}", "data_current_as_of": as_of}
    votes = pd.concat(votes, ignore_index=True)
    votes.to_csv(DATA_DIR / "dls_prop25_votes.csv", index=False)
    print(f"  {DATA_DIR / 'dls_prop25_votes.csv'}: {len(votes):,} votes: " +
          ", ".join(f"{k} {n}" for k, n in votes["Vote"].value_counts().items()))

    # Capital projects: one year and amount type at a time
    choices, as_of, table = form(CAPITAL)
    munis = [c for c in choices if c[0] != "iclYear"]
    parts = []
    for year in sorted(v for k, v in choices if k == "iclYear"):
        for amount in ("Expenditures", "Revenues"):
            df = tidy(export(CAPITAL, table, munis + [("iclYear", year), ("islAmountType", amount)]))
            df.insert(1, "Type", amount)
            parts.append(df)
    cap = pd.concat(parts, ignore_index=True)
    cap.to_csv(DATA_DIR / "dls_capital_projects.csv", index=False)
    print(f"  {DATA_DIR / 'dls_capital_projects.csv'}: {len(cap):,} rows, current as of {as_of}")
    sources["capital_projects"] = {"source": f"{BASE}?rdReport={CAPITAL}", "data_current_as_of": as_of}

    proof, as_of = free_cash_proof()
    proof.to_csv(DATA_DIR / "dls_free_cash_proof.csv", index=False)
    print(f"  {DATA_DIR / 'dls_free_cash_proof.csv'}: {len(proof):,} town-years, FY{proof['FY'].min()}-FY{proof['FY'].max()}")
    sources["free_cash_proof"] = {"source": f"{BASE}?rdReport={PROOF}", "data_current_as_of": as_of}

    SOURCES_JSON.write_text(json.dumps(sources, indent=1) + "\n")
    print(f"  {SOURCES_JSON}")


if __name__ == "__main__":
    main()
