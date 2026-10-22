"""
Download the property values behind each Massachusetts municipality's tax base
from the DLS Municipal Databank: parcel counts and assessed values by property
use (the assessors' LA-4), FY2003 onward; the Commissioner of Revenue's
equalized valuations (EQV), 1970 onward; the average single-family tax
bill, 1988 onward; and each town's Proposition 2 1/2 levy limit, levy, and
excess levy capacity, FY2007 onward.

    data/dls_la4_parcels.csv
        DOR_CODE, Municipality, County, FY, Category, Parcels, Value
    data/dls_eqv.csv
        DOR_CODE, Municipality, County, Year, EQV
    data/dls_single_family_tax_bill.csv
        DOR_CODE, Municipality, FY, Single-Family Values, Single-Family Parcels,
        Average Single-Family Value, Single-Family Tax Bill, Single-Family Tax
        Bill as % of Value, DOR Income Per Capita, Average Tax Bill as a % of Income
    data/dls_levy_capacity.csv
        DOR_CODE, Municipality, FY, Levy Limit w/o Debt & Capital Exclusions,
        Maximum Levy Limit, Total Tax Levy, Excess Levy Capacity, Levy Ceiling,
        Override Capacity, Total Assessed Value, ...

The LA-4 is each board of assessors' yearly report of parcels and assessed
values by property use code, as of January 1 before the fiscal year: the FY2025
LA-4 values property as of January 1, 2024. Its categories are DLS's groups of
use codes, such as "Single Family 101" or "Commercial 300-393", and count
personal property accounts (501-508, 550-552) alongside parcels. Its taxable
categories add up to the town's total assessed value; exempt property is a
category of its own, outside that total. A fiscal year appears once a town's
assessors have filed, so the latest years cover only some towns; see the counts
printed at the end.

EQV is the Commissioner's estimate of the full and fair cash value of all
taxable property in each municipality as of January 1 of even-numbered years
(G.L. c. 58, sections 9-10C). The Bureau of Local Assessment divides each
property class's assessed value by its level of assessment, measured from
sales, and adds projected new growth. Chapter 70 applies one statewide
percentage to it in setting each town's target local contribution.

The average single-family tax bill is DLS's measure of what a typical homeowner
pays: single-family assessed value divided by single-family parcels, times the
residential tax rate. DLS sets it beside the Department of Revenue's income per
capita and reports the bill as a percentage of that income.

Under Proposition 2 1/2 a town's levy limit grows 2.5% a year plus new growth,
and voters can raise it with an override or exclude debt from it; the maximum
levy limit includes those exclusions. Excess levy capacity is what a town could
raise but doesn't: the maximum levy limit less the levy it set. The levy
ceiling, 2.5% of total assessed value, caps the levy limit itself.

    python update_dls_property_data.py
"""

import datetime as dt
import io
import json
import re
import time
from pathlib import Path

import pandas as pd

from update_dls_data import BASE, PAUSE, fetch

DATA_DIR = Path("./data")
LA4 = "PropertyTaxInformation.LA4.Parcel_counts_vals"
EQV = "PropertyTaxInformation.EQV.EQV"
SF_BILL = "AverageSingleTaxBill.SingleFamTaxBill_wRange"
LEVY = "Prop2.5.ExcessLevyCapandOverride_10_pres"
LA4_TABLES = {"Parcels": "xtParcels", "Value": "xtVals"}  # measure: the report's table for it
OUT_LA4 = DATA_DIR / "dls_la4_parcels.csv"
OUT_EQV = DATA_DIR / "dls_eqv.csv"
OUT_SF_BILL = DATA_DIR / "dls_single_family_tax_bill.csv"
OUT_LEVY = DATA_DIR / "dls_levy_capacity.csv"
SOURCES_JSON = DATA_DIR / "dls_property_sources.json"
ID_COLS = ["DOR Code", "Municipality", "County"]


def form(report: str) -> tuple[dict[str, list[str]], str]:
    """The report form's municipality, county and year choices, and its 'data current as of' date."""
    page = fetch(f"{BASE}?rdReport={report}").decode("utf-8", errors="ignore")
    fields = {name: re.findall(rf'name="{name}"[^>]*value="([^"]*)"', page) for name in ["iclMuni", "iclCounty", "iclYear"]}
    select = re.search(r'NAME="islYear".*?</SELECT>', page, re.S)  # LA-4: one year at a time
    fields["islYear"] = re.findall(r'VALUE="(\d{4})"', select.group(0)) if select else []
    current = re.search(r"current as of\s*(\d{1,2}/\d{1,2}/\d{4})", re.sub(r"<[^>]+>", " ", page))
    as_of = dt.datetime.strptime(current.group(1), "%m/%d/%Y").date().isoformat() if current else ""
    if not fields["iclMuni"] or not (fields["iclYear"] or fields["islYear"]):
        raise ValueError(f"The {report} form lists no municipalities or years; has the Gateway page changed?")
    return fields, as_of


def export(report: str, table: str, data: list[tuple[str, str]]) -> pd.DataFrame:
    """One of the report's tables as a spreadsheet, without its totals row."""
    raw = fetch(f"{BASE}?rdReport={report}&rdReportFormat=NativeExcel&rdExportTableID={table}"
                "&rdExcelOutputFormat=Excel2007", data)
    if not raw.startswith(b"PK"):  # an .xlsx is a zip; anything else is the Gateway's error page
        raise ValueError(f"{report} {table}: the Gateway returned an error page, not a spreadsheet")
    df = pd.read_excel(io.BytesIO(raw), dtype={"DOR Code": str})
    return df[df["DOR Code"].str.fullmatch(r"\d+", na=False)]


def tidy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"DOR Code": "DOR_CODE"})
    df["DOR_CODE"] = df["DOR_CODE"].str.zfill(3)
    df["Municipality"] = df["Municipality"].str.strip()
    df["County"] = df["County"].str.strip().str.title()
    return df


def la4(fields: dict[str, list[str]]) -> pd.DataFrame:
    """Every year's parcel counts and values by use category, for every municipality."""
    places = [("iclMuni", m) for m in fields["iclMuni"]] + [("iclCounty", c) for c in fields["iclCounty"]]
    frames = []
    for year in sorted(int(y) for y in fields["islYear"]):
        parts = []
        for measure, table in LA4_TABLES.items():
            wide = export(LA4, table, places + [("islYear", str(year))])
            long = wide.melt(id_vars=ID_COLS + ["Fiscal Year"], var_name="Category", value_name=measure)
            long["Category"] = long["Category"].str.replace(r"\s+", " ", regex=True).str.strip()
            parts.append(long.set_index(ID_COLS + ["Fiscal Year", "Category"])[measure].pipe(pd.to_numeric, errors="coerce"))
            time.sleep(PAUSE)
        frames.append(pd.concat(parts, axis=1).reset_index())
        print(f"  FY{year}", end="", flush=True)
    print()
    df = tidy(pd.concat(frames, ignore_index=True).rename(columns={"Fiscal Year": "FY"}))
    df["FY"] = df["FY"].astype(int)
    return df[["DOR_CODE", "Municipality", "County", "FY", "Category", "Parcels", "Value"]]


def eqv(fields: dict[str, list[str]]) -> pd.DataFrame:
    """Every equalized valuation, all municipalities and years in one request."""
    wide = export(EQV, "xtEQV", [("iclMuni", m) for m in fields["iclMuni"]]
                  + [("iclCounty", c) for c in fields["iclCounty"]] + [("iclYear", y) for y in fields["iclYear"]])
    long = wide.melt(id_vars=ID_COLS, var_name="Year", value_name="EQV")
    long["Year"] = long["Year"].astype(str).str.extract(r"(\d{4})", expand=False).astype(int)
    long["EQV"] = pd.to_numeric(long["EQV"], errors="coerce")
    return tidy(long)[["DOR_CODE", "Municipality", "County", "Year", "EQV"]]


def single_family_bills(fields: dict[str, list[str]]) -> pd.DataFrame:
    """Every year's average single-family tax bill, all municipalities in one request."""
    df = export(SF_BILL, "tblSinglefamtaxbill", [("iclMuni", m) for m in fields["iclMuni"]]
                + [("iclYear", y) for y in fields["iclYear"]])
    df = df.rename(columns=lambda c: re.sub(r"\s+", " ", str(c)).strip())
    df = df.rename(columns={"DOR Code": "DOR_CODE", "Fiscal Year": "FY"})
    df["DOR_CODE"] = df["DOR_CODE"].str.zfill(3)
    df["Municipality"] = df["Municipality"].str.strip()
    df["FY"] = df["FY"].astype(int)
    df = df.drop(columns=["Rank"], errors="ignore")
    for c in df.columns[3:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[df["Single-Family Tax Bill"].notna()]


def levy_capacity(fields: dict[str, list[str]]) -> pd.DataFrame:
    """Every year's levy limits, levy, and excess capacity, all municipalities in one request."""
    df = export(LEVY, "tblExcess", [("iclMuni", m) for m in fields["iclMuni"]] + [("iclYear", y) for y in fields["iclYear"]])
    df = df.rename(columns=lambda c: re.sub(r"\s+", " ", str(c)).strip())
    df = df.rename(columns={"DOR Code": "DOR_CODE", "Fiscal Year": "FY"})
    df["DOR_CODE"] = df["DOR_CODE"].str.zfill(3)
    df["Municipality"] = df["Municipality"].str.strip()
    df["FY"] = df["FY"].astype(int)
    for c in df.columns[3:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[df["Total Tax Levy"].notna()]


def main():
    today = dt.date.today()
    print("[DLS] Reading the LA-4 and EQV forms...")
    la4_fields, la4_as_of = form(LA4)
    eqv_fields, eqv_as_of = form(EQV)
    years = sorted(int(y) for y in la4_fields["islYear"])
    print(f"  LA-4: {len(la4_fields['iclMuni'])} municipalities, FY{years[0]}-FY{years[-1]}, current as of {la4_as_of}")
    parcels = la4(la4_fields).sort_values(["FY", "DOR_CODE", "Category"])
    parcels.to_csv(OUT_LA4, index=False)
    taxable = parcels[~parcels["Category"].str.startswith("Exempt")]
    filed = taxable[taxable["Value"] > 0].groupby("FY")["DOR_CODE"].nunique()
    print(f"  {OUT_LA4}: {len(parcels):,} rows")
    print("  Municipalities with taxable value, by year:")
    print("    " + filed.to_string().replace("\n", "\n    "))

    print(f"  EQV: {len(eqv_fields['iclYear'])} years, current as of {eqv_as_of}")
    vals = eqv(eqv_fields).sort_values(["Year", "DOR_CODE"])
    vals.to_csv(OUT_EQV, index=False)
    print(f"  {OUT_EQV}: {len(vals):,} rows, {vals['Year'].min()}-{vals['Year'].max()}")

    sf_fields, sf_as_of = form(SF_BILL)
    bills = single_family_bills(sf_fields).sort_values(["FY", "DOR_CODE"])
    bills.to_csv(OUT_SF_BILL, index=False)
    print(f"  {OUT_SF_BILL}: {len(bills):,} rows, FY{bills['FY'].min()}-FY{bills['FY'].max()}, current as of {sf_as_of}")

    levy_fields, levy_as_of = form(LEVY)
    levy = levy_capacity(levy_fields).sort_values(["FY", "DOR_CODE"])
    levy.to_csv(OUT_LEVY, index=False)
    print(f"  {OUT_LEVY}: {len(levy):,} rows, FY{levy['FY'].min()}-FY{levy['FY'].max()}, current as of {levy_as_of}")

    SOURCES_JSON.write_text(json.dumps({
        "la4": {"source": f"DLS Municipal Databank, Parcel Counts and Values by Usage Code: {BASE}?rdReport={LA4}",
                "data_current_as_of": la4_as_of, "first_year": years[0], "latest_year": years[-1]},
        "eqv": {"source": f"DLS Municipal Databank, Equalized Valuations (EQV): {BASE}?rdReport={EQV}",
                "data_current_as_of": eqv_as_of,
                "first_year": int(vals["Year"].min()), "latest_year": int(vals["Year"].max())},
        "single_family_tax_bill": {
            "source": f"DLS Municipal Databank, Average Single Family Tax Bill: {BASE}?rdReport={SF_BILL}",
            "data_current_as_of": sf_as_of,
            "first_year": int(bills["FY"].min()), "latest_year": int(bills["FY"].max())},
        "levy_capacity": {
            "source": f"DLS Municipal Databank, Excess Levy Capacity and Override Capacity: {BASE}?rdReport={LEVY}",
            "data_current_as_of": levy_as_of,
            "first_year": int(levy["FY"].min()), "latest_year": int(levy["FY"].max())},
        "downloaded": today.isoformat(),
    }, indent=1) + "\n")
    print(f"  {SOURCES_JSON}")


if __name__ == "__main__":
    main()
