"""
Inflation addendum (draft): how the report's per-pupil expenditure (PPE)
findings read once prices are taken into account.

The main report's tables are nominal. This mini-report restates its PPE levels
and growth rates against three price indexes (data/inflation_indices.csv,
fiscal-year averages, from update_inflation_data.py):

- CPI-U Northeast: general consumer prices in the region.
- State and local government deflator (BEA): prices of what state and local
  governments buy, mostly labor. Chapter 70's foundation inflation index is
  built on it (M.G.L. c.70, s.2).
- CPI-U U.S. city average: shown for reference.

Usage:
    python inflation_addendum.py                      # the current edition (LATEST_YEAR)
    PPE_LATEST_YEAR=2024 python inflation_addendum.py # the first edition's window, in output/

Outputs (in the edition's output folder, OUTPUT_DIR):
    WMPPE Inflation Addendum.pdf (or "... (FY2025 data).pdf" for later editions)
    inflation_indices_vs_ppe.png
    inflation_real_ppe_districts.png
    inflation_addendum_table.csv
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

from school_shared import (DISTRICTS_OF_INTEREST, LATEST_YEAR, OUTPUT_DIR, START_YEAR, WINDOW_YEARS,
                           edition_pdf_name, load_data,
                           get_western_cohort_districts_for_year,
                           weighted_epp_aggregation)

DATA_DIR = Path("./data")
INDEX_CSV = DATA_DIR / "inflation_indices.csv"
PDF_NAME = edition_pdf_name("WMPPE Inflation Addendum")

INDEXES = {
    "cpi_u_northeast": "CPI-U Northeast",
    "sl_gov_deflator": "State & local gov't deflator",
    "cpi_u_us": "CPI-U U.S.",
}
MAIN_INDEXES = ["cpi_u_northeast", "sl_gov_deflator"]
COHORT_ORDER = [("TINY", "Western MA Tiny"), ("SMALL", "Western MA Small"),
                ("MEDIUM", "Western MA Medium"), ("LARGE", "Western MA Large"),
                ("SPRINGFIELD", "Springfield")]
DISTRICT_LABELS = {"Amherst-Pelham": "Amherst-Pelham Regional"}


def cagr(start: float, end: float, years: int) -> float:
    return (end / start) ** (1 / years) - 1


def ppe_total(pivot: pd.DataFrame) -> pd.Series:
    return pivot.sum(axis=1)


def build_series(df, reg, latest):
    """PPE totals by year for Western MA groups and the districts of interest."""
    cohorts = get_western_cohort_districts_for_year(df, reg, latest)
    series = {}
    western = [d for key, members in cohorts.items() if key != "SPRINGFIELD" for d in members]
    piv, _, _ = weighted_epp_aggregation(df, western)
    series["Western MA (all, excl. Springfield)"] = ppe_total(piv)
    for key, label in COHORT_ORDER:
        if cohorts.get(key):
            piv, _, _ = weighted_epp_aggregation(df, cohorts[key])
            series[label] = ppe_total(piv)
    for dist in DISTRICTS_OF_INTEREST:
        piv, _, _ = weighted_epp_aggregation(df, [dist])
        series[DISTRICT_LABELS.get(dist, dist)] = ppe_total(piv)
    sizes = {label: len(cohorts.get(key, [])) for key, label in COHORT_ORDER}
    return series, sizes


def build_table(series, idx, latest):
    start = START_YEAR
    rows = []
    for name, s in series.items():
        if start not in s.index or latest not in s.index:
            continue
        row = {"group": name, "start_nominal": s[start], "latest_nominal": s[latest]}
        for n in (WINDOW_YEARS, 10, 5):
            row[f"nominal_{n}y"] = cagr(s[latest - n], s[latest], n)
        for key in INDEXES:
            ratio = idx.loc[latest, key] / idx.loc[start, key]
            row[f"start_real_{key}"] = s[start] * ratio  # start-year PPE in latest-year dollars
            for n in (WINDOW_YEARS, 10, 5):
                i_ratio = idx.loc[latest, key] / idx.loc[latest - n, key]
                row[f"real_{key}_{n}y"] = ((s[latest] / s[latest - n]) / i_ratio) ** (1 / n) - 1
            row[f"real_change_{key}"] = s[latest] / row[f"start_real_{key}"] - 1
        rows.append(row)
    return pd.DataFrame(rows)


def plot_indexes(series, idx, latest, path):
    start = START_YEAR
    years = list(range(start, latest + 1))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ppe = series["Western MA (all, excl. Springfield)"].reindex(years)
    ax.plot(years, ppe / ppe[start] * 100, color="#1b6ca8", lw=3,
            label="Western MA PPE (nominal)")
    styles = {"cpi_u_northeast": ("#d1495b", "-"), "sl_gov_deflator": ("#2e8b57", "-"),
              "cpi_u_us": ("#888888", "--")}
    for key, label in INDEXES.items():
        col, ls = styles[key]
        s = idx.loc[years, key]
        ax.plot(years, s / s[start] * 100, color=col, ls=ls, lw=2, label=label)
    ax.axhline(100, color="#cccccc", lw=1)
    ax.set_ylabel(f"Index, FY{start} = 100", fontsize=12)
    ax.set_xlabel("Fiscal year", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=11, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_real_districts(series, idx, latest, path):
    start = START_YEAR
    years = list(range(start, latest + 1))
    key = "cpi_u_northeast"
    fig, ax = plt.subplots(figsize=(10, 5.5))
    palette = {"Western MA (all, excl. Springfield)": ("#555555", "--"),
               "Western MA Tiny": ("#999999", ":"),
               "Amherst-Pelham Regional": ("#1b6ca8", "-"), "Amherst": ("#7b4ea3", "-"),
               "Leverett": ("#d1495b", "-"), "Pelham": ("#edae49", "-"),
               "Shutesbury": ("#2e8b57", "-")}
    for name, (col, ls) in palette.items():
        if name not in series:
            continue
        s = series[name].reindex(years)
        real = s * idx.loc[latest, key] / idx.loc[years, key].values
        ax.plot(years, real, color=col, ls=ls, lw=2.5 if ls == "-" else 2, label=name)
    ax.set_ylabel(f"PPE in FY{latest} dollars (CPI-U Northeast)", fontsize=12)
    ax.set_xlabel("Fiscal year", fontsize=12)
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("${x:,.0f}"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=10, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def money(v):
    return f"${v:,.0f}"


def pct(v):
    return f"{v * 100:+.1f}%"


def compose_pdf(table, idx, latest, sizes, out_pdf, fig1, fig2):
    start = START_YEAR
    ss = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=ss["BodyText"], fontSize=10, leading=14)
    small = ParagraphStyle("small", parent=body, fontSize=8, leading=10)
    h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=18)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=13)
    num = ParagraphStyle("num", parent=small, alignment=TA_RIGHT)
    hdr = ParagraphStyle("hdr", parent=small, fontName="Helvetica-Bold")
    hdr_r = ParagraphStyle("hdr_r", parent=hdr, alignment=TA_RIGHT)

    infl = {k: idx.loc[latest, k] / idx.loc[start, k] - 1 for k in INDEXES}
    infl_cagr = {k: cagr(idx.loc[start, k], idx.loc[latest, k], WINDOW_YEARS) for k in INDEXES}
    wm = table.set_index("group").loc["Western MA (all, excl. Springfield)"]

    story = [
        Paragraph("Western MA Per Pupil Expenditure Report", h1),
        Paragraph(f"Inflation Addendum — DRAFT, {dt.date.today():%B %-d, %Y}", h2),
        Spacer(1, 6),
        Paragraph(
            f"The main report compares per-pupil expenditure (PPE) in nominal dollars. Over "
            f"FY{start}–FY{latest}, prices rose too: {infl['cpi_u_northeast'] * 100:.0f}% by the "
            f"CPI-U for the Northeast ({infl_cagr['cpi_u_northeast'] * 100:.1f}% a year) and "
            f"{infl['sl_gov_deflator'] * 100:.0f}% by the federal price index for state and local "
            f"government purchases ({infl_cagr['sl_gov_deflator'] * 100:.1f}% a year). This addendum "
            f"restates the report's PPE levels and growth rates against both.", body),
        Spacer(1, 6),
        Paragraph(
            f"Western MA PPE grew {wm[f'nominal_{WINDOW_YEARS}y'] * 100:.1f}% a year in nominal terms. After "
            f"Northeast consumer inflation, that is {wm[f'real_cpi_u_northeast_{WINDOW_YEARS}y'] * 100:.1f}% a year; "
            f"after state and local government inflation, {wm[f'real_sl_gov_deflator_{WINDOW_YEARS}y'] * 100:.1f}% "
            f"a year.", body),
        Spacer(1, 10),
        Paragraph("Which price index", h2),
        Paragraph(
            "<b>CPI-U Northeast</b> measures what households in the region pay for goods and "
            "services. It answers the taxpayer's question: has school spending outpaced the cost "
            "of living? <b>The state and local government deflator</b> (BEA, NIPA Table 1.1.9) "
            "measures the prices state and local governments pay, mostly for labor. It answers the "
            "budget's question: has spending outpaced the cost of what schools buy? The Commonwealth "
            "uses it to inflate the Chapter 70 foundation budget, with a 4.5% annual cap (M.G.L. "
            "c.70, s.2). Each index is averaged over the July–June fiscal year, the same period as "
            "DESE's school year.", body),
        Spacer(1, 8),
        Image(str(fig1), width=6.5 * inch, height=3.575 * inch),
        Paragraph(f"Figure 1. Western MA PPE and price indexes, FY{start} = 100.", small),
        PageBreak(),
        Paragraph(f"PPE growth, nominal and real, FY{start}–FY{latest}", h2),
    ]

    cols = ["Cohort/District", f"FY{start} $/pupil", f"FY{start} in FY{latest} $",
            f"FY{latest} $/pupil", f"Nominal {WINDOW_YEARS}y", f"Real {WINDOW_YEARS}y (CPI-NE)",
            f"Real {WINDOW_YEARS}y (S&L defl.)", "Real 10y (CPI-NE)", "Real 5y (CPI-NE)"]
    data = [[Paragraph(f"<b>{c}</b>", hdr if i == 0 else hdr_r) for i, c in enumerate(cols)]]
    for _, r in table.iterrows():
        label = r["group"]
        if label in sizes and sizes[label]:
            label = f"{label} ({sizes[label]})"
        data.append([Paragraph(label, small),
                     Paragraph(money(r["start_nominal"]), num),
                     Paragraph(money(r["start_real_cpi_u_northeast"]), num),
                     Paragraph(money(r["latest_nominal"]), num),
                     Paragraph(pct(r[f"nominal_{WINDOW_YEARS}y"]), num),
                     Paragraph(pct(r[f"real_cpi_u_northeast_{WINDOW_YEARS}y"]), num),
                     Paragraph(pct(r[f"real_sl_gov_deflator_{WINDOW_YEARS}y"]), num),
                     Paragraph(pct(r["real_cpi_u_northeast_10y"]), num),
                     Paragraph(pct(r["real_cpi_u_northeast_5y"]), num)])
    t = Table(data, colWidths=[1.55 * inch] + [0.74 * inch] * 8, repeatRows=1)
    t.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.black),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, 6), (-1, 6), 0.6, colors.HexColor("#999999")),
    ]))
    story += [t, Spacer(1, 4), Paragraph(
        "Growth rates are compound annual rates. Real rates divide each year's PPE by the "
        "index for that fiscal year. Cohort averages are weighted by in-district FTE, with "
        f"cohort membership set by FY{latest} enrollment as in the main report; the number of "
        "districts is in parentheses.", small), Spacer(1, 10)]

    story += [Image(str(fig2), width=6.5 * inch, height=3.575 * inch),
              Paragraph(f"Figure 2. PPE in constant FY{latest} dollars (CPI-U Northeast), "
                        "ARPS districts and Western MA averages.", small),
              Spacer(1, 10),
              Paragraph("Notes and open questions", h2),
              Paragraph(
                  "This draft covers PPE only. Chapter 70 aid and net school spending can be "
                  "restated the same way. The main report's original CPI file used calendar-year "
                  "averages; this addendum uses fiscal-year averages, which match the school year. "
                  f"Inflation series: BLS CUUR0100SA0 and CUUR0000SA0 (monthly), BEA A829RD "
                  f"(quarterly), retrieved {dt.date.today():%B %-d, %Y}.", body)]

    doc = SimpleDocTemplate(str(out_pdf), pagesize=letter, leftMargin=0.6 * inch,
                            rightMargin=0.6 * inch, topMargin=0.6 * inch, bottomMargin=0.6 * inch,
                            title="WMPPE Inflation Addendum (draft)", author="Tim Shores")
    doc.build(story)


def main():
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()

    df, reg, c70 = load_data()
    latest = LATEST_YEAR
    df = df[df["YEAR"] <= latest].copy()

    idx = pd.read_csv(INDEX_CSV).set_index("fiscal_year")
    missing = [k for k in INDEXES if pd.isna(idx.loc[latest, k])]
    if missing:
        raise SystemExit(f"No FY{latest} value for {missing}; run update_inflation_data.py")

    series, sizes = build_series(df, reg, latest)
    table = build_table(series, idx, latest)
    OUTPUT_DIR.mkdir(exist_ok=True)
    table.to_csv(OUTPUT_DIR / "inflation_addendum_table.csv", index=False)
    fig1 = OUTPUT_DIR / "inflation_indices_vs_ppe.png"
    fig2 = OUTPUT_DIR / "inflation_real_ppe_districts.png"
    plot_indexes(series, idx, latest, fig1)
    plot_real_districts(series, idx, latest, fig2)
    out_pdf = OUTPUT_DIR / PDF_NAME
    compose_pdf(table, idx, latest, sizes, out_pdf, fig1, fig2)
    print(f"[Inflation addendum] FY{START_YEAR}-FY{latest}: {out_pdf}")


if __name__ == "__main__":
    main()
