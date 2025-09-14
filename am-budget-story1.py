#!/usr/bin/env python3
"""
story1_operating_share.py

Compute and visualize Story 1:
"Operating is getting a smaller slice of the General Fund pie."

Adds small value labels to each point on the chart.

INPUT
  - Excel with normalized rows:
      Year | Scenario | Category | Amount
    Categories needed: "Subtotal operating", "Total expenditures"
    Scenarios: "Budgeted", "Actual"

OUTPUT
  - CSV with operating share by year & scenario
  - PNG line chart of Budgeted vs Actual operating share (with point labels)

USAGE
  python story1_operating_share.py \
      --input "/path/to/amherst-town-budgets-fy18-25.xlsx" \
      --sheet "Sheet1" \
      --out_csv "story1_operating_share.csv" \
      --out_png "story1_operating_share.png"
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patheffects as pe


def compute_operating_share(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a tidy dataframe with:
      Year | Scenario | OperatingShare
    where OperatingShare = Subtotal operating / Total expenditures
    """
    required_cols = {"Year", "Scenario", "Category", "Amount"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    needed_cats = {"Subtotal operating", "Total expenditures"}
    sub = df[df["Category"].isin(needed_cats)].copy()

    wide = (
        sub.pivot_table(
            index=["Year", "Scenario"],
            columns="Category",
            values="Amount",
            aggfunc="sum",
        )
        .reset_index()
    )
    wide.columns.name = None

    if "Subtotal operating" not in wide.columns or "Total expenditures" not in wide.columns:
        raise ValueError(
            "Input does not contain both 'Subtotal operating' and 'Total expenditures'."
        )

    wide["OperatingShare"] = (
        wide["Subtotal operating"] / wide["Total expenditures"]
    )

    out = wide[["Year", "Scenario", "OperatingShare"]].sort_values(["Year", "Scenario"])
    return out


def _annotate_points(ax, x_vals, y_vals_pct, position="above", fmt="{:.1f}%", nudge_frac=0.03, text_color=None):
    """
    Add small labels just above each point.
    nudge_frac: fraction of y-range to offset labels upward (to avoid overlap with markers).
    """
    if len(y_vals_pct) == 0:
        return

    # Compute a vertical nudge based on data range
    ymin = np.nanmin(y_vals_pct)
    ymax = np.nanmax(y_vals_pct)
    yrng = max(1.0, ymax - ymin)  # avoid zero range
    dy = yrng * nudge_frac

    for xi, yi in zip(x_vals, y_vals_pct):
        if pd.notna(yi):
            ax.text(
                xi,
                yi - dy if position == "below" else yi + dy,
                fmt.format(yi),
                ha="center",
                va="top" if position == "below" else "bottom",
                fontsize=9, color=text_color,
                path_effects=[pe.withStroke(linewidth=2, foreground="white", alpha=0.8)],
            )


def plot_operating_share(tidy: pd.DataFrame, out_png: Path):
    """
    Plots Budgeted vs Actual operating share across years with point labels.
    """
    plot_w = tidy.pivot(index="Year", columns="Scenario", values="OperatingShare").sort_index()

    fig = plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # Collect all y values (as %) for label-range computation
    all_y_pct = []

    # Plot Budgeted
    if "Budgeted" in plot_w.columns:
        y_b = (plot_w["Budgeted"] * 100).astype(float)
        x_b = plot_w.index.values
        ax.plot(x_b, y_b, marker="o", label="Budgeted")
        all_y_pct.extend(y_b.tolist())
        _annotate_points(ax, x_b, y_b, position="above", text_color="#1f4e79")

    # Plot Actual
    if "Actual" in plot_w.columns:
        y_a = (plot_w["Actual"] * 100).astype(float)
        x_a = plot_w.index.values
        ax.plot(x_a, y_a, marker="o", label="Actual")
        all_y_pct.extend(y_a.tolist())
        _annotate_points(ax, x_a, y_a, position="below", text_color="#b04a02")

    ax.set_title("Operating Share of Total Expenditures (%)")
    ax.set_xlabel("Fiscal Year")
    ax.set_ylabel("Operating Share (%)")

    # Format y-axis as percent (0–100 scale)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100))

    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()

    if out_png:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=200)
    # plt.show()  # Uncomment to view interactively
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Story 1: Operating share of General Fund.")
    p.add_argument("--input", required=True, help="Path to Excel file")
    p.add_argument("--sheet", default="Sheet1", help="Worksheet name (default: Sheet1)")
    p.add_argument("--out_csv", default="output/story1_operating_share.csv", help="Output CSV path")
    p.add_argument("--out_png", default="output/story1_operating_share.png", help="Output PNG path")
    args = p.parse_args()

    inpath = Path(args.input)
    if not inpath.exists():
        raise FileNotFoundError(inpath)

    df = pd.read_excel(inpath, sheet_name=args.sheet)
    tidy = compute_operating_share(df)

    # Save CSV
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tidy.sort_values(["Year", "Scenario"]).to_csv(out_csv, index=False)

    # Plot PNG
    plot_operating_share(tidy, Path(args.out_png))

    # Console summary
    years = sorted(tidy["Year"].unique())
    first, last = years[0], years[-1]

    def fmt_val(y, scen):
        v = tidy.query("Year == @y and Scenario == @scen")["OperatingShare"]
        return f"{float(v.iloc[0])*100:.2f}%" if len(v) else "n/a"

    print("=== Story 1: Operating Share (% of Total Expenditures) ===")
    print(f"First year ({first}) — Budgeted: {fmt_val(first, 'Budgeted')}, Actual: {fmt_val(first, 'Actual')}")
    print(f"Last year  ({last})  — Budgeted: {fmt_val(last,  'Budgeted')}, Actual: {fmt_val(last,  'Actual')}")
    print(f"Saved CSV → {args.out_csv}")
    print(f"Saved PNG → {args.out_png}")


if __name__ == "__main__":
    main()
