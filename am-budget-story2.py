#!/usr/bin/env python3
"""
story2_municipal_overestimation.py

Analyze Story 2:
- Municipal expenditures are systematically overestimated (Actual < Budget)
- Next-year budgets often apply the SAME % increase across operating categories (budget-on-budget)
- Compare growth computed from prior Budget vs prior Actual to show base-choice effects

INPUT (Excel, normalized):
  Year | Scenario | Category | Amount
  Categories used: "Municipal operating", "ASD operating", "RSD operating", "Jones Library operating"
                   (plus "Subtotal operating", "Total expenditures" not required here)
  Scenarios: "Budgeted", "Actual"

OUTPUT:
  - CSV: output/story2_municipal_summary.csv
  - PNGs:
    output/story2_municipal_variance.png
    output/story2_budget_growth_rates.png
    output/story2_growth_budget_vs_actual.png

USAGE:
  python am_budget_story2.py \
      --input "data/amherst-town-budgets-fy10-25.xlsx" \
      --range 2010-2026 \
      --sheet "Sheet1" \
      --tolerance 0.002 \
      --out_dir "output"
"""

import argparse
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as patches

# ------------------------ CONFIG ------------------------

OPERATING_CATS = [
    "Municipal operating",
    "ASD operating",
    "RSD operating",
    "Jones Library operating",
]

DASHED_CATS = {"ASD operating", "RSD operating"}  # helps to distinguish school districts
CAT_COLORS = dict(zip(OPERATING_CATS, ["C0", "C1", "C2", "C3"]))


# ------------------------ IO & CORE ------------------------

def load_df(path: Path, sheet: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet)
    required = {"Year", "Scenario", "Category", "Amount"}
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"Missing required columns: {sorted(miss)}")
    return df


def get_amount(df: pd.DataFrame, year: int, scenario: str, category: str) -> float:
    m = (df["Year"] == year) & (df["Scenario"] == scenario) & (df["Category"] == category)
    if not m.any():
        return np.nan
    return float(df.loc[m, "Amount"].values[0])


def compute_variances(df: pd.DataFrame) -> pd.DataFrame:
    years = sorted(df["Year"].unique())
    rows = []
    for y in years:
        b = get_amount(df, y, "Budgeted", "Municipal operating")
        a = get_amount(df, y, "Actual", "Municipal operating")
        if np.isnan(b) or b == 0:
            var_pct = np.nan
            var_amt = np.nan
        else:
            var_amt = (a - b) if not np.isnan(a) else np.nan
            var_pct = (var_amt / b) if not np.isnan(var_amt) else np.nan
        rows.append({"Year": y, "Budgeted": b, "Actual": a, "MunicipalVarAmt": var_amt, "MunicipalVarPct": var_pct})
    return pd.DataFrame(rows)


def compute_budget_growth_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each transition Y -> Y+1, compute Budget growth rates per operating category:
      growth_rate = (Budget_{Y+1} - Budget_{Y}) / Budget_{Y}
    """
    years = sorted(df["Year"].unique())
    rows = []
    for i in range(len(years) - 1):
        y, y_next = years[i], years[i + 1]
        row = {"Year": y, "NextYear": y_next}
        for cat in OPERATING_CATS:
            b0 = get_amount(df, y, "Budgeted", cat)
            b1 = get_amount(df, y_next, "Budgeted", cat)
            if np.isnan(b0) or b0 == 0 or np.isnan(b1):
                gr = np.nan
            else:
                gr = (b1 - b0) / b0
            row[f"{cat}__BudgetGR"] = gr
        rows.append(row)
    return pd.DataFrame(rows)


def flag_uniform_growth(gr_df: pd.DataFrame, tolerance: float) -> pd.Series:
    """
    Flag transitions where all operating categories share (approximately) the SAME growth rate:
      max(gr) - min(gr) <= tolerance
    Returns a boolean Series indexed like gr_df.
    """
    flags = []
    for _, r in gr_df.iterrows():
        vals = [r[f"{cat}__BudgetGR"] for cat in OPERATING_CATS]
        vals = [v for v in vals if not np.isnan(v)]
        if len(vals) < len(OPERATING_CATS):
            flags.append(False)
            continue
        flags.append((max(vals) - min(vals)) <= tolerance)
    return pd.Series(flags, index=gr_df.index, name="UniformGrowthFlag")


def build_summary(df: pd.DataFrame, gr_df: pd.DataFrame, uniform_flag: pd.Series) -> pd.DataFrame:
    """Combine key metrics into one tidy table keyed by Year (and NextYear where relevant)."""
    var_df = compute_variances(df)
    out = var_df.merge(
        gr_df.assign(UniformGrowthFlag=uniform_flag),
        how="left",
        on="Year"
    )

    # Order columns for readability
    ordered = [
        "Year", "NextYear",
        "Budgeted", "Actual",
        "MunicipalVarAmt", "MunicipalVarPct",
        *(f"{cat}__BudgetGR" for cat in OPERATING_CATS),
        "UniformGrowthFlag",
    ]
    cols = [c for c in ordered if c in out.columns] + [c for c in out.columns if c not in ordered]
    return out[cols]


# ------------------------ HELPERS ------------------------

def parse_year_range(rng: Optional[str]) -> Optional[Tuple[int, int]]:
    if not rng:
        return None
    try:
        start_s, end_s = rng.split("-")
        return int(start_s), int(end_s)
    except Exception as e:
        raise ValueError("--range must look like START-END, e.g., 2010-2026") from e


def apply_year_range(df: pd.DataFrame, year_range: Optional[Tuple[int, int]]) -> pd.DataFrame:
    if not year_range:
        return df
    lo, hi = year_range
    return df[(df["Year"] >= lo) & (df["Year"] <= hi)].copy()


def sparse_xticks(ax, positions, labels, max_labels: int = 10, rotation: int = 0, fontsize: int = 10):
    """
    Reduce clutter by keeping ~max_labels labels. Others are blank.
    """
    n = len(positions)
    if n <= max_labels:
        shown = labels
    else:
        step = int(np.ceil(n / max_labels))
        shown = [lbl if (i % step == 0) else "" for i, lbl in enumerate(labels)]
    ax.set_xticks(positions)
    ax.set_xticklabels(shown, rotation=rotation, ha="right" if rotation else "center", fontsize=fontsize)


# ------------------------ PLOTS ------------------------

def plot_municipal_variance(df: pd.DataFrame, out_png: Path):
    """Four stacked panels: % variance (Actual vs Budgeted) for each operating category."""
    years = sorted(df["Year"].unique())

    # Build % variance series for each category (in percentage points)
    series = {}
    all_vals = []
    for cat in OPERATING_CATS:
        vals = []
        for y in years:
            b = get_amount(df, y, "Budgeted", cat)
            a = get_amount(df, y, "Actual",   cat)
            if np.isnan(b) or b == 0 or np.isnan(a):
                v = np.nan
            else:
                v = (a - b) / b * 100.0
            vals.append(v)
        arr = np.array(vals, dtype=float)
        series[cat] = arr
        all_vals.extend([v for v in arr if not np.isnan(v)])

    # Consistent y-limits across all panels
    ymin = min(all_vals + [0.0])
    ymax = max(all_vals + [0.0])
    pad = 0.05 * (ymax - ymin if ymax > ymin else 1.0)
    ylim = (ymin - pad, ymax + pad)

    fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(11, 9))
    for ax, cat in zip(axes, OPERATING_CATS):
        y = series[cat]
        bars = ax.bar(years, y, color=CAT_COLORS.get(cat, None))
        # label inside the bar when magnitude ≥ 2.0%
        for xi, yi, b in zip(years, y, bars):
            if not np.isnan(yi) and abs(yi) >= 2.0:
                ax.text(
                    xi, yi * 0.5, f"{yi:.1f}%",
                    ha="center", va="center", fontsize=9, color="white",
                )
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title(cat, fontsize=11)
        ax.set_ylabel("% Var")
        ax.set_ylim(*ylim) 
            # Vertical shading for odd fiscal years
        ymin, ymax = ax.get_ylim()
        for xi in years:
            if int(xi) % 2 == 1:
                ax.add_patch(
                    patches.Rectangle((xi - 0.5, ymin),
                                      1.0, ymax - ymin,
                                      alpha=0.08, zorder=0)
                )
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100))
        ax.grid(True, linestyle="--", alpha=0.2)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    axes[-1].set_xlabel("Fiscal Year")
    fig.suptitle("Amherst — % Variance (Actual vs Budgeted) by Operating Category", y=0.99, fontsize=17)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_budget_growth_rates(df: pd.DataFrame, gr_df: pd.DataFrame, out_png: Path):
    """
    Two vertically aligned plots: top = growth from prior Budget (Budgeted),
    bottom = growth from prior Actual; both axes share the same y-scale.
    Automatic odd-year vertical shading.
    """
    years = gr_df["Year"].astype(int).values
    fig, (ax_top, ax_bot) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(11, 9))

    # Precompute actual-anchored growth rates: (Budget_{Y+1} - Actual_{Y}) / Actual_{Y}
    y_actual = {cat: [] for cat in OPERATING_CATS}
    for y in years:
        for cat in OPERATING_CATS:
            a0 = get_amount(df, int(y), "Actual",   cat)
            b1 = get_amount(df, int(y)+1, "Budgeted", cat)
            if (a0 is None) or np.isnan(a0) or a0 == 0 or (b1 is None) or np.isnan(b1):
                y_actual[cat].append(np.nan)
            else:
                y_actual[cat].append((b1 - a0) / a0 * 100.0)

    # TOP: budget-anchored lines (from prior Budget)
    for cat in OPERATING_CATS:
        yvals = (gr_df[f"{cat}__BudgetGR"] * 100).astype(float).values
        style = dict(linewidth=2.8, alpha=0.75, markersize=6,
                     markeredgecolor="white", markeredgewidth=0.8)
        if cat in DASHED_CATS:
            style.update(linestyle="--", zorder=6, linewidth=3.2, alpha=0.9)
        else:
            style.update(zorder=4)
        ax_top.plot(years, yvals, marker="o", label=cat, **style)

    # BOTTOM: actual-anchored lines (from prior Actual)
    for cat in OPERATING_CATS:
        yvals = np.array(y_actual[cat], dtype=float)
        style = dict(linewidth=2.8, alpha=0.75, markersize=6,
                     markeredgecolor="white", markeredgewidth=0.8)
        if cat in DASHED_CATS:
            style.update(linestyle="--", zorder=6, linewidth=3.2, alpha=0.9)
        else:
            style.update(zorder=4)
        ax_bot.plot(years, yvals, marker="o", label=cat, **style)

    # Equal y-scale on both plots
    for ax in (ax_top, ax_bot):
        ax.set_ylim(-5, 12)
        ax.set_ylabel("Budget Growth Rate (%)")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100))
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    # Automatic shading for odd base years
    for ax in (ax_top, ax_bot):
        ymin, ymax = ax.get_ylim()
        for xi in years:
            if int(xi) % 2 == 1:  # odd years
                ax.add_patch(
                    patches.Rectangle((xi - 0.5, ymin),
                                      1.0, ymax - ymin,
                                      alpha=0.08, zorder=0)
                )

    # Titles/labels
    ax_top.set_title("Growth from prior Budget")
    ax_bot.set_title("Growth from prior Actual")
    ax_bot.set_xlabel("Base Fiscal Year")

    # Legend tightened up and overall top margin reduced
    handles, labels = ax_top.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center", bbox_to_anchor=(0.5, 0.965),
        ncol=4, frameon=False, title="Operating category",
        columnspacing=1.5, handlelength=2.2, handletextpad=0.6, borderaxespad=0.4
    )
    fig.suptitle("Amherst — Operating Budget Growth Rates (base year → next year)", y=0.992, fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    # De-clutter x ticks
    sparse_xticks(ax_bot, years, [str(y) for y in years], max_labels=12, rotation=0, fontsize=9)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_growth_from_budget_vs_actual(df: pd.DataFrame, out_png: Path):
    """
    For each base fiscal year Y and each operating category:
      - Budget-anchored growth: (Budget_{Y+1} - Budget_{Y}) / Budget_{Y}
      - Actual-anchored  growth: (Budget_{Y+1} - Actual_{Y}) / Actual_{Y}
    Produces a 4×1 stacked chart; each panel shows paired bars per year.
    """
    years_all = sorted(df["Year"].unique())
    base_years = [y for y in years_all if (y + 1) in years_all]
    x = np.arange(len(base_years))
    width = 0.38

    # Precompute growth arrays (in %)
    growth_budget = {}
    growth_actual = {}
    for cat in OPERATING_CATS:
        gb, ga = [], []
        for y in base_years:
            b0 = get_amount(df, y,       "Budgeted", cat)
            a0 = get_amount(df, y,       "Actual",   cat)
            b1 = get_amount(df, y + 1,   "Budgeted", cat)

            # Budget-anchored
            if np.isnan(b0) or b0 == 0 or np.isnan(b1):
                gb.append(np.nan)
            else:
                gb.append((b1 - b0) / b0 * 100.0)

            # Actual-anchored
            if np.isnan(a0) or a0 == 0 or np.isnan(b1):
                ga.append(np.nan)
            else:
                ga.append((b1 - a0) / a0 * 100.0)

        growth_budget[cat] = np.array(gb, dtype=float)
        growth_actual[cat] = np.array(ga, dtype=float)

    # Y-limits consistent across panels
    all_vals = []
    for cat in OPERATING_CATS:
        all_vals.extend([v for v in growth_budget[cat] if not np.isnan(v)])
        all_vals.extend([v for v in growth_actual[cat] if not np.isnan(v)])
    ymin = min(all_vals + [0.0])
    ymax = max(all_vals + [0.0])
    pad = 0.05 * (ymax - ymin if ymax > ymin else 1.0)
    ylim = (ymin - pad, ymax + pad)

    fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(12, 11))
    for i, (ax, cat) in enumerate(zip(axes, OPERATING_CATS)):
        col = CAT_COLORS.get(cat, None)
        y_b = growth_budget[cat]
        y_a = growth_actual[cat]

        bars_b = ax.bar(x - width/2, y_b, width=width, color=col, label="Growth from prior Budget")
        bars_a = ax.bar(x + width/2, y_a, width=width, color=col, alpha=0.55, hatch="//",
                        edgecolor="black", linewidth=0.4, label="Growth from prior Actual")

        # Inset legend at top-left: two lines with swatches and 'Avg: X%'
        avg_b = np.nanmean(y_b)  # budget-anchored (%)
        avg_a = np.nanmean(y_a)  # actual-anchored (%)

        h_budget = patches.Patch(facecolor=col, edgecolor="black", linewidth=0.4)
        h_actual = patches.Patch(facecolor=col, alpha=0.55, hatch="//",
                                 edgecolor="black", linewidth=0.4)

        ax.legend(
            [h_budget, h_actual],
            [f"Growth from prior Budget — Avg: {avg_b:.1f}%",
             f"Growth from prior Actual — Avg: {avg_a:.1f}%"],
            loc="upper left", fontsize=9, frameon=False, ncol=1,
            borderaxespad=0.3, handlelength=1.8, handletextpad=0.6, labelspacing=0.3
        )

        ax.axhline(0, color="black", linewidth=1)
        ax.set_ylim(*ylim)
        ax.set_ylabel("%")
        ax.set_title(cat, fontsize=11)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100))
        ax.grid(True, linestyle="--", alpha=0.2)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        # Optional: labels inside bars for big magnitudes (|%| ≥ 2)
        for bars, lbl_color in ((bars_b, "white"), (bars_a, "black")):
            for b in bars:
                yi = b.get_height()
                if not np.isnan(yi) and abs(yi) >= 2.0:
                    ax.text(b.get_x() + b.get_width()/2, yi*0.5, f"{yi:.1f}%",
                            ha="center", va="center", fontsize=8, color=lbl_color)
        
        # === Top-right summary box for 2021–2025 averages (all categories) ===
        yrs = np.array(base_years)
        mask = (yrs >= 2021) & (yrs <= 2025)
        if mask.any():
            avg_b_2021_25 = float(np.nanmean(y_b[mask]))
            avg_a_2021_25 = float(np.nanmean(y_a[mask]))
            txt = (f"2021–2025 Avg growth from prior Budget — {avg_b_2021_25:.1f}%\n"
                   f"2021–2025 Avg growth from prior Actual — {avg_a_2021_25:.1f}%")
            ax.text(0.98, 1.05, txt,
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, lw=0.6))


    # X-axis: show every window label, angled
    tick_labels = [f"{y}→{y+1}" for y in base_years]
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=9)
    axes[-1].set_xlabel("Budget growth window (base FY → next FY)")


        # Vertical shading for odd base years on all four panels
    for ax in axes:
        ymin, ymax = ax.get_ylim()
        for xpos, y in zip(x, base_years):
            if int(y) % 2 == 1:
                ax.add_patch(
                    patches.Rectangle((xpos - 0.5, ymin),
                                      1.0, ymax - ymin,
                                      alpha=0.08, zorder=0)
                )

    fig.suptitle("Amherst — Operating Budget Growth from Prior Actual? Or Prior Budget?", y=0.99, fontsize=17)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close(fig)


# ------------------------ MAIN ------------------------

def main():
    ap = argparse.ArgumentParser(description="Story 2: Municipal overestimation, uniform %-increase across operating categories, and budget growth comparisons vs prior Budget vs prior Actual.")
    ap.add_argument("--input", required=True, help="Path to Excel file")
    ap.add_argument("--sheet", default="Sheet1", help="Worksheet name (default: Sheet1)")
    ap.add_argument("--tolerance", type=float, default=0.002, help="Uniform growth tolerance (abs diff). 0.002 = 0.2%")
    ap.add_argument("--out_dir", default="output", help="Directory for outputs")
    ap.add_argument("--range", dest="year_range", help="Restrict Year to START-END (inclusive), e.g. 2010-2026", default=None)
    args = ap.parse_args()

    inpath = Path(args.input)
    out_dir = Path(args.out_dir)

    if not inpath.exists():
        raise FileNotFoundError(inpath)

    df = load_df(inpath, args.sheet)

    # Optionally clip by year range
    rng = parse_year_range(args.year_range)
    if rng:
        df = apply_year_range(df, rng)

    # Sanity: ensure the categories we need exist
    missing_cats = [c for c in OPERATING_CATS if c not in df["Category"].unique()]
    if missing_cats:
        raise ValueError(f"Missing required operating categories: {missing_cats}")

    # Core computations
    var_df = compute_variances(df)
    gr_df = compute_budget_growth_rates(df)
    uniform_flag = flag_uniform_growth(gr_df, args.tolerance)
    summary = build_summary(df, gr_df, uniform_flag)

    # Save summary CSV
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "story2_municipal_summary.csv"
    summary.to_csv(csv_path, index=False)

    # Charts
    plot_municipal_variance(df, out_dir / "story2_municipal_variance.png")
    plot_budget_growth_rates(df, gr_df, out_dir / "story2_budget_growth_rates.png")
    plot_growth_from_budget_vs_actual(df, out_dir / "story2_growth_budget_vs_actual.png")

    # Console digest
    print("=== Story 2: Municipal Overestimation & Same-% Increase ===")
    # Years with municipal under-spend (Actual < Budget)
    under_years = var_df[var_df["MunicipalVarPct"] < 0][["Year", "MunicipalVarPct"]]
    if not under_years.empty:
        pairs = ", ".join([f"{int(y)}: {pct*100:.1f}%" for y, pct in under_years.values])
        print(f"Years Municipal Actual < Budget: {pairs}")
    else:
        print("No years with Municipal Actual < Budget found.")

    # Uniform-growth years
    uniform_years = gr_df.loc[uniform_flag, "Year"].astype(int).tolist()
    print(f"Uniform-growth (≈ same % increase across operating categories) base years → next: {uniform_years or 'None'}")

    print(f"Saved CSV → {csv_path}")
    print(f"Saved charts → {out_dir}")

if __name__ == "__main__":
    main()
