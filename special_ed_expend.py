"""Visualize Massachusetts special education spending trends for selected districts."""

from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import StrMethodFormatter
from matplotlib.patches import Rectangle

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "special-ed-spending-trends.xlsx"
OUTPUT_DIR = BASE_DIR / "output"
TARGET_DISTRICTS = [
    "Amherst",
    "Amherst-Pelham",
    "Leverett",
    "Pelham",
    "Shutesbury",
]
DISTRICT_ALIASES: Dict[str, List[str]] = {
    "Amherst-Pelham": ["Amherst Pelham"],
}
SUBCATEGORY_COLUMNS: Dict[str, str] = {
    "indistrict_expend_teaching": "In-district teaching",
    "indistrict_expend_other_instructional": "In-district other instructional",
    "indistrict_expend_transportation": "In-district transportation",
    "out_of_district_expend_ma_public_schools_and_collaboratives": "Out-of-district MA public & collaboratives",
    "out_of_district_expend_mass_private_and_out_of_state_schools": "Out-of-district private & out-of-state",
    "out_of_district_expend_transportation": "Out-of-district transportation",
    "other_expend_non_public_health_services": "Non-public health services",
    "other_expend_grants_and_revolving": "Grants & revolving funds",
}
BAR_COLOR = "#f4a261"
SMALL_BAR_COLOR = "#fbd1a2"
LINE_COLOR = "#1f77b4"
CALLOUT_TEXT_COLOR = "#0f1f3d"
FOOTER_TEXT = "Data source: https://www.doe.mass.edu/research/radar/"
GROWTH_HEADER = "SPED expend total growth\nFY15–24"
AXIS_LABEL_FONTSIZE = 12
TITLE_FONTSIZE = 13
CALLOUT_FONTSIZE = 16
GROWTH_FONTSIZE = 18
ROLLING_WINDOW = 3

# Orange growth column styling
GROWTH_BOX_COLOR = "#f5a86d"      # soft orange
GROWTH_BOX_ALPHA = 0.10           # faint
GROWTH_BOX_PAD_RIGHT = 0.07       # base pad past numbers (axes coords)
GROWTH_BOX_SHIFT = 0.04           # shift column right (axes coords)
GROWTH_BOX_CHAR_PAD = 0.006       # extra width per character in growth string
NAME_X = 1.01                     # x for vertically rotated district names (axes coords)
WHITE_SPACER = 0.012              # thin white gutter between names and box
K_THRESHOLD_M = 1.0               # use K units when |Δ| < $1M


def load_data() -> pd.DataFrame:
    """Load spreadsheet and keep rows for the target districts."""
    df = pd.read_excel(DATA_PATH, header=4)
    alias_map = {
        alias: canonical
        for canonical, aliases in DISTRICT_ALIASES.items()
        for alias in aliases
    }
    valid_names = set(TARGET_DISTRICTS) | set(alias_map)
    df = df[df["district_name"].isin(valid_names)].copy()
    if alias_map:
        df["district_name"].replace(alias_map, inplace=True)
    df["fiscal_year"] = df["fiscal_year"].astype(int)
    df.sort_values(["district_name", "fiscal_year"], inplace=True)
    return df


def compute_growth_text(pivot_millions: pd.DataFrame) -> str:
    """Return a string like '+3.6M (+68%)' or '+950K (+35%)' for FY15→FY24."""
    totals = pivot_millions.sum(axis=1)
    totals = totals[totals.index >= 2015].dropna()
    if len(totals) < 2:
        return "n/a"
    start = totals.iloc[0]
    end = totals.iloc[-1]
    delta_m = end - start
    pct = (delta_m / start) if start not in (0, np.nan) else np.nan

    if abs(delta_m) < K_THRESHOLD_M:
        delta_k = delta_m * 1_000  # M → K
        delta_str = f"{delta_k:+.0f}K"
    else:
        delta_str = f"{delta_m:+.1f}M"

    pct_str = f" ({pct:+.0%})" if not np.isnan(pct) else ""
    return f"{delta_str}{pct_str}"


def plot_total_special_ed(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    for district in TARGET_DISTRICTS:
        district_df = df[df["district_name"] == district].dropna(subset=["sum_sped_expend"])
        if district_df.empty:
            continue
        ax.plot(
            district_df["fiscal_year"],
            district_df["sum_sped_expend"] / 1_000_000,
            marker="o",
            label=district,
        )
    ax.set_title("Total special education spending", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Fiscal year")
    ax.set_ylabel("Spending (millions USD)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.legend()
    ax.grid(alpha=0.3)
    # Trim padding at the ends
    year_min = int(df["fiscal_year"].min())
    year_max = int(df["fiscal_year"].max())
    ax.set_xlim(year_min - 0.5, year_max + 0.5)
    ax.margins(x=0)
    fig.tight_layout(rect=(0.03, 0.08, 0.97, 0.9))
    fig.text(0.03, 0.03, FOOTER_TEXT, ha="left", fontsize=12)
    return fig


def plot_sped_share(df: pd.DataFrame) -> plt.Figure:
    rows = len(TARGET_DISTRICTS)
    fig, axes = plt.subplots(rows, 1, figsize=(10, 3.4 * rows), sharex=True, sharey=True)
    if rows == 1:
        axes = [axes]

    left_label_ax = None
    right_label_ax = None
    for ax, district in zip(axes, TARGET_DISTRICTS):
        district_df = df[df["district_name"] == district].dropna(subset=["sped_pct_of_total", "sum_sped_expend"])
        if district_df.empty:
            continue

        years = district_df["fiscal_year"].to_numpy()
        share = district_df["sped_pct_of_total"].to_numpy() * 100
        dollars = district_df["sum_sped_expend"].to_numpy() / 1_000_000
        bar_color = SMALL_BAR_COLOR if np.nanmax(dollars) < 1 else BAR_COLOR

        ax2 = ax.twinx()
        ax2.bar(years, dollars, width=0.6, color=bar_color, alpha=0.85)
        if right_label_ax is None:
            ax2.set_ylabel("Special education spending (millions USD)", fontsize=AXIS_LABEL_FONTSIZE)
            right_label_ax = ax2
        else:
            ax2.set_ylabel("")
        ax2.yaxis.set_major_formatter(StrMethodFormatter("${x:.1f}M"))
        ax2.set_zorder(ax.get_zorder() - 1)

        ax.plot(years, share, color=LINE_COLOR, marker="o", linewidth=2)
        if left_label_ax is None:
            ax.set_ylabel("Share (%)", fontsize=AXIS_LABEL_FONTSIZE)
            left_label_ax = ax
        else:
            ax.set_ylabel("")
        ax.set_title(district, fontweight="bold", fontsize=TITLE_FONTSIZE)
        ax.grid(alpha=0.3)
        ax.set_xlim(years.min() - 0.5, years.max() + 0.5)
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)

    axes[-1].set_xlabel("Fiscal year")
    if left_label_ax is not None:
        left_label_ax.yaxis.set_label_coords(-0.07, 0.5)
    if right_label_ax is not None:
        right_label_ax.yaxis.set_label_coords(1.08, 0.5)
    fig.suptitle("Special education spending and share of total school budget", fontsize=TITLE_FONTSIZE + 1)
    fig.tight_layout(rect=(0.05, 0.1, 0.95, 0.92), h_pad=1.3)
    fig.text(0.03, 0.03, FOOTER_TEXT, ha="left", fontsize=12)
    return fig


def reshape_subcategories(df: pd.DataFrame) -> pd.DataFrame:
    long_df = df.melt(
        id_vars=["district_name", "fiscal_year"],
        value_vars=list(SUBCATEGORY_COLUMNS.keys()),
        var_name="subcategory",
        value_name="amount",
    )
    long_df["subcategory"] = long_df["subcategory"].map(SUBCATEGORY_COLUMNS)
    long_df.dropna(subset=["amount"], inplace=True)
    return long_df


def plot_subcategory_trends(long_df: pd.DataFrame, districts: Iterable[str]) -> plt.Figure:
    districts = list(districts)
    if not districts:
        raise ValueError("No districts supplied for plotting")

    categories = list(SUBCATEGORY_COLUMNS.values())
    cmap = plt.get_cmap("tab10")
    color_cycle = {category: cmap(idx % cmap.N) for idx, category in enumerate(categories)}

    base_height = 2.2
    legend_top = 0.93
    fig_height = base_height * len(districts) + 0.8
    fig, axes = plt.subplots(len(districts), 1, figsize=(14, fig_height))
    if len(districts) == 1:
        axes = [axes]

    last_axis = axes[-1]
    growth_x = 1.07

    # ---------- Compute a COMMON dynamic pad so every row's orange box aligns ----------
    projected_lengths = []
    for _district in districts:
        _ddf = long_df[long_df["district_name"] == _district]
        _pv = (
            _ddf.pivot(index="fiscal_year", columns="subcategory", values="amount")
            .sort_index()
            .reindex(columns=categories)
        ) / 1_000_000
        projected_lengths.append(len(compute_growth_text(_pv)))
    COMMON_DYN_PAD = GROWTH_BOX_PAD_RIGHT + max(0, max(projected_lengths, default=10) - 10) * GROWTH_BOX_CHAR_PAD
    # -----------------------------------------------------------------------------------

    for ax, district in zip(axes, districts):
        district_df = long_df[long_df["district_name"] == district]
        if district_df.empty:
            continue

        pivot = (
            district_df.pivot(index="fiscal_year", columns="subcategory", values="amount")
            .sort_index()
            .reindex(columns=categories)
        )
        pivot_millions = pivot / 1_000_000
        years = pivot.index.to_numpy()
        x = np.arange(len(years))
        width = 0.8 / len(categories)

        for idx, category in enumerate(categories):
            values = pivot_millions[category].to_numpy()
            offsets = (idx - (len(categories) - 1) / 2) * width
            ax.bar(
                x + offsets,
                np.nan_to_num(values, nan=0.0),
                width=width,
                label=category,
                color=color_cycle[category],
            )

        teaching_series = pd.Series(pivot_millions["In-district teaching"].to_numpy(), index=years)
        rolling = teaching_series.rolling(window=ROLLING_WINDOW, min_periods=1).mean()
        ax.plot(
            x,
            rolling.to_numpy(),
            color="#444444",
            linewidth=2.3,
            linestyle="--",
            label="In-district teaching trend",
            zorder=5,
        )

        ax.set_ylabel("Millions USD")
        ax.set_xticks(x)
        if ax is last_axis:
            ax.set_xticklabels(years.astype(int), rotation=45)
            ax.tick_params(axis="x", labelsize=12)
        else:
            ax.set_xticklabels([])
            ax.tick_params(axis="x", labelbottom=False)
        # Trim left/right padding around the first/last fiscal years
        ax.set_xlim(-0.5, len(years) - 0.5)
        ax.margins(x=0)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", length=0)

        growth_text = compute_growth_text(pivot_millions)

        # Faint orange column behind the growth numbers (full-height box)
        box_left = NAME_X + WHITE_SPACER + GROWTH_BOX_SHIFT
        box_right = growth_x + COMMON_DYN_PAD + GROWTH_BOX_SHIFT
        ax.add_patch(
            Rectangle(
                (box_left, -0.02),
                box_right - box_left,
                1.04,
                transform=ax.transAxes,
                facecolor=GROWTH_BOX_COLOR,
                alpha=GROWTH_BOX_ALPHA,
                edgecolor="none",
                zorder=0,
                clip_on=False,
            )
        )

        # Thin white spacer between names (at x≈1.01) and the orange box
        ax.add_patch(
            Rectangle(
                (NAME_X, -0.02),
                WHITE_SPACER,
                1.04,
                transform=ax.transAxes,
                facecolor="white",
                edgecolor="none",
                zorder=1,
                clip_on=False,
            )
        )

        # District name (vertical) and growth text
        ax.text(
            1.01,
            0.5,
            district,
            transform=ax.transAxes,
            rotation=-90,
            ha="center",
            va="center",
            fontweight="bold",
            fontsize=CALLOUT_FONTSIZE,
            color=CALLOUT_TEXT_COLOR,
            clip_on=False,
        )
        ax.text(
            growth_x,
            0.5,
            growth_text,
            transform=ax.transAxes,
            rotation=0,
            ha="left",
            va="center",
            fontweight="bold",
            fontsize=GROWTH_FONTSIZE,
            color=CALLOUT_TEXT_COLOR,
            clip_on=False,
        )

    # Layout + legend
    fig.subplots_adjust(left=0.08, right=growth_x + 0.1, top=legend_top - 0.07, bottom=0.12, hspace=0.08)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(0.5, legend_top),
        frameon=False,
        fontsize=10,
        columnspacing=0.9,
    )

    # -------- Center the 2-line header over the orange column (in FIGURE coords) --------
    ref_ax = axes[0]
    ax_bbox = ref_ax.get_position()          # figure fractions
    ax_left, ax_width = ax_bbox.x0, ax_bbox.width
    fig_box_left = ax_left + ax_width * (NAME_X + WHITE_SPACER + GROWTH_BOX_SHIFT)
    fig_box_right = ax_left + ax_width * (growth_x + COMMON_DYN_PAD + GROWTH_BOX_SHIFT)
    header_center_x = 0.5 * (fig_box_left + fig_box_right)

    fig.text(
        header_center_x,
        legend_top,
        GROWTH_HEADER,
        ha="center",
        va="center",
        fontsize=GROWTH_FONTSIZE,
        fontweight="bold",
        transform=fig.transFigure,
    )
    # ------------------------------------------------------------------------------------

    fig.suptitle("Special education subcategory expenses", fontsize=17, fontweight="bold", y=legend_top + 0.045)
    fig.supxlabel("Fiscal year", x=0.44, y=0.05, fontsize=14, fontweight="bold")
    fig.text(0.08, 0.025, FOOTER_TEXT, ha="left", fontsize=12)
    return fig


def main() -> None:
    plt.style.use("seaborn-v0_8")
    df = load_data()
    total_fig = plot_total_special_ed(df)
    share_fig = plot_sped_share(df)
    subcategory_fig = plot_subcategory_trends(reshape_subcategories(df), TARGET_DISTRICTS)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure_paths = {
        "total": OUTPUT_DIR / "special_ed_total_spending.png",
        "share": OUTPUT_DIR / "special_ed_spending_share.png",
        "subcategories": OUTPUT_DIR / "special_ed_subcategories.png",
    }

    total_fig.savefig(figure_paths["total"], dpi=300, bbox_inches="tight")
    share_fig.savefig(figure_paths["share"], dpi=300, bbox_inches="tight")
    subcategory_fig.savefig(figure_paths["subcategories"], dpi=300, bbox_inches="tight")

    for fig in (total_fig, share_fig, subcategory_fig):
        plt.close(fig)

    for label, path in figure_paths.items():
        print(f"Saved {label} figure to {path}")


if __name__ == "__main__":
    main()
