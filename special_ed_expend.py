"""Visualize Massachusetts special education spending trends for selected districts."""

from pathlib import Path
from typing import Dict, Iterable, List
import re
import shutil
import tempfile
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import StrMethodFormatter
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

# -------------------- Paths, constants, styling --------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "special-ed-spending-trends.xlsx"
CB_DATA_PATH = BASE_DIR / "data" / "ma-schools-circuit-breaker-history.xlsx"
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

# Colors for CB figure
BLUE = "#1f77b4"
SPEND_GRAY = "#444444"
CIRCLE_GREEN = "#1b7a1b"
CIRCLE_EDGE = "#115511"


# -------------------- Utilities --------------------
def _read_excel_resilient(path: Path, tries: int = 3, sleep_sec: float = 0.6) -> pd.DataFrame:
    """Read an Excel file even if OneDrive/Excel briefly locks it."""
    last_err = None
    for _ in range(tries):
        try:
            return pd.read_excel(path)
        except PermissionError as e:
            last_err = e
            with tempfile.NamedTemporaryFile(delete=False, suffix=path.suffix) as tmp:
                tmp_path = Path(tmp.name)
            try:
                shutil.copy2(path, tmp_path)
                try:
                    df = pd.read_excel(tmp_path)
                    return df
                finally:
                    try:
                        tmp_path.unlink(missing_ok=True)
                    except Exception:
                        pass
            except PermissionError:
                time.sleep(sleep_sec)
            except FileNotFoundError:
                raise
    raise PermissionError(f"Could not read '{path}' due to file locking") from last_err


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase columns, collapse non-alphanumerics to underscores."""
    def norm(c: str) -> str:
        c = c.strip().lower()
        c = re.sub(r"[^a-z0-9]+", "_", c)
        return c.strip("_")
    return df.rename(columns={c: norm(c) for c in df.columns})


def _pick(df: pd.DataFrame, candidates: List[str]) -> str | None:
    """Return the first normalized column present from candidates."""
    for name in candidates:
        if name in df.columns:
            return name
    return None


# -------------------- SPED spending core --------------------
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
        # avoid chained-assignment FutureWarning
        df["district_name"] = df["district_name"].replace(alias_map)
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

        # Bars on right y-axis
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

        # Share line on left y-axis
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

    # Shared legend under the title
    share_proxy = Line2D([], [], color=LINE_COLOR, marker="o", linewidth=2, label="SPED share of total (%)")
    spend_proxy = Patch(facecolor=BAR_COLOR, alpha=0.85, label="SPED spending ($M)")
    fig.legend(
        [share_proxy, spend_proxy],
        ["SPED share of total (%)", "SPED spending ($M)"],
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.97),
        frameon=False,
        columnspacing=1.2,
    )

    fig.suptitle("Special education spending and share of total school budget", fontsize=TITLE_FONTSIZE + 1, y=0.995)

    # Leave room for legend/title
    fig.tight_layout(rect=(0.05, 0.08, 0.95, 0.90), h_pad=1.3)
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

    # ---------- COMMON dynamic pad so every row's orange box aligns ----------
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
    # -----------------------------------------------------------------------

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
        ax.set_xlim(-0.5, len(years) - 0.5)
        ax.margins(x=0)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", length=0)

        growth_text = compute_growth_text(pivot_millions)

        # Faint orange column behind the growth numbers
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
        # Thin white spacer between names and orange box
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

        # District name (vertical) + growth text
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
    fig.subplots_adjust(left=0.08, right=growth_x + 0.1, top=0.93 - 0.07, bottom=0.12, hspace=0.08)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(0.5, 0.93),
        frameon=False,
        fontsize=10,
        columnspacing=0.9,
    )

    ref_ax = axes[0]
    ax_bbox = ref_ax.get_position()
    ax_left, ax_width = ax_bbox.x0, ax_bbox.width
    fig_box_left = ax_left + ax_width * (NAME_X + WHITE_SPACER + GROWTH_BOX_SHIFT)
    fig_box_right = ax_left + ax_width * (1.07 + COMMON_DYN_PAD + GROWTH_BOX_SHIFT)
    header_center_x = 0.5 * (fig_box_left + fig_box_right)

    fig.text(
        header_center_x,
        0.93,
        GROWTH_HEADER,
        ha="center",
        va="center",
        fontsize=GROWTH_FONTSIZE,
        fontweight="bold",
        transform=fig.transFigure,
    )

    fig.suptitle("Special education subcategory expenses", fontsize=17, fontweight="bold", y=0.975)
    fig.supxlabel("Fiscal year", x=0.44, y=0.05, fontsize=14, fontweight="bold")
    fig.text(0.08, 0.025, FOOTER_TEXT, ha="left", fontsize=12)
    return fig


# -------------------- Circuit Breaker: robust load (lean columns OK) --------------------
def load_circuit_breaker() -> pd.DataFrame:
    """
    Load circuit-breaker history with flexible headers.

    We PREFER reimbursement finance year:
      fiscal_year := reimb_fiscal_year, else fallback to 'fiscal_year'/'fy'/'year',
      else last resort 'sd_fiscal_year'.

    Eligible counts:
      If only sd_fiscal_year is present, shift eligible + net-claim forward by +1 to align
      with reimbursement year. If eligible counts are already aligned to reimbursement
      year (i.e., only reimb_fiscal_year exists), no shift is applied.

    Reimbursement dollars:
      Use 'cb_reimb_used' if present; else prefer 'reimb_total_adjusted_reimb';
      else 'reimb_total_reimbursement'. If none present, remains NaN.
    """
    raw = _read_excel_resilient(CB_DATA_PATH)
    df = _normalize_cols(raw)

    # Map whatever variants exist
    rename_map: Dict[str, str] = {}

    def add_first(canonical: str, keys: List[str]):
        found = _pick(df, keys)
        if found:
            rename_map[found] = canonical

    add_first("district_name", [
        "district_name", "district", "organization_name", "org_name", "districts", "district_name__text"
    ])
    add_first("reimb_fiscal_year", [
        "reimb_fiscal_year", "reimbursement_fiscal_year", "reimb_fy", "reimbursement_fy"
    ])
    add_first("fiscal_year_fallback", ["fiscal_year", "fy", "fiscalyear", "year"])
    add_first("sd_fiscal_year", ["sd_fiscal_year", "student_data_fiscal_year", "enrollment_fiscal_year"])

    # Eligible / claims (optional)
    add_first("eligible_students", ["eligible_students", "sd_eligible_students_claimed", "eligible_students_claimed"])
    add_first("sd_net_claim", ["sd_net_claim", "net_claim", "net_claim_amount"])

    # Reimbursement totals
    add_first("cb_reimb_used", ["cb_reimb_used"])  # allow precomputed field
    add_first("reimb_total_adjusted_reimb", ["reimb_total_adjusted_reimb", "adjusted_reimbursement", "total_adjusted_reimbursement", "reimb_adjusted", "adjusted_total"])
    add_first("reimb_total_reimbursement", ["reimb_total_reimbursement", "total_reimbursement", "reimbursement_total", "reimb_total", "total_reimb"])

    # Need at least district + some year
    if "district_name" not in rename_map or not any(
        k in rename_map for k in ("reimb_fiscal_year", "fiscal_year_fallback", "sd_fiscal_year")
    ):
        cols_preview = ", ".join(df.columns[:12])
        raise ValueError(
            "Circuit-breaker file missing district and/or year columns. "
            f"Saw columns: {cols_preview} ..."
        )

    df = df.rename(columns=rename_map)

    # Ensure numerics for known fields if present
    for c in ["reimb_fiscal_year", "fiscal_year_fallback", "sd_fiscal_year",
              "eligible_students", "sd_net_claim",
              "cb_reimb_used", "reimb_total_adjusted_reimb", "reimb_total_reimbursement"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Build reimbursement dollars with a simple, resilient chain
    if "cb_reimb_used" in df.columns:
        cb_used = df["cb_reimb_used"]
    else:
        cb_used = pd.Series(np.nan, index=df.index)
        if "reimb_total_adjusted_reimb" in df.columns:
            cb_used = cb_used.where(cb_used.notna(), df["reimb_total_adjusted_reimb"])
        if "reimb_total_reimbursement" in df.columns:
            cb_used = cb_used.where(cb_used.notna(), df["reimb_total_reimbursement"])
    df["cb_reimb_used"] = cb_used

    # Determine plotting finance year
    fiscal_year = df.get("reimb_fiscal_year")
    if fiscal_year is None or fiscal_year.isna().all():
        fiscal_year = df.get("fiscal_year_fallback")
    if fiscal_year is None or fiscal_year.isna().all():
        fiscal_year = df.get("sd_fiscal_year")  # last resort
    df["fiscal_year"] = fiscal_year

    # Eligible students alignment:
    # If we only have sd_fiscal_year for eligible counts, shift to reimb year (+1).
    core = df[["district_name", "fiscal_year", "cb_reimb_used"]].copy()
    core = core.dropna(subset=["fiscal_year"])
    core["fiscal_year"] = core["fiscal_year"].astype("Int64").astype(int)

    elig_series = None
    if "eligible_students" in df.columns:
        # If we have sd_fiscal_year and no reimb_fiscal_year for elig entries, assume sd-year and shift +1.
        if "sd_fiscal_year" in df.columns and ("reimb_fiscal_year" not in df.columns):
            tmp = df[["district_name", "sd_fiscal_year", "eligible_students"]].dropna(subset=["sd_fiscal_year"])
            tmp = tmp.rename(columns={"sd_fiscal_year": "sd_year"})
            tmp["fiscal_year"] = tmp["sd_year"].astype("Int64") + 1
            elig_series = tmp.groupby(["district_name", "fiscal_year"], as_index=False)["eligible_students"].sum()
        else:
            # Assume values are already aligned to fiscal_year
            tmp = df[["district_name", "fiscal_year", "eligible_students"]].dropna(subset=["fiscal_year"])
            elig_series = tmp.groupby(["district_name", "fiscal_year"], as_index=False)["eligible_students"].sum()

    if elig_series is not None:
        core = core.merge(elig_series, on=["district_name", "fiscal_year"], how="left")
    else:
        core["eligible_students"] = np.nan

    # Apply aliases / tag targets
    alias_map = {alias: canonical for canonical, aliases in DISTRICT_ALIASES.items() for alias in aliases}
    core["district_name"] = core["district_name"].replace(alias_map)
    core["is_target"] = core["district_name"].isin(TARGET_DISTRICTS)

    # Final schema
    out = core[["district_name", "fiscal_year", "eligible_students", "cb_reimb_used", "is_target"]]
    return out.sort_values(["district_name", "fiscal_year"]).reset_index(drop=True)


# -------------------- Circuit Breaker figure --------------------
from matplotlib.lines import Line2D  # keep near other imports

def plot_cb_timeseries(sped_df: pd.DataFrame, cb_df: pd.DataFrame) -> plt.Figure:
    """
    Per target district:
      - LEFT y-axis ($K): blue bars = Reimbursement; dashed line = SPED spend
      - Eligible students: solid dark-green circles with white numbers
        • vertical position: staggered in a mid band (panel-local)
        • size: GLOBAL scaling across all panels (area ~ sqrt(count))
      - No right axis. Shared legend below title. Vertical FY guides only.
      - If CB/eligible exist, drop earlier years; otherwise show spend across its range.
      - Labels above bars show reimbursement dollars ($, nearest).
    """
    rows = len(TARGET_DISTRICTS)
    fig, axes = plt.subplots(rows, 1, figsize=(12, 3.6 * rows), sharex=True)
    if rows == 1:
        axes = [axes]

    BLUE = "#1f77b4"
    SPEND_GRAY = "#444444"
    CIRCLE_GREEN = "#1b7a1b"
    CIRCLE_EDGE = "#115511"

    # ---------- GLOBAL circle-size scaling (across all panels) ----------
    cb_targets = cb_df[cb_df["district_name"].isin(TARGET_DISTRICTS)]
    e_all = cb_targets["eligible_students"].dropna().to_numpy()
    if e_all.size == 0:
        sqrt_min, sqrt_max = 0.0, 1.0
    else:
        # area-perception friendly: scale by sqrt
        sqrt_min = float(np.sqrt(max(np.min(e_all), 0)))
        sqrt_max = float(np.sqrt(np.max(e_all)))
        if np.isclose(sqrt_max, sqrt_min):
            sqrt_max = sqrt_min + 1.0  # avoid divide-by-zero
    # Visual limits so circles never get too big/small
    PAD_MIN, PAD_MAX = 0.26, 0.52     # bbox "circle,pad"
    FONTSZ_MIN, FONTSZ_MAX = 9, 12    # font size
    # -------------------------------------------------------------------

    proxy_handles = None

    for ax, district in zip(axes, TARGET_DISTRICTS):
        s = (
            sped_df[sped_df["district_name"] == district][["fiscal_year", "sum_sped_expend"]]
            .dropna(subset=["fiscal_year"])
            .rename(columns={"sum_sped_expend": "sped_spend"})
        )
        c = cb_df[cb_df["district_name"] == district][
            ["fiscal_year", "cb_reimb_used", "eligible_students"]
        ]

        # If any CB/eligible present, trim spend to finance window start
        if not c.dropna(subset=["fiscal_year", "cb_reimb_used", "eligible_students"], how="all").empty:
            min_finance_year = int(c["fiscal_year"].min())
            s = s[s["fiscal_year"] >= min_finance_year]

        # Outer merge keeps spend even if CB/elig missing (e.g., Pelham)
        df = pd.merge(s, c, on="fiscal_year", how="outer").sort_values("fiscal_year")
        if df.empty:
            ax.set_title(f"{district} (no data)")
            ax.axis("off")
            continue

        # Drop leading rows with neither reimbursement nor eligible students
        signal_mask = df["cb_reimb_used"].notna() | df["eligible_students"].notna()
        if signal_mask.any():
            first_idx = signal_mask.idxmax()
            df = df.loc[first_idx:]

        years   = df["fiscal_year"].astype(int).to_numpy()
        reimb   = df["cb_reimb_used"].to_numpy()               # dollars (for labels)
        reimb_k = (df["cb_reimb_used"] / 1_000).to_numpy()     # $K for plotting
        sped_k  = (df["sped_spend"]    / 1_000).to_numpy()
        elig    = df["eligible_students"].to_numpy()

        # Bars: Reimbursement ($K)
        bars = ax.bar(years, np.nan_to_num(reimb_k, nan=0.0), width=0.65, color=BLUE, alpha=0.95, label="Reimbursement ($K)")

        # Label bars with nearest-dollar values
        ymin_init, ymax_init = ax.get_ylim()
        bar_pad_k = max((ymax_init - ymin_init) * 0.015, 3.0)  # ≥ 3K pad
        for x, yk, dollars in zip(years, reimb_k, reimb):
            if np.isnan(yk) or np.isclose(yk, 0.0):
                continue
            ax.text(
                x, yk + bar_pad_k,
                f"${dollars:,.0f}",
                ha="center", va="bottom",
                fontsize=9, color="#0f1f3d",
                zorder=6,
            )

        # Line: SPED spend ($K), dashed
        spend_line, = ax.plot(years, sped_k, linestyle="--", linewidth=2.0, color=SPEND_GRAY, marker=None, label="SPED spend ($K)")

        # Cosmetics
        ax.set_ylabel("Dollars ($K)")
        ax.set_title("Pelham" if district == "Pelham" else district, fontweight="bold")
        ax.set_xlim(years.min() - 0.5, years.max() + 0.5)
        ax.grid(False)

        # Vertical FY guides behind series
        ymin, ymax = ax.get_ylim()
        for x in years:
            ax.vlines(x, ymin, ymax, colors="white", linewidth=1.2, zorder=0)
        ax.set_ylim(ymin, ymax)

        # --- Eligible students: GLOBAL size scaling, panel-local vertical staggering ---
        y0, y1 = ax.get_ylim()
        yrange = max(y1 - y0, 1e-9)
        band_low, band_high = 0.35, 0.75  # vertical mid band

        # Local range for vertical staggering only
        valid_e_local = np.array([e for e in elig if pd.notna(e)])
        e_min_local = float(valid_e_local.min()) if valid_e_local.size else 0.0
        e_max_local = float(valid_e_local.max()) if valid_e_local.size else 1.0
        e_span_local = (e_max_local - e_min_local) if (e_max_local - e_min_local) != 0 else 1.0

        for X, e in zip(years, elig):
            if pd.isna(e):
                continue

            # size scale: GLOBAL sqrt normalization
            s_e = (np.sqrt(float(e)) - sqrt_min) / (sqrt_max - sqrt_min)
            s_e = float(np.clip(s_e, 0.0, 1.0))

            pad = PAD_MIN + (PAD_MAX - PAD_MIN) * s_e
            fsz = FONTSZ_MIN + (FONTSZ_MAX - FONTSZ_MIN) * s_e

            # vertical position: panel-local staggering to avoid collisions
            e_norm_local = (float(e) - e_min_local) / e_span_local
            frac = band_low + e_norm_local * (band_high - band_low)
            y_pos = y0 + frac * yrange

            ax.text(
                X, y_pos, f"{int(round(e))}",
                ha="center", va="center",
                fontsize=fsz, color="white",
                bbox=dict(boxstyle=f"circle,pad={pad:.2f}", facecolor=CIRCLE_GREEN, edgecolor=CIRCLE_EDGE, linewidth=1.2),
                zorder=5,
            )

        # Build shared legend proxies once
        if proxy_handles is None:
            eligible_proxy = Line2D([], [], linestyle="none", marker="o", markersize=8,
                                    markerfacecolor=CIRCLE_GREEN, markeredgecolor=CIRCLE_EDGE,
                                    label="Eligible students (count)")
            proxy_handles = [bars, spend_line, eligible_proxy]

        # Pelham annotation if no reimbursement at all
        if district == "Pelham":
            no_reimb = np.all(np.isnan(reimb_k)) or (np.nan_to_num(reimb_k).sum() == 0)
            if no_reimb:
                ax.text(
                    0.5, 0.5,
                    "Pelham has no records in Circuit Breaker Reimbursement Payments data during this period",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=11,
                    color="#7a1f1f",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff6f6", edgecolor="#e0b3b3")
                )

    # Shared x-label + title + legend
    axes[-1].set_xlabel("Fiscal year")

    fig.suptitle("Special Ed Spending, Eligible Students, and Circuit Breaker Reimbursements",
                 fontsize=16, fontweight="bold", y=0.985)

    if proxy_handles is not None:
        labels = [h.get_label() for h in proxy_handles]
        fig.legend(
            proxy_handles, labels,
            loc="upper center", ncol=3,
            bbox_to_anchor=(0.5, 0.955),  # just under the title
            frameon=False,
            columnspacing=1.2,
        )

    # Make the bottom-axis tick labels (fiscal years) bigger on every subplot
    for ax in axes:
        ax.tick_params(axis="x", labelsize=12)   # <- bump this number to taste
    # Make the shared x-axis label bigger
    axes[-1].set_xlabel("Fiscal year", fontsize=14, labelpad=4)  # <- bump fontsize/labelpad
    fig.tight_layout(rect=(0.04, 0.06, 0.98, 0.90), h_pad=1.1)
    fig.text(0.04, 0.03, FOOTER_TEXT, ha="left", fontsize=11)
    return fig


# -------------------- Main --------------------
def main() -> None:
    plt.style.use("seaborn-v0_8")

    # SPED spending data + standard figs
    df = load_data()
    total_fig = plot_total_special_ed(df)
    share_fig = plot_sped_share(df)
    subcategory_fig = plot_subcategory_trends(reshape_subcategories(df), TARGET_DISTRICTS)

    # Circuit breaker time-series
    cb = load_circuit_breaker()
    cb_timeseries_fig = plot_cb_timeseries(df, cb)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure_paths = {
        "total": OUTPUT_DIR / "special_ed_total_spending.png",
        "share": OUTPUT_DIR / "special_ed_spending_share.png",
        "subcategories": OUTPUT_DIR / "special_ed_subcategories.png",
        "cb_timeseries": OUTPUT_DIR / "special_ed_cb_timeseries.png",
    }

    total_fig.savefig(figure_paths["total"], dpi=300, bbox_inches="tight")
    share_fig.savefig(figure_paths["share"], dpi=300, bbox_inches="tight")
    subcategory_fig.savefig(figure_paths["subcategories"], dpi=300, bbox_inches="tight")
    cb_timeseries_fig.savefig(figure_paths["cb_timeseries"], dpi=300, bbox_inches="tight")

    for fig in (total_fig, share_fig, subcategory_fig, cb_timeseries_fig):
        plt.close(fig)

    for label, path in figure_paths.items():
        print(f"Saved {label} figure to {path}")


if __name__ == "__main__":
    main()
