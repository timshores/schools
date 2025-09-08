# Bigger, clearer typography for both paired figures.
# - Bump global font sizes via rcParams (titles/labels/ticks/legend).
# - Slightly larger figure and markers for readability.
# - Re-render nominal & inflation-adjusted with shared bar y-scale, thermo own scale.
#
# Outputs:
# - paired_bars_and_thermometer_nominal_v6_bigfonts.png
# - paired_bars_and_thermometer_inflation_adjusted_v6_bigfonts.png

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter, AutoMinorLocator

# ----------------------
# CONFIG
# ----------------------
DATA_DIR   = Path.cwd() / "data"
OUTPUT_DIR = Path.cwd() / "output"
EXCEL_FILE = DATA_DIR / "E2C_Hub_MA_DESE_Data.xlsx"
REGIONS_CSV = DATA_DIR / "ma_districts_with_eohhs_regions_complete.csv"
CPI_CSV    = DATA_DIR / "cpi_annual.csv"

# Fallbacks for this environment
if not EXCEL_FILE.exists():
    EXCEL_FILE = Path("/mnt/data/E2C_Hub_MA_DESE_Data_08122025.xlsx")
    DATA_DIR = Path("/mnt/data")
    OUTPUT_DIR = Path("/mnt/data/output")
    CPI_CSV = DATA_DIR / "cpi_annual.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Font scaling (tweak here if you want even bigger)
plt.rcParams.update({
    "font.size": 14,          # base
    "axes.titlesize": 20,     # panel titles
    "axes.labelsize": 16,     # y-axis label
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})

TARGET_SUBCATS = [
    "Administration",
    "Guidance, Counseling and Testing",
    "Instructional Leadership",
    "Instructional Materials, Equipment and Technology",
    "Insurance, Retirement Programs and Other",
    "Operations and Maintenance",
    "Other Teaching Services",
    "Professional Development",
    "Pupil Services",
    "Teachers",
]

ABBR = {
    "Administration": "Admin",
    "Guidance, Counseling and Testing": "Guidance/\nTesting",
    "Instructional Leadership": "Instr\nLead",
    "Instructional Materials, Equipment and Technology": "Materials/\nEquip/Tech",
    "Insurance, Retirement Programs and Other": "Insurance/\nRetire",
    "Operations and Maintenance": "Ops & Maint",
    "Other Teaching Services": "Other Teach",
    "Professional Development": "Prof Dev",
    "Pupil Services": "Pupil Svcs",
    "Teachers": "Teachers",
}

DISTRICTS = {
    6050000: "Amherst-Pelham",
    80000:   "Amherst",
    1540000: "Leverett",
    2300000: "Pelham",
    2720000: "Shutesbury",
}

def compute_avg_change(df, value_col="IND_VALUE", year_col="SY", group_col="IND_SUBCAT"):
    out = []
    for key, g in df.groupby(group_col):
        g = g.sort_values(year_col)
        if g.empty:
            continue
        start_year, end_year = int(g[year_col].iloc[0]), int(g[year_col].iloc[-1])
        start_val, end_val = float(g[value_col].iloc[0]), float(g[value_col].iloc[-1])
        years = end_year - start_year
        if years > 0:
            out.append((key, (end_val - start_val) / years, start_year, end_year))
    return pd.DataFrame(out, columns=[group_col, "Delta_per_year", "start_year", "end_year"])

def total_and_years_overall(df, value_col="IND_VALUE", year_col="SY"):
    if df.empty:
        return np.nan, 1
    s = df.groupby(year_col)[value_col].sum().sort_index()
    if len(s) < 2:
        return np.nan, 1
    years = int(s.index[-1] - s.index[0])
    return float(s.iloc[-1] - s.iloc[0]), max(years, 1)

def paired_bars_and_thermometer(comp_df, totals_by_series, years_by_series,
                                title_left, title_right, outfile, bars_ylim=None):
    series = comp_df.columns.tolist()
    n_series = len(series)
    x = list(range(len(comp_df)))
    bar_w = 0.8 / n_series

    # ---- Figure (constrained layout) ----
    fig = plt.figure(figsize=(19, 10), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.02, hspace=0.02)
    gs  = GridSpec(nrows=1, ncols=16, figure=fig,
                   width_ratios=[1]*12 + [0.7] + [0.22]*3, wspace=0.02)
    ax_left  = fig.add_subplot(gs[0, :12])
    ax_sp    = fig.add_subplot(gs[0, 12])
    ax_therm = fig.add_subplot(gs[0, 13:])

    # ---- Bars (left) ----
    for idx in x:
        if idx % 2 == 1:
            ax_left.axvspan(idx - 0.5, idx + 0.5, color="0.95", zorder=0)

    # y-limits: provided shared limits or data-driven with padding
    if bars_ylim is None:
        data_min = float(np.nanmin(comp_df.values))
        data_max = float(np.nanmax(comp_df.values))
        lo = min(0.0, data_min); hi = max(0.0, data_max)
        rng = max(1.0, hi - lo); pad = max(5.0, rng * 0.06)
        y_min, y_max = lo - pad, hi + pad
    else:
        y_min, y_max = bars_ylim
    ax_left.set_ylim(y_min, y_max)

    # Y-grid
    ax_left.set_axisbelow(True)
    ax_left.grid(which="major", axis="y", linestyle=":", linewidth=1.0, color="0.82")
    ax_left.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax_left.grid(which="minor", axis="y", linestyle=":", linewidth=0.5, color="0.90")

    # Bars + colors
    series_colors = {}
    for i, s in enumerate(series):
        offsets = [xi - 0.4 + i*bar_w + bar_w/2 for xi in x]
        vals = comp_df[s].values
        bars = ax_left.bar(offsets, vals, width=bar_w, label=s, zorder=2)
        if len(bars.patches):
            series_colors[s] = bars.patches[0].get_facecolor()

    # Top x labels (bigger)
    ax_left.set_xticks(x)
    abbr_labels = [ABBR.get(cat, cat) for cat in comp_df.index.tolist()]
    ax_left.set_xticklabels(abbr_labels)
    ax_left.tick_params(axis="x", bottom=False, labelbottom=False, top=True, labeltop=True)
    ax_left.spines["bottom"].set_visible(False)

    # Left y-axis only
    fmt = FuncFormatter(lambda v, pos: f"${v:,.0f}")
    ax_left.yaxis.set_major_formatter(fmt)
    ax_left.tick_params(axis="y", left=True, labelleft=True, right=False, labelright=False)
    ax_left.spines["left"].set_visible(True)
    ax_left.spines["right"].set_visible(False)
    ax_left.set_ylabel("Avg annual change (dollars per pupil per year)")
    ax_left.set_title(title_left, pad=12)
    handles, labels = ax_left.get_legend_handles_labels()

    # Spacer
    ax_sp.axis("off")

    # ---- Thermometer (right) ----
    vals_total = np.array([totals_by_series.get(s, np.nan) for s in series], dtype=float)
    vals_years = np.array([years_by_series.get(s, np.nan) for s in series], dtype=float)
    per_year = vals_total / vals_years  # position in $/yr

    # Therm own scale
    if np.all(np.isnan(per_year)):
        t_ymin, t_ymax = -1, 1
    else:
        tmin = np.nanmin(per_year); tmax = np.nanmax(per_year)
        tmin = min(0, tmin);        tmax = max(0, tmax)
        trng = max(1.0, tmax - tmin); tpad = max(10.0, trng * 0.08)
        t_ymin, t_ymax = tmin - tpad, tmax + tpad
    ax_therm.set_xlim(0, 1); ax_therm.set_ylim(t_ymin, t_ymax)

    # Thicker center line; no axes
    ax_therm.axvline(0.5, color="0.80", linewidth=7, zorder=0)
    for side in ["top", "right", "bottom", "left"]:
        ax_therm.spines[side].set_visible(False)
    ax_therm.set_xticks([]); ax_therm.set_yticks([])

    # Nudge overlaps
    def nudge(yvals, min_gap):
        order = np.argsort(yvals)
        ys = [yvals[i] for i in order]
        out, last = [], None
        for y in ys:
            out.append(y if last is None else max(y, last + min_gap))
            last = out[-1]
        y_new = np.zeros_like(yvals, dtype=float); moved = np.zeros_like(yvals, dtype=bool)
        for idx, i in enumerate(order):
            y_new[i] = out[idx]; moved[i] = abs(y_new[i] - yvals[i]) > 1e-9
        return y_new, moved

    t_min_gap = (t_ymax - t_ymin) * 0.06
    y_nudged, moved = nudge(per_year, min_gap=t_min_gap)

    x_dot, x_label, x_line_end = 0.5, 0.70, 0.66  # push text a bit farther with bigger font
    for s, y0, y1, tot in zip(series, per_year, y_nudged, vals_total):
        if np.isnan(y0):
            continue
        color = series_colors.get(s, None)
        ax_therm.plot([x_dot], [y0], marker="o", markersize=11, color=color, zorder=3)
        ax_therm.text(x_label, y1, f"{s}: ${tot:,.0f}", va="center", ha="left",
                      fontsize=12, color=color, zorder=4)
        if abs(y1 - y0) > 1e-9:
            ax_therm.plot([x_dot+0.02, x_line_end], [y0, y1], color=color, linewidth=1.8, zorder=3)

    ax_therm.set_title(title_right, pad=8)

    # Figure-level legend
    if handles and labels:
        fig.legend(handles, labels, ncol=min(len(series), 8), fontsize=12, frameon=False,
                   loc="lower center")

    fig.savefig(outfile, dpi=220, bbox_inches="tight")
    plt.close(fig)

# ----------------------
# Load per‑pupil data and compute
# ----------------------
expend_df = pd.read_excel(EXCEL_FILE, sheet_name="District Expend by Category")
pp = expend_df[expend_df["IND_SUBCAT"].isin(TARGET_SUBCATS)][["SY","DIST_CODE","IND_SUBCAT","IND_VALUE"]].copy()

# Western/State membership
if REGIONS_CSV.exists():
    r = pd.read_csv(REGIONS_CSV)
    dc = next((c for c in r.columns if c.lower().strip() in {"district_code","dist_code","code"}))
    rg = next((c for c in r.columns if c.lower().strip() in {"eohhs_region","region"}))
    regions_df = r[[dc, rg]].rename(columns={dc:"district_code", rg:"eohhs_region"}).copy()
    regions_df["district_code"] = regions_df["district_code"].astype(int)
else:
    regions_df = pd.read_excel(EXCEL_FILE, sheet_name="District Regions")[["district_code","eohhs_region"]].copy()
    regions_df["district_code"] = regions_df["district_code"].astype(int)

western_codes = regions_df.loc[regions_df["eohhs_region"].str.strip().eq("Western"), "district_code"].unique()

# ---------- NOMINAL ----------
series_map_nom = {}
totals_nom = {}
years_nom = {}

for code, name in DISTRICTS.items():
    ddf = pp[pp["DIST_CODE"]==code]
    if ddf.empty: continue
    series_map_nom[name] = compute_avg_change(ddf).set_index("IND_SUBCAT")["Delta_per_year"]
    tot, yrs = total_and_years_overall(ddf)
    totals_nom[name] = tot; years_nom[name] = yrs

western_mean = pp[pp["DIST_CODE"].isin(western_codes)].groupby(["SY","IND_SUBCAT"])["IND_VALUE"].mean().reset_index()
state_mean   = pp.groupby(["SY","IND_SUBCAT"])["IND_VALUE"].mean().reset_index()

series_map_nom["Western Avg"] = compute_avg_change(western_mean).set_index("IND_SUBCAT")["Delta_per_year"]
series_map_nom["State Avg"]   = compute_avg_change(state_mean).set_index("IND_SUBCAT")["Delta_per_year"]

totals_nom["Western Avg"], years_nom["Western Avg"] = total_and_years_overall(western_mean)
totals_nom["State Avg"],   years_nom["State Avg"]   = total_and_years_overall(state_mean)

comp_nom = pd.DataFrame(series_map_nom).reindex(sorted(TARGET_SUBCATS)).dropna(how="all")

state_series_nom = comp_nom.get("State Avg", pd.Series(dtype=float))
if len(state_series_nom):
    top2 = state_series_nom.sort_values(ascending=False).head(2)
    fast_nom = [ABBR.get(c, c) for c in top2.index]
    title_left_nom = fr"Avg annual \$ change per pupil — fastest: {fast_nom[0]}, {fast_nom[1]}"
else:
    title_left_nom = "Avg annual $ change per pupil"
title_right_nom = "Total change per pupil\n(entire period, dollars)"

# ---------- INFLATION-ADJUSTED ----------
cpi = pd.read_csv(CPI_CSV)
cpi = cpi.rename(columns=lambda s: s.strip().lower()).rename(columns={"year":"SY","cpi":"CPI"})
cpi = cpi.dropna(subset=["SY","CPI"]).copy()
cpi["SY"] = cpi["SY"].astype(int)
cpi["CPI"] = pd.to_numeric(cpi["CPI"], errors="coerce")
cpi = cpi.groupby("SY", as_index=False)["CPI"].mean()

latest_sy = int(pp["SY"].max())
cpi_latest = float(cpi.loc[cpi["SY"]==latest_sy, "CPI"].iloc[0])
cpi["deflator"] = cpi_latest / cpi["CPI"]

pp_real = pp.merge(cpi[["SY","deflator"]], on="SY", how="inner").copy()
pp_real["IND_VALUE"] = pp_real["IND_VALUE"] * pp_real["deflator"]

series_map_real = {}
totals_real = {}
years_real = {}

for code, name in DISTRICTS.items():
    ddf = pp_real[pp_real["DIST_CODE"]==code]
    if ddf.empty: continue
    series_map_real[name] = compute_avg_change(ddf).set_index("IND_SUBCAT")["Delta_per_year"]
    tot, yrs = total_and_years_overall(ddf)
    totals_real[name] = tot; years_real[name] = yrs

western_real = pp_real[pp_real["DIST_CODE"].isin(western_codes)].groupby(["SY","IND_SUBCAT"])["IND_VALUE"].mean().reset_index()
state_real   = pp_real.groupby(["SY","IND_SUBCAT"])["IND_VALUE"].mean().reset_index()

series_map_real["Western Avg"] = compute_avg_change(western_real).set_index("IND_SUBCAT")["Delta_per_year"]
series_map_real["State Avg"]   = compute_avg_change(state_real).set_index("IND_SUBCAT")["Delta_per_year"]

totals_real["Western Avg"], years_real["Western Avg"] = total_and_years_overall(western_real)
totals_real["State Avg"],   years_real["State Avg"]   = total_and_years_overall(state_real)

comp_real = pd.DataFrame(series_map_real).reindex(sorted(TARGET_SUBCATS)).dropna(how="all")

state_series_real = comp_real.get("State Avg", pd.Series(dtype=float))
if len(state_series_real):
    top2r = state_series_real.sort_values(ascending=False).head(2)
    fast_real = [ABBR.get(c, c) for c in top2r.index]
    title_left_real = fr"Inflation adjusted: Avg annual \$ change per pupil — fastest: {fast_real[0]}, {fast_real[1]}"
else:
    title_left_real = "Inflation adjusted: Avg annual $ change per pupil"
title_right_real = f"Total change per pupil\n(inflation-adjusted {latest_sy} $)"

# ---------- Shared bar y-scale across both figures ----------
all_bar_vals = np.concatenate([comp_nom.values.flatten(), comp_real.values.flatten()])
all_bar_vals = all_bar_vals[~np.isnan(all_bar_vals)]
if all_bar_vals.size:
    lo = min(0.0, float(all_bar_vals.min()))
    hi = max(0.0, float(all_bar_vals.max()))
    rng = max(1.0, hi - lo)
    pad = max(5.0, rng * 0.06)
    shared_ylim = (lo - pad, hi + pad)
else:
    shared_ylim = None

# ---------- Render both figures ----------
out_nom = OUTPUT_DIR / "paired_bars_and_thermometer_nominal_v6_bigfonts.png"
paired_bars_and_thermometer(comp_nom, totals_nom, years_nom,
                             title_left_nom, title_right_nom, out_nom,
                             bars_ylim=shared_ylim)

out_real = OUTPUT_DIR / "paired_bars_and_thermometer_inflation_adjusted_v6_bigfonts.png"
paired_bars_and_thermometer(comp_real, totals_real, years_real,
                             title_left_real, title_right_real, out_real,
                             bars_ylim=shared_ylim)

[str(out_nom), str(out_real)]
