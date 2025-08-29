#!/Users/tshores/anaconda3/bin/python

### Note that line 1 makes it executable, after changing perms:
### ### chmod +x /Users/tshores/Documents/pythong/enrollmentplot.py
### execute with: ./enrollmentplot.py

# Enrollment plot with annotated trend-lines
# Works with the file: ./data/arps_schools.xlsx

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
from matplotlib.ticker import FuncFormatter

# ----------------------
# Config
# ----------------------
FILE_PATH = "./data/arps_schools.xlsx"
ROLLING_YEARS = 5          # centered rolling window for trend
SHOW_LAG = False           # set True to overlay primary total shifted forward
LAG_YEARS = 5              # visual alignment idea for the feeder relationship

# ----------------------
# Load
# ----------------------
df = pd.read_excel(FILE_PATH, sheet_name="Sheet1")
df = df.sort_values(["District", "School Year"]).reset_index(drop=True)

# ----------------------
# Aggregations
# ----------------------
primary_df = df[df["Type"] == "Primary"]
secondary_df = df[df["Type"] == "Secondary"]

primary_total = (
    primary_df.groupby("School Year", as_index=False)["Total Enrollment"].sum()
    .rename(columns={"Total Enrollment": "Primary Total"})
)

secondary_total = (
    secondary_df.groupby("School Year", as_index=False)["Total Enrollment"].sum()
    .rename(columns={"Total Enrollment": "Secondary Total"})
)

# Trend lines: centered rolling mean = smooths out short-term bumps
primary_total["Primary Trend"] = (
    primary_total["Primary Total"].rolling(window=ROLLING_YEARS, center=True, min_periods=1).mean()
)
secondary_total["Secondary Trend"] = (
    secondary_total["Secondary Total"].rolling(window=ROLLING_YEARS, center=True, min_periods=1).mean()
)

# Optional: overlay a lagged primary series (to line up with secondary a few years later)
if SHOW_LAG:
    primary_total_lagged = primary_total.copy()
    primary_total_lagged["School Year"] = primary_total_lagged["School Year"] + LAG_YEARS


# ======================
# SIDE-BY-SIDE PLOTS + A-P TREND FORECAST FROM ELEMENTARY (+5y) TO 2030
# ======================

# --- Build stacked primary pivot (stable order for legend) ---
primary_pivot = (
    df[df["Type"] == "Primary"]
    .pivot(index="School Year", columns="District", values="Total Enrollment")
    .fillna(0)
    .sort_index()
)
preferred_order = ["Amherst", "Pelham", "Leverett", "Shutesbury"]
cols = [c for c in preferred_order if c in primary_pivot.columns] + \
       [c for c in primary_pivot.columns if c not in preferred_order]
primary_pivot = primary_pivot[cols]

# --- Area colors for primaries (Okabe–Ito palette) ---
area_palette = ["#E69F00", "#009E73", "#F0E442", "#D55E00"]  # orange, green, yellow, vermilion
area_colors = [area_palette[i % len(area_palette)] for i in range(primary_pivot.shape[1])]

# --- Line/Trend colors (matches your last script; purple for forecast) ---
COLORS = {
    "amherst_total": "#000000",  # A-P Total (bold black)
    "amherst_trend": "#009E73",  # A-P Trend (you can swap to your preferred light blue/gray)
    "primary_trend": "#E69F00",  # Elementary Trend (orange)
    "ap_trend_forecast": "#CC79A7",  # Forecast (PURPLE)
}

# --- Halo path effects so lines stay legible ---
halo       = [pe.Stroke(linewidth=4.0, foreground="white"), pe.Normal()]
halo_thick = [pe.Stroke(linewidth=6.0, foreground="white"), pe.Normal()]
halo_soft  = [pe.Stroke(linewidth=5.0, foreground=(1, 1, 1, 0.85)), pe.Normal()]

# --- Create side-by-side axes ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), sharey=True)

# ===== LEFT PANEL: stacked primaries + trends + A-P total =====
primary_pivot.plot.area(ax=ax1, stacked=True, linewidth=0, alpha=0.25, color=area_colors)

ax1.plot(
    secondary_total["School Year"], secondary_total["Secondary Total"],
    color=COLORS["amherst_total"], linewidth=4.0, label="Amherst-Pelham Total",
    zorder=8, path_effects=halo_thick
)
ax1.plot(
    primary_total["School Year"], primary_total["Primary Trend"],
    color=COLORS["primary_trend"], linestyle="--", linewidth=2.2,
    label=f"Elementary Trend ({ROLLING_YEARS}-yr centered)", zorder=9, path_effects=halo
)
ax1.plot(
    secondary_total["School Year"], secondary_total["Secondary Trend"],
    color=COLORS["amherst_trend"], linestyle="--", linewidth=2.8, alpha=0.6,
    label=f"Amherst-Pelham Trend ({ROLLING_YEARS}-yr centered)", zorder=10, path_effects=halo_soft
)

ax1.set_title("Stacked Elementary by District + Trends (1994–2025)")
ax1.set_xlabel("School Year")
ax1.set_ylabel("Total Enrollment")
ax1.grid(alpha=0.25)
ax1.set_facecolor("white")
for s in ax1.spines.values():
    s.set_alpha(0.4)

# Legend + LEFT annotation
area_handles = [Patch(facecolor=col, edgecolor='none', alpha=0.25, label=name)
                for name, col in zip(primary_pivot.columns, area_colors)]
line_handles = [
    Line2D([0], [0], color=COLORS["amherst_total"], linewidth=4.0, label="Amherst-Pelham Total"),
    Line2D([0], [0], color=COLORS["primary_trend"], linestyle="--", linewidth=2.2,
           label=f"Elementary Trend ({ROLLING_YEARS}-yr)"),
    Line2D([0], [0], color=COLORS["amherst_trend"], linestyle="--", linewidth=2.8, alpha=0.6,
           label=f"Amherst-Pelham Trend ({ROLLING_YEARS}-yr)"),
]
ax1.legend(handles=area_handles + line_handles, loc="best", ncol=2)

annotation_text_left = (
    f"Trend lines use a centered {ROLLING_YEARS}-year rolling mean:\n"
    "• For year t, average enrollment from t−k … t … t+k, k=(window−1)/2.\n"
    "• Smooths cohort bumps to reveal the long-run signal.\n"
    "• Endpoints shrink the window when data are limited (min_periods=1)."
)
ax1.annotate(
    annotation_text_left,
    xy=(0.01, 0.02), xycoords="axes fraction",
    va="bottom", ha="left", fontsize=9,
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, pad=0.5)
)

# ===== RIGHT PANEL: Trends only — Elementary (+LAG) ORANGE to 2030,
#                                  A-P GREEN through last year,
#                                  PURPLE forecast extends A-P via OLS on shifted Elementary =====

FORECAST_TO_YEAR = 2030
AP_FORECAST_COLOR = COLORS.get("ap_trend_forecast", "#CC79A7")  # purple

# Clear and restyle right axes
ax2.cla()
ax2.set_title(f"Trends Only: Elementary (+{LAG_YEARS}y) vs A-P Trend + Forecast")
ax2.set_xlabel("School Year")
ax2.grid(alpha=0.25)
ax2.set_facecolor("white")
for s in ax2.spines.values():
    s.set_alpha(0.4)

# --- Build series ---
# A-P trend (historical)
ap_trend = secondary_total[["School Year", "Secondary Trend"]].dropna().copy()
last_ap_year = int(ap_trend["School Year"].max())
ap_last_y = ap_trend.loc[ap_trend["School Year"].eq(last_ap_year), "Secondary Trend"].to_numpy().item()

# Elementary trend shifted +LAG (t maps to t+LAG)
elem_shifted = primary_total[["School Year", "Primary Trend"]].dropna().copy()
elem_shifted["School Year"] = elem_shifted["School Year"] + LAG_YEARS
colname = f"Elementary Trend (+{LAG_YEARS}y)"
elem_shifted = elem_shifted.rename(columns={"Primary Trend": colname})

# Draw Elementary (+5y) as continuous ORANGE line through 2030
elem_to_2030 = elem_shifted[elem_shifted["School Year"] <= FORECAST_TO_YEAR]
ax2.plot(
    elem_to_2030["School Year"], elem_to_2030[colname],
    color=COLORS["primary_trend"], linestyle="--", linewidth=3.0, alpha=0.9,
    label=f"Elementary Trend shifted +{LAG_YEARS} yrs", path_effects=halo, zorder=8
)

# Draw A-P GREEN only through last observed year
ap_hist = ap_trend[ap_trend["School Year"] <= last_ap_year]
ax2.plot(
    ap_hist["School Year"], ap_hist["Secondary Trend"],
    color=COLORS["amherst_trend"], linestyle="--", linewidth=2.8, alpha=0.9,
    label=f"Amherst-Pelham Trend ({ROLLING_YEARS}-yr centered)", path_effects=halo, zorder=9
)

# ---------- OLS fit: A-P_t ~ a + b * ElemShifted_t (overlapping historical years) ----------
overlap = elem_shifted.merge(ap_trend, on="School Year", how="inner")
if len(overlap) >= 2:
    x = overlap[colname].to_numpy()
    y = overlap["Secondary Trend"].to_numpy()
    b, a = np.polyfit(x, y, 1)  # y ≈ a + b*x (note: np.polyfit returns [slope, intercept])

    # R^2 for reporting
    y_hat = a + b * x
    ss_res = np.sum((y - y_hat)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan

    # Forecast years and driver x values from shifted Elementary
    future_years_all = np.arange(last_ap_year + 1, FORECAST_TO_YEAR + 1)
    x_future = elem_shifted.set_index("School Year")[colname].reindex(future_years_all).to_numpy()

    mask = ~np.isnan(x_future)
    future_years = future_years_all[mask]
    x_future = x_future[mask]

    if future_years.size > 0:
        # Forecast extends the A-P line forward from last observed point
        y_future = a + b * x_future
        x_line = np.concatenate(([last_ap_year], future_years))
        y_line = np.concatenate(([ap_last_y], y_future))

        ax2.plot(
            x_line, y_line,
            color=AP_FORECAST_COLOR, linestyle="--", linewidth=3.2, alpha=0.95,
            label=f"A-P Trend Forecast (OLS on Elementary, to {FORECAST_TO_YEAR})",
            path_effects=halo, zorder=10,
        )

        # Mark the boundary where forecast begins
        ax2.axvline(last_ap_year + 0.5, color=AP_FORECAST_COLOR,
                    linestyle=":", linewidth=1.2, alpha=0.7, zorder=5)

    # Annotation: method + coefficients
    ann = (
        f"Ordinary Least Squares (OLS) forecast method:\n"
        f"Fit window: {overlap['School Year'].min()}–{overlap['School Year'].max()}   |   "
        f"a = {a:.1f},  b = {b:.3f},  R² ≈ {r2:.2f}\n"        
        f"1) Look at years where both series overlap and fit a straight line that converts the\n"
        f"   Elementary(+{LAG_YEARS}) trend into the A-P trend:  A-P ≈ a + b × Elementary(+{LAG_YEARS}).\n"
        f"   • a (intercept) captures the typical level gap between A-P and Elementary.\n"
        f"   • b (slope) says how much A-P changes when Elementary changes by 1 student.\n"
        f"2) Use future Elementary(+{LAG_YEARS}) values (2026–{FORECAST_TO_YEAR}) in that line to extend A-P."
        )
else:
    ann = (
        f"Insufficient overlap to fit OLS (need ≥2 years).\n"
        f"Orange: Elementary (+{LAG_YEARS}y) through {FORECAST_TO_YEAR}.\n"
        f"Green: A-P through {last_ap_year}."
    )

# Legend + annotation
ax2.legend(loc="best")
ax2.annotate(
    ann,
    xy=(0.01, 0.02), xycoords="axes fraction",
    va="bottom", ha="left", fontsize=9,
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, pad=0.5)
)

# Mirror Y-axis on the right (stays synced with the left)
secax = ax2.secondary_yaxis('right', functions=(lambda y: y, lambda y: y))
secax.set_ylabel("Total Enrollment")
secax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:,.0f}"))
secax.grid(False)

plt.tight_layout()
plt.show()

# ======================
# NEW PLOT: Leverett, Pelham, Shutesbury (lines only)
# ======================
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

targets = ["Amherst", "Leverett", "Pelham", "Shutesbury"]
present = [d for d in targets if d in df["District"].unique()]
if present:
    # Colorblind-friendly (Okabe–Ito) picks, distinct from earlier semantics
    color_map = {
        "Amherst":   "#009E73",  # green
        "Leverett":  "#0072B2",  # blue
        "Pelham":     "#D55E00",  # vermilion
        "Shutesbury": "#CC79A7",  # purple
    }

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    for d in present:
        g = (
            df[df["District"] == d]
            .sort_values("School Year")
            [["School Year", "Total Enrollment"]]
        )
        ax.plot(
            g["School Year"],
            g["Total Enrollment"],
            linewidth=2.5,
            label=d,
            color=color_map.get(d, None),
        )
        # If you prefer markers as well, uncomment:
        # ax.plot(g["School Year"], g["Total Enrollment"], marker="o", linewidth=0, color=color_map.get(d, None))

    # Formatting
    ax.set_title("Elementary Enrollment Over Time")
    ax.set_xlabel("School Year")
    ax.set_ylabel("Total Enrollment")
    ax.grid(alpha=0.25)
    ax.set_facecolor("white")
    for s in ax.spines.values():
        s.set_alpha(0.4)
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()
else:
    print("None of Amherst, Leverett, Pelham, or Shutesbury are present in the data.")
