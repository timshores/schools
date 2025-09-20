from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from school_shared import (
    OUTPUT_DIR, load_data, create_or_load_color_map, color_for,
    context_for_district, context_for_western,
    prepare_district_epp_lines, prepare_western_epp_lines,
    DISTRICTS_OF_INTEREST, ENROLL_KEYS,
    compute_global_dollar_ylim, compute_districts_fte_ylim,
    LINE_COLORS_DIST, LINE_COLORS_WESTERN,
    canonical_order_bottom_to_top,
    add_alps_pk12, EXCLUDE_SUBCATS, aggregate_to_canonical,
)

# ===== version footer for images =====
CODE_VERSION = "v2025.09.19-ALPS-6"

def _stamp(fig):
    fig.text(0.99, 0.01, f"Code: {CODE_VERSION}", ha="right", va="bottom", fontsize=8.5, color="#666666")

def _boost_plot_fonts():
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "axes.titlesize": 14,  # we won't use titles on charts, but keep consistent
    })

def comma_formatter():
    return FuncFormatter(lambda x, pos: f"{x:,.0f}")

def plot_one(out_path: Path, epp_pivot: pd.DataFrame, lines: Dict[str, pd.Series],
             context: str, right_ylim: float, left_ylim: float | None,
             line_colors: Dict[str, str], cmap_all: Dict[str, Dict[str, str]]):

    cols = list(epp_pivot.columns) if (epp_pivot is not None and not epp_pivot.empty) else []
    sub_order_bottom_top = canonical_order_bottom_to_top(cols)
    years = (epp_pivot.index.tolist() if cols
             else sorted(set().union(*(set(s.index) for s in lines.values() if s is not None))))

    fig, axL = plt.subplots(figsize=(11.0, 6.6))
    axR = axL.twinx()

    # Lines in FRONT
    axL.set_zorder(3); axR.set_zorder(2)
    axL.patch.set_alpha(0.0)

    # Stacked bars (right axis)
    if sub_order_bottom_top:
        bottom = np.zeros(len(years))
        for sc in sub_order_bottom_top:
            vals = epp_pivot[sc].reindex(years).fillna(0.0).values
            col = color_for(cmap_all, context, sc)
            axR.bar(years, vals, bottom=bottom, color=col, width=0.8,
                    edgecolor="white", linewidth=0.5, zorder=1)
            bottom = bottom + vals

    # Lines (left axis)
    for _key, label in ENROLL_KEYS:
        s = lines.get(label)
        if s is None or s.empty: continue
        y = s.reindex(years).values
        lc = line_colors[label]
        axL.plot(years, y, color=lc, lw=3.2, marker="o", ms=7.5,
                 markerfacecolor="white", markeredgecolor=lc, markeredgewidth=2.0,
                 zorder=6, clip_on=False)

    axL.set_xlabel("School Year")
    axL.set_ylabel("Pupils (FTE)")
    axR.set_ylabel("$ per pupil")
    axL.yaxis.set_major_formatter(comma_formatter())
    axR.yaxis.set_major_formatter(comma_formatter())

    if right_ylim is not None: axR.set_ylim(0, right_ylim)
    if left_ylim is not None:  axL.set_ylim(0, left_ylim)

    axL.grid(False); axR.grid(False)
    axL.margins(x=0.02); axR.margins(x=0.02)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=320, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {out_path}")

# ===== PPE comparative bars (5-year change) with enrollment mini-areas =====
def _total_ppe_series_from_pivot(piv: pd.DataFrame) -> pd.Series:
    return piv.sum(axis=1).sort_index() if (piv is not None and not piv.empty) else pd.Series(dtype=float)

def _western_all_total_series(df: pd.DataFrame, reg: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    mask = (reg["EOHHS_REGION"].str.lower() == "western") & (reg["SCHOOL_TYPE"].str.lower() == "traditional")
    members = sorted(set(reg[mask]["DIST_NAME"].str.lower()))
    present = set(df["DIST_NAME"].str.lower())
    members = [m for m in members if m in present]
    if not members:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    # Weighted EPP by subcat
    epp = df[(df["IND_CAT"].str.lower() == "expenditures per pupil") & (df["DIST_NAME"].str.lower().isin(members))][["DIST_NAME","YEAR","IND_SUBCAT","IND_VALUE"]].copy()
    epp = epp[~epp["IND_SUBCAT"].str.lower().isin(EXCLUDE_SUBCATS)].copy()
    wts = df[(df["IND_CAT"].str.lower() == "student enrollment")
             & (df["IND_SUBCAT"].str.lower() == "in-district fte pupils")
             & (df["DIST_NAME"].str.lower().isin(members))][["DIST_NAME","YEAR","IND_VALUE"]].rename(columns={"IND_VALUE":"WEIGHT"}).copy()
    m = epp.merge(wts, on=["DIST_NAME","YEAR"], how="left")
    m["WEIGHT"] = pd.to_numeric(m["WEIGHT"], errors="coerce").fillna(0.0)
    m["P"] = m["IND_VALUE"] * m["WEIGHT"]
    out = m.groupby(["YEAR","IND_SUBCAT"], as_index=False).agg(NUM=("P","sum"), DEN=("WEIGHT","sum"), MEAN=("IND_VALUE","mean"))
    out["VALUE"] = np.where(out["DEN"]>0, out["NUM"]/out["DEN"], out["MEAN"])
    piv_raw = out.pivot(index="YEAR", columns="IND_SUBCAT", values="VALUE").sort_index().fillna(0.0)
    piv = aggregate_to_canonical(piv_raw)
    # Enrollment sum (In-District FTE)
    enr = df[(df["IND_CAT"].str.lower()=="student enrollment")
             & (df["IND_SUBCAT"].str.lower()=="in-district fte pupils")
             & (df["DIST_NAME"].str.lower().isin(members))][["YEAR","IND_VALUE"]]
    enroll_sum = enr.groupby("YEAR")["IND_VALUE"].sum().sort_index()
    return _total_ppe_series_from_pivot(piv), enroll_sum

def plot_ppe_change_bars(out_path: Path, df: pd.DataFrame, reg: pd.DataFrame,
                         districts: list[str], year_lag: int = 5,
                         title: str | None = None):  # title intentionally unused (removed from PNG)
    BLUE_BASE   = "#8fbcd4"
    BLUE_DELTA  = "#1b6ca8"
    PURP_DECL   = "#955196"
    MINI_FILL   = "#F5CBA7"  # pale orange fill for enrollment mini-areas
    MINI_EDGE   = "#C97E2C"  # darker orange for stroke/markers
    bar_width   = 0.7
    dot_offset  = 0.22

    latest = int(df["YEAR"].max())
    t0 = latest - year_lag

    labels, p0s, p1s = [], [], []
    series_list: List[pd.Series | None] = []

    def add_district(name: str):
        piv, lines = prepare_district_epp_lines(df, name)
        if piv.empty: return
        total = _total_ppe_series_from_pivot(piv)
        p0 = total.get(t0, np.nan)
        p1 = total.get(latest, np.nan)
        s = lines.get("In-District FTE Pupils", None)
        if np.isnan(p0) or np.isnan(p1): return
        labels.append(name)
        p0s.append(float(p0)); p1s.append(float(p1))
        series_list.append(s.sort_index() if isinstance(s, pd.Series) and not s.empty else None)

    for d in districts:
        if d.lower() != "western ma (aggregate)":
            add_district(d)

    ppe_w, enr_w = _western_all_total_series(df, reg)
    if not ppe_w.empty and (t0 in ppe_w.index) and (latest in ppe_w.index):
        labels.append("Western MA (aggregate)")
        p0s.append(float(ppe_w.loc[t0])); p1s.append(float(ppe_w.loc[latest]))
        series_list.append(enr_w.sort_index())

    if not labels:
        print("[WARN] comparative plot: no districts with both years.")
        return

    p0_arr = np.array(p0s); p1_arr = np.array(p1s)
    delta  = p1_arr - p0_arr

    order = np.argsort(p1_arr)
    labels = [labels[i] for i in order]
    p0_arr = p0_arr[order]; p1_arr = p1_arr[order]
    delta  = delta[order]
    series_list = [series_list[i] for i in order]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(12.0, 1.05*len(labels)+4), 10.4))  # taller + room for larger fonts

    ax.bar(x, p0_arr, width=bar_width, color=BLUE_BASE, edgecolor="white", linewidth=0.8, label=f"{t0} PPE")

    pos = np.clip(delta, 0, None)
    pos_lbl = f"{latest} PPE increase from {t0} (↑)"
    if np.any(pos > 0):
        ax.bar(x, pos, bottom=p0_arr, width=bar_width, color=BLUE_DELTA, edgecolor="white", linewidth=0.8, label=pos_lbl)

    neg = np.clip(delta, None, 0)
    has_neg = np.any(neg < 0)
    if has_neg:
        ax.bar(x, neg, bottom=p0_arr, width=bar_width, color=PURP_DECL, edgecolor="white", linewidth=0.8,
               label=f"{latest} PPE decrease from {t0} (↓)")

    bar_tops = np.maximum(p0_arr + np.clip(delta, 0, None), p0_arr)
    Ymax = np.nanmax(np.maximum(p0_arr, p1_arr))
    y_gap = 0.08 * Ymax
    amp   = 0.20 * Ymax

    handles: List = [Patch(facecolor=BLUE_BASE, edgecolor="white", label=f"{t0} PPE")]
    if np.any(pos > 0):
        handles.append(Patch(facecolor=BLUE_DELTA, edgecolor="white", label=pos_lbl))
    if has_neg:
        handles.append(Patch(facecolor=PURP_DECL, edgecolor="white", label=f"{latest} PPE decrease from {t0} (↓)"))
    handles.append(Line2D([0], [0], color=MINI_EDGE, lw=2.0, marker="o",
                          markerfacecolor="white", markeredgecolor=MINI_EDGE,
                          label=f"Enrollment change (FTE) {t0} -> {latest}"))

    for i, s in enumerate(series_list):
        if s is None or len(s) < 2:
            continue
        s = s.dropna()
        ys = s.values.astype(float)
        span = float(ys.max() - ys.min()) if len(ys) else 0.0
        y_norm = (ys - ys.min()) / (span if span > 0 else 1.0)
        y_base = bar_tops[i] + y_gap
        y_area = y_base + y_norm * amp
        xs = np.linspace(x[i] - dot_offset, x[i] + dot_offset, num=len(ys))
        ax.fill_between(xs, y_base, y_area, color=MINI_FILL, alpha=0.35, linewidth=0.0, zorder=4)
        ax.plot(xs, y_area, color=MINI_EDGE, lw=1.6, zorder=5)

        # Endpoints: dots + thousands labels
        left_y, right_y = y_area[0], y_area[-1]
        ax.scatter([xs[0], xs[-1]], [left_y, right_y], s=28, color=MINI_EDGE, zorder=6)
        ax.text(xs[0]-0.035, left_y + 0.012*Ymax, f"{int(round(ys[0])):,}", ha="right", va="bottom", fontsize=11, color=MINI_EDGE)
        ax.text(xs[-1]+0.035, right_y + 0.012*Ymax, f"{int(round(ys[-1])):,}", ha="left", va="bottom", fontsize=11, color=MINI_EDGE)

    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylabel("$ per pupil")
    # no PNG title (moved to PDF page title)
    ax.yaxis.set_major_formatter(comma_formatter())
    ax.set_ylim(0, Ymax * 1.30)
    ax.grid(axis="y", alpha=0.12); ax.set_axisbelow(True)

    ax.legend(handles=handles, frameon=False, loc="upper left", ncols=2)

    _stamp(fig)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=320, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {out_path}")

# ---- main ----
if __name__ == "__main__":
    _boost_plot_fonts()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df, reg = load_data()
    df = add_alps_pk12(df)
    cmap_all = create_or_load_color_map(df)

    pivots_all, district_lines_all = [], []

    # Western first
    western_prepared = {}
    for bucket in ("le_500", "gt_500"):
        title, piv, lines_sum, _ = prepare_western_epp_lines(df, reg, bucket)
        western_prepared[bucket] = (title, piv, lines_sum)
        if not piv.empty: pivots_all.append(piv)

    # Districts (five)
    district_prepared = {}
    for dist in DISTRICTS_OF_INTEREST:
        piv, lines = prepare_district_epp_lines(df, dist)
        district_prepared[dist] = (piv, lines)
        if not piv.empty: pivots_all.append(piv)
        district_lines_all.append(lines)

    right_ylim = compute_global_dollar_ylim(pivots_all, pad=1.06, step=500)
    left_ylim_districts = compute_districts_fte_ylim(district_lines_all, pad=1.06, step=50)

    # Western plots
    for bucket in ("le_500", "gt_500"):
        _title, piv, lines_sum = western_prepared[bucket]
        context = context_for_western(bucket)
        out = OUTPUT_DIR / f"regional_expenditures_per_pupil_Western_Traditional_{bucket}.png"
        plot_one(out, piv, lines_sum, context, right_ylim, None, LINE_COLORS_WESTERN, cmap_all)

    # District plots (uniform left axis)
    ordered = ["Amherst-Pelham"] + [d for d in DISTRICTS_OF_INTEREST if d != "Amherst-Pelham"]
    for dist in ordered:
        piv, lines = district_prepared[dist]
        context = context_for_district(df, dist)
        out = OUTPUT_DIR / f"expenditures_per_pupil_vs_enrollment_{dist.replace(' ', '_')}.png"
        plot_one(out, piv, lines, context, right_ylim, left_ylim_districts, LINE_COLORS_DIST, cmap_all)

    # ALPS-only plot: use its OWN enrollment y-scale so the line sits nicely on-chart
    piv_alps, lines_alps = prepare_district_epp_lines(df, "ALPS PK-12")
    ctx_alps = context_for_district(df, "ALPS PK-12")
    out_alps = OUTPUT_DIR / "expenditures_per_pupil_vs_enrollment_ALPS_PK-12.png"
    left_ylim_alps = compute_districts_fte_ylim([lines_alps], pad=1.08, step=50)
    plot_one(out_alps, piv_alps, lines_alps, ctx_alps, right_ylim, left_ylim_alps, LINE_COLORS_DIST, cmap_all)

    # Comparative PPE bars incl. ALPS & peer PK-12 districts (taller)
    peers = [
        "ALPS PK-12",
        "Greenfield", "Easthampton", "South Hadley", "Northampton",
        "East Longmeadow", "Longmeadow", "Agawam", "Hadley",
        "Hampden-Wilbraham",
        "Western MA (aggregate)",
    ]
    plot_ppe_change_bars(
        OUTPUT_DIR / "ppe_change_bars_ALPS_and_peers.png",
        df, reg, peers, year_lag=5,
        title=None  # PDF will provide the title
    )
