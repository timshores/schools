from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
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
    canonical_order_bottom_to_top,
    add_alps_pk12, EXCLUDE_SUBCATS, aggregate_to_canonical,
    FTE_LINE_COLORS, PPE_PEERS_YMAX, PPE_PEERS_REMOVE_SPINES, PPE_PEERS_BAR_EDGES,
    MICRO_AREA_FILL, MICRO_AREA_EDGE,
)

# ===== version footer for images =====
CODE_VERSION = "v2025.09.20-ALPS-11"

def _stamp(fig):
    fig.text(0.99, 0.01, f"Code: {CODE_VERSION}", ha="right", va="bottom", fontsize=10.5, color="#666666")

def _boost_plot_fonts():
    plt.rcParams.update({
        "font.size": 15,
        "axes.labelsize": 20,
        "axes.titlesize": 20,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    })

def comma_formatter():
    return FuncFormatter(lambda x, pos: f"{x:,.0f}")

# ---------- Helpers used across charts ----------
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

def _weighted_total_series_for_list(df: pd.DataFrame, districts: List[str]) -> tuple[pd.Series, pd.Series]:
    """Weighted total PPE (sum of canonical EPP, weighted by In-District FTE) and enrollment sum for a given list of districts."""
    names_low = {d.lower() for d in districts}
    present = set(df["DIST_NAME"].str.lower())
    members = [n for n in names_low if n in present]
    if not members:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    # Weighted EPP per subcat
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

# ---------- Main district page plot ----------
def plot_one(out_path: Path, epp_pivot: pd.DataFrame, lines: Dict[str, pd.Series],
             context: str, right_ylim: float, left_ylim: float | None,
             line_colors: Dict[str, str], cmap_all: Dict[str, Dict[str, str]]):

    cols = list(epp_pivot.columns) if (epp_pivot is not None and not epp_pivot.empty) else []
    sub_order_bottom_top = canonical_order_bottom_to_top(cols)
    years = (epp_pivot.index.tolist() if cols
             else sorted(set().union(*(set(s.index) for s in lines.values() if s is not None))))

    fig, axL = plt.subplots(figsize=(11.8, 7.4))
    axR = axL.twinx()

    axL.set_zorder(3); axR.set_zorder(2)
    axL.patch.set_alpha(0.0)

    if sub_order_bottom_top:
        bottom = np.zeros(len(years))
        for sc in sub_order_bottom_top:
            vals = epp_pivot[sc].reindex(years).fillna(0.0).values
            col = color_for(cmap_all, context, sc)
            axR.bar(years, vals, bottom=bottom, color=col, width=0.8,
                    edgecolor="white", linewidth=0.5, zorder=1)
            bottom = bottom + vals

    for _key, label in ENROLL_KEYS:
        s = lines.get(label)
        if s is None or s.empty: continue
        y = s.reindex(years).values
        lc = line_colors[label]
        axL.plot(years, y, color=lc, lw=3.4, marker="o", ms=8.0,
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
def plot_ppe_change_bars(out_path: Path, df: pd.DataFrame, reg: pd.DataFrame,
                         districts: list[str], year_lag: int = 5,
                         title: str | None = None):

    # Peers (blue) palette
    BLUE_BASE   = "#8fbcd4"
    BLUE_DELTA  = "#1b6ca8"
    PURP_DECL   = "#955196"

    # Aggregates (Western + PK-12 Aggregate) palette
    AGG_BASE   = "#b3c3c9"  # desaturated slate
    AGG_DELTA  = "#4b5563"  # dark slate
    AGG_DECL   = "#8e6aa6"  # muted mauve
    AGG_AREA_FILL = "#DBEAFE"  # pale blue
    AGG_AREA_EDGE = "#1D4ED8"  # blue edge

    bar_width   = 0.72
    dot_offset  = 0.22

    latest = int(df["YEAR"].max()); t0 = latest - year_lag

    # Separate requested inputs
    peers_input = [d for d in districts if d.lower() not in {"western ma (aggregate)", "pk-12 district aggregate"}]

    # Collect peer bars
    peer_labels, peer_p0, peer_p1, peer_series = [], [], [], []
    def add_peer(name: str):
        piv, lines = prepare_district_epp_lines(df, name)
        if piv.empty: return
        total = _total_ppe_series_from_pivot(piv)
        p0 = total.get(t0, np.nan); p1 = total.get(latest, np.nan)
        s = lines.get("In-District FTE Pupils")
        if np.isnan(p0) or np.isnan(p1): return
        peer_labels.append(name); peer_p0.append(float(p0)); peer_p1.append(float(p1))
        peer_series.append(s.sort_index() if isinstance(s, pd.Series) and not s.empty else None)
    for d in peers_input:
        add_peer(d)

    # Western MA aggregate
    ppe_w, enr_w = _western_all_total_series(df, reg)
    has_west = (not ppe_w.empty) and (t0 in ppe_w.index) and (latest in ppe_w.index)
    west_p0 = float(ppe_w.loc[t0]) if has_west else np.nan
    west_p1 = float(ppe_w.loc[latest]) if has_west else np.nan

    # PK-12 District Aggregate (weighted) — the named PK-12 districts on the plot
    pk12_list = [
        "ALPS PK-12", "Easthampton", "Longmeadow", "Hampden-Wilbraham",
        "East Longmeadow", "South Hadley", "Agawam", "Northampton", "Greenfield", "Hadley"
    ]
    ppe_pk12, enr_pk12 = _weighted_total_series_for_list(df, pk12_list)
    has_pk12 = (not ppe_pk12.empty) and (t0 in ppe_pk12.index) and (latest in ppe_pk12.index)
    pk12_p0 = float(ppe_pk12.loc[t0]) if has_pk12 else np.nan
    pk12_p1 = float(ppe_pk12.loc[latest]) if has_pk12 else np.nan

    if not peer_labels and not (has_west or has_pk12):
        print("[WARN] comparative plot: nothing to draw.")
        return

    # Order peers by latest PPE
    if peer_labels:
        peer_p0 = np.array(peer_p0); peer_p1 = np.array(peer_p1)
        order = np.argsort(peer_p1)
        peer_labels = [peer_labels[i] for i in order]
        peer_p0 = peer_p0[order]; peer_p1 = peer_p1[order]
        peer_series = [peer_series[i] for i in order]
    else:
        peer_p0 = np.array([]); peer_p1 = np.array([]); peer_series = []

    # Build full X positions: peers | divider | Western | PK-12 Aggregate
    n_peers = len(peer_labels)
    x_peers = np.arange(n_peers)
    x_blocks = []
    labels = []
    series_list: List[pd.Series | None] = []

    # Peers first
    labels.extend(peer_labels)
    series_list.extend(peer_series)

    # Aggregates on the right
    if has_west:
        labels.append("Western MA (aggregate)")
        series_list.append(enr_w.sort_index())
        x_blocks.append("WEST")
    if has_pk12:
        labels.append("PK-12 District Aggregate")
        series_list.append(enr_pk12.sort_index())
        x_blocks.append("PK12")

    # X coordinates for aggregates: append to the right of peers
    x_west = n_peers if has_west else None
    x_pk12 = n_peers + (1 if has_west else 0) if has_pk12 else None
    x_all = np.concatenate([x_peers, np.array([v for v in [x_west, x_pk12] if v is not None])])

    # Prepare bar numbers aligned with x_all
    p0_all = list(peer_p0)
    p1_all = list(peer_p1)
    if has_west:
        p0_all.append(west_p0); p1_all.append(west_p1)
    if has_pk12:
        p0_all.append(pk12_p0); p1_all.append(pk12_p1)
    p0_all = np.array(p0_all); p1_all = np.array(p1_all)
    delta_all = p1_all - p0_all

    # Figure — taller for breathing room
    fig_height = 12.6
    fig, ax = plt.subplots(figsize=(max(13.0, 1.05*len(labels)+5), fig_height))
    if PPE_PEERS_REMOVE_SPINES:
        for s in ("top","right","left","bottom"):
            ax.spines[s].set_visible(False)

    # Divider line between peers and aggregates
    if len(x_all) > n_peers and n_peers > 0:
        x_sep = n_peers - 0.5
        ax.axvline(x_sep, color="#9AA0A6", linestyle=":", linewidth=1.5, alpha=0.6, zorder=2)

    edge = "white" if PPE_PEERS_BAR_EDGES else None

    # Draw peer bars (blue scheme)
    if n_peers:
        pos = np.clip(delta_all[:n_peers], 0, None)
        neg = np.clip(delta_all[:n_peers], None, 0)
        ax.bar(x_peers, p0_all[:n_peers], width=bar_width, color=BLUE_BASE, edgecolor=edge, linewidth=0.0, zorder=1, label=None)
        if np.any(pos > 0):
            ax.bar(x_peers, pos, bottom=p0_all[:n_peers], width=bar_width, color=BLUE_DELTA, edgecolor=edge, linewidth=0.0, zorder=1, label=None)
        if np.any(neg < 0):
            ax.bar(x_peers, neg, bottom=p0_all[:n_peers], width=bar_width, color=PURP_DECL, edgecolor=edge, linewidth=0.0, zorder=1, label=None)

    # Draw aggregate bars (distinct aggregate palette)
    # Western
    if has_west:
        i = x_west
        d = delta_all[n_peers]  # first aggregate delta
        base = p0_all[n_peers]
        ax.bar([i], [base], width=bar_width, color=AGG_BASE, edgecolor=edge, linewidth=0.0, zorder=1)
        if d >= 0:
            ax.bar([i], [d], bottom=[base], width=bar_width, color=AGG_DELTA, edgecolor=edge, linewidth=0.0, zorder=1)
        else:
            ax.bar([i], [d], bottom=[base], width=bar_width, color=AGG_DECL, edgecolor=edge, linewidth=0.0, zorder=1)
    # PK-12 Aggregate
    if has_pk12:
        idx = n_peers + (1 if has_west else 0)
        d = delta_all[idx]
        base = p0_all[idx]
        ax.bar([x_pk12], [base], width=bar_width, color=AGG_BASE, edgecolor=edge, linewidth=0.0, zorder=1)
        if d >= 0:
            ax.bar([x_pk12], [d], bottom=[base], width=bar_width, color=AGG_DELTA, edgecolor=edge, linewidth=0.0, zorder=1)
        else:
            ax.bar([x_pk12], [d], bottom=[base], width=bar_width, color=AGG_DECL, edgecolor=edge, linewidth=0.0, zorder=1)

    # Y limits and grid
    ax.set_ylim(0, PPE_PEERS_YMAX)
    ax.grid(axis="y", alpha=0.12); ax.set_axisbelow(True)

    # Micro-areas (orange for peers, blue for aggregates)
    bar_tops = np.maximum(p0_all + np.clip(delta_all, 0, None), p0_all)
    headroom = np.maximum(0.0, PPE_PEERS_YMAX - bar_tops)
    default_gap, default_amp = 2400.0, 6000.0
    min_gap, min_amp = 400.0, 800.0

    for i_global, s in enumerate(series_list):
        if s is None or len(s) < 2: continue
        ys = s.dropna().values.astype(float)
        span = float(ys.max() - ys.min()); y_norm = (ys - ys.min()) / (span if span > 0 else 1.0)

        gap_i = min(default_gap, max(min_gap, headroom[i_global] * 0.22))
        amp_i = min(default_amp, max(min_amp, headroom[i_global] - gap_i - 300.0))
        y_base = bar_tops[i_global] + gap_i
        y_area = y_base + y_norm * max(0.0, amp_i)

        x_pos = x_all[i_global]
        xs = np.linspace(x_pos - dot_offset, x_pos + dot_offset, num=len(ys))

        # Choose palette based on whether this index is peer or aggregate
        if i_global < n_peers:
            fill_c, edge_c = MICRO_AREA_FILL, MICRO_AREA_EDGE
        else:
            fill_c, edge_c = AGG_AREA_FILL, AGG_AREA_EDGE

        ax.fill_between(xs, y_base, y_area, color=fill_c, alpha=0.35, linewidth=0.0, zorder=4)
        ax.plot(xs, y_area, color=edge_c, lw=1.9, zorder=5)
        ax.scatter([xs[0], xs[-1]], [y_area[0], y_area[-1]], s=34, color=edge_c, zorder=6)
        ax.text(xs[0]-0.038, y_area[0] + 130, f"{int(round(ys[0])):,}", ha="right", va="bottom", fontsize=13, color=edge_c)
        ax.text(xs[-1]+0.038, y_area[-1] + 130, f"{int(round(ys[-1])):,}", ha="left", va="bottom", fontsize=13, color=edge_c)

    # X labels
    ax.set_xticks(x_all, labels, rotation=30, ha="right")
    ax.set_ylabel("$ per pupil")
    ax.yaxis.set_major_formatter(comma_formatter())

    # Legend OUTSIDE (single line spanning figure width)
    handles = [
        Patch(facecolor=BLUE_BASE, edgecolor=edge, label=f"{t0} PPE"),
        Patch(facecolor=BLUE_DELTA, edgecolor=edge, label=f"{latest} increase from {t0}"),
        Patch(facecolor=PURP_DECL, edgecolor=edge, label=f"{latest} decrease from {t0}"),
        Line2D([0],[0], color=MICRO_AREA_EDGE, lw=2.1, marker="o", markerfacecolor="white", markeredgecolor=MICRO_AREA_EDGE,
               label=f"Enrollment change (FTE) {t0}→{latest}"),
    ]
    # Place legend at top-center, outside axes
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995),
               ncol=len(handles), frameon=False)
    plt.subplots_adjust(top=0.92)  # leave space for legend above

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
        plot_one(out, piv, lines_sum, context, right_ylim, None, FTE_LINE_COLORS, cmap_all)

    # District plots (uniform left axis)
    ordered = ["Amherst-Pelham"] + [d for d in DISTRICTS_OF_INTEREST if d != "Amherst-Pelham"]
    for dist in ordered:
        piv, lines = district_prepared[dist]
        context = context_for_district(df, dist)
        out = OUTPUT_DIR / f"expenditures_per_pupil_vs_enrollment_{dist.replace(' ', '_')}.png"
        plot_one(out, piv, lines, context, right_ylim, left_ylim_districts, FTE_LINE_COLORS, cmap_all)

    # ALPS-only plot: own enrollment y-scale
    piv_alps, lines_alps = prepare_district_epp_lines(df, "ALPS PK-12")
    ctx_alps = context_for_district(df, "ALPS PK-12")
    out_alps = OUTPUT_DIR / "expenditures_per_pupil_vs_enrollment_ALPS_PK-12.png"
    left_ylim_alps = compute_districts_fte_ylim([lines_alps], pad=1.08, step=50)
    plot_one(out_alps, piv_alps, lines_alps, ctx_alps, right_ylim, left_ylim_alps, FTE_LINE_COLORS, cmap_all)

    # Comparative PPE bars incl. ALPS & peer PK-12 districts, plus two aggregates on the right
    peers = [
        "ALPS PK-12",
        "Greenfield", "Easthampton", "South Hadley", "Northampton",
        "East Longmeadow", "Longmeadow", "Agawam", "Hadley",
        "Hampden-Wilbraham",
        "Western MA (aggregate)",
        "PK-12 District Aggregate",
    ]
    plot_ppe_change_bars(
        OUTPUT_DIR / "ppe_change_bars_ALPS_and_peers.png",
        df, reg, peers, year_lag=5, title=None
    )
