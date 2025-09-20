# district_expend_pp_stack.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

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
CODE_VERSION = "v2025.09.19-ALPS-1"
def _stamp(fig):
    fig.text(0.99, 0.01, f"Code: {CODE_VERSION}", ha="right", va="bottom", fontsize=7, color="#666666")


def comma_formatter():
    return FuncFormatter(lambda x, pos: f"{x:,.0f}")

def plot_one(out_path: Path, epp_pivot: pd.DataFrame, lines: Dict[str, pd.Series],
             context: str, right_ylim: float, left_ylim: float | None,
             line_colors: Dict[str, str], cmap_all: Dict[str, Dict[str, str]]):

    cols = list(epp_pivot.columns) if (epp_pivot is not None and not epp_pivot.empty) else []
    sub_order_bottom_top = canonical_order_bottom_to_top(cols)
    years = (epp_pivot.index.tolist() if cols
             else sorted(set().union(*(set(s.index) for s in lines.values() if s is not None))))

    fig, axL = plt.subplots(figsize=(10, 6))
    axR = axL.twinx()

    # Lines in FRONT
    axL.set_zorder(3); axR.set_zorder(2)
    axL.patch.set_alpha(0.0)

    # Stacked bars (right axis), iterate bottom->top with "Other" last
    if sub_order_bottom_top:
        bottom = np.zeros(len(years))
        for sc in sub_order_bottom_top:
            vals = epp_pivot[sc].reindex(years).fillna(0.0).values
            col = color_for(cmap_all, context, sc)
            axR.bar(years, vals, bottom=bottom, color=col, width=0.8,
                    edgecolor="white", linewidth=0.3, zorder=1)
            bottom = bottom + vals

    # Lines (left axis) with white centers
    for _key, label in ENROLL_KEYS:
        s = lines.get(label)
        if s is None or s.empty: continue
        y = s.reindex(years).values
        lc = line_colors[label]
        axL.plot(years, y, color=lc, lw=3.0, marker="o", ms=7,
                 markerfacecolor="white", markeredgecolor=lc, markeredgewidth=1.8,
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
    if axL.get_legend(): axL.get_legend().remove()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {out_path}")

# ===== PPE comparative bars (5-year change) with enrollment dots =====
def _total_ppe_series_from_pivot(piv: pd.DataFrame) -> pd.Series:
    return piv.sum(axis=1).sort_index() if (piv is not None and not piv.empty) else pd.Series(dtype=float)

def _western_all_total_series(df: pd.DataFrame, reg: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Weighted Western MA traditional aggregate (all sizes): returns (ppe_total_series, enroll_series)
    """
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
    piv = canonical_order_bottom_to_top(piv_raw.columns.tolist())
    piv = aggregate_to_canonical(piv_raw)
    # Enrollment sum (In-District FTE)
    enr = df[(df["IND_CAT"].str.lower()=="student enrollment")
             & (df["IND_SUBCAT"].str.lower()=="in-district fte pupils")
             & (df["DIST_NAME"].str.lower().isin(members))][["YEAR","IND_VALUE"]]
    enroll_sum = enr.groupby("YEAR")["IND_VALUE"].sum().sort_index()
    return _total_ppe_series_from_pivot(piv), enroll_sum

def plot_ppe_change_bars(out_path: Path, df: pd.DataFrame, reg: pd.DataFrame,
                         districts: list[str], year_lag: int = 5,
                         title: str = "PPE, Five-Year Change — ALPS & Peers"):
    import matplotlib.pyplot as plt
    BLUE_BASE   = "#8fbcd4"
    BLUE_DELTA  = "#1b6ca8"
    PURP_DECL   = "#955196"
    DOT_COLOR   = "#444444"
    LINE_COLOR  = "#444444"
    bar_width   = 0.7
    dot_offset  = 0.18

    latest = int(df["YEAR"].max())
    t0 = latest - year_lag

    labels, p0s, p1s, e0s, e1s = [], [], [], [], []

    # Helper for a regular district label
    def add_district(name: str):
        piv, lines = prepare_district_epp_lines(df, name)
        if piv.empty: return
        p0 = _total_ppe_series_from_pivot(piv).get(t0, np.nan)
        p1 = _total_ppe_series_from_pivot(piv).get(latest, np.nan)
        s = lines.get("In-District FTE Pupils", None)
        if isinstance(s, pd.Series) and not s.empty:
            e0 = s.get(t0, np.nan)
            e1 = s.get(latest, np.nan)
        else:
            e0 = np.nan
            e1 = np.nan
        if np.isnan(p0) or np.isnan(p1): return
        labels.append(name); p0s.append(float(p0)); p1s.append(float(p1)); e0s.append(float(e0) if not np.isnan(e0) else np.nan); e1s.append(float(e1) if not np.isnan(e1) else np.nan)

    # Fill list in order
    for d in districts:
        if d.lower() != "western ma (aggregate)":
            add_district(d)

    # Western MA aggregate (all traditional)
    ppe_w, enr_w = _western_all_total_series(df, reg)
    if not ppe_w.empty:
        if (t0 in ppe_w.index) and (latest in ppe_w.index):
            labels.append("Western MA (aggregate)")
            p0s.append(float(ppe_w.loc[t0])); p1s.append(float(ppe_w.loc[latest]))
            e0s.append(float(enr_w.get(t0, np.nan))); e1s.append(float(enr_w.get(latest, np.nan)))

    if not labels:
        print("[WARN] comparative plot: no districts with both years.")
        return

    x = np.arange(len(labels))
    p0_arr = np.array(p0s); p1_arr = np.array(p1s); delta = p1_arr - p0_arr
    e0_arr = np.array(e0s); e1_arr = np.array(e1s)

    fig, ax = plt.subplots(figsize=(max(10, 0.8*len(labels)+4), 6))

    # Base bar (earlier PPE)
    ax.bar(x, p0_arr, width=bar_width, color=BLUE_BASE, edgecolor="white", linewidth=0.8, label=f"{t0} PPE")

    # Positive stack
    pos = np.clip(delta, 0, None)
    ax.bar(x, pos, bottom=p0_arr, width=bar_width, color=BLUE_DELTA, edgecolor="white", linewidth=0.8, label=f"Change to {latest} (↑)")

    # Negative "subtractive" segment
    neg = np.clip(delta, None, 0)
    ax.bar(x, neg, bottom=p0_arr, width=bar_width, color=PURP_DECL, edgecolor="white", linewidth=0.8, label=f"Change to {latest} (↓)")

    # Enrollment dots + short connector (kept on one y-line for compactness; labels carry the numbers)
    bar_tops = np.maximum(p0_arr, p0_arr + pos)
    yb = bar_tops + 0.02 * np.nanmax(bar_tops)
    # Scale enrollment difference into a small vertical span so the mini-line tilts
    enr_delta = np.abs(e1_arr - e0_arr)
    max_enr_delta = np.nanmax(enr_delta) if enr_delta.size else np.nan
    # Map the biggest enrollment change to ~12% of bar height; smaller differences scale proportionally
    scale = 0.12 * np.nanmax(bar_tops) / max_enr_delta if (max_enr_delta and max_enr_delta == max_enr_delta and max_enr_delta > 0) else 0.0
    for i in range(len(labels)):
    # y positions for left/right dots; center the segment around yb[i]
        if scale > 0 and (not np.isnan(e0_arr[i])) and (not np.isnan(e1_arr[i])):
            dy = (e1_arr[i] - e0_arr[i]) * scale
            y_left  = yb[i] - 0.5 * dy
            y_right = yb[i] + 0.5 * dy
        else:
            y_left = y_right = yb[i]

    ax.plot([x[i]-dot_offset, x[i]+dot_offset], [y_left, y_right], lw=1.5, color=LINE_COLOR)
    ax.scatter([x[i]-dot_offset, x[i]+dot_offset], [y_left, y_right], s=18, color=DOT_COLOR, zorder=5)

    if not np.isnan(e0_arr[i]):
        ax.text(x[i]-dot_offset, y_left, f"{int(round(e0_arr[i]))}", ha="center", va="bottom", fontsize=8, color=DOT_COLOR)
    if not np.isnan(e1_arr[i]):
        ax.text(x[i]+dot_offset, y_right, f"{int(round(e1_arr[i]))}", ha="center", va="bottom", fontsize=8, color=DOT_COLOR)

    dir_char = "↑" if delta[i] > 0 else ("↓" if delta[i] < 0 else "→")
    ax.text(x[i], max(y_left, y_right) + 0.005*np.nanmax(bar_tops), dir_char,
            ha="center", va="bottom", fontsize=8, color=DOT_COLOR)

    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylabel("$ per pupil")
    ax.set_title(f"{title}  ({t0} → {latest})", pad=10)
    ax.yaxis.set_major_formatter(comma_formatter())

    handles, lab = ax.get_legend_handles_labels()
    hl = {}
    for h, l in zip(handles, lab): hl[l] = h
    ax.legend(hl.values(), hl.keys(), frameon=False, loc="upper left", ncols=2)

    ax.grid(axis="y", alpha=0.12); ax.set_axisbelow(True)
    ax.set_ylim(0, np.nanmax(np.maximum(p0_arr, p1_arr)) * 1.18)

    _stamp(fig)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {out_path}")


# ---- main ----
if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df, reg = load_data()
    # ===== add ALPS PK-12 to the dataset for downstream use =====
    df = add_alps_pk12(df)
    cmap_all = create_or_load_color_map(df)

    pivots_all, district_lines_all = [], []

    # Western first
    western_prepared = {}
    for bucket in ("le_500", "gt_500"):
        title, piv, lines_sum, _ = prepare_western_epp_lines(df, reg, bucket)
        western_prepared[bucket] = (title, piv, lines_sum)
        if not piv.empty: pivots_all.append(piv)

    # Districts
    district_prepared = {}
    for dist in DISTRICTS_OF_INTEREST:
        piv, lines = prepare_district_epp_lines(df, dist)
        district_prepared[dist] = (piv, lines)
        if not piv.empty: pivots_all.append(piv)
        district_lines_all.append(lines)

    right_ylim = compute_global_dollar_ylim(pivots_all, pad=1.06, step=500)
    left_ylim_districts = compute_districts_fte_ylim(district_lines_all, pad=1.06, step=50)

    # Western plots (left axis proportional)
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

    # ===== Comparative PPE bars incl. ALPS & peer PK-12 districts =====
    peers = [
        "ALPS PK-12",
        "Greenfield", "Easthampton", "South Hadley", "Northampton",
        "East Longmeadow", "Longmeadow", "Agawam", "Hadley",
        "Hampden-Wilbraham",
        "Western MA (aggregate)",  # synthesized inside the function
    ]
    plot_ppe_change_bars(
        OUTPUT_DIR / "ppe_change_bars_ALPS_and_peers.png",
        df, reg, peers, year_lag=5,
        title="Per-Pupil Expenditure: Five-Year Change — ALPS PK-12 & Peers"
    )
