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
)

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

# ---- main ----
if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df, reg = load_data()
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
