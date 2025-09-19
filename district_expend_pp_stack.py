# district_expend_pp_stack.py
# Render charts only (no tables). Uses school_shared.py for everything shared.

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

from school_shared import (
    OUTPUT_DIR, load_data, create_or_load_color_map, color_for, context_for_district,
    context_for_western, prepare_district_epp_lines, prepare_western_epp_lines,
    DISTRICTS_OF_INTEREST, ENROLL_KEYS, compute_global_dollar_ylim, compute_districts_fte_ylim,
    LINE_COLORS_DIST, LINE_COLORS_WESTERN, N_THRESHOLD
)

# -------- helpers --------
def order_subcats_by_mean(piv: pd.DataFrame) -> List[str]:
    if piv.shape[1] <= 1:
        return list(piv.columns)
    return list(piv.mean(axis=0).sort_values(ascending=False).index)

def comma_formatter():
    return FuncFormatter(lambda x, pos: f"{x:,.0f}")

def plot_one(
    out_path: Path,
    epp_pivot: pd.DataFrame,
    lines: Dict[str, pd.Series],
    context: str,  # "SMALL" or "LARGE"
    right_ylim: float,
    left_ylim: float | None,
    line_colors: Dict[str, str],
):
    if epp_pivot is None:
        epp_pivot = pd.DataFrame()
    sub_order = order_subcats_by_mean(epp_pivot) if not epp_pivot.empty else []
    years = epp_pivot.index.tolist() if not epp_pivot.empty else sorted(set().union(*(set(s.index) for s in lines.values() if s is not None)))

    fig, axL = plt.subplots(figsize=(10, 6))
    axR = axL.twinx()

    # Make lines *on top* of bars: raise axL zorder and make its face transparent
    axL.set_zorder(3)
    axR.set_zorder(2)
    axL.patch.set_alpha(0.0)

    # Stacked bars on right axis
    if sub_order:
        bottom = np.zeros(len(years))
        for sc in sub_order:
            vals = epp_pivot[sc].reindex(years).fillna(0.0).values
            col = color_for(cmap_all, context, sc)
            axR.bar(years, vals, bottom=bottom, color=col, width=0.8, edgecolor="white", linewidth=0.3, zorder=1)
            bottom = bottom + vals

    # Enrollment lines on left axis
    for key, label in ENROLL_KEYS:
        s = lines.get(label)
        if s is None or s.empty:
            continue
        y = s.reindex(years).values
        style = dict(color=line_colors[label], lw=3.0, marker="o", ms=6, zorder=5, clip_on=False)
        axL.plot(years, y, label=label, **style)

    # Axis labels and formatting
    axL.set_xlabel("School Year")
    axL.set_ylabel("Pupils (FTE)")
    axR.set_ylabel("$ per pupil")

    axL.yaxis.set_major_formatter(comma_formatter())
    axR.yaxis.set_major_formatter(comma_formatter())

    # Uniform right axis across all plots
    if right_ylim is not None:
        axR.set_ylim(0, right_ylim)

    # Left axis: uniform for districts; None means auto (used for Western)
    if left_ylim is not None:
        axL.set_ylim(0, left_ylim)

    # No gridlines, small margins
    axL.grid(False)
    axR.grid(False)
    axL.margins(x=0.02)
    axR.margins(x=0.02)

    # Legend is omitted (handled in PDF tables)
    if axL.get_legend():
        axL.get_legend().remove()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {out_path}")

# -------- main workflow --------
if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df, reg = load_data()
    cmap_all = create_or_load_color_map(df)

    # Prepare all pages up front to compute global axis limits
    pivots_all: List[pd.DataFrame] = []
    district_lines_all: List[Dict[str, pd.Series]] = []

    # Western: <=500 and >500
    western_prepared = {}
    for bucket in ("le_500", "gt_500"):
        title, piv, lines_sum, _lines_mean = prepare_western_epp_lines(df, reg, bucket)
        western_prepared[bucket] = (title, piv, lines_sum)
        if not piv.empty:
            pivots_all.append(piv)

    # Districts of interest
    district_prepared = {}
    for dist in DISTRICTS_OF_INTEREST:
        piv, lines = prepare_district_epp_lines(df, dist)
        district_prepared[dist] = (piv, lines)
        if not piv.empty:
            pivots_all.append(piv)
        district_lines_all.append(lines)

    # Compute axis limits
    right_ylim = compute_global_dollar_ylim(pivots_all, pad=1.06, step=500)
    left_ylim_districts = compute_districts_fte_ylim(district_lines_all, pad=1.06, step=50)

    # Western plots first (both buckets) — left axis auto, right axis uniform
    for bucket in ("le_500", "gt_500"):
        title, piv, lines_sum = western_prepared[bucket]
        context = context_for_western(bucket)  # SMALL for le_500, LARGE for gt_500
        out = OUTPUT_DIR / f"regional_expenditures_per_pupil_Western_Traditional_{bucket}.png"
        plot_one(
            out_path=out,
            epp_pivot=piv,
            lines=lines_sum,
            context=context,
            right_ylim=right_ylim,
            left_ylim=None,  # proportional for Western
            line_colors=LINE_COLORS_WESTERN,
        )

    # District plots — left axis uniform for all districts
    ordered = ["Amherst-Pelham"] + [d for d in DISTRICTS_OF_INTEREST if d != "Amherst-Pelham"]
    for dist in ordered:
        piv, lines = district_prepared[dist]
        context = context_for_district(df, dist)  # SMALL or LARGE by N_THRESHOLD
        out = OUTPUT_DIR / f"expenditures_per_pupil_vs_enrollment_{dist.replace(' ', '_')}.png"
        plot_one(
            out_path=out,
            epp_pivot=piv,
            lines=lines,
            context=context,
            right_ylim=right_ylim,
            left_ylim=left_ylim_districts,
            line_colors=LINE_COLORS_DIST,
        )
