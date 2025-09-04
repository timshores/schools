#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Western Region Enrollment Visualization — End-to-end

FLOW
1) CONFIG: point to .\data\E2C_Hub_MA_DESE_Data.xlsx and (optionally) the regions CSV.
2) LOAD: read enrollment from the Excel workbook (explicit overrides or auto-detect).
3) REGION JOIN: attach EOHHS region from Excel sheet "District Regions" (preferred) or fallback CSV.
4) FILTER: keep only EOHHS 'Western' districts.
5) AGGREGATE:
   - overlay: (District, School Year, Total Enrollment) table.
   - classify districts into trend buckets: increasing / flat / decreasing via pct change over full span.
6) PLOTS (all with y-axis starting at 0 where stacking/total):
   a) western_enrollment_total.png           — total region enrollment over time (single line).
   b) western_enrollment_stacked_area.png    — stacked area by district (Top K + Others) for the region.
   c) western_enrollment_grouped_lines.png   — stacked totals of the three trend buckets.
   d) western_enrollment_stacked_increasing.png   — stacked area for “increasing” districts only (Top K + Others).
   e) western_enrollment_stacked_flat.png         — stacked area for “flat” districts only (Top K + Others).
   f) western_enrollment_stacked_decreasing.png   — stacked area for “decreasing” districts only (Top K + Others).
   g) western_enrollment_small_multiples.pdf — small multiples per district (readability).
7) OUTPUTS:
   - western_enrollment_trends.csv          — District, bucket, pct_change, slope (for QA/table use).

Run:
  python western_enrollment_plot.py

Dependencies:
  pandas, numpy, matplotlib, openpyxl
  (install:  python -m pip install pandas numpy matplotlib openpyxl)
"""

from pathlib import Path
import re, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import StrMethodFormatter

# ----------------------
# CONFIG
# ----------------------
DATA_DIR   = Path.cwd() / "data"
OUTPUT_DIR = Path.cwd() / "output"
EXCEL_FILE = DATA_DIR / "E2C_Hub_MA_DESE_Data.xlsx"
REGIONS_CSV = DATA_DIR / "ma_districts_with_eohhs_regions_complete.csv"  # optional

# >>> OVERRIDES (fill these in from inspect_excel.py if auto-detect fails) <<<
SHEET_OVERRIDE        = "Enrollment"
DISTRICT_COL_OVERRIDE = "DIST_NAME"
YEAR_COL_OVERRIDE     = "SY"
ENROLL_COL_OVERRIDE   = "TOTAL_CNT"
REGION_COL_OVERRIDE   = ""   # leave blank (will map via the regions sheet below)

# Auto-detect candidates (used only if overrides are blank)
DISTRICT_CANDS = ["District","district","ORG_NAME","Org Name","LEA Name","DISTRICT_NAME","OrgName"]
YEAR_CANDS     = ["School Year","school_year","Year","Fiscal Year","year","SY"]
ENROLL_CANDS   = ["Total Enrollment","Enrollment","Enroll","total_enrollment","TOTAL_ENROLLMENT","TotalEnrollment"]
REGION_CANDS   = ["eohhs_region","EOHHS Region","Region","region"]

WESTERN_ALIASES = {"Western","WESTERN","western","West","WEST"}

PALETTE = [
    "#000000","#E69F00","#56B4E9","#009E73","#F0E442",
    "#0072B2","#D55E00","#CC79A7","#999999"
]
BG = "white"

# ----------------------
# Helpers
# ----------------------
def find_first_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    lower_map = {col.lower(): col for col in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None

def coerce_year(y):
    if pd.isna(y): return None
    s = str(y).strip()
    m = re.match(r"^(\d{4})", s)
    if m: return int(m.group(1))
    try: return int(float(s))
    except: return None

def load_enrollment(xlsx_path: Path):
    xls = pd.ExcelFile(xlsx_path, engine="openpyxl")
    sheet_names = xls.sheet_names

    # Use override sheet if supplied
    sheets_try = [SHEET_OVERRIDE] if SHEET_OVERRIDE else sheet_names

    best = None
    best_rows = 0
    region_col_found = None
    used_sheet = None

    for sheet in sheets_try:
        if not sheet:
            continue
        if sheet not in sheet_names:
            if SHEET_OVERRIDE:
                raise RuntimeError(f"SHEET_OVERRIDE '{SHEET_OVERRIDE}' not found. Available: {sheet_names}")
            continue

        df = pd.read_excel(xls, sheet_name=sheet, engine="openpyxl")
        if df.empty:
            continue
        df = df.loc[:, ~df.columns.duplicated()].copy()

        # Use overrides if provided
        dcol = DISTRICT_COL_OVERRIDE or find_first_col(df, DISTRICT_CANDS)
        ycol = YEAR_COL_OVERRIDE     or find_first_col(df, YEAR_CANDS)
        ecol = ENROLL_COL_OVERRIDE   or find_first_col(df, ENROLL_CANDS)
        if not (dcol and ycol and ecol):
            if SHEET_OVERRIDE:
                raise RuntimeError(
                    f"Column detection failed in sheet '{sheet}'. "
                    f"Found columns: {list(df.columns)[:20]}...\n"
                    f"Set DISTRICT_COL_OVERRIDE, YEAR_COL_OVERRIDE, ENROLL_COL_OVERRIDE at the top of the script."
                )
            continue

        keep = [dcol, ycol, ecol]
        # Region: prefer explicit override, else any candidate present
        rcol = REGION_COL_OVERRIDE or next((c for c in REGION_CANDS if c in df.columns), None)
        if rcol: keep.append(rcol)

        tmp = df[keep].copy()
        tmp = tmp.rename(columns={dcol: "District", ycol: "School Year", ecol: "Total Enrollment"})
        if rcol: tmp = tmp.rename(columns={rcol: "eohhs_region"})

        tmp["School Year"] = tmp["School Year"].map(coerce_year)
        tmp["Total Enrollment"] = pd.to_numeric(tmp["Total Enrollment"], errors="coerce")
        tmp = tmp.dropna(subset=["District","School Year","Total Enrollment"])

        if len(tmp) > best_rows:
            best, best_rows = tmp, len(tmp)
            region_col_found = "eohhs_region" if rcol else None
            used_sheet = sheet

        if SHEET_OVERRIDE and DISTRICT_COL_OVERRIDE and YEAR_COL_OVERRIDE and ENROLL_COL_OVERRIDE:
            break

    if best is None or best.empty:
        raise RuntimeError(
            "Could not find an enrollment table.\n"
            f"Sheets available: {sheet_names}\n"
            "Use inspect_excel.py to identify the sheet and exact column names,\n"
            "then set SHEET_OVERRIDE and the *_COL_OVERRIDE values at the top of this script."
        )

    best["District"] = best["District"].astype(str).str.strip()
    best = best.drop_duplicates(subset=["District","School Year"], keep="last")
    print(f"Using sheet: {used_sheet} | rows: {len(best)} | region_in_excel: {bool(region_col_found)}")
    return best, region_col_found

def attach_regions(enroll_df: pd.DataFrame, region_col_in_excel: bool, regions_csv: Path) -> pd.DataFrame:
    # If the enrollment sheet already had a region column (via REGION_COL_OVERRIDE), just use it.
    if region_col_in_excel:
        return enroll_df.copy()

    # Prefer the mapping embedded in the workbook (sheet: "District Regions")
    xlsx_path = EXCEL_FILE
    try:
        xls = pd.ExcelFile(xlsx_path, engine="openpyxl")
        if "District Regions" in xls.sheet_names:
            reg_df = pd.read_excel(xls, sheet_name="District Regions", engine="openpyxl", dtype=str)
            cols = {c.lower(): c for c in reg_df.columns}
            name_col   = cols.get("district_name")
            region_col = cols.get("eohhs_region")
            type_col   = cols.get("school_type")  # <-- NEW (optional)
            if name_col and region_col:
                left = enroll_df.copy()
                left["District_key"] = left["District"].str.upper().str.strip()
                keep_cols = [name_col, region_col]
                if type_col:  # carry school_type if present
                    keep_cols.append(type_col)
                right = reg_df[keep_cols].copy()
                right["District_key"] = right[name_col].str.upper().str.strip()
                out = left.merge(right, on="District_key", how="left")
                out = out.drop(columns=["District_key"])
                out = out.rename(columns={region_col: "eohhs_region"})
                if type_col and type_col in out.columns:
                    out.rename(columns={type_col: "school_type"}, inplace=True)
                return out
    except Exception:
        pass  # fall back to CSV

    # CSV fallback
    if regions_csv.exists():
        map_df = pd.read_csv(regions_csv, dtype=str)
        cols = {c.lower(): c for c in map_df.columns}
        name_col   = cols.get("district_name")
        region_col = cols.get("eohhs_region")
        type_col   = cols.get("school_type")  # optional
        if not (name_col and region_col):
            raise RuntimeError(f"Mapping CSV at {regions_csv} is missing district_name/eohhs_region.")
        left = enroll_df.copy()
        left["District_key"] = left["District"].str.upper().str.strip()
        keep_cols = [name_col, region_col]
        if type_col:
            keep_cols.append(type_col)
        right = map_df[keep_cols].copy()
        right["District_key"] = right[name_col].str.upper().str.strip()
        out = left.merge(right, on="District_key", how="left")
        out = out.drop(columns=["District_key"]).rename(columns={region_col: "eohhs_region"})
        if type_col and type_col in out.columns:
            out.rename(columns={type_col: "school_type"}, inplace=True)
        return out

    raise RuntimeError(
        "No region info found. Keep the workbook's 'District Regions' sheet (now with school_type), "
        "or provide ma_districts_with_eohhs_regions_complete.csv in .\\data (with school_type)."
    )

def filter_western(df: pd.DataFrame) -> pd.DataFrame:
    df["eohhs_region_norm"] = df["eohhs_region"].astype(str).str.strip()
    return df[df["eohhs_region_norm"].isin(WESTERN_ALIASES)].copy()

def okabe_ito_colors(n):
    return [PALETTE[i % len(PALETTE)] for i in range(n)]

# ----------------------
# Main
# ----------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not EXCEL_FILE.exists():
        raise FileNotFoundError(f"Workbook not found: {EXCEL_FILE}")

    # 1-4) Load, map regions, filter Western, and prepare overlay
    enroll_df, region_in_excel = load_enrollment(EXCEL_FILE)
    enroll_df = attach_regions(enroll_df, region_in_excel, REGIONS_CSV)
    west = filter_western(enroll_df)
    if west.empty:
        raise RuntimeError("No rows mapped to EOHHS 'Western'. Check your region mapping or REGION_COL_OVERRIDE.")

    yr_min, yr_max = int(west["School Year"].min()), int(west["School Year"].max())
    overlay = west.groupby(["District","School Year"], as_index=False)["Total Enrollment"].sum()
    districts = sorted(overlay["District"].unique().tolist())

    # Helper: classify district trend over the full span by percent change
    def classify_trend(df_one_district, pct_threshold=0.05):
        """
        > +5% => 'increasing', < -5% => 'decreasing', else 'flat'.
        Also returns simple linear slope for reference.
        """
        g = df_one_district.sort_values("School Year")
        y0 = g["Total Enrollment"].iloc[0]
        y1 = g["Total Enrollment"].iloc[-1]
        years = g["School Year"].to_numpy()
        vals  = g["Total Enrollment"].to_numpy()
        pct_change = (y1 - y0) / y0 if pd.notna(y0) and y0 != 0 else np.nan
        slope = float(np.polyfit(years, vals, 1)[0]) if len(years) >= 2 else np.nan
        if pd.notna(pct_change):
            if pct_change >= pct_threshold:
                bucket = "increasing"
            elif pct_change <= -pct_threshold:
                bucket = "decreasing"
            else:
                bucket = "flat"
        else:
            bucket = "flat"
        return bucket, pct_change, slope

    # 6a) Total region enrollment over time (start at zero)
    region_total = overlay.groupby("School Year", as_index=False)["Total Enrollment"].sum()
    plt.figure(figsize=(12, 7))
    ax = plt.gca()
    ax.plot(region_total["School Year"], region_total["Total Enrollment"], linewidth=3)
    ax.set_title(f"Western Region — Total Enrollment ({yr_min}–{yr_max})", pad=10)
    ax.set_xlabel("School Year"); ax.set_ylabel("Total Enrollment")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25); ax.set_facecolor(BG)
    [s.set_alpha(0.5) for s in ax.spines.values()]
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    plt.tight_layout()
    out_total = OUTPUT_DIR / "western_enrollment_total.png"
    plt.savefig(out_total, dpi=200, bbox_inches="tight")
    plt.close()

    # 6b) Stacked area by district (Top K + Others)
    K = 50
    pivot = overlay.pivot(index="School Year", columns="District", values="Total Enrollment").fillna(0).sort_index()
    last = pivot.iloc[-1].copy()
    topk = last.sort_values(ascending=False).head(K).index.tolist()
    others = [c for c in pivot.columns if c not in topk]
    stack_df = pivot[topk].copy()
    if others:
        stack_df["Others"] = pivot[others].sum(axis=1)

    labels = list(stack_df.columns)
    series = [stack_df[c].to_numpy() for c in labels]
    years = stack_df.index.to_numpy()

    plt.figure(figsize=(14, 8))
    ax = plt.gca()
    ax.stackplot(years, series, labels=labels, linewidth=0.5)
    ax.set_title(f"Western Region — Stacked Enrollment by District (Top {K} + Others), {yr_min}–{yr_max}", pad=10)
    ax.set_xlabel("School Year"); ax.set_ylabel("Total Enrollment")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25); ax.set_facecolor(BG)
    [s.set_alpha(0.5) for s in ax.spines.values()]
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    plt.tight_layout()
    out_stack = OUTPUT_DIR / "western_enrollment_stacked_area.png"
    plt.savefig(out_stack, dpi=200, bbox_inches="tight")
    plt.close()

    # 5) Trend classification table
    trend_rows = []
    for d in districts:
        g = overlay.loc[overlay["District"] == d, ["School Year","Total Enrollment"]]
        bucket, pct_change, slope = classify_trend(g)
        trend_rows.append((d, bucket, pct_change, slope))
    trend_df = pd.DataFrame(trend_rows, columns=["District","bucket","pct_change","slope"])

    # Save trend table
    trend_csv = OUTPUT_DIR / "western_enrollment_trends.csv"
    trend_df.sort_values(["bucket","District"]).to_csv(trend_csv, index=False)

    # 6c) Stacked totals by trend group (decreasing / flat / increasing)
    group_totals = (
        overlay.merge(trend_df[["District","bucket"]], on="District", how="left")
               .groupby(["School Year","bucket"], as_index=False)["Total Enrollment"].sum()
    )
    bucket_order = ["decreasing","flat","increasing"]  # left-to-right stack order
    gt_pivot = (
        group_totals.pivot(index="School Year", columns="bucket", values="Total Enrollment")
                    .reindex(columns=bucket_order).fillna(0).sort_index()
    )
    years = gt_pivot.index.to_numpy()
    series = [gt_pivot[b].to_numpy() for b in bucket_order if b in gt_pivot.columns]
    labels = [b.title() for b in bucket_order if b in gt_pivot.columns]
    bucket_colors = {"increasing": "#009E73", "flat": "#999999", "decreasing": "#D55E00"}
    facecolors = [bucket_colors[b] for b in bucket_order if b in gt_pivot.columns]

    plt.figure(figsize=(14, 8))
    ax = plt.gca()
    ax.stackplot(years, series, labels=labels, colors=facecolors, linewidth=0.5)
    ax.set_title(f"Western Region — Stacked Totals by Trend Group ({yr_min}–{yr_max})", pad=10)
    ax.set_xlabel("School Year"); ax.set_ylabel("Total Enrollment")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25); ax.set_facecolor(BG)
    [s.set_alpha(0.5) for s in ax.spines.values()]
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    plt.tight_layout()
    out_grouped = OUTPUT_DIR / "western_enrollment_grouped_lines.png"  # grouped stacks
    plt.savefig(out_grouped, dpi=200, bbox_inches="tight")
    plt.close()

    # 6d) Per-bucket stacked area plots (increasing / flat / decreasing)
    def stacked_by_group(overlay_df, trend_table, bucket_label, top_k=20):
        districts_in_bucket = trend_table.loc[trend_table["bucket"] == bucket_label, "District"].unique().tolist()
        if not districts_in_bucket:
            return None, None

        sub = overlay_df[overlay_df["District"].isin(districts_in_bucket)]
        if sub.empty:
            return None, None

        pv = sub.pivot(index="School Year", columns="District", values="Total Enrollment").fillna(0).sort_index()
        last = pv.iloc[-1].copy()
        topk_local = last.sort_values(ascending=False).head(top_k).index.tolist()
        others_local = [c for c in pv.columns if c not in topk_local]

        stack_df_local = pv[topk_local].copy()
        if others_local:
            stack_df_local["Others"] = pv[others_local].sum(axis=1)

        labels_local = list(stack_df_local.columns)
        series_local = [stack_df_local[c].to_numpy() for c in labels_local]
        years_local = stack_df_local.index.to_numpy()
        colors_local = [PALETTE[i % len(PALETTE)] for i in range(len(labels_local))]
        titlesafe = bucket_label.title()

        plt.figure(figsize=(14, 8))
        ax = plt.gca()
        ax.stackplot(years_local, series_local, labels=labels_local, colors=colors_local, linewidth=0.5)
        ax.set_title(f"Western Region — Stacked Enrollment ({titlesafe}) Districts, {yr_min}–{yr_max}", pad=10)
        ax.set_xlabel("School Year"); ax.set_ylabel("Total Enrollment")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25); ax.set_facecolor(BG)
        [s.set_alpha(0.5) for s in ax.spines.values()]
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        plt.tight_layout()

        out_path = OUTPUT_DIR / f"western_enrollment_stacked_{bucket_label}.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        return out_path, len(districts_in_bucket)

    out_inc, n_inc = stacked_by_group(overlay, trend_df, "increasing", top_k=80)
    out_flat, n_flat = stacked_by_group(overlay, trend_df, "flat", top_k=80)
    out_dec,  n_dec  = stacked_by_group(overlay, trend_df, "decreasing", top_k=80)

    # 6e) Western region stacked by school_type (Traditional / Charter / Vocational) ---
    # --- with 2025 % labels + dimension arrows ---
    # --- Build pivot for school_type stacks ---
    west_types = west.copy()
    if "school_type" not in west_types.columns:
        west_types["school_type"] = "Unknown"
    west_types["school_type"] = west_types["school_type"].fillna("Unknown").str.title().str.strip()

    type_totals = (
        west_types.groupby(["School Year", "school_type"], as_index=False)["Total Enrollment"].sum()
    )

    desired_order = ["Traditional", "Charter", "Vocational", "Unknown"]
    type_pivot = (
        type_totals.pivot(index="School Year", columns="school_type", values="Total Enrollment")
                  .reindex(columns=[c for c in desired_order if c in set(type_totals["school_type"])])
                  .fillna(0)
                  .sort_index()
    )

    labels = list(type_pivot.columns)
    years = type_pivot.index.to_numpy()
    facecolors = [
        {"Traditional": "#0072B2", "Charter": "#009E73", "Vocational": "#D55E00", "Unknown": "#999999"}.get(c, "#666666")
        for c in labels
    ]
    plt.figure(figsize=(14, 8))
    ax = plt.gca()
    ax.stackplot(years, [type_pivot[c].to_numpy() for c in labels], labels=labels, colors=facecolors, linewidth=0.5)
    ax.set_title(f"Western Region — Stacked Enrollment by School Type ({yr_min}–{yr_max})", pad=10)
    ax.set_xlabel("School Year"); ax.set_ylabel("Total Enrollment")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25); ax.set_facecolor(BG)
    [s.set_alpha(0.5) for s in ax.spines.values()]
    leg = ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))

    # ---- Annotate 2025 slice with percentages + dimension arrows ----
    year_target = 2025
    if year_target in type_pivot.index:
        # Values at 2025 by type (in the current column order `labels`)
        vals_2025 = type_pivot.loc[year_target, labels].fillna(0).to_numpy()
        total_2025 = float(vals_2025.sum())
        if total_2025 > 0:
            # Cumulative bounds for the vertical stack at 2025
            cum = np.cumsum(vals_2025)
            bottoms = np.concatenate(([0], cum[:-1]))
            tops = cum

            # Where to place the annotations/arrows on the x-axis
            x_label = year_target  # labels centered on the slice
            x_arrow = year_target + 0.35  # small offset to draw dimension arrows beside the slice

            for i, (lab, v, y0, y1, color) in enumerate(zip(labels, vals_2025, bottoms, tops, facecolors)):
                if v <= 0:
                    continue
                pct = 100.0 * v / total_2025
                y_mid = (y0 + y1) / 2.0

                # Percentage label inside/near the slice
                ax.text(x_label, y_mid, f"{lab}: {pct:.1f}%",
                        ha="center", va="center",
                        fontsize=9, color="black",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.65, edgecolor="none"))

                # Dimension arrow (double-headed) showing the “thickness” of this band at 2025
                # A tiny vertical wiggle so short segments still show arrowheads
                y0_draw = y0 + max(0.002 * total_2025, 5)
                y1_draw = y1 - max(0.002 * total_2025, 5)
                if y1_draw > y0_draw:
                    ax.annotate("",
                                xy=(x_arrow, y1_draw), xytext=(x_arrow, y0_draw),
                                arrowprops=dict(arrowstyle="<->", lw=1.5, color=color, shrinkA=0, shrinkB=0))
                    # Small tick marks at ends (like technical drawings)
                    tick_w = 0.08
                    ax.plot([x_arrow - tick_w, x_arrow + tick_w], [y0_draw, y0_draw], color=color, lw=1.2)
                    ax.plot([x_arrow - tick_w, x_arrow + tick_w], [y1_draw, y1_draw], color=color, lw=1.2)

            # Add a subtle vertical guide at 2025
            ax.axvline(year_target, color="k", lw=1, alpha=0.25, zorder=0)

    plt.tight_layout()
    out_type = OUTPUT_DIR / "western_enrollment_stacked_by_school_type.png"
    plt.savefig(out_type, dpi=200, bbox_inches="tight")
    plt.close()

    # 6e) Custom stack for selected districts of interest ---
    selected_raw = [
        "Amherst",
        "Leverett",
        "Pelham",
        "Shutesbury",
        "Amherst-Pelham",
        "Hilltown Cooperative Charter Public (District)",
        "Pioneer Valley Chinese Immersion Charter (District)",
        "Pioneer Valley Performing Arts Charter Public (District)",
    ]

    # Light aliasing to improve matches
    alias_map = {
        "SHUTEBURY": "SHUTESBURY",  # fix common typo
        # add more aliases here if needed
    }

    # Build lookup for districts present in overlay (case-insensitive)
    present_keys = {d.upper().strip(): d for d in overlay["District"].unique()}

    resolved = []
    missing = []
    for name in selected_raw:
        key = name.upper().strip()
        key = alias_map.get(key, key)
        # try exact (case-insensitive) first
        if key in present_keys:
            resolved.append(present_keys[key])
            continue
        # try a loose contains match among available districts
        candidates = [present_keys[k] for k in present_keys if key in k]
        if len(candidates) == 1:
            resolved.append(candidates[0])
        elif len(candidates) > 1:
            # pick the shortest name (usually the canonical) if ambiguous
            resolved.append(sorted(candidates, key=len)[0])
        else:
            missing.append(name)

    if missing:
        print(f"[warn] Requested districts not found: {missing}")

    # If nothing resolved, skip plotting
    if resolved:
        # Pivot to year x district for just these selections
        sel = overlay[overlay["District"].isin(resolved)]
        sel_pivot = (sel.pivot(index="School Year", columns="District", values="Total Enrollment")
                        .fillna(0)
                        .sort_index())

        # Keep the order as the user listed (for legend readability)
        ordered_cols = [c for c in [present_keys.get(alias_map.get(n.upper().strip(), n.upper().strip()), n) for n in selected_raw]
                        if c in sel_pivot.columns]
        # Fallback to alphabetical for anything unmatched by order list
        ordered_cols += [c for c in sel_pivot.columns if c not in ordered_cols]

        sel_pivot = sel_pivot[ordered_cols]
        years_sel = sel_pivot.index.to_numpy()
        series_sel = [sel_pivot[c].to_numpy() for c in sel_pivot.columns]

        # Color cycle
        colors_sel = [PALETTE[i % len(PALETTE)] for i in range(len(sel_pivot.columns))]

        plt.figure(figsize=(14, 8))
        ax = plt.gca()
        ax.stackplot(years_sel, series_sel, labels=list(sel_pivot.columns), colors=colors_sel, linewidth=0.5)
        ax.set_title(f"Western Region — Stacked Enrollment (Selected Districts), {yr_min}–{yr_max}", pad=10)
        ax.set_xlabel("School Year"); ax.set_ylabel("Total Enrollment")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25); ax.set_facecolor(BG)
        [s.set_alpha(0.5) for s in ax.spines.values()]
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        plt.tight_layout()
        out_selected = OUTPUT_DIR / "western_enrollment_stacked_selected_districts.png"
        plt.savefig(out_selected, dpi=200, bbox_inches="tight")
        plt.close()

        # Optional: write a tiny QA CSV of what was included/missing
        qa_csv = OUTPUT_DIR / "western_enrollment_selected_districts_qa.csv"
        pd.DataFrame({
            "requested": selected_raw,
            "resolved_or_missing": [present_keys.get(alias_map.get(n.upper().strip(), n.upper().strip()), n) for n in selected_raw],
            "status": ["resolved" if (alias_map.get(n.upper().strip(), n.upper().strip()) in present_keys) or
                                any(alias_map.get(n.upper().strip(), n.upper().strip()) in k for k in present_keys)
                    else "missing" for n in selected_raw]
        }).to_csv(qa_csv, index=False)

        print(f"✓ Wrote: {out_selected}")
        print(f"✓ Wrote: {qa_csv}")
    else:
        print("[warn] No selected districts could be resolved; skipping selected stack plot.")


    # 6g) Small multiples PDF (per-district)
    per_page, rows, cols = 12, 4, 3
    pages = int(math.ceil(len(districts)/per_page))
    out_pdf = OUTPUT_DIR / "western_enrollment_small_multiples.pdf"
    with PdfPages(out_pdf) as pdf:
        for p in range(pages):
            start, end = p*per_page, min((p+1)*per_page, len(districts))
            subset = districts[start:end]
            fig, axes = plt.subplots(rows, cols, figsize=(17, 11), sharex=True, sharey=False)
            axes = axes.flatten()
            for ax_idx, d in enumerate(subset):
                ax = axes[ax_idx]
                g = overlay[overlay["District"] == d].sort_values("School Year")
                ax.plot(g["School Year"], g["Total Enrollment"], linewidth=2.0,
                        color=PALETTE[(start+ax_idx) % len(PALETTE)])
                ax.set_title(d, fontsize=10)
                ax.grid(alpha=0.25); ax.set_facecolor(BG)
                [s.set_alpha(0.4) for s in ax.spines.values()]
                ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
            for k in range(len(subset), len(axes)):
                axes[k].axis("off")
            fig.suptitle(f"Western Region — Enrollment by District ({yr_min}–{yr_max})", fontsize=14, y=0.98)
            for ax in axes[-cols:]: ax.set_xlabel("School Year")
            for i in range(0, len(axes), cols): axes[i].set_ylabel("Total Enrollment")
            plt.tight_layout(rect=[0, 0.03, 1, 0.96])
            pdf.savefig(fig, dpi=200, bbox_inches="tight")
            plt.close(fig)

    # Final prints
    print(f"✓ Wrote: {out_total}")
    print(f"✓ Wrote: {out_stack}")
    print(f"✓ Wrote: {out_grouped}")
    if out_inc:  print(f"✓ Wrote: {out_inc}  (districts: {n_inc})")
    if out_flat: print(f"✓ Wrote: {out_flat} (districts: {n_flat})")
    if out_dec:  print(f"✓ Wrote: {out_dec}  (districts: {n_dec})")
    print(f"✓ Wrote: {trend_csv}")
    print(f"✓ Wrote: {out_pdf}")

if __name__ == "__main__":
    main()
