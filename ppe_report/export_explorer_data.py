"""
Export the data behind the interactive explorer at timshores.com/massppe.

One JSON file with every Western MA traditional district in the report, each
cohort aggregate, and the Western MA aggregate, FY START_YEAR..LATEST_YEAR:
per-pupil expenditure by category, enrollment, and Chapter 70 / net school
spending per foundation pupil. Values come from the same functions the PDF
report uses, so the explorer and the report agree.

    python export_explorer_data.py ../../mywebsite/public/massppe/data/explorer.json
"""

import datetime as dt
import json
import re
import sys
from pathlib import Path

import pandas as pd

from school_shared import (CANON_CATS_BOTTOM_TO_TOP, LATEST_YEAR, START_YEAR, get_cohort_label,
                           get_cohort_range, get_cohort_short_label, get_indistrict_fte_for_year, get_western_cohort_districts, load_data,
                           prepare_aggregate_nss_ch70_weighted, prepare_district_epp_lines,
                           prepare_district_nss_ch70, prepare_western_epp_lines)

YEARS = list(range(START_YEAR, LATEST_YEAR + 1))
ENROLL = {"In-District FTE Pupils": "in_district", "Foundation Enrollment": "foundation",
          "Out-of-District FTE Pupils": "out_of_district"}
NSS = {"Ch70 Aid": "ch70_aid", "Req NSS (minus Ch70)": "required_local",
       "Actual NSS (minus Req NSS)": "above_required"}
COHORT_ORDER = ["TINY", "SMALL", "MEDIUM", "LARGE", "SPRINGFIELD"]


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def series(s, digits):
    """Values for YEARS, None where missing."""
    if s is None or len(s) == 0:
        return [None] * len(YEARS)
    s = s.copy(); s.index = [int(i) for i in s.index]
    return [None if (y not in s.index or pd.isna(s[y])) else round(float(s[y]), digits) for y in YEARS]


def pack(piv, lines, nss):
    return {
        "ppe": {c: series(piv[c] if c in piv.columns else None, 0) for c in CANON_CATS_BOTTOM_TO_TOP},
        "enroll": {key: series(lines.get(label), 1) for label, key in ENROLL.items()},
        "nss": {key: series(nss[col] if col in nss.columns else None, 0) for col, key in NSS.items()},
    }


def main(out_path: Path):
    df, reg, c70 = load_data()
    cohorts = get_western_cohort_districts(df, reg)
    proper = {n.lower(): n for n in df["DIST_NAME"].unique()}
    idx = pd.read_csv("data/inflation_indices.csv").set_index("fiscal_year")["cpi_u_northeast"]

    entities = []
    # Aggregates first: Western MA (excluding Springfield) and each cohort
    western = [d for k in ["TINY", "SMALL", "MEDIUM", "LARGE"] for d in cohorts.get(k, [])]
    groups = [("western-ma", "Western MA average", "western", None, western)]
    groups += [(f"cohort-{k.lower()}", f"{get_cohort_short_label(k)} cohort average", "cohort", k, cohorts[k])
               for k in ["TINY", "SMALL", "MEDIUM", "LARGE"] if cohorts.get(k)]
    for eid, name, kind, cohort, members in groups:
        bucket = "all_western" if kind == "western" else cohort.lower()
        _t, piv, _sum, lines_mean = prepare_western_epp_lines(df, reg, bucket, c70, districts=members)
        nss, _e, foundation = prepare_aggregate_nss_ch70_weighted(df, c70, members)
        lines = dict(lines_mean)
        if foundation is not None and len(foundation):
            lines["Foundation Enrollment"] = foundation
        entities.append({"id": eid, "name": name, "kind": kind, "cohort": cohort, "members": len(members),
                         **pack(piv, lines, nss)})

    # Districts, grouped by cohort
    for k in COHORT_ORDER:
        for d in sorted(cohorts.get(k, [])):
            name = proper.get(d, d.title())
            piv, lines = prepare_district_epp_lines(df, name, c70)
            nss, _e, _f = prepare_district_nss_ch70(df, c70, name)
            entities.append({"id": slug(name), "name": name, "kind": "district", "cohort": k,
                             "fte": round(get_indistrict_fte_for_year(df, d, LATEST_YEAR), 1),
                             "compare": f"cohort-{k.lower()}" if k != "SPRINGFIELD" else "western-ma",
                             **pack(piv, lines, nss)})

    out = {
        "meta": {
            "title": "Western MA per-pupil expenditure",
            "years": YEARS, "latest": LATEST_YEAR, "start": START_YEAR,
            "built": dt.date.today().isoformat(),
            "categories": CANON_CATS_BOTTOM_TO_TOP,
            "cohorts": {k: {"label": get_cohort_label(k), "short": get_cohort_short_label(k),
                            "range": [r if r != float("inf") else None for r in get_cohort_range(k)]}
                        for k in COHORT_ORDER},
            "cpi": {str(y): float(idx[y]) for y in YEARS if y in idx.index},
            "sources": {
                "expenditures": "DESE District Expenditures by Spending Category (E2C Hub er3w-dyti)",
                "chapter70": "DESE Chapter 70 District Profiles",
                "cpi": "BLS CPI-U Northeast (CUUR0100SA0), July-June fiscal-year averages",
            },
        },
        "entities": entities,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, separators=(",", ":")))
    n_d = sum(e["kind"] == "district" for e in entities)
    print(f"{out_path}: {n_d} districts, {len(entities) - n_d} aggregates, "
          f"{out_path.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output/explorer.json"))
