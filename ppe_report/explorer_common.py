"""
What export_explorer_data.py and explorer_history.py share (#237): the explorer's years, its regions and the
districts it leaves out, town and district keys, yearly series, the test for a regional district, and the Hub's
rows with each district's figures prepared once for both files.

explorer_history.history() once took twelve parameters, among them norm, REGIONS and EXCLUDE_DISTRICTS, because
it could not import the export, which imports it; both now import these from here.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from school_shared import (CANON_CATS_BOTTOM_TO_TOP, LATEST_YEAR, coalesce_year_column,
                           prepare_aggregate_nss_ch70_weighted, prepare_district_epp_lines, prepare_district_nss_ch70,
                           prepare_western_epp_lines)

HIST_FIRST = 1990
FUNCTIONAL_FROM = 2005
# The explorer's own years begin with DESE's functional measure; the report keeps FY2009
# (school_shared.START_YEAR)
EXPLORER_START = FUNCTIONAL_FROM
HIST_YEARS = list(range(HIST_FIRST, EXPLORER_START))
YEARS = list(range(EXPLORER_START, LATEST_YEAR + 1))
# FY2005-FY2008, DESE's functional measure from its archive (data/dese_archive/README.md)
FUNC_CSV = Path("data/dese_expenditures_fy2005_2008.csv")
OUT_OF_DISTRICT = "Payments To Out-Of-District Schools"

ENROLL = {"In-District FTE Pupils": "in_district", "Foundation Enrollment": "foundation",
          "Out-of-District FTE Pupils": "out_of_district"}
NSS = {"Ch70 Aid": "ch70_aid", "Req NSS (minus Ch70)": "required_local",
       "Actual NSS (minus Req NSS)": "above_required"}
# EOHHS regions, as DESE assigns districts to them (data/cache/regional_data.csv)
REGIONS = {"Western": "Western MA", "Central": "Central MA", "Northeast": "Northeast MA",
           "Metro West": "Metro West", "Southeast": "Southeast MA", "Boston": "Boston area"}
# Districts: every traditional, vocational and charter district with latest-year
# enrollment and spending, less these special-purpose state schools
EXCLUDE_DISTRICTS = {"institutional schools", "ma academy for math and science",
                     "tec connections academy commonwealth virtual school district",
                     "greater commonwealth virtual district"}
# DESE numbers regional academic districts from 0600; DIST_CODE is that times 10000
REGIONAL_MIN_CODE = 6000000
# DESE's Chapter 70 tables call Manchester-by-the-Sea "Manchester"
TOWN_ALIASES = {"manchester": "manchesterbythesea"}


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def norm(name: str) -> str:
    """A town name as DLS, DESE and Census all spell it ("Manchester By The Sea", "Manchester-by-the-Sea")."""
    key = re.sub(r"[^a-z]", "", name.lower())
    return TOWN_ALIASES.get(key, key)


def is_regional(d):
    """A regional academic or vocational district: d has its "type" and DESE "code". Charter schools are a category
    of their own, though DESE numbers its newer ones from 3500, past the regional districts' 0600 (#237): taken by
    the code alone, 13 counted as regional and were given revenue series the Census has no figures for. The former
    districts (explorer_history) are tested the same way, by the type they are given."""
    return d["type"] != "charter" and d["code"] >= REGIONAL_MIN_CODE


def series(s, digits, ys=YEARS):
    """Values for the years ys (the explorer's own, unless given), None where missing."""
    if s is None or len(s) == 0:
        return [None] * len(ys)
    s = s.copy()
    s.index = [int(i) for i in s.index]
    return [None if (y not in s.index or pd.isna(s[y])) else round(float(s[y]), digits) for y in ys]


def pack(piv, lines, nss, ys=YEARS, spent=None):
    """Spending by category, enrollment and Chapter 70 for the years ys. With spent (the years a
    district reported spending), Chapter 70 shows only in those: DESE's profiles keep a row for a
    town whose pupils all go to a regional district, with leftover sums in it."""
    if spent is not None:
        nss = nss[nss.index.isin(spent)] if len(nss) else nss
        lines = {k: (v[v.index.isin(spent)] if k == "Foundation Enrollment" else v) for k, v in lines.items()}
    return {
        "ppe": {c: series(piv[c] if c in piv.columns else None, 0, ys) for c in CANON_CATS_BOTTOM_TO_TOP},
        "enroll": {key: series(lines.get(label), 1, ys) for label, key in ENROLL.items()},
        "nss": {key: series(nss[col] if col in nss.columns else None, 0, ys) for col, key in NSS.items()},
    }


def with_functional(df, func):
    """The Hub's rows with FY2005-FY2008 (func: FUNC_CSV as read, DIST_CODE as text) stacked under them, for the
    shared functions. The Hub's other columns stay, empty in the stacked years. The file's payments to
    out-of-district schools, per out-of-district pupil, are left out: the Hub has no such row, and the shared
    functions would fold it into "Other"."""
    h = func[func["IND_SUBCAT"] != OUT_OF_DISTRICT]
    h = coalesce_year_column(h)
    for c in ("DIST_NAME", "IND_CAT", "IND_SUBCAT"):
        h[c] = h[c].astype(str).str.strip()
    h["IND_VALUE"] = pd.to_numeric(h["IND_VALUE"], errors="coerce")
    h["DIST_CODE"] = h["DIST_CODE"].astype(df["DIST_CODE"].dtype)
    cols = ["DIST_CODE", "DIST_NAME", "IND_CAT", "IND_SUBCAT", "IND_VALUE", "YEAR"]
    return pd.concat([h[cols], df], ignore_index=True)


# What DESE reports as each pupil-count and per-pupil total, for payments to out-of-district schools
OOD_PARTS = {"total expenditures": "tot", "total in-district expenditures": "ind_pp", "total fte pupils": "tot_fte",
             "in-district fte pupils": "ind_fte", "out-of-district fte pupils": "ood_fte"}


def ood_per_pupil(df, names):
    """Payments to out-of-district schools per out-of-district FTE pupil by year, pooled over the districts named
    (lowercase); None without the figures. DESE's Hub has no row for them: they are total spending (per pupil,
    times all FTE pupils) less in-district spending (per in-district pupil, times those pupils), over
    out-of-district pupils. DESE printed the figure for FY2005-FY2008, but those years' in-district totals were
    computed from it (update_dese_history.py), so this gives it back by construction and checks nothing; no DESE
    figure tests the method from FY2009 (#242)."""
    sub = df[df["DIST_NAME"].str.lower().isin(names) & df["IND_SUBCAT"].str.lower().isin(OOD_PARTS)]
    if sub.empty:
        return None
    w = sub.assign(k=sub["IND_SUBCAT"].str.lower().map(OOD_PARTS)).pivot_table(
        index=["DIST_NAME", "YEAR"], columns="k", values="IND_VALUE", aggfunc="first")
    w = w.dropna(subset=[c for c in OOD_PARTS.values() if c in w.columns])
    if not set(OOD_PARTS.values()) <= set(w.columns) or w.empty:
        return None
    w = w.assign(dollars=w["tot"] * w["tot_fte"] - w["ind_pp"] * w["ind_fte"])
    by = w.groupby(level="YEAR")[["dollars", "ood_fte"]].sum()
    by = by[by["ood_fte"] > 0]
    return by["dollars"] / by["ood_fte"]


class Hub:
    """The Hub's rows, FY2005-FY2008 stacked under them (with_functional), and the Chapter 70 profiles (load_data's
    c70), with each district's rows found once and each district's and average's figures prepared once (#237).

    school_shared's functions find a district's rows by lower-casing the names, categories and subcategories of
    every row on every call: 196,000 rows, three times a call, two or three calls a district, and the same again
    for the earlier years (explorer_history), which took most of the export's five minutes. Handed only the rows
    of the districts asked for, in the frame's own order (so each sum adds the same figures in the same order),
    they return what they would from the whole frame, and the PDF report still calls them on the whole frame.
    cat and subcat are the whole frame's categories, lower-cased once, for the export's own filters."""

    def __init__(self, df, reg, c70):
        self.df, self.reg, self.c70 = df, reg, c70
        self.name = df["DIST_NAME"].str.lower()
        self.cat = df["IND_CAT"].str.lower()
        self.subcat = df["IND_SUBCAT"].str.lower()
        self._at = pd.Series(np.arange(len(df))).groupby(self.name.to_numpy()).indices
        self._c70_at = pd.Series(np.arange(len(c70))).groupby(c70["DIST_NAME"].str.lower().to_numpy()).indices
        self._district, self._aggregate, self._ood = {}, {}, {}

    @staticmethod
    def _take(frame, at, keys):
        pos = [at[k] for k in keys if k in at]
        return frame.iloc[np.unique(np.concatenate(pos)) if pos else np.array([], dtype=int)]

    def rows(self, keys):
        """The rows of the districts keys (lowercase names), in the frame's order"""
        return self._take(self.df, self._at, keys)

    def c70_rows(self, keys):
        return self._take(self.c70, self._c70_at, keys)

    def district(self, name):
        """One district's (spending by category, enrollment lines, Chapter 70 and NSS per foundation pupil), as
        prepare_district_epp_lines and prepare_district_nss_ch70 give them; the export and the earlier years
        read the same figures, so they are prepared once."""
        key = name.lower()
        if key not in self._district:
            df, c70 = self.rows([key]), self.c70_rows([key])
            piv, lines = prepare_district_epp_lines(df, name, c70)
            nss, _e, _f = prepare_district_nss_ch70(df, c70, name)
            self._district[key] = piv, lines, nss
        return self._district[key]

    def aggregate(self, keys):
        """An average's (spending by category, enrollment lines, Chapter 70 and NSS per foundation pupil) over the
        districts keys: spending weighted by in-district FTE (prepare_western_epp_lines, given the members), the
        average member's pupils, and NSS weighted by foundation enrollment, with the average member's foundation
        enrollment"""
        k = tuple(keys)
        if k not in self._aggregate:
            df, c70 = self.rows(k), self.c70_rows(k)
            _t, piv, _sum, lines_mean = prepare_western_epp_lines(df, self.reg, "all_western", c70, districts=list(k))
            nss, _e, foundation = prepare_aggregate_nss_ch70_weighted(df, c70, list(k))
            lines = dict(lines_mean)
            if foundation is not None and len(foundation):
                lines["Foundation Enrollment"] = foundation
            self._aggregate[k] = piv, lines, nss
        return self._aggregate[k]

    def ood(self, keys, ys=YEARS):
        """Payments to out-of-district schools per out-of-district FTE pupil (ood_per_pupil), pooled over the
        districts keys, for the years ys"""
        k = tuple(keys)
        if k not in self._ood:
            self._ood[k] = ood_per_pupil(self.rows(k), list(k))
        by = self._ood[k]
        return [None] * len(ys) if by is None else series(by, 0, ys)
