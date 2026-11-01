"""
The explorer's earlier years, FY1990 to FY2008, for its "Since 1990" switch.

export_explorer_data.py calls history() once it has built the explorer's districts and averages.
Every district and average gets a record of yearly series over HIST_YEARS, which the explorer puts
in front of its FY2009-on series when the switch is on, in files of their own
(districts-history/<chunk>.json) that load only then.

Two measures (data/dese_archive/README.md):
- Functional, FY2005-FY2008 (data/dese_expenditures_fy2005_2008.csv): the explorer's own measure,
  so these years extend its spending columns, enrollment and averages, computed with the same
  functions as FY2009 on. The file's payments to out-of-district schools, per out-of-district
  pupil, are left out: the Hub has no such row, and the shared functions would fold it into
  "Other".
- Day program, FY1990-FY2006 (data/dese_day_program_fy1990_2006.csv): DESE's older, narrower
  measure, kept apart as its own series. A district's total and its programs; an average's total
  only, weighted by DESE's average membership from FY1997 and, before that, by NCES fall
  enrollment (data/nces_ma_enrollment_fy1990_2006.csv), since DESE's reports for those years give
  no pupils. Program averages would need each program's pupils, which no source gives.

Chapter 70 reaches back to FY1993, DESE's first profile year.

Former districts: every district in the older files or the Hub that the explorer's FY2025 list
doesn't have (towns that ran their own schools, regional, vocational and charter districts that
closed), with all their years and a note on what became of each. The note is built from the
figures and DESE's regional members list, or from a source in FORMER_NOTES.

Every district and average also gets its payments to out-of-district schools (ood_series).
"""

import re
from pathlib import Path

import pandas as pd

from school_shared import (START_YEAR, coalesce_year_column, prepare_aggregate_nss_ch70_weighted,
                           prepare_district_epp_lines, prepare_district_nss_ch70, prepare_western_epp_lines)

HIST_FIRST = 1990
HIST_YEARS = list(range(HIST_FIRST, START_YEAR))
FUNCTIONAL_FROM = 2005
DAY_LAST = 2006
MEMBERSHIP_FROM = 1997
CHAPTER70_FROM = 1993
FUNC_CSV = Path("data/dese_expenditures_fy2005_2008.csv")
DAY_CSV = Path("data/dese_day_program_fy1990_2006.csv")
NCES_CSV = Path("data/nces_ma_enrollment_fy1990_2006.csv")
OUT_OF_DISTRICT = "Payments To Out-Of-District Schools"
TOTAL_DAY = "Total, All Day Programs"
PROGRAMS = {"regular": "Regular Day", "special": "Special Needs", "bilingual": "Bilingual",
            "occupational": "Occupational Day"}
ENROLL = {"In-District FTE Pupils": "in_district", "Foundation Enrollment": "foundation",
          "Out-of-District FTE Pupils": "out_of_district"}
NSS = {"Ch70 Aid": "ch70_aid", "Req NSS (minus Ch70)": "required_local",
       "Actual NSS (minus Req NSS)": "above_required"}
# The scans print regional names without their hyphens
FORMER_NAMES = {"Ashfield Plainfield": "Ashfield-Plainfield", "Buckland Shelburne": "Buckland-Shelburne",
                "Plymouth Carver": "Plymouth-Carver"}
VOCATIONAL_CODES = {406, 408}
# Why a former district ended, where a source says so: DESE code -> (sentence, source URL)
FORMER_NOTES = {
    35010000: ("Its board of trustees voted on January 17, 2023, to surrender the school's charter, effective June 30, "
               "2023. DESE's commissioner wrote that the school \u201chas been unable to attract enrollment sufficient to "
               "maintain financial viability.\u201d",
               "https://www.doe.mass.edu/bese/docs/fy2023/2023-01/item5.docx"),
}


def english_list(items):
    items = list(items)
    return items[0] if len(items) == 1 else " and ".join(items) if len(items) == 2 else \
        f"{', '.join(items[:-1])}, and {items[-1]}"


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def yseries(s, digits, ys):
    """Values for the years ys, None where missing."""
    if s is None or len(s) == 0:
        return [None] * len(ys)
    s = s.copy()
    s.index = [int(i) for i in s.index]
    return [None if (y not in s.index or pd.isna(s[y])) else round(float(s[y]), digits) for y in ys]


def from_dict(d, digits):
    """{fy: value} to values for HIST_YEARS."""
    return [None if d.get(y) is None else round(float(d[y]), digits) for y in HIST_YEARS]


def pack(piv, lines, nss, categories, ys=HIST_YEARS, spent=None):
    """Spending by category, enrollment and Chapter 70 for the years ys. With spent (the years a
    district reported spending), Chapter 70 shows only in those: DESE's profiles keep a row for a
    town whose pupils all go to a regional district, with leftover sums in it."""
    if spent is not None:
        nss = nss[nss.index.isin(spent)] if len(nss) else nss
        lines = {k: (v[v.index.isin(spent)] if k == "Foundation Enrollment" else v) for k, v in lines.items()}
    return {
        "ppe": {c: yseries(piv[c] if c in piv.columns else None, 0, ys) for c in categories},
        "enroll": {key: yseries(lines.get(label), 1, ys) for label, key in ENROLL.items()},
        "nss": {key: yseries(nss[col] if col in nss.columns else None, 0, ys) for col, key in NSS.items()},
    }


# What DESE reports as each pupil-count and per-pupil total, for payments to out-of-district schools
OOD_PARTS = {"total expenditures": "tot", "total in-district expenditures": "ind_pp", "total fte pupils": "tot_fte",
             "in-district fte pupils": "ind_fte", "out-of-district fte pupils": "ood_fte"}


def ood_series(df, names, ys):
    """Payments to out-of-district schools per out-of-district FTE pupil, pooled over the districts named
    (lowercase), for the years ys. DESE's Hub has no row for them: they are total spending (per pupil,
    times all FTE pupils) less in-district spending (per in-district pupil, times those pupils), over
    out-of-district pupils. For FY2005-FY2008, where DESE printed the figure, this agrees with it to
    within 0.7%."""
    sub = df[df["DIST_NAME"].str.lower().isin(names) & df["IND_SUBCAT"].str.lower().isin(OOD_PARTS)]
    if sub.empty:
        return [None] * len(ys)
    w = sub.assign(k=sub["IND_SUBCAT"].str.lower().map(OOD_PARTS)).pivot_table(
        index=["DIST_NAME", "YEAR"], columns="k", values="IND_VALUE", aggfunc="first")
    w = w.dropna(subset=[c for c in OOD_PARTS.values() if c in w.columns])
    if not set(OOD_PARTS.values()) <= set(w.columns) or w.empty:
        return [None] * len(ys)
    w = w.assign(dollars=w["tot"] * w["tot_fte"] - w["ind_pp"] * w["ind_fte"])
    by = w.groupby(level="YEAR")[["dollars", "ood_fte"]].sum()
    by = by[by["ood_fte"] > 0]
    return yseries(by["dollars"] / by["ood_fte"], 0, ys)


def with_functional(df):
    """The Hub's rows with FY2005-FY2008 stacked under them, for the shared functions."""
    h = pd.read_csv(FUNC_CSV, dtype={"DIST_CODE": str})
    h = h[h["IND_SUBCAT"] != OUT_OF_DISTRICT]
    h = coalesce_year_column(h)
    for c in ("DIST_NAME", "IND_CAT", "IND_SUBCAT"):
        h[c] = h[c].astype(str).str.strip()
    h["IND_VALUE"] = pd.to_numeric(h["IND_VALUE"], errors="coerce")
    cols = ["DIST_NAME", "IND_CAT", "IND_SUBCAT", "IND_VALUE", "YEAR"]
    return pd.concat([h[cols], df[cols]], ignore_index=True)


def load_day():
    """{code: {"total"/program key/"memb": {fy: value}}, "name": ...}, code as DESE's DIST_CODE number."""
    day = {}
    sub = {v: k for k, v in PROGRAMS.items()} | {TOTAL_DAY: "total"}
    for r in pd.read_csv(DAY_CSV, dtype={"DIST_CODE": str}).itertuples():
        code = int(r.DIST_CODE)
        rec = day.setdefault(code, {"name": r.DIST_NAME})
        if r.IND_CAT == "Day Program Expenditures Per Pupil":
            rec.setdefault(sub[r.IND_SUBCAT], {})[int(r.SY)] = r.IND_VALUE
        elif r.IND_CAT == "Day Program Enrollment":
            rec.setdefault("memb", {})[int(r.SY)] = r.IND_VALUE
    return day


def load_nces():
    """{code: {fy: fall enrollment}}, FY1990 to MEMBERSHIP_FROM - 1, the years it weights."""
    out = {}
    if not NCES_CSV.exists():
        return out
    for r in pd.read_csv(NCES_CSV, dtype={"DIST_CODE": str}).itertuples():
        if r.FY < MEMBERSHIP_FROM:
            out.setdefault(int(r.DIST_CODE), {})[int(r.FY)] = r.enrollment
    return out


def day_fields(rec, nces_d):
    """A district's day-program series: total and programs per pupil, DESE's average membership, and
    NCES fall enrollment where DESE gives no pupils."""
    if not rec:
        return {}
    day = {"total": from_dict(rec.get("total", {}), 0)}
    for k in PROGRAMS:
        if rec.get(k):
            day[k] = from_dict(rec[k], 0)
    return {"day": day, "day_memb": from_dict(rec.get("memb", {}), 1), "nces": from_dict(nces_d or {}, 0)}


def day_average(codes, day, nces):
    """An average's day-program total per pupil, each year over its members that reported one, weighted
    by DESE's average membership from FY1997 and NCES fall enrollment before; with the average member's
    pupils by each count, and how many districts each year counts."""
    total, memb, fall, n = {}, {}, {}, {}
    for fy in range(HIST_FIRST, DAY_LAST + 1):
        pairs = []
        for c in codes:
            v = day.get(c, {}).get("total", {}).get(fy)
            w = day.get(c, {}).get("memb", {}).get(fy) if fy >= MEMBERSHIP_FROM else nces.get(c, {}).get(fy)
            if v is not None and w:
                pairs.append((v, w))
        if not pairs:
            continue
        wsum = sum(w for _, w in pairs)
        total[fy] = sum(v * w for v, w in pairs) / wsum
        (memb if fy >= MEMBERSHIP_FROM else fall)[fy] = wsum / len(pairs)
        n[fy] = len(pairs)
    return {"day": {"total": from_dict(total, 0)}, "day_memb": from_dict(memb, 1), "nces": from_dict(fall, 0),
            "day_n": from_dict(n, 0)}


def history(df, reg, c70, dists, groups, full, towns, categories, ys, regions, norm, exclude):
    """({chunk: [records]}, the meta's "history" entry, the former districts' menu list).

    dists: export_explorer_data.explorer_districts() with ids; groups: its averages, as
    (id, name, scope, cohort, region, members, cmp); full: the explorer's district entities, with
    their towns; towns: every full town (for former districts' regions and links); ys: the explorer's
    own years, which former districts' Hub figures fill; regions: the EOHHS regions; norm: the
    export's town-name key; exclude: the districts the explorer leaves out, lowercase."""
    df_all = with_functional(df)
    day, nces = load_day(), load_nces()
    names_2008 = set(df_all.loc[df_all["YEAR"] == START_YEAR - 1, "DIST_NAME"].str.lower())
    matched = sum(d["key"] in names_2008 for d in dists)
    print(f"  history: {matched} of {len(dists)} districts in the FY{START_YEAR - 1} functional data; "
          f"{sum(d['code'] in day for d in dists)} with day-program figures")

    chunk_of = {e["id"]: e["chunk"] for e in full}
    chunks = {}

    def spent(piv, code):
        return set(int(y) for y in piv.index) | set(day.get(code, {}).get("total", {}))

    # Each district's earlier years, and its first year with figures, to find a district that begins as
    # a former one ends
    first_of = {}
    for d in dists:
        piv, lines = prepare_district_epp_lines(df_all, d["name"], c70)
        years_d = spent(piv, d["code"])
        if years_d:
            first_of[d["id"]] = min(years_d)
        nss, _e, _f = prepare_district_nss_ch70(df_all, c70, d["name"])
        rec = {"id": d["id"], **pack(piv, lines, nss, categories, spent=spent(piv, d["code"])),
               "ood": ood_series(df_all, [d["key"]], HIST_YEARS), **day_fields(day.get(d["code"]), nces.get(d["code"]))}
        chunks.setdefault(chunk_of[d["id"]], []).append(rec)

    by_key = {d["key"]: d for d in dists}
    for eid, _name, _scope, _cohort, region, members, _cmp in groups:
        keys = [d["key"] for d in members]
        _t, piv, _sum, lines_mean = prepare_western_epp_lines(df_all, reg, "all_western", c70, districts=keys)
        nss, _e, foundation = prepare_aggregate_nss_ch70_weighted(df_all, c70, keys)
        lines = dict(lines_mean)
        if foundation is not None and len(foundation):
            lines["Foundation Enrollment"] = foundation
        rec = {"id": eid, **pack(piv, lines, nss, categories), "ood": ood_series(df_all, keys, HIST_YEARS),
               **day_average([by_key[k]["code"] for k in keys], day, nces)}
        chunks.setdefault(slug(region) if region else "statewide", []).append(rec)

    # Former districts: every district in the older files or the Hub that the explorer doesn't list,
    # because it has no FY{latest} figures, less the special-purpose schools it leaves out
    current = {d["code"] for d in dists}
    names = {int(c): n for c, n in zip(df["DIST_CODE"], df["DIST_NAME"])}
    func = pd.read_csv(FUNC_CSV, dtype={"DIST_CODE": str})
    for c, n in zip(func["DIST_CODE"], func["DIST_NAME"]):
        names.setdefault(int(c), n)
    for c, rec in day.items():
        names.setdefault(c, FORMER_NAMES.get(rec["name"], rec["name"]))
    types = {n.lower(): t for n, t in zip(reg["DIST_NAME"], reg["SCHOOL_TYPE"])}
    regions_of = {n.lower(): r for n, r in zip(reg["DIST_NAME"], reg["EOHHS_REGION"])}
    town_by_key = {norm(t["name"]): t for t in towns if t["kind"] == "town"}
    # Each town's districts today, from the districts' own town lists
    districts_of = {}
    for e in full:
        kind = "vocational" if e["type"] == "vocational" else "regional" if e["regional"] else "town"
        for t in e.get("towns", []):
            districts_of.setdefault(t["id"], []).append({"name": e["name"], "id": e["id"], "kind": kind})
    ids = {d["id"] for d in dists}
    former = []
    for code, name in sorted(names.items(), key=lambda kv: kv[1]):
        org = code // 10000
        if code in current or org == 999 or name.lower() in exclude:
            continue
        piv, lines = prepare_district_epp_lines(df_all, name, c70)
        years = sorted(spent(piv, code))
        if not years:
            continue
        kind = types.get(name.lower(), "")
        vocational = kind == "Vocational" or org in VOCATIONAL_CODES or 800 <= org < 1000
        charter = kind == "Charter" or (not vocational and (400 <= org < 600 or org >= 3500))
        regional = org >= 600 and not charter
        member_names = name.split("-") if regional else [name]
        member_towns = [town_by_key[norm(m)] for m in member_names if norm(m) in town_by_key]
        # Worcester Trade: the city's own vocational school
        home = member_towns[0] if member_towns else town_by_key.get(norm(name.split()[0]))
        region = regions_of.get(name.lower())
        region = region if region in regions else home["region"] if home else None
        fid = slug(name) if slug(name) not in ids else f"{slug(name)}-former"
        ids.add(fid)
        t = "vocational" if vocational else "charter" if charter else "traditional"
        cmp = {"state": "statewide", "region": f"region-{slug(region)}" if region and t == "traditional" else None,
               "cohort_state": {"vocational": "cohort-vocational", "charter": "cohort-charter"}.get(t)}
        nss, _e, _f = prepare_district_nss_ch70(df_all, c70, name)
        entity = {"id": fid, "name": name, "kind": "district", "type": t, "cohort": None, "region": region,
                  "regional": regional, "gateway": False, "cmp": cmp, "chunk": "former",
                  "former": {"first": years[0], "last": years[-1]},
                  "main": {**pack(piv, lines, nss, categories, ys, spent=set(years)),
                           "ood": ood_series(df_all, [name.lower()], ys)},
                  **pack(piv, lines, nss, categories, spent=set(years)), "ood": ood_series(df_all, [name.lower()], HIST_YEARS),
                  **day_fields(day.get(code), nces.get(code))}
        if member_towns and t == "traditional":
            entity["towns"] = [{"id": x["id"], "name": x["name"], "contribution": None,
                                "also": districts_of.get(x["id"], [])} for x in member_towns]
            entity["towns_kind"] = "members" if len(member_towns) > 1 else "town"
        # A district its towns belong to that begins in the year or two after this one ends
        last = years[-1]
        entity["_succ"] = sorted({a["id"]: a for x in member_towns for a in districts_of.get(x["id"], [])
                                  if last < first_of.get(a["id"], 0) <= last + 2}.values(), key=lambda a: a["name"])
        entity["_code"] = code
        chunks.setdefault("former", []).append(entity)
        group = "early" if last < START_YEAR else "charter" if t == "charter" else "later"
        former.append({"id": fid, "name": name, "chunk": "former", "type": t, "cohort": None, "region": region,
                       "regional": regional, "former": {**entity["former"], "group": group}})
    print(f"  history: {len(former)} former districts")

    # What became of each, from the figures: its last year; a district its towns belong to that begins
    # just after, with the other former districts that end the same year in it; or the districts its
    # towns belong to today. A sourced reason, where there is one.
    formers = chunks.get("former", [])
    for f in formers:
        last = f["former"]["last"]
        parts = [f"Its last figures are for FY{last}."]
        succ = f.pop("_succ")
        code = f.pop("_code")
        towns = f.get("towns", [])
        if succ:
            s0 = succ[0]
            begins = first_of[s0["id"]]
            partners = [g for g in formers if g["former"]["last"] == last and any(
                a["id"] == s0["id"] for x in g.get("towns", []) for a in x["also"])]
            names = list(dict.fromkeys(x["name"] for g in partners for x in g.get("towns", [])))
            parts.append(f"From FY{begins}, {english_list(names)} belong{'s' if len(names) == 1 else ''} to "
                         f"{s0['name']}, a district whose figures begin that year.")
            if begins > last + 1:
                parts.append(f"No figures cover FY{last + 1}.")
        elif towns:
            now = list(dict.fromkeys(a["name"] for x in towns for a in x["also"]))
            if now:
                who = english_list([x["name"] for x in towns])
                parts.append(f"{who} now belong{'s' if len(towns) == 1 else ''} to {english_list(now)}.")
        source = None
        if code in FORMER_NOTES:
            text, source = FORMER_NOTES[code]
            parts.append(text)
        f["note"] = {"text": " ".join(parts), "source": source}

    meta = {"years": HIST_YEARS, "functional_from": FUNCTIONAL_FROM, "day": [HIST_FIRST, DAY_LAST],
            "membership_from": MEMBERSHIP_FROM, "chapter70_from": CHAPTER70_FROM,
            "nces": [HIST_FIRST, MEMBERSHIP_FROM - 1]}
    return chunks, meta, former
