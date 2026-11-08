"""
The explorer's earlier years, FY1990 to FY2004, for its "Since 1990" switch.

export_explorer_data.py calls history() once it has built the explorer's districts and averages.
Every district and average gets a record of yearly series over HIST_YEARS, which the explorer puts
in front of its own years, FY2005 on, when the switch is on, in files of their own
(districts-history/<chunk>.json) that load only then.

Two measures (data/dese_archive/README.md):
- Functional, FY2005-FY2008 (data/dese_expenditures_fy2005_2008.csv): the explorer's own measure,
  so the export stacks these years under the Hub's (explorer_common.with_functional) and the
  explorer's own years begin with them, EXPLORER_START. The file's payments to out-of-district schools, per
  out-of-district pupil, are left out: the Hub has no such row, and the shared functions would
  fold it into "Other".
- Day program, FY1990-FY2006 (data/dese_day_program_fy1990_2006.csv): DESE's older, narrower
  measure, kept apart as its own series over DAY_YEARS, which run two years into the explorer's
  own, so the switch shows both measures for FY2005 and FY2006. A district's total and its programs; an average's total
  only, weighted by DESE's average membership from FY1997 and, before that, by NCES fall
  enrollment (data/nces_ma_enrollment_fy1990_2006.csv), since DESE's reports for those years give
  no pupils. Program averages would need each program's pupils, which no source gives.

Chapter 70 reaches back to FY1993, DESE's first profile year.

Former districts: every district in the older files or the Hub that the explorer's FY2025 list
doesn't have (towns that ran their own schools, regional, vocational and charter districts that
closed), with all their years and a note on what became of each. The note is built from the
figures and DESE's regional members list, or from a source in FORMER_NOTES.

Every district and average also gets its payments to out-of-district schools (explorer_common.ood_per_pupil).
Its spending, enrollment and Chapter 70 figures are the ones the export prepared (explorer_common.Hub), cut to
the earlier years.
"""

from pathlib import Path

import pandas as pd

from explorer_common import EXCLUDE_DISTRICTS, EXPLORER_START, FUNCTIONAL_FROM, HIST_FIRST, HIST_YEARS, REGIONS, \
    is_regional, norm, pack, slug

DAY_LAST = 2006
DAY_YEARS = list(range(HIST_FIRST, DAY_LAST + 1))
MEMBERSHIP_FROM = 1997
CHAPTER70_FROM = 1993
DAY_CSV = Path("data/dese_day_program_fy1990_2006.csv")
NCES_CSV = Path("data/nces_ma_enrollment_fy1990_2006.csv")
TOTAL_DAY = "Total, All Day Programs"
PROGRAMS = {"regular": "Regular Day", "special": "Special Needs", "bilingual": "Bilingual",
            "occupational": "Occupational Day"}
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


def from_dict(d, digits, ys=HIST_YEARS):
    """{fy: value} to values for the years ys."""
    return [None if d.get(y) is None else round(float(d[y]), digits) for y in ys]


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
    """{code: {fy: fall enrollment}}, FY1990 to MEMBERSHIP_FROM - 1, the years it weights. The file is committed,
    and the export checks for it with its other inputs; without it the averages' early years would weight
    nothing (#237)."""
    out = {}
    for r in pd.read_csv(NCES_CSV, dtype={"DIST_CODE": str}).itertuples():
        if r.FY < MEMBERSHIP_FROM:
            out.setdefault(int(r.DIST_CODE), {})[int(r.FY)] = r.enrollment
    return out


def drop_empty(rec):
    """A current district's or an average's record less the series with no figure in them. Before FY2005 its
    spending comes from DESE's day-program reports alone, so the series cut like the explorer's own years (spending
    by category, in- and out-of-district pupils, out-of-district payments) were all null: 487 KB of nulls (#237).
    The explorer reads a missing series as nulls. A former district's record keeps its whole shape."""
    def empty(v):
        return (isinstance(v, list) and all(x is None for x in v)) or (isinstance(v, dict) and all(map(empty, v.values())))
    out = {}
    for k, v in rec.items():
        if isinstance(v, dict):
            kept = {kk: vv for kk, vv in v.items() if not empty(vv)}
            if kept:
                out[k] = kept
        elif not empty(v):
            out[k] = v
    return out


def day_fields(rec, nces_d):
    """A district's day-program series: total and programs per pupil, DESE's average membership, and
    NCES fall enrollment where DESE gives no pupils."""
    if not rec:
        return {}
    day = {"total": from_dict(rec.get("total", {}), 0, DAY_YEARS)}
    for k in PROGRAMS:
        if rec.get(k):
            day[k] = from_dict(rec[k], 0, DAY_YEARS)
    return {"day": day, "day_memb": from_dict(rec.get("memb", {}), 1, DAY_YEARS), "nces": from_dict(nces_d or {}, 0)}


def day_average(codes, day, nces):
    """An average's day-program total per pupil, each year over its members that reported one, weighted
    by DESE's average membership from FY1997 and NCES fall enrollment before; with the average member's
    pupils by each count."""
    total, memb, fall = {}, {}, {}
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
    return {"day": {"total": from_dict(total, 0, DAY_YEARS)}, "day_memb": from_dict(memb, 1, DAY_YEARS),
            "nces": from_dict(fall, 0)}


def history(hub, dists, groups, full, towns, func):
    """({chunk: [records]}, the meta's "history" entry, the former districts' menu list).

    hub: explorer_common.Hub over the Hub's rows with FY2005-FY2008 stacked under them (with_functional),
    whose districts' and averages' figures the export has already prepared; dists:
    export_explorer_data.explorer_districts() with ids; groups: its averages (export_explorer_data.Group);
    full: the explorer's district entities, with their towns; towns: every full town (for former districts'
    regions and links); func: FUNC_CSV as read, for the names of districts only it has."""
    df, reg = hub.df, hub.reg
    day, nces = load_day(), load_nces()
    names_first = set(hub.name[df["YEAR"] == EXPLORER_START])
    matched = sum(d["key"] in names_first for d in dists)
    print(f"  history: {matched} of {len(dists)} districts in the FY{EXPLORER_START} functional data; "
          f"{sum(d['code'] in day for d in dists)} with day-program figures")

    chunk_of = {e["id"]: e["chunk"] for e in full}
    chunks = {}

    def spent(piv, code):
        return set(int(y) for y in piv.index) | set(day.get(code, {}).get("total", {}))

    # Each district's earlier years, and its first year with figures, to find a district that begins as
    # a former one ends
    first_of = {}
    for d in dists:
        piv, lines, nss = hub.district(d["name"])
        years_d = spent(piv, d["code"])
        if years_d:
            first_of[d["id"]] = min(years_d)
        rec = {"id": d["id"], **pack(piv, lines, nss, HIST_YEARS, spent=spent(piv, d["code"])),
               "ood": hub.ood([d["key"]], HIST_YEARS), **day_fields(day.get(d["code"]), nces.get(d["code"]))}
        chunks.setdefault(chunk_of[d["id"]], []).append(drop_empty(rec))

    by_key = {d["key"]: d for d in dists}
    for g in groups:
        keys = [d["key"] for d in g.members]
        rec = {"id": g.id, **pack(*hub.aggregate(keys), HIST_YEARS), "ood": hub.ood(keys, HIST_YEARS),
               **day_average([by_key[k]["code"] for k in keys], day, nces)}
        chunks.setdefault(slug(g.region) if g.region else "statewide", []).append(drop_empty(rec))

    # Former districts: every district in the older files or the Hub that the explorer doesn't list,
    # because it has no FY{latest} figures, less the special-purpose schools it leaves out
    current = {d["code"] for d in dists}
    names = {int(c): n for c, n in zip(df["DIST_CODE"], df["DIST_NAME"])}
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
        if code in current or org == 999 or name.lower() in EXCLUDE_DISTRICTS:
            continue
        piv, lines, nss = hub.district(name)
        years = sorted(spent(piv, code))
        if not years:
            continue
        kind = types.get(name.lower(), "")
        vocational = kind == "Vocational" or org in VOCATIONAL_CODES or 800 <= org < 1000
        charter = kind == "Charter" or (not vocational and (400 <= org < 600 or org >= 3500))
        t = "vocational" if vocational else "charter" if charter else "traditional"
        regional = is_regional({"type": t, "code": code})
        member_names = name.split("-") if regional else [name]
        member_towns = [town_by_key[norm(m)] for m in member_names if norm(m) in town_by_key]
        # Worcester Trade: the city's own vocational school
        home = member_towns[0] if member_towns else town_by_key.get(norm(name.split()[0]))
        region = regions_of.get(name.lower())
        region = region if region in REGIONS else home["region"] if home else None
        fid = slug(name) if slug(name) not in ids else f"{slug(name)}-former"
        ids.add(fid)
        cmp = {"state": "statewide", "region": f"region-{slug(region)}" if region and t == "traditional" else None,
               "cohort_state": {"vocational": "cohort-vocational", "charter": "cohort-charter"}.get(t)}
        entity = {"id": fid, "name": name, "kind": "district", "type": t, "cohort": None, "region": region,
                  "regional": regional, "gateway": False, "cmp": cmp, "chunk": "former",
                  "former": {"first": years[0], "last": years[-1]},
                  "main": {**pack(piv, lines, nss, spent=set(years)), "ood": hub.ood([name.lower()])},
                  **pack(piv, lines, nss, HIST_YEARS, spent=set(years)), "ood": hub.ood([name.lower()], HIST_YEARS),
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
        group = "early" if last < EXPLORER_START else "charter" if t == "charter" else "later"
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
