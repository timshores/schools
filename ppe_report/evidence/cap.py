"""
The evidence record for the explorer's cap tab (Project Regain Custody, Passes 2 and 3; seahag prompt log #242,
#244). The cap tab says what each town's Chapter 70 target local contribution would be under another cap, or none,
keeping the statewide total. For each year it shows, this checks every link from DESE's own spreadsheet to the
figures the site draws. It writes what it found to evidence/cap.md, and a summary to the site's
src/data/massppe/evidence.json, which the tab's "Show the work" reads.

    1. DESE's workbook -> data/dese_c70_townwide.csv: every town's figures, read again from the cached workbook
       cell by cell (openpyxl), not by update_c70_formula.py's pandas path
    2. DESE's formula, restated: yield = the two percentages times EQV and income; target = the lesser of the
       yield and the maximum (82.5% of the foundation budget); the targets total 59% of all foundation budgets
    3. the CSV -> the site's cap.json, town by town
    4. the model, two ways: the pipeline's closed form (nocap_factor) and a bisection like the cap tab's
       (cap.ts solve), with no cap and with a 100% cap, solved a third way for the cap, exactly
    5. an outside calculation, where one exists: Michael Naughton's FY2026 workbook, which MASS Promise to Invest
       posted
    6. the screen: the cap tab's own model (mywebsite src/lib/massppe/capModel.ts), run on cap.json by mywebsite's
       check:units, which every build runs: its no-cap targets and required contributions against the pipeline's

These are checks an AI wrote of figures from one DESE spreadsheet. Only link 5 is someone else's calculation, and
it starts from the same spreadsheet. None of this is independent verification of DESE's figures. A year counts as
reviewed only when a person has read its record and accepted it (REVIEWED).

    python evidence/cap.py [SITE_ROOT]    # from ppe_report; SITE_ROOT defaults to ../../mywebsite
Downloads the two workbooks into data/c70_raw/ (git ignores it) if they aren't there. Exits 1 if a link fails.
"""

import csv
import datetime as dt
import hashlib
import json
import math
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

import openpyxl

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RAW = ROOT / "data" / "c70_raw"
DESE_URL = "https://www.doe.mass.edu/finance/chapter70/fy{fy}/chapter-{fy}.xlsm"
# Outside calculations by year: Michael Naughton's workbook, which MASS Promise to Invest posted on its information page
NAUGHTON = {2026: "https://masspromise2invest.org/wp-content/uploads/2025/10/fy26-chapter-70-cap-comparisons.xlsx"}
NAUGHTON_PAGE = "https://masspromise2invest.org/information-2/"
RECORD_URL = "https://github.com/timshores/schools/blob/main/ppe_report/evidence/cap.md"
# Years a person has read the record for and accepted: fiscal year -> who and when. Empty until then; the site says
# "checked, not yet reviewed" for a year that isn't here.
REVIEWED: dict[int, str] = {}
TOWNWIDE_CSV = ROOT / "data" / "dese_c70_townwide.csv"
POP_CSV = ROOT / "data" / "census_town_population.csv"
SHEET = "Townwide Contributions"
OUT = HERE / "cap.md"


def cached(url, name):
    """The file at url, kept in data/c70_raw/; its path and SHA-256"""
    path = RAW / name
    if not path.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=300) as r:
            path.write_bytes(r.read())
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def num(v):
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def dese_sheet(path):
    """Every town's row of the Townwide Contributions sheet, by LEA, read cell by cell, with the year's two
    percentages. Columns by their header words, as DESE labels them."""
    ws = openpyxl.load_workbook(path, read_only=True, data_only=True, keep_vba=False)[SHEET]
    rows = list(ws.iter_rows(values_only=True))
    head = next(i for i, r in enumerate(rows) if isinstance(r[0], str) and r[0].strip() == "LEA")
    words = [str(c or "").lower() for c in rows[head]]
    col = lambda *ws_: next(i for i, w in enumerate(words) if all(x in w for x in ws_))  # noqa: E731
    at = {"eqv": col("eqv"), "income": 4, "cey": col("combined effort yield"), "foundation": col("foundation"),
          "max_local": col("maximum local"), "target": col("target local contribution"), "required": len(words) - 1}
    assert "required contribution" in words[at["required"]] or "required" in words[23], "required column moved"
    at["required"] = 23
    rate_row = next(i for i, r in enumerate(rows) if isinstance(r[3], str) and "uniform property percentage" in r[3].lower())
    rates = {"property": num(rows[rate_row + 1][3]), "income": num(rows[rate_row + 1][5])}
    towns = {}
    for r in rows[head + 1:]:
        lea = num(r[0])
        if lea and 1 <= lea <= 351 and isinstance(r[1], str) and num(r[2]) is not None:
            towns[int(lea)] = {"town": r[1].strip(), **{k: num(r[i]) for k, i in at.items()}}
    return towns, rates


def naughton(path):
    """His target local contribution (column K) for each LEA, on each of his three sheets"""
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    out = {}
    for key, sheet in ((0.825, "Townwide Contributions - 82.5%"), (1.0, "Townwide Contributions - 100%"),
                       (None, "Townwide Contributions - NO Cap")):
        out[key] = {int(r[0]): r[10] for r in wb[sheet].iter_rows(min_row=10, values_only=True)
                    if num(r[0]) and 1 <= r[0] <= 351 and num(r[10]) is not None}
    # His required contributions, on the Comparisons sheet: P, R and W, at 82.5%, 100% and no cap (#246)
    for key, col in (("req", 15), ("req100", 17), ("reqnone", 22)):
        out[key] = {int(r[0]): r[col] for r in wb["Comparisons"].iter_rows(min_row=10, values_only=True)
                    if num(r[0]) and 1 <= r[0] <= 351 and num(r[col]) is not None}
    return out


def required(t, target, yield_pct, cap90=True):
    """The required contribution under the year's rules for a target (DESE's Townwide Contributions sheet, columns
    P-X; capModel.requiredFor in the site): last year's grown by the growth factor, down to the target if above it,
    up 1% or 2% of last year's if it falls short by more than 2.5 or 7.5 points of foundation, at least the target
    where the yield is over 175% of foundation, and, with cap90, at most 90% of foundation (column X, the parameters
    sheet's "Cap contributions at fixed percentage of foundation")"""
    if t["preliminary"] is None or t["prior"] is None:
        return None
    if not t["foundation"]:
        return 0
    P, H = t["preliminary"], t["foundation"]
    short = js_round(target * 100 / H * 100) / 100 - P * 100 / H
    step = 0.02 if short > 7.5 else 0.01 if short > 2.5 else 0
    T = js_round(step * t["prior"])
    V = max(P - target, 0)
    W = max(target - (P + T), 0) if yield_pct > 1.75 else 0
    r = js_round(P + T - V + W)
    return min(H * 0.9, r) if cap90 else r


def js_round(v):
    """Math.round, as the cap tab rounds"""
    return math.floor(v + 0.5)


def bisect(towns, total, cap):
    """The cap tab's solve(): the factor on both percentages that keeps the targets at the year's total, by 80 halvings
    of [0, 4]. A town with no foundation budget has no target under any cap."""
    lo, hi = 0.0, 4.0
    for _ in range(80):
        f = (lo + hi) / 2
        s = sum(0 if not t["foundation"] else t["cey"] * f if cap is None else min(t["cey"] * f, cap * t["foundation"])
                for t in towns)
        lo, hi = (f, hi) if s < total else (lo, f)
    return (lo + hi) / 2


def exact(towns, total, cap):
    """The same factor with a cap, solved exactly: sorted by where each town reaches the cap (cap x foundation /
    yield), the total is linear in f between those points"""
    ts = sorted((t for t in towns if t["foundation"]), key=lambda t: cap * t["foundation"] / t["cey"])
    capped = 0.0
    free = sum(t["cey"] for t in ts)
    for t in ts:
        f = (total - capped) / free
        if f <= cap * t["foundation"] / t["cey"]:
            return f
        capped += cap * t["foundation"]
        free -= t["cey"]
    raise ValueError("no factor reaches the total")


def check_year(FY, site, model):
    """One year's links: (the record's lines, the failed links' numbers, how many links ran)"""
    lines, failed, ran = [], [], []

    def link(n, title, ok, *body):
        lines.extend([f"### {n}. {title}: {'PASS' if ok else 'FAIL'}", "", *body, ""])
        ran.append(n)
        if not ok:
            failed.append(n)

    dese_path, dese_sha = cached(DESE_URL.format(fy=FY), f"chapter-{FY}.xlsm")
    dese, rates = dese_sheet(dese_path)
    csv_rows = {int(r["lea"]): r for r in csv.DictReader(TOWNWIDE_CSV.open()) if int(r["fy"]) == FY}
    cap = json.loads((site / "src/data/massppe/cap.json").read_text())
    year = cap["years"][str(FY)]
    site_towns = {t["lea"]: t for t in year["towns"]}

    # 1. The workbook against the CSV
    fields = ("eqv", "income", "cey", "foundation", "max_local", "target", "required")
    off = [(lea, f) for lea, d in dese.items() for f in fields
           if d[f] is not None and abs(d[f] - float(csv_rows[lea][f] or "nan")) > 0.5]
    link(1, "DESE's workbook to the pipeline's CSV", len(dese) == 351 and not off and set(dese) == set(csv_rows),
         f"- Workbook: {DESE_URL.format(fy=FY)}, SHA-256 `{dese_sha}`, read cell by cell.",
         f"- {len(dese)} towns; {len(fields)} figures each ({', '.join(fields)}); {len(off)} differ from "
         f"`data/dese_c70_townwide.csv` by more than $0.50." + (f" First: {off[:3]}" if off else ""))

    # 2. DESE's formula, restated
    yield_off = [lea for lea, d in dese.items() if abs(d["eqv"] * rates["property"] + d["income"] * rates["income"] - d["cey"]) > 1]
    target_off = [lea for lea, d in dese.items() if abs(min(d["cey"], d["max_local"]) - d["target"]) > 1]
    max_off = [lea for lea, d in dese.items() if abs(0.825 * d["foundation"] - d["max_local"]) > 1]
    share = sum(d["target"] for d in dese.values()) / sum(d["foundation"] for d in dese.values())
    link(2, "DESE's formula, restated", not (yield_off or target_off or max_off) and abs(share - 0.59) < 0.0005,
         f"- Percentages from the sheet: {rates['property'] * 100:.6f}% of EQV, {rates['income'] * 100:.6f}% of income.",
         f"- Yield = the two percentages times EQV and income, within $1: {351 - len(yield_off)} of 351 towns.",
         f"- Maximum = 82.5% of the foundation budget, within $1: {351 - len(max_off)} of 351.",
         f"- Target = the lesser of the yield and the maximum, within $1: {351 - len(target_off)} of 351.",
         f"- Targets total {share * 100:.3f}% of all foundation budgets (DESE's rule: 59%).")

    # 3. The CSV against cap.json
    pairs = (("cey", "cey"), ("foundation", "foundation"), ("max_local", "max"), ("target", "target"), ("required", "required"))
    off3 = [(lea, s) for lea, r in csv_rows.items() for c, s in pairs
            if r[c] and site_towns[lea][s] is not None and abs(float(r[c]) - site_towns[lea][s]) > 1]
    link(3, "The pipeline's CSV to the site's cap.json", set(site_towns) == set(csv_rows) and not off3,
         f"- {len(site_towns)} towns in cap.json FY{FY}; {len(pairs)} figures each; {len(off3)} differ by more than $1."
         + (f" First: {off3[:3]}" if off3 else ""),
         f"- cap.json built {cap['meta']['built']}.")

    # 4. The model, two ways
    towns = [dict(lea=lea, cey=d["cey"], foundation=d["foundation"]) for lea, d in dese.items()]
    total = sum(d["target"] for d in dese.values())
    has = [t for t in towns if t["foundation"]]
    closed = sum(d["target"] for d in dese.values() if d["foundation"]) / sum(t["cey"] for t in has)
    f_none = bisect(towns, total, None)
    nocap = {t["lea"]: js_round(t["cey"] * f_none) for t in towns}
    off4 = [lea for lea, v in nocap.items() if abs(v - site_towns[lea]["nocap"]) > 1]
    f100, f100x = bisect(towns, total, 1.0), exact(towns, total, 1.0)
    f825 = bisect(towns, total, 0.825)
    link(4, "The model, two ways", abs(f_none - closed) < 1e-9 and not off4 and abs(f100 - f100x) < 1e-9 and abs(f825 - 1) < 1e-6,
         f"- No cap: the closed form's factor {closed:.10f}, the bisection's {f_none:.10f}.",
         f"- No-cap targets by the bisection against cap.json's: {351 - len(off4)} of 351 within $1.",
         f"- A 100% cap: the bisection's factor {f100:.10f}, the exact solution's {f100x:.10f}.",
         f"- At DESE's 82.5% cap the bisection gives {f825:.10f}: today's percentages, as it should.",
         "- A town with a foundation budget of $0 (Gosnold) takes no part in setting the factor in either; with no cap "
         "its target is its yield times the factor, as in Naughton's workbook (Tim, #243), and under any cap $0.")

    # The model's required contributions, here, for links 5 to 7: each scenario's target, then the year's rules
    st = list(site_towns.values())
    def scenario_required(f, cap_, cap90=True):
        out = {}
        for t in st:
            tg = js_round(t["cey"] * f if cap_ is None else (0 if not t["foundation"] else min(t["cey"] * f, cap_ * t["foundation"])))
            out[t["lea"]] = (tg, required(t, tg, t["cey"] * f / t["foundation"] if t["foundation"] else 0, cap90))
        return out
    mine = {None: scenario_required(f_none, None), 1.0: scenario_required(f100, 1.0)}

    # 5. Naughton's workbook, for the years he figured
    if FY not in NAUGHTON:
        lines.extend(["### 5. An outside calculation: none for this year", "",
                      f"- No one outside this project has published FY{FY} targets under another cap to compare with.", ""])
    else:
        outside(FY, lines, link, dese, towns, total, f100, nocap, mine, scenario_required(f_none, None, False),
                scenario_required(f100, 1.0, False))

    # 6. The screen: the cap tab's own TypeScript (capModel.ts), run on cap.json, against the record's own arithmetic
    tm = model[str(FY)]
    off6 = [(lea, k) for k, c in (("nocap", None), ("cap100", 1.0)) for lea, (tg, rq) in mine[c].items()
            if abs(tm[k][str(lea)][0] - tg) > 1 or (rq is not None and abs((tm[k][str(lea)][1] or 0) - rq) > 1)
            or (rq is None) != (tm[k][str(lea)][1] is None)]
    link(6, "The cap tab's own model", not off6,
         "- `scripts/check/cap-model.mjs` in mywebsite runs `capModel.ts`, the model the cap tab draws with, on cap.json "
         "and prints every town's target and required contribution with no cap and with a 100% cap.",
         f"- Against this record's own arithmetic: {2 * 351 - len(off6)} of {2 * 351} town-scenarios agree within $1, "
         "target and required contribution both." + (f" First: {off6[:3]}" if off6 else ""))

    # 7. The income fifths, as the cap tab's "By income" table groups the towns
    pop = {}
    for r in csv.DictReader(POP_CSV.open()):
        pop[(re.sub(r"[^a-z]", "", r["Town"].lower()), int(r["FY"]))] = float(r["Population"])
    inc_year = int(year["income_year"])
    off7 = [t["name"] for t in st if t["pop_income"]
            and abs(pop.get((re.sub(r"[^a-z]", "", t["name"].lower()), inc_year + 1), -1) - t["pop_income"]) > 0.5]
    ranked = sorted((t for t in st if t["pop_income"] and t["income"]), key=lambda t: t["income"] / t["pop_income"])
    n = len(ranked)
    fifth = {t["lea"]: min(4, k * 5 // n) for k, t in enumerate(ranked)}
    off_ts = [lea for lea, q in fifth.items() if tm["fifth"][str(lea)] != q]
    groups = json.loads((site / "src/data/massppe/groups.json").read_text())
    explorer = {m["id"]: int(g["id"][-1]) - 1 for g in groups["towns"] if g["id"].startswith("towns-fifth-") for m in g["members"]}
    differ = sum(1 for t in ranked if explorer.get(t["id"]) != fifth[t["lea"]])
    rows = []
    for q in range(4, -1, -1):
        part = [t for t in ranked if fifth[t["lea"]] == q]
        ch = sum(mine[None][t["lea"]][0] - t["target"] for t in part)
        rows.append(f"| {['Lowest', 'Second', 'Middle', 'Fourth', 'Highest'][q]} | {len(part)} | "
                    f"{sum(1 for t in part if t['capped'])} | {'+' if ch > 0 else '−' if ch < 0 else ''}${abs(ch):,.0f} |")
    link(7, "The income fifths", not off7 and not off_ts and n == 351,
         f"- Income per capita: each town's {inc_year} income from DESE's sheet over the Census Bureau's population "
         f"estimate for July {inc_year} (`data/census_town_population.csv`): {len(st) - len(off7)} of {len(st)} towns' "
         "populations in cap.json match the Census file." + (f" Differ: {off7[:5]}" if off7 else ""),
         f"- Ranked and cut into five groups of 70 or 71: the cap tab's code puts {n - len(off_ts)} of {n} towns in the "
         "same fifth as this record.",
         f"- The explorer's income-fifth groups use the latest year's income; {differ} of {n} towns sit in a different "
         f"fifth there{' than in this year' if differ else ''}" + (", by design: each year here is cut on its own income." if differ else "."),
         f"- The change in targets with no cap, by fifth, highest first:", "",
         "| Fifth | Towns | At today's cap | Change in targets |", "| --- | --- | --- | --- |", *rows)
    return lines, failed, ran


def outside(FY, lines, link, dese, towns, total, f100, nocap, mine, mine_no90_none, mine_no90_100):
    """Link 5: Naughton's workbook against the model, targets and required contributions"""
    n_path, n_sha = cached(NAUGHTON[FY], Path(NAUGHTON[FY]).name)
    nw = naughton(n_path)
    alt100 = {t["lea"]: js_round(0 if not t["foundation"] else min(t["cey"] * f100, t["foundation"])) for t in towns}
    d100 = [abs(alt100[l] - v) for l, v in nw[1.0].items()]
    dnone = [(abs(nocap[l] - v), l) for l, v in nw[None].items()]
    d825 = [abs(dese[l]["target"] - v) for l, v in nw[0.825].items()]
    zero = {l for l, d in dese.items() if not d["foundation"]}
    rest = dnone
    worst = max(rest)
    sums = sum(nw[None].values()), sum(nocap.values())
    link(5, "An outside calculation: Michael Naughton's workbook", max(d100) <= 1 and worst[0] <= 1 and max(d825) <= 1,
         f"- Workbook: {NAUGHTON[FY]} (posted on {NAUGHTON_PAGE}), SHA-256 `{n_sha}`.",
         f"- At DESE's 82.5% cap: {sum(1 for d in d825 if d == 0)} of {len(d825)} towns match DESE's targets exactly.",
         f"- At a 100% cap: {sum(1 for d in d100 if d == 0)} of {len(d100)} towns match exactly, all within ${max(d100):,.0f}.",
         f"- With no cap: {sum(1 for d, _ in rest if d == 0)} of {len(rest)} towns match exactly, all within ${worst[0]:,.0f}.",
         *[f"- {dese[l]['town']}, with a foundation budget of $0: his no-cap target ${nw[None].get(l, 0):,.0f}, the model's "
           f"${nocap[l]:,.0f}. Both leave it out of the factor and then give it its yield, so with no cap the targets "
           f"total more than DESE's: his ${sums[0]:,.0f}, the model's ${sums[1]:,.0f} (DESE's, ${total:,.0f})."
           for l in sorted(zero)],
         "- His workbook starts from the same DESE spreadsheet, so agreement shows the arithmetic is reproducible, not "
         "that DESE's figures are right.")
    # Required contributions: the model keeps DESE's 90% rule; his figures leave it out
    def compare(ours, his_):
        both = [(l, ours[l][1], v) for l, v in his_.items() if ours.get(l) and ours[l][1] is not None]
        return both, [b for b in both if abs(b[1] - b[2]) > 1]
    for label, key, c, no90 in (("with no cap", "reqnone", None, mine_no90_none), ("with a 100% cap", "req100", 1.0, mine_no90_100)):
        both, off = compare(mine[c], nw[key])
        _, off_no90 = compare(no90, nw[key])
        binds = sum(1 for l in mine[c] if mine[c][l][1] is not None and no90[l][1] is not None and no90[l][1] > mine[c][l][1] + 1)
        ok = not off_no90 and len(off) == binds
        link("5b" if c is None else "5c", f"Naughton's required contributions {label}", ok,
             f"- {len(both)} towns with figures in both. With DESE's rule that a required contribution can't exceed 90% "
             f"of the foundation budget (the spreadsheet's column X), which the model keeps: {len(both) - len(off)} within "
             f"$1; the other {len(off)} are the towns where that rule applies.",
             f"- Without the rule, as his workbook figures them: {len(both) - len(off_no90)} of {len(both)} within $1.",
             *[f"- {dese[l]['town']}: DESE's FY{FY} sheet gives its prior-year figures as text, so the model has no "
               "required contribution for it." for l in sorted(set(nw[key]) - {b[0] for b in both})])


def main():
    site = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent.parent / "mywebsite"
    cap = json.loads((site / "src/data/massppe/cap.json").read_text())
    years = sorted(int(y) for y in cap["years"])
    # The cap tab's own model, run on cap.json by mywebsite's scripts/check/cap-model.mjs (link 6)
    printed = subprocess.run(["node", "scripts/check/cap-model.mjs"], cwd=site, capture_output=True, text=True, check=True)
    model = json.loads(printed.stdout)
    # 8. The site's own checks, once for every year: check:units, which every build runs
    run = subprocess.run(["npm", "run", "--silent", "check:units"], cwd=site, capture_output=True, text=True)
    said = (run.stdout + run.stderr).strip().splitlines()
    cap_fails = [l.strip() for l in said if "cap model" in l]
    screen_ok = run.returncode == 0 and not cap_fails
    screen = [f"## 8. The site's build checks, every year: {'PASS' if screen_ok else 'FAIL'}", "",
              "- `npm run check:units` in mywebsite runs `capModel.ts`, the model the cap tab draws with, on cap.json, for "
              "every year it has: each town's no-cap target against the pipeline's, within $1, and each required "
              "contribution the model's rules reproduce from today's targets.",
              f"- It reported: {said[-1] if said else 'nothing'}" + (f" Failures: {cap_fails}" if cap_fails else ""), ""]
    today = dt.date.today().isoformat()
    lines, summary = [], {}
    for fy in years:
        ls, failed, ran = check_year(fy, site, model)
        n = len(ran) + 1
        passed = n - len(failed) - (0 if screen_ok else 1)
        lines += [f"## FY{fy}: {passed} of {n} links pass" + (f"; reviewed by {REVIEWED[fy]}" if fy in REVIEWED else
                  "; not yet reviewed by a person"), "", *ls]
        summary[str(fy)] = {"links": n, "passed": passed, "outside": fy in NAUGHTON and 5 not in failed,
                            "outside_required": fy in NAUGHTON and "5b" not in failed and "5c" not in failed,
                            "fifths": 7 not in failed, "reviewed": REVIEWED.get(fy)}
    head = ["# Evidence record: the cap tab's targets under another cap", "",
            f"Written by `evidence/cap.py` on {today}. What this is and isn't: see the script's docstring. These are "
            "AI-written checks of DESE's spreadsheets; only link 5, where there is one, is another person's "
            "calculation, and it starts from the same spreadsheet.", "",
            "**To accept a year:** read its links below, then add the year to `REVIEWED` in `evidence/cap.py` with "
            "who and when (`2026: \"Tim Shores, 2026-11-09\"`), or ask Claude to, and rerun the script. The year's "
            "heading here, and the cap tab's Show the work, then say it was reviewed instead of \"not yet reviewed\".", ""]
    OUT.write_text("\n".join(head + lines + screen))
    (site / "src/data/massppe/evidence.json").write_text(json.dumps(
        {"cap": {"checked": today, "record": RECORD_URL, "years": summary}}, indent=1) + "\n")
    ok = all(v["passed"] == v["links"] for v in summary.values())
    print(f"[evidence] cap: " + "; ".join(f"FY{y} {v['passed']} of {v['links']}" for y, v in summary.items())
          + f" -> {OUT.relative_to(ROOT)}, {site.name}/src/data/massppe/evidence.json")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
