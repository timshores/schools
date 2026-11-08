"""
The evidence record for the explorer's no-cap counterfactual, FY2026 (Project Regain Custody, Pass 2; seahag
prompt log #242). The cap tab says what each town's Chapter 70 target local contribution would be under another
cap, or none, keeping the statewide total. This checks every link from DESE's own spreadsheet to the figures the
site draws, and writes what it found to evidence/cap_fy2026.md.

    1. DESE's workbook -> data/dese_c70_townwide.csv: every town's figures, read again from the cached workbook
       cell by cell (openpyxl), not by update_c70_formula.py's pandas path
    2. DESE's formula, restated: yield = the two percentages times EQV and income; target = the lesser of the
       yield and the maximum (82.5% of the foundation budget); the targets total 59% of all foundation budgets
    3. the CSV -> the site's cap.json, town by town
    4. the model, two ways: the pipeline's closed form (nocap_factor) and a bisection like the cap tab's
       (cap.ts solve), with no cap and with a 100% cap, solved a third way for the cap, exactly
    5. an outside calculation: Michael Naughton's FY2026 workbook, which MASS Promise to Invest posted
    6. the screen: the cap tab's own model (mywebsite src/lib/massppe/capModel.ts), run on cap.json by mywebsite's
       check:units, which every build runs: its no-cap targets and required contributions against the pipeline's

These are checks an AI wrote of figures from one DESE spreadsheet. Only link 5 is someone else's calculation, and
it starts from the same spreadsheet. None of this is independent verification of DESE's figures.

    python evidence/cap_fy2026.py [SITE_ROOT]    # from ppe_report; SITE_ROOT defaults to ../../mywebsite
Downloads the two workbooks into data/c70_raw/ (git ignores it) if they aren't there. Exits 1 if a link fails.
"""

import csv
import datetime as dt
import hashlib
import json
import math
import subprocess
import sys
import urllib.request
from pathlib import Path

import openpyxl

FY = 2026
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RAW = ROOT / "data" / "c70_raw"
DESE_URL = f"https://www.doe.mass.edu/finance/chapter70/fy{FY}/chapter-{FY}.xlsm"
NAUGHTON_URL = "https://masspromise2invest.org/wp-content/uploads/2025/10/fy26-chapter-70-cap-comparisons.xlsx"
NAUGHTON_PAGE = "https://masspromise2invest.org/information-2/"
TOWNWIDE_CSV = ROOT / "data" / "dese_c70_townwide.csv"
SHEET = "Townwide Contributions"
OUT = HERE / f"cap_fy{FY}.md"


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
    return out


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


def main():
    site = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent.parent / "mywebsite"
    lines, failed = [], []

    def link(n, title, ok, *body):
        lines.extend([f"## {n}. {title}: {'PASS' if ok else 'FAIL'}", "", *body, ""])
        if not ok:
            failed.append(n)

    dese_path, dese_sha = cached(DESE_URL, f"chapter-{FY}.xlsm")
    n_path, n_sha = cached(NAUGHTON_URL, "fy26-chapter-70-cap-comparisons.xlsx")
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
         f"- Workbook: {DESE_URL}, SHA-256 `{dese_sha}`, read cell by cell.",
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
    nocap = {t["lea"]: (js_round(t["cey"] * f_none) if t["foundation"] else 0) for t in towns}
    off4 = [lea for lea, v in nocap.items() if abs(v - site_towns[lea]["nocap"]) > 1]
    f100, f100x = bisect(towns, total, 1.0), exact(towns, total, 1.0)
    f825 = bisect(towns, total, 0.825)
    link(4, "The model, two ways", abs(f_none - closed) < 1e-9 and not off4 and abs(f100 - f100x) < 1e-9 and abs(f825 - 1) < 1e-6,
         f"- No cap: the closed form's factor {closed:.10f}, the bisection's {f_none:.10f}.",
         f"- No-cap targets by the bisection against cap.json's: {351 - len(off4)} of 351 within $1.",
         f"- A 100% cap: the bisection's factor {f100:.10f}, the exact solution's {f100x:.10f}.",
         f"- At DESE's 82.5% cap the bisection gives {f825:.10f}: today's percentages, as it should.",
         f"- Gosnold, with a foundation budget of $0, has no target under any cap in both.")

    # 5. Naughton's workbook
    nw = naughton(n_path)
    alt100 = {t["lea"]: js_round(0 if not t["foundation"] else min(t["cey"] * f100, t["foundation"])) for t in towns}
    d100 = [abs(alt100[l] - v) for l, v in nw[1.0].items()]
    dnone = [(abs(nocap[l] - v), l) for l, v in nw[None].items()]
    d825 = [abs(dese[l]["target"] - v) for l, v in nw[0.825].items()]
    # A town with no foundation budget is where the two models differ by design: he gives it a no-cap target
    zero = {l for l, d in dese.items() if not d["foundation"]}
    rest = [(d, l) for d, l in dnone if l not in zero]
    worst = max(rest)
    sums = sum(nw[None].values()), sum(nocap.values())
    link(5, "An outside calculation: Michael Naughton's workbook", max(d100) <= 1 and worst[0] <= 1 and max(d825) <= 1,
         f"- Workbook: {NAUGHTON_URL} (posted on {NAUGHTON_PAGE}), SHA-256 `{n_sha}`.",
         f"- At DESE's 82.5% cap: {sum(1 for d in d825 if d == 0)} of {len(d825)} towns match DESE's targets exactly.",
         f"- At a 100% cap: {sum(1 for d in d100 if d == 0)} of {len(d100)} towns match exactly, all within ${max(d100):,.0f}.",
         f"- With no cap: {sum(1 for d, _ in rest if d == 0)} of the {len(rest)} towns with a foundation budget match "
         f"exactly, all within ${worst[0]:,.0f}.",
         *[f"- {dese[l]['town']}, with a foundation budget of $0: he gives it a no-cap target of ${nw[None].get(l, 0):,.0f}; "
           "the model gives it none, since the formula asks a town for a share of its foundation budget. That accounts "
           f"for the difference in the totals, give or take rounding: his ${sums[0]:,.0f}, the model's ${sums[1]:,.0f} "
           f"(DESE's, ${total:,.0f})."
           for l in sorted(zero)],
         "- His workbook starts from the same DESE spreadsheet, so agreement shows the arithmetic is reproducible, not "
         "that DESE's figures are right.")

    # 6. The screen: the browser's model, run by mywebsite's own check
    run = subprocess.run(["npm", "run", "--silent", "check:units"], cwd=site, capture_output=True, text=True)
    said = (run.stdout + run.stderr).strip().splitlines()
    cap_fails = [l.strip() for l in said if "cap model" in l]
    link(6, "The cap tab's own model", run.returncode == 0 and not cap_fails,
         "- `npm run check:units` in mywebsite runs `capModel.ts`, the model the cap tab draws with, on cap.json, for "
         "every year it has: each town's no-cap target against the pipeline's, within $1, and each required "
         "contribution the model's rules reproduce from today's targets.",
         f"- It reported: {said[-1] if said else 'nothing'}" + (f" Failures: {cap_fails}" if cap_fails else ""))

    head = [f"# Evidence record: the no-cap counterfactual, FY{FY}", "",
            f"Written by `evidence/cap_fy{FY}.py` on {dt.date.today().isoformat()}. "
            f"{'Every link passed.' if not failed else f'Failed: links {failed}.'} What this is and isn't: see the "
            "script's docstring. These are AI-written checks of one DESE spreadsheet's figures; only link 5 is "
            "another person's calculation, and it starts from the same spreadsheet.", ""]
    OUT.write_text("\n".join(head + lines))
    print(f"[evidence] cap FY{FY}: {6 - len(failed)} of 6 links pass -> {OUT.relative_to(ROOT)}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
