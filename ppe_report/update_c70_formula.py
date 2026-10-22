"""
Download DESE's complete Chapter 70 formula spreadsheets and keep each town's
local contribution calculation, the "Townwide Contributions" sheet:

    data/dese_c70_townwide.csv
        fy, lea, town, eqv (and eqv_year), income (and income_year), the
        statewide property and income percentages, combined effort yield,
        foundation budget, maximum local contribution (82.5% of foundation),
        target local contribution, last year's required contribution, MRGF,
        increments and reductions, and the required contribution.

The target local contribution is the combined effort yield (property
percentage x EQV + income percentage x income) or the maximum, 82.5% of the
town's foundation budget, whichever is less. DESE sets the two percentages so
that the targets, after the cap, add up to 59% of all foundation budgets; the
white paper for each year gives them. The workbooks themselves (.xlsm, with
macros that are never run here) are cached in data/c70_raw/.

    python update_c70_formula.py            # FY2026 and FY2027
    python update_c70_formula.py 2025 2026  # other years
"""

import datetime as dt
import json
import re
import sys
import urllib.request
from pathlib import Path

import pandas as pd

DATA_DIR = Path("./data")
RAW_DIR = DATA_DIR / "c70_raw"
URL = "https://www.doe.mass.edu/finance/chapter70/fy{fy}/chapter-{fy}.xlsm"
OUT = DATA_DIR / "dese_c70_townwide.csv"
SOURCES_JSON = DATA_DIR / "dese_c70_townwide_sources.json"
SHEET = "Townwide Contributions"
# Sheet column: field. The header row (row 9) is checked against these words.
COLUMNS = {
    "A": ("lea", "LEA"), "B": ("town", "Town"), "C": ("eqv", "EQV"), "D": ("property_effort", "property"),
    "E": ("income", "income"), "F": ("income_effort", "income"), "G": ("cey", "combined effort yield"),
    "H": ("foundation", "foundation"), "I": ("cey_pct", "CEY"), "J": ("max_local", "maximum local"),
    "K": ("target", "target local contribution"), "L": ("target_share", "target local share"),
    "N": ("rlc_prior", "required local contribution"), "O": ("mrgf", "mrgf"),
    "P": ("preliminary", "preliminary contribution"), "S": ("pct_increment", "percentage increment"),
    "T": ("increment", "dollar increment"), "V": ("excess_reduction", "excess effort times"),
    "W": ("floor_increment", "175%"), "X": ("required", "required contribution"),
}


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def townwide(fy: int) -> tuple[pd.DataFrame, dict]:
    """One year's sheet: a row per town, and the year's percentages and labels."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"chapter-{fy}.xlsm"
    path.write_bytes(fetch(URL.format(fy=fy)))
    df = pd.read_excel(path, sheet_name=SHEET, header=None, usecols="A:X", engine="openpyxl")
    df.columns = [chr(65 + i) for i in range(df.shape[1])]
    # Find rows by their labels: pandas skips some of the sheet's blank rows
    head_at = df.index[df["A"].astype(str).str.strip() == "LEA"][0]
    rates_at = df.index[df["D"].astype(str).str.contains("uniform property percentage", na=False)][0] + 1
    header = {c: str(df.at[head_at, c]) for c in COLUMNS}
    for c, (_, words) in COLUMNS.items():
        if words.lower() not in header[c].lower():
            raise ValueError(f"FY{fy} {SHEET} column {c} reads {header[c]!r}, not {words!r}; has the sheet changed?")
    eqv_year = int(re.search(r"(\d{4})", header["C"]).group(1))
    income_year = int(re.search(r"(\d{4})", header["E"]).group(1))
    rates = {"property_pct": float(df.at[rates_at, "D"]), "income_pct": float(df.at[rates_at, "F"])}
    towns = df[pd.to_numeric(df["A"], errors="coerce").between(1, 351) & df["B"].map(lambda v: isinstance(v, str))
               & pd.to_numeric(df["C"], errors="coerce").notna()]
    out = towns[list(COLUMNS)].rename(columns={c: f for c, (f, _) in COLUMNS.items()})
    out["town"] = out["town"].str.strip()
    # A few cells hold text, not numbers; report them and leave them blank
    for f in [f for f, _ in COLUMNS.values() if f not in ("lea", "town")]:
        num = pd.to_numeric(out[f], errors="coerce")
        for town, v in out.loc[num.isna() & out[f].notna(), ["town", f]].itertuples(index=False):
            print(f"  [WARN] FY{fy} {town} {f}: {v!r} is not a number; left blank")
        out[f] = num
    out.insert(0, "fy", fy)
    out.insert(4, "eqv_year", eqv_year)
    out.insert(6, "income_year", income_year)
    # The cap, as each town's maximum over its foundation budget (the sheet's own cell is blank some years)
    has = out["foundation"] > 0
    rates["max_pct"] = round(float((out.loc[has, "max_local"] / out.loc[has, "foundation"]).median()), 4)
    for k, v in rates.items():
        out[k] = v
    out["source_url"] = URL.format(fy=fy)
    if len(out) != 351:
        raise ValueError(f"FY{fy}: {len(out)} towns, not 351")
    return out, {"eqv_year": eqv_year, "income_year": income_year, **rates}


def main(years: list[int]):
    frames, meta = [], {}
    for fy in years:
        print(f"[DESE] FY{fy} complete formula spreadsheet...")
        df, m = townwide(fy)
        frames.append(df)
        meta[fy] = m
        share = df["target"].sum() / df["foundation"].sum()
        print(f"  {len(df)} towns; EQV {m['eqv_year']}, income {m['income_year']}; property {m['property_pct']:.6%}, "
              f"income {m['income_pct']:.6%}; targets {df['target'].sum() / 1e9:.3f}B = {share:.2%} of foundation; "
              f"{((df['target'] >= df['max_local']) & (df['foundation'] > 0)).sum()} at the maximum")
    pd.concat(frames, ignore_index=True).to_csv(OUT, index=False)
    SOURCES_JSON.write_text(json.dumps({
        "source": "DESE Chapter 70 complete formula spreadsheet, Townwide Contributions sheet",
        "urls": {str(fy): URL.format(fy=fy) for fy in years},
        "years": {str(fy): m for fy, m in meta.items()},
        "downloaded": dt.date.today().isoformat(),
    }, indent=1) + "\n")
    print(f"  {OUT}; {SOURCES_JSON}")


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [2026, 2027])
