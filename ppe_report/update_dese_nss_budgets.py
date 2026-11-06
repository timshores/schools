"""
Download DESE's net school spending compliance summaries for every
Massachusetts district, for the explorer's budgeted and actual net school
spending panel.

Each year's summary (one workbook) gives every district's required and actual
net school spending (NSS) for one fiscal year, and its required and budgeted
NSS for the next, as the district reported them on its End-of-Year Pupil and
Financial Report. So FY2025's file holds FY2025's actual and FY2026's budget.
DESE links FY2021 on from its compliance page; the earlier files, FY2010 to
FY2020 under older names, survive only on the Internet Archive, some only in
later captures (an earlier one can be DESE's HTML page instead). An archived
copy is kept and reused; a year with no workbook anywhere is reported.

The workbooks go to data/dese_nss_compliance/ as they came; their rows, one
per district and fiscal year, to data/dese_nss_budgets.csv:

    lea, district, fy, budgeted_nss, budget_required_nss, actual_nss,
    actual_required_nss, budget_file, actual_file

budgeted_nss and budget_required_nss come from the file before fy's (the
budget and the requirement as they stood then), actual_nss and
actual_required_nss from fy's own.

    python update_dese_nss_budgets.py
"""

import datetime as dt
import io
import json
import re
import time
import urllib.request
from pathlib import Path

import pandas as pd

DATA_DIR = Path("./data")
RAW_DIR = DATA_DIR / "dese_nss_compliance"
OUT = DATA_DIR / "dese_nss_budgets.csv"
SOURCES_JSON = DATA_DIR / "dese_nss_budget_sources.json"
PAGE = "https://www.doe.mass.edu/finance/chapter70/compliance.html"
BASE = "https://www.doe.mass.edu/finance/chapter70/"
# Each file's name by the fiscal year of its actual spending: the current naming from FY2021, the older before
NAMES = {**{y: f"comply{y % 100:02d}.{'xls' if y < 2012 else 'xlsx'}" for y in range(2010, 2020)},
         **{y: f"comply-fy{y}.xlsx" for y in range(2020, 2031)}}
CDX = "https://web.archive.org/cdx/search/cdx?url={url}&output=json&fl=timestamp,statuscode&filter=statuscode:200"


def fetch(url: str, tries: int = 1) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for k in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except OSError:
            if k == tries - 1:
                raise
            time.sleep(5 * (k + 1))
    raise OSError(url)


def is_workbook(b: bytes) -> bool:
    """An xlsx (a zip) or an xls (an OLE file), not the HTML page a missing file returns"""
    return b[:4] == b"PK\x03\x04" or b[:8] == bytes.fromhex("d0cf11e0a1b11ae1")


def download(year: int, before: dict) -> dict | None:
    """The year's workbook: DESE's own, or else the Internet Archive's latest copy that is a workbook. An archived
    copy doesn't change, so one already here, from the run before, is used as it is."""
    name = NAMES[year]
    if "archived" in before and (RAW_DIR / name).exists():
        return {k: before[k] for k in ("file", "url", "archived")}
    try:
        b = fetch(BASE + name)
        if is_workbook(b):
            (RAW_DIR / name).write_bytes(b)
            return {"file": name, "url": BASE + name}
    except OSError:
        pass
    for scheme in ("https://www.", "http://www."):
        try:
            caps = json.loads(fetch(CDX.format(url=f"{scheme}doe.mass.edu/finance/chapter70/{name}"), 3))[1:]
        except (OSError, ValueError):
            continue
        for ts, _status in reversed(caps):
            url = f"https://web.archive.org/web/{ts}id_/{scheme}doe.mass.edu/finance/chapter70/{name}"
            try:
                b = fetch(url, 3)
            except OSError:
                continue
            if is_workbook(b):
                (RAW_DIR / name).write_bytes(b)
                return {"file": name, "url": url, "archived": ts[:8]}
    return None


def header_text(raw: pd.DataFrame, first: int) -> list[str]:
    """Each column's header, its words above the first district joined"""
    return [" ".join(str(v).replace("\n", " ") for v in raw.iloc[:first, j] if pd.notna(v)).lower()
            for j in range(raw.shape[1])]


def parse(path: Path) -> tuple[int, pd.DataFrame]:
    """The file's year and its districts: lea, district, actual and required NSS for the year, budgeted and
    required NSS for the next. Columns found by their headers, which moved over the years."""
    raw = pd.read_excel(path, header=None)
    top = " ".join(str(v) for v in raw.iloc[:10].values.ravel() if pd.notna(v))
    m = re.search(r"FY\s?(\d{2,4}) and FY\s?(\d{2,4})", top)
    if not m:
        raise ValueError(f"{path.name}: no 'FYxx and FYyy' in its title")
    year = 2000 + int(m.group(1)) % 100
    code = raw.iloc[:, 0].map(lambda v: str(int(v)) if isinstance(v, (int, float)) and pd.notna(v) else str(v).strip())
    rows = code.str.fullmatch(r"\d{1,4}")
    first = int(rows.idxmax())
    heads = header_text(raw, first)
    nss = lambda h: ("net school" in h or "nss" in h) and not any(w in h for w in ("%", "pct", "over", "chg"))
    cols = {"budgeted": [j for j, h in enumerate(heads) if "budgeted" in h and nss(h)],
            "actual": [j for j, h in enumerate(heads) if "actual" in h and nss(h)],
            "required": [j for j, h in enumerate(heads) if "required" in h and nss(h)]}
    if len(cols["budgeted"]) != 1 or len(cols["actual"]) != 1 or len(cols["required"]) != 2:
        raise ValueError(f"{path.name}: unexpected headers {cols}")
    b, a = cols["budgeted"][0], cols["actual"][0]
    r0, r1 = cols["required"]
    if not r0 < a < r1 < b:
        raise ValueError(f"{path.name}: columns out of order {cols}")
    df = raw[rows].copy()
    out = pd.DataFrame({"lea": code[rows].str.zfill(4), "district": df.iloc[:, 1].astype(str).str.strip(),
                        "actual_nss": pd.to_numeric(df.iloc[:, a], errors="coerce"),
                        "actual_required_nss": pd.to_numeric(df.iloc[:, r0], errors="coerce"),
                        "budgeted_nss": pd.to_numeric(df.iloc[:, b], errors="coerce"),
                        "budget_required_nss": pd.to_numeric(df.iloc[:, r1], errors="coerce")})
    # The totals rows (999, 9999) aren't districts
    out = out[~out["lea"].isin(["0999", "9999"]) & (out["district"] != "nan")]
    return year, out


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    today = dt.date.today()
    got, missing, parts = {}, [], []
    before = json.loads(SOURCES_JSON.read_text())["files"] if SOURCES_JSON.exists() else {}
    for year in sorted(NAMES):
        if year > today.year:
            break
        src = download(year, before.get(str(year), {}))
        if not src:
            missing.append(year)
            continue
        y, df = parse(RAW_DIR / src["file"])
        if y != year:
            raise ValueError(f"{src['file']} is FY{y}'s, not FY{year}'s")
        got[year] = src | {"districts": len(df)}
        parts.append(df.assign(year=year, file=src["file"]))
        print(f"  FY{year}: {src['file']}{' (Internet Archive, ' + src['archived'] + ')' if 'archived' in src else ''}, "
              f"{len(df)} districts, FY{year + 1} budgets {df['budgeted_nss'].notna().sum()}")
    # A year past the newest file isn't missing, only not out yet
    missing = [y for y in missing if y < max(got)]
    allf = pd.concat(parts)
    actual = allf[["lea", "district", "year", "actual_nss", "actual_required_nss", "file"]].rename(
        columns={"year": "fy", "file": "actual_file"})
    budget = allf[["lea", "district", "year", "budgeted_nss", "budget_required_nss", "file"]].rename(
        columns={"year": "fy", "file": "budget_file"})
    budget["fy"] += 1
    out = actual.merge(budget, on=["lea", "fy"], how="outer", suffixes=("", "_b"))
    out["district"] = out["district"].fillna(out.pop("district_b"))
    out = out[["lea", "district", "fy", "budgeted_nss", "budget_required_nss", "actual_nss", "actual_required_nss",
               "budget_file", "actual_file"]].sort_values(["fy", "lea"])
    for c in ("budgeted_nss", "budget_required_nss", "actual_nss", "actual_required_nss"):
        out[c] = out[c].round(2)
    out.to_csv(OUT, index=False)
    b = out.dropna(subset=["budgeted_nss"])
    gap = f" (none for FY{', FY'.join(str(y + 1) for y in missing)})" if missing else ""
    print(f"  {OUT}: {len(out):,} rows; budgets FY{b['fy'].min()}-FY{b['fy'].max()}{gap}, {b['lea'].nunique()} districts")
    SOURCES_JSON.write_text(json.dumps({
        "source": "DESE, Compliance With Net School Spending Requirements: " + PAGE,
        "files": {str(k): v for k, v in got.items()},
        "not_found": missing,
        "downloaded": today.isoformat(),
    }, indent=1) + "\n")
    print(f"  {SOURCES_JSON}")


if __name__ == "__main__":
    main()
