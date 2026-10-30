#!/usr/bin/env python3
"""update_dese_history.py: DESE's per-pupil expenditure files from before FY2009, normalized.

DESE's Education-to-Career Hub dataset behind data/dese_expenditures.csv starts in FY2009. The
archive in data/dese_archive/ came from DESE's Public Records Center and reaches back to FY1990, in
two formats: scanned and OCR'd PDF reports for FY1990 to FY1999, and the Office of School Finance's
Excel workbooks for FY2001 to FY2008. This script reads both and writes three CSVs:

  data/dese_expenditures_fy2005_2008.csv
      Spending per pupil by DESE's functional category, and FTE enrollment, FY2005 to FY2008. Same
      columns and the same IND_CAT/IND_SUBCAT labels as dese_expenditures.csv, so the two can be
      stacked. DESE's workbooks call the teachers category "Classroom and Specialist Teachers";
      the Hub calls it "Teachers", and so does this file.
  data/dese_day_program_fy1990_2006.csv
      DESE's older measure, "day program" spending per pupil by program (regular day, special
      needs, bilingual, occupational day) and in total, FY1990 to FY2006; total spending and
      average membership from FY1997. It is a narrower measure than the functional total (see
      data/dese_archive/README.md) and is kept apart for that reason.
  data/dese_day_program_state_fy1989_2006.csv
      DESE's published statewide day-program averages, FY1989 to FY2006.

Every row names the archive file and the sheet and row, or the page, it came from. The OCR'd pages
are checked three ways and the results written to data/dese_archive/qa/: the name printed beside
each code must match DESE's name for that code; each total must fall within the range of the
district's program figures, since it is their weighted average; and FY1997 to FY1999, which
appear both as PDF pages and in the workbooks, must agree. Values read by eye from a page image
are listed in CORRECTIONS with the reason.

Usage (from ppe_report/):  python update_dese_history.py
Needs pdftotext (poppler) and xlrd.
"""
import csv
import math
import difflib
import html
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import xlrd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
ARCHIVE = DATA / "dese_archive"
QA = ARCHIVE / "qa"
MODERN = DATA / "dese_expenditures.csv"
OUT_FUNCTIONAL = DATA / "dese_expenditures_fy2005_2008.csv"
OUT_DAY = DATA / "dese_day_program_fy1990_2006.csv"
OUT_STATE = DATA / "dese_day_program_state_fy1989_2006.csv"

REPO_URL = "https://github.com/timshores/schools/blob/main/ppe_report/data/dese_archive/"
RECEIVED = "2026-10-28"  # the date the archive files were received from DESE's Public Records Center
SCRIPT = "update_dese_history.py"

FIELDS = ["SY", "DIST_CODE", "DIST_NAME", "IND_CAT", "IND_SUBCAT", "IND_VALUE", "IND_VALUE_TYPE",
          "Source_and_Data_Dictionary", "Last_Updated", "Download_Date", "Download_By",
          "SOURCE_FILE", "SOURCE_LOCATION", "SOURCE_NAME"]

# ---------------------------------------------------------------------------------------------
# Sources

# FY: (file, first page, last page) of the "Summary of All Day Programs" table in each report
PDF_SUMMARY = {
    1990: ("PER_PUPIL_EXPENDITURE_1989-1990.pdf", 13, 21),
    1991: ("PER_PUPIL_EXPENDITURE_1990-1991_17351741.pdf", 12, 21),
    1992: ("PER_PUPIL_EXPENDITURE_1991-1992.pdf", 13, 21),
    1993: ("PER_PUPIL_EXPENDITURE_1992-1993.pdf", 13, 20),
    1994: ("PER_PUPIL_EXPENDITURE_1993-1994.pdf", 19, 27),
    1995: ("PER_PUPIL_EXPENDITURE_1994-1995.pdf", 19, 27),
    1996: ("PER_PUPIL_EXPENDITURE_FY96.pdf", 20, 28),
    1997: ("PER_PUPIL_EXPENDITURE_FY97.pdf", 20, 28),
    1998: ("PER_PUPIL_EXPENDITURE_FY98.pdf", 19, 30),
    1999: ("PER_PUPIL_EXPENDITURE_FY99.pdf", 18, 27),
}
# Years whose PDF table is used only to check the workbooks, which hold the same figures unrounded
PDF_CHECK_ONLY = {1997, 1998, 1999}

# Five-year trend sheets ("5yrpp"); each year is taken from the latest workbook that has it
FIVE_YEAR = ["perpupil_post01_final.xls", "perpupil_post02_1mar04_final.xls",
             "perpupil_post03_27aug04_final.xls", "perpupil_post04_22aug05_final.xls"]
FY05_FILE = "calcfunctional05final.xls"
FY06_DAY_FILE = "ppx06_day_program_FY06_final.xls"
# FY: (file, sheet) holding the functional calculation's long table for that year
FUNCTIONAL = {
    2005: ("calcfunctional05final.xls", "Data"),
    2006: ("calcfunctional08final.xls", "data06"),
    2007: ("calcfunctional07dec.xls", "Data"),
    2008: ("calcfunctional08final.xls", "Data"),
}
FUNCTIONAL_CHECK = {2007: ("calcfunctional08final.xls", "data07")}

# When DESE published each file: the report's cover date (FY1991's cover has none, so the print
# date on its table pages), or the date in a workbook's file name. Written as Last_Updated.
PUBLISHED = {
    "PER_PUPIL_EXPENDITURE_1989-1990.pdf": "1991-07", "PER_PUPIL_EXPENDITURE_1990-1991_17351741.pdf": "1992-07",
    "PER_PUPIL_EXPENDITURE_1991-1992.pdf": "1993-07", "PER_PUPIL_EXPENDITURE_1992-1993.pdf": "1994-09",
    "PER_PUPIL_EXPENDITURE_1993-1994.pdf": "1995-12", "PER_PUPIL_EXPENDITURE_1994-1995.pdf": "1997-04",
    "PER_PUPIL_EXPENDITURE_FY96.pdf": "1998-10", "PER_PUPIL_EXPENDITURE_FY97.pdf": "1998-12",
    "PER_PUPIL_EXPENDITURE_FY98.pdf": "1999-11", "PER_PUPIL_EXPENDITURE_FY99.pdf": "2001-01",
    "perpupil_post02_1mar04_final.xls": "2004-03-01", "perpupil_post03_27aug04_final.xls": "2004-08-27",
    "perpupil_post04_22aug05_final.xls": "2005-08-22", "calcfunctional07dec.xls": "2007-12",
}

PROGRAMS = ["Regular Day", "Special Needs", "Bilingual", "Occupational Day"]
TOTAL_DAY = "Total, All Day Programs"
DAY_CAT = "Day Program Expenditures Per Pupil"

# Functional summary rows, by DESE's label, and the Hub's label for each
FUNCTIONAL_LABELS = {
    "Administration": "Administration",
    "Instructional Leadership": "Instructional Leadership",
    "Classroom and Specialist Teachers": "Teachers",
    "Other Teaching Services": "Other Teaching Services",
    "Professional Development": "Professional Development",
    "Instructional Materials, Equipment and Technology": "Instructional Materials, Equipment and Technology",
    "Guidance, Counseling and Testing": "Guidance, Counseling and Testing",
    "Pupil Services": "Pupil Services",
    "Operations and Maintenance": "Operations and Maintenance",
    "Insurance, Retirement Programs and Other": "Insurance, Retirement Programs and Other",
}

# Values read by eye from the page image where the OCR text was wrong or missing.
# (FY, code, column): (value, note).
CORRECTIONS = {
    (1990, 766, "Regular Day"): (3748, "page image reads 3,748; OCR gave '3,7.48'"),
    (1996, 57, "Special Needs"): (11057, "page image reads 11,057; OCR split it into '1' and '057'"),
}
# Rows that failed a check but were read against the page image and found printed as parsed:
# DESE's own figures, however odd (Heath's tiny regular-day figures and low totals among them).
VERIFIED = {
    (1990, 130): "page 15", (1991, 130): "page 14", (1992, 130): "page 15", (1993, 130): "page 15",
    (1992, 260): "page 17", (1993, 169): "page 15", (1994, 655): "page 26", (1995, 109): "page 20",
}


# ---------------------------------------------------------------------------------------------
# Helpers

def dese_code(lea) -> str:
    return f"{int(float(lea)):04d}0000"


def clean_name(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^A-Za-z0-9 \-'.&]", " ", str(s))).strip()


def title(s: str) -> str:
    small = {"Of", "And", "The"}
    words = [w.capitalize() for w in clean_name(s).split()]
    return " ".join(w.lower() if i and w in small else w for i, w in enumerate(words))


def num(v):
    if v in ("", None):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def modern_names() -> dict:
    """DESE's current name for each code, from the Hub dataset's latest year."""
    names = {}
    with open(MODERN, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            names[r["DIST_CODE"]] = (int(r["SY"]), r["DIST_NAME"])
    return {code: name for code, (_, name) in names.items()}


def row(sy, code, name, cat, subcat, value, vtype, src_file, location, printed, updated=""):
    return {"SY": sy, "DIST_CODE": code, "DIST_NAME": name, "IND_CAT": cat, "IND_SUBCAT": subcat,
            "IND_VALUE": value, "IND_VALUE_TYPE": vtype, "Source_and_Data_Dictionary": REPO_URL + src_file,
            "Last_Updated": updated or PUBLISHED.get(src_file, ""), "Download_Date": RECEIVED, "Download_By": SCRIPT,
            "SOURCE_FILE": src_file, "SOURCE_LOCATION": location, "SOURCE_NAME": printed}


def write_csv(path: Path, rows: list, fields=FIELDS):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path.relative_to(HERE)}: {len(rows):,} rows")


# ---------------------------------------------------------------------------------------------
# Workbooks

def read_five_year(fname):
    """Rows of a 5yrpp sheet: (fy, lea, name, programs, spending, membership, total, row index)."""
    sh = xlrd.open_workbook(ARCHIVE / fname, on_demand=True).sheet_by_name("5yrpp")
    out = []
    for r in range(sh.nrows):
        v = sh.row_values(r)
        m = re.fullmatch(r"FY(\d\d)", str(v[4]).strip())
        if not m or num(v[2]) is None:
            continue
        yy = int(m.group(1))
        fy = 1900 + yy if yy > 50 else 2000 + yy
        progs = dict(zip(PROGRAMS, (num(x) for x in v[5:9])))
        out.append((fy, int(float(v[2])), str(v[3]).strip(), progs, num(v[10]), num(v[11]), num(v[12]), r + 1))
    return out


def read_fy05_day():
    sh = xlrd.open_workbook(ARCHIVE / FY05_FILE, on_demand=True).sheet_by_name("ppx vs funct")
    out = []
    for r in range(sh.nrows):
        v = sh.row_values(r)
        if num(v[0]) is None or not str(v[1]).strip():
            continue
        out.append((int(float(v[0])), str(v[1]).strip(), num(v[2]), num(v[3]), num(v[4]), r + 1))
    return out


def read_fy06_day():
    wb = xlrd.open_workbook(ARCHIVE / FY06_DAY_FILE, on_demand=True)
    sh = wb.sheet_by_name("list")
    out = []
    for r in range(sh.nrows):
        v = sh.row_values(r)
        # Codes below 100 are text ("091"), the rest numbers (100.0); 999 is the state total
        if not re.fullmatch(r"\d{1,3}(\.0)?", str(v[0]).strip()) or int(float(v[0])) == 999:
            continue
        progs = {"Regular Day": num(v[2]), "Special Needs": num(v[3]), "Bilingual": None,
                 "Occupational Day": num(v[4])}
        out.append((int(v[0]), str(v[1]).strip(), progs, num(v[5]), num(v[6]), num(v[7]), r + 1))
    state = wb.sheet_by_name("state")
    total = None
    for r in range(state.nrows):
        v = state.row_values(r)
        if str(v[0]).strip().startswith("8. Per Pupil"):
            total = {"Regular Day": num(v[1]), "Special Needs": num(v[2]), "Occupational Day": num(v[3]),
                     TOTAL_DAY: num(v[4]), "_row": r + 1}
    return out, total


def read_functional(fname, sheet):
    """{lea: {"name", "fte": {...}, "cats": {label: (all funds, per pupil)}, "rows": {label: row}}}."""
    sh = xlrd.open_workbook(ARCHIVE / fname, on_demand=True).sheet_by_name(sheet)
    off = None
    out = {}
    for r in range(sh.nrows):
        v = sh.row_values(r)
        if off is None:
            for i in range(0, 3):
                if re.fullmatch(r"\d{3}", str(v[i]).strip()) and str(v[i + 1]).strip():
                    off = i - 1
                    break
            if off is None:
                continue
        lea, name, sum_id, label = v[1 + off], v[2 + off], num(v[3 + off]), str(v[4 + off]).strip()
        # Codes below 100 are text ("001"), the rest numbers (100.0)
        if not re.fullmatch(r"\d{1,3}(\.0)?", str(lea).strip()) or not sum_id:
            continue
        d = out.setdefault(int(lea), {"name": str(name).strip(), "fte": {}, "cats": {}, "rows": {}})
        vals = [num(x) for x in v[7 + off:13 + off]]
        d["rows"][label] = r + 1
        if "FTE" in label:
            d["fte"][label] = vals[5]
        elif label.startswith("percentage"):
            continue
        else:
            d["cats"][label] = (vals[2], vals[4])
    return out


# ---------------------------------------------------------------------------------------------
# OCR'd PDF tables

WORD = re.compile(r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">(.*?)</word>')
HEADERS = {"REGULAR": "Regular Day", "SPECIAL": "Special Needs", "BILINGUAL": "Bilingual",
           "OCCUPATIONAL": "Occupational Day", "TOTAL": TOTAL_DAY}


def page_words(pdf, page):
    out = subprocess.run(["pdftotext", "-f", str(page), "-l", str(page), "-bbox", str(ARCHIVE / pdf), "-"],
                         capture_output=True, text=True, check=True).stdout
    return [(float(a), float(b), float(c), float(d), html.unescape(t)) for a, b, c, d, t in WORD.findall(out)]


def header_anchors(words):
    """Column centres from the header words, where the OCR kept them."""
    anchors = {}
    for x0, y0, x1, y1, t in words:
        key = re.sub(r"[^A-Z]", "", t.upper())
        for h, col in HEADERS.items():
            if key and (key == h or (len(key) >= 5 and difflib.SequenceMatcher(None, key, h).ratio() > 0.8)):
                if col not in anchors or y0 < anchors[col][1]:
                    anchors[col] = ((x0 + x1) / 2, (y0 + y1) / 2)
    return anchors


def parse_number(t):
    """A printed whole-dollar figure. Returns (value, cleaned): cleaned is True when OCR noise
    (a stray mark, a comma read as a period, a lost comma) had to be removed to read it."""
    s = re.sub(r"[^\d,.]", "", t).strip(",.")
    if re.fullmatch(r"\d{1,3}[,.]\d{3}", s) or re.fullmatch(r"\d{3,5}", s):
        return int(s.replace(",", "").replace(".", "")), s != t.strip()
    return None, False


def median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else 0.0


def parse_summary(fy):
    pdf, first, last = PDF_SUMMARY[fy]
    pages = {p: page_words(pdf, p) for p in range(first, last + 1)}
    found_xy = {p: header_anchors(w) for p, w in pages.items()}
    found = {p: {c: xy[0] for c, xy in a.items()} for p, a in found_xy.items()}
    # A column's position varies little within one report; fill gaps from the report's other pages
    doc = {c: sorted(a[c] for a in found.values() if c in a) for c in HEADERS.values()}
    doc = {c: xs[len(xs) // 2] for c, xs in doc.items() if xs}
    rows = []
    for p, words in pages.items():
        anchors = {**doc, **found[p]}
        if len(anchors) < 5:
            raise SystemExit(f"FY{fy} p{p}: columns not found ({sorted(anchors)})")
        left_edge = min(anchors.values()) - 60
        # The table begins below its header: drop "PAGE 001", report codes and dates above it
        header_y = min(y for x0, y, x1, y1, t in words if re.sub(r"[^A-Z]", "", t.upper()) in ("REGULAR", "SPECIAL"))
        codes = sorted((w for w in words if re.fullmatch(r"\d{1,3}", w[4]) and w[2] < left_edge
                        and w[0] < min(anchors.values()) * 0.5 and w[1] > header_y + 5), key=lambda w: w[1])
        # OCR sometimes splits a code ("1 45"): join digit tokens that sit side by side
        merged = []
        for w in codes:
            if merged and abs(w[1] - merged[-1][1]) < 2 and 0 < w[0] - merged[-1][2] < 6:
                a = merged.pop()
                w = (a[0], min(a[1], w[1]), w[2], max(a[3], w[3]), a[4] + w[4])
            merged.append(w)
        codes = [w for w in merged if len(w[4]) <= 3]
        if not codes:
            continue
        ycs = [(w[1] + w[3]) / 2 for w in codes]
        spacing = median([b - a for a, b in zip(ycs, ycs[1:]) if b - a > 2]) or 8.0
        nearest = lambda y: min(range(len(ycs)), key=lambda i: abs(ycs[i] - y))
        # Number tokens, each placed in its column by x
        tokens = []
        for x0, y0, x1, y1, t in words:
            if x1 < left_edge + 10 or y0 < header_y + 5:
                continue
            value, odd = parse_number(t)
            if value is None and not re.search(r"\d", t):
                continue
            col = min(anchors, key=lambda c: abs(anchors[c] - (x0 + x1) / 2))
            tokens.append((col, (y0 + y1) / 2, value, odd, t))
        # Scanned pages are slightly rotated, so a column's figures sit above or below their row by
        # an amount that grows with distance from the code column, past half a row at the far
        # right of some pages. Fit each column's offset modulo one row, then unwrap the columns left
        # to right from the code column (offset 0), choosing the whole-row shift that continues
        # the trend; neighbouring columns differ by far less than half a row.
        def fine_offset(ys):
            best, best_score = 0.0, -1.0
            for k in range(-50, 51):
                d = k * spacing / 100
                score = sum(math.exp(-(((y - d) - ycs[nearest(y - d)]) / (0.15 * spacing)) ** 2) for y in ys)
                if score > best_score:
                    best, best_score = d, score
            return best
        code_x = median([(w[0] + w[2]) / 2 for w in codes])
        # The header words share one printed line, so their drift across the page gives the
        # rotation directly; with fewer than three of them, unwrap column by column instead
        hxy = list(found_xy[p].values())
        slope = None
        if len(hxy) >= 3:
            mx, my = sum(x for x, _ in hxy) / len(hxy), sum(y for _, y in hxy) / len(hxy)
            sxx = sum((x - mx) ** 2 for x, _ in hxy)
            slope = sum((x - mx) * (y - my) for x, y in hxy) / sxx if sxx else None
        fine = {}
        for c in anchors:
            ys = [y for cc, y, v, *_ in tokens if cc == c and v is not None]
            if ys:
                fine[c] = fine_offset(ys)

        def unwrap(use_slope):
            points, offset = [(code_x, 0.0)], {}
            for c in sorted(anchors, key=anchors.get):
                if c not in fine:
                    offset[c] = points[-1][1]
                    continue
                if use_slope:
                    pred = slope * (anchors[c] - code_x)
                elif len(points) >= 2:
                    (xa, da), (xb, db) = points[-2], points[-1]
                    pred = db + (db - da) / (xb - xa) * (anchors[c] - xb) if xb != xa else db
                else:
                    pred = points[-1][1]
                offset[c] = fine[c] + spacing * round((pred - fine[c]) / spacing)
                points.append((anchors[c], offset[c]))
            return offset

        def assign(offset):
            per_row = defaultdict(list)
            for col, y, value, odd, t in tokens:
                i = nearest(y - offset[col])
                if abs(y - offset[col] - ycs[i]) <= 0.45 * spacing:
                    per_row[i].append((col, value, odd, t))
            return per_row

        def consistent(per_row):
            # rows whose total lies within the range of their program figures
            n = 0
            for items in per_row.values():
                v = {c: val for c, val, *_ in items if val is not None}
                progs = [v[c] for c in PROGRAMS if c in v]
                if TOTAL_DAY in v and progs and min(progs) - 1 <= v[TOTAL_DAY] <= max(progs) + 1:
                    n += 1
            return n

        # Two estimates of the rotation: the header line's slope, and column-by-column unwrapping.
        # Keep whichever leaves more rows internally consistent.
        options = [assign(unwrap(False))]
        if slope is not None:
            options.append(assign(unwrap(True)))
        per_row = max(options, key=consistent)
        for i, (cx0, cy0, cx1, cy1, ct) in enumerate(codes):
            line = [w for w in words if abs((w[1] + w[3]) / 2 - ycs[i]) < 0.45 * spacing and w[0] > cx1 - 1
                    and w[2] < left_edge + 10]
            name = " ".join(t for x0, y0, x1, y1, t in sorted(line) if not re.fullmatch(r"[\d,.]+", t))
            vals, notes = {}, []
            for col, value, odd, t in per_row[i]:
                if value is None:
                    # a lone digit or mark is scanner noise; anything longer may be a lost value
                    notes.append(f"{'stray mark' if len(re.sub(r'[^0-9]', '', t)) <= 1 else 'unread token'} {t!r}")
                    continue
                if col in vals:
                    notes.append(f"two values for {col}")
                vals[col] = value
                if odd:
                    notes.append(f"{col} read from {t!r}")
            rows.append({"fy": fy, "lea": int(ct), "name": clean_name(name), "vals": vals, "notes": notes,
                         "file": pdf, "page": p})
    return rows


def read_state_pdf_table(pdf, page, label_map):
    """A statewide multi-year comparison table: {fy: {program: value}}."""
    text = subprocess.run(["pdftotext", "-f", str(page), "-l", str(page), "-layout", str(ARCHIVE / pdf), "-"],
                          capture_output=True, text=True, check=True).stdout
    lines = text.splitlines()
    years = None
    out = defaultdict(dict)
    for ln in lines:
        fys = re.findall(r"FY(\d\d)", ln)
        if len(fys) >= 5:
            years = [1900 + int(y) if int(y) > 50 else 2000 + int(y) for y in fys]
            continue
        if not years:
            continue
        for label, prog in label_map.items():
            if ln.strip().upper().startswith(label):
                nums = [int(n.replace(",", "")) for n in re.findall(r"\d{1,2},\d{3}", ln)]
                for fy, n in zip(years, nums):
                    out[fy][prog] = n
    return out


# ---------------------------------------------------------------------------------------------
# Build

def main():
    names = modern_names()
    QA.mkdir(parents=True, exist_ok=True)

    # Reference names for each code, from every workbook: the check on the OCR'd codes
    ref_names = defaultdict(set)

    # --- Day program: workbooks -------------------------------------------------------------
    day = {}  # (fy, lea) -> record
    for fname in FIVE_YEAR:  # oldest first, so later workbooks overwrite earlier ones
        for fy, lea, nm, progs, spend, memb, total, r in read_five_year(fname):
            ref_names[lea].add(clean_name(nm).upper())
            day[(fy, lea)] = {"name": nm, "progs": progs, "spend": spend, "memb": memb, "total": total,
                              "file": fname, "loc": f"5yrpp!row {r}"}
    for lea, nm, spend, memb, total, r in read_fy05_day():
        ref_names[lea].add(clean_name(nm).upper())
        day[(2005, lea)] = {"name": nm, "progs": {}, "spend": spend, "memb": memb, "total": total,
                            "file": FY05_FILE, "loc": f"ppx vs funct!row {r}"}
    fy06, fy06_state = read_fy06_day()
    for lea, nm, progs, spend, memb, total, r in fy06:
        ref_names[lea].add(clean_name(nm).upper())
        day[(2006, lea)] = {"name": nm, "progs": progs, "spend": spend, "memb": memb, "total": total,
                            "file": FY06_DAY_FILE, "loc": f"list!row {r}"}

    # --- Functional categories ----------------------------------------------------------------
    functional = {fy: read_functional(*src) for fy, src in FUNCTIONAL.items()}
    for fy, d in functional.items():
        for lea, rec in d.items():
            ref_names[lea].add(clean_name(rec["name"]).upper())
    for code, nm in names.items():
        ref_names[int(code) // 10000].add(clean_name(nm).upper())
    # Every town's code, including towns that stopped running schools before FY1997
    with open(DATA / "dese_c70_town_history.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ref_names[int(r["lea"])].add(clean_name(r["town"]).upper())

    def name_score(printed, lea):
        refs = ref_names.get(lea, ())
        p = re.sub(r"[^A-Z]", "", printed.upper())
        return max((difflib.SequenceMatcher(None, p, re.sub(r"[^A-Z]", "", n)).ratio() for n in refs), default=0)

    def digits_fit(ocr, true):
        """The OCR'd code's digits appear, in order, in the true code (digits lost, none invented)."""
        it = iter(str(true))
        return all(ch in it for ch in str(ocr))

    # --- Day program: OCR'd PDFs -------------------------------------------------------------
    flags = []
    pdf_rows = []
    for fy in sorted(PDF_SUMMARY):
        pdf_rows += parse_summary(fy)
    pdf_rows = [r for r in pdf_rows if not r["name"].upper().startswith("STATE")]
    # Rows with no figures carry nothing (stray "0" tokens, codes read twice)
    pdf_rows = [r for r in pdf_rows if r["vals"]]
    # A code the OCR misread is repaired from the printed name, when the name matches one district
    # strongly and the OCR'd digits fit inside that district's code
    for r in pdf_rows:
        if name_score(r["name"], r["lea"]) >= 0.8:
            continue
        cands = sorted(((name_score(r["name"], k), k) for k in ref_names), reverse=True)
        (s1, k1), (s2, _) = cands[0], cands[1]
        if s1 >= 0.85 and s1 - s2 >= 0.1 and digits_fit(r["lea"], k1):
            r["notes"].append(f"code repaired: printed {r['lea']}, name matches {k1}")
            r["lea"] = k1
    # The OCR sometimes splits one printed row in two; rejoin rows with the same code on the same
    # page whose columns do not overlap
    groups = defaultdict(list)
    for r in pdf_rows:
        groups[(r["fy"], r["lea"], r["page"])].append(r)
    merged = []
    for g in groups.values():
        if len(g) == 2 and not set(g[0]["vals"]) & set(g[1]["vals"]):
            a, b = g
            a["vals"].update(b["vals"])
            a["notes"] += b["notes"] + ["row split by OCR, rejoined"]
            merged.append(a)
        else:
            merged += g
    pdf_rows = merged
    seen = Counter((r["fy"], r["lea"]) for r in pdf_rows)
    for r in pdf_rows:
        fy, lea, vals = r["fy"], r["lea"], r["vals"]
        for (cfy, clea, col), (value, note) in CORRECTIONS.items():
            if (cfy, clea) == (fy, lea):
                vals[col] = value
                r["notes"].append(f"corrected {col}: {note}")
        corrected = any(n.startswith("corrected") for n in r["notes"])
        problems = [n for n in r["notes"] if n.startswith(("unread", "two values")) and not corrected]
        problems += [f"{c} {v} printed without a comma" for c, v in vals.items() if v > 60000
                     and any(n.startswith(f"{c} read from") and "," not in n for n in r["notes"])]
        r["info"] = [n for n in r["notes"] if n not in problems]
        if ref_names.get(lea):
            if name_score(r["name"], lea) < 0.6:
                problems.append(f"name {r['name']!r} does not match code {lea} ({'; '.join(sorted(ref_names[lea]))})")
        else:
            r["info"].append(f"code {lea} not in any later list")
        if seen[(fy, lea)] > 1:
            problems.append("code appears twice")
        progs = [vals[c] for c in PROGRAMS if c in vals]
        total = vals.get(TOTAL_DAY)
        if total is None:
            if progs:
                problems.append("no total")
        elif progs and not (min(progs) - 1 <= total <= max(progs) + 1):
            # A total is its programs' weighted average, but DESE's own workbooks show vocational
            # schools' totals running a little outside that range (2% of their rows, FY1997-FY2004)
            vocational = lea >= 800 or lea in (406, 408)
            if vocational and 0.85 * min(progs) <= total <= 1.15 * max(progs):
                r["info"].append(f"vocational total {total} outside programs {min(progs)}-{max(progs)}")
            else:
                problems.append(f"total {total} outside programs {min(progs)}-{max(progs)}")
        for c, v in vals.items():
            if v < 300 or (v > 60000 and not re.search(r"\d,\d{3}$", str(f"{v:,}"))):
                problems.append(f"{c} {v} out of range")
        if fy in PDF_CHECK_ONLY:
            wb = day.get((fy, lea))
            if not wb:
                problems.append("not in the workbooks")
            else:
                for col in PROGRAMS + [TOTAL_DAY]:
                    a = vals.get(col)
                    b = wb["total"] if col == TOTAL_DAY else wb["progs"].get(col)
                    b = round(b) if b else None
                    if (a or None) != (b or None) and not (a and b and abs(a - b) <= 1):
                        problems.append(f"{col}: PDF {a} vs workbook {b}")
        if (fy, lea) in VERIFIED and problems:
            r["info"] += [f"checked against the page image ({VERIFIED[(fy, lea)]}), printed as parsed: {p}"
                          for p in problems]
            problems = []
        r["problems"] = problems
        flags.append({"FY": fy, "code": lea, "printed_name": r["name"], "file": r["file"], "page": r["page"],
                      **{c: vals.get(c, "") for c in PROGRAMS + [TOTAL_DAY]},
                      "problems": " | ".join(problems), "notes": " | ".join(r["info"])})

    write_csv(QA / "day_program_pdf_rows.csv", flags,
              ["FY", "code", "printed_name", "file", "page"] + PROGRAMS + [TOTAL_DAY, "problems", "notes"])

    # PDF years enter the day-program table unless a check failed; flagged rows are held back
    held = []
    for r in pdf_rows:
        if r["fy"] in PDF_CHECK_ONLY:
            continue
        if r["problems"]:
            held.append(r)
            continue
        if (r["fy"], r["lea"]) in day:
            continue
        day[(r["fy"], r["lea"])] = {"name": r["name"], "progs": {c: r["vals"].get(c) for c in PROGRAMS},
                                    "spend": None, "memb": None, "total": r["vals"].get(TOTAL_DAY),
                                    "file": r["file"], "loc": f"page {r['page']}"}

    # --- One name per code ----------------------------------------------------------------------
    # DESE's current name where the Hub has the code; else the town's name; else the name DESE's
    # workbooks print (title-cased); else, for districts gone before FY1997, the name most often
    # read from the PDFs. A misread name ("FLYMOUTH CARVER") never reaches the output.
    towns = {}
    with open(DATA / "dese_c70_town_history.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            towns[int(r["lea"])] = r["town"]
    wb_names = defaultdict(Counter)
    for (fy, lea), rec in day.items():
        if not rec["file"].endswith(".pdf"):
            wb_names[lea][clean_name(rec["name"])] += 1
    for fy, d in functional.items():
        for lea, rec in d.items():
            wb_names[lea][clean_name(rec["name"])] += 1
    ocr_names = defaultdict(Counter)
    for r in pdf_rows:
        ocr_names[r["lea"]][r["name"]] += 1

    def canonical(lea):
        code = dese_code(lea)
        if code in names:
            return names[code]
        if lea in towns:
            return towns[lea]
        if wb_names[lea]:
            return title(wb_names[lea].most_common(1)[0][0])
        return title(ocr_names[lea].most_common(1)[0][0]) if ocr_names[lea] else ""

    # --- Write the day-program table ---------------------------------------------------------
    out = []
    for (fy, lea), rec in sorted(day.items()):
        code = dese_code(lea)
        name = canonical(lea)
        common = dict(src_file=rec["file"], location=rec["loc"], printed=clean_name(rec["name"]))
        for col in PROGRAMS:
            v = rec["progs"].get(col)
            if v:
                out.append(row(fy, code, name, DAY_CAT, col, round(v, 2), "Amount", **common))
        if rec["total"]:
            out.append(row(fy, code, name, DAY_CAT, TOTAL_DAY, round(rec["total"], 2), "Amount", **common))
        if rec["spend"]:
            out.append(row(fy, code, name, "Day Program Expenditures", TOTAL_DAY, round(rec["spend"], 2), "Amount", **common))
        if rec["memb"]:
            out.append(row(fy, code, name, "Day Program Enrollment", "Average Membership", round(rec["memb"], 1), "Count", **common))
    write_csv(OUT_DAY, out)

    # A misread digit shows as a one-year spike that both neighbouring years contradict. Check the
    # stable series (regular day and total) and list every such spike for a look at the page.
    series = defaultdict(dict)
    for r in out:
        if r["IND_CAT"] == DAY_CAT and r["IND_SUBCAT"] in ("Regular Day", TOTAL_DAY):
            series[(r["DIST_CODE"], r["IND_SUBCAT"])][r["SY"]] = (r["IND_VALUE"], r["SOURCE_FILE"], r["SOURCE_LOCATION"], r["DIST_NAME"])
    spikes = []
    for (code, col), ys in series.items():
        for fy, (v, src, loc, nm) in ys.items():
            a, b = ys.get(fy - 1), ys.get(fy + 1)
            if not a or not b:
                continue
            lo, hi = sorted((a[0], b[0]))
            if hi / lo < 1.2 and (v > 1.3 * hi or v < lo / 1.3):
                spikes.append({"FY": fy, "code": code, "district": nm, "column": col, "value": v,
                               "previous": a[0], "next": b[0], "file": src, "location": loc})
    write_csv(QA / "day_program_spikes.csv", spikes,
              ["FY", "code", "district", "column", "value", "previous", "next", "file", "location"])

    # --- Write the functional table -----------------------------------------------------------
    out = []
    fcheck = []
    for fy, d in sorted(functional.items()):
        fname, sheet = FUNCTIONAL[fy]
        for lea, rec in sorted(d.items()):
            ind = rec["fte"].get("In-District FTE Average Membership")
            ood = rec["fte"].get("Out-of-District FTE Average Membership")
            tot = rec["fte"].get("Total FTE Average Membership")
            total = rec["cats"].get("TOTAL EXPENDITURES", (None, None))
            if not tot or not total[0]:
                continue
            code = dese_code(lea)
            name = canonical(lea)
            loc = lambda label: f"{sheet}!row {rec['rows'].get(label, '')}"
            common = dict(src_file=fname, printed=clean_name(rec["name"]))
            for label, hub in FUNCTIONAL_LABELS.items():
                dollars, pp = rec["cats"].get(label, (None, None))
                if pp is not None and ind:
                    out.append(row(fy, code, name, "Expenditures Per Pupil", hub, round(pp, 2), "Amount",
                                   location=loc(label), **common))
            out.append(row(fy, code, name, "Expenditures Per Pupil", "Total Expenditures", round(total[1], 2),
                           "Amount", location=loc("TOTAL EXPENDITURES"), **common))
            ood_dollars = rec["cats"].get("Payments To Out-Of-District Schools", (0, None))[0] or 0
            if ind:
                out.append(row(fy, code, name, "Expenditures Per Pupil", "Total In-District Expenditures",
                               round((total[0] - ood_dollars) / ind, 2), "Amount",
                               location=f"computed: ({loc('TOTAL EXPENDITURES')} - payments out of district) / in-district FTE",
                               **common))
            if ood:
                pp = rec["cats"].get("Payments To Out-Of-District Schools", (None, None))[1]
                if pp is not None:
                    out.append(row(fy, code, name, "Expenditures Per Pupil", "Payments To Out-Of-District Schools",
                                   round(pp, 2), "Amount", location=loc("Payments To Out-Of-District Schools"), **common))
            for label, hub in [("In-District FTE Average Membership", "In-District FTE Pupils"),
                               ("Out-of-District FTE Average Membership", "Out-of-District FTE Pupils"),
                               ("Total FTE Average Membership", "Total FTE Pupils")]:
                v = rec["fte"].get(label)
                if v is not None:
                    out.append(row(fy, code, name, "Student Enrollment", hub, round(v, 1), "Count",
                                   location=loc(label), **common))
            # Each category's per-pupil figure must equal its dollars over in-district FTE
            for label in FUNCTIONAL_LABELS:
                dollars, pp = rec["cats"].get(label, (None, None))
                if dollars is not None and pp is not None and ind and abs(dollars / ind - pp) > 0.02:
                    fcheck.append({"FY": fy, "code": lea, "check": f"{label}: {dollars}/{ind} != {pp}"})
            if abs(total[0] / tot - total[1]) > 0.02:
                fcheck.append({"FY": fy, "code": lea, "check": f"total: {total[0]}/{tot} != {total[1]}"})
    for fy, (fname, sheet) in FUNCTIONAL_CHECK.items():
        other = read_functional(fname, sheet)
        for lea, rec in functional[fy].items():
            a = rec["cats"].get("TOTAL EXPENDITURES", (None,))[0]
            b = other.get(lea, {}).get("cats", {}).get("TOTAL EXPENDITURES", (None,))[0]
            if a != b:
                fcheck.append({"FY": fy, "code": lea, "check": f"total {a} vs {fname} {sheet} {b}"})
    write_csv(OUT_FUNCTIONAL, out)
    write_csv(QA / "functional_checks.csv", fcheck, ["FY", "code", "check"])

    # --- Statewide day-program averages ---------------------------------------------------------
    labels = {"TOTAL DAY": TOTAL_DAY, "DAY SUMMARY": TOTAL_DAY, "REGULAR DAY": "Regular Day",
              "SPECIAL NEEDS": "Special Needs", "BILINGUAL": "Bilingual", "OCCUPATIONAL": "Occupational Day"}
    state_sources = [("PER_PUPIL_EXPENDITURE_FY96.pdf", 19), ("PER_PUPIL_EXPENDITURE_FY98.pdf", 31),
                     ("PER_PUPIL_EXPENDITURE_FY99.pdf", 17)]
    state, state_check = {}, []
    for pdf, page in state_sources:
        for fy, progs in read_state_pdf_table(pdf, page, labels).items():
            for prog, v in progs.items():
                prior = state.get((fy, prog))
                if prior and prior[0] != v:
                    state_check.append({"FY": fy, "program": prog, "check": f"{prior[1]} {prior[0]} vs {pdf} {v}"})
                state[(fy, prog)] = (v, f"{pdf}", f"page {page}")
    for fname in FIVE_YEAR:
        sh = xlrd.open_workbook(ARCHIVE / fname, on_demand=True).sheet_by_name("pptext")
        for r in range(sh.nrows):
            v = sh.row_values(r)
            for i, cell in enumerate(v):
                if str(cell).strip().upper() in ("REGULAR DAY", "TOTAL") and i + 1 < len(v):
                    head = sh.row_values(r - 1 if str(cell).strip().upper() == "REGULAR DAY" else r - 2)
                    for j in range(i + 1, len(v)):
                        m = re.fullmatch(r"FY(\d\d)", str(head[j]).strip())
                        if m and num(v[j]):
                            fy = 2000 + int(m.group(1)) if int(m.group(1)) < 50 else 1900 + int(m.group(1))
                            prog = "Regular Day" if str(cell).strip().upper() == "REGULAR DAY" else TOTAL_DAY
                            state[(fy, prog)] = (round(num(v[j])), fname, f"pptext!row {r + 1}")
    if fy06_state:
        for prog in ["Regular Day", "Special Needs", "Occupational Day", TOTAL_DAY]:
            state[(2006, prog)] = (round(fy06_state[prog]), FY06_DAY_FILE, f"state!row {fy06_state['_row']}")
    out = [row(fy, "00000000", "Massachusetts", DAY_CAT, prog, v, "Amount", src, loc, "STATE AVERAGE")
           for (fy, prog), (v, src, loc) in sorted(state.items())]
    write_csv(OUT_STATE, out)
    write_csv(QA / "state_checks.csv", state_check, ["FY", "program", "check"])

    # --- Summary ------------------------------------------------------------------------------
    n_pdf = sum(1 for r in pdf_rows if r["fy"] not in PDF_CHECK_ONLY)
    print(f"PDF rows: {len(pdf_rows)}; in FY1990-FY1996: {n_pdf}; held back: {len(held)}")
    for fy in sorted(PDF_CHECK_ONLY):
        rs = [r for r in pdf_rows if r["fy"] == fy]
        bad = [r for r in rs if any(" vs workbook " in p for p in r["problems"])]
        cleaned = [r for r in rs if any(" read from " in n for n in r["info"])]
        repaired = [r for r in rs if any(n.startswith("code repaired") for n in r["info"])]
        print(f"  FY{fy}: {len(rs)} PDF rows; {len(bad)} disagree with the workbooks; "
              f"{len(cleaned)} had OCR noise removed, {sum(r not in bad for r in cleaned)} of them agree; "
              f"{len(repaired)} codes repaired, {sum(r not in bad for r in repaired)} of them agree")
    reasons = Counter(re.sub(r"'[^']*'|\(.*?\)|\d+", "#", p) for r in held for p in r["problems"])
    print("held back, by reason:", dict(reasons.most_common(8)))
    by_year = Counter(fy for fy, _ in day)
    print("day-program districts by year:", dict(sorted(by_year.items())))


if __name__ == "__main__":
    sys.exit(main())
