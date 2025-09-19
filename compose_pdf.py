# compose_pdf.py — A4 portrait PDF compositor using ReportLab
# Requires: reportlab  (pip install reportlab)

from __future__ import annotations
import os, re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd


from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    KeepTogether, Flowable, PageBreak 
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


# ---- Paths (adjust if needed) ----
DATA_DIR   = Path("./data")
OUTPUT_DIR = Path("./output")
EXCEL_FILE = DATA_DIR / "E2C_Hub_MA_DESE_Data.xlsx"
OUTPUT_PDF = OUTPUT_DIR / "expenditures_series.pdf"

# Where the chart PNGs live (created by district_expend_pp_stack.py)
def district_png(dist: str) -> Path:
    return OUTPUT_DIR / f"expenditures_per_pupil_vs_enrollment_{dist.replace(' ','_')}.png"

def regional_png(key: str) -> Path:
    return OUTPUT_DIR / f"regional_expenditures_per_pupil_{key}.png"

# ---- Domain constants ----
SHEET_EXPEND  = "District Expend by Category"
SHEET_REGIONS = "District Regions"

DISTRICTS = ["Amherst-Pelham", "Amherst", "Leverett", "Pelham", "Shutesbury"]

EXCLUDE_SUBCATS = {"total expenditures", "total in-district expenditures"}
ENROLL_KEYS = [
    ("in-district fte pupils", "In-District FTE Pupils"),
    ("out-of-district fte pupils", "Out-of-District FTE Pupils"),
]
TOTAL_FTE_KEY = "total fte pupils"

# Palettes (to color the Category table to match charts)
COLOR_OVERRIDES_DIST   = {"professional development": "#6A3D9A"}
PALETTE_DIST = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#F0E442", "#56B4E9",
                "#E69F00", "#999999", "#332288", "#88CCEE", "#44AA99", "#882255"]

COLOR_OVERRIDES_APEL   = {"professional development": "#2A9D8F"}
PALETTE_APEL = ["#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51", "#8AB17D",
                "#7A5195", "#4ECDC4", "#C7F464", "#FF6B6B", "#FFE66D", "#355C7D"]

COLOR_OVERRIDES_REGION = {"professional development": "#1B9E77"}
PALETTE_REGION = ["#8DD3C7", "#FFFFB3", "#BEBADA", "#FB8072", "#80B1D3", "#FDB462",
                  "#B3DE69", "#FCCDE5", "#D9D9D9", "#BC80BD", "#CCEBC5", "#FFED6F"]

LINE_COLORS_DIST  = {"In-District FTE Pupils": "#000000", "Out-of-District FTE Pupils": "#AA0000"}
LINE_COLORS_APEL  = {"In-District FTE Pupils": "#1F4E79", "Out-of-District FTE Pupils": "#D35400"}
LINE_COLORS_REGION= {"In-District FTE Pupils": "#1B7837", "Out-of-District FTE Pupils": "#5E3C99"}

# ---- Text styles ----
styles = getSampleStyleSheet()
style_h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=13, leading=16, spaceAfter=6)
style_body = ParagraphStyle("body", parent=styles["Normal"], fontSize=9, leading=12)
style_num  = ParagraphStyle("num",  parent=styles["Normal"], fontSize=9, leading=12, alignment=2)  # right
style_src  = ParagraphStyle("src",  parent=styles["Normal"], fontSize=8, leading=10)
# neutral, eye-catching color for negative values
NEG_COLOR = HexColor("#3F51B5")  # deep indigo (neutral, colorblind-friendly)
style_num_neg = ParagraphStyle("num_neg", parent=style_num, textColor=NEG_COLOR)



# ---- Flowables for legend swatches ----
class LineSwatch(Flowable):
    def __init__(self, hex_color, w=42, h=8, lw=2.8):   # match chart linewidth
        Flowable.__init__(self)
        self.c = HexColor(hex_color); self.w=w; self.h=h; self.lw=lw
        self.width=w; self.height=h
    def draw(self):
        c = self.canv
        y = self.h / 2.0
        # draw the line
        c.setStrokeColor(self.c); c.setLineWidth(self.lw)
        c.line(0, y, self.w, y)
        # draw a white-filled circle on top so the center is pure white
        c.setFillColor(colors.white)
        c.setStrokeColor(self.c)
        r = self.h * 0.35
        c.circle(self.w/2.0, y, r, stroke=1, fill=1)

class ColorSwatch(Flowable):
    def __init__(self, hex_color, w=14, h=10):
        Flowable.__init__(self)
        self.c = HexColor(hex_color); self.w=w; self.h=h
        self.width=w; self.height=h
    def draw(self):
        c = self.canv
        c.setFillColor(self.c)
        c.setStrokeColor(self.c)
        c.rect(0, 0, self.w, self.h, fill=1, stroke=0)

# ---- Data helpers (mirroring your plotting script) ----
def find_sheet_name(xls: pd.ExcelFile, target: str) -> str:
    for n in xls.sheet_names:
        if n.strip().lower() == target.strip().lower(): return n
    t = target.strip().lower()
    for n in xls.sheet_names:
        nl = n.strip().lower()
        if t in nl or nl in t: return n
    return xls.sheet_names[0]

def coalesce_year_column(df: pd.DataFrame) -> pd.DataFrame:
    name_map = {str(c).strip().lower(): c for c in df.columns}
    chosen = None
    for key in ("fiscal_year","year","sy","school_year"):
        if key in name_map: chosen = name_map[key]; break
    if chosen is None:
        first = df.columns[0]
        if str(first).strip().lower() in ("sy","school_year"): chosen = first
    if chosen is None:
        for c in df.columns:
            sample = str(df[c].dropna().astype(str).head(20).tolist())
            if re.search(r"\b(19|20)\d{2}\b", sample): chosen = c; break
    if chosen is None: raise ValueError("Could not find a year column.")
    def parse_to_year(val):
        if pd.isna(val): return np.nan
        if isinstance(val,(int,float)) and not pd.isna(val):
            v=int(val); 
            if 1900<=v<=2100: return v
        s=str(val).strip()
        m=re.search(r"\b((?:19|20)\d{2})\s*[–\-/]\s*(\d{2,4})\b", s)
        if m:
            y1=int(m.group(1)); y2raw=m.group(2)
            y2=(y1//100)*100+int(y2raw) if len(y2raw)==2 else int(y2raw)
            if len(y2raw)==2 and (y2%100)<(y1%100): y2+=100
            return y2
        m2=re.search(r"\b((?:19|20)\d{2})\b", s)
        if m2: return int(m2.group(1))
        return np.nan
    df["YEAR"]=df[chosen].apply(parse_to_year)
    df=df.dropna(subset=["YEAR"]).copy()
    df["YEAR"]=df["YEAR"].astype(int)
    return df

def compute_cagr_last(series: pd.Series, years_back: int) -> float | np.nan:
    if series is None or series.empty: return np.nan
    s = series.sort_index().astype(float).replace([np.inf,-np.inf], np.nan).dropna()
    if s.size < 2: return np.nan
    end_year = int(s.index.max())
    candidates = s[s.index<=end_year-years_back]
    start_year = int(candidates.index.max()) if not candidates.empty else int(s.index.min())
    n_years = end_year - start_year
    if n_years <= 0: return np.nan
    start_val=float(s.loc[start_year]); end_val=float(s.loc[end_year])
    if start_val<=0 or end_val<=0: return np.nan
    return (end_val/start_val)**(1.0/n_years)-1.0

def order_subcats_by_mean(piv: pd.DataFrame)->List[str]:
    return list(piv.mean(axis=0).sort_values(ascending=False).index) if piv.shape[1]>1 else list(piv.columns)

def load_data():
    xls = pd.ExcelFile(EXCEL_FILE)
    sx = find_sheet_name(xls, SHEET_EXPEND)
    df = pd.read_excel(xls, sheet_name=sx)
    df.columns=[str(c).strip() for c in df.columns]
    ren={}
    for c in df.columns:
        low=str(c).strip().lower()
        if low=="dist_name": ren[c]="DIST_NAME"
        elif low=="ind_cat": ren[c]="IND_CAT"
        elif low=="ind_subcat": ren[c]="IND_SUBCAT"
        elif low=="ind_value": ren[c]="IND_VALUE"
        elif low in ("sy","school_year","fiscal_year","year"): ren[c]=c
    df=df.rename(columns=ren)
    if "IND_VALUE" not in df.columns: raise ValueError("IND_VALUE missing")
    df=coalesce_year_column(df)
    for c in ["DIST_NAME","IND_CAT","IND_SUBCAT"]:
        if c in df.columns: df[c]=df[c].astype(str).str.strip()
    df["IND_VALUE"]=pd.to_numeric(df["IND_VALUE"], errors="coerce")
    # Regions
    sr = find_sheet_name(xls, SHEET_REGIONS)
    reg = pd.read_excel(xls, sheet_name=sr)
    reg.columns=[str(c).strip() for c in reg.columns]
    cmap={}
    for c in reg.columns:
        low=c.strip().lower()
        if low in ("district_name","dist_name","name"): cmap[c]="DIST_NAME"
        elif low in ("eohhs_region","region"): cmap[c]="EOHHS_REGION"
        elif low in ("school_type","type","district_type","org_type"): cmap[c]="SCHOOL_TYPE"
    reg=reg.rename(columns=cmap)
    for c in ["DIST_NAME","EOHHS_REGION","SCHOOL_TYPE"]:
        reg[c]=reg[c].astype(str).str.strip()
    return df, reg

def prepare_district(df: pd.DataFrame, dist: str):
    ddf = df[df["DIST_NAME"].str.lower()==dist.lower()].copy()
    epp = ddf[ddf["IND_CAT"].str.lower()=="expenditures per pupil"].copy()
    epp["IND_SUBCAT"]=epp["IND_SUBCAT"].replace({np.nan:"Unspecified"}).fillna("Unspecified")
    epp=epp[~epp["IND_SUBCAT"].str.lower().isin(EXCLUDE_SUBCATS)].copy()
    epp_pivot = epp.pivot_table(index="YEAR", columns="IND_SUBCAT", values="IND_VALUE", aggfunc="sum").sort_index().fillna(0.0)
    enroll_lines = {}
    enroll_ddf = ddf[ddf["IND_CAT"].str.lower()=="student enrollment"].copy()
    for key,label in ENROLL_KEYS:
        ser = enroll_ddf[enroll_ddf["IND_SUBCAT"].str.lower()==key][["YEAR","IND_VALUE"]]
        if not ser.empty:
            s = ser.dropna().drop_duplicates(subset=["YEAR"]).set_index("YEAR")["IND_VALUE"].sort_index()
            enroll_lines[label] = s
    return epp_pivot, enroll_lines

def prepare_western(df: pd.DataFrame, reg: pd.DataFrame, size_bucket: str, school_type: str):
    mask = (reg["EOHHS_REGION"].str.lower()=="western") & (reg["SCHOOL_TYPE"].str.lower()==school_type.lower())
    names = set(reg[mask]["DIST_NAME"].str.lower())
    present = set(df["DIST_NAME"].str.lower())
    members = sorted(n for n in names if n in present)

    enroll = df[df["IND_CAT"].str.lower()=="student enrollment"].copy()
    totals={}
    for nm in members:
        dsub=enroll[enroll["DIST_NAME"].str.lower()==nm]
        tot=dsub[dsub["IND_SUBCAT"].str.lower()==TOTAL_FTE_KEY]
        if not tot.empty:
            y=int(tot["YEAR"].max()); totals[nm]=float(tot.loc[tot["YEAR"]==y,"IND_VALUE"].iloc[0])
    if size_bucket=="le_500":
        members=[n for n,v in totals.items() if v<=500]; title_suffix="≤500 FTE"
    elif size_bucket=="gt_500":
        members=[n for n,v in totals.items() if v>500]; title_suffix=">500 FTE"
    else:
        title_suffix=None
    title = f"All Western MA {school_type.capitalize()} Districts"
    if title_suffix: title += f" {title_suffix}"
    if not members: return title, pd.DataFrame(), {}, {}
    epp = df[(df["IND_CAT"].str.lower()=="expenditures per pupil") & (df["DIST_NAME"].str.lower().isin(members))][["DIST_NAME","YEAR","IND_SUBCAT","IND_VALUE"]].copy()
    epp=epp[~epp["IND_SUBCAT"].str.lower().isin(EXCLUDE_SUBCATS)].copy()
    weights = df[(df["IND_CAT"].str.lower()=="student enrollment") &
                 (df["IND_SUBCAT"].str.lower()=="in-district fte pupils") &
                 (df["DIST_NAME"].str.lower().isin(members))][["DIST_NAME","YEAR","IND_VALUE"]].rename(columns={"IND_VALUE":"WEIGHT"})
    # weighted mean EPP
    m = epp.merge(weights, on=["DIST_NAME","YEAR"], how="left")
    m["WEIGHT"]=pd.to_numeric(m["WEIGHT"], errors="coerce").fillna(0.0)
    m["P"]=m["IND_VALUE"]*m["WEIGHT"]
    grp=m.groupby(["YEAR","IND_SUBCAT"], as_index=False)
    out=grp.agg(NUM=("P","sum"), DEN=("WEIGHT","sum"), MEAN=("IND_VALUE","mean"))
    out["VALUE"]=np.where(out["DEN"]>0, out["NUM"]/out["DEN"], out["MEAN"])
    epp_pivot=out.pivot(index="YEAR", columns="IND_SUBCAT", values="VALUE").sort_index().fillna(0.0)

    # Enrollment: sum for chart; mean for table
    lines_sum, lines_mean = {}, {}
    for key,label in ENROLL_KEYS:
        sub = df[(df["IND_CAT"].str.lower()=="student enrollment") &
                 (df["IND_SUBCAT"].str.lower()==key) &
                 (df["DIST_NAME"].str.lower().isin(members))][["DIST_NAME","YEAR","IND_VALUE"]]
        if not sub.empty:
            lines_sum[label]  = sub.groupby("YEAR")["IND_VALUE"].sum().sort_index()
            lines_mean[label] = sub.groupby("YEAR")["IND_VALUE"].mean().sort_index()
    return title, epp_pivot, lines_sum, lines_mean

def build_page_dicts(df: pd.DataFrame, reg: pd.DataFrame) -> List[dict]:
    pages = []

    # Districts
    for dist in DISTRICTS:
        epp, lines = prepare_district(df, dist)
        if epp.empty and not lines: continue
        subcats = order_subcats_by_mean(epp)
        # Palette and line colors
        if dist == "Amherst-Pelham":
            pal, overrides, line_cols = PALETTE_APEL, COLOR_OVERRIDES_APEL, LINE_COLORS_APEL
            title = "Amherst-Pelham Regional: Expenditures Per Pupil vs Enrollment"
        else:
            pal, overrides, line_cols = PALETTE_DIST, COLOR_OVERRIDES_DIST, LINE_COLORS_DIST
            title = f"{dist}: Expenditures Per Pupil vs Enrollment"

        # color map per subcat in that order
        base = (pal * ((len(subcats)//len(pal))+1))[:len(subcats)]
        color_map = {sc: base[i] for i, sc in enumerate(subcats)}
        for sc in subcats:
            key = sc.strip().lower()
            if key in overrides: color_map[sc] = overrides[key]

        years = sorted(set(epp.index) | set().union(*(set(s.index) for s in lines.values() if s is not None)))
        epp = epp.reindex(years).fillna(0.0)
        latest_year = int(epp.index.max()) if not epp.empty else 0
        # Category table rows (top-to-bottom order equals stack top-to-bottom? We’ll display top row as top of stack)
        latest_vals = [epp.loc[latest_year, sc] if sc in epp.columns else 0.0 for sc in subcats]
        c5  = [compute_cagr_last(epp[sc], 5)  for sc in subcats]
        c10 = [compute_cagr_last(epp[sc], 10) for sc in subcats]
        c15 = [compute_cagr_last(epp[sc], 15) for sc in subcats]
        def fmt_pct(v): return "—" if (v is None or (isinstance(v,float) and np.isnan(v))) else f"{v*100:+.1f}%"
        rev = list(reversed(subcats))
        cat_rows = [(sc, f"${epp.loc[latest_year, sc]:,.0f}", fmt_pct(c5[subcats.index(sc)]),
                     fmt_pct(c10[subcats.index(sc)]), fmt_pct(c15[subcats.index(sc)]), color_map[sc]) for sc in rev]
        total_latest = float(np.nansum(latest_vals)) if latest_vals else 0.0
        def mean_clean(arr):
            vs=[v for v in arr if v is not None and not (isinstance(v,float) and np.isnan(v))]
            return float(np.mean(vs)) if vs else np.nan
        cat_total = ("Total", f"${total_latest:,.0f}", fmt_pct(mean_clean(c5)), fmt_pct(mean_clean(c10)), fmt_pct(mean_clean(c15)))

        # FTE table rows (use district lines)
        fte_years = [int(s.index.max()) for s in lines.values() if s is not None and not s.empty]
        latest_fte_year = max(fte_years) if fte_years else latest_year
        fte_rows=[]
        for _k,label in ENROLL_KEYS:
            s = lines.get(label)
            if s is None or s.empty: continue
            r5=compute_cagr_last(s,5); r10=compute_cagr_last(s,10); r15=compute_cagr_last(s,15)
            fte_rows.append((line_cols[label], label,
                             ("—" if latest_fte_year not in s.index else f"{float(s.loc[latest_fte_year]):,.0f}"),
                             fmt_pct(r5), fmt_pct(r10), fmt_pct(r15)))

        pages.append(dict(
            title=title,
            chart_path=str(district_png(dist)),
            latest_year=latest_year,
            latest_year_fte=latest_fte_year,
            cat_rows=cat_rows,
            cat_total=cat_total,
            fte_rows=fte_rows
        ))

    # Western region pages
    specs = [("le_500","Traditional","Western_Traditional_le_500"),
             ("gt_500","Traditional","Western_Traditional_gt_500"),
             ("all","Charter","Western_Charter_all"),
             ("all","Vocational","Western_Vocational_all")]
    for size_bucket, school_type, key in specs:
        title, epp, lines_sum, lines_mean = prepare_western(df, reg, size_bucket, school_type)
        if epp.empty and not lines_sum: continue
        subcats = order_subcats_by_mean(epp)
        pal, overrides, line_cols = PALETTE_REGION, COLOR_OVERRIDES_REGION, LINE_COLORS_REGION
        base = (pal * ((len(subcats)//len(pal))+1))[:len(subcats)]
        color_map = {sc: base[i] for i, sc in enumerate(subcats)}
        for sc in subcats:
            k=sc.strip().lower()
            if k in overrides: color_map[sc]=overrides[k]
        years = sorted(set(epp.index) | set().union(*(set(s.index) for s in lines_sum.values() if s is not None)))
        epp=epp.reindex(years).fillna(0.0)
        latest_year = int(epp.index.max()) if not epp.empty else 0
        latest_vals = [epp.loc[latest_year, sc] if sc in epp.columns else 0.0 for sc in subcats]
        c5  = [compute_cagr_last(epp[sc], 5)  for sc in subcats]
        c10 = [compute_cagr_last(epp[sc], 10) for sc in subcats]
        c15 = [compute_cagr_last(epp[sc], 15) for sc in subcats]
        def fmt_pct(v): return "—" if (v is None or (isinstance(v,float) and np.isnan(v))) else f"{v*100:+.1f}%"
        rev=list(reversed(subcats))
        cat_rows=[(sc, f"${epp.loc[latest_year, sc]:,.0f}", fmt_pct(c5[subcats.index(sc)]),
                   fmt_pct(c10[subcats.index(sc)]), fmt_pct(c15[subcats.index(sc)]), color_map[sc]) for sc in rev]
        total_latest = float(np.nansum(latest_vals)) if latest_vals else 0.0
        def mean_clean(arr):
            vs=[v for v in arr if v is not None and not (isinstance(v,float) and np.isnan(v))]
            return float(np.mean(vs)) if vs else np.nan
        cat_total=("Total", f"${total_latest:,.0f}", fmt_pct(mean_clean(c5)), fmt_pct(mean_clean(c10)), fmt_pct(mean_clean(c15)))
        # FTE table uses MEAN (not totals)
        fte_years=[int(s.index.max()) for s in lines_mean.values() if s is not None and not s.empty]
        latest_fte_year=max(fte_years) if fte_years else latest_year
        fte_rows=[]
        for _k,label in ENROLL_KEYS:
            s = lines_mean.get(label)
            if s is None or s.empty: continue
            r5=compute_cagr_last(s,5); r10=compute_cagr_last(s,10); r15=compute_cagr_last(s,15)
            val = "—" if latest_fte_year not in s.index else f"{float(s.loc[latest_fte_year]):,.0f}"
            fte_rows.append((line_cols[label], label, val, fmt_pct(r5), fmt_pct(r10), fmt_pct(r15)))

        pages.append(dict(
            title=title,
            chart_path=str(regional_png(key)),
            latest_year=latest_year,
            latest_year_fte=latest_fte_year,
            cat_rows=cat_rows,
            cat_total=cat_total,
            fte_rows=fte_rows
        ))
    return pages

# build helper - highlight negative CAGR values
def _pnum_auto(text: str) -> Paragraph:
    t = (text or "").strip()
    # color only negatives; em dash / blanks stay default
    if t.startswith("-"):
        return Paragraph(text, style_num_neg)
    return Paragraph(text, style_num)


def build_pdf(pages: List[dict], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(out_path), pagesize=A4,
        leftMargin=0.5*inch, rightMargin=0.5*inch,
        topMargin=0.5*inch, bottomMargin=0.5*inch,
        allowSplitting=1  # allow tables to split across pages
    )

    story = []
    for idx, p in enumerate(pages):
        # Title
        story.append(Paragraph(p["title"], style_h2))
        story.append(Spacer(0, 6))

        # Chart image (scale to page width, but cap height so tables have room)
        img_path = Path(p["chart_path"])
        if not img_path.exists():
            story.append(Paragraph(f"[Missing chart image: {img_path.name}]", style_body))
        else:
            im = Image(str(img_path))
            ratio = im.imageHeight / float(im.imageWidth)
            im.drawWidth  = doc.width
            im.drawHeight = doc.width * ratio
            # cap chart height to ~40% of usable height to leave space for tables
            max_chart_h = doc.height * 0.40
            if im.drawHeight > max_chart_h:
                im.drawHeight = max_chart_h
                im.drawWidth  = im.drawHeight / ratio
            story.append(im)

        story.append(Spacer(0, 24))

        # -------- Category table (repeat header; can split) --------
        cat_header = ["", "Category", f"{p['latest_year']} $/pupil", "5y CAGR", "10y CAGR", "15y CAGR"]

        cat_rows_rl = []
        for (label, latest, c5, c10, c15, hexcol) in p["cat_rows"]:
            cat_rows_rl.append([
                ColorSwatch(hexcol),
                Paragraph(label, style_body),   # black on white
                Paragraph(latest, style_num),   # dollars
                _pnum_auto(c5),                 # CAGRs get auto-colored if negative
                _pnum_auto(c10),
                _pnum_auto(c15),
            ])

        # Total row (unchanged semantics)
        total = ["", Paragraph(p["cat_total"][0], style_body)] + [
            Paragraph(p["cat_total"][1], style_num),
            _pnum_auto(p["cat_total"][2]),
            _pnum_auto(p["cat_total"][3]),
            _pnum_auto(p["cat_total"][4]),
        ]


        cat_tbl = Table(
            [cat_header] + cat_rows_rl + [total],
            # keep total width same as before: 0.50*doc.width for label block,
            # split into 0.08 (swatch) + 0.42 (category text), then numeric columns
            colWidths=[0.08*doc.width, 0.42*doc.width, 0.18*doc.width, 0.10*doc.width, 0.10*doc.width, 0.12*doc.width],
            hAlign="LEFT",
            repeatRows=1,
            splitByRow=1
        )
        cat_tbl.setStyle(TableStyle([
            ("FONT", (0,0), (-1,0), "Helvetica-Bold", 9),
            ("ALIGN", (2,0), (-1,0), "RIGHT"),             # right-align numeric HEADERS
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("ALIGN", (2,1), (-1,-1), "RIGHT"),
            ("LINEBELOW", (0,0), (-1,0), 0.5, colors.black),
            ("ROWSPACING", (0,1), (-1,-1), 2),
            ("BACKGROUND", (0, len(cat_rows_rl)+1), (-1, len(cat_rows_rl)+1), colors.whitesmoke),
            ("FONT", (0, len(cat_rows_rl)+1), (-1, len(cat_rows_rl)+1), "Helvetica-Bold", 9),  # (optional) bold Total row
        ]))

        story.append(cat_tbl)
        story.append(Spacer(0, 24))

        # -------- FTE table (repeat header; can split) --------
        fte_header = ["", "Pupil Group", f"{p['latest_year_fte']} avg FTE", "5y CAGR", "10y CAGR", "15y CAGR"]
        fte_data = [fte_header]
        for (hexcol, label, fte, c5, c10, c15) in p["fte_rows"]:
            fte_data.append([
                LineSwatch(hexcol),
                Paragraph(label, style_body),
                Paragraph(fte,  style_num),   # absolute FTE
                _pnum_auto(c5),
                _pnum_auto(c10),
                _pnum_auto(c15),
            ])

        fte_tbl = Table(
            fte_data,
            colWidths=[0.16*doc.width, 0.34*doc.width, 0.18*doc.width, 0.10*doc.width, 0.10*doc.width, 0.12*doc.width],
            hAlign="LEFT",
            repeatRows=1,
            splitByRow=1
        )
        fte_tbl.setStyle(TableStyle([
            ("FONT", (0,0), (-1,0), "Helvetica-Bold", 9),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("ALIGN", (2,1), (-1,-1), "RIGHT"),
            ("LINEBELOW", (0,0), (-1,0), 0.5, colors.black),
            ("ROWSPACING", (0,1), (-1,-1), 2),
            ("ALIGN", (2,0), (-1,0), "RIGHT"),
        ]))

        story.append(fte_tbl)
        story.append(Spacer(0, 24)) 

        # Source
        src = Paragraph(
            "Source: DESE District Expenditures by Spending Category<br/>"
            "Last updated: August 12, 2025<br/>"
            "https://educationtocareer.data.mass.gov/Finance-and-Budget/District-Expenditures-by-Spending-Category/er3w-dyti/",
            style_src
        )
        story.append(src)

        # Page break between pages (except after the last one)
        if idx < len(pages) - 1:
            story.append(PageBreak())

    doc.build(story)
    print(f"[OK] Wrote PDF: {out_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df, reg = load_data()
    pages = build_page_dicts(df, reg)
    if not pages:
        print("[WARN] No pages to write.")
        return
    build_pdf(pages, OUTPUT_PDF)

if __name__ == "__main__":
    main()
