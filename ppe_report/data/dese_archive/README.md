# DESE per-pupil expenditure archive, FY1990 to FY2008

These are the Massachusetts Department of Elementary and Secondary Education's per-pupil
expenditure reports and workbooks from before FY2009, the first year of the Education-to-Career Hub
dataset in [`../dese_expenditures.csv`](../dese_expenditures.csv). DESE's Public Records Center
provided them to Tim Shores in October 2026 in answer to Massachusetts public records request
P004939-101625. The files are kept here exactly as received, under the names DESE gave them.

The PDFs are scans of DESE's printed reports. DESE made the scans in January 2024 and ran OCR on
them, so their text can be searched and copied, but the OCR text has errors. The workbooks are the
Office of School Finance's own Excel files.

[`../../update_dese_history.py`](../../update_dese_history.py) reads this folder and writes three
CSVs:

| File | What it holds |
| --- | --- |
| [`dese_expenditures_fy2005_2008.csv`](../dese_expenditures_fy2005_2008.csv) | Spending per pupil in DESE's functional categories, and FTE enrollment, for every district, FY2005 to FY2008. Same columns and labels as `dese_expenditures.csv`, so the two stack into one series from FY2005, plus one row the Hub doesn't have: payments to out-of-district schools per out-of-district pupil, which a stack of the two should leave out. |
| [`dese_day_program_fy1990_2006.csv`](../dese_day_program_fy1990_2006.csv) | DESE's older "day program" spending per pupil, by program and in total, for every district, FY1990 to FY2006. Total day-program spending and average membership from FY1997. |
| [`dese_day_program_state_fy1989_2006.csv`](../dese_day_program_state_fy1989_2006.csv) | DESE's published statewide day-program averages, FY1989 to FY2006. |

Every row names its source file and the page, or the sheet and row, it came from
(`SOURCE_FILE`, `SOURCE_LOCATION`), and the name printed there (`SOURCE_NAME`). `DIST_NAME` is
DESE's current name for the code, where the district still exists. `Last_Updated` is when DESE
published the source.

## Two measures

DESE has measured per-pupil spending in two ways, and they should not be joined into one line.

**Day program** (FY1990 to FY2006). In DESE's words: "'Day Program' per pupil expenditures include
all pupils enrolled in the local schools, even if they are non-residents who attend on a
tuitioned-in basis. This measure does not count children being educated outside the district.
Tuition, regional district spending, and other payments for out-of district pupils are not factored
in." It is reported by program: regular day, special needs, bilingual (through FY2003; from FY2004
DESE folded English learners' programs into regular education), and occupational day.

**Functional** (FY2005 on). Spending from all funds in ten functional categories, divided by
in-district FTE pupils; payments for pupils placed out of district, divided by those pupils; and a
total covering both, divided by all FTE pupils. The Education-to-Career Hub reports the same
categories, under the same names, from FY2009, and the explorer uses them today.

FY2005 has both, in the "ppx vs funct" sheet of `calcfunctional05final.xls`. For Abington that year
the day-program figure was $6,766 per pupil ($16.1 million over 2,373 pupils in district), and the
functional total was $8,246 ($20.0 million over 2,423.7 pupils, counting those placed out of
district). A chart can show both series, but the step between them is a change of measure, not of
spending.

## Files

| File | Fiscal year | What it is | Used for |
| --- | --- | --- | --- |
| `PER_PUPIL_EXPENDITURE_1989-1990.pdf` | FY1990 | Report, July 1991 | Day program, pages 13–21 |
| `PER_PUPIL_EXPENDITURE_1990-1991_17351741.pdf` | FY1991 | Report, printed July 1992 | Day program, pages 12–21 |
| `PER_PUPIL_EXPENDITURE_1991-1992.pdf` | FY1992 | Report, July 1993 | Day program, pages 13–21 |
| `PER_PUPIL_EXPENDITURE_1992-1993.pdf` | FY1993 | Report, September 1994 | Day program, pages 13–20 |
| `PER_PUPIL_EXPENDITURE_1993-1994.pdf` | FY1994 | Report, December 1995 | Day program, pages 19–27 |
| `PER_PUPIL_EXPENDITURE_1994-1995.pdf` | FY1995 | Report, April 1997 | Day program, pages 19–27 |
| `PER_PUPIL_EXPENDITURE_FY96.pdf` | FY1996 | Report, October 1998 | Day program, pages 20–28; statewide averages FY1989–FY1996, page 19 |
| `PER_PUPIL_EXPENDITURE_FY97.pdf` | FY1997 | Report, December 1998 | Check on the workbooks; Essex, Manchester and Worcester Trade, pages 20–28 |
| `PER_PUPIL_EXPENDITURE_FY98.pdf` | FY1998 | Report, November 1999 | Check on the workbooks; Essex, Manchester and Worcester Trade, pages 19–30; statewide averages FY1990–FY1998, page 31 |
| `PER_PUPIL_EXPENDITURE_FY99.pdf` | FY1999 | Report, January 2001 | Check on the workbooks; Essex and Manchester, pages 18–27; statewide averages FY1995–FY1999, page 17 |
| `PER_PUPIL_EXPENDITURE_BY_FUNCTION_*.pdf` | FY1990, FY1992–FY1996, FY1998 | Functional spending per pupil under DESE's older chart of accounts | Not yet read (see below) |
| `perpupil_post01_final.xls` | FY1997–FY2001 | Workbook: five-year day-program trends, functional and integrated-cost tables | Day program FY1997 |
| `perpupil_post02_1mar04_final.xls` | FY1998–FY2002 | Workbook, March 2004 | Day program FY1998 |
| `perpupil_post03_27aug04_final.xls` | FY1999–FY2003 | Workbook, August 2004 | Day program FY1999 |
| `perpupil_post04_22aug05_final.xls` | FY2000–FY2004 | Workbook, August 2005 | Day program FY2000–FY2004; statewide averages |
| `calcfunctional05final.xls` | FY2005 | Functional calculation workbook | Functional FY2005; day-program total FY2005 |
| `ppx06_day_program_FY06_final.xls` | FY2006 | Day-program workbook | Day program FY2006; statewide averages |
| `calcfunctional07dec.xls` | FY2007 | Functional calculation workbook, December 2007 | Functional FY2007 |
| `calcfunctional08final.xls` | FY2008 | Functional calculation workbook; also holds FY2006 and FY2007 data | Functional FY2008 and FY2006; check on FY2007 |

Where several workbooks hold the same year, the latest one is used: each five-year workbook revises
earlier years. Each workbook lists only the districts still operating when it was made, so for
FY1997 to FY1999 the PDFs supply the three it leaves out: Essex and Manchester (the workbooks
start Manchester Essex in FY2001) and Worcester Trade (last printed in FY1998). Essex and
Manchester have no FY2000 figure, since there is no FY2000 report.

## How the PDFs were read, and checked

The day-program tables in the FY1990–FY1999 reports were read from the OCR text with each word's
position on the page, placing each figure in its column by where it sits. The scans are slightly
rotated, so figures drift above or below their row toward the right edge of a page; the script
measures that drift on each page and corrects for it. Every row was then checked:

- **Name and code.** The name printed beside each code must match DESE's name for it in the
  workbooks, the Hub, or the list of 351 towns. Where the OCR dropped digits from a code ("1 GRANBY"
  for 111), the code was restored from the name, but only when the name matched one district
  clearly and the misread digits fit inside its code.
- **Total against programs.** A district's total is the weighted average of its program figures, so
  it must fall between the lowest and highest of them. DESE's own workbooks show vocational schools'
  totals running a little outside that range in about 2% of their rows; vocational rows within 15%
  are accepted and noted.
- **Against the workbooks.** FY1997, FY1998 and FY1999 appear both as PDF reports and in the
  workbooks. Of 990 rows and 3,272 figures read from those PDFs, 21 rows differ from the workbooks.
  Two are figures the OCR could not read, which the script skips rather than guesses. The other 19
  look like DESE's later revisions: the differences are small or move together across a row's
  columns, and the three checked against the page image (Foxborough FY1997, Seekonk FY1998,
  Orange FY1999) each print what was read.
- **Year to year.** A misread digit would show as a one-year spike that both neighbouring years
  contradict. Eight such spikes turned up across FY1990–FY2006; none is an OCR error (one is Heath,
  checked against its page, and seven come from the workbooks themselves).

Of the 2,424 rows and 8,068 figures read from the FY1990–FY1996 reports, 11 rows failed a check and
were read against the page image. Two figures were corrected by hand: Southwick-Tolland's FY1990
regular day (3,748) and Chelsea's FY1996 special needs (11,057). Eight rows are printed exactly as
read, however odd (Heath's tiny regular-day figures among them). One, Greenfield's FY1995 row, had
been split in two by the OCR, which the script now rejoins. The corrections and the rows confirmed
as printed are listed in `update_dese_history.py` (`CORRECTIONS`, `VERIFIED`). Smaller repairs,
such as a comma read as a period, are recorded row by row in `qa/day_program_pdf_rows.csv`.

The workbooks' figures were checked arithmetically: every functional category's per-pupil figure
equals its dollars divided by in-district FTE pupils, every total equals total spending divided by
total FTE, and the FY2007 figures in `calcfunctional07dec.xls` match the copy in
`calcfunctional08final.xls`. All pass (`qa/functional_checks.csv` is empty).

## Not yet read

- **The FY1990–FY1998 by-function reports.** They give functional spending per pupil under DESE's
  older chart of accounts (administration, instruction, pupil services, transportation, operations
  and maintenance, fixed charges). Their OCR is much rougher than the day-program reports': figures
  wrap onto other lines and decimal points are lost. Reading them needs an arithmetic check on
  every row (the functions must sum to the total), and the categories map only roughly onto today's.
- **The integrated cost tables.** Each report and the FY2001 workbook give an "integrated" cost per
  pupil for each of the 351 towns, counting what a town spends on its pupils wherever they attend.
  That fits the explorer's town view.
- **The workbooks' other sheets.** Special education by prototype, the FY2001 functional tables, and
  the line-by-line functional detail behind each category.
