# Evidence record: the single-family tax bill as a share of income per capita

Written by `evidence/bill_income.py` on 2026-11-09. 4 of 4 links pass; not yet reviewed by a person. These are AI-written checks of one DLS report; the outside report is a newspaper's reading of the same figure.

**To accept it:** read the links below, then set `REVIEWED` in `evidence/bill_income.py` to who and when, or ask Claude to, and rerun the script.

## 1. DLS's percentage is its bill over its income per capita: PASS

- `data/dls_single_family_tax_bill.csv`, DLS's report (https://dls-gw.dor.state.ma.us/reports/rdPage.aspx?rdReport=AverageSingleTaxBill.SingleFamTaxBill_wRange): 6812 town-years with a bill, an income per capita and DLS's percentage; 6812 agree to DLS's two decimals.

## 2. DLS's report to the site's town files: PASS

- 6439 town-years, FY2005 to FY2025; 6439 of the site's shares equal DLS's bill over its income per capita, to four decimals.

## 3. The highest share, FY2025: PASS

- DLS's figures rank Amherst first in FY2025, at 40.76% (a $9,693 bill over $23,779 of income per capita), then Tisbury, 39.86%, and Goshen, 25.90%; 351 towns in all.
- The map's data, which About the data reads for its sentence, ranks Amherst first at 40.76%.

## 4. An outside report: the Amherst Indy: PASS

- The Amherst Indy (https://www.amherstindy.org/2025/08/01/by-this-measure-amherst-has-the-least-affordable-property-taxes-in-massachusetts/) reported Amherst's FY2025 bill at 40.76% of income per capita, the highest in the state; DLS's figure is 40.76%.
- The newspaper read the same DLS report, so agreement shows the figure is DLS's, not that DLS's inputs are right.
