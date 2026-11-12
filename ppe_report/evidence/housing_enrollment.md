# Evidence record: the Housing and enrollment tab

Written by `evidence/housing_enrollment.py` on 2026-11-12. 9 of 9 links pass; not yet reviewed by a person. These are AI-written checks of the site's figures against the ones MAPC, AFT, DESE and DLS published.

**To accept it:** read the links below, then set `REVIEWED` in `evidence/housing_enrollment.py` to who and when, or ask Claude to, and rerun the script.

## 1. The towns' housing units add up to the state's: PASS

- 2010: the 351 towns sum to 2,808,254; the file's state record says 2,808,254.
- 2020: the 351 towns sum to 2,998,537; the file's state record says 2,998,537.

## 2. DESE's enrollment against MAPC's 2024 counts: PASS

- MAPC's update (https://www.mapc.org/learn/research-analysis/enrollment/) and the site's 231 districts, DESE's counts for 2010/11 and 2019/20:
  - districts: MAPC 231, the site 231
  - declined: MAPC 155, the site 155
  - grew: MAPC 76, the site 76
  - over_10: MAPC 28, the site 28
  - under_20: MAPC 22, the site 22
  - statewide: MAPC -0.7, the site -0.7
- MAPC counts the one district with no change among those that grew; every public school's students, charter schools included, for the statewide change.

## 3. MAPC's region by community type: PASS

- Housing change, enrollment change, districts, summed over each type:
  - Inner Core: MAPC 9.3%, 2.2%, 16 districts; the site 9.3%, 2.2%, 16
  - Regional Urban Center: MAPC 6.3%, 4.7%, 11 districts; the site 6.3%, 4.7%, 11
  - Maturing Suburb: MAPC 6.0%, -2.5%, 43 districts; the site 6.0%, -2.5%, 43
  - Developing Suburb: MAPC 9.6%, -8.2%, 23 districts; the site 9.6%, -8.2%, 23
  - All: MAPC 7.9%, -0.3%, 93 districts; the site 7.9%, -0.3%, 93

## 4. The straight line: PASS

- Least squares over 231 districts: slope 0.180, p = 0.177, R² = 0.008; MAPC: "the association between the two variables is not different from zero."
- Where it crosses zero housing growth: -6.6%; MAPC: "declined 6 percent."
- Without the districts drawn off the chart (Gosnold, Hancock): slope 0.278, p = 0.136.

## 5. MAPC's 2017 data against its brief: PASS

- The data file (https://www.mapc.org/learn/research-analysis/enrollment-2017/): 234 districts, 159 lost students; the brief: 234 and 159.
- Hopkinton: permits 18.23% of its 2010 housing, enrollment +0.23%; the brief: 18% and +0.23%.

## 6. AFT's medians, to within a cent: PASS

- 175 rows outside Indiana, read from the fact sheet's table (https://farmlandinfo.org/wp-content/uploads/sites/2/2019/09/cost-of-community-services-fact-sheet.pdf).
  - res: the table's median 1.16; AFT's chart 1.16
  - ci: the table's median 0.30; AFT's chart 0.30
  - wol: the table's median 0.38; AFT's chart 0.37
- Working and open land is a cent off AFT's chart; the chart may count a different set of studies.

## 7. DLS's new growth: PASS

- 7370 town-years: 7370 of the site's shares equal DLS's own percentage, new growth over the prior year's levy limit, to DLS's two decimals.

## 8. Chapter 70, FY2027: PASS

- DESE's spreadsheet (https://www.doe.mass.edu/finance/chapter70/fy2027/chapter-2027.xlsm): 263 towns below target, 87 above, the property percentage 0.3310% of 2024 valuation.
- DESE's summary (https://www.doe.mass.edu/finance/chapter70/fy2027/chapter-2027-whitepaper.docx): 263, 87, 0.331% of 2024.

## 9. Hancock's students: PASS

- MAPC gives 41 to 47. DESE counts 41.0 in 2009/10, 48.0 in 2010/11 and 47.0 in 2019/20: MAPC's first figure is the year before the others it gives. The site uses 2010/11, as MAPC's other counts do.
