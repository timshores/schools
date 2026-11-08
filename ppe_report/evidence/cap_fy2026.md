# Evidence record: the no-cap counterfactual, FY2026

Written by `evidence/cap_fy2026.py` on 2026-11-08. Every link passed. What this is and isn't: see the script's docstring. These are AI-written checks of one DESE spreadsheet's figures; only link 5 is another person's calculation, and it starts from the same spreadsheet.

## 1. DESE's workbook to the pipeline's CSV: PASS

- Workbook: https://www.doe.mass.edu/finance/chapter70/fy2026/chapter-2026.xlsm, SHA-256 `ff1dd669f3ae83d85822ed0dc77e30b9d80525569441b0ac52442bcdbe62520f`, read cell by cell.
- 351 towns; 7 figures each (eqv, income, cey, foundation, max_local, target, required); 0 differ from `data/dese_c70_townwide.csv` by more than $0.50.

## 2. DESE's formula, restated: PASS

- Percentages from the sheet: 0.324287% of EQV, 1.569938% of income.
- Yield = the two percentages times EQV and income, within $1: 351 of 351 towns.
- Maximum = 82.5% of the foundation budget, within $1: 351 of 351.
- Target = the lesser of the yield and the maximum, within $1: 351 of 351.
- Targets total 59.000% of all foundation budgets (DESE's rule: 59%).

## 3. The pipeline's CSV to the site's cap.json: PASS

- 351 towns in cap.json FY2026; 5 figures each; 0 differ by more than $1.
- cap.json built 2026-11-08.

## 4. The model, two ways: PASS

- No cap: the closed form's factor 0.7126452229, the bisection's 0.7126452229.
- No-cap targets by the bisection against cap.json's: 351 of 351 within $1.
- A 100% cap: the bisection's factor 0.8298916680, the exact solution's 0.8298916680.
- At DESE's 82.5% cap the bisection gives 0.9999999988: today's percentages, as it should.
- Gosnold, with a foundation budget of $0, has no target under any cap in both.

## 5. An outside calculation: Michael Naughton's workbook: PASS

- Workbook: https://masspromise2invest.org/wp-content/uploads/2025/10/fy26-chapter-70-cap-comparisons.xlsx (posted on https://masspromise2invest.org/information-2/), SHA-256 `5087dc59fb14e8e67c23149cc8890a969424e2e4e14e14b1d2fc7b0d94b3555b`.
- At DESE's 82.5% cap: 351 of 351 towns match DESE's targets exactly.
- At a 100% cap: 297 of 351 towns match exactly, all within $1.
- With no cap: 281 of the 350 towns with a foundation budget match exactly, all within $1.
- Gosnold, with a foundation budget of $0: he gives it a no-cap target of $563,230; the model gives it none, since the formula asks a town for a share of its foundation budget. That accounts for the difference in the totals, give or take rounding: his $9,036,315,385, the model's $9,035,752,144 (DESE's, $9,035,752,151).
- His workbook starts from the same DESE spreadsheet, so agreement shows the arithmetic is reproducible, not that DESE's figures are right.

## 6. The cap tab's own model: PASS

- `npm run check:units` in mywebsite runs `capModel.ts`, the model the cap tab draws with, on cap.json, for every year it has: each town's no-cap target against the pipeline's, within $1, and each required contribution the model's rules reproduce from today's targets.
- It reported: [check:units] 69 of 69 cases pass.
