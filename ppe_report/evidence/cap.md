# Evidence record: the cap tab's targets under another cap

Written by `evidence/cap.py` on 2026-11-08. What this is and isn't: see the script's docstring. These are AI-written checks of DESE's spreadsheets; only link 5, where there is one, is another person's calculation, and it starts from the same spreadsheet.

**To accept a year:** read its links below, then add the year to `REVIEWED` in `evidence/cap.py` with who and when (`2026: "Tim Shores, 2026-11-09"`), or ask Claude to, and rerun the script. The year's heading here, and the cap tab's Show the work, then say it was reviewed instead of "not yet reviewed".

## FY2026: 10 of 10 links pass; not yet reviewed by a person

### 1. DESE's workbook to the pipeline's CSV: PASS

- Workbook: https://www.doe.mass.edu/finance/chapter70/fy2026/chapter-2026.xlsm, SHA-256 `ff1dd669f3ae83d85822ed0dc77e30b9d80525569441b0ac52442bcdbe62520f`, read cell by cell.
- 351 towns; 7 figures each (eqv, income, cey, foundation, max_local, target, required); 0 differ from `data/dese_c70_townwide.csv` by more than $0.50.

### 2. DESE's formula, restated: PASS

- Percentages from the sheet: 0.324287% of EQV, 1.569938% of income.
- Yield = the two percentages times EQV and income, within $1: 351 of 351 towns.
- Maximum = 82.5% of the foundation budget, within $1: 351 of 351.
- Target = the lesser of the yield and the maximum, within $1: 351 of 351.
- Targets total 59.000% of all foundation budgets (DESE's rule: 59%).

### 3. The pipeline's CSV to the site's cap.json: PASS

- 351 towns in cap.json FY2026; 5 figures each; 0 differ by more than $1.
- cap.json built 2026-11-08.

### 4. The model, two ways: PASS

- No cap: the closed form's factor 0.7126452229, the bisection's 0.7126452229.
- No-cap targets by the bisection against cap.json's: 351 of 351 within $1.
- A 100% cap: the bisection's factor 0.8298916680, the exact solution's 0.8298916680.
- At DESE's 82.5% cap the bisection gives 0.9999999988: today's percentages, as it should.
- A town with a foundation budget of $0 (Gosnold) takes no part in setting the factor in either; with no cap its target is its yield times the factor, as in Naughton's workbook (Tim, #243), and under any cap $0.

### 5. An outside calculation: Michael Naughton's workbook: PASS

- Workbook: https://masspromise2invest.org/wp-content/uploads/2025/10/fy26-chapter-70-cap-comparisons.xlsx (posted on https://masspromise2invest.org/information-2/), SHA-256 `5087dc59fb14e8e67c23149cc8890a969424e2e4e14e14b1d2fc7b0d94b3555b`.
- At DESE's 82.5% cap: 351 of 351 towns match DESE's targets exactly.
- At a 100% cap: 297 of 351 towns match exactly, all within $1.
- With no cap: 282 of 351 towns match exactly, all within $1.
- Gosnold, with a foundation budget of $0: his no-cap target $563,230, the model's $563,230. Both leave it out of the factor and then give it its yield, so with no cap the targets total more than DESE's: his $9,036,315,385, the model's $9,036,315,374 (DESE's, $9,035,752,151).
- His workbook starts from the same DESE spreadsheet, so agreement shows the arithmetic is reproducible, not that DESE's figures are right.

### 5b. Naughton's required contributions with no cap: PASS

- 350 towns with figures in both. With DESE's rule that a required contribution can't exceed 90% of the foundation budget (the spreadsheet's column X), which the model keeps: 317 within $1; the other 33 are the towns where that rule applies.
- Without the rule, as his workbook figures them: 350 of 350 within $1.
- Acton: DESE's FY2026 sheet gives its prior-year figures as text, so the model has no required contribution for it.

### 5c. Naughton's required contributions with a 100% cap: PASS

- 350 towns with figures in both. With DESE's rule that a required contribution can't exceed 90% of the foundation budget (the spreadsheet's column X), which the model keeps: 310 within $1; the other 40 are the towns where that rule applies.
- Without the rule, as his workbook figures them: 350 of 350 within $1.
- Acton: DESE's FY2026 sheet gives its prior-year figures as text, so the model has no required contribution for it.

### 6. The cap tab's own model: PASS

- `scripts/check/cap-model.mjs` in mywebsite runs `capModel.ts`, the model the cap tab draws with, on cap.json and prints every town's target and required contribution with no cap and with a 100% cap.
- Against this record's own arithmetic: 702 of 702 town-scenarios agree within $1, target and required contribution both.

### 7. The income fifths: PASS

- Income per capita: each town's 2022 income from DESE's sheet over the Census Bureau's population estimate for July 2022 (`data/census_town_population.csv`): 351 of 351 towns' populations in cap.json match the Census file.
- Ranked and cut into five groups of 70 or 71: the cap tab's code puts 351 of 351 towns in the same fifth as this record.
- The explorer's income-fifth groups use the latest year's income; 40 of 351 towns sit in a different fifth there than in this year, by design: each year here is cut on its own income.
- The change in targets with no cap, by fifth, highest first:

| Fifth | Towns | At today's cap | Change in targets |
| --- | --- | --- | --- |
| Highest | 70 | 70 | +$1,015,523,810 |
| Fourth | 70 | 62 | +$161,299 |
| Middle | 70 | 33 | −$314,175,749 |
| Second | 70 | 12 | −$322,973,222 |
| Lowest | 71 | 15 | −$377,972,915 |

## FY2027: 7 of 7 links pass; not yet reviewed by a person

### 1. DESE's workbook to the pipeline's CSV: PASS

- Workbook: https://www.doe.mass.edu/finance/chapter70/fy2027/chapter-2027.xlsm, SHA-256 `b36eab3616b053fbe438374bb5d60d5aa35d5174e18dafb9d8e847be5e3b7985`, read cell by cell.
- 351 towns; 7 figures each (eqv, income, cey, foundation, max_local, target, required); 0 differ from `data/dese_c70_townwide.csv` by more than $0.50.

### 2. DESE's formula, restated: PASS

- Percentages from the sheet: 0.330978% of EQV, 1.589878% of income.
- Yield = the two percentages times EQV and income, within $1: 351 of 351 towns.
- Maximum = 82.5% of the foundation budget, within $1: 351 of 351.
- Target = the lesser of the yield and the maximum, within $1: 351 of 351.
- Targets total 59.000% of all foundation budgets (DESE's rule: 59%).

### 3. The pipeline's CSV to the site's cap.json: PASS

- 351 towns in cap.json FY2027; 5 figures each; 0 differ by more than $1.
- cap.json built 2026-11-08.

### 4. The model, two ways: PASS

- No cap: the closed form's factor 0.7217778080, the bisection's 0.7217778080.
- No-cap targets by the bisection against cap.json's: 351 of 351 within $1.
- A 100% cap: the bisection's factor 0.8351362353, the exact solution's 0.8351362353.
- At DESE's 82.5% cap the bisection gives 1.0000000003: today's percentages, as it should.
- A town with a foundation budget of $0 (Gosnold) takes no part in setting the factor in either; with no cap its target is its yield times the factor, as in Naughton's workbook (Tim, #243), and under any cap $0.

### 5. An outside calculation: none for this year

- No one outside this project has published FY2027 targets under another cap to compare with.

### 6. The cap tab's own model: PASS

- `scripts/check/cap-model.mjs` in mywebsite runs `capModel.ts`, the model the cap tab draws with, on cap.json and prints every town's target and required contribution with no cap and with a 100% cap.
- Against this record's own arithmetic: 702 of 702 town-scenarios agree within $1, target and required contribution both.

### 7. The income fifths: PASS

- Income per capita: each town's 2023 income from DESE's sheet over the Census Bureau's population estimate for July 2023 (`data/census_town_population.csv`): 351 of 351 towns' populations in cap.json match the Census file.
- Ranked and cut into five groups of 70 or 71: the cap tab's code puts 351 of 351 towns in the same fifth as this record.
- The explorer's income-fifth groups use the latest year's income; 0 of 351 towns sit in a different fifth there.
- The change in targets with no cap, by fifth, highest first:

| Fifth | Towns | At today's cap | Change in targets |
| --- | --- | --- | --- |
| Highest | 70 | 70 | +$1,000,375,716 |
| Fourth | 70 | 60 | +$6,320,084 |
| Middle | 70 | 34 | −$304,240,154 |
| Second | 70 | 15 | −$321,461,148 |
| Lowest | 71 | 14 | −$380,411,852 |

## 8. The site's build checks, every year: PASS

- `npm run check:units` in mywebsite runs `capModel.ts`, the model the cap tab draws with, on cap.json, for every year it has: each town's no-cap target against the pipeline's, within $1, and each required contribution the model's rules reproduce from today's targets.
- It reported: [check:units] 69 of 69 cases pass.
