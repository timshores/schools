# Source data

The files here are the public data behind the
[Massachusetts school spending explorer](https://timshores.com/mass-school-spending/) and the
Western Massachusetts per-pupil expenditure report. Each comes from a state or federal agency, and
the script named beside it downloads it again. The scripts change only the layout: rows are kept as
each agency published them, apart from the notes below. `export_explorer_data.py` turns these files
into the explorer's data.

"Retrieved" is the date the file was last downloaded. Where the agency states a date for the data
itself, it is given too.

## School districts

| File | What it holds | Source | Retrieved | Script |
| --- | --- | --- | --- | --- |
| [dese_expenditures.csv](dese_expenditures.csv) | Spending per pupil by category, enrollment, and DESE's other district indicators (staffing, salaries, demographics, MCAS) for every district, FY2009 on | DESE, [District Expenditures by Spending Category](https://educationtocareer.data.mass.gov/d/er3w-dyti) (Education-to-Career Hub dataset er3w-dyti); data last updated August 7, 2026 | October 18, 2026 | `update_dese_data.py` |
| [dese_c70_profile.csv](dese_c70_profile.csv) | Each district's foundation enrollment, foundation budget, required local contribution, Chapter 70 aid, and required and actual net school spending, by year | DESE, [Chapter 70 district profiles](https://www.doe.mass.edu/finance/chapter70/default.html) (`profile.xlsx`, DataC70 sheet), dated March 26, 2026; the downloaded workbook is [dese_c70_profile.xlsx](dese_c70_profile.xlsx) | October 18, 2026 | `update_dese_data.py` |
| [dese_c70_regional_members.csv](dese_c70_regional_members.csv) | Each regional district's member towns and each town's FY2025 required contribution | DESE, [FY2025 Chapter 70 regional district summary](https://www.doe.mass.edu/finance/chapter70/fy2025/chapter-2025.html) (`chapter-2025-reg.xlsx`, saved as [dese_c70_regional_fy2025.xlsx](dese_c70_regional_fy2025.xlsx)) | October 21, 2026 | `update_dese_data.py` |
| [census_f33_ma.csv](census_f33_ma.csv) | Revenue by source and spending for every Massachusetts school system, FY2009 to FY2024, with all of the Census Bureau's columns (dollars in thousands) | Census Bureau, [Annual Survey of School System Finances](https://www.census.gov/programs-surveys/school-finances.html) (F-33); Massachusetts rows of each year's national file | October 21, 2026 | `update_census_data.py` |
| [nces_ma_districts.csv](nces_ma_districts.csv) | The crosswalk from Census and NCES district IDs to DESE's district codes | NCES [Common Core of Data](https://nces.ed.gov/ccd/) district directory, through the Urban Institute's [Education Data Portal](https://educationdata.urban.org/) | October 21, 2026 | `update_census_data.py` |

## Chapter 70 formula

| File | What it holds | Source | Retrieved | Script |
| --- | --- | --- | --- | --- |
| [dese_c70_townwide.csv](dese_c70_townwide.csv) | Every town's local contribution calculation for FY2026 and FY2027: EQV, income, combined effort yield, foundation budget, the 82.5% maximum, target and required contribution | DESE's complete Chapter 70 formula spreadsheets, Townwide Contributions sheet ([FY2026](https://www.doe.mass.edu/finance/chapter70/fy2026/chapter-2026.xlsm), [FY2027](https://www.doe.mass.edu/finance/chapter70/fy2027/chapter-2027.xlsm)) | October 21, 2026 | `update_c70_formula.py` |
| [dese_c70_town_history.csv](dese_c70_town_history.csv) | Every town's yield, foundation budget, target and required contribution, FY2007 on | DESE, [Chapter 70 Trends in Aid and Local Contribution](https://www.doe.mass.edu/finance/chapter70/keyfactors.xlsx) (`keyfactors.xlsx`, dataContribution sheet) | October 21, 2026 | `update_c70_formula.py` |

## Cities and towns

| File | What it holds | Source | Retrieved | Script |
| --- | --- | --- | --- | --- |
| [dls_schedule_a_general_fund.csv](dls_schedule_a_general_fund.csv) | Every municipality's general fund revenue by source and spending by function, FY2002 on | Division of Local Services, [Schedule A General Fund](https://dls-gw.dor.state.ma.us/reports/rdPage.aspx?rdReport=ScheduleA.GenFund_MAIN) (Municipal Databank); data current as of October 21, 2026 | October 21, 2026 | `update_dls_data.py` |
| [dls_la4_parcels.csv](dls_la4_parcels.csv) | Parcels and assessed values by property use, FY2003 on | Division of Local Services, [LA-4 parcel counts and values](https://dls-gw.dor.state.ma.us/reports/rdPage.aspx?rdReport=PropertyTaxInformation.LA4.Parcel_counts_vals) | October 21, 2026 | `update_dls_property_data.py` |
| [dls_eqv.csv](dls_eqv.csv) | Equalized valuations, 1970 on | Division of Local Services, [Equalized Valuations](https://dls-gw.dor.state.ma.us/reports/rdPage.aspx?rdReport=PropertyTaxInformation.EQV.EQV) | October 21, 2026 | `update_dls_property_data.py` |
| [dls_single_family_tax_bill.csv](dls_single_family_tax_bill.csv) | Average single-family value and tax bill, and DOR income per capita, 1988 on | Division of Local Services, [Average Single Family Tax Bill](https://dls-gw.dor.state.ma.us/reports/rdPage.aspx?rdReport=AverageSingleTaxBill.SingleFamTaxBill_wRange) | October 21, 2026 | `update_dls_property_data.py` |
| [dls_levy_capacity.csv](dls_levy_capacity.csv) | Each town's Proposition 2½ levy limit, levy, excess capacity and ceiling, FY2007 on | Division of Local Services, [Excess Levy Capacity, Levy Ceiling and Overrides](https://dls-gw.dor.state.ma.us/reports/rdPage.aspx?rdReport=Prop2.5.ExcessLevyCapandOverride_10_pres) | October 21, 2026 | `update_dls_property_data.py` |
| [census_town_population.csv](census_town_population.csv) | Every town's population on July 1 at the start of each fiscal year, FY2009 on | Census Bureau, [subcounty population estimates](https://www.census.gov/programs-surveys/popest.html) (2000–2010 intercensal, 2010–2020, and 2020s vintages) | October 21, 2026 | `update_census_data.py` |
| [census_town_households.csv](census_town_households.csv) | Every town's households, each five-year release from 2005–2009 on | Census Bureau, [American Community Survey](https://www.census.gov/programs-surveys/acs) five-year summary files, table B11001 | October 21, 2026 | `update_acs_households.py` |
| [town_boundaries.json](town_boundaries.json) | Town outlines for the explorer's map, and each town's land area | Census Bureau, [2023 cartographic boundary file](https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_25_cousub_500k.zip) of county subdivisions, redrawn as SVG paths | October 22, 2026 | `update_town_boundaries.py` |

## Prices

| File | What it holds | Source | Retrieved | Script |
| --- | --- | --- | --- | --- |
| [inflation_indices.csv](inflation_indices.csv) | Fiscal-year averages of the Northeast and U.S. consumer price indexes and the state and local government price deflator | Bureau of Labor Statistics, [CPI-U Northeast](https://data.bls.gov/timeseries/CUUR0100SA0) (CUUR0100SA0) and [U.S. city average](https://data.bls.gov/timeseries/CUUR0000SA0); Bureau of Economic Analysis, [NIPA quarterly data](https://apps.bea.gov/national/Release/TXT/NipaDataQ.txt), Table 1.1.9, line 26 | October 18, 2026 | `update_inflation_data.py` |

## Notes

- **What the scripts change.** `update_census_data.py` keeps only Massachusetts rows and adds DESE's
  district codes. `update_dls_*` and `update_acs_households.py` stack each year's download into one
  long table. `update_inflation_data.py` averages monthly or quarterly index values over each
  July–June fiscal year. Nothing else is altered.
- **Not kept here.** The Census Bureau's national F-33 files (`census_raw/`, about 130 MB) and
  DESE's formula spreadsheets (`c70_raw/`) are cached locally and not committed; the links above
  download them.
- **The report's workbooks.** `E2C_Hub_MA_DESE_Data.xlsx`, `Ch 70 District Profiles Actual NSS Over
  Required.xlsx`, `keyfactors.xlsx`, `dese_c70_fy2023_reference.csv`, the school district shapefiles in
  `shapefiles/`, and `cache/` serve the PDF report, which was built before the explorer.
