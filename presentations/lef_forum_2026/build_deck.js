// Leverett Education Foundation Education Forum, November 2026: seven-slide deck.
// Every number comes from deck_data.json (deck_data.py). Rebuild:
//   cd ppe_report && python ../presentations/lef_forum_2026/deck_data.py
//   cd ../presentations/lef_forum_2026 && node build_deck.js

const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");

const D = JSON.parse(fs.readFileSync(path.join(__dirname, "deck_data.json"), "utf8"));
const OUT = path.join(__dirname, "Leverett school spending in context - LEF forum 2026-11.pptx");

const L = String(D.latest_year), F = String(D.first_year);
const N_YEARS = D.latest_year - D.first_year;
const PEERS = "Western MA Tiny";
const usd = (v) => "$" + Math.round(v).toLocaleString("en-US");
const pct = (v, d = 0) => (v >= 0 ? "+" : "\u2212") + Math.abs(v * 100).toFixed(d) + "%";
const rate = (a, b, n) => Math.pow(b / a, 1 / n) - 1;
const ppe = (g, y) => D.ppe[g][String(y)];
const real = (g, y) => D.ppe_real[g][String(y)];
const realChange = (g) => ppe(g, L) / real(g, F) - 1;
const nomChange = (g) => ppe(g, L) / ppe(g, F) - 1;
const cpiFactor = D.cpi_u_northeast_fy[L] / D.cpi_u_northeast_fy[F];
const TB = D.town_budgets.towns;
const TOWN_PEERS = "Western MA towns of 1,000-2,499";
const share = (v) => Math.round(v * 100) + "%";
const nomRate = (g) => rate(ppe(g, F), ppe(g, L), N_YEARS);
const salary = (g) => D.teacher_pay[g].avg_salary[L];
const levRank = D.tiny_ranking.findIndex((r) => r.district === "Leverett") + 1;

const THEME = {
  name: "Leverett Forum",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1F2A2E", lt1: "FFFFFF", dk2: "23423A", lt2: "EEF2EF",
    accent1: "C8553D", accent2: "2F6F62", accent3: "9DB5AC", accent4: "E0A458",
    accent5: "5B6C8F", accent6: "7A8580", hlink: "2F6F62", folHlink: "5B6C8F",
  },
};
const HEX = THEME.colors;
const GRID = "D9DFDC";
const MUTED = "5E6B66";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.author = "Tim Shores";
pres.title = "Leverett school spending in context";
const C = pres.SchemeColor;

pres.defineSlideMaster({
  title: "Title dark",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.7, y: 1.0, w: 7.2, h: 2.2,
      fontSize: 40, bold: true, color: C.background1, valign: "bottom", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.7, y: 3.45, w: 7.0, h: 1.3,
      fontSize: 18, color: C.background2, valign: "top", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "byline", type: "body", x: 0.7, y: 6.35, w: 8.0, h: 0.5,
      fontSize: 14, color: C.background2, valign: "top", margin: 0 }, text: "" } },
  ],
});

pres.defineSlideMaster({
  title: "Chart and commentary",
  background: { color: C.background1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 1.05,
      fontSize: 32, bold: true, color: C.text2, valign: "middle", align: "left", margin: 0 }, text: "" } },
    { text: { text: `Data: Massachusetts DESE, FY${F}–FY${L}. Analysis: Tim Shores, with Claude Code. Report: timshores.com/portfolio · Explorer: timshores.com/mass-school-spending`,
      options: { x: 0.6, y: 6.95, w: 11.2, h: 0.3, fontSize: 10, color: C.accent6, margin: 0 } } },
  ],
  slideNumber: { x: 12.2, y: 6.95, w: 0.5, h: 0.3, fontSize: 10, color: C.accent6, align: "right" },
});

const chartText = { catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
  legendFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt", titleFontFace: "+mn-lt" };
const axisStyle = () => ({
  ...chartText,
  catAxisLabelColor: MUTED, valAxisLabelColor: MUTED, catAxisLabelFontSize: 12, valAxisLabelFontSize: 12,
  valGridLine: { color: GRID, size: 0.75 }, catGridLine: { style: "none" },
  catAxisLineShow: true, catAxisLineColor: GRID, valAxisLineShow: false,
  legendColor: HEX.dk1, legendFontSize: 13,
});

function callout(slide, { x, y, w, value, label, valueColor, name }) {
  slide.addText([
    { text: value, options: { fontSize: 30, bold: true, color: valueColor || C.text2, breakLine: true,
      fontFace: THEME.headFontFace } },
    { text: label, options: { fontSize: 14, color: C.text1 } },
  ], { x, y, w, h: 1.05, margin: 0, valign: "top", isTextBox: true, objectName: name });
}

function notes(slide, lines) { slide.addNotes(lines.join("\n\n")); }

// ---------- Slide 1: title and the message ----------
pres.addSection({ title: "Opening" });
{
  const s = pres.addSlide({ masterName: "Title dark", sectionTitle: "Opening" });
  s.addText("Leverett's school spending in context", { placeholder: "title" });
  s.addText(`Per-pupil spending at Leverett Elementary, ${F}–${L}, compared with the ARPS towns and Western Massachusetts`,
    { placeholder: "body" });
  s.addText("Tim Shores  ·  Leverett Education Foundation Education Forum  ·  November 2026",
    { placeholder: "byline" });

  const stats = [
    [usd(ppe("Leverett", L)), `Leverett, spending per pupil, FY${L}`],
    [usd(ppe(PEERS, L)), `Average for the ${D.tiny_cohort_size} Western MA districts with 200 or fewer students`],
    [`${pct(realChange("Leverett"))} vs ${pct(realChange(PEERS))}`,
      `Growth since ${F} after inflation: Leverett vs those districts`],
  ];
  stats.forEach(([v, lab], i) => {
    const y = 1.0 + i * 1.75;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 8.55, y, w: 4.2, h: 1.5, rectRadius: 0.12,
      fill: { color: C.background1, transparency: 88 }, line: { color: C.background1, transparency: 70, width: 0.75 },
      objectName: `stat card ${i + 1}` });
    s.addText([
      { text: v, options: { fontSize: 30, bold: true, color: C.background1, breakLine: true, fontFace: THEME.headFontFace } },
      { text: lab, options: { fontSize: 13, color: C.background2 } },
    ], { x: 8.8, y: y + 0.15, w: 3.75, h: 1.2, margin: 0, valign: "middle", isTextBox: true, objectName: `stat ${i + 1}` });
  });
  notes(s, [
    "Per-pupil expenditure (PPE) is a flawed lens on school budgets: one number hides student needs, buildings, governance and history. It is also the standard the state reports and the public uses, so it is worth understanding in full.",
    `The comparison group is the ${D.tiny_cohort_size} Western MA traditional districts with 200 or fewer in-district students in FY${L}, the report's "Tiny" cohort definition. Leverett ranks ${levRank} of ${D.tiny_cohort_size} on FY${L} PPE.`,
    "Inflation uses the CPI-U for the Northeast, averaged over each July–June fiscal year.",
    `Source: Western MA Per Pupil Expenditure Report, FY${F}–FY${L} edition. As its "How this report was made" statement says, Tim designed the analysis and is responsible for it; Claude Code helped write the code and draft the text. This deck was built the same way.`,
  ]);
}

// ---------- Slide 2: Western MA backdrop ----------
pres.addSection({ title: "Western MA" });
{
  const s = pres.addSlide({ masterName: "Chart and commentary", sectionTitle: "Western MA" });
  s.addText("Small districts spend more per pupil, and all of them grew", { placeholder: "title" });
  const years = D.years.map(String);
  s.addChart(pres.charts.LINE, [
    { name: `Districts with 200 or fewer students (${D.tiny_cohort_size})`, labels: years, values: years.map((y) => ppe(PEERS, y)) },
    { name: `All Western MA districts (${D.western_district_count})`, labels: years, values: years.map((y) => ppe("Western MA", y)) },
  ], {
    x: 0.6, y: 1.55, w: 7.6, h: 5.2, ...axisStyle(),
    chartColors: [HEX.accent2, HEX.accent6], lineSize: 3, lineDataSymbol: "none",
    showLegend: true, legendPos: "b", valAxisLabelFormatCode: "$#,##0", valAxisMinVal: 0,
    catAxisLabelFrequency: 2, showTitle: true, title: "Spending per pupil, nominal dollars",
    titleFontSize: 14, titleColor: HEX.dk1, objectName: "Western MA PPE chart",
  });
  callout(s, { x: 8.7, y: 1.7, w: 4.0, value: `${(nomRate("Western MA") * 100).toFixed(1)}% a year`,
    label: `Average growth in Western MA per-pupil spending, ${F}–${L}`, name: "callout growth" });
  callout(s, { x: 8.7, y: 3.15, w: 4.0, value: usd(ppe(PEERS, L) - ppe("Western MA", L)),
    label: `More per pupil in districts with 200 or fewer students, FY${L}: fixed costs spread over fewer students`, name: "callout small premium" });
  callout(s, { x: 8.7, y: 4.6, w: 4.0, value: pct(realChange("Western MA")),
    label: `Western MA growth since ${F} after inflation`, name: "callout real" });
  notes(s, [
    "Schools are labor-intensive: most of a budget is salaries and benefits, which follow regional labor markets rather than local revenue. Staffing minimums, buildings, transportation and special education services do not shrink in proportion to enrollment, so small districts spend more per pupil.",
    `Western MA averages are weighted by enrollment; Springfield is excluded as an outlier. ${D.western_district_count} traditional districts in FY${L}.`,
  ]);
}

// ---------- Slide 3: ARPS backdrop ----------
pres.addSection({ title: "ARPS" });
{
  const s = pres.addSlide({ masterName: "Chart and commentary", sectionTitle: "ARPS" });
  s.addText("The ARPS towns started high and grew at ordinary rates", { placeholder: "title" });
  const groups = [["Western MA", "Western MA"], [PEERS, "200 or fewer"],
    ["Amherst-Pelham Regional", "A-P Regional"], ["Amherst", "Amherst"], ["Shutesbury", "Shutesbury"],
    ["Pelham", "Pelham"], ["Leverett", "Leverett"]];
  s.addChart(pres.charts.BAR, [
    { name: `FY${F}`, labels: groups.map((g) => g[1]), values: groups.map((g) => ppe(g[0], F)) },
    { name: `FY${L}`, labels: groups.map((g) => g[1]), values: groups.map((g) => ppe(g[0], L)) },
  ], {
    x: 0.6, y: 1.55, w: 7.9, h: 5.2, barDir: "col", barGrouping: "clustered", barGapWidthPct: 60,
    ...axisStyle(), catAxisLabelFontSize: 12, catAxisLabelRotate: 0, chartColors: [HEX.accent3, HEX.dk2],
    showLegend: true, legendPos: "b", valAxisLabelFormatCode: "$#,##0", valAxisMinVal: 0,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: '$#,##0,"K"', dataLabelFontSize: 10,
    dataLabelColor: HEX.dk1, showTitle: true, title: "Spending per pupil, nominal dollars",
    titleFontSize: 14, titleColor: HEX.dk1, objectName: "ARPS PPE chart",
  });
  const arps = ["Amherst-Pelham Regional", "Amherst", "Shutesbury", "Leverett", "Pelham"];
  const rates = arps.map((g) => nomRate(g) * 100);
  callout(s, { x: 9.0, y: 1.7, w: 3.8, value: usd(Math.min(...arps.map((g) => ppe(g, F)))) + "+",
    label: `Every ARPS district spent more per pupil in ${F} than the Western MA average of ${usd(ppe("Western MA", F))}`,
    name: "callout start" });
  callout(s, { x: 9.0, y: 3.25, w: 3.8, value: `${Math.min(...rates).toFixed(1)}–${Math.max(...rates).toFixed(1)}% a year`,
    label: `ARPS growth since ${F}, against ${(nomRate("Western MA") * 100).toFixed(1)}% for Western MA`, name: "callout rates" });
  s.addText("Today's high spending is a legacy of a high starting point, not runaway growth.", {
    x: 9.0, y: 4.85, w: 3.8, h: 1.2, fontSize: 16, italic: true, color: C.accent2, margin: 0, valign: "top",
    isTextBox: true, objectName: "takeaway" });
  notes(s, [
    "This is the main finding of the Western MA Per Pupil Expenditure Report: in 2009 the ARPS districts already spent well above regional averages, and ordinary growth applied to a high base keeps them above average. The question shifts from 'why are costs so high?' to 'how do we manage the system we have inherited at ordinary growth rates?'",
    ...arps.map((g) => `${g}: ${usd(ppe(g, F))} to ${usd(ppe(g, L))}, ${(nomRate(g) * 100).toFixed(1)}% a year.`),
  ]);
}

// ---------- Slide 4: what inflation does to the same numbers ----------
pres.addSection({ title: "Inflation" });
{
  const s = pres.addSlide({ masterName: "Chart and commentary", sectionTitle: "Inflation" });
  s.addText(`Most of the growth since ${F} is inflation`, { placeholder: "title" });
  const groups = [["Leverett", "Leverett"], ["Amherst-Pelham Regional", "Amherst-Pelham Regional"]];
  s.addChart(pres.charts.BAR, [
    { name: `FY${F}, as spent`, labels: groups.map((g) => g[1]), values: groups.map((g) => ppe(g[0], F)) },
    { name: `FY${F} in FY${L} dollars`, labels: groups.map((g) => g[1]), values: groups.map((g) => real(g[0], F)) },
    { name: `FY${L}`, labels: groups.map((g) => g[1]), values: groups.map((g) => ppe(g[0], L)) },
  ], {
    x: 0.6, y: 1.55, w: 7.9, h: 5.2, barDir: "col", barGrouping: "clustered", barGapWidthPct: 55,
    ...axisStyle(), catAxisLabelFontSize: 13, chartColors: [HEX.accent3, HEX.accent4, HEX.dk2],
    showLegend: true, legendPos: "b", valAxisLabelFormatCode: "$#,##0", valAxisMinVal: 0,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "$#,##0", dataLabelFontSize: 11,
    dataLabelColor: HEX.dk1, showTitle: true, title: "Spending per pupil: as spent, and restated in today's dollars",
    titleFontSize: 14, titleColor: HEX.dk1, objectName: "inflation demo chart",
  });
  callout(s, { x: 9.0, y: 1.7, w: 3.8, value: `$1.00 → $${cpiFactor.toFixed(2)}`,
    label: `A dollar spent in FY${F} buys what $${cpiFactor.toFixed(2)} buys in FY${L}, by Northeast consumer prices`,
    name: "callout dollar" });
  callout(s, { x: 9.0, y: 3.15, w: 3.8, value: `${pct(nomChange("Leverett"))} → ${pct(realChange("Leverett"))}`,
    label: `Leverett's growth in spending per pupil since FY${F}, as spent, then after inflation`,
    valueColor: C.accent1, name: "callout Leverett" });
  callout(s, { x: 9.0, y: 4.6, w: 3.8, value: `${pct(nomChange("Amherst-Pelham Regional"))} → ${pct(realChange("Amherst-Pelham Regional"))}`,
    label: "The same for Amherst-Pelham Regional", name: "callout ARPS" });
  const inflShare = (g) => Math.round((real(g, F) - ppe(g, F)) / (ppe(g, L) - ppe(g, F)) * 100);
  notes(s, [
    `How to read it: the first bar is what each district spent per pupil in FY${F}. The second is the same spending in FY${L} dollars, what it would take today to buy what that money bought then. The third is what the district spent in FY${L}. The gap between the first two bars is inflation; only the gap between the last two is growth.`,
    `Leverett: ${usd(ppe("Leverett", F))} in FY${F} is ${usd(real("Leverett", F))} in FY${L} dollars, against ${usd(ppe("Leverett", L))} spent in FY${L}. Inflation accounts for ${inflShare("Leverett")}% of the increase. Amherst-Pelham Regional: ${usd(ppe("Amherst-Pelham Regional", F))}, ${usd(real("Amherst-Pelham Regional", F))}, ${usd(ppe("Amherst-Pelham Regional", L))}; inflation is ${inflShare("Amherst-Pelham Regional")}% of the increase.`,
    `Prices here are the CPI-U for the Northeast, averaged over each July–June fiscal year. The report's inflation addendum also uses the federal price index for what state and local governments buy, mostly labor; by that measure the real growth is smaller still.`,
    "The FY2025 dollars switch in the online explorer does this for every district and town: timshores.com/mass-school-spending.",
  ]);
}

// ---------- Slide 5: Leverett ----------
pres.addSection({ title: "Leverett" });
{
  const s = pres.addSlide({ masterName: "Chart and commentary", sectionTitle: "Leverett" });
  s.addText("Leverett went from above its peers to below them", { placeholder: "title" });
  const years = D.years.map(String);
  s.addChart(pres.charts.LINE, [
    { name: "Leverett", labels: years, values: years.map((y) => real("Leverett", y)) },
    { name: "Districts with 200 or fewer students", labels: years, values: years.map((y) => real(PEERS, y)) },
    { name: "All Western MA districts", labels: years, values: years.map((y) => real("Western MA", y)) },
  ], {
    x: 0.6, y: 1.55, w: 7.6, h: 5.2, ...axisStyle(),
    chartColors: [HEX.accent1, HEX.accent2, HEX.accent6], lineSize: 3, lineDataSymbol: "none",
    showLegend: true, legendPos: "b", valAxisLabelFormatCode: "$#,##0", valAxisMinVal: 0,
    catAxisLabelFrequency: 2, showTitle: true, title: `Spending per pupil in FY${L} dollars (adjusted for inflation)`,
    titleFontSize: 14, titleColor: HEX.dk1, objectName: "Leverett real PPE chart",
  });
  const enr = D.enrollment.Leverett;
  callout(s, { x: 8.7, y: 1.7, w: 4.0, value: `${levRank} of ${D.tiny_cohort_size}`,
    label: `Leverett's rank on FY${L} spending per pupil among districts with 200 or fewer students`,
    valueColor: C.accent1, name: "callout rank" });
  callout(s, { x: 8.7, y: 3.15, w: 4.0, value: pct(ppe("Leverett", L) / real("Leverett", "2014") - 1),
    label: `Change in Leverett's spending per pupil since 2014, after inflation`, valueColor: C.accent1,
    name: "callout since 2014" });
  callout(s, { x: 8.7, y: 4.6, w: 4.0, value: `${Math.round(enr[F])} → ${Math.round(enr["2014"])} → ${Math.round(enr[L])}`,
    label: `Students at Leverett Elementary, ${F}, 2014 and ${L}, residents and school-choice students together`,
    name: "callout enrollment" });
  notes(s, [
    `In ${F} Leverett spent ${usd(ppe("Leverett", F))} per pupil, above the ${usd(ppe(PEERS, F))} average for the districts that are its size today. In FY${L} it spent ${usd(ppe("Leverett", L))}, against ${usd(ppe(PEERS, L))}.`,
    `When enrollment fell from ${Math.round(enr[F])} to ${Math.round(enr["2014"])}, fixed costs were spread over fewer students and per-pupil spending jumped. Enrollment recovered, and since 2014 spending per pupil has grown ${(rate(ppe("Leverett", 2014), ppe("Leverett", L), D.latest_year - 2014) * 100).toFixed(1)}% a year in nominal dollars, from ${usd(ppe("Leverett", 2014))} to ${usd(ppe("Leverett", L))}: slower than prices, so it fell after inflation.`,
    `Enrollment here is DESE in-district FTE: Leverett residents plus school-choice students. Foundation enrollment, the residents the town pays for, was ${Math.round(D.leverett_foundation_enrollment[L])} in FY${L}.`,
  ]);
}

// ---------- Slide 6: pay and who pays ----------
pres.addSection({ title: "Pay and funding" });
{
  const s = pres.addSlide({ masterName: "Chart and commentary", sectionTitle: "Pay and funding" });
  s.addText("Lower teacher pay has held Leverett's costs down", { placeholder: "title" });
  const groups = [["Amherst", "Amherst"], ["Amherst-Pelham Regional", "Amherst-Pelham Reg."],
    ["Western MA", "Western MA"], ["Pelham", "Pelham"], ["Shutesbury", "Shutesbury"],
    [PEERS, "200 or fewer"], ["Leverett", "Leverett"]];
  s.addChart(pres.charts.BAR, [
    { name: `Average teacher salary, FY${L}`, labels: groups.map((g) => g[1]), values: groups.map((g) => salary(g[0])) },
  ], {
    x: 0.6, y: 1.55, w: 7.6, h: 5.2, barDir: "bar", ...axisStyle(), catAxisLabelFontSize: 12,
    chartColors: groups.map((g) => (g[0] === "Leverett" ? HEX.accent1 : (g[0] === "Western MA" || g[0] === PEERS) ? HEX.accent3 : HEX.accent2)),
    showLegend: false, valAxisHidden: true, valGridLine: { style: "none" }, valAxisMinVal: 0,
    valAxisMaxVal: Math.ceil(Math.max(...groups.map((g) => salary(g[0]))) * 1.18 / 10000) * 10000, barGapWidthPct: 45,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "$#,##0", dataLabelFontSize: 12,
    dataLabelColor: HEX.dk1, showTitle: true, title: `Average teacher salary, FY${L}`,
    titleFontSize: 14, titleColor: HEX.dk1, objectName: "teacher salary chart",
  });
  const c70 = D.c70;
  callout(s, { x: 8.7, y: 1.7, w: 4.0, value: usd(salary(PEERS) - salary("Leverett")),
    label: `Below the average teacher salary in districts with 200 or fewer students, FY${L}`, valueColor: C.accent1,
    name: "callout pay gap" });
  callout(s, { x: 8.7, y: 3.15, w: 4.0, value: `${usd(c70.Leverett[L].aid)} vs ${usd(c70[PEERS][L].aid)}`,
    label: "State Chapter 70 aid per pupil: Leverett vs districts with 200 or fewer students", name: "callout aid" });
  s.addText("The town carries more of the cost, and its educators absorb part of it in pay. The structural fix sits with the state's funding formula.", {
    x: 8.7, y: 4.6, w: 4.0, h: 1.6, fontSize: 16, italic: true, color: C.accent2, margin: 0, valign: "top",
    isTextBox: true, objectName: "takeaway" });
  notes(s, [
    `Average teacher salary is DESE's figure for each district. It reflects experience and credentials as well as pay scales, and with about ${Math.round(D.teacher_pay.Leverett.teacher_fte[L])} teachers Leverett's average moves with a few hires or retirements. FY2018 is the first year DESE reports it in this dataset.`,
    `Leverett: ${usd(D.teacher_pay.Leverett.avg_salary["2018"])} in FY2018, ${usd(salary("Leverett"))} in FY${L}. Districts its size: ${usd(D.teacher_pay[PEERS].avg_salary["2018"])} to ${usd(salary(PEERS))}. Peer and Western MA averages are weighted by teacher FTE.`,
    `Leverett funds ${usd(c70.Leverett[L].above_required)} per pupil above the state's required minimum (net school spending), against ${usd(c70[PEERS][L].above_required)} for districts its size. Chapter 70 aid and net school spending are per foundation pupil, the residents the district pays for.`,
    "Closing point from the report: structural funding challenges will persist until the Commonwealth changes policy that no region, town or district can change alone. See Rural Schools MA and the July 2022 report A Sustainable Future for Rural Schools.",
  ]);
}

// ---------- Slide 7: the town budget ----------
pres.addSection({ title: "Town budget" });
{
  const s = pres.addSlide({ masterName: "Chart and commentary", sectionTitle: "Town budget" });
  s.addText("Leverett puts more of its budget into schools", { placeholder: "title" });
  const groups = [["Leverett", "Leverett"], ["Shutesbury", "Shutesbury"], ["Pelham", "Pelham"], ["Amherst", "Amherst"],
    [TOWN_PEERS, "Towns of 1,000–2,499"], ["Western MA towns", "All Western MA towns"]];
  s.addChart(pres.charts.BAR, [
    { name: `Education's share of general fund spending, FY${L}`, labels: groups.map((g) => g[1]),
      values: groups.map((g) => TB[g[0]].education_share) },
  ], {
    x: 0.6, y: 1.55, w: 7.6, h: 4.95, barDir: "bar", ...axisStyle(), catAxisLabelFontSize: 12,
    chartColors: groups.map((g) => (g[0] === "Leverett" ? HEX.accent1 : g[0] === TOWN_PEERS || g[0] === "Western MA towns" ? HEX.accent3 : HEX.accent2)),
    showLegend: false, valAxisHidden: true, valGridLine: { style: "none" }, valAxisMinVal: 0, valAxisMaxVal: 0.8,
    barGapWidthPct: 45, showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0%", dataLabelFontSize: 12,
    dataLabelColor: HEX.dk1, showTitle: true, title: `Education's share of town general fund spending, FY${L}`,
    titleFontSize: 14, titleColor: HEX.dk1, objectName: "education share chart",
  });
  s.addText("Town budgets: state Division of Local Services (Schedule A). Population: Census Bureau.", {
    x: 0.6, y: 6.55, w: 7.6, h: 0.3, fontSize: 10, color: C.accent6, margin: 0, isTextBox: true, objectName: "town source" });
  const lev = TB.Leverett, peers = TB[TOWN_PEERS];
  callout(s, { x: 8.7, y: 1.7, w: 4.0, value: `${share(lev.education_share)} vs ${share(peers.education_share)}`,
    label: `Education's share of spending: Leverett vs the ${peers.members} Western MA towns of 1,000–2,499 residents`,
    valueColor: C.accent1, name: "callout share" });
  callout(s, { x: 8.7, y: 3.15, w: 4.0, value: `${usd(lev.spend_per_resident)} vs ${usd(peers.spend_per_resident)}`,
    label: "Total general fund spending per resident: about the same as those towns", name: "callout spending" });
  callout(s, { x: 8.7, y: 4.6, w: 4.0, value: `${usd(lev.taxes_per_resident)} vs ${usd(peers.taxes_per_resident)}`,
    label: "Taxes per resident, the same comparison", name: "callout taxes" });
  const contrib = lev.districts.map((d) => `${usd(d.contribution)} to ${d.kind === "town" ? "Leverett Elementary" : "Amherst-Pelham Regional"}`);
  notes(s, [
    `Leverett's town budget is about the size of similar towns' per resident (${usd(lev.spend_per_resident)} against ${usd(peers.spend_per_resident)}), and its residents pay about the same in taxes (${usd(lev.taxes_per_resident)} against ${usd(peers.taxes_per_resident)}). Within that budget, education takes ${share(lev.education_share)}, against ${share(peers.education_share)} for the ${peers.members} Western MA towns of 1,000–2,499 residents: ${usd(lev.education_per_resident)} per resident against ${usd(peers.education_per_resident)}.`,
    `Education in Schedule A is the town's own school department plus its assessments from regional and vocational districts. Health insurance and pensions for a town's school employees usually sit in fixed costs instead, so education's true share is if anything larger.`,
    `What the state formula requires: in FY${L}, Chapter 70 set Leverett's required local contribution at ${contrib.join(" and ")}.`,
    `Per-resident figures divide each town's FY${L} dollars by its Census population estimate (Leverett: ${Math.round(lev.population).toLocaleString("en-US")}). Every town is in the online explorer's Towns tab: timshores.com/mass-school-spending.`,
  ]);
}

// pptxgenjs writes Office's stock color scheme; scheme colors only take this deck's
// palette once theme1.xml carries it. jszip ships with pptxgenjs.
async function writeThemeColors(file, theme) {
  const JSZip = require(require.resolve("jszip", { paths: [require.resolve("pptxgenjs")] }));
  const zip = await JSZip.loadAsync(fs.readFileSync(file));
  const part = "ppt/theme/theme1.xml";
  const xml = await zip.file(part).async("string");
  const slots = ["dk1", "lt1", "dk2", "lt2", "accent1", "accent2", "accent3", "accent4",
    "accent5", "accent6", "hlink", "folHlink"];
  const scheme = `<a:clrScheme name="${theme.name}">` +
    slots.map((k) => `<a:${k}><a:srgbClr val="${theme.colors[k]}"/></a:${k}>`).join("") + "</a:clrScheme>";
  const out = xml.replace(/<a:clrScheme\b[\s\S]*?<\/a:clrScheme>/, scheme);
  if (!out.includes(scheme)) throw new Error(`${part} has no color scheme to replace`);
  zip.file(part, out);
  fs.writeFileSync(file, await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  await writeThemeColors(OUT, THEME);
  console.log(`[deck] ${OUT}`);
})();
