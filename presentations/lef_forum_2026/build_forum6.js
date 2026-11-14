// Leverett Education Foundation forum, "Why does it cost so much to educate our children?", November 2026:
// Tim's part in ten slides, each a headline, one chart and a figure or two; the detail is in the speaker notes.
// Every number comes from forum6_data.json (forum6_data.py, from the explorer's data).
//   python presentations/lef_forum_2026/forum6_data.py && node presentations/lef_forum_2026/build_forum6.js
// The look is the earlier deck's (build_deck.js): Leverett Elementary's navy, leaf green and teal.

const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");

const D = JSON.parse(fs.readFileSync(path.join(__dirname, "forum6_data.json"), "utf8"));
const OUT = path.join(__dirname, "Why does school cost so much - compared with what - LEF forum 2026-11.pptx");

const F = D.first, L = D.latest, N = D.years.length - 1;
const G = D.groups;
const usd = (v) => "$" + Math.round(v).toLocaleString("en-US");
const pct = (v, d = 0) => (v >= 0 ? "+" : "−") + Math.abs(v * 100).toFixed(d) + "%";
const share = (v) => Math.round(v * 100) + "%";
const PEERS = "cohort-tiny-western";
const FT = D.four_towns;
const WL = D.wealth;
const nth = (n) => n + (n % 100 > 10 && n % 100 < 14 ? "th" : ["th", "st", "nd", "rd"][n % 10] || "th");

const THEME = { headFontFace: "Cambria", bodyFontFace: "Calibri",
  colors: { dk1: "141F3E", lt2: "DCE9EA", accent2: "9DC058", accent3: "CBE793", accent5: "035D58", accent6: "6E7681" } };
const HEX = THEME.colors;
const GRID = "E3E3E3", MUTED = "5F6670", LIGHT_GREEN = "CBE793", WHITE = "FFFFFF";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.author = "Tim Shores";
pres.title = "Why does school cost so much? Compared with what?";

const SOURCE = "Data: Massachusetts DESE, FY" + F + "\u2013FY" + L + ", after inflation by the Northeast CPI  \u00b7  timshores.com/muni-spending";
pres.defineSlideMaster({
  title: "Title dark", background: { color: HEX.dk1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.7, y: 0.55, w: 11.9, h: 2.1, fontSize: 40, bold: true,
      color: WHITE, valign: "bottom", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.7, y: 2.9, w: 11.9, h: 0.9, fontSize: 18, color: LIGHT_GREEN,
      valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "Content", background: { color: WHITE },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 1.05, fontSize: 30, bold: true,
      color: HEX.dk1, valign: "middle", align: "left", margin: 0 }, text: "" } },
    { text: { text: SOURCE, options: { x: 0.6, y: 6.95, w: 11.2, h: 0.3, fontSize: 10, color: HEX.accent6, margin: 0 } } },
  ],
  slideNumber: { x: 12.2, y: 6.95, w: 0.5, h: 0.3, fontSize: 10, color: HEX.accent6, align: "right" },
});
// The growth factor slide's figures aren't adjusted for inflation and run to the latest Chapter 70 year
const SOURCE_C70 = "Data: Massachusetts DESE, Chapter 70 formula, FY" + D.mrgf.years[0] + "\u2013FY" + D.mrgf.years[D.mrgf.years.length - 1] + "  \u00b7  timshores.com/muni-spending";
pres.defineSlideMaster({
  title: "Content c70", background: { color: WHITE },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 1.05, fontSize: 30, bold: true,
      color: HEX.dk1, valign: "middle", align: "left", margin: 0 }, text: "" } },
    { text: { text: SOURCE_C70, options: { x: 0.6, y: 6.95, w: 11.2, h: 0.3, fontSize: 10, color: HEX.accent6, margin: 0 } } },
  ],
  slideNumber: { x: 12.2, y: 6.95, w: 0.5, h: 0.3, fontSize: 10, color: HEX.accent6, align: "right" },
});
// The tax base slide's figures are the Four Towns Fiscal Study Group's, from the Division of Local Services
pres.defineSlideMaster({
  title: "Content four towns", background: { color: WHITE },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 1.05, fontSize: 30, bold: true,
      color: HEX.dk1, valign: "middle", align: "left", margin: 0 }, text: "" } },
    { text: { text: "Data: Four Towns Fiscal Study Group, Report to the Community, October 2026, from the Mass. Division of Local Services, FY2025",
      options: { x: 0.6, y: 6.95, w: 11.2, h: 0.3, fontSize: 10, color: HEX.accent6, margin: 0 } } },
  ],
  slideNumber: { x: 12.2, y: 6.95, w: 0.5, h: 0.3, fontSize: 10, color: HEX.accent6, align: "right" },
});
// The map slide has no headline (Tim, 2026-11-12), so the map can fill the slide
pres.defineSlideMaster({
  title: "No title", background: { color: WHITE },
  objects: [{ text: { text: SOURCE, options: { x: 0.6, y: 6.95, w: 11.2, h: 0.3, fontSize: 10, color: HEX.accent6, margin: 0 } } }],
  slideNumber: { x: 12.2, y: 6.95, w: 0.5, h: 0.3, fontSize: 10, color: HEX.accent6, align: "right" },
});

const chartText = { catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", legendFontFace: "+mn-lt",
  dataLabelFontFace: "+mn-lt", titleFontFace: "+mn-lt" };
const axisStyle = () => ({ ...chartText, catAxisLabelColor: MUTED, valAxisLabelColor: MUTED, catAxisLabelFontSize: 12,
  valAxisLabelFontSize: 12, valGridLine: { color: GRID, size: 0.75 }, catGridLine: { style: "none" },
  catAxisLineShow: true, catAxisLineColor: GRID, valAxisLineShow: false, legendColor: HEX.dk1, legendFontSize: 13,
  titleFontSize: 14, titleColor: HEX.dk1, showTitle: true });

function callout(slide, { x, y, w, value, label, name }) {
  slide.addText([
    { text: value, options: { fontSize: 30, bold: true, color: HEX.dk1, breakLine: true, fontFace: THEME.headFontFace } },
    { text: label, options: { fontSize: 14, color: "1F1F1C" } },
  ], { x, y, w, h: 1.2, margin: 0, valign: "top", isTextBox: true, objectName: name });
}
function takeaway(slide, text, { x, y, w, h = 1.1 }) {
  slide.addText(text, { x, y, w, h, fontSize: 16, italic: true, color: HEX.accent5, margin: 0, valign: "top",
    isTextBox: true, objectName: "takeaway" });
}
const notes = (slide, lines) => slide.addNotes(lines.join("\n\n"));
const lev = G.leverett, peers = G[PEERS], wma = G["region-western"], state = G["statewide-all"];

// The slides carry a headline, one chart and at most two figures (Tim, 2026-11-11: far fewer words and elements on
// the slides; the detail is in the speaker notes, which export as the PDF's notes pages).
const big = (slide, { x, y, w, value, label, name, color = HEX.dk1 }) => slide.addText([
  { text: value, options: { fontSize: 44, bold: true, color, breakLine: true, fontFace: THEME.headFontFace } },
  { text: label, options: { fontSize: 18, color: MUTED } },
], { x, y, w, h: 1.6, margin: 0, valign: "top", isTextBox: true, objectName: name });

// ---------- 1. The question ----------
{
  const s = pres.addSlide({ masterName: "Title dark" });
  s.addText("Why does school cost so much?\nSmart shoppers compare costs.", { placeholder: "title" });
  s.addText("Tim Shores  ·  Leverett's representative on the Amherst-Pelham Regional School Committee", { placeholder: "body" });
  // What a 1% rise is worth if prices rise as they did in the latest year, or at the twenty-year average
  const REAL1 = 1.01 / (1 + D.cpi_rate_latest) - 1, REAL1_AVG = 1.01 / (1 + D.cpi_rate_avg) - 1;
  const tiles = [
    [`${WL.income_vs_western.toFixed(1)}\u00d7`, "Leverett's income per capita, against Western MA's", "Mass. Department of Revenue"],
    [`${WL.eqv_vs_western.toFixed(1)}\u00d7`, "Leverett's property value per capita, against Western MA's", "Mass. Department of Revenue"],
    [share(FT.bill_pct_of_cpi), "Leverett's average tax bill, against inflation since 1992", "Four Towns Fiscal Study Group"],
    ["1%", "FY28 limit on Lev budgets. After inflation,", "Leverett Finance Committee guidance", `reduction of about \u2212${Math.round(-REAL1 * 100)}%`],
  ];
  tiles.forEach(([value, label, source, warn], i) => {
    const x = 0.7 + i * 3.02;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 4.45, w: 2.87, h: 2.35, rectRadius: 0.08, fill: { color: HEX.dk1 }, line: { color: "3A4666", width: 0.75 },
      objectName: `tile ${i + 1}` });
    s.addText([
      { text: value, options: { fontSize: 38, bold: true, color: WHITE, breakLine: true, fontFace: THEME.headFontFace } },
      { text: label, options: { fontSize: 14, color: HEX.lt2, breakLine: true } },
      ...(warn ? [{ text: warn, options: { fontSize: 14, color: "FFC000", breakLine: true } }] : []),
      { text: source, options: { fontSize: 10.5, color: LIGHT_GREEN } },
    ], { x: x + 0.2, y: 4.55, w: 2.47, h: 2.15, margin: 0, valign: "middle", paraSpaceAfter: 5, isTextBox: true, objectName: `figure ${i + 1}` });
  });
  notes(s, [
    "You've heard what it takes to run Leverett Elementary. My part is the view from further back: what that picture looks like when we compare it with something.",
    `Four numbers. By Western Mass standards Leverett is well-off: income per capita is ${WL.income_vs_western.toFixed(1)} times the region's, property value ${WL.eqv_vs_western.toFixed(1)} times. Against the whole state we're ordinary, ${nth(WL.income_rank_state)} of ${WL.state_towns} towns in income.`,
    `The third is from the Four Towns Fiscal Study Group's report, "The Walls are Closing In." Leverett's average single-family tax bill was ${usd(FT.bill_1992)} in 1992 and is ${usd(FT.bill_2026)} now, ${share(FT.bill_pct_of_cpi)} of what inflation alone would have made it.`,
    "The fourth I learned this week. The Finance Committee's guidance for FY28 is a 1% limit on growth. As it was described to me: largely because of health insurance, Leverett has used more than half of its excess levy capacity, and past the top of it Town Meeting would have to vote an override.",
    `And 1% is before inflation. If prices keep rising as they did in FY${L}, ${(D.cpi_rate_latest * 100).toFixed(1)}%, a budget that grows 1% buys about ${(-REAL1 * 100).toFixed(0)}% less.`,
    "So: a town with more income and property than most of its neighbors, a tax bill well ahead of inflation, and 1%. What's up with that? My answer is that it depends on what you compare with.",
  ]);
}

// Slides 2 and 3 (Tim, 2026-11-12): the enrollment trend as a dashed arrow over the spending chart. The chart's plot
// area and axis are fixed so the arrow can be placed: it starts at the height of $8,000, under both spending lines, and
// falls by the trend's percentage, so against the $0 baseline its drop is in proportion.
const TREND_GRAY = "5F6670", LINE_MAX = 35000, LINE_LAY = { x: 0.11, y: 0.1, w: 0.86, h: 0.7 }, LINE_BOX = { x: 0.6, y: 1.5, w: 8.3, h: 5.3 };
function enrollmentTrend(slide, g) {
  const n = D.years.length, px = LINE_BOX.x + LINE_LAY.x * LINE_BOX.w, pw = LINE_LAY.w * LINE_BOX.w;
  const py = LINE_BOX.y + LINE_LAY.y * LINE_BOX.h, ph = LINE_LAY.h * LINE_BOX.h;
  const yAt = (v) => py + ph * (1 - v / LINE_MAX), start = 8000, end = start * (1 + g.enroll_trend.change);
  const x1 = px + (0.5 / n) * pw, x2 = px + ((n - 0.5) / n) * pw;
  slide.addShape(pres.shapes.LINE, { x: x1, y: yAt(start), w: x2 - x1, h: yAt(end) - yAt(start), line: { color: TREND_GRAY, width: 2.5, dashType: "dash", endArrowType: "triangle" },
    objectName: "enrollment trend" });
  slide.addText("Enrollment trend", { x: x1, y: yAt(start) - 0.36, w: 2.4, h: 0.3, fontSize: 13, color: TREND_GRAY, margin: 0, isTextBox: true, objectName: "enrollment trend label" });
}
const three = (slide, rows) => rows.forEach(([value, label, color, name], i) => big(slide, { x: 9.3, y: 1.6 + i * 1.7, w: 3.6, value, label, name, color }));

// ---------- 2. Leverett against inflation ----------
{
  const s = pres.addSlide({ masterName: "Content" });
  s.addText("Leverett Elementary: most of our budget growth per pupil is due to inflation and enrollment decline", { placeholder: "title" });
  const years = D.years.map(String);
  s.addChart(pres.charts.LINE, [
    { name: "Nominal", labels: years, values: lev.ppe },
    { name: `FY${L} dollars`, labels: years, values: lev.ppe_real },
  ], { ...LINE_BOX, layout: LINE_LAY, ...axisStyle(), chartColors: [HEX.accent2, HEX.dk1], lineSize: 3.5, lineDataSymbol: "none",
    showLegend: true, legendPos: "b", legendFontSize: 15, valAxisLabelFormatCode: "$#,##0", valAxisMinVal: 0, valAxisMaxVal: LINE_MAX, valAxisMajorUnit: 5000, catAxisLabelFrequency: 5,
    catAxisLabelFontSize: 14, valAxisLabelFontSize: 14, title: "Spending per pupil", titleFontSize: 16, objectName: "Leverett nominal and real" });
  enrollmentTrend(s, lev);
  three(s, [[pct(lev.nominal_change), "Nominal dollars", HEX.accent2, "nominal"], [pct(lev.real_change), "Inflation-adjusted", HEX.dk1, "real"],
    [pct(lev.enroll_trend.change), "Enrollment trend", TREND_GRAY, "enrollment"]]);
  notes(s, [
    "Leverett Elementary, spending per pupil, the state's standard measure.",
    `The green line is nominal dollars: ${usd(lev.ppe[0])} per pupil in FY${F}, ${usd(lev.ppe[N])} in FY${L}, up ${pct(lev.nominal_change).replace("+", "")}.`,
    `The navy line is the same spending in FY${L} dollars. After inflation, Leverett is up ${pct(lev.real_change).replace("+", "")} in twenty years, about ${(lev.real_rate * 100).toFixed(1)}% a year.`,
    `Most of each year's percentage is prices going up. Enrollment is the other part, the dashed line: ${Math.round(Math.max(...lev.enroll))} pupils at the peak in FY${D.years[lev.enroll.indexOf(Math.max(...lev.enroll))]}, ${Math.round(lev.enroll_latest)} in FY${L}. On trend that's ${share(-lev.enroll_trend.change)} fewer.`,
  ]);
}

// ---------- 2b. The Region against inflation, in the Region's maroon and gold ----------
{
  const s = pres.addSlide({ masterName: "Content" });
  const ap = G["amherst-pelham"], MAROON = "7A1F2E", GOLD = "D4A72C";
  s.addText("Amherst-Pelham Regional: more growth, greater decline", { placeholder: "title" });
  const years = D.years.map(String);
  s.addChart(pres.charts.LINE, [
    { name: "Nominal", labels: years, values: ap.ppe },
    { name: `FY${L} dollars`, labels: years, values: ap.ppe_real },
  ], { ...LINE_BOX, layout: LINE_LAY, ...axisStyle(), chartColors: [GOLD, MAROON], lineSize: 3.5, lineDataSymbol: "none",
    showLegend: true, legendPos: "b", legendFontSize: 15, valAxisLabelFormatCode: "$#,##0", valAxisMinVal: 0, valAxisMaxVal: LINE_MAX, valAxisMajorUnit: 5000, catAxisLabelFrequency: 5,
    catAxisLabelFontSize: 14, valAxisLabelFontSize: 14, title: "Spending per pupil, grades 7\u201312", titleFontSize: 16, objectName: "Region nominal and real" });
  enrollmentTrend(s, ap);
  three(s, [[pct(ap.nominal_change), "Nominal dollars", GOLD, "nominal"], [pct(ap.real_change), "Inflation-adjusted", MAROON, "real"],
    [pct(ap.enroll_trend.change), "Enrollment trend", TREND_GRAY, "enrollment"]]);
  notes(s, [
    "Leverett pays for a second district: the Amherst-Pelham Region, grades 7 to 12. Same chart, in the Region's colors.",
    `Nominal, up ${pct(ap.nominal_change).replace("+", "")}. After inflation, up ${pct(ap.real_change).replace("+", "")}, about ${(ap.real_rate * 100).toFixed(1)}% a year.`,
    `The difference is enrollment. The Region had ${Math.round(ap.enroll_first).toLocaleString("en-US")} pupils in FY${F} and ${Math.round(ap.enroll_latest).toLocaleString("en-US")} in FY${L}, ${share(-ap.enroll_change)} fewer, and ${share(-ap.enroll_trend.change)} on the dashed trend line. Two school buildings don't shrink by ${share(-ap.enroll_change)}, so the cost per pupil climbs.`,
  ]);
}

// ---------- 3. Our two districts, each against districts its size ----------
{
  const s = pres.addSlide({ masterName: "Content" });
  s.addText("Comparisons with districts & groups of similar size", { placeholder: "title" });
  const MED = "cohort-medium-western";
  const bars = (keys, names, focus, x, w, title, name) => s.addChart(pres.charts.BAR, [{ name: "Growth after inflation", labels: names,
    values: keys.map((k) => Math.round(G[k].real_change * 100)) }],
  { x, y: 1.5, w, h: 4.85, barDir: "bar", barGapWidthPct: 45, ...axisStyle(), catAxisLabelFontSize: 15, catAxisOrientation: "maxMin",
    chartColors: keys.map((k) => (focus.includes(k) ? HEX.dk1 : k === PEERS || k === MED ? HEX.accent5 : HEX.accent2)), showLegend: false,
    valAxisHidden: true, valGridLine: { style: "none" }, valAxisMinVal: 0, valAxisMaxVal: 70, showValue: true, dataLabelPosition: "outEnd",
    dataLabelFormatCode: '"+"0"%"', dataLabelFontSize: 16, dataLabelColor: HEX.dk1, title, titleFontSize: 16, objectName: name });
  bars(["leverett", "pelham", "shutesbury", PEERS], ["Leverett", "Pelham", "Shutesbury", "Western MA schools their size"], ["leverett"],
    0.6, 6.0, "Elementary: Leverett and its neighbors", "elementary bars");
  bars(["amherst-pelham", MED, "region-western", "statewide-all"], ["Amherst-Pelham Regional", "Western MA districts its size", "Western MA", "Massachusetts"],
    ["amherst-pelham"], 6.8, 6.0, "Grades 7–12: Amherst-Pelham Regional", "regional bars");
  s.addText(`Growth in spending per pupil after inflation, FY${F}–FY${L}`, { x: 0.6, y: 6.55, w: 12.1, h: 0.3, fontSize: 24, bold: true, color: "7A1F2E",
    align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "measure" });
  const med = G[MED], ap = G["amherst-pelham"];
  const plain = (v) => pct(v).replace("+", "");
  notes(s, [
    "Now compare each of our districts with districts its size. Same measure, same twenty years, after inflation.",
    `Elementary: Leverett is up ${plain(lev.real_change)}, Pelham ${plain(G.pelham.real_change)}, Shutesbury ${plain(G.shutesbury.real_change)}. The ${peers.members} Western Mass districts with 200 or fewer pupils are up ${plain(peers.real_change)}. They lost ${share(-peers.enroll_change)} of their enrollment, and a small school's fixed costs don't shrink with it.`,
    `The Region is up ${plain(ap.real_change)}. So are the ${med.members} Western Mass districts its size, and Western Mass and the state are up ${plain(wma.real_change)}. The Region grew at the going rate while losing more students than its peers.`,
    "I'm not saying either spends too little or too much. The comparison you pick gives you the answer, so say which one you're using.",
  ]);
}

// ---------- 4. Zoom out: a map of the regions over their enrollment ----------
{
  const s = pres.addSlide({ masterName: "No title" });
  const M = D.map;
  const mh = 4.6, mw = (mh * M.width) / M.height, mx = 3.5, my = 0.25;
  s.addImage({ path: path.join(__dirname, "region_map.png"), x: mx, y: my, w: mw, h: mh,
    altText: "Massachusetts in six regions, shaded light to dark by yearly growth in spending per pupil after inflation, from 1.6% to 1.9%" });
  // Each region's yearly rate on the map: where its label sits, as fractions of the map. The Boston area's sits beside
  // the map; Metro West, too narrow for a label, has its below the map with a line to it.
  const at = { "region-western": [0.2, 0.3], "region-central": [0.435, 0.3], "region-northeast": [0.708, 0.1], "region-metro-west": [0.461, 0.657],
    "region-boston": [0.84, 0.31], "region-southeast": [0.715, 0.68] };
  const yearly = (k) => `${(G[k].real_rate * 100).toFixed(1)}%`;
  for (const [k, [fx, fy]] of Object.entries(at)) {
    const outside = k === "region-boston" || k === "region-metro-west";
    s.addText([
      { text: yearly(k), options: { fontSize: 20, bold: true, breakLine: true, fontFace: THEME.headFontFace } },
      { text: G[k].name.replace(" MA", ""), options: { fontSize: 11 } },
    ], { x: mx + fx * mw - 0.75, y: my + fy * mh - 0.36, w: 1.5, h: 0.72, align: "center", valign: "middle", margin: 0,
      color: outside || !M.regions[k].dark ? HEX.dk1 : WHITE, isTextBox: true, objectName: `map label ${G[k].name}` });
  }
  const [x1, y1, x2, y2] = [0.496, 0.586, 0.617, 0.466].map((f, i) => (i % 2 ? my + f * mh : mx + f * mw));
  s.addShape(pres.shapes.LINE, { x: x1, y: y2, w: x2 - x1, h: y1 - y2, flipV: true, line: { color: HEX.dk1, width: 1 }, objectName: "Metro West leader" });
  const cap = (text, y, h = 1.3, fontSize = 18) => s.addText(text, { x: 0.6, y, w: 2.7, h, fontSize, bold: true, color: HEX.dk1, fontFace: THEME.headFontFace,
    margin: 0, valign: "top", isTextBox: true, objectName: "caption" });
  cap("Spending per pupil: yearly growth after inflation", 1.09, 0.7, 20);
  cap(`Total, FY${F}\u2013FY${L}`, 5.07, 0.3);
  cap("Change in students per district", 5.72, 0.6);
  // Beneath the map, west to east as the map runs: each region's twenty-year total, then its enrollment change.
  // The chart's plot area spans its full width (layout), so each total sits over its region's column.
  const order = ["region-western", "region-central", "region-metro-west", "region-northeast", "region-boston", "region-southeast"];
  const ex = mx - 0.2, ew = mw + 0.4, colW = ew / order.length;
  order.forEach((k, i) => s.addText(pct(G[k].real_change), { shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.08, x: ex + i * colW + 0.1, y: 4.98,
    w: colW - 0.2, h: 0.45, fontSize: 17, bold: true, fill: { color: M.regions[k].color }, line: { color: M.regions[k].color },
    color: M.regions[k].dark ? WHITE : HEX.dk1, fontFace: THEME.headFontFace, align: "center", valign: "middle", margin: 0,
    objectName: `total ${G[k].name}` }));
  s.addChart(pres.charts.BAR, [{ name: "Enrollment change", labels: order.map((k) => G[k].name.replace(" MA", "")),
    values: order.map((k) => Math.round(G[k].enroll_change * 100)) }],
  { x: ex, y: 5.45, w: ew, h: 1.47, layout: { x: 0, y: 0.04, w: 1, h: 0.74 }, barDir: "col", barGapWidthPct: 45, ...axisStyle(), showTitle: false,
    catAxisLabelFontSize: 12, catAxisLabelPos: "low", chartColors: order.map((k) => (k === "region-western" ? HEX.dk1 : HEX.accent2)), showLegend: false,
    valAxisHidden: true, valGridLine: { style: "none" }, valAxisMinVal: -50, valAxisMaxVal: 30, showValue: true, dataLabelFormatCode: '0"%"',
    dataLabelFontSize: 14, dataLabelColor: HEX.dk1, objectName: "enrollment by region" });
  const rates = order.map((k) => G[k].real_rate * 100);
  const ch = order.map((k) => G[k].real_change);
  notes(s, [
    `Zoom out to the state's six regions. The number on each is how fast spending per pupil grew each year, after inflation.`,
    `Read the numbers, not the shading: ${Math.min(...rates).toFixed(1)}% to ${Math.max(...rates).toFixed(1)}% a year, everywhere, rich or poor. Schools are people and their health insurance, and those costs rise faster than prices in general. No town sets that rate.`,
    `What differs is enrollment, at the bottom. The average Western Mass district has ${share(-wma.enroll_change)} fewer students than in FY${F}. The Boston area has ${share(G["region-boston"].enroll_change)} more.`,
    "School costs rise close to 2% a year above inflation in every region. The FY28 guidance is 1%, before inflation. In one town's budget that looks like a local problem. In the pooled data it looks like weather.",
  ]);
}

// ---------- 5. State aid ----------
{
  const s = pres.addSlide({ masterName: "Content" });
  s.addText("State aid can mitigate the risk of school cost growth\nSome groups have fallen behind", { placeholder: "title" });
  const GATE = "cohort-gateway", gate = G[GATE];
  const RURAL = "type-rural", rural = G[RURAL];
  // Most state aid first, by the latest year's share
  const subjects = [["leverett", "Leverett"], [RURAL, "Rural schools"], [GATE, "Gateway cities"], ["statewide-all", "Massachusetts"]]
    .sort((a, b) => G[b[0]].nss_shares[String(L)].ch70_aid - G[a[0]].nss_shares[String(L)].ch70_aid);
  const cols = subjects.flatMap(([k]) => [[k, F], [k, L]]);
  const lab = cols.map(([, y]) => String(y));
  const part = (key) => cols.map(([k, y]) => Math.round(G[k].nss_shares[String(y)][key] * 100));
  // A shaded band behind each subject's pair of bars, its name over it. The chart's plot area is fixed (layout), so
  // the bands can be set under its six bars.
  const cx = 0.6, cy = 1.5, cw = 12.1, chh = 5.3, lay = { x: 0.03, y: 0.12, w: 0.94, h: 0.7 };
  const px = cx + lay.x * cw, pw = lay.w * cw, bandW = pw / subjects.length;
  subjects.forEach(([, name], i) => {
    s.addShape(pres.shapes.RECTANGLE, { x: px + i * bandW + 0.06, y: cy + 0.05, w: bandW - 0.12, h: chh - 0.75, fill: { color: i % 2 ? "EEF1F4" : HEX.lt2 },
      line: { color: i % 2 ? "EEF1F4" : HEX.lt2 }, objectName: `band ${name}` });
    s.addText(name, { x: px + i * bandW + 0.06, y: cy + 0.1, w: bandW - 0.12, h: 0.45, fontSize: 16, bold: true, color: HEX.dk1, align: "center",
      valign: "middle", margin: 0, fontFace: THEME.headFontFace, isTextBox: true, objectName: `band label ${name}` });
  });
  s.addChart(pres.charts.BAR, [
    { name: "State aid (Chapter 70)", labels: lab, values: part("ch70_aid") },
    { name: "Local, required by the state", labels: lab, values: part("required_local") },
    { name: "Local, above the requirement", labels: lab, values: part("above_required") },
  ], { x: cx, y: cy, w: cw, h: chh, layout: lay, barDir: "col", barGrouping: "percentStacked", barGapWidthPct: 45, ...axisStyle(), showTitle: false,
    chartColors: [HEX.accent2, "7F8A99", HEX.dk1], showLegend: true, legendPos: "b", legendFontSize: 18, valAxisHidden: true,
    valGridLine: { style: "none" }, catAxisLabelFontSize: 15, showValue: true, dataLabelFormatCode: '0"%"', dataLabelFontSize: 16,
    dataLabelColor: WHITE, objectName: "who pays" });
  const C = D.cap;
  const sh = (g, y, k) => share(g.nss_shares[String(y)][k]);
  notes(s, [
    "State aid is a kind of insurance: a cost rises for everyone, so you pool it across the state. Chapter 70 is ours.",
    `The chart is who pays for school spending in FY${F} and FY${L}, with the most state aid on the left. Green is state aid. Gray is what the state requires of the town. Navy is what the town spends above that, on its own property tax.`,
    `Gateway cities: Chapter 70 pays ${sh(gate, L, "ch70_aid")}, up from ${sh(gate, F, "ch70_aid")}. That's the formula working as designed, and the 2019 Student Opportunity Act sent its new money there. I don't begrudge them a dollar. It shows the formula can share risk.`,
    `The state as a whole: ${sh(state, L, "ch70_aid")}. The ${rural.members} rural districts: ${sh(rural, L, "ch70_aid")}, down from ${sh(rural, F, "ch70_aid")}. Leverett: ${sh(lev, L, "ch70_aid")}.`,
    `Now the navy. In FY${F}, ${sh(lev, F, "above_required")} of Leverett's school spending was above what the formula required. Now it's ${sh(lev, L, "above_required")}. Rural schools went from ${sh(rural, F, "above_required")} to ${sh(rural, L, "above_required")}, the state from ${sh(state, F, "above_required")} to ${sh(state, L, "above_required")}. The formula's idea of what schooling costs has fallen behind, and aid doesn't follow spending the formula doesn't see.`,
    "These are my readings of state data, not a position of the Regional School Committee.",
  ]);
}

// ---------- 5b. The municipal revenue growth factor ----------
{
  const s = pres.addSlide({ masterName: "Content c70" });
  const R = D.mrgf, MAROON = "7A1F2E", GRAY = "7F8A99";
  const p1 = (v) => (v * 100).toFixed(1) + "%";
  const yy = R.years.map(String), y0 = R.years[0], y1 = R.years[R.years.length - 1];
  const ls = R.steps.leverett, as = R.steps.amherst, a = ls[String(y1)], b = ls[String(y1 - 1)], am = as[String(y1)];
  s.addText(`Chapter 70 assumes Leverett revenue grows ~${p1(R.avg.leverett)} a year`, { placeholder: "title" });
  s.addText("The municipal revenue growth factor: the state's yearly estimate of how fast a town's own revenue grows. Chapter 70 raises the town's required school contribution by it.",
    { x: 0.6, y: 1.45, w: 12.1, h: 0.8, fontSize: 18, color: MUTED, margin: 0, valign: "top", isTextBox: true, objectName: "what the factor is" });
  const avgOf = (k) => R[k].reduce((t, v) => t + v, 0) / R[k].length;
  const rows = [["Leverett", R.avg.leverett, HEX.dk1], ["Amherst", R.avg.amherst, MAROON], ["Pelham", avgOf("pelham"), GRAY], ["Shutesbury", avgOf("shutesbury"), GRAY],
    ["The median Western MA town", R.avg.western_median, GRAY], ["The median Massachusetts town", R.avg.median, GRAY],
    ["Leverett's budget limit for FY28", 0.01, HEX.accent2]];
  // pptxgenjs draws a horizontal bar chart's first category at the bottom, so the rows go in reversed
  const rev = rows.slice().reverse();
  s.addChart(pres.charts.BAR, [{ name: "Growth a year", labels: rev.map((r) => r[0]), values: rev.map((r) => Math.round(r[1] * 1000) / 10) }],
    { x: 0.6, y: 2.35, w: 12.1, h: 4.5, barDir: "bar", barGapWidthPct: 40, ...axisStyle(), showTitle: true, title: `Average growth a year, FY${y0}–FY${y1}`, titleFontSize: 16,
      chartColors: rev.map((r) => r[2]), showLegend: false, valAxisHidden: true, valGridLine: { style: "none" }, valAxisMinVal: 0, valAxisMaxVal: 4.5,
      catAxisLabelFontSize: 16, catAxisLabelColor: HEX.dk1, showValue: true, dataLabelFormatCode: '0.0"%"', dataLabelFontSize: 18, dataLabelColor: HEX.dk1,
      dataLabelPosition: "outEnd", objectName: "growth factors" });
  notes(s, [
    "One more number from the state's formula: the municipal revenue growth factor. Each year the Department of Revenue estimates how fast each town's own revenue grows: the 2.5% of Proposition 2½, new growth, unrestricted state aid and local receipts.",
    `Leverett averages ${p1(R.avg.leverett)} a year since FY${y0}, about the same as the median Massachusetts town. Amherst averages ${p1(R.avg.amherst)}. The green bar is the 1% guidance, for scale.`,
    "Chapter 70 raises each town's required school contribution by this factor every year, then adjusts it toward the town's target. It's also the state's yardstick for effort: a town whose school spending grows more slowly than its factor is giving schools a shrinking share of its revenue.",
    `So the squeeze: the formula assumes our revenue grows almost ${Math.round(R.avg.leverett * 100)}% a year. School costs rise about 2% above inflation. The guidance is 1%. The factor counts revenue the town is allowed to raise, whether or not health insurance takes it first.`,
    "[Check before Saturday: whether the Finance Committee's 1% applies to the school assessment as well as town departments.]",
  ]);
}

// ---------- 5c. The tax base: homes ----------
{
  const s = pres.addSlide({ masterName: "Content four towns" });
  const T = D.tax_base, GRAY = "7F8A99";
  s.addText("In our four towns, property tax is just housing", { placeholder: "title" });
  s.addChart(pres.charts.BAR, [{ name: "Commercial and industrial share of the levy", labels: T.ci_share.map((r) => r[0]), values: T.ci_share.map((r) => Math.round(r[1] * 1000) / 10) }],
    { x: 0.6, y: 1.5, w: 8.3, h: 5.3, barDir: "bar", barGapWidthPct: 40, catAxisOrientation: "maxMin", ...axisStyle(), showTitle: true,
      title: "Commercial and industrial share of the tax levy, FY25", titleFontSize: 16, chartColors: T.ci_share.map((r) => (T.ours.includes(r[0]) ? HEX.dk1 : GRAY)),
      showLegend: false, valAxisHidden: true, valGridLine: { style: "none" }, valAxisMinVal: 0, valAxisMaxVal: 32, catAxisLabelFontSize: 15, catAxisLabelColor: HEX.dk1,
      showValue: true, dataLabelFormatCode: '0.0"%"', dataLabelFontSize: 16, dataLabelColor: HEX.dk1, dataLabelPosition: "outEnd", objectName: "commercial and industrial share" });
  const r = T.residential;
  s.addText([
    { text: "Residential share of the levy", options: { fontSize: 18, color: MUTED, breakLine: true } },
    ...T.ours.map((n, k) => ({ text: `${share(r[n])}  ${n}`, options: { fontSize: 26, bold: true, color: HEX.dk1, fontFace: THEME.headFontFace, breakLine: true } })),
  ], { x: 9.3, y: 1.7, w: 3.6, h: 2.6, margin: 0, valign: "top", paraSpaceAfter: 4, isTextBox: true, objectName: "residential shares" });
  s.addText([
    { text: share(T.amherst_exempt), options: { fontSize: 36, bold: true, color: "7A1F2E", fontFace: THEME.headFontFace, breakLine: true } },
    { text: "of Amherst's real property value is tax exempt", options: { fontSize: 16, color: MUTED } },
  ], { x: 9.3, y: 4.75, w: 3.6, h: 1.5, margin: 0, valign: "top", isTextBox: true, objectName: "Amherst exempt" });
  notes(s, [
    "This is from the Four Towns report, and it explains the kind of revenue growth our towns get.",
    `In Leverett and Amherst, ${share(r.Leverett)} of the property tax falls on homes. In Pelham ${share(r.Pelham)}, in Shutesbury ${share(r.Shutesbury)}. Commercial and industrial property pays about 1% of the levy in the three small towns and 6% in Amherst. In Hadley it pays 27%.`,
    `A town with stores and industry gets new growth that raises its levy limit with no vote, and its homeowners carry less of each increase. We have little of that. The report says the chance for it doesn't exist in Leverett, Pelham or Shutesbury, and ${share(T.amherst_exempt)} of Amherst's real property value is tax exempt, most of it the colleges'.`,
    "So when the state's formula assumes our revenue grows almost 4% a year, here that growth comes from homeowners' bills.",
  ]);
}

// ---------- 6. Where to push ----------
{
  const s = pres.addSlide({ masterName: "Content" });
  s.addText("Six systems-level leverage points?", { placeholder: "title" });
  const C0 = D.cap, round50 = (n) => Math.round(n / 50) * 50;
  // Three at the state level on the left and three at the local level on the right, each side in its own color (Tim,
  // 2026-11-12). Each: its name, the ask in lines (a middle part of the first line in bold), and the bill or who decides.
  const sides = [
    ["State level", HEX.lt2, [
      ["The 82.5% cap in Ch. 70 target local contribution", [[`The wealthiest ${round50(C0.lowered)} towns receive `, `$${(C0.shifted / 1e9).toFixed(1)}B in effective aid`,
        ` — that could be redistributed to the ${round50(C0.raised)} other towns`]], ""],
      ["Rebalance state aid", [["Rural aid in full; Ch. 70 aid for fixed costs, less focus on head counts"]], "S.394  ·  H.517 / S.314, the rural schools bill"],
      ["Charter schools", [["Reimburse the towns that send students"]], "HD.4439"],
    ]],
    ["Local level", "E9F3D3", [
      ["Capital improvements that cut operating costs", [["A district and school budget strategy:"], ["invest once to spend less every year"]], "School committees and towns"],
      ["The ARPS regional agreement", [["Amend it to right-size the regional district to our four towns"]],
        "Regional School Committee design and approval, then a vote in each of the four towns, then approval by the state Commissioner"],
      ["Commercial and industrial tax base diversity", [["Plan zoning changes and water and sewer infrastructure that create opportunities for it"]], "Planning boards, selectboards and town meetings"],
    ]],
  ];
  sides.forEach(([side, fill, levers], c) => {
    const x = 0.6 + c * 6.15;
    s.addText(side, { x, y: 1.4, w: 5.95, h: 0.35, fontSize: 16, bold: true, color: c ? "4E6B1F" : HEX.accent5, margin: 0, valign: "middle", isTextBox: true,
      objectName: `${side} heading` });
    levers.forEach(([head, ask, bill], r) => {
      const y = 1.85 + r * 1.66, n = c * 3 + r + 1;
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 5.95, h: 1.54, rectRadius: 0.1, fill: { color: fill }, line: { color: fill }, objectName: `lever card ${n}` });
      const runs = ask.flatMap((line, li) => line.map((t, k) => ({ text: t, options: { fontSize: 15, bold: line.length > 1 && k === 1, color: c ? "35520F" : HEX.accent5,
        breakLine: k === line.length - 1 && (li < ask.length - 1 || !!bill) } })));
      s.addText([
        { text: head, options: { fontSize: 20, bold: true, color: HEX.dk1, fontFace: THEME.headFontFace, breakLine: true } },
        ...runs,
        ...(bill ? [{ text: bill, options: { fontSize: 12, color: MUTED } }] : []),
      ], { x: x + 0.3, y: y + 0.07, w: 5.35, h: 1.4, margin: 0, valign: "middle", paraSpaceAfter: 3, isTextBox: true, objectName: `lever ${n}` });
    });
  });
  const C = D.cap;
  notes(s, [
    "Six places where the system could change: three in state law on the left, three of our own on the right.",
    `The cap. No town's target contribution can exceed 82.5% of its foundation budget. That lowers the targets of ${C.lowered} of the wealthiest towns by $${(C.shifted / 1e9).toFixed(1)} billion a year and raises the other ${C.raised} towns' to cover it. Leverett's is ${usd(-C.leverett_target_change)} higher. I haven't found a bill that changes it.`,
    "Rebalance state aid. Rural school aid at the $60 million the Legislature's own commission recommended, and a formula that pays for a small school's fixed costs with less focus on head counts. That's the rural schools bill, House 517 and Senate 314, filed by Natalie Blais and Jo Comerford, which has a declining enrollment fund. Senate 394 would raise the state's share of the foundation budget.",
    "Charter schools. The town pays the tuition and the state reimburses the increase for three years. House docket 4439 would have the state carry more of it.",
    "Now ours. Capital improvements that cut operating costs: spend once on a building, a roof or a heating system and spend less every year after.",
    "The regional agreement. Amending it could right-size the Region for our four towns. The Regional School Committee designs and approves an amendment, each of the four towns votes on it, and the state Commissioner of Elementary and Secondary Education approves it. The review starts this winter.",
    "And the tax base from the last slide. Zoning changes and water and sewer planning are how a town makes room for commercial and industrial property. It's slow, and in our three small towns the room is limited, but it's the one of the six that changes who pays.",
    "The legislative session ends in January, and whatever hasn't passed must be filed again. This winter is the time to ask.",
    "[Check before Saturday: whether S.2870 or HD.4439 has moved; what to say about a specific capital project and about the scope of the regional agreement review.]",
  ]);
}

// ---------- 7. The people working on it ----------
{
  const s = pres.addSlide({ masterName: "Content" });
  s.addText("People working to push those leverage points", { placeholder: "title" });
  const people = [
    ["Our school administrators and committee members", "Budgeting, line by line, in search of cost efficiencies year in and year out"],
    ["Leverett Education Foundation", "Knowledge-sharing (like today)\nCharitable giving provides for students when taxes fail"],
    ["Rural Schools MA", "ruralschoolsma.org"],
    ["MASS Promise to Invest", "The 82.5% cap campaign\nmasspromise2invest.org"],
    ["Four Towns Fiscal Study Group", "“The walls are closing in … we believe the first step is to clearly understand the challenges we face”"],
    ["Our legislators", "Sen. Jo Comerford  \u00b7  Rep. Mindy Domb\nRep.-elect Lora Wondolowski"],
  ];
  people.forEach(([name, what], i) => {
    const x = 0.6 + (i % 3) * 4.1, y = 1.6 + Math.floor(i / 3) * 1.95;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 3.85, h: 1.75, rectRadius: 0.1, fill: { color: HEX.lt2 }, line: { color: HEX.lt2 },
      objectName: `people card ${i + 1}` });
    s.addText([
      { text: name, options: { fontSize: 20, bold: true, color: HEX.dk1, fontFace: THEME.headFontFace, breakLine: true } },
      { text: what, options: { fontSize: 15, color: HEX.accent5 } },
    ], { x: x + 0.25, y: y + 0.12, w: 3.35, h: 1.5, margin: 0, valign: "middle", paraSpaceAfter: 5, isTextBox: true, objectName: `people ${i + 1}` });
  });
  s.addText([
    { text: "Smart shoppers compare costs, explore comparisons for yourself:", options: { color: MUTED, fontSize: 20, breakLine: true } },
    { text: "timshores.com/muni-spending", options: { bold: true, color: HEX.accent5, fontSize: 30, fontFace: THEME.headFontFace } },
  ], { x: 0.6, y: 5.65, w: 12.1, h: 0.9, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "explorer" });
  notes(s, [
    "Who is already working on this.",
    "Our school administrators and committee members, who work the budget line by line inside limits they didn't set. The Leverett Education Foundation, which put tonight together and pays for what the budget can't reach.",
    "Rural Schools MA keeps the rural schools bill in front of the Legislature. MASS Promise to Invest is the 82.5% cap campaign; Michael Naughton of Montague worked out what the cap costs, town by town, and my figures began with his.",
    "The Four Towns Fiscal Study Group, with Patricia Duffy and Steve Weiss for Leverett. Read their report in full.",
    "Our legislators: Senator Jo Comerford, Representative Mindy Domb, and Lora Wondolowski, elected last week to Natalie Blais's seat. She takes office in January, when these bills have to be filed again.",
    "And the site: every figure I've shown, for every district and all 351 towns. Please check my work, and use it.",
    "Three asks for budget season. Compare before you judge. Plan for the rate: school costs rise about " + (state.real_rate * 100).toFixed(1) + "% a year above inflation, and a 1% limit doesn't change that; it decides what gives way. And push on how the risk is shared, in Boston and here.",
    "This is my view, not a committee position. If asked about AI: I designed the analysis and am responsible for it; Claude Code helped write the code and draft the slides.",
    "[Check before Saturday: that Leverett's state senator is Jo Comerford; Mindy Domb's district; how the Leverett Education Foundation describes its own work.]",
  ]);
}

pres.writeFile({ fileName: OUT }).then((f) => console.log("wrote", f));
