"""The six regions as a map for slide 4 of the forum deck: each town filled with its region's shade, by the
region's yearly growth in spending per pupil after inflation (compound, FY2005 to FY2025), on a scale stretched across the
regions' own range (1.5% to 1.95% a year) so the regions can be told apart; the printed rates carry the point that
they are close. Each region's color is written to forum6_data.json for the slide's row of totals. Writes region_map.svg and adds each region's label position to forum6_data.json.

    python presentations/lef_forum_2026/forum6_data.py && python presentations/lef_forum_2026/forum6_map.py
    node presentations/lef_forum_2026/forum6_map.mjs      (rasterizes the SVG to region_map.png)

Outlines and regions are the explorer's (mywebsite/src/data/massppe/map.json; Census cartographic boundaries).
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SITE = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/projects/mywebsite/src/data/massppe")
m = json.load(open(os.path.join(SITE, "map.json")))
D = json.load(open(os.path.join(HERE, "forum6_data.json")))
W, H = m["meta"]["width"], m["meta"]["height"]
region_of = next(g for g in m["groupings"]["towns"] if g["key"] == "region")["of"]

LIGHT, DARK = (0xC2, 0xDE, 0xDB), (0x02, 0x3F, 0x3C)  # a pale teal to a teal deeper than the deck's


LOW, TOP = 0.015, 0.0195


def shade(v):
    f = max(0.0, min(1.0, (v - LOW) / (TOP - LOW)))
    return "#%02x%02x%02x" % tuple(round(a + (b - a) * f) for a, b in zip(LIGHT, DARK))


def points(d):
    """A path's points: one M, then relative l steps (the explorer's outlines use only M, l and z)"""
    out, x, y = [], 0.0, 0.0
    for cmd, a, b in re.findall(r"([Ml])(-?[\d.]+) (-?[\d.]+)", d):
        x, y = (float(a), float(b)) if cmd == "M" else (x + float(a), y + float(b))
        out.append((x, y))
    return out


by_region, boxes = {}, {}
for t in m["towns"]:
    rid = region_of[t["id"]].replace("towns-", "")
    by_region.setdefault(rid, []).append(t["d"])
    for x, y in points(t["d"]):
        b = boxes.setdefault(rid, [x, y, x, y, 0.0, 0.0, 0])
        b[0], b[1], b[2], b[3] = min(b[0], x), min(b[1], y), max(b[2], x), max(b[3], y)
        b[4] += x; b[5] += y; b[6] += 1
# A white line where two regions meet: each region in turn lays a wide white outline under its towns, then its
# towns in its own shade. The white that reaches past a region's edge stays on the regions drawn before it.
parts = []
for rid, ds in by_region.items():
    c = shade(D["groups"][rid]["real_rate"])
    parts += [f'<path d="{d}" fill="none" stroke="#ffffff" stroke-width="9" stroke-linejoin="round"/>' for d in ds]
    parts += [f'<path d="{d}" fill="{c}" stroke="{c}" stroke-width="1.2" stroke-linejoin="round"/>' for d in ds]
svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">' + "".join(parts) + "</svg>"
open(os.path.join(HERE, "region_map.svg"), "w").write(svg)
D["map"] = {"width": W, "height": H,
            "regions": {rid: {"x": b[4] / b[6] / W, "y": b[5] / b[6] / H, "left": b[0] / W, "right": b[2] / W,
                              "color": shade(D["groups"][rid]["real_rate"])[1:].upper(),
                              "dark": (D["groups"][rid]["real_rate"] - LOW) / (TOP - LOW) > 0.45} for rid, b in boxes.items()}}
json.dump(D, open(os.path.join(HERE, "forum6_data.json"), "w"), indent=1)
for rid, r in D["map"]["regions"].items():
    print(f"{rid:20s} label at {r['x']:.2f}, {r['y']:.2f}   {D['groups'][rid]['real_rate']:+.2%} a year  {shade(D['groups'][rid]['real_rate'])}")
