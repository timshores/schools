"""Checks of the explorer data export_explorer_data.py writes (#237).

  python check_explorer_data.py shape SITE            the JSON's shape: no NaN or infinity anywhere; every series as
                                                      long as the years it runs over; every dict of series with the
                                                      same keys in every district
  python check_explorer_data.py diff BEFORE AFTER     what differs between two site roots' data: each JSON file as
                                                      data (meta.built dropped), each CSV as bytes; --show N paths

SITE, BEFORE and AFTER are site roots, as the export takes: mywebsite, or a scratch directory. To check a change
to the export, write it to a scratch directory and compare with the committed data:

  python export_explorer_data.py /tmp/site && python check_explorer_data.py shape /tmp/site \\
      && python check_explorer_data.py diff ~/projects/mywebsite /tmp/site

Each exits 1 on a problem or a difference.
"""
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

JSON = Path("src/data/massppe")
CSV = Path("public/muni-spending/data")


def series(x, path=()):
    """(path, list) for every list of numbers and nulls in x"""
    if isinstance(x, dict):
        for k, v in x.items():
            yield from series(v, path + (k,))
    elif isinstance(x, list):
        if x and all(v is None or isinstance(v, (int, float)) and not isinstance(v, bool) for v in x):
            yield path, x
        else:
            for v in x:
                yield from series(v, path)


def numbers(x, path=()):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from numbers(v, path + (k,))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from numbers(v, path + (i,))
    elif isinstance(x, float):
        yield path, x


def shape(site):
    root = site / JSON
    problems = []
    meta = json.loads((root / "explorer.json").read_text())["meta"]
    n = len(meta["years"])
    hist = meta["history"]
    nh = len(hist["years"])
    nday = hist["day"][1] - hist["day"][0] + 1

    for f in sorted(root.rglob("*.json")):
        for path, v in numbers(json.loads(f.read_text())):
            if not math.isfinite(v):
                problems.append(f"{f.relative_to(root)} {'.'.join(map(str, path))}: {v}")

    def lengths(files, want, label):
        bad = Counter()
        for f in files:
            for path, s in series(json.loads(f.read_text())):
                ok = want(path)
                if ok is not None and len(s) != ok:
                    bad[f"{label} {'.'.join(p for p in path if isinstance(p, str))}: {len(s)} long, not {ok}"] += 1
        problems.extend(f"{k} ({c}x)" for k, c in bad.items())

    # The explorer's own years, FY2005 on
    lengths(sorted(root.glob("districts/*.json")) + [root / "commonwealth.json", root / "cap.json"], lambda p: n, "districts")
    # A town's votes are [held, won] pairs
    lengths(sorted(root.glob("towns/*.json")), lambda p: 2 if "votes" in p else n, "towns")
    # The earlier years, FY1990-FY2004; the day-program series run to FY2006; a former district's record its own years
    lengths(sorted(root.glob("districts-history/*.json")),
            lambda p: n if "main" in p else nday if p and p[-1] in ("day_memb",) or "day" in p else nh, "history")

    # Each dict of series has the same keys in every district and average
    keysets = defaultdict(Counter)
    for f in sorted(root.glob("districts/*.json")):
        for e in json.loads(f.read_text())["entities"]:
            for k, v in e.items():
                if isinstance(v, dict) and v and all(isinstance(x, list) for x in v.values()):
                    keysets[k][tuple(sorted(v))] += 1
    for k, sets in keysets.items():
        if len(sets) > 1:
            problems.append(f"districts {k}: {len(sets)} different sets of keys: {[list(s) for s in sets]}")

    for p in problems[:40]:
        print(f"  {p}")
    print(f"[shape] {len(problems)} problem(s)")
    return 1 if problems else 0


def walk(a, b, path, out, limit):
    if len(out) >= limit:
        return
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in a:
                out.append(f"{path}.{k}: added {short(b[k])}")
            elif k not in b:
                out.append(f"{path}.{k}: removed {short(a[k])}")
            else:
                walk(a[k], b[k], f"{path}.{k}", out, limit)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} -> {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]", out, limit)
    elif a != b:
        out.append(f"{path}: {short(a)} -> {short(b)}")


def short(v):
    s = json.dumps(v)
    return s if len(s) <= 80 else s[:77] + "..."


def diff(a, b, show):
    n = 0
    names = sorted({p.relative_to(a / JSON) for p in (a / JSON).rglob("*.json")} |
                   {p.relative_to(b / JSON) for p in (b / JSON).rglob("*.json")})
    for f in names:
        pa, pb = a / JSON / f, b / JSON / f
        if not pa.exists() or not pb.exists():
            print(f"only in {'after' if pb.exists() else 'before'}: {f}")
            n += 1
            continue
        x, y = json.loads(pa.read_text()), json.loads(pb.read_text())
        for d in (x, y):
            if isinstance(d, dict) and isinstance(d.get("meta"), dict):
                d["meta"].pop("built", None)
        if x != y:
            out = []
            walk(x, y, "", out, show)
            print(f"differs: {f}")
            for line in out:
                print(f"    {line}")
            n += 1
    for f in sorted({p.name for p in (a / CSV).glob("*")} | {p.name for p in (b / CSV).glob("*")}):
        pa, pb = a / CSV / f, b / CSV / f
        if not pa.exists() or not pb.exists() or pa.read_bytes() != pb.read_bytes():
            print(f"differs: {CSV / f}")
            n += 1
    print(f"[diff] {n} file(s) differ")
    return 1 if n else 0


if __name__ == "__main__":
    args = sys.argv[1:]
    show = int(args[args.index("--show") + 1]) if "--show" in args else 6
    if len(args) >= 2 and args[0] == "shape":
        sys.exit(shape(Path(args[1]).expanduser()))
    if len(args) >= 3 and args[0] == "diff":
        sys.exit(diff(Path(args[1]).expanduser(), Path(args[2]).expanduser(), show))
    print(__doc__)
    sys.exit(2)
