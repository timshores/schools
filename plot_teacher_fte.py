
import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python plot_teacher_fte.py <path_to_csv>")
    sys.exit(1)

csv = Path(sys.argv[1])
df = pd.read_csv(csv)

# Normalize column names
cols = {c.lower().strip(): c for c in df.columns}
year_col = cols.get("year") or list(df.columns)[0]
fte_col  = cols.get("teacher_fte") or list(df.columns)[1]

df = df[[year_col, fte_col]].rename(columns={year_col: "Year", fte_col: "FTE"})
df = df.dropna().astype({"Year": int, "FTE": float}).sort_values("Year")

# endpoints
start, end = 2005, 2024
sy = df.loc[(df["Year"] - start).abs().idxmin(), "Year"]
ey = df.loc[(df["Year"] - end).abs().idxmin(), "Year"]
s  = float(df.loc[df["Year"] == sy, "FTE"].iloc[0])
e  = float(df.loc[df["Year"] == ey, "FTE"].iloc[0])

years = ey - sy if ey > sy else 1
total_pct = (e / s - 1) * 100
cagr_pct  = ((e / s) ** (1 / years) - 1) * 100

fig, ax = plt.subplots(figsize=(8, 4.8))
ax.plot(df["Year"], df["FTE"], marker="o")
ax.set_xlabel("Year")
ax.set_ylabel("Teacher FTE")
ax.set_title("Teacher FTE Over Time")

annotation = f"Period: {sy}–{ey}\\nTotal change: {total_pct:+.1f}%\\nCAGR: {cagr_pct:+.2f}%/yr"
ax.text(0.02, 0.98, annotation, transform=ax.transAxes, va="top", ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="black", linewidth=0.5))
ax.grid(True, linestyle="--", alpha=0.3)
fig.tight_layout()

out = csv.with_suffix(".png")
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved chart -> {out}")
