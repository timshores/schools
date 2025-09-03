# inspect_excel.py
from pathlib import Path
import pandas as pd

EXCEL_FILE = Path.cwd() / "data" / "E2C_Hub_MA_DESE_Data.xlsx"

def main():
    if not EXCEL_FILE.exists():
        raise FileNotFoundError(f"Not found: {EXCEL_FILE}")
    xls = pd.ExcelFile(EXCEL_FILE, engine="openpyxl")
    print("\nSheets:")
    for i, s in enumerate(xls.sheet_names, 1):
        print(f"  {i}. {s}")

    print("\n--- Column previews (first 30 cols) ---")
    for s in xls.sheet_names:
        try:
            df = pd.read_excel(xls, sheet_name=s, nrows=8, engine="openpyxl")
            print(f"\n[{s}]")
            cols = list(df.columns)
            if not cols:
                print("  (no columns?)")
                continue
            for j, c in enumerate(cols[:30], 1):
                print(f"  {j:2d}. {c}")
            print("\n  Sample rows:")
            print(df.head(5).to_string(index=False))
        except Exception as e:
            print(f"\n[{s}] -> read error: {e}")

if __name__ == "__main__":
    main()
