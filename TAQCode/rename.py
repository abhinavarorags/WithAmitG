# rename.py — Rename trade columns to UPPERCASE (schema preserved)
import polars as pl
from pathlib import Path
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--TRADES_DIR", required=True)
parser.add_argument("--OUT_DIR", required=True)
args = parser.parse_args()

TRADES_DIR = Path(args.TRADES_DIR)
OUT_DIR = Path(args.OUT_DIR)
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"🔁 Renaming columns to UPPERCASE")
print(f"   Source: {TRADES_DIR}")
print(f"   Output: {OUT_DIR}")

for file in tqdm(sorted(TRADES_DIR.glob("*.parquet")), desc="Renaming"):
    try:
        # Lazy scan preserves schema exactly
        lf = pl.scan_parquet(file)

        # Rename columns ONLY (no casts, no computation)
        lf = lf.rename({c: c.upper() for c in lf.columns})

        # Write back without touching dtypes
        out_file = OUT_DIR / file.name
        lf.collect(engine="streaming").write_parquet(
            out_file,
            compression="zstd",
            statistics=True
        )

    except Exception as e:
        print(f"⚠️ Skipped {file.name}: {e}")

print("✅ Done. Column names updated, dtypes unchanged.")
