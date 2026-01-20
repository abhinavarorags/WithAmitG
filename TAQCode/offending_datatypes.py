#offending_datatypes.py
# find_schema_mismatch.py — identify which parquet chunk(s) have mismatched dtypes (e.g., SIZE Int32 vs Int64)

from pathlib import Path
import polars as pl
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--TRADES_DIR")
args = parser.parse_args()
TRADES_DIR = Path(args.TRADES_DIR)

files = sorted(TRADES_DIR.glob("*.parquet"))
print(f"Found {len(files)} parquet files in {TRADES_DIR}")

# Track dtypes per column across files
col_types = {}   # col -> {dtype_str -> [filenames]}
bad_files = []   # (file, err)

for i, f in enumerate(files, 1):
    try:
        # reads ONLY schema/metadata (fast) — no full data load
        sch = pl.read_parquet_schema(str(f))  # dict[col -> polars dtype]
        for c, dt in sch.items():
            col_types.setdefault(c, {}).setdefault(str(dt), []).append(f.name)
    except Exception as e:
        bad_files.append((f.name, str(e)))

    if i % 200 == 0 or i == len(files):
        print(f"  scanned schemas: {i}/{len(files)}")

print("\n=== Columns with >1 dtype across files ===")
mismatched_cols = []
for c, dmap in col_types.items():
    if len(dmap) > 1:
        mismatched_cols.append(c)
        print(f"\nCOLUMN: {c}")
        for dt, fnames in dmap.items():
            print(f"  dtype {dt}: {len(fnames)} files (e.g. {fnames[0]})")

# Focus on SIZE specifically (your error)
print("\n=== SIZE dtype breakdown ===")
if "SIZE" in col_types:
    for dt, fnames in col_types["SIZE"].items():
        print(f"SIZE dtype {dt}: {len(fnames)} files")
        print("  examples:", fnames[:10])
else:
    print("No column named SIZE found in schemas (check case).")

# Print exact offenders: files where SIZE is Int32 vs Int64
def offenders(col="SIZE", want=None):
    if col not in col_types:
        return []
    out = []
    for dt, fnames in col_types[col].items():
        if want is None or dt == want:
            out.extend(fnames)
    return out

print("\n=== Offending files (SIZE != Int64) ===")
if "SIZE" in col_types:
    for dt, fnames in col_types["SIZE"].items():
        if dt != "Int64":
            print(f"\nFiles where SIZE is {dt}:")
            for n in fnames[:50]:
                print(" ", n)
            if len(fnames) > 50:
                print(f"  ... and {len(fnames)-50} more")
else:
    print("No SIZE column found.")

if bad_files:
    print("\n=== Files whose schema could not be read ===")
    for name, err in bad_files[:50]:
        print(f"  {name}: {err}")
    if len(bad_files) > 50:
        print(f"  ... and {len(bad_files)-50} more")
