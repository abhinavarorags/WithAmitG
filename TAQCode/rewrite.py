#rewrite.py
#BUG ERROR somehow duplicating rows

#!/usr/bin/env python3
from pathlib import Path
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# -------------------- CONFIG --------------------
IN_DIR  = Path("/home/amazon/Documents/TAQData/2024_03_15/processed_output_trades_upper/")
OUT_DIR = Path("/home/amazon/Documents/TAQData/2024_03_15/processed_output_trades_upper_clean/")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------- TARGET SCHEMAS --------------------
TARGET_PL = {
    "DATE": pl.Date,
    "TIME_M": pl.Time,
    "EX": pl.Utf8,
    "SYM_ROOT": pl.Utf8,
    "SYM_SUFFIX": pl.Utf8,
    "TR_SCOND": pl.Utf8,
    "SIZE": pl.Int64,
    "PRICE": pl.Float64,
    "TR_STOP_IND": pl.Utf8,
    "TR_CORR": pl.Utf8,
    "TR_SEQNUM": pl.Int64,
    "TR_ID": pl.Int64,
    "TR_SOURCE": pl.Utf8,
    "TR_RF": pl.Utf8,
}

TARGET_PA = pa.schema([
    pa.field("DATE", pa.date32()),
    pa.field("TIME_M", pa.time64("ns")),
    pa.field("EX", pa.string()),
    pa.field("SYM_ROOT", pa.string()),
    pa.field("SYM_SUFFIX", pa.string()),
    pa.field("TR_SCOND", pa.string()),
    pa.field("SIZE", pa.int64()),
    pa.field("PRICE", pa.float64()),
    pa.field("TR_STOP_IND", pa.string()),
    pa.field("TR_CORR", pa.string()),
    pa.field("TR_SEQNUM", pa.int64()),
    pa.field("TR_ID", pa.int64()),
    pa.field("TR_SOURCE", pa.string()),
    pa.field("TR_RF", pa.string()),
])

RELAX_NUMERIC = {"SIZE", "PRICE", "TR_SEQNUM", "TR_ID"}

# -------------------- FILE DISCOVERY --------------------
files = sorted(IN_DIR.glob("*.parquet"))
if not files:
    raise RuntimeError(f"No parquet files found in {IN_DIR}")

print(f"Found {len(files)} parquet files")

# -------------------- REWRITE WITH PROGRESS + SAFETY --------------------
skipped = []

for f in tqdm(files, desc="Rewriting parquet chunks"):
    try:
        df = pl.read_parquet(f)

        # Ensure all expected columns exist
        for c, dt in TARGET_PL.items():
            if c not in df.columns:
                df = df.with_columns(pl.lit(None, dtype=dt).alias(c))

        # Cast to target dtypes
        df = (
            df.with_columns([
                pl.col(c).cast(dt, strict=(c not in RELAX_NUMERIC)).alias(c)
                for c, dt in TARGET_PL.items()
            ])
            .select(list(TARGET_PL.keys()))
        )

        # Force Arrow schema
        t = df.to_arrow().cast(TARGET_PA)

        tmp = OUT_DIR / (f.name + ".tmp")
        out = OUT_DIR / f.name

        pq.write_table(
            t,
            tmp,
            compression="zstd",
            write_statistics=True,
            use_dictionary=False,
            row_group_size=t.num_rows,
        )
        tmp.replace(out)

    except Exception as e:
        skipped.append((f.name, str(e)))

# -------------------- SUMMARY --------------------
print(f"\nDone. Written {len(files) - len(skipped)} / {len(files)} files")
if skipped:
    print("\nSkipped files:")
    for name, err in skipped:
        print(f"  {name}: {err}")

# -------------------- OPTIONAL SANITY CHECK --------------------
_ = pl.scan_parquet(str(OUT_DIR / "*.parquet"), schema=TARGET_PL).head(1).collect()
print("Sanity check passed.")
