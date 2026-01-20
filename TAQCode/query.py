#query.py
import polars as pl
from datetime import date, datetime, timedelta  # (not strictly needed here, kept as you had it)
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--TRADES_UPPER")
parser.add_argument("--QUOTES_OLD")
parser.add_argument("--OUTPUT_FILE")
args = parser.parse_args()

TRADES_UPPER = args.TRADES_UPPER  # e.g. "/home/amazon/Documents/TAQData/processed_output_trades_upper/*.parquet"
QUOTES_OLD   = args.QUOTES_OLD    # e.g. "/home/amazon/Documents/TAQData/processed_output_quotes/*.parquet"
OUTPUT_FILE  = args.OUTPUT_FILE   # e.g. "taq_analysis_output.txt"

# --- Caution: delete existing output file before processing ---
out_path = Path(OUTPUT_FILE)
if out_path.exists():
    out_path.unlink()
    print(f"[INFO] Deleted existing output file before processing: {out_path}")

# Scan lazily
quotes_lf = pl.scan_parquet(QUOTES_OLD)
trades_lf = pl.scan_parquet(TRADES_UPPER)

pl.Config.set_tbl_cols(50)
pl.Config.set_tbl_width_chars(200)

# Build timestamp (TS) columns
quotes_lf = quotes_lf.with_columns([
    (pl.col("DATE").cast(pl.Datetime("us")) + pl.duration(
        hours=pl.col("TIME_M").dt.hour(),
        minutes=pl.col("TIME_M").dt.minute(),
        seconds=pl.col("TIME_M").dt.second(),
        microseconds=(pl.col("TIME_M").dt.nanosecond() // 1000)
    )).alias("TS")
])

trades_lf = trades_lf.with_columns([
    (pl.col("DATE").cast(pl.Datetime("us")) + pl.duration(
        hours=pl.col("TIME_M").dt.hour(),
        minutes=pl.col("TIME_M").dt.minute(),
        seconds=pl.col("TIME_M").dt.second(),
        microseconds=(pl.col("TIME_M").dt.nanosecond() // 1000)
    )).alias("TS")
])

# Perform asof join
joined_lf = trades_lf.join_asof(
    quotes_lf,
    left_on="TS",
    right_on="TS",
    by=["SYM_ROOT", "SYM_SUFFIX"],
    strategy="nearest",
    tolerance="1s"
)

# Write some samples
with open(OUTPUT_FILE, "w") as f:
    print(trades_lf.fetch(n_rows=10), file=f)
    print(quotes_lf.fetch(n_rows=10), file=f)
    # If you want, you can also print joined:
    # print(joined_lf.fetch(n_rows=10), file=f)

print(f"\nAnalysis successfully executed and results written to {OUTPUT_FILE}")
