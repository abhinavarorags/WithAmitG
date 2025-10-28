# top50_trades_corresponding_quotes.py
# ▶ Filter top-50 symbols into new trades/quotes folders using a SYMBOLS DATAFRAME (no lists), with verbose progress.

"""
STEPS:
1) Read stats_out_simple/top50_trades_by_volume.csv and build a Polars DF of (SYM_ROOT, SYM_SUFFIX) for the top-50.
2) For every Parquet row-group in TRADES source, semi-join on that DF and write only matching rows to /processed_output_trades_upper_top50/.
3) Do the same for QUOTES into /processed_output_quotes_top50/.
4) Write a small merged summary CSV to stats_out_simple/merged_top50_sorted.csv (contains totals and keeps top-50 order).
"""

import time
from pathlib import Path

import pyarrow.parquet as pq
import polars as pl

# ---------- PATHS ----------
OUT_DIR = Path("./stats_out_simple")
TRADES_TOP50_CSV = OUT_DIR / "top50_trades_by_volume.csv"  # input with top-50 (SYM_ROOT, SYM_SUFFIX, TRADES_VOL, TRADES_ROWS)

TRADES_SRC_DIR = Path("/home/amazon/Documents/TAQData/processed_output_trades_upper/")
TRADES_OUT_DIR = Path("/home/amazon/Documents/TAQData/processed_output_trades_upper_top50/")

QUOTES_SRC_DIR = Path("/home/amazon/Documents/TAQData/processed_output_quotes/")
QUOTES_OUT_DIR = Path("/home/amazon/Documents/TAQData/processed_output_quotes_top50/")

MERGED_OUT_CSV = OUT_DIR / "merged_top50_sorted.csv"

TRADES_OUT_DIR.mkdir(parents=True, exist_ok=True)
QUOTES_OUT_DIR.mkdir(parents=True, exist_ok=True)

pl.Config.set_tbl_cols(100)
pl.Config.set_tbl_width_chars(200)

def list_parquet_files(d: Path):
    return sorted(p for p in d.glob("*.parquet") if p.is_file())

print("\n🔧 Paths configured:")
print(f"  • Top-50 CSV        : {TRADES_TOP50_CSV}")
print(f"  • Trades src        : {TRADES_SRC_DIR}")
print(f"  • Trades out (top50): {TRADES_OUT_DIR}")
print(f"  • Quotes src        : {QUOTES_SRC_DIR}")
print(f"  • Quotes out (top50): {QUOTES_OUT_DIR}")
print(f"  • Summary CSV       : {MERGED_OUT_CSV}")

# ---------- 1) LOAD TOP-50 SYMBOLS AS A DATAFRAME ----------
print("\n📄 Loading top-50 symbols…")
top50 = pl.read_csv(TRADES_TOP50_CSV)
if top50.height == 0:
    raise SystemExit("❌ No rows in top50_trades_by_volume.csv")

# Normalize suffix to empty string and uppercase roots; then keep only keys needed for joins
symbols_df = (
    top50.select([
        pl.col("SYM_ROOT").cast(pl.Utf8).str.to_uppercase(),
        pl.col("SYM_SUFFIX").cast(pl.Utf8).fill_null("")
    ])
    .unique(maintain_order=True)
)
print(f"✅ Symbols DF ready: {symbols_df.height} rows (expected 50)")
print(symbols_df.head(10))

# ---------- 2) FILTER TRADES CHUNKS VIA SEMI-JOIN ----------
print("\n🚀 Filtering TRADES into top-50 subset (semi-join per chunk)…")
trade_files = list_parquet_files(TRADES_SRC_DIR)
if not trade_files:
    raise SystemExit(f"❌ No parquet files found in {TRADES_SRC_DIR}")

total_trades_saved = 0
t0 = time.time()

for fidx, fpath in enumerate(trade_files, 1):
    pf = pq.ParquetFile(str(fpath))
    n_rg = pf.num_row_groups
    print(f"\n  → TRADES file {fidx}/{len(trade_files)}: {fpath.name} with {n_rg} row groups")
    for rg in range(n_rg):
        # Read one row group
        pa_tbl = pf.read_row_group(rg)
        df = pl.from_arrow(pa_tbl)

        # Normalize keys for safe joining
        df = df.with_columns([
            pl.col("SYM_ROOT").cast(pl.Utf8).str.to_uppercase(),
            pl.col("SYM_SUFFIX").cast(pl.Utf8).fill_null("")
        ])

        # Keep only rows whose (SYM_ROOT, SYM_SUFFIX) are in top-50 (no lists; use semi-join)
        filtered = df.join(symbols_df, on=["SYM_ROOT", "SYM_SUFFIX"], how="semi")

        if filtered.height > 0:
            out_file = TRADES_OUT_DIR / f"trades_chunk_{fidx:03d}_{rg:04d}.parquet"
            filtered.write_parquet(out_file)
            total_trades_saved += filtered.height
            print(f"    ✅ RG {rg+1:04d}/{n_rg:04d}: wrote {filtered.height:,} rows → {out_file.name}")
        else:
            if rg % 50 == 0 or rg == n_rg - 1:
                print(f"    … RG {rg+1:04d}/{n_rg:04d}: no matches")

print(f"\n🎯 TRADES filtering complete in {time.time()-t0:.1f}s — total saved: {total_trades_saved:,} rows")

# ---------- 3) FILTER QUOTES CHUNKS VIA SEMI-JOIN ----------
print("\n🚀 Filtering QUOTES into top-50 subset (semi-join per chunk)…")
quote_files = list_parquet_files(QUOTES_SRC_DIR)
if not quote_files:
    raise SystemExit(f"❌ No parquet files found in {QUOTES_SRC_DIR}")

total_quotes_saved = 0
t1 = time.time()

for fidx, fpath in enumerate(quote_files, 1):
    pf = pq.ParquetFile(str(fpath))
    n_rg = pf.num_row_groups
    print(f"\n  → QUOTES file {fidx}/{len(quote_files)}: {fpath.name} with {n_rg} row groups")
    for rg in range(n_rg):
        pa_tbl = pf.read_row_group(rg)
        df = pl.from_arrow(pa_tbl)

        df = df.with_columns([
            pl.col("SYM_ROOT").cast(pl.Utf8).str.to_uppercase(),
            pl.col("SYM_SUFFIX").cast(pl.Utf8).fill_null("")
        ])

        filtered = df.join(symbols_df, on=["SYM_ROOT", "SYM_SUFFIX"], how="semi")

        if filtered.height > 0:
            out_file = QUOTES_OUT_DIR / f"quotes_chunk_{fidx:03d}_{rg:04d}.parquet"
            filtered.write_parquet(out_file)
            total_quotes_saved += filtered.height
            print(f"    ✅ RG {rg+1:04d}/{n_rg:04d}: wrote {filtered.height:,} rows → {out_file.name}")
        else:
            if rg % 50 == 0 or rg == n_rg - 1:
                print(f"    … RG {rg+1:04d}/{n_rg:04d}: no matches")

print(f"\n🎯 QUOTES filtering complete in {time.time()-t1:.1f}s — total saved: {total_quotes_saved:,} rows")

# ---------- 4) SUMMARY CSV ----------
print("\n🧮 Writing merged summary CSV (keeps original top-50 order)…")
summary = (
    top50
    .with_columns([
        pl.lit(int(total_trades_saved)).alias("TRADES_ROWS_SAVED_TOTAL"),
        pl.lit(int(total_quotes_saved)).alias("QUOTES_ROWS_SAVED_TOTAL"),
    ])
)
summary.write_csv(MERGED_OUT_CSV)

print("\n✅ Done.")
print(f"   • TRADES saved rows total: {total_trades_saved:,}")
print(f"   • QUOTES saved rows total: {total_quotes_saved:,}")
print(f"   • Summary CSV: {MERGED_OUT_CSV.resolve()}")

