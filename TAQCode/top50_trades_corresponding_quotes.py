# top50_trades_corresponding_quotes.py
# ▶ Filter top-50 symbols into new trades/quotes folders using a SYMBOLS DATAFRAME (no lists), with verbose progress.
# ▶ Also writes merged summary CSV with per-symbol QUOTES_ROWS and QUOTES_COLS.
# ▶ NEW: deletes TRADES_OUT_DIR and QUOTES_OUT_DIR at startup (prevents mixed-schema leftovers).

"""
STEPS:
1) Read stats_out_simple/top50_trades_by_volume.csv and build a Polars DF of (SYM_ROOT, SYM_SUFFIX) for the top-50.
2) For every Parquet row-group in TRADES source, semi-join on that DF and write only matching rows to TRADES_OUT_DIR.
3) Do the same for QUOTES into QUOTES_OUT_DIR, while accumulating per-symbol QUOTES_ROWS and QUOTES_COLS.
4) Write merged summary CSV to MERGED_OUT_CSV (keeps original top-50 order).
"""

import time
import shutil
from pathlib import Path
import argparse

import pyarrow.parquet as pq
import polars as pl

parser = argparse.ArgumentParser()
parser.add_argument("--OUT_DIR")
parser.add_argument("--TRADES_TOP50_CSV")
parser.add_argument("--TRADES_SRC_DIR")
parser.add_argument("--TRADES_OUT_DIR")
parser.add_argument("--QUOTES_SRC_DIR")
parser.add_argument("--QUOTES_OUT_DIR")
parser.add_argument("--MERGED_OUT_CSV")
args = parser.parse_args()

OUT_DIR = Path(args.OUT_DIR)
TRADES_TOP50_CSV = f"{OUT_DIR}/{args.TRADES_TOP50_CSV}"

TRADES_SRC_DIR = Path(args.TRADES_SRC_DIR)
TRADES_OUT_DIR = Path(args.TRADES_OUT_DIR)

QUOTES_SRC_DIR = Path(args.QUOTES_SRC_DIR)
QUOTES_OUT_DIR = Path(args.QUOTES_OUT_DIR)

MERGED_OUT_CSV = OUT_DIR / args.MERGED_OUT_CSV

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

# --- CAUTION: delete output dirs to avoid mixed-schema leftovers ---
if TRADES_OUT_DIR.exists():
    shutil.rmtree(TRADES_OUT_DIR)
    print(f"[INFO] Deleted existing TRADES_OUT_DIR before processing: {TRADES_OUT_DIR}")
TRADES_OUT_DIR.mkdir(parents=True, exist_ok=True)

if QUOTES_OUT_DIR.exists():
    shutil.rmtree(QUOTES_OUT_DIR)
    print(f"[INFO] Deleted existing QUOTES_OUT_DIR before processing: {QUOTES_OUT_DIR}")
QUOTES_OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Caution: delete existing merged summary CSV before processing ---
if MERGED_OUT_CSV.exists():
    MERGED_OUT_CSV.unlink()
    print(f"[INFO] Deleted existing merged summary CSV before processing: {MERGED_OUT_CSV}")

# ---------- 1) LOAD TOP-50 SYMBOLS AS A DATAFRAME ----------
print("\n📄 Loading top-50 symbols…")
top50 = pl.read_csv(TRADES_TOP50_CSV)
if top50.height == 0:
    raise SystemExit("❌ No rows in top50_trades_by_volume.csv")

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
        pa_tbl = pf.read_row_group(rg)
        df = pl.from_arrow(pa_tbl)

        df = df.with_columns([
            pl.col("SYM_ROOT").cast(pl.Utf8).str.to_uppercase(),
            pl.col("SYM_SUFFIX").cast(pl.Utf8).fill_null("")
        ])

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

# ---------- 3) FILTER QUOTES CHUNKS VIA SEMI-JOIN (+ per-symbol stats) ----------
print("\n🚀 Filtering QUOTES into top-50 subset (semi-join per chunk)…")
quote_files = list_parquet_files(QUOTES_SRC_DIR)
if not quote_files:
    raise SystemExit(f"❌ No parquet files found in {QUOTES_SRC_DIR}")

total_quotes_saved = 0
t1 = time.time()

quotes_rows_by_sym = {}   # (SYM_ROOT, SYM_SUFFIX) -> rows
quotes_cols_const = None

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

        if quotes_cols_const is None:
            quotes_cols_const = len(df.columns)
            print(f"    ℹ️ QUOTES_COLS detected: {quotes_cols_const}")

        filtered = df.join(symbols_df, on=["SYM_ROOT", "SYM_SUFFIX"], how="semi")

        if filtered.height > 0:
            grp = (
                filtered.group_by(["SYM_ROOT", "SYM_SUFFIX"])
                .agg(pl.len().alias("n"))
            )
            for r in grp.iter_rows():
                k = (r[0], r[1])
                quotes_rows_by_sym[k] = quotes_rows_by_sym.get(k, 0) + int(r[2])

            out_file = QUOTES_OUT_DIR / f"quotes_chunk_{fidx:03d}_{rg:04d}.parquet"
            filtered.write_parquet(out_file)
            total_quotes_saved += filtered.height
            print(f"    ✅ RG {rg+1:04d}/{n_rg:04d}: wrote {filtered.height:,} rows → {out_file.name}")
        else:
            if rg % 50 == 0 or rg == n_rg - 1:
                print(f"    … RG {rg+1:04d}/{n_rg:04d}: no matches")

print(f"\n🎯 QUOTES filtering complete in {time.time()-t1:.1f}s — total saved: {total_quotes_saved:,} rows")

# Build quotes stats DF to join into summary
quotes_stats_df = pl.DataFrame(
    {
        "SYM_ROOT": [k[0] for k in quotes_rows_by_sym.keys()],
        "SYM_SUFFIX": [k[1] for k in quotes_rows_by_sym.keys()],
        "QUOTES_ROWS": [v for v in quotes_rows_by_sym.values()],
    }
)
if quotes_cols_const is None:
    quotes_cols_const = 0
quotes_stats_df = quotes_stats_df.with_columns(pl.lit(int(quotes_cols_const)).alias("QUOTES_COLS"))

# ---------- 4) SUMMARY CSV (keep original top-50 order) ----------
print("\n🧮 Writing merged summary CSV (keeps original top-50 order)…")
summary = (
    top50
    .with_columns([
        pl.col("SYM_ROOT").cast(pl.Utf8).str.to_uppercase(),
        pl.col("SYM_SUFFIX").cast(pl.Utf8).fill_null("")
    ])
    .join(quotes_stats_df, on=["SYM_ROOT", "SYM_SUFFIX"], how="left")
    .with_columns([
        pl.lit(int(total_trades_saved)).alias("TRADES_ROWS_SAVED_TOTAL"),
        pl.lit(int(total_quotes_saved)).alias("QUOTES_ROWS_SAVED_TOTAL"),
        pl.col("QUOTES_ROWS").fill_null(0).cast(pl.Int64),
        pl.col("QUOTES_COLS").fill_null(int(quotes_cols_const)).cast(pl.Int64),
    ])
)

summary.write_csv(MERGED_OUT_CSV)

print("\n✅ Done.")
print(f"   • TRADES saved rows total: {total_trades_saved:,}")
print(f"   • QUOTES saved rows total: {total_quotes_saved:,}")
print(f"   • Summary CSV: {MERGED_OUT_CSV.resolve()}")
