# top50_trades_corresponding_quotes_persist.py — Quote-dominant asof join; ONE Parquet file with many row groups.
# Steps: (1) load top-50 symbols; (2) per-symbol, per-hour shards; (3) build TS = DATE+TIME_M; (4) QUOTES ⟵asof⟶ TRADES @ 1 ms;
#         (5) stream each shard in 25k-row chunks into ONE Parquet file (row groups preserved).  All columns (incl. TS) are kept.

import time
from datetime import timedelta, date as date_cls
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

# -------------------- Paths & Config --------------------
TRADES_DIR    = Path("/home/amazon/Documents/TAQData/processed_output_trades_upper_top50/")
QUOTES_DIR    = Path("/home/amazon/Documents/TAQData/processed_output_quotes_top50/")
TOP50_CSV     = Path("./stats_out_simple/top50_trades_by_volume.csv")

OUT_DIR       = Path("/home/amazon/Documents/TAQData/merged_output_top50/")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE      = OUT_DIR / "merged_all.parquet"   # ONE parquet file (many row groups)

CHUNK_SIZE    = 25_000
TOLERANCE     = "1ms"

pl.Config.set_tbl_cols(120)
pl.Config.set_tbl_width_chars(220)


def build_ts(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Create TS from DATE + TIME_M (μs), lazily."""
    return lf.with_columns([
        (pl.col("DATE").cast(pl.Datetime("us")) +
         pl.duration(
            hours=pl.col("TIME_M").dt.hour(),
            minutes=pl.col("TIME_M").dt.minute(),
            seconds=pl.col("TIME_M").dt.second(),
            microseconds=(pl.col("TIME_M").dt.nanosecond() // 1000)
         )).alias("TS")
    ])


def symbol_date_range(sym_root: str, sym_suffix: str):
    """Get min/max DATE for this symbol from QUOTES (cheap aggregation)."""
    q = (
        pl.scan_parquet(str(QUOTES_DIR / "*.parquet"))
        .filter((pl.col("SYM_ROOT") == sym_root) & (pl.col("SYM_SUFFIX").fill_null("") == sym_suffix))
        .select(pl.min("DATE").alias("min_d"), pl.max("DATE").alias("max_d"))
        .collect()
    )
    if q.height == 0 or q["min_d"][0] is None or q["max_d"][0] is None:
        return None, None
    return q["min_d"][0], q["max_d"][0]


def per_symbol_per_hour_lazy(sym_root: str, sym_suffix: str, day: date_cls, hour: int) -> pl.LazyFrame:
    """Build a per-(symbol, day, hour) quote-left asof join (lazy). Keep ALL columns, including TS."""
    # Filter by day+hour BEFORE building TS to minimize data
    quotes_lf = (
        pl.scan_parquet(str(QUOTES_DIR / "*.parquet"))
        .filter(
            (pl.col("SYM_ROOT") == sym_root) &
            (pl.col("SYM_SUFFIX").fill_null("") == sym_suffix) &
            (pl.col("DATE") == pl.lit(day)) &
            (pl.col("TIME_M").dt.hour() == hour)
        )
        .select(["DATE", "TIME_M", "SYM_ROOT", "SYM_SUFFIX", "BID", "ASK", "BIDSIZ", "ASKSIZ"])
    )
    trades_lf = (
        pl.scan_parquet(str(TRADES_DIR / "*.parquet"))
        .filter(
            (pl.col("SYM_ROOT") == sym_root) &
            (pl.col("SYM_SUFFIX").fill_null("") == sym_suffix) &
            (pl.col("DATE") == pl.lit(day)) &
            (pl.col("TIME_M").dt.hour() == hour)
        )
        .select(["DATE", "TIME_M", "SYM_ROOT", "SYM_SUFFIX", "PRICE", "SIZE"])
    )

    # Add derived quote features (keep originals) + TS on both sides
    quotes_lf = build_ts(quotes_lf).with_columns([
        ((pl.col("BID") + pl.col("ASK")) / 2).alias("MID"),
        (pl.col("ASK") - pl.col("BID")).alias("SPREAD"),
        (pl.col("BIDSIZ") + pl.col("ASKSIZ")).alias("DEPTH_TOB"),
    ])
    trades_lf = build_ts(trades_lf)

    # Quote-dominant asof join; KEEP ALL COLUMNS (no select)
    joined = quotes_lf.join_asof(
        trades_lf,
        left_on="TS", right_on="TS",
        by=["SYM_ROOT", "SYM_SUFFIX"],
        strategy="nearest",
        tolerance=TOLERANCE,
    )
    return joined  # no column dropping; TS retained


def daterange(d0: date_cls, d1: date_cls):
    """Yield each date from d0 to d1 inclusive."""
    cur = d0
    while cur <= d1:
        yield cur
        cur += timedelta(days=1)


def main():
    print(f"📄 Reading symbols from: {TOP50_CSV}")
    top50 = pl.read_csv(TOP50_CSV)
    symbols_df = (
        top50.select([
            pl.col("SYM_ROOT").cast(pl.Utf8).str.to_uppercase(),
            pl.col("SYM_SUFFIX").cast(pl.Utf8).fill_null("")
        ])
        .unique(maintain_order=True)
    )
    symbols = [(r[0], r[1]) for r in symbols_df.iter_rows()]
    print(f"🔎 Symbols to process: {len(symbols)}")

    # Fresh output file
    if OUT_FILE.exists():
        OUT_FILE.unlink()
        print(f"🧹 Removed existing file: {OUT_FILE}")

    writer = None
    total_rows = 0
    row_groups = 0
    t0 = time.time()

    for sidx, (sym_root, sym_suffix) in enumerate(symbols, start=1):
        tag = f"{sym_root}{('_' + sym_suffix) if sym_suffix else ''}"
        d_min, d_max = symbol_date_range(sym_root, sym_suffix)
        if d_min is None:
            print(f"\n—— [{sidx}/{len(symbols)}] {tag}: no quotes found — skipping")
            continue

        print(f"\n—— [{sidx}/{len(symbols)}] {tag}: {d_min} → {d_max} | tol={TOLERANCE} | chunk={CHUNK_SIZE:,}")

        for d in daterange(d_min, d_max):
            for hr in range(0, 24):
                shard_tag = f"{tag} {d} {hr:02d}:00"
                # Build lazy joined shard
                shard_lf = per_symbol_per_hour_lazy(sym_root, sym_suffix, d, hr)

                # Stream-execute the shard and append in 25k chunks
                try:
                    stream_df = shard_lf.collect(streaming=True)
                except Exception as e:
                    # No data or execution failed for the shard → continue
                    continue

                wrote_any = False
                for slice_df in stream_df.iter_slices(n_rows=CHUNK_SIZE):
                    df = pl.DataFrame(slice_df)  # KEEP ALL COLUMNS
                    n = df.height
                    if n == 0:
                        continue

                    # Open writer on first non-empty chunk (schema from chunk)
                    tbl = df.to_arrow()
                    if writer is None:
                        writer = pq.ParquetWriter(
                            where=str(OUT_FILE),
                            schema=tbl.schema,
                            compression="zstd"
                        )
                        print(f"📝 Opened ParquetWriter → {OUT_FILE.name}")

                    writer.write_table(tbl)
                    total_rows += n
                    row_groups += 1
                    wrote_any = True
                    elapsed = time.time() - t0
                    print(f"    ✅ RowGroup {row_groups:05d}: {shard_tag} — wrote {n:,} rows (cum {total_rows:,}) | {elapsed:.1f}s")

                # Optional: print a dot for empty shard to show progress
                if not wrote_any:
                    print(f"    · {shard_tag}: 0 rows")

    if writer is not None:
        writer.close()
        print(f"\n🏁 Closed ParquetWriter. File: {OUT_FILE}")

    print(f"\n✅ DONE — total rows: {total_rows:,} | row groups: {row_groups:,} | elapsed: {time.time()-t0:.1f}s")
    print(f"📂 Single merged dataset (one file, many row groups): {OUT_FILE.resolve()}")


if __name__ == "__main__":
    main()

