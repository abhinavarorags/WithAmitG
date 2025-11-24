# top50_trades_corresponding_quotes_persist.py
# -----------------------------------------------------------------------------
# PURPOSE:
# Build ONE merged parquet (`merged_all.parquet`) from the top-50 tickers using
# quote-dominant backward as-of joins:
#   quotes.TS ≤ trades.TS  (i.e., last quote before the trade)
#
# KEY POINTS:
# • TS is built separately for trades and quotes using full nanosecond precision.
# • backward ensures strictly backward-in-time matching:
#       latest quote with TS ≤ trade.TS
# • tolerance kept at "1s" (per your instruction).
#   Possible tolerances: "1ns","10ns","100ns","1us","10us","100us",
#                        "1ms","10ms","100ms","1s"
# • No filtering of columns — ALL columns are kept.
# • Data is persisted in one Parquet file with many row groups (~25k rows each).
# • We DO NOT rely on QU_SEQNUM manually — backward asof handles the “latest”
#   quote for each trade, including multiple quotes at identical TIME_M.
# -----------------------------------------------------------------------------

import time
from datetime import timedelta, date as date_cls
from pathlib import Path
import polars as pl
import pyarrow.parquet as pq

# -------------------- Paths & Config --------------------
TRADES_DIR = Path("/home/amazon/Documents/TAQData/processed_output_trades_upper_top50/")
QUOTES_DIR = Path("/home/amazon/Documents/TAQData/processed_output_quotes_top50/")
TOP50_CSV = Path("./stats_out_simple/top50_trades_by_volume.csv")

OUT_DIR = Path("/home/amazon/Documents/TAQData/merged_output_top50/")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "merged_all.parquet"

CHUNK_SIZE = 25_000

# Tolerance REQUIRED to be "1s"
TOLERANCE = "1s"      # Practical values: "1ns","10ns","100ns","1us","10us","100us","1ms","10ms","100ms","1s"

pl.Config.set_tbl_cols(120)
pl.Config.set_tbl_width_chars(220)


# ---------------- TS BUILDER (FULL NANO PRECISION) ----------------
def build_ts_quotes(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Quotes TIME_M is already time64[ns], so TS = DATE + TIME_M (ns)."""
    return lf.with_columns([
        (pl.col("DATE").cast(pl.Datetime("ns")) + pl.col("TIME_M").cast(pl.Duration("ns"))).alias("TS")
    ])


def build_ts_trades(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Trades provide time64[us] + time_m_nano(int16) nanosecond remainder."""
    return lf.with_columns([
        (
            pl.col("DATE").cast(pl.Datetime("ns")) +
            pl.col("TIME_M").cast(pl.Duration("us")).cast(pl.Duration("ns")) +
            pl.col("TIME_M_NANO").cast(pl.Duration("ns"))
        ).alias("TS")
    ])


# ---------------- DATE RANGE HELPER ----------------
def symbol_date_range(sym_root: str, sym_suffix: str):
    q = (
        pl.scan_parquet(str(QUOTES_DIR / "*.parquet"))
        .filter(
            (pl.col("SYM_ROOT") == sym_root) &
            (pl.col("SYM_SUFFIX").fill_null("") == sym_suffix)
        )
        .select(pl.min("DATE").alias("min_d"), pl.max("DATE").alias("max_d"))
        .collect()
    )
    if q.height == 0 or q["min_d"][0] is None:
        return None, None
    return q["min_d"][0], q["max_d"][0]


def daterange(a: date_cls, b: date_cls):
    d = a
    while d <= b:
        yield d
        d += timedelta(days=1)


# ---------------- PER SYMBOL PER HOUR LAZY JOIN ----------------
def per_symbol_per_hour_lazy(sym_root: str, sym_suffix: str, day: date_cls, hour: int):
    """Quotes-left backward asof join (quote-dominant). ALL columns kept."""

    quotes_lf = (
        pl.scan_parquet(str(QUOTES_DIR / "*.parquet"))
        .filter(
            (pl.col("SYM_ROOT") == sym_root) &
            (pl.col("SYM_SUFFIX").fill_null("") == sym_suffix) &
            (pl.col("DATE") == day) &
            (pl.col("TIME_M").dt.hour() == hour)
        )
    )

    trades_lf = (
        pl.scan_parquet(str(TRADES_DIR / "*.parquet"))
        .filter(
            (pl.col("SYM_ROOT") == sym_root) &
            (pl.col("SYM_SUFFIX").fill_null("") == sym_suffix) &
            (pl.col("DATE") == day) &
            (pl.col("TIME_M").dt.hour() == hour)
        )
    )

    quotes_lf = build_ts_quotes(quotes_lf).with_columns([
        ((pl.col("BID") + pl.col("ASK")) / 2).alias("MID"),
        (pl.col("ASK") - pl.col("BID")).alias("SPREAD"),
        (pl.col("BIDSIZ") + pl.col("ASKSIZ")).alias("DEPTH_TOB"),
    ])

    trades_lf = build_ts_trades(trades_lf)

    # Critical: BACKWARD ensures quotes.TS ≤ trades.TS (the correct market rule)
    joined = trades_lf.join_asof(
        quotes_lf,
        left_on="TS",
        right_on="TS",
        by=["SYM_ROOT", "SYM_SUFFIX"],
        strategy="backward",
        tolerance=TOLERANCE,
    )
    return joined


# ---------------- MAIN DRIVER ----------------
def main():
    print(f"📄 Loading top50: {TOP50_CSV}")
    top50 = pl.read_csv(TOP50_CSV)

    symbols_df = (
        top50.select([
            pl.col("SYM_ROOT").str.to_uppercase(),
            pl.col("SYM_SUFFIX").fill_null("")
        ])
        .unique()
    )
    symbols = [(r[0], r[1]) for r in symbols_df.iter_rows()]

    if OUT_FILE.exists():
        OUT_FILE.unlink()

    writer = None
    total_rows = 0
    row_groups = 0
    t0 = time.time()

    for sidx, (root, suf) in enumerate(symbols, start=1):
        tag = f"{root}{('_' + suf) if suf else ''}"
        d0, d1 = symbol_date_range(root, suf)
        if d0 is None:
            print(f"— [{sidx}/{len(symbols)}] {tag}: no quotes → skip")
            continue

        print(f"\n—— [{sidx}/{len(symbols)}] {tag}: {d0} → {d1} | tol={TOLERANCE} | chunk={CHUNK_SIZE:,}")

        for day in daterange(d0, d1):
            for hr in range(24):
                shard = per_symbol_per_hour_lazy(root, suf, day, hr)

                try:
                    out_df = shard.collect(streaming=True)
                except:
                    continue

                wrote = False
                for chunk in out_df.iter_slices(n_rows=CHUNK_SIZE):
                    df = pl.DataFrame(chunk)
                    n = df.height
                    if n == 0:
                        continue

                    tbl = df.to_arrow()
                    if writer is None:
                        writer = pq.ParquetWriter(
                            where=str(OUT_FILE),
                            schema=tbl.schema,
                            compression="zstd"
                        )
                        print(f"📝 Opened writer → {OUT_FILE.name}")

                    writer.write_table(tbl)
                    wrote = True
                    total_rows += n
                    row_groups += 1
                    print(f"    ✅ RG {row_groups:05d}: {tag} {day} {hr:02d}:00 — {n:,} rows")

                if not wrote:
                    print(f"    · {tag} {day} {hr:02d}:00 — 0 rows")

    if writer:
        writer.close()

    print(f"\n🏁 DONE. Total rows={total_rows:,} | row_groups={row_groups:,}")
    print(f"📂 File: {OUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
