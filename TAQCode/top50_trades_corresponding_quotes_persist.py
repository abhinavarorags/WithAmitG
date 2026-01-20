# top50_trades_corresponding_quotes_persist.py
# -----------------------------------------------------------------------------
# TEMP DEBUG MODS INCLUDED:
#   1) No silent swallow: print exceptions (first N) with context (symbol/day/hour)
#   2) TEMP: run only 1 symbol (first symbol in top50) to debug quickly
#      (remove the TEMP block later)
# -----------------------------------------------------------------------------

import time
import shutil
from datetime import timedelta, date as date_cls
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--TRADES_DIR")
parser.add_argument("--QUOTES_DIR")
parser.add_argument("--STATS_DIR")
parser.add_argument("--TOP50_CSV")
parser.add_argument("--OUT_DIR")
parser.add_argument("--OUT_FILE")
parser.add_argument("--CHUNK_SIZE", type=int)
args = parser.parse_args()

TRADES_DIR = Path(args.TRADES_DIR)
QUOTES_DIR = Path(args.QUOTES_DIR)
TOP50_CSV  = Path(f"{args.STATS_DIR}/{args.TOP50_CSV}")

OUT_DIR = Path(args.OUT_DIR)

# -------------------- CAUTION: WIPE OUT_DIR (prevents accidental append/duplication) --------------------
if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)
    print(f"[INFO] Deleted existing OUT_DIR before processing: {OUT_DIR}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUT_DIR / args.OUT_FILE
CHUNK_SIZE = args.CHUNK_SIZE

TOLERANCE = "1s"

pl.Config.set_tbl_cols(120)
pl.Config.set_tbl_width_chars(220)

# ---------------- TS BUILDER (SAFE, MICROSECOND-BASED) ----------------
def build_ts_us(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.with_columns([
        (
            pl.col("DATE").cast(pl.Datetime("us")) +
            pl.duration(
                hours=pl.col("TIME_M").dt.hour(),
                minutes=pl.col("TIME_M").dt.minute(),
                seconds=pl.col("TIME_M").dt.second(),
                microseconds=(pl.col("TIME_M").dt.nanosecond() // 1_000),
            )
            # ---- uncomment ONLY if TIME_M_NANO exists and you want extra ns remainder ----
            # + pl.col("TIME_M_NANO").cast(pl.Duration("ns"))
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

    # Build TS consistently (us-based)
    quotes_lf = build_ts_us(quotes_lf).with_columns([
        ((pl.col("BID") + pl.col("ASK")) / 2).alias("MID"),
        (pl.col("ASK") - pl.col("BID")).alias("SPREAD"),
        (pl.col("BIDSIZ") + pl.col("ASKSIZ")).alias("DEPTH_TOB"),
    ])

    trades_lf = build_ts_us(trades_lf)

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

    # ---------------- TEMP: RUN ONLY 1 SYMBOL (REMOVE LATER) ----------------
    if not symbols:
        raise SystemExit("No symbols found in TOP50 CSV")
    # symbols = symbols[:1]
    # print(f"[TEMP] Running only 1 symbol for debug: {symbols[0]}")
    # -----------------------------------------------------------------------

    if OUT_FILE.exists():
        OUT_FILE.unlink()

    writer = None
    total_rows = 0
    row_groups = 0
    t0 = time.time()

    max_err_print = 30
    err_printed = 0

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

                # TEMP sanity counts BEFORE join collect (helps pinpoint empty vs failing)
                try:
                    q_cnt = (
                        pl.scan_parquet(str(QUOTES_DIR / "*.parquet"))
                        .filter(
                            (pl.col("SYM_ROOT") == root) &
                            (pl.col("SYM_SUFFIX").fill_null("") == suf) &
                            (pl.col("DATE") == day) &
                            (pl.col("TIME_M").dt.hour() == hr)
                        )
                        .select(pl.len().alias("n"))
                        .collect()
                    )["n"][0]
                    t_cnt = (
                        pl.scan_parquet(str(TRADES_DIR / "*.parquet"))
                        .filter(
                            (pl.col("SYM_ROOT") == root) &
                            (pl.col("SYM_SUFFIX").fill_null("") == suf) &
                            (pl.col("DATE") == day) &
                            (pl.col("TIME_M").dt.hour() == hr)
                        )
                        .select(pl.len().alias("n"))
                        .collect()
                    )["n"][0]
                except Exception as e:
                    if err_printed < max_err_print:
                        print(f"[ERR pre-count] {tag} {day} {hr:02d}:00 -> {type(e).__name__}: {e}")
                        err_printed += 1
                    continue

                if (q_cnt == 0) or (t_cnt == 0):
                    # no data to join
                    continue

                try:
                    out_df = shard.collect(engine="streaming")
                except Exception as e:
                    if err_printed < max_err_print:
                        print(f"[ERR collect] {tag} {day} {hr:02d}:00 (q={q_cnt:,} t={t_cnt:,}) -> {type(e).__name__}: {e}")
                        err_printed += 1
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
                    print(f"    · {tag} {day} {hr:02d}:00 — joined but 0 rows")

    if writer:
        writer.close()

    print(f"\n🏁 DONE. Total rows={total_rows:,} | row_groups={row_groups:,}")
    print(f"📂 File: {OUT_FILE.resolve()}")

if __name__ == "__main__":
    main()
