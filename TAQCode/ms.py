# ms.py — Microstructure stats (IS, MI, MR) for merged_all.parquet
# ------------------------------------------------------------------------------
# THINKING / DESIGN NOTES (fully auditable):
#
# 1) We analyze microstructure metrics (IS, MI, MR) from the already-constructed
#    merged_all.parquet produced by top50_trades_corresponding_quotes_persist.py.
#    That file already contains correct TS timestamps for trades & quotes.
#
# 2) Arrival quotes: backward-asof in the persist script ensures that each trade
#    gets matched to the **most recent quote at or before the trade timestamp**.
#    This means ARR_MID = MID with shift(0) is correct because the join already
#    did the sequencing. No extra lookup needed here.
#
# 3) Multiple quotes can occur at the same TIME_M; backward-asof ensures the
#    quote with the **latest QU_SEQNUM** is used. That logic lives in the persist
#    script (where the authoritative quote selection must be done).
#
# 4) IS_BPS = 10,000 * SIDE * (exec_price – arrival_mid) / arrival_mid.
#    MI_BPS = 10,000 * SIDE * (mid_now – arrival_mid) / arrival_mid.
#
# 5) MR_BPS compares current mid to a **future mid**. Therefore we must “lead”
#    the MID column — this is why:
#        MID_FWD = MID.shift(-K)
#    The negative shift means “look ahead K trades,” because mean reversion is a
#    forward-looking metric (future drift opposite to trade direction).
#
# 6) We filter out rows where MID_FWD <= 0 *before* computing MR_BPS to avoid
#    infinities and division errors.
#
# 7) Use engine="streaming" ALWAYS to avoid memory blow-ups.
# ------------------------------------------------------------------------------

import time
import argparse
import shutil
from pathlib import Path
import polars as pl
import argparse

# ---------------- CONFIG ----------------
MERGED_FILE = None
summary_df = None
metrics_history = None
K_FWD_TRADES = 10     # event-time lookahead for MR

pl.Config.set_tbl_cols(120)
pl.Config.set_tbl_width_chars(240)


def sanity_check():
    base = pl.scan_parquet(str(MERGED_FILE))

    print(f"Null TR_SEQNUM : {base.filter(pl.col('TR_SEQNUM').is_null()).count().collect(engine='streaming').item(0,0)}")
    base = base.filter(pl.col("TR_SEQNUM").is_not_null())

    print(f"Null QU_SEQNUM : {base.filter(pl.col('QU_SEQNUM').is_null()).count().collect(engine='streaming').item(0,0)}")
    base = base.filter(pl.col("QU_SEQNUM").is_not_null())

    print(f"Zero BID and ASK : {base.filter((pl.col('BID') == 0) & (pl.col('ASK') == 0)).count().collect(engine='streaming').item(0,0)}")
    base = base.filter((pl.col('BID') != 0) & (pl.col('ASK') != 0))

    return base


def analyze_ticker(base, ticker: str, out_dir: Path):
    t0 = time.time()
    ticker = ticker.upper()
    print(f"\n🔎 Running microstructure analysis for: {ticker}")

    base = (
        base.filter(pl.col("SYM_ROOT") == ticker)
            .sort("TS")
    )

    # ---- METRICS PIPELINE ----
    metrics_lf = base.with_columns([
        pl.when(pl.col("PRICE") >= pl.col("MID")).then(1).otherwise(-1).alias("SIDE"),
        pl.col("MID").shift(0).alias("ARR_MID"),
    ])

    metrics_lf = metrics_lf.with_columns([
        pl.when(pl.col("ARR_MID") > 0)
          .then(10_000 * pl.col("SIDE") * (pl.col("PRICE") - pl.col("ARR_MID")) / pl.col("ARR_MID"))
          .otherwise(None)
          .alias("IS_BPS"),

        pl.when(pl.col("ARR_MID") > 0)
          .then(10_000 * pl.col("SIDE") * (pl.col("MID") - pl.col("ARR_MID")) / pl.col("ARR_MID"))
          .otherwise(None)
          .alias("MI_BPS"),
    ])

    metrics_lf = metrics_lf.with_columns([
        pl.col("MID").shift(-K_FWD_TRADES).alias("MID_FWD")
    ])

    metrics_lf = metrics_lf.filter(pl.col("MID_FWD") > 0)

    metrics_lf = metrics_lf.with_columns([
        (10_000 * (-pl.col("SIDE")) * (pl.col("MID_FWD") - pl.col("MID")) / pl.col("MID"))
        .alias("MR_BPS")
    ])

    # ---- SUMMARY ----
    summary_lf = metrics_lf.select([
        pl.len().alias("N_ROWS"),
        pl.col("SIZE").sum().alias("TOTAL_SIZE"),
        ((pl.col("PRICE") * pl.col("SIZE")).sum() / pl.col("SIZE").sum()).alias("VWAP"),
        (pl.col("SPREAD").mean() * 10_000).alias("AVG_SPREAD_BPS"),
        pl.col("DEPTH_TOB").mean().alias("AVG_DEPTH_TOB"),

        pl.col("IS_BPS").mean().alias("IS_MEAN_BPS"),
        pl.col("IS_BPS").median().alias("IS_MEDIAN_BPS"),
        pl.col("IS_BPS").min().alias("IS_MIN_BPS"),
        pl.col("IS_BPS").max().alias("IS_MAX_BPS"),

        pl.col("MI_BPS").mean().alias("MI_MEAN_BPS"),
        pl.col("MI_BPS").median().alias("MI_MEDIAN_BPS"),
        pl.col("MI_BPS").min().alias("MI_MIN_BPS"),
        pl.col("MI_BPS").max().alias("MI_MAX_BPS"),

        pl.col("MR_BPS").mean().alias("MR_MEAN_BPS"),
        pl.col("MR_BPS").median().alias("MR_MEDIAN_BPS"),
        pl.col("MR_BPS").min().alias("MR_MIN_BPS"),
        pl.col("MR_BPS").max().alias("MR_MAX_BPS"),
    ]).with_columns([
        pl.lit(ticker).alias("TICKER"),
        pl.lit(K_FWD_TRADES).alias("FWD_TRADES_FOR_MR")
    ])

    summary_df_local = summary_lf.collect(engine="streaming")
    summary_df_local = summary_df_local.select(['TICKER', pl.all().exclude('TICKER')])
    print(summary_df_local)

    out_csv = out_dir / f"ms_summary_{ticker}.csv"
    summary_df_local.write_csv(out_csv)
    print(f"💾 Saved → {out_csv.resolve()} | ⏱ {time.time()-t0:.2f}s")
    global summary_df, metrics_history

    if summary_df is None :
        summary_df = summary_df_local
    else:
        summary_df = summary_df.vstack(summary_df_local).rechunk()

    if metrics_history is None:
        metrics_history = metrics_lf
    else:
        metrics_history = pl.concat([metrics_history,metrics_lf])
    # globals()["metrics_lf"] = globals()["metrics_lf"].vstack(metrics_lf)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--MERGED_FILE", required=True)
    parser.add_argument("--OUT_DIR", required=True)
    parser.add_argument("--TICKERS", required=True)
    args = parser.parse_args()

    out_dir = Path(args.OUT_DIR)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    global MERGED_FILE
    MERGED_FILE = Path(args.MERGED_FILE)
    base = sanity_check()
    tickers = args.TICKERS.split(",") if args.TICKERS else []

    for t in tickers:
        analyze_ticker(base, t, out_dir)


if __name__ == "__main__":
    main()
