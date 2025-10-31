# ms.py — Microstructure stats (IS, MI, MR) for a given ticker from merged_all.parquet
# ------------------------------------------------------------------------------
# THINKING / PLAN (auditable comments):
# 1) INPUT: Read the merged Parquet (quotes-left asof join) that already has TS, PRICE, SIZE,
#    BID/ASK/MID/SPREAD/DEPTH_TOB, etc.
# 2) FILTER: Work lazily in Polars; analyze one ticker at a time. For cost/impact metrics we
#    must have an actual execution → keep only rows where PRICE and SIZE are NOT NULL.
# 3) ORDERING: Sort by TS to define event-time order for lag/lead operations.
# 4) SIDE: Lee–Ready–style proxy: SIDE = +1 if PRICE ≥ MID else –1 (first pass heuristic).
# 5) ARRIVAL MID: ARR_MID = previous MID (lag); if missing (first row), fall back to current MID.
# 6) METRICS (per-trade, in bps = ×10,000):
#    • IS_BPS  = SIDE * (PRICE - ARR_MID) / ARR_MID * 10,000
#    • MI_BPS  = SIDE * (MID_now - ARR_MID) / ARR_MID * 10,000
#    • MR_BPS  = -SIDE * (MID_fwd - MID_now) / MID_now * 10,000, where MID_fwd is K-trade lead.
# 7) SUMMARY: Count, total size, VWAP, average spread/depth, and min/median/mean/max for IS/MI/MR.
# 8) NUMERICS: We don’t coerce away NaN/inf—these reveal edge cases (e.g., denominators ~0, edges).
# 9) SCALING: Output costs in basis points.
# 10) EXTENSIBILITY: Ticker list is in driver(); MR horizon K_FWD_TRADES is configurable here.
# ------------------------------------------------------------------------------

import time
from pathlib import Path
import polars as pl

# -------- CONFIG --------
MERGED_FILE = Path("/home/amazon/Documents/TAQData/merged_output_top50/merged_all.parquet")
OUT_DIR = Path("./ms_out")
OUT_DIR.mkdir(exist_ok=True)

pl.Config.set_tbl_cols(120)
pl.Config.set_tbl_width_chars(240)

# Event-time lead for MR (number of trades ahead)
K_FWD_TRADES = 10


def analyze_ticker(ticker: str):
    """
    Compute IS / MI / MR distribution stats for one ticker (lazy until collect()).

    Assumptions:
    - Dataset is quote-dominant (quotes-left asof). Rows without executions may have PRICE/SIZE null.
    - IS/MI/MR are *execution* metrics → we restrict to rows with PRICE and SIZE present.
    """
    t0 = time.time()
    ticker = ticker.upper()
    print(f"\n🔎 Microstructure analysis for: {ticker}")

    lf = pl.scan_parquet(str(MERGED_FILE))

    # Base slice for this ticker with required columns and event-time ordering
    base = (
        lf.filter(pl.col("SYM_ROOT") == ticker)
          .select([
              "TS","SYM_ROOT","SYM_SUFFIX",
              "PRICE","SIZE",
              "BID","ASK","MID","SPREAD","DEPTH_TOB","BIDSIZ","ASKSIZ"
          ])
          .sort("TS")
    )

    # Keep only rows that represent actual trades (PRICE & SIZE not null) for IS/MI/MR
    traded = base.filter(pl.col("PRICE").is_not_null() & pl.col("SIZE").is_not_null())

    metrics_lf = (
        traded.with_columns([
            # Trade sign proxy
            pl.when(pl.col("PRICE") >= pl.col("MID")).then(1).otherwise(-1).alias("SIDE"),
            # Arrival mid: previous mid; fallback to current mid
            pl.col("MID").shift(1).alias("MID_LAG"),
        ])
        .with_columns([
            pl.when(pl.col("MID_LAG").is_not_null()).then(pl.col("MID_LAG")).otherwise(pl.col("MID")).alias("ARR_MID"),
        ])
        .with_columns([
            # Implementation Shortfall (bps)
            pl.when(pl.col("ARR_MID") > 0)
              .then(1_0000 * pl.col("SIDE") * (pl.col("PRICE") - pl.col("ARR_MID")) / pl.col("ARR_MID"))
              .otherwise(None)
              .alias("IS_BPS"),
            # Instantaneous Market Impact (bps)
            pl.when(pl.col("ARR_MID") > 0)
              .then(1_0000 * pl.col("SIDE") * (pl.col("MID") - pl.col("ARR_MID")) / pl.col("ARR_MID"))
              .otherwise(None)
              .alias("MI_BPS"),
        ])
        .with_columns([
            # Forward mid: event-time lead by K trades
            pl.col("MID").shift(-K_FWD_TRADES).alias("MID_FWD"),
        ])
        .with_columns([
            # Mean Reversion (bps)
            pl.when(pl.col("MID") > 0)
              .then(1_0000 * (-pl.col("SIDE")) * (pl.col("MID_FWD") - pl.col("MID")) / pl.col("MID"))
              .otherwise(None)
              .alias("MR_BPS"),
        ])
    )

    summary_lf = (
        metrics_lf.select([
            pl.len().alias("N_ROWS"),
            pl.col("SIZE").sum().alias("TOTAL_SIZE"),
            # VWAP of executions
            ((pl.col("PRICE") * pl.col("SIZE")).sum() / pl.col("SIZE").sum()).alias("VWAP"),
            # Liquidity context from quotes at execution instants
            (pl.col("SPREAD").mean() * 1_0000).alias("AVG_SPREAD_BPS"),
            pl.col("DEPTH_TOB").mean().alias("AVG_DEPTH_TOB"),

            # IS stats
            pl.col("IS_BPS").mean().alias("IS_MEAN_BPS"),
            pl.col("IS_BPS").median().alias("IS_MEDIAN_BPS"),
            pl.col("IS_BPS").min().alias("IS_MIN_BPS"),
            pl.col("IS_BPS").max().alias("IS_MAX_BPS"),

            # MI stats
            pl.col("MI_BPS").mean().alias("MI_MEAN_BPS"),
            pl.col("MI_BPS").median().alias("MI_MEDIAN_BPS"),
            pl.col("MI_BPS").min().alias("MI_MIN_BPS"),
            pl.col("MI_BPS").max().alias("MI_MAX_BPS"),

            # MR stats
            pl.col("MR_BPS").mean().alias("MR_MEAN_BPS"),
            pl.col("MR_BPS").median().alias("MR_MEDIAN_BPS"),
            pl.col("MR_BPS").min().alias("MR_MIN_BPS"),
            pl.col("MR_BPS").max().alias("MR_MAX_BPS"),
        ])
        .with_columns([
            pl.lit(ticker).alias("TICKER"),
            pl.lit(K_FWD_TRADES).alias("FWD_TRADES_FOR_MR"),
        ])
        .select([
            "TICKER","FWD_TRADES_FOR_MR",
            "N_ROWS","TOTAL_SIZE","VWAP","AVG_SPREAD_BPS","AVG_DEPTH_TOB",
            "IS_MEAN_BPS","IS_MEDIAN_BPS","IS_MIN_BPS","IS_MAX_BPS",
            "MI_MEAN_BPS","MI_MEDIAN_BPS","MI_MIN_BPS","MI_MAX_BPS",
            "MR_MEAN_BPS","MR_MEDIAN_BPS","MR_MIN_BPS","MR_MAX_BPS",
        ])
    )

    summary_df = summary_lf.collect(streaming=True)
    print(summary_df)

    out_csv = OUT_DIR / f"ms_summary_{ticker}.csv"
    summary_df.write_csv(out_csv)
    print(f"💾 Saved → {out_csv.resolve()}  |  ⏱ {time.time()-t0:.2f}s")

    # expose for interactive sessions if desired
    globals()['summary_df'] = summary_df
    globals()['metrics_lf'] = metrics_lf


def driver():
    """Driver controls which tickers to analyze; extend this list as needed."""
    tickers = ["UBER"]
    for t in tickers:
        analyze_ticker(t)


if __name__ == "__main__":
    driver()
