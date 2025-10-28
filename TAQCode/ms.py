# ms.py — Microstructure stats (IS, MI, MR) for a given ticker from merged_all.parquet
# ------------------------------------------------------------------------------
# THINKING / PLAN (baked into comments so it’s fully auditable later):
# 1) INPUT: We read the single-file merged Parquet produced earlier (quotes-left asof join),
#    which already contains: TS, PRICE, SIZE, BID/ASK/MID/SPREAD/DEPTH_TOB, etc.
# 2) FILTER: Work lazily in Polars; filter to one ticker at a time (driver controls the list).
# 3) ORDERING: Sort by TS to define a consistent event-time ordering (needed for lag/lead ops).
# 4) SIDE: Use a simple Lee–Ready style proxy: SIDE = +1 if trade price ≥ mid; else –1.
#    (Comment: This is a common heuristic. It’s imperfect but good enough for first pass.)
# 5) ARRIVAL MID: Use previous trade’s mid as ARR_MID (lag by one). If it’s null (first row),
#    fallback to current mid. This proxies an “arrival” snapshot for cost calculations.
# 6) METRICS (per-trade, bps):
#    • IS_BPS  = 10,000 * SIDE * (PRICE - ARR_MID) / ARR_MID
#      - Interpretation: signed implementation shortfall vs. the arrival mid.
#    • MI_BPS  = 10,000 * SIDE * (MID_now - ARR_MID) / ARR_MID
#      - Interpretation: instantaneous (quote) market impact relative to arrival.
#    • MR_BPS  = 10,000 * ( -SIDE ) * ( MID_fwd - MID_now ) / MID_now
#      - Interpretation: short-horizon mean reversion in direction opposite to SIDE.
#      - We approximate “forward” mid with an event-time lead of K_FWD_TRADES trades.
# 7) SUMMARY: Report count, total size, VWAP, average spread/depth, and min/median/mean/max
#    for IS/MI/MR. (Median is more robust to outliers than mean.)
# 8) NUMERICS: We *don’t* coerce away NaN/inf here to keep the pipeline honest. If some
#    denominators are zero/missing (e.g., ARR_MID≈0 or missing MID_fwd near file edges),
#    you may see NaN/inf in summaries. It’s expected; clean downstream if needed.
# 9) SCALING: All cost metrics output in basis points (×10,000) for industry familiarity.
# 10) EXTENSIBILITY: Driver controls the ticker list; add more tickers easily. You can also
#     switch MR horizon (K_FWD_TRADES) here without touching logic elsewhere.
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

# EVENT-TIME HORIZON for MR: lead the mid by K trades (not wall-clock minutes)
K_FWD_TRADES = 10  # Tweak to 5/10/20 depending on how “short-horizon” you want MR


def analyze_ticker(ticker: str):
    """
    Compute IS / MI / MR distribution stats for one ticker (fully lazy until collect()).

    NOTE ON ASSUMPTIONS:
    - The merged parquet is quote-dominant (quotes-left asof join). There may be quotes with
      no matched trade → PRICE/SIZE can be null. That’s fine; IS/MI/MR are conditional on mids.
    - SIDE uses PRICE vs MID_now; if PRICE is null, SIDE will be -1 (via the when/otherwise),
      but cost formulas also require ARR_MID/MID_now denominators > 0 to emit a value.
    - We *do not* fill NaNs here to avoid hiding data issues; downstream cleaning is optional.
    """
    t0 = time.time()
    ticker = ticker.upper()
    print(f"\n🔎 Microstructure analysis for: {ticker}")

    # Load lazily
    lf = pl.scan_parquet(str(MERGED_FILE))

    # Filter + select only needed columns; sort by TS for lag/lead semantics
    base = (
        lf.filter(pl.col("SYM_ROOT") == ticker)
          .select([
              "TS","SYM_ROOT","SYM_SUFFIX",
              "PRICE","SIZE",
              "BID","ASK","MID","SPREAD","DEPTH_TOB","BIDSIZ","ASKSIZ"
          ])
          .sort("TS")  # IMPORTANT: define event-time order for lag/lead ops
    )

    # Build SIDE and ARR_MID, then compute IS/MI (bps); finally compute forward MID and MR (bps)
    metrics_lf = (
        base.with_columns([
            # SIDE heuristic: +1 buy if execution at/above mid; else -1 sell
            pl.when(pl.col("PRICE") >= pl.col("MID")).then(1).otherwise(-1).alias("SIDE"),

            # Previous mid as arrival; fallback to current mid at sequence start
            pl.col("MID").shift(1).alias("MID_LAG"),
        ])
        .with_columns([
            pl.when(pl.col("MID_LAG").is_not_null()).then(pl.col("MID_LAG")).otherwise(pl.col("MID")).alias("ARR_MID"),
        ])
        .with_columns([
            # Implementation Shortfall (bps): SIDE * (exec - arrival_mid) / arrival_mid
            pl.when(pl.col("ARR_MID") > 0)
              .then(1_0000 * pl.col("SIDE") * (pl.col("PRICE") - pl.col("ARR_MID")) / pl.col("ARR_MID"))
              .otherwise(None)
              .alias("IS_BPS"),

            # Market Impact (bps): SIDE * (mid_now - arrival_mid) / arrival_mid
            pl.when(pl.col("ARR_MID") > 0)
              .then(1_0000 * pl.col("SIDE") * (pl.col("MID") - pl.col("ARR_MID")) / pl.col("ARR_MID"))
              .otherwise(None)
              .alias("MI_BPS"),
        ])
        .with_columns([
            # Forward mid (event time): lead by K_FWD_TRADES trades
            pl.col("MID").shift(-K_FWD_TRADES).alias("MID_FWD"),
        ])
        .with_columns([
            # Mean Reversion (bps): negative direction of SIDE * relative forward move
            # MR_BPS = -SIDE * (MID_fwd - MID_now) / MID_now * 10,000
            pl.when(pl.col("MID") > 0)
              .then(1_0000 * (-pl.col("SIDE")) * (pl.col("MID_FWD") - pl.col("MID")) / pl.col("MID"))
              .otherwise(None)
              .alias("MR_BPS"),
        ])
    )

    # Summarize distribution statistics for IS/MI/MR (min/median/mean/max) + basic liquidity context
    summary_lf = (
        metrics_lf.select([
            pl.len().alias("N_ROWS"),
            pl.col("SIZE").sum().alias("TOTAL_SIZE"),
            # VWAP with SIZE as weight (guarded by denominator via Polars’ NA handling)
            ((pl.col("PRICE") * pl.col("SIZE")).sum() / pl.col("SIZE").sum()).alias("VWAP"),
            # Average quoted spread (bps) and depth at top-of-book
            (pl.col("SPREAD").mean() * 1_0000).alias("AVG_SPREAD_BPS"),
            pl.col("DEPTH_TOB").mean().alias("AVG_DEPTH_TOB"),

            # Implementation Shortfall stats
            pl.col("IS_BPS").mean().alias("IS_MEAN_BPS"),
            pl.col("IS_BPS").median().alias("IS_MEDIAN_BPS"),
            pl.col("IS_BPS").min().alias("IS_MIN_BPS"),
            pl.col("IS_BPS").max().alias("IS_MAX_BPS"),

            # Market Impact stats
            pl.col("MI_BPS").mean().alias("MI_MEAN_BPS"),
            pl.col("MI_BPS").median().alias("MI_MEDIAN_BPS"),
            pl.col("MI_BPS").min().alias("MI_MIN_BPS"),
            pl.col("MI_BPS").max().alias("MI_MAX_BPS"),

            # Mean Reversion stats
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

    # Execute lazily with streaming
    summary_df = summary_lf.collect(streaming=True)
    print(summary_df)

    # Persist summary
    out_csv = OUT_DIR / f"ms_summary_{ticker}.csv"
    summary_df.write_csv(out_csv)
    print(f"💾 Saved → {out_csv.resolve()}  |  ⏱ {time.time()-t0:.2f}s")


def driver():
    """
    Driver controls which tickers to analyze.
    - Add/remove symbols here (e.g., ["UBER", "AAPL", "SPY"]).
    - This avoids CLI parsing and keeps runbooks simple/explicit.
    """
    tickers = ["UBER"]
    for t in tickers:
        analyze_ticker(t)


if __name__ == "__main__":
    driver()

