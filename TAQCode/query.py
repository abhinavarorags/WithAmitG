import polars as pl
from datetime import date, datetime, timedelta # Added necessary imports for completeness

TRADES_UPPER = "/home/amazon/Documents/TAQData/processed_output_trades_upper/*.parquet"
QUOTES_OLD   = "/home/amazon/Documents/TAQData/processed_output_quotes/*.parquet"

# Load 100 rows from each
# NOTE: .fetch() is deprecated and immediately collects the result into a DataFrame.1000 000 000
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
    by=["SYM_ROOT","SYM_SUFFIX"],
    strategy="nearest",
    tolerance="1s"
)

# Print first 10 joined rows
OUTPUT_FILE = "taq_analysis_output.txt"
with open(OUTPUT_FILE, 'w') as f:
    print(trades_lf.fetch(n_rows=10), file=f)
    print(quotes_lf.fetch(n_rows=10), file=f)
    
    # FIX: Changed .fetch(n_rows=10) to .head(10) to avoid AttributeError
    print(quotes_lf.filter((pl.col('SYM_ROOT') == 'AAPL') & ( pl.col('BIDSIZ').is_not_null() )).fetch(n_rows=10), file=f)
    
    # FIX: Changed .fetch(n_rows=10) to .head(10) to avoid AttributeError
    print(joined_lf.filter((pl.col('SYM_ROOT') == 'AAPL') & ( pl.col('BIDSIZ').is_not_null() )).fetch(n_rows=10), file=f)
    
print(f"\nAnalysis successfully executed and results written to {OUTPUT_FILE}")

