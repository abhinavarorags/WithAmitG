import polars as pl
from pathlib import Path
from tqdm import tqdm
import os

TRADES_GLOB = "/home/amazon/Documents/TAQData/processed_output_trades/*.parquet"
OUT_DIR = Path("/home/amazon/Documents/TAQData/processed_output_trades_upper/")
OUT_DIR.mkdir(parents=True, exist_ok=True)

for file in tqdm(list(Path("/home/amazon/Documents/TAQData/processed_output_trades/").glob("*.parquet")),
                 desc="Renaming to uppercase"):
    try:
        df = pl.read_parquet(file)
        df = df.rename({c: c.upper() for c in df.columns})
        out_file = OUT_DIR / file.name
        df.write_parquet(out_file, compression="zstd")
    except Exception as e:
        print(f"⚠️ Skipped {file.name}: {e}")

print(f"✅ Done. Uppercase trades parquet files saved in: {OUT_DIR}")

