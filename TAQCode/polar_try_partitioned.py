import polars as pl
import pyarrow.parquet as pq
import pyarrow.dataset as ds
from pathlib import Path
import os
import time

# --- Configuration ---
#FILE_PATH = "/home/amazon/Documents/TAQData/data_trades_2023_12_15.parquet"
#OUTPUT_DIR = "/home/amazon/Documents/TAQData/processed_output_trades/"
FILE_PATH = "/home/amazon/Documents/TAQData/data_quotes_2023_12_15.parquet"
OUTPUT_DIR = "/home/amazon/Documents/TAQData/processed_output_quotes/"

# ALL columns will be used since the 'columns' argument is omitted in read_row_group.
# REQUIRED_COLUMNS = ["timestamp", "symbol", "price", "size"]
# ---

# 1. Setup the output directory
output_path = Path(OUTPUT_DIR)
output_path.mkdir(parents=True, exist_ok=True) 

print(f"Starting chunked read from: {FILE_PATH}")
print(f"Saving output to directory: {OUTPUT_DIR}")

total_rows_read = 0
chunk_count = 0
start_time = time.time()

try:
    parquet_file = pq.ParquetFile(FILE_PATH)
    
    # Iterate through each Row Group in the source Parquet file
    num_row_groups = parquet_file.num_row_groups
    print(f"File contains {num_row_groups} row groups (chunks).")

    for i in range(num_row_groups):
        
        # 2. Read one Row Group (the chunk)
        # Reading ALL columns as requested by the user.
        # This may increase memory usage compared to only reading a subset of columns.
        pa_table = parquet_file.read_row_group(i)
        
        # 3. Convert to Polars DataFrame (Optional, but useful for processing)
        batch_df = pl.from_arrow(pa_table)
        
        # --- Perform any necessary processing (e.g., filtering, calculation) ---
        # processed_df = batch_df.filter(pl.col("price") > 100)
        processed_df = batch_df # Using the raw chunk for this example
        # -----------------------------------------------------------------------

        # 4. Save the chunk as a new, small Parquet file
        chunk_count += 1
        output_file = output_path / f"chunk_{chunk_count:04d}.parquet"
        
        # Write the processed Polars DataFrame back to a Parquet file
        processed_df.write_parquet(output_file)
        
        chunk_size = len(processed_df)
        total_rows_read += chunk_size
        
        print(f"✅ Chunk {chunk_count:04d}/{num_row_groups:04d} saved: {chunk_size} rows.")
        
    end_time = time.time()
    
    print("-" * 30)
    print(f"🎉 Finished reading and saving.")
    print(f"Total rows saved: {total_rows_read}")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    print(f"The final dataset consists of {chunk_count} files in the output directory.")
    
except Exception as e:
    print(f"❌ An error occurred during file reading or writing: {e}")

