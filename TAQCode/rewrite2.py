#!/usr/bin/env python3
# rewrite2.py
# Read ONE Parquet file -> write many chunk Parquet files, enforcing schema, preserving TIME_M in us,
# clearing OUT_DIR first, tqdm progress, per-batch try/except, atomic tmp rename, and rowcount check.

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


# -------------------- TARGET SCHEMAS --------------------
TARGET_PL = {
    "DATE": pl.Date,
    "TIME_M": pl.Time,
    "EX": pl.Utf8,
    "SYM_ROOT": pl.Utf8,
    "SYM_SUFFIX": pl.Utf8,
    "TR_SCOND": pl.Utf8,
    "SIZE": pl.Int64,
    "PRICE": pl.Float64,
    "TR_STOP_IND": pl.Utf8,
    "TR_CORR": pl.Utf8,
    "TR_SEQNUM": pl.Int64,
    "TR_ID": pl.Int64,
    "TR_SOURCE": pl.Utf8,
    "TR_RF": pl.Utf8,
}

# IMPORTANT: TIME_M is microseconds (us) to match your original input
TARGET_PA = pa.schema([
    pa.field("DATE", pa.date32()),
    pa.field("TIME_M", pa.time64("us")),
    pa.field("EX", pa.string()),
    pa.field("SYM_ROOT", pa.string()),
    pa.field("SYM_SUFFIX", pa.string()),
    pa.field("TR_SCOND", pa.string()),
    pa.field("SIZE", pa.int64()),
    pa.field("PRICE", pa.float64()),
    pa.field("TR_STOP_IND", pa.string()),
    pa.field("TR_CORR", pa.string()),
    pa.field("TR_SEQNUM", pa.int64()),
    pa.field("TR_ID", pa.int64()),
    pa.field("TR_SOURCE", pa.string()),
    pa.field("TR_RF", pa.string()),
])

# Columns that can arrive as Utf8/Int32/etc; strict=False => unparsable -> null instead of crashing
RELAX_NUMERIC = {"SIZE", "PRICE", "TR_SEQNUM", "TR_ID"}


def clear_out_dir(out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def parquet_rows(file_path: Path) -> int:
    return pq.ParquetFile(file_path).metadata.num_rows


def sum_parquet_rows(dir_path: Path) -> int:
    total = 0
    for f in dir_path.glob("*.parquet"):
        total += pq.ParquetFile(f).metadata.num_rows
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--IN_FILE", required=True)
    parser.add_argument("--OUT_DIR", required=True)
    parser.add_argument("--BATCH_ROWS", type=int, default=25_000)
    args = parser.parse_args()

    in_file = Path(args.IN_FILE)
    out_dir = Path(args.OUT_DIR)
    batch_rows = int(args.BATCH_ROWS)

    if not in_file.exists():
        raise FileNotFoundError(f"Input parquet file not found: {in_file}")

    # Clear OUT_DIR first to avoid "doubling" from old chunks
    clear_out_dir(out_dir)

    pf = pq.ParquetFile(in_file)
    total_rows = pf.metadata.num_rows if pf.metadata is not None else None
    total_batches = (total_rows + batch_rows - 1) // batch_rows if total_rows else None

    skipped = []
    written = 0

    pbar = tqdm(total=total_batches, desc="Rewrite2 (single->chunks)") if total_batches else tqdm(desc="Rewrite2 (single->chunks)")

    # Read in Arrow batches directly from Parquet
    for i, rb in enumerate(pf.iter_batches(batch_size=batch_rows), start=1):
        try:
            # RecordBatch -> Table
            t_in = pa.Table.from_batches([rb])

            # Use Polars to add missing columns and cast reliably
            df = pl.from_arrow(t_in)

            # Add missing cols as typed nulls
            for c, dt in TARGET_PL.items():
                if c not in df.columns:
                    df = df.with_columns(pl.lit(None, dtype=dt).alias(c))

            # Cast to target dtypes
            df = (
                df.with_columns([
                    pl.col(c).cast(dt, strict=(c not in RELAX_NUMERIC)).alias(c)
                    for c, dt in TARGET_PL.items()
                ])
                .select(list(TARGET_PL.keys()))
            )

            # Force Arrow schema (authoritative)
            t_out = df.to_arrow().cast(TARGET_PA)

            out_name = f"chunk_{i:06d}.parquet"
            out_file = out_dir / out_name
            tmp_file = out_dir / (out_name + ".tmp")

            pq.write_table(
                t_out,
                tmp_file,
                compression="zstd",
                write_statistics=True,
                use_dictionary=False,
                row_group_size=t_out.num_rows,  # single row group per chunk
            )
            tmp_file.replace(out_file)

            written += 1

        except Exception as e:
            skipped.append((i, str(e)))

        pbar.update(1)

    pbar.close()

    print(f"\nDone. Written chunks: {written}")
    if skipped:
        print(f"Skipped chunks: {len(skipped)}")
        for idx, err in skipped[:20]:
            print(f"  chunk_{idx:06d}: {err}")
        if len(skipped) > 20:
            print("  ... (showing first 20)")

    # -------- Row count comparison (metadata-based) --------
    in_rows = parquet_rows(in_file)
    out_rows = sum_parquet_rows(out_dir)

    print("\nRow count check:")
    print(f"  IN_FILE : {in_rows}")
    print(f"  OUT_DIR : {out_rows}")
    print(f"  DIFF    : {out_rows - in_rows}")

    # Optional: quick scan sanity check
    _ = pl.scan_parquet(str(out_dir / "*.parquet"), schema=TARGET_PL).head(1).collect()
    print("Sanity check passed: scan_parquet works on rewritten chunks.")


if __name__ == "__main__":
    main()
# main(
    # IN_FILE="/Users/wanganbo/Desktop/new/data_trades_2024_03_15_upper.parquet",
    # OUT_DIR="/Users/wanganbo/Desktop/new/processed_output_trades_upper_clean_from_single/",
    # BATCH_ROWS=25_000,
#)
#
