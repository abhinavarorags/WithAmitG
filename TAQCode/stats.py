# stats_simple.py — Top-50 by volume (TRADES) and top-50 by rows (QUOTES), chunked & 8GB-friendly

import os
import time
from pathlib import Path
from collections import defaultdict

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import polars as pl

TRADES_DIR  = Path("/home/amazon/Documents/TAQData/processed_output_trades_upper")  # all-cols UPPERCASE
QUOTES_DIR  = Path("/home/amazon/Documents/TAQData/processed_output_quotes")
OUT_DIR     = Path("./stats_out_simple")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRADES_OUT_CSV = OUT_DIR / "top50_trades_by_volume.csv"
QUOTES_OUT_CSV = OUT_DIR / "top50_quotes_by_rows.csv"

def human(n: int) -> str:
    return f"{n:,}"

def list_parquet_files(d: Path):
    return sorted([p for p in d.glob("*.parquet") if p.is_file()])

def ensure_string(pa_tbl: pa.Table, col: str) -> pa.Table:
    if col in pa_tbl.column_names and not pa.types.is_string(pa_tbl[col].type):
        pa_tbl = pa_tbl.set_column(
            pa_tbl.schema.get_field_index(col), col, pc.cast(pa_tbl[col], pa.string())
        )
    return pa_tbl

# ----------------------------- TRADES: Top-50 by volume -----------------------------
def compute_trades_top50(trades_dir: Path):
    files = list_parquet_files(trades_dir)
    if not files:
        raise SystemExit(f"No parquet files in {trades_dir}")

    agg = defaultdict(lambda: {"rows": 0, "vol": 0})
    t0 = time.time()
    print(f"[TRADES] scanning {len(files)} files in {trades_dir} …")

    for fidx, f in enumerate(files, start=1):
        pf = pq.ParquetFile(str(f))
        n_rg = pf.num_row_groups
        print(f"  → {fidx}/{len(files)} {f.name}: {n_rg} row-groups")
        for rg in range(n_rg):
            # Only columns we need for volume/rows
            pa_tbl = pf.read_row_group(rg, columns=["SYM_ROOT", "SYM_SUFFIX", "SIZE"])
            # Ensure string dtypes; uppercase & fill suffix
            pa_tbl = ensure_string(pa_tbl, "SYM_ROOT")
            pa_tbl = ensure_string(pa_tbl, "SYM_SUFFIX")

            pl_df = pl.from_arrow(pa_tbl).with_columns([
                pl.col("SYM_ROOT").str.to_uppercase(),
                pl.col("SYM_SUFFIX").fill_null("").cast(pl.Utf8)
            ])

            chunk_agg = (
                pl_df.group_by(["SYM_ROOT", "SYM_SUFFIX"])
                     .agg([pl.len().alias("rows"), pl.col("SIZE").sum().alias("vol")])
            )

            for sym_root, sym_suffix, rows, vol in chunk_agg.iter_rows():
                key = (sym_root, sym_suffix)
                agg[key]["rows"] += int(rows or 0)
                agg[key]["vol"]  += int(vol  or 0)

            if (rg + 1) % 50 == 0 or rg == n_rg - 1:
                print(f"    RG {rg+1}/{n_rg} … unique symbols so far: {human(len(agg))}")

    elapsed = time.time() - t0
    print(f"[TRADES] done in {elapsed:.1f}s — symbols: {human(len(agg))}")

    rows = [(k[0], k[1], v["rows"], v["vol"]) for k, v in agg.items()]
    df = pl.DataFrame(rows, schema=["SYM_ROOT", "SYM_SUFFIX", "TRADES_ROWS", "TRADES_VOL"])
    df_top50 = df.sort("TRADES_VOL", descending=True).head(50)
    df_top50.write_csv(TRADES_OUT_CSV)
    print("\nTop-10 TRADES by volume:")
    print(df_top50.head(10))
    print(f"💾 Saved → {TRADES_OUT_CSV}")
    return df_top50

# ----------------------------- QUOTES: Top-50 by row count -----------------------------
def compute_quotes_top50(quotes_dir: Path):
    files = list_parquet_files(quotes_dir)
    if not files:
        raise SystemExit(f"No parquet files in {quotes_dir}")

    counts = defaultdict(int)
    t0 = time.time()
    print(f"\n[QUOTES] scanning {len(files)} files in {quotes_dir} …")

    for fidx, f in enumerate(files, start=1):
        pf = pq.ParquetFile(str(f))
        n_rg = pf.num_row_groups
        print(f"  → {fidx}/{len(files)} {f.name}: {n_rg} row-groups")
        for rg in range(n_rg):
            pa_tbl = pf.read_row_group(rg, columns=["SYM_ROOT", "SYM_SUFFIX"])
            pa_tbl = ensure_string(pa_tbl, "SYM_ROOT")
            pa_tbl = ensure_string(pa_tbl, "SYM_SUFFIX")

            pl_df = pl.from_arrow(pa_tbl).with_columns([
                pl.col("SYM_ROOT").str.to_uppercase(),
                pl.col("SYM_SUFFIX").fill_null("").cast(pl.Utf8)
            ])

            chunk_counts = (
                pl_df.group_by(["SYM_ROOT", "SYM_SUFFIX"])
                     .agg(pl.len().alias("rows"))
            )
            for sym_root, sym_suffix, rows in chunk_counts.iter_rows():
                counts[(sym_root, sym_suffix)] += int(rows or 0)

            if (rg + 1) % 100 == 0 or rg == n_rg - 1:
                seen = sum(counts.values())
                print(f"    RG {rg+1}/{n_rg} … cumulative QUOTES rows: {human(seen)}")

    elapsed = time.time() - t0
    print(f"[QUOTES] done in {elapsed:.1f}s — symbols: {human(len(counts))}")

    rows = [(k[0], k[1], v) for k, v in counts.items()]
    df = pl.DataFrame(rows, schema=["SYM_ROOT", "SYM_SUFFIX", "QUOTES_ROWS"])
    df_top50 = df.sort("QUOTES_ROWS", descending=True).head(50)
    df_top50.write_csv(QUOTES_OUT_CSV)
    print("\nTop-10 QUOTES by rows:")
    print(df_top50.head(10))
    print(f"💾 Saved → {QUOTES_OUT_CSV}")
    return df_top50

# ----------------------------- MAIN -----------------------------
if __name__ == "__main__":
    print(f"Output dir: {OUT_DIR.resolve()}")
    top50_trades = compute_trades_top50(TRADES_DIR)
    top50_quotes = compute_quotes_top50(QUOTES_DIR)
    print("\n✅ Done.")

