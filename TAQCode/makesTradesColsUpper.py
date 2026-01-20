# makesTradesColsUpper.py (type-conscious; preserves Arrow schema/time units)
#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def upper_schema(schema: pa.Schema) -> pa.Schema:
    # Keep exact field types/metadata; only rename
    fields = [pa.field(f.name.upper(), f.type, f.nullable, f.metadata) for f in schema]
    return pa.schema(fields, metadata=schema.metadata)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--INPUT", required=True)
    parser.add_argument("--OUTPUT", required=True)
    parser.add_argument("--ROW_GROUP_SIZE", type=int, default=256_000)  # tuning only
    args = parser.parse_args()

    in_path = Path(args.INPUT)
    out_path = Path(args.OUTPUT)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    pf = pq.ParquetFile(in_path)
    in_schema = pf.schema_arrow  # Arrow schema from file footer (authoritative)
    out_schema = upper_schema(in_schema)

    # Progress based on row groups (cheap)
    num_rgs = pf.num_row_groups
    pbar = tqdm(total=num_rgs, desc="Uppercasing column names (preserve types)")

    writer = pq.ParquetWriter(
        where=str(tmp_path),
        schema=out_schema,
        compression="zstd",
        write_statistics=True,
        use_dictionary=False,
    )

    try:
        for rg in range(num_rgs):
            # Read exactly one row group to avoid schema/unit changes
            table = pf.read_row_group(rg)  # preserves original Arrow types (e.g., time64[us])

            # Rename columns only (no casting)
            new_names = [name.upper() for name in table.schema.names]
            table2 = table.rename_columns(new_names)

            # Ensure it matches the declared output schema names/types
            # (types should already match; this mainly aligns schema object)
            table2 = table2.cast(out_schema)

            writer.write_table(table2, row_group_size=args.ROW_GROUP_SIZE)
            pbar.update(1)
    finally:
        writer.close()
        pbar.close()

    # Atomic replace
    tmp_path.replace(out_path)
    print(f"Done. Wrote: {out_path}")
    print("Input TIME_M type:", in_schema.field(in_schema.get_field_index("TIME_M")).type if "TIME_M" in in_schema.names else "N/A")
    print("Output TIME_M type:", out_schema.field(out_schema.get_field_index("TIME_M".upper())).type if "TIME_M" in in_schema.names else "N/A")


if __name__ == "__main__":
    main()
