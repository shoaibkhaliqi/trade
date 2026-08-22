"""Parquet persistence for processed OHLCV datasets.

One file per (symbol, timeframe): data/processed/{SYMBOL}_{timeframe}.parquet
Dataset lineage (source, window, row count, save time) is embedded into the
Parquet file metadata so a stray file can always be identified.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from darwin.data.schema import OHLCV_COLUMNS, TIMESTAMP_COL, to_canonical_timestamps


class DataStorage:
    """Reads/writes canonical OHLCV frames with embedded lineage metadata."""

    def __init__(self, base_dir: str | Path = "data/processed") -> None:
        self.base_dir = Path(base_dir)

    def path_for(self, symbol: str, timeframe: str) -> Path:
        return self.base_dir / f"{symbol.upper()}_{timeframe}.parquet"

    def save(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        metadata: dict[str, Any] | None = None,
        *,
        required_columns: list[str] | None = None,
    ) -> Path:
        req = OHLCV_COLUMNS if required_columns is None else required_columns
        missing = [c for c in req if c not in df.columns]
        if missing:
            msg = f"refusing to save: missing columns {missing}"
            raise ValueError(msg)

        path = self.path_for(symbol, timeframe)
        path.parent.mkdir(parents=True, exist_ok=True)

        meta = {
            "symbol": symbol.upper(),
            "timeframe": timeframe,
            "rows": len(df),
            "first_timestamp": str(df[TIMESTAMP_COL].iloc[0]) if len(df) else "",
            "last_timestamp": str(df[TIMESTAMP_COL].iloc[-1]) if len(df) else "",
            "saved_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        if metadata:
            meta.update(metadata)
        arrow_meta = {
            f"darwin.{k}": json.dumps(v, default=str) for k, v in meta.items()
        }

        table = pa.Table.from_pandas(df, preserve_index=False)
        table = table.replace_schema_metadata(
            {**table.schema.metadata, **{k.encode(): v.encode() for k, v in arrow_meta.items()}}
        )
        pq.write_table(table, path)
        return path

    def load(self, symbol: str, timeframe: str) -> pd.DataFrame:
        path = self.path_for(symbol, timeframe)
        if not path.exists():
            msg = f"dataset not found: {path}"
            raise FileNotFoundError(msg)
        df = pq.read_table(path).to_pandas()
        # normalize dtype that Parquet may round-trip at microsecond resolution
        df[TIMESTAMP_COL] = to_canonical_timestamps(df[TIMESTAMP_COL])
        return df.sort_values(TIMESTAMP_COL).reset_index(drop=True)

    def read_lineage(self, symbol: str, timeframe: str) -> dict[str, Any]:
        """Return the darwin.* metadata embedded in the file, if any."""
        path = self.path_for(symbol, timeframe)
        schema_meta = pq.read_schema(path).metadata or {}
        out: dict[str, Any] = {}
        for key_bytes, value_bytes in schema_meta.items():
            key = key_bytes.decode()
            if key.startswith("darwin."):
                try:
                    out[key.removeprefix("darwin.")] = json.loads(value_bytes)
                except json.JSONDecodeError:
                    out[key.removeprefix("darwin.")] = value_bytes.decode(errors="replace")
        return out
