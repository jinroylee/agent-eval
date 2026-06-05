"""Tabular dataset adapter: CSV / Excel / Parquet via pandas."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path

import pandas as pd

from agent_eval.core.contracts import EvalContext
from agent_eval.core.errors import ConfigError
from agent_eval.datasets.base import record_to_context

_EXT_FORMAT = {
    ".csv": "csv",
    ".tsv": "tsv",
    ".xlsx": "excel",
    ".xls": "excel",
    ".parquet": "parquet",
}


class TabularAdapter:
    def __init__(
        self,
        path: str,
        field_map: Mapping[str, str] | None = None,
        format: str | None = None,
        include_unmapped: bool = True,
    ) -> None:
        self.path = path
        self.field_map = field_map or {}
        self.format = format
        self.include_unmapped = include_unmapped

    def _read(self) -> pd.DataFrame:
        fmt = self.format or _EXT_FORMAT.get(Path(self.path).suffix.lower())
        if fmt == "csv":
            return pd.read_csv(self.path)
        if fmt == "tsv":
            return pd.read_csv(self.path, sep="\t")
        if fmt == "excel":
            return pd.read_excel(self.path)
        if fmt == "parquet":
            return pd.read_parquet(self.path)
        raise ConfigError(f"unknown/unsupported tabular format for {self.path!r}: {fmt!r}")

    def __iter__(self) -> Iterator[EvalContext]:
        df = self._read()
        # Normalize pandas NaN/NA to None so "missing" reads as missing, not a float nan.
        df = df.astype(object).where(pd.notnull(df), None)
        for rec in df.to_dict(orient="records"):
            yield record_to_context(rec, self.field_map, self.include_unmapped)
