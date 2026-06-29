"""Tabular dataset adapter: CSV / TSV / Excel / Parquet via pandas."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path

import pandas as pd

from agent_eval.core.contracts import EvalContext
from agent_eval.core.errors import ConfigError
from agent_eval.datasets.base import record_to_context

_EXT_FORMAT = {".csv": "csv", ".tsv": "tsv", ".xlsx": "excel", ".xls": "excel", ".parquet": "parquet"}


def _read_df(path: str, format: str | None) -> pd.DataFrame:
    fmt = format or _EXT_FORMAT.get(Path(path).suffix.lower())
    if fmt == "csv":
        return pd.read_csv(path)
    if fmt == "tsv":
        return pd.read_csv(path, sep="\t")
    if fmt == "excel":
        return pd.read_excel(path)
    if fmt == "parquet":
        return pd.read_parquet(path)
    raise ConfigError(f"unknown/unsupported tabular format for {path!r}: {fmt!r}")


def read_tabular_records(path: str, format: str | None = None) -> list[dict]:
    """Read a tabular file into raw dict records (pandas NaN/NA normalized to None)."""
    df = _read_df(path, format)
    df = df.astype(object).where(pd.notnull(df), None)
    return df.to_dict(orient="records")


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

    def __iter__(self) -> Iterator[EvalContext]:
        for rec in read_tabular_records(self.path, self.format):
            yield record_to_context(rec, self.field_map, self.include_unmapped)
