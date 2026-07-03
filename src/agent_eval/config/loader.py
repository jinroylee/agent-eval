"""Load, validate, and project the YAML config into runnable framework objects."""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import yaml

from agent_eval.config.schema import ConfigModel
from agent_eval.core.errors import ConfigError
from agent_eval.core.gate import GatePolicy
from agent_eval.core.registry import BuildContext, MetricRegistry
from agent_eval.core.suite import Suite
from agent_eval.datasets.base import DatasetSpec


def load_config(path: str | Path) -> ConfigModel:
    """Read YAML (with ``${ENV}`` interpolation) and validate it against the schema."""
    text = Path(path).read_text(encoding="utf-8")
    text = os.path.expandvars(text)  # ${VAR} / $VAR from the environment
    data = yaml.safe_load(text) or {}
    return ConfigModel(**data)


def import_attr(dotted: str) -> Any:
    """Resolve an attribute from either a dotted module path or a file path.

    Accepts ``"package.module:attr"`` (installed/importable code) or ``"path/to/file.py:attr"``
    (a script, e.g. an example agent). For a file path the script's directory is put on ``sys.path``
    first, so it can import its own siblings.
    """
    module_part, sep, attr = dotted.partition(":")
    if not sep:
        module_part, _, attr = dotted.rpartition(".")
    if not module_part or not attr:
        raise ConfigError(f"expected 'module:attr' (or 'file.py:attr'), got {dotted!r}")

    if module_part.endswith(".py") or "/" in module_part or os.sep in module_part:
        path = Path(module_part).resolve()
        if not path.exists():
            raise ConfigError(f"graph/judge file not found: {path}")
        sys.path.insert(0, str(path.parent))
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise ConfigError(f"could not load module from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_part)
    return getattr(module, attr)


def build_dataset_spec(cfg: ConfigModel, name: str) -> DatasetSpec:
    if name not in cfg.datasets:
        raise ConfigError(f"unknown dataset {name!r}; known: {sorted(cfg.datasets)}")
    dm = cfg.datasets[name]
    return DatasetSpec(
        adapter=dm.adapter,
        path=dm.path,
        format=dm.format,
        field_map=dict(dm.field_map),
        include_unmapped=dm.include_unmapped,
    )


def resolve_judge(cfg: ConfigModel) -> tuple[Any, tuple[Any, ...]]:
    """Resolve ``(primary_backend, panel)`` from ``cfg.judge.factory``; ``(None, ())`` if unset.

    The factory may be a JudgeBackend, a zero-arg callable returning one, or a list (= a PoLL panel).
    """
    if not cfg.judge or not cfg.judge.factory:
        return None, ()
    obj = import_attr(cfg.judge.factory)
    result = obj() if (callable(obj) and not hasattr(obj, "evaluate")) else obj
    if isinstance(result, (list, tuple)):
        backends = list(result)
        if not backends:
            raise ConfigError("judge factory returned an empty panel")
        return backends[0], tuple(backends[1:])
    return result, ()


def build_context(cfg: ConfigModel) -> BuildContext:
    judge, panel = resolve_judge(cfg)
    return BuildContext(judge=judge, panel=panel, defaults=dict(cfg.defaults))


def build_suite(cfg: ConfigModel, category: str, registry: MetricRegistry) -> Suite:
    if category not in cfg.suites:
        raise ConfigError(f"unknown suite {category!r}; known: {sorted(cfg.suites)}")
    sm = cfg.suites[category]
    bctx = build_context(cfg)
    metrics = [registry.build(spec, bctx) for spec in sm.metrics]
    gate = GatePolicy(thresholds=dict(sm.gate.thresholds), require_pass=list(sm.gate.require_pass))
    return Suite(cfg.agent_type, category, metrics, gate)
