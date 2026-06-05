"""Load, validate, and project the YAML config into runnable framework objects.

The same config file is read by the offline runner and (later) the runtime critic; calibrated
thresholds round-trip back into ``runtime_critic.tau`` via ``write_calibrated_tau`` so offline and
runtime stay consistent.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

import yaml

from agent_eval.config.schema import ConfigModel
from agent_eval.core.contracts import EvalTarget, Level
from agent_eval.core.gate import GatePolicy
from agent_eval.core.registry import MetricRegistry
from agent_eval.core.suite import Suite
from agent_eval.datasets.base import DatasetSpec


def load_config(path: str | Path) -> ConfigModel:
    """Read YAML (with ``${ENV}`` interpolation) and validate it against the schema."""
    text = Path(path).read_text()
    text = os.path.expandvars(text)  # ${VAR} / $VAR from the environment
    data = yaml.safe_load(text) or {}
    return ConfigModel(**data)


def build_dataset_spec(cfg: ConfigModel, name: str) -> DatasetSpec:
    dm = cfg.datasets[name]
    return DatasetSpec(
        adapter=dm.adapter,
        path=dm.path,
        format=dm.format,
        field_map=dict(dm.field_map),
        include_unmapped=dm.include_unmapped,
    )


def build_suite(cfg: ConfigModel, category: str, registry: MetricRegistry) -> Suite:
    sm = cfg.suites[category]
    metrics = [registry.build(spec) for spec in sm.metrics]
    gate = GatePolicy(
        thresholds=dict(sm.gate.thresholds),
        require_pass=list(sm.gate.require_pass),
        min_effect=sm.gate.min_effect,
        significance_alpha=sm.gate.significance_alpha,
        fdr=sm.gate.fdr,
    )
    target = EvalTarget(
        level=Level(sm.target.level), selector=sm.target.selector, attach=sm.target.attach
    )
    return Suite(cfg.agent_type, category, metrics, gate, target)


def write_calibrated_tau(path: str | Path, tau: Mapping[str, float]) -> None:
    """Round-trip calibrated operating points back into ``runtime_critic.tau`` of the config.

    This is the offline->runtime link: the critic reads exactly what the offline study calibrated.
    """
    p = Path(path)
    data = yaml.safe_load(p.read_text()) or {}
    data.setdefault("runtime_critic", {})["tau"] = dict(tau)
    p.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
