"""The single ``Metric`` contract every scorer implements, plus ``BaseMetric`` with shared plumbing.

In-house checks and OSS adapters (BERTScore, an LLM judge) all become metrics by subclassing
:class:`BaseMetric` and implementing ``_compute``. BaseMetric handles required-field validation, the
sync/async bridge, optional [0, 1] normalization, latency capture, and turning exceptions into
``MetricResult(error=...)`` so one metric can never crash a whole run.
"""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter
from typing import Protocol, runtime_checkable

from agent_eval.core.contracts import Aggregation, CostClass, EvalContext, MetricResult


@runtime_checkable
class Metric(Protocol):
    """Structural type for anything the runner can score with."""

    name: str
    requires: frozenset[str]
    cost_class: CostClass
    aggregation: Aggregation
    higher_is_better: bool
    unit_interval: bool  # are scores bounded to [0, 1]? (False for latency/token magnitudes)

    def score(self, ctx: EvalContext) -> MetricResult: ...
    async def ascore(self, ctx: EvalContext) -> MetricResult: ...


class BaseMetric:
    """Concrete base implementing the shared metric plumbing.

    Subclasses set the class attributes and implement ``_compute`` (sync). Override ``_acompute``
    for a genuinely async implementation; by default it bridges to ``_compute``.
    """

    name: str = ""
    requires: frozenset[str] = frozenset()
    cost_class: CostClass = CostClass.FREE
    aggregation: Aggregation = Aggregation.MEAN
    higher_is_better: bool = True
    unit_interval: bool = True  # clamp score to [0, 1]? (False for raw values like latency/tokens)

    # --- subclass hooks ---------------------------------------------------
    def _compute(self, ctx: EvalContext) -> MetricResult:
        raise NotImplementedError

    async def _acompute(self, ctx: EvalContext) -> MetricResult:
        return self._compute(ctx)

    # --- public API -------------------------------------------------------
    def score(self, ctx: EvalContext) -> MetricResult:
        missing = self._missing(ctx)
        if missing:
            return self._missing_result(missing)
        t0 = perf_counter()
        try:
            result = self._compute(ctx)
        except Exception as exc:  # never let one metric crash a whole run
            return MetricResult(self.name, 0.0, None, error=repr(exc))
        return self._finalize(result, (perf_counter() - t0) * 1000.0)

    async def ascore(self, ctx: EvalContext) -> MetricResult:
        missing = self._missing(ctx)
        if missing:
            return self._missing_result(missing)
        t0 = perf_counter()
        try:
            result = await self._acompute(ctx)
        except Exception as exc:
            return MetricResult(self.name, 0.0, None, error=repr(exc))
        return self._finalize(result, (perf_counter() - t0) * 1000.0)

    # --- helpers ----------------------------------------------------------
    def _missing(self, ctx: EvalContext) -> list[str]:
        missing: list[str] = []
        for name in self.requires:
            val = getattr(ctx, name, None)
            if val is None:
                missing.append(name)
            elif isinstance(val, (str, tuple, list)) and len(val) == 0:
                missing.append(name)
        return missing

    def _missing_result(self, missing: list[str]) -> MetricResult:
        return MetricResult(
            self.name, 0.0, None, error=f"missing required context fields: {sorted(missing)}"
        )

    def _finalize(self, result: MetricResult, latency_ms: float) -> MetricResult:
        score = float(result.score)
        if self.unit_interval:
            score = min(max(score, 0.0), 1.0)
        cost = result.cost
        if cost.latency_ms == 0.0:
            cost = replace(cost, latency_ms=latency_ms)
        return replace(result, score=score, cost=cost)
