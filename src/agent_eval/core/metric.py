"""The single Metric contract every scorer implements, plus a BaseMetric with shared plumbing.

OSS libraries (DeepEval, RAGAS, agentevals, Inspect) and in-house checks all become metrics by
subclassing ``BaseMetric`` and implementing ``_compute`` (and optionally ``_acompute``). BaseMetric
handles requirement-checking, the sync/async bridge, score normalization, latency capture, and
turning exceptions into ``MetricResult(error=...)`` rather than crashing a run.
"""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter
from typing import Protocol, runtime_checkable

from agent_eval.core.contracts import CostClass, EvalContext, MetricResult, Mode, Tier

_DEFAULT_MODES = frozenset({Mode.OFFLINE, Mode.RUNTIME})


@runtime_checkable
class Metric(Protocol):
    """Structural type for anything the runner/critic can call."""

    name: str
    tier: Tier
    modes: frozenset[Mode]
    requires: frozenset[str]
    cost_class: CostClass

    def score(self, ctx: EvalContext) -> MetricResult: ...
    async def ascore(self, ctx: EvalContext) -> MetricResult: ...


class BaseMetric:
    """Concrete base implementing the shared metric plumbing.

    Subclasses set the class attributes and implement ``_compute`` (sync). Override ``_acompute``
    for a genuinely async implementation; by default it bridges to ``_compute``.
    """

    name: str = ""
    tier: Tier = Tier.DETERMINISTIC
    modes: frozenset[Mode] = _DEFAULT_MODES
    requires: frozenset[str] = frozenset()
    cost_class: CostClass = CostClass.CHEAP

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
        score = min(max(float(result.score), 0.0), 1.0)  # normalize to [0, 1]
        cost = result.cost
        if cost.latency_ms == 0.0:
            cost = replace(cost, latency_ms=latency_ms)
        return replace(result, score=score, cost=cost)
