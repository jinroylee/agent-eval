"""Performance metrics — token/latency attribution, tail latencies, reliability, budgets.

Report tail latency (p95/p99), not means (averaging hides mid-stream stalls); attribute tokens and
latency per graph node so you optimize the right one (multi-agent runs can be ~15x single-call
tokens). LatencyBudget/TokenBudget are the runtime per-step gates.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from agent_eval.core.contracts import CostClass, EvalContext, MetricResult, Tier
from agent_eval.core.metric import BaseMetric
from agent_eval.stats.agents import PassHatKResult, pass_hat_k

_FIELDS = ("tokens", "latency_ms", "usd", "calls")


def attribute(spans: Sequence[dict]) -> dict:
    """Aggregate per-node token/latency/cost from spans; the graph total reconciles to the sum."""
    nodes: dict[str, dict] = {}
    for span in spans:
        node = span.get("node", "?")
        acc = nodes.setdefault(node, {"tokens": 0, "latency_ms": 0.0, "usd": 0.0, "calls": 0})
        acc["tokens"] += span.get("tokens", 0)
        acc["latency_ms"] += span.get("latency_ms", 0.0)
        acc["usd"] += span.get("usd", 0.0)
        acc["calls"] += 1
    graph = {f: sum(acc[f] for acc in nodes.values()) for f in _FIELDS}
    return {"nodes": nodes, "graph": graph}


def latency_percentiles(latencies: Sequence[float], ps: Sequence[int] = (50, 95, 99)) -> dict:
    arr = np.asarray(latencies, dtype=float)
    return {f"p{p}": float(np.percentile(arr, p)) for p in ps}


def scenario_pass_hat_k(
    per_task_runs: Sequence[Sequence[bool]], k: int, alpha: float = 0.05
) -> PassHatKResult:
    """Reliability gate over repeated end-to-end runs: pass^k (all-k-pass), not pass@k."""
    successes = [sum(1 for r in runs if r) for runs in per_task_runs]
    trials = [len(runs) for runs in per_task_runs]
    return pass_hat_k(successes, trials, k, alpha=alpha)


class LatencyBudget(BaseMetric):
    name = "latency_budget"
    tier = Tier.DETERMINISTIC
    cost_class = CostClass.FREE

    def __init__(self, max_ms: float = 2000.0) -> None:
        self.max_ms = max_ms

    def _compute(self, ctx: EvalContext) -> MetricResult:
        latency = float(ctx.metadata.get("latency_ms", 0.0))
        ok = latency <= self.max_ms
        return MetricResult(self.name, 1.0 if ok else 0.0, passed=ok, detail={"latency_ms": latency})


class TokenBudget(BaseMetric):
    name = "token_budget"
    tier = Tier.DETERMINISTIC
    cost_class = CostClass.FREE

    def __init__(self, max_tokens: int = 100_000) -> None:
        self.max_tokens = max_tokens

    def _compute(self, ctx: EvalContext) -> MetricResult:
        tokens = int(ctx.metadata.get("tokens", 0))
        ok = tokens <= self.max_tokens
        return MetricResult(self.name, 1.0 if ok else 0.0, passed=ok, detail={"tokens": tokens})


def register_perf_metrics(registry) -> None:
    registry.register("latency_budget", lambda p: LatencyBudget(**p))
    registry.register("token_budget", lambda p: TokenBudget(**p))
