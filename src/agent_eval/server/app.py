"""The FastAPI app — every served metric as one endpoint, built from the catalog table.

A thin wrapper over the existing framework: per request it builds the metric via the registry
(the same factories the YAML path uses), scores each posted context once, and replays those
results through the public :func:`agent_eval.offline.runner.evaluate` for the aggregate, so the
CI statistics live in exactly one place. See docs/api-server.md.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Iterator, Sequence
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version

from fastapi import FastAPI, HTTPException

from agent_eval.core.contracts import EvalContext, MetricResult
from agent_eval.core.errors import ConfigError
from agent_eval.core.gate import GatePolicy
from agent_eval.core.metric import Metric
from agent_eval.core.registry import BuildContext, MetricRegistry, default_registry
from agent_eval.core.suite import Suite
from agent_eval.judges.backend import JudgeBackend
from agent_eval.offline.runner import evaluate
from agent_eval.server.catalog import ENDPOINTS, EndpointSpec, describe
from agent_eval.server.judge import ResolvedJudge, resolve_judge_from_env
from agent_eval.server.schemas import (
    AggregateOut,
    CatalogEntry,
    HealthOut,
    JudgeInfo,
    MetricRequest,
    MetricResponse,
    MetricResultOut,
)

try:
    _VERSION = pkg_version("agent-eval")
except PackageNotFoundError:  # pragma: no cover - not installed as a distribution
    _VERSION = "0.0.0"


def _bertscore_available() -> bool:
    return importlib.util.find_spec("bert_score") is not None


def _unavailable_hint(spec: EndpointSpec, registry: MetricRegistry) -> str | None:
    """Why this endpoint cannot score right now (a missing optional extra), or None if it can."""
    if spec.type not in registry.types():
        if spec.extra:
            return (
                f"metric type {spec.type!r} is not available on this server; "
                f"install the optional extra: pip install 'agent-eval[{spec.extra}]'"
            )
        return f"metric type {spec.type!r} is not registered on this server"
    if spec.type == "bertscore" and not _bertscore_available():
        return "bertscore needs the 'bert-score' package: pip install 'agent-eval[bertscore]'"
    return None


class _ReplayMetric:
    """Feed already-computed per-item results back through the public ``evaluate()``.

    ``evaluate`` both scores and aggregates; scoring again would double-charge judge metrics one
    LLM call per item. This shim mirrors the real metric's aggregation attributes and replays the
    results in order, so ``evaluate`` contributes only the aggregation/CI math — which therefore
    lives in exactly one place (the runner), never re-implemented here.
    """

    def __init__(self, metric: Metric, results: Sequence[MetricResult]) -> None:
        self.name = metric.name
        self.requires: frozenset[str] = frozenset()
        self.cost_class = metric.cost_class
        self.aggregation = metric.aggregation
        self.higher_is_better = metric.higher_is_better
        self.unit_interval = getattr(metric, "unit_interval", True)
        self._results: Iterator[MetricResult] = iter(results)

    def score(self, ctx: EvalContext) -> MetricResult:
        return self._next()

    async def ascore(self, ctx: EvalContext) -> MetricResult:
        return self._next()

    def _next(self) -> MetricResult:
        try:
            return next(self._results)
        except StopIteration:  # pragma: no cover - guards a cross-module iteration contract
            raise RuntimeError(
                "replay exhausted: evaluate() requested more scores than were precomputed "
                "(the runner's one-score-per-context iteration contract changed)"
            ) from None


def _coerce_params(params: dict) -> dict:
    """JSON-shape fixups + shape checks for constructor params (JSON has no tuples: ``scale: [1, 5]``).

    Only *wire-shape* concerns live here — a list arriving where the constructor wants a
    ``(lo, hi)`` tuple, an int that must be positive; metric semantics stay in the metric classes.
    Raises ``ValueError`` (→ 422 via the caller's build guard) so a bad request fails the same
    way no matter which judge backend is deployed.
    """
    # An explicit JSON null means "use the default" — same as omitting the key.
    out = {key: value for key, value in params.items() if value is not None}
    scale = out.get("scale")
    if scale is not None:
        if not (
            isinstance(scale, list)
            and len(scale) == 2
            and all(isinstance(v, int) and not isinstance(v, bool) for v in scale)
        ):
            raise ValueError(f"'scale' must be a [lo, hi] pair of integers; got {scale!r}")
        lo, hi = scale
        if lo >= hi:
            raise ValueError(f"'scale' must have lo < hi; got {scale!r}")
        out["scale"] = (lo, hi)
    k = out.get("k")
    if k is not None and (isinstance(k, bool) or not isinstance(k, int) or k < 1):
        raise ValueError(f"'k' must be an integer >= 1; got {k!r}")
    return out


def create_app(
    registry: MetricRegistry | None = None,
    judge: JudgeBackend | None = None,
    panel: Sequence[JudgeBackend] = (),
) -> FastAPI:
    """Build the API server. ``registry``/``judge`` are injectable for tests; by default the
    full built-in registry and the env-resolved judge (see server.judge) are used."""
    reg = registry if registry is not None else default_registry()
    if judge is not None:
        resolved = ResolvedJudge(judge, tuple(panel), kind="injected", detail=type(judge).__name__)
    else:
        resolved = resolve_judge_from_env()
    bctx = BuildContext(judge=resolved.backend, panel=resolved.panel)

    app = FastAPI(
        title="agent-eval metric API",
        version=_VERSION,
        description=(
            "Each agent-eval metric as an endpoint, prefixed by agent-type family "
            "(/common, /rag, /t2s). POST 1..N contexts; get per-item results plus the "
            "aggregated final value with a 95% CI. See GET /catalog."
        ),
    )

    @app.get("/health", response_model=HealthOut, tags=["meta"])
    def health() -> HealthOut:
        return HealthOut(
            status="ok",
            version=_VERSION,
            judge=JudgeInfo(
                kind=resolved.kind, detail=resolved.detail, panel_size=1 + len(resolved.panel)
            ),
            available={
                # all-or-nothing probe: default_registry() registers the t2s family atomically
                "t2s": "soft_f1" in reg.types(),
                "bertscore": _bertscore_available(),
            },
        )

    @app.get("/catalog", response_model=list[CatalogEntry], tags=["meta"])
    def catalog() -> list[CatalogEntry]:
        entries: list[CatalogEntry] = []
        for spec in ENDPOINTS:
            hint = _unavailable_hint(spec, reg)
            cost_class = aggregation = None
            if spec.type in reg.types():  # registered → attrs are knowable even if unavailable
                metric = reg.build({"type": spec.type, "name": spec.type}, bctx)
                cost_class, aggregation = str(metric.cost_class), str(metric.aggregation)
            entries.append(
                CatalogEntry(
                    family=spec.family,
                    path=spec.path,
                    type=spec.type,
                    requires=list(spec.requires),
                    optional=list(spec.optional),
                    metadata_keys=list(spec.metadata_keys),
                    metadata_optional=list(spec.metadata_optional),
                    params=list(spec.params),
                    judge_based=spec.judge_based,
                    cost_class=cost_class,
                    aggregation=aggregation,
                    available=hint is None,
                    unavailable_hint=hint,
                )
            )
        return entries

    def _make_handler(spec: EndpointSpec):
        def handler(request: MetricRequest) -> MetricResponse:
            hint = _unavailable_hint(spec, reg)
            if hint:
                raise HTTPException(status_code=501, detail=hint)
            unknown = set(request.params) - set(spec.params)
            if unknown:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"unknown params for {spec.path}: {sorted(unknown)}; "
                        f"allowed: {sorted(spec.params)}"
                    ),
                )
            try:
                metric = reg.build(
                    {"type": spec.type, "name": spec.type, "params": _coerce_params(request.params)},
                    bctx,
                )
            except (ConfigError, TypeError, ValueError) as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc

            contexts = [c.to_context() for c in request.contexts]
            results = [metric.score(ctx) for ctx in contexts]  # each item scored exactly once
            suite = Suite(spec.family, spec.type, [_ReplayMetric(metric, results)], GatePolicy())
            aggregate = evaluate(suite, contexts).aggregates[0]
            return MetricResponse(
                family=spec.family,
                metric=spec.type,
                n=len(contexts),
                n_errors=sum(1 for r in results if r.error is not None),
                results=[MetricResultOut.from_result(r) for r in results],
                aggregate=AggregateOut.from_aggregate(aggregate),
            )

        handler.__name__ = f"score_{spec.family}_{spec.name}"
        return handler

    for spec in ENDPOINTS:
        app.add_api_route(
            spec.path,
            _make_handler(spec),
            methods=["POST"],
            response_model=MetricResponse,
            summary=f"Score {spec.type}",
            description=describe(spec),
            tags=[spec.family],
        )

    return app
