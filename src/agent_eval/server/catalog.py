"""The declarative endpoint table — the single source of truth the server is built from.

One :class:`EndpointSpec` per served metric: URL (``/{family}/{name}``), the registry type it
builds, what each posted context must carry, and which ``params`` keys the endpoint accepts.
Routes, OpenAPI descriptions, ``params`` validation, and ``GET /catalog`` are all projections of
this table, so adding an endpoint is one new entry.

Deliberately absent: ``p95_latency`` and ``token_usage`` — they read harness-produced metadata
only (no input/output to score), so they stay CLI/library-side.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EndpointSpec:
    """One served metric: its URL, the registry type it builds, and its request contract."""

    family: str  # URL prefix = metric family: common (all agent types) | rag | t2s
    name: str  # URL segment under the family
    type: str  # registry metric type this endpoint builds
    requires: tuple[str, ...] = ()  # EvalContext fields the metric needs
    optional: tuple[str, ...] = ()  # EvalContext fields it uses when present
    metadata_keys: tuple[str, ...] = ()  # required metadata[...] keys
    metadata_optional: tuple[str, ...] = ()  # optional metadata[...] keys
    params: tuple[str, ...] = ()  # allowed request `params` keys
    judge_based: bool = False  # scored by the server's configured LLM judge?
    extra: str | None = None  # optional-dependency extra that provides it

    @property
    def path(self) -> str:
        return f"/{self.family}/{self.name}"


ENDPOINTS: tuple[EndpointSpec, ...] = (
    EndpointSpec(
        family="common", name="llm_judge", type="llm_judge",
        requires=("input", "output"), optional=("expected",),
        params=("criteria", "scale"), judge_based=True,
    ),
    EndpointSpec(
        family="common", name="bertscore", type="bertscore",
        requires=("output", "expected"),
        params=("lang", "model_type", "rescale_with_baseline"), extra="bertscore",
    ),
    EndpointSpec(
        family="rag", name="recall_at_k", type="recall_at_k",
        metadata_keys=("retrieved_ids", "relevant_ids"), params=("k",),
    ),
    EndpointSpec(
        family="rag", name="precision_at_k", type="precision_at_k",
        metadata_keys=("retrieved_ids", "relevant_ids"), params=("k",),
    ),
    EndpointSpec(
        family="rag", name="ndcg_at_k", type="ndcg_at_k",
        metadata_keys=("retrieved_ids", "relevant_ids"),
        metadata_optional=("relevance",), params=("k",),
    ),
    EndpointSpec(
        family="rag", name="faithfulness", type="faithfulness",
        requires=("output", "retrieved_context"), judge_based=True,
    ),
    EndpointSpec(
        family="rag", name="consistency", type="consistency",
        optional=("input",), metadata_keys=("repeated_outputs",), judge_based=True,
    ),
    EndpointSpec(
        family="t2s", name="soft_f1", type="soft_f1",
        metadata_keys=("execution_result", "gold_execution_result"),
        params=("result_policy",), extra="t2s",
    ),
    EndpointSpec(
        family="t2s", name="component_match", type="component_match",
        metadata_keys=("sql", "gold_sql"), params=("dialect",), extra="t2s",
    ),
    EndpointSpec(
        family="t2s", name="ast_valid", type="ast_valid",
        metadata_keys=("sql",), params=("dialect",), extra="t2s",
    ),
    EndpointSpec(
        family="t2s", name="faithfulness", type="t2s_faithfulness",
        requires=("output",), optional=("input",), metadata_keys=("execution_result",),
        params=("sample_rows", "max_distinct", "max_columns"), judge_based=True, extra="t2s",
    ),
    EndpointSpec(
        family="t2s", name="consistency", type="t2s_consistency",
        optional=("input",), metadata_keys=("repeated_sql",), judge_based=True, extra="t2s",
    ),
)


def describe(spec: EndpointSpec) -> str:
    """The OpenAPI description for one endpoint — generated so /docs never drifts from the table."""
    lines = [
        f"Scores the `{spec.type}` metric for each posted context and returns per-item results "
        f"plus the aggregated final value with a 95% CI.",
    ]
    if spec.requires:
        lines.append(f"Required context fields: {', '.join(spec.requires)}.")
    if spec.optional:
        lines.append(f"Optional context fields: {', '.join(spec.optional)}.")
    if spec.metadata_keys:
        lines.append(f"Required `metadata` keys: {', '.join(spec.metadata_keys)}.")
    if spec.metadata_optional:
        lines.append(f"Optional `metadata` keys: {', '.join(spec.metadata_optional)}.")
    lines.append(f"Allowed `params` keys: {', '.join(spec.params) if spec.params else '(none)'}.")
    if spec.judge_based:
        lines.append("Judge-based: scored by the server's configured LLM judge (see GET /health).")
    if spec.extra:
        lines.append(f"Needs the optional `[{spec.extra}]` extra installed on the server.")
    return "\n\n".join(lines)
