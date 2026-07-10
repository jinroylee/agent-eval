# Metric API Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the 12 input/output-scoring agent-eval metrics as FastAPI endpoints (`/common`, `/rag`, `/t2s`), purely additive to the repo, with a Dockerfile.

**Architecture:** A new `agent_eval.server` subpackage wraps the existing framework: per request it builds the metric via `default_registry()` (the same factories the YAML path uses), scores each posted context once, then replays those results through the public `evaluate()` for the aggregated final value with a 95% CI. A declarative endpoint table (`catalog.py`) is the single source of truth for routes, OpenAPI descriptions, `params` validation, and `/catalog`. The judge is resolved once at startup from env (factory → OpenAI-compatible → lexical stub).

**Tech Stack:** FastAPI + uvicorn + httpx (new `[server]` extra), pydantic v2 (already a base dep), existing `agent_eval` core/metrics/judges/offline modules. Tests with `fastapi.testclient` (skipped if the extra is absent).

**Spec:** `docs/superpowers/specs/2026-07-09-metric-api-server-design.md`

**Constraints (from the spec — re-read before starting):**
- Purely additive. The ONLY existing files modified: `pyproject.toml` (new extra + new script, additive lines) and `uv.lock` (regenerated). Never touch `src/agent_eval/{core,metrics,judges,offline,config,...}`, existing tests, docs, or examples.
- `p95_latency` / `token_usage` get **no** endpoint.
- Per-item metric failures are 200-with-`error`-in-body (framework semantics), not HTTP errors.

---

### Task 1: Packaging — `[server]` extra + console script

**Files:**
- Modify: `pyproject.toml` (two additive edits)

- [ ] **Step 1: Add the extra and the script**

In `pyproject.toml`, append to `[project.optional-dependencies]` (after the `bertscore` line):

```toml
# Metric API server (FastAPI wrapper around the metrics; see docs/api-server.md):
server = ["fastapi>=0.111", "uvicorn[standard]>=0.30", "httpx>=0.27"]
```

And in `[project.scripts]`, add below the existing `agent-eval` line:

```toml
agent-eval-serve = "agent_eval.server.__main__:main"
```

- [ ] **Step 2: Re-lock and sync**

Run: `cd /Users/makinarocks/workspace/agent-eval && uv lock && uv sync --extra server --extra t2s --extra langgraph`
Expected: lock updates with fastapi/uvicorn/httpx; sync succeeds. (If resolution conflicts with the pinned `langgraph-api`, relax to `fastapi>=0.100` — do not touch other dep pins.)

- [ ] **Step 3: Verify imports**

Run: `uv run python -c "import fastapi, uvicorn, httpx; print('server deps ok')"`
Expected: `server deps ok`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "feat(server): add [server] extra and agent-eval-serve script

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: The declarative endpoint catalog

**Files:**
- Create: `src/agent_eval/server/__init__.py` (placeholder for now)
- Create: `src/agent_eval/server/catalog.py`
- Test: `tests/unit/test_server.py` (new file, first tests)

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_server.py`:

```python
"""API server tests — TestClient over ``create_app`` with deterministic judges.

The module skips entirely when the ``[server]`` extra is not installed, so the base suite is
unaffected. The default app fixture clears the judge env vars → the lexical stub, which makes
judge-metric scores exact and assertable.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from agent_eval.server.catalog import ENDPOINTS  # noqa: E402


# --------------------------------------------------------------------------- catalog table
def test_catalog_has_the_12_served_metrics():
    assert len(ENDPOINTS) == 12
    paths = [spec.path for spec in ENDPOINTS]
    assert len(set(paths)) == 12  # unique URLs
    # prefixed by agent-type family
    assert {spec.family for spec in ENDPOINTS} == {"common", "rag", "t2s"}
    # metadata-only metrics are deliberately NOT served
    served_types = {spec.type for spec in ENDPOINTS}
    assert "p95_latency" not in served_types and "token_usage" not in served_types


def test_catalog_t2s_paths_drop_the_redundant_prefix():
    by_path = {spec.path: spec for spec in ENDPOINTS}
    assert by_path["/t2s/faithfulness"].type == "t2s_faithfulness"
    assert by_path["/t2s/consistency"].type == "t2s_consistency"


def test_catalog_declares_params_and_judges():
    by_path = {spec.path: spec for spec in ENDPOINTS}
    assert by_path["/rag/recall_at_k"].params == ("k",)
    assert by_path["/common/llm_judge"].params == ("criteria", "scale")
    assert by_path["/t2s/soft_f1"].params == ("result_policy",)
    judge_paths = {s.path for s in ENDPOINTS if s.judge_based}
    assert judge_paths == {
        "/common/llm_judge", "/rag/faithfulness", "/rag/consistency",
        "/t2s/faithfulness", "/t2s/consistency",
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_server.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent_eval.server'`

- [ ] **Step 3: Implement the catalog**

Create `src/agent_eval/server/__init__.py`:

```python
"""agent-eval metric API server (optional ``[server]`` extra). See docs/api-server.md."""
```

(The `create_app` re-export is added in Task 5, after `app.py` exists.)

Create `src/agent_eval/server/catalog.py`:

```python
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
        requires=("output", "retrieved_context"), judge_based=True,
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
        requires=("output",), optional=("input",), metadata_keys=("execution_result",),
        params=("sample_rows", "max_distinct", "max_columns"), judge_based=True, extra="t2s",
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_server.py -q`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/agent_eval/server/__init__.py src/agent_eval/server/catalog.py tests/unit/test_server.py
git commit -m "feat(server): declarative endpoint catalog (12 served metrics)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Wire schemas (pydantic request/response models)

**Files:**
- Create: `src/agent_eval/server/schemas.py`
- Test: `tests/unit/test_server.py` (append)

- [ ] **Step 1: Write the failing tests** (append to `tests/unit/test_server.py`)

```python
# --------------------------------------------------------------------------- wire schemas
def test_context_in_converts_to_frozen_eval_context():
    from agent_eval.core.contracts import EvalContext
    from agent_eval.server.schemas import EvalContextIn

    ctx = EvalContextIn(
        input="q", output="a", expected="ref",
        retrieved_context=["c1", "c2"], metadata={"retrieved_ids": ["d1"]},
    ).to_context()
    assert isinstance(ctx, EvalContext)
    assert ctx.input == "q" and ctx.output == "a" and ctx.expected == "ref"
    assert ctx.retrieved_context == ("c1", "c2")
    assert ctx.metadata == {"retrieved_ids": ["d1"]}
    # everything is optional on the wire — the metric's `requires` decides
    empty = EvalContextIn().to_context()
    assert empty.input is None and empty.output is None


def test_result_and_aggregate_mirror_core_contracts():
    from agent_eval.core.contracts import Cost, MetricResult
    from agent_eval.core.gate import MetricAggregate
    from agent_eval.server.schemas import AggregateOut, MetricResultOut

    out = MetricResultOut.from_result(
        MetricResult("m", 0.5, passed=True, confidence=0.9,
                     cost=Cost(latency_ms=1.5), detail={"reason": "r"}, error=None)
    )
    assert (out.metric, out.score, out.passed, out.confidence) == ("m", 0.5, True, 0.9)
    assert out.cost.latency_ms == 1.5 and out.detail == {"reason": "r"} and out.error is None

    from agent_eval.core.contracts import Aggregation

    agg = AggregateOut.from_aggregate(
        MetricAggregate("m", 0.8, 0.6, 0.9, 10, 2, Aggregation.MEAN, True)
    )
    assert (agg.metric, agg.value, agg.ci_low, agg.ci_high) == ("m", 0.8, 0.6, 0.9)
    assert (agg.n, agg.n_errors, agg.aggregation, agg.higher_is_better) == (10, 2, "mean", True)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_server.py -q`
Expected: 2 new FAIL — `No module named 'agent_eval.server.schemas'`

- [ ] **Step 3: Implement the schemas**

Create `src/agent_eval/server/schemas.py`:

```python
"""Pydantic request/response models — the wire mirror of the core contracts.

Requests carry 1..N :class:`EvalContextIn` items (every field optional; each metric's ``requires``
decides what is enough); responses mirror :class:`MetricResult` per item plus the runner's
:class:`MetricAggregate` as the final value. Field names match the core dataclasses exactly so
payloads read like the framework's own objects.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agent_eval.core.contracts import EvalContext, MetricResult
from agent_eval.core.gate import MetricAggregate


class EvalContextIn(BaseModel):
    """JSON mirror of :class:`EvalContext` (see MetaKey for the conventional metadata keys)."""

    input: Any = None
    output: Any = None
    expected: Any = None
    retrieved_context: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_context(self) -> EvalContext:
        return EvalContext(
            input=self.input,
            output=self.output,
            expected=self.expected,
            retrieved_context=tuple(self.retrieved_context),
            metadata=dict(self.metadata),
        )


class MetricRequest(BaseModel):
    """The uniform request body: 1..N contexts (repeated runs post N) + optional metric params."""

    contexts: list[EvalContextIn] = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)


class CostOut(BaseModel):
    tokens: int = 0
    usd: float = 0.0
    latency_ms: float = 0.0


class MetricResultOut(BaseModel):
    """One item's score. ``error`` set means the metric could not run for this item (it is then
    excluded from the aggregate's denominator) — never conflate it with a low score."""

    metric: str
    score: float
    passed: bool | None = None
    confidence: float | None = None
    cost: CostOut = Field(default_factory=CostOut)
    detail: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None

    @classmethod
    def from_result(cls, r: MetricResult) -> MetricResultOut:
        return cls(
            metric=r.metric, score=r.score, passed=r.passed, confidence=r.confidence,
            cost=CostOut(tokens=r.cost.tokens, usd=r.cost.usd, latency_ms=r.cost.latency_ms),
            detail=dict(r.detail), error=r.error,
        )


class AggregateOut(BaseModel):
    """Mirrors :class:`agent_eval.core.gate.MetricAggregate` field-for-field."""

    metric: str
    value: float
    ci_low: float
    ci_high: float
    n: int
    n_errors: int
    aggregation: str
    higher_is_better: bool

    @classmethod
    def from_aggregate(cls, a: MetricAggregate) -> AggregateOut:
        return cls(
            metric=a.metric, value=a.value, ci_low=a.ci_low, ci_high=a.ci_high,
            n=a.n, n_errors=a.n_errors, aggregation=str(a.aggregation),
            higher_is_better=a.higher_is_better,
        )


class MetricResponse(BaseModel):
    family: str
    metric: str  # the registry type
    n: int  # contexts posted
    n_errors: int  # items whose metric errored
    results: list[MetricResultOut]  # one per posted context, same order
    aggregate: AggregateOut  # the final value (always present; n=1 is degenerate but well-defined)


class JudgeInfo(BaseModel):
    kind: str  # stub | factory | openai_compatible | injected
    detail: str = ""
    panel_size: int = 1  # judges consulted per item (primary + panel)


class HealthOut(BaseModel):
    status: str
    version: str
    judge: JudgeInfo
    available: dict[str, bool]  # optional capabilities: {"t2s": ..., "bertscore": ...}


class CatalogEntry(BaseModel):
    family: str
    path: str
    type: str
    requires: list[str]
    optional: list[str]
    metadata_keys: list[str]
    metadata_optional: list[str]
    params: list[str]
    judge_based: bool
    cost_class: str | None = None  # None when the metric is unavailable on this server
    aggregation: str | None = None
    available: bool
    unavailable_hint: str | None = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_server.py -q`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/agent_eval/server/schemas.py tests/unit/test_server.py
git commit -m "feat(server): wire schemas mirroring the core contracts

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Judge resolution from environment

**Files:**
- Create: `src/agent_eval/server/judge.py`
- Test: `tests/unit/test_server.py` (append)

- [ ] **Step 1: Write the failing tests** (append to `tests/unit/test_server.py`)

```python
# --------------------------------------------------------------------------- judge resolution
def test_judge_defaults_to_lexical_stub():
    from agent_eval.server import judge as judge_mod

    resolved = judge_mod.resolve_judge_from_env(environ={})
    assert resolved.kind == "stub"
    assert resolved.panel == ()
    # duck-typed JudgeBackend
    from agent_eval.judges.backend import JudgeRequest

    verdict = resolved.backend.evaluate(JudgeRequest(instruction="i", response="x", reference="x"))
    assert verdict.score == 1.0  # full token overlap with the reference


def test_judge_base_url_and_model_must_come_together():
    from agent_eval.server import judge as judge_mod

    with pytest.raises(RuntimeError, match="AGENT_EVAL_JUDGE_MODEL"):
        judge_mod.resolve_judge_from_env(environ={"AGENT_EVAL_JUDGE_BASE_URL": "http://x/v1"})
    with pytest.raises(RuntimeError, match="AGENT_EVAL_JUDGE_BASE_URL"):
        judge_mod.resolve_judge_from_env(environ={"AGENT_EVAL_JUDGE_MODEL": "m"})


def test_openai_compatible_judge_calls_chat_completions():
    import json

    import httpx

    from agent_eval.judges.backend import JudgeRequest
    from agent_eval.server import judge as judge_mod

    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "SCORE: 4\nREASON: solid"}}]}
        )

    resolved = judge_mod.resolve_judge_from_env(
        environ={
            "AGENT_EVAL_JUDGE_BASE_URL": "http://llm.test/v1",
            "AGENT_EVAL_JUDGE_MODEL": "test-model",
            "AGENT_EVAL_JUDGE_API_KEY": "sk-test",
        },
        transport=httpx.MockTransport(handler),
    )
    assert resolved.kind == "openai_compatible"
    assert "test-model" in resolved.detail

    verdict = resolved.backend.evaluate(JudgeRequest(instruction="rate", response="hi"))
    assert verdict.score == pytest.approx(0.75)  # SCORE: 4 on the default 1-5 scale
    assert verdict.reason == "solid"
    assert captured["url"] == "http://llm.test/v1/chat/completions"
    assert captured["auth"] == "Bearer sk-test"
    assert captured["body"]["model"] == "test-model"
    assert captured["body"]["temperature"] == 0


def test_judge_factory_env_uses_the_yaml_factory_convention(tmp_path):
    from agent_eval.server import judge as judge_mod

    factory_file = tmp_path / "my_judges.py"
    factory_file.write_text(
        "from agent_eval.judges.backend import FunctionJudge\n"
        "def make_panel():\n"
        "    return [FunctionJudge(lambda r: 1.0), FunctionJudge(lambda r: 0.5)]\n",
        encoding="utf-8",
    )
    resolved = judge_mod.resolve_judge_from_env(
        environ={"AGENT_EVAL_JUDGE_FACTORY": f"{factory_file}:make_panel"}
    )
    assert resolved.kind == "factory"
    assert len(resolved.panel) == 1  # primary + 1 panel member
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_server.py -q`
Expected: 4 new FAIL — `No module named 'agent_eval.server.judge'`

- [ ] **Step 3: Implement judge resolution**

Create `src/agent_eval/server/judge.py`:

```python
"""Resolve the server's LLM judge from environment variables (startup-time, server-wide).

Precedence:

1. ``AGENT_EVAL_JUDGE_FACTORY`` — the exact YAML ``judge.factory`` convention (``module:attr`` or
   ``path/to/file.py:attr``; may return a backend, a zero-arg callable, or a list = PoLL panel).
   Resolved through :func:`agent_eval.config.loader.resolve_judge`, so semantics never drift.
2. ``AGENT_EVAL_JUDGE_BASE_URL`` + ``AGENT_EVAL_JUDGE_MODEL`` (+ optional ``_API_KEY``,
   ``_TIMEOUT`` seconds) — any OpenAI-compatible ``/chat/completions`` (vLLM, an internal
   gateway). The framework's own :class:`LLMJudge` renders the prompt and parses ``SCORE:``;
   this module contributes only the ``complete(prompt) -> str`` callable.
3. Neither — the deterministic lexical-overlap stub (offline; NOT a real judgment — ``/health``
   reports the active kind so stub scores are never mistaken for real ones).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from agent_eval.config.loader import resolve_judge
from agent_eval.config.schema import ConfigModel, JudgeModel
from agent_eval.judges.backend import FunctionJudge, LLMJudge, lexical_overlap_judge

ENV_FACTORY = "AGENT_EVAL_JUDGE_FACTORY"
ENV_BASE_URL = "AGENT_EVAL_JUDGE_BASE_URL"
ENV_MODEL = "AGENT_EVAL_JUDGE_MODEL"
ENV_API_KEY = "AGENT_EVAL_JUDGE_API_KEY"
ENV_TIMEOUT = "AGENT_EVAL_JUDGE_TIMEOUT"


@dataclass(frozen=True)
class ResolvedJudge:
    """The judge the server will inject into every metric build, plus how it was chosen."""

    backend: Any  # a judges.backend.JudgeBackend
    panel: tuple[Any, ...] = ()
    kind: str = "stub"  # stub | factory | openai_compatible | injected
    detail: str = ""


def openai_complete(
    base_url: str, model: str, api_key: str | None, timeout: float, transport: Any | None = None
):
    """A ``complete(prompt) -> str`` callable against an OpenAI-compatible endpoint.

    ``transport`` is a test hook (``httpx.MockTransport``); leave ``None`` in production.
    """
    import httpx  # deferred: only this judge path needs it

    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def complete(prompt: str) -> str:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }
        with httpx.Client(timeout=timeout, transport=transport) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    return complete


def resolve_judge_from_env(
    environ: Mapping[str, str] | None = None, transport: Any | None = None
) -> ResolvedJudge:
    """Resolve the judge per the documented precedence. ``environ`` overrides ``os.environ`` (tests)."""
    env = os.environ if environ is None else environ

    factory = env.get(ENV_FACTORY)
    if factory:
        backend, panel = resolve_judge(ConfigModel(judge=JudgeModel(factory=factory)))
        if backend is None:
            raise RuntimeError(f"{ENV_FACTORY}={factory!r} did not produce a judge backend")
        return ResolvedJudge(backend, tuple(panel), kind="factory", detail=factory)

    base_url, model = env.get(ENV_BASE_URL), env.get(ENV_MODEL)
    if base_url or model:
        if not base_url:
            raise RuntimeError(f"{ENV_MODEL} is set but {ENV_BASE_URL} is missing")
        if not model:
            raise RuntimeError(f"{ENV_BASE_URL} is set but {ENV_MODEL} is missing")
        timeout = float(env.get(ENV_TIMEOUT, "60"))
        complete = openai_complete(base_url, model, env.get(ENV_API_KEY), timeout, transport)
        return ResolvedJudge(
            LLMJudge(complete, name=model),
            kind="openai_compatible",
            detail=f"{model} @ {base_url}",
        )

    return ResolvedJudge(
        FunctionJudge(lexical_overlap_judge), kind="stub", detail="lexical_overlap_judge"
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_server.py -q`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add src/agent_eval/server/judge.py tests/unit/test_server.py
git commit -m "feat(server): env-resolved judge (factory / OpenAI-compatible / stub)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: App skeleton — `create_app`, `/health`, `/catalog`

**Files:**
- Create: `src/agent_eval/server/app.py`
- Modify: `src/agent_eval/server/__init__.py`
- Test: `tests/unit/test_server.py` (append)

- [ ] **Step 1: Write the failing tests** (append to `tests/unit/test_server.py`)

```python
# --------------------------------------------------------------------------- app: health + catalog
@pytest.fixture()
def client(monkeypatch):
    """A TestClient with all judge env cleared → the deterministic lexical stub."""
    from fastapi.testclient import TestClient

    from agent_eval.server import judge as judge_mod
    from agent_eval.server.app import create_app

    for var in (
        judge_mod.ENV_FACTORY, judge_mod.ENV_BASE_URL, judge_mod.ENV_MODEL,
        judge_mod.ENV_API_KEY, judge_mod.ENV_TIMEOUT,
    ):
        monkeypatch.delenv(var, raising=False)
    return TestClient(create_app())


def test_health_reports_status_judge_and_availability(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["version"]
    assert body["judge"]["kind"] == "stub"
    assert body["judge"]["panel_size"] == 1
    assert set(body["available"]) == {"t2s", "bertscore"}
    assert all(isinstance(v, bool) for v in body["available"].values())


def test_catalog_endpoint_lists_every_served_metric(client):
    entries = client.get("/catalog").json()
    assert len(entries) == 12
    by_path = {e["path"]: e for e in entries}
    assert by_path["/t2s/faithfulness"]["type"] == "t2s_faithfulness"
    assert by_path["/rag/recall_at_k"]["params"] == ["k"]
    assert by_path["/common/llm_judge"]["judge_based"] is True
    # sqlglot is installed in the dev env → t2s metrics are registered and available
    assert by_path["/t2s/soft_f1"]["available"] is True
    assert by_path["/t2s/soft_f1"]["aggregation"] == "mean"


def test_openapi_exposes_all_metric_routes(client):
    paths = set(client.get("/openapi.json").json()["paths"])
    from agent_eval.server.catalog import ENDPOINTS as _eps

    assert {spec.path for spec in _eps} <= paths
```

Note: the third test will only pass after Task 6 registers the metric routes — expect exactly that one to keep failing at the end of this task.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_server.py -q`
Expected: 3 new FAIL — `No module named 'agent_eval.server.app'` (plus the 9 existing passes)

- [ ] **Step 3: Implement the app skeleton**

Create `src/agent_eval/server/app.py`:

```python
"""The FastAPI app — every served metric as one endpoint, built from the catalog table.

A thin wrapper over the existing framework: per request it builds the metric via the registry
(the same factories the YAML path uses), scores each posted context once, and replays those
results through the public :func:`agent_eval.offline.runner.evaluate` for the aggregate, so the
CI statistics live in exactly one place. See docs/api-server.md.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Iterator, Sequence
from importlib.metadata import PackageNotFoundError, version as pkg_version

from fastapi import FastAPI, HTTPException

from agent_eval.core.contracts import EvalContext, MetricResult
from agent_eval.core.errors import ConfigError
from agent_eval.core.gate import GatePolicy
from agent_eval.core.metric import Metric
from agent_eval.core.registry import BuildContext, MetricRegistry, default_registry
from agent_eval.core.suite import Suite
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
        extra = spec.extra or "t2s"
        return (
            f"metric type {spec.type!r} is not available on this server; "
            f"install the optional extra: pip install 'agent-eval[{extra}]'"
        )
    if spec.type == "bertscore" and not _bertscore_available():
        return "bertscore needs the 'bert-score' package: pip install 'agent-eval[bertscore]'"
    return None


def create_app(
    registry: MetricRegistry | None = None,
    judge: object | None = None,
    panel: Sequence[object] = (),
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

    return app
```

Replace the whole `src/agent_eval/server/__init__.py` with:

```python
"""agent-eval metric API server (optional ``[server]`` extra). See docs/api-server.md."""

from agent_eval.server.app import create_app

__all__ = ["create_app"]
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/test_server.py -q`
Expected: 11 passed, 1 failed — only `test_openapi_exposes_all_metric_routes` (metric routes land in Task 6)

- [ ] **Step 5: Commit**

```bash
git add src/agent_eval/server/app.py src/agent_eval/server/__init__.py tests/unit/test_server.py
git commit -m "feat(server): create_app with /health and /catalog

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Metric endpoints — score once, replay for the aggregate

**Files:**
- Modify: `src/agent_eval/server/app.py`
- Test: `tests/unit/test_server.py` (append)

- [ ] **Step 1: Write the failing tests** (append to `tests/unit/test_server.py`)

```python
# --------------------------------------------------------------------------- metric endpoints
# Deterministic stub-judge expectations: lexical_overlap_judge scores the share of response
# tokens covered by (context ∪ reference) tokens.
RAG_IDS = {"retrieved_ids": ["d1", "d4", "d9"], "relevant_ids": ["d1", "d4"]}

HAPPY_CASES = [
    (
        "/common/llm_judge",
        {"input": "capital of France?", "output": "Paris", "expected": "Paris"},
        1.0,  # response fully covered by the reference
    ),
    ("/rag/recall_at_k", {"metadata": RAG_IDS}, 1.0),  # both gold ids in top-5
    ("/rag/precision_at_k", {"metadata": RAG_IDS}, 2 / 3),  # 2 relevant of 3 retrieved
    ("/rag/ndcg_at_k", {"metadata": RAG_IDS}, 1.0),  # relevant docs ranked first
    ("/rag/faithfulness", {"output": "alpha delta", "retrieved_context": ["alpha beta gamma"]}, 0.5),
    ("/rag/consistency", {"output": "alpha delta", "retrieved_context": ["alpha beta gamma"]}, 0.5),
]

T2S_HAPPY_CASES = [
    (
        "/t2s/soft_f1",
        {"metadata": {
            "execution_result": [{"name": "Alice"}, {"name": "Carol"}],
            "gold_execution_result": [{"name": "Alice"}, {"name": "Carol"}],
        }},
        1.0,
    ),
    (
        "/t2s/component_match",
        {"metadata": {"sql": "SELECT name FROM employee", "gold_sql": "SELECT name FROM employee"}},
        1.0,
    ),
    ("/t2s/ast_valid", {"metadata": {"sql": "SELECT 1"}}, 1.0),
    ("/t2s/faithfulness", {"input": "q", "output": "3", "metadata": {"execution_result": [{"count": 3}]}}, 1.0),
    ("/t2s/consistency", {"input": "q", "output": "3", "metadata": {"execution_result": [{"count": 3}]}}, 1.0),
]


@pytest.mark.parametrize(("path", "ctx", "expected"), HAPPY_CASES)
def test_endpoint_scores_single_context(client, path, ctx, expected):
    resp = client.post(path, json={"contexts": [ctx]})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["family"] == path.split("/")[1]
    assert body["n"] == 1 and body["n_errors"] == 0
    assert body["results"][0]["error"] is None
    assert body["results"][0]["score"] == pytest.approx(expected)
    assert body["aggregate"]["value"] == pytest.approx(expected)
    assert body["aggregate"]["n"] == 1


@pytest.mark.parametrize(("path", "ctx", "expected"), T2S_HAPPY_CASES)
def test_t2s_endpoint_scores_single_context(client, path, ctx, expected):
    pytest.importorskip("sqlglot")
    resp = client.post(path, json={"contexts": [ctx]})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["results"][0]["error"] is None
    assert body["results"][0]["score"] == pytest.approx(expected)


def test_params_are_plumbed_into_the_metric(client):
    # k=1 → only d1 in the window → recall drops from 1.0 to 0.5
    body = client.post(
        "/rag/recall_at_k", json={"contexts": [{"metadata": RAG_IDS}], "params": {"k": 1}}
    ).json()
    assert body["results"][0]["score"] == pytest.approx(0.5)


def test_batch_aggregate_matches_the_public_evaluate(client):
    from agent_eval.core.contracts import EvalContext
    from agent_eval.core.gate import GatePolicy
    from agent_eval.core.registry import default_registry
    from agent_eval.core.suite import Suite
    from agent_eval.offline.runner import evaluate

    metas = [
        {"retrieved_ids": ["d1", "d4", "d9"], "relevant_ids": ["d1", "d4"]},  # 1.0
        {"retrieved_ids": ["d8", "d9"], "relevant_ids": ["d1", "d4"]},  # 0.0
        {"retrieved_ids": ["d1", "d9"], "relevant_ids": ["d1", "d4"]},  # 0.5
    ]
    body = client.post(
        "/rag/recall_at_k", json={"contexts": [{"metadata": m} for m in metas]}
    ).json()
    assert [r["score"] for r in body["results"]] == [
        pytest.approx(1.0), pytest.approx(0.0), pytest.approx(0.5),
    ]

    expected = evaluate(
        Suite("rag", "recall_at_k", [default_registry().build("recall_at_k")], GatePolicy()),
        [EvalContext(input=None, metadata=m) for m in metas],
    ).aggregates[0]
    agg = body["aggregate"]
    assert agg["value"] == pytest.approx(expected.value)
    assert agg["ci_low"] == pytest.approx(expected.ci_low)
    assert agg["ci_high"] == pytest.approx(expected.ci_high)
    assert (agg["n"], agg["aggregation"]) == (3, "mean")


def test_rate_metric_batch_uses_wilson_pass_rate(client):
    pytest.importorskip("sqlglot")
    sqls = ["SELECT 1", "SELECT 2", "SELECT FROM WHERE"]  # 2 valid, 1 broken
    body = client.post(
        "/t2s/ast_valid", json={"contexts": [{"metadata": {"sql": s}} for s in sqls]}
    ).json()
    assert [r["passed"] for r in body["results"]] == [True, True, False]
    assert body["aggregate"]["aggregation"] == "rate"
    assert body["aggregate"]["value"] == pytest.approx(2 / 3)
    assert body["aggregate"]["ci_low"] < 2 / 3 < body["aggregate"]["ci_high"]


def test_consistency_accepts_repeated_runs_and_returns_final_mean(client):
    runs = [
        {"output": "alpha beta", "retrieved_context": ["alpha beta gamma"]},  # 1.0
        {"output": "alpha delta", "retrieved_context": ["alpha beta gamma"]},  # 0.5
        {"output": "delta epsilon", "retrieved_context": ["alpha beta gamma"]},  # 0.0
    ]
    body = client.post("/rag/consistency", json={"contexts": runs}).json()
    assert body["n"] == 3
    assert body["aggregate"]["value"] == pytest.approx(0.5)


def test_judge_metrics_score_each_item_exactly_once():
    from fastapi.testclient import TestClient

    from agent_eval.judges.backend import FunctionJudge
    from agent_eval.server.app import create_app

    calls = {"n": 0}

    def counting_judge(request):
        calls["n"] += 1
        return 1.0

    app = create_app(judge=FunctionJudge(counting_judge))
    body = TestClient(app).post(
        "/rag/faithfulness",
        json={"contexts": [{"output": "x", "retrieved_context": ["x"]} for _ in range(3)]},
    ).json()
    assert body["n"] == 3
    assert calls["n"] == 3  # replayed for aggregation, never re-scored
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_server.py -q`
Expected: the new tests FAIL with 404s (routes not registered yet); `test_openapi_exposes_all_metric_routes` still failing.

- [ ] **Step 3: Implement the replay metric, handler, and route registration**

In `src/agent_eval/server/app.py`, add after `_unavailable_hint`:

```python
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
        return next(self._results)

    async def ascore(self, ctx: EvalContext) -> MetricResult:
        return next(self._results)


def _coerce_params(params: dict) -> dict:
    """JSON-shape fixups for constructor params (JSON has no tuples: ``scale: [1, 5]``)."""
    out = dict(params)
    if isinstance(out.get("scale"), list):
        out["scale"] = tuple(out["scale"])
    return out
```

Inside `create_app`, add after the `catalog` route (still inside the function):

```python
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
```

(Keep `return app` as the last line of `create_app`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_server.py -q`
Expected: all pass (incl. the previously-failing openapi test). If a stub-judge expectation is off by exact value, print `resp.json()` and check the token sets — do NOT loosen to `> 0` without understanding why.

- [ ] **Step 5: Commit**

```bash
git add src/agent_eval/server/app.py tests/unit/test_server.py
git commit -m "feat(server): 12 metric endpoints with replayed evaluate() aggregation

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: Error model — per-item errors, 422s, 501s

**Files:**
- Modify: `src/agent_eval/server/app.py` (only if a test exposes a gap — the checks were built in Task 6)
- Test: `tests/unit/test_server.py` (append)

- [ ] **Step 1: Write the tests** (append to `tests/unit/test_server.py`)

```python
# --------------------------------------------------------------------------- error model
def test_item_error_is_reported_not_scored(client):
    body = client.post(
        "/rag/recall_at_k",
        json={"contexts": [
            {"metadata": {"retrieved_ids": ["d1"], "relevant_ids": ["d1"]}},
            {"metadata": {}},  # missing ids → this item errors, the request does not
        ]},
    ).json()
    assert body["n"] == 2 and body["n_errors"] == 1
    assert body["results"][0]["error"] is None
    assert body["results"][1]["error"]
    # errored item excluded from the denominator — the aggregate is over the 1 valid item
    assert body["aggregate"]["n"] == 1
    assert body["aggregate"]["value"] == pytest.approx(1.0)


def test_all_items_errored_yields_empty_aggregate(client):
    body = client.post("/rag/recall_at_k", json={"contexts": [{"metadata": {}}]}).json()
    assert body["n_errors"] == 1
    assert body["aggregate"]["n"] == 0 and body["aggregate"]["value"] == 0.0


def test_empty_contexts_is_a_422(client):
    assert client.post("/rag/recall_at_k", json={"contexts": []}).status_code == 422


def test_unknown_param_is_a_422(client):
    resp = client.post(
        "/rag/recall_at_k",
        json={"contexts": [{"metadata": RAG_IDS}], "params": {"kk": 3}},
    )
    assert resp.status_code == 422
    assert "kk" in resp.json()["detail"]


def test_bad_param_value_is_a_422(client):
    resp = client.post(
        "/rag/recall_at_k",
        json={"contexts": [{"metadata": RAG_IDS}], "params": {"k": "not-a-number"}},
    )
    assert resp.status_code == 422


def test_t2s_endpoints_501_when_extra_missing():
    from fastapi.testclient import TestClient

    from agent_eval.core.registry import MetricRegistry
    from agent_eval.judges.backend import FunctionJudge, lexical_overlap_judge
    from agent_eval.metrics import common as common_metrics, rag as rag_metrics
    from agent_eval.server.app import create_app

    reg = MetricRegistry()  # a registry as it would look without sqlglot installed
    common_metrics.register(reg)
    rag_metrics.register(reg)
    client = TestClient(create_app(registry=reg, judge=FunctionJudge(lexical_overlap_judge)))

    resp = client.post("/t2s/soft_f1", json={"contexts": [{"metadata": {}}]})
    assert resp.status_code == 501
    assert "agent-eval[t2s]" in resp.json()["detail"]
    entry = next(e for e in client.get("/catalog").json() if e["path"] == "/t2s/soft_f1")
    assert entry["available"] is False and entry["unavailable_hint"]


def test_bertscore_501_when_package_missing(client, monkeypatch):
    from agent_eval.server import app as app_mod

    monkeypatch.setattr(app_mod, "_bertscore_available", lambda: False)
    resp = client.post("/common/bertscore", json={"contexts": [{"output": "a", "expected": "a"}]})
    assert resp.status_code == 501
    assert "bert-score" in resp.json()["detail"]
```

- [ ] **Step 2: Run the tests**

Run: `uv run pytest tests/unit/test_server.py -q`
Expected: all pass (Task 6 built the checks; these tests pin the contract). If any fail, fix `app.py` — e.g. a missed exception type in the `reg.build` try/except.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_server.py
git commit -m "test(server): pin the error model (item errors, 422, 501)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: Judge wiring through the app (factory env + counting)

**Files:**
- Test: `tests/unit/test_server.py` (append)

- [ ] **Step 1: Write the tests** (append)

```python
# --------------------------------------------------------------------------- judge through the app
def test_factory_env_drives_the_app_judge(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from agent_eval.server import judge as judge_mod
    from agent_eval.server.app import create_app

    factory_file = tmp_path / "my_judges.py"
    factory_file.write_text(
        "from agent_eval.judges.backend import FunctionJudge\n"
        "def make_panel():\n"
        "    return [FunctionJudge(lambda r: 1.0), FunctionJudge(lambda r: 0.5)]\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(judge_mod.ENV_FACTORY, f"{factory_file}:make_panel")
    client = TestClient(create_app())

    info = client.get("/health").json()["judge"]
    assert info["kind"] == "factory" and info["panel_size"] == 2

    body = client.post(
        "/rag/faithfulness", json={"contexts": [{"output": "x", "retrieved_context": ["y"]}]}
    ).json()
    # PoLL panel: mean of (1.0, 0.5) with agreement 1 - spread
    assert body["results"][0]["score"] == pytest.approx(0.75)
    assert body["results"][0]["confidence"] == pytest.approx(0.5)
```

- [ ] **Step 2: Run the tests**

Run: `uv run pytest tests/unit/test_server.py -q`
Expected: all pass (this exercises existing code paths end-to-end; no implementation expected).

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_server.py
git commit -m "test(server): factory-env judge and PoLL panel through the app

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 9: `agent-eval-serve` entrypoint

**Files:**
- Create: `src/agent_eval/server/__main__.py`

- [ ] **Step 1: Implement the entrypoint**

Create `src/agent_eval/server/__main__.py`:

```python
"""Serve the metric API: ``agent-eval-serve`` (or ``python -m agent_eval.server``).

Defaults to loopback; pass ``--host 0.0.0.0`` to expose (the Dockerfile does). The judge is
configured via environment variables — see agent_eval/server/judge.py and docs/api-server.md.
"""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="agent-eval-serve", description="Run the agent-eval metric API server."
    )
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="port (default: 8000)")
    parser.add_argument("--workers", type=int, default=1, help="uvicorn workers (default: 1)")
    args = parser.parse_args(argv)

    import uvicorn  # deferred so `--help` works even on a broken uvicorn install

    uvicorn.run(
        "agent_eval.server.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test the CLI**

Run: `uv run agent-eval-serve --help && uv run python -m agent_eval.server --help`
Expected: both print the usage text with `--host`, `--port`, `--workers`.

- [ ] **Step 3: Boot smoke test**

Run: `cd /Users/makinarocks/workspace/agent-eval && (uv run agent-eval-serve --port 8123 &) && sleep 3 && curl -s http://127.0.0.1:8123/health && curl -s -X POST http://127.0.0.1:8123/rag/recall_at_k -H 'Content-Type: application/json' -d '{"contexts":[{"metadata":{"retrieved_ids":["d1"],"relevant_ids":["d1"]}}]}' && pkill -f "agent-eval-serve --port 8123"`
Expected: health JSON with `"judge":{"kind":"stub"...}`, then a metric response with `"value":1.0`.

- [ ] **Step 4: Commit**

```bash
git add src/agent_eval/server/__main__.py
git commit -m "feat(server): agent-eval-serve entrypoint

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 10: Dockerfile + .dockerignore

**Files:**
- Create: `Dockerfile`
- Create: `.dockerignore`

- [ ] **Step 1: Write the Dockerfile**

Create `Dockerfile`:

```dockerfile
# agent-eval metric API server.
#
# Default extras keep the image lean (no torch). To bake BERTScore in (multi-GB):
#   docker build --build-arg EXTRAS="server,t2s,bertscore" -t agent-eval-api .
# Closed network: point pip at your mirror/wheelhouse, e.g.
#   docker build --build-arg PIP_ARGS="--no-index --find-links=/wheels" ...
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Everything the wheel build needs (hatchling reads pyproject + README; code lives in src/).
COPY pyproject.toml README.md ./
COPY src ./src

ARG EXTRAS="server,t2s"
ARG PIP_ARGS=""
RUN pip install ${PIP_ARGS} ".[${EXTRAS}]"

RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]

CMD ["uvicorn", "agent_eval.server.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 2: Write .dockerignore**

Create `.dockerignore`:

```
.git
.venv
__pycache__
*.pyc
*.egg-info
.pytest_cache
.mypy_cache
.ruff_cache
.hypothesis
dist
build
docs
docs_kr
examples
tests
.env
.env.*
```

- [ ] **Step 3: Build and smoke-test the image (skip gracefully if Docker is unavailable)**

Run: `docker build -t agent-eval-api /Users/makinarocks/workspace/agent-eval && docker run -d --rm -p 18000:8000 --name agent-eval-api-smoke agent-eval-api && sleep 5 && curl -s http://127.0.0.1:18000/health && docker rm -f agent-eval-api-smoke`
Expected: build succeeds; health returns `{"status":"ok",...}`. If the docker daemon is not available on this machine, note it in the final report and rely on the Task 9 boot smoke test.

- [ ] **Step 4: Commit**

```bash
git add Dockerfile .dockerignore
git commit -m "feat(server): Dockerfile (EXTRAS build-arg, non-root, healthcheck)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 11: Documentation (EN + KR)

**Files:**
- Create: `docs/api-server.md`
- Create: `docs_kr/api-server.md`

- [ ] **Step 1: Write the English doc**

Create `docs/api-server.md` with exactly this content:

````markdown
# Metric API server

Every input/output-scoring metric as an HTTP endpoint, so other services can score agent
responses without embedding the framework. The server is a thin wrapper: metric semantics,
statistics, and judge behavior are exactly the library's (`registry` → `metric.score` →
`evaluate()` aggregation).

```
POST /{family}/{metric}   ←  1..N EvalContexts (JSON)
                          →  per-item results + ONE aggregated value with a 95% CI
```

`p95_latency` and `token_usage` have **no endpoint** — they read harness-produced metadata
(no input/output to score) and stay CLI/library-side.

## Install & run

```bash
uv sync --extra server --extra t2s      # or: pip install 'agent-eval[server,t2s]'
uv run agent-eval-serve                 # 127.0.0.1:8000; --host/--port/--workers
```

Docker:

```bash
docker build -t agent-eval-api .
docker run -p 8000:8000 -e AGENT_EVAL_JUDGE_BASE_URL=... -e AGENT_EVAL_JUDGE_MODEL=... agent-eval-api
# BERTScore baked in (pulls torch, multi-GB):
docker build --build-arg EXTRAS="server,t2s,bertscore" -t agent-eval-api .
```

Interactive API docs at `/docs`; machine-readable map at `GET /catalog`; liveness + active judge
at `GET /health`.

## Endpoints

| Path | Metric | Each context needs | `params` |
|---|---|---|---|
| `POST /common/llm_judge` | `llm_judge` | `input`, `output` (opt `expected`) | `criteria`, `scale` |
| `POST /common/bertscore` | `bertscore` | `output`, `expected` | `lang`, `model_type`, `rescale_with_baseline` |
| `POST /rag/recall_at_k` | `recall_at_k` | `metadata.retrieved_ids`, `.relevant_ids` | `k` |
| `POST /rag/precision_at_k` | `precision_at_k` | `metadata.retrieved_ids`, `.relevant_ids` | `k` |
| `POST /rag/ndcg_at_k` | `ndcg_at_k` | same (+ opt `.relevance`) | `k` |
| `POST /rag/faithfulness` | `faithfulness` | `output`, `retrieved_context` | — |
| `POST /rag/consistency` | `consistency` | `output`, `retrieved_context` | — |
| `POST /t2s/soft_f1` | `soft_f1` | `metadata.execution_result`, `.gold_execution_result` | `result_policy` |
| `POST /t2s/component_match` | `component_match` | `metadata.sql`, `.gold_sql` | `dialect` |
| `POST /t2s/ast_valid` | `ast_valid` | `metadata.sql` | `dialect` |
| `POST /t2s/faithfulness` | `t2s_faithfulness` | `output`, `metadata.execution_result` (opt `input`) | `sample_rows`, `max_distinct`, `max_columns` |
| `POST /t2s/consistency` | `t2s_consistency` | `output`, `metadata.execution_result` (opt `input`) | `sample_rows`, `max_distinct`, `max_columns` |

`result_policy` accepts the `ResultSetPolicy` fields: `row_order` (`ignore|strict`), `duplicates`
(`keep|dedup`), `nulls` (`distinct|coalesce`), `column_order` (`ignore|strict`), `float_tolerance`.

## Request / response

One request shape for every endpoint — post one context, or post N (e.g. repeated runs of the
same input for a consistency check) and read the final aggregated value:

```bash
curl -s -X POST http://127.0.0.1:8000/rag/recall_at_k \
  -H 'Content-Type: application/json' \
  -d '{
        "contexts": [
          {"metadata": {"retrieved_ids": ["d1","d4","d9"], "relevant_ids": ["d1","d4"]}},
          {"metadata": {"retrieved_ids": ["d8","d9"],      "relevant_ids": ["d1","d4"]}}
        ],
        "params": {"k": 5}
      }'
```

```jsonc
{
  "family": "rag", "metric": "recall_at_k", "n": 2, "n_errors": 0,
  "results": [
    {"metric": "recall_at_k", "score": 1.0, "passed": null, "confidence": null,
     "cost": {"tokens": 0, "usd": 0.0, "latency_ms": 0.05}, "detail": {}, "error": null},
    {"metric": "recall_at_k", "score": 0.0, "passed": null, "confidence": null,
     "cost": {"tokens": 0, "usd": 0.0, "latency_ms": 0.03}, "detail": {}, "error": null}
  ],
  "aggregate": {"metric": "recall_at_k", "value": 0.5, "ci_low": 0.0, "ci_high": 1.0,
                "n": 2, "n_errors": 0, "aggregation": "mean", "higher_is_better": true}
}
```

The aggregate is computed by the library's own `evaluate()` — pass rate with a Wilson interval
for binary (RATE) metrics, mean with a cluster-robust CI for graded (MEAN) metrics (set
`metadata.cluster_id` on non-iid items). Every item is scored exactly once; judge endpoints call
the LLM once per posted context.

## Error model

| Condition | Result |
|---|---|
| An item can't be scored (missing field/metadata) | **200**; that item's `results[i].error` is set and it is excluded from the aggregate's denominator (`aggregate.n` counts only valid items). Errored ≠ scored 0. |
| Malformed body, empty `contexts`, unknown/bad `params` | **422** with detail |
| Metric's optional dependency missing on the server | **501** with the exact `pip install 'agent-eval[...]'` hint |

## Judge configuration (env, at startup)

| Variable | Meaning |
|---|---|
| `AGENT_EVAL_JUDGE_FACTORY` | `module:attr` or `path/to/file.py:attr` — the YAML `judge.factory` convention. May return a backend, a zero-arg callable, or a list (= PoLL panel). Wins over everything. |
| `AGENT_EVAL_JUDGE_BASE_URL` | OpenAI-compatible base URL (e.g. `http://vllm.internal:8000/v1`). Requires `_MODEL`. |
| `AGENT_EVAL_JUDGE_MODEL` | Model name sent to `{base_url}/chat/completions`. |
| `AGENT_EVAL_JUDGE_API_KEY` | Optional `Authorization: Bearer` token. |
| `AGENT_EVAL_JUDGE_TIMEOUT` | Judge HTTP timeout in seconds (default 60). |

Nothing set → the deterministic lexical-overlap **stub** (offline; not a real judgment).
`GET /health` reports which judge is active — check it before trusting judge-based scores.

## Operational notes

- No auth/TLS — built for a closed network; front with a gateway if exposure matters.
- No batch cap: memory and latency scale with `len(contexts)`; judge endpoints make one LLM call
  per item. Scale out with `--workers` (scoring is synchronous per request).
- Closed-network image builds: `--build-arg PIP_ARGS="--no-index --find-links=/wheels"` with a
  vendored wheelhouse, or point pip at your internal mirror.
````

- [ ] **Step 2: Write the Korean twin**

Create `docs_kr/api-server.md`: a **full, natural Korean translation** of `docs/api-server.md`
(not machine-literal). Match the existing `docs_kr/` style: declarative "~다" endings, keep code
blocks/tables/URLs/env names verbatim, translate prose and table headers. Keep structural parity:
same heading count, same table row count, same code blocks.

- [ ] **Step 3: Verify structural parity**

Run: `grep -c '^#' docs/api-server.md docs_kr/api-server.md && grep -c '^|' docs/api-server.md docs_kr/api-server.md`
Expected: equal counts per file pair.

- [ ] **Step 4: Commit**

```bash
git add docs/api-server.md docs_kr/api-server.md
git commit -m "docs(server): API server guide (EN + KR)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 12: Full verification

- [ ] **Step 1: Whole test suite** (nothing existing may break)

Run: `uv run pytest -q`
Expected: all pass (existing suite + the new `test_server.py`).

- [ ] **Step 2: Lint + types on the new code**

Run: `uv run ruff check src/agent_eval/server tests/unit/test_server.py && uv run mypy`
Expected: clean (mypy runs on the whole `agent_eval` package per pyproject config).

- [ ] **Step 3: Diff audit — additive-only promise**

Run: `git status --short && git log --oneline -12`
Expected modified-only files across the whole feature: `pyproject.toml`, `uv.lock`. Everything
else new: `src/agent_eval/server/*`, `tests/unit/test_server.py`, `Dockerfile`, `.dockerignore`,
`docs/api-server.md`, `docs_kr/api-server.md`, the spec + this plan. If anything else shows as
modified, STOP and investigate before proceeding.

---

## Self-review notes (spec → plan coverage)

- Spec §3 architecture (build → score once → replay through `evaluate()`): Task 6 (`_ReplayMetric`, counting-judge test).
- Spec §4 endpoint map + meta endpoints: Tasks 2, 5, 6 (catalog table = §4 verbatim; `/health`, `/catalog`, `/docs`).
- Spec §5 schemas: Task 3 (field-for-field `MetricAggregate` mirror asserted).
- Spec §6 error model: Task 7 (200-with-error, 422 ×3, 501 ×2).
- Spec §7 judge resolution: Task 4 (unit) + Task 8 (through the app).
- Spec §8 packaging: Task 1 (extra + script), Task 9 (entrypoint).
- Spec §9 Docker: Task 10 (EXTRAS + PIP_ARGS build-args, non-root, healthcheck, factory CMD).
- Spec §10 testing: distributed across Tasks 2–8; importorskip guard in Task 2 Step 1.
- Spec §11 non-goals honored: no auth, no batch cap, no README index edits, no existing-file changes beyond `pyproject.toml`/`uv.lock`.
