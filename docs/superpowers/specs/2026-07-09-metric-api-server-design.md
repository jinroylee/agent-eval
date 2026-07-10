# Metric API server — design

**Date:** 2026-07-09
**Status:** approved (design review in chat); implementation pending

## 1. Goal

Expose every input/output-scoring metric in `agent-eval` as an HTTP endpoint, so other services
(in a closed network) can score agent responses without embedding the framework. This is a **pure
wrapper**: a new `agent_eval.server` subpackage + a Dockerfile. No existing evaluation
method/module changes; the CLI, YAML config path, and library API keep working exactly as today.

## 2. Decisions (settled with the user)

| Question | Decision |
|---|---|
| Packaging | **In-package** `agent_eval.server` subpackage + a `[server]` optional extra. One wheel (`pip install agent-eval[server]`) — fits the closed-network wheelhouse story. |
| Batch shape | **Uniform 1..N everywhere.** Every endpoint accepts `contexts: [1+]`; the response always carries per-item results **and** one final aggregate with a 95% CI (the runner's own statistics). "Multiple runs" (e.g. repeated generations for a consistency check) are simply posted as N contexts. |
| Judge config | Env-resolved at startup, precedence: `AGENT_EVAL_JUDGE_FACTORY` (same `module:attr` convention as YAML `judge.factory`, list ⇒ PoLL panel) → OpenAI-compatible HTTP judge (`AGENT_EVAL_JUDGE_BASE_URL` + `_MODEL`, optional `_API_KEY`/`_TIMEOUT`) → deterministic lexical stub. |
| Excluded metrics | `p95_latency`, `token_usage` get **no endpoint** (they read harness metadata only, no input/output to score). |
| URL prefixes | `/common`, `/rag`, `/t2s` (the metric families; "common" applies to every agent type). T2S paths drop the redundant `t2s_` type prefix. |
| Errors | Framework-faithful: per-item `error` in a 200 body (errored items excluded from the aggregate denominator), 422 for malformed requests/params, 501 for missing optional extras. |

## 3. Architecture — thin wrapper, no metric math in the server

Per request, the server runs the exact code path the CLI uses:

1. `default_registry().build({"type": <registry type>, "params": <request params>}, BuildContext(judge, panel))`
2. `metric.score(ctx)` once per posted context → the per-item results.
3. Aggregation **replays** those results through the public runner:
   `evaluate(Suite(family, type, [replay_metric], GatePolicy()), contexts)` where `replay_metric`
   is a tiny wrapper that returns the already-computed `MetricResult`s in order (and mirrors the
   real metric's `name`/`aggregation`/`higher_is_better`/`unit_interval`). This keeps Wilson /
   cluster-robust / bootstrap CI logic in exactly one place **and** never scores an item twice —
   judge metrics must not pay the LLM twice per item.

The registry and judge are resolved once at app startup. `create_app(registry=None, judge=None,
panel=())` accepts injected overrides for tests; by default it builds `default_registry()` and
resolves the judge from env.

## 4. Endpoint map (12 metric endpoints + 2 meta)

| Path | Registry type | Required per context | Allowed `params` keys |
|---|---|---|---|
| `POST /common/llm_judge` | `llm_judge` | `input`, `output` (opt `expected`) | `criteria` (str), `scale` ([lo, hi]) |
| `POST /common/bertscore` | `bertscore` | `output`, `expected` | `lang`, `model_type`, `rescale_with_baseline` |
| `POST /rag/recall_at_k` | `recall_at_k` | `metadata.retrieved_ids`, `metadata.relevant_ids` | `k` (int) |
| `POST /rag/precision_at_k` | `precision_at_k` | same as above | `k` |
| `POST /rag/ndcg_at_k` | `ndcg_at_k` | same (+ opt `metadata.relevance`) | `k` |
| `POST /rag/faithfulness` | `faithfulness` | `output`, `retrieved_context` | — |
| `POST /rag/consistency` | `consistency` | `output`, `retrieved_context` | — |
| `POST /t2s/soft_f1` | `soft_f1` | `metadata.execution_result`, `metadata.gold_execution_result` | `result_policy` (object → `ResultSetPolicy`) |
| `POST /t2s/component_match` | `component_match` | `metadata.sql`, `metadata.gold_sql` | `dialect` (str) |
| `POST /t2s/ast_valid` | `ast_valid` | `metadata.sql` | `dialect` |
| `POST /t2s/faithfulness` | `t2s_faithfulness` | `output`, `metadata.execution_result` (opt `input`) | `sample_rows`, `max_distinct`, `max_columns` |
| `POST /t2s/consistency` | `t2s_consistency` | same as above | same as above |

Meta endpoints:

- `GET /health` — `{status, version, judge: {kind: stub|factory|openai_compatible, detail, panel_size}, available: {t2s: bool, bertscore: bool}}`. Surfacing the judge kind is deliberate: stub scores must never be mistaken for real LLM judgments.
- `GET /catalog` — one entry per metric endpoint: `{family, path, type, requires, metadata_keys, params, judge_based, cost_class, aggregation, available, unavailable_hint}`.
- FastAPI's built-in `/docs` (Swagger) / `/openapi.json`; each route carries a description generated from the catalog entry.

The endpoint table lives in **one declarative module** (`catalog.py`); routes are registered from
it in a loop, so adding a metric later is a one-line table entry.

## 5. Request / response schema

Request (identical for all 12 endpoints):

```jsonc
{
  "contexts": [                       // min 1 item
    {
      "input": "...",                 // any JSON; optional (metric `requires` decides)
      "output": "...",
      "expected": "...",
      "retrieved_context": ["..."],   // list[str], default []
      "metadata": { "...": "..." }    // MetaKey names, default {}
    }
  ],
  "params": { "k": 5 }                // optional; keys validated against the endpoint's allowed set
}
```

Response (200):

```jsonc
{
  "family": "rag",
  "metric": "recall_at_k",            // the registry type
  "n": 3,                             // contexts posted
  "n_errors": 0,                      // items whose metric errored (excluded from aggregate)
  "results": [                        // one per posted context, same order
    { "metric": "recall_at_k", "score": 0.8, "passed": null, "confidence": null,
      "detail": {}, "error": null,
      "cost": { "tokens": 0, "usd": 0.0, "latency_ms": 0.41 } }
  ],
  "aggregate": {                      // mirrors core.gate.MetricAggregate exactly
    "metric": "recall_at_k", "value": 0.8, "ci_low": 0.61, "ci_high": 0.92,
    "n": 3, "n_errors": 0, "aggregation": "mean", "higher_is_better": true
  }
}
```

- The aggregate is **always** present (n=1 gives a degenerate but well-defined interval).
- Pydantic models: `EvalContextIn` (all fields optional/defaulted; converted to the frozen
  `EvalContext` dataclass), `MetricRequest`, `CostOut`, `MetricResultOut`, `AggregateOut`,
  `MetricResponse`, plus health/catalog models.
- `metadata.cluster_id` keeps its runner meaning (cluster-robust CI for MEAN metrics).

## 6. Error model

| Condition | HTTP | Body |
|---|---|---|
| Item can't be scored (missing field/metadata, judge parse failure, …) | 200 | that item's `results[i].error` set (metric name, `score: 0.0`); item excluded from the aggregate denominator; `n_errors` incremented — the runner's "errored ≠ scored 0" semantics |
| Malformed JSON / schema violation (`contexts` empty, wrong types) | 422 | FastAPI/pydantic detail |
| Unknown `params` key for that endpoint, or a param value the metric constructor rejects (`TypeError`/`ValueError`/`ConfigError`) | 422 | `{"detail": "<message>"}` |
| Endpoint whose optional dependency is missing (`bertscore` without `bert-score`; any `/t2s/*` without `sqlglot`) | 501 | `{"detail": "... pip install 'agent-eval[t2s]'"}` — the route always exists; availability is checked per request |
| Unexpected exception outside `metric.score` | 500 | generic detail |

## 7. Judge resolution (startup)

```
AGENT_EVAL_JUDGE_FACTORY="module:attr" | "path/to/file.py:attr"
    → resolved via config.loader (same unwrap semantics as YAML judge.factory:
      backend | zero-arg callable | list ⇒ (primary, panel))
else AGENT_EVAL_JUDGE_BASE_URL + AGENT_EVAL_JUDGE_MODEL
    → LLMJudge(complete=<httpx POST {base_url}/chat/completions>)
      (temperature 0; optional AGENT_EVAL_JUDGE_API_KEY → Authorization: Bearer;
       AGENT_EVAL_JUDGE_TIMEOUT seconds, default 60)
else → FunctionJudge(lexical_overlap_judge)   # deterministic offline stub
```

- `BASE_URL` set without `MODEL` (or vice versa) → fail fast at startup with a clear message.
- The OpenAI-compatible judge reuses the existing `LLMJudge` prompt rendering and `SCORE:/REASON:`
  parsing — the server contributes only the `complete(prompt) -> str` callable (httpx).
- `/health` reports which path won.

## 8. Files & packaging (all additive)

```
src/agent_eval/server/__init__.py     # exports create_app
src/agent_eval/server/app.py          # create_app(), route registration, handlers
src/agent_eval/server/schemas.py      # pydantic request/response models
src/agent_eval/server/catalog.py      # declarative endpoint table (single source of truth)
src/agent_eval/server/judge.py        # env resolution + OpenAI-compatible complete()
src/agent_eval/server/__main__.py     # `agent-eval-serve` / `python -m agent_eval.server` → uvicorn
tests/unit/test_server.py             # TestClient suite (skips cleanly if fastapi not installed)
Dockerfile
.dockerignore
docs/api-server.md                    # usage doc (EN)
docs_kr/api-server.md                 # usage doc (KR, natural translation)
```

`pyproject.toml` — **additive edits only**:

- `[project.optional-dependencies] server = ["fastapi>=0.111", "uvicorn[standard]>=0.30", "httpx>=0.27"]`
- `[project.scripts] agent-eval-serve = "agent_eval.server.__main__:main"`

`uv.lock` is regenerated (`uv lock`) so `uv sync --extra server` works. No other existing file is
touched.

`agent-eval-serve` args: `--host` (default `127.0.0.1`), `--port` (default `8000`), `--workers`
(default 1); runs uvicorn with the `create_app` factory. The Dockerfile passes `--host 0.0.0.0`.

## 9. Dockerfile

- `FROM python:3.12-slim` (matches `requires-python >= 3.12`).
- `ARG EXTRAS="server,t2s"` → `pip install ".[${EXTRAS}]"`; build with
  `--build-arg EXTRAS="server,t2s,bertscore"` to bake in BERTScore (pulls torch — multi-GB — so
  it is off by default; the endpoint answers 501 until installed).
- Non-root user, `EXPOSE 8000`, `HEALTHCHECK` hitting `/health` via stdlib urllib (slim has no curl),
  `CMD ["uvicorn", "agent_eval.server.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]`.
- Closed-network note (docs): build against an internal mirror or a vendored wheelhouse
  (`pip install --no-index --find-links=/wheels ".[server,t2s]"`).

## 10. Testing

`tests/unit/test_server.py`, module-level `pytest.importorskip("fastapi")` so the existing suite
still passes without the extra. Stub judge by default (deterministic). Cases:

1. Happy path for each of the 12 endpoints (minimal valid contexts).
2. Batch: N=3 → `aggregate.n == 3`; **aggregate equals `evaluate()`** run directly on the same
   suite/contexts (proves the replay wrapper is faithful).
3. Consistency-as-multi-run: 3 repeated-run contexts → one final mean.
4. Per-item error: missing metadata → 200, `error` set, `n_errors == 1`, aggregate over the rest;
   all-errored → aggregate `n == 0`, `value == 0.0` (runner behavior).
5. 422: empty `contexts`, unknown `params` key, bad param value.
6. 501: injected registry without T2S types; monkeypatched missing `bert_score`.
7. `/health` + `/catalog` shapes; judge kind reporting (stub vs env-configured).
8. OpenAI-compatible judge: `httpx.MockTransport` returning `SCORE: 4\nREASON: ok` → 0.75 with
   the default 1–5 scale; auth header and model propagation asserted.
9. Factory env: points at a test factory returning `FunctionJudge`s (single + panel).

## 11. Non-goals / constraints

- **No existing behavior changes** — server is additive; metric semantics stay in `metrics/*`.
- No auth/TLS/rate limiting (closed-network internal tool; front with a gateway if needed).
- No batch size cap (documented: memory/latency scale with N; judge endpoints call the LLM N times).
- No `/plain` alias for `/common`; no per-request judge override (server-level only, by design).
- No README index edits (`docs/README.md`, `docs_kr/README.md`) — offered separately to keep
  existing files untouched.
- Scoring is synchronous per request (FastAPI threadpool); scale out with `--workers`.
