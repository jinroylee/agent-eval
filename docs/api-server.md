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

Interactive API docs at `/docs`; machine-readable map at `GET /catalog`; liveness, the active
judge, and optional-capability availability (`t2s`/`bertscore`) at `GET /health`.

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
An explicit JSON `null` for any param means "use the default" — same as omitting the key.

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
`metadata.cluster_id` on non-iid items). Every item is scored exactly once; judge endpoints run
the judge once per posted context — one LLM call per configured judge, so a PoLL panel
multiplies that by its size. `n` counts posted contexts; `aggregate.n` counts only the items
that actually scored (errored items are excluded from the denominator).

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
  per item per configured judge (a PoLL panel multiplies accordingly). Scale out with
  `--workers` (scoring is synchronous per request).
- `uvicorn[standard]` ships compiled wheels (`uvloop`, `httptools`, `watchfiles`, `websockets`).
  If your internal mirror can't serve them for your platform, install plain `uvicorn` instead —
  the server runs fine on it.

## Closed-network image builds

Vendor a wheelhouse on a connected machine, ship it into the build context, and point pip at it:

```bash
# connected machine — resolve the full closure for linux/py3.12
docker run --rm -v "$PWD:/w" -w /w python:3.12-slim \
  pip download ".[server,t2s]" -d wheelhouse/

# closed network — add `COPY wheelhouse /wheels` above the pip install line in the
# Dockerfile, then:
docker build --build-arg PIP_ARGS="--no-index --find-links=/wheels" -t agent-eval-api .
```

Or point `PIP_ARGS` at an internal mirror (`--index-url https://pypi.internal/simple`).
**Never embed credentials in `PIP_ARGS`** — build args are recorded in the image history
(`docker history`); use an unauthenticated mirror, or a BuildKit secret mount for `pip.conf`.
